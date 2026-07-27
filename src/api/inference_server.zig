const std = @import("std");
const net = std.net;
const mem = std.mem;
const fs = std.fs;
const Thread = std.Thread;
const Allocator = mem.Allocator;
const RSFLayer = @import("../processor/rsf.zig").RSFLayer;
const Ranker = @import("../ranker/ranker.zig").Ranker;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const SSI = @import("../index/ssi.zig").SSI;
const Tensor = @import("../core/tensor.zig").Tensor;
const ModelFormat = @import("../core/model_io.zig").ModelFormat;
const importModel = @import("../core/model_io.zig").importModel;
const LearnedEmbedding = @import("../core/learned_embedding.zig").LearnedEmbedding;
const OFTB = @import("../processor/oftb.zig").OFTB;
const core_relational = @import("../core_relational/mod.zig");
const ChaosCoreKernel = core_relational.ChaosCoreKernel;
const CREVPipeline = core_relational.CREVPipeline;
const EntangledStochasticSymmetryOptimizer = core_relational.EntangledStochasticSymmetryOptimizer;
const ReasoningOrchestrator = core_relational.ReasoningOrchestrator;
const SurpriseMemoryManager = core_relational.SurpriseMemoryManager;
const TemporalGraph = core_relational.TemporalGraph;
const VerifiedInferenceEngine = core_relational.VerifiedInferenceEngine;
const SignalPropagationEngine = core_relational.SignalPropagationEngine;
const ZRuntime = core_relational.ZRuntime;
const SelfSimilarRelationalGraph = core_relational.SelfSimilarRelationalGraph;
const QuantumState = core_relational.QuantumState;
const FractalLPU = @import("../hw/accel/fractal_lpu.zig").FractalLPU;
const RelationalGraphProcessingUnit = core_relational.RelationalGraphProcessingUnit;
const VPU = core_relational.VPU;
const FNDSManager = core_relational.FNDSManager;
const PatternLocation = core_relational.PatternLocation;
const FormalVerificationEngine = core_relational.FormalVerificationEngine;
const SecurityProofEngine = core_relational.SecurityProofEngine;
const QuantumTaskAdapter = core_relational.QuantumTaskAdapter;
const QuantumSubgraph = core_relational.QuantumSubgraph;

const RESERVED_TOKEN_COUNT: usize = 4;

pub const ServerConfig = struct {
    port: u16 = 8080,
    host: []const u8 = "127.0.0.1",
    max_connections: u32 = 100,
    request_timeout_ms: u64 = 30000,
    batch_size: usize = 32,
    model_path: ?[]const u8 = null,
    dataset_path: ?[]const u8 = null,
    rate_limit_per_minute: u32 = 10,
    max_request_size_bytes: usize = 16 * 1024 * 1024,
    require_api_key: bool = true,
    num_worker_threads: u32 = 4,
    keep_alive_timeout_ms: u64 = 5000,
    esso_initial_temp: f64 = 100.0,
    esso_cooling_rate: f64 = 0.95,
    esso_max_iterations: usize = 10000,
};

const RateLimiter = struct {
    const RequestLog = struct {
        timestamps: std.ArrayList(i64),
        mutex: Thread.Mutex,
    };

    logs: std.StringHashMap(RequestLog),
    key_storage: std.ArrayList([]u8),
    allocator: Allocator,
    mutex: Thread.Mutex,
    window_seconds: u64,
    max_requests: u32,

    pub fn init(allocator: Allocator, max_requests_per_minute: u32) RateLimiter {
        return RateLimiter{
            .logs = std.StringHashMap(RequestLog).init(allocator),
            .key_storage = std.ArrayList([]u8).init(allocator),
            .allocator = allocator,
            .mutex = Thread.Mutex{},
            .window_seconds = 60,
            .max_requests = max_requests_per_minute,
        };
    }

    pub fn deinit(self: *RateLimiter) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var iter = self.logs.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.timestamps.deinit();
        }
        self.logs.deinit();

        for (self.key_storage.items) |key| {
            self.allocator.free(key);
        }
        self.key_storage.deinit();
    }

    pub fn checkAndRecord(self: *RateLimiter, ip_address: []const u8) !bool {
        const now = std.time.timestamp();
        const cutoff = now - @as(i64, @intCast(self.window_seconds));

        self.mutex.lock();
        defer self.mutex.unlock();

        const result = self.logs.getOrPut(ip_address) catch return error.OutOfMemory;
        if (!result.found_existing) {
            const owned_key = self.allocator.dupe(u8, ip_address) catch return error.OutOfMemory;
            self.key_storage.append(owned_key) catch {
                self.allocator.free(owned_key);
                return error.OutOfMemory;
            };
            result.key_ptr.* = owned_key;
            result.value_ptr.* = RequestLog{
                .timestamps = std.ArrayList(i64).init(self.allocator),
                .mutex = Thread.Mutex{},
            };
        }

        var log = result.value_ptr;
        log.mutex.lock();
        defer log.mutex.unlock();

        var i: usize = 0;
        while (i < log.timestamps.items.len) {
            if (log.timestamps.items[i] < cutoff) {
                _ = log.timestamps.orderedRemove(i);
            } else {
                i += 1;
            }
        }

        if (log.timestamps.items.len >= self.max_requests) {
            return false;
        }

        log.timestamps.append(now) catch return error.OutOfMemory;
        return true;
    }
};

pub const InferenceRequest = struct {
    text: []const u8,
    max_tokens: ?usize = null,
    return_embeddings: bool = false,

    pub fn fromJson(allocator: Allocator, json: []const u8) !InferenceRequest {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return error.InvalidJson;
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.InvalidJson;

        const text_val = root.object.get("text") orelse return error.MissingTextField;
        if (text_val != .string) return error.InvalidTextField;

        var max_tokens: ?usize = null;
        if (root.object.get("max_tokens")) |mt| {
            if (mt == .integer) {
                if (mt.integer < 0) return error.InvalidMaxTokens;
                if (mt.integer > 1000000) return error.MaxTokensTooLarge;
                max_tokens = @intCast(mt.integer);
            }
        }

        var return_embeddings = false;
        if (root.object.get("return_embeddings")) |re| {
            if (re == .bool) {
                return_embeddings = re.bool;
            }
        }

        return InferenceRequest{
            .text = try allocator.dupe(u8, text_val.string),
            .max_tokens = max_tokens,
            .return_embeddings = return_embeddings,
        };
    }

    pub fn deinit(self: *InferenceRequest, allocator: Allocator) void {
        allocator.free(self.text);
    }
};

pub const InferenceResponse = struct {
    tokens: []u32,
    text: ?[]const u8 = null,
    input_tokens: ?[]u32 = null,
    embeddings: ?[]f32 = null,
    processing_time_ms: f64,

    pub fn toJson(self: *const InferenceResponse, allocator: Allocator) ![]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();
        var writer = list.writer();

        try writer.writeAll("{\"tokens\":[");
        var i: usize = 0;
        while (i < self.tokens.len) : (i += 1) {
            if (i > 0) try writer.writeAll(",");
            try writer.print("{d}", .{self.tokens[i]});
        }
        try writer.writeAll("]");

        if (self.text) |t| {
            try writer.writeAll(",\"text\":\"");
            var ci: usize = 0;
            while (ci < t.len) : (ci += 1) {
                const c = t[ci];
                switch (c) {
                    '"' => try writer.writeAll("\\\""),
                    '\\' => try writer.writeAll("\\\\"),
                    '\n' => try writer.writeAll("\\n"),
                    '\r' => try writer.writeAll("\\r"),
                    '\t' => try writer.writeAll("\\t"),
                    else => {
                        if (c >= 0x20 and c < 0x7f) {
                            try writer.writeByte(c);
                        } else {
                            try writer.print("\\u{d:0>4}", .{@as(u16, @intCast(c))});
                        }
                    },
                }
            }
            try writer.writeAll("\"");
        }

        if (self.input_tokens) |it| {
            try writer.writeAll(",\"input_tokens\":[");
            var j: usize = 0;
            while (j < it.len) : (j += 1) {
                if (j > 0) try writer.writeAll(",");
                try writer.print("{d}", .{it[j]});
            }
            try writer.writeAll("]");
        }

        if (self.embeddings) |emb| {
            try writer.writeAll(",\"embeddings\":[");
            var j: usize = 0;
            while (j < emb.len) : (j += 1) {
                if (j > 0) try writer.writeAll(",");
                try writer.print("{d:.6}", .{emb[j]});
            }
            try writer.writeAll("]");
        }

        try writer.print(",\"processing_time_ms\":{d:.2}", .{self.processing_time_ms});
        try writer.writeAll("}");

        return try list.toOwnedSlice();
    }

    pub fn deinit(self: *InferenceResponse, allocator: Allocator) void {
        allocator.free(self.tokens);
        if (self.text) |t| {
            allocator.free(t);
        }
        if (self.input_tokens) |it| {
            allocator.free(it);
        }
        if (self.embeddings) |emb| {
            allocator.free(emb);
        }
    }
};

pub const HealthResponse = struct {
    status: []const u8 = "healthy",
    uptime_seconds: u64,
    model_loaded: bool,
    version: []const u8 = "1.0.0",

    pub fn toJson(self: *const HealthResponse, allocator: Allocator) ![]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();
        var writer = list.writer();

        try writer.writeAll("{");
        try writer.print("\"status\":\"{s}\",", .{self.status});
        try writer.print("\"uptime_seconds\":{d},", .{self.uptime_seconds});
        try writer.print("\"model_loaded\":{s},", .{if (self.model_loaded) "true" else "false"});
        try writer.print("\"version\":\"{s}\"", .{self.version});
        try writer.writeAll("}");

        return try list.toOwnedSlice();
    }
};

fn boostAboveMean(data: []f32) void {
    if (data.len == 0) return;
    var mean: f64 = 0.0;
    for (data) |v| {
        mean += @as(f64, v);
    }
    mean /= @as(f64, @floatFromInt(data.len));
    const mean_f32: f32 = @floatCast(mean);
    var i: usize = 0;
    while (i < data.len) : (i += 1) {
        if (data[i] > mean_f32) {
            data[i] *= 1.05;
        }
    }
}

const ConnectionContext = struct {
    server: *InferenceServer,
    stream: net.Stream,
    client_addr: net.Address,
};

pub const InferenceServer = struct {
    allocator: Allocator,
    config: ServerConfig,
    model: ?ModelFormat = null,
    ssi: ?SSI = null,
    ranker: ?Ranker = null,
    request_count: std.atomic.Value(u64),
    inference_mutex: Thread.Mutex,
    start_time: i64,
    running: std.atomic.Value(bool),
    embedding: ?LearnedEmbedding,
    nsir_graph: ?SelfSimilarRelationalGraph,
    chaos_kernel: ?ChaosCoreKernel,
    esso: ?EntangledStochasticSymmetryOptimizer,
    surprise_memory: ?SurpriseMemoryManager,
    temporal_graph: ?TemporalGraph,
    verifier: ?*VerifiedInferenceEngine,
    signal_engine: ?SignalPropagationEngine,
    z_runtime: ?*ZRuntime,
    rate_limiter: RateLimiter,
    api_key: ?[]const u8,
    fractal_lpu: ?FractalLPU,
    r_gpu: ?RelationalGraphProcessingUnit,
    active_connections: std.atomic.Value(u32),
    thread_pool: ?*Thread.Pool,
    vpu: ?VPU,
    fnds_manager: ?FNDSManager,
    crev_pipeline: ?CREVPipeline,
    formal_verifier: ?*FormalVerificationEngine,
    security_engine: ?*SecurityProofEngine,
    quantum_adapter: ?QuantumTaskAdapter,

    pub fn init(allocator: Allocator, config: ServerConfig) !InferenceServer {
        var api_key: ?[]const u8 = null;
        if (config.require_api_key) {
            if (std.posix.getenv("JAIDE_API_KEY")) |env_key| {
                api_key = try allocator.dupe(u8, env_key);
            } else {
                return error.ApiKeyRequired;
            }
        }

        var model_path_owned: ?[]const u8 = null;
        if (config.model_path) |mp| {
            model_path_owned = try allocator.dupe(u8, mp);
        } else if (std.posix.getenv("JAIDE_MODEL_PATH")) |env_path| {
            model_path_owned = try allocator.dupe(u8, env_path);
        }

        var effective_config = config;
        if (model_path_owned) |mp| {
            effective_config.model_path = mp;
        }

        return InferenceServer{
            .allocator = allocator,
            .config = effective_config,
            .embedding = null,
            .nsir_graph = null,
            .chaos_kernel = null,
            .esso = null,
            .surprise_memory = null,
            .temporal_graph = null,
            .verifier = null,
            .signal_engine = null,
            .z_runtime = null,
            .request_count = std.atomic.Value(u64).init(0),
            .inference_mutex = Thread.Mutex{},
            .start_time = std.time.milliTimestamp(),
            .running = std.atomic.Value(bool).init(false),
            .rate_limiter = RateLimiter.init(allocator, config.rate_limit_per_minute),
            .api_key = api_key,
            .fractal_lpu = null,
            .r_gpu = null,
            .active_connections = std.atomic.Value(u32).init(0),
            .thread_pool = null,
            .vpu = null,
            .fnds_manager = null,
            .crev_pipeline = null,
            .formal_verifier = null,
            .security_engine = null,
            .quantum_adapter = null,
        };
    }

    pub fn deinit(self: *InferenceServer) void {
        if (self.model) |*model| {
            model.deinit();
        }
        if (self.ssi) |*ssi| {
            ssi.deinit();
        }
        if (self.ranker) |*r| {
            r.deinit();
        }
        if (self.embedding) |*emb| {
            emb.deinit();
        }
        if (self.quantum_adapter) |*qa| {
            qa.deinit();
        }
        if (self.security_engine) |se| {
            se.deinit();
            self.allocator.destroy(se);
        }
        if (self.formal_verifier) |fv| {
            fv.deinit();
            self.allocator.destroy(fv);
        }
        if (self.crev_pipeline) |*cp| {
            cp.deinit();
        }
        if (self.fnds_manager) |*fm| {
            fm.deinit();
        }
        if (self.vpu) |*v| {
            v.deinit();
        }
        if (self.signal_engine) |*se| {
            se.deinit();
        }
        if (self.temporal_graph) |*tg| {
            tg.deinit();
        }
        if (self.surprise_memory) |*sm| {
            sm.deinit();
        }
        if (self.esso) |*esso_opt| {
            esso_opt.deinit();
        }
        if (self.verifier) |v| {
            v.deinit();
        }
        if (self.chaos_kernel) |*kernel| {
            kernel.deinit();
        }
        if (self.nsir_graph) |*graph| {
            graph.deinit();
        }
        if (self.z_runtime) |zr| {
            zr.deinit();
        }
        if (self.api_key) |key| {
            self.allocator.free(key);
        }
        if (self.config.model_path) |mp| {
            self.allocator.free(mp);
        }
        if (self.fractal_lpu) |*fl| {
            fl.deinit();
        }
        if (self.r_gpu) |*rg| {
            rg.deinit();
        }
        if (self.thread_pool) |pool| {
            pool.deinit();
            self.allocator.destroy(pool);
        }
        self.rate_limiter.deinit();
    }

    pub fn loadModel(self: *InferenceServer, path: []const u8) !void {
        self.model = try importModel(path, self.allocator);
        self.ssi = SSI.init(self.allocator);
        self.ranker = try Ranker.init(self.allocator, 3, 8, 42);

        const dim = if (self.model.?.rsf) |rsf| (rsf.ctrl orelse return).dim else 256;

        const source_vocab_size: usize = blk: {
            if (self.model.?.mgt) |mgt| {
                const tokenizer_vocab = mgt.vocabSize();
                if (tokenizer_vocab > 0) break :blk tokenizer_vocab;
            }
            const metadata_vocab = self.model.?.metadata.mgt_vocab_size;
            if (metadata_vocab > 0) break :blk metadata_vocab;
            break :blk 50000;
        };

        self.embedding = try LearnedEmbedding.init(self.allocator, source_vocab_size, dim * 2, 42);

        self.nsir_graph = try SelfSimilarRelationalGraph.init(self.allocator);
        self.chaos_kernel = ChaosCoreKernel.init(self.allocator);
        self.esso = EntangledStochasticSymmetryOptimizer.init(
            self.allocator,
            self.config.esso_initial_temp,
            self.config.esso_cooling_rate,
            self.config.esso_max_iterations,
        );
        self.surprise_memory = SurpriseMemoryManager.init(self.allocator, &self.chaos_kernel.?.storage, &self.chaos_kernel.?.flow_analyzer);
        self.temporal_graph = TemporalGraph.init(self.allocator);
        self.signal_engine = SignalPropagationEngine.init(self.allocator, &self.nsir_graph.?, &self.chaos_kernel.?.flow_analyzer);

        if (std.posix.getenv("JAIDE_VERIFY")) |v| {
            if (std.mem.eql(u8, v, "1")) {
                self.verifier = try VerifiedInferenceEngine.init(self.allocator);
            }
        }

        self.z_runtime = try ZRuntime.init(self.allocator);

        self.fractal_lpu = FractalLPU.init(self.allocator, 65536, 1.5) catch null;
        self.r_gpu = RelationalGraphProcessingUnit.init(self.allocator, 4, 4) catch null;

        self.vpu = VPU.init(self.allocator) catch null;
        self.fnds_manager = FNDSManager.init(self.allocator) catch null;
        if (self.chaos_kernel) |*ck| {
            self.crev_pipeline = CREVPipeline.init(self.allocator, ck) catch null;
        }
        const fv_ptr = self.allocator.create(FormalVerificationEngine) catch null;
        if (fv_ptr) |fv| {
            if (FormalVerificationEngine.init(self.allocator)) |fv_val| {
                fv.* = fv_val;
                self.formal_verifier = fv;
            } else |_| {
                self.allocator.destroy(fv);
            }
        }
        const se_ptr = self.allocator.create(SecurityProofEngine) catch null;
        if (se_ptr) |se| {
            se.* = try SecurityProofEngine.init(self.allocator, .INTERNAL);
            self.security_engine = se;
        }
        if (self.nsir_graph) |*graph| {
            self.quantum_adapter = QuantumTaskAdapter.init(self.allocator, graph);
        }
    }

    fn isModelLoaded(self: *const InferenceServer) bool {
        if (self.model == null) return false;
        if (self.model.?.mgt == null) return false;
        if (self.model.?.rsf == null) return false;
        if (self.model.?.rsf.?.ctrl == null) return false;
        if (self.embedding == null) return false;
        if (self.ssi == null) return false;
        if (self.ranker == null) return false;
        return true;
    }

    pub fn start(self: *InferenceServer) !void {
        const address = try net.Address.parseIp(self.config.host, self.config.port);
        var server = address.listen(.{
            .reuse_address = true,
        }) catch |err| {
            std.debug.print("Failed to listen: {}\n", .{err});
            return err;
        };
        defer server.deinit();

        const pool = try self.allocator.create(Thread.Pool);
        errdefer self.allocator.destroy(pool);
        try pool.init(.{
            .allocator = self.allocator,
            .n_jobs = @intCast(self.config.num_worker_threads),
        });
        self.thread_pool = pool;
        self.running.store(true, .seq_cst);

        std.debug.print("Security configuration:\n", .{});
        std.debug.print("   - API key auth: {s}\n", .{if (self.api_key != null) "ENABLED" else "DISABLED"});
        std.debug.print("   - Rate limiting: {d} requests/min per IP\n", .{self.config.rate_limit_per_minute});
        std.debug.print("   - Max request size: {d} bytes\n", .{self.config.max_request_size_bytes});
        std.debug.print("   - Worker threads: {d}\n", .{self.config.num_worker_threads});
        std.debug.print("\n", .{});
        std.debug.print("Inference server listening on {s}:{d}\n", .{ self.config.host, self.config.port });

        while (self.running.load(.seq_cst)) {
            const connection = server.accept() catch |err| {
                std.debug.print("Failed to accept connection: {}\n", .{err});
                continue;
            };

            const active = self.active_connections.load(.monotonic);
            if (active >= self.config.max_connections) {
                _ = connection.stream.writeAll("HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: 39\r\n\r\n{\"error\":\"Too many connections\"}") catch {};
                connection.stream.close();
                continue;
            }

            const ctx = self.allocator.create(ConnectionContext) catch {
                connection.stream.close();
                continue;
            };
            ctx.* = .{
                .server = self,
                .stream = connection.stream,
                .client_addr = connection.address,
            };

            _ = self.active_connections.fetchAdd(1, .monotonic);

            self.thread_pool.?.spawn(handleConnectionWork, .{ctx}) catch {
                self.allocator.destroy(ctx);
                connection.stream.close();
                _ = self.active_connections.fetchSub(1, .monotonic);
                continue;
            };
        }
    }

    fn handleConnectionWork(ctx: *ConnectionContext) void {
        const server = ctx.server;
        defer {
            server.allocator.destroy(ctx);
            _ = server.active_connections.fetchSub(1, .monotonic);
        }

        server.handleStreamConnection(ctx.stream, ctx.client_addr) catch {};
    }

    pub fn stop(self: *InferenceServer) void {
        self.running.store(false, .seq_cst);
    }

    fn setSocketTimeout(stream: net.Stream, timeout_ms: u64) void {
        const sec: i64 = @intCast(timeout_ms / 1000);
        const usec: i64 = @intCast((timeout_ms % 1000) * 1000);
        var tv = std.posix.timeval{ .sec = sec, .usec = usec };
        const optval = std.mem.asBytes(&tv);
        _ = std.posix.setsockopt(stream.handle, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, optval) catch {};
    }

    fn handleStreamConnection(self: *InferenceServer, stream: net.Stream, client_addr: net.Address) !void {
        defer stream.close();

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const temp_allocator = arena.allocator();

        var ip_buf: [128]u8 = undefined;
        const ip_len = std.fmt.bufPrint(&ip_buf, "{}", .{client_addr}) catch return;
        const ip_str = try temp_allocator.dupe(u8, ip_len);

        var request_buf = std.ArrayList(u8).init(self.allocator);
        defer request_buf.deinit();

        var tmp_buf = try self.allocator.alloc(u8, 8192);
        defer self.allocator.free(tmp_buf);

        var is_first_request = true;

        while (true) {
            request_buf.clearRetainingCapacity();

            if (is_first_request) {
                setSocketTimeout(stream, self.config.request_timeout_ms);
            } else {
                setSocketTimeout(stream, self.config.keep_alive_timeout_ms);
            }
            is_first_request = false;

            var headers_found = false;
            var total_read: usize = 0;
            while (total_read < self.config.max_request_size_bytes) {
                const bytes_read = stream.read(tmp_buf) catch break;
                if (bytes_read == 0) return;
                try request_buf.appendSlice(tmp_buf[0..bytes_read]);
                total_read += bytes_read;

                if (mem.indexOf(u8, request_buf.items, "\r\n\r\n")) |_| {
                    headers_found = true;
                    break;
                }
            }

            if (!headers_found or request_buf.items.len == 0) return;

            if (request_buf.items.len >= self.config.max_request_size_bytes) {
                try self.sendError(stream, "Request too large", 413);
                return;
            }

            const request_data = request_buf.items;

            const method_end = mem.indexOf(u8, request_data, " ") orelse return error.InvalidRequest;
            const method = request_data[0..method_end];

            const path_start = method_end + 1;
            const path_end = mem.indexOfPos(u8, request_data, path_start, " ") orelse return error.InvalidRequest;
            const path = request_data[path_start..path_end];

            const headers_end = mem.indexOf(u8, request_data, "\r\n\r\n") orelse return error.InvalidRequest;
            const headers = request_data[0..headers_end];

            var content_length: usize = 0;
            var header_lines = mem.splitSequence(u8, headers, "\r\n");
            while (header_lines.next()) |line| {
                if (line.len >= 16) {
                    var lower_buf: [64]u8 = undefined;
                    const prefix = line[0..@min(line.len, 15)];
                    var pi: usize = 0;
                    while (pi < prefix.len) : (pi += 1) {
                        lower_buf[pi] = std.ascii.toLower(prefix[pi]);
                    }
                    if (mem.eql(u8, lower_buf[0..15], "content-length:")) {
                        const colon_idx = mem.indexOf(u8, line, ":") orelse continue;
                        const val = mem.trim(u8, line[colon_idx + 1 ..], " \t");
                        content_length = std.fmt.parseInt(usize, val, 10) catch 0;
                    }
                }
            }

            const body_start = headers_end + 4;
            const body = if (body_start < request_data.len) request_data[body_start..] else "";

            if (body.len < content_length and content_length <= self.config.max_request_size_bytes) {
                var total_body_read: usize = body.len;
                while (total_body_read < content_length) {
                    const remaining = content_length - total_body_read;
                    const to_read = @min(remaining, tmp_buf.len);
                    const bytes_read = stream.read(tmp_buf[0..to_read]) catch break;
                    if (bytes_read == 0) break;
                    try request_buf.appendSlice(tmp_buf[0..bytes_read]);
                    total_body_read += bytes_read;
                    _ = request_buf.items[body_start..];
                }
            }

            var connection_keep_alive = false;
            var conn_header_lines = mem.splitSequence(u8, headers, "\r\n");
            while (conn_header_lines.next()) |line| {
                if (line.len >= 12) {
                    var lower_buf2: [64]u8 = undefined;
                    const prefix2 = line[0..@min(line.len, 11)];
                    var pi2: usize = 0;
                    while (pi2 < prefix2.len) : (pi2 += 1) {
                        lower_buf2[pi2] = std.ascii.toLower(prefix2[pi2]);
                    }
                    if (mem.eql(u8, lower_buf2[0..11], "connection:")) {
                        const colon_idx2 = mem.indexOf(u8, line, ":") orelse continue;
                        const val2 = mem.trim(u8, line[colon_idx2 + 1 ..], " \t");
                        var lower_val: [32]u8 = undefined;
                        var vi: usize = 0;
                        while (vi < @min(val2.len, 31)) : (vi += 1) {
                            lower_val[vi] = std.ascii.toLower(val2[vi]);
                        }
                        if (mem.indexOf(u8, lower_val[0..vi], "keep-alive") != null) {
                            connection_keep_alive = true;
                        }
                    }
                }
            }

            if (mem.eql(u8, method, "GET") and mem.eql(u8, path, "/v1/health")) {
                try self.handleHealth(stream, temp_allocator, connection_keep_alive);
            } else if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/v1/inference")) {
                const rate_allowed = self.rate_limiter.checkAndRecord(ip_str) catch false;
                if (!rate_allowed) {
                    try self.sendError(stream, "Rate limit exceeded", 429);
                    return;
                }

                if (self.api_key) |expected_key| {
                    const auth_valid = self.checkAuthorization(headers, expected_key);
                    if (!auth_valid) {
                        try self.sendError(stream, "Unauthorized - Invalid or missing API key", 401);
                        return;
                    }
                }

                try self.handleInference(stream, body, temp_allocator, connection_keep_alive);
            } else if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/v1/batch_inference")) {
                const rate_allowed = self.rate_limiter.checkAndRecord(ip_str) catch false;
                if (!rate_allowed) {
                    try self.sendError(stream, "Rate limit exceeded", 429);
                    return;
                }

                if (self.api_key) |expected_key| {
                    const auth_valid = self.checkAuthorization(headers, expected_key);
                    if (!auth_valid) {
                        try self.sendError(stream, "Unauthorized - Invalid or missing API key", 401);
                        return;
                    }
                }

                try self.handleBatchInference(stream, body, temp_allocator, connection_keep_alive);
            } else {
                try self.sendNotFound(stream);
                return;
            }

            if (!connection_keep_alive) return;
        }
    }

    fn checkAuthorization(self: *InferenceServer, headers: []const u8, expected_key: []const u8) bool {
        _ = self;

        var lines = mem.splitSequence(u8, headers, "\r\n");
        while (lines.next()) |line| {
            if (line.len < 14) continue;

            var lower_buf: [64]u8 = undefined;
            const prefix = line[0..@min(line.len, 13)];
            var pi: usize = 0;
            while (pi < prefix.len) : (pi += 1) {
                lower_buf[pi] = std.ascii.toLower(prefix[pi]);
            }

            if (mem.eql(u8, lower_buf[0..13], "authorization:")) {
                const value_start = mem.indexOf(u8, line, ":") orelse continue;
                const value = mem.trim(u8, line[value_start + 1 ..], " \t");

                if (value.len > 7) {
                    var prefix_lower: [6]u8 = undefined;
                    var bi: usize = 0;
                    while (bi < 6) : (bi += 1) {
                        prefix_lower[bi] = std.ascii.toLower(value[bi]);
                    }
                    if (mem.eql(u8, prefix_lower[0..6], "bearer") and (value[6] == ' ' or value[6] == '\t')) {
                        const token = mem.trim(u8, value[7..], " \t");
                        return mem.eql(u8, token, expected_key);
                    }
                }
            }
        }

        return false;
    }

    fn handleHealth(self: *InferenceServer, stream: net.Stream, allocator: Allocator, keep_alive: bool) !void {
        const start_ms = self.start_time;
        const now_ms = std.time.milliTimestamp();
        const uptime_ms = now_ms - start_ms;
        const uptime_seconds: u64 = if (uptime_ms >= 0) @intCast(@divTrunc(uptime_ms, 1000)) else 0;

        const response = HealthResponse{
            .uptime_seconds = uptime_seconds,
            .model_loaded = self.isModelLoaded(),
        };

        const json = try response.toJson(allocator);
        defer allocator.free(json);

        var response_buf = std.ArrayList(u8).init(allocator);
        defer response_buf.deinit();
        var writer = response_buf.writer();

        try writer.writeAll("HTTP/1.1 200 OK\r\n");
        try writer.writeAll("Content-Type: application/json\r\n");
        try writer.writeAll("Cache-Control: no-cache\r\n");
        try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
        if (keep_alive) {
            try writer.writeAll("Connection: keep-alive\r\n");
        } else {
            try writer.writeAll("Connection: close\r\n");
        }
        try writer.print("Content-Length: {d}\r\n", .{json.len});
        try writer.writeAll("\r\n");
        try writer.writeAll(json);

        _ = stream.writeAll(response_buf.items) catch {};
    }

    fn handleInference(self: *InferenceServer, stream: net.Stream, body: []const u8, allocator: Allocator, keep_alive: bool) !void {
        if (!self.isModelLoaded()) {
            try self.sendError(stream, "Model not loaded", 503);
            return;
        }

        const start_time = std.time.milliTimestamp();

        var request = InferenceRequest.fromJson(allocator, body) catch {
            try self.sendError(stream, "Invalid JSON request", 400);
            return;
        };
        defer request.deinit(allocator);

        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();
        self.model.?.mgt.?.encode(request.text, &tokens) catch {
            try self.sendError(stream, "Encoding failed", 500);
            return;
        };

        const max_tokens = request.max_tokens orelse tokens.items.len;
        const final_tokens = if (tokens.items.len > max_tokens)
            tokens.items[0..max_tokens]
        else
            tokens.items;

        var embeddings: ?[]f32 = null;
        errdefer if (embeddings) |emb| allocator.free(emb);

        self.inference_mutex.lock();
        defer self.inference_mutex.unlock();

        if (self.model.?.rsf != null) {
            const dim = (self.model.?.rsf.?.ctrl orelse {
                try self.sendError(stream, "RSF not initialized", 500);
                return;
            }).dim;

            var emb_tensor = if (self.embedding) |*emb|
                emb.forward(allocator, final_tokens, final_tokens.len) catch null
            else
                null;
            defer if (emb_tensor) |*t| t.deinit();

            var input_tensor = if (emb_tensor) |et| blk: {
                var t = try Tensor.init(allocator, &.{ 1, dim * 2 });
                const copy_len = @min(et.data.len, t.data.len);
                @memcpy(t.data[0..copy_len], et.data[0..copy_len]);
                if (copy_len < t.data.len) @memset(t.data[copy_len..], 0.0);
                break :blk t;
            } else blk: {
                var t = try Tensor.init(allocator, &.{ 1, dim * 2 });
                var k: usize = 0;
                while (k < t.data.len) : (k += 1) {
                    t.data[k] = if (k < final_tokens.len)
                        @as(f32, @floatFromInt(final_tokens[k])) / 1000.0
                    else
                        0.0;
                }
                break :blk t;
            };
            defer input_tensor.deinit();

            self.model.?.rsf.?.forward(&input_tensor) catch {
                try self.sendError(stream, "Forward pass failed", 500);
                return;
            };

            if (self.nsir_graph) |*graph| {
                const tensor_bytes = std.mem.sliceAsBytes(input_tensor.data);
                _ = graph.encodeInformation(tensor_bytes) catch {};
            }

            if (self.fractal_lpu) |*fl| {
                if (self.nsir_graph) |*graph| {
                    var node_iter = graph.nodes.iterator();
                    while (node_iter.next()) |entry| {
                        const node = entry.value_ptr;
                        var hasher = std.hash.Wyhash.init(0);
                        hasher.update(node.id);
                        const node_hash = hasher.final();
                        fl.mapNode(node_hash, 1.0) catch {};
                    }
                    fl.balanceAllTiles();
                }
            }

            if (self.r_gpu) |*rg| {
                if (self.nsir_graph) |*graph| {
                    rg.distributeGraph(graph) catch {};
                }
            }

            if (self.esso) |*esso_opt| {
                if (self.nsir_graph) |*graph| {
                    if (self.chaos_kernel) |*kernel| {
                        var orchestrator = ReasoningOrchestrator.init(self.allocator, graph, esso_opt, kernel);
                        defer orchestrator.deinit();
                        var reasoning_cycles: usize = 1;
                        if (std.posix.getenv("JAIDE_REASONING_CYCLES")) |cycles_str| {
                            reasoning_cycles = std.fmt.parseInt(usize, cycles_str, 10) catch 1;
                        }
                        _ = orchestrator.runHierarchicalReasoning(reasoning_cycles) catch {};
                    }
                }
            }

            if (self.surprise_memory) |*sm| {
                const tensor_bytes = std.mem.sliceAsBytes(input_tensor.data);
                _ = sm.storeWithSurprise(tensor_bytes, null) catch {};
            }

            if (self.temporal_graph) |*tg| {
                const now_ns: i64 = @truncate(std.time.nanoTimestamp());
                if (self.nsir_graph) |*graph| {
                    var node_iter = graph.nodes.iterator();
                    while (node_iter.next()) |entry| {
                        const node = entry.value_ptr;
                        const qs = QuantumState.init(
                            node.qubit.a.re,
                            node.qubit.a.im,
                            node.qubit.b.re,
                            node.qubit.b.im,
                            node.phase,
                            0.0,
                        );
                        tg.addNodeAtTime(node.id, qs, now_ns) catch {};
                    }
                }
                const after_ns: i64 = @truncate(std.time.nanoTimestamp());
                tg.advanceTime(after_ns - now_ns);
            }

            if (self.verifier) |v| {
                const output_buf = allocator.alloc(f32, input_tensor.data.len) catch null;
                if (output_buf) |obuf| {
                    defer allocator.free(obuf);
                    v.performVerifiedInference(input_tensor.data, obuf) catch {};
                }
            }

            if (self.signal_engine) |*se| {
                se.propagateStep() catch {};
            }

            if (self.z_runtime) |zr| {
                var name_buf: [64]u8 = undefined;
                const req_count = self.request_count.load(.monotonic);
                const var_name = std.fmt.bufPrint(&name_buf, "inf_{}", .{req_count}) catch "inf";
                _ = zr.createVariable(var_name, null) catch {};
            }

            if (self.vpu) |*v| {
                if (self.nsir_graph) |*graph| {
                    var graph_embeddings_opt = v.computeGraphEmbeddings(graph) catch null;
                    if (graph_embeddings_opt) |*embs| {
                        defer embs.deinit();
                        if (embs.items.len > 0) {
                            const req_count = self.request_count.load(.monotonic);
                            const theta: f64 = @as(f64, @floatFromInt(req_count % 314)) / 100.0;
                            const phi: f64 = @as(f64, @floatFromInt(req_count % 628)) / 100.0;
                            v.quantumVectorOps(embs.items, theta, phi);
                            if (embs.items.len >= 2) {
                                var sim_matrix_opt = v.computeSimilarityMatrix(embs.items) catch null;
                                if (sim_matrix_opt) |*sm| {
                                    defer {
                                        for (sm.items) |*row| row.deinit();
                                        sm.deinit();
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (self.fnds_manager) |*fm| {
                const tree_id_opt = fm.createTree(4, 3) catch null;
                if (tree_id_opt) |tid| {
                    const req_bytes = std.mem.sliceAsBytes(input_tensor.data);
                    var node_id_buf: [64]u8 = undefined;
                    const req_count = self.request_count.load(.monotonic);
                    const node_id = std.fmt.bufPrint(&node_id_buf, "inf_{d}", .{req_count}) catch "inf_node";
                    _ = fm.insertIntoTree(tid, node_id, req_bytes, 0) catch {};
                    fm.createIndex("inference_patterns") catch {};
                    const pattern_len = @min(req_bytes.len, 32);
                    if (pattern_len > 0) {
                        var loc = PatternLocation.init(
                            self.allocator,
                            tid,
                            0,
                            node_id,
                            0,
                            pattern_len,
                            1.0,
                        ) catch null;
                        if (loc) |*loc_val| {
                            var loc_added = false;
                            defer if (!loc_added) loc_val.deinit();
                            fm.addPatternToIndex("inference_patterns", req_bytes[0..pattern_len], loc_val.*) catch {};
                            loc_added = true;
                        }
                    }
                }
            }

            if (self.crev_pipeline) |*cp| {
                _ = cp.processTextStream(request.text) catch {};
            }

            if (self.formal_verifier) |fv| {
                if (self.nsir_graph) |*graph| {
                    _ = fv.verifyGraph(graph) catch {};
                }
            }

            if (self.security_engine) |se| {
                if (self.nsir_graph) |*graph| {
                    _ = se.proveInformationFlowSecurity(graph) catch {};
                }
            }

            if (self.quantum_adapter) |*qa| {
                var subgraphs_opt = qa.identifyQuantumSubgraphs() catch null;
                if (subgraphs_opt) |*sgs| {
                    defer {
                        for (sgs.items) |*sg| sg.deinit();
                        sgs.deinit();
                    }
                    for (sgs.items) |*sg| {
                        var task_result = qa.executeQuantumTask(sg) catch continue;
                        defer task_result.deinit();
                        if (task_result.success) {
                            qa.applyResultsToGraph(sg, &task_result) catch {};
                        }
                    }
                }
            }

            boostAboveMean(input_tensor.data);

            embeddings = try allocator.alloc(f32, @min(dim, 128));
            var m: usize = 0;
            while (m < embeddings.?.len) : (m += 1) {
                embeddings.?[m] = if (m < input_tensor.data.len) input_tensor.data[m] else 0.0;
            }
        }

        if (self.ssi) |*ssi_idx| {
            const is_anchor = (self.request_count.load(.monotonic) % 10 == 0);
            ssi_idx.addSequence(final_tokens, self.request_count.load(.monotonic), is_anchor) catch {};
        }
        _ = self.request_count.fetchAdd(1, .monotonic);

        var owned_embeddings: ?[]f32 = embeddings;
        var owned_input_tokens: ?[]u32 = null;
        var owned_generated_text: ?[]const u8 = null;
        var owned_generated_tokens: ?[]u32 = null;
        errdefer {
            if (owned_embeddings) |emb| allocator.free(emb);
            if (owned_input_tokens) |it| allocator.free(it);
            if (owned_generated_text) |gt| allocator.free(gt);
            if (owned_generated_tokens) |tk| allocator.free(tk);
        }

        const input_tokens_copy = try allocator.dupe(u32, final_tokens);
        owned_input_tokens = input_tokens_copy;

        var generated = std.ArrayList(u32).init(allocator);
        defer generated.deinit();
        try generated.appendSlice(final_tokens);

        const max_new_tokens = request.max_tokens orelse 32;
        var gen_step: usize = 0;
        while (gen_step < max_new_tokens) : (gen_step += 1) {
            var next_token: u32 = 0;

            if (self.model.?.rsf != null and self.embedding != null) {
                const dim = (self.model.?.rsf.?.ctrl orelse break).dim;
                const flow_width = dim * 2;

                const context_start = if (generated.items.len > 0) generated.items.len - 1 else 0;
                const context_tokens = generated.items[context_start..];

                var step_emb = self.embedding.?.forward(allocator, context_tokens, context_tokens.len) catch null;
                defer if (step_emb) |*t| t.deinit();

                if (step_emb) |et| {
                    var step_tensor = Tensor.init(allocator, &.{ 1, flow_width }) catch break;
                    defer step_tensor.deinit();

                    const row_start = if (et.data.len >= flow_width) et.data.len - flow_width else 0;
                    const source_row = et.data[row_start..];
                    const copy_len = @min(source_row.len, flow_width);
                    @memcpy(step_tensor.data[0..copy_len], source_row[0..copy_len]);
                    if (copy_len < flow_width) @memset(step_tensor.data[copy_len..], 0.0);

                    const forward_ok = if (self.model.?.rsf.?.forward(&step_tensor)) |_| true else |_| false;

                    if (forward_ok) {
                        if (self.nsir_graph) |*graph| {
                            const tensor_bytes = std.mem.sliceAsBytes(step_tensor.data);
                            _ = graph.encodeInformation(tensor_bytes) catch {};
                        }

                        next_token = self.selectTokenFromFlowState(step_tensor.data) catch 0;
                    }
                }
            }

            if (self.ssi) |*ssi_idx| {
                if (self.ranker) |*rnk| {
                    const top_candidates = rnk.topKHeap(ssi_idx, generated.items, 5, allocator) catch null;
                    if (top_candidates) |cands| {
                        defer {
                            for (cands) |*c| {
                                c.deinit(allocator);
                            }
                            allocator.free(cands);
                        }
                        if (cands.len > 0 and cands[0].tokens.len > 0) {
                            rnk.rankCandidatesWithQuery(cands, generated.items, ssi_idx, allocator) catch {};
                            next_token = cands[0].tokens[0];
                        }
                    }
                }
            }

            if (next_token == 0) {
                if (self.ssi) |*ssi_idx| {
                    if (generated.items.len > 0) {
                        const start_idx = if (generated.items.len > 4) generated.items.len - 4 else 0;
                        const recent = generated.items[start_idx..];
                        const candidates = ssi_idx.retrieveTopK(recent, 5, allocator) catch null;
                        if (candidates) |cands| {
                            defer {
                                for (cands) |*c| {
                                    c.deinit(allocator);
                                }
                                allocator.free(cands);
                            }
                            if (cands.len > 0) {
                                var best_score: f64 = -std.math.inf(f64);
                                var best_token: u32 = 0;
                                for (cands) |cand| {
                                    if (cand.tokens.len > 0 and cand.score > best_score) {
                                        best_score = cand.score;
                                        best_token = cand.tokens[0];
                                    }
                                }
                                if (best_token != 0) {
                                    next_token = best_token;
                                }
                            }
                        }
                    }
                }
            }

            if (next_token == 0) break;

            try generated.append(next_token);

            if (self.ssi) |*ssi_idx| {
                ssi_idx.addSequence(&[_]u32{next_token}, self.request_count.load(.monotonic), false) catch {};
            }
        }

        var generated_text: ?[]const u8 = null;
        if (generated.items.len > final_tokens.len) {
            const new_tokens = generated.items[final_tokens.len..];
            var text_buf = std.ArrayList(u8).init(allocator);
            self.model.?.mgt.?.decode(new_tokens, &text_buf) catch {};
            generated_text = try text_buf.toOwnedSlice();
        }
        owned_generated_text = generated_text;

        const generated_tokens_slice = if (generated.items.len > final_tokens.len)
            try allocator.dupe(u32, generated.items[final_tokens.len..])
        else
            try allocator.alloc(u32, 0);
        owned_generated_tokens = generated_tokens_slice;

        const end_time = std.time.milliTimestamp();
        const processing_time = @as(f64, @floatFromInt(end_time - start_time));

        var response = InferenceResponse{
            .tokens = generated_tokens_slice,
            .text = generated_text,
            .input_tokens = input_tokens_copy,
            .embeddings = embeddings,
            .processing_time_ms = processing_time,
        };
        owned_embeddings = null;
        owned_input_tokens = null;
        owned_generated_text = null;
        owned_generated_tokens = null;
        defer response.deinit(allocator);

        const json = try response.toJson(allocator);
        defer allocator.free(json);

        var response_buf = std.ArrayList(u8).init(allocator);
        defer response_buf.deinit();
        var writer = response_buf.writer();

        try writer.writeAll("HTTP/1.1 200 OK\r\n");
        try writer.writeAll("Content-Type: application/json\r\n");
        try writer.writeAll("Cache-Control: no-cache\r\n");
        try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
        if (keep_alive) {
            try writer.writeAll("Connection: keep-alive\r\n");
        } else {
            try writer.writeAll("Connection: close\r\n");
        }
        try writer.print("Content-Length: {d}\r\n", .{json.len});
        try writer.writeAll("\r\n");
        try writer.writeAll(json);

        _ = stream.writeAll(response_buf.items) catch {};
    }

    fn handleBatchInference(self: *InferenceServer, stream: net.Stream, body: []const u8, allocator: Allocator, keep_alive: bool) !void {
        if (!self.isModelLoaded()) {
            try self.sendError(stream, "Model not loaded", 503);
            return;
        }

        var batch_req = BatchInferenceRequest.fromJson(allocator, body) catch {
            try self.sendError(stream, "Invalid JSON request", 400);
            return;
        };
        defer batch_req.deinit(allocator);

        var results = std.ArrayList([]u8).init(allocator);
        defer {
            for (results.items) |r| {
                allocator.free(r);
            }
            results.deinit();
        }

        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();
        var writer = list.writer();

        try writer.writeAll("{\"results\":[");

        var ti: usize = 0;
        while (ti < batch_req.texts.len) : (ti += 1) {
            if (ti > 0) try writer.writeAll(",");

            const batch_start_time = std.time.milliTimestamp();

            var tokens = std.ArrayList(u32).init(allocator);
            defer tokens.deinit();
            self.model.?.mgt.?.encode(batch_req.texts[ti], &tokens) catch {
                try writer.writeAll("{\"error\":\"Encoding failed\"}");
                continue;
            };

            const max_gen_tokens = batch_req.max_tokens orelse 32;

            self.inference_mutex.lock();

            if (self.model.?.rsf != null) {
                const dim = (self.model.?.rsf.?.ctrl orelse {
                    self.inference_mutex.unlock();
                    try writer.writeAll("{\"error\":\"RSF not initialized\"}");
                    continue;
                }).dim;

                var emb_tensor = if (self.embedding) |*emb|
                    emb.forward(allocator, tokens.items, tokens.items.len) catch null
                else
                    null;
                defer if (emb_tensor) |*t| t.deinit();

                if (emb_tensor) |et| {
                    var input_tensor = blk: {
                        var t = Tensor.init(allocator, &.{ 1, dim * 2 }) catch break;
                        const copy_len = @min(et.data.len, t.data.len);
                        @memcpy(t.data[0..copy_len], et.data[0..copy_len]);
                        if (copy_len < t.data.len) @memset(t.data[copy_len..], 0.0);
                        break :blk t;
                    };
                    defer input_tensor.deinit();

                    self.model.?.rsf.?.forward(&input_tensor) catch {};

                    if (self.nsir_graph) |*graph| {
                        const tensor_bytes = std.mem.sliceAsBytes(input_tensor.data);
                        _ = graph.encodeInformation(tensor_bytes) catch {};
                    }

                    if (self.vpu) |*v| {
                        if (self.nsir_graph) |*graph| {
                            var graph_embeddings_opt = v.computeGraphEmbeddings(graph) catch null;
                            if (graph_embeddings_opt) |*embs| {
                                defer embs.deinit();
                                if (embs.items.len > 0) {
                                    const req_count = self.request_count.load(.monotonic);
                                    const theta: f64 = @as(f64, @floatFromInt(req_count % 314)) / 100.0;
                                    const phi: f64 = @as(f64, @floatFromInt(req_count % 628)) / 100.0;
                                    v.quantumVectorOps(embs.items, theta, phi);
                                    if (embs.items.len >= 2) {
                                        var sim_matrix_opt = v.computeSimilarityMatrix(embs.items) catch null;
                                        if (sim_matrix_opt) |*sm| {
                                            defer {
                                                for (sm.items) |*row| row.deinit();
                                                sm.deinit();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (self.fnds_manager) |*fm| {
                        const tree_id_opt = fm.createTree(4, 3) catch null;
                        if (tree_id_opt) |tid| {
                            const req_bytes = std.mem.sliceAsBytes(input_tensor.data);
                            var node_id_buf: [64]u8 = undefined;
                            const req_count = self.request_count.load(.monotonic);
                            const node_id = std.fmt.bufPrint(&node_id_buf, "batch_{d}_{d}", .{ req_count, ti }) catch "batch_node";
                            _ = fm.insertIntoTree(tid, node_id, req_bytes, 0) catch {};
                            fm.createIndex("batch_inference_patterns") catch {};
                            const pattern_len = @min(req_bytes.len, 32);
                            if (pattern_len > 0) {
                                var loc = PatternLocation.init(
                                    self.allocator,
                                    tid,
                                    0,
                                    node_id,
                                    0,
                                    pattern_len,
                                    1.0,
                                ) catch null;
                                if (loc) |*loc_val| {
                                    var loc_added = false;
                                    defer if (!loc_added) loc_val.deinit();
                                    fm.addPatternToIndex("batch_inference_patterns", req_bytes[0..pattern_len], loc_val.*) catch {};
                                    loc_added = true;
                                }
                            }
                        }
                    }

                    if (self.crev_pipeline) |*cp| {
                        _ = cp.processTextStream(batch_req.texts[ti]) catch {};
                    }

                    if (self.formal_verifier) |fv| {
                        if (self.nsir_graph) |*graph| {
                            _ = fv.verifyGraph(graph) catch {};
                        }
                    }

                    if (self.security_engine) |se| {
                        if (self.nsir_graph) |*graph| {
                            _ = se.proveInformationFlowSecurity(graph) catch {};
                        }
                    }

                    if (self.quantum_adapter) |*qa| {
                        var subgraphs_opt = qa.identifyQuantumSubgraphs() catch null;
                        if (subgraphs_opt) |*sgs| {
                            defer {
                                for (sgs.items) |*sg| sg.deinit();
                                sgs.deinit();
                            }
                            for (sgs.items) |*sg| {
                                var task_result = qa.executeQuantumTask(sg) catch continue;
                                defer task_result.deinit();
                                if (task_result.success) {
                                    qa.applyResultsToGraph(sg, &task_result) catch {};
                                }
                            }
                        }
                    }
                }
            }

            var generated = std.ArrayList(u32).init(allocator);
            defer generated.deinit();
            try generated.appendSlice(tokens.items);

            var gen_step: usize = 0;
            while (gen_step < max_gen_tokens) : (gen_step += 1) {
                var next_token: u32 = 0;

                if (self.model.?.rsf != null and self.embedding != null) {
                    const dim = (self.model.?.rsf.?.ctrl orelse break).dim;
                    const flow_width = dim * 2;

                    const context_start = if (generated.items.len > 0) generated.items.len - 1 else 0;
                    const context_tokens = generated.items[context_start..];

                    var step_emb = self.embedding.?.forward(allocator, context_tokens, context_tokens.len) catch null;
                    defer if (step_emb) |*t| t.deinit();

                    if (step_emb) |et| {
                        var step_tensor = Tensor.init(allocator, &.{ 1, flow_width }) catch break;
                        defer step_tensor.deinit();

                        const row_start = if (et.data.len >= flow_width) et.data.len - flow_width else 0;
                        const source_row = et.data[row_start..];
                        const copy_len = @min(source_row.len, flow_width);
                        @memcpy(step_tensor.data[0..copy_len], source_row[0..copy_len]);
                        if (copy_len < flow_width) @memset(step_tensor.data[copy_len..], 0.0);

                        const forward_ok = if (self.model.?.rsf.?.forward(&step_tensor)) |_| true else |_| false;

                        if (forward_ok) {
                            if (self.nsir_graph) |*graph| {
                                const tensor_bytes = std.mem.sliceAsBytes(step_tensor.data);
                                _ = graph.encodeInformation(tensor_bytes) catch {};
                            }

                            next_token = self.selectTokenFromFlowState(step_tensor.data) catch 0;
                        }
                    }
                }

                if (self.ssi) |*ssi_idx| {
                    if (self.ranker) |*rnk| {
                        const top_candidates = rnk.topKHeap(ssi_idx, generated.items, 5, allocator) catch null;
                        if (top_candidates) |cands| {
                            defer {
                                for (cands) |*c| {
                                    c.deinit(allocator);
                                }
                                allocator.free(cands);
                            }
                            if (cands.len > 0 and cands[0].tokens.len > 0) {
                                rnk.rankCandidatesWithQuery(cands, generated.items, ssi_idx, allocator) catch {};
                                next_token = cands[0].tokens[0];
                            }
                        }
                    }
                }

                if (next_token == 0) {
                    if (self.ssi) |*ssi_idx| {
                        if (generated.items.len > 0) {
                            const start_idx = if (generated.items.len > 4) generated.items.len - 4 else 0;
                            const recent = generated.items[start_idx..];
                            const candidates = ssi_idx.retrieveTopK(recent, 5, allocator) catch null;
                            if (candidates) |cands| {
                                defer {
                                    for (cands) |*c| {
                                        c.deinit(allocator);
                                    }
                                    allocator.free(cands);
                                }
                                if (cands.len > 0) {
                                    var best_score: f64 = -std.math.inf(f64);
                                    var best_token: u32 = 0;
                                    for (cands) |cand| {
                                        if (cand.tokens.len > 0 and cand.score > best_score) {
                                            best_score = cand.score;
                                            best_token = cand.tokens[0];
                                        }
                                    }
                                    if (best_token != 0) {
                                        next_token = best_token;
                                    }
                                }
                            }
                        }
                    }
                }

                if (next_token == 0) break;
                try generated.append(next_token);

                if (self.ssi) |*ssi_idx| {
                    ssi_idx.addSequence(&[_]u32{next_token}, self.request_count.load(.monotonic), false) catch {};
                }
            }

            self.inference_mutex.unlock();

            _ = self.request_count.fetchAdd(1, .monotonic);

            var generated_text: ?[]const u8 = null;
            if (generated.items.len > tokens.items.len) {
                const new_tokens = generated.items[tokens.items.len..];
                var text_buf = std.ArrayList(u8).init(allocator);
                self.model.?.mgt.?.decode(new_tokens, &text_buf) catch {};
                generated_text = try text_buf.toOwnedSlice();
            }
            defer if (generated_text) |gt| allocator.free(gt);

            const batch_end_time = std.time.milliTimestamp();
            const processing_time = @as(f64, @floatFromInt(batch_end_time - batch_start_time));

            try writer.writeAll("{\"tokens\":[");
            var gi: usize = 0;
            while (gi < generated.items.len) : (gi += 1) {
                if (gi > 0) try writer.writeAll(",");
                try writer.print("{d}", .{generated.items[gi]});
            }
            try writer.writeAll("]");
            if (generated_text) |gt| {
                try writer.writeAll(",\"text\":\"");
                var ci: usize = 0;
                while (ci < gt.len) : (ci += 1) {
                    const c = gt[ci];
                    switch (c) {
                        '"' => try writer.writeAll("\\\""),
                        '\\' => try writer.writeAll("\\\\"),
                        '\n' => try writer.writeAll("\\n"),
                        '\r' => try writer.writeAll("\\r"),
                        '\t' => try writer.writeAll("\\t"),
                        else => {
                            if (c >= 0x20 and c < 0x7f) {
                                try writer.writeByte(c);
                            }
                        },
                    }
                }
                try writer.writeAll("\"");
            }
            try writer.print(",\"processing_time_ms\":{d:.2}", .{processing_time});
            try writer.writeByte('}');
        }

        try writer.writeAll("]}");

        const json = try list.toOwnedSlice();
        defer allocator.free(json);

        var response_buf = std.ArrayList(u8).init(allocator);
        defer response_buf.deinit();
        var rwriter = response_buf.writer();

        try rwriter.writeAll("HTTP/1.1 200 OK\r\n");
        try rwriter.writeAll("Content-Type: application/json\r\n");
        try rwriter.writeAll("Cache-Control: no-cache\r\n");
        try rwriter.writeAll("Access-Control-Allow-Origin: *\r\n");
        if (keep_alive) {
            try rwriter.writeAll("Connection: keep-alive\r\n");
        } else {
            try rwriter.writeAll("Connection: close\r\n");
        }
        try rwriter.print("Content-Length: {d}\r\n", .{json.len});
        try rwriter.writeAll("\r\n");
        try rwriter.writeAll(json);

        _ = stream.writeAll(response_buf.items) catch {};
    }

    fn selectTokenFromFlowState(self: *InferenceServer, flow_state: []const f32) !u32 {
        if (self.embedding == null) return 0;
        const source = &self.embedding.?;
        const width = source.dim;
        if (width == 0 or flow_state.len < width) return 0;
        if (source.vocab_size <= RESERVED_TOKEN_COUNT) return 0;

        const state_row = flow_state[0..width];
        var state_norm_sq: f64 = 0.0;
        for (state_row) |value| {
            if (!std.math.isFinite(value)) return 0;
            state_norm_sq += @as(f64, value) * @as(f64, value);
        }
        if (!(state_norm_sq > 0.0)) return 0;
        const state_norm = @sqrt(state_norm_sq);

        var best_token: u32 = 0;
        var best_similarity: f64 = -std.math.inf(f64);
        var found_candidate = false;

        var token_index: usize = RESERVED_TOKEN_COUNT;
        while (token_index < source.vocab_size) : (token_index += 1) {
            const row_start = token_index * width;
            const row_end = row_start + width;
            if (row_end > source.weight.data.len) break;
            const candidate_row = source.weight.data[row_start..row_end];

            var dot: f64 = 0.0;
            var candidate_norm_sq: f64 = 0.0;
            var row_finite = true;
            for (candidate_row, 0..) |value, position| {
                if (!std.math.isFinite(value)) {
                    row_finite = false;
                    break;
                }
                const promoted: f64 = @floatCast(value);
                dot += promoted * @as(f64, state_row[position]);
                candidate_norm_sq += promoted * promoted;
            }
            if (!row_finite) continue;
            if (!(candidate_norm_sq > 0.0)) continue;

            const similarity = dot / (state_norm * @sqrt(candidate_norm_sq));
            if (!std.math.isFinite(similarity)) continue;
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_token = @intCast(token_index);
                found_candidate = true;
            }
        }

        if (!found_candidate) return 0;
        return best_token;
    }

    fn sendError(self: *InferenceServer, stream: net.Stream, message: []const u8, status_code: u16) !void {
        var json_list = std.ArrayList(u8).init(self.allocator);
        defer json_list.deinit();
        var jwriter = json_list.writer();

        try jwriter.writeAll("{\"error\":\"");
        var mi: usize = 0;
        while (mi < message.len) : (mi += 1) {
            const c = message[mi];
            switch (c) {
                '"' => try jwriter.writeAll("\\\""),
                '\\' => try jwriter.writeAll("\\\\"),
                else => try jwriter.writeByte(c),
            }
        }
        try jwriter.writeAll("\"}");

        const json = json_list.items;

        const status_text = switch (status_code) {
            400 => "Bad Request",
            401 => "Unauthorized",
            404 => "Not Found",
            413 => "Payload Too Large",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            else => "Error",
        };

        var resp_list = std.ArrayList(u8).init(self.allocator);
        defer resp_list.deinit();
        var rwriter = resp_list.writer();

        try rwriter.writeAll("HTTP/1.1 ");
        try rwriter.print("{d} {s}\r\n", .{ status_code, status_text });
        try rwriter.writeAll("Content-Type: application/json\r\n");
        try rwriter.writeAll("Cache-Control: no-cache\r\n");
        try rwriter.writeAll("Access-Control-Allow-Origin: *\r\n");
        try rwriter.writeAll("Connection: close\r\n");
        try rwriter.print("Content-Length: {d}\r\n", .{json.len});
        try rwriter.writeAll("\r\n");
        try rwriter.writeAll(json);

        _ = stream.writeAll(resp_list.items) catch {};
    }

    fn sendNotFound(self: *InferenceServer, stream: net.Stream) !void {
        try self.sendError(stream, "Endpoint not found", 404);
    }
};

pub const BatchInferenceRequest = struct {
    texts: [][]const u8,
    max_tokens: ?usize = null,
    return_embeddings: bool = false,

    pub fn fromJson(allocator: Allocator, json: []const u8) !BatchInferenceRequest {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return error.InvalidJson;
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.InvalidJson;

        const texts_array = root.object.get("texts") orelse return error.MissingTextsField;
        if (texts_array != .array) return error.InvalidTextsField;

        var texts = try allocator.alloc([]const u8, texts_array.array.items.len);
        var n: usize = 0;
        errdefer {
            var ci: usize = 0;
            while (ci < n) : (ci += 1) {
                allocator.free(texts[ci]);
            }
            allocator.free(texts);
        }
        while (n < texts_array.array.items.len) : (n += 1) {
            if (texts_array.array.items[n] != .string) {
                return error.InvalidTextsField;
            }
            texts[n] = try allocator.dupe(u8, texts_array.array.items[n].string);
        }

        var max_tokens: ?usize = null;
        if (root.object.get("max_tokens")) |mt| {
            if (mt == .integer) {
                if (mt.integer < 0) return error.InvalidMaxTokens;
                if (mt.integer > 1000000) return error.MaxTokensTooLarge;
                max_tokens = @intCast(mt.integer);
            }
        }

        var return_embeddings = false;
        if (root.object.get("return_embeddings")) |re| {
            if (re == .bool) {
                return_embeddings = re.bool;
            }
        }

        return BatchInferenceRequest{
            .texts = texts,
            .max_tokens = max_tokens,
            .return_embeddings = return_embeddings,
        };
    }

    pub fn deinit(self: *BatchInferenceRequest, allocator: Allocator) void {
        for (self.texts) |text| {
            allocator.free(text);
        }
        allocator.free(self.texts);
    }
};
