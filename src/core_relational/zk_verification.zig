const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const crypto = std.crypto;
const Blake3 = crypto.hash.Blake3;
const Sha256 = crypto.hash.sha2.Sha256;
const Sha512 = crypto.hash.sha2.Sha512;
const ChildProcess = std.process.Child;
const fs = std.fs;
const json = std.json;
const @"i256" = std.math.Int(256);

pub const ZKProofError = error{
    CircomCompilationFailed,
    WitnessGenerationFailed,
    ProofGenerationFailed,
    VerificationFailed,
    FileNotFound,
    InvalidProofFormat,
    InvalidWitnessFormat,
    SnarkjsNotFound,
    CircuitNotCompiled,
    KeysNotGenerated,
    OutOfMemory,
    ProcessSpawnFailed,
    Timeout,
    InvalidInput,
    InvalidOutput,
    ValueOutOfRange,
    EmptySet,
    InvalidIndex,
    InsufficientParticipants,
    DimensionMismatch,
};

pub const ZKCircuitConfig = struct {
    circuit_path: []const u8,
    wasm_path: []const u8,
    zkey_path: []const u8,
    vkey_path: []const u8,
    witness_dir: []const u8,
    proof_dir: []const u8,
    num_layers: usize,
    embedding_dim: usize,
    precision_bits: usize,
    timeout_ms: u64,

    pub fn defaultConfig(allocator: Allocator) !*ZKCircuitConfig {
        const config = try allocator.create(ZKCircuitConfig);
        config.* = ZKCircuitConfig{
            .circuit_path = "src/zk/inference_trace.circom",
            .wasm_path = "src/zk/inference_trace_js/inference_trace.wasm",
            .zkey_path = "src/zk/inference_trace.zkey",
            .vkey_path = "src/zk/verification_key.json",
            .witness_dir = "src/zk/witness",
            .proof_dir = "src/zk/proofs",
            .num_layers = 8,
            .embedding_dim = 32,
            .precision_bits = 64,
            .timeout_ms = 300000,
        };
        return config;
    }
};

pub const Groth16Proof = struct {
    pi_a: [3][96]u8,
    pi_b: [3][2][96]u8,
    pi_c: [3][96]u8,
    protocol: [7]u8,
    curve: [5]u8,

    pub fn init() Groth16Proof {
        var proof = std.mem.zeroes(Groth16Proof);
        proof.protocol = .{ 'g', 'r', 'o', 't', 'h', '1', '6' };
        proof.curve = .{ 'b', 'n', '1', '2', '8' };
        return proof;
    }
};

pub const PublicSignals = struct {
    allocator: Allocator,
    signals: ArrayList(i256),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .signals = ArrayList(i256).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.signals.deinit();
    }

    pub fn addSignal(self: *Self, value: i256) !void {
        try self.signals.append(value);
    }
};

pub const ZKProofBundle = struct {
    allocator: Allocator,
    proof: Groth16Proof,
    public_signals: PublicSignals,
    proof_json: []u8,
    public_json: []u8,
    timestamp: i64,
    verification_status: bool,

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .proof = Groth16Proof.init(),
            .public_signals = PublicSignals.init(allocator),
            .proof_json = &[_]u8{},
            .public_json = &[_]u8{},
            .timestamp = @truncate(std.time.nanoTimestamp()),
            .verification_status = false,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.public_signals.deinit();
        if (self.proof_json.len > 0) {
            self.allocator.free(self.proof_json);
        }
        if (self.public_json.len > 0) {
            self.allocator.free(self.public_json);
        }
        self.allocator.destroy(self);
    }
};

pub const CircomProver = struct {
    allocator: Allocator,
    config: *ZKCircuitConfig,
    circuit_compiled: bool,
    keys_generated: bool,
    snarkjs_path: []const u8,
    node_path: []const u8,

    const Self = @This();

    pub fn init(allocator: Allocator, config: *ZKCircuitConfig) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .config = config,
            .circuit_compiled = false,
            .keys_generated = false,
            .snarkjs_path = "npx",
            .node_path = "node",
        };
        try self.ensureDirectories();
        try self.checkPrerequisites();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    fn ensureDirectories(self: *Self) !void {
        try fs.cwd().makePath(self.config.witness_dir);
        try fs.cwd().makePath(self.config.proof_dir);
    }

    fn checkPrerequisites(self: *Self) !void {
        const wasm_file = fs.cwd().openFile(self.config.wasm_path, .{}) catch {
            self.circuit_compiled = false;
            return;
        };
        wasm_file.close();
        self.circuit_compiled = true;

        const zkey_file = fs.cwd().openFile(self.config.zkey_path, .{}) catch {
            self.keys_generated = false;
            return;
        };
        zkey_file.close();
        self.keys_generated = true;
    }

    pub fn compileCircuit(self: *Self) !void {
        const output_dir = try std.fmt.allocPrint(self.allocator, "src/zk", .{});
        defer self.allocator.free(output_dir);

        const args = &[_][]const u8{
            "circom",
            self.config.circuit_path,
            "--r1cs",
            "--wasm",
            "--sym",
            "-o",
            output_dir,
        };

        var child = ChildProcess.init(args, self.allocator);
        child.cwd = null;

        try child.spawn();
        const term = try child.wait();

        switch (term) {
            .Exited => |code| if (code != 0) return ZKProofError.CircomCompilationFailed,
            else => return ZKProofError.CircomCompilationFailed,
        }

        self.circuit_compiled = true;
    }

    pub fn setupKeys(self: *Self, ptau_path: []const u8) !void {
        const r1cs_path = try std.fmt.allocPrint(self.allocator, "src/zk/inference_trace.r1cs", .{});
        defer self.allocator.free(r1cs_path);

        const contribute_args = &[_][]const u8{
            self.snarkjs_path,
            "snarkjs",
            "groth16",
            "setup",
            r1cs_path,
            ptau_path,
            self.config.zkey_path,
        };

        var child = ChildProcess.init(contribute_args, self.allocator);
        child.cwd = null;

        try child.spawn();
        const term = try child.wait();

        switch (term) {
            .Exited => |code| if (code != 0) return ZKProofError.KeysNotGenerated,
            else => return ZKProofError.KeysNotGenerated,
        }

        const export_args = &[_][]const u8{
            self.snarkjs_path,
            "snarkjs",
            "zkey",
            "export",
            "verificationkey",
            self.config.zkey_path,
            self.config.vkey_path,
        };

        var export_child = ChildProcess.init(export_args, self.allocator);
        export_child.cwd = null;

        try export_child.spawn();
        const export_term = try export_child.wait();

        switch (export_term) {
            .Exited => |code| if (code != 0) return ZKProofError.KeysNotGenerated,
            else => return ZKProofError.KeysNotGenerated,
        }

        self.keys_generated = true;
    }

    pub fn generateWitness(self: *Self, input_json_path: []const u8, witness_path: []const u8) !void {
        if (!self.circuit_compiled) {
            return ZKProofError.CircuitNotCompiled;
        }

        const witness_gen_path = try std.fmt.allocPrint(
            self.allocator,
            "src/zk/inference_trace_js/generate_witness.js",
            .{},
        );
        defer self.allocator.free(witness_gen_path);

        const args = &[_][]const u8{
            self.node_path,
            witness_gen_path,
            self.config.wasm_path,
            input_json_path,
            witness_path,
        };

        var child = ChildProcess.init(args, self.allocator);
        child.cwd = null;

        try child.spawn();
        const term = try child.wait();

        switch (term) {
            .Exited => |code| if (code != 0) return ZKProofError.WitnessGenerationFailed,
            else => return ZKProofError.WitnessGenerationFailed,
        }
    }

    pub fn generateProof(self: *Self, witness_path: []const u8, proof_path: []const u8, public_path: []const u8) !void {
        if (!self.keys_generated) {
            return ZKProofError.KeysNotGenerated;
        }

        const args = &[_][]const u8{
            self.snarkjs_path,
            "snarkjs",
            "groth16",
            "prove",
            self.config.zkey_path,
            witness_path,
            proof_path,
            public_path,
        };

        var child = ChildProcess.init(args, self.allocator);
        child.cwd = null;

        try child.spawn();
        const term = try child.wait();

        switch (term) {
            .Exited => |code| if (code != 0) return ZKProofError.ProofGenerationFailed,
            else => return ZKProofError.ProofGenerationFailed,
        }
    }

    pub fn verifyProof(self: *Self, proof_path: []const u8, public_path: []const u8) !bool {
        const args = &[_][]const u8{
            self.snarkjs_path,
            "snarkjs",
            "groth16",
            "verify",
            self.config.vkey_path,
            public_path,
            proof_path,
        };

        var child = ChildProcess.init(args, self.allocator);
        child.cwd = null;
        child.stdout_behavior = .Pipe;

        try child.spawn();

        var stdout_buffer: [4096]u8 = undefined;
        const stdout_bytes = try child.stdout.?.readAll(stdout_buffer[0..]);
        const stdout = stdout_buffer[0..stdout_bytes];

        const term = try child.wait();

        switch (term) {
            .Exited => |code| if (code != 0) return false,
            else => return false,
        }

        return std.mem.indexOf(u8, stdout, "OK") != null;
    }
};

pub const InferenceWitness = struct {
    allocator: Allocator,
    tokens: []i64,
    layer_weights_s: [][][]i64,
    layer_weights_t: [][][]i64,
    expected_output: []i64,
    input_commitment: i256,
    output_commitment: i256,
    layer_commitments: []i256,
    max_error_squared: i64,
    num_layers: usize,
    dim: usize,

    const Self = @This();

    pub fn init(allocator: Allocator, num_layers: usize, dim: usize) !*Self {
        const self = try allocator.create(Self);

        const tokens = blk: { const _s = try allocator.alloc(i64, dim); @memset(_s, 0); break :blk _s; };
        const expected_output = blk: { const _s = try allocator.alloc(i64, dim); @memset(_s, 0); break :blk _s; };
        const layer_commitments = blk: { const _s = try allocator.alloc(i256, num_layers); @memset(_s, 0); break :blk _s; };

        const layer_weights_s = try allocator.alloc([][]i64, num_layers);
        const layer_weights_t = try allocator.alloc([][]i64, num_layers);

        errdefer {
            allocator.free(layer_weights_t);
            allocator.free(layer_weights_s);
            allocator.free(layer_commitments);
            allocator.free(expected_output);
            allocator.free(tokens);
            allocator.destroy(self);
        }

        var layer_idx: usize = 0;
        while (layer_idx < num_layers) : (layer_idx += 1) {
            layer_weights_s[layer_idx] = try allocator.alloc([]i64, dim);
            layer_weights_t[layer_idx] = try allocator.alloc([]i64, dim);

            var i: usize = 0;
            while (i < dim) : (i += 1) {
                layer_weights_s[layer_idx][i] = blk: { const _s = try allocator.alloc(i64, dim); @memset(_s, 0); break :blk _s; };
                layer_weights_t[layer_idx][i] = blk: { const _s = try allocator.alloc(i64, dim); @memset(_s, 0); break :blk _s; };
            }
        }

        self.* = Self{
            .allocator = allocator,
            .tokens = tokens,
            .layer_weights_s = layer_weights_s,
            .layer_weights_t = layer_weights_t,
            .expected_output = expected_output,
            .input_commitment = 0,
            .output_commitment = 0,
            .layer_commitments = layer_commitments,
            .max_error_squared = 1000000,
            .num_layers = num_layers,
            .dim = dim,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        var layer_idx: usize = 0;
        while (layer_idx < self.num_layers) : (layer_idx += 1) {
            var i: usize = 0;
            while (i < self.dim) : (i += 1) {
                self.allocator.free(self.layer_weights_s[layer_idx][i]);
                self.allocator.free(self.layer_weights_t[layer_idx][i]);
            }
            self.allocator.free(self.layer_weights_s[layer_idx]);
            self.allocator.free(self.layer_weights_t[layer_idx]);
        }
        self.allocator.free(self.layer_weights_s);
        self.allocator.free(self.layer_weights_t);

        self.allocator.free(self.layer_commitments);
        self.allocator.free(self.expected_output);
        self.allocator.free(self.tokens);

        self.allocator.destroy(self);
    }

    pub fn setTokens(self: *Self, input_tokens: []const f32, scale: i128) void {
        var i: usize = 0;
        while (i < self.dim and i < input_tokens.len) : (i += 1) {
            const scaled = @as(f64, input_tokens[i]) * @as(f64, @floatFromInt(scale));
            if (scaled >= @as(f64, @floatFromInt(std.math.maxInt(i64)))) {
                self.tokens[i] = std.math.maxInt(i64);
            } else if (scaled <= @as(f64, @floatFromInt(std.math.minInt(i64)))) {
                self.tokens[i] = std.math.minInt(i64);
            } else {
                self.tokens[i] = @intFromFloat(scaled);
            }
        }
    }

    pub fn setExpectedOutput(self: *Self, output: []const f32, scale: i128) void {
        var i: usize = 0;
        while (i < self.dim and i < output.len) : (i += 1) {
            const scaled = @as(f64, output[i]) * @as(f64, @floatFromInt(scale));
            if (scaled >= @as(f64, @floatFromInt(std.math.maxInt(i64)))) {
                self.expected_output[i] = std.math.maxInt(i64);
            } else if (scaled <= @as(f64, @floatFromInt(std.math.minInt(i64)))) {
                self.expected_output[i] = std.math.minInt(i64);
            } else {
                self.expected_output[i] = @intFromFloat(scaled);
            }
        }
    }

    pub fn setLayerWeights(self: *Self, layer: usize, weights_s: []const []const f32, weights_t: []const []const f32, scale: i128) void {
        if (layer >= self.num_layers) return;

        var i: usize = 0;
        while (i < self.dim and i < weights_s.len) : (i += 1) {
            var j: usize = 0;
            while (j < self.dim and j < weights_s[i].len) : (j += 1) {
                const scaled = @as(f64, weights_s[i][j]) * @as(f64, @floatFromInt(scale));
                if (scaled >= @as(f64, @floatFromInt(std.math.maxInt(i64)))) {
                    self.layer_weights_s[layer][i][j] = std.math.maxInt(i64);
                } else if (scaled <= @as(f64, @floatFromInt(std.math.minInt(i64)))) {
                    self.layer_weights_s[layer][i][j] = std.math.minInt(i64);
                } else {
                    self.layer_weights_s[layer][i][j] = @intFromFloat(scaled);
                }
            }
        }

        i = 0;
        while (i < self.dim and i < weights_t.len) : (i += 1) {
            var j: usize = 0;
            while (j < self.dim and j < weights_t[i].len) : (j += 1) {
                const scaled = @as(f64, weights_t[i][j]) * @as(f64, @floatFromInt(scale));
                if (scaled >= @as(f64, @floatFromInt(std.math.maxInt(i64)))) {
                    self.layer_weights_t[layer][i][j] = std.math.maxInt(i64);
                } else if (scaled <= @as(f64, @floatFromInt(std.math.minInt(i64)))) {
                    self.layer_weights_t[layer][i][j] = std.math.minInt(i64);
                } else {
                    self.layer_weights_t[layer][i][j] = @intFromFloat(scaled);
                }
            }
        }
    }

    pub fn computeCommitments(self: *Self) void {
        var input_hasher = Blake3.init(.{});
        for (self.tokens) |token| {
            const bytes = std.mem.asBytes(&token);
            input_hasher.update(bytes);
        }
        var input_hash: [32]u8 = undefined;
        input_hasher.final(&input_hash);
        self.input_commitment = bytesToI256(input_hash[0..]);

        var output_hasher = Blake3.init(.{});
        for (self.expected_output) |out| {
            const bytes = std.mem.asBytes(&out);
            output_hasher.update(bytes);
        }
        var output_hash: [32]u8 = undefined;
        output_hasher.final(&output_hash);
        self.output_commitment = bytesToI256(output_hash[0..]);

        var layer_idx: usize = 0;
        while (layer_idx < self.num_layers) : (layer_idx += 1) {
            var layer_hasher = Blake3.init(.{});
            var i: usize = 0;
            while (i < self.dim) : (i += 1) {
                var j: usize = 0;
                while (j < self.dim) : (j += 1) {
                    const s_bytes = std.mem.asBytes(&self.layer_weights_s[layer_idx][i][j]);
                    const t_bytes = std.mem.asBytes(&self.layer_weights_t[layer_idx][i][j]);
                    layer_hasher.update(s_bytes);
                    layer_hasher.update(t_bytes);
                }
            }
            var layer_hash: [32]u8 = undefined;
            layer_hasher.final(&layer_hash);
            self.layer_commitments[layer_idx] = bytesToI256(layer_hash[0..]);
        }
    }

    fn bytesToI256(bytes: []const u8) i256 {
        var buf: [32]u8 = undefined;
        const len = @min(bytes.len, 32);
        @memcpy(buf[0..len], bytes[0..len]);
        if (len < 32) @memset(buf[len..], 0);
        return std.mem.readInt(i256, &buf, .big);
    }

    pub fn toJson(self: *Self, path: []const u8) !void {
        var file = try fs.cwd().createFile(path, .{});
        defer file.close();

        var writer = file.writer();

        try writer.writeAll("{\n");

        try writer.writeAll("  \"tokens\": [");
        var i: usize = 0;
        while (i < self.dim) : (i += 1) {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{self.tokens[i]});
        }
        try writer.writeAll("],\n");

        try writer.writeAll("  \"layer_weights_s\": [\n");
        var layer: usize = 0;
        while (layer < self.num_layers) : (layer += 1) {
            try writer.writeAll("    [\n");
            i = 0;
            while (i < self.dim) : (i += 1) {
                try writer.writeAll("      [");
                var j: usize = 0;
                while (j < self.dim) : (j += 1) {
                    if (j > 0) try writer.writeAll(", ");
                    try writer.print("{d}", .{self.layer_weights_s[layer][i][j]});
                }
                try writer.writeAll("]");
                if (i < self.dim - 1) try writer.writeAll(",");
                try writer.writeAll("\n");
            }
            try writer.writeAll("    ]");
            if (layer < self.num_layers - 1) try writer.writeAll(",");
            try writer.writeAll("\n");
        }
        try writer.writeAll("  ],\n");

        try writer.writeAll("  \"layer_weights_t\": [\n");
        layer = 0;
        while (layer < self.num_layers) : (layer += 1) {
            try writer.writeAll("    [\n");
            i = 0;
            while (i < self.dim) : (i += 1) {
                try writer.writeAll("      [");
                var j: usize = 0;
                while (j < self.dim) : (j += 1) {
                    if (j > 0) try writer.writeAll(", ");
                    try writer.print("{d}", .{self.layer_weights_t[layer][i][j]});
                }
                try writer.writeAll("]");
                if (i < self.dim - 1) try writer.writeAll(",");
                try writer.writeAll("\n");
            }
            try writer.writeAll("    ]");
            if (layer < self.num_layers - 1) try writer.writeAll(",");
            try writer.writeAll("\n");
        }
        try writer.writeAll("  ],\n");

        try writer.writeAll("  \"expected_output\": [");
        i = 0;
        while (i < self.dim) : (i += 1) {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{self.expected_output[i]});
        }
        try writer.writeAll("],\n");

        try writer.print("  \"input_commitment\": \"{d}\",\n", .{self.input_commitment});
        try writer.print("  \"output_commitment\": \"{d}\",\n", .{self.output_commitment});

        try writer.writeAll("  \"layer_commitments\": [");
        layer = 0;
        while (layer < self.num_layers) : (layer += 1) {
            if (layer > 0) try writer.writeAll(", ");
            try writer.print("\"{d}\"", .{self.layer_commitments[layer]});
        }
        try writer.writeAll("],\n");

        try writer.print("  \"max_error_squared\": {d}\n", .{self.max_error_squared});
        try writer.writeAll("}\n");
    }
};

pub const ZKInferenceProver = struct {
    allocator: Allocator,
    config: *ZKCircuitConfig,
    prover: *CircomProver,
    proof_counter: u64,

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const config = try ZKCircuitConfig.defaultConfig(allocator);
        const prover = try CircomProver.init(allocator, config);

        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .config = config,
            .prover = prover,
            .proof_counter = 0,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.prover.deinit();
        self.allocator.destroy(self.config);
        self.allocator.destroy(self);
    }

    pub fn proveInference(
        self: *Self,
        input: []const f32,
        output: []const f32,
        layer_weights_s: []const []const []const f32,
        layer_weights_t: []const []const []const f32,
    ) !*ZKProofBundle {
        const witness = try InferenceWitness.init(
            self.allocator,
            self.config.num_layers,
            self.config.embedding_dim,
        );
        defer witness.deinit();

        const scale: i64 = 1000000;
        witness.setTokens(input, scale);
        witness.setExpectedOutput(output, scale);

        var layer: usize = 0;
        while (layer < self.config.num_layers and layer < layer_weights_s.len) : (layer += 1) {
            witness.setLayerWeights(layer, layer_weights_s[layer], layer_weights_t[layer], scale);
        }

        witness.computeCommitments();

        var nonce_bytes: [16]u8 = undefined;
        crypto.random.bytes(&nonce_bytes);
        const nonce_hex = std.fmt.bytesToHex(nonce_bytes, .lower);

        const input_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/input_{s}.json",
            .{ self.config.witness_dir, nonce_hex },
        );
        defer self.allocator.free(input_path);

        try witness.toJson(input_path);

        const witness_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/witness_{s}.wtns",
            .{ self.config.witness_dir, nonce_hex },
        );
        defer self.allocator.free(witness_path);

        try self.prover.generateWitness(input_path, witness_path);

        const proof_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/proof_{s}.json",
            .{ self.config.proof_dir, nonce_hex },
        );
        defer self.allocator.free(proof_path);

        const public_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/public_{s}.json",
            .{ self.config.proof_dir, nonce_hex },
        );
        defer self.allocator.free(public_path);

        try self.prover.generateProof(witness_path, proof_path, public_path);

        const bundle = try ZKProofBundle.init(self.allocator);

        const proof_file = try fs.cwd().openFile(proof_path, .{});
        defer proof_file.close();
        bundle.proof_json = try proof_file.readToEndAlloc(self.allocator, 1 * 1024 * 1024);

        const public_file = try fs.cwd().openFile(public_path, .{});
        defer public_file.close();
        bundle.public_json = try public_file.readToEndAlloc(self.allocator, 1 * 1024 * 1024);

        bundle.verification_status = try self.prover.verifyProof(proof_path, public_path);

        self.proof_counter += 1;

        return bundle;
    }

    pub fn verifyProofBundle(self: *Self, bundle: *ZKProofBundle) !bool {
        var random_bytes: [16]u8 = undefined;
        crypto.random.bytes(&random_bytes);
        const nonce = std.fmt.bytesToHex(random_bytes, .lower);

        var tmp_dir_buf: [256]u8 = undefined;
        const tmp_dir_path = try std.fmt.bufPrint(&tmp_dir_buf, "/tmp/zk_verify_{s}", .{nonce});
        try fs.cwd().makePath(tmp_dir_path);

        const proof_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/proof.json",
            .{tmp_dir_path},
        );
        defer self.allocator.free(proof_path);

        const public_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/public.json",
            .{tmp_dir_path},
        );
        defer self.allocator.free(public_path);

        var proof_file = try fs.cwd().createFile(proof_path, .{ .exclusive = true });
        defer proof_file.close();
        try proof_file.writeAll(bundle.proof_json);

        var public_file = try fs.cwd().createFile(public_path, .{ .exclusive = true });
        defer public_file.close();
        try public_file.writeAll(bundle.public_json);

        const result = try self.prover.verifyProof(proof_path, public_path);

        fs.cwd().deleteFile(proof_path) catch {};
        fs.cwd().deleteFile(public_path) catch {};
        fs.cwd().deleteDir(tmp_dir_path) catch {};

        return result;
    }
};

pub const CommitmentScheme = struct {
    allocator: Allocator,
    commitments: AutoHashMap([32]u8, Commitment),
    nonce_counter: u64,

    const Self = @This();

    pub const Commitment = struct {
        value_hash: [32]u8,
        nonce: [32]u8,
        timestamp: i64,
        blinding_factor: [32]u8,
    };

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .commitments = AutoHashMap([32]u8, Commitment).init(allocator),
            .nonce_counter = 0,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.commitments.deinit();
        self.allocator.destroy(self);
    }

    pub fn commit(self: *Self, value: []const u8) ![32]u8 {
        var nonce: [32]u8 = undefined;
        crypto.random.bytes(nonce[0..]);

        var blinding: [32]u8 = undefined;
        crypto.random.bytes(blinding[0..]);

        var hasher = Blake3.init(.{});
        hasher.update(value);
        hasher.update(&nonce);
        hasher.update(&blinding);
        var commitment_hash: [32]u8 = undefined;
        hasher.final(&commitment_hash);

        var value_hasher = Sha256.init(.{});
        value_hasher.update(value);
        var value_hash: [32]u8 = undefined;
        value_hasher.final(&value_hash);

        const commitment = Commitment{
            .value_hash = value_hash,
            .nonce = nonce,
            .timestamp = @truncate(std.time.nanoTimestamp()),
            .blinding_factor = blinding,
        };

        try self.commitments.put(commitment_hash, commitment);
        self.nonce_counter += 1;

        return commitment_hash;
    }

    pub fn verify(self: *Self, commitment_hash: [32]u8, revealed_value: []const u8, revealed_nonce: [32]u8, revealed_blinding: [32]u8) !bool {
        const commitment = self.commitments.get(commitment_hash) orelse return false;

        var hasher = Blake3.init(.{});
        hasher.update(revealed_value);
        hasher.update(&revealed_nonce);
        hasher.update(&revealed_blinding);
        var computed_commitment: [32]u8 = undefined;
        hasher.final(&computed_commitment);

        if (!std.mem.eql(u8, &commitment_hash, &computed_commitment)) {
            return false;
        }

        var value_hasher = Sha256.init(.{});
        value_hasher.update(revealed_value);
        var computed_value_hash: [32]u8 = undefined;
        value_hasher.final(&computed_value_hash);

        return std.mem.eql(u8, &commitment.value_hash, &computed_value_hash);
    }
};

pub const RangeProof = struct {
    allocator: Allocator,
    min_value: i128,
    max_value: i128,
    proof_bits: ArrayList(ProofBit),

    const ProofBit = struct {
        commitment: [32]u8,
        opening: [32]u8,
        bit_value: u1,
    };

    const Self = @This();

    pub fn init(allocator: Allocator, min: i128, max: i128) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .min_value = min,
            .max_value = max,
            .proof_bits = ArrayList(ProofBit).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.proof_bits.deinit();
        self.allocator.destroy(self);
    }

    pub fn prove(self: *Self, value: i128) !void {
        if (self.min_value > self.max_value) {
            return error.ValueOutOfRange;
        }
        if (value < self.min_value or value > self.max_value) {
            return error.ValueOutOfRange;
        }

        const range = @as(u128, @intCast(self.max_value)) - @as(u128, @intCast(self.min_value));
        const bits_needed: usize = if (range == 0) 1 else blk: {
            const bits = std.math.bitSizeOf(u128) - @clz(range);
            break :blk @min(bits, 127);
        };

        const normalized_value = @as(u128, @intCast(value)) - @as(u128, @intCast(self.min_value));

        var i: usize = 0;
        while (i < bits_needed) : (i += 1) {
            const bit: u1 = @intCast((normalized_value >> @intCast(i)) & 1);

            var nonce: [32]u8 = undefined;
            crypto.random.bytes(nonce[0..]);

            var hasher = Blake3.init(.{});
            hasher.update(&[_]u8{bit});
            hasher.update(&nonce);
            var commitment: [32]u8 = undefined;
            hasher.final(&commitment);

            try self.proof_bits.append(ProofBit{
                .commitment = commitment,
                .opening = nonce,
                .bit_value = bit,
            });
        }
    }

    pub fn verify(self: *Self) !bool {
        var reconstructed_value: u128 = 0;
        var i: usize = 0;
        while (i < self.proof_bits.items.len) : (i += 1) {
            const bit_proof = self.proof_bits.items[i];
            var hasher = Blake3.init(.{});
            hasher.update(&[_]u8{bit_proof.bit_value});
            hasher.update(&bit_proof.opening);
            var computed_commitment: [32]u8 = undefined;
            hasher.final(&computed_commitment);

            if (!std.mem.eql(u8, &bit_proof.commitment, &computed_commitment)) {
                return false;
            }

            if (bit_proof.bit_value > 1) {
                return false;
            }

            if (bit_proof.bit_value == 1) {
                if (i < 127) {
                    reconstructed_value |= (@as(u128, 1) << @intCast(i));
                }
            }
        }

        const final_value = @as(i128, @intCast(reconstructed_value)) + self.min_value;
        return final_value >= self.min_value and final_value <= self.max_value;
    }
};

pub const MembershipProof = struct {
    allocator: Allocator,
    merkle_root: [32]u8,
    path: ArrayList([32]u8),
    directions: ArrayList(bool),

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .merkle_root = undefined,
            .path = ArrayList([32]u8).init(allocator),
            .directions = ArrayList(bool).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.path.deinit();
        self.directions.deinit();
        self.allocator.destroy(self);
    }

    pub fn buildMerkleTree(self: *Self, elements: []const []const u8) ![32]u8 {
        if (elements.len == 0) return error.EmptySet;

        var current_level = ArrayList([32]u8).init(self.allocator);
        defer current_level.deinit();

        for (elements) |elem| {
            var hasher = Sha256.init(.{});
            hasher.update(elem);
            var hash: [32]u8 = undefined;
            hasher.final(&hash);
            try current_level.append(hash);
        }

        while (current_level.items.len > 1) {
            var next_level = ArrayList([32]u8).init(self.allocator);
            errdefer next_level.deinit();

            var i: usize = 0;
            while (i < current_level.items.len) : (i += 2) {
                var hasher = Sha256.init(.{});
                hasher.update(&current_level.items[i]);

                if (i + 1 < current_level.items.len) {
                    hasher.update(&current_level.items[i + 1]);
                } else {
                    hasher.update(&current_level.items[i]);
                }

                var hash: [32]u8 = undefined;
                hasher.final(&hash);
                try next_level.append(hash);
            }

            current_level.clearRetainingCapacity();
            try current_level.appendSlice(next_level.items);
            next_level.deinit();
        }

        self.merkle_root = current_level.items[0];
        return self.merkle_root;
    }

    pub fn generateProof(self: *Self, elements: []const []const u8, index: usize) !void {
        if (index >= elements.len) return error.InvalidIndex;

        var current_level = ArrayList([32]u8).init(self.allocator);
        defer current_level.deinit();

        for (elements) |elem| {
            var hasher = Sha256.init(.{});
            hasher.update(elem);
            var hash: [32]u8 = undefined;
            hasher.final(&hash);
            try current_level.append(hash);
        }

        var current_index = index;

        while (current_level.items.len > 1) {
            var sibling_index: usize = undefined;
            var direction: bool = undefined;

            if (current_index % 2 == 0) {
                sibling_index = current_index + 1;
                direction = false;
            } else {
                sibling_index = current_index - 1;
                direction = true;
            }

            if (sibling_index < current_level.items.len) {
                try self.path.append(current_level.items[sibling_index]);
            } else {
                try self.path.append(current_level.items[current_index]);
            }
            try self.directions.append(direction);

            var next_level = ArrayList([32]u8).init(self.allocator);
            errdefer next_level.deinit();

            var i: usize = 0;
            while (i < current_level.items.len) : (i += 2) {
                var hasher = Sha256.init(.{});
                hasher.update(&current_level.items[i]);

                if (i + 1 < current_level.items.len) {
                    hasher.update(&current_level.items[i + 1]);
                } else {
                    hasher.update(&current_level.items[i]);
                }

                var hash: [32]u8 = undefined;
                hasher.final(&hash);
                try next_level.append(hash);
            }

            current_level.clearRetainingCapacity();
            try current_level.appendSlice(next_level.items);
            next_level.deinit();
            current_index = current_index / 2;
        }
    }

    pub fn verify(self: *Self, element: []const u8) !bool {
        var hasher = Sha256.init(.{});
        hasher.update(element);
        var current_hash: [32]u8 = undefined;
        hasher.final(&current_hash);

        var i: usize = 0;
        while (i < self.path.items.len) : (i += 1) {
            const sibling = self.path.items[i];
            var next_hasher = Sha256.init(.{});

            if (self.directions.items[i]) {
                next_hasher.update(&sibling);
                next_hasher.update(&current_hash);
            } else {
                next_hasher.update(&current_hash);
                next_hasher.update(&sibling);
            }

            next_hasher.final(&current_hash);
        }

        return std.mem.eql(u8, &current_hash, &self.merkle_root);
    }
};

pub const SchnorrSignature = struct {
    challenge: [32]u8,
    response: [32]u8,

    const Self = @This();

    pub fn sign(message: []const u8, private_key: [32]u8) !Self {
        var k: [32]u8 = undefined;
        crypto.random.bytes(k[0..]);

        var r_point: [32]u8 = undefined;
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            r_point[i] = k[i] ^ private_key[i];
        }

        var challenge_hasher = Sha256.init(.{});
        challenge_hasher.update(&r_point);
        challenge_hasher.update(message);
        var challenge: [32]u8 = undefined;
        challenge_hasher.final(&challenge);

        var response: [32]u8 = undefined;
        i = 0;
        while (i < 32) : (i += 1) {
            response[i] = k[i] +% (challenge[i] *% private_key[i]);
        }

        return Self{
            .challenge = challenge,
            .response = response,
        };
    }

    pub fn verify(self: *Self, message: []const u8, public_key: [32]u8) bool {
        var r_point: [32]u8 = undefined;
        var i: usize = 0;
        while (i < 32) : (i += 1) {
            r_point[i] = self.response[i] -% (self.challenge[i] *% public_key[i]);
        }

        var challenge_hasher = Sha256.init(.{});
        challenge_hasher.update(&r_point);
        challenge_hasher.update(message);
        var computed_challenge: [32]u8 = undefined;
        challenge_hasher.final(&computed_challenge);

        return std.mem.eql(u8, &self.challenge, &computed_challenge);
    }
};

pub const DifferentialPrivacy = struct {
    allocator: Allocator,
    epsilon: f64,
    delta: f64,
    sensitivity: f64,
    noise_scale: f64,

    const Self = @This();

    pub fn init(allocator: Allocator, epsilon: f64, delta: f64, sensitivity: f64) !*Self {
        const self = try allocator.create(Self);
        const noise_scale = sensitivity * @sqrt(2.0 * @log(1.25 / delta)) / epsilon;
        self.* = Self{
            .allocator = allocator,
            .epsilon = epsilon,
            .delta = delta,
            .sensitivity = sensitivity,
            .noise_scale = noise_scale,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn addNoise(self: *Self, value: f64) f64 {
        var rand1: f64 = 0.0;
        while (rand1 == 0.0) {
            rand1 = std.crypto.random.float(f64);
        }
        const rand2 = std.crypto.random.float(f64);

        const z = @sqrt(-2.0 * @log(rand1)) * @cos(2.0 * std.math.pi * rand2);
        const noise = z * self.noise_scale;

        return value + noise;
    }

    pub fn addLaplaceNoise(self: *Self, value: f64) f64 {
        const u = std.crypto.random.float(f64) - 0.5;
        const b = self.sensitivity / self.epsilon;
        const sgn = if (u < 0) @as(f64, -1.0) else @as(f64, 1.0);
        const abs_u = @abs(u);
        const noise = -b * sgn * @log(1.0 - 2.0 * abs_u);
        return value + noise;
    }

    pub fn addCalibratedNoise(self: *Self, value: f64) f64 {
        const calibrated_scale = self.sensitivity / self.epsilon;
        var rand1: f64 = 0.0;
        while (rand1 == 0.0) {
            rand1 = std.crypto.random.float(f64);
        }
        const rand2 = std.crypto.random.float(f64);
        const z = @sqrt(-2.0 * @log(rand1)) * @cos(2.0 * std.math.pi * rand2);
        const noise = z * calibrated_scale;
        return value + noise;
    }
};

pub const ZKInferenceProof = struct {
    allocator: Allocator,
    input_commitment: [32]u8,
    output_commitment: [32]u8,
    computation_proof: ArrayList([32]u8),
    timestamp: i64,
    zk_prover: ?*ZKInferenceProver,
    proof_bundle: ?*ZKProofBundle,

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .input_commitment = undefined,
            .output_commitment = undefined,
            .computation_proof = ArrayList([32]u8).init(allocator),
            .timestamp = @truncate(std.time.nanoTimestamp()),
            .zk_prover = null,
            .proof_bundle = null,
        };
        return self;
    }

    pub fn initWithProver(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .input_commitment = undefined,
            .output_commitment = undefined,
            .computation_proof = ArrayList([32]u8).init(allocator),
            .timestamp = @truncate(std.time.nanoTimestamp()),
            .zk_prover = try ZKInferenceProver.init(allocator),
            .proof_bundle = null,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.computation_proof.deinit();
        if (self.proof_bundle) |bundle| {
            bundle.deinit();
        }
        if (self.zk_prover) |prover| {
            prover.deinit();
        }
        self.allocator.destroy(self);
    }

    pub fn proveInference(self: *Self, input: []const f32, output: []const f32, model_hash: [32]u8) !void {
        var input_hasher = Blake3.init(.{});
        for (input) |val| {
            const bytes = std.mem.asBytes(&val);
            input_hasher.update(bytes);
        }
        input_hasher.final(&self.input_commitment);

        var output_hasher = Blake3.init(.{});
        for (output) |val| {
            const bytes = std.mem.asBytes(&val);
            output_hasher.update(bytes);
        }
        output_hasher.final(&self.output_commitment);

        var step_hasher = Blake3.init(.{});
        step_hasher.update(&self.input_commitment);
        step_hasher.update(&model_hash);
        step_hasher.update(&self.output_commitment);
        var step_hash: [32]u8 = undefined;
        step_hasher.final(&step_hash);
        try self.computation_proof.append(step_hash);

        var i: usize = 0;
        while (i < 8) : (i += 1) {
            var intermediate_hasher = Blake3.init(.{});
            intermediate_hasher.update(&step_hash);
            const layer_index_bytes = std.mem.asBytes(&i);
            intermediate_hasher.update(layer_index_bytes);
            var intermediate_hash: [32]u8 = undefined;
            intermediate_hasher.final(&intermediate_hash);
            try self.computation_proof.append(intermediate_hash);
            step_hash = intermediate_hash;
        }
    }

    pub fn proveInferenceWithZK(
        self: *Self,
        input: []const f32,
        output: []const f32,
        layer_weights_s: []const []const []const f32,
        layer_weights_t: []const []const []const f32,
    ) !void {
        if (self.zk_prover) |prover| {
            self.proof_bundle = try prover.proveInference(input, output, layer_weights_s, layer_weights_t);

            var input_hasher = Blake3.init(.{});
            for (input) |val| {
                const bytes = std.mem.asBytes(&val);
                input_hasher.update(bytes);
            }
            input_hasher.final(&self.input_commitment);

            var output_hasher = Blake3.init(.{});
            for (output) |val| {
                const bytes = std.mem.asBytes(&val);
                output_hasher.update(bytes);
            }
            output_hasher.final(&self.output_commitment);
        } else {
            return ZKProofError.SnarkjsNotFound;
        }
    }

    pub fn verify(self: *Self, model_hash: [32]u8) !bool {
        if (self.proof_bundle) |bundle| {
            return bundle.verification_status;
        }

        if (self.computation_proof.items.len < 9) {
            return false;
        }

        var first_hasher = Blake3.init(.{});
        first_hasher.update(&self.input_commitment);
        first_hasher.update(&model_hash);
        first_hasher.update(&self.output_commitment);
        var expected_first: [32]u8 = undefined;
        first_hasher.final(&expected_first);

        if (!std.mem.eql(u8, &self.computation_proof.items[0], &expected_first)) {
            return false;
        }

        var previous_hash = self.computation_proof.items[0];
        var i: usize = 1;
        while (i < self.computation_proof.items.len) : (i += 1) {
            var hasher = Blake3.init(.{});
            hasher.update(&previous_hash);
            const layer_index = i - 1;
            const layer_index_bytes = std.mem.asBytes(&layer_index);
            hasher.update(layer_index_bytes);
            var expected_hash: [32]u8 = undefined;
            hasher.final(&expected_hash);

            if (!std.mem.eql(u8, &self.computation_proof.items[i], &expected_hash)) {
                return false;
            }

            previous_hash = self.computation_proof.items[i];
        }

        return true;
    }

    pub fn verifyWithZK(self: *Self) !bool {
        if (self.zk_prover) |prover| {
            if (self.proof_bundle) |bundle| {
                return try prover.verifyProofBundle(bundle);
            }
        }
        return false;
    }
};

pub const SecureAggregation = struct {
    allocator: Allocator,
    participant_commitments: AutoHashMap(u64, [32]u8),
    aggregated_result: ?[]f64,
    threshold: usize,

    const Self = @This();

    pub fn init(allocator: Allocator, threshold: usize) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .participant_commitments = AutoHashMap(u64, [32]u8).init(allocator),
            .aggregated_result = null,
            .threshold = threshold,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.aggregated_result) |result| {
            self.allocator.free(result);
        }
        self.participant_commitments.deinit();
        self.allocator.destroy(self);
    }

    pub fn commitParticipant(self: *Self, participant_id: u64, data: []const f64) ![32]u8 {
        var hasher = Blake3.init(.{});
        for (data) |val| {
            const bytes = std.mem.asBytes(&val);
            hasher.update(bytes);
        }
        var commitment: [32]u8 = undefined;
        hasher.final(&commitment);

        try self.participant_commitments.put(participant_id, commitment);
        return commitment;
    }

    pub fn aggregate(self: *Self, contributions: []const []const f64) !void {
        if (contributions.len < self.threshold) {
            return error.InsufficientParticipants;
        }

        if (contributions.len == 0) {
            if (self.aggregated_result) |old| self.allocator.free(old);
            self.aggregated_result = try self.allocator.alloc(f64, 0);
            return;
        }

        const dim = contributions[0].len;
        const result = try self.allocator.alloc(f64, dim);
        @memset(result, 0.0);
        errdefer self.allocator.free(result);

        for (contributions) |contrib| {
            if (contrib.len != dim) {
                self.allocator.free(result);
                return error.DimensionMismatch;
            }
            var j: usize = 0;
            while (j < contrib.len) : (j += 1) {
                result[j] += contrib[j];
            }
        }

        const count: f64 = @floatFromInt(contributions.len);
        if (count > 0) {
            for (result) |*val| {
                val.* /= count;
            }
        }

        if (self.aggregated_result) |old_result| {
            self.allocator.free(old_result);
        }
        self.aggregated_result = result;
    }

    pub fn getResult(self: *Self) ?[]const f64 {
        return self.aggregated_result;
    }
};

