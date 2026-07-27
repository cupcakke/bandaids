const std = @import("std");
const http = std.http;

pub const IBMQuantumClient = struct {
    allocator: std.mem.Allocator,
    api_token: []const u8,
    crn: []const u8,
    http_client: http.Client,
    owns_crn: bool,
    backend_name: []const u8,
    owns_backend_name: bool,

    pub fn init(allocator: std.mem.Allocator, api_token: []const u8) !IBMQuantumClient {
        return initWithCrn(allocator, api_token, null, null);
    }

    pub fn initWithBackend(allocator: std.mem.Allocator, api_token: []const u8, backend: []const u8) !IBMQuantumClient {
        return initWithCrn(allocator, api_token, null, backend);
    }

    pub fn initWithCrn(allocator: std.mem.Allocator, api_token: []const u8, crn_override: ?[]const u8, backend_override: ?[]const u8) !IBMQuantumClient {
        const crn = if (crn_override) |c|
            try allocator.dupe(u8, c)
        else if (std.posix.getenv("IBM_QUANTUM_CRN")) |env_crn|
            try allocator.dupe(u8, env_crn)
        else
            return error.MissingIBMQuantumCRN;

        const backend = if (backend_override) |b|
            try allocator.dupe(u8, b)
        else if (std.posix.getenv("IBM_QUANTUM_BACKEND")) |env_backend|
            try allocator.dupe(u8, env_backend)
        else
            try allocator.dupe(u8, "ibm_brisbane");

        return .{
            .allocator = allocator,
            .api_token = try allocator.dupe(u8, api_token),
            .crn = crn,
            .http_client = http.Client{ .allocator = allocator },
            .owns_crn = true,
            .backend_name = backend,
            .owns_backend_name = true,
        };
    }

    pub fn setBackendName(self: *IBMQuantumClient, name: []const u8) !void {
        if (self.owns_backend_name) {
            self.allocator.free(self.backend_name);
        }
        self.backend_name = try self.allocator.dupe(u8, name);
        self.owns_backend_name = true;
    }

    pub fn deinit(self: *IBMQuantumClient) void {
        self.zeroSensitiveData();
        self.allocator.free(self.api_token);
        if (self.owns_crn) {
            self.allocator.free(self.crn);
        }
        if (self.owns_backend_name) {
            self.allocator.free(self.backend_name);
        }
        self.http_client.deinit();
    }

    fn escapeForJson(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
        var buf = std.ArrayList(u8).init(allocator);
        errdefer buf.deinit();
        for (input) |ch| {
            switch (ch) {
                '"' => {
                    try buf.appendSlice("\\\"");
                },
                '\\' => {
                    try buf.appendSlice("\\\\");
                },
                '\n' => {
                    try buf.appendSlice("\\n");
                },
                '\r' => {
                    try buf.appendSlice("\\r");
                },
                '\t' => {
                    try buf.appendSlice("\\t");
                },
                '\x08' => {
                    try buf.appendSlice("\\b");
                },
                '\x0C' => {
                    try buf.appendSlice("\\f");
                },
                else => {
                    if (ch < 0x20) {
                        var fmt_buf: [8]u8 = undefined;
                        const seq = std.fmt.bufPrint(&fmt_buf, "\\u{X:0>4}", .{ch}) catch unreachable;
                        try buf.appendSlice(seq);
                    } else {
                        try buf.append(ch);
                    }
                },
            }
        }
        return try buf.toOwnedSlice();
    }

    pub fn submitJob(self: *IBMQuantumClient, qasm: []const u8) ![]const u8 {
        return self.submitJobWithBackend(qasm, self.backend_name, 1024);
    }

    pub fn submitJobWithBackend(self: *IBMQuantumClient, qasm: []const u8, backend: []const u8, shots: u32) ![]const u8 {
        const uri = try std.Uri.parse("https://cloud.ibm.com/quantum/api/v1/jobs");

        var auth_buf: [4096]u8 = undefined;
        const auth_value = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{self.api_token}) catch return error.OutOfMemory;

        const escaped_qasm = try escapeForJson(self.allocator, qasm);
        defer self.allocator.free(escaped_qasm);

        const escaped_backend = try escapeForJson(self.allocator, backend);
        defer self.allocator.free(escaped_backend);

        const payload = try std.fmt.allocPrint(self.allocator,
            \\{{"qasm": "{s}", "backend": "{s}", "shots": {d}}}
        , .{ escaped_qasm, escaped_backend, shots });
        defer self.allocator.free(payload);

        var server_header_buffer: [16 * 1024]u8 = undefined;
        var req = try self.http_client.open(.POST, uri, .{
            .server_header_buffer = &server_header_buffer,
            .extra_headers = &[_]http.Header{
                .{ .name = "Authorization", .value = auth_value },
                .{ .name = "Content-Type", .value = "application/json" },
            },
        });
        defer req.deinit();

        req.transfer_encoding = .chunked;
        try req.send();
        try req.writeAll(payload);
        try req.finish();
        try req.wait();

        const body = try req.reader().readAllAlloc(self.allocator, 1024 * 1024);
        return body;
    }

    pub fn getJobResult(self: *IBMQuantumClient, job_id: []const u8) ![]const u8 {
        const uri_str = try std.fmt.allocPrint(self.allocator, "https://cloud.ibm.com/quantum/api/v1/jobs/{s}", .{job_id});
        defer self.allocator.free(uri_str);

        const uri = try std.Uri.parse(uri_str);

        var auth_buf: [4096]u8 = undefined;
        const auth_value = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{self.api_token}) catch return error.OutOfMemory;

        var server_header_buffer: [16 * 1024]u8 = undefined;
        var req = try self.http_client.open(.GET, uri, .{
            .server_header_buffer = &server_header_buffer,
            .extra_headers = &[_]http.Header{
                .{ .name = "Authorization", .value = auth_value },
            },
        });
        defer req.deinit();

        try req.send();
        try req.finish();
        try req.wait();

        const body = try req.reader().readAllAlloc(self.allocator, 1024 * 1024);
        return body;
    }

    pub fn zeroSensitiveData(self: *IBMQuantumClient) void {
        if (self.api_token.len > 0) {
            @memset(@constCast(self.api_token), 0);
        }
        if (self.crn.len > 0) {
            @memset(@constCast(self.crn), 0);
        }
    }
};

pub const QuantumTaskResult = struct {
    subgraph_id: u64,
    success: bool,
    quantum_states: std.ArrayList(std.math.Complex(f64)),
    correlations: std.ArrayList(f64),
    execution_time_ms: i64,
    backend_name: ?[]const u8,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, subgraph_id: u64) Self {
        return Self{
            .subgraph_id = subgraph_id,
            .success = false,
            .quantum_states = std.ArrayList(std.math.Complex(f64)).init(allocator),
            .correlations = std.ArrayList(f64).init(allocator),
            .execution_time_ms = 0,
            .backend_name = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.quantum_states.deinit();
        self.correlations.deinit();
        if (self.backend_name) |bn| {
            self.allocator.free(bn);
            self.backend_name = null;
        }
    }

    pub fn setBackendName(self: *Self, name: []const u8) !void {
        if (self.backend_name) |bn| {
            self.allocator.free(bn);
        }
        self.backend_name = try self.allocator.dupe(u8, name);
    }
};
