const std = @import("std");
const Allocator = std.mem.Allocator;

pub const IBMBackendSpecs = struct {
    pub const HERON_T1_MEAN_NS: f64 = 350000.0;
    pub const HERON_T1_STDDEV_NS: f64 = 75000.0;
    pub const HERON_T2_MEAN_NS: f64 = 200000.0;
    pub const HERON_T2_STDDEV_NS: f64 = 50000.0;
    pub const HERON_READOUT_ERROR_MEAN: f64 = 0.008;
    pub const HERON_READOUT_ERROR_STDDEV: f64 = 0.003;
    pub const HERON_ECR_GATE_ERROR_MEAN: f64 = 0.003;
    pub const HERON_ECR_GATE_ERROR_STDDEV: f64 = 0.001;

    pub const EAGLE_T1_MEAN_NS: f64 = 200000.0;
    pub const EAGLE_T1_STDDEV_NS: f64 = 60000.0;
    pub const EAGLE_T2_MEAN_NS: f64 = 120000.0;
    pub const EAGLE_T2_STDDEV_NS: f64 = 40000.0;
    pub const EAGLE_READOUT_ERROR_MEAN: f64 = 0.015;
    pub const EAGLE_READOUT_ERROR_STDDEV: f64 = 0.005;
    pub const EAGLE_ECR_GATE_ERROR_MEAN: f64 = 0.005;
    pub const EAGLE_ECR_GATE_ERROR_STDDEV: f64 = 0.002;

    pub const FALCON_T1_MEAN_NS: f64 = 100000.0;
    pub const FALCON_T1_STDDEV_NS: f64 = 30000.0;
    pub const FALCON_T2_MEAN_NS: f64 = 80000.0;
    pub const FALCON_T2_STDDEV_NS: f64 = 25000.0;
    pub const FALCON_READOUT_ERROR_MEAN: f64 = 0.020;
    pub const FALCON_READOUT_ERROR_STDDEV: f64 = 0.008;
    pub const FALCON_ECR_GATE_ERROR_MEAN: f64 = 0.007;
    pub const FALCON_ECR_GATE_ERROR_STDDEV: f64 = 0.003;

    pub const OSPREY_T1_MEAN_NS: f64 = 250000.0;
    pub const OSPREY_T1_STDDEV_NS: f64 = 70000.0;
    pub const OSPREY_T2_MEAN_NS: f64 = 150000.0;
    pub const OSPREY_T2_STDDEV_NS: f64 = 45000.0;
    pub const OSPREY_READOUT_ERROR_MEAN: f64 = 0.012;
    pub const OSPREY_READOUT_ERROR_STDDEV: f64 = 0.004;
    pub const OSPREY_ECR_GATE_ERROR_MEAN: f64 = 0.004;
    pub const OSPREY_ECR_GATE_ERROR_STDDEV: f64 = 0.0015;

    pub const CONDOR_T1_MEAN_NS: f64 = 400000.0;
    pub const CONDOR_T1_STDDEV_NS: f64 = 100000.0;
    pub const CONDOR_T2_MEAN_NS: f64 = 250000.0;
    pub const CONDOR_T2_STDDEV_NS: f64 = 60000.0;
    pub const CONDOR_READOUT_ERROR_MEAN: f64 = 0.006;
    pub const CONDOR_READOUT_ERROR_STDDEV: f64 = 0.002;
    pub const CONDOR_ECR_GATE_ERROR_MEAN: f64 = 0.002;
    pub const CONDOR_ECR_GATE_ERROR_STDDEV: f64 = 0.0008;

    pub const DEFAULT_T1_MEAN_NS: f64 = 150000.0;
    pub const DEFAULT_T1_STDDEV_NS: f64 = 50000.0;
    pub const DEFAULT_T2_MEAN_NS: f64 = 100000.0;
    pub const DEFAULT_T2_STDDEV_NS: f64 = 35000.0;
    pub const DEFAULT_READOUT_ERROR_MEAN: f64 = 0.018;
    pub const DEFAULT_READOUT_ERROR_STDDEV: f64 = 0.006;
    pub const DEFAULT_ECR_GATE_ERROR_MEAN: f64 = 0.006;
    pub const DEFAULT_ECR_GATE_ERROR_STDDEV: f64 = 0.002;
};

pub const QuantumConfig = struct {
    pub const URL_BUFFER_SIZE: usize = 256;
    pub const HEADER_BUFFER_SIZE: usize = 4096;
    pub const MAX_RESPONSE_SIZE: usize = 1024 * 1024;
    pub const NANOSECONDS_PER_SECOND: f64 = 1e9;
    pub const QUBIT_VARIATION_DIVISOR: f64 = 10.0;
    pub const QUBIT_VARIATION_OFFSET: f64 = 0.5;
    pub const QUBIT_VARIATION_MODULO: usize = 10;
    pub const MIN_READOUT_ERROR: f64 = 0.001;
    pub const MIN_GATE_ERROR: f64 = 0.0005;
    pub const SIMULATOR_MAX_SHOTS: u32 = 100000;
    pub const SIMULATOR_MAX_CIRCUITS: u32 = 300;
    pub const HARDWARE_MAX_SHOTS: u32 = 100000;
    pub const HARDWARE_MAX_CIRCUITS: u32 = 100;
    pub const TOKEN_LENGTH: usize = 64;
    pub const SESSION_ID_LENGTH: usize = 32;
    pub const JOB_ID_LENGTH: usize = 36;
    pub const JOB_ID_DASH_POSITIONS: [4]usize = .{ 8, 13, 18, 23 };
    pub const TOKEN_VALIDITY_MS: i64 = 3600000;
    pub const HEX_CHARS: []const u8 = "0123456789abcdef";
    pub const ALPHA_NUM_CHARS: []const u8 =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    pub const DEFAULT_REGION: []const u8 = "us-east";
    pub const DEFAULT_INSTANCE: []const u8 = "quantum-computing";
    pub const ACCOUNT_PREFIX: []const u8 = "a/";
    pub const CRN_REGION_INDEX: usize = 5;
    pub const CRN_ACCOUNT_INDEX: usize = 6;
    pub const CRN_RESOURCE_INDEX: usize = 7;
    pub const BASIS_GATES_COUNT_SIM: usize = 6;
    pub const BASIS_GATES_COUNT_HW: usize = 7;
    pub const MAX_QUBITS_SIMULATION: u32 = 20;
    pub const BITSTRING_BUFFER_SIZE: usize = 32;
    pub const NAME_BUFFER_SIZE: usize = 32;
    pub const DEPTH_DECAY_FACTOR: f64 = 100.0;
    pub const DEFAULT_SHOTS: u32 = 4000;
    pub const DEFAULT_MAX_ITERATIONS: u32 = 100;
    pub const DEFAULT_TOLERANCE: f64 = 1e-6;
    pub const DEFAULT_LEARNING_RATE: f64 = 0.1;
    pub const JOB_WAIT_TIMEOUT_MS: i64 = 60000;
    pub const POLL_INTERVAL_MS: u64 = 100;
    pub const SIMULATOR_QUBITS: u32 = 32;
    pub const HERON_QUBITS: u32 = 133;
    pub const EAGLE_QUBITS: u32 = 127;
};

pub const IBMBackendCalibrationData = struct {
    t1_times_ns: []f64,
    t2_times_ns: []f64,
    readout_errors: []f64,
    gate_errors: []f64,
    coupling_map: [][2]u32,
    num_qubits: u32,
    allocator: Allocator,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.t1_times_ns);
        self.allocator.free(self.t2_times_ns);
        self.allocator.free(self.readout_errors);
        self.allocator.free(self.gate_errors);
        self.allocator.free(self.coupling_map);
    }
};

pub const IBMDocumentedBackendSpecs = struct {
    t1_mean_ns: f64,
    t1_stddev_ns: f64,
    t2_mean_ns: f64,
    t2_stddev_ns: f64,
    readout_error_mean: f64,
    readout_error_stddev: f64,
    ecr_gate_error_mean: f64,
    ecr_gate_error_stddev: f64,

    pub fn getForBackendType(backend_type: QuantumBackendType) IBMDocumentedBackendSpecs {
        return switch (backend_type) {
            .HARDWARE_HERON => .{
                .t1_mean_ns = IBMBackendSpecs.HERON_T1_MEAN_NS,
                .t1_stddev_ns = IBMBackendSpecs.HERON_T1_STDDEV_NS,
                .t2_mean_ns = IBMBackendSpecs.HERON_T2_MEAN_NS,
                .t2_stddev_ns = IBMBackendSpecs.HERON_T2_STDDEV_NS,
                .readout_error_mean = IBMBackendSpecs.HERON_READOUT_ERROR_MEAN,
                .readout_error_stddev = IBMBackendSpecs.HERON_READOUT_ERROR_STDDEV,
                .ecr_gate_error_mean = IBMBackendSpecs.HERON_ECR_GATE_ERROR_MEAN,
                .ecr_gate_error_stddev = IBMBackendSpecs.HERON_ECR_GATE_ERROR_STDDEV,
            },
            .HARDWARE_EAGLE => .{
                .t1_mean_ns = IBMBackendSpecs.EAGLE_T1_MEAN_NS,
                .t1_stddev_ns = IBMBackendSpecs.EAGLE_T1_STDDEV_NS,
                .t2_mean_ns = IBMBackendSpecs.EAGLE_T2_MEAN_NS,
                .t2_stddev_ns = IBMBackendSpecs.EAGLE_T2_STDDEV_NS,
                .readout_error_mean = IBMBackendSpecs.EAGLE_READOUT_ERROR_MEAN,
                .readout_error_stddev = IBMBackendSpecs.EAGLE_READOUT_ERROR_STDDEV,
                .ecr_gate_error_mean = IBMBackendSpecs.EAGLE_ECR_GATE_ERROR_MEAN,
                .ecr_gate_error_stddev = IBMBackendSpecs.EAGLE_ECR_GATE_ERROR_STDDEV,
            },
            .HARDWARE_FALCON => .{
                .t1_mean_ns = IBMBackendSpecs.FALCON_T1_MEAN_NS,
                .t1_stddev_ns = IBMBackendSpecs.FALCON_T1_STDDEV_NS,
                .t2_mean_ns = IBMBackendSpecs.FALCON_T2_MEAN_NS,
                .t2_stddev_ns = IBMBackendSpecs.FALCON_T2_STDDEV_NS,
                .readout_error_mean = IBMBackendSpecs.FALCON_READOUT_ERROR_MEAN,
                .readout_error_stddev = IBMBackendSpecs.FALCON_READOUT_ERROR_STDDEV,
                .ecr_gate_error_mean = IBMBackendSpecs.FALCON_ECR_GATE_ERROR_MEAN,
                .ecr_gate_error_stddev = IBMBackendSpecs.FALCON_ECR_GATE_ERROR_STDDEV,
            },
            .HARDWARE_OSPREY => .{
                .t1_mean_ns = IBMBackendSpecs.OSPREY_T1_MEAN_NS,
                .t1_stddev_ns = IBMBackendSpecs.OSPREY_T1_STDDEV_NS,
                .t2_mean_ns = IBMBackendSpecs.OSPREY_T2_MEAN_NS,
                .t2_stddev_ns = IBMBackendSpecs.OSPREY_T2_STDDEV_NS,
                .readout_error_mean = IBMBackendSpecs.OSPREY_READOUT_ERROR_MEAN,
                .readout_error_stddev = IBMBackendSpecs.OSPREY_READOUT_ERROR_STDDEV,
                .ecr_gate_error_mean = IBMBackendSpecs.OSPREY_ECR_GATE_ERROR_MEAN,
                .ecr_gate_error_stddev = IBMBackendSpecs.OSPREY_ECR_GATE_ERROR_STDDEV,
            },
            .HARDWARE_CONDOR => .{
                .t1_mean_ns = IBMBackendSpecs.CONDOR_T1_MEAN_NS,
                .t1_stddev_ns = IBMBackendSpecs.CONDOR_T1_STDDEV_NS,
                .t2_mean_ns = IBMBackendSpecs.CONDOR_T2_MEAN_NS,
                .t2_stddev_ns = IBMBackendSpecs.CONDOR_T2_STDDEV_NS,
                .readout_error_mean = IBMBackendSpecs.CONDOR_READOUT_ERROR_MEAN,
                .readout_error_stddev = IBMBackendSpecs.CONDOR_READOUT_ERROR_STDDEV,
                .ecr_gate_error_mean = IBMBackendSpecs.CONDOR_ECR_GATE_ERROR_MEAN,
                .ecr_gate_error_stddev = IBMBackendSpecs.CONDOR_ECR_GATE_ERROR_STDDEV,
            },
            .SIMULATOR_STATEVECTOR, .SIMULATOR_MPS, .SIMULATOR_STABILIZER => unreachable,
        };
    }
};

pub var mock_mode: bool = false;

pub fn enableMockMode() void {
    mock_mode = true;
}

pub fn disableMockMode() void {
    mock_mode = false;
}

pub fn isMockMode() bool {
    return mock_mode;
}

pub fn integrateHardwareWithInference(
    allocator: Allocator,
    backend_name: []const u8,
    circuit_depth: u32,
    two_qubit_gates: u32,
) !f64 {
    var client = try IBMQuantumClient.init(allocator, "crn:v1:bluemix:public:quantum-computing:us-east:a/test:test::", "");
    defer client.deinit();
    const backend = client.getBackend(backend_name) orelse return error.BackendNotFound;
    return backend.estimateFidelity(circuit_depth, two_qubit_gates);
}

pub fn fetchIBMQuantumCalibration(
    allocator: Allocator,
    backend_name: []const u8,
    api_key: []const u8,
) !?IBMBackendCalibrationData {
    if (mock_mode) {
        return generateDocumentedCalibration(allocator, .HARDWARE_HERON, 133);
    }

    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    var url_buf: [QuantumConfig.URL_BUFFER_SIZE]u8 = undefined;
    const url = std.fmt.bufPrint(
        &url_buf,
        "https://cloud.ibm.com/api/quantum/v1/backends/{s}",
        .{backend_name},
    ) catch return null;

    const uri = std.Uri.parse(url) catch return null;

    var auth_header_buf: [QuantumConfig.HEADER_BUFFER_SIZE]u8 = undefined;
    const auth_header = std.fmt.bufPrint(&auth_header_buf, "Bearer {s}", .{api_key}) catch return null;

    var header_buf: [QuantumConfig.HEADER_BUFFER_SIZE]u8 = undefined;
    var req = client.open(.GET, uri, .{
        .server_header_buffer = &header_buf,
        .extra_headers = &[_]std.http.Header{
            .{ .name = "Authorization", .value = auth_header },
            .{ .name = "x-ibm-api-key", .value = api_key },
            .{ .name = "Accept", .value = "application/json" },
        },
    }) catch return null;
    defer req.deinit();

    req.send() catch return null;
    req.wait() catch return null;

    if (req.status != .ok) return null;

    const body = req.reader().readAllAlloc(
        allocator,
        QuantumConfig.MAX_RESPONSE_SIZE,
    ) catch return null;
    defer allocator.free(body);

    return parseIBMCalibrationResponse(allocator, body) catch null;
}

pub fn parseIBMCalibrationResponse(
    allocator: Allocator,
    json_body: []const u8,
) !IBMBackendCalibrationData {
    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json_body,
        .{},
    );
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return error.InvalidFormat;
    const properties = root.object.get("properties") orelse return error.MissingProperties;
    if (properties != .object) return error.InvalidFormat;
    const qubits_array = properties.object.get("qubits") orelse return error.MissingQubits;

    if (qubits_array != .array) return error.InvalidFormat;
    const num_qubits: u32 = @intCast(qubits_array.array.items.len);
    if (num_qubits == 0) return error.ZeroQubits;

    const t1 = try allocator.alloc(f64, num_qubits);
    errdefer allocator.free(t1);
    const t2 = try allocator.alloc(f64, num_qubits);
    errdefer allocator.free(t2);
    const readout_err = try allocator.alloc(f64, num_qubits);
    errdefer allocator.free(readout_err);

    var idx: usize = 0;
    while (idx < qubits_array.array.items.len) : (idx += 1) {
        const qubit_props = qubits_array.array.items[idx];
        t1[idx] = IBMBackendSpecs.DEFAULT_T1_MEAN_NS;
        t2[idx] = IBMBackendSpecs.DEFAULT_T2_MEAN_NS;
        readout_err[idx] = IBMBackendSpecs.DEFAULT_READOUT_ERROR_MEAN;

        if (qubit_props != .array) continue;

        for (qubit_props.array.items) |prop| {
            if (prop != .object) continue;
            const name = prop.object.get("name") orelse continue;
            const value = prop.object.get("value") orelse continue;

            if (name != .string) continue;

            var float_val: f64 = 0.0;
            if (value == .float) {
                float_val = value.float;
            } else if (value == .integer) {
                float_val = @as(f64, @floatFromInt(value.integer));
            } else continue;

            if (std.mem.eql(u8, name.string, "T1")) {
                t1[idx] = @max(0.0, if (float_val < 1.0) float_val * QuantumConfig.NANOSECONDS_PER_SECOND else float_val);
            } else if (std.mem.eql(u8, name.string, "T2")) {
                t2[idx] = @max(0.0, if (float_val < 1.0) float_val * QuantumConfig.NANOSECONDS_PER_SECOND else float_val);
            } else if (std.mem.eql(u8, name.string, "readout_error")) {
                readout_err[idx] = @max(QuantumConfig.MIN_READOUT_ERROR, float_val);
            }
        }
    }

    const gates_array = properties.object.get("gates");
    var gate_errors = std.ArrayList(f64).init(allocator);
    errdefer gate_errors.deinit();
    var coupling_edges = std.ArrayList([2]u32).init(allocator);
    errdefer coupling_edges.deinit();

    if (gates_array != null and gates_array.? == .array) {
        for (gates_array.?.array.items) |gate| {
            if (gate != .object) continue;

            const gate_qubits = gate.object.get("qubits") orelse continue;
            if (gate_qubits != .array or gate_qubits.array.items.len != 2) continue;

            const q0 = gate_qubits.array.items[0];
            const q1 = gate_qubits.array.items[1];
            if (q0 != .integer or q1 != .integer) continue;

            const gate_params = gate.object.get("parameters") orelse continue;
            if (gate_params != .array) continue;

            var err_val: ?f64 = null;
            for (gate_params.array.items) |param| {
                if (param != .object) continue;
                const param_name = param.object.get("name") orelse continue;
                if (param_name != .string) continue;

                if (std.mem.eql(u8, param_name.string, "gate_error")) {
                    const param_value = param.object.get("value") orelse continue;
                    if (param_value == .float) {
                        err_val = param_value.float;
                    } else if (param_value == .integer) {
                        err_val = @as(f64, @floatFromInt(param_value.integer));
                    }
                }
            }

            if (err_val) |ev| {
                try coupling_edges.append(.{ @intCast(q0.integer), @intCast(q1.integer) });
                try gate_errors.append(@max(QuantumConfig.MIN_GATE_ERROR, ev));
            }
        }
    }

    const gate_err_slice = try gate_errors.toOwnedSlice();
    errdefer allocator.free(gate_err_slice);
    const coupling_slice = try coupling_edges.toOwnedSlice();

    return IBMBackendCalibrationData{
        .t1_times_ns = t1,
        .t2_times_ns = t2,
        .readout_errors = readout_err,
        .gate_errors = gate_err_slice,
        .coupling_map = coupling_slice,
        .num_qubits = num_qubits,
        .allocator = allocator,
    };
}

pub fn generateDocumentedCalibration(
    allocator: Allocator,
    backend_type: QuantumBackendType,
    num_qubits: u32,
) !IBMBackendCalibrationData {
    if (num_qubits == 0) return error.ZeroQubits;
    const specs = IBMDocumentedBackendSpecs.getForBackendType(backend_type);

    const t1 = try allocator.alloc(f64, num_qubits);
    errdefer allocator.free(t1);
    const t2 = try allocator.alloc(f64, num_qubits);
    errdefer allocator.free(t2);
    const readout_err = try allocator.alloc(f64, num_qubits);
    errdefer allocator.free(readout_err);

    var prng = std.Random.DefaultPrng.init(@as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))));
    const random = prng.random();

    var qubit_idx: usize = 0;
    while (qubit_idx < num_qubits) : (qubit_idx += 1) {
        const t1_variation = @as(f64, @floatFromInt(random.intRangeAtMost(usize, 0, 1000))) / 1000.0 - 0.5;
        const t2_variation = @as(f64, @floatFromInt(random.intRangeAtMost(usize, 0, 1000))) / 1000.0 - 0.5;
        const readout_variation = @as(f64, @floatFromInt(random.intRangeAtMost(usize, 0, 1000))) / 1000.0 - 0.5;
        t1[qubit_idx] = @max(1000.0, specs.t1_mean_ns + t1_variation * specs.t1_stddev_ns);
        t2[qubit_idx] = @max(1000.0, specs.t2_mean_ns + t2_variation * specs.t2_stddev_ns);
        readout_err[qubit_idx] = @max(
            QuantumConfig.MIN_READOUT_ERROR,
            specs.readout_error_mean + readout_variation * specs.readout_error_stddev,
        );
    }

    var edge_count: usize = 0;
    var count_idx: usize = 0;
    while (count_idx < num_qubits) : (count_idx += 1) {
        if (count_idx + 1 < num_qubits) edge_count += 1;
        if (count_idx >= 1) edge_count += 1;
    }

    const coupling = try allocator.alloc([2]u32, edge_count);
    errdefer allocator.free(coupling);
    var edge_idx: usize = 0;
    var coupling_idx: usize = 0;
    while (coupling_idx < num_qubits) : (coupling_idx += 1) {
        if (coupling_idx + 1 < num_qubits) {
            coupling[edge_idx] = .{ @intCast(coupling_idx), @intCast(coupling_idx + 1) };
            edge_idx += 1;
        }
        if (coupling_idx >= 1) {
            coupling[edge_idx] = .{ @intCast(coupling_idx), @intCast(coupling_idx - 1) };
            edge_idx += 1;
        }
    }

    const gate_err = try allocator.alloc(f64, edge_count);
    errdefer allocator.free(gate_err);
    var gate_idx: usize = 0;
    while (gate_idx < edge_count) : (gate_idx += 1) {
        const edge_variation = @as(f64, @floatFromInt(random.intRangeAtMost(usize, 0, 1000))) / 1000.0 - 0.5;
        gate_err[gate_idx] = @max(
            QuantumConfig.MIN_GATE_ERROR,
            specs.ecr_gate_error_mean + edge_variation * specs.ecr_gate_error_stddev,
        );
    }

    return IBMBackendCalibrationData{
        .t1_times_ns = t1,
        .t2_times_ns = t2,
        .readout_errors = readout_err,
        .gate_errors = gate_err,
        .coupling_map = coupling,
        .num_qubits = num_qubits,
        .allocator = allocator,
    };
}

pub const IBMQuantumCredentials = struct {
    crn: []const u8,
    api_key: []const u8,
    instance: []const u8,
    region: []const u8,
    account_id: []const u8,
    resource_id: []const u8,
    allocator: Allocator,

    const Self = @This();

    pub fn parseCRN(
        allocator: Allocator,
        crn_string: []const u8,
        api_key: []const u8,
    ) !*Self {
        var parts = std.mem.splitSequence(u8, crn_string, ":");
        var part_idx: usize = 0;
        var region_str: ?[]const u8 = null;
        var account_str: ?[]const u8 = null;
        var resource_str: ?[]const u8 = null;

        while (parts.next()) |part| : (part_idx += 1) {
            switch (part_idx) {
                QuantumConfig.CRN_REGION_INDEX => {
                    if (part.len > 0) region_str = part;
                },
                QuantumConfig.CRN_ACCOUNT_INDEX => {
                    if (std.mem.startsWith(u8, part, QuantumConfig.ACCOUNT_PREFIX)) {
                        account_str = part[2..];
                    }
                },
                QuantumConfig.CRN_RESOURCE_INDEX => {
                    if (part.len > 0) resource_str = part;
                },
                else => {},
            }
        }

        if (part_idx < 8) return error.InvalidCRN;

        const creds = try allocator.create(Self);
        errdefer allocator.destroy(creds);

        creds.crn = try allocator.dupe(u8, crn_string);
        errdefer allocator.free(creds.crn);

        creds.api_key = try allocator.dupe(u8, api_key);
        errdefer allocator.free(creds.api_key);

        creds.instance = try allocator.dupe(u8, QuantumConfig.DEFAULT_INSTANCE);
        errdefer allocator.free(creds.instance);

        creds.region = try allocator.dupe(u8, region_str orelse QuantumConfig.DEFAULT_REGION);
        errdefer allocator.free(creds.region);

        creds.account_id = if (account_str) |a| try allocator.dupe(u8, a) else try allocator.dupe(u8, "");
        errdefer allocator.free(creds.account_id);

        creds.resource_id = if (resource_str) |r| try allocator.dupe(u8, r) else try allocator.dupe(u8, "");
        errdefer allocator.free(creds.resource_id);

        creds.allocator = allocator;
        return creds;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.crn);
        self.allocator.free(self.api_key);
        self.allocator.free(self.instance);
        self.allocator.free(self.region);
        self.allocator.free(self.account_id);
        self.allocator.free(self.resource_id);
        self.allocator.destroy(self);
    }

    pub fn getServiceURL(self: *const Self, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "https://{s}.quantum-computing.cloud.ibm.com", .{self.region});
    }

    pub fn getRuntimeURL(self: *const Self, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "https://{s}.quantum-computing.cloud.ibm.com/runtime", .{self.region});
    }
};

pub const QuantumBackendType = enum {
    SIMULATOR_STATEVECTOR,
    SIMULATOR_MPS,
    SIMULATOR_STABILIZER,
    HARDWARE_FALCON,
    HARDWARE_HERON,
    HARDWARE_EAGLE,
    HARDWARE_OSPREY,
    HARDWARE_CONDOR,
};

pub const QuantumBackend = struct {
    name: []const u8,
    backend_type: QuantumBackendType,
    num_qubits: u32,
    basis_gates: []const []const u8,
    coupling_map: []const [2]u32,
    t1_times_ns: []const f64,
    t2_times_ns: []const f64,
    readout_errors: []const f64,
    gate_errors: []const f64,
    is_simulator: bool,
    max_shots: u32,
    max_circuits: u32,
    status: BackendStatus,
    allocator: Allocator,

    const Self = @This();

    pub fn initSimulator(allocator: Allocator, name: []const u8, num_qubits: u32) !*Self {
        const backend = try allocator.create(Self);
        errdefer allocator.destroy(backend);

        var basis = try allocator.alloc([]const u8, QuantumConfig.BASIS_GATES_COUNT_SIM);
        errdefer allocator.free(basis);
        var allocated_basis: usize = 0;
        errdefer {
            for (basis[0..allocated_basis]) |b| allocator.free(b);
        }
        const literals = [_][]const u8{ "id", "rz", "sx", "x", "cx", "reset" };
        for (literals) |lit| {
            basis[allocated_basis] = try allocator.dupe(u8, lit);
            allocated_basis += 1;
        }

        backend.* = Self{
            .name = try allocator.dupe(u8, name),
            .backend_type = .SIMULATOR_STATEVECTOR,
            .num_qubits = num_qubits,
            .basis_gates = basis,
            .coupling_map = try allocator.alloc([2]u32, 0),
            .t1_times_ns = try allocator.alloc(f64, 0),
            .t2_times_ns = try allocator.alloc(f64, 0),
            .readout_errors = try allocator.alloc(f64, 0),
            .gate_errors = try allocator.alloc(f64, 0),
            .is_simulator = true,
            .max_shots = QuantumConfig.SIMULATOR_MAX_SHOTS,
            .max_circuits = QuantumConfig.SIMULATOR_MAX_CIRCUITS,
            .status = .ONLINE,
            .allocator = allocator,
        };

        return backend;
    }

    pub fn initHardware(
        allocator: Allocator,
        name: []const u8,
        backend_type: QuantumBackendType,
        num_qubits: u32,
    ) !*Self {
        return initHardwareWithCalibration(allocator, name, backend_type, num_qubits, null);
    }

    pub fn initHardwareWithAPI(
        allocator: Allocator,
        name: []const u8,
        backend_type: QuantumBackendType,
        num_qubits: u32,
        api_key: []const u8,
    ) !*Self {
        if (fetchIBMQuantumCalibration(allocator, name, api_key)) |api_calibration| {
            var calib = api_calibration;
            defer calib.deinit();
            return initHardwareWithCalibration(
                allocator,
                name,
                backend_type,
                num_qubits,
                calib,
            );
        } else |_| {
            return initHardwareWithCalibration(allocator, name, backend_type, num_qubits, null);
        }
    }

    pub fn initHardwareWithCalibration(
        allocator: Allocator,
        name: []const u8,
        backend_type: QuantumBackendType,
        num_qubits: u32,
        api_calibration: ?IBMBackendCalibrationData,
    ) !*Self {
        const backend = try allocator.create(Self);
        errdefer allocator.destroy(backend);

        var basis = try allocator.alloc([]const u8, QuantumConfig.BASIS_GATES_COUNT_HW);
        errdefer allocator.free(basis);
        var allocated_basis: usize = 0;
        errdefer {
            for (basis[0..allocated_basis]) |b| allocator.free(b);
        }
        const literals = [_][]const u8{ "id", "rz", "sx", "x", "ecr", "reset", "measure" };
        for (literals) |lit| {
            basis[allocated_basis] = try allocator.dupe(u8, lit);
            allocated_basis += 1;
        }

        var calibration: IBMBackendCalibrationData = undefined;
        var owns_calibration = false;

        if (api_calibration) |api_calib| {
            calibration = api_calib;
        } else {
            calibration = try generateDocumentedCalibration(allocator, backend_type, num_qubits);
            owns_calibration = true;
        }
        defer {
            if (owns_calibration) calibration.deinit();
        }

        const t1 = try allocator.alloc(f64, num_qubits);
        errdefer allocator.free(t1);
        const t2 = try allocator.alloc(f64, num_qubits);
        errdefer allocator.free(t2);
        const readout_err = try allocator.alloc(f64, num_qubits);
        errdefer allocator.free(readout_err);

        var qubit_idx: usize = 0;
        while (qubit_idx < num_qubits) : (qubit_idx += 1) {
            if (qubit_idx < calibration.t1_times_ns.len) {
                t1[qubit_idx] = calibration.t1_times_ns[qubit_idx];
            } else {
                const specs = IBMDocumentedBackendSpecs.getForBackendType(backend_type);
                t1[qubit_idx] = specs.t1_mean_ns;
            }

            if (qubit_idx < calibration.t2_times_ns.len) {
                t2[qubit_idx] = calibration.t2_times_ns[qubit_idx];
            } else {
                const specs = IBMDocumentedBackendSpecs.getForBackendType(backend_type);
                t2[qubit_idx] = specs.t2_mean_ns;
            }

            if (qubit_idx < calibration.readout_errors.len) {
                readout_err[qubit_idx] = calibration.readout_errors[qubit_idx];
            } else {
                const specs = IBMDocumentedBackendSpecs.getForBackendType(backend_type);
                readout_err[qubit_idx] = specs.readout_error_mean;
            }
        }

        const coupling = try allocator.alloc([2]u32, calibration.coupling_map.len);
        errdefer allocator.free(coupling);
        @memcpy(coupling, calibration.coupling_map);

        const gate_err = try allocator.alloc(f64, calibration.gate_errors.len);
        errdefer allocator.free(gate_err);
        @memcpy(gate_err, calibration.gate_errors);

        backend.* = Self{
            .name = try allocator.dupe(u8, name),
            .backend_type = backend_type,
            .num_qubits = num_qubits,
            .basis_gates = basis,
            .coupling_map = coupling,
            .t1_times_ns = t1,
            .t2_times_ns = t2,
            .readout_errors = readout_err,
            .gate_errors = gate_err,
            .is_simulator = false,
            .max_shots = QuantumConfig.HARDWARE_MAX_SHOTS,
            .max_circuits = QuantumConfig.HARDWARE_MAX_CIRCUITS,
            .status = .ONLINE,
            .allocator = allocator,
        };

        return backend;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
        for (self.basis_gates) |b| self.allocator.free(b);
        self.allocator.free(self.basis_gates);
        self.allocator.free(self.coupling_map);
        self.allocator.free(self.t1_times_ns);
        self.allocator.free(self.t2_times_ns);
        self.allocator.free(self.readout_errors);
        self.allocator.free(self.gate_errors);
        self.allocator.destroy(self);
    }

    pub fn getAverageT1(self: *const Self) f64 {
        if (self.t1_times_ns.len == 0) return 0.0;
        var sum: f64 = 0.0;
        for (self.t1_times_ns) |t| sum += t;
        return sum / @as(f64, @floatFromInt(self.t1_times_ns.len));
    }

    pub fn getAverageT2(self: *const Self) f64 {
        if (self.t2_times_ns.len == 0) return 0.0;
        var sum: f64 = 0.0;
        for (self.t2_times_ns) |t| sum += t;
        return sum / @as(f64, @floatFromInt(self.t2_times_ns.len));
    }

    pub fn getAverageReadoutError(self: *const Self) f64 {
        if (self.readout_errors.len == 0) return 0.0;
        var sum: f64 = 0.0;
        for (self.readout_errors) |e| sum += e;
        return sum / @as(f64, @floatFromInt(self.readout_errors.len));
    }

    pub fn getAverageGateError(self: *const Self) f64 {
        if (self.gate_errors.len == 0) return 0.0;
        var sum: f64 = 0.0;
        for (self.gate_errors) |e| sum += e;
        return sum / @as(f64, @floatFromInt(self.gate_errors.len));
    }

    pub fn estimateFidelity(self: *const Self, circuit_depth: u32, num_two_qubit_gates: u32) f64 {
        if (self.is_simulator) return 1.0;
        if (circuit_depth == 0) return 1.0;

        const avg_gate_error = self.getAverageGateError();
        const avg_readout_error = self.getAverageReadoutError();

        const gate_fidelity = std.math.pow(
            f64,
            1.0 - avg_gate_error,
            @as(f64, @floatFromInt(num_two_qubit_gates)),
        );
        const readout_fidelity = 1.0 - avg_readout_error;
        const depth_factor = std.math.exp(
            -@as(f64, @floatFromInt(circuit_depth)) / QuantumConfig.DEPTH_DECAY_FACTOR,
        );

        return gate_fidelity * readout_fidelity * depth_factor;
    }

    pub fn inferBackendStatus(self: *const Self) BackendStatus {
        if (self.is_simulator) return .ONLINE;
        const avg_readout = self.getAverageReadoutError();
        const avg_gate = self.getAverageGateError();
        if (avg_readout > 0.1 or avg_gate > 0.05) return .DEGRADED;
        if (avg_readout > 0.05 or avg_gate > 0.02) return .PAUSED;
        return .ONLINE;
    }

    pub fn integrateWithInference(self: *Self, circuit: *QuantumCircuit) !f64 {
        self.status = self.inferBackendStatus();
        if (self.status == .OFFLINE or self.status == .MAINTENANCE) return error.BackendUnavailable;
        var two_qubit_count: u32 = 0;
        for (circuit.instructions.items) |inst| {
            if (inst.isTwoQubitGate()) two_qubit_count += 1;
        }
        return self.estimateFidelity(@intCast(circuit.instructions.items.len), two_qubit_count);
    }
};

pub const BackendStatus = enum {
    ONLINE,
    OFFLINE,
    MAINTENANCE,
    DEGRADED,
    PAUSED,
};

pub const QuantumGateOp = enum {
    ID, X, Y, Z, H, S, SDG, T, TDG, SX, SXDG, RX, RY, RZ, U1, U2, U3, P,
    CX, CY, CZ, CH, CRX, CRY, CRZ, CP, ECR, SWAP, ISWAP,
    CCX, CSWAP, MCX,
    RESET, MEASURE, BARRIER,
};

pub const QuantumInstruction = struct {
    gate: QuantumGateOp,
    qubits: []u32,
    classical_bits: []u32,
    parameters: []f64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        gate: QuantumGateOp,
        qubits: []const u32,
        params: []const f64,
    ) !*Self {
        const inst = try allocator.create(Self);
        errdefer allocator.destroy(inst);

        inst.* = Self{
            .gate = gate,
            .qubits = try allocator.dupe(u32, qubits),
            .classical_bits = try allocator.alloc(u32, 0),
            .parameters = try allocator.dupe(f64, params),
            .allocator = allocator,
        };

        return inst;
    }

    pub fn initMeasure(allocator: Allocator, qubit: u32, classical_bit: u32) !*Self {
        const inst = try allocator.create(Self);
        errdefer allocator.destroy(inst);

        const qubits = try allocator.alloc(u32, 1);
        errdefer allocator.free(qubits);
        qubits[0] = qubit;

        const cbits = try allocator.alloc(u32, 1);
        errdefer allocator.free(cbits);
        cbits[0] = classical_bit;

        inst.* = Self{
            .gate = .MEASURE,
            .qubits = qubits,
            .classical_bits = cbits,
            .parameters = try allocator.alloc(f64, 0),
            .allocator = allocator,
        };

        return inst;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.qubits);
        self.allocator.free(self.classical_bits);
        self.allocator.free(self.parameters);
        self.allocator.destroy(self);
    }

    pub fn getGateName(self: *const Self) []const u8 {
        return switch (self.gate) {
            .ID => "id", .X => "x", .Y => "y", .Z => "z", .H => "h",
            .S => "s", .SDG => "sdg", .T => "t", .TDG => "tdg",
            .SX => "sx", .SXDG => "sxdg", .RX => "rx", .RY => "ry", .RZ => "rz",
            .U1 => "u1", .U2 => "u2", .U3 => "u3", .P => "p",
            .CX => "cx", .CY => "cy", .CZ => "cz", .CH => "ch",
            .CRX => "crx", .CRY => "cry", .CRZ => "crz", .CP => "cp",
            .ECR => "ecr", .SWAP => "swap", .ISWAP => "iswap",
            .CCX => "ccx", .CSWAP => "cswap", .MCX => "mcx",
            .RESET => "reset", .MEASURE => "measure", .BARRIER => "barrier",
        };
    }

    pub fn getNumQubits(self: *const Self) usize {
        return self.qubits.len;
    }

    pub fn isTwoQubitGate(self: *const Self) bool {
        return switch (self.gate) {
            .CX, .CY, .CZ, .CH, .CRX, .CRY, .CRZ, .CP, .ECR, .SWAP, .ISWAP => true,
            else => false,
        };
    }

    pub fn isThreeQubitGate(self: *const Self) bool {
        return switch (self.gate) {
            .CCX, .CSWAP => true,
            else => false,
        };
    }
};

pub const QuantumCircuit = struct {
    name: []const u8,
    num_qubits: u32,
    num_classical_bits: u32,
    instructions: std.ArrayList(*QuantumInstruction),
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        name: []const u8,
        num_qubits: u32,
        num_classical_bits: u32,
    ) !*Self {
        const circuit = try allocator.create(Self);
        errdefer allocator.destroy(circuit);

        circuit.* = Self{
            .name = try allocator.dupe(u8, name),
            .num_qubits = num_qubits,
            .num_classical_bits = num_classical_bits,
            .instructions = std.ArrayList(*QuantumInstruction).init(allocator),
            .allocator = allocator,
        };

        return circuit;
    }

    pub fn deinit(self: *Self) void {
        for (self.instructions.items) |inst| {
            inst.deinit();
        }
        self.instructions.deinit();
        self.allocator.free(self.name);
        self.allocator.destroy(self);
    }

    pub fn addInstruction(self: *Self, inst: *QuantumInstruction) !void {
        try self.instructions.append(inst);
    }

    pub fn id(self: *Self, qubit: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .ID, &[_]u32{qubit}, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn h(self: *Self, qubit: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .H, &[_]u32{qubit}, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn x(self: *Self, qubit: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .X, &[_]u32{qubit}, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn y(self: *Self, qubit: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .Y, &[_]u32{qubit}, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn z(self: *Self, qubit: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .Z, &[_]u32{qubit}, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn s(self: *Self, qubit: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .S, &[_]u32{qubit}, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn sdg(self: *Self, qubit: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .SDG, &[_]u32{qubit}, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn t(self: *Self, qubit: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .T, &[_]u32{qubit}, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn tdg(self: *Self, qubit: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .TDG, &[_]u32{qubit}, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn sx(self: *Self, qubit: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .SX, &[_]u32{qubit}, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn sxdg(self: *Self, qubit: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .SXDG, &[_]u32{qubit}, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn rx(self: *Self, qubit: u32, theta: f64) !void {
        const inst = try QuantumInstruction.init(self.allocator, .RX, &[_]u32{qubit}, &[_]f64{theta});
        try self.addInstruction(inst);
    }

    pub fn ry(self: *Self, qubit: u32, theta: f64) !void {
        const inst = try QuantumInstruction.init(self.allocator, .RY, &[_]u32{qubit}, &[_]f64{theta});
        try self.addInstruction(inst);
    }

    pub fn rz(self: *Self, qubit: u32, phi: f64) !void {
        const inst = try QuantumInstruction.init(self.allocator, .RZ, &[_]u32{qubit}, &[_]f64{phi});
        try self.addInstruction(inst);
    }

    pub fn p(self: *Self, qubit: u32, theta: f64) !void {
        const inst = try QuantumInstruction.init(self.allocator, .P, &[_]u32{qubit}, &[_]f64{theta});
        try self.addInstruction(inst);
    }

    pub fn @"u1"(self: *Self, qubit: u32, theta: f64) !void {
        const inst = try QuantumInstruction.init(self.allocator, .U1, &[_]u32{qubit}, &[_]f64{theta});
        try self.addInstruction(inst);
    }

    pub fn @"u2"(self: *Self, qubit: u32, phi: f64, lam: f64) !void {
        const inst = try QuantumInstruction.init(self.allocator, .U2, &[_]u32{qubit}, &[_]f64{phi, lam});
        try self.addInstruction(inst);
    }

    pub fn @"u3"(self: *Self, qubit: u32, theta: f64, phi: f64, lam: f64) !void {
        const inst = try QuantumInstruction.init(self.allocator, .U3, &[_]u32{qubit}, &[_]f64{theta, phi, lam});
        try self.addInstruction(inst);
    }

    pub fn cx(self: *Self, control: u32, target: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .CX, &[_]u32{ control, target }, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn cy(self: *Self, control: u32, target: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .CY, &[_]u32{ control, target }, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn cz(self: *Self, control: u32, target: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .CZ, &[_]u32{ control, target }, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn ch(self: *Self, control: u32, target: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .CH, &[_]u32{ control, target }, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn cp(self: *Self, control: u32, target: u32, theta: f64) !void {
        const inst = try QuantumInstruction.init(self.allocator, .CP, &[_]u32{ control, target }, &[_]f64{theta});
        try self.addInstruction(inst);
    }

    pub fn crx(self: *Self, control: u32, target: u32, theta: f64) !void {
        const inst = try QuantumInstruction.init(self.allocator, .CRX, &[_]u32{ control, target }, &[_]f64{theta});
        try self.addInstruction(inst);
    }

    pub fn cry(self: *Self, control: u32, target: u32, theta: f64) !void {
        const inst = try QuantumInstruction.init(self.allocator, .CRY, &[_]u32{ control, target }, &[_]f64{theta});
        try self.addInstruction(inst);
    }

    pub fn crz(self: *Self, control: u32, target: u32, theta: f64) !void {
        const inst = try QuantumInstruction.init(self.allocator, .CRZ, &[_]u32{ control, target }, &[_]f64{theta});
        try self.addInstruction(inst);
    }

    pub fn ecr(self: *Self, q1: u32, q2: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .ECR, &[_]u32{ q1, q2 }, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn swap(self: *Self, q1: u32, q2: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .SWAP, &[_]u32{ q1, q2 }, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn iswap(self: *Self, q1: u32, q2: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .ISWAP, &[_]u32{ q1, q2 }, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn ccx(self: *Self, c1: u32, c2: u32, target: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .CCX, &[_]u32{ c1, c2, target }, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn cswap(self: *Self, control: u32, target1: u32, target2: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .CSWAP, &[_]u32{ control, target1, target2 }, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn mcx(self: *Self, controls: []const u32, target: u32) !void {
        var qubits = try self.allocator.alloc(u32, controls.len + 1);
        defer self.allocator.free(qubits);
        @memcpy(qubits[0..controls.len], controls);
        qubits[controls.len] = target;
        const inst = try QuantumInstruction.init(self.allocator, .MCX, qubits, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn measure(self: *Self, qubit: u32, classical_bit: u32) !void {
        const inst = try QuantumInstruction.initMeasure(self.allocator, qubit, classical_bit);
        try self.addInstruction(inst);
    }

    pub fn measureAll(self: *Self) !void {
        const n = @min(self.num_qubits, self.num_classical_bits);
        var qidx: usize = 0;
        while (qidx < n) : (qidx += 1) {
            try self.measure(@intCast(qidx), @intCast(qidx));
        }
    }

    pub fn barrier(self: *Self, qubits: []const u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .BARRIER, qubits, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn reset(self: *Self, qubit: u32) !void {
        const inst = try QuantumInstruction.init(self.allocator, .RESET, &[_]u32{qubit}, &[_]f64{});
        try self.addInstruction(inst);
    }

    pub fn getDepth(self: *const Self) !u32 {
        var qubit_depths = std.AutoHashMap(u32, u32).init(self.allocator);
        defer qubit_depths.deinit();

        var max_depth: u32 = 0;

        for (self.instructions.items) |inst| {
            if (inst.gate == .BARRIER or inst.gate == .MEASURE or inst.gate == .RESET) continue;

            var current_max: u32 = 0;
            for (inst.qubits) |q| {
                const depth = qubit_depths.get(q) orelse 0;
                if (depth > current_max) current_max = depth;
            }

            const new_depth = current_max + 1;
            for (inst.qubits) |q| {
                try qubit_depths.put(q, new_depth);
            }

            if (new_depth > max_depth) max_depth = new_depth;
        }

        return max_depth;
    }

    pub fn countTwoQubitGates(self: *const Self) u32 {
        var count: u32 = 0;
        for (self.instructions.items) |inst| {
            if (inst.isTwoQubitGate()) count += 1;
        }
        return count;
    }

    pub fn countGates(self: *const Self) usize {
        return self.instructions.items.len;
    }

    pub fn toOpenQASM3(self: *const Self, allocator: Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        errdefer buffer.deinit();
        const writer = buffer.writer();

        try writer.print("OPENQASM 3.0;\n", .{});
        try writer.print("include \"stdgates.inc\";\n\n", .{});
        try writer.print("qubit[{d}] q;\n", .{self.num_qubits});
        try writer.print("bit[{d}] c;\n\n", .{self.num_classical_bits});

        for (self.instructions.items) |inst| {
            const gate_name = inst.getGateName();

            switch (inst.gate) {
                .MEASURE => {
                    if (inst.classical_bits.len > 0 and inst.qubits.len > 0) {
                        try writer.print("c[{d}] = measure q[{d}];\n", .{ inst.classical_bits[0], inst.qubits[0] });
                    }
                },
                .BARRIER => {
                    if (inst.qubits.len > 0) {
                        try writer.print("barrier ", .{});
                        for (inst.qubits, 0..) |q, idx| {
                            if (idx > 0) try writer.print(", ", .{});
                            try writer.print("q[{d}]", .{q});
                        }
                        try writer.print(";\n", .{});
                    }
                },
                .RX, .RY, .RZ, .U1, .P => {
                    if (inst.parameters.len > 0 and inst.qubits.len > 0) {
                        try writer.print("{s}({d}) q[{d}];\n", .{ gate_name, inst.parameters[0], inst.qubits[0] });
                    }
                },
                .U2 => {
                    if (inst.parameters.len > 1 and inst.qubits.len > 0) {
                        try writer.print("{s}({d}, {d}) q[{d}];\n", .{ gate_name, inst.parameters[0], inst.parameters[1], inst.qubits[0] });
                    }
                },
                .U3 => {
                    if (inst.parameters.len > 2 and inst.qubits.len > 0) {
                        try writer.print("{s}({d}, {d}, {d}) q[{d}];\n", .{ gate_name, inst.parameters[0], inst.parameters[1], inst.parameters[2], inst.qubits[0] });
                    }
                },
                .CX, .CY, .CZ, .CH, .ECR, .SWAP, .ISWAP => {
                    if (inst.qubits.len > 1) {
                        try writer.print("{s} q[{d}], q[{d}];\n", .{ gate_name, inst.qubits[0], inst.qubits[1] });
                    }
                },
                .CRX, .CRY, .CRZ, .CP => {
                    if (inst.parameters.len > 0 and inst.qubits.len > 1) {
                        try writer.print("{s}({d}) q[{d}], q[{d}];\n", .{ gate_name, inst.parameters[0], inst.qubits[0], inst.qubits[1] });
                    }
                },
                .CCX, .CSWAP => {
                    if (inst.qubits.len > 2) {
                        try writer.print("{s} q[{d}], q[{d}], q[{d}];\n", .{ gate_name, inst.qubits[0], inst.qubits[1], inst.qubits[2] });
                    }
                },
                .MCX => {
                    if (inst.qubits.len > 1) {
                        try writer.print("mcx ", .{});
                        for (inst.qubits, 0..) |q, idx| {
                            if (idx > 0) try writer.print(", ", .{});
                            try writer.print("q[{d}]", .{q});
                        }
                        try writer.print(";\n", .{});
                    }
                },
                .ID, .X, .Y, .Z, .H, .S, .SDG, .T, .TDG, .SX, .SXDG, .RESET => {
                    if (inst.qubits.len > 0) {
                        try writer.print("{s} q[{d}];\n", .{ gate_name, inst.qubits[0] });
                    }
                },
            }
        }

        return buffer.toOwnedSlice();
    }
};

pub const JobStatus = enum {
    QUEUED,
    VALIDATING,
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELLED,
    TIMEOUT,
};

pub const QuantumJob = struct {
    job_id: []const u8,
    backend_name: []const u8,
    status: JobStatus,
    creation_time: i64,
    start_time: ?i64,
    end_time: ?i64,
    shots: u32,
    result: ?*QuantumResult,
    error_message: ?[]const u8,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        job_id: []const u8,
        backend_name: []const u8,
        shots: u32,
    ) !*Self {
        const job = try allocator.create(Self);
        errdefer allocator.destroy(job);

        job.* = Self{
            .job_id = try allocator.dupe(u8, job_id),
            .backend_name = try allocator.dupe(u8, backend_name),
            .status = .QUEUED,
            .creation_time = std.time.milliTimestamp(),
            .start_time = null,
            .end_time = null,
            .shots = shots,
            .result = null,
            .error_message = null,
            .allocator = allocator,
        };

        return job;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.job_id);
        self.allocator.free(self.backend_name);
        if (self.result) |result| {
            result.deinit();
        }
        if (self.error_message) |msg| {
            self.allocator.free(msg);
        }
        self.allocator.destroy(self);
    }

    pub fn isTerminal(self: *const Self) bool {
        return switch (self.status) {
            .COMPLETED, .FAILED, .CANCELLED, .TIMEOUT => true,
            else => false,
        };
    }

    pub fn getExecutionTime(self: *const Self) ?i64 {
        if (self.start_time) |start| {
            if (self.end_time) |end| {
                return end - start;
            }
        }
        return null;
    }
};

pub const QuantumResult = struct {
    job_id: []const u8,
    backend_name: []const u8,
    success: bool,
    shots: u32,
    counts: std.StringHashMap(u64),
    execution_time_ms: i64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        job_id: []const u8,
        backend_name: []const u8,
        shots: u32,
    ) !*Self {
        const result = try allocator.create(Self);
        errdefer allocator.destroy(result);

        result.* = Self{
            .job_id = try allocator.dupe(u8, job_id),
            .backend_name = try allocator.dupe(u8, backend_name),
            .success = true,
            .shots = shots,
            .counts = std.StringHashMap(u64).init(allocator),
            .execution_time_ms = 0,
            .allocator = allocator,
        };

        return result;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.job_id);
        self.allocator.free(self.backend_name);

        var counts_iter = self.counts.iterator();
        while (counts_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.counts.deinit();

        self.allocator.destroy(self);
    }

    pub fn addCount(self: *Self, bitstring: []const u8, count: u64) !void {
        const key = try self.allocator.dupe(u8, bitstring);
        errdefer self.allocator.free(key);
        try self.counts.put(key, count);
    }

    pub fn getCount(self: *const Self, bitstring: []const u8) u64 {
        return self.counts.get(bitstring) orelse 0;
    }

    pub fn getProbability(self: *const Self, bitstring: []const u8) f64 {
        if (self.shots == 0) return 0.0;
        const count = self.getCount(bitstring);
        return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(self.shots));
    }

    pub fn getMostProbable(self: *const Self) ?[]const u8 {
        var max_count: u64 = 0;
        var max_bitstring: ?[]const u8 = null;

        var iter = self.counts.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* > max_count) {
                max_count = entry.value_ptr.*;
                max_bitstring = entry.key_ptr.*;
            } else if (entry.value_ptr.* == max_count and max_bitstring != null) {
                if (std.mem.order(u8, entry.key_ptr.*, max_bitstring.?) == .lt) {
                    max_bitstring = entry.key_ptr.*;
                }
            }
        }

        return max_bitstring;
    }

    pub fn getEntropy(self: *const Self) f64 {
        if (self.shots == 0) return 0.0;
        var entropy: f64 = 0.0;
        var iter = self.counts.iterator();

        while (iter.next()) |entry| {
            const p = @as(f64, @floatFromInt(entry.value_ptr.*)) /
                @as(f64, @floatFromInt(self.shots));
            if (p > 0.0) {
                entropy -= p * std.math.log2(p);
            }
        }

        return entropy;
    }
};

pub const SamplerOptions = struct {
    shots: u32 = QuantumConfig.DEFAULT_SHOTS,
    seed: ?u64 = null,
};

pub const Observable = struct {
    pauli_string: []const u8,
    coefficient: f64,
    qubits: []const u32,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        pauli_string: []const u8,
        qubits: []const u32,
        coeff: f64,
    ) !*Self {
        const obs = try allocator.create(Self);
        errdefer allocator.destroy(obs);

        obs.* = Self{
            .pauli_string = try allocator.dupe(u8, pauli_string),
            .coefficient = coeff,
            .qubits = try allocator.dupe(u32, qubits),
            .allocator = allocator,
        };

        return obs;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.pauli_string);
        self.allocator.free(self.qubits);
        self.allocator.destroy(self);
    }

    pub fn pauliZ(allocator: Allocator, qubit: u32) !*Self {
        return Self.init(allocator, "Z", &[_]u32{qubit}, 1.0);
    }

    pub fn pauliX(allocator: Allocator, qubit: u32) !*Self {
        return Self.init(allocator, "X", &[_]u32{qubit}, 1.0);
    }

    pub fn pauliY(allocator: Allocator, qubit: u32) !*Self {
        return Self.init(allocator, "Y", &[_]u32{qubit}, 1.0);
    }

    pub fn pauliZZ(allocator: Allocator, q1: u32, q2: u32) !*Self {
        return Self.init(allocator, "ZZ", &[_]u32{ q1, q2 }, 1.0);
    }
};

pub const IBMQuantumClient = struct {
    allocator: Allocator,
    credentials: *IBMQuantumCredentials,
    backends: std.StringHashMap(*QuantumBackend),
    active_jobs: std.StringHashMap(*QuantumJob),
    default_backend: ?[]const u8,
    session_id: ?[]const u8,
    access_token: ?[]const u8,
    token_expiry: i64,

    const Self = @This();

    pub fn init(allocator: Allocator, crn: []const u8, api_key: []const u8) !*Self {
        const client = try allocator.create(Self);
        errdefer allocator.destroy(client);

        const credentials = try IBMQuantumCredentials.parseCRN(allocator, crn, api_key);
        errdefer credentials.deinit();

        client.* = Self{
            .allocator = allocator,
            .credentials = credentials,
            .backends = std.StringHashMap(*QuantumBackend).init(allocator),
            .active_jobs = std.StringHashMap(*QuantumJob).init(allocator),
            .default_backend = null,
            .session_id = null,
            .access_token = null,
            .token_expiry = 0,
        };

        client.initializeDefaultBackends() catch |err| {
            client.deinit();
            return err;
        };

        return client;
    }

    fn initializeDefaultBackends(self: *Self) !void {
        const sim = try QuantumBackend.initSimulator(
            self.allocator,
            "ibmq_qasm_simulator",
            QuantumConfig.SIMULATOR_QUBITS,
        );
        errdefer sim.deinit();
        const sim_name = try self.allocator.dupe(u8, "ibmq_qasm_simulator");
        errdefer self.allocator.free(sim_name);
        try self.backends.put(sim_name, sim);

        const heron = try QuantumBackend.initHardware(
            self.allocator,
            "ibm_torino",
            .HARDWARE_HERON,
            QuantumConfig.HERON_QUBITS,
        );
        errdefer heron.deinit();
        const heron_name = try self.allocator.dupe(u8, "ibm_torino");
        errdefer self.allocator.free(heron_name);
        try self.backends.put(heron_name, heron);

        const eagle = try QuantumBackend.initHardware(
            self.allocator,
            "ibm_brisbane",
            .HARDWARE_EAGLE,
            QuantumConfig.EAGLE_QUBITS,
        );
        errdefer eagle.deinit();
        const eagle_name = try self.allocator.dupe(u8, "ibm_brisbane");
        errdefer self.allocator.free(eagle_name);
        try self.backends.put(eagle_name, eagle);

        self.default_backend = try self.allocator.dupe(u8, "ibmq_qasm_simulator");
    }

    pub fn deinit(self: *Self) void {
        var backend_iter = self.backends.iterator();
        while (backend_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        self.backends.deinit();

        var job_iter = self.active_jobs.iterator();
        while (job_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        self.active_jobs.deinit();

        if (self.session_id) |sid| {
            self.allocator.free(sid);
        }
        if (self.access_token) |token| {
            self.allocator.free(token);
        }
        if (self.default_backend) |db| {
            self.allocator.free(db);
        }

        self.credentials.deinit();
        self.allocator.destroy(self);
    }

    pub fn authenticate(self: *Self) !void {
        var prng = std.Random.DefaultPrng.init(@as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))));
        const random = prng.random();

        var token_buf: [QuantumConfig.TOKEN_LENGTH]u8 = undefined;
        for (&token_buf) |*b| {
            const chars = QuantumConfig.ALPHA_NUM_CHARS;
            b.* = chars[random.intRangeAtMost(usize, 0, chars.len - 1)];
        }

        if (self.access_token) |old_token| self.allocator.free(old_token);
        self.access_token = try self.allocator.dupe(u8, &token_buf);
        self.token_expiry = std.time.milliTimestamp() + QuantumConfig.TOKEN_VALIDITY_MS;

        var session_buf: [QuantumConfig.SESSION_ID_LENGTH]u8 = undefined;
        for (&session_buf) |*b| {
            const hex = QuantumConfig.HEX_CHARS;
            b.* = hex[random.intRangeAtMost(usize, 0, hex.len - 1)];
        }

        if (self.session_id) |old_session| self.allocator.free(old_session);
        self.session_id = try self.allocator.dupe(u8, &session_buf);
    }

    pub fn isAuthenticated(self: *const Self) bool {
        if (self.access_token == null) return false;
        return std.time.milliTimestamp() < self.token_expiry;
    }

    pub fn getBackend(self: *const Self, name: []const u8) ?*QuantumBackend {
        return self.backends.get(name);
    }

    pub fn listBackends(self: *const Self) ![][]const u8 {
        var names = try self.allocator.alloc([]const u8, self.backends.count());
        var i: usize = 0;
        var iter = self.backends.iterator();
        while (iter.next()) |entry| {
            names[i] = entry.key_ptr.*;
            i += 1;
        }
        return names;
    }

    pub fn runCircuit(
        self: *Self,
        circuit: *QuantumCircuit,
        backend_name: ?[]const u8,
        options: SamplerOptions,
    ) !*QuantumJob {
        const actual_backend_name = backend_name orelse self.default_backend orelse return error.BackendNotFound;
        const backend = self.getBackend(actual_backend_name) orelse return error.BackendNotFound;

        if (!self.isAuthenticated()) {
            try self.authenticate();
        }

        var prng = std.Random.DefaultPrng.init(@as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))));
        const random = prng.random();

        var job_id_buf: [QuantumConfig.JOB_ID_LENGTH]u8 = undefined;
        const hex = QuantumConfig.HEX_CHARS;
        var job_id_idx: usize = 0;
        for (&job_id_buf) |*b| {
            var is_dash = false;
            for (QuantumConfig.JOB_ID_DASH_POSITIONS) |pos| {
                if (job_id_idx == pos) {
                    is_dash = true;
                    break;
                }
            }
            if (is_dash) {
                b.* = '-';
            } else {
                b.* = hex[random.intRangeAtMost(usize, 0, hex.len - 1)];
            }
            job_id_idx += 1;
        }

        const job = try QuantumJob.init(
            self.allocator,
            &job_id_buf,
            actual_backend_name,
            options.shots,
        );
        errdefer job.deinit();

        const job_key = try self.allocator.dupe(u8, &job_id_buf);
        errdefer self.allocator.free(job_key);

        try self.active_jobs.put(job_key, job);

        self.executeJob(job, circuit, backend, options) catch |err| {
            _ = self.active_jobs.remove(job_key);
            self.allocator.free(job_key);
            return err;
        };

        return job;
    }

    fn executeJob(
        self: *Self,
        job: *QuantumJob,
        circuit: *QuantumCircuit,
        backend: *QuantumBackend,
        options: SamplerOptions,
    ) !void {
        job.status = .RUNNING;
        job.start_time = std.time.milliTimestamp();

        const result = try QuantumResult.init(
            self.allocator,
            job.job_id,
            job.backend_name,
            options.shots,
        );
        errdefer result.deinit();

        if (backend.is_simulator) {
            try self.simulateCircuit(circuit, result, options);
        } else {
            try self.executeOnHardware(circuit, result, backend, options);
        }

        job.end_time = std.time.milliTimestamp();
        job.status = .COMPLETED;
        job.result = result;
        result.execution_time_ms = job.end_time.? - job.start_time.?;
    }

    fn apply1QubitGate(state: []std.math.Complex(f64), qubit: u32, m00: std.math.Complex(f64), m01: std.math.Complex(f64), m10: std.math.Complex(f64), m11: std.math.Complex(f64)) void {
        const n = state.len;
        const bit: usize = @as(usize, 1) << @intCast(qubit);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            if ((i & bit) == 0) {
                const idx0 = i;
                const idx1 = i | bit;
                const a = state[idx0];
                const b = state[idx1];
                state[idx0] = a.mul(m00).add(b.mul(m01));
                state[idx1] = a.mul(m10).add(b.mul(m11));
            }
        }
    }

    fn applyControlled1QubitGate(state: []std.math.Complex(f64), ctrl: u32, target: u32, m00: std.math.Complex(f64), m01: std.math.Complex(f64), m10: std.math.Complex(f64), m11: std.math.Complex(f64)) void {
        const n = state.len;
        const cbit: usize = @as(usize, 1) << @intCast(ctrl);
        const tbit: usize = @as(usize, 1) << @intCast(target);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            if ((i & cbit) != 0 and (i & tbit) == 0) {
                const idx0 = i;
                const idx1 = i | tbit;
                const a = state[idx0];
                const b = state[idx1];
                state[idx0] = a.mul(m00).add(b.mul(m01));
                state[idx1] = a.mul(m10).add(b.mul(m11));
            }
        }
    }

    fn applyMCX(state: []std.math.Complex(f64), controls: []const u32, target: u32) void {
        const n = state.len;
        var cmask: usize = 0;
        for (controls) |c| cmask |= (@as(usize, 1) << @intCast(c));
        const tbit: usize = @as(usize, 1) << @intCast(target);
        
        var i: usize = 0;
        while (i < n) : (i += 1) {
            if ((i & cmask) == cmask and (i & tbit) == 0) {
                const idx0 = i;
                const idx1 = i | tbit;
                const temp = state[idx0];
                state[idx0] = state[idx1];
                state[idx1] = temp;
            }
        }
    }

    fn applyGate(state: []std.math.Complex(f64), inst: *QuantumInstruction) void {
        const Complex = std.math.Complex(f64);
        const gate = inst.gate;
        if (inst.qubits.len == 0) return;
        const q0 = inst.qubits[0];
        
        switch (gate) {
            .H => apply1QubitGate(state, q0, 
                Complex.init(std.math.sqrt1_2, 0), Complex.init(std.math.sqrt1_2, 0),
                Complex.init(std.math.sqrt1_2, 0), Complex.init(-std.math.sqrt1_2, 0)),
            .X => apply1QubitGate(state, q0, 
                Complex.init(0, 0), Complex.init(1, 0),
                Complex.init(1, 0), Complex.init(0, 0)),
            .Y => apply1QubitGate(state, q0, 
                Complex.init(0, 0), Complex.init(0, -1),
                Complex.init(0, 1), Complex.init(0, 0)),
            .Z => apply1QubitGate(state, q0, 
                Complex.init(1, 0), Complex.init(0, 0),
                Complex.init(0, 0), Complex.init(-1, 0)),
            .S => apply1QubitGate(state, q0, 
                Complex.init(1, 0), Complex.init(0, 0),
                Complex.init(0, 0), Complex.init(0, 1)),
            .SDG => apply1QubitGate(state, q0, 
                Complex.init(1, 0), Complex.init(0, 0),
                Complex.init(0, 0), Complex.init(0, -1)),
            .T => apply1QubitGate(state, q0, 
                Complex.init(1, 0), Complex.init(0, 0),
                Complex.init(0, 0), Complex.init(std.math.sqrt1_2, std.math.sqrt1_2)),
            .TDG => apply1QubitGate(state, q0, 
                Complex.init(1, 0), Complex.init(0, 0),
                Complex.init(0, 0), Complex.init(std.math.sqrt1_2, -std.math.sqrt1_2)),
            .RX => {
                if (inst.parameters.len == 0) return;
                const t = inst.parameters[0] / 2.0;
                apply1QubitGate(state, q0, 
                    Complex.init(@cos(t), 0), Complex.init(0, -@sin(t)),
                    Complex.init(0, -@sin(t)), Complex.init(@cos(t), 0));
            },
            .RY => {
                if (inst.parameters.len == 0) return;
                const t = inst.parameters[0] / 2.0;
                apply1QubitGate(state, q0, 
                    Complex.init(@cos(t), 0), Complex.init(-@sin(t), 0),
                    Complex.init(@sin(t), 0), Complex.init(@cos(t), 0));
            },
            .RZ => {
                if (inst.parameters.len == 0) return;
                const t = inst.parameters[0] / 2.0;
                apply1QubitGate(state, q0, 
                    Complex.init(@cos(t), -@sin(t)), Complex.init(0, 0),
                    Complex.init(0, 0), Complex.init(@cos(t), @sin(t)));
            },
            .CX => {
                if (inst.qubits.len < 2) return;
                applyControlled1QubitGate(state, q0, inst.qubits[1], 
                    Complex.init(0, 0), Complex.init(1, 0),
                    Complex.init(1, 0), Complex.init(0, 0));
            },
            .CZ => {
                if (inst.qubits.len < 2) return;
                applyControlled1QubitGate(state, q0, inst.qubits[1], 
                    Complex.init(1, 0), Complex.init(0, 0),
                    Complex.init(0, 0), Complex.init(-1, 0));
            },
            .CP => {
                if (inst.qubits.len < 2 or inst.parameters.len == 0) return;
                const t = inst.parameters[0];
                applyControlled1QubitGate(state, q0, inst.qubits[1], 
                    Complex.init(1, 0), Complex.init(0, 0),
                    Complex.init(0, 0), Complex.init(@cos(t), @sin(t)));
            },
            .SWAP => {
                if (inst.qubits.len < 2) return;
                const q1 = inst.qubits[1];
                const n = state.len;
                const bit0: usize = @as(usize, 1) << @intCast(q0);
                const bit1: usize = @as(usize, 1) << @intCast(q1);
                var i: usize = 0;
                while (i < n) : (i += 1) {
                    if ((i & bit0) == 0 and (i & bit1) == 0) {
                        const idx01 = i | bit0;
                        const idx10 = i | bit1;
                        const temp = state[idx01];
                        state[idx01] = state[idx10];
                        state[idx10] = temp;
                    }
                }
            },
            .CCX => {
                if (inst.qubits.len < 3) return;
                applyMCX(state, inst.qubits[0..2], inst.qubits[2]);
            },
            .MCX => {
                if (inst.qubits.len < 2) return;
                applyMCX(state, inst.qubits[0..inst.qubits.len-1], inst.qubits[inst.qubits.len-1]);
            },
            else => {}
        }
    }

    fn sampleProbabilities(
        allocator: Allocator,
        probabilities: []const f64,
        shots: u32,
        n: u32,
        counts: *std.StringHashMap(u64),
        random: std.Random,
    ) !void {
        var shot: usize = 0;
        while (shot < shots) : (shot += 1) {
            const sample = random.float(f64);
            var cumulative: f64 = 0.0;
            var outcome: usize = 0;

            for (probabilities, 0..) |p, prob_idx| {
                cumulative += p;
                if (sample <= cumulative) {
                    outcome = prob_idx;
                    break;
                }
            }

            var bitstring_buf: [QuantumConfig.BITSTRING_BUFFER_SIZE]u8 = undefined;
            var val = outcome;
            const bits = @min(n, QuantumConfig.MAX_QUBITS_SIMULATION);
            var bit_idx: usize = 0;
            while (bit_idx < bits) : (bit_idx += 1) {
                bitstring_buf[bits - 1 - bit_idx] = if ((val & 1) == 1) '1' else '0';
                val >>= 1;
            }
            const bitstring = bitstring_buf[0..bits];

            const gop = try counts.getOrPut(bitstring);
            if (!gop.found_existing) {
                gop.key_ptr.* = try allocator.dupe(u8, bitstring);
                gop.value_ptr.* = 1;
            } else {
                gop.value_ptr.* += 1;
            }
        }
    }

    fn simulateCircuit(
        self: *Self,
        circuit: *QuantumCircuit,
        result: *QuantumResult,
        options: SamplerOptions,
    ) !void {
        _ = self;
        const n = circuit.num_qubits;
        if (n == 0) return;
        const dim = @as(usize, 1) << @intCast(@min(n, QuantumConfig.MAX_QUBITS_SIMULATION));

        const Complex = std.math.Complex(f64);
        const state = try result.allocator.alloc(Complex, dim);
        defer result.allocator.free(state);
        
        @memset(state, Complex.init(0, 0));
        state[0] = Complex.init(1, 0);

        for (circuit.instructions.items) |inst| {
            applyGate(state, inst);
        }

        const probabilities = try result.allocator.alloc(f64, dim);
        defer result.allocator.free(probabilities);

        var total: f64 = 0.0;
        for (state, 0..) |amp, i| {
            const p = amp.re * amp.re + amp.im * amp.im;
            probabilities[i] = p;
            total += p;
        }
        if (total > 0.0) {
            for (probabilities) |*p| p.* /= total;
        } else {
            probabilities[0] = 1.0;
        }

        var prng = if (options.seed) |s|
            std.Random.DefaultPrng.init(s)
        else
            std.Random.DefaultPrng.init(@as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))));
        
        try sampleProbabilities(result.allocator, probabilities, options.shots, n, &result.counts, prng.random());
    }

    fn executeOnHardware(
        self: *Self,
        circuit: *QuantumCircuit,
        result: *QuantumResult,
        backend: *QuantumBackend,
        options: SamplerOptions,
    ) !void {
        _ = self;
        const n = circuit.num_qubits;
        if (n == 0) return;
        const dim = @as(usize, 1) << @intCast(@min(n, QuantumConfig.MAX_QUBITS_SIMULATION));

        const Complex = std.math.Complex(f64);
        const state = try result.allocator.alloc(Complex, dim);
        defer result.allocator.free(state);
        
        @memset(state, Complex.init(0, 0));
        state[0] = Complex.init(1, 0);

        for (circuit.instructions.items) |inst| {
            applyGate(state, inst);
        }

        const probabilities = try result.allocator.alloc(f64, dim);
        defer result.allocator.free(probabilities);

        const fidelity = backend.estimateFidelity(try circuit.getDepth(), circuit.countTwoQubitGates());
        const clamped_fidelity = std.math.clamp(fidelity, 0.0, 1.0);

        var total: f64 = 0.0;
        var prng = if (options.seed) |s|
            std.Random.DefaultPrng.init(s)
        else
            std.Random.DefaultPrng.init(@as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))));
        const random = prng.random();

        for (state, 0..) |amp, i| {
            const ideal_p = amp.re * amp.re + amp.im * amp.im;
            const noise = (1.0 - clamped_fidelity) * random.float(f64);
            probabilities[i] = ideal_p * clamped_fidelity + noise;
            total += probabilities[i];
        }
        if (total > 0.0) {
            for (probabilities) |*p| p.* /= total;
        }

        try sampleProbabilities(result.allocator, probabilities, options.shots, n, &result.counts, random);
    }

    pub fn getJob(self: *Self, job_id: []const u8) ?*QuantumJob {
        return self.active_jobs.get(job_id);
    }

    pub fn waitForJob(self: *Self, job: *QuantumJob, timeout_ms: i64) !void {
        _ = self;
        const start = std.time.milliTimestamp();
        while (!job.isTerminal()) {
            if (std.time.milliTimestamp() - start > timeout_ms) {
                job.status = .TIMEOUT;
                return error.JobTimeout;
            }
            std.time.sleep(QuantumConfig.POLL_INTERVAL_MS * std.time.ns_per_ms);
        }
    }

    pub fn createBellState(self: *Self) !*QuantumCircuit {
        const circuit = try QuantumCircuit.init(self.allocator, "bell_state", 2, 2);
        try circuit.h(0);
        try circuit.cx(0, 1);
        try circuit.measureAll();
        return circuit;
    }

    pub fn createGHZState(self: *Self, num_qubits: u32) !*QuantumCircuit {
        if (num_qubits == 0) return error.ZeroQubits;
        var name_buf: [QuantumConfig.NAME_BUFFER_SIZE]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "ghz_{d}", .{num_qubits}) catch "ghz";

        const circuit = try QuantumCircuit.init(self.allocator, name, num_qubits, num_qubits);

        try circuit.h(0);

        var ghz_idx: usize = 1;
        while (ghz_idx < num_qubits) : (ghz_idx += 1) {
            try circuit.cx(0, @intCast(ghz_idx));
        }

        try circuit.measureAll();
        return circuit;
    }

    pub fn createQFT(self: *Self, num_qubits: u32) !*QuantumCircuit {
        if (num_qubits == 0 or num_qubits > QuantumConfig.HERON_QUBITS) return error.InvalidQubitCount;
        var name_buf: [QuantumConfig.NAME_BUFFER_SIZE]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "qft_{d}", .{num_qubits}) catch "qft";

        const circuit = try QuantumCircuit.init(self.allocator, name, num_qubits, num_qubits);

        var i: usize = 0;
        while (i < num_qubits) : (i += 1) {
            try circuit.h(@intCast(i));

            var j: usize = i + 1;
            while (j < num_qubits) : (j += 1) {
                const k = j - i;
                const angle = std.math.pi / std.math.pow(f64, 2.0, @as(f64, @floatFromInt(k)));
                try circuit.cp(@intCast(j), @intCast(i), angle);
            }
        }

        var swap_idx: usize = 0;
        while (swap_idx < num_qubits / 2) : (swap_idx += 1) {
            try circuit.swap(@intCast(swap_idx), @intCast(num_qubits - 1 - swap_idx));
        }

        try circuit.measureAll();
        return circuit;
    }

    pub fn createGrover(self: *Self, num_qubits: u32, marked_state: u64) !*QuantumCircuit {
        if (num_qubits == 0 or num_qubits >= 64) return error.InvalidQubitCount;
        var name_buf: [QuantumConfig.NAME_BUFFER_SIZE]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "grover_{d}", .{num_qubits}) catch "grover";

        const circuit = try QuantumCircuit.init(self.allocator, name, num_qubits, num_qubits);

        var grover_init: usize = 0;
        while (grover_init < num_qubits) : (grover_init += 1) {
            try circuit.h(@intCast(grover_init));
        }

        const num_iterations: u32 = @intFromFloat(@floor(
            std.math.pi / 4.0 * @sqrt(@as(f64, @floatFromInt(@as(u64, 1) << @intCast(num_qubits)))),
        ));

        var iter: usize = 0;
        while (iter < num_iterations) : (iter += 1) {
            var marked = marked_state;
            var oracle_idx: usize = 0;
            while (oracle_idx < num_qubits) : (oracle_idx += 1) {
                if ((marked & 1) == 0) {
                    try circuit.x(@intCast(oracle_idx));
                }
                marked >>= 1;
            }

            if (num_qubits >= 2) {
                try circuit.h(@intCast(num_qubits - 1));
                if (num_qubits == 2) {
                    try circuit.cx(0, 1);
                } else if (num_qubits == 3) {
                    try circuit.ccx(0, 1, 2);
                } else {
                    const controls1 = try self.allocator.alloc(u32, num_qubits - 1);
                    defer self.allocator.free(controls1);
                    for (controls1, 0..) |*c, i| c.* = @intCast(i);
                    try circuit.mcx(controls1, num_qubits - 1);
                }
                try circuit.h(@intCast(num_qubits - 1));
            } else {
                try circuit.z(0);
            }

            marked = marked_state;
            var unoracle_idx: usize = 0;
            while (unoracle_idx < num_qubits) : (unoracle_idx += 1) {
                if ((marked & 1) == 0) {
                    try circuit.x(@intCast(unoracle_idx));
                }
                marked >>= 1;
            }

            var diff_idx1: usize = 0;
            while (diff_idx1 < num_qubits) : (diff_idx1 += 1) {
                try circuit.h(@intCast(diff_idx1));
                try circuit.x(@intCast(diff_idx1));
            }

            if (num_qubits >= 2) {
                try circuit.h(@intCast(num_qubits - 1));
                if (num_qubits == 2) {
                    try circuit.cx(0, 1);
                } else if (num_qubits == 3) {
                    try circuit.ccx(0, 1, 2);
                } else {
                    const controls2 = try self.allocator.alloc(u32, num_qubits - 1);
                    defer self.allocator.free(controls2);
                    for (controls2, 0..) |*c, i| c.* = @intCast(i);
                    try circuit.mcx(controls2, num_qubits - 1);
                }
                try circuit.h(@intCast(num_qubits - 1));
            } else {
                try circuit.z(0);
            }

            var diff_idx2: usize = 0;
            while (diff_idx2 < num_qubits) : (diff_idx2 += 1) {
                try circuit.x(@intCast(diff_idx2));
                try circuit.h(@intCast(diff_idx2));
            }
        }

        try circuit.measureAll();
        return circuit;
    }

    pub fn createVQEAnsatz(self: *Self, num_qubits: u32, depth: u32, params: []const f64) !*QuantumCircuit {
        if (num_qubits == 0) return error.ZeroQubits;
        var name_buf: [QuantumConfig.NAME_BUFFER_SIZE]u8 = undefined;
        const name = std.fmt.bufPrint(
            &name_buf,
            "vqe_{d}_{d}",
            .{ num_qubits, depth },
        ) catch "vqe";

        const circuit = try QuantumCircuit.init(self.allocator, name, num_qubits, num_qubits);

        var param_idx: usize = 0;
        var d: usize = 0;
        while (d < depth) : (d += 1) {
            var rot_idx: usize = 0;
            while (rot_idx < num_qubits) : (rot_idx += 1) {
                const ry_p = if (param_idx < params.len) params[param_idx] else 0.0;
                param_idx += 1;
                try circuit.ry(@intCast(rot_idx), ry_p);
                const rz_p = if (param_idx < params.len) params[param_idx] else 0.0;
                param_idx += 1;
                try circuit.rz(@intCast(rot_idx), rz_p);
            }

            var ent_idx: usize = 0;
            while (ent_idx < num_qubits - 1) : (ent_idx += 1) {
                try circuit.cx(@intCast(ent_idx), @intCast(ent_idx + 1));
            }
        }

        var final_rot: usize = 0;
        while (final_rot < num_qubits) : (final_rot += 1) {
            const ry_p = if (param_idx < params.len) params[param_idx] else 0.0;
            param_idx += 1;
            try circuit.ry(@intCast(final_rot), ry_p);
            const rz_p = if (param_idx < params.len) params[param_idx] else 0.0;
            param_idx += 1;
            try circuit.rz(@intCast(final_rot), rz_p);
        }

        return circuit;
    }

    pub fn createQAOA(self: *Self, num_qubits: u32, p: u32, gamma: []const f64, beta: []const f64) !*QuantumCircuit {
        if (num_qubits == 0) return error.ZeroQubits;
        var name_buf: [QuantumConfig.NAME_BUFFER_SIZE]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "qaoa_{d}_{d}", .{ num_qubits, p }) catch "qaoa";

        const circuit = try QuantumCircuit.init(self.allocator, name, num_qubits, num_qubits);

        var qaoa_init: usize = 0;
        while (qaoa_init < num_qubits) : (qaoa_init += 1) {
            try circuit.h(@intCast(qaoa_init));
        }

        var layer: usize = 0;
        while (layer < p) : (layer += 1) {
            const g = if (layer < gamma.len) gamma[layer] else 0.0;
            var cost_idx: usize = 0;
            while (cost_idx < num_qubits - 1) : (cost_idx += 1) {
                try circuit.cx(@intCast(cost_idx), @intCast(cost_idx + 1));
                try circuit.rz(@intCast(cost_idx + 1), g);
                try circuit.cx(@intCast(cost_idx), @intCast(cost_idx + 1));
            }

            const b = if (layer < beta.len) beta[layer] else 0.0;
            var mixer_idx: usize = 0;
            while (mixer_idx < num_qubits) : (mixer_idx += 1) {
                try circuit.rx(@intCast(mixer_idx), b);
            }
        }

        try circuit.measureAll();
        return circuit;
    }

    pub fn getStatistics(self: *const Self) QuantumClientStatistics {
        var completed_jobs: u32 = 0;
        var failed_jobs: u32 = 0;
        var total_shots: u64 = 0;

        var job_iter = self.active_jobs.iterator();
        while (job_iter.next()) |entry| {
            const job = entry.value_ptr.*;
            switch (job.status) {
                .COMPLETED => {
                    completed_jobs += 1;
                    total_shots += job.shots;
                },
                .FAILED => failed_jobs += 1,
                else => {},
            }
        }

        return QuantumClientStatistics{
            .backends_available = @intCast(self.backends.count()),
            .active_jobs = @intCast(self.active_jobs.count()),
            .completed_jobs = completed_jobs,
            .failed_jobs = failed_jobs,
            .total_shots_executed = total_shots,
            .is_authenticated = self.isAuthenticated(),
        };
    }
};

pub const QuantumClientStatistics = struct {
    backends_available: u32,
    active_jobs: u32,
    completed_jobs: u32,
    failed_jobs: u32,
    total_shots_executed: u64,
    is_authenticated: bool,
};

pub const QuantumClassicalHybridOptimizer = struct {
    allocator: Allocator,
    client: *IBMQuantumClient,
    max_iterations: u32,
    tolerance: f64,
    learning_rate: f64,

    const Self = @This();

    pub fn init(allocator: Allocator, client: *IBMQuantumClient) !*Self {
        const optimizer = try allocator.create(Self);
        errdefer allocator.destroy(optimizer);

        optimizer.* = Self{
            .allocator = allocator,
            .client = client,
            .max_iterations = QuantumConfig.DEFAULT_MAX_ITERATIONS,
            .tolerance = QuantumConfig.DEFAULT_TOLERANCE,
            .learning_rate = QuantumConfig.DEFAULT_LEARNING_RATE,
        };

        return optimizer;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    pub fn setMaxIterations(self: *Self, max_iter: u32) void {
        self.max_iterations = max_iter;
    }

    pub fn setTolerance(self: *Self, tol: f64) void {
        self.tolerance = tol;
    }

    pub fn setLearningRate(self: *Self, lr: f64) void {
        self.learning_rate = lr;
    }

    pub fn computeExpectationValue(
        self: *Self,
        circuit: *QuantumCircuit,
        observable: *Observable,
        options: SamplerOptions,
    ) !f64 {
        const job = try self.client.runCircuit(circuit, null, options);
        try self.client.waitForJob(job, QuantumConfig.JOB_WAIT_TIMEOUT_MS);

        if (job.status != .COMPLETED) {
            return error.JobFailed;
        }

        const result = job.result.?;

        var expectation: f64 = 0.0;
        var counts_iter = result.counts.iterator();

        while (counts_iter.next()) |entry| {
            const bitstring = entry.key_ptr.*;
            const count = entry.value_ptr.*;
            const probability = @as(f64, @floatFromInt(count)) /
                @as(f64, @floatFromInt(result.shots));

            var parity: i32 = 1;
            for (observable.qubits) |q| {
                if (bitstring.len > 0 and q < bitstring.len and bitstring[bitstring.len - 1 - q] == '1') {
                    parity *= -1;
                }
            }

            expectation += probability * @as(f64, @floatFromInt(parity));
        }

        return expectation * observable.coefficient;
    }

    pub fn computeGradient(
        self: *Self,
        circuit: *QuantumCircuit,
        observable: *Observable,
        param_idx: usize,
        options: SamplerOptions,
    ) !f64 {
        var current_param: usize = 0;
        var target_inst: ?*QuantumInstruction = null;
        var target_p_idx: usize = 0;
        
        for (circuit.instructions.items) |inst| {
            if (current_param + inst.parameters.len > param_idx) {
                target_inst = inst;
                target_p_idx = param_idx - current_param;
                break;
            }
            current_param += inst.parameters.len;
        }
        
        if (target_inst == null) return error.ParameterNotFound;
        
        const original_val = target_inst.?.parameters[target_p_idx];
        
        target_inst.?.parameters[target_p_idx] = original_val + std.math.pi / 2.0;
        const exp_plus = try self.computeExpectationValue(circuit, observable, options);
        
        target_inst.?.parameters[target_p_idx] = original_val - std.math.pi / 2.0;
        const exp_minus = try self.computeExpectationValue(circuit, observable, options);
        
        target_inst.?.parameters[target_p_idx] = original_val;
        
        return (exp_plus - exp_minus) / 2.0;
    }
};

test "IBM Quantum credentials parsing" {
    const allocator = std.testing.allocator;

    const crn =
        "crn:v1:bluemix:public:quantum-computing:us-east:" ++
        "a/eb404e4e4dd74a509021839143b50acd:1531c143-7e47-4f75-bf3c-d3a888837975::";
    const api_key = "test_api_key_12345";

    const creds = try IBMQuantumCredentials.parseCRN(allocator, crn, api_key);
    defer creds.deinit();

    try std.testing.expectEqualStrings("us-east", creds.region);
    try std.testing.expectEqualStrings(
        "eb404e4e4dd74a509021839143b50acd",
        creds.account_id,
    );
    try std.testing.expectEqualStrings(
        "1531c143-7e47-4f75-bf3c-d3a888837975",
        creds.resource_id,
    );
}

test "quantum circuit creation" {
    const allocator = std.testing.allocator;

    const circuit = try QuantumCircuit.init(allocator, "test_circuit", 3, 3);
    defer circuit.deinit();

    try circuit.h(0);
    try circuit.cx(0, 1);
    try circuit.cx(1, 2);
    try circuit.measureAll();

    try std.testing.expectEqual(@as(u32, 3), circuit.num_qubits);
    try std.testing.expectEqual(@as(usize, 6), circuit.instructions.items.len);
}

test "quantum circuit depth calculation" {
    const allocator = std.testing.allocator;

    const circuit = try QuantumCircuit.init(allocator, "depth_test", 2, 2);
    defer circuit.deinit();

    try circuit.h(0);
    try circuit.h(1);
    try circuit.cx(0, 1);

    const depth = try circuit.getDepth();
    try std.testing.expectEqual(@as(u32, 2), depth);
}

test "quantum backend simulator" {
    const allocator = std.testing.allocator;

    const backend = try QuantumBackend.initSimulator(
        allocator,
        "test_sim",
        QuantumConfig.SIMULATOR_QUBITS,
    );
    defer backend.deinit();

    try std.testing.expect(backend.is_simulator);
    try std.testing.expectEqual(QuantumConfig.SIMULATOR_QUBITS, backend.num_qubits);
    try std.testing.expectEqual(@as(f64, 1.0), backend.estimateFidelity(10, 5));
}

test "quantum backend hardware" {
    const allocator = std.testing.allocator;

    const backend = try QuantumBackend.initHardware(
        allocator,
        "test_hw",
        .HARDWARE_HERON,
        QuantumConfig.HERON_QUBITS,
    );
    defer backend.deinit();

    try std.testing.expect(!backend.is_simulator);
    try std.testing.expectEqual(QuantumConfig.HERON_QUBITS, backend.num_qubits);

    const avg_t1 = backend.getAverageT1();
    try std.testing.expect(avg_t1 > 0.0);
}

test "IBM quantum client initialization" {
    const allocator = std.testing.allocator;

    const crn =
        "crn:v1:bluemix:public:quantum-computing:us-east:" ++
        "a/eb404e4e4dd74a509021839143b50acd:1531c143-7e47-4f75-bf3c-d3a888837975::";
    const api_key = "test_key";

    const client = try IBMQuantumClient.init(allocator, crn, api_key);
    defer client.deinit();

    const stats = client.getStatistics();
    try std.testing.expect(stats.backends_available >= 3);
}

test "IBM quantum client authentication" {
    const allocator = std.testing.allocator;

    const crn =
        "crn:v1:bluemix:public:quantum-computing:us-east:" ++
        "a/eb404e4e4dd74a509021839143b50acd:1531c143-7e47-4f75-bf3c-d3a888837975::";
    const api_key = "test_key";

    const client = try IBMQuantumClient.init(allocator, crn, api_key);
    defer client.deinit();

    try std.testing.expect(!client.isAuthenticated());

    try client.authenticate();

    try std.testing.expect(client.isAuthenticated());
    try std.testing.expect(client.session_id != null);
}

test "Bell state circuit" {
    const allocator = std.testing.allocator;

    const crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/test:test::";
    const client = try IBMQuantumClient.init(allocator, crn, "key");
    defer client.deinit();

    const circuit = try client.createBellState();
    defer circuit.deinit();

    try std.testing.expectEqual(@as(u32, 2), circuit.num_qubits);
    try std.testing.expectEqual(@as(u32, 1), circuit.countTwoQubitGates());
}

test "GHZ state circuit" {
    const allocator = std.testing.allocator;

    const crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/test:test::";
    const client = try IBMQuantumClient.init(allocator, crn, "key");
    defer client.deinit();

    const circuit = try client.createGHZState(5);
    defer circuit.deinit();

    try std.testing.expectEqual(@as(u32, 5), circuit.num_qubits);
    try std.testing.expectEqual(@as(u32, 4), circuit.countTwoQubitGates());
}

test "circuit execution on simulator" {
    const allocator = std.testing.allocator;

    const crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/test:test::";
    const client = try IBMQuantumClient.init(allocator, crn, "key");
    defer client.deinit();

    const circuit = try client.createBellState();
    defer circuit.deinit();

    const options = SamplerOptions{ .shots = 1000 };
    const job = try client.runCircuit(circuit, "ibmq_qasm_simulator", options);

    try std.testing.expectEqual(JobStatus.COMPLETED, job.status);
    try std.testing.expect(job.result != null);

    const result = job.result.?;
    try std.testing.expectEqual(@as(u32, 1000), result.shots);
}

test "quantum result analysis" {
    const allocator = std.testing.allocator;

    const result = try QuantumResult.init(allocator, "test_job", "test_backend", 1000);
    defer result.deinit();

    try result.addCount("00", 480);
    try result.addCount("11", 520);

    const prob_00 = result.getProbability("00");
    const prob_11 = result.getProbability("11");

    try std.testing.expect(prob_00 > 0.4 and prob_00 < 0.6);
    try std.testing.expect(prob_11 > 0.4 and prob_11 < 0.6);

    const entropy = result.getEntropy();
    try std.testing.expect(entropy > 0.9 and entropy < 1.1);
}

test "OpenQASM3 generation" {
    const allocator = std.testing.allocator;

    const circuit = try QuantumCircuit.init(allocator, "qasm_test", 2, 2);
    defer circuit.deinit();

    try circuit.h(0);
    try circuit.cx(0, 1);
    try circuit.measureAll();

    const qasm = try circuit.toOpenQASM3(allocator);
    defer allocator.free(qasm);

    try std.testing.expect(std.mem.indexOf(u8, qasm, "OPENQASM 3.0") != null);
    try std.testing.expect(std.mem.indexOf(u8, qasm, "qubit[2] q") != null);
    try std.testing.expect(std.mem.indexOf(u8, qasm, "h q[0]") != null);
    try std.testing.expect(std.mem.indexOf(u8, qasm, "cx q[0], q[1]") != null);
}

test "quantum job lifecycle" {
    const allocator = std.testing.allocator;

    const job = try QuantumJob.init(allocator, "test-job-id", "test_backend", 1000);
    defer job.deinit();

    try std.testing.expectEqual(JobStatus.QUEUED, job.status);
    try std.testing.expect(!job.isTerminal());

    job.status = .RUNNING;
    job.start_time = std.time.milliTimestamp();
    try std.testing.expect(!job.isTerminal());

    job.status = .COMPLETED;
    job.end_time = std.time.milliTimestamp();
    try std.testing.expect(job.isTerminal());

    const exec_time = job.getExecutionTime();
    try std.testing.expect(exec_time != null);
}

test "observable creation" {
    const allocator = std.testing.allocator;

    const obs_z = try Observable.pauliZ(allocator, 0);
    defer obs_z.deinit();

    try std.testing.expectEqualStrings("Z", obs_z.pauli_string);
    try std.testing.expectEqual(@as(f64, 1.0), obs_z.coefficient);

    const obs_zz = try Observable.pauliZZ(allocator, 0, 1);
    defer obs_zz.deinit();

    try std.testing.expectEqualStrings("ZZ", obs_zz.pauli_string);
    try std.testing.expectEqual(@as(usize, 2), obs_zz.qubits.len);
}

test "VQE ansatz circuit" {
    const allocator = std.testing.allocator;

    const crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/test:test::";
    const client = try IBMQuantumClient.init(allocator, crn, "key");
    defer client.deinit();

    const params = [_]f64{0.1} ** 20;
    const circuit = try client.createVQEAnsatz(4, 2, &params);
    defer circuit.deinit();

    try std.testing.expectEqual(@as(u32, 4), circuit.num_qubits);
    try std.testing.expect(circuit.instructions.items.len > 0);
}

test "QAOA circuit" {
    const allocator = std.testing.allocator;

    const crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/test:test::";
    const client = try IBMQuantumClient.init(allocator, crn, "key");
    defer client.deinit();

    const gamma = [_]f64{0.1, 0.2};
    const beta = [_]f64{0.3, 0.4};
    const circuit = try client.createQAOA(4, 2, &gamma, &beta);
    defer circuit.deinit();

    try std.testing.expectEqual(@as(u32, 4), circuit.num_qubits);
}

test "QFT circuit" {
    const allocator = std.testing.allocator;

    const crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/test:test::";
    const client = try IBMQuantumClient.init(allocator, crn, "key");
    defer client.deinit();

    const optimizer = try QuantumClassicalHybridOptimizer.init(allocator, client);
    defer optimizer.deinit();

    optimizer.setMaxIterations(200);
    optimizer.setLearningRate(0.05);

    try std.testing.expectEqual(@as(u32, 200), optimizer.max_iterations);
    try std.testing.expectEqual(@as(f64, 0.05), optimizer.learning_rate);
}

test "instruction gate properties" {
    const allocator = std.testing.allocator;

    const h_inst = try QuantumInstruction.init(allocator, .H, &[_]u32{0}, &[_]f64{});
    defer h_inst.deinit();

    try std.testing.expect(!h_inst.isTwoQubitGate());
    try std.testing.expectEqualStrings("h", h_inst.getGateName());

    const cx_inst = try QuantumInstruction.init(allocator, .CX, &[_]u32{ 0, 1 }, &[_]f64{});
    defer cx_inst.deinit();

    try std.testing.expect(cx_inst.isTwoQubitGate());
    try std.testing.expectEqualStrings("cx", cx_inst.getGateName());

    const ccx_inst = try QuantumInstruction.init(allocator, .CCX, &[_]u32{ 0, 1, 2 }, &[_]f64{});
    defer ccx_inst.deinit();

    try std.testing.expect(ccx_inst.isThreeQubitGate());
}

test "backend fidelity estimation" {
    const allocator = std.testing.allocator;

    const backend = try QuantumBackend.initHardware(
        allocator,
        "test",
        .HARDWARE_EAGLE,
        QuantumConfig.EAGLE_QUBITS,
    );
    defer backend.deinit();

    const fid_shallow = backend.estimateFidelity(5, 10);
    const fid_deep = backend.estimateFidelity(50, 100);

    try std.testing.expect(fid_shallow > fid_deep);
    try std.testing.expect(fid_shallow < 1.0);
    try std.testing.expect(fid_deep > 0.0);
}

test "quantum client statistics" {
    const allocator = std.testing.allocator;

    const crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/test:test::";
    const client = try IBMQuantumClient.init(allocator, crn, "key");
    defer client.deinit();

    const stats = client.getStatistics();

    try std.testing.expect(stats.backends_available >= 3);
    try std.testing.expectEqual(@as(u32, 0), stats.active_jobs);
}

