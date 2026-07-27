const std = @import("std");
const nsir_core = @import("nsir_core.zig");

const Allocator = std.mem.Allocator;
const Complex = std.math.Complex;
const ArrayList = std.ArrayList;

const MAX_STRING_SCAN: usize = 65536;
const MAX_DATA_SCAN: usize = 1048576;
const FLOAT_EPSILON: f64 = std.math.floatEps(f64);
const EXP_UNDERFLOW_THRESHOLD: f64 = -745.0;

const VERSION_MAJOR: c_int = 4;
const VERSION_MINOR: c_int = 1;
const VERSION_PATCH: c_int = 0;

const DEFAULT_INITIAL_TEMP: f64 = 1.0;
const DEFAULT_COOLING_RATE: f64 = 0.95;
const DEFAULT_MAX_ITERATIONS: usize = 1000;
const DEFAULT_MIN_TEMP: f64 = 0.001;
const DEFAULT_REHEAT_THRESHOLD: f64 = 0.1;
const DEFAULT_REHEAT_FACTOR: f64 = 2.0;

const DEFAULT_EDGE_WEIGHT: f64 = 1.0;
const DEFAULT_EDGE_FRACTAL_DIM: f64 = 0.0;

const DEFAULT_PERTURB_EDGE_PROB: f64 = 0.3;
const DEFAULT_PERTURB_NODE_PROB: f64 = 0.2;
const DEFAULT_PERTURB_EDGE_FACTOR: f64 = 0.1;
const DEFAULT_PERTURB_NODE_FACTOR: f64 = 0.05;

const ACCEPTANCE_RATE_HIGH: f64 = 0.6;
const ACCEPTANCE_RATE_LOW: f64 = 0.2;
const COOLING_ADJUST_FAST: f64 = 0.98;
const COOLING_ADJUST_SLOW: f64 = 1.02;
const COOLING_RATE_MIN: f64 = 0.8;
const COOLING_RATE_MAX: f64 = 0.999;

pub const JAIDE_SUCCESS: c_int = 0;
pub const JAIDE_ERROR_NULL_POINTER: c_int = -1;
pub const JAIDE_ERROR_ALLOCATION: c_int = -2;
pub const JAIDE_ERROR_NODE_NOT_FOUND: c_int = -3;
pub const JAIDE_ERROR_EDGE_NOT_FOUND: c_int = -4;
pub const JAIDE_ERROR_INVALID_QUALITY: c_int = -5;
pub const JAIDE_ERROR_OPTIMIZATION_FAILED: c_int = -6;
pub const JAIDE_ERROR_INVALID_STRING: c_int = -7;
pub const JAIDE_ERROR_OPERATION_FAILED: c_int = -8;
pub const JAIDE_ERROR_DUPLICATE_NODE: c_int = -9;
pub const JAIDE_ERROR_DUPLICATE_EDGE: c_int = -10;
pub const JAIDE_ERROR_INVALID_PARAMETER: c_int = -11;
pub const JAIDE_ERROR_MATH_ERROR: c_int = -12;
pub const JAIDE_ERROR_NOT_INITIALIZED: c_int = -13;
pub const JAIDE_ERROR_SELF_REFERENCE: c_int = -14;
pub const JAIDE_ERROR_INVALID_STATE: c_int = -15;
pub const JAIDE_ERROR_THREADING: c_int = -16;
pub const JAIDE_ERROR_UNKNOWN_GATE: c_int = -17;
pub const JAIDE_ERROR_OUT_OF_MEMORY: c_int = -18;

pub const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
pub const Node = nsir_core.Node;
pub const Edge = nsir_core.Edge;
pub const EdgeQuality = nsir_core.EdgeQuality;
pub const EdgeKey = nsir_core.EdgeKey;

pub const CQuantumState = extern struct {
    real: f64,
    imag: f64,
};

const GraphContext = struct {
    inner: SelfSimilarRelationalGraph,
    lock: std.Thread.Mutex,
    allocator: Allocator,
};

pub const CGraph = opaque {
    pub fn fromInternal(ctx: *GraphContext) *CGraph {
        return @ptrCast(ctx);
    }
    pub fn toInternal(self: *CGraph) *GraphContext {
        return @ptrCast(@alignCast(self));
    }
    pub fn toInternalConst(self: *const CGraph) *const GraphContext {
        return @ptrCast(@alignCast(self));
    }
};

pub const COptimizer = opaque {
    pub fn fromInternal(opt: *EntangledStochasticSymmetryOptimizer) *COptimizer {
        return @ptrCast(opt);
    }
    pub fn toInternal(self: *COptimizer) *EntangledStochasticSymmetryOptimizer {
        return @ptrCast(@alignCast(self));
    }
};

pub const GateType = enum(c_int) {
    Identity = 0,
    Hadamard = 1,
    PauliX = 2,
    PauliY = 3,
    PauliZ = 4,
};

const PerturbationRecord = struct {
    type: enum { Node, Edge },
    target_id: []const u8,
    target_id2: []const u8,
    old_weight: f64,
    old_real: f64,
    old_imag: f64,
};

pub const OptimizationStatistics = struct {
    iterations_completed: usize,
    moves_accepted: usize,
    moves_rejected: usize,
    best_energy: f64,
    current_energy: f64,
    temperature: f64,
    acceptance_rate: f64,
    lock: std.Thread.Mutex,

    pub fn init() OptimizationStatistics {
        return .{
            .iterations_completed = 0,
            .moves_accepted = 0,
            .moves_rejected = 0,
            .best_energy = std.math.inf(f64),
            .current_energy = std.math.inf(f64),
            .temperature = 0.0,
            .acceptance_rate = 0.0,
            .lock = .{},
        };
    }

    pub fn updateAcceptanceRate(self: *OptimizationStatistics) void {
        self.lock.lock();
        defer self.lock.unlock();
        const total_moves = self.moves_accepted + self.moves_rejected;
        if (total_moves > 0) {
            self.acceptance_rate = @as(f64, @floatFromInt(self.moves_accepted)) / @as(f64, @floatFromInt(total_moves));
        } else {
            self.acceptance_rate = 0.0;
        }
    }

    pub fn recordAccepted(self: *OptimizationStatistics) void {
        self.lock.lock();
        defer self.lock.unlock();
        self.moves_accepted += 1;
    }

    pub fn recordRejected(self: *OptimizationStatistics) void {
        self.lock.lock();
        defer self.lock.unlock();
        self.moves_rejected += 1;
    }
};

fn complexMagnitude(c: Complex(f64)) f64 {
    return @sqrt(c.re * c.re + c.im * c.im);
}

pub const EntangledStochasticSymmetryOptimizer = struct {
    temperature: f64,
    initial_temperature: f64,
    min_temperature: f64,
    cooling_rate: f64,
    max_iterations: usize,
    perturb_edge_prob: f64,
    perturb_node_prob: f64,
    perturb_edge_factor: f64,
    perturb_node_factor: f64,
    reheat_threshold: f64,
    reheat_factor: f64,
    adaptive_cooling: bool,
    prng: std.Random.DefaultPrng,
    allocator: Allocator,
    statistics: OptimizationStatistics,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        initial_temp: f64,
        cooling_rate: f64,
        max_iterations: usize,
    ) Self {
        var seed: u64 = undefined;
        std.crypto.random.bytes(std.mem.asBytes(&seed));

        return Self{
            .temperature = if (std.math.isFinite(initial_temp)) initial_temp else DEFAULT_INITIAL_TEMP,
            .initial_temperature = if (std.math.isFinite(initial_temp)) initial_temp else DEFAULT_INITIAL_TEMP,
            .min_temperature = DEFAULT_MIN_TEMP,
            .cooling_rate = if (std.math.isFinite(cooling_rate)) cooling_rate else DEFAULT_COOLING_RATE,
            .max_iterations = if (max_iterations > 0) max_iterations else DEFAULT_MAX_ITERATIONS,
            .perturb_edge_prob = DEFAULT_PERTURB_EDGE_PROB,
            .perturb_node_prob = DEFAULT_PERTURB_NODE_PROB,
            .perturb_edge_factor = DEFAULT_PERTURB_EDGE_FACTOR,
            .perturb_node_factor = DEFAULT_PERTURB_NODE_FACTOR,
            .reheat_threshold = DEFAULT_REHEAT_THRESHOLD,
            .reheat_factor = DEFAULT_REHEAT_FACTOR,
            .adaptive_cooling = true,
            .prng = std.Random.DefaultPrng.init(seed),
            .allocator = allocator,
            .statistics = OptimizationStatistics.init(),
        };
    }

    pub fn setConfig(self: *Self, key: []const u8, value: f64) bool {
        if (std.mem.eql(u8, key, "perturb_edge_prob")) {
            self.perturb_edge_prob = value;
        } else if (std.mem.eql(u8, key, "perturb_node_prob")) {
            self.perturb_node_prob = value;
        } else if (std.mem.eql(u8, key, "perturb_edge_factor")) {
            self.perturb_edge_factor = value;
        } else if (std.mem.eql(u8, key, "perturb_node_factor")) {
            self.perturb_node_factor = value;
        } else if (std.mem.eql(u8, key, "reheat_threshold")) {
            self.reheat_threshold = value;
        } else if (std.mem.eql(u8, key, "reheat_factor")) {
            self.reheat_factor = value;
        } else if (std.mem.eql(u8, key, "min_temperature")) {
            self.min_temperature = value;
        } else {
            return false;
        }
        return true;
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    const KahanAccumulator = struct {
        sum: f64,
        c: f64,

        fn init() KahanAccumulator {
            return .{ .sum = 0.0, .c = 0.0 };
        }

        fn add(self: *KahanAccumulator, val: f64) void {
            if (!std.math.isFinite(val)) return;
            const y = val - self.c;
            const t = self.sum + y;
            self.c = (t - self.sum) - y;
            self.sum = t;
        }
    };

    fn computeEnergy(graph: *const SelfSimilarRelationalGraph) f64 {
        var acc = KahanAccumulator.init();
        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |edge| {
                if (std.math.isFinite(edge.weight) and std.math.isFinite(edge.fractal_dimension)) {
                    acc.add(edge.weight * edge.fractal_dimension);
                }
                const m = complexMagnitude(edge.quantum_correlation);
                acc.add(m);
            }
        }
        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const q = entry.value_ptr.qubit;
            const m = @sqrt(q.a.re * q.a.re + q.a.im * q.a.im + q.b.re * q.b.re + q.b.im * q.b.im);
            acc.add(m);
        }
        return acc.sum;
    }

    fn proposePerturbation(self: *Self, graph: *SelfSimilarRelationalGraph, records: *ArrayList(PerturbationRecord)) !void {
        records.clearRetainingCapacity();
        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |*edge| {
                if (self.prng.random().float(f64) < self.perturb_edge_prob) {
                    const perturbation = (self.prng.random().float(f64) - 0.5) * self.temperature * self.perturb_edge_factor;
                    try records.append(.{
                        .type = .Edge,
                        .target_id = edge.source,
                        .target_id2 = edge.target,
                        .old_weight = edge.weight,
                        .old_real = 0.0,
                        .old_imag = 0.0,
                    });
                    const w = edge.weight + perturbation;
                    edge.weight = @max(0.0, @min(1.0, w));
                }
            }
        }
        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            if (self.prng.random().float(f64) < self.perturb_node_prob) {
                const state = entry.value_ptr.qubit.a;
                const angle = self.prng.random().float(f64) * 2.0 * std.math.pi;
                const perturbation = self.temperature * self.perturb_node_factor;
                try records.append(.{
                    .type = .Node,
                    .target_id = entry.key_ptr.*,
                    .target_id2 = "",
                    .old_weight = 0.0,
                    .old_real = state.re,
                    .old_imag = state.im,
                });
                const new_real = state.re + perturbation * @cos(angle);
                const new_imag = state.im + perturbation * @sin(angle);
                if (std.math.isFinite(new_real) and std.math.isFinite(new_imag)) {
                    entry.value_ptr.qubit = nsir_core.Qubit.init(Complex(f64).init(new_real, new_imag), entry.value_ptr.qubit.b);
                }
            }
        }
    }

    fn rollbackPerturbation(graph: *SelfSimilarRelationalGraph, records: []const PerturbationRecord) void {
        for (records) |rec| {
            if (rec.type == .Edge) {
                const key = EdgeKey{ .source = rec.target_id, .target = rec.target_id2 };
                if (graph.edges.getPtr(key)) |list| {
                    if (list.items.len > 0) {
                        list.items[0].weight = rec.old_weight;
                    }
                }
            } else if (rec.type == .Node) {
                if (graph.nodes.getPtr(rec.target_id)) |node| {
                    node.qubit = nsir_core.Qubit.init(Complex(f64).init(rec.old_real, rec.old_imag), node.qubit.b);
                }
            }
        }
    }

    fn acceptMove(self: *Self, delta_energy: f64) bool {
        if (!std.math.isFinite(delta_energy)) return false;
        if (delta_energy < 0.0) return true;
        if (delta_energy <= FLOAT_EPSILON) return true;
        if (self.temperature <= FLOAT_EPSILON) return false;
        const x = -delta_energy / self.temperature;
        if (x < EXP_UNDERFLOW_THRESHOLD) return false;
        const acceptance_probability = @exp(x);
        const random_value = self.prng.random().float(f64);
        return random_value < acceptance_probability;
    }

    fn coolTemperature(self: *Self) void {
        if (self.adaptive_cooling) {
            var adjusted_rate = self.cooling_rate;
            const rate = self.statistics.acceptance_rate;
            if (rate > ACCEPTANCE_RATE_HIGH) {
                adjusted_rate *= COOLING_ADJUST_FAST;
            } else if (rate < ACCEPTANCE_RATE_LOW) {
                adjusted_rate *= COOLING_ADJUST_SLOW;
            }
            adjusted_rate = @max(COOLING_RATE_MIN, @min(COOLING_RATE_MAX, adjusted_rate));
            self.temperature *= adjusted_rate;
        } else {
            self.temperature *= self.cooling_rate;
        }
        if (self.temperature < self.min_temperature) self.temperature = self.min_temperature;
    }

    pub fn optimize(self: *Self, graph_ctx: *GraphContext) !void {
        graph_ctx.lock.lock();
        defer graph_ctx.lock.unlock();
        const graph = &graph_ctx.inner;
        self.statistics = OptimizationStatistics.init();
        self.temperature = self.initial_temperature;
        self.statistics.temperature = self.temperature;
        self.statistics.updateAcceptanceRate();
        var current_energy = computeEnergy(graph);
        var best_energy = current_energy;
        self.statistics.current_energy = current_energy;
        self.statistics.best_energy = best_energy;
        var perturbation_records = ArrayList(PerturbationRecord).init(self.allocator);
        defer perturbation_records.deinit();
        var stagnation_counter: usize = 0;
        const stagnation_limit: usize = @max(@as(usize, 1), self.max_iterations / 10);
        var iteration: usize = 0;
        while (iteration < self.max_iterations) : (iteration += 1) {
            try self.proposePerturbation(graph, &perturbation_records);
            const candidate_energy = computeEnergy(graph);
            const delta_energy = candidate_energy - current_energy;
            if (self.acceptMove(delta_energy)) {
                current_energy = candidate_energy;
                self.statistics.recordAccepted();
                if (candidate_energy < best_energy) {
                    best_energy = candidate_energy;
                    self.statistics.best_energy = best_energy;
                    stagnation_counter = 0;
                } else {
                    stagnation_counter += 1;
                }
            } else {
                self.statistics.recordRejected();
                stagnation_counter += 1;
                rollbackPerturbation(graph, perturbation_records.items);
            }
            self.statistics.current_energy = current_energy;
            self.statistics.temperature = self.temperature;
            self.statistics.iterations_completed = iteration + 1;
            self.statistics.updateAcceptanceRate();
            self.coolTemperature();
            if (stagnation_counter >= stagnation_limit) {
                if (self.temperature < self.initial_temperature * self.reheat_threshold + FLOAT_EPSILON) {
                    self.temperature = @max(self.min_temperature, self.temperature * self.reheat_factor);
                    stagnation_counter = 0;
                }
            }
            if (self.temperature <= self.min_temperature + FLOAT_EPSILON) break;
        }
    }
};

fn getGlobalAllocator() Allocator {
    const GPA = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true });
    const gpa_state = struct { var gpa: GPA = .{}; };
    return gpa_state.gpa.allocator();
}

fn cStringToSlice(ptr: [*c]const u8) ?[]const u8 {
    if (ptr == null) return null;
    return std.mem.span(ptr);
}

fn cStringToSliceMax(ptr: [*c]const u8, max: usize) ?[]const u8 {
    if (ptr == null) return null;
    const len = std.mem.indexOfScalar(u8, ptr[0..max], 0) orelse return null;
    return ptr[0..len];
}

fn cStringToNonEmptySliceMax(ptr: [*c]const u8, max: usize) ?[]const u8 {
    const s = cStringToSliceMax(ptr, max) orelse return null;
    if (s.len == 0) return null;
    return s;
}

fn clamp01(x: f64) f64 {
    if (!std.math.isFinite(x)) return 0.0;
    return @max(0.0, @min(1.0, x));
}

fn normalizePhase(phi: f64) f64 {
    if (!std.math.isFinite(phi)) return 0.0;
    const two_pi = 2.0 * std.math.pi;
    var x = @mod(phi, two_pi);
    if (x < 0.0) x += two_pi;
    return x;
}

export fn jaide_get_error_string(code: c_int) callconv(.C) [*c]const u8 {
    return switch (code) {
        JAIDE_SUCCESS => "Success",
        JAIDE_ERROR_NULL_POINTER => "Null Pointer",
        JAIDE_ERROR_ALLOCATION => "Allocation Failed",
        JAIDE_ERROR_NODE_NOT_FOUND => "Node Not Found",
        JAIDE_ERROR_EDGE_NOT_FOUND => "Edge Not Found",
        JAIDE_ERROR_INVALID_QUALITY => "Invalid Quality",
        JAIDE_ERROR_OPTIMIZATION_FAILED => "Optimization Failed",
        JAIDE_ERROR_INVALID_STRING => "Invalid String",
        JAIDE_ERROR_OPERATION_FAILED => "Operation Failed",
        JAIDE_ERROR_DUPLICATE_NODE => "Duplicate Node",
        JAIDE_ERROR_DUPLICATE_EDGE => "Duplicate Edge",
        JAIDE_ERROR_INVALID_PARAMETER => "Invalid Parameter",
        JAIDE_ERROR_MATH_ERROR => "Math Error",
        JAIDE_ERROR_NOT_INITIALIZED => "Not Initialized",
        JAIDE_ERROR_SELF_REFERENCE => "Self Reference",
        JAIDE_ERROR_INVALID_STATE => "Invalid State",
        JAIDE_ERROR_THREADING => "Threading Error",
        JAIDE_ERROR_UNKNOWN_GATE => "Unknown Gate",
        JAIDE_ERROR_OUT_OF_MEMORY => "Out Of Memory",
        else => "Unknown Error",
    };
}

export fn jaide_version_major() callconv(.C) c_int { return VERSION_MAJOR; }
export fn jaide_version_minor() callconv(.C) c_int { return VERSION_MINOR; }
export fn jaide_version_patch() callconv(.C) c_int { return VERSION_PATCH; }

export fn jaide_create_graph() callconv(.C) ?*CGraph {
    const allocator = getGlobalAllocator();
    const ctx = allocator.create(GraphContext) catch return null;
    errdefer allocator.destroy(ctx);
    const inner_graph = SelfSimilarRelationalGraph.init(allocator) catch return null;
    ctx.* = GraphContext{
        .inner = inner_graph,
        .lock = .{},
        .allocator = allocator,
    };
    return CGraph.fromInternal(ctx);
}

export fn jaide_destroy_graph(handle: ?*CGraph) callconv(.C) void {
    if (handle == null) return;
    const ctx = handle.?.toInternal();
    const allocator = ctx.allocator;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    ctx.inner.deinit();
    allocator.destroy(ctx);
}

export fn jaide_add_node(
    graph: ?*CGraph,
    id: [*c]const u8,
    type_name: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    const ctx = graph.?.toInternal();
    const id_slice = cStringToNonEmptySliceMax(id, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    const type_slice = cStringToSliceMax(type_name, MAX_STRING_SCAN) orelse "";
    ctx.lock.lock();
    defer ctx.lock.unlock();
    if (ctx.inner.nodes.get(id_slice) != null) return JAIDE_ERROR_DUPLICATE_NODE;
    const qubit = nsir_core.Qubit.init(Complex(f64).init(1.0, 0.0), Complex(f64).init(0.0, 0.0));
    var node = Node.init(ctx.allocator, id_slice, type_slice, qubit, 0.0) catch return JAIDE_ERROR_OUT_OF_MEMORY;
    ctx.inner.addNode(node) catch {
        node.deinit();
        return JAIDE_ERROR_OPERATION_FAILED;
    };
    return JAIDE_SUCCESS;
}

export fn jaide_remove_node(
    graph: ?*CGraph,
    id: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    const ctx = graph.?.toInternal();
    const id_slice = cStringToNonEmptySliceMax(id, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    if (ctx.inner.nodes.get(id_slice) == null) return JAIDE_ERROR_NODE_NOT_FOUND;
    var edges_to_remove = ArrayList(EdgeKey).init(ctx.allocator);
    defer edges_to_remove.deinit();
    var edge_iter = ctx.inner.edges.iterator();
    while (edge_iter.next()) |entry| {
        const key = entry.key_ptr.*;
        if (std.mem.eql(u8, key.source, id_slice) or std.mem.eql(u8, key.target, id_slice)) {
            edges_to_remove.append(key) catch return JAIDE_ERROR_OUT_OF_MEMORY;
        }
    }
    for (edges_to_remove.items) |key| {
        if (ctx.inner.edges.fetchRemove(key)) |kv| {
            var list = kv.value;
            for (list.items) |*edge| edge.deinit();
            list.deinit();
        }
    }
    if (ctx.inner.nodes.fetchRemove(id_slice)) |kv| {
        var node = kv.value;
        node.deinit();
    }
    _ = ctx.inner.quantum_register.fetchRemove(id_slice);
    return JAIDE_SUCCESS;
}

export fn jaide_set_node_quantum_state(
    graph: ?*CGraph,
    id: [*c]const u8,
    real: f64,
    imag: f64,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    if (!std.math.isFinite(real) or !std.math.isFinite(imag)) return JAIDE_ERROR_INVALID_PARAMETER;
    const ctx = graph.?.toInternal();
    const id_slice = cStringToNonEmptySliceMax(id, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    if (ctx.inner.nodes.get(id_slice) == null) return JAIDE_ERROR_NODE_NOT_FOUND;
    const mag2 = real * real + imag * imag;
    const qc = if (mag2 > 0 and std.math.isFinite(mag2))
        Complex(f64).init(real / @sqrt(mag2), imag / @sqrt(mag2))
    else
        Complex(f64).init(1.0, 0.0);
    const qubit_state = nsir_core.Qubit.init(qc, Complex(f64).init(0.0, 0.0));
    ctx.inner.setQuantumState(id_slice, qubit_state) catch return JAIDE_ERROR_OPERATION_FAILED;
    return JAIDE_SUCCESS;
}

export fn jaide_get_node_quantum_state(
    graph: ?*CGraph,
    id: [*c]const u8,
    out_state: ?*CQuantumState,
) callconv(.C) c_int {
    if (graph == null or out_state == null) return JAIDE_ERROR_NULL_POINTER;
    const ctx = graph.?.toInternal();
    const id_slice = cStringToNonEmptySliceMax(id, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const node = ctx.inner.nodes.getPtr(id_slice) orelse return JAIDE_ERROR_NODE_NOT_FOUND;
    out_state.?.real = node.qubit.a.re;
    out_state.?.imag = node.qubit.a.im;
    return JAIDE_SUCCESS;
}

export fn jaide_apply_gate(
    graph: ?*CGraph,
    node_id: [*c]const u8,
    gate_type: c_int,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    const ctx = graph.?.toInternal();
    const id_slice = cStringToNonEmptySliceMax(node_id, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    if (ctx.inner.nodes.get(id_slice) == null) return JAIDE_ERROR_NODE_NOT_FOUND;
    switch (gate_type) {
        0 => ctx.inner.applyQuantumGate(id_slice, &nsir_core.identityGate) catch return JAIDE_ERROR_MATH_ERROR,
        1 => ctx.inner.applyQuantumGate(id_slice, &nsir_core.hadamardGate) catch return JAIDE_ERROR_MATH_ERROR,
        2 => ctx.inner.applyQuantumGate(id_slice, &nsir_core.pauliXGate) catch return JAIDE_ERROR_MATH_ERROR,
        3 => ctx.inner.applyQuantumGate(id_slice, &nsir_core.pauliYGate) catch return JAIDE_ERROR_MATH_ERROR,
        4 => ctx.inner.applyQuantumGate(id_slice, &nsir_core.pauliZGate) catch return JAIDE_ERROR_MATH_ERROR,
        else => return JAIDE_ERROR_UNKNOWN_GATE,
    }
    return JAIDE_SUCCESS;
}

export fn jaide_add_edge(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
    weight: f64,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    const ctx = graph.?.toInternal();
    const source_slice = cStringToNonEmptySliceMax(source, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    const target_slice = cStringToNonEmptySliceMax(target, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    if (ctx.inner.nodes.get(source_slice) == null or ctx.inner.nodes.get(target_slice) == null) return JAIDE_ERROR_NODE_NOT_FOUND;
    if (std.mem.eql(u8, source_slice, target_slice)) return JAIDE_ERROR_SELF_REFERENCE;
    var edge = Edge.init(
        ctx.allocator,
        source_slice,
        target_slice,
        .coherent,
        clamp01(weight),
        Complex(f64).init(0.0, 0.0),
        DEFAULT_EDGE_FRACTAL_DIM,
    );
    ctx.inner.addEdge(source_slice, target_slice, edge) catch {
        edge.deinit();
        return JAIDE_ERROR_OPERATION_FAILED;
    };
    return JAIDE_SUCCESS;
}

export fn jaide_remove_edge(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    const ctx = graph.?.toInternal();
    const source_slice = cStringToNonEmptySliceMax(source, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    const target_slice = cStringToNonEmptySliceMax(target, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const key = EdgeKey{ .source = source_slice, .target = target_slice };
    if (ctx.inner.edges.fetchRemove(key)) |kv| {
        var list = kv.value;
        for (list.items) |*edge| edge.deinit();
        list.deinit();
        return JAIDE_SUCCESS;
    }
    return JAIDE_ERROR_EDGE_NOT_FOUND;
}

export fn jaide_get_edge_weight(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
) callconv(.C) f64 {
    if (graph == null) return -1.0;
    const ctx = graph.?.toInternal();
    const source_slice = cStringToNonEmptySliceMax(source, MAX_STRING_SCAN) orelse return -1.0;
    const target_slice = cStringToNonEmptySliceMax(target, MAX_STRING_SCAN) orelse return -1.0;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const edge_list = ctx.inner.getEdgesConst(source_slice, target_slice) orelse return -1.0;
    if (edge_list.len == 0) return -1.0;
    return edge_list[0].weight;
}

export fn jaide_set_edge_weight(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
    weight: f64,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    const ctx = graph.?.toInternal();
    const source_slice = cStringToNonEmptySliceMax(source, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    const target_slice = cStringToNonEmptySliceMax(target, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const key = EdgeKey{ .source = source_slice, .target = target_slice };
    const edges = ctx.inner.edges.getPtr(key) orelse return JAIDE_ERROR_EDGE_NOT_FOUND;
    if (edges.items.len == 0) return JAIDE_ERROR_EDGE_NOT_FOUND;
    const w = clamp01(weight);
    for (edges.items) |*edge| edge.weight = w;
    return JAIDE_SUCCESS;
}

export fn jaide_entangle_nodes(
    graph: ?*CGraph,
    node1: [*c]const u8,
    node2: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    const ctx = graph.?.toInternal();
    const n1 = cStringToNonEmptySliceMax(node1, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    const n2 = cStringToNonEmptySliceMax(node2, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    if (std.mem.eql(u8, n1, n2)) return JAIDE_ERROR_SELF_REFERENCE;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    if (ctx.inner.nodes.get(n1) == null or ctx.inner.nodes.get(n2) == null) return JAIDE_ERROR_NODE_NOT_FOUND;
    ctx.inner.entangleNodes(n1, n2) catch return JAIDE_ERROR_OPERATION_FAILED;
    return JAIDE_SUCCESS;
}

export fn jaide_encode_information(
    graph: ?*CGraph,
    data: [*c]const u8,
    out_node_id: [*c]u8,
    max_len: usize,
) callconv(.C) c_int {
    if (graph == null or data == null or out_node_id == null) return JAIDE_ERROR_NULL_POINTER;
    if (max_len == 0) return JAIDE_ERROR_INVALID_PARAMETER;
    const ctx = graph.?.toInternal();
    @memset(out_node_id[0..max_len], 0);
    const data_slice = cStringToNonEmptySliceMax(data, MAX_DATA_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const node_id = ctx.inner.encodeInformation(data_slice) catch return JAIDE_ERROR_OPERATION_FAILED;
    defer ctx.allocator.free(node_id);
    const copy_len = @min(node_id.len, max_len - 1);
    @memcpy(out_node_id[0..copy_len], node_id[0..copy_len]);
    return JAIDE_SUCCESS;
}

export fn jaide_decode_information(
    graph: ?*CGraph,
    node_id: [*c]const u8,
    out_data: [*c]u8,
    max_len: usize,
) callconv(.C) c_int {
    if (graph == null or node_id == null or out_data == null) return JAIDE_ERROR_NULL_POINTER;
    if (max_len == 0) return JAIDE_ERROR_INVALID_PARAMETER;
    const ctx = graph.?.toInternal();
    @memset(out_data[0..max_len], 0);
    const id_slice = cStringToNonEmptySliceMax(node_id, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const data = ctx.inner.decodeInformation(id_slice) orelse return JAIDE_ERROR_NODE_NOT_FOUND;
    const copy_len = @min(data.len, max_len - 1);
    @memcpy(out_data[0..copy_len], data[0..copy_len]);
    return JAIDE_SUCCESS;
}

export fn jaide_graph_node_count(graph: ?*CGraph) callconv(.C) c_int {
    if (graph == null) return 0;
    const ctx = graph.?.toInternal();
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const n = ctx.inner.nodeCount();
    return @intCast(@min(n, @as(usize, @intCast(std.math.maxInt(c_int)))));
}

export fn jaide_graph_edge_count(graph: ?*CGraph) callconv(.C) c_int {
    if (graph == null) return 0;
    const ctx = graph.?.toInternal();
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const n = ctx.inner.edgeCount();
    return @intCast(@min(n, @as(usize, @intCast(std.math.maxInt(c_int)))));
}

export fn jaide_get_topology_hash(
    graph: ?*CGraph,
    out_hash: [*c]u8,
    max_len: usize,
) callconv(.C) c_int {
    if (graph == null or out_hash == null) return JAIDE_ERROR_NULL_POINTER;
    if (max_len == 0) return JAIDE_ERROR_INVALID_PARAMETER;
    const ctx = graph.?.toInternal();
    @memset(out_hash[0..max_len], 0);
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const hash_hex = ctx.inner.getTopologyHashHex();
    const copy_len = @min(hash_hex.len, max_len - 1);
    @memcpy(out_hash[0..copy_len], hash_hex[0..copy_len]);
    return JAIDE_SUCCESS;
}

export fn jaide_measure_node(graph: ?*CGraph, node_id: [*c]const u8) callconv(.C) f64 {
    if (graph == null) return -1.0;
    const ctx = graph.?.toInternal();
    const id_slice = cStringToNonEmptySliceMax(node_id, MAX_STRING_SCAN) orelse return -1.0;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    if (ctx.inner.nodes.get(id_slice) == null) return -1.0;
    const bit = ctx.inner.measure(id_slice) catch return -1.0;
    return @as(f64, @floatFromInt(bit));
}

export fn jaide_get_node_probability(graph: ?*CGraph, id: [*c]const u8) callconv(.C) f64 {
    if (graph == null) return -1.0;
    const ctx = graph.?.toInternal();
    const id_slice = cStringToNonEmptySliceMax(id, MAX_STRING_SCAN) orelse return -1.0;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const node = ctx.inner.nodes.getPtr(id_slice) orelse return -1.0;
    const re = node.qubit.a.re;
    const im = node.qubit.a.im;
    const p = re * re + im * im;
    return clamp01(p);
}

export fn jaide_has_node(graph: ?*CGraph, id: [*c]const u8) callconv(.C) c_int {
    if (graph == null) return 0;
    const ctx = graph.?.toInternal();
    const id_slice = cStringToNonEmptySliceMax(id, MAX_STRING_SCAN) orelse return 0;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    return @intFromBool(ctx.inner.nodes.get(id_slice) != null);
}

export fn jaide_has_edge(graph: ?*CGraph, source: [*c]const u8, target: [*c]const u8) callconv(.C) c_int {
    if (graph == null) return 0;
    const ctx = graph.?.toInternal();
    const s = cStringToNonEmptySliceMax(source, MAX_STRING_SCAN) orelse return 0;
    const t = cStringToNonEmptySliceMax(target, MAX_STRING_SCAN) orelse return 0;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    return @intFromBool(ctx.inner.getEdgesConst(s, t) != null);
}

export fn jaide_create_optimizer(
    temp: f64,
    cooling: f64,
    max_iter: c_int,
) callconv(.C) ?*COptimizer {
    const allocator = getGlobalAllocator();
    if (max_iter <= 0) return null;
    const opt = allocator.create(EntangledStochasticSymmetryOptimizer) catch return null;
    opt.* = EntangledStochasticSymmetryOptimizer.init(
        allocator,
        temp,
        cooling,
        @intCast(max_iter),
    );
    return COptimizer.fromInternal(opt);
}

export fn jaide_destroy_optimizer(opt: ?*COptimizer) callconv(.C) void {
    if (opt == null) return;
    const internal = opt.?.toInternal();
    const allocator = internal.allocator;
    internal.deinit();
    allocator.destroy(internal);
}

export fn jaide_optimize_graph(
    opt: ?*COptimizer,
    graph: ?*CGraph,
) callconv(.C) c_int {
    if (opt == null or graph == null) return JAIDE_ERROR_NULL_POINTER;
    const internal_opt = opt.?.toInternal();
    const ctx = graph.?.toInternal();
    internal_opt.optimize(ctx) catch return JAIDE_ERROR_OPTIMIZATION_FAILED;
    return JAIDE_SUCCESS;
}

export fn jaide_set_optimizer_config(
    opt: ?*COptimizer,
    key: [*c]const u8,
    value: f64,
) callconv(.C) c_int {
    if (opt == null or key == null) return JAIDE_ERROR_NULL_POINTER;
    const internal = opt.?.toInternal();
    const key_slice = cStringToNonEmptySliceMax(key, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    if (!internal.setConfig(key_slice, value)) return JAIDE_ERROR_INVALID_PARAMETER;
    return JAIDE_SUCCESS;
}

export fn jaide_get_optimizer_statistics(
    opt: ?*COptimizer,
    out_iterations: ?*c_int,
    out_best_energy: ?*f64,
    out_acceptance_rate: ?*f64,
) callconv(.C) c_int {
    if (opt == null) return JAIDE_ERROR_NULL_POINTER;
    const internal = opt.?.toInternal();
    internal.statistics.lock.lock();
    defer internal.statistics.lock.unlock();
    if (out_iterations) |ptr| {
        const it = internal.statistics.iterations_completed;
        ptr.* = @intCast(@min(it, @as(usize, @intCast(std.math.maxInt(c_int)))));
    }
    if (out_best_energy) |ptr| ptr.* = internal.statistics.best_energy;
    if (out_acceptance_rate) |ptr| ptr.* = internal.statistics.acceptance_rate;
    return JAIDE_SUCCESS;
}

export fn jaide_get_node_phase(graph: ?*CGraph, id: [*c]const u8) callconv(.C) f64 {
    if (graph == null) return -1.0;
    const ctx = graph.?.toInternal();
    const id_slice = cStringToNonEmptySliceMax(id, MAX_STRING_SCAN) orelse return -1.0;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const node = ctx.inner.nodes.getPtr(id_slice) orelse return -1.0;
    return normalizePhase(node.phase);
}

export fn jaide_set_node_phase(graph: ?*CGraph, id: [*c]const u8, phase: f64) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    const ctx = graph.?.toInternal();
    const id_slice = cStringToNonEmptySliceMax(id, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const node = ctx.inner.nodes.getPtr(id_slice) orelse return JAIDE_ERROR_NODE_NOT_FOUND;
    node.phase = normalizePhase(phase);
    return JAIDE_SUCCESS;
}

export fn jaide_get_edge_quality(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
) callconv(.C) c_int {
    if (graph == null) return -1;
    const ctx = graph.?.toInternal();
    const source_slice = cStringToNonEmptySliceMax(source, MAX_STRING_SCAN) orelse return -1;
    const target_slice = cStringToNonEmptySliceMax(target, MAX_STRING_SCAN) orelse return -1;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const edge_list = ctx.inner.getEdgesConst(source_slice, target_slice) orelse return -1;
    if (edge_list.len == 0) return -1;
    return @intFromEnum(edge_list[0].quality);
}

export fn jaide_set_edge_quality(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
    quality: c_int,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    const ctx = graph.?.toInternal();
    const source_slice = cStringToNonEmptySliceMax(source, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    const target_slice = cStringToNonEmptySliceMax(target, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const key = EdgeKey{ .source = source_slice, .target = target_slice };
    const edges = ctx.inner.edges.getPtr(key) orelse return JAIDE_ERROR_EDGE_NOT_FOUND;
    if (edges.items.len == 0) return JAIDE_ERROR_EDGE_NOT_FOUND;
    const new_quality: EdgeQuality = switch (quality) {
        0 => .superposition,
        1 => .entangled,
        2 => .coherent,
        3 => .collapsed,
        4 => .fractal,
        else => return JAIDE_ERROR_INVALID_QUALITY,
    };
    for (edges.items) |*edge| {
        edge.quality = new_quality;
    }
    return JAIDE_SUCCESS;
}

export fn jaide_get_node_data(
    graph: ?*CGraph,
    id: [*c]const u8,
    out_data: [*c]u8,
    max_len: usize,
) callconv(.C) c_int {
    if (graph == null or id == null or out_data == null) return JAIDE_ERROR_NULL_POINTER;
    if (max_len == 0) return JAIDE_ERROR_INVALID_PARAMETER;
    const ctx = graph.?.toInternal();
    @memset(out_data[0..max_len], 0);
    const id_slice = cStringToNonEmptySliceMax(id, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const node = ctx.inner.nodes.getPtr(id_slice) orelse return JAIDE_ERROR_NODE_NOT_FOUND;
    if (node.data.len > 0) {
        const copy_len = @min(node.data.len, max_len - 1);
        @memcpy(out_data[0..copy_len], node.data[0..copy_len]);
    }
    return JAIDE_SUCCESS;
}

export fn jaide_get_edge_fractal_dimension(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
) callconv(.C) f64 {
    if (graph == null) return -1.0;
    const ctx = graph.?.toInternal();
    const source_slice = cStringToNonEmptySliceMax(source, MAX_STRING_SCAN) orelse return -1.0;
    const target_slice = cStringToNonEmptySliceMax(target, MAX_STRING_SCAN) orelse return -1.0;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const edge_list = ctx.inner.getEdgesConst(source_slice, target_slice) orelse return -1.0;
    if (edge_list.len == 0) return -1.0;
    const d = edge_list[0].fractal_dimension;
    if (!std.math.isFinite(d) or d < 0.0) return -1.0;
    return d;
}

export fn jaide_set_edge_fractal_dimension(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
    dimension: f64,
) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    if (!std.math.isFinite(dimension) or dimension < 0.0) return JAIDE_ERROR_INVALID_PARAMETER;
    const ctx = graph.?.toInternal();
    const source_slice = cStringToNonEmptySliceMax(source, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    const target_slice = cStringToNonEmptySliceMax(target, MAX_STRING_SCAN) orelse return JAIDE_ERROR_INVALID_STRING;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const key = EdgeKey{ .source = source_slice, .target = target_slice };
    const edges = ctx.inner.edges.getPtr(key) orelse return JAIDE_ERROR_EDGE_NOT_FOUND;
    if (edges.items.len == 0) return JAIDE_ERROR_EDGE_NOT_FOUND;
    for (edges.items) |*edge| {
        edge.fractal_dimension = dimension;
    }
    return JAIDE_SUCCESS;
}

export fn jaide_get_fractal_dimension(graph: ?*CGraph) callconv(.C) f64 {
    if (graph == null) return -1.0;
    const ctx = graph.?.toInternal();
    ctx.lock.lock();
    defer ctx.lock.unlock();
    var total: f64 = 0.0;
    var count: usize = 0;
    var edge_iter = ctx.inner.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            if (std.math.isFinite(edge.fractal_dimension) and edge.fractal_dimension >= 0.0) {
                total += edge.fractal_dimension;
                count += 1;
            }
        }
    }
    if (count == 0) return 0.0;
    const v = total / @as(f64, @floatFromInt(count));
    if (!std.math.isFinite(v) or v < 0.0) return -1.0;
    return v;
}

export fn jaide_get_node_magnitude(
    graph: ?*CGraph,
    id: [*c]const u8,
) callconv(.C) f64 {
    if (graph == null) return -1.0;
    const ctx = graph.?.toInternal();
    const id_slice = cStringToNonEmptySliceMax(id, MAX_STRING_SCAN) orelse return -1.0;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const node = ctx.inner.nodes.getPtr(id_slice) orelse return -1.0;
    const q = node.qubit;
    const m = @sqrt(q.a.re * q.a.re + q.a.im * q.a.im + q.b.re * q.b.re + q.b.im * q.b.im);
    if (!std.math.isFinite(m)) return -1.0;
    return m;
}

export fn jaide_get_edge_correlation_magnitude(
    graph: ?*CGraph,
    source: [*c]const u8,
    target: [*c]const u8,
) callconv(.C) f64 {
    if (graph == null) return -1.0;
    const ctx = graph.?.toInternal();
    const source_slice = cStringToNonEmptySliceMax(source, MAX_STRING_SCAN) orelse return -1.0;
    const target_slice = cStringToNonEmptySliceMax(target, MAX_STRING_SCAN) orelse return -1.0;
    ctx.lock.lock();
    defer ctx.lock.unlock();
    const edge_list = ctx.inner.getEdgesConst(source_slice, target_slice) orelse return -1.0;
    if (edge_list.len == 0) return -1.0;
    const qc = edge_list[0].quantum_correlation;
    const v = @sqrt(qc.re * qc.re + qc.im * qc.im);
    if (!std.math.isFinite(v)) return -1.0;
    return @max(0.0, v);
}

export fn jaide_clear_graph(graph: ?*CGraph) callconv(.C) c_int {
    if (graph == null) return JAIDE_ERROR_NULL_POINTER;
    const ctx = graph.?.toInternal();
    ctx.lock.lock();
    defer ctx.lock.unlock();
    ctx.inner.clear() catch return JAIDE_ERROR_OPERATION_FAILED;
    return JAIDE_SUCCESS;
}

export fn jaide_apply_hadamard(
    graph: ?*CGraph,
    node_id: [*c]const u8,
) callconv(.C) c_int {
    return jaide_apply_gate(graph, node_id, 1);
}

export fn jaide_apply_pauli_x(
    graph: ?*CGraph,
    node_id: [*c]const u8,
) callconv(.C) c_int {
    return jaide_apply_gate(graph, node_id, 2);
}

export fn jaide_apply_pauli_y(
    graph: ?*CGraph,
    node_id: [*c]const u8,
) callconv(.C) c_int {
    return jaide_apply_gate(graph, node_id, 3);
}

export fn jaide_apply_pauli_z(
    graph: ?*CGraph,
    node_id: [*c]const u8,
) callconv(.C) c_int {
    return jaide_apply_gate(graph, node_id, 4);
}

export fn jaide_apply_identity_gate(
    graph: ?*CGraph,
    node_id: [*c]const u8,
) callconv(.C) c_int {
    return jaide_apply_gate(graph, node_id, 0);
}

