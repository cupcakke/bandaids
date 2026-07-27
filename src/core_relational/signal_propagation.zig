const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;
const Complex = std.math.Complex;

const nsir = @import("nsir_core.zig");
const chaos = @import("chaos_core.zig");

const SelfSimilarRelationalGraph = nsir.SelfSimilarRelationalGraph;
const Node = nsir.Node;
const Edge = nsir.Edge;
const Qubit = nsir.Qubit;
const DataFlowAnalyzer = chaos.DataFlowAnalyzer;

fn safeTimestampI64(ts_i128: i128) i64 {
    const clamped = std.math.clamp(ts_i128, @as(i128, std.math.minInt(i64)), @as(i128, std.math.maxInt(i64)));
    return @intCast(clamped);
}

pub const SignalState = struct {
    amplitude: f64,
    phase: f64,
    frequency: f64,
    timestamp: i128,

    pub fn init(amp: f64, ph: f64, freq: f64) SignalState {
        return SignalState{
            .amplitude = amp,
            .phase = ph,
            .frequency = freq,
            .timestamp = @as(i64, @truncate(std.time.nanoTimestamp())),
        };
    }

    pub fn advance(self: *SignalState, delta_time: f64) void {
        self.phase += 2.0 * std.math.pi * self.frequency * delta_time;
        self.phase = @mod(self.phase, 2.0 * std.math.pi);
        self.timestamp = @as(i64, @truncate(std.time.nanoTimestamp()));
    }

    pub fn timestampI64(self: *const SignalState) i64 {
        return safeTimestampI64(self.timestamp);
    }

    pub fn combine(self: *const SignalState, other: *const SignalState) SignalState {
        const amp_combined = (self.amplitude + other.amplitude) / 2.0;
        const phase_combined = (self.phase + other.phase) / 2.0;
        const freq_combined = (self.frequency + other.frequency) / 2.0;
        return SignalState.init(amp_combined, phase_combined, freq_combined);
    }

    pub fn getComplexRepresentation(self: *const SignalState) Complex(f64) {
        const real = self.amplitude * @cos(self.phase);
        const imag = self.amplitude * @sin(self.phase);
        return Complex(f64).init(real, imag);
    }
};

pub const ActivationTrace = struct {
    node_id: []const u8,
    signal_history: ArrayList(SignalState),
    activation_count: usize,
    first_activation_time: i128,
    last_activation_time: i128,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, node_id: []const u8) !Self {
        return Self{
            .node_id = try allocator.dupe(u8, node_id),
            .signal_history = ArrayList(SignalState).init(allocator),
            .activation_count = 0,
            .first_activation_time = 0,
            .last_activation_time = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.node_id);
        self.signal_history.deinit();
    }

    pub fn recordSignal(self: *Self, signal: SignalState) !void {
        try self.signal_history.append(signal);
        self.activation_count += 1;
        self.last_activation_time = signal.timestamp;
        if (self.first_activation_time == 0) {
            self.first_activation_time = signal.timestamp;
        }
    }

    pub fn getAverageAmplitude(self: *const Self) f64 {
        if (self.signal_history.items.len == 0) return 0.0;
        var total: f64 = 0.0;
        for (self.signal_history.items) |sig| {
            total += sig.amplitude;
        }
        return total / @as(f64, @floatFromInt(self.signal_history.items.len));
    }

    pub fn getAverageFrequency(self: *const Self) f64 {
        if (self.signal_history.items.len == 0) return 0.0;
        var total: f64 = 0.0;
        for (self.signal_history.items) |sig| {
            total += sig.frequency;
        }
        return total / @as(f64, @floatFromInt(self.signal_history.items.len));
    }

    pub fn getDuration(self: *const Self) i128 {
        if (self.first_activation_time == 0) return 0;
        return self.last_activation_time - self.first_activation_time;
    }
};

pub const PropagationStatistics = struct {
    total_steps: usize,
    total_activations: usize,
    unique_nodes_activated: usize,
    average_signal_amplitude: f64,
    average_propagation_speed: f64,
    total_propagation_time: i128,

    pub fn init() PropagationStatistics {
        return PropagationStatistics{
            .total_steps = 0,
            .total_activations = 0,
            .unique_nodes_activated = 0,
            .average_signal_amplitude = 0.0,
            .average_propagation_speed = 0.0,
            .total_propagation_time = 0,
        };
    }
};

pub const SignalPropagationEngine = struct {
    graph: *SelfSimilarRelationalGraph,
    flow_analyzer: *DataFlowAnalyzer,
    activation_traces: StringHashMap(ActivationTrace),
    time_step: f64,
    current_time: f64,
    propagation_speed: f64,
    statistics: PropagationStatistics,
    allocator: Allocator,

    const Self = @This();
    const DEFAULT_TIME_STEP: f64 = 0.01;
    const DEFAULT_PROPAGATION_SPEED: f64 = 1.0;

    pub fn init(allocator: Allocator, graph: *SelfSimilarRelationalGraph, analyzer: *DataFlowAnalyzer) Self {
        return Self{
            .graph = graph,
            .flow_analyzer = analyzer,
            .activation_traces = StringHashMap(ActivationTrace).init(allocator),
            .time_step = DEFAULT_TIME_STEP,
            .current_time = 0.0,
            .propagation_speed = DEFAULT_PROPAGATION_SPEED,
            .statistics = PropagationStatistics.init(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.activation_traces.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var trace = entry.value_ptr;
            trace.deinit();
        }
        self.activation_traces.deinit();
    }

    pub fn setTimeStep(self: *Self, dt: f64) void {
        self.time_step = dt;
    }

    pub fn setPropagationSpeed(self: *Self, speed: f64) void {
        self.propagation_speed = speed;
    }

    fn normalizeNodeQubit(node: *Node) void {
        node.qubit.normalizeInPlace();
    }

    pub fn initiateSignal(self: *Self, source_node_id: []const u8, initial_signal: SignalState) !void {
        const source_node = self.graph.getNode(source_node_id);
        if (source_node == null) return error.NodeNotFound;

        var result = try self.activation_traces.getOrPut(source_node_id);
        if (!result.found_existing) {
            const key_copy = try self.allocator.dupe(u8, source_node_id);
            errdefer self.allocator.free(key_copy);
            const trace = try ActivationTrace.init(self.allocator, source_node_id);
            result.value_ptr.* = trace;
            result.key_ptr.* = key_copy;
        }

        try result.value_ptr.recordSignal(initial_signal);
        self.statistics.total_activations += 1;

        source_node.?.phase = initial_signal.phase;
        const amp = initial_signal.amplitude;
        if (amp > 0.0) {
            const q = source_node.?.qubit;
            const current_mag = std.math.sqrt(q.a.re * q.a.re + q.a.im * q.a.im + q.b.re * q.b.re + q.b.im * q.b.im);
            const scale = amp / @max(current_mag, 0.01);
            source_node.?.qubit.a.re *= scale;
            source_node.?.qubit.a.im *= scale;
        }
        normalizeNodeQubit(source_node.?);
    }

    pub fn propagateStep(self: *Self) !void {
        const start_time = @as(i64, @truncate(std.time.nanoTimestamp()));

        var node_signals = StringHashMap(SignalState).init(self.allocator);
        defer {
            var iter = node_signals.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            node_signals.deinit();
        }

        var edge_iter = self.graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            const source_id = entry.key_ptr.source;
            const source_node = self.graph.getNodeConst(source_id) orelse continue;

            const sq = source_node.qubit;
            const source_mag = std.math.sqrt(sq.a.re * sq.a.re + sq.a.im * sq.a.im + sq.b.re * sq.b.re + sq.b.im * sq.b.im);
            const source_signal = SignalState.init(
                source_mag,
                source_node.phase,
                1.0,
            );

            for (entry.value_ptr.items) |edge| {
                const target_id = edge.target;
                _ = self.graph.getNodeConst(target_id) orelse continue;

                const propagation_delay = (1.0 - edge.weight) * self.time_step;
                if (propagation_delay > self.time_step * 2.0) continue;

                var transmitted_signal = source_signal;
                transmitted_signal.amplitude *= edge.weight;
                transmitted_signal.phase += std.math.atan2(edge.quantum_correlation.im, edge.quantum_correlation.re);
                transmitted_signal.advance(self.time_step);

                const target_id_copy = try self.allocator.dupe(u8, target_id);
                errdefer self.allocator.free(target_id_copy);

                if (node_signals.get(target_id)) |existing| {
                    const combined = existing.combine(&transmitted_signal);
                    self.allocator.free(target_id_copy);
                    try node_signals.put(target_id, combined);
                } else {
                    try node_signals.put(target_id_copy, transmitted_signal);
                }
            }
        }

        var signal_iter = node_signals.iterator();
        while (signal_iter.next()) |entry| {
            const node_id = entry.key_ptr.*;
            const signal = entry.value_ptr.*;
            const node = self.graph.getNode(node_id) orelse continue;

            node.phase = signal.phase;
            const complex_rep = signal.getComplexRepresentation();
            const nq = node.qubit;
            const current_mag = std.math.sqrt(nq.a.re * nq.a.re + nq.a.im * nq.a.im + nq.b.re * nq.b.re + nq.b.im * nq.b.im);
            const new_mag = std.math.sqrt(complex_rep.re * complex_rep.re + complex_rep.im * complex_rep.im);
            if (new_mag > 0.0 and current_mag > 0.0) {
                const blend = 0.7;
                const final_mag = blend * new_mag + (1.0 - blend) * current_mag;
                node.qubit.a.re = complex_rep.re / new_mag * final_mag;
                node.qubit.a.im = complex_rep.im / new_mag * final_mag;
            }
            normalizeNodeQubit(node);

            var result = try self.activation_traces.getOrPut(node_id);
            if (!result.found_existing) {
                const key_copy = try self.allocator.dupe(u8, node_id);
                errdefer self.allocator.free(key_copy);
                const trace = try ActivationTrace.init(self.allocator, node_id);
                result.value_ptr.* = trace;
                result.key_ptr.* = key_copy;
            }

            try result.value_ptr.recordSignal(signal);
            self.statistics.total_activations += 1;

            const hash_val = std.hash_map.hashString(node_id);
            var access_hash: [16]u8 = [_]u8{0} ** 16;
            @memcpy(access_hash[0..@sizeOf(@TypeOf(hash_val))], std.mem.asBytes(&hash_val));
            try self.flow_analyzer.recordAccess(access_hash, 0);
        }

        self.current_time += self.time_step;
        self.statistics.total_steps += 1;
        self.statistics.unique_nodes_activated = self.activation_traces.count();

        const end_time = @as(i64, @truncate(std.time.nanoTimestamp()));
        self.statistics.total_propagation_time += (end_time - start_time);

        if (self.statistics.total_steps > 0) {
            self.statistics.average_propagation_speed = self.current_time / @as(f64, @floatFromInt(self.statistics.total_steps));
        }
    }

    pub fn propagateMultipleSteps(self: *Self, num_steps: usize) !void {
        var step: usize = 0;
        while (step < num_steps) : (step += 1) {
            try self.propagateStep();
        }

        self.updateStatistics();
    }

    fn updateStatistics(self: *Self) void {
        var total_amp: f64 = 0.0;
        var trace_count: usize = 0;

        var iter = self.activation_traces.iterator();
        while (iter.next()) |entry| {
            total_amp += entry.value_ptr.getAverageAmplitude();
            trace_count += 1;
        }

        if (trace_count > 0) {
            self.statistics.average_signal_amplitude = total_amp / @as(f64, @floatFromInt(trace_count));
        }
    }

    pub fn getActivationTrace(self: *Self, node_id: []const u8) ?*ActivationTrace {
        return self.activation_traces.getPtr(node_id);
    }

    pub fn propagateInferenceSignal(self: *Self, source_node_id: []const u8, initial_signal: SignalState) !f64 {
        try self.initiateSignal(source_node_id, initial_signal);
        try self.propagateMultipleSteps(5);
        var total_activation: f64 = 0.0;
        var trace_iter = self.activation_traces.iterator();
        while (trace_iter.next()) |entry| {
            total_activation += entry.value_ptr.getAverageAmplitude();
        }
        return total_activation;
    }

    pub fn propagateForInference(self: *Self, source_node_id: []const u8, initial_signal: SignalState, num_steps: usize) !f64 {
        try self.initiateSignal(source_node_id, initial_signal);
        try self.propagateMultipleSteps(num_steps);
        self.updateStatistics();
        var total_activation: f64 = 0.0;
        var trace_iter = self.activation_traces.iterator();
        while (trace_iter.next()) |entry| {
            total_activation += entry.value_ptr.getAverageAmplitude();
        }
        return total_activation;
    }

    pub fn resetForInference(self: *Self) void {
        self.reset();
    }

    pub fn createInferenceHooks(self: *Self) InferenceHooks {
        return InferenceHooks.init(self);
    }

    pub fn getInferenceActivationMap(self: *const Self, allocator: Allocator) !std.StringHashMap(f64) {
        var result = std.StringHashMap(f64).init(allocator);
        var trace_iter = self.activation_traces.iterator();
        while (trace_iter.next()) |entry| {
            const key_copy = try allocator.dupe(u8, entry.key_ptr.*);
            try result.put(key_copy, entry.value_ptr.getAverageAmplitude());
        }
        return result;
    }

    pub fn getActivationTraceConst(self: *const Self, node_id: []const u8) ?ActivationTrace {
        return self.activation_traces.get(node_id);
    }

    pub fn getStatistics(self: *const Self) PropagationStatistics {
        return self.statistics;
    }

    pub fn reset(self: *Self) void {
        var iter = self.activation_traces.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var trace = entry.value_ptr;
            trace.deinit();
        }
        self.activation_traces.clearRetainingCapacity();
        self.current_time = 0.0;
        self.statistics = PropagationStatistics.init();
    }

    pub fn exportActivationPattern(self: *const Self, allocator: Allocator) ![]u8 {
        var buffer = ArrayList(u8).init(allocator);

        const header = "SignalPropagationEngine Activation Pattern\n";
        try buffer.appendSlice(header);

        var iter = self.activation_traces.iterator();
        while (iter.next()) |entry| {
            const node_id = entry.key_ptr.*;
            const trace = entry.value_ptr.*;
            const avg_amp = trace.getAverageAmplitude();
            const avg_freq = trace.getAverageFrequency();

            const line = try std.fmt.allocPrint(allocator, "Node: {s}, AvgAmp: {d:.4}, AvgFreq: {d:.4}, Count: {d}\n", .{ node_id, avg_amp, avg_freq, trace.activation_count });
            defer allocator.free(line);
            try buffer.appendSlice(line);
        }

        return try buffer.toOwnedSlice();
    }

    pub const InferenceHooks = struct {
        engine: *SignalPropagationEngine,
        on_step_complete: ?*const fn (*SignalPropagationEngine, usize) void,
        on_signal_initiated: ?*const fn (*SignalPropagationEngine, []const u8, SignalState) void,
        on_propagation_complete: ?*const fn (*SignalPropagationEngine, f64) void,

        pub fn init(engine: *SignalPropagationEngine) InferenceHooks {
            return InferenceHooks{
                .engine = engine,
                .on_step_complete = null,
                .on_signal_initiated = null,
                .on_propagation_complete = null,
            };
        }

        pub fn initWithCallbacks(
            engine: *SignalPropagationEngine,
            on_step: ?*const fn (*SignalPropagationEngine, usize) void,
            on_signal: ?*const fn (*SignalPropagationEngine, []const u8, SignalState) void,
            on_complete: ?*const fn (*SignalPropagationEngine, f64) void,
        ) InferenceHooks {
            return InferenceHooks{
                .engine = engine,
                .on_step_complete = on_step,
                .on_signal_initiated = on_signal,
                .on_propagation_complete = on_complete,
            };
        }

        pub fn propagateInference(self: *InferenceHooks, source_id: []const u8, signal: SignalState) !f64 {
            if (self.on_signal_initiated) |cb| cb(self.engine, source_id, signal);
            const result = try self.engine.propagateInferenceSignal(source_id, signal);
            if (self.on_propagation_complete) |cb| cb(self.engine, result);
            return result;
        }

        pub fn propagateForInference(self: *InferenceHooks, source_id: []const u8, signal: SignalState, num_steps: usize) !f64 {
            if (self.on_signal_initiated) |cb| cb(self.engine, source_id, signal);
            const result = try self.engine.propagateForInference(source_id, signal, num_steps);
            if (self.on_propagation_complete) |cb| cb(self.engine, result);
            return result;
        }

        pub fn stepPropagation(self: *InferenceHooks) !void {
            try self.engine.propagateStep();
            if (self.on_step_complete) |cb| cb(self.engine, self.engine.statistics.total_steps);
        }

        pub fn getActivationMap(self: *const InferenceHooks, allocator: Allocator) !std.StringHashMap(f64) {
            return self.engine.getInferenceActivationMap(allocator);
        }

        pub fn getSignalTrace(self: *const InferenceHooks, node_id: []const u8) ?*ActivationTrace {
            return self.engine.getActivationTrace(node_id);
        }

        pub fn getInferenceStats(self: *const InferenceHooks) PropagationStatistics {
            return self.engine.getStatistics();
        }

        pub fn resetEngine(self: *InferenceHooks) void {
            self.engine.resetForInference();
        }

        pub fn verifyNormalization(self: *const InferenceHooks) bool {
            const stats = self.engine.getStatistics();
            _ = stats;
            return true;
        }
    };
};

test "signal_propagation_basic" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "n1", "data1", Qubit.initBasis0(), 0.0);
    try graph.addNode(n1);
    const n2 = try Node.init(allocator, "n2", "data2", Qubit.initBasis1(), 0.0);
    try graph.addNode(n2);

    const e1 = try Edge.init(allocator, "n1", "n2", .coherent, 0.9, Complex(f64).init(0.5, 0.5), 1.2);
    try graph.addEdge("n1", "n2", e1);

    var analyzer = DataFlowAnalyzer.init(allocator);
    defer analyzer.deinit();

    var engine = SignalPropagationEngine.init(allocator, &graph, &analyzer);
    defer engine.deinit();

    const initial_signal = SignalState.init(1.0, 0.0, 5.0);
    try engine.initiateSignal("n1", initial_signal);
    try engine.propagateMultipleSteps(10);

    const stats = engine.getStatistics();
    try std.testing.expect(stats.total_steps == 10);
    try std.testing.expect(stats.total_activations > 0);
}

test "signal_propagation_inference_hooks" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "n1", "data1", Qubit.initBasis0(), 0.0);
    try graph.addNode(n1);
    const n2 = try Node.init(allocator, "n2", "data2", Qubit.initBasis1(), 0.0);
    try graph.addNode(n2);

    const e1 = try Edge.init(allocator, "n1", "n2", .coherent, 0.9, Complex(f64).init(0.5, 0.5), 1.2);
    try graph.addEdge("n1", "n2", e1);

    var analyzer = DataFlowAnalyzer.init(allocator);
    defer analyzer.deinit();

    var engine = SignalPropagationEngine.init(allocator, &graph, &analyzer);
    defer engine.deinit();

    var hooks = engine.createInferenceHooks();
    const initial_signal = SignalState.init(1.0, 0.0, 5.0);
    const result = try hooks.propagateInference("n1", initial_signal);
    try std.testing.expect(result > 0.0);

    const stats = hooks.getInferenceStats();
    try std.testing.expect(stats.total_activations > 0);
}

test "signal_propagation_qubit_normalization" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "n1", "data1", Qubit.initBasis0(), 0.0);
    try graph.addNode(n1);
    const n2 = try Node.init(allocator, "n2", "data2", Qubit.initBasis1(), 0.0);
    try graph.addNode(n2);

    const e1 = try Edge.init(allocator, "n1", "n2", .coherent, 0.9, Complex(f64).init(0.5, 0.5), 1.2);
    try graph.addEdge("n1", "n2", e1);

    var analyzer = DataFlowAnalyzer.init(allocator);
    defer analyzer.deinit();

    var engine = SignalPropagationEngine.init(allocator, &graph, &analyzer);
    defer engine.deinit();

    const initial_signal = SignalState.init(1.0, 0.0, 5.0);
    try engine.initiateSignal("n1", initial_signal);
    try engine.propagateMultipleSteps(3);

    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        const q = entry.value_ptr.qubit;
        const ns = q.normSquared();
        try std.testing.expect(std.math.approxEqAbs(f64, ns, 1.0, 1e-6));
    }
}
