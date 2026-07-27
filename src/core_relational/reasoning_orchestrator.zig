const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;

const nsir = @import("nsir_core.zig");
const esso = @import("esso_optimizer.zig");
const chaos = @import("chaos_core.zig");
const fnds = @import("fnds.zig");
const quantum = @import("quantum_logic.zig");
const Complex = std.math.Complex;

const SelfSimilarRelationalGraph = nsir.SelfSimilarRelationalGraph;
const Node = nsir.Node;
const Edge = nsir.Edge;
const Qubit = nsir.Qubit;
const EntangledStochasticSymmetryOptimizer = esso.EntangledStochasticSymmetryOptimizer;
const ChaosCoreKernel = chaos.ChaosCoreKernel;
const FractalTree = fnds.FractalTree;
const SymmetryPattern = esso.SymmetryPattern;

pub const OrchestratorIntegrationTypes = struct {
    pub const GraphEdge = Edge;
    pub const FractalStructure = FractalTree;
    pub const DetectedSymmetryPattern = SymmetryPattern;
};

pub const PatternId = [32]u8;

pub const ThoughtLevel = enum(u8) {
    local = 0,
    global = 1,
    meta = 2,

    pub fn toString(self: ThoughtLevel) []const u8 {
        return switch (self) {
            .local => "local",
            .global => "global",
            .meta => "meta",
        };
    }
};

pub const ReasoningPhase = struct {
    phase_id: u64,
    level: ThoughtLevel,
    configured_inner_iterations: usize,
    configured_outer_iterations: usize,
    inner_iterations: usize,
    outer_iterations: usize,
    target_energy: f64,
    current_energy: f64,
    previous_energy: f64,
    convergence_threshold: f64,
    phase_start_time: i64,
    phase_end_time: i64,
    pattern_captures: ArrayList(PatternId),

    const Self = @This();

    pub fn init(allocator: Allocator, level: ThoughtLevel, inner: usize, outer: usize, phase_id: u64) Self {
        return Self{
            .phase_id = phase_id,
            .level = level,
            .configured_inner_iterations = inner,
            .configured_outer_iterations = outer,
            .inner_iterations = 0,
            .outer_iterations = 0,
            .target_energy = 0.01,
            .current_energy = 1e6,
            .previous_energy = 1e6,
            .convergence_threshold = 1e-6,
            .phase_start_time = @as(i64, @intCast(std.time.nanoTimestamp())),
            .phase_end_time = 0,
            .pattern_captures = ArrayList(PatternId).init(allocator),
        };
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var cloned = Self{
            .phase_id = self.phase_id,
            .level = self.level,
            .configured_inner_iterations = self.configured_inner_iterations,
            .configured_outer_iterations = self.configured_outer_iterations,
            .inner_iterations = self.inner_iterations,
            .outer_iterations = self.outer_iterations,
            .target_energy = self.target_energy,
            .current_energy = self.current_energy,
            .previous_energy = self.previous_energy,
            .convergence_threshold = self.convergence_threshold,
            .phase_start_time = self.phase_start_time,
            .phase_end_time = self.phase_end_time,
            .pattern_captures = ArrayList(PatternId).init(allocator),
        };
        errdefer cloned.deinit();
        try cloned.pattern_captures.appendSlice(self.pattern_captures.items);
        return cloned;
    }

    pub fn deinit(self: *Self) void {
        self.pattern_captures.deinit();
    }

    pub fn recordPattern(self: *Self, pattern_id: PatternId) !void {
        try self.pattern_captures.append(pattern_id);
    }

    pub fn recordPatternAssumeCapacity(self: *Self, pattern_id: PatternId) void {
        self.pattern_captures.appendAssumeCapacity(pattern_id);
    }

    pub fn patternDigest(self: *const Self) PatternId {
        var digest = [_]u8{0} ** 32;
        for (self.pattern_captures.items, 0..) |pattern_id, pattern_index| {
            for (pattern_id, 0..) |byte, byte_index| {
                const mix = byte +% @as(u8, @truncate((pattern_index + byte_index) & 0xff));
                digest[byte_index] = digest[byte_index] ^ mix;
            }
        }
        return digest;
    }

    pub fn hasConverged(self: *const Self) bool {
        if (!std.math.isFinite(self.current_energy) or !std.math.isFinite(self.previous_energy)) {
            return false;
        }
        const delta = @abs(self.current_energy - self.previous_energy);
        const denom = @max(@abs(self.previous_energy), 1.0);
        return std.math.isFinite(delta) and std.math.isFinite(denom) and denom > 0.0 and (delta / denom) < self.convergence_threshold;
    }

    pub fn hasMetTargetEnergy(self: *const Self) bool {
        return std.math.isFinite(self.current_energy) and std.math.isFinite(self.target_energy) and self.current_energy <= self.target_energy;
    }

    pub fn canRunInnerIteration(self: *const Self) bool {
        return self.inner_iterations < self.configured_inner_iterations;
    }

    pub fn canRunOuterIteration(self: *const Self) bool {
        return self.outer_iterations < self.configured_outer_iterations;
    }

    pub fn updateEnergy(self: *Self, new_energy: f64) void {
        self.previous_energy = self.current_energy;
        self.current_energy = if (std.math.isFinite(new_energy)) new_energy else 1e6;
    }

    pub fn finalize(self: *Self) void {
        self.phase_end_time = @as(i64, @intCast(std.time.nanoTimestamp()));
    }

    pub fn getDuration(self: *const Self) i64 {
        if (self.phase_end_time > 0) {
            return self.phase_end_time - self.phase_start_time;
        }
        return @as(i64, @intCast(std.time.nanoTimestamp())) - self.phase_start_time;
    }
};

pub const OrchestratorStatistics = struct {
    total_phases: usize,
    local_phases: usize,
    global_phases: usize,
    meta_phases: usize,
    total_inner_loops: usize,
    total_outer_loops: usize,
    average_convergence_time: f64,
    best_energy_achieved: f64,
    patterns_discovered: usize,
    orchestration_start_time: i64,

    pub fn init() OrchestratorStatistics {
        return OrchestratorStatistics{
            .total_phases = 0,
            .local_phases = 0,
            .global_phases = 0,
            .meta_phases = 0,
            .total_inner_loops = 0,
            .total_outer_loops = 0,
            .average_convergence_time = 0.0,
            .best_energy_achieved = std.math.inf(f64),
            .patterns_discovered = 0,
            .orchestration_start_time = @as(i64, @intCast(std.time.nanoTimestamp())),
        };
    }

    pub fn recordPhase(self: *OrchestratorStatistics, phase: *const ReasoningPhase) void {
        self.total_phases += 1;
        switch (phase.level) {
            .local => self.local_phases += 1,
            .global => self.global_phases += 1,
            .meta => self.meta_phases += 1,
        }
        self.total_inner_loops += phase.inner_iterations;
        self.total_outer_loops += phase.outer_iterations;
        if (std.math.isFinite(phase.current_energy) and phase.current_energy < self.best_energy_achieved) {
            self.best_energy_achieved = phase.current_energy;
        }
        self.patterns_discovered += phase.pattern_captures.items.len;

        const duration = @as(f64, @floatFromInt(phase.getDuration()));
        const n = @as(f64, @floatFromInt(self.total_phases));
        if (std.math.isFinite(duration) and n > 0.0) {
            self.average_convergence_time += (duration - self.average_convergence_time) / n;
        }
    }
};

pub const ReasoningResult = struct {
    best_energy: f64,
    modulation_factor: f64,
    phases_completed: usize,
    patterns_found: usize,
};

const NodeStateSnapshot = struct {
    node: *Node,
    qubit: Qubit,
    phase: f64,
};

const EdgeStateSnapshot = struct {
    edge: *Edge,
    weight: f64,
    fractal_dimension: f64,
    quantum_correlation: Complex(f64),
};

pub const ReasoningOrchestrator = struct {
    graph: *SelfSimilarRelationalGraph,
    esso: *EntangledStochasticSymmetryOptimizer,
    chaos_kernel: *ChaosCoreKernel,
    phase_history: ArrayList(ReasoningPhase),
    statistics: OrchestratorStatistics,
    phase_level_counts: StringHashMap(usize),
    pattern_energy_index: AutoHashMap(PatternId, f64),
    fast_inner_steps: usize,
    slow_outer_steps: usize,
    hierarchical_depth: usize,
    perturb_node_limit: usize,
    update_edge_limit: usize,
    transform_node_limit: usize,
    next_phase_id: u64,
    allocator: Allocator,

    const Self = @This();

    const default_target_energy: f64 = 0.01;
    const default_convergence_threshold: f64 = 1e-6;
    const qubit_epsilon: f64 = 1e-12;
    const max_quantum_correlation_magnitude: f64 = 1.0;

    pub fn init(
        allocator: Allocator,
        graph: *SelfSimilarRelationalGraph,
        esso_opt: *EntangledStochasticSymmetryOptimizer,
        kernel: *ChaosCoreKernel,
    ) Self {
        return Self{
            .graph = graph,
            .esso = esso_opt,
            .chaos_kernel = kernel,
            .phase_history = ArrayList(ReasoningPhase).init(allocator),
            .statistics = OrchestratorStatistics.init(),
            .phase_level_counts = StringHashMap(usize).init(allocator),
            .pattern_energy_index = AutoHashMap(PatternId, f64).init(allocator),
            .fast_inner_steps = 5,
            .slow_outer_steps = 2,
            .hierarchical_depth = 2,
            .perturb_node_limit = 10,
            .update_edge_limit = 10,
            .transform_node_limit = 5,
            .next_phase_id = 1,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.phase_history.items) |*phase| {
            phase.deinit();
        }
        self.phase_history.deinit();
        self.phase_level_counts.deinit();
        self.pattern_energy_index.deinit();
    }

    pub fn setParameters(self: *Self, inner: usize, outer: usize, depth: usize) void {
        self.fast_inner_steps = inner;
        self.slow_outer_steps = outer;
        self.hierarchical_depth = depth;
    }

    pub fn setProcessingLimits(self: *Self, perturb_nodes: usize, update_edges: usize, transform_nodes: usize) void {
        self.perturb_node_limit = perturb_nodes;
        self.update_edge_limit = update_edges;
        self.transform_node_limit = transform_nodes;
    }

    pub fn getRecordedPhaseCountForLevel(self: *const Self, level: ThoughtLevel) usize {
        return self.phase_level_counts.get(level.toString()) orelse 0;
    }

    pub fn getIndexedPatternEnergy(self: *const Self, pattern_id: PatternId) ?f64 {
        return self.pattern_energy_index.get(pattern_id);
    }

    fn allocatePhaseId(self: *Self) u64 {
        const id = self.next_phase_id;
        self.next_phase_id += 1;
        return id;
    }

    fn createPhase(self: *Self, level: ThoughtLevel, inner: usize, outer: usize, record: bool) ReasoningPhase {
        const phase_id = if (record) self.allocatePhaseId() else 0;
        var phase = ReasoningPhase.init(self.allocator, level, inner, outer, phase_id);
        phase.target_energy = default_target_energy;
        phase.convergence_threshold = default_convergence_threshold;
        return phase;
    }

    fn storePhase(self: *Self, phase: *const ReasoningPhase) !void {
        var stored = try phase.clone(self.allocator);
        errdefer stored.deinit();

        try self.phase_history.ensureUnusedCapacity(1);
        try self.phase_level_counts.ensureUnusedCapacity(1);
        if (phase.pattern_captures.items.len > 0) {
            try self.pattern_energy_index.ensureUnusedCapacity(1);
        }

        const level_key = phase.level.toString();
        const level_count = (self.phase_level_counts.get(level_key) orelse 0) + 1;
        self.phase_history.appendAssumeCapacity(stored);
        self.phase_level_counts.putAssumeCapacity(level_key, level_count);

        if (phase.pattern_captures.items.len > 0) {
            self.pattern_energy_index.putAssumeCapacity(phase.patternDigest(), phase.current_energy);
        }

        self.statistics.recordPhase(phase);
    }

    fn detectSymmetryAllocator(self: *Self) Allocator {
        if (@hasField(EntangledStochasticSymmetryOptimizer, "allocator")) {
            return self.esso.allocator;
        }
        return self.allocator;
    }

    fn freeDetectedSymmetries(self: *Self, transforms: anytype) void {
        self.detectSymmetryAllocator().free(transforms);
    }

    fn isFiniteComplex(value: Complex(f64)) bool {
        return std.math.isFinite(value.re) and std.math.isFinite(value.im);
    }

    fn sanitizeComplex(value: Complex(f64)) Complex(f64) {
        return Complex(f64).init(
            if (std.math.isFinite(value.re)) value.re else 0.0,
            if (std.math.isFinite(value.im)) value.im else 0.0,
        );
    }

    fn clampComplexMagnitude(value: Complex(f64), max_magnitude: f64) Complex(f64) {
        var sanitized = sanitizeComplex(value);
        const magnitude = sanitized.magnitude();
        if (!std.math.isFinite(magnitude)) {
            return Complex(f64).init(0.0, 0.0);
        }
        if (magnitude > max_magnitude and magnitude > 0.0) {
            const scale = max_magnitude / magnitude;
            sanitized.re *= scale;
            sanitized.im *= scale;
        }
        return sanitized;
    }

    fn sanitizeWeight(value: f64) f64 {
        if (!std.math.isFinite(value)) {
            return 0.0;
        }
        return std.math.clamp(value, 0.0, 1.0);
    }

    fn sanitizeFractalDimension(value: f64) f64 {
        if (!std.math.isFinite(value)) {
            return 1.0;
        }
        return std.math.clamp(value, 1.0, 3.0);
    }

    fn normalizeQubit(qubit: *Qubit) void {
        const re_a = if (std.math.isFinite(qubit.a.re)) qubit.a.re else 1.0;
        const im_a = if (std.math.isFinite(qubit.a.im)) qubit.a.im else 0.0;
        const re_b = if (std.math.isFinite(qubit.b.re)) qubit.b.re else 0.0;
        const im_b = if (std.math.isFinite(qubit.b.im)) qubit.b.im else 0.0;
        const mag = std.math.sqrt(re_a * re_a + im_a * im_a + re_b * re_b + im_b * im_b);

        if (std.math.isFinite(mag) and mag > qubit_epsilon) {
            qubit.a.re = re_a / mag;
            qubit.a.im = im_a / mag;
            qubit.b.re = re_b / mag;
            qubit.b.im = im_b / mag;
        } else {
            qubit.a.re = 1.0;
            qubit.a.im = 0.0;
            qubit.b.re = 0.0;
            qubit.b.im = 0.0;
        }
    }

    fn isNormalizedQubit(qubit: *const Qubit) bool {
        const mag_sq = qubit.a.re * qubit.a.re + qubit.a.im * qubit.a.im + qubit.b.re * qubit.b.re + qubit.b.im * qubit.b.im;
        return std.math.isFinite(mag_sq) and @abs(mag_sq - 1.0) < 1e-6;
    }

    fn restoreNodeSnapshots(snapshots: []const NodeStateSnapshot) void {
        for (snapshots) |snapshot| {
            snapshot.node.qubit = snapshot.qubit;
            snapshot.node.phase = snapshot.phase;
        }
    }

    fn restoreEdgeSnapshots(snapshots: []const EdgeStateSnapshot) void {
        for (snapshots) |snapshot| {
            snapshot.edge.weight = snapshot.weight;
            snapshot.edge.fractal_dimension = snapshot.fractal_dimension;
            snapshot.edge.quantum_correlation = snapshot.quantum_correlation;
        }
    }

    fn captureLocalNodeSnapshots(self: *Self, snapshots: *ArrayList(NodeStateSnapshot)) !usize {
        var node_iter = self.graph.nodes.iterator();
        var count: usize = 0;
        while (node_iter.next()) |entry| {
            if (count >= self.perturb_node_limit) break;
            const node = entry.value_ptr;
            try snapshots.append(NodeStateSnapshot{
                .node = node,
                .qubit = node.qubit,
                .phase = node.phase,
            });
            count += 1;
        }
        return count;
    }

    fn captureLocalEdgeSnapshots(self: *Self, snapshots: *ArrayList(EdgeStateSnapshot)) !usize {
        var edge_iter = self.graph.edges.iterator();
        var modified_edges: usize = 0;
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |*edge| {
                if (modified_edges >= self.update_edge_limit) break;
                try snapshots.append(EdgeStateSnapshot{
                    .edge = edge,
                    .weight = edge.weight,
                    .fractal_dimension = edge.fractal_dimension,
                    .quantum_correlation = edge.quantum_correlation,
                });
                modified_edges += 1;
            }
            if (modified_edges >= self.update_edge_limit) break;
        }
        return modified_edges;
    }

    fn captureFullGraphState(self: *Self, node_snapshots: *ArrayList(NodeStateSnapshot), edge_snapshots: *ArrayList(EdgeStateSnapshot)) !void {
        var node_iter = self.graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            try node_snapshots.append(NodeStateSnapshot{
                .node = node,
                .qubit = node.qubit,
                .phase = node.phase,
            });
        }

        var edge_iter = self.graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |*edge| {
                try edge_snapshots.append(EdgeStateSnapshot{
                    .edge = edge,
                    .weight = edge.weight,
                    .fractal_dimension = edge.fractal_dimension,
                    .quantum_correlation = edge.quantum_correlation,
                });
            }
        }
    }

    fn perturbLocalNodes(self: *Self, snapshots: []const NodeStateSnapshot) void {
        for (snapshots) |snapshot| {
            const node = snapshot.node;
            const perturbation_a = (self.esso.prng.random().float(f64) - 0.5) * 0.1;
            const perturbation_b = (self.esso.prng.random().float(f64) - 0.5) * 0.1;
            const phase_delta = (perturbation_a + perturbation_b) * 0.5;

            node.phase = if (std.math.isFinite(node.phase + phase_delta)) node.phase + phase_delta else 0.0;

            const perturb_scale = 0.01;
            node.qubit.a.re = if (std.math.isFinite(node.qubit.a.re + perturbation_a * perturb_scale)) node.qubit.a.re + perturbation_a * perturb_scale else 1.0;
            node.qubit.a.im = if (std.math.isFinite(node.qubit.a.im + perturbation_a * perturb_scale)) node.qubit.a.im + perturbation_a * perturb_scale else 0.0;
            node.qubit.b.re = if (std.math.isFinite(node.qubit.b.re + perturbation_b * perturb_scale)) node.qubit.b.re + perturbation_b * perturb_scale else 0.0;
            node.qubit.b.im = if (std.math.isFinite(node.qubit.b.im + perturbation_b * perturb_scale)) node.qubit.b.im + perturbation_b * perturb_scale else 0.0;

            normalizeQubit(&node.qubit);
        }
    }

    fn updateLocalEdges(self: *Self, snapshots: []const EdgeStateSnapshot) void {
        for (snapshots) |snapshot| {
            const edge = snapshot.edge;
            const delta = (self.esso.prng.random().float(f64) - 0.5) * 0.05;

            if (std.math.isFinite(delta)) {
                edge.weight = std.math.clamp(sanitizeWeight(edge.weight) + delta, 0.0, 1.0);

                const corr_delta = delta * 0.1;
                const base_corr = sanitizeComplex(edge.quantum_correlation);
                edge.quantum_correlation = clampComplexMagnitude(
                    Complex(f64).init(base_corr.re + corr_delta, base_corr.im + corr_delta * 0.5),
                    max_quantum_correlation_magnitude,
                );
            } else {
                edge.weight = sanitizeWeight(edge.weight);
                edge.quantum_correlation = clampComplexMagnitude(edge.quantum_correlation, max_quantum_correlation_magnitude);
            }

            edge.fractal_dimension = sanitizeFractalDimension(edge.fractal_dimension);
        }
    }

    fn executeLocalPhaseInternal(self: *Self, record: bool) !f64 {
        var phase = self.createPhase(.local, self.fast_inner_steps, 1, record);
        defer phase.deinit();

        const initial_energy = self.computeGraphEnergy();
        phase.previous_energy = initial_energy;
        phase.current_energy = initial_energy;

        var node_snapshots = ArrayList(NodeStateSnapshot).init(self.allocator);
        defer node_snapshots.deinit();

        var edge_snapshots = ArrayList(EdgeStateSnapshot).init(self.allocator);
        defer edge_snapshots.deinit();

        if (!phase.hasMetTargetEnergy() and phase.configured_inner_iterations > 0) {
            phase.outer_iterations = 1;
        }

        while (phase.canRunInnerIteration()) {
            node_snapshots.clearRetainingCapacity();
            edge_snapshots.clearRetainingCapacity();

            _ = try self.captureLocalNodeSnapshots(&node_snapshots);
            _ = try self.captureLocalEdgeSnapshots(&edge_snapshots);

            const energy_before = phase.current_energy;
            self.perturbLocalNodes(node_snapshots.items);
            self.updateLocalEdges(edge_snapshots.items);

            const proposed_energy = self.computeGraphEnergy();
            if (std.math.isFinite(proposed_energy) and proposed_energy <= energy_before) {
                phase.updateEnergy(proposed_energy);
            } else {
                restoreNodeSnapshots(node_snapshots.items);
                restoreEdgeSnapshots(edge_snapshots.items);
                phase.updateEnergy(energy_before);
            }

            phase.inner_iterations += 1;

            if (phase.hasMetTargetEnergy()) {
                break;
            }

            if (phase.inner_iterations > 1 and phase.hasConverged()) {
                break;
            }
        }

        phase.finalize();
        const final_energy = phase.current_energy;

        if (record) {
            try self.storePhase(&phase);
        }

        return final_energy;
    }

    pub fn executeLocalPhase(self: *Self) !f64 {
        return self.executeLocalPhaseInternal(true);
    }

    fn writeU64Little(id: *PatternId, offset: usize, value: u64) void {
        var i: usize = 0;
        while (i < 8) : (i += 1) {
            const shift: u6 = @intCast(i * 8);
            id[offset + i] = @as(u8, @truncate(value >> shift));
        }
    }

    fn patternIdFromBytes(bytes: []const u8, index: usize) PatternId {
        var id = [_]u8{0} ** 32;
        const seed0 = @as(u64, @intCast(index));
        const seed1 = seed0 ^ 0x9e3779b97f4a7c15;
        const seed2 = seed0 ^ 0xbf58476d1ce4e5b9;
        const seed3 = seed0 ^ 0x94d049bb133111eb;
        writeU64Little(&id, 0, std.hash.Wyhash.hash(seed0, bytes));
        writeU64Little(&id, 8, std.hash.Wyhash.hash(seed1, bytes));
        writeU64Little(&id, 16, std.hash.Wyhash.hash(seed2, bytes));
        writeU64Little(&id, 24, std.hash.Wyhash.hash(seed3, bytes));
        return id;
    }

    fn patternIdForTransform(transform: anytype, index: usize) PatternId {
        var buffer = [_]u8{0} ** 1024;
        const rendered = std.fmt.bufPrint(buffer[0..], "{any}", .{transform}) catch buffer[0..];
        return patternIdFromBytes(rendered, index);
    }

    fn transformSymmetryPatterns(self: *Self, phase: *ReasoningPhase) !void {
        const transforms = try self.esso.detectSymmetries(self.graph);
        defer self.freeDetectedSymmetries(transforms);

        try phase.pattern_captures.ensureUnusedCapacity(transforms.len);

        for (transforms, 0..) |transform, transform_index| {
            phase.recordPatternAssumeCapacity(patternIdForTransform(transform, transform_index));

            var node_iter = self.graph.nodes.iterator();
            var count: usize = 0;
            while (node_iter.next()) |entry| {
                if (count >= self.transform_node_limit) break;

                const node = entry.value_ptr;
                const quantum_state = quantum.QuantumState{
                    .amplitudes = .{
                        Complex(f64).init(node.qubit.a.re, node.qubit.a.im),
                        Complex(f64).init(node.qubit.b.re, node.qubit.b.im),
                    },
                    .phase = node.phase,
                    .entanglement_degree = 0.0,
                };

                const transformed = transform.applyToQuantumState(&quantum_state);

                if (std.math.isFinite(transformed.amplitudes[0].re)) {
                    node.qubit.a.re = transformed.amplitudes[0].re;
                }
                if (std.math.isFinite(transformed.amplitudes[0].im)) {
                    node.qubit.a.im = transformed.amplitudes[0].im;
                }
                if (std.math.isFinite(transformed.amplitudes[1].re)) {
                    node.qubit.b.re = transformed.amplitudes[1].re;
                }
                if (std.math.isFinite(transformed.amplitudes[1].im)) {
                    node.qubit.b.im = transformed.amplitudes[1].im;
                }
                if (std.math.isFinite(transformed.phase)) {
                    node.phase = transformed.phase;
                }

                normalizeQubit(&node.qubit);
                count += 1;
            }
        }

        var norm_iter = self.graph.nodes.iterator();
        while (norm_iter.next()) |entry| {
            const node = entry.value_ptr;
            normalizeQubit(&node.qubit);
        }
    }

    fn rebalanceFractalStructures(self: *Self) void {
        var edge_iter = self.graph.edges.iterator();
        var total_dimension: f64 = 0.0;
        var edge_count: usize = 0;

        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |edge| {
                total_dimension += sanitizeFractalDimension(edge.fractal_dimension);
                edge_count += 1;
            }
        }

        if (edge_count > 0) {
            const avg_dimension = total_dimension / @as(f64, @floatFromInt(edge_count));
            edge_iter = self.graph.edges.iterator();
            while (edge_iter.next()) |entry| {
                for (entry.value_ptr.items) |*edge| {
                    const current_dimension = sanitizeFractalDimension(edge.fractal_dimension);
                    const adjustment = (avg_dimension - current_dimension) * 0.1;
                    edge.fractal_dimension = std.math.clamp(current_dimension + adjustment, 1.0, 3.0);
                    edge.weight = sanitizeWeight(edge.weight);
                    edge.quantum_correlation = clampComplexMagnitude(edge.quantum_correlation, max_quantum_correlation_magnitude);
                }
            }
        }
    }

    fn applyChaosRelaxationToMeasuredGraph(self: *Self, cycle_index: usize) void {
        const phase_scale = 0.999;
        const correlation_scale = 0.999;
        const phase_bias = @sin(@as(f64, @floatFromInt(cycle_index + 1)) * 0.6180339887498948482) * 1e-6;

        var node_iter = self.graph.nodes.iterator();
        var node_count: usize = 0;
        while (node_iter.next()) |entry| {
            if (node_count >= self.transform_node_limit) break;
            const node = entry.value_ptr;
            node.phase = if (std.math.isFinite(node.phase)) node.phase * phase_scale + phase_bias else phase_bias;
            normalizeQubit(&node.qubit);
            node_count += 1;
        }

        var edge_iter = self.graph.edges.iterator();
        var edge_count: usize = 0;
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |*edge| {
                if (edge_count >= self.update_edge_limit) break;
                edge.weight = sanitizeWeight(edge.weight);
                edge.fractal_dimension = sanitizeFractalDimension(edge.fractal_dimension);

                const corr = clampComplexMagnitude(edge.quantum_correlation, max_quantum_correlation_magnitude);
                edge.quantum_correlation = clampComplexMagnitude(
                    Complex(f64).init(corr.re * correlation_scale, corr.im * correlation_scale),
                    max_quantum_correlation_magnitude,
                );

                edge_count += 1;
            }
            if (edge_count >= self.update_edge_limit) break;
        }
    }

    fn executeMeasuredChaosCycle(self: *Self, cycle_index: usize) !void {
        try self.chaos_kernel.executeCycle();
        self.applyChaosRelaxationToMeasuredGraph(cycle_index);
    }

    fn executeGlobalPhaseInternal(self: *Self, record: bool) !f64 {
        var phase = self.createPhase(.global, self.fast_inner_steps, self.slow_outer_steps, record);
        defer phase.deinit();

        const initial_energy = self.computeGraphEnergy();
        phase.previous_energy = initial_energy;
        phase.current_energy = initial_energy;

        var node_snapshots = ArrayList(NodeStateSnapshot).init(self.allocator);
        defer node_snapshots.deinit();

        var edge_snapshots = ArrayList(EdgeStateSnapshot).init(self.allocator);
        defer edge_snapshots.deinit();

        while (phase.canRunOuterIteration()) {
            if (phase.hasMetTargetEnergy()) {
                break;
            }

            node_snapshots.clearRetainingCapacity();
            edge_snapshots.clearRetainingCapacity();
            try self.captureFullGraphState(&node_snapshots, &edge_snapshots);

            {
                var rollback_active = true;
                errdefer if (rollback_active) {
                    restoreNodeSnapshots(node_snapshots.items);
                    restoreEdgeSnapshots(edge_snapshots.items);
                };

                const energy_before = phase.current_energy;

                try self.transformSymmetryPatterns(&phase);
                self.rebalanceFractalStructures();

                var inner_this_outer: usize = 0;
                while (inner_this_outer < phase.configured_inner_iterations) : (inner_this_outer += 1) {
                    try self.executeMeasuredChaosCycle(inner_this_outer);
                    phase.inner_iterations += 1;
                }

                const proposed_energy = self.computeGraphEnergy();
                if (std.math.isFinite(proposed_energy) and proposed_energy <= energy_before) {
                    phase.updateEnergy(proposed_energy);
                } else {
                    restoreNodeSnapshots(node_snapshots.items);
                    restoreEdgeSnapshots(edge_snapshots.items);
                    phase.updateEnergy(energy_before);
                }

                rollback_active = false;
            }

            phase.outer_iterations += 1;

            if (phase.hasMetTargetEnergy()) {
                break;
            }

            if (phase.outer_iterations > 1 and phase.hasConverged()) {
                break;
            }
        }

        phase.finalize();
        const final_energy = phase.current_energy;

        if (record) {
            try self.storePhase(&phase);
        }

        return final_energy;
    }

    pub fn executeGlobalPhase(self: *Self) !f64 {
        return self.executeGlobalPhaseInternal(true);
    }

    pub fn executeMetaPhase(self: *Self) !f64 {
        var phase = self.createPhase(.meta, self.fast_inner_steps, self.hierarchical_depth, true);
        defer phase.deinit();

        const initial_energy = self.computeGraphEnergy();
        phase.previous_energy = initial_energy;
        phase.current_energy = initial_energy;

        while (phase.outer_iterations < self.hierarchical_depth) {
            if (phase.hasMetTargetEnergy()) {
                break;
            }

            const sub_energy = if (phase.outer_iterations % 2 == 0)
                try self.executeLocalPhaseInternal(true)
            else
                try self.executeGlobalPhaseInternal(true);

            phase.updateEnergy(sub_energy);
            phase.outer_iterations += 1;

            if (phase.hasMetTargetEnergy()) {
                break;
            }

            if (phase.outer_iterations > 1 and phase.hasConverged()) {
                break;
            }
        }

        phase.finalize();
        const final_energy = phase.current_energy;
        try self.storePhase(&phase);

        return final_energy;
    }

    fn computeGraphEnergy(self: *Self) f64 {
        var total_energy: f64 = 0.0;
        var count: usize = 0;

        var edge_iter = self.graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |edge| {
                const weight = sanitizeWeight(edge.weight);
                const fractal_dimension = sanitizeFractalDimension(edge.fractal_dimension);
                const structural_energy = weight * fractal_dimension;

                var correlation_energy: f64 = 1e6;
                if (isFiniteComplex(edge.quantum_correlation)) {
                    const corr = clampComplexMagnitude(edge.quantum_correlation, max_quantum_correlation_magnitude);
                    const magnitude = corr.magnitude();
                    if (std.math.isFinite(magnitude)) {
                        correlation_energy = magnitude;
                    }
                }

                total_energy += structural_energy + correlation_energy;
                count += 1;

                if (!std.math.isFinite(total_energy)) {
                    return 1e6;
                }
            }
        }

        var node_iter = self.graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            if (std.math.isFinite(node.phase)) {
                const cos_phase = @cos(node.phase);
                if (std.math.isFinite(cos_phase)) {
                    total_energy += (1.0 - cos_phase * cos_phase) / 2.0;
                } else {
                    total_energy += 1.0;
                }
            } else {
                total_energy += 1.0;
            }
            count += 1;

            if (!std.math.isFinite(total_energy)) {
                return 1e6;
            }
        }

        if (count > 0 and std.math.isFinite(total_energy)) {
            return total_energy / @as(f64, @floatFromInt(count));
        }
        return 1e6;
    }

    pub fn runHierarchicalReasoning(self: *Self, max_cycles: usize) !f64 {
        const result = try self.runHierarchicalReasoningFull(max_cycles);
        return result.best_energy;
    }

    fn safeModulationFromEnergy(energy: f64) f64 {
        const safe_energy = if (std.math.isFinite(energy) and energy >= 0.0) energy else 1e6;
        const modulation = 1.0 / (1.0 + safe_energy);
        if (std.math.isFinite(modulation) and modulation > 0.0) {
            return modulation;
        }
        return 1.0;
    }

    pub fn runHierarchicalReasoningFull(self: *Self, max_cycles: usize) !ReasoningResult {
        const starting_phases = self.statistics.total_phases;
        const starting_patterns = self.statistics.patterns_discovered;

        var best_energy = self.computeGraphEnergy();
        if (!std.math.isFinite(best_energy)) {
            best_energy = 1e6;
        }

        if (max_cycles == 0 or best_energy <= default_target_energy) {
            return ReasoningResult{
                .best_energy = best_energy,
                .modulation_factor = safeModulationFromEnergy(best_energy),
                .phases_completed = self.statistics.total_phases - starting_phases,
                .patterns_found = self.statistics.patterns_discovered - starting_patterns,
            };
        }

        var prev_combined = best_energy;
        var cycle: usize = 0;

        while (cycle < max_cycles) : (cycle += 1) {
            var combined_sum: f64 = 0.0;
            var combined_count: usize = 0;

            if (best_energy <= default_target_energy) {
                break;
            }

            const local_e = try self.executeLocalPhase();
            if (std.math.isFinite(local_e)) {
                combined_sum += local_e;
                combined_count += 1;
                if (local_e < best_energy) {
                    best_energy = local_e;
                }
            }

            if (best_energy <= default_target_energy) {
                break;
            }

            const global_e = try self.executeGlobalPhase();
            if (std.math.isFinite(global_e)) {
                combined_sum += global_e;
                combined_count += 1;
                if (global_e < best_energy) {
                    best_energy = global_e;
                }
            }

            if (best_energy <= default_target_energy) {
                break;
            }

            if (max_cycles > 1) {
                const meta_e = try self.executeMetaPhase();
                if (std.math.isFinite(meta_e)) {
                    combined_sum += meta_e;
                    combined_count += 1;
                    if (meta_e < best_energy) {
                        best_energy = meta_e;
                    }
                }
            }

            const combined = if (combined_count > 0) combined_sum / @as(f64, @floatFromInt(combined_count)) else best_energy;
            if (std.math.isFinite(combined) and combined < best_energy) {
                best_energy = combined;
            }

            if (cycle > 0 and std.math.isFinite(combined) and std.math.isFinite(prev_combined)) {
                const delta = @abs(combined - prev_combined);
                const denom = @max(@abs(prev_combined), 1.0);
                if (std.math.isFinite(delta) and std.math.isFinite(denom) and denom > 0.0 and (delta / denom) < default_convergence_threshold) {
                    break;
                }
            }

            if (std.math.isFinite(combined)) {
                prev_combined = combined;
            }

            if (std.math.isFinite(combined) and combined <= default_target_energy) {
                break;
            }
        }

        if (!std.math.isFinite(best_energy)) {
            best_energy = self.computeGraphEnergy();
            if (!std.math.isFinite(best_energy)) {
                best_energy = 1e6;
            }
        }

        return ReasoningResult{
            .best_energy = best_energy,
            .modulation_factor = safeModulationFromEnergy(best_energy),
            .phases_completed = self.statistics.total_phases - starting_phases,
            .patterns_found = self.statistics.patterns_discovered - starting_patterns,
        };
    }

    pub fn modulateTensor(_: *const Self, data: []f32, modulation: f64) void {
        if (data.len == 0) return;

        const max_f32_as_f64: f64 = 3.4028234663852886e38;
        var safe_scale64 = if (std.math.isFinite(modulation) and @abs(modulation) > 1e-12) modulation else 1.0;
        safe_scale64 = std.math.clamp(safe_scale64, -max_f32_as_f64, max_f32_as_f64);

        var scale: f32 = @as(f32, @floatCast(safe_scale64));
        if (!std.math.isFinite(scale) or scale == 0.0) {
            scale = 1.0;
        }

        var i: usize = 0;
        while (i < data.len) : (i += 1) {
            data[i] *= scale;
        }
    }

    fn applyScalarModulationToGraph(self: *Self, modulation: f64) void {
        const scale = if (std.math.isFinite(modulation) and modulation > 1e-12) modulation else 1.0;

        var edge_iter = self.graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |*edge| {
                edge.weight = sanitizeWeight(edge.weight) * scale;

                const corr = clampComplexMagnitude(edge.quantum_correlation, max_quantum_correlation_magnitude);
                edge.quantum_correlation = clampComplexMagnitude(
                    Complex(f64).init(corr.re * scale, corr.im * scale),
                    max_quantum_correlation_magnitude,
                );

                edge.fractal_dimension = sanitizeFractalDimension(edge.fractal_dimension);
            }
        }

        var node_iter = self.graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            node.phase = if (std.math.isFinite(node.phase)) node.phase * scale else 0.0;
            normalizeQubit(&node.qubit);
        }
    }

    pub fn applyModulationToGraph(self: *Self) !f64 {
        const result = try self.runHierarchicalReasoningFull(1);
        self.applyScalarModulationToGraph(result.modulation_factor);
        return result.modulation_factor;
    }

    pub fn getStatistics(self: *const Self) OrchestratorStatistics {
        return self.statistics;
    }

    pub fn getPhaseHistory(self: *const Self) []const ReasoningPhase {
        return self.phase_history.items;
    }
};

test "reasoning_orchestrator_local_phase" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "n1", "data1", Qubit.initBasis0(), 0.1);
    try graph.addNode(n1);
    const n2 = try Node.init(allocator, "n2", "data2", Qubit.initBasis1(), 0.2);
    try graph.addNode(n2);

    var esso_opt = EntangledStochasticSymmetryOptimizer.initWithSeed(allocator, 10.0, 0.9, 100, 12345);
    defer esso_opt.deinit();

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var orchestrator = ReasoningOrchestrator.init(allocator, &graph, &esso_opt, &kernel);
    defer orchestrator.deinit();

    orchestrator.setProcessingLimits(10, 10, 10);

    const initial_energy = orchestrator.computeGraphEnergy();
    const energy = try orchestrator.executeLocalPhase();

    try std.testing.expect(std.math.isFinite(energy));
    try std.testing.expect(energy <= initial_energy or @abs(energy - initial_energy) < 1e-9);

    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        try std.testing.expect(ReasoningOrchestrator.isNormalizedQubit(&entry.value_ptr.qubit));
        try std.testing.expect(std.math.isFinite(entry.value_ptr.phase));
    }

    const stats = orchestrator.getStatistics();
    try std.testing.expect(stats.total_phases == 1);
    try std.testing.expect(stats.local_phases == 1);
    try std.testing.expect(stats.total_inner_loops <= orchestrator.fast_inner_steps);
    try std.testing.expect(stats.total_outer_loops <= 1);
    try std.testing.expect(orchestrator.getPhaseHistory().len == 1);
    try std.testing.expect(orchestrator.getRecordedPhaseCountForLevel(.local) == 1);
}

test "reasoning_orchestrator_global_phase" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "n1", "data1", Qubit.initBasis0(), 0.1);
    try graph.addNode(n1);

    var esso_opt = EntangledStochasticSymmetryOptimizer.initWithSeed(allocator, 10.0, 0.9, 50, 42);
    defer esso_opt.deinit();

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var orchestrator = ReasoningOrchestrator.init(allocator, &graph, &esso_opt, &kernel);
    defer orchestrator.deinit();

    orchestrator.setParameters(4, 3, 3);
    orchestrator.setProcessingLimits(10, 10, 10);

    const initial_energy = orchestrator.computeGraphEnergy();
    const energy = try orchestrator.executeGlobalPhase();

    try std.testing.expect(std.math.isFinite(energy));
    try std.testing.expect(energy <= initial_energy or @abs(energy - initial_energy) < 1e-9);

    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        try std.testing.expect(ReasoningOrchestrator.isNormalizedQubit(&entry.value_ptr.qubit));
        try std.testing.expect(std.math.isFinite(entry.value_ptr.phase));
    }

    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            try std.testing.expect(std.math.isFinite(edge.weight));
            try std.testing.expect(edge.weight >= 0.0 and edge.weight <= 1.0);
            try std.testing.expect(std.math.isFinite(edge.fractal_dimension));
            try std.testing.expect(edge.fractal_dimension >= 1.0 and edge.fractal_dimension <= 3.0);
            try std.testing.expect(std.math.isFinite(edge.quantum_correlation.re));
            try std.testing.expect(std.math.isFinite(edge.quantum_correlation.im));
            try std.testing.expect(edge.quantum_correlation.magnitude() <= 1.0 + 1e-9);
        }
    }

    const stats = orchestrator.getStatistics();
    try std.testing.expect(stats.total_phases == 1);
    try std.testing.expect(stats.global_phases == 1);
    try std.testing.expect(stats.total_outer_loops <= orchestrator.slow_outer_steps);
    try std.testing.expect(stats.total_inner_loops <= orchestrator.slow_outer_steps * orchestrator.fast_inner_steps);
    try std.testing.expect(orchestrator.getPhaseHistory().len == 1);
    try std.testing.expect(orchestrator.getRecordedPhaseCountForLevel(.global) == 1);
}

test "reasoning_orchestrator_hierarchical_reasoning_zero_cycles" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "n1", "data1", Qubit.initBasis0(), 0.1);
    try graph.addNode(n1);

    var esso_opt = EntangledStochasticSymmetryOptimizer.initWithSeed(allocator, 10.0, 0.9, 50, 99);
    defer esso_opt.deinit();

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var orchestrator = ReasoningOrchestrator.init(allocator, &graph, &esso_opt, &kernel);
    defer orchestrator.deinit();

    const result = try orchestrator.runHierarchicalReasoningFull(0);
    try std.testing.expect(std.math.isFinite(result.best_energy));
    try std.testing.expect(std.math.isFinite(result.modulation_factor));
    try std.testing.expect(result.modulation_factor > 0.0);
    try std.testing.expect(result.phases_completed == 0);
    try std.testing.expect(result.patterns_found == 0);
}

test "reasoning_orchestrator_meta_phase_uses_hierarchical_depth" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "n1", "data1", Qubit.initBasis0(), 0.1);
    try graph.addNode(n1);
    const n2 = try Node.init(allocator, "n2", "data2", Qubit.initBasis1(), 0.2);
    try graph.addNode(n2);

    var esso_opt = EntangledStochasticSymmetryOptimizer.initWithSeed(allocator, 10.0, 0.9, 50, 777);
    defer esso_opt.deinit();

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var orchestrator = ReasoningOrchestrator.init(allocator, &graph, &esso_opt, &kernel);
    defer orchestrator.deinit();

    orchestrator.setParameters(2, 2, 2);
    orchestrator.setProcessingLimits(2, 2, 2);

    const energy = try orchestrator.executeMetaPhase();
    try std.testing.expect(std.math.isFinite(energy));

    const stats = orchestrator.getStatistics();
    try std.testing.expect(stats.meta_phases == 1);
    try std.testing.expect(stats.total_phases >= 1);
    try std.testing.expect(orchestrator.getRecordedPhaseCountForLevel(.meta) == 1);
}

test "reasoning_orchestrator_modulate_tensor_invertible_scale" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    var esso_opt = EntangledStochasticSymmetryOptimizer.initWithSeed(allocator, 10.0, 0.9, 50, 7);
    defer esso_opt.deinit();

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var orchestrator = ReasoningOrchestrator.init(allocator, &graph, &esso_opt, &kernel);
    defer orchestrator.deinit();

    var data = [_]f32{ 1.0, -2.0, 3.5 };
    orchestrator.modulateTensor(data[0..], 0.0);
    try std.testing.expect(data[0] == 1.0);
    try std.testing.expect(data[1] == -2.0);
    try std.testing.expect(data[2] == 3.5);

    orchestrator.modulateTensor(data[0..], 0.5);
    try std.testing.expect(data[0] == 0.5);
    try std.testing.expect(data[1] == -1.0);
    try std.testing.expect(data[2] == 1.75);
}
