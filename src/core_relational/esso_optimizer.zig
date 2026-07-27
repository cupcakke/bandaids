const std = @import("std");
const nsir_core = @import("nsir_core.zig");
const quantum_logic = @import("quantum_logic.zig");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Complex = std.math.Complex;

const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
const Node = nsir_core.Node;
const Edge = nsir_core.Edge;
const EdgeQuality = nsir_core.EdgeQuality;
const EdgeKey = nsir_core.EdgeKey;
const Qubit = nsir_core.Qubit;

const QuantumState = quantum_logic.QuantumState;
const RelationalQuantumLogic = quantum_logic.RelationalQuantumLogic;
const LogicGate = quantum_logic.LogicGate;

pub const ObjectiveFunction = *const fn (*const OptimizationState) f64;

fn nowNs() i64 {
    const t: i128 = std.time.nanoTimestamp();
    const max_i64: i128 = std.math.maxInt(i64);
    const min_i64: i128 = std.math.minInt(i64);
    if (t > max_i64) return std.math.maxInt(i64);
    if (t < min_i64) return std.math.minInt(i64);
    return @intCast(t);
}

fn normalizeAngle(angle: f64) f64 {
    var result = @mod(angle, 2.0 * std.math.pi);
    if (result < 0.0) result += 2.0 * std.math.pi;
    return result;
}

fn finiteOr(value: f64, fallback: f64) f64 {
    return if (std.math.isFinite(value)) value else fallback;
}

pub fn cloneGraph(allocator: Allocator, source: *const SelfSimilarRelationalGraph) !SelfSimilarRelationalGraph {
    var new_graph = try SelfSimilarRelationalGraph.init(allocator);
    errdefer new_graph.deinit();

    var node_iter = source.nodes.iterator();
    while (node_iter.next()) |entry| {
        var cloned_node = try entry.value_ptr.clone(allocator);
        var added = false;
        errdefer if (!added) cloned_node.deinit();
        try new_graph.addNode(cloned_node);
        added = true;
    }

    var edge_iter = source.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            var cloned_edge = try edge.clone(allocator);
            var added = false;
            errdefer if (!added) cloned_edge.deinit();
            try new_graph.addEdge(cloned_edge.source, cloned_edge.target, cloned_edge);
            added = true;
        }
    }

    try new_graph.updateTopologyHash();

    return new_graph;
}

pub const SymmetryGroup = enum(u8) {
    identity = 0,
    reflection = 1,
    rotation_90 = 2,
    rotation_180 = 3,
    rotation_270 = 4,
    translation = 5,
    custom_rotation = 6,

    pub fn toString(self: SymmetryGroup) []const u8 {
        return switch (self) {
            .identity => "identity",
            .reflection => "reflection",
            .rotation_90 => "rotation_90",
            .rotation_180 => "rotation_180",
            .rotation_270 => "rotation_270",
            .translation => "translation",
            .custom_rotation => "custom_rotation",
        };
    }

    pub fn fromString(s: []const u8) ?SymmetryGroup {
        if (std.mem.eql(u8, s, "identity")) return .identity;
        if (std.mem.eql(u8, s, "reflection")) return .reflection;
        if (std.mem.eql(u8, s, "rotation_90")) return .rotation_90;
        if (std.mem.eql(u8, s, "rotation_180")) return .rotation_180;
        if (std.mem.eql(u8, s, "rotation_270")) return .rotation_270;
        if (std.mem.eql(u8, s, "translation")) return .translation;
        if (std.mem.eql(u8, s, "custom_rotation")) return .custom_rotation;
        return null;
    }

    pub fn getAngle(self: SymmetryGroup) f64 {
        return switch (self) {
            .identity => 0.0,
            .reflection => 0.0,
            .rotation_90 => std.math.pi / 2.0,
            .rotation_180 => std.math.pi,
            .rotation_270 => 3.0 * std.math.pi / 2.0,
            .translation => 0.0,
            .custom_rotation => 0.0,
        };
    }

    pub fn getOrder(self: SymmetryGroup) usize {
        return switch (self) {
            .identity => 1,
            .reflection => 2,
            .rotation_90 => 4,
            .rotation_180 => 2,
            .rotation_270 => 4,
            .translation => 1,
            .custom_rotation => 2,
        };
    }

    pub fn effectiveOrder(angle: f64) usize {
        const two_pi = 2.0 * std.math.pi;
        if (@abs(angle) < 1e-10) return 1;
        const ratio = two_pi / angle;
        const rounded = @round(ratio);
        if (@abs(ratio - rounded) < 0.01 and rounded > 0 and rounded < 1000) {
            return @as(usize, @intFromFloat(rounded));
        }
        return 0;
    }
};

const SymmetryTransform = struct {
    group: SymmetryGroup,
    origin_x: f64,
    origin_y: f64,
    parameters: [4]f64,
    scale_factor: f64,

    const Self = @This();

    const Affine = struct {
        m: [2][2]f64,
        tx: f64,
        ty: f64,
    };

    pub fn init(group: SymmetryGroup) Self {
        return Self{
            .group = group,
            .origin_x = 0.0,
            .origin_y = 0.0,
            .parameters = [4]f64{ 0.0, 0.0, 1.0, 0.0 },
            .scale_factor = 1.0,
        };
    }

    pub fn initWithParams(group: SymmetryGroup, params: [4]f64) Self {
        const safe_scale = if (std.math.isFinite(params[2]) and params[2] > 0.0) params[2] else 1.0;
        return Self{
            .group = group,
            .origin_x = finiteOr(params[0], 0.0),
            .origin_y = finiteOr(params[1], 0.0),
            .parameters = [4]f64{ finiteOr(params[0], 0.0), finiteOr(params[1], 0.0), safe_scale, finiteOr(params[3], 0.0) },
            .scale_factor = safe_scale,
        };
    }

    pub fn effectiveAngle(self: *const Self) f64 {
        return switch (self.group) {
            .identity, .reflection, .translation => 0.0,
            .rotation_90 => std.math.pi / 2.0,
            .rotation_180 => std.math.pi,
            .rotation_270 => 3.0 * std.math.pi / 2.0,
            .custom_rotation => self.parameters[3],
        };
    }

    pub fn apply(self: *const Self, x: f64, y: f64) struct { x: f64, y: f64 } {
        const dx = x - self.origin_x;
        const dy = y - self.origin_y;

        return switch (self.group) {
            .identity => .{
                .x = self.origin_x + dx * self.scale_factor,
                .y = self.origin_y + dy * self.scale_factor,
            },
            .reflection => .{
                .x = self.origin_x + (dx * @cos(2.0 * self.parameters[3]) + dy * @sin(2.0 * self.parameters[3])) * self.scale_factor,
                .y = self.origin_y + (dx * @sin(2.0 * self.parameters[3]) - dy * @cos(2.0 * self.parameters[3])) * self.scale_factor,
            },
            .rotation_90 => .{
                .x = self.origin_x - dy * self.scale_factor,
                .y = self.origin_y + dx * self.scale_factor,
            },
            .rotation_180 => .{
                .x = self.origin_x - dx * self.scale_factor,
                .y = self.origin_y - dy * self.scale_factor,
            },
            .rotation_270 => .{
                .x = self.origin_x + dy * self.scale_factor,
                .y = self.origin_y - dx * self.scale_factor,
            },
            .custom_rotation => blk: {
                const c = @cos(self.parameters[3]);
                const s = @sin(self.parameters[3]);
                break :blk .{
                    .x = self.origin_x + (dx * c - dy * s) * self.scale_factor,
                    .y = self.origin_y + (dx * s + dy * c) * self.scale_factor,
                };
            },
            .translation => .{
                .x = x + self.parameters[0] * self.scale_factor,
                .y = y + self.parameters[1] * self.scale_factor,
            },
        };
    }

    pub fn applyToComplex(self: *const Self, z: Complex(f64)) Complex(f64) {
        const result = self.apply(z.re, z.im);
        return Complex(f64).init(result.x, result.y);
    }

    pub fn applyToQuantumState(self: *const Self, state: *const QuantumState) QuantumState {
        var new_real = state.amplitudes[0].re;
        var new_imag = state.amplitudes[0].im;
        var new_phase = state.phase;

        switch (self.group) {
            .identity, .translation => {},
            .reflection => {
                const a = 2.0 * self.parameters[3];
                const ca = @cos(a);
                const sa = @sin(a);
                const r = state.amplitudes[0].re;
                const im = state.amplitudes[0].im;
                new_real = r * ca + im * sa;
                new_imag = r * sa - im * ca;
                new_phase = -state.phase + 2.0 * self.parameters[3];
            },
            .rotation_90, .rotation_180, .rotation_270, .custom_rotation => {
                new_phase = state.phase + self.effectiveAngle();
            },
        }

        new_phase = normalizeAngle(new_phase);

        return QuantumState{
            .amplitudes = .{
                Complex(f64).init(new_real, new_imag),
                state.amplitudes[1],
            },
            .phase = new_phase,
            .entanglement_degree = state.entanglement_degree,
        };
    }

    pub fn inverse(self: *const Self) Self {
        const inv_scale = if (self.scale_factor != 0.0 and std.math.isFinite(self.scale_factor)) 1.0 / self.scale_factor else 1.0;
        return switch (self.group) {
            .identity => Self{
                .group = .identity,
                .origin_x = self.origin_x,
                .origin_y = self.origin_y,
                .parameters = [4]f64{ self.origin_x, self.origin_y, inv_scale, self.parameters[3] },
                .scale_factor = inv_scale,
            },
            .reflection => Self{
                .group = .reflection,
                .origin_x = self.origin_x,
                .origin_y = self.origin_y,
                .parameters = [4]f64{ self.origin_x, self.origin_y, inv_scale, self.parameters[3] },
                .scale_factor = inv_scale,
            },
            .rotation_90 => Self{
                .group = .rotation_270,
                .origin_x = self.origin_x,
                .origin_y = self.origin_y,
                .parameters = [4]f64{ self.origin_x, self.origin_y, inv_scale, self.parameters[3] },
                .scale_factor = inv_scale,
            },
            .rotation_180 => Self{
                .group = .rotation_180,
                .origin_x = self.origin_x,
                .origin_y = self.origin_y,
                .parameters = [4]f64{ self.origin_x, self.origin_y, inv_scale, self.parameters[3] },
                .scale_factor = inv_scale,
            },
            .rotation_270 => Self{
                .group = .rotation_90,
                .origin_x = self.origin_x,
                .origin_y = self.origin_y,
                .parameters = [4]f64{ self.origin_x, self.origin_y, inv_scale, self.parameters[3] },
                .scale_factor = inv_scale,
            },
            .custom_rotation => Self{
                .group = .custom_rotation,
                .origin_x = self.origin_x,
                .origin_y = self.origin_y,
                .parameters = [4]f64{ self.origin_x, self.origin_y, inv_scale, -self.parameters[3] },
                .scale_factor = inv_scale,
            },
            .translation => Self{
                .group = .translation,
                .origin_x = self.origin_x,
                .origin_y = self.origin_y,
                .parameters = [4]f64{ -self.parameters[0] * self.scale_factor, -self.parameters[1] * self.scale_factor, 1.0, self.parameters[3] },
                .scale_factor = 1.0,
            },
        };
    }

    fn affine(self: *const Self) Affine {
        const mat = self.getRotationMatrix();
        return switch (self.group) {
            .translation => .{
                .m = mat,
                .tx = self.parameters[0] * self.scale_factor,
                .ty = self.parameters[1] * self.scale_factor,
            },
            else => .{
                .m = mat,
                .tx = self.origin_x - (mat[0][0] * self.origin_x + mat[0][1] * self.origin_y),
                .ty = self.origin_y - (mat[1][0] * self.origin_x + mat[1][1] * self.origin_y),
            },
        };
    }

    fn originForMatrix(m: [2][2]f64, tx: f64, ty: f64, fallback_x: f64, fallback_y: f64) struct { x: f64, y: f64 } {
        const a00 = 1.0 - m[0][0];
        const a01 = -m[0][1];
        const a10 = -m[1][0];
        const a11 = 1.0 - m[1][1];
        const det = a00 * a11 - a01 * a10;
        if (@abs(det) > 1e-12) {
            return .{
                .x = (tx * a11 - a01 * ty) / det,
                .y = (a00 * ty - tx * a10) / det,
            };
        }
        return .{ .x = fallback_x, .y = fallback_y };
    }

    pub fn compose(self: *const Self, other: *const Self) Self {
        const a1 = self.affine();
        const a2 = other.affine();

        const m_out = [2][2]f64{
            [2]f64{ a1.m[0][0] * a2.m[0][0] + a1.m[0][1] * a2.m[1][0], a1.m[0][0] * a2.m[0][1] + a1.m[0][1] * a2.m[1][1] },
            [2]f64{ a1.m[1][0] * a2.m[0][0] + a1.m[1][1] * a2.m[1][0], a1.m[1][0] * a2.m[0][1] + a1.m[1][1] * a2.m[1][1] },
        };

        const tx_out = a1.m[0][0] * a2.tx + a1.m[0][1] * a2.ty + a1.tx;
        const ty_out = a1.m[1][0] * a2.tx + a1.m[1][1] * a2.ty + a1.ty;

        const det = m_out[0][0] * m_out[1][1] - m_out[0][1] * m_out[1][0];
        var scale = std.math.sqrt(@abs(det));
        if (!std.math.isFinite(scale) or scale <= 1e-12) scale = 1.0;

        const is_identity_matrix =
            @abs(m_out[0][0] - 1.0) < 1e-10 and
            @abs(m_out[0][1]) < 1e-10 and
            @abs(m_out[1][0]) < 1e-10 and
            @abs(m_out[1][1] - 1.0) < 1e-10;

        if (is_identity_matrix) {
            if (@abs(tx_out) > 1e-10 or @abs(ty_out) > 1e-10) {
                return Self{
                    .group = .translation,
                    .origin_x = 0.0,
                    .origin_y = 0.0,
                    .parameters = [4]f64{ tx_out, ty_out, 1.0, 0.0 },
                    .scale_factor = 1.0,
                };
            }
            return Self.init(.identity);
        }

        const is_scaled_identity =
            @abs(m_out[0][0] - scale) < 1e-10 and
            @abs(m_out[0][1]) < 1e-10 and
            @abs(m_out[1][0]) < 1e-10 and
            @abs(m_out[1][1] - scale) < 1e-10;

        if (is_scaled_identity) {
            const origin = originForMatrix(m_out, tx_out, ty_out, self.origin_x, self.origin_y);
            return Self{
                .group = .identity,
                .origin_x = origin.x,
                .origin_y = origin.y,
                .parameters = [4]f64{ origin.x, origin.y, scale, 0.0 },
                .scale_factor = scale,
            };
        }

        var group: SymmetryGroup = .identity;
        var angle: f64 = 0.0;

        if (det > 0.0) {
            angle = normalizeAngle(std.math.atan2(m_out[1][0], m_out[0][0]));
            if (angle < 0.01 or angle > 2.0 * std.math.pi - 0.01) {
                group = .identity;
                angle = 0.0;
            } else if (@abs(angle - std.math.pi / 2.0) < 0.01) {
                group = .rotation_90;
                angle = 0.0;
            } else if (@abs(angle - std.math.pi) < 0.01) {
                group = .rotation_180;
                angle = 0.0;
            } else if (@abs(angle - 3.0 * std.math.pi / 2.0) < 0.01) {
                group = .rotation_270;
                angle = 0.0;
            } else {
                group = .custom_rotation;
            }
        } else {
            group = .reflection;
            angle = std.math.atan2(m_out[1][0], m_out[0][0]) / 2.0;
        }

        const origin = originForMatrix(m_out, tx_out, ty_out, self.origin_x, self.origin_y);
        return Self{
            .group = group,
            .origin_x = origin.x,
            .origin_y = origin.y,
            .parameters = [4]f64{ origin.x, origin.y, scale, angle },
            .scale_factor = scale,
        };
    }

    pub fn getRotationMatrix(self: *const Self) [2][2]f64 {
        switch (self.group) {
            .identity => return [2][2]f64{ [2]f64{ self.scale_factor, 0.0 }, [2]f64{ 0.0, self.scale_factor } },
            .reflection => return [2][2]f64{
                [2]f64{ @cos(2.0 * self.parameters[3]) * self.scale_factor, @sin(2.0 * self.parameters[3]) * self.scale_factor },
                [2]f64{ @sin(2.0 * self.parameters[3]) * self.scale_factor, -@cos(2.0 * self.parameters[3]) * self.scale_factor },
            },
            .rotation_90, .rotation_180, .rotation_270 => {
                const angle = self.group.getAngle();
                const cos_a = @cos(angle) * self.scale_factor;
                const sin_a = @sin(angle) * self.scale_factor;
                return [2][2]f64{ [2]f64{ cos_a, -sin_a }, [2]f64{ sin_a, cos_a } };
            },
            .custom_rotation => {
                const cos_a = @cos(self.parameters[3]) * self.scale_factor;
                const sin_a = @sin(self.parameters[3]) * self.scale_factor;
                return [2][2]f64{ [2]f64{ cos_a, -sin_a }, [2]f64{ sin_a, cos_a } };
            },
            .translation => return [2][2]f64{ [2]f64{ 1.0, 0.0 }, [2]f64{ 0.0, 1.0 } },
        }
    }

    pub fn determinant(self: *const Self) f64 {
        const mat = self.getRotationMatrix();
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    }

    pub fn isIsometry(self: *const Self) bool {
        const det = self.determinant();
        const abs_det = if (det < 0.0) -det else det;
        return (if (abs_det - 1.0 < 0.0) -(abs_det - 1.0) else abs_det - 1.0) < 1e-10;
    }
};

pub const NodePairKey = struct {
    node1: []const u8,
    node2: []const u8,
};

pub const NodePairKeyContext = struct {
    pub fn hash(self: @This(), key: NodePairKey) u64 {
        _ = self;
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&key.node1.len));
        hasher.update(key.node1);
        hasher.update(std.mem.asBytes(&key.node2.len));
        hasher.update(key.node2);
        return hasher.final();
    }

    pub fn eql(self: @This(), a: NodePairKey, b: NodePairKey) bool {
        _ = self;
        return std.mem.eql(u8, a.node1, b.node1) and std.mem.eql(u8, a.node2, b.node2);
    }
};

pub const EntanglementInfo = struct {
    correlation_strength: f64,
    phase_difference: f64,
    creation_time: i64,
    last_update_time: i64,
    interaction_count: usize,

    pub fn init(correlation: f64, phase_diff: f64) EntanglementInfo {
        const now: i64 = nowNs();
        return EntanglementInfo{
            .correlation_strength = finiteOr(correlation, 0.0),
            .phase_difference = normalizeAngle(finiteOr(phase_diff, 0.0)),
            .creation_time = now,
            .last_update_time = now,
            .interaction_count = 1,
        };
    }

    pub fn update(self: *EntanglementInfo, new_correlation: f64, new_phase: f64) void {
        const count = @as(f64, @floatFromInt(self.interaction_count));
        const denom = count + 1.0;
        self.correlation_strength = (self.correlation_strength * count + finiteOr(new_correlation, self.correlation_strength)) / denom;

        const normalized_phase = normalizeAngle(finiteOr(new_phase, self.phase_difference));
        const x = (@cos(self.phase_difference) * count + @cos(normalized_phase)) / denom;
        const y = (@sin(self.phase_difference) * count + @sin(normalized_phase)) / denom;
        var new_p = std.math.atan2(y, x);
        if (new_p < 0.0) new_p += 2.0 * std.math.pi;
        self.phase_difference = new_p;

        if (self.interaction_count < std.math.maxInt(usize)) {
            self.interaction_count += 1;
        }
        self.last_update_time = nowNs();
    }

    pub fn getAge(self: *const EntanglementInfo) i64 {
        const now = nowNs();
        if (now <= self.creation_time) return 0;
        return now - self.creation_time;
    }

    pub fn getDecayFactor(self: *const EntanglementInfo, half_life_ms: i64) f64 {
        if (half_life_ms <= 0) return 1.0;
        const now: i64 = nowNs();
        if (now <= self.last_update_time) return 1.0;
        const elapsed_ns = now - self.last_update_time;
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const half_life = @as(f64, @floatFromInt(half_life_ms));
        return @exp(-std.math.ln2 * (elapsed_ms / half_life));
    }
};

pub const OptimizationState = struct {
    graph: *SelfSimilarRelationalGraph,
    energy: f64,
    entanglement_percentage: f64,
    iteration: usize,
    allocator: Allocator,
    owns_graph: bool,
    entanglement_map: std.HashMap(NodePairKey, EntanglementInfo, NodePairKeyContext, std.hash_map.default_max_load_percentage),

    const Self = @This();

    pub fn init(allocator: Allocator, graph: *SelfSimilarRelationalGraph, energy: f64, owns_graph: bool) Self {
        return Self{
            .graph = graph,
            .energy = finiteOr(energy, 0.0),
            .entanglement_percentage = 0.0,
            .iteration = 0,
            .allocator = allocator,
            .owns_graph = owns_graph,
            .entanglement_map = std.HashMap(NodePairKey, EntanglementInfo, NodePairKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.entanglement_map.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.node1);
            self.allocator.free(entry.key_ptr.node2);
        }
        self.entanglement_map.deinit();
        if (self.owns_graph) {
            self.graph.deinit();
            self.allocator.destroy(self.graph);
        }
    }

    pub fn refreshEntanglementPercentage(self: *Self) void {
        const node_count = self.graph.nodeCount();
        if (node_count < 2) {
            self.entanglement_percentage = 0.0;
            return;
        }
        const node_count_f = @as(f64, @floatFromInt(node_count));
        const max_edges_f = node_count_f * (node_count_f - 1.0) / 2.0;
        if (max_edges_f <= 0.0) {
            self.entanglement_percentage = 0.0;
            return;
        }
        const ratio = @as(f64, @floatFromInt(self.entanglement_map.count())) / max_edges_f;
        self.entanglement_percentage = if (ratio > 1.0) 1.0 else ratio;
    }

    pub fn addEntanglement(self: *Self, node1: []const u8, node2: []const u8, info: EntanglementInfo) !void {
        var n1 = node1;
        var n2 = node2;
        if (std.mem.order(u8, n1, n2) == .gt) {
            n1 = node2;
            n2 = node1;
        }

        const pair_key = NodePairKey{ .node1 = n1, .node2 = n2 };
        if (self.entanglement_map.getPtr(pair_key)) |existing| {
            const preserved_creation = existing.creation_time;
            existing.* = info;
            existing.creation_time = preserved_creation;
            self.refreshEntanglementPercentage();
            return;
        }

        const key1 = try self.allocator.dupe(u8, n1);
        errdefer self.allocator.free(key1);
        const key2 = try self.allocator.dupe(u8, n2);
        errdefer self.allocator.free(key2);

        const new_pair_key = NodePairKey{ .node1 = key1, .node2 = key2 };
        try self.entanglement_map.put(new_pair_key, info);
        self.refreshEntanglementPercentage();
    }

    pub fn getEntanglement(self: *const Self, node1: []const u8, node2: []const u8) ?EntanglementInfo {
        var n1 = node1;
        var n2 = node2;
        if (std.mem.order(u8, n1, n2) == .gt) {
            n1 = node2;
            n2 = node1;
        }
        const pair_key = NodePairKey{ .node1 = n1, .node2 = n2 };
        return self.entanglement_map.get(pair_key);
    }

    pub fn hasEntanglement(self: *const Self, node1: []const u8, node2: []const u8) bool {
        return self.getEntanglement(node1, node2) != null;
    }

    pub fn entangledPairsCount(self: *const Self) usize {
        return self.entanglement_map.count();
    }

    pub fn updateEntanglement(self: *Self, node1: []const u8, node2: []const u8, new_correlation: f64, new_phase: f64) void {
        var n1 = node1;
        var n2 = node2;
        if (std.mem.order(u8, n1, n2) == .gt) {
            n1 = node2;
            n2 = node1;
        }
        const pair_key = NodePairKey{ .node1 = n1, .node2 = n2 };
        if (self.entanglement_map.getPtr(pair_key)) |info| {
            info.update(new_correlation, new_phase);
            self.refreshEntanglementPercentage();
        }
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        const new_graph = try allocator.create(SelfSimilarRelationalGraph);
        var graph_initialized = false;
        errdefer {
            if (graph_initialized) new_graph.deinit();
            allocator.destroy(new_graph);
        }
        new_graph.* = try cloneGraph(allocator, self.graph);
        graph_initialized = true;

        var new_state = Self{
            .graph = new_graph,
            .energy = self.energy,
            .entanglement_percentage = self.entanglement_percentage,
            .iteration = self.iteration,
            .allocator = allocator,
            .owns_graph = true,
            .entanglement_map = std.HashMap(NodePairKey, EntanglementInfo, NodePairKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
        };
        var state_owns = true;
        errdefer if (state_owns) new_state.deinit();

        var iter = self.entanglement_map.iterator();
        while (iter.next()) |entry| {
            const key1 = try allocator.dupe(u8, entry.key_ptr.node1);
            var key1_owned = true;
            errdefer if (key1_owned) allocator.free(key1);
            const key2 = try allocator.dupe(u8, entry.key_ptr.node2);
            var key2_owned = true;
            errdefer if (key2_owned) allocator.free(key2);
            const new_key = NodePairKey{ .node1 = key1, .node2 = key2 };
            try new_state.entanglement_map.put(new_key, entry.value_ptr.*);
            key1_owned = false;
            key2_owned = false;
        }

        state_owns = false;
        return new_state;
    }

    pub fn averageEntanglement(self: *const Self) f64 {
        if (self.entanglement_map.count() == 0) return 0.0;
        var total: f64 = 0.0;
        var iter = self.entanglement_map.iterator();
        while (iter.next()) |entry| {
            total += entry.value_ptr.correlation_strength;
        }
        return total / @as(f64, @floatFromInt(self.entanglement_map.count()));
    }
};

pub const OptimizationStatistics = struct {
    iterations_completed: usize,
    moves_accepted: usize,
    moves_rejected: usize,
    best_energy: f64,
    current_energy: f64,
    symmetries_detected: usize,
    entangled_pairs: usize,
    elapsed_time_ms: i64,
    start_time_ns: i64,
    acceptance_rate: f64,
    cooling_factor_applied: usize,
    local_minima_escapes: usize,
    convergence_delta: f64,
    temperature: f64,
    total_energy_evaluations: usize,
    average_move_delta: f64,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .iterations_completed = 0,
            .moves_accepted = 0,
            .moves_rejected = 0,
            .best_energy = std.math.inf(f64),
            .current_energy = std.math.inf(f64),
            .symmetries_detected = 0,
            .entangled_pairs = 0,
            .elapsed_time_ms = 0,
            .start_time_ns = nowNs(),
            .acceptance_rate = 0.0,
            .cooling_factor_applied = 0,
            .local_minima_escapes = 0,
            .convergence_delta = 0.0,
            .temperature = 0.0,
            .total_energy_evaluations = 0,
            .average_move_delta = 0.0,
        };
    }

    pub fn updateAcceptanceRate(self: *Self) void {
        const accepted = @as(f64, @floatFromInt(self.moves_accepted));
        const rejected = @as(f64, @floatFromInt(self.moves_rejected));
        const total_moves = accepted + rejected;
        if (total_moves > 0.0) {
            self.acceptance_rate = accepted / total_moves;
        } else {
            self.acceptance_rate = 0.0;
        }
    }

    pub fn updateElapsedTime(self: *Self) void {
        const now: i64 = nowNs();
        if (now <= self.start_time_ns) {
            self.elapsed_time_ms = 0;
        } else {
            self.elapsed_time_ms = @divTrunc(now - self.start_time_ns, 1_000_000);
        }
    }

    pub fn iterationsPerSecond(self: *const Self) f64 {
        if (self.elapsed_time_ms <= 0) return 0.0;
        return @as(f64, @floatFromInt(self.iterations_completed)) * 1000.0 / @as(f64, @floatFromInt(self.elapsed_time_ms));
    }

    pub fn isConverged(self: *const Self, threshold: f64) bool {
        const safe_threshold = if (std.math.isFinite(threshold) and threshold > 0.0) threshold else 0.0;
        const abs_delta = if (self.convergence_delta < 0.0) -self.convergence_delta else self.convergence_delta;
        return self.moves_accepted > 0 and abs_delta < safe_threshold and self.iterations_completed > 10;
    }
};

pub const SymmetryPattern = struct {
    pattern_id: [16]u8,
    transform: SymmetryTransform,
    nodes: ArrayList([]const u8),
    symmetry_score: f64,
    resonance_frequency: f64,
    creation_timestamp: i64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, transform: SymmetryTransform) Self {
        var hasher = Sha256.init(.{});
        const timestamp: i64 = nowNs();
        hasher.update(std.mem.asBytes(&timestamp));
        hasher.update(std.mem.asBytes(&transform.group));
        hasher.update(std.mem.asBytes(&transform.origin_x));
        hasher.update(std.mem.asBytes(&transform.origin_y));
        hasher.update(std.mem.asBytes(&transform.scale_factor));
        hasher.update(std.mem.asBytes(&transform.parameters));
        const hash_result = hasher.finalResult();
        const id: [16]u8 = hash_result[0..16].*;

        return Self{
            .pattern_id = id,
            .transform = transform,
            .nodes = ArrayList([]const u8).init(allocator),
            .symmetry_score = 0.0,
            .resonance_frequency = 0.0,
            .creation_timestamp = timestamp,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.nodes.items) |node_id| {
            self.allocator.free(node_id);
        }
        self.nodes.deinit();
    }

    pub fn addNode(self: *Self, node_id: []const u8) !void {
        const id_copy = try self.allocator.dupe(u8, node_id);
        errdefer self.allocator.free(id_copy);
        try self.nodes.append(id_copy);
    }

    pub fn getPatternIdHex(self: *const Self) [32]u8 {
        const hex_chars = "0123456789abcdef";
        var result: [32]u8 = undefined;
        for (self.pattern_id, 0..) |byte, i| {
            result[i * 2] = hex_chars[(byte >> 4) & 0x0F];
            result[i * 2 + 1] = hex_chars[byte & 0x0F];
        }
        return result;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var new_pattern = Self{
            .pattern_id = self.pattern_id,
            .transform = self.transform,
            .nodes = ArrayList([]const u8).init(allocator),
            .symmetry_score = self.symmetry_score,
            .resonance_frequency = self.resonance_frequency,
            .creation_timestamp = self.creation_timestamp,
            .allocator = allocator,
        };
        errdefer new_pattern.deinit();
        for (self.nodes.items) |node_id| {
            const id_copy = try allocator.dupe(u8, node_id);
            var id_copy_owned = true;
            errdefer if (id_copy_owned) allocator.free(id_copy);
            try new_pattern.nodes.append(id_copy);
            id_copy_owned = false;
        }
        return new_pattern;
    }
};

const UndoLog = struct {
    move_type: usize,
    edge_weights: ArrayList(struct { source: []const u8, target: []const u8, index: usize, weight: f64, fractal_dimension: f64 }),
    node_states: ArrayList(struct { id: []const u8, phase: f64, qubit_a: Complex(f64), qubit_b: Complex(f64) }),
    added_entanglements: ArrayList(NodePairKey),
    old_graph: ?*SelfSimilarRelationalGraph,
    allocator: Allocator,

    pub fn init(allocator: Allocator) UndoLog {
        return .{
            .move_type = 0,
            .edge_weights = ArrayList(struct { source: []const u8, target: []const u8, index: usize, weight: f64, fractal_dimension: f64 }).init(allocator),
            .node_states = ArrayList(struct { id: []const u8, phase: f64, qubit_a: Complex(f64), qubit_b: Complex(f64) }).init(allocator),
            .added_entanglements = ArrayList(NodePairKey).init(allocator),
            .old_graph = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *UndoLog) void {
        self.edge_weights.deinit();
        self.node_states.deinit();
        self.added_entanglements.deinit();
        if (self.old_graph) |old_g| {
            old_g.deinit();
            self.allocator.destroy(old_g);
        }
    }

    pub fn clear(self: *UndoLog) void {
        self.edge_weights.clearRetainingCapacity();
        self.node_states.clearRetainingCapacity();
        self.added_entanglements.clearRetainingCapacity();
        if (self.old_graph) |old_g| {
            old_g.deinit();
            self.allocator.destroy(old_g);
            self.old_graph = null;
        }
    }
};

pub const EntangledStochasticSymmetryOptimizer = struct {
    initial_temperature: f64,
    temperature: f64,
    cooling_rate: f64,
    max_iterations: usize,
    current_iteration: usize,
    min_temperature: f64,
    current_state: ?OptimizationState,
    best_state: ?OptimizationState,
    detected_patterns: ArrayList(SymmetryPattern),
    symmetry_transforms: ArrayList(SymmetryTransform),
    allocator: Allocator,
    statistics: OptimizationStatistics,
    objective_fn: ?ObjectiveFunction,
    prng: std.Random.DefaultPrng,
    seed: u64,
    energy_history: ArrayList(f64),
    temperature_history: ArrayList(f64),
    reheat_factor: f64,
    entanglement_decay_half_life: i64,
    symmetry_detection_interval: usize,
    convergence_threshold: f64,
    adaptive_cooling: bool,

    const Self = @This();
    const DEFAULT_INITIAL_TEMP: f64 = 100.0;
    const DEFAULT_COOLING_RATE: f64 = 0.95;
    const DEFAULT_MAX_ITERATIONS: usize = 10000;
    const DEFAULT_MIN_TEMP: f64 = 0.001;
    const DEFAULT_REHEAT_FACTOR: f64 = 2.0;
    const DEFAULT_ENTANGLEMENT_HALF_LIFE: i64 = 60000;
    const DEFAULT_SYMMETRY_INTERVAL: usize = 50;
    const DEFAULT_CONVERGENCE_THRESHOLD: f64 = 1e-8;

    pub fn init(allocator: Allocator, initial_temp: f64, cooling_rate: f64, max_iterations: usize) Self {
        const ts = std.time.nanoTimestamp();
        const seed = @as(u64, @truncate(@as(u128, @bitCast(ts))));

        const safe_initial_temp = if (!std.math.isFinite(initial_temp) or initial_temp <= 0.0) DEFAULT_INITIAL_TEMP else initial_temp;
        const safe_cooling_rate = if (!std.math.isFinite(cooling_rate) or cooling_rate <= 0.0 or cooling_rate > 1.0) DEFAULT_COOLING_RATE else cooling_rate;

        return Self{
            .initial_temperature = safe_initial_temp,
            .temperature = safe_initial_temp,
            .cooling_rate = safe_cooling_rate,
            .max_iterations = max_iterations,
            .current_iteration = 0,
            .min_temperature = DEFAULT_MIN_TEMP,
            .current_state = null,
            .best_state = null,
            .detected_patterns = ArrayList(SymmetryPattern).init(allocator),
            .symmetry_transforms = ArrayList(SymmetryTransform).init(allocator),
            .allocator = allocator,
            .statistics = OptimizationStatistics.init(),
            .objective_fn = null,
            .prng = std.Random.DefaultPrng.init(seed),
            .seed = seed,
            .energy_history = ArrayList(f64).init(allocator),
            .temperature_history = ArrayList(f64).init(allocator),
            .reheat_factor = DEFAULT_REHEAT_FACTOR,
            .entanglement_decay_half_life = DEFAULT_ENTANGLEMENT_HALF_LIFE,
            .symmetry_detection_interval = DEFAULT_SYMMETRY_INTERVAL,
            .convergence_threshold = DEFAULT_CONVERGENCE_THRESHOLD,
            .adaptive_cooling = true,
        };
    }

    pub fn initWithSeed(allocator: Allocator, initial_temp: f64, cooling_rate: f64, max_iterations: usize, seed: u64) Self {
        var optimizer = Self.init(allocator, initial_temp, cooling_rate, max_iterations);
        optimizer.prng = std.Random.DefaultPrng.init(seed);
        optimizer.seed = seed;
        return optimizer;
    }

    pub fn initDefault(allocator: Allocator) Self {
        return Self.init(allocator, DEFAULT_INITIAL_TEMP, DEFAULT_COOLING_RATE, DEFAULT_MAX_ITERATIONS);
    }

    pub fn deinit(self: *Self) void {
        if (self.current_state) |*state| {
            state.deinit();
        }
        self.current_state = null;
        if (self.best_state) |*state| {
            state.deinit();
        }
        self.best_state = null;
        for (self.detected_patterns.items) |*pattern| {
            pattern.deinit();
        }
        self.detected_patterns.deinit();
        self.symmetry_transforms.deinit();
        self.energy_history.deinit();
        self.temperature_history.deinit();
    }

    pub fn setObjectiveFunction(self: *Self, obj_fn: ObjectiveFunction) void {
        self.objective_fn = obj_fn;
    }

    pub fn setAdaptiveCooling(self: *Self, enabled: bool) void {
        self.adaptive_cooling = enabled;
    }

    pub fn setMinTemperature(self: *Self, min_temp: f64) void {
        self.min_temperature = if (!std.math.isFinite(min_temp) or min_temp < 1e-12) 1e-12 else min_temp;
        if (self.temperature < self.min_temperature) {
            self.temperature = self.min_temperature;
        }
    }

    pub fn setReheatFactor(self: *Self, factor: f64) void {
        self.reheat_factor = if (!std.math.isFinite(factor) or factor <= 1.0) 1.1 else factor;
    }

    pub fn setSymmetryDetectionInterval(self: *Self, interval: usize) void {
        self.symmetry_detection_interval = if (interval == 0) 1 else interval;
        if (self.current_iteration > 0 and self.current_iteration % self.symmetry_detection_interval == 0) {
            self.statistics.symmetries_detected = self.symmetry_transforms.items.len;
        }
    }

    fn clearOwnedOptimizationState(self: *Self) void {
        if (self.current_state) |*state| {
            state.deinit();
        }
        self.current_state = null;
        if (self.best_state) |*state| {
            state.deinit();
        }
        self.best_state = null;
        for (self.detected_patterns.items) |*pattern| {
            pattern.deinit();
        }
        self.detected_patterns.clearRetainingCapacity();
        self.symmetry_transforms.clearRetainingCapacity();
        self.energy_history.clearRetainingCapacity();
        self.temperature_history.clearRetainingCapacity();
    }

    fn appendDetectedPattern(self: *Self, graph: *const SelfSimilarRelationalGraph, transform: SymmetryTransform) !void {
        var pattern = SymmetryPattern.init(self.allocator, transform);
        errdefer pattern.deinit();
        var pnode_iter = graph.nodes.iterator();
        while (pnode_iter.next()) |entry| {
            try pattern.addNode(entry.key_ptr.*);
        }
        try self.symmetry_transforms.append(transform);
        errdefer _ = self.symmetry_transforms.pop();
        try self.detected_patterns.append(pattern);
    }

    pub fn optimize(self: *Self, graph: *const SelfSimilarRelationalGraph, objective_fn: ?ObjectiveFunction) !*SelfSimilarRelationalGraph {
        if (objective_fn) |obj| {
            self.objective_fn = obj;
        } else if (self.objective_fn == null) {
            self.objective_fn = defaultGraphObjective;
        }

        self.current_iteration = 0;
        self.temperature = self.initial_temperature;

        self.clearOwnedOptimizationState();

        self.statistics = OptimizationStatistics.init();

        var optimize_completed = false;
        errdefer if (!optimize_completed) self.clearOwnedOptimizationState();

        var graph_owned_by_state = false;
        const initial_graph = try self.allocator.create(SelfSimilarRelationalGraph);
        var graph_initialized = false;
        errdefer {
            if (!graph_owned_by_state) {
                if (graph_initialized) initial_graph.deinit();
                self.allocator.destroy(initial_graph);
            }
        }
        initial_graph.* = try cloneGraph(self.allocator, graph);
        graph_initialized = true;

        self.current_state = OptimizationState.init(self.allocator, initial_graph, 0.0, true);
        graph_owned_by_state = true;

        self.current_state.?.energy = self.computeEnergy(&self.current_state.?);
        self.best_state = try self.current_state.?.clone(self.allocator);

        self.statistics.best_energy = self.current_state.?.energy;
        self.statistics.current_energy = self.current_state.?.energy;
        self.statistics.total_energy_evaluations = 1;

        const initial_transforms = try self.detectSymmetries(graph);
        defer self.allocator.free(initial_transforms);
        for (initial_transforms) |transform| {
            try self.appendDetectedPattern(graph, transform);
        }
        self.statistics.symmetries_detected = self.symmetry_transforms.items.len;

        if (self.max_iterations == 0) {
            self.statistics.temperature = self.temperature;
            self.statistics.entangled_pairs = self.current_state.?.entangledPairsCount();
            self.statistics.updateElapsedTime();
            const ret_graph = try self.allocator.create(SelfSimilarRelationalGraph);
            errdefer self.allocator.destroy(ret_graph);
            ret_graph.* = try cloneGraph(self.allocator, self.best_state.?.graph);
            optimize_completed = true;
            return ret_graph;
        }

        var stagnation_counter: usize = 0;
        const stagnation_limit = @max(@as(usize, 1), self.max_iterations / 10);
        var previous_energy = self.current_state.?.energy;

        var log = UndoLog.init(self.allocator);
        defer log.deinit();

        while (self.current_iteration < self.max_iterations) : (self.current_iteration += 1) {
            try self.updateEntanglementMap(&self.current_state.?);

            if (self.current_iteration % self.symmetry_detection_interval == 0 and self.current_iteration > 0) {
                const new_transforms = try self.detectSymmetries(self.current_state.?.graph);
                defer self.allocator.free(new_transforms);
                for (new_transforms) |transform| {
                    var is_duplicate = false;
                    for (self.symmetry_transforms.items) |existing| {
                        if (existing.group == transform.group and
                            @abs(existing.origin_x - transform.origin_x) < 1e-6 and
                            @abs(existing.origin_y - transform.origin_y) < 1e-6 and
                            @abs(existing.scale_factor - transform.scale_factor) < 1e-6 and
                            @abs(existing.parameters[3] - transform.parameters[3]) < 1e-6)
                        {
                            is_duplicate = true;
                            break;
                        }
                    }
                    if (!is_duplicate) {
                        try self.appendDetectedPattern(self.current_state.?.graph, transform);
                    }
                }
                self.statistics.symmetries_detected = self.symmetry_transforms.items.len;
            }

            try self.applyMove(&log);
            self.current_state.?.energy = self.computeEnergy(&self.current_state.?);
            self.statistics.total_energy_evaluations += 1;

            const attempted_energy = self.current_state.?.energy;
            const delta_energy = attempted_energy - previous_energy;
            const attempted_energy_change = @abs(delta_energy);

            if (self.acceptMove(delta_energy, &log)) {
                self.statistics.moves_accepted += 1;

                if (self.current_state.?.energy < self.best_state.?.energy) {
                    const new_best = try self.current_state.?.clone(self.allocator);
                    if (self.best_state) |*old_best| old_best.deinit();
                    self.best_state = new_best;
                    self.statistics.best_energy = self.current_state.?.energy;
                    stagnation_counter = 0;
                } else {
                    stagnation_counter += 1;
                }
            } else {
                self.statistics.moves_rejected += 1;
                stagnation_counter += 1;
                self.current_state.?.energy = previous_energy;
            }

            self.statistics.current_energy = self.current_state.?.energy;
            self.statistics.entangled_pairs = self.current_state.?.entangledPairsCount();
            self.statistics.convergence_delta = attempted_energy_change;
            previous_energy = self.current_state.?.energy;

            self.statistics.updateAcceptanceRate();

            if (self.adaptive_cooling) {
                self.adaptiveCoolTemperature();
            } else {
                self.coolTemperature();
            }

            if (stagnation_counter > stagnation_limit) {
                self.temperature *= self.reheat_factor;
                if (!std.math.isFinite(self.temperature)) {
                    self.temperature = self.initial_temperature;
                }
                self.statistics.local_minima_escapes += 1;
                stagnation_counter = 0;
            }

            self.statistics.iterations_completed = self.current_iteration + 1;
            self.statistics.temperature = self.temperature;
            self.statistics.updateElapsedTime();

            try self.energy_history.append(self.current_state.?.energy);
            try self.temperature_history.append(self.temperature);

            if (self.statistics.moves_accepted > 0 and self.statistics.isConverged(self.convergence_threshold)) {
                break;
            }
        }

        if (self.energy_history.items.len > 1) {
            var total_delta: f64 = 0.0;
            var delta_idx: usize = 1;
            while (delta_idx < self.energy_history.items.len) : (delta_idx += 1) {
                total_delta += @abs(self.energy_history.items[delta_idx] - self.energy_history.items[delta_idx - 1]);
            }
            self.statistics.average_move_delta = total_delta / @as(f64, @floatFromInt(self.energy_history.items.len - 1));
        }

        const ret_graph = try self.allocator.create(SelfSimilarRelationalGraph);
        errdefer self.allocator.destroy(ret_graph);
        ret_graph.* = try cloneGraph(self.allocator, self.best_state.?.graph);
        optimize_completed = true;
        return ret_graph;
    }

    fn computeEnergy(self: *Self, state: *const OptimizationState) f64 {
        const energy = if (self.objective_fn) |obj_fn| obj_fn(state) else defaultGraphObjective(state);
        return if (std.math.isFinite(energy)) energy else std.math.inf(f64);
    }

    fn normalizeQubit(node: *Node) void {
        const re_a = node.qubit.a.re;
        const im_a = node.qubit.a.im;
        const re_b = node.qubit.b.re;
        const im_b = node.qubit.b.im;
        const mag = std.math.sqrt(re_a * re_a + im_a * im_a + re_b * re_b + im_b * im_b);
        if (mag > 1e-12) {
            node.qubit.a = Complex(f64).init(re_a / mag, im_a / mag);
            node.qubit.b = Complex(f64).init(re_b / mag, im_b / mag);
        } else {
            node.qubit.a = Complex(f64).init(1.0, 0.0);
            node.qubit.b = Complex(f64).init(0.0, 0.0);
        }
    }

    fn applyMove(self: *Self, log: *UndoLog) !void {
        const state = &self.current_state.?;
        const graph = state.graph;
        log.clear();
        errdefer self.undoMove(log);

        const move_type = self.prng.random().uintLessThan(usize, 7);
        log.move_type = move_type;

        switch (move_type) {
            0 => {
                var edge_iter = graph.edges.iterator();
                while (edge_iter.next()) |entry| {
                    for (entry.value_ptr.items, 0..) |*edge, edge_index| {
                        try log.edge_weights.append(.{ .source = edge.source, .target = edge.target, .index = edge_index, .weight = edge.weight, .fractal_dimension = edge.fractal_dimension });
                        const perturbation = (self.prng.random().float(f64) - 0.5) * self.temperature * 0.1;
                        edge.weight = @max(0.0, @min(1.0, edge.weight + perturbation));
                    }
                }
            },
            1 => {
                var node_iter = graph.nodes.iterator();
                while (node_iter.next()) |entry| {
                    const node = entry.value_ptr;
                    try log.node_states.append(.{ .id = entry.key_ptr.*, .phase = node.phase, .qubit_a = node.qubit.a, .qubit_b = node.qubit.b });
                    const phase_delta = (self.prng.random().float(f64) - 0.5) * self.temperature * 0.2;
                    node.phase = normalizeAngle(node.phase + phase_delta);
                }
            },
            2 => {
                try self.createNewEntanglementInPlace(state, log);
            },
            3 => {
                if (self.symmetry_transforms.items.len > 0) {
                    var valid_transforms = ArrayList(SymmetryTransform).init(self.allocator);
                    defer valid_transforms.deinit();
                    for (self.symmetry_transforms.items) |t| {
                        if (t.group != .identity) try valid_transforms.append(t);
                    }
                    if (valid_transforms.items.len > 0) {
                        const transform_idx = self.prng.random().uintLessThan(usize, valid_transforms.items.len);
                        const transform = valid_transforms.items[transform_idx];

                        var node_iter = graph.nodes.iterator();
                        while (node_iter.next()) |entry| {
                            const node = entry.value_ptr;
                            try log.node_states.append(.{ .id = entry.key_ptr.*, .phase = node.phase, .qubit_a = node.qubit.a, .qubit_b = node.qubit.b });

                            const transformed_a = transform.applyToQuantumState(&QuantumState{
                                .amplitude_real = node.qubit.a.re,
                                .amplitude_imag = node.qubit.a.im,
                                .phase = node.phase,
                                .entanglement_degree = 0.0,
                            });
                            const transformed_b = transform.applyToQuantumState(&QuantumState{
                                .amplitude_real = node.qubit.b.re,
                                .amplitude_imag = node.qubit.b.im,
                                .phase = node.phase,
                                .entanglement_degree = 0.0,
                            });

                            node.qubit.a = Complex(f64).init(transformed_a.amplitude_real, transformed_a.amplitude_imag);
                            node.qubit.b = Complex(f64).init(transformed_b.amplitude_real, transformed_b.amplitude_imag);

                            const sx = @sin(transformed_a.phase) + @sin(transformed_b.phase);
                            const cx = @cos(transformed_a.phase) + @cos(transformed_b.phase);
                            node.phase = normalizeAngle(std.math.atan2(sx, cx));

                            normalizeQubit(node);
                        }
                    }
                }
            },
            4 => {
                var node_iter = graph.nodes.iterator();
                while (node_iter.next()) |entry| {
                    const node = entry.value_ptr;
                    try log.node_states.append(.{ .id = entry.key_ptr.*, .phase = node.phase, .qubit_a = node.qubit.a, .qubit_b = node.qubit.b });

                    const perturbation = self.temperature * 0.05;

                    const angle_a = self.prng.random().float(f64) * 2.0 * std.math.pi;
                    var new_re_a = node.qubit.a.re + perturbation * @cos(angle_a);
                    const new_im_a = node.qubit.a.im + perturbation * @sin(angle_a);
                    if (new_re_a == 0.0 and new_im_a == 0.0) new_re_a = 1e-6;

                    const angle_b = self.prng.random().float(f64) * 2.0 * std.math.pi;
                    var new_re_b = node.qubit.b.re + perturbation * @cos(angle_b);
                    const new_im_b = node.qubit.b.im + perturbation * @sin(angle_b);
                    if (new_re_b == 0.0 and new_im_b == 0.0) new_re_b = 1e-6;

                    const mag = std.math.sqrt(new_re_a * new_re_a + new_im_a * new_im_a + new_re_b * new_re_b + new_im_b * new_im_b);
                    if (mag > 1e-12) {
                        node.qubit.a = Complex(f64).init(new_re_a / mag, new_im_a / mag);
                        node.qubit.b = Complex(f64).init(new_re_b / mag, new_im_b / mag);
                    } else {
                        node.qubit.a = Complex(f64).init(1.0, 0.0);
                        node.qubit.b = Complex(f64).init(0.0, 0.0);
                    }
                }
            },
            5 => {
                var edge_iter = graph.edges.iterator();
                while (edge_iter.next()) |entry| {
                    for (entry.value_ptr.items, 0..) |*edge, edge_index| {
                        try log.edge_weights.append(.{ .source = edge.source, .target = edge.target, .index = edge_index, .weight = edge.weight, .fractal_dimension = edge.fractal_dimension });
                        const delta = (self.prng.random().float(f64) - 0.5) * self.temperature * 0.02;
                        edge.fractal_dimension = @max(0.0, @min(3.0, edge.fractal_dimension + delta));
                    }
                }
            },
            6 => {
                try self.toggleRandomEdge(state, log);
            },
            else => unreachable,
        }
    }

    fn createNewEntanglementInPlace(self: *Self, state: *OptimizationState, log: *UndoLog) !void {
        const graph = state.graph;
        const node_count = graph.nodeCount();
        if (node_count < 2) return;

        var node_ids = ArrayList([]const u8).init(self.allocator);
        defer node_ids.deinit();
        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try node_ids.append(entry.key_ptr.*);
        }
        if (node_ids.items.len < 2) return;

        const idx1 = self.prng.random().uintLessThan(usize, node_ids.items.len);
        var idx2 = self.prng.random().uintLessThan(usize, node_ids.items.len);
        var attempts: usize = 0;
        while (idx2 == idx1 and attempts < 16) : (attempts += 1) {
            idx2 = self.prng.random().uintLessThan(usize, node_ids.items.len);
        }
        if (idx2 == idx1) {
            idx2 = (idx1 + 1) % node_ids.items.len;
        }

        const n1_id = node_ids.items[idx1];
        const n2_id = node_ids.items[idx2];
        if (std.mem.eql(u8, n1_id, n2_id)) return;
        if (state.hasEntanglement(n1_id, n2_id)) return;

        const node1 = graph.getNode(n1_id) orelse return;
        const node2 = graph.getNode(n2_id) orelse return;

        try log.node_states.append(.{ .id = n1_id, .phase = node1.phase, .qubit_a = node1.qubit.a, .qubit_b = node1.qubit.b });
        try log.node_states.append(.{ .id = n2_id, .phase = node2.phase, .qubit_a = node2.qubit.a, .qubit_b = node2.qubit.b });

        var phase_diff = @abs(node1.phase - node2.phase);
        if (phase_diff > std.math.pi) phase_diff = 2.0 * std.math.pi - phase_diff;

        const correlation = self.prng.random().float(f64) * 0.5 + 0.5;
        const info = EntanglementInfo.init(correlation, phase_diff);

        var key1 = n1_id;
        var key2 = n2_id;
        if (std.mem.order(u8, key1, key2) == .gt) {
            key1 = n2_id;
            key2 = n1_id;
        }
        try log.added_entanglements.append(NodePairKey{ .node1 = key1, .node2 = key2 });
        try state.addEntanglement(n1_id, n2_id, info);
    }

    fn toggleRandomEdge(self: *Self, state: *OptimizationState, log: *UndoLog) !void {
        const old_graph = state.graph;
        const node_count = old_graph.nodeCount();
        if (node_count < 2) return;

        const new_graph = try self.allocator.create(SelfSimilarRelationalGraph);
        var graph_assigned = false;
        errdefer if (!graph_assigned) self.allocator.destroy(new_graph);
        new_graph.* = try cloneGraph(self.allocator, old_graph);
        errdefer if (!graph_assigned) new_graph.deinit();

        var node_ids = ArrayList([]const u8).init(self.allocator);
        defer node_ids.deinit();
        var node_iter = new_graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try node_ids.append(entry.key_ptr.*);
        }
        if (node_ids.items.len < 2) {
            new_graph.deinit();
            self.allocator.destroy(new_graph);
            graph_assigned = true;
            return;
        }

        const idx1 = self.prng.random().uintLessThan(usize, node_ids.items.len);
        var idx2 = self.prng.random().uintLessThan(usize, node_ids.items.len);
        var attempts: usize = 0;
        while (idx2 == idx1 and attempts < 16) : (attempts += 1) {
            idx2 = self.prng.random().uintLessThan(usize, node_ids.items.len);
        }
        if (idx2 == idx1) idx2 = (idx1 + 1) % node_ids.items.len;

        const n1 = node_ids.items[idx1];
        const n2 = node_ids.items[idx2];

        var exists = false;
        var existing_source = n1;
        var existing_target = n2;
        var edge_iter = new_graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            if ((std.mem.eql(u8, entry.key_ptr.source, n1) and std.mem.eql(u8, entry.key_ptr.target, n2)) or
                (std.mem.eql(u8, entry.key_ptr.source, n2) and std.mem.eql(u8, entry.key_ptr.target, n1)))
            {
                exists = true;
                existing_source = entry.key_ptr.source;
                existing_target = entry.key_ptr.target;
                break;
            }
        }

        if (exists) {
            try new_graph.removeEdge(existing_source, existing_target);
        } else {
            const weight = self.prng.random().float(f64);
            var edge = Edge.init(self.allocator, n1, n2, .coherent, weight, Complex(f64).init(0.0, 0.0), 1.5);
            var edge_added = false;
            errdefer if (!edge_added) edge.deinit();
            try new_graph.addEdge(n1, n2, edge);
            edge_added = true;
        }

        log.old_graph = old_graph;
        state.graph = new_graph;
        graph_assigned = true;
    }

    fn undoMove(self: *Self, log: *UndoLog) void {
        if (self.current_state == null) return;

        const state = &self.current_state.?;
        const graph = state.graph;

        if (log.move_type == 6) {
            if (log.old_graph) |old_g| {
                state.graph.deinit();
                self.allocator.destroy(state.graph);
                state.graph = old_g;
                log.old_graph = null;
            }
            return;
        }

        for (log.edge_weights.items) |saved| {
            var edge_iter = graph.edges.iterator();
            while (edge_iter.next()) |entry| {
                const direct_match = std.mem.eql(u8, entry.key_ptr.source, saved.source) and std.mem.eql(u8, entry.key_ptr.target, saved.target);
                const reverse_match = std.mem.eql(u8, entry.key_ptr.source, saved.target) and std.mem.eql(u8, entry.key_ptr.target, saved.source);
                if (direct_match or reverse_match) {
                    if (saved.index < entry.value_ptr.items.len) {
                        entry.value_ptr.items[saved.index].weight = saved.weight;
                        entry.value_ptr.items[saved.index].fractal_dimension = saved.fractal_dimension;
                    }
                    break;
                }
            }
        }

        for (log.node_states.items) |saved| {
            if (graph.getNode(saved.id)) |node| {
                node.phase = saved.phase;
                node.qubit.a = saved.qubit_a;
                node.qubit.b = saved.qubit_b;
            }
        }

        for (log.added_entanglements.items) |key| {
            if (state.entanglement_map.fetchRemove(key)) |kv| {
                self.allocator.free(kv.key.node1);
                self.allocator.free(kv.key.node2);
            }
        }
        state.refreshEntanglementPercentage();
    }

    fn acceptMove(self: *Self, delta_energy: f64, log: *UndoLog) bool {
        var accepted = false;
        if (delta_energy < 0.0) {
            accepted = true;
        } else if (self.temperature > 1e-12 and std.math.isFinite(delta_energy)) {
            const acceptance_probability = @exp(-delta_energy / self.temperature);
            const random_value = self.prng.random().float(f64);
            if (random_value < acceptance_probability) {
                accepted = true;
            }
        }

        if (accepted) {
            if (log.move_type == 6) {
                if (log.old_graph) |old_g| {
                    old_g.deinit();
                    self.allocator.destroy(old_g);
                    log.old_graph = null;
                }
            }
            return true;
        } else {
            self.undoMove(log);
            return false;
        }
    }

    fn coolTemperature(self: *Self) void {
        if (self.temperature > self.min_temperature) {
            self.temperature *= self.cooling_rate;
            if (self.temperature < self.min_temperature) {
                self.temperature = self.min_temperature;
            }
            self.statistics.cooling_factor_applied += 1;
        }
    }

    fn adaptiveCoolTemperature(self: *Self) void {
        const acceptance_rate = self.statistics.acceptance_rate;

        var rate = self.cooling_rate;
        if (acceptance_rate > 0.6) {
            rate = self.cooling_rate * 0.98;
        } else if (acceptance_rate < 0.2) {
            rate = self.cooling_rate * 1.02;
        }
        if (rate > 0.999) rate = 0.999;
        if (rate >= 1.0) rate = 0.999;
        if (rate < 0.001) rate = 0.001;

        if (self.temperature > self.min_temperature) {
            self.temperature *= rate;
            if (self.temperature < self.min_temperature) {
                self.temperature = self.min_temperature;
            }
            self.statistics.cooling_factor_applied += 1;
        }
    }

    pub fn detectSymmetries(self: *Self, graph: *const SelfSimilarRelationalGraph) ![]SymmetryTransform {
        const SymmetryPosition = struct {
            x: f64,
            y: f64,
            phase: f64,
            key: u64,
        };

        var transforms = ArrayList(SymmetryTransform).init(self.allocator);
        errdefer transforms.deinit();

        var positions = ArrayList(SymmetryPosition).init(self.allocator);
        defer positions.deinit();

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const raw_x = entry.value_ptr.qubit.a.re;
            const raw_y = entry.value_ptr.qubit.a.im;
            const raw_phase = entry.value_ptr.phase;
            try positions.append(.{
                .x = if (std.math.isFinite(raw_x)) raw_x else 0.0,
                .y = if (std.math.isFinite(raw_y)) raw_y else 0.0,
                .phase = if (std.math.isFinite(raw_phase)) raw_phase else 0.0,
                .key = std.hash.Wyhash.hash(0x4A4149444553594D, entry.key_ptr.*),
            });
        }

        std.mem.sort(SymmetryPosition, positions.items, {}, struct {
            fn lessThan(_: void, left: SymmetryPosition, right: SymmetryPosition) bool {
                if (left.key != right.key) return left.key < right.key;
                const left_x: u64 = @bitCast(left.x);
                const right_x: u64 = @bitCast(right.x);
                if (left_x != right_x) return left_x < right_x;
                const left_y: u64 = @bitCast(left.y);
                const right_y: u64 = @bitCast(right.y);
                return left_y < right_y;
            }
        }.lessThan);

        const maximum_sample_count: usize = 100;
        const sampled_count = @min(positions.items.len, maximum_sample_count);
        const sampled_positions = positions.items[0..sampled_count];

        var cx: f64 = 0.0;
        var cy: f64 = 0.0;
        for (sampled_positions) |position| {
            cx += position.x;
            cy += position.y;
        }
        if (sampled_count > 0) {
            const denominator = @as(f64, @floatFromInt(sampled_count));
            cx /= denominator;
            cy /= denominator;
        }

        try transforms.append(SymmetryTransform.initWithParams(.identity, [4]f64{ cx, cy, 1.0, 0.0 }));

        if (sampled_count < 2) {
            return transforms.toOwnedSlice();
        }

        var centroid_moment_xx: f64 = 0.0;
        var centroid_moment_xy: f64 = 0.0;
        var centroid_moment_yy: f64 = 0.0;
        for (sampled_positions) |position| {
            const dx = position.x - cx;
            const dy = position.y - cy;
            centroid_moment_xx += dx * dx;
            centroid_moment_xy += dx * dy;
            centroid_moment_yy += dy * dy;
        }

        const trace = centroid_moment_xx + centroid_moment_yy;
        const determinant = centroid_moment_xx * centroid_moment_yy - centroid_moment_xy * centroid_moment_xy;
        const discriminant = trace * trace - 4.0 * determinant;
        const principal_angle = if (discriminant >= 0.0 and @abs(centroid_moment_xy) > 1e-12)
            0.5 * std.math.atan2(2.0 * centroid_moment_xy, centroid_moment_xx - centroid_moment_yy)
        else
            0.0;
        const eccentricity = if (trace > 1e-12)
            @sqrt(@abs(discriminant)) / trace
        else
            0.0;
        const reflection_angle = normalizeAngle(principal_angle);

        var reflected_positions = ArrayList(struct { x: f64, y: f64 }).init(self.allocator);
        defer reflected_positions.deinit();
        try reflected_positions.ensureTotalCapacity(sampled_count);
        for (sampled_positions) |position| {
            const dx = position.x - cx;
            const dy = position.y - cy;
            const rx = dx * @cos(2.0 * reflection_angle) + dy * @sin(2.0 * reflection_angle);
            const ry = dx * @sin(2.0 * reflection_angle) - dy * @cos(2.0 * reflection_angle);
            reflected_positions.appendAssumeCapacity(.{ .x = cx + rx, .y = cy + ry });
        }

        var symmetry_score_reflection: f64 = 0.0;
        for (reflected_positions.items) |reflected_position| {
            var best_distance: f64 = std.math.inf(f64);
            for (sampled_positions) |position| {
                const dx = position.x - reflected_position.x;
                const dy = position.y - reflected_position.y;
                const distance = dx * dx + dy * dy;
                if (distance < best_distance) best_distance = distance;
            }
            if (best_distance < 0.01) {
                symmetry_score_reflection += 1.0;
            }
        }
        symmetry_score_reflection /= @as(f64, @floatFromInt(sampled_count));

        if (symmetry_score_reflection > 0.3) {
            try transforms.append(SymmetryTransform.initWithParams(.reflection, [4]f64{ cx, cy, 1.0, reflection_angle }));
        }

        var best_rotation_order: usize = 0;
        var best_rotation_score: f64 = 0.0;
        const candidate_orders = [_]usize{ 2, 3, 4, 6 };
        for (candidate_orders) |order| {
            const test_angle = 2.0 * std.math.pi / @as(f64, @floatFromInt(order));
            const cosine = @cos(test_angle);
            const sine = @sin(test_angle);
            var rotation_score: f64 = 0.0;

            for (sampled_positions) |position| {
                const dx = position.x - cx;
                const dy = position.y - cy;
                const rotated_x = cx + dx * cosine - dy * sine;
                const rotated_y = cy + dx * sine + dy * cosine;
                var best_distance: f64 = std.math.inf(f64);

                for (sampled_positions) |candidate| {
                    const candidate_dx = candidate.x - rotated_x;
                    const candidate_dy = candidate.y - rotated_y;
                    const distance = candidate_dx * candidate_dx + candidate_dy * candidate_dy;
                    if (distance < best_distance) best_distance = distance;
                }

                if (best_distance < 0.01) {
                    rotation_score += 1.0;
                }
            }

            rotation_score /= @as(f64, @floatFromInt(sampled_count));
            if (rotation_score > best_rotation_score) {
                best_rotation_score = rotation_score;
                best_rotation_order = order;
            }
        }

        if (best_rotation_score > 0.3) {
            switch (best_rotation_order) {
                4 => try transforms.append(SymmetryTransform.initWithParams(.rotation_90, [4]f64{ cx, cy, 1.0, 0.0 })),
                2 => try transforms.append(SymmetryTransform.initWithParams(.rotation_180, [4]f64{ cx, cy, 1.0, 0.0 })),
                3, 6 => try transforms.append(SymmetryTransform.initWithParams(.custom_rotation, [4]f64{
                    cx,
                    cy,
                    1.0,
                    2.0 * std.math.pi / @as(f64, @floatFromInt(best_rotation_order)),
                })),
                else => try transforms.append(SymmetryTransform.initWithParams(.rotation_180, [4]f64{ cx, cy, 1.0, 0.0 })),
            }
        }

        var phase_coherence_sum_sin: f64 = 0.0;
        var phase_coherence_sum_cos: f64 = 0.0;
        for (sampled_positions) |position| {
            phase_coherence_sum_sin += @sin(position.phase);
            phase_coherence_sum_cos += @cos(position.phase);
        }

        const phase_coherence = std.math.sqrt(
            phase_coherence_sum_sin * phase_coherence_sum_sin +
                phase_coherence_sum_cos * phase_coherence_sum_cos,
        ) / @as(f64, @floatFromInt(sampled_count));

        if (phase_coherence > 0.5) {
            var average_phase = std.math.atan2(phase_coherence_sum_sin, phase_coherence_sum_cos);
            if (average_phase < 0.0) average_phase += 2.0 * std.math.pi;
            try transforms.append(SymmetryTransform.initWithParams(.custom_rotation, [4]f64{ cx, cy, 1.0, average_phase }));
        }

        if (eccentricity > 0.1) {
            try transforms.append(SymmetryTransform.initWithParams(.translation, [4]f64{ cx * 0.01, cy * 0.01, 1.0, 0.0 }));
        }

        return transforms.toOwnedSlice();
    }

    fn updateEntanglementMap(self: *Self, state: *OptimizationState) !void {
        var removals = ArrayList(NodePairKey).init(self.allocator);
        defer removals.deinit();

        var iter = state.entanglement_map.iterator();
        while (iter.next()) |entry| {
            const info = entry.value_ptr;
            const decay_factor = info.getDecayFactor(self.entanglement_decay_half_life);
            info.correlation_strength *= decay_factor;

            info.phase_difference = normalizeAngle(info.phase_difference + self.temperature * 0.01);
            info.last_update_time = nowNs();

            if (info.correlation_strength < 0.01) {
                try removals.append(entry.key_ptr.*);
            }
        }

        for (removals.items) |key| {
            if (state.entanglement_map.fetchRemove(key)) |kv| {
                self.allocator.free(kv.key.node1);
                self.allocator.free(kv.key.node2);
            }
        }

        var node_iter = state.graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node_id = entry.key_ptr.*;
            var entanglement_sum: f64 = 0.0;
            var entanglement_count: usize = 0;

            var ent_iter = state.entanglement_map.iterator();
            while (ent_iter.next()) |ent_entry| {
                if (std.mem.eql(u8, ent_entry.key_ptr.node1, node_id) or
                    std.mem.eql(u8, ent_entry.key_ptr.node2, node_id))
                {
                    entanglement_sum += ent_entry.value_ptr.correlation_strength;
                    entanglement_count += 1;
                }
            }

            if (entanglement_count > 0) {
                const avg_entanglement = entanglement_sum / @as(f64, @floatFromInt(entanglement_count));
                const phase_adjustment = avg_entanglement * 0.1;
                entry.value_ptr.phase = normalizeAngle(entry.value_ptr.phase + phase_adjustment);
            }
        }
        state.refreshEntanglementPercentage();
    }

    pub fn runQuickOptimization(self: *Self, graph: *SelfSimilarRelationalGraph, iterations: usize) !f64 {
        if (graph.nodeCount() == 0) return 0.0;

        const saved_max_iterations = self.max_iterations;
        const saved_temperature = self.temperature;
        const saved_iteration = self.current_iteration;
        defer {
            self.max_iterations = saved_max_iterations;
            self.temperature = saved_temperature;
            self.current_iteration = saved_iteration;
        }

        self.max_iterations = iterations;
        self.temperature = self.initial_temperature * 0.1;
        self.current_iteration = 0;

        const optimized_graph = self.optimize(graph, null) catch {
            self.max_iterations = saved_max_iterations;
            self.temperature = saved_temperature;
            self.current_iteration = saved_iteration;
            return std.math.inf(f64);
        };
        defer {
            optimized_graph.deinit();
            self.allocator.destroy(optimized_graph);
        }

        var temp_state = OptimizationState.init(self.allocator, optimized_graph, 0.0, false);
        defer temp_state.deinit();
        temp_state.energy = self.computeEnergy(&temp_state);
        return temp_state.energy;
    }

    pub fn modulateInferenceTensor(self: *const Self, data: []f32) void {
        if (data.len == 0) return;
        var avg_correlation: f64 = 0.0;
        var count: usize = 0;
        if (self.best_state) |state| {
            var iter = state.entanglement_map.iterator();
            while (iter.next()) |entry| {
                avg_correlation += entry.value_ptr.correlation_strength;
                count += 1;
            }
            if (count > 0) {
                avg_correlation /= @as(f64, @floatFromInt(count));
            }
        }
        const symmetry_boost = @as(f64, @floatFromInt(self.statistics.symmetries_detected)) * 0.01;
        const modulation = 1.0 + avg_correlation * 0.1 + symmetry_boost;
        const scale: f32 = @floatCast(@min(modulation, 2.0));
        var i: usize = 0;
        while (i < data.len) : (i += 1) {
            data[i] *= scale;
        }
    }

    pub fn getStatistics(self: *const Self) OptimizationStatistics {
        return self.statistics;
    }

    pub fn getBestGraph(self: *const Self) ?*const SelfSimilarRelationalGraph {
        if (self.best_state) |state| {
            return state.graph;
        }
        return null;
    }

    pub fn getCurrentTemperature(self: *const Self) f64 {
        return self.temperature;
    }

    pub fn getCurrentIteration(self: *const Self) usize {
        return self.current_iteration;
    }

    pub fn getEnergyHistory(self: *const Self) []const f64 {
        return self.energy_history.items;
    }

    pub fn getTemperatureHistory(self: *const Self) []const f64 {
        return self.temperature_history.items;
    }

    pub fn getDetectedPatterns(self: *const Self) []const SymmetryPattern {
        return self.detected_patterns.items;
    }

    pub fn reset(self: *Self) void {
        self.current_iteration = 0;
        self.temperature = self.initial_temperature;

        if (self.current_state) |*state| {
            state.deinit();
        }
        self.current_state = null;

        if (self.best_state) |*state| {
            state.deinit();
        }
        self.best_state = null;

        self.energy_history.clearRetainingCapacity();
        for (self.detected_patterns.items) |*pattern| {
            pattern.deinit();
        }
        self.detected_patterns.clearRetainingCapacity();
        self.symmetry_transforms.clearRetainingCapacity();
        self.temperature_history.clearRetainingCapacity();
        self.statistics = OptimizationStatistics.init();
        self.prng = std.Random.DefaultPrng.init(self.seed);
    }
};

pub fn defaultGraphObjective(state: *const OptimizationState) f64 {
    const graph = state.graph;
    var total_energy: f64 = 0.0;

    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            total_energy += edge.weight * edge.fractal_dimension;
            total_energy += edge.quantum_correlation.abs();
        }
    }

    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        const node = entry.value_ptr;
        total_energy += (1.0 - @cos(node.phase)) / 2.0;
        total_energy += std.math.sqrt(node.qubit.a.re * node.qubit.a.re + node.qubit.a.im * node.qubit.a.im);
        total_energy += std.math.sqrt(node.qubit.b.re * node.qubit.b.re + node.qubit.b.im * node.qubit.b.im);
    }

    total_energy += state.averageEntanglement();

    return total_energy;
}

pub fn connectivityObjective(state: *const OptimizationState) f64 {
    const graph = state.graph;
    const node_count = graph.nodeCount();
    const edge_count = graph.edgeCount();

    if (node_count == 0) return 0.0;

    const node_count_f = @as(f64, @floatFromInt(node_count));
    const max_edges_f = if (node_count >= 2) node_count_f * (node_count_f - 1.0) / 2.0 else 0.0;
    const connectivity_ratio = if (max_edges_f > 0.0)
        @as(f64, @floatFromInt(edge_count)) / max_edges_f
    else
        0.0;
    const clamped_ratio = if (connectivity_ratio > 1.0) 1.0 else connectivity_ratio;

    var total_weight: f64 = 0.0;
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            total_weight += edge.weight;
        }
    }

    const avg_weight = if (edge_count > 0) total_weight / @as(f64, @floatFromInt(edge_count)) else 0.0;

    return (1.0 - clamped_ratio) + (1.0 - avg_weight);
}

pub fn quantumCoherenceObjective(state: *const OptimizationState) f64 {
    const graph = state.graph;
    var total_coherence: f64 = 0.0;
    var node_count: usize = 0;

    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        const node = entry.value_ptr;
        const mag_a = std.math.sqrt(node.qubit.a.re * node.qubit.a.re + node.qubit.a.im * node.qubit.a.im);
        const mag_b = std.math.sqrt(node.qubit.b.re * node.qubit.b.re + node.qubit.b.im * node.qubit.b.im);
        const magnitude_coherence = (mag_a + mag_b) / 2.0;
        const phase_coherence = (@cos(node.phase) + 1.0) / 2.0;
        total_coherence += (magnitude_coherence + phase_coherence) / 2.0;
        node_count += 1;
    }

    var total_correlation: f64 = 0.0;
    var edge_count: usize = 0;

    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            total_correlation += edge.quantum_correlation.abs();
            edge_count += 1;
        }
    }

    const avg_coherence = if (node_count > 0) total_coherence / @as(f64, @floatFromInt(node_count)) else 0.0;
    const avg_correlation = if (edge_count > 0) total_correlation / @as(f64, @floatFromInt(edge_count)) else 0.0;

    const obj = 1.0 - (avg_coherence + avg_correlation) / 2.0;
    return @max(0.0, @min(1.0, obj));
}

pub fn fractalDimensionObjective(state: *const OptimizationState) f64 {
    const graph = state.graph;
    var total_dimension: f64 = 0.0;
    var edge_count: usize = 0;

    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            total_dimension += edge.fractal_dimension;
            edge_count += 1;
        }
    }

    if (edge_count == 0) return 0.0;

    const avg_dimension = total_dimension / @as(f64, @floatFromInt(edge_count));
    const target_dimension: f64 = 1.5;

    return @abs(avg_dimension - target_dimension);
}

test "SymmetryGroup basic operations" {
    const identity = SymmetryGroup.identity;
    try std.testing.expectEqualStrings("identity", identity.toString());
    try std.testing.expectEqual(@as(usize, 1), identity.getOrder());

    const rotation = SymmetryGroup.rotation_90;
    const expected_angle: f64 = std.math.pi / 2.0;
    const actual_angle = rotation.getAngle();
    try std.testing.expectApproxEqAbs(expected_angle, actual_angle, 0.001);

    try std.testing.expectEqual(@as(usize, 1), SymmetryGroup.translation.getOrder());
    try std.testing.expectEqual(@as(usize, 2), SymmetryGroup.custom_rotation.getOrder());
}

test "SymmetryTransform apply" {
    const identity = SymmetryTransform.init(.identity);
    const result = identity.apply(1.0, 2.0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.x, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result.y, 0.001);

    const rotation_180 = SymmetryTransform.init(.rotation_180);
    const rotated = rotation_180.apply(1.0, 0.0);
    try std.testing.expectApproxEqAbs(@as(f64, -1.0), rotated.x, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rotated.y, 0.001);
}

test "SymmetryTransform inverse" {
    const rotation_90 = SymmetryTransform.init(.rotation_90);
    const inv = rotation_90.inverse();
    try std.testing.expectEqual(SymmetryGroup.rotation_270, inv.group);

    const identity = SymmetryTransform.init(.identity);
    const id_inv = identity.inverse();
    try std.testing.expectEqual(SymmetryGroup.identity, id_inv.group);
}

test "SymmetryTransform compose" {
    const rot90 = SymmetryTransform.init(.rotation_90);
    const composed = rot90.compose(&rot90);
    try std.testing.expectEqual(SymmetryGroup.rotation_180, composed.group);

    const identity = SymmetryTransform.init(.identity);
    const with_id = rot90.compose(&identity);
    try std.testing.expectEqual(SymmetryGroup.rotation_90, with_id.group);
}

test "OptimizationState basic" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const node = try Node.init(allocator, "test", "data", Qubit.initBasis0(), 0.0);
    try graph.addNode(node);

    var state = OptimizationState.init(allocator, &graph, 1.5, false);
    defer state.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 1.5), state.energy, 0.001);
    try std.testing.expectEqual(@as(usize, 0), state.iteration);
}

test "OptimizationState entanglement" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "n1", "d1", Qubit.initBasis0(), 0.0);
    try graph.addNode(n1);
    const n2 = try Node.init(allocator, "n2", "d2", Qubit.initBasis1(), 0.5);
    try graph.addNode(n2);

    var state = OptimizationState.init(allocator, &graph, 1.0, false);
    defer state.deinit();

    const info = EntanglementInfo.init(0.8, 0.5);
    try state.addEntanglement("n1", "n2", info);

    try std.testing.expect(state.hasEntanglement("n1", "n2"));
    try std.testing.expectEqual(@as(usize, 1), state.entangledPairsCount());

    const retrieved = state.getEntanglement("n1", "n2");
    try std.testing.expect(retrieved != null);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), retrieved.?.correlation_strength, 0.001);
}

test "OptimizationStatistics" {
    var stats = OptimizationStatistics.init();

    stats.moves_accepted = 75;
    stats.moves_rejected = 25;
    stats.updateAcceptanceRate();

    try std.testing.expectApproxEqAbs(@as(f64, 0.75), stats.acceptance_rate, 0.001);
}

test "EntangledStochasticSymmetryOptimizer init" {
    const allocator = std.testing.allocator;

    var optimizer = EntangledStochasticSymmetryOptimizer.init(allocator, 100.0, 0.95, 1000);
    defer optimizer.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 100.0), optimizer.temperature, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.95), optimizer.cooling_rate, 0.001);
    try std.testing.expectEqual(@as(usize, 1000), optimizer.max_iterations);
}

test "EntangledStochasticSymmetryOptimizer coolTemperature" {
    const allocator = std.testing.allocator;

    var optimizer = EntangledStochasticSymmetryOptimizer.init(allocator, 100.0, 0.9, 1000);
    defer optimizer.deinit();

    optimizer.coolTemperature();
    try std.testing.expectApproxEqAbs(@as(f64, 90.0), optimizer.temperature, 0.001);

    optimizer.coolTemperature();
    try std.testing.expectApproxEqAbs(@as(f64, 81.0), optimizer.temperature, 0.001);
}
