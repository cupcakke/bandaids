const std = @import("std");
const nsir_core = @import("nsir_core.zig");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Complex = std.math.Complex;
const Random = std.crypto.random;

pub const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
pub const Node = nsir_core.Node;
pub const Edge = nsir_core.Edge;
pub const EdgeQuality = nsir_core.EdgeQuality;
pub const EdgeKey = nsir_core.EdgeKey;

pub const FNDSError = error{
    PatternLengthOutOfRange,
    InvalidArgument,
    InvalidScale,
    InvalidWeight,
    InvalidConfidence,
    InvalidBranchingFactor,
    InvalidMaxDepth,
    InvalidCapacity,
    AllocatorMismatch,
    CycleDetected,
    DuplicateChild,
    NodeNotFound,
    TreeNotFound,
    IndexNotFound,
    Overflow,
};

fn satAddUsize(a: usize, b: usize) usize {
    const r = @addWithOverflow(a, b);
    if (r[1] != 0) return std.math.maxInt(usize);
    return r[0];
}

fn satSubUsize(a: usize, b: usize) usize {
    if (b > a) return 0;
    return a - b;
}

fn canonicalF64Bytes(v: f64) [8]u8 {
    var x = v;
    if (std.math.isNan(x)) {
        x = std.math.nan(f64);
    } else if (x == 0.0) {
        x = 0.0;
    }
    var out: [8]u8 = undefined;
    const bits = @as(u64, @bitCast(x));
    std.mem.writeInt(u64, &out, bits, .little);
    return out;
}

fn isFiniteF64(v: f64) bool {
    return !std.math.isNan(v) and !std.math.isInf(v);
}

pub const FNDSStatistics = struct {
    total_trees: usize,
    total_indices: usize,
    cache_hits: usize,
    cache_misses: usize,
    average_tree_depth: f64,
    memory_used: usize,
    total_nodes_across_trees: usize,
    total_patterns_indexed: usize,
    total_pattern_locations_indexed: usize,
    cache_hit_ratio: f64,
    last_operation_time_ns: u64,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .total_trees = 0,
            .total_indices = 0,
            .cache_hits = 0,
            .cache_misses = 0,
            .average_tree_depth = 0.0,
            .memory_used = 0,
            .total_nodes_across_trees = 0,
            .total_patterns_indexed = 0,
            .total_pattern_locations_indexed = 0,
            .cache_hit_ratio = 0.0,
            .last_operation_time_ns = 0,
        };
    }

    pub fn updateCacheHitRatio(self: *Self) void {
        const total = satAddUsize(self.cache_hits, self.cache_misses);
        if (total > 0) {
            self.cache_hit_ratio = @as(f64, @floatFromInt(self.cache_hits)) / @as(f64, @floatFromInt(total));
        } else {
            self.cache_hit_ratio = 0.0;
        }
    }

    pub fn recordCacheHit(self: *Self) void {
        self.cache_hits = satAddUsize(self.cache_hits, 1);
        self.updateCacheHitRatio();
    }

    pub fn recordCacheMiss(self: *Self) void {
        self.cache_misses = satAddUsize(self.cache_misses, 1);
        self.updateCacheHitRatio();
    }

    pub fn updateAverageTreeDepth(self: *Self, depths: []const usize) void {
        if (depths.len == 0) {
            self.average_tree_depth = 0.0;
            return;
        }
        var sum: f64 = 0.0;
        for (depths) |d| {
            sum += @as(f64, @floatFromInt(d));
        }
        self.average_tree_depth = sum / @as(f64, @floatFromInt(depths.len));
    }
};

pub const FractalNodeData = struct {
    id: []const u8,
    data: []const u8,
    weight: f64,
    scale: f64,
    fractal_signature: [32]u8,
    children_count: usize,
    metadata: StringHashMap([]const u8),
    metadata_keys_owned: ArrayList([]u8),
    allocator: Allocator,

    const Self = @This();

    fn computeSignature(id: []const u8, data: []const u8, weight: f64, scale: f64) [32]u8 {
        var signature: [32]u8 = undefined;
        var hasher = Sha256.init(.{});
        hasher.update(id);
        hasher.update(data);
        const wb = canonicalF64Bytes(weight);
        const sb = canonicalF64Bytes(scale);
        hasher.update(&wb);
        hasher.update(&sb);
        const hash_result = hasher.finalResult();
        @memcpy(signature[0..], hash_result[0..]);
        return signature;
    }

    pub fn init(allocator: Allocator, id: []const u8, data: []const u8, weight: f64, scale: f64) !Self {
        if (!isFiniteF64(weight)) return FNDSError.InvalidWeight;
        if (!isFiniteF64(scale)) return FNDSError.InvalidScale;

        const id_copy = try allocator.dupe(u8, id);
        errdefer allocator.free(id_copy);
        const data_copy = try allocator.dupe(u8, data);
        errdefer allocator.free(data_copy);

        const signature = computeSignature(id, data, weight, scale);

        return Self{
            .id = id_copy,
            .data = data_copy,
            .weight = weight,
            .scale = scale,
            .fractal_signature = signature,
            .children_count = 0,
            .metadata = StringHashMap([]const u8).init(allocator),
            .metadata_keys_owned = ArrayList([]u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.id);
        self.allocator.free(self.data);
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
        for (self.metadata_keys_owned.items) |k| {
            self.allocator.free(k);
        }
        self.metadata_keys_owned.deinit();
        self.id = &[_]u8{};
        self.data = &[_]u8{};
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        const id_copy = try allocator.dupe(u8, self.id);
        errdefer allocator.free(id_copy);
        const data_copy = try allocator.dupe(u8, self.data);
        errdefer allocator.free(data_copy);

        var new_metadata = StringHashMap([]const u8).init(allocator);
        var new_keys = ArrayList([]u8).init(allocator);
        errdefer {
            var it = new_metadata.iterator();
            while (it.next()) |e| allocator.free(e.value_ptr.*);
            new_metadata.deinit();
            for (new_keys.items) |k| allocator.free(k);
            new_keys.deinit();
        }

        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            const key_copy = try allocator.dupe(u8, entry.key_ptr.*);
            var key_added = false;
            errdefer if (!key_added) allocator.free(key_copy);

            const val_copy = try allocator.dupe(u8, entry.value_ptr.*);
            errdefer allocator.free(val_copy);

            try new_keys.append(key_copy);
            key_added = true;
            try new_metadata.put(key_copy, val_copy);
        }

        return Self{
            .id = id_copy,
            .data = data_copy,
            .weight = self.weight,
            .scale = self.scale,
            .fractal_signature = self.fractal_signature,
            .children_count = self.children_count,
            .metadata = new_metadata,
            .metadata_keys_owned = new_keys,
            .allocator = allocator,
        };
    }

    pub fn setMetadata(self: *Self, key: []const u8, value: []const u8) !void {
        const val_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(val_copy);

        if (self.metadata.getEntry(key)) |entry| {
            const old_val = entry.value_ptr.*;
            entry.value_ptr.* = val_copy;
            self.allocator.free(old_val);
            return;
        }

        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        try self.metadata_keys_owned.append(key_copy);
        errdefer _ = self.metadata_keys_owned.pop();
        try self.metadata.put(key_copy, val_copy);
        self.refreshSignature();
    }

    fn refreshSignature(self: *Self) void {
        self.fractal_signature = computeSignature(self.id, self.data, self.weight, self.scale);
    }

    pub fn getMetadata(self: *const Self, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }

    pub fn computeHash(self: *const Self) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(self.id);
        hasher.update(self.data);
        const wb = canonicalF64Bytes(self.weight);
        const sb = canonicalF64Bytes(self.scale);
        hasher.update(&wb);
        hasher.update(&sb);
        hasher.update(&self.fractal_signature);

        var keys = ArrayList([]const u8).init(self.allocator);
        defer keys.deinit();
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            keys.append(entry.key_ptr.*) catch return hasher.final();
        }
        std.mem.sort([]const u8, keys.items, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        for (keys.items) |k| {
            hasher.update(k);
            const v = self.metadata.get(k) orelse continue;
            hasher.update(v);
        }
        return hasher.final();
    }
};

pub const FractalEdgeData = struct {
    source_id: []const u8,
    target_id: []const u8,
    weight: f64,
    scale_ratio: f64,
    edge_type: EdgeType,
    fractal_correlation: f64,
    allocator: Allocator,

    pub const EdgeType = enum(u8) {
        hierarchical = 0,
        sibling = 1,
        cross_level = 2,
        self_similar = 3,

        pub fn toString(self: EdgeType) []const u8 {
            return switch (self) {
                .hierarchical => "hierarchical",
                .sibling => "sibling",
                .cross_level => "cross_level",
                .self_similar => "self_similar",
            };
        }
    };

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        source_id: []const u8,
        target_id: []const u8,
        weight: f64,
        scale_ratio: f64,
        edge_type: EdgeType,
    ) !Self {
        if (!isFiniteF64(weight)) return FNDSError.InvalidWeight;
        if (!isFiniteF64(scale_ratio)) return FNDSError.InvalidScale;

        const src = try allocator.dupe(u8, source_id);
        errdefer allocator.free(src);
        const tgt = try allocator.dupe(u8, target_id);
        errdefer allocator.free(tgt);

        return Self{
            .source_id = src,
            .target_id = tgt,
            .weight = weight,
            .scale_ratio = scale_ratio,
            .edge_type = edge_type,
            .fractal_correlation = 1.0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.source_id);
        self.allocator.free(self.target_id);
        self.source_id = &[_]u8{};
        self.target_id = &[_]u8{};
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        const src = try allocator.dupe(u8, self.source_id);
        errdefer allocator.free(src);
        const tgt = try allocator.dupe(u8, self.target_id);
        errdefer allocator.free(tgt);
        return Self{
            .source_id = src,
            .target_id = tgt,
            .weight = self.weight,
            .scale_ratio = self.scale_ratio,
            .edge_type = self.edge_type,
            .fractal_correlation = self.fractal_correlation,
            .allocator = allocator,
        };
    }
};

pub const FractalLevel = struct {
    level: usize,
    scale_factor: f64,
    nodes: StringHashMap(FractalNodeData),
    node_keys_owned: ArrayList([]u8),
    edges: StringHashMap(ArrayList(FractalEdgeData)),
    edge_keys_owned: ArrayList([]u8),
    parent_level: ?*FractalLevel,
    child_levels: ArrayList(*FractalLevel),
    node_count: usize,
    edge_count: usize,
    fractal_dimension: f64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, level: usize, scale_factor: f64) !Self {
        if (!isFiniteF64(scale_factor)) return FNDSError.InvalidScale;
        return Self{
            .level = level,
            .scale_factor = scale_factor,
            .nodes = StringHashMap(FractalNodeData).init(allocator),
            .node_keys_owned = ArrayList([]u8).init(allocator),
            .edges = StringHashMap(ArrayList(FractalEdgeData)).init(allocator),
            .edge_keys_owned = ArrayList([]u8).init(allocator),
            .parent_level = null,
            .child_levels = ArrayList(*FractalLevel).init(allocator),
            .node_count = 0,
            .edge_count = 0,
            .fractal_dimension = 1.0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var visited = std.AutoHashMap(*FractalLevel, void).init(self.allocator);
        defer visited.deinit();
        self.deinitInternal(&visited);
    }

    fn deinitInternal(self: *Self, visited: *std.AutoHashMap(*FractalLevel, void)) void {
        if (visited.contains(self)) return;
        visited.put(self, {}) catch {};

        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.nodes.deinit();
        for (self.node_keys_owned.items) |k| self.allocator.free(k);
        self.node_keys_owned.deinit();

        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            const edge_list = entry.value_ptr;
            for (edge_list.items) |*edge| {
                edge.deinit();
            }
            edge_list.deinit();
        }
        self.edges.deinit();
        for (self.edge_keys_owned.items) |k| self.allocator.free(k);
        self.edge_keys_owned.deinit();

        const children_copy = self.child_levels.toOwnedSlice() catch &[_]*FractalLevel{};
        defer if (children_copy.len > 0) self.allocator.free(children_copy);

        for (children_copy) |child| {
            child.deinitInternal(visited);
            self.allocator.destroy(child);
        }
    }

    pub fn getNode(self: *Self, node_id: []const u8) ?*FractalNodeData {
        return self.nodes.getPtr(node_id);
    }

    pub fn getNodeConst(self: *const Self, node_id: []const u8) ?*const FractalNodeData {
        return self.nodes.getPtr(node_id);
    }

    pub fn addNode(self: *Self, node: FractalNodeData) !void {
        if (node.allocator.ptr != self.allocator.ptr or node.allocator.vtable != self.allocator.vtable) {
            return FNDSError.AllocatorMismatch;
        }

        if (self.nodes.getPtr(node.id)) |existing| {
            var mut_node = node;
            existing.deinit();
            existing.* = mut_node;
            _ = &mut_node;
            return;
        }

        const key_copy = try self.allocator.dupe(u8, node.id);
        errdefer self.allocator.free(key_copy);
        try self.node_keys_owned.append(key_copy);
        errdefer _ = self.node_keys_owned.pop();
        try self.nodes.put(key_copy, node);
        self.node_count = satAddUsize(self.node_count, 1);
    }

    pub fn removeNode(self: *Self, node_id: []const u8) bool {
        const entry = self.nodes.getEntry(node_id) orelse return false;
        const stored_key = entry.key_ptr.*;
        var stored_val = entry.value_ptr.*;
        _ = self.nodes.remove(node_id);
        stored_val.deinit();
        for (self.node_keys_owned.items, 0..) |k, i| {
            if (k.ptr == stored_key.ptr and k.len == stored_key.len) {
                _ = self.node_keys_owned.swapRemove(i);
                self.allocator.free(k);
                break;
            }
        }
        self.node_count = satSubUsize(self.node_count, 1);

        if (self.edges.getEntry(node_id)) |eentry| {
            const ekey = eentry.key_ptr.*;
            var elist = eentry.value_ptr.*;
            _ = self.edges.remove(node_id);
            for (elist.items) |*edge| {
                edge.deinit();
                self.edge_count = satSubUsize(self.edge_count, 1);
            }
            elist.deinit();
            for (self.edge_keys_owned.items, 0..) |k, i| {
                if (k.ptr == ekey.ptr and k.len == ekey.len) {
                    _ = self.edge_keys_owned.swapRemove(i);
                    self.allocator.free(k);
                    break;
                }
            }
        }

        var it = self.edges.iterator();
        while (it.next()) |e| {
            var list = e.value_ptr;
            var i: usize = 0;
            while (i < list.items.len) {
                if (std.mem.eql(u8, list.items[i].target_id, node_id)) {
                    var removed = list.orderedRemove(i);
                    removed.deinit();
                    self.edge_count = satSubUsize(self.edge_count, 1);
                } else {
                    i += 1;
                }
            }
        }

        return true;
    }

    pub fn getEdges(self: *Self, source_id: []const u8) ?*ArrayList(FractalEdgeData) {
        return self.edges.getPtr(source_id);
    }

    pub fn getEdgesConst(self: *const Self, source_id: []const u8) ?[]const FractalEdgeData {
        if (self.edges.getPtr(source_id)) |list| {
            return list.items;
        }
        return null;
    }

    pub fn addEdge(self: *Self, edge: FractalEdgeData) !void {
        if (edge.allocator.ptr != self.allocator.ptr or edge.allocator.vtable != self.allocator.vtable) {
            return FNDSError.AllocatorMismatch;
        }

        if (self.edges.getPtr(edge.source_id)) |list| {
            try list.append(edge);
            self.edge_count = satAddUsize(self.edge_count, 1);
            return;
        }

        const key_copy = try self.allocator.dupe(u8, edge.source_id);
        errdefer self.allocator.free(key_copy);
        try self.edge_keys_owned.append(key_copy);
        errdefer _ = self.edge_keys_owned.pop();
        var list = ArrayList(FractalEdgeData).init(self.allocator);
        errdefer list.deinit();
        try list.append(edge);
        try self.edges.put(key_copy, list);
        self.edge_count = satAddUsize(self.edge_count, 1);
    }

    pub fn removeEdge(self: *Self, source_id: []const u8, target_id: []const u8) bool {
        const list_ptr = self.edges.getPtr(source_id) orelse return false;
        var i: usize = 0;
        var removed_any = false;
        while (i < list_ptr.items.len) {
            if (std.mem.eql(u8, list_ptr.items[i].target_id, target_id)) {
                var removed = list_ptr.orderedRemove(i);
                removed.deinit();
                self.edge_count = satSubUsize(self.edge_count, 1);
                removed_any = true;
            } else {
                i += 1;
            }
        }
        if (removed_any and list_ptr.items.len == 0) {
            const entry = self.edges.getEntry(source_id).?;
            const stored_key = entry.key_ptr.*;
            var stored_list = entry.value_ptr.*;
            _ = self.edges.remove(source_id);
            stored_list.deinit();
            for (self.edge_keys_owned.items, 0..) |k, i2_idx| {
                if (k.ptr == stored_key.ptr and k.len == stored_key.len) {
                    _ = self.edge_keys_owned.swapRemove(i2_idx);
                    self.allocator.free(k);
                    break;
                }
            }
        }
        return removed_any;
    }

    pub fn addChildLevel(self: *Self, child: *FractalLevel) !void {
        if (child == self) return FNDSError.CycleDetected;
        if (child.allocator.ptr != self.allocator.ptr or child.allocator.vtable != self.allocator.vtable) {
            return FNDSError.AllocatorMismatch;
        }
        for (self.child_levels.items) |existing| {
            if (existing == child) return FNDSError.DuplicateChild;
        }
        var ancestor: ?*FractalLevel = self.parent_level;
        while (ancestor) |a| : (ancestor = a.parent_level) {
            if (a == child) return FNDSError.CycleDetected;
        }
        child.parent_level = self;
        try self.child_levels.append(child);
    }

    pub fn getChildLevel(self: *Self, index: usize) ?*FractalLevel {
        if (index < self.child_levels.items.len) {
            return self.child_levels.items[index];
        }
        return null;
    }

    pub fn computeLocalFractalDimension(self: *Self) f64 {
        if (self.node_count < 2) {
            self.fractal_dimension = 0.0;
            return 0.0;
        }

        const box_sizes = [_]usize{ 1, 2, 4, 8 };
        var log_n_sum: f64 = 0.0;
        var log_r_sum: f64 = 0.0;
        var log_nr_sum: f64 = 0.0;
        var log_r2_sum: f64 = 0.0;
        var count: usize = 0;

        for (box_sizes) |size| {
            const box_count = self.estimateBoxCount(size);
            if (box_count > 0 and size > 0) {
                const log_n = @log(@as(f64, @floatFromInt(box_count)));
                const log_r = @log(1.0 / @as(f64, @floatFromInt(size)));
                log_n_sum += log_n;
                log_r_sum += log_r;
                log_nr_sum += log_n * log_r;
                log_r2_sum += log_r * log_r;
                count += 1;
            }
        }

        if (count < 2) {
            self.fractal_dimension = 1.0;
            return 1.0;
        }

        const n = @as(f64, @floatFromInt(count));
        const denominator = n * log_r2_sum - log_r_sum * log_r_sum;
        if (@abs(denominator) < 1e-10) {
            self.fractal_dimension = 1.0;
            return 1.0;
        }

        const slope = (n * log_nr_sum - log_n_sum * log_r_sum) / denominator;
        const result = @abs(slope);
        if (!isFiniteF64(result)) {
            self.fractal_dimension = 1.0;
            return 1.0;
        }
        self.fractal_dimension = result;
        return result;
    }

    fn estimateBoxCount(self: *Self, box_size: usize) usize {
        if (box_size == 0) return 0;
        if (self.node_count == 0) return 0;
        var node_positions = ArrayList(usize).init(self.allocator);
        defer node_positions.deinit();
        var edge_positions = ArrayList(usize).init(self.allocator);
        defer edge_positions.deinit();
        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            const weight_bits = canonicalF64Bytes(node.weight);
            var h1 = std.hash.Wyhash.init(0);
            h1.update(node.id);
            h1.update(&weight_bits);
            const h1_val = h1.final();
            var h2 = std.hash.Wyhash.init(h1_val);
            h2.update(node.data);
            const position = h2.final();
            try node_positions.append(@as(usize, @truncate(position)));
        }
        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |edge| {
                const src_bits = canonicalF64Bytes(edge.weight);
                var h1 = std.hash.Wyhash.init(0);
                h1.update(edge.source_id);
                h1.update(edge.target_id);
                h1.update(&src_bits);
                const h1_val = h1.final();
                var h2 = std.hash.Wyhash.init(h1_val);
                h2.update(&canonicalF64Bytes(edge.fractal_correlation));
                const position = h2.final();
                try edge_positions.append(@as(usize, @truncate(position)));
            }
        }
        var occupied = std.AutoHashMap(usize, void).init(self.allocator);
        defer occupied.deinit();
        const scale = @as(f64, @floatFromInt(box_size));
        for (node_positions.items) |pos| {
            const scaled = @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(pos)) / scale)));
            occupied.put(scaled, {}) catch return self.node_count;
        }
        for (edge_positions.items) |pos| {
            const scaled = @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(pos)) / scale)));
            occupied.put(scaled, {}) catch {};
        }
        return occupied.count();
    }

    pub fn getDepth(self: *const Self) usize {
        var visited = std.AutoHashMap(*const FractalLevel, void).init(self.allocator);
        defer visited.deinit();
        return self.getDepthInternal(&visited);
    }

    fn getDepthInternal(self: *const Self, visited: *std.AutoHashMap(*const FractalLevel, void)) usize {
        if (visited.contains(self)) return 0;
        visited.put(self, {}) catch return 1;
        if (self.child_levels.items.len == 0) return 1;
        var max_child: usize = 0;
        for (self.child_levels.items) |child| {
            const d = child.getDepthInternal(visited);
            if (d > max_child) max_child = d;
        }
        return satAddUsize(max_child, 1);
    }

    pub fn getTotalNodeCount(self: *const Self) usize {
        var visited = std.AutoHashMap(*const FractalLevel, void).init(self.allocator);
        defer visited.deinit();
        return self.getTotalNodeCountInternal(&visited);
    }

    fn getTotalNodeCountInternal(self: *const Self, visited: *std.AutoHashMap(*const FractalLevel, void)) usize {
        if (visited.contains(self)) return 0;
        visited.put(self, {}) catch return self.node_count;
        var total = self.node_count;
        for (self.child_levels.items) |child| {
            total = satAddUsize(total, child.getTotalNodeCountInternal(visited));
        }
        return total;
    }
};

pub const TraversalOrder = enum {
    pre_order,
    post_order,
    level_order,
    fractal_order,
};

pub const TraversalCallback = *const fn (*FractalLevel, usize, ?*anyopaque) void;

pub const FractalTree = struct {
    root: *FractalLevel,
    max_depth: usize,
    branching_factor: usize,
    total_nodes: usize,
    tree_id: [32]u8,
    creation_time: u64,
    last_modified: u64,
    allocator: Allocator,
    is_balanced: bool,
    depth_cache: ?usize,

    const Self = @This();

    pub fn init(allocator: Allocator, max_depth: usize, branching_factor: usize) !Self {
        if (branching_factor < 2) return FNDSError.InvalidBranchingFactor;
        if (max_depth == 0) return FNDSError.InvalidMaxDepth;

        const root = try allocator.create(FractalLevel);
        errdefer allocator.destroy(root);
        root.* = try FractalLevel.init(allocator, 0, 1.0);
        errdefer root.deinit();

        var tree_id: [32]u8 = undefined;
        Random.bytes(&tree_id);

        const now = @as(u64, @intCast(std.time.nanoTimestamp()));

        return Self{
            .root = root,
            .max_depth = max_depth,
            .branching_factor = branching_factor,
            .total_nodes = 0,
            .tree_id = tree_id,
            .creation_time = now,
            .last_modified = now,
            .allocator = allocator,
            .is_balanced = true,
            .depth_cache = null,
        };
    }

    pub fn deinit(self: *Self) void {
        self.root.deinit();
        self.allocator.destroy(self.root);
    }

    pub fn insert(self: *Self, node_id: []const u8, data: []const u8, target_level: usize) !bool {
        if (target_level > self.max_depth) return false;
        if (node_id.len == 0) return FNDSError.InvalidArgument;

        if (self.searchLevelForNode(self.root, node_id)) |found| {
            const new_data = try self.allocator.dupe(u8, data);
            self.allocator.free(@constCast(found.data));
            found.data = new_data;
            found.refreshSignature();
            self.last_modified = @as(u64, @intCast(std.time.nanoTimestamp()));
            return true;
        }

        var current_level = self.root;
        var depth: usize = 0;

        while (depth < target_level) : (depth += 1) {
            if (current_level.child_levels.items.len < self.branching_factor) {
                const child_scale = current_level.scale_factor / @as(f64, @floatFromInt(self.branching_factor));
                const new_child = try self.allocator.create(FractalLevel);
                errdefer self.allocator.destroy(new_child);
                new_child.* = try FractalLevel.init(self.allocator, depth + 1, child_scale);
                errdefer new_child.deinit();
                try current_level.addChildLevel(new_child);
            }

            const child_index = self.computeChildIndex(node_id, depth, current_level.child_levels.items.len);
            current_level = current_level.child_levels.items[child_index];
        }

        var node = try FractalNodeData.init(self.allocator, node_id, data, 1.0, current_level.scale_factor);
        errdefer node.deinit();
        try current_level.addNode(node);

        self.total_nodes = satAddUsize(self.total_nodes, 1);
        self.last_modified = @as(u64, @intCast(std.time.nanoTimestamp()));
        self.depth_cache = null;

        if (self.total_nodes % 100 == 0) {
            self.checkBalance();
        }

        return true;
    }

    fn computeChildIndex(self: *const Self, node_id: []const u8, depth: usize, max_children: usize) usize {
        if (max_children == 0) return 0;
        var hasher = std.hash.Wyhash.init(@as(u64, @intCast(depth)) ^ std.mem.readInt(u64, self.tree_id[0..8], .little));
        hasher.update(node_id);
        const hash = hasher.final();
        return @as(usize, @intCast(hash % @as(u64, @intCast(max_children))));
    }

    pub fn search(self: *Self, node_id: []const u8) ?*FractalNodeData {
        return self.searchInternal(self.root, node_id);
    }

    fn searchInternal(self: *Self, level: *FractalLevel, node_id: []const u8) ?*FractalNodeData {
        if (level.getNode(node_id)) |node| return node;
        var visited = std.AutoHashMap(*FractalLevel, void).init(self.allocator);
        defer visited.deinit();
        var fifo = std.fifo.LinearFifo(*FractalLevel, .Dynamic).init(self.allocator);
        defer fifo.deinit();
        visited.put(level, {}) catch return null;
        for (level.child_levels.items) |child| {
            fifo.writeItem(child) catch return null;
        }
        while (fifo.readItem()) |curr| {
            if (visited.contains(curr)) continue;
            visited.put(curr, {}) catch return null;
            if (curr.getNode(node_id)) |node| return node;
            for (curr.child_levels.items) |child| {
                fifo.writeItem(child) catch return null;
            }
        }
        return null;
    }

    fn searchLevelForNode(self: *Self, level: *FractalLevel, node_id: []const u8) ?*FractalNodeData {
        _ = self;
        return level.getNode(node_id);
    }

    pub fn searchConst(self: *const Self, node_id: []const u8) ?*const FractalNodeData {
        if (self.root.getNodeConst(node_id)) |n| return n;
        var visited = std.AutoHashMap(*const FractalLevel, void).init(self.allocator);
        defer visited.deinit();
        var stack = std.ArrayList(*const FractalLevel).init(self.allocator);
        defer stack.deinit();
        visited.put(self.root, {}) catch return null;
        for (self.root.child_levels.items) |child| {
            stack.append(child) catch return null;
        }
        while (stack.pop()) |curr| {
            if (visited.contains(curr)) continue;
            visited.put(curr, {}) catch return null;
            if (curr.getNodeConst(node_id)) |n| return n;
            for (curr.child_levels.items) |child| {
                stack.append(child) catch return null;
            }
        }
        return null;
    }

    pub fn delete(self: *Self, node_id: []const u8) bool {
        if (self.deleteFromLevel(self.root, node_id)) {
            self.total_nodes = satSubUsize(self.total_nodes, 1);
            self.last_modified = @as(u64, @intCast(std.time.nanoTimestamp()));
            self.depth_cache = null;
            self.checkBalance();
            return true;
        }
        return false;
    }

    fn deleteFromLevel(self: *Self, root: *FractalLevel, node_id: []const u8) bool {
        var visited = std.AutoHashMap(*FractalLevel, void).init(self.allocator);
        defer visited.deinit();
        var stack = std.ArrayList(*FractalLevel).init(self.allocator);
        defer stack.deinit();
        stack.append(root) catch return false;
        while (stack.pop()) |level| {
            if (visited.contains(level)) continue;
            visited.put(level, {}) catch return false;
            if (level.removeNode(node_id)) return true;
            for (level.child_levels.items) |child| {
                stack.append(child) catch return false;
            }
        }
        return false;
    }

    pub fn traverse(self: *Self, order: TraversalOrder, callback: TraversalCallback, ctx: ?*anyopaque) !void {
        var visited = std.AutoHashMap(*FractalLevel, void).init(self.allocator);
        defer visited.deinit();
        switch (order) {
            .pre_order => try self.traversePreOrder(self.root, 0, callback, ctx, &visited),
            .post_order => try self.traversePostOrder(self.root, 0, callback, ctx, &visited),
            .level_order => try self.traverseLevelOrder(callback, ctx),
            .fractal_order => try self.traverseFractalOrder(self.root, 0, callback, ctx, &visited),
        }
    }

    fn traversePreOrder(self: *Self, level: *FractalLevel, depth: usize, callback: TraversalCallback, ctx: ?*anyopaque, visited: *std.AutoHashMap(*FractalLevel, void)) !void {
        if (visited.contains(level)) return;
        try visited.put(level, {});
        callback(level, depth, ctx);
        for (level.child_levels.items) |child| {
            try self.traversePreOrder(child, satAddUsize(depth, 1), callback, ctx, visited);
        }
    }

    fn traversePostOrder(self: *Self, level: *FractalLevel, depth: usize, callback: TraversalCallback, ctx: ?*anyopaque, visited: *std.AutoHashMap(*FractalLevel, void)) !void {
        if (visited.contains(level)) return;
        try visited.put(level, {});
        for (level.child_levels.items) |child| {
            try self.traversePostOrder(child, satAddUsize(depth, 1), callback, ctx, visited);
        }
        callback(level, depth, ctx);
    }

    fn traverseLevelOrder(self: *Self, callback: TraversalCallback, ctx: ?*anyopaque) !void {
        var visited = std.AutoHashMap(*FractalLevel, void).init(self.allocator);
        defer visited.deinit();
        var fifo = std.fifo.LinearFifo(struct { level: *FractalLevel, depth: usize }, .Dynamic).init(self.allocator);
        defer fifo.deinit();
        try fifo.writeItem(.{ .level = self.root, .depth = 0 });
        while (fifo.readItem()) |item| {
            if (visited.contains(item.level)) continue;
            try visited.put(item.level, {});
            callback(item.level, item.depth, ctx);
            for (item.level.child_levels.items) |child| {
                try fifo.writeItem(.{ .level = child, .depth = satAddUsize(item.depth, 1) });
            }
        }
    }

    fn traverseFractalOrder(self: *Self, level: *FractalLevel, depth: usize, callback: TraversalCallback, ctx: ?*anyopaque, visited: *std.AutoHashMap(*FractalLevel, void)) !void {
        if (visited.contains(level)) return;
        try visited.put(level, {});
        callback(level, depth, ctx);

        const len = level.child_levels.items.len;
        if (len == 0) return;

        const mid = len / 2;
        var i: usize = 0;
        while (i < mid) : (i += 1) {
            try self.traverseFractalOrder(level.child_levels.items[i], satAddUsize(depth, 1), callback, ctx, visited);
        }
        i = len;
        while (i > mid) {
            i -= 1;
            try self.traverseFractalOrder(level.child_levels.items[i], satAddUsize(depth, 1), callback, ctx, visited);
        }
    }

    pub fn getDepth(self: *Self) usize {
        if (self.depth_cache) |cached| return cached;
        const depth = self.root.getDepth();
        self.depth_cache = depth;
        return depth;
    }

    pub fn getDepthConst(self: *const Self) usize {
        return self.root.getDepth();
    }

    pub fn balance(self: *Self) !void {
        var all_nodes = ArrayList(FractalNodeData).init(self.allocator);
        errdefer {
            for (all_nodes.items) |*n| n.deinit();
            all_nodes.deinit();
        }

        try self.collectAllNodes(self.root, &all_nodes);

        var sorted_ids = try self.allocator.alloc([]const u8, all_nodes.items.len);
        defer self.allocator.free(sorted_ids);
        var idx: usize = 0;
        while (idx < all_nodes.items.len) : (idx += 1) {
            sorted_ids[idx] = all_nodes.items[idx].id;
        }
        std.mem.sort([]const u8, sorted_ids, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan);

        const new_root = try self.allocator.create(FractalLevel);
        errdefer self.allocator.destroy(new_root);
        new_root.* = try FractalLevel.init(self.allocator, 0, 1.0);
        errdefer new_root.deinit();

        const old_root = self.root;
        self.root = new_root;
        const old_total = self.total_nodes;
        self.total_nodes = 0;

        var rebuilt = false;
        defer if (!rebuilt) {
            new_root.deinit();
            self.allocator.destroy(new_root);
            self.root = old_root;
            self.total_nodes = old_total;
        };

        var max_depth = self.computeOptimalDepth();
        if (max_depth == 0) max_depth = 1;

        for (all_nodes.items, 0..) |*node, node_idx| {
            const target_level = (node_idx * max_depth) / all_nodes.items.len;
            var current_level = self.root;
            var depth: usize = 0;
            while (depth < target_level) : (depth += 1) {
                if (current_level.child_levels.items.len < self.branching_factor) {
                    const child_scale = current_level.scale_factor / @as(f64, @floatFromInt(self.branching_factor));
                    const new_child = try self.allocator.create(FractalLevel);
                    errdefer self.allocator.destroy(new_child);
                    new_child.* = try FractalLevel.init(self.allocator, depth + 1, child_scale);
                    errdefer new_child.deinit();
                    try current_level.addChildLevel(new_child);
                }
                const child_index = (node_idx + depth) % current_level.child_levels.items.len;
                current_level = current_level.child_levels.items[child_index];
            }
            const cloned = try node.clone(self.allocator);
            try current_level.addNode(cloned);
            node.deinit();
            self.total_nodes = satAddUsize(self.total_nodes, 1);
        }

        all_nodes.clearRetainingCapacity();
        rebuilt = true;
        old_root.deinit();
        self.allocator.destroy(old_root);
        self.is_balanced = true;
        self.depth_cache = null;
        self.last_modified = @as(u64, @intCast(std.time.nanoTimestamp()));
    }

    fn collectAllNodes(self: *Self, root: *FractalLevel, nodes: *ArrayList(FractalNodeData)) !void {
        var visited = std.AutoHashMap(*FractalLevel, void).init(self.allocator);
        defer visited.deinit();
        var stack = std.ArrayList(*FractalLevel).init(self.allocator);
        defer stack.deinit();
        try stack.append(root);
        while (stack.pop()) |curr| {
            if (visited.contains(curr)) continue;
            try visited.put(curr, {});
            var iter = curr.nodes.iterator();
            while (iter.next()) |entry| {
                var cloned = try entry.value_ptr.clone(self.allocator);
                errdefer cloned.deinit();
                try nodes.append(cloned);
            }
            for (curr.child_levels.items) |child| {
                try stack.append(child);
            }
        }
    }

    fn computeOptimalLevel(self: *const Self, node_id: []const u8) usize {
        var hasher = std.hash.Wyhash.init(std.mem.readInt(u64, self.tree_id[0..8], .little));
        hasher.update(node_id);
        const hash = hasher.final();
        const cap = if (self.max_depth == std.math.maxInt(usize)) self.max_depth else self.max_depth + 1;
        const level = @as(usize, @intCast(hash % @as(u64, @intCast(cap))));
        return @min(level, self.max_depth);
    }

    fn checkBalance(self: *Self) void {
        const depth = self.getDepth();
        const optimal_depth = self.computeOptimalDepth();
        const threshold = satAddUsize(optimal_depth, 2);
        self.is_balanced = (depth <= threshold);
    }

    fn computeOptimalDepth(self: *const Self) usize {
        if (self.total_nodes == 0) return 0;
        if (self.total_nodes == 1) return 1;
        const log_base = @log(@as(f64, @floatFromInt(@max(2, self.branching_factor))));
        const log_nodes = @log(@as(f64, @floatFromInt(self.total_nodes)));
        if (log_base <= 0.0) return self.max_depth;
        const result = @ceil(log_nodes / log_base);
        if (!isFiniteF64(result) or result < 0.0) return 0;
        const max_f = @as(f64, @floatFromInt(std.math.maxInt(usize)));
        if (result > max_f) return std.math.maxInt(usize);
        return @as(usize, @intFromFloat(result));
    }

    pub fn getTreeIdHex(self: *const Self) [64]u8 {
        var hex_buf: [64]u8 = undefined;
        const hex_chars = "0123456789abcdef";
        for (self.tree_id, 0..) |b, i| {
            hex_buf[i * 2] = hex_chars[(b >> 4) & 0xF];
            hex_buf[i * 2 + 1] = hex_chars[b & 0xF];
        }
        return hex_buf;
    }

    pub fn computeFractalDimension(self: *Self) f64 {
        var visited = std.AutoHashMap(*FractalLevel, void).init(self.allocator);
        defer visited.deinit();
        return self.computeLevelDimension(self.root, &visited);
    }

    fn computeLevelDimension(self: *Self, level: *FractalLevel, visited: *std.AutoHashMap(*FractalLevel, void)) f64 {
        if (visited.contains(level)) return 0.0;
        visited.put(level, {}) catch return 0.0;

        const local_dim = level.computeLocalFractalDimension();
        if (level.child_levels.items.len == 0) return local_dim;

        var child_dim_sum: f64 = 0.0;
        var counted: f64 = 0.0;
        for (level.child_levels.items) |child| {
            child_dim_sum += self.computeLevelDimension(child, visited);
            counted += 1.0;
        }
        if (counted == 0.0) return local_dim;
        const child_avg = child_dim_sum / counted;
        return (local_dim + child_avg) / 2.0;
    }
};

pub const PatternLocation = struct {
    tree_id: [32]u8,
    level: usize,
    node_id: []const u8,
    offset: usize,
    length: usize,
    confidence: f64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        tree_id: [32]u8,
        level: usize,
        node_id: []const u8,
        offset: usize,
        length: usize,
        confidence: f64,
    ) !Self {
        if (!isFiniteF64(confidence)) return FNDSError.InvalidConfidence;
        if (confidence < 0.0 or confidence > 1.0) return FNDSError.InvalidConfidence;
        const overflow = @addWithOverflow(offset, length);
        if (overflow[1] != 0) return FNDSError.Overflow;

        const id_copy = try allocator.dupe(u8, node_id);
        errdefer allocator.free(id_copy);

        return Self{
            .tree_id = tree_id,
            .level = level,
            .node_id = id_copy,
            .offset = offset,
            .length = length,
            .confidence = confidence,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.node_id);
        self.node_id = &[_]u8{};
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        const id_copy = try allocator.dupe(u8, self.node_id);
        errdefer allocator.free(id_copy);
        return Self{
            .tree_id = self.tree_id,
            .level = self.level,
            .node_id = id_copy,
            .offset = self.offset,
            .length = self.length,
            .confidence = self.confidence,
            .allocator = allocator,
        };
    }
};

pub const SimilarPatternEntry = struct {
    pattern: []const u8,
    similarity: f64,
};

pub const SelfSimilarIndex = struct {
    patterns: StringHashMap(ArrayList(PatternLocation)),
    pattern_keys: ArrayList([]u8),
    dimension_estimate: f64,
    pattern_count: usize,
    total_locations: usize,
    min_pattern_length: usize,
    max_pattern_length: usize,
    similarity_threshold: f64,
    allocator: Allocator,
    creation_time: u64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .patterns = StringHashMap(ArrayList(PatternLocation)).init(allocator),
            .pattern_keys = ArrayList([]u8).init(allocator),
            .dimension_estimate = 0.0,
            .pattern_count = 0,
            .total_locations = 0,
            .min_pattern_length = 1,
            .max_pattern_length = 256,
            .similarity_threshold = 0.8,
            .allocator = allocator,
            .creation_time = @as(u64, @intCast(std.time.nanoTimestamp())),
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.patterns.iterator();
        while (iter.next()) |entry| {
            const locations = entry.value_ptr;
            for (locations.items) |*loc| loc.deinit();
            locations.deinit();
        }
        self.patterns.deinit();
        for (self.pattern_keys.items) |k| self.allocator.free(k);
        self.pattern_keys.deinit();
    }

    pub fn setSimilarityThreshold(self: *Self, t: f64) !void {
        if (!isFiniteF64(t) or t < 0.0 or t > 1.0) return FNDSError.InvalidArgument;
        self.similarity_threshold = t;
    }

    pub fn setLengthBounds(self: *Self, min_len: usize, max_len: usize) !void {
        if (min_len > max_len) return FNDSError.InvalidArgument;
        self.min_pattern_length = min_len;
        self.max_pattern_length = max_len;
    }

    pub fn addPattern(self: *Self, pattern: []const u8, location: PatternLocation) !void {
        var loc_mut = location;
        if (pattern.len < self.min_pattern_length or pattern.len > self.max_pattern_length) {
            loc_mut.deinit();
            return FNDSError.PatternLengthOutOfRange;
        }
        if (loc_mut.allocator.ptr != self.allocator.ptr or loc_mut.allocator.vtable != self.allocator.vtable) {
            loc_mut.deinit();
            return FNDSError.AllocatorMismatch;
        }

        if (self.patterns.getPtr(pattern)) |list| {
            try list.append(loc_mut);
            self.total_locations = satAddUsize(self.total_locations, 1);
            return;
        }

        const pattern_copy = try self.allocator.dupe(u8, pattern);
        errdefer self.allocator.free(pattern_copy);
        try self.pattern_keys.append(pattern_copy);
        errdefer _ = self.pattern_keys.pop();
        var list = ArrayList(PatternLocation).init(self.allocator);
        errdefer list.deinit();
        try list.append(loc_mut);
        try self.patterns.put(pattern_copy, list);
        self.pattern_count = satAddUsize(self.pattern_count, 1);
        self.total_locations = satAddUsize(self.total_locations, 1);
    }

    pub fn findPattern(self: *Self, pattern: []const u8) []const PatternLocation {
        if (self.patterns.getPtr(pattern)) |locations| {
            return locations.items;
        }
        return &[_]PatternLocation{};
    }

    pub fn findSimilarPatterns(self: *Self, pattern: []const u8, results: *ArrayList(SimilarPatternEntry)) !void {
        var iter = self.patterns.iterator();
        while (iter.next()) |entry| {
            const similarity = self.computeSimilarity(pattern, entry.key_ptr.*);
            if (similarity >= self.similarity_threshold) {
                try results.append(.{ .pattern = entry.key_ptr.*, .similarity = similarity });
            }
        }
        std.mem.sort(SimilarPatternEntry, results.items, {}, struct {
            fn cmp(_: void, a: SimilarPatternEntry, b: SimilarPatternEntry) bool {
                return a.similarity > b.similarity;
            }
        }.cmp);
    }

    fn computeSimilarity(self: *const Self, a: []const u8, b: []const u8) f64 {
        _ = self;
        if (a.len == 0 and b.len == 0) return 1.0;
        if (a.len == 0 or b.len == 0) return 0.0;
        if (std.mem.eql(u8, a, b)) return 1.0;

        const max_len = @max(a.len, b.len);
        const min_len = @min(a.len, b.len);

        var matches: usize = 0;
        var i: usize = 0;
        while (i < min_len) : (i += 1) {
            if (a[i] == b[i]) matches += 1;
        }

        const length_ratio = @as(f64, @floatFromInt(min_len)) / @as(f64, @floatFromInt(max_len));
        const match_ratio = @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(min_len));
        return (length_ratio + match_ratio) / 2.0;
    }

    pub fn computeFractalDimension(self: *Self) f64 {
        if (self.pattern_count < 2) {
            self.dimension_estimate = 0.0;
            return 0.0;
        }

        var length_counts = AutoHashMap(usize, usize).init(self.allocator);
        defer length_counts.deinit();

        var iter = self.patterns.iterator();
        while (iter.next()) |entry| {
            const len = entry.key_ptr.len;
            const current = length_counts.get(len) orelse 0;
            length_counts.put(len, satAddUsize(current, 1)) catch continue;
        }

        var log_l_sum: f64 = 0.0;
        var log_n_sum: f64 = 0.0;
        var log_ln_sum: f64 = 0.0;
        var log_l2_sum: f64 = 0.0;
        var count: usize = 0;

        var lc_iter = length_counts.iterator();
        while (lc_iter.next()) |entry| {
            const len = entry.key_ptr.*;
            const cnt = entry.value_ptr.*;
            if (len > 0 and cnt > 0) {
                const log_l = @log(@as(f64, @floatFromInt(len)));
                const log_n = @log(@as(f64, @floatFromInt(cnt)));
                log_l_sum += log_l;
                log_n_sum += log_n;
                log_ln_sum += log_l * log_n;
                log_l2_sum += log_l * log_l;
                count += 1;
            }
        }

        if (count < 2) {
            self.dimension_estimate = 0.0;
            return 0.0;
        }

        const n = @as(f64, @floatFromInt(count));
        const denominator = n * log_l2_sum - log_l_sum * log_l_sum;
        if (@abs(denominator) < 1e-10) {
            self.dimension_estimate = 0.0;
            return 0.0;
        }

        const slope = (n * log_ln_sum - log_l_sum * log_n_sum) / denominator;
        const result = @abs(slope);
        if (!isFiniteF64(result)) {
            self.dimension_estimate = 0.0;
            return 0.0;
        }
        self.dimension_estimate = result;
        return result;
    }

    pub fn removePattern(self: *Self, pattern: []const u8) bool {
        const entry = self.patterns.getEntry(pattern) orelse return false;
        const stored_key = entry.key_ptr.*;
        var locations = entry.value_ptr.*;
        _ = self.patterns.remove(pattern);

        for (locations.items) |*loc| {
            loc.deinit();
            self.total_locations = satSubUsize(self.total_locations, 1);
        }
        locations.deinit();

        for (self.pattern_keys.items, 0..) |k, i| {
            if (k.ptr == stored_key.ptr and k.len == stored_key.len) {
                _ = self.pattern_keys.swapRemove(i);
                self.allocator.free(k);
                break;
            }
        }
        self.pattern_count = satSubUsize(self.pattern_count, 1);
        return true;
    }

    pub fn getPatternCount(self: *const Self) usize {
        return self.pattern_count;
    }

    pub fn getTotalLocations(self: *const Self) usize {
        return self.total_locations;
    }
};

pub fn CoalescedEntry(comptime K: type, comptime V: type) type {
    return struct {
        key: K,
        value: V,
        next_index: ?usize,
        is_primary: bool,
    };
}

pub fn CoalescedHashMap(comptime K: type, comptime V: type) type {
    return struct {
        buckets: []?Entry,
        capacity: usize,
        size: usize,
        load_factor: f64,
        max_load_factor: f64,
        cellar_start: usize,
        free_list: ArrayList(usize),
        allocator: Allocator,
        seed: u64,

        const Entry = CoalescedEntry(K, V);
        const Self = @This();
        const DEFAULT_CAPACITY: usize = 16;
        const DEFAULT_MAX_LOAD_FACTOR: f64 = 0.86;
        const CELLAR_RATIO: f64 = 0.14;

        pub fn init(allocator: Allocator) !Self {
            return initWithCapacity(allocator, DEFAULT_CAPACITY);
        }

        pub fn initWithCapacity(allocator: Allocator, initial_capacity: usize) !Self {
            const capacity = @max(initial_capacity, 8);
            const buckets = try allocator.alloc(?Entry, capacity);
            @memset(buckets, null);

            const cellar_size_f = @as(f64, @floatFromInt(capacity)) * CELLAR_RATIO;
            const cellar_size = @as(usize, @intFromFloat(@floor(cellar_size_f)));
            const cellar_actual = @max(cellar_size, 1);
            const cellar_start = if (cellar_actual >= capacity) capacity / 2 else capacity - cellar_actual;

            var seed_bytes: [8]u8 = undefined;
            Random.bytes(&seed_bytes);
            const seed = std.mem.readInt(u64, &seed_bytes, .little);

            return Self{
                .buckets = buckets,
                .capacity = capacity,
                .size = 0,
                .load_factor = 0.0,
                .max_load_factor = DEFAULT_MAX_LOAD_FACTOR,
                .cellar_start = cellar_start,
                .free_list = ArrayList(usize).init(allocator),
                .allocator = allocator,
                .seed = seed,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buckets);
            self.free_list.deinit();
        }

        fn hash(self: *const Self, key: K) usize {
            var hasher = std.hash.Wyhash.init(self.seed);
            std.hash.autoHash(&hasher, key);
            return @as(usize, @truncate(hasher.final()));
        }

        fn eql(self: *const Self, a: K, b: K) bool {
            _ = self;
            return std.meta.eql(a, b);
        }

        const PutError = FNDSError || Allocator.Error;

        pub fn put(self: *Self, key: K, value: V) PutError!void {
            if (self.cellar_start == 0) return FNDSError.InvalidCapacity;
            if (self.load_factor >= self.max_load_factor) {
                try self.resize(satAddUsize(self.capacity, self.capacity));
            }
            try self.putInternal(key, value);
        }

        fn putInternal(self: *Self, key: K, value: V) PutError!void {
            const index = self.hash(key) % self.cellar_start;

            if (self.buckets[index] == null) {
                self.buckets[index] = Entry{
                    .key = key,
                    .value = value,
                    .next_index = null,
                    .is_primary = true,
                };
                self.size = satAddUsize(self.size, 1);
                self.updateLoadFactor();
                return;
            }

            var current_idx = index;
            while (true) {
                const entry = self.buckets[current_idx].?;
                if (self.eql(entry.key, key)) {
                    self.buckets[current_idx].?.value = value;
                    return;
                }
                if (entry.next_index) |next| {
                    current_idx = next;
                } else break;
            }

            const slot_opt = self.findEmptySlot();
            if (slot_opt) |slot| {
                self.buckets[slot] = Entry{
                    .key = key,
                    .value = value,
                    .next_index = null,
                    .is_primary = false,
                };
                self.buckets[current_idx].?.next_index = slot;
                self.size = satAddUsize(self.size, 1);
                self.updateLoadFactor();
            } else {
                try self.resize(satAddUsize(self.capacity, self.capacity));
                try self.putInternal(key, value);
            }
        }

        fn findEmptySlot(self: *Self) ?usize {
            if (self.free_list.items.len > 0) {
                return self.free_list.pop();
            }
            var i: usize = self.cellar_start;
            while (i < self.capacity) : (i += 1) {
                if (self.buckets[i] == null) return i;
            }
            i = 0;
            while (i < self.cellar_start) : (i += 1) {
                if (self.buckets[i] == null) return i;
            }
            return null;
        }

        pub fn get(self: *const Self, key: K) ?V {
            if (self.cellar_start == 0) return null;
            const index = self.hash(key) % self.cellar_start;
            var current_idx: ?usize = index;
            while (current_idx) |idx| {
                if (self.buckets[idx]) |entry| {
                    if (self.eql(entry.key, key)) return entry.value;
                    current_idx = entry.next_index;
                } else return null;
            }
            return null;
        }

        pub fn getPtr(self: *Self, key: K) ?*V {
            if (self.cellar_start == 0) return null;
            const index = self.hash(key) % self.cellar_start;
            var current_idx: ?usize = index;
            while (current_idx) |idx| {
                if (self.buckets[idx]) |*entry| {
                    if (self.eql(entry.key, key)) return &entry.value;
                    current_idx = entry.next_index;
                } else return null;
            }
            return null;
        }

        pub fn contains(self: *const Self, key: K) bool {
            if (self.cellar_start == 0) return false;
            const index = self.hash(key) % self.cellar_start;
            var current_idx: ?usize = index;
            while (current_idx) |idx| {
                if (self.buckets[idx]) |entry| {
                    if (self.eql(entry.key, key)) return true;
                    current_idx = entry.next_index;
                } else return false;
            }
            return false;
        }

        pub fn remove(self: *Self, key: K) bool {
            if (self.cellar_start == 0) return false;
            const index = self.hash(key) % self.cellar_start;
            if (self.buckets[index] == null) return false;

            if (self.eql(self.buckets[index].?.key, key)) {
                if (self.buckets[index].?.next_index) |next| {
                    var moved = self.buckets[next].?;
                    moved.is_primary = true;
                    self.buckets[index] = moved;
                    self.buckets[next] = null;
                    self.free_list.append(next) catch {};
                } else {
                    self.buckets[index] = null;
                }
                self.size = satSubUsize(self.size, 1);
                self.updateLoadFactor();
                return true;
            }

            var prev_idx = index;
            var current_idx_opt = self.buckets[index].?.next_index;
            while (current_idx_opt) |idx| {
                if (self.buckets[idx]) |entry| {
                    if (self.eql(entry.key, key)) {
                        self.buckets[prev_idx].?.next_index = entry.next_index;
                        self.buckets[idx] = null;
                        self.free_list.append(idx) catch {};
                        self.size = satSubUsize(self.size, 1);
                        self.updateLoadFactor();
                        return true;
                    }
                    prev_idx = idx;
                    current_idx_opt = entry.next_index;
                } else break;
            }
            return false;
        }

        pub fn resize(self: *Self, new_capacity: usize) PutError!void {
            const min_required = satAddUsize(self.size, 1);
            const target = @max(new_capacity, satAddUsize(min_required, min_required));
            const capacity = @max(target, 8);

            const new_buckets = try self.allocator.alloc(?Entry, capacity);
            errdefer self.allocator.free(new_buckets);
            @memset(new_buckets, null);

            const cellar_size_f = @as(f64, @floatFromInt(capacity)) * CELLAR_RATIO;
            const cellar_size = @as(usize, @intFromFloat(@floor(cellar_size_f)));
            const cellar_actual = @max(cellar_size, 1);
            const new_cellar_start = if (cellar_actual >= capacity) capacity / 2 else capacity - cellar_actual;

            const old_buckets = self.buckets;
            const old_capacity = self.capacity;
            const old_cellar_start = self.cellar_start;
            const old_size = self.size;
            var old_free = self.free_list;

            self.buckets = new_buckets;
            self.capacity = capacity;
            self.cellar_start = new_cellar_start;
            self.size = 0;
            self.free_list = ArrayList(usize).init(self.allocator);
            self.load_factor = 0.0;

            var rebuilt = false;
            defer if (!rebuilt) {
                self.free_list.deinit();
                self.allocator.free(self.buckets);
                self.buckets = old_buckets;
                self.capacity = old_capacity;
                self.cellar_start = old_cellar_start;
                self.size = old_size;
                self.free_list = old_free;
            };

            var i: usize = 0;
            while (i < old_capacity) : (i += 1) {
                if (old_buckets[i]) |entry| {
                    try self.putInternal(entry.key, entry.value);
                }
            }

            rebuilt = true;
            old_free.deinit();
            self.allocator.free(old_buckets);
        }

        fn updateLoadFactor(self: *Self) void {
            if (self.capacity == 0) {
                self.load_factor = 0.0;
                return;
            }
            self.load_factor = @as(f64, @floatFromInt(self.size)) / @as(f64, @floatFromInt(self.capacity));
        }

        pub fn getLoadFactor(self: *const Self) f64 {
            return self.load_factor;
        }

        pub fn count(self: *const Self) usize {
            return self.size;
        }

        pub fn getCapacity(self: *const Self) usize {
            return self.capacity;
        }

        pub fn clear(self: *Self) void {
            @memset(self.buckets, null);
            self.size = 0;
            self.load_factor = 0.0;
            self.free_list.clearAndFree();
        }
    };
}

pub const LRUCache = struct {
    const Entry = struct {
        key: []u8,
        value: []u8,
        prev: ?*Entry = null,
        next: ?*Entry = null,
    };

    map: StringHashMap(*Entry),
    head: ?*Entry,
    tail: ?*Entry,
    capacity: usize,
    current_size: usize,
    max_memory: usize,
    current_memory: usize,
    hits: usize,
    misses: usize,
    evictions: usize,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, capacity: usize, max_memory: usize) !Self {
        if (capacity == 0) return FNDSError.InvalidCapacity;
        if (max_memory == 0) return FNDSError.InvalidCapacity;
        return Self{
            .map = StringHashMap(*Entry).init(allocator),
            .head = null,
            .tail = null,
            .capacity = capacity,
            .current_size = 0,
            .max_memory = max_memory,
            .current_memory = 0,
            .hits = 0,
            .misses = 0,
            .evictions = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.map.iterator();
        while (iter.next()) |entry| {
            const e = entry.value_ptr.*;
            self.allocator.free(e.key);
            self.allocator.free(e.value);
            self.allocator.destroy(e);
        }
        self.map.deinit();
        self.head = null;
        self.tail = null;
        self.current_size = 0;
        self.current_memory = 0;
    }

    fn unlink(self: *Self, node: *Entry) void {
        if (node.prev) |prev| {
            prev.next = node.next;
        } else {
            self.head = node.next;
        }
        if (node.next) |next| {
            next.prev = node.prev;
        } else {
            self.tail = node.prev;
        }
        node.prev = null;
        node.next = null;
    }

    fn insertHead(self: *Self, node: *Entry) void {
        node.prev = null;
        node.next = self.head;
        if (self.head) |h| h.prev = node;
        self.head = node;
        if (self.tail == null) self.tail = node;
    }

    pub fn get(self: *Self, key: []const u8) ?[]const u8 {
        if (self.map.get(key)) |node| {
            self.unlink(node);
            self.insertHead(node);
            self.hits = satAddUsize(self.hits, 1);
            return node.value;
        }
        self.misses = satAddUsize(self.misses, 1);
        return null;
    }

    pub fn put(self: *Self, key: []const u8, value: []const u8) !void {
        const entry_size = satAddUsize(key.len, value.len);
        if (entry_size > self.max_memory) return FNDSError.InvalidArgument;

        if (self.map.get(key)) |node| {
            const new_value = try self.allocator.dupe(u8, value);
            errdefer self.allocator.free(new_value);

            const old_total = satAddUsize(node.key.len, node.value.len);
            const new_total = satAddUsize(node.key.len, value.len);

            while (self.current_memory - old_total + new_total > self.max_memory) {
                if (self.tail == node or self.tail == null) break;
                if (!self.evict()) break;
            }

            self.allocator.free(node.value);
            node.value = new_value;
            self.current_memory = self.current_memory - old_total + new_total;

            self.unlink(node);
            self.insertHead(node);
            return;
        }

        while (self.current_size >= self.capacity or satAddUsize(self.current_memory, entry_size) > self.max_memory) {
            if (self.current_size == 0) break;
            if (!self.evict()) break;
        }

        const node = try self.allocator.create(Entry);
        errdefer self.allocator.destroy(node);

        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        const val_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(val_copy);

        node.* = Entry{
            .key = key_copy,
            .value = val_copy,
            .prev = null,
            .next = null,
        };

        try self.map.put(key_copy, node);
        self.current_memory = satAddUsize(self.current_memory, entry_size);
        self.insertHead(node);
        self.current_size = satAddUsize(self.current_size, 1);
    }

    fn evict(self: *Self) bool {
        const lru = self.tail orelse return false;
        self.unlink(lru);
        const total = satAddUsize(lru.key.len, lru.value.len);
        const removed = self.map.remove(lru.key);
        if (!removed) {
            self.allocator.free(lru.key);
            self.allocator.free(lru.value);
            self.allocator.destroy(lru);
            return false;
        }
        self.allocator.free(lru.key);
        self.allocator.free(lru.value);
        self.allocator.destroy(lru);
        self.current_size = satSubUsize(self.current_size, 1);
        self.current_memory = satSubUsize(self.current_memory, total);
        self.evictions = satAddUsize(self.evictions, 1);
        return true;
    }

    pub fn remove(self: *Self, key: []const u8) bool {
        if (self.map.fetchRemove(key)) |kv| {
            const node = kv.value;
            self.unlink(node);
            const total = satAddUsize(node.key.len, node.value.len);
            self.allocator.free(node.key);
            self.allocator.free(node.value);
            self.allocator.destroy(node);
            self.current_size = satSubUsize(self.current_size, 1);
            self.current_memory = satSubUsize(self.current_memory, total);
            return true;
        }
        return false;
    }

    pub fn clear(self: *Self) void {
        var iter = self.map.iterator();
        while (iter.next()) |entry| {
            const node = entry.value_ptr.*;
            self.allocator.free(node.key);
            self.allocator.free(node.value);
            self.allocator.destroy(node);
        }
        self.map.clearRetainingCapacity();
        self.head = null;
        self.tail = null;
        self.current_size = 0;
        self.current_memory = 0;
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }

    pub fn getHitRatio(self: *const Self) f64 {
        const total = satAddUsize(self.hits, self.misses);
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
    }

    pub fn getSize(self: *const Self) usize {
        return self.current_size;
    }

    pub fn getMemoryUsage(self: *const Self) usize {
        return self.current_memory;
    }
};

pub const TreeIdContext = struct {
    pub fn hash(_: @This(), key: [32]u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(key[0..]);
        return hasher.final();
    }

    pub fn eql(_: @This(), a: [32]u8, b: [32]u8) bool {
        return std.mem.eql(u8, a[0..], b[0..]);
    }
};

pub const FNDSManager = struct {
    fractal_trees: std.HashMap([32]u8, FractalTree, TreeIdContext, std.hash_map.default_max_load_percentage),
    indices: StringHashMap(SelfSimilarIndex),
    index_keys_owned: ArrayList([]u8),
    cache: LRUCache,
    statistics: FNDSStatistics,
    allocator: Allocator,
    creation_time: u64,

    const Self = @This();
    const DEFAULT_CACHE_CAPACITY: usize = 1000;
    const DEFAULT_CACHE_MEMORY: usize = 10 * 1024 * 1024;

    pub fn init(allocator: Allocator) !Self {
        return initWithCache(allocator, DEFAULT_CACHE_CAPACITY, DEFAULT_CACHE_MEMORY);
    }

    pub fn initWithCache(allocator: Allocator, cache_capacity: usize, cache_memory: usize) !Self {
        return Self{
            .fractal_trees = std.HashMap([32]u8, FractalTree, TreeIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .indices = StringHashMap(SelfSimilarIndex).init(allocator),
            .index_keys_owned = ArrayList([]u8).init(allocator),
            .cache = try LRUCache.init(allocator, cache_capacity, cache_memory),
            .statistics = FNDSStatistics.init(),
            .allocator = allocator,
            .creation_time = @as(u64, @intCast(std.time.nanoTimestamp())),
        };
    }

    pub fn deinit(self: *Self) void {
        var tree_iter = self.fractal_trees.iterator();
        while (tree_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.fractal_trees.deinit();

        var index_iter = self.indices.iterator();
        while (index_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.indices.deinit();
        for (self.index_keys_owned.items) |k| self.allocator.free(k);
        self.index_keys_owned.deinit();

        self.cache.deinit();
    }

    pub fn createTree(self: *Self, max_depth: usize, branching_factor: usize) ![32]u8 {
        const start_time = @as(u64, @intCast(std.time.nanoTimestamp()));

        var tree = try FractalTree.init(self.allocator, max_depth, branching_factor);
        errdefer tree.deinit();

        var tree_id = tree.tree_id;
        var attempts: usize = 0;
        while (self.fractal_trees.contains(tree_id)) {
            if (attempts > 16) return FNDSError.Overflow;
            Random.bytes(&tree_id);
            tree.tree_id = tree_id;
            attempts += 1;
        }

        try self.fractal_trees.put(tree_id, tree);

        self.statistics.total_trees = satAddUsize(self.statistics.total_trees, 1);
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        self.statistics.last_operation_time_ns = if (now > start_time) now - start_time else 0;

        return tree_id;
    }

    pub fn getTree(self: *Self, tree_id: [32]u8) ?*FractalTree {
        return self.fractal_trees.getPtr(tree_id);
    }

    pub fn getTreeConst(self: *const Self, tree_id: [32]u8) ?*const FractalTree {
        if (self.fractal_trees.getPtr(tree_id)) |p| return p;
        return null;
    }

    pub fn removeTree(self: *Self, tree_id: [32]u8) bool {
        const start_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        if (self.fractal_trees.fetchRemove(tree_id)) |removed| {
            var tree = removed.value;
            self.statistics.total_nodes_across_trees = satSubUsize(self.statistics.total_nodes_across_trees, tree.total_nodes);
            tree.deinit();
            self.statistics.total_trees = satSubUsize(self.statistics.total_trees, 1);
            const now = @as(u64, @intCast(std.time.nanoTimestamp()));
            self.statistics.last_operation_time_ns = if (now > start_time) now - start_time else 0;
            return true;
        }
        return false;
    }

    pub fn createIndex(self: *Self, index_id: []const u8) !void {
        const start_time = @as(u64, @intCast(std.time.nanoTimestamp()));

        if (self.indices.contains(index_id)) return FNDSError.InvalidArgument;

        const id_copy = try self.allocator.dupe(u8, index_id);
        errdefer self.allocator.free(id_copy);
        try self.index_keys_owned.append(id_copy);
        errdefer _ = self.index_keys_owned.pop();

        const index = SelfSimilarIndex.init(self.allocator);
        try self.indices.put(id_copy, index);

        self.statistics.total_indices = satAddUsize(self.statistics.total_indices, 1);
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        self.statistics.last_operation_time_ns = if (now > start_time) now - start_time else 0;
    }

    pub fn getIndex(self: *Self, index_id: []const u8) ?*SelfSimilarIndex {
        return self.indices.getPtr(index_id);
    }

    pub fn getIndexConst(self: *const Self, index_id: []const u8) ?*const SelfSimilarIndex {
        if (self.indices.getPtr(index_id)) |p| return p;
        return null;
    }

    pub fn removeIndex(self: *Self, index_id: []const u8) bool {
        const start_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        const entry = self.indices.getEntry(index_id) orelse return false;
        const stored_key = entry.key_ptr.*;
        var index = entry.value_ptr.*;
        _ = self.indices.remove(index_id);

        self.statistics.total_patterns_indexed = satSubUsize(self.statistics.total_patterns_indexed, index.pattern_count);
        self.statistics.total_pattern_locations_indexed = satSubUsize(self.statistics.total_pattern_locations_indexed, index.total_locations);
        index.deinit();

        for (self.index_keys_owned.items, 0..) |k, i| {
            if (k.ptr == stored_key.ptr and k.len == stored_key.len) {
                _ = self.index_keys_owned.swapRemove(i);
                self.allocator.free(k);
                break;
            }
        }
        self.statistics.total_indices = satSubUsize(self.statistics.total_indices, 1);
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        self.statistics.last_operation_time_ns = if (now > start_time) now - start_time else 0;
        return true;
    }

    pub fn cacheGet(self: *Self, key: []const u8) ?[]const u8 {
        if (self.cache.get(key)) |value| {
            self.statistics.recordCacheHit();
            return value;
        }
        self.statistics.recordCacheMiss();
        return null;
    }

    pub fn cachePut(self: *Self, key: []const u8, value: []const u8) !void {
        try self.cache.put(key, value);
    }

    pub fn cacheRemove(self: *Self, key: []const u8) bool {
        return self.cache.remove(key);
    }

    pub fn cacheClear(self: *Self) void {
        self.cache.clear();
    }

    pub fn getStatistics(self: *Self) FNDSStatistics {
        self.updateStatistics();
        return self.statistics;
    }

    fn updateStatistics(self: *Self) void {
        self.statistics.total_trees = self.fractal_trees.count();
        self.statistics.total_indices = self.indices.count();
        self.statistics.updateCacheHitRatio();

        var total_nodes: usize = 0;
        var total_patterns: usize = 0;
        var total_locations: usize = 0;
        var depths = ArrayList(usize).init(self.allocator);
        defer depths.deinit();

        var tree_iter = self.fractal_trees.iterator();
        while (tree_iter.next()) |entry| {
            const tree = entry.value_ptr;
            total_nodes = satAddUsize(total_nodes, tree.total_nodes);
            depths.append(tree.getDepth()) catch {};
        }

        var index_iter = self.indices.iterator();
        while (index_iter.next()) |entry| {
            total_patterns = satAddUsize(total_patterns, entry.value_ptr.pattern_count);
            total_locations = satAddUsize(total_locations, entry.value_ptr.total_locations);
        }

        self.statistics.total_nodes_across_trees = total_nodes;
        self.statistics.total_patterns_indexed = total_patterns;
        self.statistics.total_pattern_locations_indexed = total_locations;
        self.statistics.updateAverageTreeDepth(depths.items);

        var memory_estimate: usize = 0;
        memory_estimate = satAddUsize(memory_estimate, self.cache.current_memory);
        memory_estimate = satAddUsize(memory_estimate, self.fractal_trees.count() * @sizeOf(FractalTree));
        memory_estimate = satAddUsize(memory_estimate, self.indices.count() * @sizeOf(SelfSimilarIndex));
        self.statistics.memory_used = memory_estimate;
    }

    pub fn getTreeCount(self: *const Self) usize {
        return self.fractal_trees.count();
    }

    pub fn getIndexCount(self: *const Self) usize {
        return self.indices.count();
    }

    pub fn getCacheHitRatio(self: *const Self) f64 {
        return self.cache.getHitRatio();
    }

    pub fn insertIntoTree(self: *Self, tree_id: [32]u8, node_id: []const u8, data: []const u8, level: usize) !bool {
        const start_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        const tree = self.fractal_trees.getPtr(tree_id) orelse return FNDSError.TreeNotFound;
        const existed = tree.searchInternal(tree.root, node_id) != null;
        const result = try tree.insert(node_id, data, level);
        if (result and !existed) {
            self.statistics.total_nodes_across_trees = satAddUsize(self.statistics.total_nodes_across_trees, 1);
        }
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        self.statistics.last_operation_time_ns = if (now > start_time) now - start_time else 0;
        return result;
    }

    pub fn searchInTree(self: *Self, tree_id: [32]u8, node_id: []const u8) ?*FractalNodeData {
        const tree = self.fractal_trees.getPtr(tree_id) orelse return null;
        return tree.search(node_id);
    }

    pub fn addPatternToIndex(self: *Self, index_id: []const u8, pattern: []const u8, location: PatternLocation) !void {
        const start_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        const index = self.indices.getPtr(index_id) orelse {
            var loc_mut = location;
            loc_mut.deinit();
            return FNDSError.IndexNotFound;
        };
        const before_patterns = index.pattern_count;
        try index.addPattern(pattern, location);
        if (index.pattern_count > before_patterns) {
            self.statistics.total_patterns_indexed = satAddUsize(self.statistics.total_patterns_indexed, 1);
        }
        self.statistics.total_pattern_locations_indexed = satAddUsize(self.statistics.total_pattern_locations_indexed, 1);
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        self.statistics.last_operation_time_ns = if (now > start_time) now - start_time else 0;
    }

    pub fn findPatternInIndex(self: *Self, index_id: []const u8, pattern: []const u8) []const PatternLocation {
        if (self.indices.getPtr(index_id)) |index| {
            return index.findPattern(pattern);
        }
        return &[_]PatternLocation{};
    }

    pub fn computeGlobalFractalDimension(self: *Self) f64 {
        var sum: f64 = 0.0;
        var count: usize = 0;
        var tree_iter = self.fractal_trees.iterator();
        while (tree_iter.next()) |entry| {
            const d = entry.value_ptr.computeFractalDimension();
            if (isFiniteF64(d)) {
                sum += d;
                count += 1;
            }
        }
        var index_iter = self.indices.iterator();
        while (index_iter.next()) |entry| {
            const d = entry.value_ptr.computeFractalDimension();
            if (isFiniteF64(d)) {
                sum += d;
                count += 1;
            }
        }
        if (count == 0) return 0.0;
        return sum / @as(f64, @floatFromInt(count));
    }
};

test "FractalLevel basic operations" {
    const allocator = std.testing.allocator;

    var level = try FractalLevel.init(allocator, 0, 1.0);
    defer level.deinit();

    const node1 = try FractalNodeData.init(allocator, "node1", "data1", 1.0, 1.0);
    try level.addNode(node1);

    try std.testing.expect(level.node_count == 1);
    try std.testing.expect(level.getNode("node1") != null);

    const edge = try FractalEdgeData.init(allocator, "node1", "node2", 0.5, 1.0, .hierarchical);
    try level.addEdge(edge);

    try std.testing.expect(level.edge_count == 1);

    try std.testing.expect(level.removeNode("node1") == true);
    try std.testing.expect(level.node_count == 0);
    try std.testing.expect(level.edge_count == 0);
}

test "FractalTree insert and search" {
    const allocator = std.testing.allocator;

    var tree = try FractalTree.init(allocator, 5, 4);
    defer tree.deinit();

    const inserted = try tree.insert("test_node", "test_data", 2);
    try std.testing.expect(inserted == true);
    try std.testing.expect(tree.total_nodes == 1);

    const found = tree.search("test_node");
    try std.testing.expect(found != null);
    try std.testing.expect(std.mem.eql(u8, found.?.data, "test_data"));

    const deleted = tree.delete("test_node");
    try std.testing.expect(deleted == true);
    try std.testing.expect(tree.total_nodes == 0);
}

test "SelfSimilarIndex pattern operations" {
    const allocator = std.testing.allocator;

    var index = SelfSimilarIndex.init(allocator);
    defer index.deinit();

    var tree_id: [32]u8 = undefined;
    @memset(&tree_id, 0);

    const location = try PatternLocation.init(allocator, tree_id, 0, "node1", 0, 5, 1.0);
    try index.addPattern("test_pattern", location);

    try std.testing.expect(index.pattern_count == 1);

    const found = index.findPattern("test_pattern");
    try std.testing.expect(found.len == 1);
}

test "CoalescedHashMap operations" {
    const allocator = std.testing.allocator;

    var map = try CoalescedHashMap(u64, u64).init(allocator);
    defer map.deinit();

    try map.put(1, 100);
    try map.put(2, 200);
    try map.put(3, 300);

    try std.testing.expect(map.count() == 3);
    try std.testing.expect(map.get(1).? == 100);
    try std.testing.expect(map.get(2).? == 200);
    try std.testing.expect(map.get(3).? == 300);

    try std.testing.expect(map.remove(2) == true);
    try std.testing.expect(map.count() == 2);
    try std.testing.expect(map.get(2) == null);
}

test "LRUCache operations" {
    const allocator = std.testing.allocator;

    var cache = try LRUCache.init(allocator, 3, 1024);
    defer cache.deinit();

    try cache.put("key1", "value1");
    try cache.put("key2", "value2");
    try cache.put("key3", "value3");

    try std.testing.expect(cache.getSize() == 3);

    const v1 = cache.get("key1");
    try std.testing.expect(v1 != null);
    try std.testing.expect(std.mem.eql(u8, v1.?, "value1"));

    try cache.put("key4", "value4");
    try std.testing.expect(cache.getSize() == 3);
    try std.testing.expect(cache.get("key2") == null);
}

test "FNDSManager full workflow" {
    const allocator = std.testing.allocator;

    var manager = try FNDSManager.init(allocator);
    defer manager.deinit();

    const tree_id = try manager.createTree(5, 4);
    try std.testing.expect(manager.getTreeCount() == 1);

    _ = try manager.insertIntoTree(tree_id, "node1", "data1", 1);

    try manager.createIndex("test_index");
    try std.testing.expect(manager.getIndexCount() == 1);

    try manager.cachePut("cache_key", "cache_value");
    const cached = manager.cacheGet("cache_key");
    try std.testing.expect(cached != null);

    const stats = manager.getStatistics();
    try std.testing.expect(stats.total_trees == 1);
    try std.testing.expect(stats.total_indices == 1);
    try std.testing.expect(stats.cache_hits == 1);
}

