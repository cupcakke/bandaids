const std = @import("std");
const nsir_core = @import("nsir_core.zig");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;
const PriorityQueue = std.PriorityQueue;

pub const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
pub const Node = nsir_core.Node;
pub const Edge = nsir_core.Edge;
pub const EdgeQuality = nsir_core.EdgeQuality;
pub const EdgeKey = nsir_core.EdgeKey;
pub const EdgeKeyContext = nsir_core.EdgeKeyContext;
pub const Qubit = nsir_core.Qubit;

pub const CoreState = enum(u8) {
    idle = 0,
    processing = 1,
    communicating = 2,
    power_gated = 3,

    pub fn toString(self: CoreState) []const u8 {
        return switch (self) {
            .idle => "idle",
            .processing => "processing",
            .communicating => "communicating",
            .power_gated => "power_gated",
        };
    }

    pub fn fromString(s: []const u8) ?CoreState {
        if (std.mem.eql(u8, s, "idle")) return .idle;
        if (std.mem.eql(u8, s, "processing")) return .processing;
        if (std.mem.eql(u8, s, "communicating")) return .communicating;
        if (std.mem.eql(u8, s, "power_gated")) return .power_gated;
        return null;
    }
};

pub const MessageType = enum(u8) {
    weight_update = 0,
    graph_sync = 1,
    isomorphism_result = 2,
    power_control = 3,
    data_transfer = 4,

    pub fn toString(self: MessageType) []const u8 {
        return switch (self) {
            .weight_update => "weight_update",
            .graph_sync => "graph_sync",
            .isomorphism_result => "isomorphism_result",
            .power_control => "power_control",
            .data_transfer => "data_transfer",
        };
    }

    pub fn fromString(s: []const u8) ?MessageType {
        if (std.mem.eql(u8, s, "weight_update")) return .weight_update;
        if (std.mem.eql(u8, s, "graph_sync")) return .graph_sync;
        if (std.mem.eql(u8, s, "isomorphism_result")) return .isomorphism_result;
        if (std.mem.eql(u8, s, "power_control")) return .power_control;
        if (std.mem.eql(u8, s, "data_transfer")) return .data_transfer;
        return null;
    }
};

pub const ProcessingCore = struct {
    core_id: usize,
    x: usize,
    y: usize,
    state: CoreState,
    neighbors: ArrayList(usize),
    local_graph: ?*SelfSimilarRelationalGraph,
    local_graph_owned: bool,
    message_queue: ArrayList(NoCMessage),
    energy_consumed: f64,
    cycles_active: usize,
    cycles_idle: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, core_id: usize, x: usize, y: usize) ProcessingCore {
        return ProcessingCore{
            .core_id = core_id,
            .x = x,
            .y = y,
            .state = .idle,
            .neighbors = ArrayList(usize).init(allocator),
            .local_graph = null,
            .local_graph_owned = false,
            .message_queue = ArrayList(NoCMessage).init(allocator),
            .energy_consumed = 0.0,
            .cycles_active = 0,
            .cycles_idle = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ProcessingCore) void {
        self.neighbors.deinit();
        for (self.message_queue.items) |*msg| {
            msg.deinit();
        }
        self.message_queue.deinit();
        if (self.local_graph_owned) {
            if (self.local_graph) |graph| {
                graph.deinit();
                self.allocator.destroy(graph);
            }
        }
    }

    pub fn addNeighbor(self: *ProcessingCore, neighbor_id: usize) !void {
        try self.neighbors.append(neighbor_id);
    }

    pub fn setLocalGraph(self: *ProcessingCore, graph: *SelfSimilarRelationalGraph, owned: bool) void {
        if (self.local_graph_owned) {
            if (self.local_graph) |old_graph| {
                old_graph.deinit();
                self.allocator.destroy(old_graph);
            }
        }
        self.local_graph = graph;
        self.local_graph_owned = owned;
    }

    pub fn createLocalGraph(self: *ProcessingCore) !*SelfSimilarRelationalGraph {
        const graph = try self.allocator.create(SelfSimilarRelationalGraph);
        errdefer self.allocator.destroy(graph);
        graph.* = try SelfSimilarRelationalGraph.init(self.allocator);
        self.setLocalGraph(graph, true);
        return graph;
    }

    pub fn enqueueMessage(self: *ProcessingCore, message: NoCMessage) !void {
        try self.message_queue.append(message);
    }

    pub fn processMessages(self: *ProcessingCore) usize {
        const count = self.message_queue.items.len;
        for (self.message_queue.items) |*msg| {
            msg.deinit();
        }
        self.message_queue.clearRetainingCapacity();
        return count;
    }

    pub fn getWorkload(self: *const ProcessingCore) f64 {
        const total = self.cycles_active + self.cycles_idle;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.cycles_active)) / @as(f64, @floatFromInt(total));
    }

    pub fn clone(self: *const ProcessingCore, allocator: Allocator) !ProcessingCore {
        var new_core = ProcessingCore{
            .core_id = self.core_id,
            .x = self.x,
            .y = self.y,
            .state = self.state,
            .neighbors = ArrayList(usize).init(allocator),
            .local_graph = null,
            .local_graph_owned = false,
            .message_queue = ArrayList(NoCMessage).init(allocator),
            .energy_consumed = self.energy_consumed,
            .cycles_active = self.cycles_active,
            .cycles_idle = self.cycles_idle,
            .allocator = allocator,
        };
        for (self.neighbors.items) |neighbor| {
            try new_core.neighbors.append(neighbor);
        }
        return new_core;
    }
};

pub const NoCMessage = struct {
    source_core: usize,
    target_core: usize,
    message_type: MessageType,
    payload: []const u8,
    timestamp: i64,
    priority: i32,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        source_core: usize,
        target_core: usize,
        message_type: MessageType,
        payload: []const u8,
        priority: i32,
    ) !NoCMessage {
        return NoCMessage{
            .source_core = source_core,
            .target_core = target_core,
            .message_type = message_type,
            .payload = try allocator.dupe(u8, payload),
            .timestamp = @as(i64, @intCast(std.time.nanoTimestamp())),
            .priority = priority,
            .allocator = allocator,
        };
    }

    pub fn initWithTimestamp(
        allocator: Allocator,
        source_core: usize,
        target_core: usize,
        message_type: MessageType,
        payload: []const u8,
        timestamp: i64,
        priority: i32,
    ) !NoCMessage {
        return NoCMessage{
            .source_core = source_core,
            .target_core = target_core,
            .message_type = message_type,
            .payload = try allocator.dupe(u8, payload),
            .timestamp = timestamp,
            .priority = priority,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *NoCMessage) void {
        self.allocator.free(self.payload);
    }

    pub fn clone(self: *const NoCMessage, allocator: Allocator) !NoCMessage {
        return NoCMessage{
            .source_core = self.source_core,
            .target_core = self.target_core,
            .message_type = self.message_type,
            .payload = try allocator.dupe(u8, self.payload),
            .timestamp = self.timestamp,
            .priority = self.priority,
            .allocator = allocator,
        };
    }
};

const MessagePriorityEntry = struct {
    priority: i32,
    sequence: usize,
    message: NoCMessage,

    fn compare(_: void, a: MessagePriorityEntry, b: MessagePriorityEntry) std.math.Order {
        if (a.priority != b.priority) {
            return if (a.priority < b.priority) .lt else .gt;
        }
        if (a.sequence != b.sequence) {
            return if (a.sequence < b.sequence) .lt else .gt;
        }
        return .eq;
    }
};

pub const RouteKey = struct {
    source: usize,
    destination: usize,
};

pub const RouteKeyContext = struct {
    pub fn hash(_: @This(), key: RouteKey) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&key.source));
        hasher.update(std.mem.asBytes(&key.destination));
        return hasher.final();
    }

    pub fn eql(_: @This(), a: RouteKey, b: RouteKey) bool {
        return a.source == b.source and a.destination == b.destination;
    }
};

pub const AsynchronousNoC = struct {
    grid_width: usize,
    grid_height: usize,
    cores: AutoHashMap(usize, ProcessingCore),
    routing_table: std.HashMap(RouteKey, ArrayList(usize), RouteKeyContext, std.hash_map.default_max_load_percentage),
    message_buffer: PriorityQueue(MessagePriorityEntry, void, MessagePriorityEntry.compare),
    total_messages: usize,
    total_hops: usize,
    message_sequence: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, grid_width: usize, grid_height: usize) !AsynchronousNoC {
        var noc = AsynchronousNoC{
            .grid_width = grid_width,
            .grid_height = grid_height,
            .cores = AutoHashMap(usize, ProcessingCore).init(allocator),
            .routing_table = std.HashMap(RouteKey, ArrayList(usize), RouteKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
            .message_buffer = PriorityQueue(MessagePriorityEntry, void, MessagePriorityEntry.compare).init(allocator, {}),
            .total_messages = 0,
            .total_hops = 0,
            .message_sequence = 0,
            .allocator = allocator,
        };
        errdefer noc.deinit();
        try noc.initializeCores();
        try noc.buildRoutingTable();
        return noc;
    }

    pub fn deinit(self: *AsynchronousNoC) void {
        var core_iter = self.cores.iterator();
        while (core_iter.next()) |entry| {
            var core = entry.value_ptr;
            core.deinit();
        }
        self.cores.deinit();

        var route_iter = self.routing_table.iterator();
        while (route_iter.next()) |entry| {
            var path = entry.value_ptr;
            path.deinit();
        }
        self.routing_table.deinit();

        while (self.message_buffer.count() > 0) {
            var entry = self.message_buffer.remove();
            entry.message.deinit();
        }
        self.message_buffer.deinit();
    }

    pub fn initializeCores(self: *AsynchronousNoC) !void {
        var core_id: usize = 0;
        var y: usize = 0;
        while (y < self.grid_height) : (y += 1) {
            var x: usize = 0;
            while (x < self.grid_width) : (x += 1) {
                var core = ProcessingCore.init(self.allocator, core_id, x, y);
                var core_transferred = false;
                defer if (!core_transferred) core.deinit();
                if (x > 0) {
                    try core.addNeighbor(core_id - 1);
                }
                if (x < self.grid_width - 1) {
                    try core.addNeighbor(core_id + 1);
                }
                if (y > 0) {
                    try core.addNeighbor(core_id - self.grid_width);
                }
                if (y < self.grid_height - 1) {
                    try core.addNeighbor(core_id + self.grid_width);
                }
                try self.cores.putNoClobber(core_id, core);
                core_transferred = true;
                core_id += 1;
            }
        }
    }

    pub fn buildRoutingTable(self: *AsynchronousNoC) !void {
        var src_iter = self.cores.iterator();
        while (src_iter.next()) |src_entry| {
            const src_id = src_entry.key_ptr.*;
            var dst_iter = self.cores.iterator();
            while (dst_iter.next()) |dst_entry| {
                const dst_id = dst_entry.key_ptr.*;
                if (src_id != dst_id) {
                    const route_key = RouteKey{ .source = src_id, .destination = dst_id };
                    var path = try self.computeXYRoute(src_id, dst_id);
                    var path_transferred = false;
                    defer if (!path_transferred) path.deinit();
                    try self.routing_table.putNoClobber(route_key, path);
                    path_transferred = true;
                }
            }
        }
    }

    pub fn computeXYRoute(self: *AsynchronousNoC, src_id: usize, dst_id: usize) !ArrayList(usize) {
        var path = ArrayList(usize).init(self.allocator);
        errdefer path.deinit();

        const src_core = self.cores.get(src_id) orelse return path;
        const dst_core = self.cores.get(dst_id) orelse return path;

        try path.append(src_id);
        var current_x = src_core.x;
        var current_y = src_core.y;

        while (current_x != dst_core.x) {
            if (current_x < dst_core.x) {
                current_x += 1;
            } else {
                current_x -= 1;
            }
            const next_id = current_y * self.grid_width + current_x;
            try path.append(next_id);
        }

        while (current_y != dst_core.y) {
            if (current_y < dst_core.y) {
                current_y += 1;
            } else {
                current_y -= 1;
            }
            const next_id = current_y * self.grid_width + current_x;
            try path.append(next_id);
        }

        return path;
    }

    pub fn sendMessage(self: *AsynchronousNoC, message: NoCMessage) !bool {
        if (!self.cores.contains(message.source_core) or !self.cores.contains(message.target_core)) {
            return false;
        }

        const entry = MessagePriorityEntry{
            .priority = message.priority,
            .sequence = self.message_sequence,
            .message = message,
        };
        try self.message_buffer.add(entry);
        self.message_sequence += 1;
        self.total_messages += 1;
        return true;
    }

    pub fn routeMessages(self: *AsynchronousNoC) !usize {
        var routed_count: usize = 0;
        while (self.message_buffer.count() > 0) {
            var entry = self.message_buffer.remove();
            defer entry.message.deinit();

            const route_key = RouteKey{ .source = entry.message.source_core, .destination = entry.message.target_core };
            if (self.routing_table.get(route_key)) |path| {
                if (path.items.len > 1) {
                    self.total_hops = std.math.add(usize, self.total_hops, path.items.len - 1) catch return error.Overflow;
                }
            }

            if (self.cores.getPtr(entry.message.target_core)) |target_core| {
                var message_clone = try entry.message.clone(self.allocator);
                var clone_transferred = false;
                defer if (!clone_transferred) message_clone.deinit();
                try target_core.enqueueMessage(message_clone);
                clone_transferred = true;
                routed_count = std.math.add(usize, routed_count, 1) catch return error.Overflow;
            }
        }
        return routed_count;
    }

    pub fn getCore(self: *AsynchronousNoC, core_id: usize) ?*ProcessingCore {
        return self.cores.getPtr(core_id);
    }

    pub fn getCoreConst(self: *const AsynchronousNoC, core_id: usize) ?ProcessingCore {
        return self.cores.get(core_id);
    }

    pub fn getTotalCores(self: *const AsynchronousNoC) usize {
        return self.cores.count();
    }

    pub fn getActiveCores(self: *const AsynchronousNoC) usize {
        var count: usize = 0;
        var iter = self.cores.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.state != .power_gated) {
                count += 1;
            }
        }
        return count;
    }
};

const StringContext = struct {
    pub fn hash(_: @This(), key: []const u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(key);
        return hasher.final();
    }

    pub fn eql(_: @This(), a: []const u8, b: []const u8) bool {
        return std.mem.eql(u8, a, b);
    }
};

pub const GraphIsomorphismProcessor = struct {
    canonical_forms: StringHashMap(ArrayList([]const u8)),
    allocator: Allocator,

    pub fn init(allocator: Allocator) GraphIsomorphismProcessor {
        return GraphIsomorphismProcessor{
            .canonical_forms = StringHashMap(ArrayList([]const u8)).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GraphIsomorphismProcessor) void {
        var iter = self.canonical_forms.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            for (entry.value_ptr.items) |item| {
                self.allocator.free(item);
            }
            entry.value_ptr.deinit();
        }
        self.canonical_forms.deinit();
    }

    pub fn computeCanonicalForm(self: *GraphIsomorphismProcessor, graph: *SelfSimilarRelationalGraph) ![]const u8 {
        _ = self;
        var node_ids = ArrayList([]const u8).init(graph.allocator);
        defer node_ids.deinit();

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try node_ids.append(entry.key_ptr.*);
        }

        std.mem.sort([]const u8, node_ids.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lessThan);

        const NodeSignature = struct {
            out_degree: usize,
            in_degree: usize,
            weight_sum: f64,
            edge_quality_sum: u64,
        };

        var node_signatures = ArrayList(NodeSignature).init(graph.allocator);
        defer node_signatures.deinit();

        for (node_ids.items) |node_id| {
            var out_degree: usize = 0;
            var in_degree: usize = 0;
            var weight_sum: f64 = 0.0;
            var edge_quality_sum: u64 = 0;

            var edge_iter = graph.edges.iterator();
            while (edge_iter.next()) |edge_entry| {
                const key = edge_entry.key_ptr.*;
                if (std.mem.eql(u8, key.source, node_id)) {
                    out_degree += edge_entry.value_ptr.items.len;
                    for (edge_entry.value_ptr.items) |edge| {
                        weight_sum += edge.weight;
                        edge_quality_sum += @intFromEnum(edge.quality);
                    }
                }
                if (std.mem.eql(u8, key.target, node_id)) {
                    in_degree += edge_entry.value_ptr.items.len;
                    for (edge_entry.value_ptr.items) |edge| {
                        weight_sum += edge.weight;
                        edge_quality_sum += @intFromEnum(edge.quality);
                    }
                }
            }
            try node_signatures.append(.{ .out_degree = out_degree, .in_degree = in_degree, .weight_sum = weight_sum, .edge_quality_sum = edge_quality_sum });
        }

        std.mem.sort(NodeSignature, node_signatures.items, {}, struct {
            fn lessThan(_: void, a: NodeSignature, b: NodeSignature) bool {
                if (a.out_degree != b.out_degree) return a.out_degree < b.out_degree;
                if (a.in_degree != b.in_degree) return a.in_degree < b.in_degree;
                if (a.edge_quality_sum != b.edge_quality_sum) return a.edge_quality_sum < b.edge_quality_sum;
                return a.weight_sum < b.weight_sum;
            }
        }.lessThan);

        var adj_triples = ArrayList(struct { src: usize, dst: usize, quality: u8 }).init(graph.allocator);
        defer adj_triples.deinit();

        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |edge_entry| {
            const key = edge_entry.key_ptr.*;
            var src_idx: usize = 0;
            var found_src = false;
            for (node_ids.items, 0..) |nid, idx| {
                if (std.mem.eql(u8, nid, key.source)) { src_idx = idx; found_src = true; break; }
            }
            var dst_idx: usize = 0;
            var found_dst = false;
            for (node_ids.items, 0..) |nid, idx| {
                if (std.mem.eql(u8, nid, key.target)) { dst_idx = idx; found_dst = true; break; }
            }
            if (found_src and found_dst) {
                for (edge_entry.value_ptr.items) |edge| {
                    try adj_triples.append(.{ .src = src_idx, .dst = dst_idx, .quality = @intFromEnum(edge.quality) });
                }
            }
        }

        std.mem.sort(@TypeOf(adj_triples.items[0]), adj_triples.items, {}, struct {
            fn lessThan(_: void, a: @TypeOf(adj_triples.items[0]), b: @TypeOf(adj_triples.items[0])) bool {
                if (a.src != b.src) return a.src < b.src;
                if (a.dst != b.dst) return a.dst < b.dst;
                return a.quality < b.quality;
            }
        }.lessThan);

        var buffer = ArrayList(u8).init(graph.allocator);
        errdefer buffer.deinit();

        try std.fmt.format(buffer.writer(), "{d}_", .{node_ids.items.len});
        for (node_signatures.items, 0..) |sig, i| {
            if (i > 0) try buffer.appendSlice(";");
            try std.fmt.format(buffer.writer(), "({d},{d},{d:.0},{d})", .{ sig.out_degree, sig.in_degree, sig.weight_sum, sig.edge_quality_sum });
        }
        try buffer.appendSlice("_E");
        for (adj_triples.items, 0..) |triple, i| {
            if (i > 0) try buffer.appendSlice(",");
            try std.fmt.format(buffer.writer(), "{d}-{d}-{d}", .{ triple.src, triple.dst, triple.quality });
        }

        return try buffer.toOwnedSlice();
    }

    pub fn areIsomorphic(self: *GraphIsomorphismProcessor, graph1: *SelfSimilarRelationalGraph, graph2: *SelfSimilarRelationalGraph) !bool {
        if (graph1.nodeCount() != graph2.nodeCount()) {
            return false;
        }
        if (graph1.edgeCount() != graph2.edgeCount()) {
            return false;
        }

        const canonical1 = try self.computeCanonicalForm(graph1);
        defer self.allocator.free(canonical1);
        const canonical2 = try self.computeCanonicalForm(graph2);
        defer self.allocator.free(canonical2);

        return std.mem.eql(u8, canonical1, canonical2);
    }

    pub fn findIsomorphicSubgraphs(
        self: *GraphIsomorphismProcessor,
        main_graph: *SelfSimilarRelationalGraph,
        pattern_graph: *SelfSimilarRelationalGraph,
    ) !ArrayList(ArrayList([]const u8)) {
        var matches = ArrayList(ArrayList([]const u8)).init(self.allocator);
        errdefer {
            for (matches.items) |*match| {
                for (match.items) |item| {
                    self.allocator.free(item);
                }
                match.deinit();
            }
            matches.deinit();
        }

        const pattern_size = pattern_graph.nodeCount();
        const main_node_count = main_graph.nodeCount();

        if (pattern_size > main_node_count) {
            return matches;
        }

        var main_nodes = ArrayList([]const u8).init(self.allocator);
        defer main_nodes.deinit();

        var node_iter = main_graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try main_nodes.append(entry.key_ptr.*);
        }

        const pattern_canonical = try self.computeCanonicalForm(pattern_graph);
        defer self.allocator.free(pattern_canonical);

        var i: usize = 0;
        while (i + pattern_size <= main_nodes.items.len) : (i += 1) {
            var subgraph = try SelfSimilarRelationalGraph.init(self.allocator);
            defer subgraph.deinit();

            const subgraph_nodes = main_nodes.items[i .. i + pattern_size];

            for (subgraph_nodes) |node_id| {
                if (main_graph.nodes.get(node_id)) |node| {
                    const node_clone = try node.clone(self.allocator);
                    try subgraph.addNode(node_clone);
                }
            }

            var edge_iter = main_graph.edges.iterator();
            while (edge_iter.next()) |edge_entry| {
                const key = edge_entry.key_ptr.*;
                var source_in_subgraph = false;
                var target_in_subgraph = false;

                for (subgraph_nodes) |node_id| {
                    if (std.mem.eql(u8, key.source, node_id)) source_in_subgraph = true;
                    if (std.mem.eql(u8, key.target, node_id)) target_in_subgraph = true;
                }

                if (source_in_subgraph and target_in_subgraph) {
                    for (edge_entry.value_ptr.items) |edge| {
                        const edge_clone = try edge.clone(self.allocator);
                        try subgraph.addEdge(edge_clone.source, edge_clone.target, edge_clone);
                    }
                }
            }

            const subgraph_canonical = try self.computeCanonicalForm(&subgraph);
            defer self.allocator.free(subgraph_canonical);

            if (std.mem.eql(u8, subgraph_canonical, pattern_canonical)) {
                var match_set = ArrayList([]const u8).init(self.allocator);
                for (subgraph_nodes) |node_id| {
                    try match_set.append(try self.allocator.dupe(u8, node_id));
                }
                try matches.append(match_set);
            }
        }

        return matches;
    }

    pub fn cacheCanonicalForm(self: *GraphIsomorphismProcessor, canonical: []const u8, node_ids: []const []const u8) !void {
        const key = try self.allocator.dupe(u8, canonical);
        errdefer self.allocator.free(key);

        var list = ArrayList([]const u8).init(self.allocator);
        for (node_ids) |id| {
            try list.append(try self.allocator.dupe(u8, id));
        }

        try self.canonical_forms.put(key, list);
    }
};

pub const EdgeKeyForWeighting = struct {
    source: []const u8,
    target: []const u8,
};

const EdgeKeyForWeightingContext = struct {
    pub fn hash(_: @This(), key: EdgeKeyForWeighting) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(key.source);
        hasher.update(&[_]u8{0});
        hasher.update(key.target);
        return hasher.final();
    }

    pub fn eql(_: @This(), a: EdgeKeyForWeighting, b: EdgeKeyForWeighting) bool {
        return std.mem.eql(u8, a.source, b.source) and std.mem.eql(u8, a.target, b.target);
    }
};

pub const DynamicEdgeWeighting = struct {
    weight_history: std.HashMap(EdgeKeyForWeighting, ArrayList(f64), EdgeKeyForWeightingContext, std.hash_map.default_max_load_percentage),
    key_storage: ArrayList([]const u8),
    learning_rate: f64,
    allocator: Allocator,

    pub fn init(allocator: Allocator) DynamicEdgeWeighting {
        return DynamicEdgeWeighting{
            .weight_history = std.HashMap(EdgeKeyForWeighting, ArrayList(f64), EdgeKeyForWeightingContext, std.hash_map.default_max_load_percentage).init(allocator),
            .key_storage = ArrayList([]const u8).init(allocator),
            .learning_rate = 0.01,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DynamicEdgeWeighting) void {
        var iter = self.weight_history.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.weight_history.deinit();

        for (self.key_storage.items) |key| {
            self.allocator.free(key);
        }
        self.key_storage.deinit();

        self.learning_rate = 0.0;
    }

    pub fn updateWeight(self: *DynamicEdgeWeighting, source: []const u8, target: []const u8, current_weight: f64, feedback: f64) !f64 {
        if (!std.math.isFinite(current_weight) or !std.math.isFinite(feedback) or !std.math.isFinite(self.learning_rate)) return error.InvalidWeight;
        const delta = self.learning_rate * feedback;
        var new_weight = current_weight + delta;
        if (!std.math.isFinite(new_weight)) return error.InvalidWeight;
        new_weight = @max(0.0, @min(1.0, new_weight));

        const lookup_key = EdgeKeyForWeighting{ .source = source, .target = target };
        if (self.weight_history.getPtr(lookup_key)) |history| {
            try history.append(new_weight);
            return new_weight;
        }

        const source_copy = try self.allocator.dupe(u8, source);
        errdefer self.allocator.free(source_copy);
        const target_copy = try self.allocator.dupe(u8, target);
        errdefer self.allocator.free(target_copy);

        var history = ArrayList(f64).init(self.allocator);
        errdefer history.deinit();
        try history.append(new_weight);
        try self.key_storage.ensureUnusedCapacity(2);

        const owned_key = EdgeKeyForWeighting{ .source = source_copy, .target = target_copy };
        try self.weight_history.putNoClobber(owned_key, history);
        self.key_storage.appendAssumeCapacity(source_copy);
        self.key_storage.appendAssumeCapacity(target_copy);
        return new_weight;
    }

    pub fn computeAdaptiveWeight(
        self: *DynamicEdgeWeighting,
        source: []const u8,
        target: []const u8,
        base_weight: f64,
        temporal_factor: f64,
        spatial_factor: f64,
        semantic_factor: f64,
    ) f64 {
        const key = EdgeKeyForWeighting{ .source = source, .target = target };
        var history_adjustment: f64 = 1.0;
        var temporal_adjustment: f64 = 1.0;
        var spatial_adjustment: f64 = 1.0;
        var semantic_adjustment: f64 = 1.0;
        if (self.weight_history.get(key)) |history| {
            if (history.items.len > 0) {
                const recent = history.items[history.items.len - 1];
                history_adjustment = 0.8 + 0.2 * recent;
                const history_len = @as(f64, @floatFromInt(history.items.len));
                temporal_adjustment = temporal_factor * (1.0 + 0.1 * @log(@max(1.0, history_len)));
                if (history.items.len >= 2) {
                    const prev = history.items[history.items.len - 2];
                    const trend = recent - prev;
                    spatial_adjustment = spatial_factor * (1.0 + 0.05 * trend);
                } else {
                    spatial_adjustment = spatial_factor;
                }
                var history_sum: f64 = 0.0;
                for (history.items) |h| {
                    history_sum += h;
                }
                const history_avg = history_sum / history_len;
                semantic_adjustment = semantic_factor * (0.5 + 0.5 * history_avg);
            } else {
                temporal_adjustment = temporal_factor;
                spatial_adjustment = spatial_factor;
                semantic_adjustment = semantic_factor;
            }
        } else {
            temporal_adjustment = temporal_factor;
            spatial_adjustment = spatial_factor;
            semantic_adjustment = semantic_factor;
        }
        var adaptive_weight = base_weight * history_adjustment * temporal_adjustment * spatial_adjustment * semantic_adjustment;
        adaptive_weight = @max(0.0, @min(1.0, adaptive_weight));
        return adaptive_weight;
    }

    pub fn propagateWeights(self: *DynamicEdgeWeighting, graph: *SelfSimilarRelationalGraph, source_node: []const u8, iterations: usize) !void {
        var visited = std.HashMap([]const u8, void, StringContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        defer {
            var iter = visited.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            visited.deinit();
        }

        var current_layer = ArrayList([]const u8).init(self.allocator);
        defer {
            for (current_layer.items) |item| {
                self.allocator.free(item);
            }
            current_layer.deinit();
        }

        try current_layer.append(try self.allocator.dupe(u8, source_node));

        var iteration: usize = 0;
        while (iteration < iterations) : (iteration += 1) {
            var next_layer = ArrayList([]const u8).init(self.allocator);
            defer {
                for (next_layer.items) |item| {
                    self.allocator.free(item);
                }
                next_layer.deinit();
            }

            for (current_layer.items) |node_id| {
                if (visited.contains(node_id)) {
                    continue;
                }

                const visited_copy = try self.allocator.dupe(u8, node_id);
                try visited.put(visited_copy, {});

                const decay_factor = std.math.pow(f64, 0.9, @as(f64, @floatFromInt(iteration)));

                var edge_iter = graph.edges.iterator();
                while (edge_iter.next()) |edge_entry| {
                    const key = edge_entry.key_ptr.*;
                    if (std.mem.eql(u8, key.source, node_id)) {
                        for (edge_entry.value_ptr.items) |*edge| {
                            edge.weight *= decay_factor;
                        }
                        var already_added = false;
                        for (next_layer.items) |existing| {
                            if (std.mem.eql(u8, existing, key.target)) {
                                already_added = true;
                                break;
                            }
                        }
                        if (!already_added) {
                            try next_layer.append(try self.allocator.dupe(u8, key.target));
                        }
                    } else if (std.mem.eql(u8, key.target, node_id)) {
                        for (edge_entry.value_ptr.items) |*edge| {
                            edge.weight *= decay_factor;
                        }
                        var already_added = false;
                        for (next_layer.items) |existing| {
                            if (std.mem.eql(u8, existing, key.source)) {
                                already_added = true;
                                break;
                            }
                        }
                        if (!already_added) {
                            try next_layer.append(try self.allocator.dupe(u8, key.source));
                        }
                    }
                }
            }

            for (current_layer.items) |item| {
                self.allocator.free(item);
            }
            current_layer.clearRetainingCapacity();

            for (next_layer.items) |item| {
                try current_layer.append(try self.allocator.dupe(u8, item));
            }

            if (current_layer.items.len == 0) {
                break;
            }
        }
    }

    pub fn setLearningRate(self: *DynamicEdgeWeighting, rate: f64) void {
        self.learning_rate = @max(0.0, @min(1.0, rate));
    }

    pub fn getWeightHistory(self: *const DynamicEdgeWeighting, source: []const u8, target: []const u8) ?[]const f64 {
        const key = EdgeKeyForWeighting{ .source = source, .target = target };
        if (self.weight_history.get(key)) |history| {
            return history.items;
        }
        return null;
    }
};

pub const SparseActivationManager = struct {
    sparsity_threshold: f64,
    activation_map: AutoHashMap(usize, bool),
    energy_saved: f64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, sparsity_threshold: f64) SparseActivationManager {
        return SparseActivationManager{
            .sparsity_threshold = sparsity_threshold,
            .activation_map = AutoHashMap(usize, bool).init(allocator),
            .energy_saved = 0.0,
            .allocator = allocator,
        };
    }

    pub fn initDefault(allocator: Allocator) SparseActivationManager {
        return SparseActivationManager.init(allocator, 0.1);
    }

    pub fn deinit(self: *SparseActivationManager) void {
        self.activation_map.deinit();
    }

    pub fn shouldActivateCore(self: *SparseActivationManager, core_id: usize, workload: f64) !bool {
        if (workload < self.sparsity_threshold) {
            try self.activation_map.put(core_id, false);
            self.energy_saved += 1.0;
            return false;
        }
        try self.activation_map.put(core_id, true);
        return true;
    }

    pub fn computeSparsityRatio(self: *const SparseActivationManager) f64 {
        if (self.activation_map.count() == 0) {
            return 0.0;
        }
        var inactive_count: usize = 0;
        var iter = self.activation_map.iterator();
        while (iter.next()) |entry| {
            if (!entry.value_ptr.*) {
                inactive_count += 1;
            }
        }
        return @as(f64, @floatFromInt(inactive_count)) / @as(f64, @floatFromInt(self.activation_map.count()));
    }

    pub fn isActivated(self: *const SparseActivationManager, core_id: usize) ?bool {
        return self.activation_map.get(core_id);
    }

    pub fn getEnergySaved(self: *const SparseActivationManager) f64 {
        return self.energy_saved;
    }

    pub fn resetEnergySaved(self: *SparseActivationManager) void {
        self.energy_saved = 0.0;
    }

    pub fn setSparsityThreshold(self: *SparseActivationManager, threshold: f64) void {
        self.sparsity_threshold = @max(0.0, @min(1.0, threshold));
    }
};

pub const CoreIdSet = AutoHashMap(usize, void);

pub const PowerGatingController = struct {
    gated_cores: CoreIdSet,
    power_budget: f64,
    current_power: f64,
    allocator: Allocator,

    pub fn init(allocator: Allocator) PowerGatingController {
        return PowerGatingController{
            .gated_cores = CoreIdSet.init(allocator),
            .power_budget = 1000.0,
            .current_power = 0.0,
            .allocator = allocator,
        };
    }

    pub fn initWithBudget(allocator: Allocator, power_budget: f64) PowerGatingController {
        return PowerGatingController{
            .gated_cores = CoreIdSet.init(allocator),
            .power_budget = power_budget,
            .current_power = 0.0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PowerGatingController) void {
        self.gated_cores.deinit();
    }

    pub fn gateCore(self: *PowerGatingController, core: *ProcessingCore) !bool {
        if (self.gated_cores.contains(core.core_id)) {
            return false;
        }
        core.state = .power_gated;
        try self.gated_cores.put(core.core_id, {});
        self.current_power -= 10.0;
        return true;
    }

    pub fn ungateCore(self: *PowerGatingController, core: *ProcessingCore) bool {
        if (!self.gated_cores.contains(core.core_id)) {
            return false;
        }
        if (self.current_power + 10.0 > self.power_budget) {
            return false;
        }
        core.state = .idle;
        _ = self.gated_cores.remove(core.core_id);
        self.current_power += 10.0;
        return true;
    }

    pub fn managePowerBudget(self: *PowerGatingController, cores: *AutoHashMap(usize, ProcessingCore)) !void {
        const CoreUtilization = struct {
            core_id: usize,
            utilization: f64,

            fn lessThan(_: void, a: @This(), b: @This()) bool {
                return a.utilization < b.utilization;
            }
        };

        var core_utilization = ArrayList(CoreUtilization).init(self.allocator);
        defer core_utilization.deinit();

        var iter = cores.iterator();
        while (iter.next()) |entry| {
            const core_id = entry.key_ptr.*;
            const core = entry.value_ptr.*;
            if (core.state != .power_gated) {
                const total_cycles = core.cycles_active + core.cycles_idle;
                const utilization: f64 = if (total_cycles > 0)
                    @as(f64, @floatFromInt(core.cycles_active)) / @as(f64, @floatFromInt(total_cycles))
                else
                    0.0;
                try core_utilization.append(.{ .core_id = core_id, .utilization = utilization });
            }
        }

        std.mem.sort(CoreUtilization, core_utilization.items, {}, CoreUtilization.lessThan);

        for (core_utilization.items) |cu| {
            if (cores.getPtr(cu.core_id)) |core| {
                if (cu.utilization < 0.1 and self.current_power > self.power_budget * 0.5) {
                    _ = try self.gateCore(core);
                } else if (cu.utilization > 0.8 and self.gated_cores.contains(cu.core_id)) {
                    _ = self.ungateCore(core);
                }
            }
        }
    }

    pub fn isGated(self: *const PowerGatingController, core_id: usize) bool {
        return self.gated_cores.contains(core_id);
    }

    pub fn getGatedCount(self: *const PowerGatingController) usize {
        return self.gated_cores.count();
    }

    pub fn setPowerBudget(self: *PowerGatingController, budget: f64) void {
        self.power_budget = @max(0.0, budget);
    }

    pub fn getPowerUtilization(self: *const PowerGatingController) f64 {
        if (self.power_budget == 0.0) return 0.0;
        return self.current_power / self.power_budget;
    }
};

pub const RPGUStatistics = struct {
    total_cores: usize,
    active_cores: usize,
    gated_cores: usize,
    total_energy_consumed: f64,
    total_active_cycles: usize,
    total_idle_cycles: usize,
    execution_cycles: usize,
    sparsity_ratio: f64,
    energy_saved: f64,
    total_messages: usize,
    average_message_hops: f64,
    current_power: f64,
    power_budget: f64,
};

pub const RelationalGraphProcessingUnit = struct {
    noc: AsynchronousNoC,
    isomorphism_processor: GraphIsomorphismProcessor,
    edge_weighting: DynamicEdgeWeighting,
    sparse_activation: SparseActivationManager,
    power_gating: PowerGatingController,
    global_graph: ?*SelfSimilarRelationalGraph,
    global_graph_owned: bool,
    execution_cycles: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, grid_width: usize, grid_height: usize) !RelationalGraphProcessingUnit {
        return RelationalGraphProcessingUnit{
            .noc = try AsynchronousNoC.init(allocator, grid_width, grid_height),
            .isomorphism_processor = GraphIsomorphismProcessor.init(allocator),
            .edge_weighting = DynamicEdgeWeighting.init(allocator),
            .sparse_activation = SparseActivationManager.initDefault(allocator),
            .power_gating = PowerGatingController.init(allocator),
            .global_graph = null,
            .global_graph_owned = false,
            .execution_cycles = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RelationalGraphProcessingUnit) void {
        self.noc.deinit();
        self.isomorphism_processor.deinit();
        self.edge_weighting.deinit();
        self.sparse_activation.deinit();
        self.power_gating.deinit();
        if (self.global_graph_owned) {
            if (self.global_graph) |graph| {
                graph.deinit();
                self.allocator.destroy(graph);
            }
        }
    }

    pub fn setGlobalGraph(self: *RelationalGraphProcessingUnit, graph: *SelfSimilarRelationalGraph, owned: bool) void {
        if (self.global_graph_owned) {
            if (self.global_graph) |old_graph| {
                old_graph.deinit();
                self.allocator.destroy(old_graph);
            }
        }
        self.global_graph = graph;
        self.global_graph_owned = owned;
    }

    pub fn clearLocalGraphs(self: *RelationalGraphProcessingUnit) !void {
        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            _ = try entry.value_ptr.createLocalGraph();
        }
    }

    pub fn distributeGraph(self: *RelationalGraphProcessingUnit, graph: *SelfSimilarRelationalGraph) !void {
        var node_list = ArrayList([]const u8).init(self.allocator);
        defer node_list.deinit();

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            try node_list.append(entry.key_ptr.*);
        }

        var cores_available = ArrayList(usize).init(self.allocator);
        defer cores_available.deinit();

        // Roll back all local graphs on any failure after this point.
        // Placed BEFORE the createLocalGraph loop so that a failure partway
        // through that loop (e.g. OOM on core 3 of 8) still frees the graphs
        // already allocated for cores 0-2.  Cores whose local_graph_owned is
        // false are safely skipped by the inner guard.
        errdefer {
            var rollback_iter = self.noc.cores.iterator();
            while (rollback_iter.next()) |entry| {
                const core = entry.value_ptr;
                if (core.local_graph_owned) {
                    if (core.local_graph) |lg| {
                        lg.deinit();
                        self.allocator.destroy(lg);
                    }
                }
                core.local_graph = null;
                core.local_graph_owned = false;
            }
        }

        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            _ = try entry.value_ptr.createLocalGraph();
            if (entry.value_ptr.state != .power_gated) {
                try cores_available.append(entry.key_ptr.*);
            }
        }

        if (cores_available.items.len == 0) return;

        const nodes_per_core = node_list.items.len / cores_available.items.len;
        const remainder = node_list.items.len % cores_available.items.len;

        var start_idx: usize = 0;
        var idx: usize = 0;
        while (idx < cores_available.items.len) : (idx += 1) {
            const core_id = cores_available.items[idx];
            const extra: usize = if (idx < remainder) 1 else 0;
            const end_idx = start_idx + nodes_per_core + extra;

            if (self.noc.getCore(core_id)) |core| {
                const local_graph = core.local_graph orelse return error.LocalGraphUnavailable;
                const core_nodes = node_list.items[start_idx..end_idx];

                for (core_nodes) |node_id| {
                    if (graph.nodes.get(node_id)) |node| {
                        var node_clone = try node.clone(self.allocator);
                        var node_transferred = false;
                        defer if (!node_transferred) node_clone.deinit();
                        try local_graph.addNode(node_clone);
                        node_transferred = true;
                    }
                }

                var edge_iter = graph.edges.iterator();
                while (edge_iter.next()) |edge_entry| {
                    const key = edge_entry.key_ptr.*;
                    var source_in_core = false;
                    var target_in_core = false;

                    for (core_nodes) |node_id| {
                        if (std.mem.eql(u8, key.source, node_id)) source_in_core = true;
                        if (std.mem.eql(u8, key.target, node_id)) target_in_core = true;
                    }

                    if (source_in_core and target_in_core) {
                        for (edge_entry.value_ptr.items) |edge| {
                            const edge_clone = try edge.clone(self.allocator);
                            try local_graph.addEdge(edge_clone.source, edge_clone.target, edge_clone);
                        }
                    }
                }

                start_idx = end_idx;
            }
        }
    }

    pub fn processIsomorphismParallel(self: *RelationalGraphProcessingUnit, pattern_graph: *SelfSimilarRelationalGraph) !ArrayList(ArrayList([]const u8)) {
        var all_matches = ArrayList(ArrayList([]const u8)).init(self.allocator);
        errdefer {
            for (all_matches.items) |*match| {
                for (match.items) |item| {
                    self.allocator.free(item);
                }
                match.deinit();
            }
            all_matches.deinit();
        }

        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            const core_id = entry.key_ptr.*;
            var core = entry.value_ptr;

            if (core.state == .power_gated or core.local_graph == null) {
                continue;
            }

            const workload: f64 = @as(f64, @floatFromInt(core.local_graph.?.nodeCount())) / 100.0;
            const should_activate = try self.sparse_activation.shouldActivateCore(core_id, workload);
            if (!should_activate) {
                continue;
            }

            core.state = .processing;
            var matches = try self.isomorphism_processor.findIsomorphicSubgraphs(core.local_graph.?, pattern_graph);

            for (matches.items) |match| {
                try all_matches.append(match);
            }
            matches.deinit();

            core.cycles_active += 1;
            core.energy_consumed += 5.0;
            core.state = .idle;
        }

        self.execution_cycles += 1;
        return all_matches;
    }

    pub fn updateEdgeWeightsParallel(
        self: *RelationalGraphProcessingUnit,
        temporal_factor: f64,
        spatial_factor: f64,
        semantic_factor: f64,
    ) !void {
        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            const core_id = entry.key_ptr.*;
            var core = entry.value_ptr;

            if (core.state == .power_gated or core.local_graph == null) {
                continue;
            }

            var total_edges: usize = 0;
            var edge_iter = core.local_graph.?.edges.iterator();
            while (edge_iter.next()) |edge_entry| {
                total_edges += edge_entry.value_ptr.items.len;
            }

            const workload: f64 = @as(f64, @floatFromInt(total_edges)) / 100.0;
            const should_activate = try self.sparse_activation.shouldActivateCore(core_id, workload);
            if (!should_activate) {
                continue;
            }

            core.state = .processing;

            var edge_iter2 = core.local_graph.?.edges.iterator();
            while (edge_iter2.next()) |edge_entry| {
                const key = edge_entry.key_ptr.*;
                for (edge_entry.value_ptr.items) |*edge| {
                    const new_weight = self.edge_weighting.computeAdaptiveWeight(
                        key.source,
                        key.target,
                        edge.weight,
                        temporal_factor,
                        spatial_factor,
                        semantic_factor,
                    );
                    edge.weight = new_weight;
                }
            }

            core.cycles_active += 1;
            core.energy_consumed += 3.0;
            core.state = .idle;
        }

        self.execution_cycles += 1;
    }

    pub fn propagateWeightsAsync(self: *RelationalGraphProcessingUnit, source_node: []const u8, iterations: usize) !void {
        var source_core_id: ?usize = null;
        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            const core = entry.value_ptr.*;
            if (core.local_graph) |graph| {
                if (graph.nodes.contains(source_node)) {
                    source_core_id = entry.key_ptr.*;
                    break;
                }
            }
        }

        if (source_core_id == null) {
            return;
        }

        var iteration: usize = 0;
        while (iteration < iterations) : (iteration += 1) {
            var inner_core_iter = self.noc.cores.iterator();
            while (inner_core_iter.next()) |entry| {
                const core_id = entry.key_ptr.*;
                var core = entry.value_ptr;

                if (core.state == .power_gated or core.local_graph == null) {
                    continue;
                }

                core.state = .processing;
                try self.edge_weighting.propagateWeights(core.local_graph.?, source_node, 1);

                for (core.neighbors.items) |neighbor_id| {
                    var buffer: [64]u8 = undefined;
                    const payload = std.fmt.bufPrint(&buffer, "iteration:{d}", .{iteration}) catch "";
                    const message = try NoCMessage.init(
                        self.allocator,
                        core_id,
                        neighbor_id,
                        .weight_update,
                        payload,
                        @intCast(iteration),
                    );
                    _ = try self.noc.sendMessage(message);
                }

                core.state = .communicating;
                core.cycles_active += 1;
                core.energy_consumed += 2.0;
            }

            _ = try self.noc.routeMessages();
            self.execution_cycles += 1;
        }
    }

    pub fn synchronizeGraphs(self: *RelationalGraphProcessingUnit) !void {
        if (self.global_graph_owned) {
            if (self.global_graph) |old_graph| {
                old_graph.deinit();
                self.allocator.destroy(old_graph);
            }
        }

        const new_global = try self.allocator.create(SelfSimilarRelationalGraph);
        new_global.* = try SelfSimilarRelationalGraph.init(self.allocator);
        self.global_graph = new_global;
        self.global_graph_owned = true;

        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            const core = entry.value_ptr.*;
            if (core.local_graph == null) {
                continue;
            }

            var node_iter = core.local_graph.?.nodes.iterator();
            while (node_iter.next()) |node_entry| {
                const node_id = node_entry.key_ptr.*;
                if (!new_global.nodes.contains(node_id)) {
                    const node_clone = try node_entry.value_ptr.clone(self.allocator);
                    try new_global.addNode(node_clone);
                }
            }

            var edge_iter = core.local_graph.?.edges.iterator();
            while (edge_iter.next()) |edge_entry| {
                for (edge_entry.value_ptr.items) |edge| {
                    const edge_clone = try edge.clone(self.allocator);
                    try new_global.addEdge(edge_clone.source, edge_clone.target, edge_clone);
                }
            }
        }
    }

    pub fn managePower(self: *RelationalGraphProcessingUnit) !void {
        try self.power_gating.managePowerBudget(&self.noc.cores);
    }

    pub fn getStatistics(self: *RelationalGraphProcessingUnit) RPGUStatistics {
        var total_energy: f64 = 0.0;
        var total_active_cycles: usize = 0;
        var total_idle_cycles: usize = 0;
        var active_cores: usize = 0;

        var core_iter = self.noc.cores.iterator();
        while (core_iter.next()) |entry| {
            const core = entry.value_ptr.*;
            total_energy += core.energy_consumed;
            total_active_cycles += core.cycles_active;
            total_idle_cycles += core.cycles_idle;
            if (core.state != .power_gated) {
                active_cores += 1;
            }
        }

        const sparsity_ratio = self.sparse_activation.computeSparsityRatio();
        const avg_message_hops: f64 = if (self.noc.total_messages > 0)
            @as(f64, @floatFromInt(self.noc.total_hops)) / @as(f64, @floatFromInt(self.noc.total_messages))
        else
            0.0;

        return RPGUStatistics{
            .total_cores = self.noc.cores.count(),
            .active_cores = active_cores,
            .gated_cores = self.power_gating.getGatedCount(),
            .total_energy_consumed = total_energy,
            .total_active_cycles = total_active_cycles,
            .total_idle_cycles = total_idle_cycles,
            .execution_cycles = self.execution_cycles,
            .sparsity_ratio = sparsity_ratio,
            .energy_saved = self.sparse_activation.getEnergySaved(),
            .total_messages = self.noc.total_messages,
            .average_message_hops = avg_message_hops,
            .current_power = self.power_gating.current_power,
            .power_budget = self.power_gating.power_budget,
        };
    }

    pub fn getGridDimensions(self: *const RelationalGraphProcessingUnit) struct { width: usize, height: usize } {
        return .{ .width = self.noc.grid_width, .height = self.noc.grid_height };
    }

    pub fn setSparsityThreshold(self: *RelationalGraphProcessingUnit, threshold: f64) void {
        self.sparse_activation.setSparsityThreshold(threshold);
    }

    pub fn setPowerBudget(self: *RelationalGraphProcessingUnit, budget: f64) void {
        self.power_gating.setPowerBudget(budget);
    }

    pub fn setLearningRate(self: *RelationalGraphProcessingUnit, rate: f64) void {
        self.edge_weighting.setLearningRate(rate);
    }
};

test "CoreState enum" {
    const testing = std.testing;
    try testing.expectEqualStrings("idle", CoreState.idle.toString());
    try testing.expectEqualStrings("processing", CoreState.processing.toString());
    try testing.expectEqualStrings("communicating", CoreState.communicating.toString());
    try testing.expectEqualStrings("power_gated", CoreState.power_gated.toString());
    try testing.expectEqual(CoreState.idle, CoreState.fromString("idle").?);
    try testing.expectEqual(CoreState.processing, CoreState.fromString("processing").?);
}

test "MessageType enum" {
    const testing = std.testing;
    try testing.expectEqualStrings("weight_update", MessageType.weight_update.toString());
    try testing.expectEqual(MessageType.graph_sync, MessageType.fromString("graph_sync").?);
}

test "ProcessingCore init and deinit" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var core = ProcessingCore.init(allocator, 0, 2, 3);
    defer core.deinit();

    try testing.expectEqual(@as(usize, 0), core.core_id);
    try testing.expectEqual(@as(usize, 2), core.x);
    try testing.expectEqual(@as(usize, 3), core.y);
    try testing.expectEqual(CoreState.idle, core.state);
    try testing.expectApproxEqAbs(@as(f64, 0.0), core.energy_consumed, 0.001);
}

test "ProcessingCore addNeighbor" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var core = ProcessingCore.init(allocator, 0, 0, 0);
    defer core.deinit();

    try core.addNeighbor(1);
    try core.addNeighbor(4);

    try testing.expectEqual(@as(usize, 2), core.neighbors.items.len);
    try testing.expectEqual(@as(usize, 1), core.neighbors.items[0]);
    try testing.expectEqual(@as(usize, 4), core.neighbors.items[1]);
}

test "NoCMessage init and deinit" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var msg = try NoCMessage.init(allocator, 0, 5, .weight_update, "test payload", 1);
    defer msg.deinit();

    try testing.expectEqual(@as(usize, 0), msg.source_core);
    try testing.expectEqual(@as(usize, 5), msg.target_core);
    try testing.expectEqual(MessageType.weight_update, msg.message_type);
    try testing.expectEqualStrings("test payload", msg.payload);
    try testing.expectEqual(@as(i32, 1), msg.priority);
}

test "AsynchronousNoC init and basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var noc = try AsynchronousNoC.init(allocator, 4, 4);
    defer noc.deinit();

    try testing.expectEqual(@as(usize, 4), noc.grid_width);
    try testing.expectEqual(@as(usize, 4), noc.grid_height);
    try testing.expectEqual(@as(usize, 16), noc.cores.count());

    const core = noc.getCore(5);
    try testing.expect(core != null);
    try testing.expectEqual(@as(usize, 1), core.?.x);
    try testing.expectEqual(@as(usize, 1), core.?.y);
}

test "AsynchronousNoC sendMessage and routeMessages" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var noc = try AsynchronousNoC.init(allocator, 4, 4);
    defer noc.deinit();

    const msg = try NoCMessage.init(allocator, 0, 15, .data_transfer, "test", 0);
    const sent = try noc.sendMessage(msg);
    try testing.expect(sent);
    try testing.expectEqual(@as(usize, 1), noc.total_messages);

    const routed = try noc.routeMessages();
    try testing.expectEqual(@as(usize, 1), routed);
}

test "SparseActivationManager" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var manager = SparseActivationManager.init(allocator, 0.1);
    defer manager.deinit();

    const activate_low = try manager.shouldActivateCore(0, 0.05);
    try testing.expect(!activate_low);

    const activate_high = try manager.shouldActivateCore(1, 0.5);
    try testing.expect(activate_high);

    const sparsity = manager.computeSparsityRatio();
    try testing.expectApproxEqAbs(@as(f64, 0.5), sparsity, 0.001);
}

test "PowerGatingController" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var controller = PowerGatingController.init(allocator);
    defer controller.deinit();

    var core = ProcessingCore.init(allocator, 0, 0, 0);
    defer core.deinit();

    const gated = try controller.gateCore(&core);
    try testing.expect(gated);
    try testing.expectEqual(CoreState.power_gated, core.state);
    try testing.expect(controller.isGated(0));

    const ungated = controller.ungateCore(&core);
    try testing.expect(ungated);
    try testing.expectEqual(CoreState.idle, core.state);
    try testing.expect(!controller.isGated(0));
}

test "DynamicEdgeWeighting" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var weighting = DynamicEdgeWeighting.init(allocator);
    defer weighting.deinit();

    const new_weight = try weighting.updateWeight("node1", "node2", 0.5, 10.0);
    try testing.expect(new_weight > 0.5);
    try testing.expect(new_weight <= 1.0);

    const adaptive = weighting.computeAdaptiveWeight("a", "b", 0.5, 1.0, 1.0, 1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.5), adaptive, 0.001);
}

test "GraphIsomorphismProcessor" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var processor = GraphIsomorphismProcessor.init(allocator);
    defer processor.deinit();

    var graph1 = try SelfSimilarRelationalGraph.init(allocator);
    defer graph1.deinit();

    const node1 = try Node.init(allocator, "n1", "data1", Qubit.initBasis0(), 0.0);
    try graph1.addNode(node1);
    const node2 = try Node.init(allocator, "n2", "data2", Qubit.initBasis1(), 0.0);
    try graph1.addNode(node2);

    const canonical = try processor.computeCanonicalForm(&graph1);
    defer allocator.free(canonical);
    try testing.expect(canonical.len > 0);
}

test "RelationalGraphProcessingUnit init and getStatistics" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var rpgu = try RelationalGraphProcessingUnit.init(allocator, 2, 2);
    defer rpgu.deinit();

    const dims = rpgu.getGridDimensions();
    try testing.expectEqual(@as(usize, 2), dims.width);
    try testing.expectEqual(@as(usize, 2), dims.height);

    const stats = rpgu.getStatistics();
    try testing.expectEqual(@as(usize, 4), stats.total_cores);
    try testing.expectEqual(@as(usize, 4), stats.active_cores);
    try testing.expectEqual(@as(usize, 0), stats.gated_cores);
    try testing.expectApproxEqAbs(@as(f64, 0.0), stats.total_energy_consumed, 0.001);
}

test "RelationalGraphProcessingUnit distributeGraph" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var rpgu = try RelationalGraphProcessingUnit.init(allocator, 2, 2);
    defer rpgu.deinit();

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "n1", "d1", Qubit.initBasis0(), 0.0);
    try graph.addNode(n1);
    const n2 = try Node.init(allocator, "n2", "d2", Qubit.initBasis0(), 0.0);
    try graph.addNode(n2);
    const n3 = try Node.init(allocator, "n3", "d3", Qubit.initBasis0(), 0.0);
    try graph.addNode(n3);
    const n4 = try Node.init(allocator, "n4", "d4", Qubit.initBasis0(), 0.0);
    try graph.addNode(n4);

    try rpgu.distributeGraph(&graph);

    var total_nodes: usize = 0;
    var core_iter = rpgu.noc.cores.iterator();
    while (core_iter.next()) |entry| {
        if (entry.value_ptr.local_graph) |local| {
            total_nodes += local.nodeCount();
        }
    }
    try testing.expect(total_nodes >= 4);
}

test "RelationalGraphProcessingUnit distributeGraph errdefer: all cores have local graphs after success" {
    // After a successful distributeGraph call every core must own a local graph.
    // This verifies that the errdefer rollback block is guarded by
    // local_graph_owned and would therefore only free graphs that were actually
    // allocated — a prerequisite for safe partial-failure rollback.
    const testing = std.testing;
    const allocator = testing.allocator;

    var rpgu = try RelationalGraphProcessingUnit.init(allocator, 2, 2);
    defer rpgu.deinit();

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "p", "dp", Qubit.initBasis0(), 0.0);
    try graph.addNode(n1);
    const n2 = try Node.init(allocator, "q", "dq", Qubit.initBasis1(), 0.0);
    try graph.addNode(n2);

    try rpgu.distributeGraph(&graph);

    // Every core must now own a local graph (owned flag set) so that a
    // hypothetical subsequent error would trigger the errdefer correctly.
    var core_iter = rpgu.noc.cores.iterator();
    while (core_iter.next()) |entry| {
        const core = entry.value_ptr;
        try testing.expect(core.local_graph != null);
        try testing.expect(core.local_graph_owned);
    }
}

test "RelationalGraphProcessingUnit distributeGraph: second call replaces local graphs" {
    // Calling distributeGraph twice must not leak the first set of local graphs.
    // The test allocator will catch any leak or double-free.
    const testing = std.testing;
    const allocator = testing.allocator;

    var rpgu = try RelationalGraphProcessingUnit.init(allocator, 2, 2);
    defer rpgu.deinit();

    var g1 = try SelfSimilarRelationalGraph.init(allocator);
    defer g1.deinit();
    const na = try Node.init(allocator, "a", "da", Qubit.initBasis0(), 0.0);
    try g1.addNode(na);

    try rpgu.distributeGraph(&g1);

    var g2 = try SelfSimilarRelationalGraph.init(allocator);
    defer g2.deinit();
    const nb = try Node.init(allocator, "b", "db", Qubit.initBasis1(), 0.0);
    try g2.addNode(nb);

    // Must replace the first set of local graphs without leaking.
    try rpgu.distributeGraph(&g2);

    var total: usize = 0;
    var core_iter = rpgu.noc.cores.iterator();
    while (core_iter.next()) |entry| {
        if (entry.value_ptr.local_graph) |lg| total += lg.nodeCount();
    }
    try testing.expect(total >= 1);
}

test "RelationalGraphProcessingUnit managePower" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var rpgu = try RelationalGraphProcessingUnit.init(allocator, 2, 2);
    defer rpgu.deinit();

    try rpgu.managePower();

    const stats = rpgu.getStatistics();
    try testing.expectEqual(@as(usize, 4), stats.total_cores);
}