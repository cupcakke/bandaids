const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const StringHashMap = std.StringHashMap;

const nsir_core = @import("nsir_core.zig");
const quantum_logic = @import("quantum_logic.zig");

pub const EdgeQuality = nsir_core.EdgeQuality;
pub const EdgeKey = nsir_core.EdgeKey;
pub const EdgeKeyContext = nsir_core.EdgeKeyContext;
pub const QuantumState = quantum_logic.QuantumState;

pub const Timestamp = i64;

pub const MIN_TIMESTAMP: Timestamp = std.math.minInt(Timestamp);
pub const MAX_TIMESTAMP: Timestamp = std.math.maxInt(Timestamp);

const StringContext = struct {
    pub fn hash(self: @This(), key: []const u8) u64 {
        _ = self;
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(key);
        return hasher.final();
    }

    pub fn eql(self: @This(), a: []const u8, b: []const u8) bool {
        _ = self;
        return std.mem.eql(u8, a, b);
    }
};

pub const NodeVersion = struct {
    version: usize,
    timestamp: Timestamp,
    data: QuantumState,
    properties: StringHashMap([]const u8),
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        version_num: usize,
        timestamp_ns: Timestamp,
        quantum_data: QuantumState,
    ) Self {
        return Self{
            .version = version_num,
            .timestamp = timestamp_ns,
            .data = quantum_data,
            .properties = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn initWithProperties(
        allocator: Allocator,
        version_num: usize,
        timestamp_ns: Timestamp,
        quantum_data: QuantumState,
        props: []const struct { key: []const u8, value: []const u8 },
    ) !Self {
        var self = Self{
            .version = version_num,
            .timestamp = timestamp_ns,
            .data = quantum_data,
            .properties = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
        for (props) |prop| {
            try self.setProperty(prop.key, prop.value);
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        var iter = self.properties.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.properties.deinit();
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var new_version = Self{
            .version = self.version,
            .timestamp = self.timestamp,
            .data = self.data.clone(),
            .properties = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
        errdefer new_version.deinit();
        var iter = self.properties.iterator();
        while (iter.next()) |entry| {
            const key_copy = try allocator.dupe(u8, entry.key_ptr.*);
            errdefer allocator.free(key_copy);
            const val_copy = try allocator.dupe(u8, entry.value_ptr.*);
            errdefer allocator.free(val_copy);
            try new_version.properties.put(key_copy, val_copy);
        }
        return new_version;
    }

    pub fn setProperty(self: *Self, key: []const u8, value: []const u8) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        const val_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(val_copy);

        if (self.properties.fetchRemove(key)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value);
        }
        try self.properties.put(key_copy, val_copy);
    }

    pub fn getProperty(self: *const Self, key: []const u8) ?[]const u8 {
        return self.properties.get(key);
    }

    pub fn removeProperty(self: *Self, key: []const u8) bool {
        if (self.properties.fetchRemove(key)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value);
            return true;
        }
        return false;
    }

    pub fn propertyCount(self: *const Self) usize {
        return self.properties.count();
    }

    pub fn probability(self: *const Self) f64 {
        return self.data.probability();
    }

    pub fn magnitude(self: *const Self) f64 {
        return self.data.magnitude();
    }
};

pub const EdgeVersion = struct {
    version: usize,
    timestamp: Timestamp,
    weight: f64,
    quality: EdgeQuality,
    metadata: StringHashMap([]const u8),
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        version_num: usize,
        timestamp_ns: Timestamp,
        edge_weight: f64,
        edge_quality: EdgeQuality,
    ) Self {
        return Self{
            .version = version_num,
            .timestamp = timestamp_ns,
            .weight = edge_weight,
            .quality = edge_quality,
            .metadata = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var new_version = Self{
            .version = self.version,
            .timestamp = self.timestamp,
            .weight = self.weight,
            .quality = self.quality,
            .metadata = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
        errdefer new_version.deinit();
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            const key_copy = try allocator.dupe(u8, entry.key_ptr.*);
            errdefer allocator.free(key_copy);
            const val_copy = try allocator.dupe(u8, entry.value_ptr.*);
            errdefer allocator.free(val_copy);
            try new_version.metadata.put(key_copy, val_copy);
        }
        return new_version;
    }

    pub fn setMetadata(self: *Self, key: []const u8, value: []const u8) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        const val_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(val_copy);

        if (self.metadata.fetchRemove(key)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value);
        }
        try self.metadata.put(key_copy, val_copy);
    }

    pub fn getMetadata(self: *const Self, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }

    pub fn qualityToString(self: *const Self) []const u8 {
        return self.quality.toString();
    }
};

pub const TemporalNode = struct {
    node_id: []const u8,
    versions: ArrayList(NodeVersion),
    current_version: usize,
    created_at: Timestamp,
    last_modified: Timestamp,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        initial_state: QuantumState,
        timestamp_ns: Timestamp,
    ) !Self {
        var self = Self{
            .node_id = try allocator.dupe(u8, id),
            .versions = ArrayList(NodeVersion).init(allocator),
            .current_version = 0,
            .created_at = timestamp_ns,
            .last_modified = timestamp_ns,
            .allocator = allocator,
        };
        const initial_version = NodeVersion.init(allocator, 0, timestamp_ns, initial_state);
        try self.versions.append(initial_version);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.node_id);
        for (self.versions.items) |*version| {
            version.deinit();
        }
        self.versions.deinit();
    }

    pub fn addVersion(self: *Self, state: QuantumState, timestamp_ns: Timestamp) !usize {
        const new_version_num = self.versions.items.len;
        const new_version = NodeVersion.init(self.allocator, new_version_num, timestamp_ns, state);
        try self.versions.append(new_version);
        self.current_version = new_version_num;
        self.last_modified = timestamp_ns;
        return new_version_num;
    }

    pub fn addVersionWithProperties(
        self: *Self,
        state: QuantumState,
        timestamp_ns: Timestamp,
        props: []const struct { key: []const u8, value: []const u8 },
    ) !usize {
        const new_version_num = self.versions.items.len;
        const new_version = try NodeVersion.initWithProperties(
            self.allocator,
            new_version_num,
            timestamp_ns,
            state,
            props,
        );
        try self.versions.append(new_version);
        self.current_version = new_version_num;
        self.last_modified = timestamp_ns;
        return new_version_num;
    }

    pub fn getVersion(self: *const Self, version_num: usize) ?*const NodeVersion {
        if (version_num >= self.versions.items.len) {
            return null;
        }
        return &self.versions.items[version_num];
    }

    pub fn getVersionMut(self: *Self, version_num: usize) ?*NodeVersion {
        if (version_num >= self.versions.items.len) {
            return null;
        }
        return &self.versions.items[version_num];
    }

    pub fn getCurrentVersion(self: *const Self) ?*const NodeVersion {
        return self.getVersion(self.current_version);
    }

    pub fn getCurrentVersionMut(self: *Self) ?*NodeVersion {
        return self.getVersionMut(self.current_version);
    }

    pub fn getVersionAt(self: *const Self, timestamp_ns: Timestamp) ?*const NodeVersion {
        var best_match: ?*const NodeVersion = null;
        var best_timestamp: Timestamp = MIN_TIMESTAMP;

        for (self.versions.items) |*version| {
            if (version.timestamp <= timestamp_ns and version.timestamp > best_timestamp) {
                best_match = version;
                best_timestamp = version.timestamp;
            }
        }
        return best_match;
    }

    pub fn rollback(self: *Self, target_version: usize) bool {
        if (target_version >= self.versions.items.len) {
            return false;
        }
        self.current_version = target_version;
        if (self.getVersion(target_version)) |version| {
            self.last_modified = version.timestamp;
        }
        return true;
    }

    pub fn rollbackToTime(self: *Self, timestamp_ns: Timestamp) bool {
        if (self.getVersionAt(timestamp_ns)) |version| {
            return self.rollback(version.version);
        }
        return false;
    }

    pub fn versionCount(self: *const Self) usize {
        return self.versions.items.len;
    }

    pub fn getVersionHistory(self: *const Self, allocator: Allocator) !ArrayList(usize) {
        var history = ArrayList(usize).init(allocator);
        var idx: usize = 0;
        while (idx < self.versions.items.len) : (idx += 1) {
            try history.append(idx);
        }
        return history;
    }

    pub fn getVersionsInRange(
        self: *const Self,
        start_time: Timestamp,
        end_time: Timestamp,
        allocator: Allocator,
    ) !ArrayList(*const NodeVersion) {
        var result = ArrayList(*const NodeVersion).init(allocator);
        for (self.versions.items) |*version| {
            if (version.timestamp >= start_time and version.timestamp <= end_time) {
                try result.append(version);
            }
        }
        return result;
    }

    pub fn getCurrentState(self: *const Self) ?QuantumState {
        if (self.getCurrentVersion()) |version| {
            return version.data;
        }
        return null;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var new_node = Self{
            .node_id = try allocator.dupe(u8, self.node_id),
            .versions = ArrayList(NodeVersion).init(allocator),
            .current_version = self.current_version,
            .created_at = self.created_at,
            .last_modified = self.last_modified,
            .allocator = allocator,
        };
        errdefer {
            for (new_node.versions.items) |*v| {
                v.deinit();
            }
            new_node.versions.deinit();
            allocator.free(new_node.node_id);
        }
        for (self.versions.items) |*version| {
            const cloned_version = try version.clone(allocator);
            try new_node.versions.append(cloned_version);
        }
        return new_node;
    }
};

pub const TemporalEdge = struct {
    edge_id: []const u8,
    source: []const u8,
    target: []const u8,
    versions: ArrayList(EdgeVersion),
    current_version: usize,
    valid_from: Timestamp,
    valid_to: Timestamp,
    created_at: Timestamp,
    last_modified: Timestamp,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        id: []const u8,
        source_id: []const u8,
        target_id: []const u8,
        initial_weight: f64,
        initial_quality: EdgeQuality,
        timestamp_ns: Timestamp,
    ) !Self {
        var self = Self{
            .edge_id = try allocator.dupe(u8, id),
            .source = try allocator.dupe(u8, source_id),
            .target = try allocator.dupe(u8, target_id),
            .versions = ArrayList(EdgeVersion).init(allocator),
            .current_version = 0,
            .valid_from = timestamp_ns,
            .valid_to = MAX_TIMESTAMP,
            .created_at = timestamp_ns,
            .last_modified = timestamp_ns,
            .allocator = allocator,
        };
        const initial_version = EdgeVersion.init(
            allocator,
            0,
            timestamp_ns,
            initial_weight,
            initial_quality,
        );
        try self.versions.append(initial_version);
        return self;
    }

    pub fn initWithTimeRange(
        allocator: Allocator,
        id: []const u8,
        source_id: []const u8,
        target_id: []const u8,
        initial_weight: f64,
        initial_quality: EdgeQuality,
        from_timestamp: Timestamp,
        to_timestamp: Timestamp,
    ) !Self {
        var self = try Self.init(
            allocator,
            id,
            source_id,
            target_id,
            initial_weight,
            initial_quality,
            from_timestamp,
        );
        self.valid_to = to_timestamp;
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.edge_id);
        self.allocator.free(self.source);
        self.allocator.free(self.target);
        for (self.versions.items) |*version| {
            version.deinit();
        }
        self.versions.deinit();
    }

    pub fn isValidAt(self: *const Self, timestamp_ns: Timestamp) bool {
        return timestamp_ns >= self.valid_from and timestamp_ns <= self.valid_to;
    }

    pub fn setValidityRange(self: *Self, from: Timestamp, to: Timestamp) void {
        self.valid_from = from;
        self.valid_to = to;
    }

    pub fn invalidate(self: *Self, timestamp_ns: Timestamp) void {
        self.valid_to = timestamp_ns;
    }

    pub fn addVersion(
        self: *Self,
        weight: f64,
        quality: EdgeQuality,
        timestamp_ns: Timestamp,
    ) !usize {
        const new_version_num = self.versions.items.len;
        const new_version = EdgeVersion.init(
            self.allocator,
            new_version_num,
            timestamp_ns,
            weight,
            quality,
        );
        try self.versions.append(new_version);
        self.current_version = new_version_num;
        self.last_modified = timestamp_ns;
        return new_version_num;
    }

    pub fn getVersion(self: *const Self, version_num: usize) ?*const EdgeVersion {
        if (version_num >= self.versions.items.len) {
            return null;
        }
        return &self.versions.items[version_num];
    }

    pub fn getVersionMut(self: *Self, version_num: usize) ?*EdgeVersion {
        if (version_num >= self.versions.items.len) {
            return null;
        }
        return &self.versions.items[version_num];
    }

    pub fn getCurrentVersion(self: *const Self) ?*const EdgeVersion {
        return self.getVersion(self.current_version);
    }

    pub fn getVersionAt(self: *const Self, timestamp_ns: Timestamp) ?*const EdgeVersion {
        if (!self.isValidAt(timestamp_ns)) {
            return null;
        }

        var best_match: ?*const EdgeVersion = null;
        var best_timestamp: Timestamp = MIN_TIMESTAMP;

        for (self.versions.items) |*version| {
            if (version.timestamp <= timestamp_ns and version.timestamp > best_timestamp) {
                best_match = version;
                best_timestamp = version.timestamp;
            }
        }
        return best_match;
    }

    pub fn rollback(self: *Self, target_version: usize) bool {
        if (target_version >= self.versions.items.len) {
            return false;
        }
        self.current_version = target_version;
        if (self.getVersion(target_version)) |version| {
            self.last_modified = version.timestamp;
        }
        return true;
    }

    pub fn versionCount(self: *const Self) usize {
        return self.versions.items.len;
    }

    pub fn getEdgeKey(self: *const Self) EdgeKey {
        return EdgeKey{ .source = self.source, .target = self.target };
    }

    pub fn getCurrentWeight(self: *const Self) f64 {
        if (self.getCurrentVersion()) |version| {
            return version.weight;
        }
        return 0.0;
    }

    pub fn getCurrentQuality(self: *const Self) EdgeQuality {
        if (self.getCurrentVersion()) |version| {
            return version.quality;
        }
        return .collapsed;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var new_edge = Self{
            .edge_id = try allocator.dupe(u8, self.edge_id),
            .source = try allocator.dupe(u8, self.source),
            .target = try allocator.dupe(u8, self.target),
            .versions = ArrayList(EdgeVersion).init(allocator),
            .current_version = self.current_version,
            .valid_from = self.valid_from,
            .valid_to = self.valid_to,
            .created_at = self.created_at,
            .last_modified = self.last_modified,
            .allocator = allocator,
        };
        errdefer {
            for (new_edge.versions.items) |*v| {
                v.deinit();
            }
            new_edge.versions.deinit();
            allocator.free(new_edge.edge_id);
            allocator.free(new_edge.source);
            allocator.free(new_edge.target);
        }
        for (self.versions.items) |*version| {
            const cloned_version = try version.clone(allocator);
            try new_edge.versions.append(cloned_version);
        }
        return new_edge;
    }
};

pub const GraphSnapshot = struct {
    snapshot_id: usize,
    timestamp: Timestamp,
    node_versions: std.HashMap([]const u8, usize, StringContext, std.hash_map.default_max_load_percentage),
    edge_versions: std.HashMap(EdgeKey, usize, EdgeKeyContext, std.hash_map.default_max_load_percentage),
    metadata: StringHashMap([]const u8),
    allocator: Allocator,
    allocated_keys: ArrayList([]const u8),

    const Self = @This();

    pub fn init(allocator: Allocator, id: usize, timestamp_ns: Timestamp) Self {
        return Self{
            .snapshot_id = id,
            .timestamp = timestamp_ns,
            .node_versions = std.HashMap([]const u8, usize, StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .edge_versions = std.HashMap(EdgeKey, usize, EdgeKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
            .metadata = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
            .allocated_keys = ArrayList([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.node_versions.deinit();
        self.edge_versions.deinit();

        for (self.allocated_keys.items) |key| {
            self.allocator.free(key);
        }
        self.allocated_keys.deinit();

        var meta_iter = self.metadata.iterator();
        while (meta_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }

    pub fn recordNodeVersion(self: *Self, node_id: []const u8, version: usize) !void {
        const key_copy = try self.allocator.dupe(u8, node_id);
        try self.allocated_keys.append(key_copy);
        try self.node_versions.put(key_copy, version);
    }

    pub fn recordEdgeVersion(self: *Self, edge_key: EdgeKey, version: usize) !void {
        const source_copy = try self.allocator.dupe(u8, edge_key.source);
        try self.allocated_keys.append(source_copy);
        const target_copy = try self.allocator.dupe(u8, edge_key.target);
        try self.allocated_keys.append(target_copy);
        const new_key = EdgeKey{ .source = source_copy, .target = target_copy };
        try self.edge_versions.put(new_key, version);
    }

    pub fn getNodeVersion(self: *const Self, node_id: []const u8) ?usize {
        return self.node_versions.get(node_id);
    }

    pub fn getEdgeVersion(self: *const Self, edge_key: EdgeKey) ?usize {
        return self.edge_versions.get(edge_key);
    }

    pub fn setMetadata(self: *Self, key: []const u8, value: []const u8) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        const val_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(val_copy);

        if (self.metadata.fetchRemove(key)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value);
        }
        try self.metadata.put(key_copy, val_copy);
    }

    pub fn nodeCount(self: *const Self) usize {
        return self.node_versions.count();
    }

    pub fn edgeCount(self: *const Self) usize {
        return self.edge_versions.count();
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var new_snapshot = Self.init(allocator, self.snapshot_id, self.timestamp);

        var node_iter = self.node_versions.iterator();
        while (node_iter.next()) |entry| {
            try new_snapshot.recordNodeVersion(entry.key_ptr.*, entry.value_ptr.*);
        }

        var edge_iter = self.edge_versions.iterator();
        while (edge_iter.next()) |entry| {
            try new_snapshot.recordEdgeVersion(entry.key_ptr.*, entry.value_ptr.*);
        }

        var meta_iter = self.metadata.iterator();
        while (meta_iter.next()) |entry| {
            try new_snapshot.setMetadata(entry.key_ptr.*, entry.value_ptr.*);
        }

        return new_snapshot;
    }
};

pub const TemporalQueryResult = struct {
    nodes: ArrayList(*const TemporalNode),
    edges: ArrayList(*const TemporalEdge),
    query_time: Timestamp,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, query_time: Timestamp) Self {
        return Self{
            .nodes = ArrayList(*const TemporalNode).init(allocator),
            .edges = ArrayList(*const TemporalEdge).init(allocator),
            .query_time = query_time,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.nodes.deinit();
        self.edges.deinit();
    }

    pub fn nodeCount(self: *const Self) usize {
        return self.nodes.items.len;
    }

    pub fn edgeCount(self: *const Self) usize {
        return self.edges.items.len;
    }
};

pub const NodeFilterFn = *const fn (*const TemporalNode) bool;
pub const EdgeFilterFn = *const fn (*const TemporalEdge) bool;

pub const TemporalQuery = struct {
    start_time: Timestamp,
    end_time: Timestamp,
    node_filter: ?NodeFilterFn,
    edge_filter: ?EdgeFilterFn,
    include_invalidated_edges: bool,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, start: Timestamp, end: Timestamp) Self {
        return Self{
            .start_time = start,
            .end_time = end,
            .node_filter = null,
            .edge_filter = null,
            .include_invalidated_edges = false,
            .allocator = allocator,
        };
    }

    pub fn initWithFilters(
        allocator: Allocator,
        start: Timestamp,
        end: Timestamp,
        node_filter: ?NodeFilterFn,
        edge_filter: ?EdgeFilterFn,
    ) Self {
        return Self{
            .start_time = start,
            .end_time = end,
            .node_filter = node_filter,
            .edge_filter = edge_filter,
            .include_invalidated_edges = false,
            .allocator = allocator,
        };
    }

    pub fn setNodeFilter(self: *Self, filter: NodeFilterFn) void {
        self.node_filter = filter;
    }

    pub fn setEdgeFilter(self: *Self, filter: EdgeFilterFn) void {
        self.edge_filter = filter;
    }

    pub fn setIncludeInvalidatedEdges(self: *Self, include: bool) void {
        self.include_invalidated_edges = include;
    }

    pub fn execute(self: *const Self, graph: *const TemporalGraph) !TemporalQueryResult {
        var result = TemporalQueryResult.init(self.allocator, self.end_time);

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            if (self.nodeMatchesTimeRange(node)) {
                if (self.node_filter) |filter| {
                    if (filter(node)) {
                        try result.nodes.append(node);
                    }
                } else {
                    try result.nodes.append(node);
                }
            }
        }

        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            const edge = entry.value_ptr;
            if (self.edgeMatchesTimeRange(edge)) {
                if (self.edge_filter) |filter| {
                    if (filter(edge)) {
                        try result.edges.append(edge);
                    }
                } else {
                    try result.edges.append(edge);
                }
            }
        }

        return result;
    }

    fn nodeMatchesTimeRange(self: *const Self, node: *const TemporalNode) bool {
        return node.created_at <= self.end_time and node.last_modified >= self.start_time;
    }

    fn edgeMatchesTimeRange(self: *const Self, edge: *const TemporalEdge) bool {
        if (!self.include_invalidated_edges) {
            if (!edge.isValidAt(self.end_time) and edge.valid_to < self.start_time) {
                return false;
            }
        }
        return edge.created_at <= self.end_time and
            (edge.valid_to >= self.start_time or self.include_invalidated_edges);
    }
};

pub const HistoryEntry = struct {
    timestamp: Timestamp,
    operation: HistoryOperation,
    entity_type: EntityType,
    entity_id: []const u8,
    version_before: ?usize,
    version_after: usize,
    allocator: Allocator,

    const Self = @This();

    pub const HistoryOperation = enum(u8) {
        create = 0,
        update = 1,
        rollback = 2,
        invalidate = 3,

        pub fn toString(self: HistoryOperation) []const u8 {
            return switch (self) {
                .create => "create",
                .update => "update",
                .rollback => "rollback",
                .invalidate => "invalidate",
            };
        }
    };

    pub const EntityType = enum(u8) {
        node = 0,
        edge = 1,
        snapshot = 2,

        pub fn toString(self: EntityType) []const u8 {
            return switch (self) {
                .node => "node",
                .edge => "edge",
                .snapshot => "snapshot",
            };
        }
    };

    pub fn init(
        allocator: Allocator,
        timestamp_ns: Timestamp,
        op: HistoryOperation,
        entity: EntityType,
        id: []const u8,
        version_before: ?usize,
        version_after: usize,
    ) !Self {
        return Self{
            .timestamp = timestamp_ns,
            .operation = op,
            .entity_type = entity,
            .entity_id = try allocator.dupe(u8, id),
            .version_before = version_before,
            .version_after = version_after,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.entity_id);
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        return Self{
            .timestamp = self.timestamp,
            .operation = self.operation,
            .entity_type = self.entity_type,
            .entity_id = try allocator.dupe(u8, self.entity_id),
            .version_before = self.version_before,
            .version_after = self.version_after,
            .allocator = allocator,
        };
    }
};

pub const TemporalGraph = struct {
    nodes: std.HashMap([]const u8, TemporalNode, StringContext, std.hash_map.default_max_load_percentage),
    edges: std.HashMap(EdgeKey, TemporalEdge, EdgeKeyContext, std.hash_map.default_max_load_percentage),
    current_time: Timestamp,
    snapshots: ArrayList(GraphSnapshot),
    history: ArrayList(HistoryEntry),
    next_snapshot_id: usize,
    next_edge_id: usize,
    allocator: Allocator,
    node_key_allocator: ArrayList([]const u8),
    edge_key_allocator: ArrayList([]const u8),

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .nodes = std.HashMap([]const u8, TemporalNode, StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .edges = std.HashMap(EdgeKey, TemporalEdge, EdgeKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
            .current_time = blk: {
                const ts = std.time.nanoTimestamp();
                break :blk if (ts > std.math.maxInt(Timestamp)) std.math.maxInt(Timestamp) else if (ts < std.math.minInt(Timestamp)) std.math.minInt(Timestamp) else @as(Timestamp, @intCast(ts));
            },
            .snapshots = ArrayList(GraphSnapshot).init(allocator),
            .history = ArrayList(HistoryEntry).init(allocator),
            .next_snapshot_id = 0,
            .next_edge_id = 0,
            .allocator = allocator,
            .node_key_allocator = ArrayList([]const u8).init(allocator),
            .edge_key_allocator = ArrayList([]const u8).init(allocator),
        };
    }

    pub fn initWithTime(allocator: Allocator, initial_time: Timestamp) Self {
        var graph = Self.init(allocator);
        graph.current_time = initial_time;
        return graph;
    }

    pub fn deinit(self: *Self) void {
        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            var node = entry.value_ptr;
            node.deinit();
        }
        self.nodes.deinit();

        for (self.node_key_allocator.items) |key| {
            self.allocator.free(key);
        }
        self.node_key_allocator.deinit();

        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            var edge = entry.value_ptr;
            edge.deinit();
        }
        self.edges.deinit();

        for (self.edge_key_allocator.items) |key| {
            self.allocator.free(key);
        }
        self.edge_key_allocator.deinit();

        for (self.snapshots.items) |*snapshot| {
            snapshot.deinit();
        }
        self.snapshots.deinit();

        for (self.history.items) |*entry| {
            entry.deinit();
        }
        self.history.deinit();
    }

    pub fn setCurrentTime(self: *Self, timestamp_ns: Timestamp) void {
        self.current_time = timestamp_ns;
    }

    pub fn advanceTime(self: *Self, delta_ns: Timestamp) void {
        self.current_time += delta_ns;
    }

    pub fn getCurrentTime(self: *const Self) Timestamp {
        return self.current_time;
    }

    pub fn addNode(
        self: *Self,
        node_id: []const u8,
        initial_state: QuantumState,
    ) !void {
        try self.addNodeAtTime(node_id, initial_state, self.current_time);
    }

    pub fn addNodeAtTime(
        self: *Self,
        node_id: []const u8,
        initial_state: QuantumState,
        timestamp_ns: Timestamp,
    ) !void {
        if (self.nodes.contains(node_id)) {
            return error.NodeAlreadyExists;
        }

        const id_copy = try self.allocator.dupe(u8, node_id);
        errdefer self.allocator.free(id_copy);
        try self.node_key_allocator.append(id_copy);

        const node = try TemporalNode.init(self.allocator, node_id, initial_state, timestamp_ns);
        try self.nodes.put(id_copy, node);

        const history_entry = try HistoryEntry.init(
            self.allocator,
            timestamp_ns,
            .create,
            .node,
            node_id,
            null,
            0,
        );
        try self.history.append(history_entry);
    }

    pub fn addEdge(
        self: *Self,
        source_id: []const u8,
        target_id: []const u8,
        weight: f64,
        quality: EdgeQuality,
    ) !void {
        try self.addEdgeAtTime(source_id, target_id, weight, quality, self.current_time);
    }

    pub fn addEdgeAtTime(
        self: *Self,
        source_id: []const u8,
        target_id: []const u8,
        weight: f64,
        quality: EdgeQuality,
        timestamp_ns: Timestamp,
    ) !void {
        const edge_key = EdgeKey{ .source = source_id, .target = target_id };
        if (self.edges.contains(edge_key)) {
            return error.EdgeAlreadyExists;
        }

        const source_copy = try self.allocator.dupe(u8, source_id);
        errdefer self.allocator.free(source_copy);
        try self.edge_key_allocator.append(source_copy);

        const target_copy = try self.allocator.dupe(u8, target_id);
        errdefer self.allocator.free(target_copy);
        try self.edge_key_allocator.append(target_copy);

        var edge_id_buf: [64]u8 = undefined;
        const edge_id = std.fmt.bufPrint(&edge_id_buf, "edge_{d}_{s}_{s}", .{
            self.next_edge_id,
            source_id,
            target_id,
        }) catch "edge_unknown";
        self.next_edge_id += 1;

        const new_key = EdgeKey{ .source = source_copy, .target = target_copy };
        const edge = try TemporalEdge.init(
            self.allocator,
            edge_id,
            source_id,
            target_id,
            weight,
            quality,
            timestamp_ns,
        );
        try self.edges.put(new_key, edge);

        const history_entry = try HistoryEntry.init(
            self.allocator,
            timestamp_ns,
            .create,
            .edge,
            edge_id,
            null,
            0,
        );
        try self.history.append(history_entry);
    }

    pub fn addEdgeWithTimeRange(
        self: *Self,
        source_id: []const u8,
        target_id: []const u8,
        weight: f64,
        quality: EdgeQuality,
        valid_from: Timestamp,
        valid_to: Timestamp,
    ) !void {
        const edge_key = EdgeKey{ .source = source_id, .target = target_id };
        if (self.edges.contains(edge_key)) {
            return error.EdgeAlreadyExists;
        }

        const source_copy = try self.allocator.dupe(u8, source_id);
        errdefer self.allocator.free(source_copy);
        try self.edge_key_allocator.append(source_copy);

        const target_copy = try self.allocator.dupe(u8, target_id);
        errdefer self.allocator.free(target_copy);
        try self.edge_key_allocator.append(target_copy);

        var edge_id_buf: [64]u8 = undefined;
        const edge_id = std.fmt.bufPrint(&edge_id_buf, "edge_{d}_{s}_{s}", .{
            self.next_edge_id,
            source_id,
            target_id,
        }) catch "edge_unknown";
        self.next_edge_id += 1;

        const new_key = EdgeKey{ .source = source_copy, .target = target_copy };
        const edge = try TemporalEdge.initWithTimeRange(
            self.allocator,
            edge_id,
            source_id,
            target_id,
            weight,
            quality,
            valid_from,
            valid_to,
        );
        try self.edges.put(new_key, edge);

        const history_entry = try HistoryEntry.init(
            self.allocator,
            valid_from,
            .create,
            .edge,
            edge_id,
            null,
            0,
        );
        try self.history.append(history_entry);
    }

    pub fn updateNode(
        self: *Self,
        node_id: []const u8,
        new_state: QuantumState,
    ) !usize {
        return self.updateNodeAtTime(node_id, new_state, self.current_time);
    }

    pub fn updateNodeAtTime(
        self: *Self,
        node_id: []const u8,
        new_state: QuantumState,
        timestamp_ns: Timestamp,
    ) !usize {
        if (self.nodes.getPtr(node_id)) |node| {
            const version_before = node.current_version;
            const new_version = try node.addVersion(new_state, timestamp_ns);

            const history_entry = try HistoryEntry.init(
                self.allocator,
                timestamp_ns,
                .update,
                .node,
                node_id,
                version_before,
                new_version,
            );
            try self.history.append(history_entry);

            return new_version;
        }
        return error.NodeNotFound;
    }

    pub fn updateEdge(
        self: *Self,
        source_id: []const u8,
        target_id: []const u8,
        new_weight: f64,
        new_quality: EdgeQuality,
    ) !usize {
        return self.updateEdgeAtTime(source_id, target_id, new_weight, new_quality, self.current_time);
    }

    pub fn updateEdgeAtTime(
        self: *Self,
        source_id: []const u8,
        target_id: []const u8,
        new_weight: f64,
        new_quality: EdgeQuality,
        timestamp_ns: Timestamp,
    ) !usize {
        const edge_key = EdgeKey{ .source = source_id, .target = target_id };
        if (self.edges.getPtr(edge_key)) |edge| {
            const version_before = edge.current_version;
            const new_version = try edge.addVersion(new_weight, new_quality, timestamp_ns);

            const history_entry = try HistoryEntry.init(
                self.allocator,
                timestamp_ns,
                .update,
                .edge,
                edge.edge_id,
                version_before,
                new_version,
            );
            try self.history.append(history_entry);

            return new_version;
        }
        return error.EdgeNotFound;
    }

    pub fn invalidateEdge(
        self: *Self,
        source_id: []const u8,
        target_id: []const u8,
    ) !void {
        try self.invalidateEdgeAtTime(source_id, target_id, self.current_time);
    }

    pub fn invalidateEdgeAtTime(
        self: *Self,
        source_id: []const u8,
        target_id: []const u8,
        timestamp_ns: Timestamp,
    ) !void {
        const edge_key = EdgeKey{ .source = source_id, .target = target_id };
        if (self.edges.getPtr(edge_key)) |edge| {
            edge.invalidate(timestamp_ns);

            const history_entry = try HistoryEntry.init(
                self.allocator,
                timestamp_ns,
                .invalidate,
                .edge,
                edge.edge_id,
                edge.current_version,
                edge.current_version,
            );
            try self.history.append(history_entry);
        } else {
            return error.EdgeNotFound;
        }
    }

    pub fn getNode(self: *Self, node_id: []const u8) ?*TemporalNode {
        return self.nodes.getPtr(node_id);
    }

    pub fn getNodeConst(self: *const Self, node_id: []const u8) ?*const TemporalNode {
        if (self.nodes.getPtr(node_id)) |ptr| {
            return ptr;
        }
        return null;
    }

    pub fn getNodeAt(self: *const Self, node_id: []const u8, timestamp_ns: Timestamp) ?*const NodeVersion {
        if (self.nodes.getPtr(node_id)) |node| {
            return node.getVersionAt(timestamp_ns);
        }
        return null;
    }

    pub fn getEdge(self: *Self, source_id: []const u8, target_id: []const u8) ?*TemporalEdge {
        const edge_key = EdgeKey{ .source = source_id, .target = target_id };
        return self.edges.getPtr(edge_key);
    }

    pub fn getEdgeConst(self: *const Self, source_id: []const u8, target_id: []const u8) ?*const TemporalEdge {
        const edge_key = EdgeKey{ .source = source_id, .target = target_id };
        if (self.edges.getPtr(edge_key)) |ptr| {
            return ptr;
        }
        return null;
    }

    pub fn getEdgeAt(
        self: *const Self,
        source_id: []const u8,
        target_id: []const u8,
        timestamp_ns: Timestamp,
    ) ?*const EdgeVersion {
        const edge_key = EdgeKey{ .source = source_id, .target = target_id };
        if (self.edges.getPtr(edge_key)) |edge| {
            return edge.getVersionAt(timestamp_ns);
        }
        return null;
    }

    pub fn createSnapshot(self: *Self) !usize {
        return self.createSnapshotAtTime(self.current_time);
    }

    pub fn createSnapshotAtTime(self: *Self, timestamp_ns: Timestamp) !usize {
        const snapshot_id = self.next_snapshot_id;
        self.next_snapshot_id += 1;

        var snapshot = GraphSnapshot.init(self.allocator, snapshot_id, timestamp_ns);

        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            if (node.getVersionAt(timestamp_ns)) |version| {
                try snapshot.recordNodeVersion(node.node_id, version.version);
            }
        }

        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            const edge = entry.value_ptr;
            if (edge.isValidAt(timestamp_ns)) {
                if (edge.getVersionAt(timestamp_ns)) |version| {
                    try snapshot.recordEdgeVersion(edge.getEdgeKey(), version.version);
                }
            }
        }

        try self.snapshots.append(snapshot);

        const history_entry = try HistoryEntry.init(
            self.allocator,
            timestamp_ns,
            .create,
            .snapshot,
            "snapshot",
            null,
            snapshot_id,
        );
        try self.history.append(history_entry);

        return snapshot_id;
    }

    pub fn getSnapshot(self: *const Self, snapshot_id: usize) ?*const GraphSnapshot {
        for (self.snapshots.items) |*snapshot| {
            if (snapshot.snapshot_id == snapshot_id) {
                return snapshot;
            }
        }
        return null;
    }

    pub fn restoreSnapshot(self: *Self, snapshot_id: usize) !void {
        const snapshot = self.getSnapshot(snapshot_id) orelse return error.SnapshotNotFound;

        var node_iter = snapshot.node_versions.iterator();
        while (node_iter.next()) |entry| {
            if (self.nodes.getPtr(entry.key_ptr.*)) |node| {
                const target_version = entry.value_ptr.*;
                if (!node.rollback(target_version)) {
                    return error.InvalidVersionRollback;
                }
            }
        }

        var edge_iter = snapshot.edge_versions.iterator();
        while (edge_iter.next()) |entry| {
            if (self.edges.getPtr(entry.key_ptr.*)) |edge| {
                const target_version = entry.value_ptr.*;
                if (!edge.rollback(target_version)) {
                    return error.InvalidVersionRollback;
                }
            }
        }

        const history_entry = try HistoryEntry.init(
            self.allocator,
            self.current_time,
            .rollback,
            .snapshot,
            "snapshot_restore",
            null,
            snapshot_id,
        );
        try self.history.append(history_entry);
    }

    pub fn getHistory(self: *const Self, allocator: Allocator) !ArrayList(HistoryEntry) {
        var result = ArrayList(HistoryEntry).init(allocator);
        for (self.history.items) |*entry| {
            const cloned = try entry.clone(allocator);
            try result.append(cloned);
        }
        return result;
    }

    pub fn getHistoryInRange(
        self: *const Self,
        start_time: Timestamp,
        end_time: Timestamp,
        allocator: Allocator,
    ) !ArrayList(HistoryEntry) {
        var result = ArrayList(HistoryEntry).init(allocator);
        for (self.history.items) |*entry| {
            if (entry.timestamp >= start_time and entry.timestamp <= end_time) {
                const cloned = try entry.clone(allocator);
                try result.append(cloned);
            }
        }
        return result;
    }

    pub fn queryTimeRange(
        self: *const Self,
        start_time: Timestamp,
        end_time: Timestamp,
    ) !TemporalQueryResult {
        const query = TemporalQuery.init(self.allocator, start_time, end_time);
        return query.execute(self);
    }

    pub fn queryWithFilters(
        self: *const Self,
        start_time: Timestamp,
        end_time: Timestamp,
        node_filter: ?NodeFilterFn,
        edge_filter: ?EdgeFilterFn,
    ) !TemporalQueryResult {
        const query = TemporalQuery.initWithFilters(
            self.allocator,
            start_time,
            end_time,
            node_filter,
            edge_filter,
        );
        return query.execute(self);
    }

    pub fn nodeCount(self: *const Self) usize {
        return self.nodes.count();
    }

    pub fn edgeCount(self: *const Self) usize {
        return self.edges.count();
    }

    pub fn snapshotCount(self: *const Self) usize {
        return self.snapshots.items.len;
    }

    pub fn historyCount(self: *const Self) usize {
        return self.history.items.len;
    }

    pub fn getValidEdgesAt(
        self: *const Self,
        timestamp_ns: Timestamp,
        allocator: Allocator,
    ) !ArrayList(*const TemporalEdge) {
        var result = ArrayList(*const TemporalEdge).init(allocator);
        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            const edge = entry.value_ptr;
            if (edge.isValidAt(timestamp_ns)) {
                try result.append(edge);
            }
        }
        return result;
    }

    pub fn getNodeNeighborsAt(
        self: *const Self,
        node_id: []const u8,
        timestamp_ns: Timestamp,
        allocator: Allocator,
    ) !ArrayList([]const u8) {
        var neighbors = ArrayList([]const u8).init(allocator);
        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            const edge = entry.value_ptr;
            if (edge.isValidAt(timestamp_ns)) {
                if (std.mem.eql(u8, edge.source, node_id)) {
                    try neighbors.append(edge.target);
                } else if (std.mem.eql(u8, edge.target, node_id)) {
                    try neighbors.append(edge.source);
                }
            }
        }
        return neighbors;
    }

    pub fn computeGraphStateAt(
        self: *const Self,
        timestamp_ns: Timestamp,
        allocator: Allocator,
    ) !struct {
        node_states: StringHashMap(QuantumState),
        edge_weights: std.HashMap(EdgeKey, f64, EdgeKeyContext, std.hash_map.default_max_load_percentage),
    } {
        var node_states = StringHashMap(QuantumState).init(allocator);
        var edge_weights = std.HashMap(EdgeKey, f64, EdgeKeyContext, std.hash_map.default_max_load_percentage).init(allocator);

        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            if (node.getVersionAt(timestamp_ns)) |version| {
                try node_states.put(node.node_id, version.data);
            }
        }

        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            const edge = entry.value_ptr;
            if (edge.isValidAt(timestamp_ns)) {
                if (edge.getVersionAt(timestamp_ns)) |version| {
                    try edge_weights.put(edge.getEdgeKey(), version.weight);
                }
            }
        }

        return .{
            .node_states = node_states,
            .edge_weights = edge_weights,
        };
    }

    pub fn recordInferenceObservation(self: *Self, node_id: []const u8, state: QuantumState, timestamp_ns: Timestamp) !usize {
        if (self.nodes.getPtr(node_id)) |node| {
            return node.addVersion(state, timestamp_ns);
        }
        try self.addNodeAtTime(node_id, state, timestamp_ns);
        return 0;
    }

    pub fn getInferenceContext(self: *const Self, node_id: []const u8) ?QuantumState {
        if (self.nodes.getPtr(node_id)) |node| {
            return node.getCurrentState();
        }
        return null;
    }

    pub fn getInferenceEdgeWeight(self: *const Self, source_id: []const u8, target_id: []const u8) ?f64 {
        const edge_key = EdgeKey{ .source = source_id, .target = target_id };
        if (self.edges.getPtr(edge_key)) |edge| {
            return edge.getCurrentWeight();
        }
        return null;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var new_graph = Self.init(allocator);
        new_graph.current_time = self.current_time;
        new_graph.next_snapshot_id = self.next_snapshot_id;
        new_graph.next_edge_id = self.next_edge_id;
        errdefer new_graph.deinit();

        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr;
            const cloned_node = try node.clone(allocator);
            const id_copy = try allocator.dupe(u8, node.node_id);
            try new_graph.node_key_allocator.append(id_copy);
            try new_graph.nodes.put(id_copy, cloned_node);
        }

        var edge_iter = self.edges.iterator();
        while (edge_iter.next()) |entry| {
            const edge = entry.value_ptr;
            const cloned_edge = try edge.clone(allocator);
            const source_copy = try allocator.dupe(u8, edge.source);
            try new_graph.edge_key_allocator.append(source_copy);
            const target_copy = try allocator.dupe(u8, edge.target);
            try new_graph.edge_key_allocator.append(target_copy);
            const new_key = EdgeKey{ .source = source_copy, .target = target_copy };
            try new_graph.edges.put(new_key, cloned_edge);
        }

        for (self.snapshots.items) |*snapshot| {
            const cloned_snapshot = try snapshot.clone(allocator);
            try new_graph.snapshots.append(cloned_snapshot);
        }

        for (self.history.items) |*entry| {
            const cloned_entry = try entry.clone(allocator);
            try new_graph.history.append(cloned_entry);
        }

        return new_graph;
    }

    pub fn createInferenceContext(self: *Self) InferenceContext {
        return InferenceContext.init(self);
    }

    pub fn inferenceRecordNode(self: *Self, node_id: []const u8, state: QuantumState) !usize {
        return self.recordInferenceObservation(node_id, state, self.current_time);
    }

    pub fn inferenceQueryEdgeWeight(self: *const Self, source_id: []const u8, target_id: []const u8) ?f64 {
        return self.getInferenceEdgeWeight(source_id, target_id);
    }

    pub fn inferenceQueryNodeState(self: *const Self, node_id: []const u8) ?QuantumState {
        return self.getInferenceContext(node_id);
    }

    pub fn inferenceUpdateAndSnapshot(self: *Self, node_id: []const u8, new_state: QuantumState) !struct { version: usize, snapshot_id: usize } {
        const version = try self.recordInferenceObservation(node_id, new_state, self.current_time);
        const snapshot_id = try self.createSnapshot();
        return .{ .version = version, .snapshot_id = snapshot_id };
    }
};

pub fn defaultNodeFilter(node: *const TemporalNode) bool {
    _ = node;
    return true;
}

pub fn defaultEdgeFilter(edge: *const TemporalEdge) bool {
    _ = edge;
    return true;
}

pub fn filterByEntanglement(node: *const TemporalNode) bool {
    if (node.getCurrentVersion()) |version| {
        return version.data.entanglement_degree > 0.5;
    }
    return false;
}

pub fn filterBySuperposition(edge: *const TemporalEdge) bool {
    if (edge.getCurrentVersion()) |version| {
        return version.quality == .superposition;
    }
    return false;
}

pub fn filterByCoherence(edge: *const TemporalEdge) bool {
    if (edge.getCurrentVersion()) |version| {
        return version.quality == .coherent or version.quality == .superposition;
    }
    return false;
}

pub fn getCurrentTimestamp() Timestamp {
    const ts = std.time.nanoTimestamp();
    return if (ts > std.math.maxInt(Timestamp)) std.math.maxInt(Timestamp) else if (ts < std.math.minInt(Timestamp)) std.math.minInt(Timestamp) else @as(Timestamp, @intCast(ts));
}

pub fn timestampToMillis(timestamp_ns: Timestamp) i64 {
    return @divFloor(timestamp_ns, 1_000_000);
}

pub fn millisToTimestamp(millis: i64) Timestamp {
    return millis * 1_000_000;
}

pub fn timestampToSeconds(timestamp_ns: Timestamp) f64 {
    return @as(f64, @floatFromInt(timestamp_ns)) / 1_000_000_000.0;
}

pub fn secondsToTimestamp(seconds: f64) Timestamp {
    return @intFromFloat(seconds * 1_000_000_000.0);
}

pub const InferenceContext = struct {
    graph: *TemporalGraph,

    pub fn init(graph: *TemporalGraph) InferenceContext {
        return InferenceContext{ .graph = graph };
    }

    pub fn getNodeState(self: *const InferenceContext, node_id: []const u8) ?QuantumState {
        return self.graph.getInferenceContext(node_id);
    }

    pub fn applyInferenceUpdate(self: *InferenceContext, node_id: []const u8, new_state: QuantumState) !usize {
        return self.graph.updateNode(node_id, new_state);
    }

    pub fn queryActiveEdges(self: *const InferenceContext, node_id: []const u8, allocator: Allocator) !ArrayList(*const TemporalEdge) {
        var result = ArrayList(*const TemporalEdge).init(allocator);
        errdefer result.deinit();
        var iter = self.graph.edges.iterator();
        while (iter.next()) |entry| {
            const edge = entry.value_ptr;
            if (edge.isValidAt(self.graph.current_time)) {
                if (std.mem.eql(u8, edge.source, node_id) or std.mem.eql(u8, edge.target, node_id)) {
                    try result.append(edge);
                }
            }
        }
        return result;
    }

    pub fn getNodeVersionAtTime(self: *const InferenceContext, node_id: []const u8, timestamp_ns: Timestamp) ?*const NodeVersion {
        return self.graph.getNodeAt(node_id, timestamp_ns);
    }

    pub fn getEdgeVersionAtTime(self: *const InferenceContext, source_id: []const u8, target_id: []const u8, timestamp_ns: Timestamp) ?*const EdgeVersion {
        return self.graph.getEdgeAt(source_id, target_id, timestamp_ns);
    }

    pub fn snapshotForInference(self: *InferenceContext) !usize {
        return self.graph.createSnapshot();
    }

    pub fn getSnapshotForInference(self: *InferenceContext, snapshot_id: usize) ?*const GraphSnapshot {
        return self.graph.getSnapshot(snapshot_id);
    }

    pub fn recordObservation(self: *InferenceContext, node_id: []const u8, state: QuantumState) !usize {
        return self.graph.recordInferenceObservation(node_id, state, self.graph.current_time);
    }

    pub fn getEdgeWeightForInference(self: *const InferenceContext, source_id: []const u8, target_id: []const u8) ?f64 {
        return self.graph.getInferenceEdgeWeight(source_id, target_id);
    }

    pub fn advanceTimeBy(self: *InferenceContext, delta_ns: Timestamp) void {
        self.graph.advanceTime(delta_ns);
    }

    pub fn queryGraphState(self: *const InferenceContext, start_time: Timestamp, end_time: Timestamp) !TemporalQueryResult {
        return self.graph.queryTimeRange(start_time, end_time);
    }
};

test "NodeVersion basic operations" {
    const allocator = std.testing.allocator;

    const state = QuantumState.init(0.707, 0.0, 0.707, 0.0, 0.0, 0.5);
    var version = NodeVersion.init(allocator, 0, 1000, state);
    defer version.deinit();

    try std.testing.expect(version.version == 0);
    try std.testing.expect(version.timestamp == 1000);

    try version.setProperty("key1", "value1");
    const val = version.getProperty("key1");
    try std.testing.expect(val != null);
    try std.testing.expectEqualStrings("value1", val.?);

    var cloned = try version.clone(allocator);
    defer cloned.deinit();
    try std.testing.expect(cloned.version == 0);
}

test "EdgeVersion basic operations" {
    const allocator = std.testing.allocator;

    var version = EdgeVersion.init(allocator, 0, 2000, 1.5, .entangled);
    defer version.deinit();

    try std.testing.expect(version.version == 0);
    try std.testing.expect(version.weight == 1.5);
    try std.testing.expect(version.quality == .entangled);

    var cloned = try version.clone(allocator);
    defer cloned.deinit();
    try std.testing.expect(cloned.weight == 1.5);
}

test "TemporalNode versioning" {
    const allocator = std.testing.allocator;

    const initial_state = QuantumState.init(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var node = try TemporalNode.init(allocator, "test_node", initial_state, 1000);
    defer node.deinit();

    try std.testing.expect(node.versionCount() == 1);
    try std.testing.expect(node.current_version == 0);

    const new_state = QuantumState.init(0.5, 0.0, 0.5, 0.0, 0.1, 0.3);
    const v1 = try node.addVersion(new_state, 2000);
    try std.testing.expect(v1 == 1);
    try std.testing.expect(node.current_version == 1);

    const version_at = node.getVersionAt(1500);
    try std.testing.expect(version_at != null);
    try std.testing.expect(version_at.?.version == 0);

    try std.testing.expect(node.rollback(0));
    try std.testing.expect(node.current_version == 0);
}

test "TemporalEdge time range validity" {
    const allocator = std.testing.allocator;

    var edge = try TemporalEdge.initWithTimeRange(
        allocator,
        "edge_1",
        "node_a",
        "node_b",
        1.0,
        .coherent,
        1000,
        5000,
    );
    defer edge.deinit();

    try std.testing.expect(edge.isValidAt(1000));
    try std.testing.expect(edge.isValidAt(3000));
    try std.testing.expect(edge.isValidAt(5000));
    try std.testing.expect(!edge.isValidAt(500));
    try std.testing.expect(!edge.isValidAt(6000));

    edge.invalidate(4000);
    try std.testing.expect(edge.isValidAt(3000));
    try std.testing.expect(!edge.isValidAt(4500));
}

test "TemporalGraph operations" {
    const allocator = std.testing.allocator;

    var graph = TemporalGraph.initWithTime(allocator, 1000);
    defer graph.deinit();

    const state_a = QuantumState.init(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const state_b = QuantumState.init(0.707, 0.0, 0.707, 0.0, 0.0, 0.5);

    try graph.addNodeAtTime("node_a", state_a, 1000);
    try graph.addNodeAtTime("node_b", state_b, 1000);

    try std.testing.expect(graph.nodeCount() == 2);

    try graph.addEdgeAtTime("node_a", "node_b", 1.0, .entangled, 1500);
    try std.testing.expect(graph.edgeCount() == 1);

    const new_state = QuantumState.init(0.5, 0.0, 0.5, 0.0, 0.2, 0.8);
    const v1 = try graph.updateNodeAtTime("node_a", new_state, 2000);
    try std.testing.expect(v1 == 1);

    const node_version = graph.getNodeAt("node_a", 1500);
    try std.testing.expect(node_version != null);
    try std.testing.expect(node_version.?.version == 0);
}

test "TemporalGraph snapshots" {
    const allocator = std.testing.allocator;

    var graph = TemporalGraph.initWithTime(allocator, 1000);
    defer graph.deinit();

    const state = QuantumState.init(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    try graph.addNodeAtTime("node_a", state, 1000);

    const snapshot_id = try graph.createSnapshotAtTime(1000);
    try std.testing.expect(snapshot_id == 0);
    try std.testing.expect(graph.snapshotCount() == 1);

    const new_state = QuantumState.init(0.5, 0.0, 0.5, 0.0, 0.0, 0.0);
    _ = try graph.updateNodeAtTime("node_a", new_state, 2000);

    if (graph.getNode("node_a")) |node| {
        try std.testing.expect(node.current_version == 1);
    }

    try graph.restoreSnapshot(snapshot_id);

    if (graph.getNode("node_a")) |node| {
        try std.testing.expect(node.current_version == 0);
    }
}

test "TemporalQuery execution" {
    const allocator = std.testing.allocator;

    var graph = TemporalGraph.initWithTime(allocator, 1000);
    defer graph.deinit();

    const state = QuantumState.init(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    try graph.addNodeAtTime("node_a", state, 1000);
    try graph.addNodeAtTime("node_b", state, 2000);
    try graph.addEdgeAtTime("node_a", "node_b", 1.0, .coherent, 1500);

    var result = try graph.queryTimeRange(500, 3000);
    defer result.deinit();

    try std.testing.expect(result.nodeCount() == 2);
    try std.testing.expect(result.edgeCount() == 1);
}

test "timestamp utilities" {
    const ns: Timestamp = 1_500_000_000;
    const ms = timestampToMillis(ns);
    try std.testing.expect(ms == 1500);

    const back_to_ns = millisToTimestamp(ms);
    try std.testing.expect(back_to_ns == 1_500_000_000);

    const seconds = timestampToSeconds(ns);
    try std.testing.expect(seconds == 1.5);
}

