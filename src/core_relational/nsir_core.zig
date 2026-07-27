const std = @import("std");

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Complex = std.math.Complex;
const Mutex = std.Thread.Mutex;
const core_tensor = @import("../core/tensor.zig");
const core_memory = @import("../core/memory.zig");

pub const EdgeQuality = enum(u8) {
    superposition = 0,
    entangled = 1,
    coherent = 2,
    collapsed = 3,
    fractal = 4,

    pub fn toString(self: EdgeQuality) []const u8 {
        return switch (self) {
            .superposition => "superposition",
            .entangled => "entangled",
            .coherent => "coherent",
            .collapsed => "collapsed",
            .fractal => "fractal",
        };
    }

    pub fn fromString(s: []const u8) ?EdgeQuality {
        if (std.mem.eql(u8, s, "superposition")) return .superposition;
        if (std.mem.eql(u8, s, "entangled")) return .entangled;
        if (std.mem.eql(u8, s, "coherent")) return .coherent;
        if (std.mem.eql(u8, s, "collapsed")) return .collapsed;
        if (std.mem.eql(u8, s, "fractal")) return .fractal;
        return null;
    }
};

fn dupeBytes(allocator: Allocator, b: []const u8) ![]u8 {
    return try allocator.dupe(u8, b);
}

fn freeMapStringBytes(map: *StringHashMap([]u8), allocator: Allocator) void {
    var it = map.iterator();
    while (it.next()) |e| {
        allocator.free(e.key_ptr.*);
        allocator.free(e.value_ptr.*);
    }
    map.deinit();
}

fn deinitNodeMap(map: *StringHashMap(Node), allocator: Allocator) void {
    var it = map.iterator();
    while (it.next()) |e| {
        allocator.free(e.key_ptr.*);
        e.value_ptr.deinit();
    }
    map.deinit();
}

fn clearNodeMapRetainingCapacity(map: *StringHashMap(Node), allocator: Allocator) void {
    var it = map.iterator();
    while (it.next()) |e| {
        allocator.free(e.key_ptr.*);
        e.value_ptr.deinit();
    }
    map.clearRetainingCapacity();
}

fn putOwnedStringBytes(map: *StringHashMap([]u8), allocator: Allocator, key: []const u8, value: []const u8) !void {
    if (map.fetchRemove(key)) |removed| {
        allocator.free(removed.key);
        allocator.free(removed.value);
    }
    const k = try allocator.dupe(u8, key);
    const v = allocator.dupe(u8, value) catch |err| {
        allocator.free(k);
        return err;
    };
    map.put(k, v) catch |err| {
        allocator.free(k);
        allocator.free(v);
        return err;
    };
}

pub const Qubit = struct {
    a: Complex(f64),
    b: Complex(f64),

    pub fn init(a: Complex(f64), b: Complex(f64)) Qubit {
        var q = Qubit{ .a = a, .b = b };
        q.normalizeInPlace();
        return q;
    }

    pub fn initBasis0() Qubit {
        return Qubit{ .a = Complex(f64).init(1.0, 0.0), .b = Complex(f64).init(0.0, 0.0) };
    }

    pub fn initBasis1() Qubit {
        return Qubit{ .a = Complex(f64).init(0.0, 0.0), .b = Complex(f64).init(1.0, 0.0) };
    }

    pub fn normSquared(self: Qubit) f64 {
        return (self.a.re * self.a.re + self.a.im * self.a.im) + (self.b.re * self.b.re + self.b.im * self.b.im);
    }

    pub fn normalizeInPlace(self: *Qubit) void {
        const ns = self.normSquared();
        if (std.math.isNan(ns) or !(ns > 0.0) or std.math.isInf(ns)) {
            self.* = Qubit.initBasis0();
            return;
        }
        const inv = 1.0 / std.math.sqrt(ns);
        const s = Complex(f64).init(inv, 0.0);
        self.a = self.a.mul(s);
        self.b = self.b.mul(s);
    }

    pub fn prob0(self: Qubit) f64 {
        var q = self;
        q.normalizeInPlace();
        return std.math.clamp(q.a.re * q.a.re + q.a.im * q.a.im, 0.0, 1.0);
    }

    pub fn prob1(self: Qubit) f64 {
        var q = self;
        q.normalizeInPlace();
        return std.math.clamp(q.b.re * q.b.re + q.b.im * q.b.im, 0.0, 1.0);
    }
};

pub const Node = struct {
    id: []u8,
    data: []u8,
    qubit: Qubit,
    phase: f64,
    coherence: f64,
    metadata: StringHashMap([]u8),
    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, data: []const u8, qubit: Qubit, phase: f64) !Node {
        const dup_id = try dupeBytes(allocator, id);
        errdefer allocator.free(dup_id);
        const dup_data = try dupeBytes(allocator, data);
        errdefer allocator.free(dup_data);
        return Node{
            .id = dup_id,
            .data = dup_data,
            .qubit = qubit,
            .phase = phase,
            .coherence = 1.0,
            .metadata = StringHashMap([]u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn magnitude(self: Node) f64 {
        return @sqrt(self.qubit.normSquared());
    }

    pub fn quantumState(self: Node) Complex(f64) {
        return self.qubit.a;
    }

    pub fn deinit(self: *Node) void {
        self.allocator.free(self.id);
        self.allocator.free(self.data);
        freeMapStringBytes(&self.metadata, self.allocator);
    }

    pub fn clone(self: *const Node, allocator: Allocator) !Node {
        const dup_id = try dupeBytes(allocator, self.id);
        errdefer allocator.free(dup_id);

        const dup_data = try dupeBytes(allocator, self.data);
        errdefer allocator.free(dup_data);

        var meta = StringHashMap([]u8).init(allocator);
        errdefer freeMapStringBytes(&meta, allocator);

        var it = self.metadata.iterator();
        while (it.next()) |e| {
            const k = try dupeBytes(allocator, e.key_ptr.*);
            const v = dupeBytes(allocator, e.value_ptr.*) catch |err| {
                allocator.free(k);
                return err;
            };
            meta.put(k, v) catch |err| {
                allocator.free(k);
                allocator.free(v);
                return err;
            };
        }

        return Node{
            .id = dup_id,
            .data = dup_data,
            .qubit = self.qubit,
            .phase = self.phase,
            .coherence = self.coherence,
            .metadata = meta,
            .allocator = allocator,
        };
    }

    pub fn setMetadata(self: *Node, key: []const u8, value: []const u8) !void {
        try putOwnedStringBytes(&self.metadata, self.allocator, key, value);
    }

    pub fn getMetadata(self: *const Node, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }
};

pub const Edge = struct {
    source: []const u8,
    target: []const u8,
    quality: EdgeQuality,
    weight: f64,
    quantum_correlation: Complex(f64),
    fractal_dimension: f64,
    metadata: StringHashMap([]u8),
    allocator: Allocator,
    owns_source: bool,
    owns_target: bool,

    pub fn init(
        allocator: Allocator,
        source: []const u8,
        target: []const u8,
        quality: EdgeQuality,
        weight: f64,
        quantum_correlation: Complex(f64),
        fractal_dimension: f64,
    ) !Edge {
        const dup_source = try allocator.dupe(u8, source);
        errdefer allocator.free(dup_source);
        const dup_target = try allocator.dupe(u8, target);
        errdefer allocator.free(dup_target);
        return Edge{
            .source = dup_source,
            .target = dup_target,
            .quality = quality,
            .weight = weight,
            .quantum_correlation = quantum_correlation,
            .fractal_dimension = fractal_dimension,
            .metadata = StringHashMap([]u8).init(allocator),
            .allocator = allocator,
            .owns_source = true,
            .owns_target = true,
        };
    }

    pub fn initBorrowed(
        allocator: Allocator,
        source: []const u8,
        target: []const u8,
        quality: EdgeQuality,
        weight: f64,
        quantum_correlation: Complex(f64),
        fractal_dimension: f64,
    ) Edge {
        return Edge{
            .source = source,
            .target = target,
            .quality = quality,
            .weight = weight,
            .quantum_correlation = quantum_correlation,
            .fractal_dimension = fractal_dimension,
            .metadata = StringHashMap([]u8).init(allocator),
            .allocator = allocator,
            .owns_source = false,
            .owns_target = false,
        };
    }

    pub fn deinit(self: *Edge) void {
        if (self.owns_source) {
            self.allocator.free(@as([]u8, @constCast(self.source)));
        }
        if (self.owns_target) {
            self.allocator.free(@as([]u8, @constCast(self.target)));
        }
        freeMapStringBytes(&self.metadata, self.allocator);
    }

    pub fn clone(self: *const Edge, allocator: Allocator) !Edge {
        const dup_source = try dupeBytes(allocator, self.source);
        errdefer allocator.free(dup_source);
        const dup_target = try dupeBytes(allocator, self.target);
        errdefer allocator.free(dup_target);

        var e = Edge{
            .source = dup_source,
            .target = dup_target,
            .quality = self.quality,
            .weight = self.weight,
            .quantum_correlation = self.quantum_correlation,
            .fractal_dimension = self.fractal_dimension,
            .metadata = StringHashMap([]u8).init(allocator),
            .allocator = allocator,
            .owns_source = true,
            .owns_target = true,
        };
        errdefer freeMapStringBytes(&e.metadata, allocator);

        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            const k = try dupeBytes(allocator, entry.key_ptr.*);
            const v = dupeBytes(allocator, entry.value_ptr.*) catch |err| {
                allocator.free(k);
                return err;
            };
            e.metadata.put(k, v) catch |err| {
                allocator.free(k);
                allocator.free(v);
                return err;
            };
        }

        return e;
    }

    pub fn setSource(self: *Edge, source: []const u8) !void {
        if (self.owns_source) {
            self.allocator.free(@as([]u8, @constCast(self.source)));
            self.owns_source = false;
        }
        const dup = try self.allocator.dupe(u8, source);
        self.source = dup;
        self.owns_source = true;
    }

    pub fn setTarget(self: *Edge, target: []const u8) !void {
        if (self.owns_target) {
            self.allocator.free(@as([]u8, @constCast(self.target)));
            self.owns_target = false;
        }
        const dup = try self.allocator.dupe(u8, target);
        self.target = dup;
        self.owns_target = true;
    }

    pub fn setMetadata(self: *Edge, key: []const u8, value: []const u8) !void {
        try putOwnedStringBytes(&self.metadata, self.allocator, key, value);
    }

    pub fn getMetadata(self: *const Edge, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }

    pub fn correlationMagnitude(self: *const Edge) f64 {
        const qc = self.quantum_correlation;
        return @sqrt(qc.re * qc.re + qc.im * qc.im);
    }
};

pub const EdgeKey = struct {
    source: []const u8,
    target: []const u8,
};

fn hashSlice(h: *std.hash.Wyhash, s: []const u8) void {
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @as(u64, @intCast(s.len)), .little);
    h.update(&len_buf);
    h.update(s);
}

pub const EdgeKeyContext = struct {
    pub fn hash(_: @This(), k: EdgeKey) u64 {
        var h = std.hash.Wyhash.init(0);
        hashSlice(&h, k.source);
        hashSlice(&h, k.target);
        return h.final();
    }

    pub fn eql(_: @This(), a: EdgeKey, b: EdgeKey) bool {
        return std.mem.eql(u8, a.source, b.source) and std.mem.eql(u8, a.target, b.target);
    }
};

pub const PairKey = struct {
    a: []const u8,
    b: []const u8,
};

pub const PairKeyContext = struct {
    pub fn hash(_: @This(), k: PairKey) u64 {
        var h = std.hash.Wyhash.init(1);
        hashSlice(&h, k.a);
        hashSlice(&h, k.b);
        return h.final();
    }

    pub fn eql(_: @This(), x: PairKey, y: PairKey) bool {
        return std.mem.eql(u8, x.a, y.a) and std.mem.eql(u8, x.b, y.b);
    }
};

pub const TwoQubit = struct {
    amps: [4]Complex(f64),

    pub fn initBellPhiPlus() TwoQubit {
        const inv_sqrt2 = 1.0 / std.math.sqrt(2.0);
        return TwoQubit{
            .amps = .{
                Complex(f64).init(inv_sqrt2, 0.0),
                Complex(f64).init(0.0, 0.0),
                Complex(f64).init(0.0, 0.0),
                Complex(f64).init(inv_sqrt2, 0.0),
            },
        };
    }

    pub fn normalizeInPlace(self: *TwoQubit) void {
        var ns: f64 = 0.0;
        for (self.amps) |amp| {
            ns += amp.re * amp.re + amp.im * amp.im;
        }
        if (std.math.isNan(ns) or !(ns > 0.0) or std.math.isInf(ns)) {
            self.* = TwoQubit.initBellPhiPlus();
            return;
        }
        const inv = 1.0 / std.math.sqrt(ns);
        const s = Complex(f64).init(inv, 0.0);
        for (&self.amps) |*amp| {
            amp.* = amp.mul(s);
        }
    }
};

pub const Gate = *const fn (q: Qubit) Qubit;

pub fn hadamardGate(q: Qubit) Qubit {
    const inv_sqrt2 = 1.0 / std.math.sqrt(2.0);
    const s = Complex(f64).init(inv_sqrt2, 0.0);
    const a = q.a.add(q.b).mul(s);
    const b = q.a.sub(q.b).mul(s);
    return Qubit.init(a, b);
}

pub fn pauliXGate(q: Qubit) Qubit {
    return Qubit.init(q.b, q.a);
}

pub fn pauliYGate(q: Qubit) Qubit {
    const i = Complex(f64).init(0.0, 1.0);
    const minus_i = Complex(f64).init(0.0, -1.0);
    return Qubit.init(minus_i.mul(q.b), i.mul(q.a));
}

pub fn pauliZGate(q: Qubit) Qubit {
    return Qubit.init(q.a, q.b.mul(Complex(f64).init(-1.0, 0.0)));
}

pub fn identityGate(q: Qubit) Qubit {
    return q;
}

pub fn phaseGate(comptime phase: f64) Gate {
    const S = struct {
        const c_val = std.math.cos(phase);
        const s_val = std.math.sin(phase);
        fn apply(q: Qubit) Qubit {
            const factor = Complex(f64).init(c_val, s_val);
            return Qubit.init(q.a, q.b.mul(factor));
        }
    };
    return &S.apply;
}

pub fn runtimePhaseGate(q: Qubit, phase: f64) Qubit {
    const factor = Complex(f64).init(std.math.cos(phase), std.math.sin(phase));
    return Qubit.init(q.a, q.b.mul(factor));
}

fn floatBits(v: f64) u64 {
    const canonical = if (v == 0.0) 0.0 else v;
    return @as(u64, @bitCast(canonical));
}

const EdgeMap = std.HashMap(EdgeKey, ArrayList(Edge), EdgeKeyContext, std.hash_map.default_max_load_percentage);
const EntMap = std.HashMap(PairKey, TwoQubit, PairKeyContext, std.hash_map.default_max_load_percentage);

pub const SelfSimilarRelationalGraph = struct {
    allocator: Allocator,
    nodes: StringHashMap(Node),
    edges: EdgeMap,
    entanglements: EntMap,
    quantum_register: StringHashMap(Qubit),
    topology_hash: [Sha256.digest_length]u8,
    topology_hash_hex: [Sha256.digest_length * 2]u8,
    topology_hash_dirty: bool,
    rng: std.Random.DefaultPrng,
    rng_mutex: Mutex,

    pub fn init(allocator: Allocator) !SelfSimilarRelationalGraph {
        const ts = @as(i64, @truncate(std.time.nanoTimestamp()));
        const seed: u64 = std.hash.Wyhash.hash(0, std.mem.asBytes(&ts));
        var g = SelfSimilarRelationalGraph{
            .allocator = allocator,
            .nodes = StringHashMap(Node).init(allocator),
            .edges = EdgeMap.init(allocator),
            .entanglements = EntMap.init(allocator),
            .quantum_register = StringHashMap(Qubit).init(allocator),
            .topology_hash = [_]u8{0} ** Sha256.digest_length,
            .topology_hash_hex = [_]u8{0} ** (Sha256.digest_length * 2),
            .topology_hash_dirty = true,
            .rng = std.Random.DefaultPrng.init(seed),
            .rng_mutex = Mutex{},
        };
        try g.ensureTopologyHash();
        return g;
    }

    pub fn initWithArena(arena: *core_memory.ArenaAllocator) !SelfSimilarRelationalGraph {
        return init(arena.allocator());
    }

    pub fn initWithPool(pool: *core_memory.PoolAllocator) !SelfSimilarRelationalGraph {
        return init(pool.allocator());
    }

    pub fn initWithBuddy(buddy: *core_memory.BuddyAllocator) !SelfSimilarRelationalGraph {
        return init(buddy.allocator());
    }

    pub fn deinit(self: *SelfSimilarRelationalGraph) void {
        deinitNodeMap(&self.nodes, self.allocator);

        var ed_it = self.edges.iterator();
        while (ed_it.next()) |e| {
            for (e.value_ptr.items) |*edge| edge.deinit();
            e.value_ptr.deinit();
        }
        self.edges.deinit();

        self.entanglements.deinit();

        var qr_it = self.quantum_register.iterator();
        while (qr_it.next()) |e| self.allocator.free(e.key_ptr.*);
        self.quantum_register.deinit();
    }

    fn canonicalIdPtr(self: *SelfSimilarRelationalGraph, id: []const u8) ?[]const u8 {
        if (self.nodes.getPtr(id)) |n| return n.id;
        return null;
    }

    fn getCanonicalEdgeKey(self: *SelfSimilarRelationalGraph, source: []const u8, target: []const u8) ?EdgeKey {
        const s = self.canonicalIdPtr(source) orelse return null;
        const t = self.canonicalIdPtr(target) orelse return null;
        return EdgeKey{ .source = s, .target = t };
    }

    fn syncQuantumRegisterValue(self: *SelfSimilarRelationalGraph, canonical_id: []const u8, q: Qubit) !void {
        if (self.quantum_register.getPtr(canonical_id)) |qptr| {
            qptr.* = q;
        } else {
            const k = try dupeBytes(self.allocator, canonical_id);
            errdefer self.allocator.free(k);
            try self.quantum_register.put(k, q);
        }
    }

    fn markTopologyDirty(self: *SelfSimilarRelationalGraph) void {
        self.topology_hash_dirty = true;
    }

    pub fn addNode(self: *SelfSimilarRelationalGraph, node_in: Node) !void {
        var node = node_in;
        const lookup_id = node.id;

        if (self.nodes.getPtr(lookup_id)) |existing| {
            self.allocator.free(existing.data);
            existing.data = node.data;
            node.data = &[_]u8{};
            existing.qubit = node.qubit;
            existing.phase = node.phase;
            freeMapStringBytes(&existing.metadata, self.allocator);
            existing.metadata = node.metadata;
            node.metadata = StringHashMap([]u8).init(self.allocator);

            try self.syncQuantumRegisterValue(existing.id, node.qubit);

            node.deinit();
        } else {
            const map_key = try dupeBytes(self.allocator, node.id);
            errdefer self.allocator.free(map_key);

            self.nodes.put(map_key, node) catch |err| {
                return err;
            };

            const entry = self.nodes.getPtr(lookup_id).?;
            errdefer {
                if (self.nodes.fetchRemove(entry.id)) |removed| {
                    self.allocator.free(removed.key);
                    var v = removed.value;
                    v.deinit();
                }
            }
            try self.syncQuantumRegisterValue(entry.id, entry.qubit);
        }

        self.markTopologyDirty();
    }

    pub fn addEdge(self: *SelfSimilarRelationalGraph, source: []const u8, target: []const u8, edge_in: Edge) !void {
        var edge = edge_in;
        defer edge.deinit();

        const s = self.canonicalIdPtr(source) orelse return error.SourceNodeNotFound;
        const t = self.canonicalIdPtr(target) orelse return error.TargetNodeNotFound;

        var stored = try edge.clone(self.allocator);
        try stored.setSource(s);
        try stored.setTarget(t);

        const key = EdgeKey{ .source = s, .target = t };
        var gop = self.edges.getOrPut(key) catch |err| {
            stored.deinit();
            return err;
        };
        if (!gop.found_existing) gop.value_ptr.* = ArrayList(Edge).init(self.allocator);

        gop.value_ptr.append(stored) catch |err| {
            stored.deinit();
            if (!gop.found_existing and gop.value_ptr.items.len == 0) {
                gop.value_ptr.deinit();
                _ = self.edges.remove(key);
            }
            return err;
        };

        self.markTopologyDirty();
    }

    pub fn removeEdge(self: *SelfSimilarRelationalGraph, source: []const u8, target: []const u8) !void {
        const key = self.getCanonicalEdgeKey(source, target) orelse return error.NodeNotFound;
        if (self.edges.fetchRemove(key)) |removed| {
            var lst = removed.value;
            for (lst.items) |*edge| edge.deinit();
            lst.deinit();
            self.markTopologyDirty();
        }
    }

    pub fn removeEdgeSingle(self: *SelfSimilarRelationalGraph, source: []const u8, target: []const u8) !void {
        const key = self.getCanonicalEdgeKey(source, target) orelse return error.NodeNotFound;
        var changed = false;
        if (self.edges.getPtr(key)) |lst| {
            if (lst.items.len > 0) {
                var e = lst.orderedRemove(lst.items.len - 1);
                e.deinit();
                changed = true;
            }
            if (lst.items.len == 0) {
                if (self.edges.fetchRemove(key)) |removed| {
                    var l = removed.value;
                    l.deinit();
                }
            }
        }
        if (changed) self.markTopologyDirty();
    }

    pub fn removeNode(self: *SelfSimilarRelationalGraph, node_id: []const u8) !void {
        const canonical = self.canonicalIdPtr(node_id) orelse return error.NodeNotFound;

        var keys_to_remove = ArrayList(EdgeKey).init(self.allocator);
        defer keys_to_remove.deinit();

        var ed_it = self.edges.iterator();
        while (ed_it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.source, canonical) or std.mem.eql(u8, e.key_ptr.target, canonical)) {
                try keys_to_remove.append(e.key_ptr.*);
            }
        }

        for (keys_to_remove.items) |k| {
            if (self.edges.fetchRemove(k)) |removed| {
                var lst = removed.value;
                for (lst.items) |*edge| edge.deinit();
                lst.deinit();
            }
        }

        var ent_keys_to_remove = ArrayList(PairKey).init(self.allocator);
        defer ent_keys_to_remove.deinit();

        var ent_it = self.entanglements.iterator();
        while (ent_it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.a, canonical) or std.mem.eql(u8, e.key_ptr.b, canonical)) {
                try ent_keys_to_remove.append(e.key_ptr.*);
            }
        }

        for (ent_keys_to_remove.items) |k| {
            _ = self.entanglements.remove(k);
        }

        if (self.quantum_register.fetchRemove(canonical)) |removed| {
            self.allocator.free(removed.key);
        }

        if (self.nodes.fetchRemove(canonical)) |removed| {
            self.allocator.free(removed.key);
            var v = removed.value;
            v.deinit();
        }

        self.markTopologyDirty();
    }

    pub fn getNode(self: *SelfSimilarRelationalGraph, node_id: []const u8) ?*Node {
        return self.nodes.getPtr(node_id);
    }

    pub fn getNodeConst(self: *const SelfSimilarRelationalGraph, node_id: []const u8) ?*const Node {
        return self.nodes.getPtr(node_id);
    }

    pub fn getEdgesConst(self: *const SelfSimilarRelationalGraph, source: []const u8, target: []const u8) ?[]const Edge {
        const s_node = self.nodes.getPtr(source) orelse return null;
        const t_node = self.nodes.getPtr(target) orelse return null;
        const key = EdgeKey{ .source = s_node.id, .target = t_node.id };
        if (self.edges.getPtr(key)) |list| return list.items;
        return null;
    }

    pub fn hasEdge(self: *const SelfSimilarRelationalGraph, source: []const u8, target: []const u8) bool {
        const s_node = self.nodes.getPtr(source) orelse return false;
        const t_node = self.nodes.getPtr(target) orelse return false;
        const key = EdgeKey{ .source = s_node.id, .target = t_node.id };
        if (self.edges.getPtr(key)) |lst| return lst.items.len > 0;
        return false;
    }

    pub fn clear(self: *SelfSimilarRelationalGraph) !void {
        clearNodeMapRetainingCapacity(&self.nodes, self.allocator);

        var ed_it = self.edges.iterator();
        while (ed_it.next()) |e| {
            for (e.value_ptr.items) |*edge| edge.deinit();
            e.value_ptr.deinit();
        }
        self.edges.clearRetainingCapacity();

        self.entanglements.clearRetainingCapacity();

        var qr_it = self.quantum_register.iterator();
        while (qr_it.next()) |e| self.allocator.free(e.key_ptr.*);
        self.quantum_register.clearRetainingCapacity();

        self.markTopologyDirty();
    }

    pub fn setQuantumState(self: *SelfSimilarRelationalGraph, node_id: []const u8, q: Qubit) !void {
        const canonical = self.canonicalIdPtr(node_id) orelse return error.NodeNotFound;
        const n = self.nodes.getPtr(canonical).?;
        n.qubit = q;
        try self.syncQuantumRegisterValue(canonical, q);
        self.markTopologyDirty();
    }

    pub fn getQuantumState(self: *const SelfSimilarRelationalGraph, node_id: []const u8) ?Qubit {
        const n = self.nodes.getPtr(node_id) orelse return null;
        return self.quantum_register.get(n.id);
    }

    pub fn applyQuantumGate(self: *SelfSimilarRelationalGraph, node_id: []const u8, gate: Gate) !void {
        const canonical = self.canonicalIdPtr(node_id) orelse return error.NodeNotFound;
        const n = self.nodes.getPtr(canonical).?;
        n.qubit = gate(n.qubit);
        try self.syncQuantumRegisterValue(canonical, n.qubit);
        self.markTopologyDirty();
    }

    pub fn applyRuntimePhaseGate(self: *SelfSimilarRelationalGraph, node_id: []const u8, phase: f64) !void {
        const canonical = self.canonicalIdPtr(node_id) orelse return error.NodeNotFound;
        const n = self.nodes.getPtr(canonical).?;
        n.qubit = runtimePhaseGate(n.qubit, phase);
        try self.syncQuantumRegisterValue(canonical, n.qubit);
        self.markTopologyDirty();
    }

    fn pairKeyFor(a: []const u8, b: []const u8) PairKey {
        return if (std.mem.lessThan(u8, a, b)) PairKey{ .a = a, .b = b } else PairKey{ .a = b, .b = a };
    }

    pub fn entangleNodes(self: *SelfSimilarRelationalGraph, a_id: []const u8, b_id: []const u8) !void {
        const a = self.canonicalIdPtr(a_id) orelse return error.NodeNotFound;
        const b = self.canonicalIdPtr(b_id) orelse return error.NodeNotFound;
        const pk = pairKeyFor(a, b);

        try self.entanglements.put(pk, TwoQubit.initBellPhiPlus());

        var changed = false;

        const key_ab = EdgeKey{ .source = a, .target = b };
        const key_ba = EdgeKey{ .source = b, .target = a };

        if (!self.hasEntangledEdge(a, b)) {
            const edge_ab = try Edge.init(self.allocator, a, b, .entangled, 1.0, Complex(f64).init(1.0, 0.0), 0.0);
            try self.addEdge(a, b, edge_ab);
            changed = true;
        }

        if (!self.hasEntangledEdge(b, a)) {
            const edge_ba = try Edge.init(self.allocator, b, a, .entangled, 1.0, Complex(f64).init(1.0, 0.0), 0.0);
            self.addEdge(b, a, edge_ba) catch |err| {
                if (changed) {
                    self.removeMatchingEdgeNoHash(key_ab, .entangled);
                    _ = self.edges.getPtr(key_ba);
                    _ = self.entanglements.remove(pk);
                    self.ensureTopologyHash() catch {};
                }
                return err;
            };
            changed = true;
        }

        if (!changed) {
            self.markTopologyDirty();
        }
    }

    fn hasEntangledEdge(self: *const SelfSimilarRelationalGraph, source: []const u8, target: []const u8) bool {
        const key = EdgeKey{ .source = source, .target = target };
        if (self.edges.getPtr(key)) |lst| {
            for (lst.items) |edge| {
                if (edge.quality == .entangled) return true;
            }
        }
        return false;
    }

    fn removeMatchingEdgeNoHash(self: *SelfSimilarRelationalGraph, key: EdgeKey, quality: EdgeQuality) void {
        if (self.edges.getPtr(key)) |lst| {
            var i: usize = 0;
            while (i < lst.items.len) : (i += 1) {
                if (lst.items[i].quality == quality) {
                    var e = lst.orderedRemove(i);
                    e.deinit();
                    break;
                }
            }
            if (lst.items.len == 0) {
                if (self.edges.fetchRemove(key)) |removed| {
                    var l = removed.value;
                    l.deinit();
                }
            }
        }
    }

    pub fn measure(self: *SelfSimilarRelationalGraph, node_id: []const u8) !u1 {
        const canonical = self.canonicalIdPtr(node_id) orelse return error.NodeNotFound;

        var hit_key: ?PairKey = null;
        var hit_val: ?TwoQubit = null;

        var it = self.entanglements.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*.a, canonical) or std.mem.eql(u8, e.key_ptr.*.b, canonical)) {
                hit_key = e.key_ptr.*;
                hit_val = e.value_ptr.*;
                break;
            }
        }

        if (hit_key) |pk| {
            var state = hit_val.?;
            state.normalizeInPlace();

            self.rng_mutex.lock();
            const r = self.rng.random().float(f64);
            self.rng_mutex.unlock();

            var cum: f64 = 0.0;
            var outcome: usize = state.amps.len - 1;
            var amp_idx: usize = 0;
            while (amp_idx < state.amps.len) : (amp_idx += 1) {
                const amp = state.amps[amp_idx];
                cum += amp.re * amp.re + amp.im * amp.im;
                if (r <= cum or amp_idx + 1 == state.amps.len) {
                    outcome = amp_idx;
                    break;
                }
            }

            const a_id = pk.a;
            const b_id = pk.b;

            const a_ptr = self.nodes.getPtr(a_id).?;
            const b_ptr = self.nodes.getPtr(b_id).?;

            switch (outcome) {
                0 => {
                    a_ptr.qubit = Qubit.initBasis0();
                    b_ptr.qubit = Qubit.initBasis0();
                },
                1 => {
                    a_ptr.qubit = Qubit.initBasis0();
                    b_ptr.qubit = Qubit.initBasis1();
                },
                2 => {
                    a_ptr.qubit = Qubit.initBasis1();
                    b_ptr.qubit = Qubit.initBasis0();
                },
                else => {
                    a_ptr.qubit = Qubit.initBasis1();
                    b_ptr.qubit = Qubit.initBasis1();
                },
            }

            try self.syncQuantumRegisterValue(a_id, a_ptr.qubit);
            try self.syncQuantumRegisterValue(b_id, b_ptr.qubit);

            _ = self.entanglements.remove(pk);

            const bit: u1 = if (std.mem.eql(u8, canonical, a_id))
                @as(u1, @intCast((outcome >> 1) & 1))
            else
                @as(u1, @intCast(outcome & 1));

            if (self.edges.getPtr(EdgeKey{ .source = a_id, .target = b_id })) |lst| {
                for (lst.items) |*edge| {
                    if (edge.quality == .entangled) edge.quality = .collapsed;
                }
            }
            if (self.edges.getPtr(EdgeKey{ .source = b_id, .target = a_id })) |lst2| {
                for (lst2.items) |*edge| {
                    if (edge.quality == .entangled) edge.quality = .collapsed;
                }
            }

            self.markTopologyDirty();
            return bit;
        }

        const n = self.nodes.getPtr(canonical).?;
        const p0 = n.qubit.prob0();

        self.rng_mutex.lock();
        const r0 = self.rng.random().float(f64);
        self.rng_mutex.unlock();

        const bit: u1 = if (r0 <= p0) 0 else 1;

        n.qubit = if (bit == 0) Qubit.initBasis0() else Qubit.initBasis1();
        try self.syncQuantumRegisterValue(canonical, n.qubit);

        self.markTopologyDirty();
        return bit;
    }

    pub fn nodeCount(self: *const SelfSimilarRelationalGraph) usize {
        return self.nodes.count();
    }

    pub fn edgeCount(self: *const SelfSimilarRelationalGraph) usize {
        var c: usize = 0;
        var it = self.edges.iterator();
        while (it.next()) |e| c += e.value_ptr.items.len;
        return c;
    }

    pub fn getAllNodeIds(self: *const SelfSimilarRelationalGraph, allocator: Allocator) !ArrayList([]u8) {
        var out = ArrayList([]u8).init(allocator);
        errdefer {
            for (out.items) |id| allocator.free(id);
            out.deinit();
        }
        var it = self.nodes.iterator();
        while (it.next()) |e| {
            const copy = try allocator.dupe(u8, e.value_ptr.id);
            try out.append(copy);
        }
        std.mem.sort([]u8, out.items, {}, struct {
            fn lessThan(_: void, lhs: []u8, rhs: []u8) bool {
                return std.mem.lessThan(u8, lhs, rhs);
            }
        }.lessThan);
        return out;
    }

    fn shaUpdateU64(h: *Sha256, v: u64) void {
        var b: [8]u8 = undefined;
        std.mem.writeInt(u64, &b, v, .little);
        h.update(&b);
    }

    fn shaUpdateBytes(h: *Sha256, b: []const u8) void {
        shaUpdateU64(h, @as(u64, @intCast(b.len)));
        h.update(b);
    }

    fn shaUpdateF64(h: *Sha256, v: f64) void {
        shaUpdateU64(h, floatBits(v));
    }

    fn ensureTopologyHash(self: *SelfSimilarRelationalGraph) !void {
        if (!self.topology_hash_dirty) return;

        var node_digests = ArrayList([Sha256.digest_length]u8).init(self.allocator);
        defer node_digests.deinit();

        var edge_group_digests = ArrayList([Sha256.digest_length]u8).init(self.allocator);
        defer edge_group_digests.deinit();

        var ent_digests = ArrayList([Sha256.digest_length]u8).init(self.allocator);
        defer ent_digests.deinit();

        var node_count: u64 = 0;
        var edgekey_count: u64 = 0;
        var total_edge_count: u64 = 0;
        var ent_count: u64 = 0;

        var n_it = self.nodes.iterator();
        while (n_it.next()) |e| {
            node_count += 1;

            var meta_digests = ArrayList([Sha256.digest_length]u8).init(self.allocator);
            defer meta_digests.deinit();

            var mit = e.value_ptr.metadata.iterator();
            while (mit.next()) |me| {
                var mh = Sha256.init(.{});
                shaUpdateBytes(&mh, me.key_ptr.*);
                shaUpdateBytes(&mh, me.value_ptr.*);
                var md: [Sha256.digest_length]u8 = undefined;
                mh.final(&md);
                try meta_digests.append(md);
            }

            std.mem.sort([Sha256.digest_length]u8, meta_digests.items, {}, struct {
                fn lessThan(_: void, lhs: [Sha256.digest_length]u8, rhs: [Sha256.digest_length]u8) bool {
                    return std.mem.lessThan(u8, &lhs, &rhs);
                }
            }.lessThan);

            var h = Sha256.init(.{});
            shaUpdateBytes(&h, e.value_ptr.id);
            shaUpdateBytes(&h, e.value_ptr.data);
            shaUpdateF64(&h, e.value_ptr.phase);
            shaUpdateF64(&h, e.value_ptr.qubit.a.re);
            shaUpdateF64(&h, e.value_ptr.qubit.a.im);
            shaUpdateF64(&h, e.value_ptr.qubit.b.re);
            shaUpdateF64(&h, e.value_ptr.qubit.b.im);
            shaUpdateU64(&h, @as(u64, @intCast(meta_digests.items.len)));
            for (meta_digests.items) |md| h.update(&md);

            var d: [Sha256.digest_length]u8 = undefined;
            h.final(&d);
            try node_digests.append(d);
        }

        std.mem.sort([Sha256.digest_length]u8, node_digests.items, {}, struct {
            fn lessThan(_: void, lhs: [Sha256.digest_length]u8, rhs: [Sha256.digest_length]u8) bool {
                return std.mem.lessThan(u8, &lhs, &rhs);
            }
        }.lessThan);

        var e_it = self.edges.iterator();
        while (e_it.next()) |kv| {
            edgekey_count += 1;

            var edge_digests = ArrayList([Sha256.digest_length]u8).init(self.allocator);
            defer edge_digests.deinit();

            for (kv.value_ptr.items) |*edge| {
                total_edge_count += 1;

                var emeta_digests = ArrayList([Sha256.digest_length]u8).init(self.allocator);
                defer emeta_digests.deinit();

                var emi = edge.metadata.iterator();
                while (emi.next()) |me| {
                    var mh = Sha256.init(.{});
                    shaUpdateBytes(&mh, me.key_ptr.*);
                    shaUpdateBytes(&mh, me.value_ptr.*);
                    var md: [Sha256.digest_length]u8 = undefined;
                    mh.final(&md);
                    try emeta_digests.append(md);
                }

                std.mem.sort([Sha256.digest_length]u8, emeta_digests.items, {}, struct {
                    fn lessThan(_: void, lhs: [Sha256.digest_length]u8, rhs: [Sha256.digest_length]u8) bool {
                        return std.mem.lessThan(u8, &lhs, &rhs);
                    }
                }.lessThan);

                var eh = Sha256.init(.{});
                shaUpdateBytes(&eh, edge.source);
                shaUpdateBytes(&eh, edge.target);
                shaUpdateBytes(&eh, edge.quality.toString());
                shaUpdateF64(&eh, edge.weight);
                shaUpdateF64(&eh, edge.fractal_dimension);
                shaUpdateF64(&eh, edge.quantum_correlation.re);
                shaUpdateF64(&eh, edge.quantum_correlation.im);
                shaUpdateU64(&eh, @as(u64, @intCast(emeta_digests.items.len)));
                for (emeta_digests.items) |md| eh.update(&md);

                var ed: [Sha256.digest_length]u8 = undefined;
                eh.final(&ed);
                try edge_digests.append(ed);
            }

            std.mem.sort([Sha256.digest_length]u8, edge_digests.items, {}, struct {
                fn lessThan(_: void, lhs: [Sha256.digest_length]u8, rhs: [Sha256.digest_length]u8) bool {
                    return std.mem.lessThan(u8, &lhs, &rhs);
                }
            }.lessThan);

            var kh = Sha256.init(.{});
            shaUpdateBytes(&kh, kv.key_ptr.source);
            shaUpdateBytes(&kh, kv.key_ptr.target);
            shaUpdateU64(&kh, @as(u64, @intCast(edge_digests.items.len)));
            for (edge_digests.items) |ed| kh.update(&ed);

            var kd: [Sha256.digest_length]u8 = undefined;
            kh.final(&kd);
            try edge_group_digests.append(kd);
        }

        std.mem.sort([Sha256.digest_length]u8, edge_group_digests.items, {}, struct {
            fn lessThan(_: void, lhs: [Sha256.digest_length]u8, rhs: [Sha256.digest_length]u8) bool {
                return std.mem.lessThan(u8, &lhs, &rhs);
            }
        }.lessThan);

        var en_it = self.entanglements.iterator();
        while (en_it.next()) |kv| {
            ent_count += 1;

            var h = Sha256.init(.{});
            shaUpdateBytes(&h, kv.key_ptr.a);
            shaUpdateBytes(&h, kv.key_ptr.b);
            for (kv.value_ptr.amps) |c| {
                shaUpdateF64(&h, c.re);
                shaUpdateF64(&h, c.im);
            }

            var d: [Sha256.digest_length]u8 = undefined;
            h.final(&d);
            try ent_digests.append(d);
        }

        std.mem.sort([Sha256.digest_length]u8, ent_digests.items, {}, struct {
            fn lessThan(_: void, lhs: [Sha256.digest_length]u8, rhs: [Sha256.digest_length]u8) bool {
                return std.mem.lessThan(u8, &lhs, &rhs);
            }
        }.lessThan);

        var final_hash = Sha256.init(.{});
        shaUpdateU64(&final_hash, node_count);
        for (node_digests.items) |d| final_hash.update(&d);
        shaUpdateU64(&final_hash, edgekey_count);
        shaUpdateU64(&final_hash, total_edge_count);
        for (edge_group_digests.items) |d| final_hash.update(&d);
        shaUpdateU64(&final_hash, ent_count);
        for (ent_digests.items) |d| final_hash.update(&d);

        final_hash.final(&self.topology_hash);
        self.topology_hash_dirty = false;
    }

    pub fn getTopologyHash(self: *SelfSimilarRelationalGraph) ![Sha256.digest_length]u8 {
        try self.ensureTopologyHash();
        return self.topology_hash;
    }

    pub fn updateTopologyHash(self: *SelfSimilarRelationalGraph) !void {
        self.markTopologyDirty();
        try self.ensureTopologyHash();
    }

    pub fn getTopologyHashHex(self: *SelfSimilarRelationalGraph) ![]const u8 {
        try self.ensureTopologyHash();
        const written = try std.fmt.bufPrint(&self.topology_hash_hex, "{s}", .{std.fmt.fmtSliceHexLower(&self.topology_hash)});
        return written;
    }

    pub fn encodeInformation(self: *SelfSimilarRelationalGraph, data: []const u8) ![]const u8 {
        var hash: [Sha256.digest_length]u8 = undefined;
        Sha256.hash(data, &hash, .{});

        var id_buf: [16]u8 = undefined;
        _ = try std.fmt.bufPrint(&id_buf, "{s}", .{std.fmt.fmtSliceHexLower(hash[0..8])});

        var added_node = false;
        var added_edges = ArrayList(EdgeKey).init(self.allocator);
        defer added_edges.deinit();

        errdefer {
            for (added_edges.items) |k| {
                if (self.edges.fetchRemove(k)) |removed| {
                    var lst = removed.value;
                    for (lst.items) |*edge| edge.deinit();
                    lst.deinit();
                }
            }
            if (added_node) {
                if (self.nodes.fetchRemove(id_buf[0..16])) |removed| {
                    self.allocator.free(removed.key);
                    var v = removed.value;
                    v.deinit();
                }
                if (self.quantum_register.fetchRemove(id_buf[0..16])) |removed| {
                    self.allocator.free(removed.key);
                }
            }
            self.markTopologyDirty();
        }

        var node = try Node.init(self.allocator, id_buf[0..16], data, Qubit.initBasis0(), 0.0);

        const ts_str = std.fmt.allocPrint(self.allocator, "{d}", .{std.time.timestamp()}) catch |err| {
            node.deinit();
            return err;
        };
        defer self.allocator.free(ts_str);

        node.setMetadata("encoding_time", ts_str) catch |err| {
            node.deinit();
            return err;
        };

        try self.addNode(node);
        added_node = true;

        var ids = try self.getAllNodeIds(self.allocator);
        defer {
            for (ids.items) |id| self.allocator.free(id);
            ids.deinit();
        }

        if (ids.items.len > 1) {
            const max_links: usize = if (ids.items.len - 1 < 3) ids.items.len - 1 else 3;
            var linked: usize = 0;
            var i: usize = 0;
            while (i < ids.items.len and linked < max_links) : (i += 1) {
                const prev = ids.items[i];
                if (std.mem.eql(u8, prev, id_buf[0..16])) continue;
                const src = self.canonicalIdPtr(id_buf[0..16]).?;
                const dst = self.canonicalIdPtr(prev).?;
                const e = try Edge.init(self.allocator, src, dst, .coherent, 0.5, Complex(f64).init(0.0, 0.0), 0.0);
                try self.addEdge(src, dst, e);
                try added_edges.append(EdgeKey{ .source = src, .target = dst });
                linked += 1;
            }
        }

        return self.canonicalIdPtr(id_buf[0..16]).?;
    }

    pub fn decodeInformation(self: *const SelfSimilarRelationalGraph, node_id: []const u8) ?[]const u8 {
        if (self.nodes.getPtr(node_id)) |n| return n.data;
        return null;
    }

    pub fn exportNodeEmbeddings(self: *SelfSimilarRelationalGraph, allocator: Allocator) !core_tensor.Tensor {
        const nc = self.nodes.count();
        if (nc == 0) {
            const shape = [_]usize{ 0, 4 };
            return core_tensor.Tensor.init(allocator, &shape);
        }

        var ids = try self.getAllNodeIds(allocator);
        defer {
            for (ids.items) |id| allocator.free(id);
            ids.deinit();
        }

        const shape = [_]usize{ nc, 4 };
        var tensor = try core_tensor.Tensor.init(allocator, &shape);

        for (ids.items, 0..) |id, idx| {
            const node = self.nodes.getPtr(id).?;
            tensor.data[idx * 4 + 0] = @floatCast(node.qubit.a.re);
            tensor.data[idx * 4 + 1] = @floatCast(node.qubit.a.im);
            tensor.data[idx * 4 + 2] = @floatCast(node.qubit.b.re);
            tensor.data[idx * 4 + 3] = @floatCast(node.qubit.b.im);
        }
        return tensor;
    }

    pub fn importNodeEmbeddings(self: *SelfSimilarRelationalGraph, tensor: *const core_tensor.Tensor) !void {
        if (tensor.shape.dims.len != 2 or tensor.shape.dims[1] != 4) return;

        var ids = try self.getAllNodeIds(self.allocator);
        defer {
            for (ids.items) |id| self.allocator.free(id);
            ids.deinit();
        }

        for (ids.items, 0..) |id, idx| {
            if (idx >= tensor.shape.dims[0]) break;
            const node = self.nodes.getPtr(id).?;
            node.qubit.a.re = @floatCast(tensor.data[idx * 4 + 0]);
            node.qubit.a.im = @floatCast(tensor.data[idx * 4 + 1]);
            node.qubit.b.re = @floatCast(tensor.data[idx * 4 + 2]);
            node.qubit.b.im = @floatCast(tensor.data[idx * 4 + 3]);
            node.qubit.normalizeInPlace();
            try self.syncQuantumRegisterValue(node.id, node.qubit);
        }

        self.markTopologyDirty();
    }

    pub fn exportAdjacencyMatrix(self: *SelfSimilarRelationalGraph, node_ids: []const []const u8, allocator: Allocator) !core_tensor.Tensor {
        const n = node_ids.len;
        if (n == 0) {
            const shape = [_]usize{ 0, 0 };
            return core_tensor.Tensor.init(allocator, &shape);
        }
        const shape = [_]usize{ n, n };
        var tensor = try core_tensor.Tensor.init(allocator, &shape);
        @memset(tensor.data, 0);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const s_node = self.nodes.getPtr(node_ids[i]) orelse continue;
            var j: usize = 0;
            while (j < n) : (j += 1) {
                const t_node = self.nodes.getPtr(node_ids[j]) orelse continue;
                const key = EdgeKey{ .source = s_node.id, .target = t_node.id };
                if (self.edges.get(key)) |edge_list| {
                    var total_weight: f64 = 0.0;
                    for (edge_list.items) |edge| total_weight += edge.weight;
                    tensor.data[i * n + j] = @floatCast(total_weight);
                }
            }
        }
        return tensor;
    }

    pub fn bulkImportFromGPU(
        self: *SelfSimilarRelationalGraph,
        hashes: []const u64,
        re_a: []const f32,
        im_a: []const f32,
        re_b: []const f32,
        im_b: []const f32,
        edge_srcs: []const i64,
        edge_tgts: []const i64,
    ) !void {
        if (hashes.len == 0) return;
        if (re_a.len != hashes.len or im_a.len != hashes.len or
            re_b.len != hashes.len or im_b.len != hashes.len)
        {
            return error.InvalidDimensions;
        }
        if (edge_srcs.len != edge_tgts.len) return error.InvalidDimensions;

        var id_bufs = try self.allocator.alloc([16]u8, hashes.len);
        defer self.allocator.free(id_bufs);

        for (hashes, 0..) |hash, i| {
            _ = try std.fmt.bufPrint(&id_bufs[i], "{x:0>16}", .{hash});
        }

        // Transactional node insertion: track how many nodes were successfully
        // added so we can roll back the entire batch on any failure, leaving
        // the graph in a consistent state instead of partial state.
        var nodes_added: usize = 0;
        errdefer {
            var k: usize = 0;
            while (k < nodes_added) : (k += 1) {
                if (self.nodes.fetchRemove(id_bufs[k][0..])) |kv| {
                    var node_copy = kv.value;
                    node_copy.deinit();
                }
            }
            self.markTopologyDirty();
        }

        for (hashes, 0..) |_, i| {
            var q = Qubit.init(
                Complex(f64).init(@floatCast(re_a[i]), @floatCast(im_a[i])),
                Complex(f64).init(@floatCast(re_b[i]), @floatCast(im_b[i])),
            );
            q.normalizeInPlace();
            var node = try Node.init(self.allocator, id_bufs[i][0..], id_bufs[i][0..], q, 0.0);
            var node_transferred = false;
            defer if (!node_transferred) node.deinit();
            try self.addNode(node);
            node_transferred = true;
            nodes_added += 1;
        }

        // Edge insertion.  SourceNodeNotFound/TargetNodeNotFound are logged as
        // warnings rather than silently swallowed: given that we just inserted
        // every node in this batch, these errors should never occur.  If they
        // do, it indicates an ABI mismatch between the Futhark output and the
        // node-ID formatting above.  Edges that fail with these errors are
        // skipped (with their allocations freed) so the caller still gets a
        // valid graph; all other addEdge errors are fatal.
        var skipped_edges: usize = 0;
        for (edge_srcs, edge_tgts) |src_i64, tgt_i64| {
            if (src_i64 < 0 or tgt_i64 < 0) continue;
            const src_idx = @as(usize, @intCast(src_i64));
            const tgt_idx = @as(usize, @intCast(tgt_i64));
            if (src_idx >= hashes.len or tgt_idx >= hashes.len) continue;
            if (src_idx == tgt_idx) continue;
            var e = try Edge.init(
                self.allocator,
                id_bufs[src_idx][0..],
                id_bufs[tgt_idx][0..],
                .coherent,
                0.5,
                Complex(f64).init(0.0, 0.0),
                0.0,
            );
            var edge_transferred = false;
            defer if (!edge_transferred) e.deinit();
            self.addEdge(id_bufs[src_idx][0..], id_bufs[tgt_idx][0..], e) catch |err| switch (err) {
                error.SourceNodeNotFound, error.TargetNodeNotFound => {
                    std.debug.print(
                        "[bulkImportFromGPU] WARN: addEdge({d}->{d}) got {s}:" ++
                            " node missing after bulk insert (possible ABI mismatch)\n",
                        .{ src_idx, tgt_idx, @errorName(err) },
                    );
                    skipped_edges += 1;
                    // edge_transferred stays false; defer frees the allocation.
                },
                else => return err,
            };
            edge_transferred = true;
        }
        if (skipped_edges > 0) {
            std.debug.print(
                "[bulkImportFromGPU] WARN: {d}/{d} edges skipped (source/target node missing)\n",
                .{ skipped_edges, edge_srcs.len },
            );
        }

        self.markTopologyDirty();
    }
};

test "quantum gates basics" {
    const testing = std.testing;
    var q = Qubit.initBasis0();
    q = hadamardGate(q);
    try testing.expectApproxEqAbs(q.prob0(), 0.5, 1e-9);
    try testing.expectApproxEqAbs(q.prob1(), 0.5, 1e-9);

    var qx = Qubit.initBasis0();
    qx = pauliXGate(qx);
    try testing.expectApproxEqAbs(qx.prob0(), 0.0, 1e-9);
    try testing.expectApproxEqAbs(qx.prob1(), 1.0, 1e-9);

    var qz = Qubit.init(Complex(f64).init(0.0, 0.0), Complex(f64).init(1.0, 0.0));
    qz = pauliZGate(qz);
    try testing.expectApproxEqAbs(qz.b.re, -1.0, 1e-9);
}

test "graph basic operations" {
    const testing = std.testing;
    var g = try SelfSimilarRelationalGraph.init(testing.allocator);
    defer g.deinit();

    const n1 = try Node.init(testing.allocator, "a", "data_a", Qubit.initBasis0(), 0.0);
    const n2 = try Node.init(testing.allocator, "b", "data_b", Qubit.initBasis1(), 0.0);
    try g.addNode(n1);
    try g.addNode(n2);

    const e = try Edge.init(testing.allocator, "a", "b", .coherent, 1.0, Complex(f64).init(0.0, 0.0), 0.0);
    try g.addEdge("a", "b", e);

    try testing.expectEqual(@as(usize, 2), g.nodeCount());
    try testing.expectEqual(@as(usize, 1), g.edgeCount());
    try testing.expect(g.getEdgesConst("a", "b") != null);

    _ = try g.measure("a");
    _ = try g.getTopologyHashHex();
}

test "graph remove edge" {
    const testing = std.testing;
    var g = try SelfSimilarRelationalGraph.init(testing.allocator);
    defer g.deinit();

    const n1 = try Node.init(testing.allocator, "x", "dx", Qubit.initBasis0(), 0.0);
    const n2 = try Node.init(testing.allocator, "y", "dy", Qubit.initBasis1(), 0.0);
    try g.addNode(n1);
    try g.addNode(n2);

    const e = try Edge.init(testing.allocator, "x", "y", .coherent, 0.5, Complex(f64).init(0.0, 0.0), 1.0);
    try g.addEdge("x", "y", e);
    try testing.expectEqual(@as(usize, 1), g.edgeCount());

    try g.removeEdge("x", "y");
    try testing.expectEqual(@as(usize, 0), g.edgeCount());
}

// ---------------------------------------------------------------------------
// bulkImportFromGPU tests
// ---------------------------------------------------------------------------

test "bulkImportFromGPU: empty input returns cleanly" {
    const testing = std.testing;
    var g = try SelfSimilarRelationalGraph.init(testing.allocator);
    defer g.deinit();

    try g.bulkImportFromGPU(&[_]u64{}, &[_]f32{}, &[_]f32{}, &[_]f32{}, &[_]f32{}, &[_]i64{}, &[_]i64{});
    try testing.expectEqual(@as(usize, 0), g.nodeCount());
    try testing.expectEqual(@as(usize, 0), g.edgeCount());
}

test "bulkImportFromGPU: dimension mismatch returns InvalidDimensions" {
    const testing = std.testing;
    var g = try SelfSimilarRelationalGraph.init(testing.allocator);
    defer g.deinit();

    // re_a has 2 elements but hashes has 3 — must fail before touching the graph.
    const hashes = [_]u64{ 1, 2, 3 };
    const short = [_]f32{ 1.0, 0.0 };
    const full = [_]f32{ 0.0, 0.0, 0.0 };

    const result = g.bulkImportFromGPU(&hashes, &short, &full, &full, &full, &[_]i64{}, &[_]i64{});
    try testing.expectError(error.InvalidDimensions, result);

    // Graph must remain completely empty — no partial state.
    try testing.expectEqual(@as(usize, 0), g.nodeCount());
    try testing.expectEqual(@as(usize, 0), g.edgeCount());
}

test "bulkImportFromGPU: happy path imports nodes and valid edges" {
    const testing = std.testing;
    var g = try SelfSimilarRelationalGraph.init(testing.allocator);
    defer g.deinit();

    const hashes = [_]u64{ 0x0000000000000001, 0x0000000000000002, 0x0000000000000003 };
    const ones = [_]f32{ 1.0, 1.0, 1.0 };
    const zeros = [_]f32{ 0.0, 0.0, 0.0 };
    // Valid forward edges: 0→1, 1→2.
    const srcs = [_]i64{ 0, 1 };
    const tgts = [_]i64{ 1, 2 };

    try g.bulkImportFromGPU(&hashes, &ones, &zeros, &zeros, &zeros, &srcs, &tgts);

    try testing.expectEqual(@as(usize, 3), g.nodeCount());
    try testing.expect(g.edgeCount() >= 2);
}

test "bulkImportFromGPU: invalid edges are silently filtered" {
    const testing = std.testing;
    var g = try SelfSimilarRelationalGraph.init(testing.allocator);
    defer g.deinit();

    const hashes = [_]u64{ 0xAAAA, 0xBBBB };
    const ones = [_]f32{ 1.0, 1.0 };
    const zeros = [_]f32{ 0.0, 0.0 };
    // self-loop (0→0), negative src (-1→1), negative tgt (0→-1), out-of-range (5→1).
    const srcs = [_]i64{ 0, -1, 0, 5 };
    const tgts = [_]i64{ 0, 1, -1, 1 };

    try g.bulkImportFromGPU(&hashes, &ones, &zeros, &zeros, &zeros, &srcs, &tgts);

    try testing.expectEqual(@as(usize, 2), g.nodeCount());
    // Every edge should have been filtered — none connect distinct in-range nodes.
    try testing.expectEqual(@as(usize, 0), g.edgeCount());
}

test "bulkImportFromGPU: graph is clean after re-import of same hashes" {
    // Re-importing the same hashes on top of an existing graph must not leak
    // memory (the existing test allocator would catch double-free/leak).
    const testing = std.testing;
    var g = try SelfSimilarRelationalGraph.init(testing.allocator);
    defer g.deinit();

    const hashes = [_]u64{ 0xDEAD, 0xBEEF };
    const ones = [_]f32{ 1.0, 1.0 };
    const zeros = [_]f32{ 0.0, 0.0 };
    const srcs = [_]i64{0};
    const tgts = [_]i64{1};

    try g.bulkImportFromGPU(&hashes, &ones, &zeros, &zeros, &zeros, &srcs, &tgts);
    const first_nodes = g.nodeCount();

    // Second import — addNode should update existing entries, not duplicate.
    try g.bulkImportFromGPU(&hashes, &ones, &zeros, &zeros, &zeros, &srcs, &tgts);
    try testing.expectEqual(first_nodes, g.nodeCount());
}
