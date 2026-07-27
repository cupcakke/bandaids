const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Sha512 = std.crypto.hash.sha2.Sha512;
const Blake3 = std.crypto.hash.Blake3;
const Complex = std.math.Complex;
const timingSafeEql = std.crypto.utils.timingSafeEql;

pub const SecurityProofsConfig = struct {
    pub const SHA256_DIGEST_SIZE: usize = 32;
    pub const SHA512_DIGEST_SIZE: usize = 64;
    pub const ACCESS_RIGHT_ADMIN_BIT: u8 = 16;
    pub const ACCESS_RIGHT_DELETE_BIT: u8 = 8;
    pub const ACCESS_RIGHT_EXECUTE_BIT: u8 = 4;
    pub const ACCESS_RIGHT_WRITE_BIT: u8 = 2;
    pub const ACCESS_RIGHT_READ_BIT: u8 = 1;
    pub const SEVERITY_MULTIPLIER: u8 = 25;
    pub const SEVERITY_MAX: u8 = 100;
    pub const SECURITY_LEVEL_PUBLIC: u8 = 0;
    pub const SECURITY_LEVEL_INTERNAL: u8 = 1;
    pub const SECURITY_LEVEL_CONFIDENTIAL: u8 = 2;
    pub const SECURITY_LEVEL_RESTRICTED: u8 = 3;
    pub const SECURITY_LEVEL_TOP: u8 = 4;
    pub const INTEGRITY_LEVEL_UNTRUSTED: u8 = 0;
    pub const INTEGRITY_LEVEL_USER: u8 = 1;
    pub const INTEGRITY_LEVEL_SYSTEM: u8 = 2;
    pub const INTEGRITY_LEVEL_KERNEL: u8 = 3;
};

const nsir_core = @import("nsir_core.zig");
const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
const Node = nsir_core.Node;
const Edge = nsir_core.Edge;
const EdgeQuality = nsir_core.EdgeQuality;
const EdgeKey = nsir_core.EdgeKey;
const Qubit = nsir_core.Qubit;

pub const SecurityError = error{
    AccessDenied,
    InvalidSecurityLevel,
    IllegalInformationFlow,
    NonInterferenceViolation,
    IntegrityViolation,
    PolicyViolation,
    InvalidProof,
    InvalidHash,
    InvalidCommitment,
    MerkleVerificationFailed,
    HashChainBroken,
    SeparationOfDutiesViolation,
    LeastPrivilegeViolation,
    BellLaPadulaViolation,
    BibaViolation,
    OutOfMemory,
    InvalidPrincipal,
    InvalidObject,
    CryptographicError,
    ProofGenerationFailed,
    BisimulationFailed,
};

pub const SecurityLevel = enum(u8) {
    PUBLIC = 0,
    INTERNAL = 1,
    CONFIDENTIAL = 2,
    SECRET = 3,
    TOP_SECRET = 4,

    const Self = @This();

    pub fn toString(self: Self) []const u8 {
        return switch (self) {
            .PUBLIC => "PUBLIC",
            .INTERNAL => "INTERNAL",
            .CONFIDENTIAL => "CONFIDENTIAL",
            .SECRET => "SECRET",
            .TOP_SECRET => "TOP_SECRET",
        };
    }

    pub fn fromString(s: []const u8) ?Self {
        if (std.mem.eql(u8, s, "PUBLIC")) return .PUBLIC;
        if (std.mem.eql(u8, s, "INTERNAL")) return .INTERNAL;
        if (std.mem.eql(u8, s, "CONFIDENTIAL")) return .CONFIDENTIAL;
        if (std.mem.eql(u8, s, "SECRET")) return .SECRET;
        if (std.mem.eql(u8, s, "TOP_SECRET")) return .TOP_SECRET;
        return null;
    }

    pub fn toNumeric(self: Self) u8 {
        return @intFromEnum(self);
    }

    pub fn fromNumeric(val: u8) ?Self {
        return switch (val) {
            0 => .PUBLIC,
            1 => .INTERNAL,
            2 => .CONFIDENTIAL,
            3 => .SECRET,
            4 => .TOP_SECRET,
            else => null,
        };
    }

    pub fn lessThanOrEqual(self: Self, other: Self) bool {
        return @intFromEnum(self) <= @intFromEnum(other);
    }

    pub fn lessThan(self: Self, other: Self) bool {
        return @intFromEnum(self) < @intFromEnum(other);
    }

    pub fn greaterThanOrEqual(self: Self, other: Self) bool {
        return @intFromEnum(self) >= @intFromEnum(other);
    }

    pub fn greaterThan(self: Self, other: Self) bool {
        return @intFromEnum(self) > @intFromEnum(other);
    }

    pub fn join(self: Self, other: Self) Self {
        const max_val = @max(@intFromEnum(self), @intFromEnum(other));
        return fromNumeric(max_val).?;
    }

    pub fn meet(self: Self, other: Self) Self {
        const min_val = @min(@intFromEnum(self), @intFromEnum(other));
        return fromNumeric(min_val).?;
    }

    pub fn dominates(self: Self, other: Self) bool {
        return self.greaterThanOrEqual(other);
    }

    pub fn isDominatedBy(self: Self, other: Self) bool {
        return self.lessThanOrEqual(other);
    }
};

pub const IntegrityLevel = enum(u8) {
    UNTRUSTED = 0,
    USER = 1,
    SYSTEM = 2,
    KERNEL = 3,

    const Self = @This();

    pub fn toString(self: Self) []const u8 {
        return switch (self) {
            .UNTRUSTED => "UNTRUSTED",
            .USER => "USER",
            .SYSTEM => "SYSTEM",
            .KERNEL => "KERNEL",
        };
    }

    pub fn fromString(s: []const u8) ?Self {
        if (std.mem.eql(u8, s, "UNTRUSTED")) return .UNTRUSTED;
        if (std.mem.eql(u8, s, "USER")) return .USER;
        if (std.mem.eql(u8, s, "SYSTEM")) return .SYSTEM;
        if (std.mem.eql(u8, s, "KERNEL")) return .KERNEL;
        return null;
    }

    pub fn toNumeric(self: Self) u8 {
        return @intFromEnum(self);
    }

    pub fn fromNumeric(val: u8) ?Self {
        return switch (val) {
            0 => .UNTRUSTED,
            1 => .USER,
            2 => .SYSTEM,
            3 => .KERNEL,
            else => null,
        };
    }

    pub fn lessThanOrEqual(self: Self, other: Self) bool {
        return @intFromEnum(self) <= @intFromEnum(other);
    }

    pub fn greaterThanOrEqual(self: Self, other: Self) bool {
        return @intFromEnum(self) >= @intFromEnum(other);
    }

    pub fn join(self: Self, other: Self) Self {
        const min_val = @min(@intFromEnum(self), @intFromEnum(other));
        return fromNumeric(min_val).?;
    }

    pub fn meet(self: Self, other: Self) Self {
        const max_val = @max(@intFromEnum(self), @intFromEnum(other));
        return fromNumeric(max_val).?;
    }
};

pub const AccessRight = enum(u8) {
    NONE = 0,
    READ = 1,
    WRITE = 2,
    EXECUTE = 4,
    DELETE = 8,
    ADMIN = SecurityProofsConfig.ACCESS_RIGHT_ADMIN_BIT,

    const Self = @This();

    pub fn toString(self: Self) []const u8 {
        return switch (self) {
            .NONE => "NONE",
            .READ => "READ",
            .WRITE => "WRITE",
            .EXECUTE => "EXECUTE",
            .DELETE => "DELETE",
            .ADMIN => "ADMIN",
        };
    }

    pub fn fromString(s: []const u8) ?Self {
        if (std.mem.eql(u8, s, "NONE")) return .NONE;
        if (std.mem.eql(u8, s, "READ")) return .READ;
        if (std.mem.eql(u8, s, "WRITE")) return .WRITE;
        if (std.mem.eql(u8, s, "EXECUTE")) return .EXECUTE;
        if (std.mem.eql(u8, s, "DELETE")) return .DELETE;
        if (std.mem.eql(u8, s, "ADMIN")) return .ADMIN;
        return null;
    }

    pub fn toBitmask(self: Self) u8 {
        return @intFromEnum(self);
    }

    pub fn fromBitmask(mask: u8) Self {
        if (mask >= SecurityProofsConfig.ACCESS_RIGHT_ADMIN_BIT) return .ADMIN;
        if (mask >= SecurityProofsConfig.ACCESS_RIGHT_DELETE_BIT) return .DELETE;
        if (mask >= SecurityProofsConfig.ACCESS_RIGHT_EXECUTE_BIT) return .EXECUTE;
        if (mask >= SecurityProofsConfig.ACCESS_RIGHT_WRITE_BIT) return .WRITE;
        if (mask >= SecurityProofsConfig.ACCESS_RIGHT_READ_BIT) return .READ;
        return .NONE;
    }
};

pub const AccessRightSet = struct {
    rights: u8,

    const Self = @This();

    pub fn init() Self {
        return Self{ .rights = 0 };
    }

    pub fn initWithRights(rights: []const AccessRight) Self {
        var mask: u8 = 0;
        for (rights) |r| {
            mask |= r.toBitmask();
        }
        return Self{ .rights = mask };
    }

    pub fn add(self: *Self, right: AccessRight) void {
        self.rights |= right.toBitmask();
    }

    pub fn remove(self: *Self, right: AccessRight) void {
        self.rights &= ~right.toBitmask();
    }

    pub fn has(self: *const Self, right: AccessRight) bool {
        return (self.rights & right.toBitmask()) != 0;
    }

    pub fn hasAll(self: *const Self, rights: []const AccessRight) bool {
        for (rights) |r| {
            if (!self.has(r)) return false;
        }
        return true;
    }

    pub fn hasAny(self: *const Self, rights: []const AccessRight) bool {
        for (rights) |r| {
            if (self.has(r)) return true;
        }
        return false;
    }

    pub fn unionWith(self: *const Self, other: *const Self) Self {
        return Self{ .rights = self.rights | other.rights };
    }

    pub fn intersectWith(self: *const Self, other: *const Self) Self {
        return Self{ .rights = self.rights & other.rights };
    }

    pub fn subtract(self: *const Self, other: *const Self) Self {
        return Self{ .rights = self.rights & ~other.rights };
    }

    pub fn isEmpty(self: *const Self) bool {
        return self.rights == 0;
    }

    pub fn isSubsetOf(self: *const Self, other: *const Self) bool {
        return (self.rights & other.rights) == self.rights;
    }
};

pub const SecurityCategory = struct {
    id: u64,
    name: []const u8,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, id: u64, name: []const u8) !*Self {
        const cat = try allocator.create(Self);
        cat.* = Self{
            .id = id,
            .name = try allocator.dupe(u8, name),
            .allocator = allocator,
        };
        return cat;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        return SecurityCategory.init(allocator, self.id, self.name);
    }

    pub fn equals(self: *const Self, other: *const Self) bool {
        return self.id == other.id;
    }
};

pub const SecurityLabel = struct {
    confidentiality_level: SecurityLevel,
    integrity_level: IntegrityLevel,
    categories: ArrayList(u64),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, conf_level: SecurityLevel, int_level: IntegrityLevel) Self {
        return Self{
            .confidentiality_level = conf_level,
            .integrity_level = int_level,
            .categories = ArrayList(u64).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.categories.deinit();
    }

    pub fn addCategory(self: *Self, category_id: u64) !void {
        for (self.categories.items) |c| {
            if (c == category_id) return;
        }
        try self.categories.append(category_id);
    }

    pub fn removeCategory(self: *Self, category_id: u64) void {
        var i: usize = 0;
        while (i < self.categories.items.len) {
            if (self.categories.items[i] == category_id) {
                _ = self.categories.orderedRemove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn hasCategory(self: *const Self, category_id: u64) bool {
        for (self.categories.items) |c| {
            if (c == category_id) return true;
        }
        return false;
    }

    pub fn categoriesSubsetOf(self: *const Self, other: *const Self) bool {
        for (self.categories.items) |c| {
            if (!other.hasCategory(c)) return false;
        }
        return true;
    }

    pub fn dominates(self: *const Self, other: *const Self) bool {
        return self.confidentiality_level.dominates(other.confidentiality_level) and
            self.integrity_level.greaterThanOrEqual(other.integrity_level) and
            other.categoriesSubsetOf(self);
    }

    pub fn join(self: *const Self, other: *const Self, allocator: Allocator) !Self {
        var result = Self.init(
            allocator,
            self.confidentiality_level.join(other.confidentiality_level),
            self.integrity_level.join(other.integrity_level),
        );
        for (self.categories.items) |c| {
            try result.addCategory(c);
        }
        for (other.categories.items) |c| {
            try result.addCategory(c);
        }
        return result;
    }

    pub fn meet(self: *const Self, other: *const Self, allocator: Allocator) !Self {
        var result = Self.init(
            allocator,
            self.confidentiality_level.meet(other.confidentiality_level),
            self.integrity_level.meet(other.integrity_level),
        );
        for (self.categories.items) |c| {
            if (other.hasCategory(c)) {
                try result.addCategory(c);
            }
        }
        return result;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var result = Self.init(allocator, self.confidentiality_level, self.integrity_level);
        for (self.categories.items) |c| {
            try result.addCategory(c);
        }
        return result;
    }

    pub fn computeHash(self: *const Self) [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 {
        var hasher = Sha256.init(.{});
        hasher.update(&[_]u8{@intFromEnum(self.confidentiality_level)});
        hasher.update(&[_]u8{@intFromEnum(self.integrity_level)});
        for (self.categories.items) |c| {
            hasher.update(std.mem.asBytes(&c));
        }
        var result: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        hasher.final(&result);
        return result;
    }
};

pub const Principal = struct {
    id: u64,
    name: []const u8,
    security_level: SecurityLevel,
    clearance: SecurityLabel,
    access_rights: AccessRightSet,
    roles: ArrayList(u64),
    attributes: StringHashMap([]const u8),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, id: u64, name: []const u8, security_level: SecurityLevel) !*Self {
        const principal = try allocator.create(Self);
        principal.* = Self{
            .id = id,
            .name = try allocator.dupe(u8, name),
            .security_level = security_level,
            .clearance = SecurityLabel.init(allocator, security_level, .USER),
            .access_rights = AccessRightSet.init(),
            .roles = ArrayList(u64).init(allocator),
            .attributes = StringHashMap([]const u8).init(allocator),
            .allocator = allocator,
        };
        return principal;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
        self.clearance.deinit();
        self.roles.deinit();
        var iter = self.attributes.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.attributes.deinit();
    }

    pub fn addRole(self: *Self, role_id: u64) !void {
        for (self.roles.items) |r| {
            if (r == role_id) return;
        }
        try self.roles.append(role_id);
    }

    pub fn removeRole(self: *Self, role_id: u64) void {
        var i: usize = 0;
        while (i < self.roles.items.len) {
            if (self.roles.items[i] == role_id) {
                _ = self.roles.orderedRemove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn hasRole(self: *const Self, role_id: u64) bool {
        for (self.roles.items) |r| {
            if (r == role_id) return true;
        }
        return false;
    }

    pub fn setAttribute(self: *Self, key: []const u8, value: []const u8) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        const val_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(val_copy);

        if (self.attributes.fetchRemove(key_copy)) |removed| {
            self.allocator.free(removed.key);
            self.allocator.free(removed.value);
            self.allocator.free(key_copy);
        }
        try self.attributes.put(key_copy, val_copy);
    }

    pub fn getAttribute(self: *const Self, key: []const u8) ?[]const u8 {
        return self.attributes.get(key);
    }

    pub fn canAccess(self: *const Self, object_label: *const SecurityLabel, right: AccessRight) bool {
        if (!self.access_rights.has(right)) return false;
        return self.clearance.dominates(object_label);
    }

    pub fn computeHash(self: *const Self) [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 {
        var hasher = Sha256.init(.{});
        hasher.update(std.mem.asBytes(&self.id));
        hasher.update(self.name);
        hasher.update(&[_]u8{@intFromEnum(self.security_level)});
        const label_hash = self.clearance.computeHash();
        hasher.update(&label_hash);
        hasher.update(&[_]u8{self.access_rights.rights});
        var result: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        hasher.final(&result);
        return result;
    }
};

pub const SecureObject = struct {
    id: u64,
    name: []const u8,
    label: SecurityLabel,
    owner_id: u64,
    data_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, id: u64, name: []const u8, label: SecurityLabel, owner_id: u64) !*Self {
        const obj = try allocator.create(Self);
        obj.* = Self{
            .id = id,
            .name = try allocator.dupe(u8, name),
            .label = try label.clone(allocator),
            .owner_id = owner_id,
            .data_hash = std.mem.zeroes([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8),
            .allocator = allocator,
        };
        return obj;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
        self.label.deinit();
    }

    pub fn setDataHash(self: *Self, data: []const u8) void {
        Sha256.hash(data, &self.data_hash, .{});
    }

    pub fn verifyDataHash(self: *const Self, data: []const u8) bool {
        var computed: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(data, &computed, .{});
        return timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, self.data_hash, computed);
    }
};

pub const FlowEdge = struct {
    source_id: u64,
    target_id: u64,
    source_level: SecurityLevel,
    target_level: SecurityLevel,
    flow_type: FlowType,
    is_explicit: bool,

    pub const FlowType = enum(u8) {
        EXPLICIT = 0,
        IMPLICIT = 1,
        COVERT = 2,

        pub fn toString(self: FlowType) []const u8 {
            return switch (self) {
                .EXPLICIT => "EXPLICIT",
                .IMPLICIT => "IMPLICIT",
                .COVERT => "COVERT",
            };
        }
    };

    const Self = @This();

    pub fn isSecure(self: *const Self) bool {
        return self.source_level.lessThanOrEqual(self.target_level);
    }

    pub fn isIllegal(self: *const Self) bool {
        return self.source_level.greaterThan(self.target_level);
    }
};

pub const FlowGraph = struct {
    nodes: AutoHashMap(u64, SecurityLevel),
    edges: ArrayList(FlowEdge),
    adjacency: AutoHashMap(u64, ArrayList(u64)),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .nodes = AutoHashMap(u64, SecurityLevel).init(allocator),
            .edges = ArrayList(FlowEdge).init(allocator),
            .adjacency = AutoHashMap(u64, ArrayList(u64)).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.nodes.deinit();
        self.edges.deinit();
        var iter = self.adjacency.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.adjacency.deinit();
    }

    pub fn addNode(self: *Self, node_id: u64, level: SecurityLevel) !void {
        try self.nodes.put(node_id, level);
        if (!self.adjacency.contains(node_id)) {
            try self.adjacency.put(node_id, ArrayList(u64).init(self.allocator));
        }
    }

    pub fn addEdge(self: *Self, source_id: u64, target_id: u64, flow_type: FlowEdge.FlowType, is_explicit: bool) !void {
        const source_level = self.nodes.get(source_id) orelse return SecurityError.InvalidPrincipal;
        const target_level = self.nodes.get(target_id) orelse return SecurityError.InvalidObject;

        try self.edges.append(FlowEdge{
            .source_id = source_id,
            .target_id = target_id,
            .source_level = source_level,
            .target_level = target_level,
            .flow_type = flow_type,
            .is_explicit = is_explicit,
        });

        if (self.adjacency.getPtr(source_id)) |adj_list| {
            try adj_list.append(target_id);
        }
    }

    pub fn getSuccessors(self: *const Self, node_id: u64) []const u64 {
        if (self.adjacency.get(node_id)) |adj_list| {
            return adj_list.items;
        }
        return &[_]u64{};
    }

    pub fn nodeCount(self: *const Self) usize {
        return self.nodes.count();
    }

    pub fn edgeCount(self: *const Self) usize {
        return self.edges.items.len;
    }
};

pub const InformationFlowLattice = struct {
    bottom: SecurityLevel,
    top: SecurityLevel,
    levels: [5]SecurityLevel,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .bottom = .PUBLIC,
            .top = .TOP_SECRET,
            .levels = .{ .PUBLIC, .INTERNAL, .CONFIDENTIAL, .SECRET, .TOP_SECRET },
        };
    }

    pub fn join(self: *const Self, a: SecurityLevel, b: SecurityLevel) SecurityLevel {
        _ = self;
        return a.join(b);
    }

    pub fn meet(self: *const Self, a: SecurityLevel, b: SecurityLevel) SecurityLevel {
        _ = self;
        return a.meet(b);
    }

    pub fn lessThanOrEqual(self: *const Self, a: SecurityLevel, b: SecurityLevel) bool {
        _ = self;
        return a.lessThanOrEqual(b);
    }

    pub fn isBottom(self: *const Self, level: SecurityLevel) bool {
        return level == self.bottom;
    }

    pub fn isTop(self: *const Self, level: SecurityLevel) bool {
        return level == self.top;
    }

    pub fn covers(self: *const Self, higher: SecurityLevel, lower: SecurityLevel) bool {
        _ = self;
        const h = @intFromEnum(higher);
        const l = @intFromEnum(lower);
        return h == l + 1;
    }

    pub fn distance(self: *const Self, from: SecurityLevel, to: SecurityLevel) i8 {
        _ = self;
        const f: i8 = @intCast(@intFromEnum(from));
        const t: i8 = @intCast(@intFromEnum(to));
        return t - f;
    }
};

pub const IllegalFlow = struct {
    source_id: u64,
    target_id: u64,
    source_level: SecurityLevel,
    target_level: SecurityLevel,
    flow_type: FlowEdge.FlowType,
    severity: u8,

    const Self = @This();

    pub fn computeSeverity(self: *Self) void {
        const src = @intFromEnum(self.source_level);
        const tgt = @intFromEnum(self.target_level);
        if (src <= tgt) {
            self.severity = 0;
            return;
        }
        const diff: u8 = src - tgt;
        const product = @mulWithOverflow(diff, SecurityProofsConfig.SEVERITY_MULTIPLIER);
        if (product[1] != 0) {
            self.severity = SecurityProofsConfig.SEVERITY_MAX;
        } else {
            self.severity = @min(product[0], SecurityProofsConfig.SEVERITY_MAX);
        }
    }
};

pub const FlowProofStep = struct {
    step_number: u64,
    description: []const u8,
    from_node: u64,
    to_node: u64,
    from_level: SecurityLevel,
    to_level: SecurityLevel,
    rule_applied: []const u8,
    is_secure: bool,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        step_number: u64,
        description: []const u8,
        from_node: u64,
        to_node: u64,
        from_level: SecurityLevel,
        to_level: SecurityLevel,
        rule_applied: []const u8,
        is_secure: bool,
    ) !*Self {
        const step = try allocator.create(Self);
        step.* = Self{
            .step_number = step_number,
            .description = try allocator.dupe(u8, description),
            .from_node = from_node,
            .to_node = to_node,
            .from_level = from_level,
            .to_level = to_level,
            .rule_applied = try allocator.dupe(u8, rule_applied),
            .is_secure = is_secure,
            .allocator = allocator,
        };
        return step;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.description);
        self.allocator.free(self.rule_applied);
    }
};

pub const InformationFlowAnalysis = struct {
    flow_graph: FlowGraph,
    lattice: InformationFlowLattice,
    illegal_flows: ArrayList(IllegalFlow),
    closure_computed: bool,
    reachability: AutoHashMap(u64, ArrayList(u64)),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .flow_graph = FlowGraph.init(allocator),
            .lattice = InformationFlowLattice.init(),
            .illegal_flows = ArrayList(IllegalFlow).init(allocator),
            .closure_computed = false,
            .reachability = AutoHashMap(u64, ArrayList(u64)).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.flow_graph.deinit();
        self.illegal_flows.deinit();
        var iter = self.reachability.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.reachability.deinit();
    }

    pub fn addVariable(self: *Self, var_id: u64, level: SecurityLevel) !void {
        try self.flow_graph.addNode(var_id, level);
        self.closure_computed = false;
    }

    pub fn addFlow(self: *Self, source_id: u64, target_id: u64, flow_type: FlowEdge.FlowType, is_explicit: bool) !void {
        try self.flow_graph.addEdge(source_id, target_id, flow_type, is_explicit);
        self.closure_computed = false;
    }

    pub fn detectIllegalFlows(self: *Self) ![]const IllegalFlow {
        self.illegal_flows.clearRetainingCapacity();

        for (self.flow_graph.edges.items) |edge| {
            if (edge.isIllegal()) {
                var illegal = IllegalFlow{
                    .source_id = edge.source_id,
                    .target_id = edge.target_id,
                    .source_level = edge.source_level,
                    .target_level = edge.target_level,
                    .flow_type = edge.flow_type,
                    .severity = 0,
                };
                illegal.computeSeverity();
                try self.illegal_flows.append(illegal);
            }
        }

        if (self.closure_computed) {
            var reach_iter = self.reachability.iterator();
            while (reach_iter.next()) |entry| {
                const source_id = entry.key_ptr.*;
                const source_level = self.flow_graph.nodes.get(source_id) orelse continue;

                for (entry.value_ptr.items) |target_id| {
                    const target_level = self.flow_graph.nodes.get(target_id) orelse continue;

                    if (source_level.greaterThan(target_level)) {
                        var already_found = false;
                        for (self.illegal_flows.items) |existing| {
                            if (existing.source_id == source_id and existing.target_id == target_id) {
                                already_found = true;
                                break;
                            }
                        }
                        if (!already_found) {
                            var illegal = IllegalFlow{
                                .source_id = source_id,
                                .target_id = target_id,
                                .source_level = source_level,
                                .target_level = target_level,
                                .flow_type = .IMPLICIT,
                                .severity = 0,
                            };
                            illegal.computeSeverity();
                            try self.illegal_flows.append(illegal);
                        }
                    }
                }
            }
        }

        return self.illegal_flows.items;
    }

    pub fn computeSecurityClosure(self: *Self) !void {
        var reach_iter = self.reachability.iterator();
        while (reach_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.reachability.clearRetainingCapacity();

        var node_iter = self.flow_graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node_id = entry.key_ptr.*;
            var reachable = ArrayList(u64).init(self.allocator);
            errdefer reachable.deinit();

            var visited = AutoHashMap(u64, void).init(self.allocator);
            defer visited.deinit();

            var fifo = std.fifo.LinearFifo(u64, .Dynamic).init(self.allocator);
            defer fifo.deinit();

            try visited.put(node_id, {});
            try fifo.writeItem(node_id);

            while (fifo.readItem()) |current| {
                const successors = self.flow_graph.getSuccessors(current);
                for (successors) |succ| {
                    if (!visited.contains(succ)) {
                        try visited.put(succ, {});
                        try reachable.append(succ);
                        try fifo.writeItem(succ);
                    }
                }
            }

            try self.reachability.put(node_id, reachable);
        }

        self.closure_computed = true;
    }

    pub fn verifyNonInterference(self: *Self, observer_level: SecurityLevel) !bool {
        if (!self.closure_computed) {
            try self.computeSecurityClosure();
        }

        var high_nodes = ArrayList(u64).init(self.allocator);
        defer high_nodes.deinit();

        var low_nodes = ArrayList(u64).init(self.allocator);
        defer low_nodes.deinit();

        var node_iter = self.flow_graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node_id = entry.key_ptr.*;
            const level = entry.value_ptr.*;

            if (level.greaterThan(observer_level)) {
                try high_nodes.append(node_id);
            } else {
                try low_nodes.append(node_id);
            }
        }

        for (high_nodes.items) |high_node| {
            if (self.reachability.get(high_node)) |reachable| {
                for (reachable.items) |reach_node| {
                    for (low_nodes.items) |low_node| {
                        if (reach_node == low_node) {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }

    pub fn generateFlowProof(self: *Self, allocator: Allocator) !ArrayList(*FlowProofStep) {
        var steps = ArrayList(*FlowProofStep).init(allocator);
        var step_num: u64 = 1;

        const step1 = try FlowProofStep.init(
            allocator,
            step_num,
            "Initialize information flow analysis with lattice structure",
            0,
            0,
            .PUBLIC,
            .PUBLIC,
            "LATTICE_INIT",
            true,
        );
        try steps.append(step1);
        step_num += 1;

        for (self.flow_graph.edges.items) |edge| {
            const is_secure = edge.isSecure();
            const description = if (is_secure)
                "Verified secure flow: source level ≤ target level"
            else
                "VIOLATION: Illegal flow detected (high → low)";

            const rule = if (is_secure) "SIMPLE_SECURITY" else "VIOLATION_DETECTED";

            const step = try FlowProofStep.init(
                allocator,
                step_num,
                description,
                edge.source_id,
                edge.target_id,
                edge.source_level,
                edge.target_level,
                rule,
                is_secure,
            );
            try steps.append(step);
            step_num += 1;
        }

        const has_violations = self.illegal_flows.items.len > 0;
        const conclusion = if (has_violations)
            "PROOF FAILED: Information flow policy violations detected"
        else
            "PROOF COMPLETE: All information flows satisfy security policy";

        const final_step = try FlowProofStep.init(
            allocator,
            step_num,
            conclusion,
            0,
            0,
            .PUBLIC,
            .TOP_SECRET,
            "CONCLUSION",
            !has_violations,
        );
        try steps.append(final_step);

        return steps;
    }

    pub fn isSecure(self: *Self) !bool {
        _ = try self.detectIllegalFlows();
        return self.illegal_flows.items.len == 0;
    }

    pub fn getFlowSeverity(self: *Self, source_id: u64, target_id: u64) ?u8 {
        for (self.illegal_flows.items) |flow| {
            if (flow.source_id == source_id and flow.target_id == target_id) {
                return flow.severity;
            }
        }
        return null;
    }

    pub fn computeInferenceSeverity(self: *Self, source_id: u64, target_id: u64) u8 {
        const source_level = self.flow_graph.nodes.get(source_id) orelse return 0;
        const target_level = self.flow_graph.nodes.get(target_id) orelse return 0;
        if (!source_level.greaterThan(target_level)) return 0;
        var flow = IllegalFlow{
            .source_id = source_id,
            .target_id = target_id,
            .source_level = source_level,
            .target_level = target_level,
            .flow_type = .IMPLICIT,
            .severity = 0,
        };
        flow.computeSeverity();
        return flow.severity;
    }

    pub fn assessInferenceRisk(self: *Self) !usize {
        if (!self.closure_computed) {
            try self.computeSecurityClosure();
        }
        var high_severity_count: usize = 0;
        var reach_iter = self.reachability.iterator();
        while (reach_iter.next()) |entry| {
            const source_id = entry.key_ptr.*;
            const source_level = self.flow_graph.nodes.get(source_id) orelse continue;
            for (entry.value_ptr.items) |target_id| {
                const target_level = self.flow_graph.nodes.get(target_id) orelse continue;
                if (source_level.greaterThan(target_level)) {
                    const severity = self.computeInferenceSeverity(source_id, target_id);
                    if (severity >= SecurityProofsConfig.SEVERITY_MAX / 2) {
                        high_severity_count += 1;
                    }
                }
            }
        }
        return high_severity_count;
    }
};

pub const SystemState = struct {
    variables: AutoHashMap(u64, []const u8),
    variable_levels: AutoHashMap(u64, SecurityLevel),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .variables = AutoHashMap(u64, []const u8).init(allocator),
            .variable_levels = AutoHashMap(u64, SecurityLevel).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.variables.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.variables.deinit();
        self.variable_levels.deinit();
    }

    pub fn setVariable(self: *Self, var_id: u64, value: []const u8, level: SecurityLevel) !void {
        const val_copy = try self.allocator.dupe(u8, value);
        if (self.variables.fetchRemove(var_id)) |removed| {
            self.allocator.free(removed.value);
        }
        try self.variables.put(var_id, val_copy);
        try self.variable_levels.put(var_id, level);
    }

    pub fn getVariable(self: *const Self, var_id: u64) ?[]const u8 {
        return self.variables.get(var_id);
    }

    pub fn getLevel(self: *const Self, var_id: u64) ?SecurityLevel {
        return self.variable_levels.get(var_id);
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var new_state = Self.init(allocator);
        var iter = self.variables.iterator();
        while (iter.next()) |entry| {
            const val_copy = try allocator.dupe(u8, entry.value_ptr.*);
            try new_state.variables.put(entry.key_ptr.*, val_copy);
        }
        var level_iter = self.variable_levels.iterator();
        while (level_iter.next()) |entry| {
            try new_state.variable_levels.put(entry.key_ptr.*, entry.value_ptr.*);
        }
        return new_state;
    }

    pub fn lowProjection(self: *const Self, observer_level: SecurityLevel, allocator: Allocator) !Self {
        var projection = Self.init(allocator);
        var iter = self.variables.iterator();
        while (iter.next()) |entry| {
            const var_id = entry.key_ptr.*;
            const level = self.variable_levels.get(var_id) orelse continue;

            if (level.lessThanOrEqual(observer_level)) {
                const val_copy = try allocator.dupe(u8, entry.value_ptr.*);
                try projection.variables.put(var_id, val_copy);
                try projection.variable_levels.put(var_id, level);
            }
        }
        return projection;
    }

    pub fn equals(self: *const Self, other: *const Self) bool {
        if (self.variables.count() != other.variables.count()) return false;
        var iter = self.variables.iterator();
        while (iter.next()) |entry| {
            const other_val = other.variables.get(entry.key_ptr.*) orelse return false;
            if (!std.mem.eql(u8, entry.value_ptr.*, other_val)) return false;
        }
        return true;
    }
};

pub const NonInterferenceProperty = struct {
    observer_level: SecurityLevel,
    high_variables: ArrayList(u64),
    low_variables: ArrayList(u64),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, observer_level: SecurityLevel) Self {
        return Self{
            .observer_level = observer_level,
            .high_variables = ArrayList(u64).init(allocator),
            .low_variables = ArrayList(u64).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.high_variables.deinit();
        self.low_variables.deinit();
    }

    pub fn classifyVariable(self: *Self, var_id: u64, level: SecurityLevel) !void {
        if (level.greaterThan(self.observer_level)) {
            try self.high_variables.append(var_id);
        } else {
            try self.low_variables.append(var_id);
        }
    }

    pub fn isHighVariable(self: *const Self, var_id: u64) bool {
        for (self.high_variables.items) |h| {
            if (h == var_id) return true;
        }
        return false;
    }

    pub fn isLowVariable(self: *const Self, var_id: u64) bool {
        for (self.low_variables.items) |l| {
            if (l == var_id) return true;
        }
        return false;
    }
};

pub const BisimulationRelation = struct {
    state_pairs: ArrayList(StatePair),
    allocator: Allocator,

    pub const StatePair = struct {
        state1_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
        state2_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    };

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .state_pairs = ArrayList(StatePair).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.state_pairs.deinit();
    }

    pub fn addPair(self: *Self, state1: *const SystemState, state2: *const SystemState) !void {
        const hash1 = computeStateHash(state1);
        const hash2 = computeStateHash(state2);
        try self.state_pairs.append(StatePair{
            .state1_hash = hash1,
            .state2_hash = hash2,
        });
    }

    fn computeStateHash(state: *const SystemState) [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 {
        var hasher = Sha256.init(.{});
        var iter = state.variables.iterator();
        while (iter.next()) |entry| {
            hasher.update(std.mem.asBytes(&entry.key_ptr.*));
            hasher.update(entry.value_ptr.*);
        }
        var result: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        hasher.final(&result);
        return result;
    }

    pub fn areBisimilar(self: *const Self, state1: *const SystemState, state2: *const SystemState, observer_level: SecurityLevel, allocator: Allocator) !bool {
        _ = self;
        var proj1 = try state1.lowProjection(observer_level, allocator);
        defer proj1.deinit();
        var proj2 = try state2.lowProjection(observer_level, allocator);
        defer proj2.deinit();
        return proj1.equals(&proj2);
    }

    pub fn pairCount(self: *const Self) usize {
        return self.state_pairs.items.len;
    }
};

pub const NonInterferenceProofStep = struct {
    step_number: u64,
    lemma_name: []const u8,
    description: []const u8,
    states_examined: u64,
    is_valid: bool,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        step_number: u64,
        lemma_name: []const u8,
        description: []const u8,
        states_examined: u64,
        is_valid: bool,
    ) !*Self {
        const step = try allocator.create(Self);
        step.* = Self{
            .step_number = step_number,
            .lemma_name = try allocator.dupe(u8, lemma_name),
            .description = try allocator.dupe(u8, description),
            .states_examined = states_examined,
            .is_valid = is_valid,
            .allocator = allocator,
        };
        return step;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.lemma_name);
        self.allocator.free(self.description);
    }
};

pub const NonInterferenceProver = struct {
    property: NonInterferenceProperty,
    bisimulation: BisimulationRelation,
    proof_steps: ArrayList(*NonInterferenceProofStep),
    is_proven: bool,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, observer_level: SecurityLevel) Self {
        return Self{
            .property = NonInterferenceProperty.init(allocator, observer_level),
            .bisimulation = BisimulationRelation.init(allocator),
            .proof_steps = ArrayList(*NonInterferenceProofStep).init(allocator),
            .is_proven = false,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.property.deinit();
        self.bisimulation.deinit();
        for (self.proof_steps.items) |step| {
            step.deinit();
            self.allocator.destroy(step);
        }
        self.proof_steps.deinit();
    }

    pub fn addVariable(self: *Self, var_id: u64, level: SecurityLevel) !void {
        try self.property.classifyVariable(var_id, level);
    }

    pub fn proveNonInterference(self: *Self, initial_state: *const SystemState, alternate_state: *const SystemState) !bool {
        for (self.proof_steps.items) |step| {
            step.deinit();
            self.allocator.destroy(step);
        }
        self.proof_steps.clearRetainingCapacity();

        var step_num: u64 = 1;

        const step1 = try NonInterferenceProofStep.init(
            self.allocator,
            step_num,
            "SETUP",
            "Initialize non-interference proof with observer level classification",
            0,
            true,
        );
        try self.proof_steps.append(step1);
        step_num += 1;

        const step2 = try NonInterferenceProofStep.init(
            self.allocator,
            step_num,
            "HIGH_LOW_PARTITION",
            "Partition variables into high (above observer) and low (at or below observer) sets",
            self.property.high_variables.items.len + self.property.low_variables.items.len,
            true,
        );
        try self.proof_steps.append(step2);
        step_num += 1;

        const bisimilar = try self.bisimulation.areBisimilar(
            initial_state,
            alternate_state,
            self.property.observer_level,
            self.allocator,
        );

        const step3 = try NonInterferenceProofStep.init(
            self.allocator,
            step_num,
            "BISIMULATION_CHECK",
            if (bisimilar)
                "States are bisimilar under low projection - observer cannot distinguish"
            else
                "States differ in low projection - non-interference VIOLATED",
            2,
            bisimilar,
        );
        try self.proof_steps.append(step3);
        step_num += 1;

        if (bisimilar) {
            try self.bisimulation.addPair(initial_state, alternate_state);
        }

        const step4 = try NonInterferenceProofStep.init(
            self.allocator,
            step_num,
            "UNWINDING_LEMMA",
            if (bisimilar)
                "Unwinding lemma satisfied: high variations produce bisimilar low behaviors"
            else
                "Unwinding lemma FAILED: high input affects low output",
            self.bisimulation.pairCount(),
            bisimilar,
        );
        try self.proof_steps.append(step4);
        step_num += 1;

        const step5 = try NonInterferenceProofStep.init(
            self.allocator,
            step_num,
            "CONCLUSION",
            if (bisimilar)
                "PROOF COMPLETE: Non-interference property holds"
            else
                "PROOF FAILED: Non-interference property violated",
            0,
            bisimilar,
        );
        try self.proof_steps.append(step5);

        self.is_proven = bisimilar;
        return bisimilar;
    }

    pub fn getProofSteps(self: *const Self) []const *NonInterferenceProofStep {
        return self.proof_steps.items;
    }
};

pub const AccessControlPolicy = struct {
    id: u64,
    name: []const u8,
    policy_type: PolicyType,
    rules: ArrayList(*AccessRule),
    allocator: Allocator,

    pub const PolicyType = enum(u8) {
        RBAC = 0,
        ABAC = 1,
        MAC = 2,
        DAC = 3,

        pub fn toString(self: PolicyType) []const u8 {
            return switch (self) {
                .RBAC => "RBAC",
                .ABAC => "ABAC",
                .MAC => "MAC",
                .DAC => "DAC",
            };
        }
    };

    const Self = @This();

    pub fn init(allocator: Allocator, id: u64, name: []const u8, policy_type: PolicyType) !*Self {
        const policy = try allocator.create(Self);
        policy.* = Self{
            .id = id,
            .name = try allocator.dupe(u8, name),
            .policy_type = policy_type,
            .rules = ArrayList(*AccessRule).init(allocator),
            .allocator = allocator,
        };
        return policy;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
        for (self.rules.items) |rule| {
            rule.deinit();
            self.allocator.destroy(rule);
        }
        self.rules.deinit();
    }

    pub fn addRule(self: *Self, rule: *AccessRule) !void {
        try self.rules.append(rule);
    }

    pub fn evaluate(self: *const Self, subject: *const Principal, object: *const SecureObject, right: AccessRight) bool {
        for (self.rules.items) |rule| {
            if (rule.matches(subject, object, right)) {
                return rule.effect == .ALLOW;
            }
        }
        return false;
    }
};

pub const AccessRule = struct {
    id: u64,
    subject_pattern: SubjectPattern,
    object_pattern: ObjectPattern,
    required_rights: AccessRightSet,
    effect: Effect,
    priority: u32,
    allocator: Allocator,

    pub const Effect = enum(u8) {
        ALLOW = 0,
        DENY = 1,
    };

    pub const SubjectPattern = struct {
        role_id: ?u64,
        min_security_level: ?SecurityLevel,
        required_attribute_key: ?[]const u8,
        required_attribute_value: ?[]const u8,

        pub fn matches(self: *const SubjectPattern, subject: *const Principal) bool {
            if (self.role_id) |rid| {
                if (!subject.hasRole(rid)) return false;
            }
            if (self.min_security_level) |min_level| {
                if (!subject.security_level.greaterThanOrEqual(min_level)) return false;
            }
            if (self.required_attribute_key) |key| {
                const val = subject.getAttribute(key);
                if (self.required_attribute_value) |expected| {
                    if (val == null or !std.mem.eql(u8, val.?, expected)) return false;
                } else {
                    if (val == null) return false;
                }
            }
            return true;
        }
    };

    pub const ObjectPattern = struct {
        owner_id: ?u64,
        max_confidentiality: ?SecurityLevel,
        required_category: ?u64,

        pub fn matches(self: *const ObjectPattern, object: *const SecureObject) bool {
            if (self.owner_id) |oid| {
                if (object.owner_id != oid) return false;
            }
            if (self.max_confidentiality) |max_conf| {
                if (!object.label.confidentiality_level.lessThanOrEqual(max_conf)) return false;
            }
            if (self.required_category) |cat| {
                if (!object.label.hasCategory(cat)) return false;
            }
            return true;
        }
    };

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        id: u64,
        subject_pattern: SubjectPattern,
        object_pattern: ObjectPattern,
        required_rights: AccessRightSet,
        effect: Effect,
        priority: u32,
    ) !*Self {
        const rule = try allocator.create(Self);
        rule.* = Self{
            .id = id,
            .subject_pattern = subject_pattern,
            .object_pattern = object_pattern,
            .required_rights = required_rights,
            .effect = effect,
            .priority = priority,
            .allocator = allocator,
        };
        return rule;
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn matches(self: *const Self, subject: *const Principal, object: *const SecureObject, right: AccessRight) bool {
        if (!self.subject_pattern.matches(subject)) return false;
        if (!self.object_pattern.matches(object)) return false;
        if (!self.required_rights.has(right)) return false;
        return true;
    }
};

pub const AccessControlMatrix = struct {
    matrix: MatrixMap,
    subjects: AutoHashMap(u64, void),
    objects: AutoHashMap(u64, void),
    allocator: Allocator,

    pub const MatrixKey = struct {
        subject_id: u64,
        object_id: u64,
    };

    const MatrixKeyContext = struct {
        pub fn hash(self: @This(), key: MatrixKey) u64 {
            _ = self;
            var hasher = std.hash.Wyhash.init(0x517cc1b727220a95);
            hasher.update(std.mem.asBytes(&key.subject_id));
            hasher.update(std.mem.asBytes(&key.object_id));
            return hasher.final();
        }

        pub fn eql(self: @This(), a: MatrixKey, b: MatrixKey) bool {
            _ = self;
            return a.subject_id == b.subject_id and a.object_id == b.object_id;
        }
    };

    const MatrixMap = std.HashMap(MatrixKey, AccessRightSet, MatrixKeyContext, std.hash_map.default_max_load_percentage);

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .matrix = MatrixMap.init(allocator),
            .subjects = AutoHashMap(u64, void).init(allocator),
            .objects = AutoHashMap(u64, void).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.matrix.deinit();
        self.subjects.deinit();
        self.objects.deinit();
    }

    pub fn addSubject(self: *Self, subject_id: u64) !void {
        try self.subjects.put(subject_id, {});
    }

    pub fn addObject(self: *Self, object_id: u64) !void {
        try self.objects.put(object_id, {});
    }

    pub fn setRights(self: *Self, subject_id: u64, object_id: u64, rights: AccessRightSet) !void {
        const key = MatrixKey{ .subject_id = subject_id, .object_id = object_id };
        try self.matrix.put(key, rights);
    }

    pub fn getRights(self: *const Self, subject_id: u64, object_id: u64) AccessRightSet {
        const key = MatrixKey{ .subject_id = subject_id, .object_id = object_id };
        return self.matrix.get(key) orelse AccessRightSet.init();
    }

    pub fn hasRight(self: *const Self, subject_id: u64, object_id: u64, right: AccessRight) bool {
        const rights = self.getRights(subject_id, object_id);
        return rights.has(right);
    }

    pub fn grantRight(self: *Self, subject_id: u64, object_id: u64, right: AccessRight) !void {
        const key = MatrixKey{ .subject_id = subject_id, .object_id = object_id };
        var rights = self.matrix.get(key) orelse AccessRightSet.init();
        rights.add(right);
        try self.matrix.put(key, rights);
    }

    pub fn revokeRight(self: *Self, subject_id: u64, object_id: u64, right: AccessRight) !void {
        const key = MatrixKey{ .subject_id = subject_id, .object_id = object_id };
        if (self.matrix.getPtr(key)) |rights| {
            rights.remove(right);
        }
    }
};

pub const AccessProofStep = struct {
    step_number: u64,
    check_type: []const u8,
    subject_id: u64,
    object_id: u64,
    right: AccessRight,
    result: bool,
    reason: []const u8,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        step_number: u64,
        check_type: []const u8,
        subject_id: u64,
        object_id: u64,
        right: AccessRight,
        result: bool,
        reason: []const u8,
    ) !*Self {
        const step = try allocator.create(Self);
        step.* = Self{
            .step_number = step_number,
            .check_type = try allocator.dupe(u8, check_type),
            .subject_id = subject_id,
            .object_id = object_id,
            .right = right,
            .result = result,
            .reason = try allocator.dupe(u8, reason),
            .allocator = allocator,
        };
        return step;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.check_type);
        self.allocator.free(self.reason);
    }
};

pub const SeparationOfDutiesConstraint = struct {
    id: u64,
    conflicting_roles: ArrayList(u64),
    min_separation: u32,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, id: u64) !*Self {
        const constraint = try allocator.create(Self);
        constraint.* = Self{
            .id = id,
            .conflicting_roles = ArrayList(u64).init(allocator),
            .min_separation = 2,
            .allocator = allocator,
        };
        return constraint;
    }

    pub fn deinit(self: *Self) void {
        self.conflicting_roles.deinit();
    }

    pub fn addConflictingRole(self: *Self, role_id: u64) !void {
        try self.conflicting_roles.append(role_id);
    }

    pub fn isViolated(self: *const Self, principal: *const Principal) bool {
        var count: u32 = 0;
        for (self.conflicting_roles.items) |role_id| {
            if (principal.hasRole(role_id)) {
                count += 1;
            }
        }
        return count >= self.min_separation;
    }
};

pub const AccessControlVerifier = struct {
    policies: ArrayList(*AccessControlPolicy),
    matrix: AccessControlMatrix,
    sod_constraints: ArrayList(*SeparationOfDutiesConstraint),
    proof_steps: ArrayList(*AccessProofStep),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .policies = ArrayList(*AccessControlPolicy).init(allocator),
            .matrix = AccessControlMatrix.init(allocator),
            .sod_constraints = ArrayList(*SeparationOfDutiesConstraint).init(allocator),
            .proof_steps = ArrayList(*AccessProofStep).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.policies.items) |policy| {
            policy.deinit();
            self.allocator.destroy(policy);
        }
        self.policies.deinit();
        self.matrix.deinit();
        for (self.sod_constraints.items) |constraint| {
            constraint.deinit();
            self.allocator.destroy(constraint);
        }
        self.sod_constraints.deinit();
        for (self.proof_steps.items) |step| {
            step.deinit();
            self.allocator.destroy(step);
        }
        self.proof_steps.deinit();
    }

    pub fn addPolicy(self: *Self, policy: *AccessControlPolicy) !void {
        try self.policies.append(policy);
    }

    pub fn addSodConstraint(self: *Self, constraint: *SeparationOfDutiesConstraint) !void {
        try self.sod_constraints.append(constraint);
    }

    pub fn verifyPolicy(self: *Self, subject: *const Principal, object: *const SecureObject, right: AccessRight) !bool {
        for (self.proof_steps.items) |step| {
            step.deinit();
            self.allocator.destroy(step);
        }
        self.proof_steps.clearRetainingCapacity();

        var step_num: u64 = 1;

        const matrix_check = self.matrix.hasRight(subject.id, object.id, right);
        const step1 = try AccessProofStep.init(
            self.allocator,
            step_num,
            "MATRIX_CHECK",
            subject.id,
            object.id,
            right,
            matrix_check,
            if (matrix_check) "Access granted by matrix" else "Not found in matrix",
        );
        try self.proof_steps.append(step1);
        step_num += 1;

        var policy_allows = false;
        for (self.policies.items) |policy| {
            if (policy.evaluate(subject, object, right)) {
                policy_allows = true;
                break;
            }
        }

        const step2 = try AccessProofStep.init(
            self.allocator,
            step_num,
            "POLICY_CHECK",
            subject.id,
            object.id,
            right,
            policy_allows,
            if (policy_allows) "Access granted by policy" else "No matching allow policy",
        );
        try self.proof_steps.append(step2);
        step_num += 1;

        const clearance_check = subject.canAccess(&object.label, right);
        const step3 = try AccessProofStep.init(
            self.allocator,
            step_num,
            "CLEARANCE_CHECK",
            subject.id,
            object.id,
            right,
            clearance_check,
            if (clearance_check) "Clearance sufficient" else "Insufficient clearance",
        );
        try self.proof_steps.append(step3);
        step_num += 1;

        const final_result = matrix_check or (policy_allows and clearance_check);
        const step4 = try AccessProofStep.init(
            self.allocator,
            step_num,
            "FINAL_DECISION",
            subject.id,
            object.id,
            right,
            final_result,
            if (final_result) "ACCESS GRANTED" else "ACCESS DENIED",
        );
        try self.proof_steps.append(step4);

        return final_result;
    }

    pub fn checkSeparationOfDuties(self: *Self, principal: *const Principal) !bool {
        for (self.sod_constraints.items) |constraint| {
            if (constraint.isViolated(principal)) {
                return false;
            }
        }
        return true;
    }

    pub fn verifyLeastPrivilege(_: *Self, principal: *const Principal, required_rights: []const AccessRight) bool {
        var excess_rights: u32 = 0;
        const all_rights = [_]AccessRight{ .READ, .WRITE, .EXECUTE, .DELETE, .ADMIN };

        for (all_rights) |right| {
            if (principal.access_rights.has(right)) {
                var is_required = false;
                for (required_rights) |req| {
                    if (right == req) {
                        is_required = true;
                        break;
                    }
                }
                if (!is_required) {
                    excess_rights += 1;
                }
            }
        }

        return excess_rights == 0;
    }

    pub fn generateAccessProof(self: *const Self) []const *AccessProofStep {
        return self.proof_steps.items;
    }
};

pub const HashChainBlock = struct {
    index: u64,
    data: []const u8,
    data_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    previous_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    block_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    timestamp: i64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, index: u64, data: []const u8, previous_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8) !*Self {
        const block = try allocator.create(Self);
        var data_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(data, &data_hash, .{});

        var block_hasher = Sha256.init(.{});
        block_hasher.update(std.mem.asBytes(&index));
        block_hasher.update(data);
        block_hasher.update(&previous_hash);
        const timestamp = @as(i64, @intCast(std.time.nanoTimestamp()));
        block_hasher.update(std.mem.asBytes(&timestamp));
        var block_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        block_hasher.final(&block_hash);

        block.* = Self{
            .index = index,
            .data = try allocator.dupe(u8, data),
            .data_hash = data_hash,
            .previous_hash = previous_hash,
            .block_hash = block_hash,
            .timestamp = timestamp,
            .allocator = allocator,
        };
        return block;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.data);
    }

    pub fn verify(self: *const Self) bool {
        var computed_data_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(self.data, &computed_data_hash, .{});
        if (!timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, self.data_hash, computed_data_hash)) return false;

        var block_hasher = Sha256.init(.{});
        block_hasher.update(std.mem.asBytes(&self.index));
        block_hasher.update(self.data);
        block_hasher.update(&self.previous_hash);
        block_hasher.update(std.mem.asBytes(&self.timestamp));
        var computed_block_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        block_hasher.final(&computed_block_hash);

        return timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, self.block_hash, computed_block_hash);
    }
};

pub const HashChain = struct {
    blocks: ArrayList(*HashChainBlock),
    genesis_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) !Self {
        var genesis_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash("GENESIS_BLOCK", &genesis_hash, .{});

        var chain = Self{
            .blocks = ArrayList(*HashChainBlock).init(allocator),
            .genesis_hash = genesis_hash,
            .allocator = allocator,
        };

        const genesis_block = try HashChainBlock.init(allocator, 0, "GENESIS", std.mem.zeroes([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8));
        try chain.blocks.append(genesis_block);

        return chain;
    }

    pub fn deinit(self: *Self) void {
        for (self.blocks.items) |block| {
            block.deinit();
            self.allocator.destroy(block);
        }
        self.blocks.deinit();
    }

    pub fn append(self: *Self, data: []const u8) !void {
        const last_block = self.blocks.items[self.blocks.items.len - 1];
        const new_block = try HashChainBlock.init(
            self.allocator,
            last_block.index + 1,
            data,
            last_block.block_hash,
        );
        try self.blocks.append(new_block);
    }

    pub fn verify(self: *const Self) bool {
        if (self.blocks.items.len == 0) return false;

        var i: usize = 0;
        while (i < self.blocks.items.len) : (i += 1) {
            const block = self.blocks.items[i];
            if (!block.verify()) return false;

            if (i > 0) {
                const prev_block = self.blocks.items[i - 1];
                if (!timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, block.previous_hash, prev_block.block_hash)) {
                    return false;
                }
            }
        }

        return true;
    }

    pub fn getBlock(self: *const Self, index: u64) ?*HashChainBlock {
        for (self.blocks.items) |block| {
            if (block.index == index) return block;
        }
        return null;
    }

    pub fn length(self: *const Self) usize {
        return self.blocks.items.len;
    }

    pub fn latestHash(self: *const Self) [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 {
        if (self.blocks.items.len == 0) return self.genesis_hash;
        return self.blocks.items[self.blocks.items.len - 1].block_hash;
    }
};

pub const MerkleNode = struct {
    hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    left_idx: ?usize,
    right_idx: ?usize,
    is_leaf: bool,
    data_index: ?usize,

    const Self = @This();
};

pub const MerkleProof = struct {
    leaf_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    sibling_hashes: ArrayList([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8),
    directions: ArrayList(bool),
    root_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .leaf_hash = std.mem.zeroes([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8),
            .sibling_hashes = ArrayList([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8).init(allocator),
            .directions = ArrayList(bool).init(allocator),
            .root_hash = std.mem.zeroes([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.sibling_hashes.deinit();
        self.directions.deinit();
    }

    pub fn verify(self: *const Self, data: []const u8) bool {
        var current_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(data, &current_hash, .{});

        if (!timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, current_hash, self.leaf_hash)) return false;

        var i: usize = self.sibling_hashes.items.len;
        while (i > 0) {
            i -= 1;
            const sibling = self.sibling_hashes.items[i];
            var hasher = Sha256.init(.{});
            if (self.directions.items[i]) {
                hasher.update(&sibling);
                hasher.update(&current_hash);
            } else {
                hasher.update(&current_hash);
                hasher.update(&sibling);
            }
            hasher.final(&current_hash);
        }

        return timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, current_hash, self.root_hash);
    }
};

pub const MerkleTree = struct {
    nodes: ArrayList(MerkleNode),
    leaf_indices: ArrayList(usize),
    data_items: ArrayList([]const u8),
    root_idx: ?usize,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .nodes = ArrayList(MerkleNode).init(allocator),
            .leaf_indices = ArrayList(usize).init(allocator),
            .data_items = ArrayList([]const u8).init(allocator),
            .root_idx = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.nodes.deinit();
        self.leaf_indices.deinit();
        for (self.data_items.items) |item| {
            self.allocator.free(item);
        }
        self.data_items.deinit();
    }

    pub fn build(self: *Self, data_items: []const []const u8) !void {
        if (data_items.len == 0) return;

        var i: usize = 0;
        while (i < data_items.len) : (i += 1) {
            const item = data_items[i];
            const item_copy = try self.allocator.dupe(u8, item);
            try self.data_items.append(item_copy);

            var hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
            Sha256.hash(item, &hash, .{});

            const leaf = MerkleNode{
                .hash = hash,
                .left_idx = null,
                .right_idx = null,
                .is_leaf = true,
                .data_index = i,
            };
            try self.nodes.append(leaf);
            try self.leaf_indices.append(self.nodes.items.len - 1);
        }

        var current_level_start: usize = 0;
        var current_level_count: usize = self.nodes.items.len;

        while (current_level_count > 1) {
            const new_level_start = self.nodes.items.len;
            var level_i: usize = 0;
            while (level_i < current_level_count) : (level_i += 2) {
                const left_idx = current_level_start + level_i;
                if (level_i + 1 < current_level_count) {
                    const right_idx = current_level_start + level_i + 1;

                    var hasher = Sha256.init(.{});
                    hasher.update(&self.nodes.items[left_idx].hash);
                    hasher.update(&self.nodes.items[right_idx].hash);
                    var hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
                    hasher.final(&hash);

                    const internal = MerkleNode{
                        .hash = hash,
                        .left_idx = left_idx,
                        .right_idx = right_idx,
                        .is_leaf = false,
                        .data_index = null,
                    };
                    try self.nodes.append(internal);
                } else {
                    const internal = MerkleNode{
                        .hash = self.nodes.items[left_idx].hash,
                        .left_idx = left_idx,
                        .right_idx = null,
                        .is_leaf = false,
                        .data_index = null,
                    };
                    try self.nodes.append(internal);
                }
            }

            current_level_start = new_level_start;
            current_level_count = self.nodes.items.len - new_level_start;
        }

        if (self.nodes.items.len > 0) {
            self.root_idx = self.nodes.items.len - 1;
        }
    }

    pub fn getRootHash(self: *const Self) ?[SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 {
        if (self.root_idx) |idx| {
            return self.nodes.items[idx].hash;
        }
        return null;
    }

    pub fn generateProof(self: *const Self, data_index: usize) !MerkleProof {
        var proof = MerkleProof.init(self.allocator);

        if (data_index >= self.leaf_indices.items.len) return proof;

        const leaf_idx = self.leaf_indices.items[data_index];
        proof.leaf_hash = self.nodes.items[leaf_idx].hash;

        if (self.root_idx) |root_idx| {
            proof.root_hash = self.nodes.items[root_idx].hash;
            try self.collectProofPath(&proof, root_idx, leaf_idx);
        }

        return proof;
    }

    fn collectProofPath(self: *const Self, proof: *MerkleProof, node_idx: usize, target_leaf_idx: usize) !void {
        const node = self.nodes.items[node_idx];
        if (node.is_leaf) return;

        const left_idx = node.left_idx orelse return;

        if (node.right_idx) |right_idx| {
            if (self.isDescendant(left_idx, target_leaf_idx)) {
                try proof.sibling_hashes.append(self.nodes.items[right_idx].hash);
                try proof.directions.append(false);
                try self.collectProofPath(proof, left_idx, target_leaf_idx);
            } else {
                try proof.sibling_hashes.append(self.nodes.items[left_idx].hash);
                try proof.directions.append(true);
                try self.collectProofPath(proof, right_idx, target_leaf_idx);
            }
        } else {
            try self.collectProofPath(proof, left_idx, target_leaf_idx);
        }
    }

    fn isDescendant(self: *const Self, ancestor_idx: usize, target_idx: usize) bool {
        if (ancestor_idx == target_idx) return true;
        const node = self.nodes.items[ancestor_idx];
        if (node.is_leaf) return false;
        if (node.left_idx) |left| {
            if (self.isDescendant(left, target_idx)) return true;
        }
        if (node.right_idx) |right| {
            if (self.isDescendant(right, target_idx)) return true;
        }
        return false;
    }

    pub fn verify(self: *const Self) bool {
        if (self.root_idx == null) return true;
        return self.verifyNode(self.root_idx.?);
    }

    fn verifyNode(self: *const Self, node_idx: usize) bool {
        const node = self.nodes.items[node_idx];
        if (node.is_leaf) return true;

        const left_idx = node.left_idx orelse return false;
        const left = self.nodes.items[left_idx];

        if (node.right_idx) |right_idx| {
            const right = self.nodes.items[right_idx];

            var hasher = Sha256.init(.{});
            hasher.update(&left.hash);
            hasher.update(&right.hash);
            var computed_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
            hasher.final(&computed_hash);

            if (!timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, node.hash, computed_hash)) return false;

            return self.verifyNode(left_idx) and self.verifyNode(right_idx);
        } else {
            if (!timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, node.hash, left.hash)) return false;
            return self.verifyNode(left_idx);
        }
    }
};

pub const Commitment = struct {
    commitment_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    salt: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    is_revealed: bool,
    revealed_value: ?[]const u8,
    allocator: Allocator,

    const Self = @This();

    pub fn commit(allocator: Allocator, value: []const u8) !*Self {
        const commitment = try allocator.create(Self);

        var salt: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        std.crypto.random.bytes(&salt);

        var hasher = Sha256.init(.{});
        hasher.update(value);
        hasher.update(&salt);
        var commitment_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        hasher.final(&commitment_hash);

        commitment.* = Self{
            .commitment_hash = commitment_hash,
            .salt = salt,
            .is_revealed = false,
            .revealed_value = null,
            .allocator = allocator,
        };
        return commitment;
    }

    pub fn deinit(self: *Self) void {
        if (self.revealed_value) |val| {
            self.allocator.free(val);
        }
    }

    pub fn reveal(self: *Self, value: []const u8) !bool {
        var hasher = Sha256.init(.{});
        hasher.update(value);
        hasher.update(&self.salt);
        var computed_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        hasher.final(&computed_hash);

        if (timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, self.commitment_hash, computed_hash)) {
            self.is_revealed = true;
            self.revealed_value = try self.allocator.dupe(u8, value);
            return true;
        }
        return false;
    }

    pub fn verifyBinding(self: *const Self, claimed_value: []const u8) bool {
        var hasher = Sha256.init(.{});
        hasher.update(claimed_value);
        hasher.update(&self.salt);
        var computed_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        hasher.final(&computed_hash);
        return timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, self.commitment_hash, computed_hash);
    }
};

pub const CommitmentScheme = struct {
    commitments: AutoHashMap(u64, *Commitment),
    next_id: u64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .commitments = AutoHashMap(u64, *Commitment).init(allocator),
            .next_id = 1,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.commitments.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.commitments.deinit();
    }

    pub fn createCommitment(self: *Self, value: []const u8) !u64 {
        const id = self.next_id;
        self.next_id += 1;
        const commitment = try Commitment.commit(self.allocator, value);
        try self.commitments.put(id, commitment);
        return id;
    }

    pub fn revealCommitment(self: *Self, id: u64, value: []const u8) !bool {
        const commitment = self.commitments.get(id) orelse return SecurityError.InvalidCommitment;
        return commitment.reveal(value);
    }

    pub fn verifyCommitment(self: *const Self, id: u64, claimed_value: []const u8) bool {
        const commitment = self.commitments.get(id) orelse return false;
        return commitment.verifyBinding(claimed_value);
    }

    pub fn getCommitmentHash(self: *const Self, id: u64) ?[SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 {
        const commitment = self.commitments.get(id) orelse return null;
        return commitment.commitment_hash;
    }
};

pub const CryptographicProof = struct {
    proof_id: u64,
    proof_type: ProofType,
    statement: []const u8,
    witness_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    proof_data: [64]u8,
    timestamp: i64,
    is_valid: bool,
    allocator: Allocator,

    pub const ProofType = enum(u8) {
        KNOWLEDGE = 0,
        MEMBERSHIP = 1,
        RANGE = 2,
        EQUALITY = 3,
        INTEGRITY = 4,

        pub fn toString(self: ProofType) []const u8 {
            return switch (self) {
                .KNOWLEDGE => "KNOWLEDGE",
                .MEMBERSHIP => "MEMBERSHIP",
                .RANGE => "RANGE",
                .EQUALITY => "EQUALITY",
                .INTEGRITY => "INTEGRITY",
            };
        }
    };

    const Self = @This();

    pub fn init(allocator: Allocator, proof_id: u64, proof_type: ProofType, statement: []const u8, witness: []const u8) !*Self {
        const proof = try allocator.create(Self);

        var witness_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(witness, &witness_hash, .{});

        var proof_data: [64]u8 = undefined;
        var hasher = Sha512.init(.{});
        hasher.update(statement);
        hasher.update(&witness_hash);
        hasher.update(std.mem.asBytes(&proof_id));
        hasher.final(&proof_data);

        proof.* = Self{
            .proof_id = proof_id,
            .proof_type = proof_type,
            .statement = try allocator.dupe(u8, statement),
            .witness_hash = witness_hash,
            .proof_data = proof_data,
            .timestamp = @as(i64, @intCast(std.time.nanoTimestamp())),
            .is_valid = true,
            .allocator = allocator,
        };
        return proof;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.statement);
    }

    pub fn verify(self: *const Self, witness: []const u8) bool {
        var computed_witness_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(witness, &computed_witness_hash, .{});

        if (!timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, self.witness_hash, computed_witness_hash)) return false;

        var expected_proof_data: [64]u8 = undefined;
        var hasher = Sha512.init(.{});
        hasher.update(self.statement);
        hasher.update(&computed_witness_hash);
        hasher.update(std.mem.asBytes(&self.proof_id));
        hasher.final(&expected_proof_data);

        return timingSafeEql([64]u8, self.proof_data, expected_proof_data);
    }

    pub fn getProofHash(self: *const Self) [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 {
        var hasher = Sha256.init(.{});
        hasher.update(&self.proof_data);
        hasher.update(self.statement);
        var result: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        hasher.final(&result);
        return result;
    }
};

pub const BellLaPadula = struct {
    subjects: AutoHashMap(u64, SecurityLevel),
    objects: AutoHashMap(u64, SecurityLevel),
    current_levels: AutoHashMap(u64, SecurityLevel),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .subjects = AutoHashMap(u64, SecurityLevel).init(allocator),
            .objects = AutoHashMap(u64, SecurityLevel).init(allocator),
            .current_levels = AutoHashMap(u64, SecurityLevel).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.subjects.deinit();
        self.objects.deinit();
        self.current_levels.deinit();
    }

    pub fn addSubject(self: *Self, subject_id: u64, clearance: SecurityLevel) !void {
        try self.subjects.put(subject_id, clearance);
        try self.current_levels.put(subject_id, clearance);
    }

    pub fn addObject(self: *Self, object_id: u64, classification: SecurityLevel) !void {
        try self.objects.put(object_id, classification);
    }

    pub fn setCurrentLevel(self: *Self, subject_id: u64, level: SecurityLevel) !void {
        const clearance = self.subjects.get(subject_id) orelse return SecurityError.InvalidPrincipal;
        if (level.greaterThan(clearance)) return SecurityError.BellLaPadulaViolation;
        try self.current_levels.put(subject_id, level);
    }

    pub fn canRead(self: *const Self, subject_id: u64, object_id: u64) bool {
        const current = self.current_levels.get(subject_id) orelse return false;
        const classification = self.objects.get(object_id) orelse return false;
        return current.greaterThanOrEqual(classification);
    }

    pub fn canWrite(self: *const Self, subject_id: u64, object_id: u64) bool {
        const current = self.current_levels.get(subject_id) orelse return false;
        const classification = self.objects.get(object_id) orelse return false;
        return current.lessThanOrEqual(classification);
    }

    pub fn verifySimpleSecurityProperty(self: *const Self, subject_id: u64, object_id: u64) bool {
        return self.canRead(subject_id, object_id);
    }

    pub fn verifyStarProperty(self: *const Self, subject_id: u64, object_id: u64) bool {
        return self.canWrite(subject_id, object_id);
    }

    pub fn verifyAccess(self: *const Self, subject_id: u64, object_id: u64, is_read: bool, is_write: bool) bool {
        if (is_read and !self.verifySimpleSecurityProperty(subject_id, object_id)) return false;
        if (is_write and !self.verifyStarProperty(subject_id, object_id)) return false;
        return true;
    }
};

pub const Biba = struct {
    subjects: AutoHashMap(u64, IntegrityLevel),
    objects: AutoHashMap(u64, IntegrityLevel),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .subjects = AutoHashMap(u64, IntegrityLevel).init(allocator),
            .objects = AutoHashMap(u64, IntegrityLevel).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.subjects.deinit();
        self.objects.deinit();
    }

    pub fn addSubject(self: *Self, subject_id: u64, integrity: IntegrityLevel) !void {
        try self.subjects.put(subject_id, integrity);
    }

    pub fn addObject(self: *Self, object_id: u64, integrity: IntegrityLevel) !void {
        try self.objects.put(object_id, integrity);
    }

    pub fn canRead(self: *const Self, subject_id: u64, object_id: u64) bool {
        const subject_integrity = self.subjects.get(subject_id) orelse return false;
        const object_integrity = self.objects.get(object_id) orelse return false;
        return subject_integrity.lessThanOrEqual(object_integrity);
    }

    pub fn canWrite(self: *const Self, subject_id: u64, object_id: u64) bool {
        const subject_integrity = self.subjects.get(subject_id) orelse return false;
        const object_integrity = self.objects.get(object_id) orelse return false;
        return subject_integrity.greaterThanOrEqual(object_integrity);
    }

    pub fn canInvoke(self: *const Self, subject_id: u64, target_subject_id: u64) bool {
        const subject_integrity = self.subjects.get(subject_id) orelse return false;
        const target_integrity = self.subjects.get(target_subject_id) orelse return false;
        return subject_integrity.lessThanOrEqual(target_integrity);
    }

    pub fn verifySimpleIntegrityProperty(self: *const Self, subject_id: u64, object_id: u64) bool {
        return self.canRead(subject_id, object_id);
    }

    pub fn verifyIntegrityStarProperty(self: *const Self, subject_id: u64, object_id: u64) bool {
        return self.canWrite(subject_id, object_id);
    }

    pub fn verifyInvocationProperty(self: *const Self, subject_id: u64, target_subject_id: u64) bool {
        return self.canInvoke(subject_id, target_subject_id);
    }
};

pub const IntegrityVerifier = struct {
    hash_chain: HashChain,
    merkle_trees: AutoHashMap(u64, *MerkleTree),
    data_hashes: AutoHashMap(u64, [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8),
    next_id: u64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) !Self {
        return Self{
            .hash_chain = try HashChain.init(allocator),
            .merkle_trees = AutoHashMap(u64, *MerkleTree).init(allocator),
            .data_hashes = AutoHashMap(u64, [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8).init(allocator),
            .next_id = 1,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.hash_chain.deinit();
        var iter = self.merkle_trees.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.merkle_trees.deinit();
        self.data_hashes.deinit();
    }

    pub fn registerData(self: *Self, data: []const u8) !u64 {
        const id = self.next_id;
        self.next_id += 1;

        var hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(data, &hash, .{});
        try self.data_hashes.put(id, hash);

        try self.hash_chain.append(data);

        return id;
    }

    pub fn verifyIntegrity(self: *const Self, data_id: u64, data: []const u8) bool {
        const expected_hash = self.data_hashes.get(data_id) orelse return false;

        var computed_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(data, &computed_hash, .{});

        return timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, expected_hash, computed_hash);
    }

    pub fn verifyChainIntegrity(self: *const Self) bool {
        return self.hash_chain.verify();
    }

    pub fn createMerkleTree(self: *Self, data_items: []const []const u8) !u64 {
        const id = self.next_id;
        self.next_id += 1;

        const tree = try self.allocator.create(MerkleTree);
        tree.* = MerkleTree.init(self.allocator);
        try tree.build(data_items);
        try self.merkle_trees.put(id, tree);

        return id;
    }

    pub fn getMerkleRoot(self: *const Self, tree_id: u64) ?[SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 {
        const tree = self.merkle_trees.get(tree_id) orelse return null;
        return tree.getRootHash();
    }

    pub fn generateMerkleProof(self: *const Self, tree_id: u64, data_index: usize) !MerkleProof {
        const tree = self.merkle_trees.get(tree_id) orelse return SecurityError.InvalidHash;
        return tree.generateProof(data_index);
    }

    pub fn proveIntegrity(self: *const Self, data_id: u64, data: []const u8, allocator: Allocator) !*CryptographicProof {
        const expected_hash = self.data_hashes.get(data_id) orelse return SecurityError.InvalidHash;

        var computed_hash: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(data, &computed_hash, .{});

        if (!timingSafeEql([SecurityProofsConfig.SHA256_DIGEST_SIZE]u8, expected_hash, computed_hash)) {
            return SecurityError.IntegrityViolation;
        }

        return CryptographicProof.init(
            allocator,
            data_id,
            .INTEGRITY,
            "Data integrity verified against stored hash",
            data,
        );
    }
};

pub const SecurityProofType = enum(u8) {
    INFORMATION_FLOW = 0,
    NON_INTERFERENCE = 1,
    ACCESS_CONTROL = 2,
    INTEGRITY = 3,
    CONFIDENTIALITY = 4,
    BELL_LAPADULA = 5,
    BIBA = 6,
    CRYPTOGRAPHIC = 7,

    pub fn toString(self: SecurityProofType) []const u8 {
        return switch (self) {
            .INFORMATION_FLOW => "INFORMATION_FLOW",
            .NON_INTERFERENCE => "NON_INTERFERENCE",
            .ACCESS_CONTROL => "ACCESS_CONTROL",
            .INTEGRITY => "INTEGRITY",
            .CONFIDENTIALITY => "CONFIDENTIALITY",
            .BELL_LAPADULA => "BELL_LAPADULA",
            .BIBA => "BIBA",
            .CRYPTOGRAPHIC => "CRYPTOGRAPHIC",
        };
    }
};

pub const SecurityProofStep = struct {
    step_number: u64,
    description: []const u8,
    rule_applied: []const u8,
    is_valid: bool,
    sub_proof_hash: ?[SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    allocator: Allocator,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        step_number: u64,
        description: []const u8,
        rule_applied: []const u8,
        is_valid: bool,
    ) !*Self {
        const step = try allocator.create(Self);
        step.* = Self{
            .step_number = step_number,
            .description = try allocator.dupe(u8, description),
            .rule_applied = try allocator.dupe(u8, rule_applied),
            .is_valid = is_valid,
            .sub_proof_hash = null,
            .allocator = allocator,
        };
        return step;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.description);
        self.allocator.free(self.rule_applied);
    }
};

pub const SecurityProof = struct {
    proof_id: u64,
    proof_type: SecurityProofType,
    property_proven: []const u8,
    proof_steps: ArrayList(*SecurityProofStep),
    is_valid: bool,
    cryptographic_binding: ?[SecurityProofsConfig.SHA256_DIGEST_SIZE]u8,
    timestamp: i64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, proof_id: u64, proof_type: SecurityProofType, property_proven: []const u8) !*Self {
        const proof = try allocator.create(Self);
        proof.* = Self{
            .proof_id = proof_id,
            .proof_type = proof_type,
            .property_proven = try allocator.dupe(u8, property_proven),
            .proof_steps = ArrayList(*SecurityProofStep).init(allocator),
            .is_valid = false,
            .cryptographic_binding = null,
            .timestamp = @as(i64, @intCast(std.time.nanoTimestamp())),
            .allocator = allocator,
        };
        return proof;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.property_proven);
        for (self.proof_steps.items) |step| {
            step.deinit();
            self.allocator.destroy(step);
        }
        self.proof_steps.deinit();
    }

    pub fn addStep(self: *Self, description: []const u8, rule_applied: []const u8, is_valid: bool) !void {
        const step_num = self.proof_steps.items.len + 1;
        const step = try SecurityProofStep.init(self.allocator, step_num, description, rule_applied, is_valid);
        try self.proof_steps.append(step);
    }

    pub fn finalize(self: *Self) void {
        self.is_valid = true;
        for (self.proof_steps.items) |step| {
            if (!step.is_valid) {
                self.is_valid = false;
                break;
            }
        }

        var hasher = Sha256.init(.{});
        hasher.update(std.mem.asBytes(&self.proof_id));
        hasher.update(&[_]u8{@intFromEnum(self.proof_type)});
        hasher.update(self.property_proven);
        for (self.proof_steps.items) |step| {
            hasher.update(step.description);
        }
        var binding: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        hasher.final(&binding);
        self.cryptographic_binding = binding;
    }

    pub fn verify(self: *const Self) bool {
        if (self.cryptographic_binding == null) return false;

        var hasher = Sha256.init(.{});
        hasher.update(std.mem.asBytes(&self.proof_id));
        hasher.update(&[_]u8{@intFromEnum(self.proof_type)});
        hasher.update(self.property_proven);
        for (self.proof_steps.items) |step| {
            hasher.update(step.description);
        }
        var computed_binding: [SecurityProofsConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        hasher.final(&computed_binding);

        return std.mem.eql(u8, &self.cryptographic_binding.?, &computed_binding);
    }

    pub fn stepCount(self: *const Self) usize {
        return self.proof_steps.items.len;
    }
};

pub const SecurityProofEngine = struct {
    flow_analysis: InformationFlowAnalysis,
    ni_prover: NonInterferenceProver,
    ac_verifier: AccessControlVerifier,
    integrity_verifier: IntegrityVerifier,
    blp_model: BellLaPadula,
    biba_model: Biba,
    commitment_scheme: CommitmentScheme,
    proofs: ArrayList(*SecurityProof),
    next_proof_id: u64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, observer_level: SecurityLevel) !Self {
        return Self{
            .flow_analysis = InformationFlowAnalysis.init(allocator),
            .ni_prover = NonInterferenceProver.init(allocator, observer_level),
            .ac_verifier = AccessControlVerifier.init(allocator),
            .integrity_verifier = try IntegrityVerifier.init(allocator),
            .blp_model = BellLaPadula.init(allocator),
            .biba_model = Biba.init(allocator),
            .commitment_scheme = CommitmentScheme.init(allocator),
            .proofs = ArrayList(*SecurityProof).init(allocator),
            .next_proof_id = 1,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.flow_analysis.deinit();
        self.ni_prover.deinit();
        self.ac_verifier.deinit();
        self.integrity_verifier.deinit();
        self.blp_model.deinit();
        self.biba_model.deinit();
        self.commitment_scheme.deinit();
        for (self.proofs.items) |proof| {
            proof.deinit();
            self.allocator.destroy(proof);
        }
        self.proofs.deinit();
    }

    pub fn proveInformationFlowSecurity(self: *Self, graph: *const SelfSimilarRelationalGraph) !*SecurityProof {
        const proof_id = self.next_proof_id;
        self.next_proof_id += 1;

        const proof = try SecurityProof.init(
            self.allocator,
            proof_id,
            .INFORMATION_FLOW,
            "Information flow satisfies security lattice constraints",
        );

        try proof.addStep(
            "Initialize flow graph from relational graph",
            "GRAPH_INIT",
            true,
        );

        var node_iter = graph.nodes.iterator();
        var node_count: u64 = 0;
        while (node_iter.next()) |entry| {
            const node_hash = hashNodeId(entry.key_ptr.*);
            const level = determineSecurityLevel(entry.value_ptr);
            try self.flow_analysis.addVariable(node_hash, level);
            node_count += 1;
        }

        var desc_buf: [128]u8 = undefined;
        const desc = std.fmt.bufPrint(&desc_buf, "Classified {} nodes by security level", .{node_count}) catch "Classified nodes";
        try proof.addStep(desc, "NODE_CLASSIFICATION", true);

        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            const source_hash = hashNodeId(entry.key_ptr.source);
            const target_hash = hashNodeId(entry.key_ptr.target);
            try self.flow_analysis.addFlow(source_hash, target_hash, .EXPLICIT, true);
        }

        try proof.addStep("Added all edges as explicit flows", "FLOW_EDGES", true);

        try self.flow_analysis.computeSecurityClosure();
        try proof.addStep("Computed transitive closure of information flows", "CLOSURE", true);

        const illegal_flows = try self.flow_analysis.detectIllegalFlows();
        const is_secure = illegal_flows.len == 0;

        if (is_secure) {
            try proof.addStep(
                "No illegal information flows detected (high → low leaks)",
                "FLOW_CHECK",
                true,
            );
        } else {
            var violation_buf: [128]u8 = undefined;
            const violation_desc = std.fmt.bufPrint(&violation_buf, "Detected {} illegal information flows", .{illegal_flows.len}) catch "Detected illegal flows";
            try proof.addStep(violation_desc, "FLOW_VIOLATION", false);
        }

        try proof.addStep(
            if (is_secure) "PROOF COMPLETE: Information flow security verified" else "PROOF FAILED: Information flow violations exist",
            "CONCLUSION",
            is_secure,
        );

        proof.finalize();
        try self.proofs.append(proof);
        return proof;
    }

    pub fn proveNonInterference(self: *Self, initial_state: *const SystemState, alternate_state: *const SystemState, observer_level: SecurityLevel) !*SecurityProof {
        const proof_id = self.next_proof_id;
        self.next_proof_id += 1;

        const proof = try SecurityProof.init(
            self.allocator,
            proof_id,
            .NON_INTERFERENCE,
            "High security inputs do not affect low security observations",
        );

        try proof.addStep(
            "Initialize non-interference proof with observer level",
            "NI_INIT",
            true,
        );

        var ni_prover = NonInterferenceProver.init(self.allocator, observer_level);
        defer ni_prover.deinit();

        var var_iter = initial_state.variable_levels.iterator();
        while (var_iter.next()) |entry| {
            try ni_prover.addVariable(entry.key_ptr.*, entry.value_ptr.*);
        }

        try proof.addStep(
            "Partitioned variables into high and low sets",
            "PARTITION",
            true,
        );

        const ni_holds = try ni_prover.proveNonInterference(initial_state, alternate_state);

        for (ni_prover.getProofSteps()) |ni_step| {
            try proof.addStep(ni_step.description, ni_step.lemma_name, ni_step.is_valid);
        }

        try proof.addStep(
            if (ni_holds) "PROOF COMPLETE: Non-interference property holds" else "PROOF FAILED: Non-interference violated",
            "NI_CONCLUSION",
            ni_holds,
        );

        proof.finalize();
        try self.proofs.append(proof);
        return proof;
    }

    pub fn proveAccessControl(self: *Self, subject: *const Principal, object: *const SecureObject, right: AccessRight) !*SecurityProof {
        const proof_id = self.next_proof_id;
        self.next_proof_id += 1;

        const proof = try SecurityProof.init(
            self.allocator,
            proof_id,
            .ACCESS_CONTROL,
            "Access request satisfies all security policies",
        );

        try proof.addStep(
            "Initialize access control verification",
            "AC_INIT",
            true,
        );

        const blp_read_ok = self.blp_model.canRead(subject.id, object.id);
        const blp_write_ok = self.blp_model.canWrite(subject.id, object.id);
        const blp_ok = (right != .WRITE or blp_write_ok) and (right != .READ or blp_read_ok);

        try proof.addStep(
            if (blp_ok) "Bell-LaPadula model constraints satisfied" else "Bell-LaPadula violation detected",
            "BLP_CHECK",
            blp_ok,
        );

        const biba_read_ok = self.biba_model.canRead(subject.id, object.id);
        const biba_write_ok = self.biba_model.canWrite(subject.id, object.id);
        const biba_ok = (right != .WRITE or biba_write_ok) and (right != .READ or biba_read_ok);

        try proof.addStep(
            if (biba_ok) "Biba integrity model constraints satisfied" else "Biba integrity violation detected",
            "BIBA_CHECK",
            biba_ok,
        );

        const policy_ok = try self.ac_verifier.verifyPolicy(subject, object, right);
        try proof.addStep(
            if (policy_ok) "Policy evaluation permits access" else "Policy denies access",
            "POLICY_CHECK",
            policy_ok,
        );

        const sod_ok = try self.ac_verifier.checkSeparationOfDuties(subject);
        try proof.addStep(
            if (sod_ok) "Separation of duties constraints satisfied" else "Separation of duties violation",
            "SOD_CHECK",
            sod_ok,
        );

        const final_result = blp_ok and biba_ok and policy_ok and sod_ok;
        try proof.addStep(
            if (final_result) "PROOF COMPLETE: Access is authorized" else "PROOF FAILED: Access denied",
            "AC_CONCLUSION",
            final_result,
        );

        proof.finalize();
        try self.proofs.append(proof);
        return proof;
    }

    pub fn proveIntegrity(self: *Self, data_id: u64, data: []const u8) !*SecurityProof {
        const proof_id = self.next_proof_id;
        self.next_proof_id += 1;

        const proof = try SecurityProof.init(
            self.allocator,
            proof_id,
            .INTEGRITY,
            "Data integrity verified against stored cryptographic hash",
        );

        try proof.addStep(
            "Initialize integrity verification",
            "INT_INIT",
            true,
        );

        const hash_valid = self.integrity_verifier.verifyIntegrity(data_id, data);
        try proof.addStep(
            if (hash_valid) "Data hash matches stored reference" else "Data hash mismatch detected",
            "HASH_CHECK",
            hash_valid,
        );

        const chain_valid = self.integrity_verifier.verifyChainIntegrity();
        try proof.addStep(
            if (chain_valid) "Hash chain integrity verified" else "Hash chain corruption detected",
            "CHAIN_CHECK",
            chain_valid,
        );

        const final_result = hash_valid and chain_valid;
        try proof.addStep(
            if (final_result) "PROOF COMPLETE: Data integrity verified" else "PROOF FAILED: Integrity violation",
            "INT_CONCLUSION",
            final_result,
        );

        proof.finalize();
        try self.proofs.append(proof);
        return proof;
    }

    pub fn registerDataForIntegrity(self: *Self, data: []const u8) !u64 {
        return self.integrity_verifier.registerData(data);
    }

    pub fn addSubjectToModels(self: *Self, subject_id: u64, security_level: SecurityLevel, integrity_level: IntegrityLevel) !void {
        try self.blp_model.addSubject(subject_id, security_level);
        try self.biba_model.addSubject(subject_id, integrity_level);
    }

    pub fn addObjectToModels(self: *Self, object_id: u64, security_level: SecurityLevel, integrity_level: IntegrityLevel) !void {
        try self.blp_model.addObject(object_id, security_level);
        try self.biba_model.addObject(object_id, integrity_level);
    }

    pub fn getProof(self: *const Self, proof_id: u64) ?*SecurityProof {
        for (self.proofs.items) |proof| {
            if (proof.proof_id == proof_id) return proof;
        }
        return null;
    }

    pub fn proofCount(self: *const Self) usize {
        return self.proofs.items.len;
    }

    fn hashNodeId(node_id: []const u8) u64 {
        var hasher = std.hash.Wyhash.init(0x517cc1b727220a95);
        hasher.update(node_id);
        return hasher.final();
    }

    fn determineSecurityLevel(node: *const Node) SecurityLevel {
        const prob = std.math.sqrt(node.qubit.a.re * node.qubit.a.re + node.qubit.a.im * node.qubit.a.im);
        if (prob < 0.2) return .PUBLIC;
        if (prob < 0.4) return .INTERNAL;
        if (prob < 0.6) return .CONFIDENTIAL;
        if (prob < 0.8) return .SECRET;
        return .TOP_SECRET;
    }
};

test "SecurityLevel lattice operations" {
    try std.testing.expect(SecurityLevel.PUBLIC.lessThanOrEqual(.INTERNAL));
    try std.testing.expect(SecurityLevel.INTERNAL.lessThanOrEqual(.CONFIDENTIAL));
    try std.testing.expect(SecurityLevel.CONFIDENTIAL.lessThanOrEqual(.SECRET));
    try std.testing.expect(SecurityLevel.SECRET.lessThanOrEqual(.TOP_SECRET));
    try std.testing.expect(!SecurityLevel.TOP_SECRET.lessThanOrEqual(.PUBLIC));

    try std.testing.expect(SecurityLevel.PUBLIC.join(.SECRET) == .SECRET);
    try std.testing.expect(SecurityLevel.SECRET.meet(.INTERNAL) == .INTERNAL);

    try std.testing.expect(SecurityLevel.TOP_SECRET.dominates(.PUBLIC));
    try std.testing.expect(!SecurityLevel.PUBLIC.dominates(.SECRET));
}

test "IntegrityLevel lattice operations" {
    try std.testing.expect(IntegrityLevel.KERNEL.greaterThanOrEqual(.USER));
    try std.testing.expect(IntegrityLevel.USER.greaterThanOrEqual(.UNTRUSTED));

    try std.testing.expect(IntegrityLevel.KERNEL.join(.USER) == .USER);
    try std.testing.expect(IntegrityLevel.USER.meet(.KERNEL) == .KERNEL);
}

test "AccessRightSet bitmask operations" {
    var rights = AccessRightSet.init();
    try std.testing.expect(rights.isEmpty());

    rights.add(.READ);
    rights.add(.WRITE);
    try std.testing.expect(rights.has(.READ));
    try std.testing.expect(rights.has(.WRITE));
    try std.testing.expect(!rights.has(.DELETE));

    rights.remove(.WRITE);
    try std.testing.expect(!rights.has(.WRITE));

    const other = AccessRightSet.initWithRights(&[_]AccessRight{ .EXECUTE, .DELETE });
    const union_rights = rights.unionWith(&other);
    try std.testing.expect(union_rights.has(.READ));
    try std.testing.expect(union_rights.has(.EXECUTE));
    try std.testing.expect(union_rights.has(.DELETE));
}

test "SecurityLabel operations" {
    const allocator = std.testing.allocator;

    var label1 = SecurityLabel.init(allocator, .SECRET, .SYSTEM);
    defer label1.deinit();
    try label1.addCategory(1);
    try label1.addCategory(2);

    var label2 = SecurityLabel.init(allocator, .CONFIDENTIAL, .USER);
    defer label2.deinit();
    try label2.addCategory(1);

    try std.testing.expect(label1.dominates(&label2));
    try std.testing.expect(!label2.dominates(&label1));

    var joined = try label1.join(&label2, allocator);
    defer joined.deinit();
    try std.testing.expect(joined.confidentiality_level == .SECRET);
}

test "Principal creation and access control" {
    const allocator = std.testing.allocator;

    var principal = try Principal.init(allocator, 1, "test_user", .CONFIDENTIAL);
    defer {
        principal.deinit();
        allocator.destroy(principal);
    }

    principal.access_rights.add(.READ);
    principal.access_rights.add(.WRITE);

    try std.testing.expect(principal.access_rights.has(.READ));
    try std.testing.expect(std.mem.eql(u8, principal.name, "test_user"));
}

test "FlowGraph and edge detection" {
    const allocator = std.testing.allocator;

    var flow_graph = FlowGraph.init(allocator);
    defer flow_graph.deinit();

    try flow_graph.addNode(1, .SECRET);
    try flow_graph.addNode(2, .PUBLIC);
    try flow_graph.addEdge(1, 2, .EXPLICIT, true);

    try std.testing.expect(flow_graph.nodeCount() == 2);
    try std.testing.expect(flow_graph.edgeCount() == 1);

    const edge = flow_graph.edges.items[0];
    try std.testing.expect(edge.isIllegal());
}

test "InformationFlowAnalysis detect illegal flows" {
    const allocator = std.testing.allocator;

    var analysis = InformationFlowAnalysis.init(allocator);
    defer analysis.deinit();

    try analysis.addVariable(1, .TOP_SECRET);
    try analysis.addVariable(2, .PUBLIC);
    try analysis.addFlow(1, 2, .EXPLICIT, true);

    const illegal = try analysis.detectIllegalFlows();
    try std.testing.expect(illegal.len > 0);
    try std.testing.expect(illegal[0].source_level == .TOP_SECRET);
    try std.testing.expect(illegal[0].target_level == .PUBLIC);
}

test "InformationFlowAnalysis secure flow" {
    const allocator = std.testing.allocator;

    var analysis = InformationFlowAnalysis.init(allocator);
    defer analysis.deinit();

    try analysis.addVariable(1, .PUBLIC);
    try analysis.addVariable(2, .SECRET);
    try analysis.addFlow(1, 2, .EXPLICIT, true);

    const is_secure = try analysis.isSecure();
    try std.testing.expect(is_secure);
}

test "InformationFlowLattice operations" {
    const lattice = InformationFlowLattice.init();

    try std.testing.expect(lattice.isBottom(.PUBLIC));
    try std.testing.expect(lattice.isTop(.TOP_SECRET));
    try std.testing.expect(lattice.covers(.INTERNAL, .PUBLIC));
    try std.testing.expect(lattice.distance(.PUBLIC, .SECRET) == 3);
}

test "SystemState low projection" {
    const allocator = std.testing.allocator;

    var state = SystemState.init(allocator);
    defer state.deinit();

    try state.setVariable(1, "public_data", .PUBLIC);
    try state.setVariable(2, "secret_data", .SECRET);

    var projection = try state.lowProjection(.INTERNAL, allocator);
    defer projection.deinit();

    try std.testing.expect(projection.getVariable(1) != null);
    try std.testing.expect(projection.getVariable(2) == null);
}

test "NonInterferenceProperty classification" {
    const allocator = std.testing.allocator;

    var property = NonInterferenceProperty.init(allocator, .CONFIDENTIAL);
    defer property.deinit();

    try property.classifyVariable(1, .PUBLIC);
    try property.classifyVariable(2, .SECRET);

    try std.testing.expect(property.isLowVariable(1));
    try std.testing.expect(property.isHighVariable(2));
}

test "BisimulationRelation check" {
    const allocator = std.testing.allocator;

    var state1 = SystemState.init(allocator);
    defer state1.deinit();
    try state1.setVariable(1, "hello", .PUBLIC);
    try state1.setVariable(2, "secret1", .SECRET);

    var state2 = SystemState.init(allocator);
    defer state2.deinit();
    try state2.setVariable(1, "hello", .PUBLIC);
    try state2.setVariable(2, "secret2", .SECRET);

    var bisim = BisimulationRelation.init(allocator);
    defer bisim.deinit();

    const are_bisimilar = try bisim.areBisimilar(&state1, &state2, .INTERNAL, allocator);
    try std.testing.expect(are_bisimilar);
}

test "NonInterferenceProver full proof" {
    const allocator = std.testing.allocator;

    var state1 = SystemState.init(allocator);
    defer state1.deinit();
    try state1.setVariable(1, "public", .PUBLIC);
    try state1.setVariable(2, "secret_a", .SECRET);

    var state2 = SystemState.init(allocator);
    defer state2.deinit();
    try state2.setVariable(1, "public", .PUBLIC);
    try state2.setVariable(2, "secret_b", .SECRET);

    var prover = NonInterferenceProver.init(allocator, .INTERNAL);
    defer prover.deinit();

    try prover.addVariable(1, .PUBLIC);
    try prover.addVariable(2, .SECRET);

    const proved = try prover.proveNonInterference(&state1, &state2);
    try std.testing.expect(proved);
    try std.testing.expect(prover.getProofSteps().len > 0);
}

test "AccessControlMatrix operations" {
    const allocator = std.testing.allocator;

    var matrix = AccessControlMatrix.init(allocator);
    defer matrix.deinit();

    try matrix.addSubject(1);
    try matrix.addObject(100);

    try matrix.grantRight(1, 100, .READ);
    try matrix.grantRight(1, 100, .WRITE);

    try std.testing.expect(matrix.hasRight(1, 100, .READ));
    try std.testing.expect(matrix.hasRight(1, 100, .WRITE));
    try std.testing.expect(!matrix.hasRight(1, 100, .DELETE));

    try matrix.revokeRight(1, 100, .WRITE);
    try std.testing.expect(!matrix.hasRight(1, 100, .WRITE));
}

test "SeparationOfDutiesConstraint" {
    const allocator = std.testing.allocator;

    var constraint = try SeparationOfDutiesConstraint.init(allocator, 1);
    defer {
        constraint.deinit();
        allocator.destroy(constraint);
    }

    try constraint.addConflictingRole(10);
    try constraint.addConflictingRole(20);

    var principal = try Principal.init(allocator, 1, "test", .INTERNAL);
    defer {
        principal.deinit();
        allocator.destroy(principal);
    }

    try std.testing.expect(!constraint.isViolated(principal));

    try principal.addRole(10);
    try principal.addRole(20);
    try std.testing.expect(constraint.isViolated(principal));
}

test "HashChain integrity" {
    const allocator = std.testing.allocator;

    var chain = try HashChain.init(allocator);
    defer chain.deinit();

    try chain.append("block1");
    try chain.append("block2");
    try chain.append("block3");

    try std.testing.expect(chain.length() == 4);
    try std.testing.expect(chain.verify());

    const block = chain.getBlock(2);
    try std.testing.expect(block != null);
}

test "MerkleTree construction and verification" {
    const allocator = std.testing.allocator;

    var tree = MerkleTree.init(allocator);
    defer tree.deinit();

    const items = [_][]const u8{ "item1", "item2", "item3", "item4" };
    try tree.build(&items);

    try std.testing.expect(tree.getRootHash() != null);
    try std.testing.expect(tree.verify());
}

test "MerkleProof generation and verification" {
    const allocator = std.testing.allocator;

    var tree = MerkleTree.init(allocator);
    defer tree.deinit();

    const items = [_][]const u8{ "data1", "data2", "data3", "data4" };
    try tree.build(&items);

    var proof = try tree.generateProof(0);
    defer proof.deinit();

    try std.testing.expect(proof.verify("data1"));
    try std.testing.expect(!proof.verify("wrong_data"));
}

test "Commitment scheme" {
    const allocator = std.testing.allocator;

    var scheme = CommitmentScheme.init(allocator);
    defer scheme.deinit();

    const commit_id = try scheme.createCommitment("secret_value");
    try std.testing.expect(scheme.getCommitmentHash(commit_id) != null);

    const valid_reveal = try scheme.revealCommitment(commit_id, "secret_value");
    try std.testing.expect(valid_reveal);

    try std.testing.expect(scheme.verifyCommitment(commit_id, "secret_value"));
    try std.testing.expect(!scheme.verifyCommitment(commit_id, "wrong_value"));
}

test "CryptographicProof creation and verification" {
    const allocator = std.testing.allocator;

    var proof = try CryptographicProof.init(
        allocator,
        1,
        .KNOWLEDGE,
        "I know the secret",
        "the_secret_witness",
    );
    defer {
        proof.deinit();
        allocator.destroy(proof);
    }

    try std.testing.expect(proof.is_valid);
    try std.testing.expect(proof.verify("the_secret_witness"));
    try std.testing.expect(!proof.verify("wrong_witness"));
}

test "BellLaPadula model" {
    const allocator = std.testing.allocator;

    var blp = BellLaPadula.init(allocator);
    defer blp.deinit();

    try blp.addSubject(1, .SECRET);
    try blp.addObject(100, .CONFIDENTIAL);
    try blp.addObject(200, .TOP_SECRET);

    try std.testing.expect(blp.canRead(1, 100));
    try std.testing.expect(!blp.canRead(1, 200));

    try std.testing.expect(!blp.canWrite(1, 100));
    try std.testing.expect(blp.canWrite(1, 200));
}

test "Biba integrity model" {
    const allocator = std.testing.allocator;

    var biba = Biba.init(allocator);
    defer biba.deinit();

    try biba.addSubject(1, .SYSTEM);
    try biba.addObject(100, .USER);
    try biba.addObject(200, .KERNEL);

    try std.testing.expect(!biba.canRead(1, 100));
    try std.testing.expect(biba.canRead(1, 200));

    try std.testing.expect(biba.canWrite(1, 100));
    try std.testing.expect(!biba.canWrite(1, 200));
}

test "IntegrityVerifier operations" {
    const allocator = std.testing.allocator;

    var verifier = try IntegrityVerifier.init(allocator);
    defer verifier.deinit();

    const data_id = try verifier.registerData("important data");

    try std.testing.expect(verifier.verifyIntegrity(data_id, "important data"));
    try std.testing.expect(!verifier.verifyIntegrity(data_id, "tampered data"));

    try std.testing.expect(verifier.verifyChainIntegrity());
}

test "SecurityProof creation and finalization" {
    const allocator = std.testing.allocator;

    var proof = try SecurityProof.init(
        allocator,
        1,
        .INFORMATION_FLOW,
        "All flows are secure",
    );
    defer {
        proof.deinit();
        allocator.destroy(proof);
    }

    try proof.addStep("Step 1", "RULE_1", true);
    try proof.addStep("Step 2", "RULE_2", true);

    proof.finalize();

    try std.testing.expect(proof.is_valid);
    try std.testing.expect(proof.cryptographic_binding != null);
    try std.testing.expect(proof.verify());
}

test "SecurityProofEngine information flow proof" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const node1 = try Node.init(allocator, "node1", "data1", Qubit{ .a = Complex(f64).init(0.3, 0.0), .b = Complex(f64).init(0.0, 0.0) }, 0.0);
    try graph.addNode(node1);

    const node2 = try Node.init(allocator, "node2", "data2", Qubit{ .a = Complex(f64).init(0.5, 0.0), .b = Complex(f64).init(0.0, 0.0) }, 0.0);
    try graph.addNode(node2);

    var engine = try SecurityProofEngine.init(allocator, .INTERNAL);
    defer engine.deinit();

    const proof = try engine.proveInformationFlowSecurity(&graph);
    try std.testing.expect(proof.proof_type == .INFORMATION_FLOW);
    try std.testing.expect(proof.stepCount() > 0);
}

test "SecurityProofEngine non-interference proof" {
    const allocator = std.testing.allocator;

    var state1 = SystemState.init(allocator);
    defer state1.deinit();
    try state1.setVariable(1, "low_value", .PUBLIC);
    try state1.setVariable(2, "high_a", .SECRET);

    var state2 = SystemState.init(allocator);
    defer state2.deinit();
    try state2.setVariable(1, "low_value", .PUBLIC);
    try state2.setVariable(2, "high_b", .SECRET);

    var engine = try SecurityProofEngine.init(allocator, .INTERNAL);
    defer engine.deinit();

    const proof = try engine.proveNonInterference(&state1, &state2, .INTERNAL);
    try std.testing.expect(proof.proof_type == .NON_INTERFERENCE);
    try std.testing.expect(proof.is_valid);
}

test "SecurityProofEngine integrity proof" {
    const allocator = std.testing.allocator;

    var engine = try SecurityProofEngine.init(allocator, .INTERNAL);
    defer engine.deinit();

    const data = "critical system data";
    const data_id = try engine.registerDataForIntegrity(data);

    const proof = try engine.proveIntegrity(data_id, data);
    try std.testing.expect(proof.proof_type == .INTEGRITY);
    try std.testing.expect(proof.is_valid);
}

