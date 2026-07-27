const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const StringHashMap = std.StringHashMap;
const AutoHashMap = std.AutoHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const TypeTheoryError = error{
    TypeMismatch,
    UnificationFailure,
    LinearityViolation,
    InvalidTypeConstruction,
    VariableNotInContext,
    DuplicateBinding,
    InvalidApplication,
    InvalidProjection,
    CategoryLawViolation,
    OutOfMemory,
    InvalidIdentityComposition,
};

pub const TypeKind = enum(u8) {
    UNIT = 0,
    BOOL = 1,
    NAT = 2,
    INT = 3,
    REAL = 4,
    COMPLEX = 5,
    STRING = 6,
    ARRAY = 7,
    TUPLE = 8,
    RECORD = 9,
    SUM = 10,
    FUNCTION = 11,
    DEPENDENT_FUNCTION = 12,
    DEPENDENT_PAIR = 13,
    UNIVERSE = 14,
    IDENTITY = 15,
    QUANTUM_TYPE = 16,
    BOTTOM = 17,
    TOP = 18,
    VARIABLE = 19,
    APPLICATION = 20,

    const Self = @This();

    pub fn toString(self: Self) []const u8 {
        return switch (self) {
            .UNIT => "Unit",
            .BOOL => "Bool",
            .NAT => "Nat",
            .INT => "Int",
            .REAL => "Real",
            .COMPLEX => "Complex",
            .STRING => "String",
            .ARRAY => "Array",
            .TUPLE => "Tuple",
            .RECORD => "Record",
            .SUM => "Sum",
            .FUNCTION => "Function",
            .DEPENDENT_FUNCTION => "Pi",
            .DEPENDENT_PAIR => "Sigma",
            .UNIVERSE => "Type",
            .IDENTITY => "Id",
            .QUANTUM_TYPE => "Quantum",
            .BOTTOM => "Bottom",
            .TOP => "Top",
            .VARIABLE => "Var",
            .APPLICATION => "App",
        };
    }

    pub fn fromString(s: []const u8) ?Self {
        if (std.mem.eql(u8, s, "Unit")) return .UNIT;
        if (std.mem.eql(u8, s, "Bool")) return .BOOL;
        if (std.mem.eql(u8, s, "Nat")) return .NAT;
        if (std.mem.eql(u8, s, "Int")) return .INT;
        if (std.mem.eql(u8, s, "Real")) return .REAL;
        if (std.mem.eql(u8, s, "Complex")) return .COMPLEX;
        if (std.mem.eql(u8, s, "String")) return .STRING;
        if (std.mem.eql(u8, s, "Array")) return .ARRAY;
        if (std.mem.eql(u8, s, "Tuple")) return .TUPLE;
        if (std.mem.eql(u8, s, "Record")) return .RECORD;
        if (std.mem.eql(u8, s, "Sum")) return .SUM;
        if (std.mem.eql(u8, s, "Function")) return .FUNCTION;
        if (std.mem.eql(u8, s, "Pi")) return .DEPENDENT_FUNCTION;
        if (std.mem.eql(u8, s, "Sigma")) return .DEPENDENT_PAIR;
        if (std.mem.eql(u8, s, "Type")) return .UNIVERSE;
        if (std.mem.eql(u8, s, "Id")) return .IDENTITY;
        if (std.mem.eql(u8, s, "Quantum")) return .QUANTUM_TYPE;
        if (std.mem.eql(u8, s, "Bottom")) return .BOTTOM;
        if (std.mem.eql(u8, s, "Top")) return .TOP;
        if (std.mem.eql(u8, s, "Var")) return .VARIABLE;
        if (std.mem.eql(u8, s, "App")) return .APPLICATION;
        return null;
    }

    pub fn isBaseType(self: Self) bool {
        return switch (self) {
            .UNIT, .BOOL, .NAT, .INT, .REAL, .COMPLEX, .STRING, .BOTTOM, .TOP => true,
            else => false,
        };
    }

    pub fn isComposite(self: Self) bool {
        return switch (self) {
            .ARRAY, .TUPLE, .RECORD, .SUM, .FUNCTION, .DEPENDENT_FUNCTION, .DEPENDENT_PAIR => true,
            else => false,
        };
    }

    pub fn isDependent(self: Self) bool {
        return self == .DEPENDENT_FUNCTION or self == .DEPENDENT_PAIR or self == .IDENTITY;
    }
};

pub const RecordField = struct {
    name: []const u8,
    field_type: ?*Type,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, name: []const u8, field_type: *Type) !*Self {
        const field = try allocator.create(Self);
        errdefer allocator.destroy(field);
        field.* = Self{
            .name = try allocator.dupe(u8, name),
            .field_type = field_type,
            .allocator = allocator,
        };
        return field;
    }

    pub fn deinit(self: *Self) void {
        if (self.field_type) |ft| {
            ft.deinit();
            self.allocator.destroy(ft);
            self.field_type = null;
        }
        if (self.name.len > 0) {
            self.allocator.free(self.name);
            self.name = "";
        }
    }

    pub fn clone(self: *const Self, allocator: Allocator) error{OutOfMemory}!*Self {
        const cloned_type = try self.field_type.?.clone(allocator);
        errdefer {
            cloned_type.deinit();
            allocator.destroy(cloned_type);
        }
        return RecordField.init(allocator, self.name, cloned_type);
    }
};

pub const Type = struct {
    kind: TypeKind,
    name: []const u8,
    parameters: ArrayList(*Type),
    fields: ArrayList(*RecordField),
    universe_level: u32,
    bound_variable: ?[]const u8,
    body_type: ?*Type,
    left_type: ?*Type,
    right_type: ?*Type,
    quantum_dimension: ?u32,
    hash_cache: ?[32]u8,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, kind: TypeKind) !*Self {
        const t = try allocator.create(Self);
        errdefer allocator.destroy(t);
        t.* = Self{
            .kind = kind,
            .name = "",
            .parameters = ArrayList(*Type).init(allocator),
            .fields = ArrayList(*RecordField).init(allocator),
            .universe_level = 0,
            .bound_variable = null,
            .body_type = null,
            .left_type = null,
            .right_type = null,
            .quantum_dimension = null,
            .hash_cache = null,
            .allocator = allocator,
        };
        return t;
    }

    pub fn initUnit(allocator: Allocator) !*Self {
        return Type.init(allocator, .UNIT);
    }

    pub fn initBool(allocator: Allocator) !*Self {
        return Type.init(allocator, .BOOL);
    }

    pub fn initNat(allocator: Allocator) !*Self {
        return Type.init(allocator, .NAT);
    }

    pub fn initInt(allocator: Allocator) !*Self {
        return Type.init(allocator, .INT);
    }

    pub fn initReal(allocator: Allocator) !*Self {
        return Type.init(allocator, .REAL);
    }

    pub fn initComplex(allocator: Allocator) !*Self {
        return Type.init(allocator, .COMPLEX);
    }

    pub fn initString(allocator: Allocator) !*Self {
        return Type.init(allocator, .STRING);
    }

    pub fn initBottom(allocator: Allocator) !*Self {
        return Type.init(allocator, .BOTTOM);
    }

    pub fn initTop(allocator: Allocator) !*Self {
        return Type.init(allocator, .TOP);
    }

    pub fn initVariable(allocator: Allocator, name: []const u8) !*Self {
        const t = try Type.init(allocator, .VARIABLE);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        t.name = try allocator.dupe(u8, name);
        return t;
    }

    pub fn initArray(allocator: Allocator, element_type: *Type) !*Self {
        const t = try Type.init(allocator, .ARRAY);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        try t.parameters.append(element_type);
        return t;
    }

    pub fn initTuple(allocator: Allocator, types: []const *Type) !*Self {
        const t = try Type.init(allocator, .TUPLE);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        for (types) |ty| {
            try t.parameters.append(ty);
        }
        return t;
    }

    pub fn initRecord(allocator: Allocator, fields: []const *RecordField) !*Self {
        const t = try Type.init(allocator, .RECORD);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        for (fields) |field| {
            try t.fields.append(field);
        }
        return t;
    }

    pub fn initSum(allocator: Allocator, left: *Type, right: *Type) !*Self {
        const t = try Type.init(allocator, .SUM);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        t.left_type = left;
        t.right_type = right;
        return t;
    }

    pub fn initFunction(allocator: Allocator, domain: *Type, codomain: *Type) !*Self {
        const t = try Type.init(allocator, .FUNCTION);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        t.left_type = domain;
        t.right_type = codomain;
        return t;
    }

    pub fn initUniverse(allocator: Allocator, level: u32) !*Self {
        const t = try Type.init(allocator, .UNIVERSE);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        t.universe_level = level;
        return t;
    }

    pub fn initQuantum(allocator: Allocator, base_type: *Type, dimension: u32) !*Self {
        const t = try Type.init(allocator, .QUANTUM_TYPE);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        try t.parameters.append(base_type);
        t.quantum_dimension = dimension;
        return t;
    }

    pub fn initApplication(allocator: Allocator, func_type: *Type, arg_type: *Type) !*Self {
        const t = try Type.init(allocator, .APPLICATION);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        t.left_type = func_type;
        t.right_type = arg_type;
        return t;
    }

    pub fn deinit(self: *Self) void {
        if (self.name.len > 0) {
            self.allocator.free(self.name);
            self.name = "";
        }
        for (self.parameters.items) |param| {
            param.deinit();
            self.allocator.destroy(param);
        }
        self.parameters.deinit();
        for (self.fields.items) |field| {
            field.deinit();
            self.allocator.destroy(field);
        }
        self.fields.deinit();
        if (self.bound_variable) |bv| {
            self.allocator.free(bv);
            self.bound_variable = null;
        }
        if (self.body_type) |body| {
            body.deinit();
            self.allocator.destroy(body);
            self.body_type = null;
        }
        if (self.left_type) |left| {
            left.deinit();
            self.allocator.destroy(left);
            self.left_type = null;
        }
        if (self.right_type) |right| {
            right.deinit();
            self.allocator.destroy(right);
            self.right_type = null;
        }
        self.hash_cache = null;
    }

    pub fn clone(self: *const Self, allocator: Allocator) error{OutOfMemory}!*Self {
        const t = try allocator.create(Self);
        errdefer allocator.destroy(t);
        t.* = Self{
            .kind = self.kind,
            .name = "",
            .parameters = ArrayList(*Type).init(allocator),
            .fields = ArrayList(*RecordField).init(allocator),
            .universe_level = self.universe_level,
            .bound_variable = null,
            .body_type = null,
            .left_type = null,
            .right_type = null,
            .quantum_dimension = self.quantum_dimension,
            .hash_cache = null,
            .allocator = allocator,
        };
        errdefer t.deinit();
        if (self.name.len > 0) {
            t.name = try allocator.dupe(u8, self.name);
        }
        if (self.bound_variable) |bv| {
            t.bound_variable = try allocator.dupe(u8, bv);
        }
        if (self.body_type) |body| {
            t.body_type = try body.clone(allocator);
        }
        if (self.left_type) |left| {
            t.left_type = try left.clone(allocator);
        }
        if (self.right_type) |right| {
            t.right_type = try right.clone(allocator);
        }
        for (self.parameters.items) |param| {
            try t.parameters.append(try param.clone(allocator));
        }
        for (self.fields.items) |field| {
            try t.fields.append(try field.clone(allocator));
        }
        return t;
    }

    pub fn equals(self: *const Self, other: *const Self) bool {
        if (self.kind != other.kind) return false;
        if (self.universe_level != other.universe_level) return false;
        if (!std.mem.eql(u8, self.name, other.name)) return false;
        if (self.parameters.items.len != other.parameters.items.len) return false;
        var param_idx: usize = 0;
        while (param_idx < self.parameters.items.len) : (param_idx += 1) {
            if (!self.parameters.items[param_idx].equals(other.parameters.items[param_idx])) return false;
        }
        if (self.fields.items.len != other.fields.items.len) return false;
        var field_idx: usize = 0;
        while (field_idx < self.fields.items.len) : (field_idx += 1) {
            const field = self.fields.items[field_idx];
            if (!std.mem.eql(u8, field.name, other.fields.items[field_idx].name)) return false;
            if (!field.field_type.?.equals(other.fields.items[field_idx].field_type.?)) return false;
        }
        if (self.bound_variable) |bv1| {
            if (other.bound_variable) |bv2| {
                if (!std.mem.eql(u8, bv1, bv2)) return false;
            } else {
                return false;
            }
        } else if (other.bound_variable != null) {
            return false;
        }
        if (self.left_type != null and other.left_type != null) {
            if (!self.left_type.?.equals(other.left_type.?)) return false;
        } else if (self.left_type != null or other.left_type != null) {
            return false;
        }
        if (self.right_type != null and other.right_type != null) {
            if (!self.right_type.?.equals(other.right_type.?)) return false;
        } else if (self.right_type != null or other.right_type != null) {
            return false;
        }
        if (self.body_type != null and other.body_type != null) {
            if (!self.body_type.?.equals(other.body_type.?)) return false;
        } else if (self.body_type != null or other.body_type != null) {
            return false;
        }
        return true;
    }

    pub fn computeHash(self: *Self) [32]u8 {
        if (self.hash_cache) |cache| {
            return cache;
        }
        var hasher = Sha256.init(.{});
        hasher.update(&[_]u8{@intFromEnum(self.kind)});
        hasher.update(self.name);
        var level_bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &level_bytes, self.universe_level, .little);
        hasher.update(&level_bytes);
        for (self.parameters.items) |param| {
            const param_hash = param.computeHash();
            hasher.update(&param_hash);
        }
        for (self.fields.items) |field| {
            hasher.update(field.name);
            const field_type_hash = field.field_type.?.computeHash();
            hasher.update(&field_type_hash);
        }
        if (self.bound_variable) |bv| {
            hasher.update(bv);
        }
        if (self.left_type) |left| {
            const left_hash = left.computeHash();
            hasher.update(&left_hash);
        }
        if (self.right_type) |right| {
            const right_hash = right.computeHash();
            hasher.update(&right_hash);
        }
        if (self.body_type) |body| {
            const body_hash = body.computeHash();
            hasher.update(&body_hash);
        }
        var result: [32]u8 = undefined;
        hasher.final(&result);
        self.hash_cache = result;
        return result;
    }

    pub fn getDomain(self: *const Self) ?*Type {
        return switch (self.kind) {
            .FUNCTION, .DEPENDENT_FUNCTION => self.left_type,
            else => null,
        };
    }

    pub fn getCodomain(self: *const Self) ?*Type {
        return switch (self.kind) {
            .FUNCTION => self.right_type,
            .DEPENDENT_FUNCTION => self.body_type,
            else => null,
        };
    }

    pub fn getElementType(self: *const Self) ?*Type {
        return switch (self.kind) {
            .ARRAY => if (self.parameters.items.len > 0) self.parameters.items[0] else null,
            else => null,
        };
    }

    pub fn getUniverseLevel(self: *const Self) u32 {
        return switch (self.kind) {
            .UNIVERSE => self.universe_level,
            .DEPENDENT_FUNCTION, .DEPENDENT_PAIR => blk: {
                var max_level: u32 = 0;
                if (self.left_type) |left| {
                    max_level = @max(max_level, left.getUniverseLevel());
                }
                if (self.body_type) |body| {
                    max_level = @max(max_level, body.getUniverseLevel());
                }
                break :blk max_level;
            },
            else => 0,
        };
    }

    pub fn substitute(self: *Self, var_name: []const u8, replacement: *const Type) !void {
        if (self.kind == .VARIABLE and std.mem.eql(u8, self.name, var_name)) {
            const name_copy = if (replacement.name.len > 0) try self.allocator.dupe(u8, replacement.name) else "";
            errdefer if (name_copy.len > 0) self.allocator.free(name_copy);

            var new_parameters = ArrayList(*Type).init(self.allocator);
            errdefer new_parameters.deinit();
            for (replacement.parameters.items) |param| {
                try new_parameters.append(try param.clone(self.allocator));
            }

            var new_fields = ArrayList(*RecordField).init(self.allocator);
            errdefer new_fields.deinit();
            for (replacement.fields.items) |field| {
                try new_fields.append(try field.clone(self.allocator));
            }

            const new_bound_variable: ?[]const u8 = if (replacement.bound_variable) |bv| try self.allocator.dupe(u8, bv) else null;
            errdefer if (new_bound_variable) |bv| self.allocator.free(bv);

            const new_body_type: ?*Type = if (replacement.body_type) |body| try body.clone(self.allocator) else null;
            errdefer if (new_body_type) |bt| {
                bt.deinit();
                self.allocator.destroy(bt);
            };

            const new_left_type: ?*Type = if (replacement.left_type) |left| try left.clone(self.allocator) else null;
            errdefer if (new_left_type) |lt| {
                lt.deinit();
                self.allocator.destroy(lt);
            };

            const new_right_type: ?*Type = if (replacement.right_type) |right| try right.clone(self.allocator) else null;
            errdefer if (new_right_type) |rt| {
                rt.deinit();
                self.allocator.destroy(rt);
            };

            for (self.parameters.items) |p| {
                p.deinit();
                self.allocator.destroy(p);
            }
            self.parameters.deinit();

            for (self.fields.items) |field| {
                field.deinit();
                self.allocator.destroy(field);
            }
            self.fields.deinit();

            if (self.name.len > 0) {
                self.allocator.free(self.name);
            }
            if (self.bound_variable) |bv| {
                self.allocator.free(bv);
            }
            if (self.body_type) |body| {
                body.deinit();
                self.allocator.destroy(body);
            }
            if (self.left_type) |left| {
                left.deinit();
                self.allocator.destroy(left);
            }
            if (self.right_type) |right| {
                right.deinit();
                self.allocator.destroy(right);
            }

            self.kind = replacement.kind;
            self.name = name_copy;
            self.universe_level = replacement.universe_level;
            self.parameters = new_parameters;
            self.fields = new_fields;
            self.bound_variable = new_bound_variable;
            self.body_type = new_body_type;
            self.left_type = new_left_type;
            self.right_type = new_right_type;
            self.quantum_dimension = replacement.quantum_dimension;
            self.hash_cache = null;
        } else {
            for (self.parameters.items) |param| {
                try param.substitute(var_name, replacement);
            }
            for (self.fields.items) |field| {
                if (field.field_type) |ft| {
                    try ft.substitute(var_name, replacement);
                }
            }
            if (self.left_type) |left| {
                try left.substitute(var_name, replacement);
            }
            if (self.right_type) |right| {
                try right.substitute(var_name, replacement);
            }
            if (self.body_type) |body| {
                if (self.bound_variable == null or !std.mem.eql(u8, self.bound_variable.?, var_name)) {
                    try body.substitute(var_name, replacement);
                }
            }
            self.hash_cache = null;
        }
    }

    pub fn containsFreeVariable(self: *const Self, var_name: []const u8) bool {
        if (self.kind == .VARIABLE and std.mem.eql(u8, self.name, var_name)) {
            return true;
        }
        for (self.parameters.items) |param| {
            if (param.containsFreeVariable(var_name)) return true;
        }
        for (self.fields.items) |field| {
            if (field.field_type.?.containsFreeVariable(var_name)) return true;
        }
        if (self.left_type) |left| {
            if (left.containsFreeVariable(var_name)) return true;
        }
        if (self.right_type) |right| {
            if (right.containsFreeVariable(var_name)) return true;
        }
        if (self.body_type) |body| {
            if (self.bound_variable != null and std.mem.eql(u8, self.bound_variable.?, var_name)) {
                return false;
            }
            if (body.containsFreeVariable(var_name)) return true;
        }
        return false;
    }
};

pub const TypeBinding = struct {
    name: []const u8,
    bound_type: *Type,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, name: []const u8, bound_type: *Type) !*Self {
        const binding = try allocator.create(Self);
        errdefer allocator.destroy(binding);
        binding.* = Self{
            .name = try allocator.dupe(u8, name),
            .bound_type = bound_type,
            .allocator = allocator,
        };
        return binding;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
        self.bound_type.deinit();
        self.allocator.destroy(self.bound_type);
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const cloned_type = try self.bound_type.clone(allocator);
        errdefer {
            cloned_type.deinit();
            allocator.destroy(cloned_type);
        }
        return TypeBinding.init(allocator, self.name, cloned_type);
    }
};

pub const TypeContext = struct {
    bindings: ArrayList(*TypeBinding),
    parent: ?*TypeContext,
    owns_parent: bool,
    allocator: Allocator,
    depth: u32,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .bindings = ArrayList(*TypeBinding).init(allocator),
            .parent = null,
            .owns_parent = false,
            .allocator = allocator,
            .depth = 0,
        };
    }

    pub fn initWithParent(allocator: Allocator, parent: *TypeContext) Self {
        return Self{
            .bindings = ArrayList(*TypeBinding).init(allocator),
            .parent = parent,
            .owns_parent = false,
            .allocator = allocator,
            .depth = parent.depth + 1,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.bindings.items) |binding| {
            binding.deinit();
            self.allocator.destroy(binding);
        }
        self.bindings.deinit();
        if (self.owns_parent) {
            if (self.parent) |p| {
                p.deinit();
                self.allocator.destroy(p);
            }
        }
    }

    pub fn extend(self: *Self, name: []const u8, bound_type: *Type) !void {
        const cloned_type = try bound_type.clone(self.allocator);
        errdefer {
            cloned_type.deinit();
            self.allocator.destroy(cloned_type);
        }
        const binding = try TypeBinding.init(self.allocator, name, cloned_type);
        errdefer {
            binding.deinit();
            self.allocator.destroy(binding);
        }
        try self.bindings.append(binding);
    }

    pub fn lookup(self: *const Self, name: []const u8) ?*Type {
        var i: usize = self.bindings.items.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, self.bindings.items[i].name, name)) {
                return self.bindings.items[i].bound_type;
            }
        }
        if (self.parent) |p| {
            return p.lookup(name);
        }
        return null;
    }

    pub fn contains(self: *const Self, name: []const u8) bool {
        return self.lookup(name) != null;
    }

    pub fn size(self: *const Self) usize {
        var count = self.bindings.items.len;
        if (self.parent) |p| {
            count += p.size();
        }
        return count;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const ctx = try allocator.create(Self);
        errdefer allocator.destroy(ctx);
        const cloned_parent = if (self.parent) |p| try p.clone(allocator) else null;
        errdefer {
            if (cloned_parent) |cp| {
                cp.deinit();
                allocator.destroy(cp);
            }
        }
        ctx.* = Self{
            .bindings = ArrayList(*TypeBinding).init(allocator),
            .parent = cloned_parent,
            .owns_parent = cloned_parent != null,
            .allocator = allocator,
            .depth = self.depth,
        };
        errdefer ctx.deinit();
        for (self.bindings.items) |binding| {
            try ctx.bindings.append(try binding.clone(allocator));
        }
        return ctx;
    }

    pub fn merge(self: *Self, other: *const Self) !void {
        for (other.bindings.items) |binding| {
            if (self.contains(binding.name)) {
                return TypeTheoryError.DuplicateBinding;
            }
            try self.bindings.append(try binding.clone(self.allocator));
        }
    }
};

pub const TermKind = enum(u8) {
    VARIABLE = 0,
    LITERAL = 1,
    LAMBDA = 2,
    APPLICATION = 3,
    PAIR = 4,
    FIRST = 5,
    SECOND = 6,
    INL = 7,
    INR = 8,
    CASE = 9,
    UNIT = 10,
    REFL = 11,
    J_ELIMINATOR = 12,
    ZERO = 13,
    SUCC = 14,
    NAT_REC = 15,
    LET = 16,
    ANNOTATION = 17,

    const Self = @This();

    pub fn toString(self: Self) []const u8 {
        return switch (self) {
            .VARIABLE => "var",
            .LITERAL => "lit",
            .LAMBDA => "lam",
            .APPLICATION => "app",
            .PAIR => "pair",
            .FIRST => "fst",
            .SECOND => "snd",
            .INL => "inl",
            .INR => "inr",
            .CASE => "case",
            .UNIT => "unit",
            .REFL => "refl",
            .J_ELIMINATOR => "J",
            .ZERO => "zero",
            .SUCC => "succ",
            .NAT_REC => "natrec",
            .LET => "let",
            .ANNOTATION => "ann",
        };
    }
};

pub const Term = struct {
    kind: TermKind,
    name: []const u8,
    sub_terms: ArrayList(*Term),
    bound_variable: ?[]const u8,
    annotation_type: ?*Type,
    literal_value: ?LiteralValue,
    allocator: Allocator,

    pub const LiteralValue = union(enum) {
        bool_val: bool,
        nat_val: u64,
        int_val: i64,
        real_val: f64,
        string_val: []u8,
    };

    const Self = @This();

    pub fn init(allocator: Allocator, kind: TermKind) !*Self {
        const t = try allocator.create(Self);
        errdefer allocator.destroy(t);
        t.* = Self{
            .kind = kind,
            .name = "",
            .sub_terms = ArrayList(*Term).init(allocator),
            .bound_variable = null,
            .annotation_type = null,
            .literal_value = null,
            .allocator = allocator,
        };
        return t;
    }

    pub fn initVariable(allocator: Allocator, name: []const u8) !*Self {
        const t = try Term.init(allocator, .VARIABLE);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        t.name = try allocator.dupe(u8, name);
        return t;
    }

    pub fn initLambda(allocator: Allocator, param: []const u8, body: *Term) !*Self {
        const t = try Term.init(allocator, .LAMBDA);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        t.bound_variable = try allocator.dupe(u8, param);
        try t.sub_terms.append(body);
        return t;
    }

    pub fn initApplication(allocator: Allocator, func: *Term, arg: *Term) !*Self {
        const t = try Term.init(allocator, .APPLICATION);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        try t.sub_terms.append(func);
        try t.sub_terms.append(arg);
        return t;
    }

    pub fn initPair(allocator: Allocator, first: *Term, second: *Term) !*Self {
        const t = try Term.init(allocator, .PAIR);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        try t.sub_terms.append(first);
        try t.sub_terms.append(second);
        return t;
    }

    pub fn initFirst(allocator: Allocator, pair: *Term) !*Self {
        const t = try Term.init(allocator, .FIRST);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        try t.sub_terms.append(pair);
        return t;
    }

    pub fn initSecond(allocator: Allocator, pair: *Term) !*Self {
        const t = try Term.init(allocator, .SECOND);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        try t.sub_terms.append(pair);
        return t;
    }

    pub fn initInl(allocator: Allocator, value: *Term) !*Self {
        const t = try Term.init(allocator, .INL);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        try t.sub_terms.append(value);
        return t;
    }

    pub fn initInr(allocator: Allocator, value: *Term) !*Self {
        const t = try Term.init(allocator, .INR);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        try t.sub_terms.append(value);
        return t;
    }

    pub fn initUnit(allocator: Allocator) !*Self {
        return Term.init(allocator, .UNIT);
    }

    pub fn initRefl(allocator: Allocator, witness: *Term) !*Self {
        const t = try Term.init(allocator, .REFL);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        try t.sub_terms.append(witness);
        return t;
    }

    pub fn initZero(allocator: Allocator) !*Self {
        return Term.init(allocator, .ZERO);
    }

    pub fn initSucc(allocator: Allocator, n: *Term) !*Self {
        const t = try Term.init(allocator, .SUCC);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        try t.sub_terms.append(n);
        return t;
    }

    pub fn initLiteralNat(allocator: Allocator, value: u64) !*Self {
        const t = try Term.init(allocator, .LITERAL);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        t.literal_value = .{ .nat_val = value };
        return t;
    }

    pub fn initLiteralBool(allocator: Allocator, value: bool) !*Self {
        const t = try Term.init(allocator, .LITERAL);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        t.literal_value = .{ .bool_val = value };
        return t;
    }

    pub fn initAnnotation(allocator: Allocator, term: *Term, ann_type: *Type) !*Self {
        const t = try Term.init(allocator, .ANNOTATION);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        try t.sub_terms.append(term);
        t.annotation_type = ann_type;
        return t;
    }

    pub fn deinit(self: *Self) void {
        if (self.name.len > 0) {
            self.allocator.free(self.name);
        }
        for (self.sub_terms.items) |sub| {
            sub.deinit();
            self.allocator.destroy(sub);
        }
        self.sub_terms.deinit();
        if (self.bound_variable) |bv| {
            self.allocator.free(bv);
        }
        if (self.annotation_type) |ann| {
            ann.deinit();
            self.allocator.destroy(ann);
        }
        if (self.literal_value) |lit| {
            switch (lit) {
                .string_val => |s| self.allocator.free(s),
                else => {},
            }
        }
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const t = try allocator.create(Self);
        errdefer allocator.destroy(t);
        t.* = Self{
            .kind = self.kind,
            .name = "",
            .sub_terms = ArrayList(*Term).init(allocator),
            .bound_variable = null,
            .annotation_type = null,
            .literal_value = null,
            .allocator = allocator,
        };
        errdefer t.deinit();
        if (self.name.len > 0) {
            t.name = try allocator.dupe(u8, self.name);
        }
        if (self.bound_variable) |bv| {
            t.bound_variable = try allocator.dupe(u8, bv);
        }
        if (self.annotation_type) |ann| {
            t.annotation_type = try ann.clone(allocator);
        }
        if (self.literal_value) |lit| {
            t.literal_value = switch (lit) {
                .string_val => |s| LiteralValue{ .string_val = try allocator.dupe(u8, s) },
                else => lit,
            };
        }
        for (self.sub_terms.items) |sub| {
            try t.sub_terms.append(try sub.clone(allocator));
        }
        return t;
    }

    pub fn equals(self: *const Self, other: *const Self) bool {
        if (self.kind != other.kind) return false;
        if (!std.mem.eql(u8, self.name, other.name)) return false;
        if (self.sub_terms.items.len != other.sub_terms.items.len) return false;
        var sub_idx: usize = 0;
        while (sub_idx < self.sub_terms.items.len) : (sub_idx += 1) {
            if (!self.sub_terms.items[sub_idx].equals(other.sub_terms.items[sub_idx])) return false;
        }
        if (self.bound_variable) |bv1| {
            if (other.bound_variable) |bv2| {
                if (!std.mem.eql(u8, bv1, bv2)) return false;
            } else {
                return false;
            }
        } else if (other.bound_variable != null) {
            return false;
        }
        if (self.annotation_type) |ann1| {
            if (other.annotation_type) |ann2| {
                if (!ann1.equals(ann2)) return false;
            } else {
                return false;
            }
        } else if (other.annotation_type != null) {
            return false;
        }
        if (self.literal_value) |lit1| {
            if (other.literal_value) |lit2| {
                const Tag = std.meta.Tag(LiteralValue);
                if (@as(Tag, lit1) != @as(Tag, lit2)) return false;
                switch (lit1) {
                    .bool_val => |b1| if (lit2.bool_val != b1) return false,
                    .nat_val => |n1| if (lit2.nat_val != n1) return false,
                    .int_val => |int_val_1| if (lit2.int_val != int_val_1) return false,
                    .real_val => |r1| if (lit2.real_val != r1) return false,
                    .string_val => |s1| if (!std.mem.eql(u8, s1, lit2.string_val)) return false,
                }
            } else {
                return false;
            }
        } else if (other.literal_value != null) {
            return false;
        }
        return true;
    }
};

pub const TypeJudgment = struct {
    context: *TypeContext,
    term: *Term,
    inferred_type: *Type,
    is_valid: bool,
    derivation_depth: u32,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, context: *TypeContext, term: *Term, inferred_type: *Type) !*Self {
        const j = try allocator.create(Self);
        errdefer allocator.destroy(j);
        const ctx_clone = try context.clone(allocator);
        errdefer {
            ctx_clone.deinit();
            allocator.destroy(ctx_clone);
        }
        const term_clone = try term.clone(allocator);
        errdefer {
            term_clone.deinit();
            allocator.destroy(term_clone);
        }
        j.* = Self{
            .context = ctx_clone,
            .term = term_clone,
            .inferred_type = inferred_type,
            .is_valid = false,
            .derivation_depth = 0,
            .allocator = allocator,
        };
        return j;
    }

    pub fn deinit(self: *Self) void {
        self.context.deinit();
        self.allocator.destroy(self.context);
        self.term.deinit();
        self.allocator.destroy(self.term);
        self.inferred_type.deinit();
        self.allocator.destroy(self.inferred_type);
    }

    pub fn validate(self: *Self) bool {
        self.is_valid = self.checkWellFormedness();
        return self.is_valid;
    }

    fn checkWellFormedness(self: *const Self) bool {
        switch (self.term.kind) {
            .VARIABLE => return self.context.contains(self.term.name),
            .UNIT => return self.inferred_type.kind == .UNIT,
            .ZERO => return self.inferred_type.kind == .NAT,
            .LITERAL => return self.checkLiteralType(),
            .LAMBDA => return self.inferred_type.kind == .FUNCTION or self.inferred_type.kind == .DEPENDENT_FUNCTION,
            .APPLICATION => return true,
            .PAIR => return self.inferred_type.kind == .TUPLE or self.inferred_type.kind == .DEPENDENT_PAIR,
            .FIRST => return true,
            .SECOND => return true,
            .INL => return true,
            .INR => return true,
            .SUCC => return self.inferred_type.kind == .NAT,
            .REFL => return self.inferred_type.kind == .IDENTITY,
            .ANNOTATION => return true,
            else => return true,
        }
    }

    fn checkLiteralType(self: *const Self) bool {
        if (self.term.literal_value) |lit| {
            return switch (lit) {
                .bool_val => self.inferred_type.kind == .BOOL,
                .nat_val => self.inferred_type.kind == .NAT,
                .int_val => self.inferred_type.kind == .INT,
                .real_val => self.inferred_type.kind == .REAL,
                .string_val => self.inferred_type.kind == .STRING,
            };
        }
        return false;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        return TypeJudgment.init(
            allocator,
            self.context,
            self.term,
            try self.inferred_type.clone(allocator),
        );
    }
};

pub const DependentPi = struct {
    param_name: []const u8,
    param_type: *Type,
    return_type: *Type,
    universe_level: u32,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, param_name: []const u8, param_type: *Type, return_type: *Type) !*Self {
        const pi = try allocator.create(Self);
        errdefer allocator.destroy(pi);
        pi.* = Self{
            .param_name = try allocator.dupe(u8, param_name),
            .param_type = param_type,
            .return_type = return_type,
            .universe_level = @max(param_type.getUniverseLevel(), return_type.getUniverseLevel()),
            .allocator = allocator,
        };
        return pi;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.param_name);
        self.param_type.deinit();
        self.allocator.destroy(self.param_type);
        self.return_type.deinit();
        self.allocator.destroy(self.return_type);
    }

    pub fn toType(self: *const Self, allocator: Allocator) !*Type {
        const t = try Type.init(allocator, .DEPENDENT_FUNCTION);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        t.bound_variable = try allocator.dupe(u8, self.param_name);
        t.left_type = try self.param_type.clone(allocator);
        t.body_type = try self.return_type.clone(allocator);
        t.universe_level = self.universe_level;
        return t;
    }

    pub fn apply(self: *Self, arg: *const Type) !*Type {
        const result = try self.return_type.clone(self.allocator);
        errdefer {
            result.deinit();
            self.allocator.destroy(result);
        }
        try result.substitute(self.param_name, arg);
        return result;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        return DependentPi.init(
            allocator,
            self.param_name,
            try self.param_type.clone(allocator),
            try self.return_type.clone(allocator),
        );
    }

    pub fn equals(self: *const Self, other: *const Self) bool {
        if (!std.mem.eql(u8, self.param_name, other.param_name)) return false;
        if (!self.param_type.equals(other.param_type)) return false;
        if (!self.return_type.equals(other.return_type)) return false;
        return true;
    }
};

pub const DependentSigma = struct {
    fst_name: []const u8,
    fst_type: *Type,
    snd_type: *Type,
    universe_level: u32,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, fst_name: []const u8, fst_type: *Type, snd_type: *Type) !*Self {
        const sigma = try allocator.create(Self);
        errdefer allocator.destroy(sigma);
        sigma.* = Self{
            .fst_name = try allocator.dupe(u8, fst_name),
            .fst_type = fst_type,
            .snd_type = snd_type,
            .universe_level = @max(fst_type.getUniverseLevel(), snd_type.getUniverseLevel()),
            .allocator = allocator,
        };
        return sigma;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.fst_name);
        self.fst_type.deinit();
        self.allocator.destroy(self.fst_type);
        self.snd_type.deinit();
        self.allocator.destroy(self.snd_type);
    }

    pub fn toType(self: *const Self, allocator: Allocator) !*Type {
        const t = try Type.init(allocator, .DEPENDENT_PAIR);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        t.bound_variable = try allocator.dupe(u8, self.fst_name);
        t.left_type = try self.fst_type.clone(allocator);
        t.body_type = try self.snd_type.clone(allocator);
        t.universe_level = self.universe_level;
        return t;
    }

    pub fn getSecondType(self: *Self, first_value: *const Type) !*Type {
        const result = try self.snd_type.clone(self.allocator);
        errdefer {
            result.deinit();
            self.allocator.destroy(result);
        }
        try result.substitute(self.fst_name, first_value);
        return result;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        return DependentSigma.init(
            allocator,
            self.fst_name,
            try self.fst_type.clone(allocator),
            try self.snd_type.clone(allocator),
        );
    }

    pub fn equals(self: *const Self, other: *const Self) bool {
        if (!std.mem.eql(u8, self.fst_name, other.fst_name)) return false;
        if (!self.fst_type.equals(other.fst_type)) return false;
        if (!self.snd_type.equals(other.snd_type)) return false;
        return true;
    }
};

pub const IdentityType = struct {
    base_type: *Type,
    left_term: *Term,
    right_term: *Term,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, base_type: *Type, left: *Term, right: *Term) !*Self {
        const id = try allocator.create(Self);
        errdefer allocator.destroy(id);
        id.* = Self{
            .base_type = base_type,
            .left_term = left,
            .right_term = right,
            .allocator = allocator,
        };
        return id;
    }

    pub fn deinit(self: *Self) void {
        self.base_type.deinit();
        self.allocator.destroy(self.base_type);
        self.left_term.deinit();
        self.allocator.destroy(self.left_term);
        self.right_term.deinit();
        self.allocator.destroy(self.right_term);
    }

    pub fn toType(self: *const Self, allocator: Allocator) !*Type {
        const t = try Type.init(allocator, .IDENTITY);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        try t.parameters.append(try self.base_type.clone(allocator));
        return t;
    }

    pub fn refl(allocator: Allocator, base_type: *Type, term: *Term) !*Self {
        const cloned_term = try term.clone(allocator);
        errdefer {
            cloned_term.deinit();
            allocator.destroy(cloned_term);
        }
        return IdentityType.init(allocator, base_type, term, cloned_term);
    }

    pub fn symmetry(self: *const Self, allocator: Allocator) !*Self {
        return IdentityType.init(
            allocator,
            try self.base_type.clone(allocator),
            try self.right_term.clone(allocator),
            try self.left_term.clone(allocator),
        );
    }

    pub fn transitivity(self: *const Self, other: *const Self, allocator: Allocator) !*Self {
        if (!self.right_term.equals(other.left_term)) {
            return TypeTheoryError.InvalidIdentityComposition;
        }
        if (!self.base_type.equals(other.base_type)) {
            return TypeTheoryError.TypeMismatch;
        }
        return IdentityType.init(
            allocator,
            try self.base_type.clone(allocator),
            try self.left_term.clone(allocator),
            try other.right_term.clone(allocator),
        );
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        return IdentityType.init(
            allocator,
            try self.base_type.clone(allocator),
            try self.left_term.clone(allocator),
            try self.right_term.clone(allocator),
        );
    }

    pub fn isReflexive(self: *const Self) bool {
        return self.left_term.equals(self.right_term);
    }
};

pub const UniverseType = struct {
    level: u32,
    cumulative: bool,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, level: u32) !*Self {
        const u = try allocator.create(Self);
        errdefer allocator.destroy(u);
        u.* = Self{
            .level = level,
            .cumulative = true,
            .allocator = allocator,
        };
        return u;
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn toType(self: *const Self, allocator: Allocator) !*Type {
        return Type.initUniverse(allocator, self.level);
    }

    pub fn typeOf(self: *const Self, allocator: Allocator) !*Self {
        return UniverseType.init(allocator, self.level + 1);
    }

    pub fn contains(self: *const Self, other: *const Self) bool {
        if (self.cumulative) {
            return other.level < self.level;
        }
        return if (self.level > 0) other.level == self.level - 1 else false;
    }

    pub fn lub(self: *const Self, other: *const Self, allocator: Allocator) !*Self {
        const res = try UniverseType.init(allocator, @max(self.level, other.level));
        res.cumulative = self.cumulative and other.cumulative;
        return res;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const u = try UniverseType.init(allocator, self.level);
        u.cumulative = self.cumulative;
        return u;
    }
};

pub const InductiveType = struct {
    name: []const u8,
    constructors: ArrayList(*Constructor),
    parameters: ArrayList(*Type),
    indices: ArrayList(*Type),
    universe_level: u32,
    allocator: Allocator,

    pub const Constructor = struct {
        name: []const u8,
        arg_types: ArrayList(*Type),
        result_type: *Type,
        allocator: Allocator,

        pub fn init(allocator: Allocator, name: []const u8, result_type: *Type) !*Constructor {
            const c = try allocator.create(Constructor);
            errdefer allocator.destroy(c);
            c.* = Constructor{
                .name = try allocator.dupe(u8, name),
                .arg_types = ArrayList(*Type).init(allocator),
                .result_type = result_type,
                .allocator = allocator,
            };
            return c;
        }

        pub fn deinit(self: *Constructor) void {
            self.allocator.free(self.name);
            for (self.arg_types.items) |t| {
                t.deinit();
                self.allocator.destroy(t);
            }
            self.arg_types.deinit();
            self.result_type.deinit();
            self.allocator.destroy(self.result_type);
        }

        pub fn addArgType(self: *Constructor, arg_type: *Type) !void {
            try self.arg_types.append(arg_type);
        }

        pub fn clone(self: *const Constructor, allocator: Allocator) !*Constructor {
            const result_type_clone = try self.result_type.clone(allocator);
            errdefer {
                result_type_clone.deinit();
                allocator.destroy(result_type_clone);
            }
            const c = try Constructor.init(allocator, self.name, result_type_clone);
            errdefer {
                c.deinit();
                allocator.destroy(c);
            }
            for (self.arg_types.items) |t| {
                try c.arg_types.append(try t.clone(allocator));
            }
            return c;
        }
    };

    const Self = @This();

    pub fn init(allocator: Allocator, name: []const u8) !*Self {
        const ind = try allocator.create(Self);
        errdefer allocator.destroy(ind);
        ind.* = Self{
            .name = try allocator.dupe(u8, name),
            .constructors = ArrayList(*Constructor).init(allocator),
            .parameters = ArrayList(*Type).init(allocator),
            .indices = ArrayList(*Type).init(allocator),
            .universe_level = 0,
            .allocator = allocator,
        };
        return ind;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
        for (self.constructors.items) |c| {
            c.deinit();
            self.allocator.destroy(c);
        }
        self.constructors.deinit();
        for (self.parameters.items) |p| {
            p.deinit();
            self.allocator.destroy(p);
        }
        self.parameters.deinit();
        for (self.indices.items) |i| {
            i.deinit();
            self.allocator.destroy(i);
        }
        self.indices.deinit();
    }

    pub fn addConstructor(self: *Self, constructor: *Constructor) !void {
        try self.constructors.append(constructor);
    }

    pub fn initNat(allocator: Allocator) !*Self {
        const nat = try InductiveType.init(allocator, "Nat");
        errdefer {
            nat.deinit();
            allocator.destroy(nat);
        }
        const nat_type = try Type.initNat(allocator);
        const zero = Constructor.init(allocator, "zero", nat_type) catch |err| {
            nat_type.deinit();
            allocator.destroy(nat_type);
            return err;
        };
        nat.addConstructor(zero) catch |err| {
            zero.deinit();
            allocator.destroy(zero);
            return err;
        };
        const succ_result = try Type.initNat(allocator);
        const succ = Constructor.init(allocator, "succ", succ_result) catch |err| {
            succ_result.deinit();
            allocator.destroy(succ_result);
            return err;
        };
        const succ_arg = Type.initNat(allocator) catch |err| {
            succ.deinit();
            allocator.destroy(succ);
            return err;
        };
        succ.addArgType(succ_arg) catch |err| {
            succ_arg.deinit();
            allocator.destroy(succ_arg);
            succ.deinit();
            allocator.destroy(succ);
            return err;
        };
        nat.addConstructor(succ) catch |err| {
            succ.deinit();
            allocator.destroy(succ);
            return err;
        };
        return nat;
    }

    pub fn initBool(allocator: Allocator) !*Self {
        const bool_type = try InductiveType.init(allocator, "Bool");
        errdefer {
            bool_type.deinit();
            allocator.destroy(bool_type);
        }
        const true_type = try Type.initBool(allocator);
        const true_ctor = Constructor.init(allocator, "true", true_type) catch |err| {
            true_type.deinit();
            allocator.destroy(true_type);
            return err;
        };
        bool_type.addConstructor(true_ctor) catch |err| {
            true_ctor.deinit();
            allocator.destroy(true_ctor);
            return err;
        };
        const false_type = try Type.initBool(allocator);
        const false_ctor = Constructor.init(allocator, "false", false_type) catch |err| {
            false_type.deinit();
            allocator.destroy(false_type);
            return err;
        };
        bool_type.addConstructor(false_ctor) catch |err| {
            false_ctor.deinit();
            allocator.destroy(false_ctor);
            return err;
        };
        return bool_type;
    }

    pub fn initList(allocator: Allocator, element_type: *Type) !*Self {
        const list = try InductiveType.init(allocator, "List");
        errdefer {
            list.deinit();
            allocator.destroy(list);
        }
        const elem_clone = try element_type.clone(allocator);
        list.parameters.append(elem_clone) catch |err| {
            elem_clone.deinit();
            allocator.destroy(elem_clone);
            return err;
        };
        const nil_inner = try element_type.clone(allocator);
        const array_type = Type.initArray(allocator, nil_inner) catch |err| {
            nil_inner.deinit();
            allocator.destroy(nil_inner);
            return err;
        };
        const nil = Constructor.init(allocator, "nil", array_type) catch |err| {
            array_type.deinit();
            allocator.destroy(array_type);
            return err;
        };
        list.addConstructor(nil) catch |err| {
            nil.deinit();
            allocator.destroy(nil);
            return err;
        };
        const cons_result_inner = try element_type.clone(allocator);
        const cons_result = Type.initArray(allocator, cons_result_inner) catch |err| {
            cons_result_inner.deinit();
            allocator.destroy(cons_result_inner);
            return err;
        };
        const cons = Constructor.init(allocator, "cons", cons_result) catch |err| {
            cons_result.deinit();
            allocator.destroy(cons_result);
            return err;
        };
        const cons_arg1 = element_type.clone(allocator) catch |err| {
            cons.deinit();
            allocator.destroy(cons);
            return err;
        };
        cons.addArgType(cons_arg1) catch |err| {
            cons_arg1.deinit();
            allocator.destroy(cons_arg1);
            cons.deinit();
            allocator.destroy(cons);
            return err;
        };
        const cons_arg2_inner = element_type.clone(allocator) catch |err| {
            cons.deinit();
            allocator.destroy(cons);
            return err;
        };
        const cons_arg2 = Type.initArray(allocator, cons_arg2_inner) catch |err| {
            cons_arg2_inner.deinit();
            allocator.destroy(cons_arg2_inner);
            cons.deinit();
            allocator.destroy(cons);
            return err;
        };
        cons.addArgType(cons_arg2) catch |err| {
            cons_arg2.deinit();
            allocator.destroy(cons_arg2);
            cons.deinit();
            allocator.destroy(cons);
            return err;
        };
        list.addConstructor(cons) catch |err| {
            cons.deinit();
            allocator.destroy(cons);
            return err;
        };
        return list;
    }

    pub fn toType(self: *const Self, allocator: Allocator) !*Type {
        const t = try Type.init(allocator, .VARIABLE);
        errdefer {
            t.deinit();
            allocator.destroy(t);
        }
        t.name = try allocator.dupe(u8, self.name);
        return t;
    }

    pub fn getRecursor(self: *const Self, motive_type: *Type, allocator: Allocator) !*Type {
        const ind_type = try self.toType(allocator);
        const motive_clone = Type.clone(motive_type, allocator) catch |err| {
            ind_type.deinit();
            allocator.destroy(ind_type);
            return err;
        };
        var rec_type = Type.initFunction(allocator, ind_type, motive_clone) catch |err| {
            ind_type.deinit();
            allocator.destroy(ind_type);
            motive_clone.deinit();
            allocator.destroy(motive_clone);
            return err;
        };
        errdefer {
            rec_type.deinit();
            allocator.destroy(rec_type);
        }
        for (self.constructors.items) |ctor| {
            const ctor_case_type = try self.buildConstructorCaseType(ctor, motive_type, allocator);
            const new_rec = Type.initFunction(allocator, ctor_case_type, rec_type) catch |err| {
                ctor_case_type.deinit();
                allocator.destroy(ctor_case_type);
                return err;
            };
            rec_type = new_rec;
        }
        return rec_type;
    }

    fn buildConstructorCaseType(self: *const Self, ctor: *Constructor, motive: *Type, allocator: Allocator) !*Type {
        _ = self;
        var result = try motive.clone(allocator);
        errdefer {
            result.deinit();
            allocator.destroy(result);
        }
        var i = ctor.arg_types.items.len;
        while (i > 0) {
            i -= 1;
            const arg_clone = try ctor.arg_types.items[i].clone(allocator);
            const new_result = Type.initFunction(allocator, arg_clone, result) catch |err| {
                arg_clone.deinit();
                allocator.destroy(arg_clone);
                return err;
            };
            result = new_result;
        }
        return result;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const ind = try InductiveType.init(allocator, self.name);
        errdefer {
            ind.deinit();
            allocator.destroy(ind);
        }
        for (self.constructors.items) |c| {
            try ind.constructors.append(try c.clone(allocator));
        }
        for (self.parameters.items) |p| {
            try ind.parameters.append(try p.clone(allocator));
        }
        for (self.indices.items) |i| {
            try ind.indices.append(try i.clone(allocator));
        }
        ind.universe_level = self.universe_level;
        return ind;
    }
};

pub const TypeChecker = struct {
    context: TypeContext,
    inference_count: u64,
    check_count: u64,
    unification_count: u64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .context = TypeContext.init(allocator),
            .inference_count = 0,
            .check_count = 0,
            .unification_count = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.context.deinit();
    }

    pub fn extendContext(self: *Self, name: []const u8, bound_type: *Type) !void {
        try self.context.extend(name, bound_type);
    }

    pub const InferError = error{OutOfMemory} || TypeTheoryError;

    pub fn checkType(self: *Self, ctx: *TypeContext, term: *Term, expected: *Type) InferError!bool {
        self.check_count += 1;
        const inferred = try self.inferType(ctx, term);
        defer {
            inferred.deinit();
            self.allocator.destroy(inferred);
        }
        return self.subtype(inferred, expected);
    }

    pub fn inferType(self: *Self, ctx: *TypeContext, term: *Term) InferError!*Type {
        self.inference_count += 1;
        return switch (term.kind) {
            .VARIABLE => self.inferVariable(ctx, term),
            .LITERAL => self.inferLiteral(term),
            .LAMBDA => self.inferLambda(ctx, term),
            .APPLICATION => self.inferApplication(ctx, term),
            .PAIR => self.inferPair(ctx, term),
            .FIRST => self.inferFirst(ctx, term),
            .SECOND => self.inferSecond(ctx, term),
            .INL => self.inferInl(ctx, term),
            .INR => self.inferInr(ctx, term),
            .UNIT => Type.initUnit(self.allocator),
            .ZERO => Type.initNat(self.allocator),
            .SUCC => self.inferSucc(ctx, term),
            .REFL => self.inferRefl(ctx, term),
            .ANNOTATION => self.inferAnnotation(term),
            .CASE => TypeTheoryError.InvalidTypeConstruction,
            .J_ELIMINATOR => TypeTheoryError.InvalidTypeConstruction,
            .NAT_REC => TypeTheoryError.InvalidTypeConstruction,
            .LET => TypeTheoryError.InvalidTypeConstruction,
        };
    }

    fn inferVariable(self: *Self, ctx: *TypeContext, term: *Term) InferError!*Type {
        const lookup_result = ctx.lookup(term.name);
        if (lookup_result) |found_type| {
            return found_type.clone(self.allocator);
        }
        return TypeTheoryError.VariableNotInContext;
    }

    fn inferLiteral(self: *Self, term: *Term) InferError!*Type {
        if (term.literal_value) |lit| {
            return switch (lit) {
                .bool_val => Type.initBool(self.allocator),
                .nat_val => Type.initNat(self.allocator),
                .int_val => Type.initInt(self.allocator),
                .real_val => Type.initReal(self.allocator),
                .string_val => Type.initString(self.allocator),
            };
        }
        return TypeTheoryError.InvalidTypeConstruction;
    }

    fn inferLambda(self: *Self, ctx: *TypeContext, term: *Term) InferError!*Type {
        if (term.bound_variable == null or term.sub_terms.items.len == 0) {
            return TypeTheoryError.InvalidTypeConstruction;
        }
        var extended_ctx = TypeContext.initWithParent(self.allocator, ctx);
        defer extended_ctx.deinit();
        const param_type = if (term.annotation_type) |ann| try ann.clone(self.allocator) else try Type.initTop(self.allocator);
        defer {
            param_type.deinit();
            self.allocator.destroy(param_type);
        }
        try extended_ctx.extend(term.bound_variable.?, param_type);
        const body_type = try self.inferType(&extended_ctx, term.sub_terms.items[0]);
        errdefer {
            body_type.deinit();
            self.allocator.destroy(body_type);
        }
        const cloned_param = try param_type.clone(self.allocator);
        errdefer {
            cloned_param.deinit();
            self.allocator.destroy(cloned_param);
        }
        return Type.initFunction(self.allocator, cloned_param, body_type);
    }

    fn inferApplication(self: *Self, ctx: *TypeContext, term: *Term) InferError!*Type {
        if (term.sub_terms.items.len < 2) {
            return TypeTheoryError.InvalidApplication;
        }
        const func_type = try self.inferType(ctx, term.sub_terms.items[0]);
        defer {
            func_type.deinit();
            self.allocator.destroy(func_type);
        }
        const arg_type = try self.inferType(ctx, term.sub_terms.items[1]);
        defer {
            arg_type.deinit();
            self.allocator.destroy(arg_type);
        }
        if (func_type.kind == .FUNCTION) {
            if (func_type.left_type) |domain| {
                if (!self.subtype(arg_type, domain)) {
                    return TypeTheoryError.TypeMismatch;
                }
                if (func_type.right_type) |codomain| {
                    return codomain.clone(self.allocator);
                } else {
                    return TypeTheoryError.InvalidTypeConstruction;
                }
            } else {
                return TypeTheoryError.InvalidTypeConstruction;
            }
        } else if (func_type.kind == .DEPENDENT_FUNCTION) {
            if (func_type.body_type) |body| {
                const result = try body.clone(self.allocator);
                errdefer {
                    result.deinit();
                    self.allocator.destroy(result);
                }
                if (func_type.bound_variable) |bv| {
                    try result.substitute(bv, arg_type);
                }
                return result;
            } else {
                return TypeTheoryError.InvalidTypeConstruction;
            }
        }
        return TypeTheoryError.InvalidApplication;
    }

    fn inferPair(self: *Self, ctx: *TypeContext, term: *Term) InferError!*Type {
        if (term.sub_terms.items.len < 2) {
            return TypeTheoryError.InvalidTypeConstruction;
        }
        const fst_type = try self.inferType(ctx, term.sub_terms.items[0]);
        errdefer {
            fst_type.deinit();
            self.allocator.destroy(fst_type);
        }
        const snd_type = try self.inferType(ctx, term.sub_terms.items[1]);
        errdefer {
            snd_type.deinit();
            self.allocator.destroy(snd_type);
        }
        return Type.initTuple(self.allocator, &[_]*Type{ fst_type, snd_type });
    }

    fn inferFirst(self: *Self, ctx: *TypeContext, term: *Term) InferError!*Type {
        if (term.sub_terms.items.len == 0) {
            return TypeTheoryError.InvalidProjection;
        }
        const pair_type = try self.inferType(ctx, term.sub_terms.items[0]);
        defer {
            pair_type.deinit();
            self.allocator.destroy(pair_type);
        }
        if (pair_type.kind == .TUPLE and pair_type.parameters.items.len > 0) {
            return pair_type.parameters.items[0].clone(self.allocator);
        } else if (pair_type.kind == .DEPENDENT_PAIR) {
            if (pair_type.left_type) |left| {
                return left.clone(self.allocator);
            }
        }
        return TypeTheoryError.InvalidProjection;
    }

    fn inferSecond(self: *Self, ctx: *TypeContext, term: *Term) InferError!*Type {
        if (term.sub_terms.items.len == 0) {
            return TypeTheoryError.InvalidProjection;
        }
        const pair_type = try self.inferType(ctx, term.sub_terms.items[0]);
        defer {
            pair_type.deinit();
            self.allocator.destroy(pair_type);
        }
        if (pair_type.kind == .TUPLE and pair_type.parameters.items.len > 1) {
            return pair_type.parameters.items[1].clone(self.allocator);
        } else if (pair_type.kind == .DEPENDENT_PAIR) {
            if (pair_type.body_type) |body| {
                return body.clone(self.allocator);
            }
        }
        return TypeTheoryError.InvalidProjection;
    }

    fn inferInl(self: *Self, ctx: *TypeContext, term: *Term) InferError!*Type {
        if (term.sub_terms.items.len == 0) {
            return TypeTheoryError.InvalidTypeConstruction;
        }
        const inner_type = try self.inferType(ctx, term.sub_terms.items[0]);
        errdefer {
            inner_type.deinit();
            self.allocator.destroy(inner_type);
        }
        const bottom = try Type.initBottom(self.allocator);
        errdefer {
            bottom.deinit();
            self.allocator.destroy(bottom);
        }
        return Type.initSum(self.allocator, inner_type, bottom);
    }

    fn inferInr(self: *Self, ctx: *TypeContext, term: *Term) InferError!*Type {
        if (term.sub_terms.items.len == 0) {
            return TypeTheoryError.InvalidTypeConstruction;
        }
        const inner_type = try self.inferType(ctx, term.sub_terms.items[0]);
        errdefer {
            inner_type.deinit();
            self.allocator.destroy(inner_type);
        }
        const bottom = try Type.initBottom(self.allocator);
        errdefer {
            bottom.deinit();
            self.allocator.destroy(bottom);
        }
        return Type.initSum(self.allocator, bottom, inner_type);
    }

    fn inferSucc(self: *Self, ctx: *TypeContext, term: *Term) InferError!*Type {
        if (term.sub_terms.items.len == 0) {
            return TypeTheoryError.InvalidTypeConstruction;
        }
        const n_type = try self.inferType(ctx, term.sub_terms.items[0]);
        defer {
            n_type.deinit();
            self.allocator.destroy(n_type);
        }
        if (n_type.kind != .NAT) {
            return TypeTheoryError.TypeMismatch;
        }
        return Type.initNat(self.allocator);
    }

    fn inferRefl(self: *Self, ctx: *TypeContext, term: *Term) InferError!*Type {
        if (term.sub_terms.items.len == 0) {
            return TypeTheoryError.InvalidTypeConstruction;
        }
        const witness_type = try self.inferType(ctx, term.sub_terms.items[0]);
        defer {
            witness_type.deinit();
            self.allocator.destroy(witness_type);
        }
        const id_type = try Type.init(self.allocator, .IDENTITY);
        errdefer {
            id_type.deinit();
            self.allocator.destroy(id_type);
        }
        try id_type.parameters.append(try witness_type.clone(self.allocator));
        return id_type;
    }

    fn inferAnnotation(self: *Self, term: *Term) InferError!*Type {
        if (term.annotation_type) |ann| {
            return ann.clone(self.allocator);
        }
        return TypeTheoryError.InvalidTypeConstruction;
    }

    pub fn subtype(self: *Self, sub: *Type, super: *Type) bool {
        if (sub.equals(super)) return true;
        if (super.kind == .TOP) return true;
        if (sub.kind == .BOTTOM) return true;
        if (sub.kind == .NAT and super.kind == .INT) return true;
        if (sub.kind == .INT and super.kind == .REAL) return true;
        if (sub.kind == .REAL and super.kind == .COMPLEX) return true;
        if (sub.kind == .FUNCTION and super.kind == .FUNCTION) {
            if (sub.left_type != null and super.left_type != null and sub.right_type != null and super.right_type != null) {
                return self.subtype(super.left_type.?, sub.left_type.?) and self.subtype(sub.right_type.?, super.right_type.?);
            }
        }
        if (sub.kind == .TUPLE and super.kind == .TUPLE) {
            if (sub.parameters.items.len != super.parameters.items.len) return false;
            for (sub.parameters.items, 0..) |s_param, i| {
                if (!self.subtype(s_param, super.parameters.items[i])) return false;
            }
            return true;
        }
        if (sub.kind == .UNIVERSE and super.kind == .UNIVERSE) {
            return sub.universe_level <= super.universe_level;
        }
        return false;
    }

    pub fn unifyTypes(self: *Self, t1: *Type, t2: *Type) !*Type {
        self.unification_count += 1;
        if (t1.equals(t2)) {
            return t1.clone(self.allocator);
        }
        if (t1.kind == .VARIABLE) {
            return t2.clone(self.allocator);
        }
        if (t2.kind == .VARIABLE) {
            return t1.clone(self.allocator);
        }
        if (t1.kind == .TOP) return t2.clone(self.allocator);
        if (t2.kind == .TOP) return t1.clone(self.allocator);
        if (t1.kind == .BOTTOM) return t2.clone(self.allocator);
        if (t2.kind == .BOTTOM) return t1.clone(self.allocator);
        if (t1.kind == t2.kind) {
            switch (t1.kind) {
                .FUNCTION => {
                    if (t1.left_type != null and t2.left_type != null and t1.right_type != null and t2.right_type != null) {
                        const unified_domain = try self.unifyTypes(t1.left_type.?, t2.left_type.?);
                        errdefer {
                            unified_domain.deinit();
                            self.allocator.destroy(unified_domain);
                        }
                        const unified_codomain = try self.unifyTypes(t1.right_type.?, t2.right_type.?);
                        errdefer {
                            unified_codomain.deinit();
                            self.allocator.destroy(unified_codomain);
                        }
                        return Type.initFunction(self.allocator, unified_domain, unified_codomain);
                    }
                },
                .TUPLE => {
                    if (t1.parameters.items.len == t2.parameters.items.len) {
                        var unified_params = ArrayList(*Type).init(self.allocator);
                        errdefer {
                            for (unified_params.items) |p| {
                                p.deinit();
                                self.allocator.destroy(p);
                            }
                            unified_params.deinit();
                        }
                        for (t1.parameters.items, 0..) |p1, i| {
                            const unified = try self.unifyTypes(p1, t2.parameters.items[i]);
                            try unified_params.append(unified);
                        }
                        const result = try Type.init(self.allocator, .TUPLE);
                        result.parameters.deinit();
                        result.parameters = unified_params;
                        return result;
                    }
                },
                .ARRAY => {
                    if (t1.parameters.items.len > 0 and t2.parameters.items.len > 0) {
                        const unified_elem = try self.unifyTypes(t1.parameters.items[0], t2.parameters.items[0]);
                        errdefer {
                            unified_elem.deinit();
                            self.allocator.destroy(unified_elem);
                        }
                        return Type.initArray(self.allocator, unified_elem);
                    }
                },
                .UNIVERSE => {
                    const max_level = @max(t1.universe_level, t2.universe_level);
                    return Type.initUniverse(self.allocator, max_level);
                },
                else => {},
            }
        }
        if (self.subtype(t1, t2)) {
            return t2.clone(self.allocator);
        }
        if (self.subtype(t2, t1)) {
            return t1.clone(self.allocator);
        }
        return TypeTheoryError.UnificationFailure;
    }

    pub fn getStatistics(self: *const Self) TypeCheckerStatistics {
        return TypeCheckerStatistics{
            .inference_count = self.inference_count,
            .check_count = self.check_count,
            .unification_count = self.unification_count,
        };
    }
};

pub const TypeCheckerStatistics = struct {
    inference_count: u64,
    check_count: u64,
    unification_count: u64,
};

pub const PropositionAsType = struct {
    connective: LogicalConnective,
    sub_propositions: ArrayList(*PropositionAsType),
    bound_variable: ?[]const u8,
    predicate_type: ?*Type,
    corresponding_type: ?*Type,
    allocator: Allocator,

    pub const LogicalConnective = enum(u8) {
        CONJUNCTION = 0,
        DISJUNCTION = 1,
        IMPLICATION = 2,
        NEGATION = 3,
        UNIVERSAL = 4,
        EXISTENTIAL = 5,
        TRUE = 6,
        FALSE = 7,
        BICONDITIONAL = 8,
    };

    const Self = @This();

    pub fn init(allocator: Allocator, connective: LogicalConnective) !*Self {
        const p = try allocator.create(Self);
        errdefer allocator.destroy(p);
        p.* = Self{
            .connective = connective,
            .sub_propositions = ArrayList(*PropositionAsType).init(allocator),
            .bound_variable = null,
            .predicate_type = null,
            .corresponding_type = null,
            .allocator = allocator,
        };
        return p;
    }

    pub fn initTrue(allocator: Allocator) !*Self {
        const p = try PropositionAsType.init(allocator, .TRUE);
        errdefer {
            p.deinit();
            allocator.destroy(p);
        }
        p.corresponding_type = try Type.initUnit(allocator);
        return p;
    }

    pub fn initFalse(allocator: Allocator) !*Self {
        const p = try PropositionAsType.init(allocator, .FALSE);
        errdefer {
            p.deinit();
            allocator.destroy(p);
        }
        p.corresponding_type = try Type.initBottom(allocator);
        return p;
    }

    pub fn initConjunction(allocator: Allocator, left: *PropositionAsType, right: *PropositionAsType) !*Self {
        const p = try PropositionAsType.init(allocator, .CONJUNCTION);
        errdefer {
            p.deinit();
            allocator.destroy(p);
        }
        try p.sub_propositions.append(left);
        try p.sub_propositions.append(right);
        if (left.corresponding_type != null and right.corresponding_type != null) {
            const lc = try left.corresponding_type.?.clone(allocator);
            errdefer {
                lc.deinit();
                allocator.destroy(lc);
            }
            const rc = try right.corresponding_type.?.clone(allocator);
            errdefer {
                rc.deinit();
                allocator.destroy(rc);
            }
            p.corresponding_type = try Type.initTuple(allocator, &[_]*Type{ lc, rc });
        }
        return p;
    }

    pub fn initDisjunction(allocator: Allocator, left: *PropositionAsType, right: *PropositionAsType) !*Self {
        const p = try PropositionAsType.init(allocator, .DISJUNCTION);
        errdefer {
            p.deinit();
            allocator.destroy(p);
        }
        try p.sub_propositions.append(left);
        try p.sub_propositions.append(right);
        if (left.corresponding_type != null and right.corresponding_type != null) {
            const lc = try left.corresponding_type.?.clone(allocator);
            errdefer {
                lc.deinit();
                allocator.destroy(lc);
            }
            const rc = try right.corresponding_type.?.clone(allocator);
            errdefer {
                rc.deinit();
                allocator.destroy(rc);
            }
            p.corresponding_type = try Type.initSum(allocator, lc, rc);
        }
        return p;
    }

    pub fn initImplication(allocator: Allocator, antecedent: *PropositionAsType, consequent: *PropositionAsType) !*Self {
        const p = try PropositionAsType.init(allocator, .IMPLICATION);
        errdefer {
            p.deinit();
            allocator.destroy(p);
        }
        try p.sub_propositions.append(antecedent);
        try p.sub_propositions.append(consequent);
        if (antecedent.corresponding_type != null and consequent.corresponding_type != null) {
            const ac = try antecedent.corresponding_type.?.clone(allocator);
            errdefer {
                ac.deinit();
                allocator.destroy(ac);
            }
            const cc = try consequent.corresponding_type.?.clone(allocator);
            errdefer {
                cc.deinit();
                allocator.destroy(cc);
            }
            p.corresponding_type = try Type.initFunction(allocator, ac, cc);
        }
        return p;
    }

    pub fn initNegation(allocator: Allocator, inner: *PropositionAsType) !*Self {
        const p = try PropositionAsType.init(allocator, .NEGATION);
        errdefer {
            p.deinit();
            allocator.destroy(p);
        }
        try p.sub_propositions.append(inner);
        if (inner.corresponding_type) |inner_type| {
            const ic = try inner_type.clone(allocator);
            errdefer {
                ic.deinit();
                allocator.destroy(ic);
            }
            const bottom = try Type.initBottom(allocator);
            errdefer {
                bottom.deinit();
                allocator.destroy(bottom);
            }
            p.corresponding_type = try Type.initFunction(allocator, ic, bottom);
        }
        return p;
    }

    pub fn initUniversal(allocator: Allocator, variable: []const u8, domain: *Type, body: *PropositionAsType) !*Self {
        const p = try PropositionAsType.init(allocator, .UNIVERSAL);
        errdefer {
            p.deinit();
            allocator.destroy(p);
        }
        p.bound_variable = try allocator.dupe(u8, variable);
        p.predicate_type = try domain.clone(allocator);
        try p.sub_propositions.append(body);
        if (body.corresponding_type) |body_type| {
            const pi_type = try Type.init(allocator, .DEPENDENT_FUNCTION);
            errdefer {
                pi_type.deinit();
                allocator.destroy(pi_type);
            }
            pi_type.bound_variable = try allocator.dupe(u8, variable);
            pi_type.left_type = try domain.clone(allocator);
            pi_type.body_type = try body_type.clone(allocator);
            p.corresponding_type = pi_type;
        }
        return p;
    }

    pub fn initExistential(allocator: Allocator, variable: []const u8, domain: *Type, body: *PropositionAsType) !*Self {
        const p = try PropositionAsType.init(allocator, .EXISTENTIAL);
        errdefer {
            p.deinit();
            allocator.destroy(p);
        }
        p.bound_variable = try allocator.dupe(u8, variable);
        p.predicate_type = try domain.clone(allocator);
        try p.sub_propositions.append(body);
        if (body.corresponding_type) |body_type| {
            const sigma_type = try Type.init(allocator, .DEPENDENT_PAIR);
            errdefer {
                sigma_type.deinit();
                allocator.destroy(sigma_type);
            }
            sigma_type.bound_variable = try allocator.dupe(u8, variable);
            sigma_type.left_type = try domain.clone(allocator);
            sigma_type.body_type = try body_type.clone(allocator);
            p.corresponding_type = sigma_type;
        }
        return p;
    }

    pub fn deinit(self: *Self) void {
        for (self.sub_propositions.items) |sub| {
            sub.deinit();
            self.allocator.destroy(sub);
        }
        self.sub_propositions.deinit();
        if (self.bound_variable) |bv| {
            self.allocator.free(bv);
        }
        if (self.predicate_type) |pt| {
            pt.deinit();
            self.allocator.destroy(pt);
        }
        if (self.corresponding_type) |ct| {
            ct.deinit();
            self.allocator.destroy(ct);
        }
    }

    pub fn toType(self: *const Self, allocator: Allocator) !*Type {
        if (self.corresponding_type) |ct| {
            return ct.clone(allocator);
        }
        return Type.initUnit(allocator);
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const p = try PropositionAsType.init(allocator, self.connective);
        errdefer {
            p.deinit();
            allocator.destroy(p);
        }
        for (self.sub_propositions.items) |sub| {
            try p.sub_propositions.append(try sub.clone(allocator));
        }
        p.bound_variable = if (self.bound_variable) |bv| try allocator.dupe(u8, bv) else null;
        p.predicate_type = if (self.predicate_type) |pt| try pt.clone(allocator) else null;
        p.corresponding_type = if (self.corresponding_type) |ct| try ct.clone(allocator) else null;
        return p;
    }
};

pub const ProofTerm = struct {
    kind: ProofKind,
    proposition: *PropositionAsType,
    sub_proofs: ArrayList(*ProofTerm),
    witness_term: ?*Term,
    is_valid: bool,
    allocator: Allocator,

    pub const ProofKind = enum(u8) {
        ASSUMPTION = 0,
        INTRO_CONJUNCTION = 1,
        ELIM_CONJUNCTION_LEFT = 2,
        ELIM_CONJUNCTION_RIGHT = 3,
        INTRO_DISJUNCTION_LEFT = 4,
        INTRO_DISJUNCTION_RIGHT = 5,
        ELIM_DISJUNCTION = 6,
        INTRO_IMPLICATION = 7,
        ELIM_IMPLICATION = 8,
        INTRO_UNIVERSAL = 9,
        ELIM_UNIVERSAL = 10,
        INTRO_EXISTENTIAL = 11,
        ELIM_EXISTENTIAL = 12,
        INTRO_NEGATION = 13,
        ELIM_NEGATION = 14,
        REFLEXIVITY = 15,
        SYMMETRY = 16,
        TRANSITIVITY = 17,
    };

    const Self = @This();

    pub fn init(allocator: Allocator, kind: ProofKind, proposition: *PropositionAsType) !*Self {
        const pt = try allocator.create(Self);
        errdefer allocator.destroy(pt);
        pt.* = Self{
            .kind = kind,
            .proposition = proposition,
            .sub_proofs = ArrayList(*ProofTerm).init(allocator),
            .witness_term = null,
            .is_valid = false,
            .allocator = allocator,
        };
        return pt;
    }

    pub fn deinit(self: *Self) void {
        self.proposition.deinit();
        self.allocator.destroy(self.proposition);
        for (self.sub_proofs.items) |sub| {
            sub.deinit();
            self.allocator.destroy(sub);
        }
        self.sub_proofs.deinit();
        if (self.witness_term) |w| {
            w.deinit();
            self.allocator.destroy(w);
        }
    }

    pub fn validate(self: *Self) bool {
        self.is_valid = switch (self.kind) {
            .ASSUMPTION => true,
            .INTRO_CONJUNCTION => self.validateConjunctionIntro(),
            .ELIM_CONJUNCTION_LEFT, .ELIM_CONJUNCTION_RIGHT => self.validateConjunctionElim(),
            .INTRO_IMPLICATION => self.validateImplicationIntro(),
            .ELIM_IMPLICATION => self.validateImplicationElim(),
            .INTRO_UNIVERSAL => self.validateUniversalIntro(),
            .ELIM_UNIVERSAL => self.validateUniversalElim(),
            .REFLEXIVITY => true,
            else => self.sub_proofs.items.len > 0,
        };
        return self.is_valid;
    }

    fn validateConjunctionIntro(self: *const Self) bool {
        if (self.sub_proofs.items.len < 2) return false;
        return self.sub_proofs.items[0].is_valid and self.sub_proofs.items[1].is_valid;
    }

    fn validateConjunctionElim(self: *const Self) bool {
        if (self.sub_proofs.items.len < 1) return false;
        const premise = self.sub_proofs.items[0];
        return premise.is_valid and premise.proposition.connective == .CONJUNCTION;
    }

    fn validateImplicationIntro(self: *const Self) bool {
        if (self.sub_proofs.items.len < 1) return false;
        return self.sub_proofs.items[0].is_valid;
    }

    fn validateImplicationElim(self: *const Self) bool {
        if (self.sub_proofs.items.len < 2) return false;
        const impl_proof = self.sub_proofs.items[0];
        return impl_proof.is_valid and impl_proof.proposition.connective == .IMPLICATION;
    }

    fn validateUniversalIntro(self: *const Self) bool {
        if (self.sub_proofs.items.len < 1) return false;
        return self.sub_proofs.items[0].is_valid;
    }

    fn validateUniversalElim(self: *const Self) bool {
        if (self.sub_proofs.items.len < 1) return false;
        const univ_proof = self.sub_proofs.items[0];
        return univ_proof.is_valid and univ_proof.proposition.connective == .UNIVERSAL;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const pt = try ProofTerm.init(allocator, self.kind, try self.proposition.clone(allocator));
        errdefer {
            pt.deinit();
            allocator.destroy(pt);
        }
        for (self.sub_proofs.items) |sub| {
            try pt.sub_proofs.append(try sub.clone(allocator));
        }
        pt.witness_term = if (self.witness_term) |w| try w.clone(allocator) else null;
        pt.is_valid = self.is_valid;
        return pt;
    }
};

pub const CategoryObject = struct {
    id: u64,
    name: []const u8,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, id: u64, name: []const u8) !*Self {
        const obj = try allocator.create(Self);
        errdefer allocator.destroy(obj);
        obj.* = Self{
            .id = id,
            .name = try allocator.dupe(u8, name),
            .allocator = allocator,
        };
        return obj;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        return CategoryObject.init(allocator, self.id, self.name);
    }

    pub fn equals(self: *const Self, other: *const Self) bool {
        return self.id == other.id and std.mem.eql(u8, self.name, other.name);
    }
};

pub const Morphism = struct {
    id: u64,
    name: []const u8,
    source: *CategoryObject,
    target: *CategoryObject,
    is_identity: bool,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, id: u64, name: []const u8, source: *CategoryObject, target: *CategoryObject) !*Self {
        const m = try allocator.create(Self);
        errdefer allocator.destroy(m);
        m.* = Self{
            .id = id,
            .name = try allocator.dupe(u8, name),
            .source = source,
            .target = target,
            .is_identity = false,
            .allocator = allocator,
        };
        return m;
    }

    pub fn initIdentity(allocator: Allocator, id: u64, obj: *CategoryObject) !*Self {
        const m = try Morphism.init(allocator, id, "id", obj, obj);
        m.is_identity = true;
        return m;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const source_clone = try self.source.clone(allocator);
        errdefer {
            source_clone.deinit();
            allocator.destroy(source_clone);
        }
        const target_clone = try self.target.clone(allocator);
        errdefer {
            target_clone.deinit();
            allocator.destroy(target_clone);
        }
        const m = try Morphism.init(allocator, self.id, self.name, source_clone, target_clone);
        m.is_identity = self.is_identity;
        return m;
    }

    pub fn canCompose(self: *const Self, other: *const Self) bool {
        return self.target.equals(other.source);
    }
};

pub const Category = struct {
    name: []const u8,
    objects: ArrayList(*CategoryObject),
    morphisms: ArrayList(*Morphism),
    compositions: AutoHashMap(u128, *Morphism),
    next_object_id: u64,
    next_morphism_id: u64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, name: []const u8) !*Self {
        const cat = try allocator.create(Self);
        errdefer allocator.destroy(cat);
        cat.* = Self{
            .name = try allocator.dupe(u8, name),
            .objects = ArrayList(*CategoryObject).init(allocator),
            .morphisms = ArrayList(*Morphism).init(allocator),
            .compositions = AutoHashMap(u128, *Morphism).init(allocator),
            .next_object_id = 1,
            .next_morphism_id = 1,
            .allocator = allocator,
        };
        return cat;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
        for (self.objects.items) |obj| {
            obj.deinit();
            self.allocator.destroy(obj);
        }
        self.objects.deinit();
        for (self.morphisms.items) |m| {
            m.deinit();
            self.allocator.destroy(m);
        }
        self.morphisms.deinit();
        self.compositions.deinit();
    }

    pub fn addObject(self: *Self, name: []const u8) !*CategoryObject {
        const obj = try CategoryObject.init(self.allocator, self.next_object_id, name);
        errdefer {
            obj.deinit();
            self.allocator.destroy(obj);
        }
        try self.objects.append(obj);
        self.next_object_id += 1;
        const identity = try Morphism.initIdentity(self.allocator, self.next_morphism_id, obj);
        errdefer {
            identity.deinit();
            self.allocator.destroy(identity);
        }
        try self.morphisms.append(identity);
        self.next_morphism_id += 1;
        return obj;
    }

    pub fn addMorphism(self: *Self, name: []const u8, source: *CategoryObject, target: *CategoryObject) !*Morphism {
        const m = try Morphism.init(self.allocator, self.next_morphism_id, name, source, target);
        errdefer {
            m.deinit();
            self.allocator.destroy(m);
        }
        try self.morphisms.append(m);
        self.next_morphism_id += 1;
        return m;
    }

    pub fn compose(self: *Self, f: *Morphism, g: *Morphism) !*Morphism {
        if (!f.canCompose(g)) {
            return TypeTheoryError.CategoryLawViolation;
        }
        const comp_key: u128 = (@as(u128, f.id) << 64) | g.id;
        if (self.compositions.get(comp_key)) |cached| {
            return cached;
        }
        var name_buf: [256]u8 = undefined;
        const composed_name = std.fmt.bufPrint(&name_buf, "{s}∘{s}", .{ g.name, f.name }) catch "composed";
        const composed = try self.addMorphism(composed_name, f.source, g.target);
        try self.compositions.put(comp_key, composed);
        return composed;
    }

    pub fn getIdentity(self: *const Self, obj: *CategoryObject) ?*Morphism {
        for (self.morphisms.items) |m| {
            if (m.is_identity and m.source.equals(obj)) {
                return m;
            }
        }
        return null;
    }

    pub fn verifyAssociativity(self: *Self, f: *Morphism, g: *Morphism, h: *Morphism) !bool {
        if (!f.canCompose(g) or !g.canCompose(h)) {
            return false;
        }
        const fg = try self.compose(f, g);
        const gh = try self.compose(g, h);
        const fg_h = try self.compose(fg, h);
        const f_gh = try self.compose(f, gh);
        return fg_h.equals(f_gh);
    }

    pub fn verifyIdentityLaw(self: *const Self, f: *Morphism) bool {
        const source_id = self.getIdentity(f.source);
        const target_id = self.getIdentity(f.target);
        return source_id != null and target_id != null;
    }

    pub fn objectCount(self: *const Self) usize {
        return self.objects.items.len;
    }

    pub fn morphismCount(self: *const Self) usize {
        return self.morphisms.items.len;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const cat = try Category.init(allocator, self.name);
        errdefer {
            cat.deinit();
            allocator.destroy(cat);
        }
        for (self.objects.items) |obj| {
            try cat.objects.append(try obj.clone(allocator));
        }
        for (self.morphisms.items) |m| {
            try cat.morphisms.append(try m.clone(allocator));
        }
        cat.next_object_id = self.next_object_id;
        cat.next_morphism_id = self.next_morphism_id;
        return cat;
    }
};

pub const Functor = struct {
    name: []const u8,
    source_category: *Category,
    target_category: *Category,
    object_mapping: AutoHashMap(u64, *CategoryObject),
    morphism_mapping: AutoHashMap(u64, *Morphism),
    owns_mapped: bool,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, name: []const u8, source: *Category, target: *Category) !*Self {
        const f = try allocator.create(Self);
        errdefer allocator.destroy(f);
        f.* = Self{
            .name = try allocator.dupe(u8, name),
            .source_category = source,
            .target_category = target,
            .object_mapping = AutoHashMap(u64, *CategoryObject).init(allocator),
            .morphism_mapping = AutoHashMap(u64, *Morphism).init(allocator),
            .owns_mapped = false,
            .allocator = allocator,
        };
        return f;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
        if (self.owns_mapped) {
            var obj_iter = self.object_mapping.iterator();
            while (obj_iter.next()) |entry| {
                entry.value_ptr.*.deinit();
                self.allocator.destroy(entry.value_ptr.*);
            }
            var morph_iter = self.morphism_mapping.iterator();
            while (morph_iter.next()) |entry| {
                entry.value_ptr.*.deinit();
                self.allocator.destroy(entry.value_ptr.*);
            }
        }
        self.object_mapping.deinit();
        self.morphism_mapping.deinit();
    }

    pub fn mapObject(self: *Self, source_obj: *CategoryObject, target_obj: *CategoryObject) !void {
        try self.object_mapping.put(source_obj.id, target_obj);
    }

    pub fn mapMorphism(self: *Self, source_morph: *Morphism, target_morph: *Morphism) !void {
        try self.morphism_mapping.put(source_morph.id, target_morph);
    }

    pub fn applyToObject(self: *const Self, obj: *CategoryObject) ?*CategoryObject {
        return self.object_mapping.get(obj.id);
    }

    pub fn applyToMorphism(self: *const Self, m: *Morphism) ?*Morphism {
        return self.morphism_mapping.get(m.id);
    }

    pub fn preservesIdentity(self: *const Self, obj: *CategoryObject) bool {
        const source_id = self.source_category.getIdentity(obj);
        if (source_id == null) return false;
        const mapped_obj = self.applyToObject(obj);
        if (mapped_obj == null) return false;
        const target_id = self.target_category.getIdentity(mapped_obj.?);
        if (target_id == null) return false;
        const mapped_id = self.applyToMorphism(source_id.?);
        if (mapped_id == null) return false;
        return mapped_id.?.equals(target_id.?);
    }

    pub fn preservesComposition(self: *Self, f: *Morphism, g: *Morphism) !bool {
        if (!f.canCompose(g)) return false;
        const fg = try self.source_category.compose(f, g);
        const mapped_f = self.applyToMorphism(f);
        const mapped_g = self.applyToMorphism(g);
        const mapped_fg = self.applyToMorphism(fg);
        if (mapped_f == null or mapped_g == null or mapped_fg == null) return false;
        const composed_mapped = try self.target_category.compose(mapped_f.?, mapped_g.?);
        return composed_mapped.equals(mapped_fg.?);
    }

    pub fn verifyFunctorLaws(self: *Self) !bool {
        for (self.source_category.objects.items) |obj| {
            if (!self.preservesIdentity(obj)) {
                return false;
            }
        }
        const snapshot = try self.allocator.alloc(*Morphism, self.source_category.morphisms.items.len);
        defer self.allocator.free(snapshot);
        for (self.source_category.morphisms.items, 0..) |m, i| {
            snapshot[i] = m;
        }
        for (snapshot) |f| {
            for (snapshot) |g| {
                if (f.canCompose(g)) {
                    if (!try self.preservesComposition(f, g)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const f = try Functor.init(allocator, self.name, self.source_category, self.target_category);
        errdefer {
            f.deinit();
            allocator.destroy(f);
        }
        f.owns_mapped = true;
        var obj_iter = self.object_mapping.iterator();
        while (obj_iter.next()) |entry| {
            try f.object_mapping.put(entry.key_ptr.*, try entry.value_ptr.*.clone(allocator));
        }
        var morph_iter = self.morphism_mapping.iterator();
        while (morph_iter.next()) |entry| {
            try f.morphism_mapping.put(entry.key_ptr.*, try entry.value_ptr.*.clone(allocator));
        }
        return f;
    }
};

pub const NaturalTransformation = struct {
    name: []const u8,
    source_functor: *Functor,
    target_functor: *Functor,
    components: AutoHashMap(u64, *Morphism),
    owns_components: bool,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, name: []const u8, source: *Functor, target: *Functor) !*Self {
        const nt = try allocator.create(Self);
        errdefer allocator.destroy(nt);
        nt.* = Self{
            .name = try allocator.dupe(u8, name),
            .source_functor = source,
            .target_functor = target,
            .components = AutoHashMap(u64, *Morphism).init(allocator),
            .owns_components = false,
            .allocator = allocator,
        };
        return nt;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
        if (self.owns_components) {
            var iter = self.components.iterator();
            while (iter.next()) |entry| {
                entry.value_ptr.*.deinit();
                self.allocator.destroy(entry.value_ptr.*);
            }
        }
        self.components.deinit();
    }

    pub fn setComponent(self: *Self, obj: *CategoryObject, component: *Morphism) !void {
        try self.components.put(obj.id, component);
    }

    pub fn getComponent(self: *const Self, obj: *CategoryObject) ?*Morphism {
        return self.components.get(obj.id);
    }

    pub fn verifyNaturality(self: *Self, f: *Morphism) !bool {
        const source_comp = self.getComponent(f.source);
        const target_comp = self.getComponent(f.target);
        if (source_comp == null or target_comp == null) return false;
        const mapped_f_source = self.source_functor.applyToMorphism(f);
        const mapped_f_target = self.target_functor.applyToMorphism(f);
        if (mapped_f_source == null or mapped_f_target == null) return false;
        const left_path = try self.target_functor.target_category.compose(source_comp.?, mapped_f_target.?);
        const right_path = try self.target_functor.target_category.compose(mapped_f_source.?, target_comp.?);
        return left_path.equals(right_path);
    }

    pub fn verifyAllNaturality(self: *Self) !bool {
        for (self.source_functor.source_category.morphisms.items) |f| {
            if (!try self.verifyNaturality(f)) {
                return false;
            }
        }
        return true;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const nt = try NaturalTransformation.init(allocator, self.name, self.source_functor, self.target_functor);
        errdefer {
            nt.deinit();
            allocator.destroy(nt);
        }
        nt.owns_components = true;
        var iter = self.components.iterator();
        while (iter.next()) |entry| {
            try nt.components.put(entry.key_ptr.*, try entry.value_ptr.*.clone(allocator));
        }
        return nt;
    }
};

pub const Monad = struct {
    name: []const u8,
    endofunctor: *Functor,
    unit: *NaturalTransformation,
    multiplication: *NaturalTransformation,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, name: []const u8, t: *Functor, eta: *NaturalTransformation, mu: *NaturalTransformation) !*Self {
        const m = try allocator.create(Self);
        errdefer allocator.destroy(m);
        m.* = Self{
            .name = try allocator.dupe(u8, name),
            .endofunctor = t,
            .unit = eta,
            .multiplication = mu,
            .allocator = allocator,
        };
        return m;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
    }

    pub fn verifyLeftUnitLaw(self: *Self, obj: *CategoryObject) !bool {
        const eta_obj = self.unit.getComponent(obj);
        const mu_obj = self.multiplication.getComponent(obj);
        if (eta_obj == null or mu_obj == null) return false;
        const t_obj = self.endofunctor.applyToObject(obj);
        if (t_obj == null) return false;
        const t_eta = self.endofunctor.applyToMorphism(eta_obj.?);
        if (t_eta == null) return false;
        const left = try self.endofunctor.target_category.compose(t_eta.?, mu_obj.?);
        const id_t = self.endofunctor.target_category.getIdentity(t_obj.?);
        if (id_t == null) return false;
        return left.equals(id_t.?);
    }

    pub fn verifyRightUnitLaw(self: *Self, obj: *CategoryObject) !bool {
        const eta_t_obj = self.unit.getComponent(obj);
        const mu_obj = self.multiplication.getComponent(obj);
        if (eta_t_obj == null or mu_obj == null) return false;
        const right = try self.endofunctor.target_category.compose(eta_t_obj.?, mu_obj.?);
        const t_obj = self.endofunctor.applyToObject(obj);
        if (t_obj == null) return false;
        const id_t = self.endofunctor.target_category.getIdentity(t_obj.?);
        if (id_t == null) return false;
        return right.equals(id_t.?);
    }

    pub fn verifyAssociativityLaw(self: *Self, obj: *CategoryObject) !bool {
        const mu_obj = self.multiplication.getComponent(obj);
        if (mu_obj == null) return false;
        const t_obj = self.endofunctor.applyToObject(obj);
        if (t_obj == null) return false;
        const mu_t_obj = self.multiplication.getComponent(t_obj.?);
        if (mu_t_obj == null) return false;
        const t_mu = self.endofunctor.applyToMorphism(mu_obj.?);
        if (t_mu == null) return false;
        const left = try self.endofunctor.target_category.compose(mu_t_obj.?, mu_obj.?);
        const right = try self.endofunctor.target_category.compose(t_mu.?, mu_obj.?);
        return left.equals(right);
    }

    pub fn verifyMonadLaws(self: *Self) !bool {
        for (self.endofunctor.source_category.objects.items) |obj| {
            if (!try self.verifyLeftUnitLaw(obj)) return false;
            if (!try self.verifyRightUnitLaw(obj)) return false;
            if (!try self.verifyAssociativityLaw(obj)) return false;
        }
        return true;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        return Monad.init(allocator, self.name, try self.endofunctor.clone(allocator), try self.unit.clone(allocator), try self.multiplication.clone(allocator));
    }
};

pub const CartesianClosedCategory = struct {
    base_category: *Category,
    terminal_object: ?*CategoryObject,
    product_functor: ?*Functor,
    exponential_functor: ?*Functor,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, base: *Category) !*Self {
        const ccc = try allocator.create(Self);
        errdefer allocator.destroy(ccc);
        ccc.* = Self{
            .base_category = base,
            .terminal_object = null,
            .product_functor = null,
            .exponential_functor = null,
            .allocator = allocator,
        };
        return ccc;
    }

    pub fn deinit(self: *Self) void {
        if (self.product_functor) |pf| {
            pf.deinit();
            self.allocator.destroy(pf);
        }
        if (self.exponential_functor) |ef| {
            ef.deinit();
            self.allocator.destroy(ef);
        }
    }

    pub fn setTerminal(self: *Self, obj: *CategoryObject) void {
        self.terminal_object = obj;
    }

    pub fn hasProducts(self: *const Self) bool {
        return self.product_functor != null;
    }

    pub fn hasExponentials(self: *const Self) bool {
        return self.exponential_functor != null;
    }

    pub fn isCartesianClosed(self: *const Self) bool {
        return self.terminal_object != null and self.hasProducts() and self.hasExponentials();
    }

    pub fn modelLambdaCalculus(self: *const Self) bool {
        return self.isCartesianClosed();
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const ccc = try CartesianClosedCategory.init(allocator, try self.base_category.clone(allocator));
        errdefer {
            ccc.deinit();
            allocator.destroy(ccc);
        }
        ccc.terminal_object = self.terminal_object;
        ccc.product_functor = if (self.product_functor) |pf| try pf.clone(allocator) else null;
        ccc.exponential_functor = if (self.exponential_functor) |ef| try ef.clone(allocator) else null;
        return ccc;
    }
};

pub const LinearityMode = enum(u8) {
    LINEAR = 0,
    AFFINE = 1,
    RELEVANT = 2,
    UNRESTRICTED = 3,

    const Self = @This();

    pub fn toString(self: Self) []const u8 {
        return switch (self) {
            .LINEAR => "linear",
            .AFFINE => "affine",
            .RELEVANT => "relevant",
            .UNRESTRICTED => "unrestricted",
        };
    }

    pub fn canWeakenTo(self: Self, target: Self) bool {
        return switch (self) {
            .UNRESTRICTED => true,
            .AFFINE => target == .AFFINE or target == .UNRESTRICTED,
            .RELEVANT => target == .RELEVANT or target == .UNRESTRICTED,
            .LINEAR => target == .LINEAR,
        };
    }

    pub fn join(self: Self, other: Self) Self {
        if (self == .LINEAR or other == .LINEAR) return .LINEAR;
        if (self == .AFFINE and other == .RELEVANT) return .LINEAR;
        if (self == .RELEVANT and other == .AFFINE) return .LINEAR;
        if (self == .AFFINE or other == .AFFINE) return .AFFINE;
        if (self == .RELEVANT or other == .RELEVANT) return .RELEVANT;
        return .UNRESTRICTED;
    }
};

pub const LinearType = struct {
    base_type: *Type,
    linearity: LinearityMode,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, base_type: *Type, linearity: LinearityMode) !*Self {
        const lt = try allocator.create(Self);
        errdefer allocator.destroy(lt);
        lt.* = Self{
            .base_type = base_type,
            .linearity = linearity,
            .allocator = allocator,
        };
        return lt;
    }

    pub fn initLinear(allocator: Allocator, base_type: *Type) !*Self {
        return LinearType.init(allocator, base_type, .LINEAR);
    }

    pub fn initAffine(allocator: Allocator, base_type: *Type) !*Self {
        return LinearType.init(allocator, base_type, .AFFINE);
    }

    pub fn initRelevant(allocator: Allocator, base_type: *Type) !*Self {
        return LinearType.init(allocator, base_type, .RELEVANT);
    }

    pub fn initUnrestricted(allocator: Allocator, base_type: *Type) !*Self {
        return LinearType.init(allocator, base_type, .UNRESTRICTED);
    }

    pub fn deinit(self: *Self) void {
        self.base_type.deinit();
        self.allocator.destroy(self.base_type);
    }

    pub fn mustUseExactlyOnce(self: *const Self) bool {
        return self.linearity == .LINEAR;
    }

    pub fn canDrop(self: *const Self) bool {
        return self.linearity == .AFFINE or self.linearity == .UNRESTRICTED;
    }

    pub fn canDuplicate(self: *const Self) bool {
        return self.linearity == .RELEVANT or self.linearity == .UNRESTRICTED;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        return LinearType.init(allocator, try self.base_type.clone(allocator), self.linearity);
    }
};

pub const ResourceUsage = struct {
    variable_name: []const u8,
    usage_count: u32,
    linear_type: *LinearType,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, name: []const u8, linear_type: *LinearType) !*Self {
        const ru = try allocator.create(Self);
        errdefer allocator.destroy(ru);
        ru.* = Self{
            .variable_name = try allocator.dupe(u8, name),
            .usage_count = 0,
            .linear_type = linear_type,
            .allocator = allocator,
        };
        return ru;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.variable_name);
        self.linear_type.deinit();
        self.allocator.destroy(self.linear_type);
    }

    pub fn use(self: *Self) void {
        self.usage_count += 1;
    }

    pub fn isValid(self: *const Self) bool {
        return switch (self.linear_type.linearity) {
            .LINEAR => self.usage_count == 1,
            .AFFINE => self.usage_count <= 1,
            .RELEVANT => self.usage_count >= 1,
            .UNRESTRICTED => true,
        };
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const ru = try ResourceUsage.init(allocator, self.variable_name, try self.linear_type.clone(allocator));
        ru.usage_count = self.usage_count;
        return ru;
    }
};

pub const LinearTypeChecker = struct {
    resources: StringHashMap(*ResourceUsage),
    violation_log: ArrayList(LinearityViolation),
    check_count: u64,
    violation_count: u64,
    allocator: Allocator,

    pub const LinearityViolation = struct {
        variable_name: []const u8,
        expected_usage: LinearityMode,
        actual_count: u32,
        violation_type: ViolationType,

        pub const ViolationType = enum {
            UNUSED,
            OVERUSED,
            DROPPED,
            DUPLICATED,
        };
    };

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .resources = StringHashMap(*ResourceUsage).init(allocator),
            .violation_log = ArrayList(LinearityViolation).init(allocator),
            .check_count = 0,
            .violation_count = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.resources.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.resources.deinit();
        for (self.violation_log.items) |v| {
            self.allocator.free(v.variable_name);
        }
        self.violation_log.deinit();
    }

    pub fn introduce(self: *Self, name: []const u8, linear_type: *LinearType) !void {
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        const usage = try ResourceUsage.init(self.allocator, name, linear_type);
        errdefer {
            usage.deinit();
            self.allocator.destroy(usage);
        }
        try self.resources.put(key, usage);
    }

    pub fn use(self: *Self, name: []const u8) !void {
        if (self.resources.get(name)) |usage| {
            usage.use();
        }
    }

    pub fn checkTerm(self: *Self, term: *Term) !bool {
        self.check_count += 1;
        switch (term.kind) {
            .VARIABLE => try self.use(term.name),
            .LAMBDA => {
                if (term.sub_terms.items.len > 0) {
                    _ = try self.checkTerm(term.sub_terms.items[0]);
                }
            },
            .APPLICATION => {
                for (term.sub_terms.items) |sub| {
                    _ = try self.checkTerm(sub);
                }
            },
            .PAIR => {
                for (term.sub_terms.items) |sub| {
                    _ = try self.checkTerm(sub);
                }
            },
            else => {},
        }
        return self.validateAll();
    }

    pub fn validateAll(self: *Self) bool {
        var all_valid = true;
        var iter = self.resources.iterator();
        while (iter.next()) |entry| {
            const usage = entry.value_ptr.*;
            if (!usage.isValid()) {
                all_valid = false;
                self.violation_count += 1;
                const dup_name = self.allocator.dupe(u8, usage.variable_name) catch continue;
                const violation = LinearityViolation{
                    .variable_name = dup_name,
                    .expected_usage = usage.linear_type.linearity,
                    .actual_count = usage.usage_count,
                    .violation_type = if (usage.usage_count == 0) .UNUSED else if (usage.usage_count > 1) .OVERUSED else .DROPPED,
                };
                self.violation_log.append(violation) catch {
                    self.allocator.free(dup_name);
                };
            }
        }
        return all_valid;
    }

    pub fn reset(self: *Self) void {
        var iter = self.resources.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.resources.clearRetainingCapacity();
        for (self.violation_log.items) |v| {
            self.allocator.free(v.variable_name);
        }
        self.violation_log.clearRetainingCapacity();
    }

    pub fn getStatistics(self: *const Self) LinearCheckerStatistics {
        return LinearCheckerStatistics{
            .check_count = self.check_count,
            .violation_count = self.violation_count,
            .active_resources = self.resources.count(),
        };
    }
};

pub const LinearCheckerStatistics = struct {
    check_count: u64,
    violation_count: u64,
    active_resources: usize,
};

pub const TypeProofKind = enum(u8) {
    TYPE_JUDGMENT = 0,
    SUBTYPING = 1,
    EQUALITY = 2,
    LINEAR_USAGE = 3,
    FUNCTOR_LAW = 4,
    MONAD_LAW = 5,
    NATURALITY = 6,
    UNIVERSE_MEMBERSHIP = 7,

    const Self = @This();

    pub fn toString(self: Self) []const u8 {
        return switch (self) {
            .TYPE_JUDGMENT => "type_judgment",
            .SUBTYPING => "subtyping",
            .EQUALITY => "equality",
            .LINEAR_USAGE => "linear_usage",
            .FUNCTOR_LAW => "functor_law",
            .MONAD_LAW => "monad_law",
            .NATURALITY => "naturality",
            .UNIVERSE_MEMBERSHIP => "universe_membership",
        };
    }
};

pub const TypeProof = struct {
    proof_type: TypeProofKind,
    judgment: ?*TypeJudgment,
    sub_type: ?*Type,
    super_type: ?*Type,
    proof_term: ?*Term,
    is_valid: bool,
    derivation_steps: ArrayList([]const u8),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, proof_type: TypeProofKind) !*Self {
        const p = try allocator.create(Self);
        errdefer allocator.destroy(p);
        p.* = Self{
            .proof_type = proof_type,
            .judgment = null,
            .sub_type = null,
            .super_type = null,
            .proof_term = null,
            .is_valid = false,
            .derivation_steps = ArrayList([]const u8).init(allocator),
            .allocator = allocator,
        };
        return p;
    }

    pub fn deinit(self: *Self) void {
        if (self.sub_type) |t| {
            t.deinit();
            self.allocator.destroy(t);
        }
        if (self.super_type) |t| {
            t.deinit();
            self.allocator.destroy(t);
        }
        if (self.judgment) |j| {
            j.deinit();
            self.allocator.destroy(j);
        }
        if (self.proof_term) |t| {
            t.deinit();
            self.allocator.destroy(t);
        }
        for (self.derivation_steps.items) |step| {
            self.allocator.free(step);
        }
        self.derivation_steps.deinit();
    }

    pub fn addStep(self: *Self, step: []const u8) !void {
        try self.derivation_steps.append(try self.allocator.dupe(u8, step));
    }

    pub fn validate(self: *Self) bool {
        self.is_valid = switch (self.proof_type) {
            .TYPE_JUDGMENT => if (self.judgment) |j| j.validate() else false,
            .SUBTYPING => self.sub_type != null and self.super_type != null,
            .EQUALITY => self.sub_type != null and self.super_type != null and self.sub_type.?.equals(self.super_type.?),
            else => self.derivation_steps.items.len > 0,
        };
        return self.is_valid;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const p = try TypeProof.init(allocator, self.proof_type);
        errdefer {
            p.deinit();
            allocator.destroy(p);
        }
        p.judgment = if (self.judgment) |j| try j.clone(allocator) else null;
        p.sub_type = if (self.sub_type) |t| try t.clone(allocator) else null;
        p.super_type = if (self.super_type) |t| try t.clone(allocator) else null;
        p.proof_term = if (self.proof_term) |t| try t.clone(allocator) else null;
        p.is_valid = self.is_valid;
        for (self.derivation_steps.items) |step| {
            try p.derivation_steps.append(try allocator.dupe(u8, step));
        }
        return p;
    }
};

pub const ProofResult = struct {
    success: bool,
    proof: ?*TypeProof,
    error_message: ?[]const u8,
    owns_proof: bool,
    allocator: Allocator,

    const Self = @This();

    pub fn initSuccess(allocator: Allocator, proof: *TypeProof) !*Self {
        const r = try allocator.create(Self);
        errdefer allocator.destroy(r);
        r.* = Self{
            .success = true,
            .proof = proof,
            .error_message = null,
            .owns_proof = false,
            .allocator = allocator,
        };
        return r;
    }

    pub fn initFailure(allocator: Allocator, message: []const u8) !*Self {
        const r = try allocator.create(Self);
        errdefer allocator.destroy(r);
        r.* = Self{
            .success = false,
            .proof = null,
            .error_message = try allocator.dupe(u8, message),
            .owns_proof = false,
            .allocator = allocator,
        };
        return r;
    }

    pub fn initWithOwnedProof(allocator: Allocator, proof: *TypeProof) !*Self {
        const r = try allocator.create(Self);
        errdefer allocator.destroy(r);
        r.* = Self{
            .success = true,
            .proof = proof,
            .error_message = null,
            .owns_proof = true,
            .allocator = allocator,
        };
        return r;
    }

    pub fn deinit(self: *Self) void {
        if (self.owns_proof) {
            if (self.proof) |p| {
                p.deinit();
                self.allocator.destroy(p);
            }
        }
        if (self.error_message) |msg| {
            self.allocator.free(msg);
        }
    }
};

pub const TypeTheoryEngine = struct {
    type_checker: TypeChecker,
    linear_checker: LinearTypeChecker,
    categories: ArrayList(*Category),
    functors: ArrayList(*Functor),
    monads: ArrayList(*Monad),
    proofs: ArrayList(*TypeProof),
    proof_count: u64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .type_checker = TypeChecker.init(allocator),
            .linear_checker = LinearTypeChecker.init(allocator),
            .categories = ArrayList(*Category).init(allocator),
            .functors = ArrayList(*Functor).init(allocator),
            .monads = ArrayList(*Monad).init(allocator),
            .proofs = ArrayList(*TypeProof).init(allocator),
            .proof_count = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.type_checker.deinit();
        self.linear_checker.deinit();
        for (self.monads.items) |m| {
            m.deinit();
            self.allocator.destroy(m);
        }
        self.monads.deinit();
        for (self.functors.items) |f| {
            f.deinit();
            self.allocator.destroy(f);
        }
        self.functors.deinit();
        for (self.categories.items) |cat| {
            cat.deinit();
            self.allocator.destroy(cat);
        }
        self.categories.deinit();
        for (self.proofs.items) |p| {
            p.deinit();
            self.allocator.destroy(p);
        }
        self.proofs.deinit();
    }

    pub fn proveTypeJudgment(self: *Self, ctx: *TypeContext, term: *Term, expected_type: *Type) !*ProofResult {
        self.proof_count += 1;
        const proof = try TypeProof.init(self.allocator, .TYPE_JUDGMENT);
        errdefer {
            proof.deinit();
            self.allocator.destroy(proof);
        }
        try proof.addStep("Begin type judgment proof");
        const inferred = self.type_checker.inferType(ctx, term) catch |err| {
            try proof.addStep("Type inference failed");
            proof.is_valid = false;
            proof.deinit();
            self.allocator.destroy(proof);
            return ProofResult.initFailure(self.allocator, @errorName(err));
        };
        defer {
            inferred.deinit();
            self.allocator.destroy(inferred);
        }
        try proof.addStep("Inferred type from term");
        if (self.type_checker.subtype(inferred, expected_type)) {
            try proof.addStep("Subtyping check passed");
            const expected_clone = try expected_type.clone(self.allocator);
            const judgment = TypeJudgment.init(self.allocator, ctx, term, expected_clone) catch |err| {
                expected_clone.deinit();
                self.allocator.destroy(expected_clone);
                return err;
            };
            _ = judgment.validate();
            proof.judgment = judgment;
            proof.is_valid = true;
            try self.proofs.append(proof);
            return ProofResult.initSuccess(self.allocator, proof);
        } else {
            try proof.addStep("Subtyping check failed");
            proof.is_valid = false;
            proof.deinit();
            self.allocator.destroy(proof);
            return ProofResult.initFailure(self.allocator, "Type mismatch");
        }
    }

    pub fn proveSubtyping(self: *Self, sub: *Type, super: *Type) !*ProofResult {
        self.proof_count += 1;
        const proof = try TypeProof.init(self.allocator, .SUBTYPING);
        errdefer {
            proof.deinit();
            self.allocator.destroy(proof);
        }
        try proof.addStep("Begin subtyping proof");
        proof.sub_type = try sub.clone(self.allocator);
        proof.super_type = try super.clone(self.allocator);
        if (self.type_checker.subtype(sub, super)) {
            try proof.addStep("Subtyping relation verified");
            proof.is_valid = true;
            try self.proofs.append(proof);
            return ProofResult.initSuccess(self.allocator, proof);
        } else {
            try proof.addStep("Subtyping relation failed");
            proof.is_valid = false;
            proof.deinit();
            self.allocator.destroy(proof);
            return ProofResult.initFailure(self.allocator, "No subtyping relation exists");
        }
    }

    pub fn proveEquality(self: *Self, t1: *Type, t2: *Type) !*ProofResult {
        self.proof_count += 1;
        const proof = try TypeProof.init(self.allocator, .EQUALITY);
        errdefer {
            proof.deinit();
            self.allocator.destroy(proof);
        }
        try proof.addStep("Begin equality proof");
        proof.sub_type = try t1.clone(self.allocator);
        proof.super_type = try t2.clone(self.allocator);
        if (t1.equals(t2)) {
            try proof.addStep("Types are definitionally equal");
            proof.is_valid = true;
            try self.proofs.append(proof);
            return ProofResult.initSuccess(self.allocator, proof);
        } else {
            const unified = self.type_checker.unifyTypes(t1, t2) catch {
                try proof.addStep("Unification failed");
                proof.is_valid = false;
                proof.deinit();
                self.allocator.destroy(proof);
                return ProofResult.initFailure(self.allocator, "Types are not equal");
            };
            defer {
                unified.deinit();
                self.allocator.destroy(unified);
            }
            try proof.addStep("Types unified via type unification");
            proof.is_valid = true;
            try self.proofs.append(proof);
            return ProofResult.initSuccess(self.allocator, proof);
        }
    }

    pub fn checkLinearUsage(self: *Self, term: *Term) !*ProofResult {
        self.proof_count += 1;
        const proof = try TypeProof.init(self.allocator, .LINEAR_USAGE);
        errdefer {
            proof.deinit();
            self.allocator.destroy(proof);
        }
        try proof.addStep("Begin linear usage check");
        const valid = try self.linear_checker.checkTerm(term);
        if (valid) {
            try proof.addStep("All linear resources used correctly");
            proof.is_valid = true;
            try self.proofs.append(proof);
            return ProofResult.initSuccess(self.allocator, proof);
        } else {
            try proof.addStep("Linear usage violation detected");
            proof.is_valid = false;
            proof.deinit();
            self.allocator.destroy(proof);
            return ProofResult.initFailure(self.allocator, "Linear usage violation");
        }
    }

    pub fn functorCheck(self: *Self, f: *Functor) !*ProofResult {
        self.proof_count += 1;
        const proof = try TypeProof.init(self.allocator, .FUNCTOR_LAW);
        errdefer {
            proof.deinit();
            self.allocator.destroy(proof);
        }
        try proof.addStep("Begin functor law verification");
        const laws_hold = try f.verifyFunctorLaws();
        if (laws_hold) {
            try proof.addStep("Functor preserves identity");
            try proof.addStep("Functor preserves composition");
            proof.is_valid = true;
            try self.proofs.append(proof);
            return ProofResult.initSuccess(self.allocator, proof);
        } else {
            try proof.addStep("Functor law violation");
            proof.is_valid = false;
            proof.deinit();
            self.allocator.destroy(proof);
            return ProofResult.initFailure(self.allocator, "Functor law violation");
        }
    }

    pub fn monadLaws(self: *Self, m: *Monad) !*ProofResult {
        self.proof_count += 1;
        const proof = try TypeProof.init(self.allocator, .MONAD_LAW);
        errdefer {
            proof.deinit();
            self.allocator.destroy(proof);
        }
        try proof.addStep("Begin monad law verification");
        const laws_hold = try m.verifyMonadLaws();
        if (laws_hold) {
            try proof.addStep("Left unit law verified");
            try proof.addStep("Right unit law verified");
            try proof.addStep("Associativity law verified");
            proof.is_valid = true;
            try self.proofs.append(proof);
            return ProofResult.initSuccess(self.allocator, proof);
        } else {
            try proof.addStep("Monad law violation");
            proof.is_valid = false;
            proof.deinit();
            self.allocator.destroy(proof);
            return ProofResult.initFailure(self.allocator, "Monad law violation");
        }
    }

    pub fn createCategory(self: *Self, name: []const u8) !*Category {
        const cat = try Category.init(self.allocator, name);
        errdefer {
            cat.deinit();
            self.allocator.destroy(cat);
        }
        try self.categories.append(cat);
        return cat;
    }

    pub fn createFunctor(self: *Self, name: []const u8, source: *Category, target: *Category) !*Functor {
        const f = try Functor.init(self.allocator, name, source, target);
        errdefer {
            f.deinit();
            self.allocator.destroy(f);
        }
        try self.functors.append(f);
        return f;
    }

    pub fn createMonad(self: *Self, name: []const u8, t: *Functor, eta: *NaturalTransformation, mu: *NaturalTransformation) !*Monad {
        const m = try Monad.init(self.allocator, name, t, eta, mu);
        errdefer {
            m.deinit();
            self.allocator.destroy(m);
        }
        try self.monads.append(m);
        return m;
    }

    pub fn getStatistics(self: *const Self) TypeTheoryStatistics {
        return TypeTheoryStatistics{
            .type_checker_stats = self.type_checker.getStatistics(),
            .linear_checker_stats = self.linear_checker.getStatistics(),
            .proof_count = self.proof_count,
            .category_count = self.categories.items.len,
            .functor_count = self.functors.items.len,
            .monad_count = self.monads.items.len,
        };
    }
};

pub const TypeTheoryStatistics = struct {
    type_checker_stats: TypeCheckerStatistics,
    linear_checker_stats: LinearCheckerStatistics,
    proof_count: u64,
    category_count: usize,
    functor_count: usize,
    monad_count: usize,
};

test "type construction unit" {
    const allocator = std.testing.allocator;
    const unit_type = try Type.initUnit(allocator);
    defer {
        unit_type.deinit();
        allocator.destroy(unit_type);
    }
    try std.testing.expect(unit_type.kind == .UNIT);
    try std.testing.expect(unit_type.kind.isBaseType());
}

test "type construction nat" {
    const allocator = std.testing.allocator;
    const nat_type = try Type.initNat(allocator);
    defer {
        nat_type.deinit();
        allocator.destroy(nat_type);
    }
    try std.testing.expect(nat_type.kind == .NAT);
}

test "type construction function" {
    const allocator = std.testing.allocator;
    const nat = try Type.initNat(allocator);
    const bool_type = try Type.initBool(allocator);
    const func_type = try Type.initFunction(allocator, nat, bool_type);
    defer {
        func_type.deinit();
        allocator.destroy(func_type);
    }
    try std.testing.expect(func_type.kind == .FUNCTION);
    try std.testing.expect(func_type.getDomain().?.kind == .NAT);
    try std.testing.expect(func_type.getCodomain().?.kind == .BOOL);
}

test "type equality" {
    const allocator = std.testing.allocator;
    const nat1 = try Type.initNat(allocator);
    defer {
        nat1.deinit();
        allocator.destroy(nat1);
    }
    const nat2 = try Type.initNat(allocator);
    defer {
        nat2.deinit();
        allocator.destroy(nat2);
    }
    try std.testing.expect(nat1.equals(nat2));
}

test "type inequality" {
    const allocator = std.testing.allocator;
    const nat = try Type.initNat(allocator);
    defer {
        nat.deinit();
        allocator.destroy(nat);
    }
    const bool_type = try Type.initBool(allocator);
    defer {
        bool_type.deinit();
        allocator.destroy(bool_type);
    }
    try std.testing.expect(!nat.equals(bool_type));
}

test "type context extend and lookup" {
    const allocator = std.testing.allocator;
    var ctx = TypeContext.init(allocator);
    defer ctx.deinit();
    const nat = try Type.initNat(allocator);
    defer {
        nat.deinit();
        allocator.destroy(nat);
    }
    try ctx.extend("x", nat);
    const found = ctx.lookup("x");
    try std.testing.expect(found != null);
    try std.testing.expect(found.?.kind == .NAT);
}

test "type context lookup missing" {
    const allocator = std.testing.allocator;
    var ctx = TypeContext.init(allocator);
    defer ctx.deinit();
    const found = ctx.lookup("missing");
    try std.testing.expect(found == null);
}

test "dependent pi type" {
    const allocator = std.testing.allocator;
    const nat = try Type.initNat(allocator);
    const bool_type = try Type.initBool(allocator);
    const pi = try DependentPi.init(allocator, "n", nat, bool_type);
    defer {
        pi.deinit();
        allocator.destroy(pi);
    }
    const pi_type = try pi.toType(allocator);
    defer {
        pi_type.deinit();
        allocator.destroy(pi_type);
    }
    try std.testing.expect(pi_type.kind == .DEPENDENT_FUNCTION);
}

test "dependent sigma type" {
    const allocator = std.testing.allocator;
    const nat = try Type.initNat(allocator);
    const bool_type = try Type.initBool(allocator);
    const sigma = try DependentSigma.init(allocator, "n", nat, bool_type);
    defer {
        sigma.deinit();
        allocator.destroy(sigma);
    }
    const sigma_type = try sigma.toType(allocator);
    defer {
        sigma_type.deinit();
        allocator.destroy(sigma_type);
    }
    try std.testing.expect(sigma_type.kind == .DEPENDENT_PAIR);
}

test "identity type reflexivity" {
    const allocator = std.testing.allocator;
    const nat = try Type.initNat(allocator);
    const term = try Term.initZero(allocator);
    const id = try IdentityType.refl(allocator, nat, term);
    defer {
        id.deinit();
        allocator.destroy(id);
    }
    try std.testing.expect(id.isReflexive());
}

test "universe type hierarchy" {
    const allocator = std.testing.allocator;
    const type0 = try UniverseType.init(allocator, 0);
    defer {
        type0.deinit();
        allocator.destroy(type0);
    }
    const type1 = try type0.typeOf(allocator);
    defer {
        type1.deinit();
        allocator.destroy(type1);
    }
    try std.testing.expect(type1.level == 1);
    try std.testing.expect(type1.contains(type0));
}

test "inductive nat type" {
    const allocator = std.testing.allocator;
    const nat = try InductiveType.initNat(allocator);
    defer {
        nat.deinit();
        allocator.destroy(nat);
    }
    try std.testing.expect(nat.constructors.items.len == 2);
    try std.testing.expect(std.mem.eql(u8, nat.constructors.items[0].name, "zero"));
    try std.testing.expect(std.mem.eql(u8, nat.constructors.items[1].name, "succ"));
}

test "type checker infer variable" {
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();
    var ctx = TypeContext.init(allocator);
    defer ctx.deinit();
    const nat = try Type.initNat(allocator);
    defer {
        nat.deinit();
        allocator.destroy(nat);
    }
    try ctx.extend("x", nat);
    const var_term = try Term.initVariable(allocator, "x");
    defer {
        var_term.deinit();
        allocator.destroy(var_term);
    }
    const inferred = try checker.inferType(&ctx, var_term);
    defer {
        inferred.deinit();
        allocator.destroy(inferred);
    }
    try std.testing.expect(inferred.kind == .NAT);
}

test "type checker infer literal" {
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();
    var ctx = TypeContext.init(allocator);
    defer ctx.deinit();
    const lit = try Term.initLiteralNat(allocator, 42);
    defer {
        lit.deinit();
        allocator.destroy(lit);
    }
    const inferred = try checker.inferType(&ctx, lit);
    defer {
        inferred.deinit();
        allocator.destroy(inferred);
    }
    try std.testing.expect(inferred.kind == .NAT);
}

test "type checker subtype nat int" {
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();
    const nat = try Type.initNat(allocator);
    defer {
        nat.deinit();
        allocator.destroy(nat);
    }
    const int = try Type.initInt(allocator);
    defer {
        int.deinit();
        allocator.destroy(int);
    }
    try std.testing.expect(checker.subtype(nat, int));
    try std.testing.expect(!checker.subtype(int, nat));
}

test "type checker unify same types" {
    const allocator = std.testing.allocator;
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();
    const nat1 = try Type.initNat(allocator);
    defer {
        nat1.deinit();
        allocator.destroy(nat1);
    }
    const nat2 = try Type.initNat(allocator);
    defer {
        nat2.deinit();
        allocator.destroy(nat2);
    }
    const unified = try checker.unifyTypes(nat1, nat2);
    defer {
        unified.deinit();
        allocator.destroy(unified);
    }
    try std.testing.expect(unified.kind == .NAT);
}

test "proposition as type conjunction" {
    const allocator = std.testing.allocator;
    const p = try PropositionAsType.initTrue(allocator);
    const q = try PropositionAsType.initTrue(allocator);
    const conj = try PropositionAsType.initConjunction(allocator, p, q);
    defer {
        conj.deinit();
        allocator.destroy(conj);
    }
    try std.testing.expect(conj.connective == .CONJUNCTION);
    try std.testing.expect(conj.corresponding_type.?.kind == .TUPLE);
}

test "proposition as type implication" {
    const allocator = std.testing.allocator;
    const p = try PropositionAsType.initTrue(allocator);
    const q = try PropositionAsType.initTrue(allocator);
    const impl = try PropositionAsType.initImplication(allocator, p, q);
    defer {
        impl.deinit();
        allocator.destroy(impl);
    }
    try std.testing.expect(impl.connective == .IMPLICATION);
    try std.testing.expect(impl.corresponding_type.?.kind == .FUNCTION);
}

test "proposition as type negation" {
    const allocator = std.testing.allocator;
    const p = try PropositionAsType.initTrue(allocator);
    const neg = try PropositionAsType.initNegation(allocator, p);
    defer {
        neg.deinit();
        allocator.destroy(neg);
    }
    try std.testing.expect(neg.connective == .NEGATION);
    const neg_type = neg.corresponding_type.?;
    try std.testing.expect(neg_type.kind == .FUNCTION);
    try std.testing.expect(neg_type.right_type.?.kind == .BOTTOM);
}

test "category creation" {
    const allocator = std.testing.allocator;
    const cat = try Category.init(allocator, "Set");
    defer {
        cat.deinit();
        allocator.destroy(cat);
    }
    try std.testing.expect(std.mem.eql(u8, cat.name, "Set"));
    try std.testing.expect(cat.objectCount() == 0);
}

test "category add object with identity" {
    const allocator = std.testing.allocator;
    const cat = try Category.init(allocator, "C");
    defer {
        cat.deinit();
        allocator.destroy(cat);
    }
    const a = try cat.addObject("A");
    try std.testing.expect(cat.objectCount() == 1);
    try std.testing.expect(cat.morphismCount() == 1);
    const id = cat.getIdentity(a);
    try std.testing.expect(id != null);
    try std.testing.expect(id.?.is_identity);
}

test "category morphism composition" {
    const allocator = std.testing.allocator;
    const cat = try Category.init(allocator, "C");
    defer {
        cat.deinit();
        allocator.destroy(cat);
    }
    const a = try cat.addObject("A");
    const b = try cat.addObject("B");
    const c = try cat.addObject("C");
    const f = try cat.addMorphism("f", a, b);
    const g = try cat.addMorphism("g", b, c);
    const gf = try cat.compose(f, g);
    try std.testing.expect(gf.source.equals(a));
    try std.testing.expect(gf.target.equals(c));
}

test "functor creation" {
    const allocator = std.testing.allocator;
    const cat1 = try Category.init(allocator, "C");
    defer {
        cat1.deinit();
        allocator.destroy(cat1);
    }
    const cat2 = try Category.init(allocator, "D");
    defer {
        cat2.deinit();
        allocator.destroy(cat2);
    }
    const f = try Functor.init(allocator, "F", cat1, cat2);
    defer {
        f.deinit();
        allocator.destroy(f);
    }
    try std.testing.expect(std.mem.eql(u8, f.name, "F"));
}

test "linear type checker introduction" {
    const allocator = std.testing.allocator;
    var checker = LinearTypeChecker.init(allocator);
    defer checker.deinit();
    const nat = try Type.initNat(allocator);
    const linear_nat = try LinearType.initLinear(allocator, nat);
    try checker.introduce("x", linear_nat);
    try std.testing.expect(checker.resources.count() == 1);
}

test "linear type usage validation" {
    const allocator = std.testing.allocator;
    var checker = LinearTypeChecker.init(allocator);
    defer checker.deinit();
    const nat = try Type.initNat(allocator);
    const linear_nat = try LinearType.initLinear(allocator, nat);
    try checker.introduce("x", linear_nat);
    try checker.use("x");
    const valid = checker.validateAll();
    try std.testing.expect(valid);
}

test "linear type overuse violation" {
    const allocator = std.testing.allocator;
    var checker = LinearTypeChecker.init(allocator);
    defer checker.deinit();
    const nat = try Type.initNat(allocator);
    const linear_nat = try LinearType.initLinear(allocator, nat);
    try checker.introduce("x", linear_nat);
    try checker.use("x");
    try checker.use("x");
    const valid = checker.validateAll();
    try std.testing.expect(!valid);
}

test "affine type can drop" {
    const allocator = std.testing.allocator;
    var checker = LinearTypeChecker.init(allocator);
    defer checker.deinit();
    const nat = try Type.initNat(allocator);
    const affine_nat = try LinearType.initAffine(allocator, nat);
    try checker.introduce("x", affine_nat);
    const valid = checker.validateAll();
    try std.testing.expect(valid);
}

test "type theory engine proof judgment" {
    const allocator = std.testing.allocator;
    var engine = TypeTheoryEngine.init(allocator);
    defer engine.deinit();
    var ctx = TypeContext.init(allocator);
    defer ctx.deinit();
    const nat = try Type.initNat(allocator);
    defer {
        nat.deinit();
        allocator.destroy(nat);
    }
    try ctx.extend("x", nat);
    const var_term = try Term.initVariable(allocator, "x");
    defer {
        var_term.deinit();
        allocator.destroy(var_term);
    }
    const expected = try Type.initNat(allocator);
    defer {
        expected.deinit();
        allocator.destroy(expected);
    }
    const result = try engine.proveTypeJudgment(&ctx, var_term, expected);
    defer {
        result.deinit();
        allocator.destroy(result);
    }
    try std.testing.expect(result.success);
}

test "type theory engine prove subtyping" {
    const allocator = std.testing.allocator;
    var engine = TypeTheoryEngine.init(allocator);
    defer engine.deinit();
    const nat = try Type.initNat(allocator);
    defer {
        nat.deinit();
        allocator.destroy(nat);
    }
    const int = try Type.initInt(allocator);
    defer {
        int.deinit();
        allocator.destroy(int);
    }
    const result = try engine.proveSubtyping(nat, int);
    defer {
        result.deinit();
        allocator.destroy(result);
    }
    try std.testing.expect(result.success);
}

test "type theory engine prove equality" {
    const allocator = std.testing.allocator;
    var engine = TypeTheoryEngine.init(allocator);
    defer engine.deinit();
    const nat1 = try Type.initNat(allocator);
    defer {
        nat1.deinit();
        allocator.destroy(nat1);
    }
    const nat2 = try Type.initNat(allocator);
    defer {
        nat2.deinit();
        allocator.destroy(nat2);
    }
    const result = try engine.proveEquality(nat1, nat2);
    defer {
        result.deinit();
        allocator.destroy(result);
    }
    try std.testing.expect(result.success);
}

test "type theory engine statistics" {
    const allocator = std.testing.allocator;
    var engine = TypeTheoryEngine.init(allocator);
    defer engine.deinit();
    const stats = engine.getStatistics();
    try std.testing.expect(stats.proof_count == 0);
    try std.testing.expect(stats.category_count == 0);
}

test "universe level ordering" {
    const allocator = std.testing.allocator;
    const type0 = try Type.initUniverse(allocator, 0);
    defer {
        type0.deinit();
        allocator.destroy(type0);
    }
    const type1 = try Type.initUniverse(allocator, 1);
    defer {
        type1.deinit();
        allocator.destroy(type1);
    }
    var checker = TypeChecker.init(allocator);
    defer checker.deinit();
    try std.testing.expect(checker.subtype(type0, type1));
    try std.testing.expect(!checker.subtype(type1, type0));
}

test "term construction lambda" {
    const allocator = std.testing.allocator;
    const body = try Term.initVariable(allocator, "x");
    const lambda = try Term.initLambda(allocator, "x", body);
    defer {
        lambda.deinit();
        allocator.destroy(lambda);
    }
    try std.testing.expect(lambda.kind == .LAMBDA);
    try std.testing.expect(std.mem.eql(u8, lambda.bound_variable.?, "x"));
}

test "term construction application" {
    const allocator = std.testing.allocator;
    const func = try Term.initVariable(allocator, "f");
    const arg = try Term.initVariable(allocator, "x");
    const app = try Term.initApplication(allocator, func, arg);
    defer {
        app.deinit();
        allocator.destroy(app);
    }
    try std.testing.expect(app.kind == .APPLICATION);
    try std.testing.expect(app.sub_terms.items.len == 2);
}

