const std = @import("std");
const nsir_core = @import("nsir_core.zig");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const Complex = std.math.Complex;

pub const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
pub const Node = nsir_core.Node;
pub const Edge = nsir_core.Edge;
pub const EdgeQuality = nsir_core.EdgeQuality;

pub const VectorType = enum(u8) {
    f32x4 = 0,
    f32x8 = 1,
    f64x2 = 2,
    f64x4 = 3,
    i32x4 = 4,
    i32x8 = 5,

    pub fn toString(self: VectorType) []const u8 {
        return switch (self) {
            .f32x4 => "f32x4",
            .f32x8 => "f32x8",
            .f64x2 => "f64x2",
            .f64x4 => "f64x4",
            .i32x4 => "i32x4",
            .i32x8 => "i32x8",
        };
    }

    pub fn fromString(s: []const u8) ?VectorType {
        const normalized = blk: {
            var buf: [16]u8 = undefined;
            var i: usize = 0;
            for (s) |c| {
                if (i >= buf.len) break;
                buf[i] = std.ascii.toLower(c);
                i += 1;
            }
            break :blk buf[0..i];
        };
        if (std.mem.eql(u8, normalized, "f32x4")) return .f32x4;
        if (std.mem.eql(u8, normalized, "f32x8")) return .f32x8;
        if (std.mem.eql(u8, normalized, "f64x2")) return .f64x2;
        if (std.mem.eql(u8, normalized, "f64x4")) return .f64x4;
        if (std.mem.eql(u8, normalized, "i32x4")) return .i32x4;
        if (std.mem.eql(u8, normalized, "i32x8")) return .i32x8;
        return null;
    }

    pub fn lanes(self: VectorType) usize {
        return switch (self) {
            .f32x4 => 4,
            .f32x8 => 8,
            .f64x2 => 2,
            .f64x4 => 4,
            .i32x4 => 4,
            .i32x8 => 8,
        };
    }

    pub fn elementSize(self: VectorType) usize {
        return switch (self) {
            .f32x4, .f32x8, .i32x4, .i32x8 => 4,
            .f64x2, .f64x4 => 8,
        };
    }

    pub fn totalSize(self: VectorType) usize {
        return self.lanes() * self.elementSize();
    }

    pub fn alignment(self: VectorType) usize {
        return switch (self) {
            .f32x4, .i32x4 => 16,
            .f32x8, .i32x8 => 32,
            .f64x2 => 16,
            .f64x4 => 32,
        };
    }
};

pub const VectorError = error{
    InvalidOperation,
    TypeMismatch,
    OutOfBounds,
    DivisionByZero,
    InvalidLength,
    AllocationFailed,
};

pub fn SimdVector(comptime T: type, comptime N: usize) type {
    comptime {
        const valid_float = T == f32 or T == f64;
        const valid_int = T == i32 or T == i64 or T == u32 or T == u64;
        if (!valid_float and !valid_int) {
            @compileError("SimdVector requires numeric type: f32, f64, i32, i64, u32, or u64");
        }
    }

    return struct {
        data: @Vector(N, T),

        const Self = @This();
        const VecType = @Vector(N, T);
        const is_float = T == f32 or T == f64;
        const is_signed = T == i32 or T == i64 or T == f32 or T == f64;

        pub fn init(value: T) Self {
            return Self{ .data = @splat(value) };
        }

        pub fn initFromArray(arr: [N]T) Self {
            return Self{ .data = @as(VecType, arr) };
        }

        pub fn initFromSliceChecked(slice: []const T) VectorError!Self {
            if (slice.len < N) {
                return VectorError.InvalidLength;
            }
            var arr: [N]T = undefined;
            var i: usize = 0; while (i < N) : (i += 1) {
                arr[i] = slice[i];
            }
            return Self{ .data = @as(VecType, arr) };
        }

        pub fn initFromSlice(slice: []const T) Self {
            var arr: [N]T = undefined;
            var i: usize = 0; while (i < N) : (i += 1) {
                if (i < slice.len) {
                    arr[i] = slice[i];
                } else {
                    arr[i] = if (T == f32 or T == f64) @as(T, 0.0) else @as(T, 0);
                }
            }
            return Self{ .data = @as(VecType, arr) };
        }

        pub fn toArray(self: Self) [N]T {
            return @as([N]T, self.data);
        }

        pub fn add(self: Self, other: Self) Self {
            return Self{ .data = self.data + other.data };
        }

        pub fn sub(self: Self, other: Self) Self {
            return Self{ .data = self.data - other.data };
        }

        pub fn mul(self: Self, other: Self) Self {
            return Self{ .data = self.data * other.data };
        }

        pub fn divChecked(self: Self, other: Self) VectorError!Self {
            const other_arr = other.toArray();
            for (other_arr) |v| {
                if (is_float) {
                    const epsilon: T = if (T == f32) 1e-7 else 1e-15;
                    if (@abs(v) < epsilon) return VectorError.DivisionByZero;
                } else {
                    if (v == 0) return VectorError.DivisionByZero;
                }
            }
            return Self{ .data = self.data / other.data };
        }

        pub fn div(self: Self, other: Self) Self {
            return Self{ .data = self.data / other.data };
        }

        pub fn scale(self: Self, scalar: T) Self {
            const scalar_vec: VecType = @splat(scalar);
            return Self{ .data = self.data * scalar_vec };
        }

        pub fn dot(self: Self, other: Self) T {
            const product = self.data * other.data;
            return @reduce(.Add, product);
        }

        pub fn magnitude(self: Self) T {
            if (!is_float) {
                @compileError("magnitude requires floating-point type");
            }
            const squared = self.data * self.data;
            const sum = @reduce(.Add, squared);
            return @sqrt(sum);
        }

        pub fn normalize(self: Self) Self {
            if (!is_float) {
                @compileError("normalize requires floating-point type");
            }
            const mag = self.magnitude();
            const epsilon: T = if (T == f32) 1e-7 else 1e-15;
            if (@abs(mag) < epsilon) {
                return Self.init(0);
            }
            const mag_vec: VecType = @splat(mag);
            return Self{ .data = self.data / mag_vec };
        }

        pub fn fma(self: Self, mul_vec: Self, add_vec: Self) Self {
            if (!is_float) {
                @compileError("fma requires floating-point type");
            }
            return Self{ .data = @mulAdd(VecType, self.data, mul_vec.data, add_vec.data) };
        }

        pub fn sqrt(self: Self) Self {
            if (!is_float) {
                @compileError("sqrt requires floating-point type");
            }
            return Self{ .data = @sqrt(self.data) };
        }

        pub fn min(self: Self, other: Self) Self {
            return Self{ .data = @min(self.data, other.data) };
        }

        pub fn max(self: Self, other: Self) Self {
            return Self{ .data = @max(self.data, other.data) };
        }

        pub fn abs(self: Self) Self {
            if (!is_signed) {
                return self;
            }
            if (is_float) {
                return Self{ .data = @abs(self.data) };
            } else {
                const zero: VecType = @splat(0);
                const neg = zero - self.data;
                const is_negative = self.data < zero;
                return Self{ .data = @select(T, is_negative, neg, self.data) };
            }
        }

        pub fn reduce_add(self: Self) T {
            return @reduce(.Add, self.data);
        }

        pub fn reduce_mul(self: Self) T {
            return @reduce(.Mul, self.data);
        }

        pub fn reduce_min(self: Self) T {
            return @reduce(.Min, self.data);
        }

        pub fn reduce_max(self: Self) T {
            return @reduce(.Max, self.data);
        }

        pub fn getChecked(self: Self, index: usize) VectorError!T {
            if (index >= N) {
                return VectorError.OutOfBounds;
            }
            return self.data[index];
        }

        pub fn get(self: Self, index: usize) T {
            if (index >= N) {
                return if (T == f32 or T == f64) @as(T, 0.0) else @as(T, 0);
            }
            return self.data[index];
        }

        pub fn setChecked(self: *Self, index: usize, value: T) VectorError!void {
            if (index >= N) {
                return VectorError.OutOfBounds;
            }
            var arr = self.toArray();
            arr[index] = value;
            self.data = @as(VecType, arr);
        }

        pub fn set(self: *Self, index: usize, value: T) void {
            if (index >= N) {
                return;
            }
            var arr = self.toArray();
            arr[index] = value;
            self.data = @as(VecType, arr);
        }

        pub fn blend(self: Self, other: Self, mask: @Vector(N, bool)) Self {
            return Self{ .data = @select(T, mask, other.data, self.data) };
        }

        pub fn shuffle(self: Self, comptime indices: [N]i32) Self {
            const zero_vec: VecType = @splat(if (T == f32 or T == f64) @as(T, 0.0) else @as(T, 0));
            return Self{ .data = @shuffle(T, self.data, zero_vec, indices) };
        }

        pub fn cross3(self: Self, other: Self) VectorError!Self {
            if (N < 3) {
                return VectorError.InvalidOperation;
            }
            const a = self.toArray();
            const b = other.toArray();
            var result: [N]T = undefined;
            result[0] = a[1] * b[2] - a[2] * b[1];
            result[1] = a[2] * b[0] - a[0] * b[2];
            result[2] = a[0] * b[1] - a[1] * b[0];
            var i: usize = 3; while (i < N) : (i += 1) {
                result[i] = if (T == f32 or T == f64) @as(T, 0.0) else @as(T, 0);
            }
            return Self.initFromArray(result);
        }

        pub fn distance(self: Self, other: Self) T {
            if (!is_float) {
                @compileError("distance requires floating-point type");
            }
            return self.sub(other).magnitude();
        }

        pub fn lerp(self: Self, other: Self, t: T) Self {
            if (!is_float) {
                @compileError("lerp requires floating-point type");
            }
            const one_minus_t: VecType = @splat(1.0 - t);
            const t_vec: VecType = @splat(t);
            return Self{ .data = self.data * one_minus_t + other.data * t_vec };
        }

        pub fn clamp(self: Self, min_val: T, max_val: T) Self {
            const min_vec: VecType = @splat(min_val);
            const max_vec: VecType = @splat(max_val);
            return Self{ .data = @max(min_vec, @min(max_vec, self.data)) };
        }

        pub fn negate(self: Self) Self {
            const zero: VecType = @splat(if (T == f32 or T == f64) @as(T, 0.0) else @as(T, 0));
            return Self{ .data = zero - self.data };
        }

        pub fn reflect(self: Self, normal: Self) Self {
            if (!is_float) {
                @compileError("reflect requires floating-point type");
            }
            const normalized_normal = normal.normalize();
            const d = self.dot(normalized_normal);
            const two_d: VecType = @splat(2 * d);
            return Self{ .data = self.data - two_d * normalized_normal.data };
        }

        pub fn isFinite(self: Self) bool {
            if (!is_float) return true;
            const arr = self.toArray();
            for (arr) |v| {
                if (std.math.isNan(v) or std.math.isInf(v)) {
                    return false;
                }
            }
            return true;
        }
    };
}

pub const F32x4 = SimdVector(f32, 4);
pub const F32x8 = SimdVector(f32, 8);
pub const F64x2 = SimdVector(f64, 2);
pub const F64x4 = SimdVector(f64, 4);
pub const I32x4 = SimdVector(i32, 4);
pub const I32x8 = SimdVector(i32, 8);

pub const VectorBatchEntry = struct {
    vector_type: VectorType,
    data: []align(32) u8,
    allocator: Allocator,
    is_valid: bool,

    const Self = @This();

    pub fn init(allocator: Allocator, vector_type: VectorType) !Self {
        const size = vector_type.totalSize();
        _ = vector_type.alignment();
        const data = try allocator.alignedAlloc(u8, 32, size);
        @memset(data, 0);
        return Self{
            .vector_type = vector_type,
            .data = data,
            .allocator = allocator,
            .is_valid = true,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.is_valid) {
            self.allocator.free(self.data);
            self.data = &[_]u8{};
            self.is_valid = false;
        }
    }

    pub fn asF32x4(self: *const Self) VectorError!F32x4 {
        if (self.vector_type != .f32x4) return VectorError.TypeMismatch;
        if (!self.is_valid) return VectorError.InvalidOperation;
        const arr: *const [4]f32 = @ptrCast(@alignCast(self.data.ptr));
        return F32x4.initFromArray(arr.*);
    }

    pub fn asF32x8(self: *const Self) VectorError!F32x8 {
        if (self.vector_type != .f32x8) return VectorError.TypeMismatch;
        if (!self.is_valid) return VectorError.InvalidOperation;
        const arr: *const [8]f32 = @ptrCast(@alignCast(self.data.ptr));
        return F32x8.initFromArray(arr.*);
    }

    pub fn asF64x2(self: *const Self) VectorError!F64x2 {
        if (self.vector_type != .f64x2) return VectorError.TypeMismatch;
        if (!self.is_valid) return VectorError.InvalidOperation;
        const arr: *const [2]f64 = @ptrCast(@alignCast(self.data.ptr));
        return F64x2.initFromArray(arr.*);
    }

    pub fn asF64x4(self: *const Self) VectorError!F64x4 {
        if (self.vector_type != .f64x4) return VectorError.TypeMismatch;
        if (!self.is_valid) return VectorError.InvalidOperation;
        const arr: *const [4]f64 = @ptrCast(@alignCast(self.data.ptr));
        return F64x4.initFromArray(arr.*);
    }

    pub fn asI32x4(self: *const Self) VectorError!I32x4 {
        if (self.vector_type != .i32x4) return VectorError.TypeMismatch;
        if (!self.is_valid) return VectorError.InvalidOperation;
        const arr: *const [4]i32 = @ptrCast(@alignCast(self.data.ptr));
        return I32x4.initFromArray(arr.*);
    }

    pub fn asI32x8(self: *const Self) VectorError!I32x8 {
        if (self.vector_type != .i32x8) return VectorError.TypeMismatch;
        if (!self.is_valid) return VectorError.InvalidOperation;
        const arr: *const [8]i32 = @ptrCast(@alignCast(self.data.ptr));
        return I32x8.initFromArray(arr.*);
    }

    pub fn setF32x4(self: *Self, vec: F32x4) VectorError!void {
        if (self.vector_type != .f32x4) return VectorError.TypeMismatch;
        if (!self.is_valid) return VectorError.InvalidOperation;
        const arr = vec.toArray();
        const dest: *[4]f32 = @ptrCast(@alignCast(self.data.ptr));
        dest.* = arr;
    }

    pub fn setF32x8(self: *Self, vec: F32x8) VectorError!void {
        if (self.vector_type != .f32x8) return VectorError.TypeMismatch;
        if (!self.is_valid) return VectorError.InvalidOperation;
        const arr = vec.toArray();
        const dest: *[8]f32 = @ptrCast(@alignCast(self.data.ptr));
        dest.* = arr;
    }

    pub fn setF64x2(self: *Self, vec: F64x2) VectorError!void {
        if (self.vector_type != .f64x2) return VectorError.TypeMismatch;
        if (!self.is_valid) return VectorError.InvalidOperation;
        const arr = vec.toArray();
        const dest: *[2]f64 = @ptrCast(@alignCast(self.data.ptr));
        dest.* = arr;
    }

    pub fn setF64x4(self: *Self, vec: F64x4) VectorError!void {
        if (self.vector_type != .f64x4) return VectorError.TypeMismatch;
        if (!self.is_valid) return VectorError.InvalidOperation;
        const arr = vec.toArray();
        const dest: *[4]f64 = @ptrCast(@alignCast(self.data.ptr));
        dest.* = arr;
    }

    pub fn setI32x4(self: *Self, vec: I32x4) VectorError!void {
        if (self.vector_type != .i32x4) return VectorError.TypeMismatch;
        if (!self.is_valid) return VectorError.InvalidOperation;
        const arr = vec.toArray();
        const dest: *[4]i32 = @ptrCast(@alignCast(self.data.ptr));
        dest.* = arr;
    }

    pub fn setI32x8(self: *Self, vec: I32x8) VectorError!void {
        if (self.vector_type != .i32x8) return VectorError.TypeMismatch;
        if (!self.is_valid) return VectorError.InvalidOperation;
        const arr = vec.toArray();
        const dest: *[8]i32 = @ptrCast(@alignCast(self.data.ptr));
        dest.* = arr;
    }
};

pub const VectorBatch = struct {
    vectors: ArrayList(VectorBatchEntry),
    batch_size: usize,
    allocator: Allocator,
    processed_count: usize,
    total_unique_processed: usize,
    skipped_count: usize,

    const Self = @This();

    pub fn init(allocator: Allocator, batch_size: usize) Self {
        return Self{
            .vectors = ArrayList(VectorBatchEntry).init(allocator),
            .batch_size = batch_size,
            .allocator = allocator,
            .processed_count = 0,
            .total_unique_processed = 0,
            .skipped_count = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.vectors.items) |*entry| {
            entry.deinit();
        }
        self.vectors.deinit();
    }

    pub fn addVector(self: *Self, vector_type: VectorType) !usize {
        var entry = try VectorBatchEntry.init(self.allocator, vector_type);
        errdefer entry.deinit();
        try self.vectors.append(entry);
        return self.vectors.items.len - 1;
    }

    pub fn addF32x4(self: *Self, vec: F32x4) !usize {
        const idx = try self.addVector(.f32x4);
        try self.vectors.items[idx].setF32x4(vec);
        return idx;
    }

    pub fn addF32x8(self: *Self, vec: F32x8) !usize {
        const idx = try self.addVector(.f32x8);
        try self.vectors.items[idx].setF32x8(vec);
        return idx;
    }

    pub fn addF64x2(self: *Self, vec: F64x2) !usize {
        const idx = try self.addVector(.f64x2);
        try self.vectors.items[idx].setF64x2(vec);
        return idx;
    }

    pub fn addF64x4(self: *Self, vec: F64x4) !usize {
        const idx = try self.addVector(.f64x4);
        try self.vectors.items[idx].setF64x4(vec);
        return idx;
    }

    pub fn addI32x4(self: *Self, vec: I32x4) !usize {
        const idx = try self.addVector(.i32x4);
        try self.vectors.items[idx].setI32x4(vec);
        return idx;
    }

    pub fn addI32x8(self: *Self, vec: I32x8) !usize {
        const idx = try self.addVector(.i32x8);
        try self.vectors.items[idx].setI32x8(vec);
        return idx;
    }

    pub fn getEntry(self: *const Self, index: usize) ?*const VectorBatchEntry {
        if (index >= self.vectors.items.len) return null;
        return &self.vectors.items[index];
    }

    pub fn getEntryMut(self: *Self, index: usize) ?*VectorBatchEntry {
        if (index >= self.vectors.items.len) return null;
        return &self.vectors.items[index];
    }

    pub fn processBatch(self: *Self, operation: BatchOperation) !void {
        const process_count = @min(self.batch_size, self.vectors.items.len);
        var success_count: usize = 0;
        var skip_count: usize = 0;
        var i: usize = 0; while (i < process_count) : (i += 1) {
            const entry = &self.vectors.items[i];
            switch (operation) {
                .normalize => self.normalizeEntry(entry) catch |err| switch (err) {
                    VectorError.InvalidOperation => {
                        skip_count += 1;
                        continue;
                    },
                    else => return err,
                },
                .scale => |s| self.scaleEntry(entry, s) catch |err| switch (err) {
                    VectorError.InvalidOperation => {
                        skip_count += 1;
                        continue;
                    },
                    else => return err,
                },
                .abs => self.absEntry(entry) catch |err| switch (err) {
                    VectorError.InvalidOperation => {
                        skip_count += 1;
                        continue;
                    },
                    else => return err,
                },
                .sqrt => self.sqrtEntry(entry) catch |err| switch (err) {
                    VectorError.InvalidOperation => {
                        skip_count += 1;
                        continue;
                    },
                    else => return err,
                },
            }
            success_count += 1;
        }
        self.processed_count = success_count;
        self.skipped_count = skip_count;
        self.total_unique_processed += success_count;
    }

    fn normalizeEntry(self: *Self, entry: *VectorBatchEntry) !void {
        _ = self;
        switch (entry.vector_type) {
            .f32x4 => {
                const vec = try entry.asF32x4();
                try entry.setF32x4(vec.normalize());
            },
            .f32x8 => {
                const vec = try entry.asF32x8();
                try entry.setF32x8(vec.normalize());
            },
            .f64x2 => {
                const vec = try entry.asF64x2();
                try entry.setF64x2(vec.normalize());
            },
            .f64x4 => {
                const vec = try entry.asF64x4();
                try entry.setF64x4(vec.normalize());
            },
            .i32x4, .i32x8 => return VectorError.InvalidOperation,
        }
    }

    fn scaleEntry(self: *Self, entry: *VectorBatchEntry, scalar: f64) !void {
        _ = self;
        switch (entry.vector_type) {
            .f32x4 => {
                const vec = try entry.asF32x4();
                try entry.setF32x4(vec.scale(@floatCast(scalar)));
            },
            .f32x8 => {
                const vec = try entry.asF32x8();
                try entry.setF32x8(vec.scale(@floatCast(scalar)));
            },
            .f64x2 => {
                const vec = try entry.asF64x2();
                try entry.setF64x2(vec.scale(scalar));
            },
            .f64x4 => {
                const vec = try entry.asF64x4();
                try entry.setF64x4(vec.scale(scalar));
            },
            .i32x4 => {
                const vec = try entry.asI32x4();
                try entry.setI32x4(vec.scale(@intFromFloat(scalar)));
            },
            .i32x8 => {
                const vec = try entry.asI32x8();
                try entry.setI32x8(vec.scale(@intFromFloat(scalar)));
            },
        }
    }

    fn absEntry(self: *Self, entry: *VectorBatchEntry) !void {
        _ = self;
        switch (entry.vector_type) {
            .f32x4 => {
                const vec = try entry.asF32x4();
                try entry.setF32x4(vec.abs());
            },
            .f32x8 => {
                const vec = try entry.asF32x8();
                try entry.setF32x8(vec.abs());
            },
            .f64x2 => {
                const vec = try entry.asF64x2();
                try entry.setF64x2(vec.abs());
            },
            .f64x4 => {
                const vec = try entry.asF64x4();
                try entry.setF64x4(vec.abs());
            },
            .i32x4 => {
                const vec = try entry.asI32x4();
                try entry.setI32x4(vec.abs());
            },
            .i32x8 => {
                const vec = try entry.asI32x8();
                try entry.setI32x8(vec.abs());
            },
        }
    }

    fn sqrtEntry(self: *Self, entry: *VectorBatchEntry) !void {
        _ = self;
        switch (entry.vector_type) {
            .f32x4 => {
                const vec = try entry.asF32x4();
                try entry.setF32x4(vec.sqrt());
            },
            .f32x8 => {
                const vec = try entry.asF32x8();
                try entry.setF32x8(vec.sqrt());
            },
            .f64x2 => {
                const vec = try entry.asF64x2();
                try entry.setF64x2(vec.sqrt());
            },
            .f64x4 => {
                const vec = try entry.asF64x4();
                try entry.setF64x4(vec.sqrt());
            },
            .i32x4, .i32x8 => return VectorError.InvalidOperation,
        }
    }

    pub fn transformAll(self: *Self, transform_fn: *const fn (F64x4) F64x4) !void {
        for (self.vectors.items) |*entry| {
            if (entry.vector_type == .f64x4) {
                const vec = try entry.asF64x4();
                try entry.setF64x4(transform_fn(vec));
            }
        }
    }

    pub fn reduceAll(self: *Self) f64 {
        var total: f64 = 0;
        for (self.vectors.items) |*entry| {
            switch (entry.vector_type) {
                .f32x4 => {
                    if (entry.asF32x4()) |vec| {
                        total += @as(f64, vec.reduce_add());
                    } else |_| {}
                },
                .f32x8 => {
                    if (entry.asF32x8()) |vec| {
                        total += @as(f64, vec.reduce_add());
                    } else |_| {}
                },
                .f64x2 => {
                    if (entry.asF64x2()) |vec| {
                        total += vec.reduce_add();
                    } else |_| {}
                },
                .f64x4 => {
                    if (entry.asF64x4()) |vec| {
                        total += vec.reduce_add();
                    } else |_| {}
                },
                .i32x4 => {
                    if (entry.asI32x4()) |vec| {
                        total += @as(f64, @floatFromInt(vec.reduce_add()));
                    } else |_| {}
                },
                .i32x8 => {
                    if (entry.asI32x8()) |vec| {
                        total += @as(f64, @floatFromInt(vec.reduce_add()));
                    } else |_| {}
                },
            }
        }
        return total;
    }

    pub fn count(self: *const Self) usize {
        return self.vectors.items.len;
    }

    pub fn clear(self: *Self) void {
        for (self.vectors.items) |*entry| {
            entry.deinit();
        }
        self.vectors.clearRetainingCapacity();
        self.processed_count = 0;
    }
};

pub const BatchOperation = union(enum) {
    normalize: void,
    scale: f64,
    abs: void,
    sqrt: void,
};

pub const Matrix4x4 = struct {
    data: [4]F32x4,

    const Self = @This();

    pub fn identity() Self {
        return Self{
            .data = .{
                F32x4.initFromArray(.{ 1, 0, 0, 0 }),
                F32x4.initFromArray(.{ 0, 1, 0, 0 }),
                F32x4.initFromArray(.{ 0, 0, 1, 0 }),
                F32x4.initFromArray(.{ 0, 0, 0, 1 }),
            },
        };
    }

    pub fn zero() Self {
        return Self{
            .data = .{
                F32x4.init(0),
                F32x4.init(0),
                F32x4.init(0),
                F32x4.init(0),
            },
        };
    }

    pub fn fromRows(r0: [4]f32, r1: [4]f32, r2: [4]f32, r3: [4]f32) Self {
        return Self{
            .data = .{
                F32x4.initFromArray(r0),
                F32x4.initFromArray(r1),
                F32x4.initFromArray(r2),
                F32x4.initFromArray(r3),
            },
        };
    }

    pub fn getChecked(self: Self, row_idx: usize, col_idx: usize) VectorError!f32 {
        if (row_idx >= 4 or col_idx >= 4) {
            return VectorError.OutOfBounds;
        }
        return self.data[row_idx].get(col_idx);
    }

    pub fn get(self: Self, row_idx: usize, col_idx: usize) f32 {
        if (row_idx >= 4 or col_idx >= 4) {
            return 0;
        }
        return self.data[row_idx].get(col_idx);
    }

    pub fn setChecked(self: *Self, row_idx: usize, col_idx: usize, value: f32) VectorError!void {
        if (row_idx >= 4 or col_idx >= 4) {
            return VectorError.OutOfBounds;
        }
        self.data[row_idx].set(col_idx, value);
    }

    pub fn set(self: *Self, row_idx: usize, col_idx: usize, value: f32) void {
        if (row_idx >= 4 or col_idx >= 4) {
            return;
        }
        self.data[row_idx].set(col_idx, value);
    }

    pub fn row(self: Self, idx: usize) F32x4 {
        if (idx >= 4) return F32x4.init(0);
        return self.data[idx];
    }

    pub fn col(self: Self, idx: usize) F32x4 {
        if (idx >= 4) return F32x4.init(0);
        return F32x4.initFromArray(.{
            self.data[0].get(idx),
            self.data[1].get(idx),
            self.data[2].get(idx),
            self.data[3].get(idx),
        });
    }

    pub fn add(self: Self, other: Self) Self {
        return Self{
            .data = .{
                self.data[0].add(other.data[0]),
                self.data[1].add(other.data[1]),
                self.data[2].add(other.data[2]),
                self.data[3].add(other.data[3]),
            },
        };
    }

    pub fn sub(self: Self, other: Self) Self {
        return Self{
            .data = .{
                self.data[0].sub(other.data[0]),
                self.data[1].sub(other.data[1]),
                self.data[2].sub(other.data[2]),
                self.data[3].sub(other.data[3]),
            },
        };
    }

    pub fn scale(self: Self, scalar: f32) Self {
        return Self{
            .data = .{
                self.data[0].scale(scalar),
                self.data[1].scale(scalar),
                self.data[2].scale(scalar),
                self.data[3].scale(scalar),
            },
        };
    }

    pub fn frobeniusNorm(self: Self) f32 {
        var sum: f32 = 0;
        var i: usize = 0; while (i < 4) : (i += 1) {
            const row_vec = self.data[i];
            const squared = row_vec.mul(row_vec);
            sum += squared.reduce_add();
        }
        return @sqrt(sum);
    }
};

pub const MatrixOps = struct {
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{ .allocator = allocator };
    }

    pub fn matmul4x4(self: *Self, a: Matrix4x4, b: Matrix4x4) Matrix4x4 {
        _ = self;
        var result = Matrix4x4.zero();
        var i: usize = 0; while (i < 4) : (i += 1) {
            var j: usize = 0; while (j < 4) : (j += 1) {
                var sum: f32 = 0;
                var k: usize = 0; while (k < 4) : (k += 1) {
                    sum += a.get(i, k) * b.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    pub fn matmul4x4Simd(self: *Self, a: Matrix4x4, b: Matrix4x4) Matrix4x4 {
        const b_t = self.transpose4x4(b);
        var result: [4]F32x4 = undefined;
        var i: usize = 0; while (i < 4) : (i += 1) {
            const row_a = a.data[i];
            result[i] = F32x4.initFromArray(.{
                row_a.dot(b_t.data[0]),
                row_a.dot(b_t.data[1]),
                row_a.dot(b_t.data[2]),
                row_a.dot(b_t.data[3]),
            });
        }
        return Matrix4x4{ .data = result };
    }

    pub fn transpose4x4(self: *Self, m: Matrix4x4) Matrix4x4 {
        _ = self;
        var result = Matrix4x4.zero();
        var i: usize = 0; while (i < 4) : (i += 1) {
            var j: usize = 0; while (j < 4) : (j += 1) {
                result.set(j, i, m.get(i, j));
            }
        }
        return result;
    }

    pub fn determinant4x4(self: *Self, m: Matrix4x4) f32 {
        var det: f32 = 0;
        var i: usize = 0; while (i < 4) : (i += 1) {
            const minor = self.minor3x3(m, 0, i);
            const sign: f32 = if (i % 2 == 0) 1.0 else -1.0;
            det += sign * m.get(0, i) * minor;
        }
        return det;
    }

    fn minor3x3(self: *Self, m: Matrix4x4, skip_row: usize, skip_col: usize) f32 {
        _ = self;
        var submatrix: [3][3]f32 = undefined;
        var si: usize = 0;
        var i: usize = 0; while (i < 4) : (i += 1) {
            if (i == skip_row) continue;
            var sj: usize = 0;
            var j: usize = 0; while (j < 4) : (j += 1) {
                if (j == skip_col) continue;
                submatrix[si][sj] = m.get(i, j);
                sj += 1;
            }
            si += 1;
        }
        return submatrix[0][0] * (submatrix[1][1] * submatrix[2][2] - submatrix[1][2] * submatrix[2][1]) -
            submatrix[0][1] * (submatrix[1][0] * submatrix[2][2] - submatrix[1][2] * submatrix[2][0]) +
            submatrix[0][2] * (submatrix[1][0] * submatrix[2][1] - submatrix[1][1] * submatrix[2][0]);
    }

    pub fn inverse4x4(self: *Self, m: Matrix4x4) ?Matrix4x4 {
        const det = self.determinant4x4(m);
        const norm = m.frobeniusNorm();
        const threshold = if (norm > 1.0) 1e-10 * norm else 1e-10;
        if (if (det >= 0) det else -det < threshold) {
            return null;
        }
        var adj = Matrix4x4.zero();
        var i: usize = 0; while (i < 4) : (i += 1) {
            var j: usize = 0; while (j < 4) : (j += 1) {
                const minor = self.minor3x3(m, i, j);
                const sign: f32 = if ((i + j) % 2 == 0) 1.0 else -1.0;
                adj.set(j, i, sign * minor / det);
            }
        }
        return adj;
    }

    pub fn qr_decomposition(self: *Self, m: Matrix4x4) struct { q: Matrix4x4, r: Matrix4x4 } {
        _ = self;
        var q = Matrix4x4.identity();
        var r = m;
        var k: usize = 0;
        while (k < 4) : (k += 1) {
            var col_k: [4]f32 = undefined;
            {
                var idx: usize = 0;
                while (idx < 4) : (idx += 1) {
                    col_k[idx] = r.get(idx, k);
                }
            }
            {
                var j_idx: usize = 0;
                while (j_idx < k) : (j_idx += 1) {
                    var dot_prod: f32 = 0;
                    {
                        var idx: usize = 0;
                        while (idx < 4) : (idx += 1) {
                            dot_prod += q.get(idx, j_idx) * col_k[idx];
                        }
                    }
                    {
                        var idx: usize = 0;
                        while (idx < 4) : (idx += 1) {
                            col_k[idx] -= dot_prod * q.get(idx, j_idx);
                        }
                    }
                    {
                        var idx: usize = 0;
                        while (idx < 4) : (idx += 1) {
                            dot_prod = 0;
                            var ii: usize = 0;
                            while (ii < 4) : (ii += 1) {
                                dot_prod += q.get(ii, j_idx) * col_k[ii];
                            }
                            col_k[idx] -= dot_prod * q.get(idx, j_idx);
                        }
                    }
                }
            }
            var norm_sq: f32 = 0;
            for (col_k) |v| {
                norm_sq += v * v;
            }
            const norm_val = @sqrt(norm_sq);
            const rkk = r.get(k, k);
            const norm_threshold = 1e-10 * @max(1.0, if (rkk >= 0) rkk else -rkk);
            if (norm_val > norm_threshold) {
                var idx: usize = 0;
                while (idx < 4) : (idx += 1) {
                    q.set(idx, k, col_k[idx] / norm_val);
                }
            }
            {
                var j_idx: usize = k;
                while (j_idx < 4) : (j_idx += 1) {
                    var dot_prod: f32 = 0;
                    var idx: usize = 0;
                    while (idx < 4) : (idx += 1) {
                        dot_prod += q.get(idx, k) * r.get(idx, j_idx);
                    }
                    r.set(k, j_idx, dot_prod);
                }
            }
            {
                var idx: usize = k + 1;
                while (idx < 4) : (idx += 1) {
                    r.set(idx, k, 0);
                }
            }
        }
        return .{ .q = q, .r = r };
    }
};

pub const RelationalVectorOps = struct {
    allocator: Allocator,
    similarity_weight_phase: f64,
    similarity_weight_magnitude: f64,
    similarity_weight_quantum: f64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .allocator = allocator,
            .similarity_weight_phase = 0.3,
            .similarity_weight_magnitude = 0.3,
            .similarity_weight_quantum = 0.4,
        };
    }

    pub fn initWithWeights(allocator: Allocator, phase_w: f64, mag_w: f64, quantum_w: f64) Self {
        return Self{
            .allocator = allocator,
            .similarity_weight_phase = phase_w,
            .similarity_weight_magnitude = mag_w,
            .similarity_weight_quantum = quantum_w,
        };
    }

    pub fn computeNodeSimilarity(self: *Self, n1: *const Node, n2: *const Node) f64 {
        const pd = n1.phase - n2.phase;
        const phase_diff = if (pd >= 0) pd else -pd;
        const normalized_phase = @min(phase_diff, 2.0 * std.math.pi) / (2.0 * std.math.pi);
        const md = n1.magnitude() - n2.magnitude();
        const mag_diff = if (md >= 0) md else -md;
        const abs_n1 = if (n1.magnitude() >= 0) n1.magnitude() else -n1.magnitude();
        const abs_n2 = if (n2.magnitude() >= 0) n2.magnitude() else -n2.magnitude();
        const max_mag = @max(abs_n1, abs_n2);
        const normalized_mag = if (max_mag > 0) mag_diff / max_mag else 0.0;
        const n1_mag = n1.quantumState().magnitude();
        const n2_mag = n2.quantumState().magnitude();
        const inner_prod = n1.quantumState().mul(n2.quantumState().conjugate()).magnitude();
        const normalized_quantum = if (n1_mag > 0 and n2_mag > 0)
            inner_prod / (n1_mag * n2_mag)
        else
            0.0;
        const weighted_sum = self.similarity_weight_phase * normalized_phase +
            self.similarity_weight_magnitude * normalized_mag +
            self.similarity_weight_quantum * (1.0 - normalized_quantum);
        return @max(0.0, @min(1.0, 1.0 - weighted_sum));
    }

    pub fn computeEdgeVectorBatch(self: *Self, edges: []const *Edge) !ArrayList(F64x4) {
        var batch = ArrayList(F64x4).init(self.allocator);
        errdefer batch.deinit();
        for (edges) |edge| {
            const vec = F64x4.initFromArray(.{
                edge.weight,
                edge.quantum_coupling.re,
                edge.quantum_coupling.im,
                edge.fractal_dimension,
            });
            try batch.append(vec);
        }
        return batch;
    }

    pub fn vectorizeGraph(self: *Self, graph: *SelfSimilarRelationalGraph) !ArrayList(F64x4) {
        var embeddings = ArrayList(F64x4).init(self.allocator);
        errdefer embeddings.deinit();
        var sorted_keys = ArrayList([]const u8).init(self.allocator);
        defer sorted_keys.deinit();
        var iter = graph.nodes.iterator();
        while (iter.next()) |entry| {
            try sorted_keys.append(entry.key_ptr.*);
        }
        std.mem.sort([]const u8, sorted_keys.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lessThan);
        for (sorted_keys.items) |key| {
            if (graph.nodes.getPtr(key)) |node| {
                const vec = F64x4.initFromArray(.{
                    node.phase,
                    node.magnitude(),
                    node.quantumState().re,
                    node.quantumState().im,
                });
                try embeddings.append(vec);
            }
        }
        return embeddings;
    }

    pub fn parallelDotProduct(self: *Self, vectors1: []const F64x4, vectors2: []const F64x4) !ArrayList(f64) {
        var results = ArrayList(f64).init(self.allocator);
        _ = &self.similarity_weight_phase;
        errdefer results.deinit();
        const count = @min(vectors1.len, vectors2.len);
        var i: usize = 0; while (i < count) : (i += 1) {
            try results.append(vectors1[i].dot(vectors2[i]));
        }
        return results;
    }

    pub fn normalizeVectors(self: *Self, vectors: []F64x4) void {
        _ = self;
        for (vectors) |*vec| {
            vec.* = vec.normalize();
        }
    }

    pub fn applyQuantumRotation(self: *Self, vec: F64x4, theta: f64, phi: f64) F64x4 {
        _ = self;
        const cos_t = @cos(theta);
        const sin_t = @sin(theta);
        const cos_p = @cos(phi);
        const sin_p = @sin(phi);
        const arr = vec.toArray();
        return F64x4.initFromArray(.{
            arr[0] * cos_t - arr[1] * sin_t,
            arr[0] * sin_t + arr[1] * cos_t,
            arr[2] * cos_p - arr[3] * sin_p,
            arr[2] * sin_p + arr[3] * cos_p,
        });
    }

    pub fn computeGraphLaplacian(self: *Self, adjacency: []const []const f64, n: usize) !ArrayList(ArrayList(f64)) {
        var laplacian = ArrayList(ArrayList(f64)).init(self.allocator);
        errdefer {
            for (laplacian.items) |*row| {
                row.deinit();
            }
            laplacian.deinit();
        }
        var i: usize = 0; while (i < n) : (i += 1) {
            var row_list = ArrayList(f64).init(self.allocator);
            errdefer row_list.deinit();
            var degree: f64 = 0;
            {
                var j: usize = 0; while (j < n) : (j += 1) {
                    const adj_val = if (i < adjacency.len and j < adjacency[i].len)
                        adjacency[i][j]
                    else
                        0.0;
                    if (!std.math.isNan(adj_val) and !std.math.isInf(adj_val) and adj_val >= 0) {
                        degree += adj_val;
                    }
                }
            }
            var j: usize = 0; while (j < n) : (j += 1) {
                const adj_val = if (i < adjacency.len and j < adjacency[i].len)
                    adjacency[i][j]
                else
                    0.0;
                const safe_adj = if (std.math.isNan(adj_val) or std.math.isInf(adj_val) or adj_val < 0)
                    0.0
                else
                    adj_val;
                if (i == j) {
                    try row_list.append(degree - safe_adj);
                } else {
                    try row_list.append(-safe_adj);
                }
            }
            try laplacian.append(row_list);
        }
        return laplacian;
    }

    pub fn spectralEmbedding(self: *Self, adjacency: []const []const f64, n: usize, dimensions: usize) !ArrayList(F64x4) {
        var embeddings = ArrayList(F64x4).init(self.allocator);
        errdefer embeddings.deinit();
        var laplacian = try self.computeGraphLaplacian(adjacency, n);
        defer {
            for (laplacian.items) |*row| {
                row.deinit();
            }
            laplacian.deinit();
        }
        const actual_dims = @min(dimensions, 4);
        var i: usize = 0; while (i < n) : (i += 1) {
            var embedding: [4]f64 = .{ 0, 0, 0, 0 };
            var d: usize = 0; while (d < actual_dims) : (d += 1) {
                if (i < laplacian.items.len and d < laplacian.items[i].items.len) {
                    embedding[d] = laplacian.items[i].items[d];
                }
            }
            const vec = F64x4.initFromArray(embedding);
            try embeddings.append(vec.normalize());
        }
        return embeddings;
    }
};

pub const MemorySlice = struct {
    offset: usize,
    size: usize,
    in_use: bool,
    allocation_size: usize,
};

pub const MemoryPool = struct {
    pool: []align(32) u8,
    free_list: ArrayList(MemorySlice),
    allocator: Allocator,
    total_allocated: usize,
    pool_size: usize,

    const Self = @This();
    const MIN_ALLOCATION_SIZE: usize = 32;

    pub fn init(allocator: Allocator, pool_size: usize) !Self {
        const aligned_size = std.mem.alignForward(usize, pool_size, 32);
        const pool = try allocator.alignedAlloc(u8, 32, aligned_size);
        @memset(pool, 0);
        var free_list = ArrayList(MemorySlice).init(allocator);
        try free_list.append(MemorySlice{
            .offset = 0,
            .size = aligned_size,
            .in_use = false,
            .allocation_size = 0,
        });
        return Self{
            .pool = pool,
            .free_list = free_list,
            .allocator = allocator,
            .total_allocated = 0,
            .pool_size = aligned_size,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.pool);
        self.free_list.deinit();
    }

    pub fn alloc(self: *Self, size: usize) ?[]align(32) u8 {
        const aligned_size = std.mem.alignForward(usize, @max(size, MIN_ALLOCATION_SIZE), 32);
        var i: usize = 0; while (i < self.free_list.items.len) : (i += 1) {
            var slice = &self.free_list.items[i];
            if (!slice.in_use and slice.size >= aligned_size) {
                if (slice.size > aligned_size + MIN_ALLOCATION_SIZE) {
                    const new_slice = MemorySlice{
                        .offset = slice.offset + aligned_size,
                        .size = slice.size - aligned_size,
                        .in_use = false,
                        .allocation_size = 0,
                    };
                    self.free_list.insert(i + 1, new_slice) catch {
                        return null;
                    };
                    slice = &self.free_list.items[i];
                    slice.size = aligned_size;
                }
                slice.in_use = true;
                slice.allocation_size = aligned_size;
                self.total_allocated += aligned_size;
                const start = slice.offset;
                const ptr: [*]align(32) u8 = @ptrCast(@alignCast(self.pool.ptr + start));
                return ptr[0..aligned_size];
            }
        }
        return null;
    }

    pub fn free(self: *Self, ptr: []align(32) u8) bool {
        const ptr_addr = @intFromPtr(ptr.ptr);
        const pool_addr = @intFromPtr(self.pool.ptr);
        if (ptr_addr < pool_addr or ptr_addr >= pool_addr + self.pool_size) {
            return false;
        }
        const offset = ptr_addr - pool_addr;
        for (self.free_list.items) |*slice| {
            if (slice.offset == offset and slice.in_use) {
                const actual_size = slice.allocation_size;
                @memset(ptr[0..@min(ptr.len, actual_size)], 0);
                slice.in_use = false;
                self.total_allocated -|= actual_size;
                self.coalesceFreeBlocks();
                return true;
            }
        }
        return false;
    }

    fn coalesceFreeBlocks(self: *Self) void {
        if (self.free_list.items.len < 2) return;
        var i: usize = 0;
        while (i < self.free_list.items.len - 1) {
            const current = &self.free_list.items[i];
            const next = &self.free_list.items[i + 1];
            if (!current.in_use and !next.in_use and current.offset + current.size == next.offset) {
                current.size += next.size;
                _ = self.free_list.orderedRemove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    pub fn getAllocatedSize(self: *const Self) usize {
        return self.total_allocated;
    }

    pub fn getPoolSize(self: *const Self) usize {
        return self.pool_size;
    }

    pub fn getFreeSize(self: *const Self) usize {
        var free_contiguous: usize = 0;
        for (self.free_list.items) |slice| {
            if (!slice.in_use) {
                free_contiguous += slice.size;
            }
        }
        return free_contiguous;
    }

    pub fn reset(self: *Self) void {
        @memset(self.pool, 0);
        self.free_list.clearRetainingCapacity();
        self.free_list.append(MemorySlice{
            .offset = 0,
            .size = self.pool_size,
            .in_use = false,
            .allocation_size = 0,
        }) catch {};
        self.total_allocated = 0;
    }
};

pub const VPUStatistics = struct {
    operations_completed: usize,
    simd_instructions_used: usize,
    cache_hits: usize,
    cache_misses: usize,
    memory_allocated: usize,
    memory_freed: usize,
    vectors_processed: usize,
    matrix_operations: usize,
    graph_operations: usize,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .operations_completed = 0,
            .simd_instructions_used = 0,
            .cache_hits = 0,
            .cache_misses = 0,
            .memory_allocated = 0,
            .memory_freed = 0,
            .vectors_processed = 0,
            .matrix_operations = 0,
            .graph_operations = 0,
        };
    }

    pub fn reset(self: *Self) void {
        self.operations_completed = 0;
        self.simd_instructions_used = 0;
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.memory_allocated = 0;
        self.memory_freed = 0;
        self.vectors_processed = 0;
        self.matrix_operations = 0;
        self.graph_operations = 0;
    }

    pub fn clone(self: *const Self) Self {
        return Self{
            .operations_completed = self.operations_completed,
            .simd_instructions_used = self.simd_instructions_used,
            .cache_hits = self.cache_hits,
            .cache_misses = self.cache_misses,
            .memory_allocated = self.memory_allocated,
            .memory_freed = self.memory_freed,
            .vectors_processed = self.vectors_processed,
            .matrix_operations = self.matrix_operations,
            .graph_operations = self.graph_operations,
        };
    }

    pub fn mergeChecked(self: *Self, other: *const Self) bool {
        const max_usize = std.math.maxInt(usize);
        if (self.operations_completed > max_usize - other.operations_completed) return false;
        if (self.simd_instructions_used > max_usize - other.simd_instructions_used) return false;
        if (self.cache_hits > max_usize - other.cache_hits) return false;
        if (self.cache_misses > max_usize - other.cache_misses) return false;
        if (self.memory_allocated > max_usize - other.memory_allocated) return false;
        if (self.memory_freed > max_usize - other.memory_freed) return false;
        if (self.vectors_processed > max_usize - other.vectors_processed) return false;
        if (self.matrix_operations > max_usize - other.matrix_operations) return false;
        if (self.graph_operations > max_usize - other.graph_operations) return false;
        self.operations_completed += other.operations_completed;
        self.simd_instructions_used += other.simd_instructions_used;
        self.cache_hits += other.cache_hits;
        self.cache_misses += other.cache_misses;
        self.memory_allocated += other.memory_allocated;
        self.memory_freed += other.memory_freed;
        self.vectors_processed += other.vectors_processed;
        self.matrix_operations += other.matrix_operations;
        self.graph_operations += other.graph_operations;
        return true;
    }

    pub fn merge(self: *Self, other: *const Self) void {
        self.operations_completed +|= other.operations_completed;
        self.simd_instructions_used +|= other.simd_instructions_used;
        self.cache_hits +|= other.cache_hits;
        self.cache_misses +|= other.cache_misses;
        self.memory_allocated +|= other.memory_allocated;
        self.memory_freed +|= other.memory_freed;
        self.vectors_processed +|= other.vectors_processed;
        self.matrix_operations +|= other.matrix_operations;
        self.graph_operations +|= other.graph_operations;
    }

    pub fn getCacheHitRate(self: *const Self) f64 {
        const total = self.cache_hits + self.cache_misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.cache_hits)) / @as(f64, @floatFromInt(total));
    }

    pub fn getSimdEfficiency(self: *const Self) f64 {
        if (self.operations_completed == 0) return 0.0;
        return @as(f64, @floatFromInt(self.simd_instructions_used)) /
            @as(f64, @floatFromInt(self.operations_completed));
    }

    pub fn getNetMemoryUsage(self: *const Self) usize {
        return self.memory_allocated -| self.memory_freed;
    }
};

pub const VectorCache = struct {
    entries: std.AutoHashMap(u64, CacheEntry),
    max_entries: usize,
    lru_queue: ArrayList(u64),
    allocator: Allocator,

    const Self = @This();

    const CacheEntry = struct {
        data: []u8,
        vector_type: VectorType,
        timestamp: i64,
    };

    pub fn init(allocator: Allocator, max_entries: usize) Self {
        return Self{
            .entries = std.AutoHashMap(u64, CacheEntry).init(allocator),
            .max_entries = max_entries,
            .lru_queue = ArrayList(u64).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.data);
        }
        self.entries.deinit();
        self.lru_queue.deinit();
    }

    pub fn get(self: *Self, key: u64) ?CacheEntry {
        if (self.entries.get(key)) |entry| {
            self.updateLRU(key);
            return CacheEntry{
                .data = entry.data,
                .vector_type = entry.vector_type,
                .timestamp = entry.timestamp,
            };
        }
        return null;
    }

    pub fn put(self: *Self, key: u64, data: []const u8, vector_type: VectorType) !void {
        if (self.entries.get(key)) |existing| {
            self.allocator.free(existing.data);
            _ = self.entries.remove(key);
            var i: usize = 0; while (i < self.lru_queue.items.len) : (i += 1) {
                if (self.lru_queue.items[i] == key) {
                    _ = self.lru_queue.orderedRemove(i);
                    break;
                }
            }
        }
        if (self.entries.count() >= self.max_entries) {
            try self.evictLRU();
        }
        const data_copy = try self.allocator.dupe(u8, data);
        errdefer self.allocator.free(data_copy);
        try self.entries.put(key, CacheEntry{
            .data = data_copy,
            .vector_type = vector_type,
            .timestamp = @as(i64, @intCast(std.time.nanoTimestamp())),
        });
        try self.lru_queue.append(key);
    }

    fn updateLRU(self: *Self, key: u64) void {
        var i: usize = 0; while (i < self.lru_queue.items.len) : (i += 1) {
            if (self.lru_queue.items[i] == key) {
                _ = self.lru_queue.orderedRemove(i);
                self.lru_queue.append(key) catch {};
                return;
            }
        }
    }

    fn evictLRU(self: *Self) !void {
        while (self.lru_queue.items.len > 0 and self.entries.count() >= self.max_entries) {
            const key = self.lru_queue.orderedRemove(0);
            if (self.entries.fetchRemove(key)) |removed| {
                self.allocator.free(removed.value.data);
            }
        }
    }

    pub fn clear(self: *Self) void {
        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.data);
        }
        self.entries.clearRetainingCapacity();
        self.lru_queue.clearRetainingCapacity();
    }

    pub fn contains(self: *const Self, key: u64) bool {
        return self.entries.contains(key);
    }

    pub fn count(self: *const Self) usize {
        return self.entries.count();
    }
};

pub const VPU = struct {
    memory_pool: MemoryPool,
    vector_batch: VectorBatch,
    statistics: VPUStatistics,
    matrix_ops: MatrixOps,
    relational_ops: RelationalVectorOps,
    vector_cache: VectorCache,
    allocator: Allocator,
    cycle_count: usize,
    instruction_pointer: usize,

    const Self = @This();
    const DEFAULT_POOL_SIZE: usize = 1024 * 1024;
    const DEFAULT_BATCH_SIZE: usize = 256;
    const DEFAULT_CACHE_SIZE: usize = 1024;

    pub fn init(allocator: Allocator) !Self {
        return try initWithOptions(allocator, DEFAULT_POOL_SIZE, DEFAULT_BATCH_SIZE, DEFAULT_CACHE_SIZE);
    }

    pub fn initWithOptions(
        allocator: Allocator,
        pool_size: usize,
        batch_size: usize,
        cache_size: usize,
    ) !Self {
        var memory_pool = try MemoryPool.init(allocator, pool_size);
        errdefer memory_pool.deinit();
        var vector_batch = VectorBatch.init(allocator, batch_size);
        errdefer vector_batch.deinit();
        const matrix_ops = MatrixOps.init(allocator);
        const relational_ops = RelationalVectorOps.init(allocator);
        var vector_cache = VectorCache.init(allocator, cache_size);
        errdefer vector_cache.deinit();
        return Self{
            .memory_pool = memory_pool,
            .vector_batch = vector_batch,
            .statistics = VPUStatistics.init(),
            .matrix_ops = matrix_ops,
            .relational_ops = relational_ops,
            .vector_cache = vector_cache,
            .allocator = allocator,
            .cycle_count = 0,
            .instruction_pointer = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.memory_pool.deinit();
        self.vector_batch.deinit();
        self.vector_cache.deinit();
    }

    pub fn processVectors(self: *Self, operation: BatchOperation) !void {
        const count_before = self.vector_batch.count();
        try self.vector_batch.processBatch(operation);
        self.statistics.vectors_processed += count_before;
        self.cycle_count += 1;
    }

    pub fn batchMatmul(self: *Self, matrices_a: []const Matrix4x4, matrices_b: []const Matrix4x4) !ArrayList(Matrix4x4) {
        var results = ArrayList(Matrix4x4).init(self.allocator);
        errdefer results.deinit();
        const count_val = @min(matrices_a.len, matrices_b.len);
        var i: usize = 0; while (i < count_val) : (i += 1) {
            const result = self.matrix_ops.matmul4x4Simd(matrices_a[i], matrices_b[i]);
            try results.append(result);
            self.statistics.matrix_operations += 1;
        }
        self.cycle_count += count_val;
        return results;
    }

    pub fn computeGraphEmbeddings(self: *Self, graph: *SelfSimilarRelationalGraph) !ArrayList(F64x4) {
        const embeddings = try self.relational_ops.vectorizeGraph(graph);
        self.relational_ops.normalizeVectors(embeddings.items);
        self.statistics.graph_operations += 1;
        self.cycle_count += 1;
        return embeddings;
    }

    pub fn quantumVectorOps(self: *Self, vectors: []F64x4, theta: f64, phi: f64) void {
        for (vectors) |*vec| {
            vec.* = self.relational_ops.applyQuantumRotation(vec.*, theta, phi);
        }
        self.statistics.vectors_processed += vectors.len;
        self.cycle_count += 1;
    }

    pub fn addF32x4(self: *Self, vec: F32x4) !usize {
        const idx = try self.vector_batch.addF32x4(vec);
        self.statistics.memory_allocated += VectorType.f32x4.totalSize();
        return idx;
    }

    pub fn addF64x4(self: *Self, vec: F64x4) !usize {
        const idx = try self.vector_batch.addF64x4(vec);
        self.statistics.memory_allocated += VectorType.f64x4.totalSize();
        return idx;
    }

    pub fn computeSimilarityMatrix(self: *Self, embeddings: []const F64x4) !ArrayList(ArrayList(f64)) {
        var similarity_matrix = ArrayList(ArrayList(f64)).init(self.allocator);
        errdefer {
            for (similarity_matrix.items) |*row| {
                row.deinit();
            }
            similarity_matrix.deinit();
        }
        var magnitudes = ArrayList(f64).init(self.allocator);
        defer magnitudes.deinit();
        for (embeddings) |emb| {
            try magnitudes.append(emb.magnitude());
        }
        var i: usize = 0; while (i < embeddings.len) : (i += 1) {
            var row = ArrayList(f64).init(self.allocator);
            errdefer row.deinit();
            const mag_i = magnitudes.items[i];
            var j: usize = 0; while (j < embeddings.len) : (j += 1) {
                const dot = embeddings[i].dot(embeddings[j]);
                const mag_j = magnitudes.items[j];
                const denom = mag_i * mag_j;
                const similarity = if (denom > 1e-10 and !std.math.isNan(denom))
                    dot / denom
                else
                    0;
                try row.append(similarity);
            }
            try similarity_matrix.append(row);
        }
        self.statistics.operations_completed += 1;
        self.cycle_count += 1;
        return similarity_matrix;
    }

    pub fn powerIteration(self: *Self, m: Matrix4x4, iterations: usize) F32x4 {
        var v = F32x4.initFromArray(.{ 1, 0, 0, 0 }).normalize();
        var prev_v = v;
        var iter_idx: usize = 0; while (iter_idx < iterations) : (iter_idx += 1) {
            var result = F32x4.init(0);
            var i: usize = 0; while (i < 4) : (i += 1) {
                const row_vec = m.row(i);
                const dot = row_vec.dot(v);
                result.set(i, dot);
            }
            v = result.normalize();
            const diff = v.sub(prev_v);
            if (diff.magnitude() < 1e-10) break;
            prev_v = v;
        }
        self.statistics.operations_completed += 1;
        return v;
    }

    pub fn getStatistics(self: *const Self) VPUStatistics {
        return self.statistics.clone();
    }

    pub fn reset(self: *Self) void {
        self.statistics.reset();
        self.memory_pool.reset();
        self.vector_batch.clear();
        self.vector_cache.clear();
        self.cycle_count = 0;
        self.instruction_pointer = 0;
    }

    pub fn getCycleCount(self: *const Self) usize {
        return self.cycle_count;
    }

    pub fn getMemoryUsage(self: *const Self) usize {
        return self.memory_pool.getAllocatedSize();
    }

    pub fn allocSimdAligned(self: *Self, size: usize) ?[]align(32) u8 {
        const result = self.memory_pool.alloc(size);
        if (result) |ptr| {
            self.statistics.memory_allocated += ptr.len;
        }
        return result;
    }

    pub fn freeSimdAligned(self: *Self, ptr: []align(32) u8) void {
        if (self.memory_pool.free(ptr)) {
            self.statistics.memory_freed += ptr.len;
        }
    }
};

pub const LNSValue = struct {
    mantissa: f64,
    exponent: f64,
    sign: bool,

    const Self = @This();

    pub fn zero() Self {
        return Self{
            .mantissa = -std.math.inf(f64),
            .exponent = 1.0,
            .sign = true,
        };
    }

    pub fn fromFloat(value: f64) Self {
        if (value == 0.0 or value == -0.0) {
            return Self.zero();
        }
        const sign = value >= 0.0;
        const abs_val = if (value >= 0) value else -value;
        return Self{
            .mantissa = @log(abs_val),
            .exponent = 1.0,
            .sign = sign,
        };
    }

    pub fn toFloat(self: Self) f64 {
        if (std.math.isInf(self.mantissa) and self.mantissa < 0.0) {
            return 0.0;
        }
        const magnitude = @exp(self.mantissa * self.exponent);
        return if (self.sign) magnitude else -magnitude;
    }

    pub fn add(self: Self, other: Self) Self {
        const f1 = self.toFloat();
        const f2 = other.toFloat();
        return Self.fromFloat(f1 + f2);
    }

    pub fn mul(self: Self, other: Self) Self {
        if ((std.math.isInf(self.mantissa) and self.mantissa < 0) or
            (std.math.isInf(other.mantissa) and other.mantissa < 0))
        {
            return Self.zero();
        }
        return Self{
            .mantissa = self.mantissa + other.mantissa,
            .exponent = 1.0,
            .sign = self.sign == other.sign,
        };
    }

    pub fn divChecked(self: Self, other: Self) ?Self {
        if (std.math.isInf(other.mantissa) and other.mantissa < 0) {
            return null;
        }
        return Self{
            .mantissa = self.mantissa - other.mantissa,
            .exponent = 1.0,
            .sign = self.sign == other.sign,
        };
    }

    pub fn div(self: Self, other: Self) Self {
        if (std.math.isInf(other.mantissa) and other.mantissa < 0) {
            return Self{
                .mantissa = std.math.inf(f64),
                .exponent = 1.0,
                .sign = self.sign == other.sign,
            };
        }
        return Self{
            .mantissa = self.mantissa - other.mantissa,
            .exponent = 1.0,
            .sign = self.sign == other.sign,
        };
    }

    pub fn isZero(self: Self) bool {
        return std.math.isInf(self.mantissa) and self.mantissa < 0;
    }

    pub fn abs(self: Self) Self {
        return Self{
            .mantissa = self.mantissa,
            .exponent = self.exponent,
            .sign = true,
        };
    }

    pub fn negate(self: Self) Self {
        return Self{
            .mantissa = self.mantissa,
            .exponent = self.exponent,
            .sign = !self.sign,
        };
    }
};

pub const LNSInstruction = union(enum) {
    rsf_scatter: struct { src: usize, dst: usize, perm_indices: usize },
    rsf_affine_couple: struct { src: usize, dst: usize, s_weight: usize, t_weight: usize },
    tensor_load: struct { addr: usize, dst: usize },
    tensor_store: struct { src: usize, addr: usize },
    lns_add: struct { src1: usize, src2: usize, dst: usize },
    lns_mul: struct { src1: usize, src2: usize, dst: usize },
    graph_transform: struct { node_id: usize, transform_type: usize },
    jump: struct { offset: isize },
    conditional_jump: struct { condition_reg: usize, offset: isize },
    halt: void,
};

test "SimdVector basic operations" {
    const v1 = F32x4.initFromArray(.{ 1, 2, 3, 4 });
    const v2 = F32x4.initFromArray(.{ 5, 6, 7, 8 });
    const sum = v1.add(v2);
    const sum_arr = sum.toArray();
    try std.testing.expectEqual(@as(f32, 6), sum_arr[0]);
    try std.testing.expectEqual(@as(f32, 8), sum_arr[1]);
    try std.testing.expectEqual(@as(f32, 10), sum_arr[2]);
    try std.testing.expectEqual(@as(f32, 12), sum_arr[3]);
    const dot = v1.dot(v2);
    try std.testing.expectEqual(@as(f32, 70), dot);
    const mag = v1.magnitude();
    try std.testing.expectApproxEqAbs(@as(f32, 5.477), mag, 0.01);
}

test "Matrix4x4 operations" {
    var ops = MatrixOps.init(std.testing.allocator);
    const m1 = Matrix4x4.identity();
    const m2 = Matrix4x4.identity();
    const result = ops.matmul4x4(m1, m2);
    try std.testing.expectEqual(@as(f32, 1), result.get(0, 0));
    try std.testing.expectEqual(@as(f32, 1), result.get(1, 1));
    try std.testing.expectEqual(@as(f32, 1), result.get(2, 2));
    try std.testing.expectEqual(@as(f32, 1), result.get(3, 3));
    try std.testing.expectEqual(@as(f32, 0), result.get(0, 1));
}

test "MemoryPool allocation" {
    const allocator = std.testing.allocator;
    var pool = try MemoryPool.init(allocator, 4096);
    defer pool.deinit();
    const ptr1 = pool.alloc(128);
    try std.testing.expect(ptr1 != null);
    try std.testing.expect(pool.getAllocatedSize() >= 128);
    const ptr2 = pool.alloc(256);
    try std.testing.expect(ptr2 != null);
    if (ptr1) |p1| {
        _ = pool.free(p1);
    }
    try std.testing.expect(pool.getAllocatedSize() < pool.getPoolSize());
}

test "VPU initialization" {
    const allocator = std.testing.allocator;
    var vpu = try VPU.init(allocator);
    defer vpu.deinit();
    try std.testing.expect(vpu.cycle_count == 0);
    try std.testing.expect(vpu.statistics.operations_completed == 0);
}

test "VectorBatch operations" {
    const allocator = std.testing.allocator;
    var batch = VectorBatch.init(allocator, 16);
    defer batch.deinit();
    const v1 = F32x4.initFromArray(.{ 1, 2, 3, 4 });
    const idx = try batch.addF32x4(v1);
    try std.testing.expect(idx == 0);
    try std.testing.expect(batch.count() == 1);
}

test "LNSValue operations" {
    const v1 = LNSValue.fromFloat(2.0);
    const v2 = LNSValue.fromFloat(3.0);
    const product = v1.mul(v2);
    const result = product.toFloat();
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), result, 0.0001);
    const sum = v1.add(v2);
    const sum_result = sum.toFloat();
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), sum_result, 0.0001);
}

test "SimdVector bounds checking" {
    const v = F32x4.initFromArray(.{ 1, 2, 3, 4 });
    try std.testing.expectEqual(@as(f32, 1), v.get(0));
    try std.testing.expectEqual(@as(f32, 0), v.get(10));
    try std.testing.expectError(VectorError.OutOfBounds, v.getChecked(10));
}

test "VectorCache put and get" {
    const allocator = std.testing.allocator;
    var cache = VectorCache.init(allocator, 10);
    defer cache.deinit();
    const data = [_]u8{ 1, 2, 3, 4 };
    try cache.put(123, &data, .f32x4);
    try std.testing.expect(cache.contains(123));
    const entry = cache.get(123);
    try std.testing.expect(entry != null);
}

