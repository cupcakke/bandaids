const std = @import("std");
const math = std.math;
const mem = std.mem;
const meta = std.meta;
const testing = std.testing;
const Allocator = mem.Allocator;

pub const Tensor = @import("tensor.zig").Tensor;

pub const FixedPointError = error{ Overflow, DivisionByZero };

pub const FixedPoint16 = packed struct {
    value: i16,

    pub fn fromFloat(f: f32) FixedPointError!FixedPoint16 {
        const scaled = f * 256.0;
        if (scaled > @as(f32, @floatFromInt(math.maxInt(i16))) or scaled < @as(f32, @floatFromInt(math.minInt(i16)))) {
            return FixedPointError.Overflow;
        }
        return .{ .value = @intFromFloat(scaled) };
    }

    pub fn toFloat(self: FixedPoint16) f32 {
        return @as(f32, @floatFromInt(self.value)) / 256.0;
    }

    pub fn add(a: FixedPoint16, b: FixedPoint16) FixedPointError!FixedPoint16 {
        const r = @addWithOverflow(a.value, b.value);
        if (r[1] != 0) return FixedPointError.Overflow;
        return .{ .value = r[0] };
    }

    pub fn sub(a: FixedPoint16, b: FixedPoint16) FixedPointError!FixedPoint16 {
        const r = @subWithOverflow(a.value, b.value);
        if (r[1] != 0) return FixedPointError.Overflow;
        return .{ .value = r[0] };
    }

    pub fn mul(a: FixedPoint16, b: FixedPoint16) FixedPointError!FixedPoint16 {
        const wide: i32 = @as(i32, a.value) * @as(i32, b.value);
        const result: i32 = wide >> 8;
        if (result > math.maxInt(i16)) return FixedPointError.Overflow;
        if (result < math.minInt(i16)) return FixedPointError.Overflow;
        return .{ .value = @intCast(result) };
    }

    pub fn div(a: FixedPoint16, b: FixedPoint16) FixedPointError!FixedPoint16 {
        if (b.value == 0) return FixedPointError.DivisionByZero;
        const wide: i32 = @as(i32, a.value) << 8;
        const result: i32 = @divTrunc(wide, @as(i32, b.value));
        if (result > math.maxInt(i16)) return FixedPointError.Overflow;
        if (result < math.minInt(i16)) return FixedPointError.Overflow;
        return .{ .value = @intCast(result) };
    }
};

pub const FixedPoint32 = packed struct {
    value: i32,

    pub fn fromFloat(f: f32) FixedPointError!FixedPoint32 {
        const scaled = f * 65536.0;
        if (scaled > @as(f32, @floatFromInt(math.maxInt(i32))) or scaled < @as(f32, @floatFromInt(math.minInt(i32)))) {
            return FixedPointError.Overflow;
        }
        return .{ .value = @intFromFloat(scaled) };
    }

    pub fn toFloat(self: FixedPoint32) f32 {
        return @as(f32, @floatFromInt(self.value)) / 65536.0;
    }

    pub fn add(a: FixedPoint32, b: FixedPoint32) FixedPointError!FixedPoint32 {
        const r = @addWithOverflow(a.value, b.value);
        if (r[1] != 0) return FixedPointError.Overflow;
        return .{ .value = r[0] };
    }

    pub fn sub(a: FixedPoint32, b: FixedPoint32) FixedPointError!FixedPoint32 {
        const r = @subWithOverflow(a.value, b.value);
        if (r[1] != 0) return FixedPointError.Overflow;
        return .{ .value = r[0] };
    }

    pub fn mul(a: FixedPoint32, b: FixedPoint32) FixedPointError!FixedPoint32 {
        const wide: i64 = @as(i64, a.value) * @as(i64, b.value);
        const result: i64 = wide >> 16;
        if (result > math.maxInt(i32)) return FixedPointError.Overflow;
        if (result < math.minInt(i32)) return FixedPointError.Overflow;
        return .{ .value = @intCast(result) };
    }

    pub fn div(a: FixedPoint32, b: FixedPoint32) FixedPointError!FixedPoint32 {
        if (b.value == 0) return FixedPointError.DivisionByZero;
        const wide: i64 = @as(i64, a.value) << 16;
        const result: i64 = @divTrunc(wide, @as(i64, b.value));
        if (result > math.maxInt(i32)) return FixedPointError.Overflow;
        if (result < math.minInt(i32)) return FixedPointError.Overflow;
        return .{ .value = @intCast(result) };
    }
};

pub const FixedPoint64 = packed struct {
    value: i64,

    pub fn fromFloat(f: f64) FixedPointError!FixedPoint64 {
        const scaled = f * 4294967296.0;
        if (scaled > @as(f64, @floatFromInt(math.maxInt(i64))) or scaled < @as(f64, @floatFromInt(math.minInt(i64)))) {
            return FixedPointError.Overflow;
        }
        return .{ .value = @intFromFloat(scaled) };
    }

    pub fn toFloat(self: FixedPoint64) f64 {
        return @as(f64, @floatFromInt(self.value)) / 4294967296.0;
    }

    pub fn add(a: FixedPoint64, b: FixedPoint64) FixedPointError!FixedPoint64 {
        const r = @addWithOverflow(a.value, b.value);
        if (r[1] != 0) return FixedPointError.Overflow;
        return .{ .value = r[0] };
    }

    pub fn sub(a: FixedPoint64, b: FixedPoint64) FixedPointError!FixedPoint64 {
        const r = @subWithOverflow(a.value, b.value);
        if (r[1] != 0) return FixedPointError.Overflow;
        return .{ .value = r[0] };
    }

    pub fn mul(a: FixedPoint64, b: FixedPoint64) FixedPointError!FixedPoint64 {
        const wide: i128 = @as(i128, a.value) * @as(i128, b.value);
        const result: i128 = wide >> 32;
        if (result > math.maxInt(i64)) return FixedPointError.Overflow;
        if (result < math.minInt(i64)) return FixedPointError.Overflow;
        return .{ .value = @intCast(result) };
    }

    pub fn div(a: FixedPoint64, b: FixedPoint64) FixedPointError!FixedPoint64 {
        if (b.value == 0) return FixedPointError.DivisionByZero;
        const wide: i128 = @as(i128, a.value) << 32;
        const result: i128 = @divTrunc(wide, @as(i128, b.value));
        if (result > math.maxInt(i64)) return FixedPointError.Overflow;
        if (result < math.minInt(i64)) return FixedPointError.Overflow;
        return .{ .value = @intCast(result) };
    }
};

pub const Fixed32_32 = packed struct {
    value: i64,

    pub fn fromFloat(f: f64) FixedPointError!Fixed32_32 {
        const scaled = f * 4294967296.0;
        if (scaled > @as(f64, @floatFromInt(math.maxInt(i64))) or scaled < @as(f64, @floatFromInt(math.minInt(i64)))) {
            return FixedPointError.Overflow;
        }
        return .{ .value = @intFromFloat(scaled) };
    }

    pub fn toFloat(self: Fixed32_32) f64 {
        return @as(f64, @floatFromInt(self.value)) / 4294967296.0;
    }

    pub fn add(a: Fixed32_32, b: Fixed32_32) FixedPointError!Fixed32_32 {
        const r = @addWithOverflow(a.value, b.value);
        if (r[1] != 0) return FixedPointError.Overflow;
        return .{ .value = r[0] };
    }

    pub fn sub(a: Fixed32_32, b: Fixed32_32) FixedPointError!Fixed32_32 {
        const r = @subWithOverflow(a.value, b.value);
        if (r[1] != 0) return FixedPointError.Overflow;
        return .{ .value = r[0] };
    }

    pub fn mul(a: Fixed32_32, b: Fixed32_32) FixedPointError!Fixed32_32 {
        const wide: i128 = @as(i128, a.value) * @as(i128, b.value);
        const result: i128 = wide >> 32;
        if (result > math.maxInt(i64)) return FixedPointError.Overflow;
        if (result < math.minInt(i64)) return FixedPointError.Overflow;
        return .{ .value = @intCast(result) };
    }

    pub fn div(a: Fixed32_32, b: Fixed32_32) FixedPointError!Fixed32_32 {
        if (b.value == 0) return FixedPointError.DivisionByZero;
        const wide: i128 = @as(i128, a.value) << 32;
        const result: i128 = @divTrunc(wide, @as(i128, b.value));
        if (result > math.maxInt(i64)) return FixedPointError.Overflow;
        if (result < math.minInt(i64)) return FixedPointError.Overflow;
        return .{ .value = @intCast(result) };
    }
};

pub const GenericTensor = struct {
    data: []u8,
    shape: []usize,
    strides: []usize,
    elem_size: usize,
    allocator: Allocator,
    refcount: *usize,

    pub fn init(allocator: Allocator, shape: []const usize, comptime T: type) !GenericTensor {
        const elem_size = @sizeOf(T);
        var size: usize = 1;
        var i: usize = 0;
        while (i < shape.len) : (i += 1) {
            size *= shape[i];
        }
        const data = try allocator.alloc(u8, size * elem_size);
        @memset(data, 0);
        var strides = try allocator.alloc(usize, shape.len);
        var stride: usize = 1;
        var j = shape.len;
        while (j > 0) {
            j -= 1;
            strides[j] = stride;
            stride *= shape[j];
        }
        const refcount = try allocator.create(usize);
        refcount.* = 1;
        return .{
            .data = data,
            .shape = try allocator.dupe(usize, shape),
            .strides = strides,
            .elem_size = elem_size,
            .allocator = allocator,
            .refcount = refcount,
        };
    }

    pub fn retain(self: *GenericTensor) void {
        _ = @atomicRmw(usize, self.refcount, .Add, 1, .seq_cst);
    }

    pub fn release(self: *GenericTensor) void {
        const old = @atomicRmw(usize, self.refcount, .Sub, 1, .seq_cst);
        if (old == 1) {
            self.deinit();
        }
    }

    pub fn deinit(self: *GenericTensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
        self.allocator.destroy(self.refcount);
    }

    pub const TensorError = error{
        IndicesLengthMismatch,
        IndexOutOfBounds,
    };

    pub fn get(comptime T: type, self: *const GenericTensor, indices: []const usize) TensorError!T {
        if (indices.len != self.shape.len) return TensorError.IndicesLengthMismatch;
        var idx: usize = 0;
        var i: usize = 0;
        while (i < indices.len) : (i += 1) {
            if (indices[i] >= self.shape[i]) return TensorError.IndexOutOfBounds;
            idx += indices[i] * self.strides[i];
        }
        var total_size: usize = 1;
        i = 0;
        while (i < self.shape.len) : (i += 1) {
            total_size *= self.shape[i];
        }
        if (idx >= total_size) return TensorError.IndexOutOfBounds;
        const ptr: [*]const T = @ptrCast(@alignCast(self.data.ptr));
        return ptr[idx];
    }

    pub fn set(comptime T: type, self: *GenericTensor, indices: []const usize, value: T) TensorError!void {
        if (indices.len != self.shape.len) return TensorError.IndicesLengthMismatch;
        var idx: usize = 0;
        var i: usize = 0;
        while (i < indices.len) : (i += 1) {
            if (indices[i] >= self.shape[i]) return TensorError.IndexOutOfBounds;
            idx += indices[i] * self.strides[i];
        }
        var total_size: usize = 1;
        i = 0;
        while (i < self.shape.len) : (i += 1) {
            total_size *= self.shape[i];
        }
        if (idx >= total_size) return TensorError.IndexOutOfBounds;
        const ptr: [*]T = @ptrCast(@alignCast(self.data.ptr));
        ptr[idx] = value;
    }
};

pub const ContextWindow = struct {
    tokens: []u32,
    size: usize,
    capacity: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, capacity: usize) !ContextWindow {
        const tokens = try allocator.alloc(u32, capacity);
        return .{
            .tokens = tokens,
            .size = 0,
            .capacity = capacity,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ContextWindow) void {
        self.allocator.free(self.tokens);
    }

    pub fn add(self: *ContextWindow, token: u32) void {
        if (self.capacity == 0) return;
        if (self.size < self.capacity) {
            self.tokens[self.size] = token;
            self.size += 1;
        } else {
            var i: usize = 0;
            while (i < self.capacity - 1) : (i += 1) {
                self.tokens[i] = self.tokens[i + 1];
            }
            self.tokens[self.capacity - 1] = token;
        }
    }

    pub fn clear(self: *ContextWindow) void {
        self.size = 0;
    }

    pub fn get(self: *const ContextWindow, index: usize) ?u32 {
        if (index >= self.size) return null;
        return self.tokens[index];
    }

    pub fn slice(self: *const ContextWindow) []const u32 {
        return self.tokens[0..self.size];
    }
};

pub const RankedSegment = struct {
    tokens: []u32,
    score: f32,
    position: u64,
    anchor: bool,

    pub fn init(allocator: Allocator, tokens: []u32, score: f32, position: u64, anchor: bool) !RankedSegment {
        return .{
            .tokens = try allocator.dupe(u32, tokens),
            .score = score,
            .position = position,
            .anchor = anchor,
        };
    }

    pub fn deinit(self: *RankedSegment, allocator: Allocator) void {
        allocator.free(self.tokens);
    }

    pub fn compare(self: RankedSegment, other: RankedSegment) i32 {
        return if (self.score > other.score) -1 else if (self.score < other.score) 1 else 0;
    }
};

pub const BitSet = struct {
    bits: []u64,
    len: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, len: usize) !BitSet {
        const num_words = (len + 63) / 64;
        const bits = try allocator.alloc(u64, num_words);
        @memset(bits, 0);
        return .{ .bits = bits, .len = len, .allocator = allocator };
    }

    pub fn deinit(self: *BitSet) void {
        self.allocator.free(self.bits);
    }

    pub fn set(self: *BitSet, index: usize) void {
        if (index >= self.len) return;
        const word = index / 64;
        const bit: u6 = @intCast(index % 64);
        self.bits[word] |= @as(u64, 1) << bit;
    }

    pub fn unset(self: *BitSet, index: usize) void {
        if (index >= self.len) return;
        const word = index / 64;
        const bit: u6 = @intCast(index % 64);
        self.bits[word] &= ~(@as(u64, 1) << bit);
    }

    pub fn isSet(self: *const BitSet, index: usize) bool {
        if (index >= self.len) return false;
        const word = index / 64;
        const bit: u6 = @intCast(index % 64);
        return (self.bits[word] & (@as(u64, 1) << bit)) != 0;
    }

    pub fn count(self: *const BitSet) usize {
        var total: usize = 0;
        var i: usize = 0;
        while (i < self.bits.len) : (i += 1) {
            total += @popCount(self.bits[i]);
        }
        return total;
    }

    pub fn unionWith(self: *BitSet, other: *const BitSet) void {
        const words = @min(self.bits.len, other.bits.len);
        var i: usize = 0;
        while (i < words) : (i += 1) {
            self.bits[i] |= other.bits[i];
        }
    }

    pub fn intersectWith(self: *BitSet, other: *const BitSet) void {
        const words = @min(self.bits.len, other.bits.len);
        var i: usize = 0;
        while (i < words) : (i += 1) {
            self.bits[i] &= other.bits[i];
        }
    }

    pub fn copy(self: *const BitSet, allocator: Allocator) !BitSet {
        const bits = try allocator.alloc(u64, self.bits.len);
        @memcpy(bits, self.bits);
        return .{ .bits = bits, .len = self.len, .allocator = allocator };
    }

    pub fn clearAll(self: *BitSet) void {
        @memset(self.bits, 0);
    }

    pub fn setAll(self: *BitSet) void {
        @memset(self.bits, 0xFFFFFFFFFFFFFFFF);
        const remainder = self.len % 64;
        if (remainder != 0) {
            const last = self.bits.len - 1;
            self.bits[last] = (@as(u64, 1) << @intCast(remainder)) - 1;
        }
    }
};

pub const PRNG = struct {
    state: [4]u64,

    pub fn init(seed: u64) PRNG {
        var prng = PRNG{ .state = undefined };
        prng.srand(seed);
        return prng;
    }

    pub fn srand(self: *PRNG, seed: u64) void {
        self.state[0] = seed;
        self.state[1] = seed ^ 0x123456789ABCDEF0;
        self.state[2] = seed ^ 0xFEDCBA9876543210;
        self.state[3] = seed ^ 0x0F1E2D3C4B5A6978;
        _ = self.next();
        _ = self.next();
        _ = self.next();
        _ = self.next();
    }

    pub fn next(self: *PRNG) u64 {
        const result_star: u64 = math.rotr(u64, self.state[1] *% 5, 7) *% 9;
        const t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = math.rotr(u64, self.state[3], 45);
        return result_star;
    }

    pub fn float(self: *PRNG) f32 {
        const bits = self.next();
        const x: u32 = @intCast((bits & 0xFFFF_FFFF));
        const divisor: f32 = @floatFromInt(@as(u64, std.math.maxInt(u32)) + 1);
        return @as(f32, @floatFromInt(x)) / divisor;
    }

    pub fn float64(self: *PRNG) f64 {
        const bits = self.next();
        const divisor: f64 = @floatFromInt(@as(u128, std.math.maxInt(u64)) + 1);
        return @as(f64, @floatFromInt(bits)) / divisor;
    }

    pub fn uint64(self: *PRNG) u64 {
        return self.next();
    }

    pub fn fill(self: *PRNG, buf: []u8) void {
        var i: usize = 0;
        while (i + 8 <= buf.len) : (i += 8) {
            const val = self.next();
            mem.writeInt(u64, buf[i .. i + 8][0..8], val, .little);
        }
        if (i < buf.len) {
            const val = self.next();
            var temp_buf: [8]u8 = undefined;
            mem.writeInt(u64, &temp_buf, val, .little);
            const remaining = buf.len - i;
            @memcpy(buf[i..], temp_buf[0..remaining]);
        }
    }

    pub fn uniform(self: *PRNG, min_val: u64, max_val: u64) u64 {
        if (min_val == max_val) return min_val;
        const range = max_val - min_val;
        var val = self.next();
        const thresh = std.math.maxInt(u64) - ((std.math.maxInt(u64) % range) + 1) % range;
        var retries: usize = 0;
        const max_retries: usize = 128;
        while (val > thresh) {
            val = self.next();
            retries += 1;
            if (retries >= max_retries) break;
        }
        return min_val + (val % range);
    }

    pub fn normal(self: *PRNG, mean: f64, stddev: f64) f64 {
        const u = 1.0 - self.float64();
        const v = self.float64();
        const z = math.sqrt(-2.0 * @log(u)) * math.cos(2.0 * math.pi * v);
        return mean + stddev * z;
    }

    pub fn reseed(self: *PRNG) !void {
        var buf: [32]u8 = undefined;
        try std.crypto.random.bytes(&buf);
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        hasher.update(&buf);
        var hash: [32]u8 = undefined;
        hasher.final(&hash);
        const seed = mem.readInt(u64, hash[0..8], .little);
        self.srand(seed);
    }

    pub fn seedFromEntropy(self: *PRNG) !void {
        try self.reseed();
    }
};

pub const Error = error{
    InvalidShape,
    OutOfBounds,
    AllocationFailed,
    ShapeMismatch,
    DivideByZero,
    InvalidAxis,
    EmptyInput,
    SingularMatrix,
    InvalidReps,
    InvalidPads,
    InvalidForOneHot,
    MustBeSquare,
    InvalidConv2D,
    InvalidPool2D,
    InvalidArgument,
    WindowFull,
    Overflow,
};

pub fn clamp(comptime T: type, value: T, min_val: T, max_val: T) T {
    return if (value < min_val) min_val else if (value > max_val) max_val else value;
}

pub fn abs(comptime T: type, x: T) T {
    return if (x < 0) -x else x;
}

pub fn min(comptime T: type, a: T, b: T) T {
    return if (a < b) a else b;
}

pub fn max(comptime T: type, a: T, b: T) T {
    return if (a > b) a else b;
}

pub fn sum(comptime T: type, slice: []const T) Error!T {
    var total: T = 0;
    var i: usize = 0;
    while (i < slice.len) : (i += 1) {
        switch (@typeInfo(T)) {
            .int => {
                const r = @addWithOverflow(total, slice[i]);
                if (r[1] != 0) return Error.Overflow;
                total = r[0];
            },
            .float => {
                total += slice[i];
            },
            else => total += slice[i],
        }
    }
    return total;
}

pub fn prod(comptime T: type, slice: []const T) Error!T {
    var total: T = 1;
    var i: usize = 0;
    while (i < slice.len) : (i += 1) {
        switch (@typeInfo(T)) {
            .int => {
                const r = @mulWithOverflow(total, slice[i]);
                if (r[1] != 0) return Error.Overflow;
                total = r[0];
            },
            .float => {
                total *= slice[i];
                if (!math.isFinite(total)) return Error.Overflow;
            },
            else => total *= slice[i],
        }
    }
    return total;
}

pub fn dotProduct(comptime T: type, a: []const T, b: []const T) Error!T {
    if (a.len != b.len) return error.ShapeMismatch;
    var result: T = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 1) {
        switch (@typeInfo(T)) {
            .int => {
                const mr = @mulWithOverflow(a[i], b[i]);
                if (mr[1] != 0) return Error.Overflow;
                const ar = @addWithOverflow(result, mr[0]);
                if (ar[1] != 0) return Error.Overflow;
                result = ar[0];
            },
            .float => {
                result += a[i] * b[i];
            },
            else => result += a[i] * b[i],
        }
    }
    return result;
}

pub fn crossProduct(comptime T: type, a: [3]T, b: [3]T) [3]T {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

pub fn norm(comptime T: type, vec: []const T) f64 {
    var sq_sum: f64 = 0.0;
    var i: usize = 0;
    while (i < vec.len) : (i += 1) {
        const f: f64 = switch (@typeInfo(T)) {
            .int => @floatFromInt(vec[i]),
            .float => @floatCast(vec[i]),
            else => @compileError("norm not supported for type " ++ @typeName(T)),
        };
        sq_sum += f * f;
    }
    return math.sqrt(sq_sum);
}

pub fn lerp(comptime T: type, a: T, b: T, t: f64) T {
    return switch (@typeInfo(T)) {
        .int => blk: {
            const fa: f64 = @floatFromInt(a);
            const fb: f64 = @floatFromInt(b);
            break :blk @intFromFloat(fa + (fb - fa) * t);
        },
        .float => blk: {
            const fa: f64 = @floatCast(a);
            const fb: f64 = @floatCast(b);
            break :blk @floatCast(fa + (fb - fa) * t);
        },
        else => @compileError("lerp not supported for type " ++ @typeName(T)),
    };
}

pub fn factorial(n: usize) Error!usize {
    if (n <= 1) return 1;
    var result: usize = 1;
    var i: usize = 2;
    while (i <= n) : (i += 1) {
        const r = @mulWithOverflow(result, i);
        if (r[1] != 0) return Error.Overflow;
        result = r[0];
    }
    return result;
}

pub fn binomial(n: usize, k: usize) Error!usize {
    if (k > n) return 0;
    if (k == 0 or k == n) return 1;
    const k_opt = if (k > n - k) n - k else k;
    var result: usize = 1;
    var i: usize = 0;
    while (i < k_opt) : (i += 1) {
        const r = @mulWithOverflow(result, (n - i));
        if (r[1] != 0) return Error.Overflow;
        result = r[0];
        result /= (i + 1);
    }
    return result;
}

pub fn gcd(a: usize, b: usize) usize {
    var x = a;
    var y = b;
    while (y != 0) {
        const temp = y;
        y = x % y;
        x = temp;
    }
    return x;
}

pub fn lcm(a: usize, b: usize) Error!usize {
    if (a == 0 or b == 0) return 0;
    const g = gcd(a, b);
    const q = a / g;
    const r = @mulWithOverflow(q, b);
    if (r[1] != 0) return Error.Overflow;
    return r[0];
}

pub fn pow(comptime T: type, base: T, exp: usize) Error!T {
    const ti = @typeInfo(T);
    switch (ti) {
        .int => {
            var result: T = 1;
            var e = exp;
            var b = base;
            while (e > 0) {
                if (e % 2 == 1) {
                    const r = @mulWithOverflow(result, b);
                    if (r[1] != 0) return Error.Overflow;
                    result = r[0];
                }
                if (e > 1) {
                    const r = @mulWithOverflow(b, b);
                    if (r[1] != 0) return Error.Overflow;
                    b = r[0];
                }
                e /= 2;
            }
            return result;
        },
        .float => {
            var result: T = 1;
            var e = exp;
            var b = base;
            while (e > 0) {
                if (e % 2 == 1) {
                    result *= b;
                    if (!math.isFinite(result)) return Error.Overflow;
                }
                if (e > 1) {
                    b *= b;
                    if (!math.isFinite(b)) return Error.Overflow;
                }
                e /= 2;
            }
            return result;
        },
        else => @compileError("pow not supported for type " ++ @typeName(T)),
    }
}

pub fn log2(comptime T: type, x: T) f64 {
    return switch (@typeInfo(T)) {
        .int => @log2(@as(f64, @floatFromInt(x))),
        .float => @log2(@as(f64, @floatCast(x))),
        else => @compileError("log2 not supported for type " ++ @typeName(T)),
    };
}

pub fn isPowerOfTwo(n: usize) bool {
    return n > 0 and (n & (n - 1)) == 0;
}

pub fn nextPowerOfTwo(n: usize) usize {
    if (n == 0) return 1;
    var p: usize = 1;
    while (p < n) p <<= 1;
    return p;
}

pub fn popcount(comptime T: type, x: T) usize {
    return switch (@typeInfo(T)) {
        .int => blk: {
            var count: usize = 0;
            var val = x;
            while (val != 0) {
                count += @intCast(val & 1);
                val >>= 1;
            }
            break :blk count;
        },
        else => @compileError("popcount not supported for type " ++ @typeName(T)),
    };
}

pub fn leadingZeros(comptime T: type, x: T) usize {
    return switch (@typeInfo(T)) {
        .int => |info| blk: {
            if (x == 0) break :blk info.bits;
            var count: usize = 0;
            var val = x;
            const bits = info.bits;
            const high_bit: T = @as(T, 1) << @intCast(bits - 1);
            while ((val & high_bit) == 0 and count < bits) : (count += 1) {
                val <<= 1;
            }
            break :blk count;
        },
        else => @compileError("leadingZeros not supported for type " ++ @typeName(T)),
    };
}

pub fn trailingZeros(comptime T: type, x: T) usize {
    return switch (@typeInfo(T)) {
        .int => |info| blk: {
            if (x == 0) break :blk info.bits;
            var count: usize = 0;
            var val = x;
            while ((val & 1) == 0 and count < info.bits) : (count += 1) {
                val >>= 1;
            }
            break :blk count;
        },
        else => @compileError("trailingZeros not supported for type " ++ @typeName(T)),
    };
}

pub fn reverseBits(comptime T: type, x: T) T {
    return switch (@typeInfo(T)) {
        .int => |info| blk: {
            var rev: T = 0;
            var val = x;
            const bits = info.bits;
            var pos: usize = 0;
            while (pos < bits) : (pos += 1) {
                rev |= (val & 1) << @intCast(bits - 1 - pos);
                val >>= 1;
            }
            break :blk rev;
        },
        else => @compileError("reverseBits not supported for type " ++ @typeName(T)),
    };
}

pub fn bitReverseCopy(comptime T: type, src: []const T, dst: []T) void {
    if (src.len != dst.len) return;
    if (src.len == 0) return;
    const n = src.len;
    var log_n: usize = 0;
    {
        var v = n - 1;
        while (v > 0) : (v >>= 1) {
            log_n += 1;
        }
    }
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var rev: usize = 0;
        var val = i;
        var j: usize = 0;
        while (j < log_n) : (j += 1) {
            rev = (rev << 1) | (val & 1);
            val >>= 1;
        }
        if (rev < n) {
            dst[rev] = src[i];
        }
    }
}

pub fn hammingWeight(comptime T: type, x: T) usize {
    return popcount(T, x);
}

pub fn hammingDistance(comptime T: type, a: T, b: T) usize {
    return popcount(T, a ^ b);
}

pub fn parity(comptime T: type, x: T) bool {
    var p: u1 = 0;
    var val = x;
    while (val != 0) : (val >>= 1) {
        p ^= @intCast(val & 1);
    }
    return p == 0;
}

pub const KernelCapability = u64;

pub const IPCChannel = struct {
    cap: u64,
    buffer: []u8,
    ready: bool,
    allocator: Allocator,

    pub fn init(allocator: Allocator, size: usize) !IPCChannel {
        const buffer = try allocator.alloc(u8, size);
        return .{
            .cap = 0,
            .buffer = buffer,
            .ready = false,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *IPCChannel) void {
        self.allocator.free(self.buffer);
    }
};

pub const ComplexFixedPoint = struct {
    real: FixedPoint32,
    imag: FixedPoint32,

    pub fn mul(a: ComplexFixedPoint, b: ComplexFixedPoint) FixedPointError!ComplexFixedPoint {
        return .{
            .real = try (try a.real.mul(b.real)).sub(try a.imag.mul(b.imag)),
            .imag = try (try a.real.mul(b.imag)).add(try a.imag.mul(b.real)),
        };
    }

    pub fn add(a: ComplexFixedPoint, b: ComplexFixedPoint) FixedPointError!ComplexFixedPoint {
        return .{
            .real = try a.real.add(b.real),
            .imag = try a.imag.add(b.imag),
        };
    }

    pub fn sub(a: ComplexFixedPoint, b: ComplexFixedPoint) FixedPointError!ComplexFixedPoint {
        return .{
            .real = try a.real.sub(b.real),
            .imag = try a.imag.sub(b.imag),
        };
    }
};

pub const Vector3D = packed struct {
    x: FixedPoint16,
    y: FixedPoint16,
    z: FixedPoint16,
};

pub const Matrix4x4 = [4][4]FixedPoint32;

pub const Quaternion = struct {
    w: FixedPoint32,
    x: FixedPoint32,
    y: FixedPoint32,
    z: FixedPoint32,
};

pub const ColorRGBA = packed struct {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
};

pub const DateTime = packed struct {
    year: u16,
    month: u8,
    day: u8,
    hour: u8,
    min: u8,
    sec: u8,
    millis: u16,
};

pub const UUID = [16]u8;
pub const IPv4 = [4]u8;
pub const IPv6 = [16]u8;
pub const MACAddress = [6]u8;

pub const GeoPoint = struct {
    lat: FixedPoint32,
    lon: FixedPoint32,
};

pub const BoundingBox = struct {
    min: GeoPoint,
    max: GeoPoint,
};

pub const Polygon = []GeoPoint;
pub const LineString = []GeoPoint;

pub const TimeSeriesPoint = struct {
    time: DateTime,
    value: FixedPoint32,
};

pub const TimeSeries = []TimeSeriesPoint;

pub const GraphNode = struct {
    id: u64,
    neighbors: []u64,
};

pub const TreeNode = struct {
    value: u64,
    children: []*TreeNode,
};

pub const LinkedListNode = struct {
    value: u64,
    next: ?*LinkedListNode,
};

pub const Stack = std.ArrayList(u64);
pub const Queue = std.fifo.LinearFifo(u64, .Dynamic);

pub const BloomFilter = struct {
    bits: BitSet,
    hash_functions: u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, expected_items: usize, hash_functions: u8) !BloomFilter {
        const bit_count = @max(expected_items * 10, @as(usize, 64));
        const bset = try BitSet.init(allocator, bit_count);
        return .{ .bits = bset, .hash_functions = hash_functions, .allocator = allocator };
    }

    pub fn deinit(self: *BloomFilter) void {
        self.bits.deinit();
    }

    fn hashKey(self: *const BloomFilter, key: u64, hash_idx: u8) usize {
        var h = key +% @as(u64, hash_idx) *% 0x9E3779B97F4A7C15;
        h ^= h >> 32;
        h *%= 0xBF58476D1CE4E5B9;
        h ^= h >> 32;
        const len: u64 = @intCast(self.bits.len);
        return @intCast(h % len);
    }

    pub fn add(self: *BloomFilter, key: u64) void {
        var i: u8 = 0;
        while (i < self.hash_functions) : (i += 1) {
            self.bits.set(self.hashKey(key, i));
        }
    }

    pub fn mightContain(self: *const BloomFilter, key: u64) bool {
        var i: u8 = 0;
        while (i < self.hash_functions) : (i += 1) {
            if (!self.bits.isSet(self.hashKey(key, i))) return false;
        }
        return true;
    }

    pub fn check(self: *const BloomFilter, key: u64) bool {
        return self.mightContain(key);
    }
};

pub const VoxelGrid = struct {
    data: []u8,
    dim_x: usize,
    dim_y: usize,
    dim_z: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) !VoxelGrid {
        const dim_x: usize = 32;
        const dim_y: usize = 32;
        const dim_z: usize = 32;
        const data = try allocator.alloc(u8, dim_x * dim_y * dim_z);
        @memset(data, 0);
        return .{ .data = data, .dim_x = dim_x, .dim_y = dim_y, .dim_z = dim_z, .allocator = allocator };
    }

    pub fn deinit(self: *VoxelGrid) void {
        self.allocator.free(self.data);
    }

    pub fn get(self: *const VoxelGrid, x: usize, y: usize, z: usize) ?u8 {
        if (x >= self.dim_x or y >= self.dim_y or z >= self.dim_z) return null;
        return self.data[x * self.dim_y * self.dim_z + y * self.dim_z + z];
    }

    pub fn set(self: *VoxelGrid, x: usize, y: usize, z: usize, value: u8) void {
        if (x >= self.dim_x or y >= self.dim_y or z >= self.dim_z) return;
        self.data[x * self.dim_y * self.dim_z + y * self.dim_z + z] = value;
    }
};

pub const Particle = struct {
    position: Vector3D,
    velocity: Vector3D,
    mass: FixedPoint16,
};

pub const ParticleSystem = []Particle;

pub const Spring = struct {
    p1: usize,
    p2: usize,
    rest_length: FixedPoint16,
    stiffness: FixedPoint16,
};

pub const ClothSimulation = struct {
    particles: ParticleSystem,
    springs: []Spring,
};

pub const FluidCell = struct {
    density: FixedPoint16,
    velocity: Vector3D,
};

pub const FluidGrid = struct {
    data: []FluidCell,
    dim_x: usize,
    dim_y: usize,
    dim_z: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) !FluidGrid {
        const dim_x: usize = 64;
        const dim_y: usize = 64;
        const dim_z: usize = 64;
        const data = try allocator.alloc(FluidCell, dim_x * dim_y * dim_z);
        @memset(data, .{ .density = .{ .value = 0 }, .velocity = .{ .x = .{ .value = 0 }, .y = .{ .value = 0 }, .z = .{ .value = 0 } } });
        return .{ .data = data, .dim_x = dim_x, .dim_y = dim_y, .dim_z = dim_z, .allocator = allocator };
    }

    pub fn deinit(self: *FluidGrid) void {
        self.allocator.free(self.data);
    }

    pub fn get(self: *const FluidGrid, x: usize, y: usize, z: usize) ?*const FluidCell {
        if (x >= self.dim_x or y >= self.dim_y or z >= self.dim_z) return null;
        return &self.data[x * self.dim_y * self.dim_z + y * self.dim_z + z];
    }

    pub fn getPtr(self: *FluidGrid, x: usize, y: usize, z: usize) ?*FluidCell {
        if (x >= self.dim_x or y >= self.dim_y or z >= self.dim_z) return null;
        return &self.data[x * self.dim_y * self.dim_z + y * self.dim_z + z];
    }
};

pub const GeneticIndividual = struct {
    genome: []u8,
    fitness: FixedPoint32,
};

pub const GeneticPopulation = []GeneticIndividual;
pub const AntColonyPath = []usize;

pub const AntColony = struct {
    ants: []AntColonyPath,
    pheromone: []FixedPoint32,
    node_count: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, node_count: usize, ant_count: usize) !AntColony {
        const pheromone = try allocator.alloc(FixedPoint32, node_count * node_count);
        var idx: usize = 0;
        while (idx < pheromone.len) : (idx += 1) {
            pheromone[idx] = .{ .value = 0 };
        }
        const ants = try allocator.alloc(AntColonyPath, ant_count);
        var ant_idx: usize = 0;
        while (ant_idx < ant_count) : (ant_idx += 1) {
            ants[ant_idx] = try allocator.alloc(usize, node_count);
            @memset(ants[ant_idx], 0);
        }
        return .{
            .ants = ants,
            .pheromone = pheromone,
            .node_count = node_count,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AntColony) void {
        for (self.ants) |ant_path| {
            self.allocator.free(ant_path);
        }
        self.allocator.free(self.pheromone);
        self.allocator.free(self.ants);
    }

    pub fn getPheromone(self: *const AntColony, from: usize, to: usize) FixedPoint32 {
        return self.pheromone[from * self.node_count + to];
    }

    pub fn setPheromone(self: *AntColony, from: usize, to: usize, value: FixedPoint32) void {
        self.pheromone[from * self.node_count + to] = value;
    }
};

pub const SwarmParticle = struct {
    position: Vector3D,
    velocity: Vector3D,
    best_position: Vector3D,
};

pub const Mesh = struct {
    vertices: []Vector3D,
    indices: []u32,
};

pub const Keyframe = struct {
    time: f32,
    value: Vector3D,
};

pub const Animation = []Keyframe;

pub const Camera = struct {
    position: Vector3D,
    rotation: Quaternion,
    fov: FixedPoint16,
};

pub const Light = struct {
    position: Vector3D,
    color: ColorRGBA,
    intensity: FixedPoint16,
};

pub const Material = struct {
    ambient: ColorRGBA,
    diffuse: ColorRGBA,
    specular: ColorRGBA,
    shininess: FixedPoint16,
};

pub const Ray = struct {
    origin: Vector3D,
    direction: Vector3D,
};

pub const AABB = struct {
    min: Vector3D,
    max: Vector3D,
};

pub const Sphere = struct {
    center: Vector3D,
    radius: FixedPoint16,
};

pub const Plane = struct {
    normal: Vector3D,
    distance: FixedPoint16,
};

pub const Triangle = struct {
    vertices: [3]Vector3D,
};

pub const BVHNode = struct {
    bbox: AABB,
    left: ?*BVHNode,
    right: ?*BVHNode,
    object_id: u32,
};

pub const OctreeNode = struct {
    bbox: AABB,
    children: ?[8]*OctreeNode,
    objects: []u32,
};

pub const KDTreeNode = struct {
    split_axis: u8,
    split_value: FixedPoint32,
    left: ?*KDTreeNode,
    right: ?*KDTreeNode,
};

pub const QuadTree = struct {
    bbox: BoundingBox,
    children: ?[4]*QuadTree,
    points: []GeoPoint,
};

pub const RTree = struct {
    bbox: AABB,
    children: []RTree,
    data: []u64,
};

pub const SkipList = struct {
    levels: [][]u64,
    max_level: usize,
};

pub const Trie = struct {
    children: [26]?*Trie,
    is_end: bool,
};

pub const SuffixArray = struct {
    text: []u8,
    sa: []usize,
};

pub const FenwickTree = struct {
    tree: []i64,
};

pub const SegmentTree = struct {
    tree: []i64,
    size: usize,
};

pub const DisjointSet = struct {
    parent: []usize,
    rank: []u8,
};

pub const HuffmanNode = struct {
    freq: usize,
    char: u8,
    left: ?*HuffmanNode,
    right: ?*HuffmanNode,
};

pub const LZWDict = struct {
    entries: std.StringHashMap(u32),
    next_code: u32,
};

pub const RLE = struct {
    runs: []struct { value: u8, count: usize },
};

pub const AtomicBool = std.atomic.Value(bool);
pub const AtomicU64 = std.atomic.Value(u64);
pub const CacheLine = [64]u8;
pub const AlignedStruct = struct { data: CacheLine };
pub const SIMDVector = @Vector(8, f32);
pub const SIMDMatrix = [8]SIMDVector;

pub const GPUKernelParam = union {
    int: i32,
    float: f32,
    ptr: *anyopaque,
};

pub const GPUKernel = *const fn ([]GPUKernelParam) void;
pub const VulkanBuffer = u64;
pub const MetalShader = u64;
pub const CUDAKernel = *const fn () void;
pub const OpenCLContext = u64;
pub const FPGAConfig = []u8;
pub const ASICDesign = []u8;
pub const QuantumBit = bool;
pub const QuantumGate = *const fn (QuantumBit) QuantumBit;
pub const QuantumCircuit = []QuantumGate;

pub const RTLSignal = struct {
    width: u32,
    value: u64,
};

pub const ZKCircuitInput = struct {
    public: []u8,
    private: []u8,
};

pub const FormalProof = struct {
    theorem: []u8,
    proof: []u8,
};

pub const MAX_SHAPE_DIMS = 8;
pub const MultiDimIndex = [MAX_SHAPE_DIMS]usize;

pub const SparseTensor = struct {
    indices: []MultiDimIndex,
    values: []f32,
    shape: []usize,
};

pub const QuantizedTensor = struct {
    data: []u8,
    scale: f32,
    zero_point: u8,
    shape: []usize,
};

pub const OptimizerState = struct {
    params: []GenericTensor,
    gradients: []GenericTensor,
    fisher_diag: []f32,
};

pub const KernelInterface = opaque {};

pub const RuntimeEnv = struct {
    kernel: *KernelInterface,
    ipc: []IPCChannel,
};

pub const HardwareAccel = struct {
    rtl_modules: []RTLSignal,
};

pub const ZKProofGen = *const fn (ZKCircuitInput) []u8;

pub const VerificationEnv = struct {
    lean_proofs: []FormalProof,
    isabelle_theories: []u8,
    tla_specs: []u8,
};

pub const SSIHashTree = struct {
    root: ?*HashNode,
    allocator: Allocator,

    const HashNode = struct {
        key: u64,
        value: []RankedSegment,
        left: ?*HashNode,
        right: ?*HashNode,
    };

    pub fn init(allocator: Allocator) SSIHashTree {
        return .{ .root = null, .allocator = allocator };
    }

    pub fn deinit(self: *SSIHashTree) void {
        if (self.root) |root| {
            self.deinitNode(root);
        }
    }

    fn deinitNode(self: *SSIHashTree, node: *HashNode) void {
        if (node.left) |left| self.deinitNode(left);
        if (node.right) |right| self.deinitNode(right);
        for (node.value) |*seg| {
            self.allocator.free(seg.tokens);
        }
        self.allocator.free(node.value);
        self.allocator.destroy(node);
    }

    pub fn insert(self: *SSIHashTree, key: u64, seg: RankedSegment) !void {
        var node = &self.root;
        while (node.*) |n| {
            if (key < n.key) {
                node = &n.left;
            } else if (key > n.key) {
                node = &n.right;
            } else {
                const new_val = try self.allocator.realloc(n.value, n.value.len + 1);
                const duped_tokens = try self.allocator.dupe(u32, seg.tokens);
                new_val[new_val.len - 1] = .{
                    .tokens = duped_tokens,
                    .score = seg.score,
                    .position = seg.position,
                    .anchor = seg.anchor,
                };
                n.value = new_val;
                return;
            }
        }
        const duped_tokens = try self.allocator.dupe(u32, seg.tokens);
        const seg_copy = RankedSegment{
            .tokens = duped_tokens,
            .score = seg.score,
            .position = seg.position,
            .anchor = seg.anchor,
        };
        const new_val = try self.allocator.dupe(RankedSegment, &.{seg_copy});
        const new_node = self.allocator.create(HashNode) catch {
            self.allocator.free(duped_tokens);
            self.allocator.free(new_val);
            return error.OutOfMemory;
        };
        new_node.* = .{
            .key = key,
            .value = new_val,
            .left = null,
            .right = null,
        };
        node.* = new_node;
    }
};

pub const MorphGraphNode = struct {
    token: u32,
    edges: []u32,
};

pub const RelevanceScore = FixedPoint32;

pub const InferenceTrace = struct {
    inputs: GenericTensor,
    outputs: GenericTensor,
    proofs: []u8,
};

pub const Texture = struct {
    data: []ColorRGBA,
    width: usize,
    height: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, width: usize, height: usize) !Texture {
        const data = try allocator.alloc(ColorRGBA, width * height);
        var idx: usize = 0;
        while (idx < data.len) : (idx += 1) {
            data[idx] = .{ .r = 0, .g = 0, .b = 0, .a = 255 };
        }
        return .{ .data = data, .width = width, .height = height, .allocator = allocator };
    }

    pub fn deinit(self: *Texture) void {
        self.allocator.free(self.data);
    }

    pub fn get(self: *const Texture, x: usize, y: usize) ?ColorRGBA {
        if (x >= self.width or y >= self.height) return null;
        return self.data[y * self.width + x];
    }

    pub fn set(self: *Texture, x: usize, y: usize, color: ColorRGBA) void {
        if (x >= self.width or y >= self.height) return;
        self.data[y * self.width + x] = color;
    }
};

pub const ShaderProgram = u32;
pub const RenderPipeline = []ShaderProgram;

pub const Resolution = struct {
    width: u16,
    height: u16,
};

pub const FrameRate = u32;

pub const LoggerLevel = enum {
    debug,
    info,
    warn,
    @"error",
};

pub const LogEntry = struct {
    level: LoggerLevel,
    msg: []u8,
    timestamp: DateTime,
};

pub const Currency = u32;
pub const Wallet = Currency;

pub const Transaction = struct {
    from: []u8,
    to: []u8,
    amount: Currency,
};

pub const NFT = struct {
    id: u64,
    metadata: []u8,
};

pub const Avatar = struct {
    model: Mesh,
    textures: []Texture,
};

pub const ChatMessage = struct {
    sender: u64,
    text: []u8,
    time: DateTime,
};

pub const Version = struct {
    major: u8,
    minor: u8,
    patch: u8,
};

pub const AuthToken = [256]u8;

pub const Session = struct {
    user: u64,
    token: AuthToken,
    expiry: DateTime,
};

pub const InputKey = enum { up, down, left, right };

pub const ControllerAxis = FixedPoint16;

pub const ControllerState = struct {
    buttons: BitSet,
    axes: [4]ControllerAxis,
};

pub const VRPose = struct {
    head: Quaternion,
    hands: [2]Quaternion,
};

pub const ARMarker = struct {
    id: u32,
    pos: Vector3D,
};

pub const Drone = struct {
    position: Vector3D,
    altitude: FixedPoint16,
    battery: u8,
};

pub const RobotArm = struct {
    joints: [6]FixedPoint16,
    end_effector: Vector3D,
};

test "FixedPoint32 arithmetic" {
    const a = try FixedPoint32.fromFloat(3.5);
    const b = try FixedPoint32.fromFloat(2.0);
    const sum_result = try a.add(b);
    const diff = try a.sub(b);
    const prod_result = try a.mul(b);
    const quot = try a.div(b);

    try testing.expectApproxEqAbs(@as(f32, 5.5), sum_result.toFloat(), 0.01);
    try testing.expectApproxEqAbs(@as(f32, 1.5), diff.toFloat(), 0.01);
    try testing.expectApproxEqAbs(@as(f32, 7.0), prod_result.toFloat(), 0.01);
    try testing.expectApproxEqAbs(@as(f32, 1.75), quot.toFloat(), 0.01);
}

test "FixedPoint16 arithmetic" {
    const a = try FixedPoint16.fromFloat(1.5);
    const b = try FixedPoint16.fromFloat(0.5);
    const sum_result = try a.add(b);
    const diff = try a.sub(b);
    const prod_result = try a.mul(b);
    const quot = try a.div(b);

    try testing.expectApproxEqAbs(@as(f32, 2.0), sum_result.toFloat(), 0.01);
    try testing.expectApproxEqAbs(@as(f32, 1.0), diff.toFloat(), 0.01);
    try testing.expectApproxEqAbs(@as(f32, 0.75), prod_result.toFloat(), 0.01);
    try testing.expectApproxEqAbs(@as(f32, 3.0), quot.toFloat(), 0.01);
}

test "GenericTensor operations" {
    const allocator = testing.allocator;
    var tensor = try GenericTensor.init(allocator, &[_]usize{ 2, 3 }, f32);
    defer tensor.deinit();

    try GenericTensor.set(f32, &tensor, &[_]usize{ 0, 0 }, 1.0);
    try GenericTensor.set(f32, &tensor, &[_]usize{ 1, 2 }, 5.0);

    const val1 = try GenericTensor.get(f32, &tensor, &[_]usize{ 0, 0 });
    const val2 = try GenericTensor.get(f32, &tensor, &[_]usize{ 1, 2 });

    try testing.expectEqual(@as(f32, 1.0), val1);
    try testing.expectEqual(@as(f32, 5.0), val2);
}

test "BitSet operations" {
    const allocator = testing.allocator;
    var bitset = try BitSet.init(allocator, 128);
    defer bitset.deinit();

    bitset.set(0);
    bitset.set(64);
    bitset.set(127);

    try testing.expect(bitset.isSet(0));
    try testing.expect(bitset.isSet(64));
    try testing.expect(bitset.isSet(127));
    try testing.expect(!bitset.isSet(50));

    try testing.expectEqual(@as(usize, 3), bitset.count());

    bitset.unset(0);
    try testing.expect(!bitset.isSet(0));
    try testing.expectEqual(@as(usize, 2), bitset.count());
}

test "BitSet union and intersection" {
    const allocator = testing.allocator;
    var bs1 = try BitSet.init(allocator, 64);
    defer bs1.deinit();
    var bs2 = try BitSet.init(allocator, 64);
    defer bs2.deinit();

    bs1.set(0);
    bs1.set(10);
    bs2.set(10);
    bs2.set(20);

    bs1.unionWith(&bs2);
    try testing.expect(bs1.isSet(0));
    try testing.expect(bs1.isSet(10));
    try testing.expect(bs1.isSet(20));

    var bs3 = try BitSet.init(allocator, 64);
    defer bs3.deinit();
    var bs4 = try BitSet.init(allocator, 64);
    defer bs4.deinit();

    bs3.set(5);
    bs3.set(15);
    bs4.set(15);
    bs4.set(25);

    bs3.intersectWith(&bs4);
    try testing.expect(!bs3.isSet(5));
    try testing.expect(bs3.isSet(15));
    try testing.expect(!bs3.isSet(25));
}

test "PRNG functionality" {
    var prng = PRNG.init(42);

    const f = prng.float();
    try testing.expect(f >= 0.0 and f < 1.0);

    const f64_val = prng.float64();
    try testing.expect(f64_val >= 0.0 and f64_val < 1.0);

    const u = prng.uint64();
    try testing.expect(u > 0);

    const uniform_val = prng.uniform(10, 20);
    try testing.expect(uniform_val >= 10 and uniform_val < 20);

    var buf: [16]u8 = undefined;
    prng.fill(&buf);
    var has_nonzero = false;
    var i: usize = 0;
    while (i < buf.len) : (i += 1) {
        if (buf[i] != 0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
}

test "PRNG normal distribution" {
    var prng = PRNG.init(12345);

    const n1 = prng.normal(0.0, 1.0);
    const n2 = prng.normal(0.0, 1.0);

    try testing.expect(n1 >= -5.0 and n1 <= 5.0);
    try testing.expect(n2 >= -5.0 and n2 <= 5.0);
}

test "Utility functions" {
    try testing.expectEqual(@as(usize, 120), try factorial(5));
    try testing.expectEqual(@as(usize, 10), try binomial(5, 2));
    try testing.expectEqual(@as(usize, 6), gcd(12, 18));
    try testing.expectEqual(@as(usize, 36), try lcm(12, 18));
    try testing.expectEqual(@as(i32, 8), try pow(i32, 2, 3));
}

test "Math utility functions" {
    try testing.expectEqual(@as(i32, 5), clamp(i32, 10, 0, 5));
    try testing.expectEqual(@as(i32, 0), clamp(i32, -5, 0, 10));
    try testing.expectEqual(@as(i32, 7), clamp(i32, 7, 0, 10));

    try testing.expectEqual(@as(i32, 5), abs(i32, -5));
    try testing.expectEqual(@as(i32, 5), abs(i32, 5));

    try testing.expectEqual(@as(i32, 3), min(i32, 3, 7));
    try testing.expectEqual(@as(i32, 7), max(i32, 3, 7));

    const arr = [_]i32{ 1, 2, 3, 4, 5 };
    try testing.expectEqual(@as(i32, 15), try sum(i32, &arr));
    try testing.expectEqual(@as(i32, 120), try prod(i32, &arr));
}

test "Vector operations" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };

    const dot = try dotProduct(f32, &a, &b);
    try testing.expectApproxEqAbs(@as(f32, 32.0), dot, 0.01);

    const cross = crossProduct(f32, a, b);
    try testing.expectApproxEqAbs(@as(f32, -3.0), cross[0], 0.01);
    try testing.expectApproxEqAbs(@as(f32, 6.0), cross[1], 0.01);
    try testing.expectApproxEqAbs(@as(f32, -3.0), cross[2], 0.01);
}

test "Bit operations" {
    try testing.expect(isPowerOfTwo(16));
    try testing.expect(!isPowerOfTwo(15));

    try testing.expectEqual(@as(usize, 16), nextPowerOfTwo(15));
    try testing.expectEqual(@as(usize, 16), nextPowerOfTwo(16));
    try testing.expectEqual(@as(usize, 32), nextPowerOfTwo(17));

    try testing.expectEqual(@as(usize, 3), popcount(u8, 0b10110000));
    try testing.expectEqual(@as(usize, 3), hammingWeight(u8, 0b10110000));

    try testing.expectEqual(@as(usize, 2), hammingDistance(u8, 0b1010, 0b1100));

    try testing.expect(parity(u8, 0b11));
    try testing.expect(!parity(u8, 0b111));
}

test "ContextWindow" {
    const allocator = testing.allocator;
    var window = try ContextWindow.init(allocator, 10);
    defer window.deinit();

    window.add(1);
    window.add(2);
    window.add(3);

    try testing.expectEqual(@as(usize, 3), window.size);
    try testing.expectEqual(@as(u32, 1), window.get(0).?);
    try testing.expectEqual(@as(u32, 2), window.get(1).?);

    const s = window.slice();
    try testing.expectEqual(@as(usize, 3), s.len);

    window.clear();
    try testing.expectEqual(@as(usize, 0), window.size);
}

test "ContextWindow sliding" {
    const allocator = testing.allocator;
    var window = try ContextWindow.init(allocator, 3);
    defer window.deinit();

    window.add(1);
    window.add(2);
    window.add(3);
    try testing.expectEqual(@as(usize, 3), window.size);

    window.add(4);
    try testing.expectEqual(@as(usize, 3), window.size);
    try testing.expectEqual(@as(u32, 2), window.get(0).?);
    try testing.expectEqual(@as(u32, 3), window.get(1).?);
    try testing.expectEqual(@as(u32, 4), window.get(2).?);
}

test "RankedSegment" {
    const allocator = testing.allocator;
    const tokens1 = [_]u32{ 1, 2, 3, 4, 5 };
    const tokens2 = [_]u32{ 6, 7, 8 };

    var seg1 = try RankedSegment.init(allocator, @constCast(&tokens1), 0.8, 0, true);
    defer seg1.deinit(allocator);

    var seg2 = try RankedSegment.init(allocator, @constCast(&tokens2), 0.6, 5, false);
    defer seg2.deinit(allocator);

    try testing.expectEqual(@as(i32, -1), seg1.compare(seg2));
    try testing.expectEqual(@as(usize, 5), seg1.tokens.len);
}

test "ComplexFixedPoint" {
    const a = ComplexFixedPoint{
        .real = try FixedPoint32.fromFloat(1.0),
        .imag = try FixedPoint32.fromFloat(2.0),
    };

    const b = ComplexFixedPoint{
        .real = try FixedPoint32.fromFloat(3.0),
        .imag = try FixedPoint32.fromFloat(4.0),
    };

    const sum_result = try a.add(b);
    try testing.expectApproxEqAbs(@as(f32, 4.0), sum_result.real.toFloat(), 0.01);
    try testing.expectApproxEqAbs(@as(f32, 6.0), sum_result.imag.toFloat(), 0.01);
}

test "SSIHashTree" {
    const allocator = testing.allocator;
    var tree = SSIHashTree.init(allocator);
    defer tree.deinit();

    const tokens1 = [_]u32{ 1, 2, 3 };
    const tokens2 = [_]u32{ 4, 5, 6 };

    var seg1 = try RankedSegment.init(allocator, @constCast(&tokens1), 0.9, 0, true);
    defer seg1.deinit(allocator);

    var seg2 = try RankedSegment.init(allocator, @constCast(&tokens2), 0.7, 10, false);
    defer seg2.deinit(allocator);

    try tree.insert(100, seg1);
    try tree.insert(200, seg2);
    try tree.insert(100, seg2);

    try testing.expect(tree.root != null);
}

test "BloomFilter" {
    const allocator = testing.allocator;
    var bf = try BloomFilter.init(allocator, 100, 3);
    defer bf.deinit();

    bf.add(42);
    bf.add(100);
    bf.add(999);

    try testing.expect(bf.mightContain(42));
    try testing.expect(bf.mightContain(100));
    try testing.expect(bf.mightContain(999));
}

test "VoxelGrid" {
    const allocator = testing.allocator;
    var grid = try VoxelGrid.init(allocator);
    defer grid.deinit();

    grid.set(0, 0, 0, 42);
    grid.set(31, 31, 31, 99);

    try testing.expectEqual(@as(u8, 42), grid.get(0, 0, 0).?);
    try testing.expectEqual(@as(u8, 99), grid.get(31, 31, 31).?);
    try testing.expectEqual(@as(u8, 0), grid.get(5, 5, 5).?);
}

test "FluidGrid" {
    const allocator = testing.allocator;
    var grid = try FluidGrid.init(allocator);
    defer grid.deinit();

    grid.getPtr(0, 0, 0).?.density = try FixedPoint16.fromFloat(1.5);

    try testing.expectApproxEqAbs(@as(f32, 1.5), grid.get(0, 0, 0).?.density.toFloat(), 0.01);
}

test "AntColony" {
    const allocator = testing.allocator;
    var colony = try AntColony.init(allocator, 5, 10);
    defer colony.deinit();

    colony.setPheromone(0, 1, try FixedPoint32.fromFloat(0.5));
    const p = colony.getPheromone(0, 1);
    try testing.expectApproxEqAbs(@as(f32, 0.5), p.toFloat(), 0.01);
}

test "Texture" {
    const allocator = testing.allocator;
    var tex = try Texture.init(allocator, 64, 64);
    defer tex.deinit();

    tex.set(10, 20, .{ .r = 255, .g = 128, .b = 0, .a = 255 });
    const c = tex.get(10, 20).?;
    try testing.expectEqual(@as(u8, 255), c.r);
    try testing.expectEqual(@as(u8, 128), c.g);
}

test "factorial overflow" {
    const result = factorial(100);
    try testing.expectError(Error.Overflow, result);
}

test "pow overflow" {
    const result = pow(u8, 2, 10);
    try testing.expectError(Error.Overflow, result);
}

test "norm and lerp with floats" {
    const vec = [_]f32{ 3.0, 4.0 };
    const n = norm(f32, &vec);
    try testing.expectApproxEqAbs(@as(f64, 5.0), n, 0.01);

    const lerped = lerp(f32, 1.0, 3.0, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), lerped, 0.01);
}

test "log2 with floats" {
    const result = log2(f32, 8.0);
    try testing.expectApproxEqAbs(@as(f64, 3.0), result, 0.01);
}
