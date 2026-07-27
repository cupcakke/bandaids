const std = @import("std");
const builtin = @import("builtin");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const types = @import("types.zig");
const Error = types.Error;
const Fixed32_32 = types.Fixed32_32;
const memory = @import("memory.zig");

const alignment = 32;
const vector_width = 8;
const Vec8 = @Vector(vector_width, f32);

fn readSmallFile(path: []const u8, buf: []u8) ?[]const u8 {
    const file = std.fs.openFileAbsolute(path, .{}) catch return null;
    defer file.close();
    const n = file.read(buf) catch return null;
    if (n == 0) return null;
    return mem.trim(u8, buf[0..n], " \n\t\r");
}

fn cgroupV2CpuCount() ?usize {
    var buf: [128]u8 = undefined;
    const content = readSmallFile("/sys/fs/cgroup/cpu.max", &buf) orelse return null;
    var it = mem.splitScalar(u8, content, ' ');
    const quota_str = it.next() orelse return null;
    const period_str = it.next() orelse return null;
    if (mem.eql(u8, quota_str, "max")) return null;
    const quota = std.fmt.parseInt(i64, quota_str, 10) catch return null;
    const period = std.fmt.parseInt(i64, period_str, 10) catch return null;
    if (quota <= 0 or period <= 0) return null;
    const cpus = @divTrunc(quota, period);
    if (cpus < 1) return 1;
    return @intCast(cpus);
}

fn cgroupV1CpuCount() ?usize {
    var qbuf: [64]u8 = undefined;
    const quota_str = readSmallFile("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", &qbuf) orelse return null;
    const quota = std.fmt.parseInt(i64, quota_str, 10) catch return null;
    if (quota <= 0) return null;
    var pbuf: [64]u8 = undefined;
    const period_str = readSmallFile("/sys/fs/cgroup/cpu/cpu.cfs_period_us", &pbuf) orelse return null;
    const period = std.fmt.parseInt(i64, period_str, 10) catch return null;
    if (period <= 0) return null;
    const cpus = @divTrunc(quota, period);
    if (cpus < 1) return 1;
    return @intCast(cpus);
}

pub fn cgroupSource() []const u8 {
    if (builtin.os.tag != .linux) return "fallback";
    if (cgroupV2CpuCount() != null) return "cgroup_v2";
    if (cgroupV1CpuCount() != null) return "cgroup_v1";
    return "fallback";
}

pub fn effectiveCpuCount() usize {
    if (builtin.os.tag == .linux) {
        if (cgroupV2CpuCount()) |c| return math.clamp(c, 1, 8);
        if (cgroupV1CpuCount()) |c| return math.clamp(c, 1, 8);
    }
    const host = std.Thread.getCpuCount() catch 1;
    return math.clamp(@max(host, 1), 1, 8);
}

pub const TensorIterator = struct {
    shape: *const Shape,
    indices: [8]usize,
    offset: usize,
    done: bool,

    pub fn init(shape: *const Shape) TensorIterator {
        return .{
            .shape = shape,
            .indices = [_]usize{0} ** 8,
            .offset = 0,
            .done = false,
        };
    }

    pub fn advance(self: *TensorIterator) bool {
        if (self.done) return false;
        if (self.shape.dims.len == 0) {
            self.done = true;
            return false;
        }
        var axis: usize = self.shape.dims.len;
        while (axis > 0) {
            axis -= 1;
            self.indices[axis] += 1;
            self.offset += self.shape.strides[axis];
            if (self.indices[axis] < self.shape.dims[axis]) return true;
            self.offset -= self.shape.dims[axis] * self.shape.strides[axis];
            self.indices[axis] = 0;
        }
        self.done = true;
        return false;
    }
};

pub const Shape = struct {
    dims: []usize,
    strides: []usize,
    total_size: usize,
    freed: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    pub fn init(allocator: Allocator, dims_in: []const usize) !Shape {
        if (dims_in.len == 0 or dims_in.len > 8) return Error.InvalidShape;
        var total: usize = 1;
        for (dims_in) |dim| {
            if (dim == 0) return Error.InvalidShape;
            const result = @mulWithOverflow(total, dim);
            if (result[1] != 0) return Error.Overflow;
            total = result[0];
        }
        const dims = try allocator.alloc(usize, dims_in.len);
        errdefer allocator.free(dims);
        const strides = try allocator.alloc(usize, dims_in.len);
        errdefer allocator.free(strides);
        @memcpy(dims, dims_in);
        var stride: usize = 1;
        var i: usize = dims_in.len;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            const result = @mulWithOverflow(stride, dims[i]);
            if (result[1] != 0) return Error.Overflow;
            stride = result[0];
        }
        return .{ .dims = dims, .strides = strides, .total_size = total };
    }

    pub fn initWithStrides(allocator: Allocator, dims_in: []const usize, strides_in: []const usize) !Shape {
        if (dims_in.len == 0 or dims_in.len > 8 or dims_in.len != strides_in.len) return Error.InvalidShape;
        var total: usize = 1;
        for (dims_in) |dim| {
            if (dim == 0) return Error.InvalidShape;
            const result = @mulWithOverflow(total, dim);
            if (result[1] != 0) return Error.Overflow;
            total = result[0];
        }
        const dims = try allocator.dupe(usize, dims_in);
        errdefer allocator.free(dims);
        const strides = try allocator.dupe(usize, strides_in);
        errdefer allocator.free(strides);
        return .{ .dims = dims, .strides = strides, .total_size = total };
    }

    pub fn deinit(self: *Shape, allocator: Allocator) void {
        // A Shape can be reached by more than one release() call -- e.g. when a
        // Tensor is retained and released in place across multiple threads (see
        // Tensor.release) -- so freeing must be a thread-safe, idempotent,
        // exactly-once operation on the same Shape instance.
        if (self.freed.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) {
            return;
        }
        allocator.free(self.dims);
        allocator.free(self.strides);
    }

    pub fn copy(self: *const Shape, allocator: Allocator) !Shape {
        return Shape.initWithStrides(allocator, self.dims, self.strides);
    }

    pub fn totalSize(self: *const Shape) usize {
        return self.total_size;
    }

    pub fn equals(self: *const Shape, other: *const Shape) bool {
        return mem.eql(usize, self.dims, other.dims) and mem.eql(usize, self.strides, other.strides);
    }

    pub fn isContiguous(self: *const Shape) bool {
        var expected: usize = 1;
        var i: usize = self.dims.len;
        while (i > 0) {
            i -= 1;
            if (self.strides[i] != expected) return false;
            expected *= self.dims[i];
        }
        return true;
    }

    pub fn broadcastCompatible(self: *const Shape, target: *const Shape) bool {
        if (target.dims.len < self.dims.len) return false;
        const offset = target.dims.len - self.dims.len;
        var i: usize = 0;
        while (i < self.dims.len) : (i += 1) {
            const source_dim = self.dims[i];
            const target_dim = target.dims[offset + i];
            if (source_dim != target_dim and source_dim != 1) return false;
        }
        return true;
    }
};

pub fn MatmulComptime(comptime M: usize, comptime K: usize, comptime N: usize) type {
    return struct {
        pub fn execute(a: *const Tensor, b: *const Tensor, out: *Tensor) void {
            comptime var i: usize = 0;
            inline while (i < M) : (i += 1) {
                comptime var j: usize = 0;
                inline while (j < N) : (j += 1) {
                    var sum_value: f32 = 0.0;
                    comptime var k: usize = 0;
                    inline while (k < K) : (k += 1) {
                        sum_value += a.data[i * a.shape.strides[0] + k * a.shape.strides[1]] * b.data[k * b.shape.strides[0] + j * b.shape.strides[1]];
                    }
                    out.data[i * out.shape.strides[0] + j * out.shape.strides[1]] = sum_value;
                }
            }
        }
    };
}

pub const Tensor = struct {
    data: []align(32) f32,
    base_data: []align(32) f32,
    shape: Shape,
    allocator: Allocator,
    refcount: *usize,
    cow: *bool,

    pub fn init(allocator: Allocator, dims: []const usize) !Tensor {
        var shape = try Shape.init(allocator, dims);
        errdefer shape.deinit(allocator);
        const data = try allocator.alignedAlloc(f32, @as(?u29, alignment), shape.totalSize());
        errdefer allocator.free(data);
        @memset(data, 0.0);
        const refcount = try allocator.create(usize);
        errdefer allocator.destroy(refcount);
        refcount.* = 1;
        const cow = try allocator.create(bool);
        errdefer allocator.destroy(cow);
        cow.* = false;
        return .{ .data = data, .base_data = data, .shape = shape, .allocator = allocator, .refcount = refcount, .cow = cow };
    }

    pub fn initWithArena(arena: *memory.ArenaAllocator, dims: []const usize) !Tensor {
        return init(arena.allocator(), dims);
    }

    pub fn initWithPool(pool: *memory.PoolAllocator, dims: []const usize) !Tensor {
        return init(pool.allocator(), dims);
    }

    pub fn initWithSlab(slab: *memory.SlabAllocator, dims: []const usize) !Tensor {
        return init(slab.allocator(), dims);
    }

    pub fn initWithBuddy(buddy: *memory.BuddyAllocator, dims: []const usize) !Tensor {
        return init(buddy.allocator(), dims);
    }

    pub fn retain(self: *Tensor) void {
        _ = @atomicRmw(usize, self.refcount, .Add, 1, .acq_rel);
        self.cow.* = true;
    }

    pub fn release(self: *Tensor) void {
        // Each Tensor value (base or view) owns its own Shape independently of
        // the shared refcount, so its Shape must always be freed here (Shape.deinit
        // is idempotent, so repeated release() calls on the very same struct
        // instance -- e.g. retain()/release() pairs used directly without
        // creating a new view -- are safe).
        self.shape.deinit(self.allocator);
        const old = @atomicRmw(usize, self.refcount, .Sub, 1, .acq_rel);
        if (old == 1) {
            self.allocator.free(self.base_data);
            self.allocator.destroy(self.refcount);
            self.allocator.destroy(self.cow);
            // Only fully invalidate the struct once the underlying shared
            // data is actually gone -- a non-final release (old != 1) must
            // leave refcount/cow/data/allocator valid, since other holders
            // (or this same value, if retained again) still depend on them.
            self.* = undefined;
        }
    }

    pub fn deinit(self: *Tensor) void {
        self.release();
    }

    fn flatIndex(self: *const Tensor, indices: []const usize) !usize {
        if (indices.len != self.shape.dims.len) return Error.InvalidAxis;
        var offset: usize = 0;
        for (indices, 0..) |index, axis| {
            if (index >= self.shape.dims[axis]) return Error.OutOfBounds;
            offset += index * self.shape.strides[axis];
        }
        return offset;
    }

    fn ensureWritable(self: *Tensor) !void {
        if (@atomicLoad(usize, self.refcount, .acquire) == 1) {
            self.cow.* = false;
            return;
        }
        const total = self.shape.totalSize();
        const new_data = try self.allocator.alignedAlloc(f32, @as(?u29, alignment), total);
        errdefer self.allocator.free(new_data);
        const contiguous = self.shape.isContiguous();
        if (contiguous) {
            @memcpy(new_data, self.data[0..total]);
        } else {
            var iterator = TensorIterator.init(&self.shape);
            var i: usize = 0;
            while (i < total) : (i += 1) {
                new_data[i] = self.data[iterator.offset];
                _ = iterator.advance();
            }
        }
        const new_refcount = try self.allocator.create(usize);
        errdefer self.allocator.destroy(new_refcount);
        new_refcount.* = 1;
        const new_cow = try self.allocator.create(bool);
        errdefer self.allocator.destroy(new_cow);
        new_cow.* = false;
        const old_base_data = self.base_data;
        const old_refcount = self.refcount;
        const old_cow = self.cow;
        const old_count = @atomicRmw(usize, old_refcount, .Sub, 1, .acq_rel);
        self.data = new_data;
        self.base_data = new_data;
        self.refcount = new_refcount;
        self.cow = new_cow;
        if (old_count == 1) {
            self.allocator.free(old_base_data);
            self.allocator.destroy(old_refcount);
            self.allocator.destroy(old_cow);
        }
    }

    pub fn copy(self: *const Tensor, allocator: Allocator) !Tensor {
        var result = try Tensor.init(allocator, self.shape.dims);
        errdefer result.deinit();
        const total = self.shape.totalSize();
        const contiguous = self.shape.isContiguous();
        if (contiguous) {
            @memcpy(result.data[0..total], self.data[0..total]);
        } else {
            var iterator = TensorIterator.init(&self.shape);
            var i: usize = 0;
            while (i < total) : (i += 1) {
                result.data[i] = self.data[iterator.offset];
                _ = iterator.advance();
            }
        }
        return result;
    }

    pub fn get(self: *const Tensor, indices: []const usize) !f32 {
        return self.data[try self.flatIndex(indices)];
    }

    pub fn set(self: *Tensor, indices: []const usize, value: f32) !void {
        try self.ensureWritable();
        self.data[try self.flatIndex(indices)] = value;
    }

    pub fn fill(self: *Tensor, value: f32) !void {
        try self.ensureWritable();
        const total = self.shape.totalSize();
        const contiguous = self.shape.isContiguous();
        if (contiguous) {
            @memset(self.data[0..total], value);
            return;
        }
        var iterator = TensorIterator.init(&self.shape);
        var i: usize = 0;
        while (i < total) : (i += 1) {
            self.data[iterator.offset] = value;
            _ = iterator.advance();
        }
    }

    fn binaryFast(self: *Tensor, other: *const Tensor, comptime op: enum { add, sub, mul, div }) !void {
        if (!self.shape.equals(&other.shape)) return Error.ShapeMismatch;
        try self.ensureWritable();
        const total = self.shape.totalSize();
        const self_contiguous = self.shape.isContiguous();
        const other_contiguous = other.shape.isContiguous();
        if (self_contiguous and other_contiguous) {
            if (op == .div) {
                var i: usize = 0;
                while (i < total) : (i += 1) {
                    if (other.data[i] == 0.0) return Error.DivideByZero;
                }
                i = 0;
                const limit = total - total % vector_width;
                while (i < limit) : (i += vector_width) {
                    const a: Vec8 = self.data[i..][0..vector_width].*;
                    const b: Vec8 = other.data[i..][0..vector_width].*;
                    self.data[i..][0..vector_width].* = a / b;
                }
                while (i < total) : (i += 1) {
                    self.data[i] /= other.data[i];
                }
            } else {
                var i: usize = 0;
                const limit = total - total % vector_width;
                while (i < limit) : (i += vector_width) {
                    const a: Vec8 = self.data[i..][0..vector_width].*;
                    const b: Vec8 = other.data[i..][0..vector_width].*;
                    self.data[i..][0..vector_width].* = switch (op) {
                        .add => a + b,
                        .sub => a - b,
                        .mul => a * b,
                        .div => unreachable,
                    };
                }
                while (i < total) : (i += 1) {
                    switch (op) {
                        .add => self.data[i] += other.data[i],
                        .sub => self.data[i] -= other.data[i],
                        .mul => self.data[i] *= other.data[i],
                        .div => unreachable,
                    }
                }
            }
            return;
        }
        if (op == .div) {
            var check_iterator = TensorIterator.init(&other.shape);
            var ci: usize = 0;
            while (ci < total) : (ci += 1) {
                if (other.data[check_iterator.offset] == 0.0) return Error.DivideByZero;
                _ = check_iterator.advance();
            }
        }
        var self_iterator = TensorIterator.init(&self.shape);
        var other_iterator = TensorIterator.init(&other.shape);
        var i: usize = 0;
        while (i < total) : (i += 1) {
            switch (op) {
                .add => self.data[self_iterator.offset] += other.data[other_iterator.offset],
                .sub => self.data[self_iterator.offset] -= other.data[other_iterator.offset],
                .mul => self.data[self_iterator.offset] *= other.data[other_iterator.offset],
                .div => self.data[self_iterator.offset] /= other.data[other_iterator.offset],
            }
            _ = self_iterator.advance();
            _ = other_iterator.advance();
        }
    }

    fn scalarFast(self: *Tensor, scalar: f32, comptime op: enum { add, sub, mul, div }) !void {
        if (op == .div and scalar == 0.0) return Error.DivideByZero;
        try self.ensureWritable();
        const total = self.shape.totalSize();
        const contiguous = self.shape.isContiguous();
        if (contiguous) {
            const scalar_vector: Vec8 = @splat(scalar);
            var i: usize = 0;
            const limit = total - total % vector_width;
            while (i < limit) : (i += vector_width) {
                const a: Vec8 = self.data[i..][0..vector_width].*;
                self.data[i..][0..vector_width].* = switch (op) {
                    .add => a + scalar_vector,
                    .sub => a - scalar_vector,
                    .mul => a * scalar_vector,
                    .div => a / scalar_vector,
                };
            }
            while (i < total) : (i += 1) {
                switch (op) {
                    .add => self.data[i] += scalar,
                    .sub => self.data[i] -= scalar,
                    .mul => self.data[i] *= scalar,
                    .div => self.data[i] /= scalar,
                }
            }
            return;
        }
        var iterator = TensorIterator.init(&self.shape);
        var i: usize = 0;
        while (i < total) : (i += 1) {
            switch (op) {
                .add => self.data[iterator.offset] += scalar,
                .sub => self.data[iterator.offset] -= scalar,
                .mul => self.data[iterator.offset] *= scalar,
                .div => self.data[iterator.offset] /= scalar,
            }
            _ = iterator.advance();
        }
    }

    pub fn addFast(self: *Tensor, other: *const Tensor) !void {
        return self.binaryFast(other, .add);
    }

    pub fn subFast(self: *Tensor, other: *const Tensor) !void {
        return self.binaryFast(other, .sub);
    }

    pub fn mulFast(self: *Tensor, other: *const Tensor) !void {
        return self.binaryFast(other, .mul);
    }

    pub fn divFast(self: *Tensor, other: *const Tensor) !void {
        return self.binaryFast(other, .div);
    }

    pub fn add(self: *Tensor, other: *const Tensor) !void {
        return self.addFast(other);
    }

    pub fn sub(self: *Tensor, other: *const Tensor) !void {
        return self.subFast(other);
    }

    pub fn mul(self: *Tensor, other: *const Tensor) !void {
        return self.mulFast(other);
    }

    pub fn div(self: *Tensor, other: *const Tensor) !void {
        return self.divFast(other);
    }

    pub fn addScalarFast(self: *Tensor, scalar: f32) !void {
        return self.scalarFast(scalar, .add);
    }

    pub fn subScalarFast(self: *Tensor, scalar: f32) !void {
        return self.scalarFast(scalar, .sub);
    }

    pub fn mulScalarFast(self: *Tensor, scalar: f32) !void {
        return self.scalarFast(scalar, .mul);
    }

    pub fn divScalarFast(self: *Tensor, scalar: f32) !void {
        return self.scalarFast(scalar, .div);
    }

    pub fn addScalar(self: *Tensor, scalar: f32) !void {
        return self.addScalarFast(scalar);
    }

    pub fn subScalar(self: *Tensor, scalar: f32) !void {
        return self.subScalarFast(scalar);
    }

    pub fn mulScalar(self: *Tensor, scalar: f32) !void {
        return self.mulScalarFast(scalar);
    }

    pub fn divScalar(self: *Tensor, scalar: f32) !void {
        return self.divScalarFast(scalar);
    }

    fn unaryFast(self: *Tensor, comptime op: enum { exp, log, sin, cos, tan, sqrt, abs }) !void {
        try self.ensureWritable();
        var iterator = TensorIterator.init(&self.shape);
        const total = self.shape.totalSize();
        const contiguous = self.shape.isContiguous();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            const offset = if (contiguous) i else iterator.offset;
            self.data[offset] = switch (op) {
                .exp => @exp(self.data[offset]),
                .log => if (self.data[offset] <= 0.0) -math.inf(f32) else @log(self.data[offset]),
                .sin => @sin(self.data[offset]),
                .cos => @cos(self.data[offset]),
                .tan => @tan(self.data[offset]),
                .sqrt => if (self.data[offset] < 0.0) math.nan(f32) else @sqrt(self.data[offset]),
                .abs => @abs(self.data[offset]),
            };
            if (!contiguous) _ = iterator.advance();
        }
    }

    pub fn expFast(self: *Tensor) !void {
        return self.unaryFast(.exp);
    }

    pub fn logFast(self: *Tensor) !void {
        return self.unaryFast(.log);
    }

    pub fn sinFast(self: *Tensor) !void {
        return self.unaryFast(.sin);
    }

    pub fn cosFast(self: *Tensor) !void {
        return self.unaryFast(.cos);
    }

    pub fn tanFast(self: *Tensor) !void {
        return self.unaryFast(.tan);
    }

    pub fn sqrtFast(self: *Tensor) !void {
        return self.unaryFast(.sqrt);
    }

    pub fn absFast(self: *Tensor) !void {
        return self.unaryFast(.abs);
    }

    pub fn exp(self: *Tensor) !void {
        return self.expFast();
    }

    pub fn log(self: *Tensor) !void {
        return self.logFast();
    }

    pub fn sin(self: *Tensor) !void {
        return self.sinFast();
    }

    pub fn cos(self: *Tensor) !void {
        return self.cosFast();
    }

    pub fn tan(self: *Tensor) !void {
        return self.tanFast();
    }

    pub fn sqrt(self: *Tensor) !void {
        return self.sqrtFast();
    }

    pub fn abs(self: *Tensor) !void {
        return self.absFast();
    }

    pub fn powFast(self: *Tensor, exponent: f32) !void {
        try self.ensureWritable();
        var iterator = TensorIterator.init(&self.shape);
        const total = self.shape.totalSize();
        const contiguous = self.shape.isContiguous();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            const offset = if (contiguous) i else iterator.offset;
            self.data[offset] = math.pow(f32, self.data[offset], exponent);
            if (!contiguous) _ = iterator.advance();
        }
    }

    pub fn pow(self: *Tensor, exponent: f32) !void {
        return self.powFast(exponent);
    }

    pub fn clipFast(self: *Tensor, min_value: f32, max_value: f32) !void {
        try self.ensureWritable();
        var iterator = TensorIterator.init(&self.shape);
        const total = self.shape.totalSize();
        const contiguous = self.shape.isContiguous();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            const offset = if (contiguous) i else iterator.offset;
            self.data[offset] = math.clamp(self.data[offset], min_value, max_value);
            if (!contiguous) _ = iterator.advance();
        }
    }

    pub fn clip(self: *Tensor, min_value: f32, max_value: f32) !void {
        return self.clipFast(min_value, max_value);
    }

    pub fn reshape(self: *Tensor, new_dims: []const usize) !void {
        if (!self.shape.isContiguous()) return Error.InvalidShape;
        var new_shape = try Shape.init(self.allocator, new_dims);
        errdefer new_shape.deinit(self.allocator);
        if (new_shape.totalSize() != self.shape.totalSize()) return Error.InvalidShape;
        const old_shape = self.shape;
        self.shape = new_shape;
        old_shape.deinit(self.allocator);
    }

    pub fn view(self: *Tensor, new_dims: []const usize) !Tensor {
        if (!self.shape.isContiguous()) return Error.InvalidShape;
        var new_shape = try Shape.init(self.allocator, new_dims);
        errdefer new_shape.deinit(self.allocator);
        if (new_shape.totalSize() != self.shape.totalSize()) return Error.InvalidShape;
        self.retain();
        return .{ .data = self.data, .base_data = self.base_data, .shape = new_shape, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow };
    }

    pub fn newView(self: *Tensor, shape: Shape) !Tensor {
        if (shape.totalSize() != self.shape.totalSize()) return Error.InvalidShape;
        self.retain();
        return .{ .data = self.data, .base_data = self.base_data, .shape = shape, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow };
    }

    pub fn slice(self: *Tensor, starts: []const usize, ends: []const usize) !Tensor {
        if (starts.len != self.shape.dims.len or ends.len != self.shape.dims.len) return Error.InvalidAxis;
        var new_dims_stack: [8]usize = undefined;
        var new_strides_stack: [8]usize = undefined;
        var offset: usize = 0;
        for (starts, 0..) |start, axis| {
            if (start > ends[axis] or ends[axis] > self.shape.dims[axis] or ends[axis] == start) return Error.OutOfBounds;
            new_dims_stack[axis] = ends[axis] - start;
            new_strides_stack[axis] = self.shape.strides[axis];
            offset += start * self.shape.strides[axis];
        }
        var result = try Tensor.init(self.allocator, new_dims_stack[0..starts.len]);
        errdefer result.deinit();
        var src_shape = try Shape.initWithStrides(self.allocator, new_dims_stack[0..starts.len], new_strides_stack[0..starts.len]);
        defer src_shape.deinit(self.allocator);
        var src_iterator = TensorIterator.init(&src_shape);
        const total = src_shape.totalSize();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            result.data[i] = self.data[offset + src_iterator.offset];
            _ = src_iterator.advance();
        }
        return result;
    }

    pub fn transpose(self: *Tensor, axes: []const usize) !Tensor {
        if (axes.len != self.shape.dims.len) return Error.InvalidAxis;
        var seen = [_]bool{false} ** 8;
        var dims_stack: [8]usize = undefined;
        var strides_stack: [8]usize = undefined;
        for (axes, 0..) |axis, i| {
            if (axis >= axes.len or seen[axis]) return Error.InvalidAxis;
            seen[axis] = true;
            dims_stack[i] = self.shape.dims[axis];
            strides_stack[i] = self.shape.strides[axis];
        }
        var new_shape = try Shape.initWithStrides(self.allocator, dims_stack[0..axes.len], strides_stack[0..axes.len]);
        errdefer new_shape.deinit(self.allocator);
        self.retain();
        return .{ .data = self.data, .base_data = self.base_data, .shape = new_shape, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow };
    }

    pub fn broadcast(self: *Tensor, target_dims: []const usize) !Tensor {
        if (target_dims.len < self.shape.dims.len or target_dims.len > 8) return Error.ShapeMismatch;
        var strides_stack: [8]usize = [_]usize{0} ** 8;
        const offset = target_dims.len - self.shape.dims.len;
        var axis: usize = 0;
        while (axis < target_dims.len) : (axis += 1) {
            if (axis < offset) {
                strides_stack[axis] = 0;
            } else {
                const source_axis = axis - offset;
                const source_dim = self.shape.dims[source_axis];
                const target_dim = target_dims[axis];
                if (source_dim != target_dim and source_dim != 1) return Error.ShapeMismatch;
                strides_stack[axis] = if (source_dim == 1 and target_dim > 1) 0 else self.shape.strides[source_axis];
            }
        }
        var new_shape = try Shape.initWithStrides(self.allocator, target_dims, strides_stack[0..target_dims.len]);
        errdefer new_shape.deinit(self.allocator);
        self.retain();
        return .{ .data = self.data, .base_data = self.base_data, .shape = new_shape, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow };
    }

    pub fn unsqueeze(self: *Tensor, axis: usize) !Tensor {
        if (axis > self.shape.dims.len or self.shape.dims.len == 8) return Error.InvalidAxis;
        var dims_stack: [8]usize = undefined;
        var strides_stack: [8]usize = undefined;
        var source_axis: usize = 0;
        var target_axis: usize = 0;
        while (target_axis < self.shape.dims.len + 1) : (target_axis += 1) {
            if (target_axis == axis) {
                dims_stack[target_axis] = 1;
                strides_stack[target_axis] = if (source_axis < self.shape.strides.len) self.shape.strides[source_axis] else 1;
            } else {
                dims_stack[target_axis] = self.shape.dims[source_axis];
                strides_stack[target_axis] = self.shape.strides[source_axis];
                source_axis += 1;
            }
        }
        var new_shape = try Shape.initWithStrides(self.allocator, dims_stack[0 .. self.shape.dims.len + 1], strides_stack[0 .. self.shape.dims.len + 1]);
        errdefer new_shape.deinit(self.allocator);
        self.retain();
        return .{ .data = self.data, .base_data = self.base_data, .shape = new_shape, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow };
    }

    pub fn zeros(allocator: Allocator, dims: []const usize) !Tensor {
        return Tensor.init(allocator, dims);
    }

    pub fn ones(allocator: Allocator, dims: []const usize) !Tensor {
        var tensor = try Tensor.init(allocator, dims);
        try tensor.fill(1.0);
        return tensor;
    }

    pub fn full(allocator: Allocator, dims: []const usize, value: f32) !Tensor {
        var tensor = try Tensor.init(allocator, dims);
        try tensor.fill(value);
        return tensor;
    }

    pub fn randomUniform(allocator: Allocator, dims: []const usize, min_value: f32, max_value: f32, seed: u64) !Tensor {
        var prng = types.PRNG.init(seed);
        var tensor = try Tensor.init(allocator, dims);
        const total = tensor.shape.totalSize();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            tensor.data[i] = prng.float() * (max_value - min_value) + min_value;
        }
        return tensor;
    }

    pub fn randomNormal(allocator: Allocator, dims: []const usize, mean_value: f32, stddev_value: f32, seed: u64) !Tensor {
        var prng = types.PRNG.init(seed);
        var tensor = try Tensor.init(allocator, dims);
        const total = tensor.shape.totalSize();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            const u = 1.0 - prng.float();
            const v = 1.0 - prng.float();
            tensor.data[i] = mean_value + stddev_value * (@sqrt(-2.0 * @log(u)) * @cos(2.0 * math.pi * v));
        }
        return tensor;
    }

    pub fn identity(allocator: Allocator, n: usize) !Tensor {
        if (n == 0) return Error.InvalidShape;
        var tensor = try Tensor.init(allocator, &.{ n, n });
        var i: usize = 0;
        while (i < n) : (i += 1) tensor.data[i * n + i] = 1.0;
        return tensor;
    }

    pub fn sum(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        var dims_stack: [8]usize = undefined;
        const result_rank = if (self.shape.dims.len == 1) 1 else self.shape.dims.len - 1;
        if (self.shape.dims.len == 1) {
            dims_stack[0] = 1;
        } else {
            var j: usize = 0;
            for (self.shape.dims, 0..) |dim, i| {
                if (i != axis) {
                    dims_stack[j] = dim;
                    j += 1;
                }
            }
        }
        var result = try Tensor.init(allocator, dims_stack[0..result_rank]);
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            var result_offset: usize = 0;
            var result_axis: usize = 0;
            for (0..self.shape.dims.len) |input_axis| {
                if (input_axis != axis) {
                    result_offset += iterator.indices[input_axis] * result.shape.strides[result_axis];
                    result_axis += 1;
                }
            }
            result.data[result_offset] += self.data[iterator.offset];
            _ = iterator.advance();
        }
        return result;
    }

    pub fn mean(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var result = try self.sum(allocator, axis);
        try result.divScalar(@floatFromInt(self.shape.dims[axis]));
        return result;
    }

    pub fn max(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        var dims_stack: [8]usize = undefined;
        const result_rank = if (self.shape.dims.len == 1) 1 else self.shape.dims.len - 1;
        if (self.shape.dims.len == 1) {
            dims_stack[0] = 1;
        } else {
            var j: usize = 0;
            for (self.shape.dims, 0..) |dim, i| {
                if (i != axis) {
                    dims_stack[j] = dim;
                    j += 1;
                }
            }
        }
        var result = try Tensor.init(allocator, dims_stack[0..result_rank]);
        try result.fill(-math.inf(f32));
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            var result_offset: usize = 0;
            var result_axis: usize = 0;
            for (0..self.shape.dims.len) |input_axis| {
                if (input_axis != axis) {
                    result_offset += iterator.indices[input_axis] * result.shape.strides[result_axis];
                    result_axis += 1;
                }
            }
            result.data[result_offset] = @max(result.data[result_offset], self.data[iterator.offset]);
            _ = iterator.advance();
        }
        return result;
    }

    pub fn min(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        var dims_stack: [8]usize = undefined;
        const result_rank = if (self.shape.dims.len == 1) 1 else self.shape.dims.len - 1;
        if (self.shape.dims.len == 1) {
            dims_stack[0] = 1;
        } else {
            var j: usize = 0;
            for (self.shape.dims, 0..) |dim, i| {
                if (i != axis) {
                    dims_stack[j] = dim;
                    j += 1;
                }
            }
        }
        var result = try Tensor.init(allocator, dims_stack[0..result_rank]);
        try result.fill(math.inf(f32));
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            var result_offset: usize = 0;
            var result_axis: usize = 0;
            for (0..self.shape.dims.len) |input_axis| {
                if (input_axis != axis) {
                    result_offset += iterator.indices[input_axis] * result.shape.strides[result_axis];
                    result_axis += 1;
                }
            }
            result.data[result_offset] = @min(result.data[result_offset], self.data[iterator.offset]);
            _ = iterator.advance();
        }
        return result;
    }

    pub fn variancePopulation(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var mean_tensor = try self.mean(allocator, axis);
        defer mean_tensor.deinit();
        var result = try Tensor.init(allocator, mean_tensor.shape.dims);
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            var result_offset: usize = 0;
            var result_axis: usize = 0;
            for (0..self.shape.dims.len) |input_axis| {
                if (input_axis != axis) {
                    result_offset += iterator.indices[input_axis] * result.shape.strides[result_axis];
                    result_axis += 1;
                }
            }
            const difference = self.data[iterator.offset] - mean_tensor.data[result_offset];
            result.data[result_offset] += difference * difference;
            _ = iterator.advance();
        }
        try result.divScalar(@floatFromInt(self.shape.dims[axis]));
        return result;
    }

    pub fn varianceSample(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var mean_tensor = try self.mean(allocator, axis);
        defer mean_tensor.deinit();
        var result = try Tensor.init(allocator, mean_tensor.shape.dims);
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            var result_offset: usize = 0;
            var result_axis: usize = 0;
            for (0..self.shape.dims.len) |input_axis| {
                if (input_axis != axis) {
                    result_offset += iterator.indices[input_axis] * result.shape.strides[result_axis];
                    result_axis += 1;
                }
            }
            const difference = self.data[iterator.offset] - mean_tensor.data[result_offset];
            result.data[result_offset] += difference * difference;
            _ = iterator.advance();
        }
        const n = self.shape.dims[axis];
        if (n > 1) {
            try result.divScalar(@floatFromInt(n - 1));
        }
        return result;
    }

    pub fn stddev(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var result = try self.variancePopulation(allocator, axis);
        try result.sqrt();
        return result;
    }

    pub fn stddevSample(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var result = try self.varianceSample(allocator, axis);
        try result.sqrt();
        return result;
    }

    pub fn normL2(self: *const Tensor) !f32 {
        var result: f32 = 0.0;
        const contiguous = self.shape.isContiguous();
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            const value = if (contiguous) self.data[count] else self.data[iterator.offset];
            result += value * value;
            if (!contiguous) _ = iterator.advance();
        }
        return @sqrt(result);
    }

    pub fn norm(self: *const Tensor, order: f32) !f32 {
        if (order <= 0.0) return Error.InvalidShape;
        var result: f32 = 0.0;
        const contiguous = self.shape.isContiguous();
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            const value = if (contiguous) self.data[count] else self.data[iterator.offset];
            result += math.pow(f32, @abs(value), order);
            if (!contiguous) _ = iterator.advance();
        }
        return math.pow(f32, result, 1.0 / order);
    }

    pub fn dot(self: *const Tensor, other: *const Tensor) !f32 {
        if (self.shape.dims.len != 1 or other.shape.dims.len != 1 or self.shape.dims[0] != other.shape.dims[0]) return Error.ShapeMismatch;
        var result: f32 = 0.0;
        const n = self.shape.dims[0];
        var i: usize = 0;
        while (i < n) : (i += 1) result += self.data[i * self.shape.strides[0]] * other.data[i * other.shape.strides[0]];
        return result;
    }

    pub fn outer(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
        if (a.shape.dims.len != 1 or b.shape.dims.len != 1) return Error.ShapeMismatch;
        var result = try Tensor.init(allocator, &.{ a.shape.dims[0], b.shape.dims[0] });
        var i: usize = 0;
        while (i < a.shape.dims[0]) : (i += 1) {
            var j: usize = 0;
            while (j < b.shape.dims[0]) : (j += 1) result.data[i * result.shape.strides[0] + j * result.shape.strides[1]] = a.data[i * a.shape.strides[0]] * b.data[j * b.shape.strides[0]];
        }
        return result;
    }

    pub fn trace(self: *const Tensor) !f32 {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return Error.MustBeSquare;
        var result: f32 = 0.0;
        var i: usize = 0;
        while (i < self.shape.dims[0]) : (i += 1) result += self.data[i * self.shape.strides[0] + i * self.shape.strides[1]];
        return result;
    }

    pub fn matmul(a: *const Tensor, b: *const Tensor, allocator: Allocator) !Tensor {
        if (a.shape.dims.len != 2 or b.shape.dims.len != 2 or a.shape.dims[1] != b.shape.dims[0]) return Error.ShapeMismatch;
        const m = a.shape.dims[0];
        const k = a.shape.dims[1];
        const n = b.shape.dims[1];
        var result = try Tensor.init(allocator, &.{ m, n });
        var a_contiguous = try a.copy(allocator);
        defer a_contiguous.deinit();
        var b_mutable = try b.copy(allocator);
        defer b_mutable.deinit();
        var b_transposed_view = try b_mutable.transpose(&.{ 1, 0 });
        defer b_transposed_view.deinit();
        var b_transposed = try b_transposed_view.copy(allocator);
        defer b_transposed.deinit();
        const Worker = struct {
            fn run(a_ptr: *const Tensor, bt_ptr: *const Tensor, out_ptr: *Tensor, start: usize, end: usize, k_dim: usize, n_dim: usize) void {
                const block: usize = 32;
                var ii: usize = start;
                while (ii < end) : (ii += block) {
                    const i_end = @min(ii + block, end);
                    var jj: usize = 0;
                    while (jj < n_dim) : (jj += block) {
                        const j_end = @min(jj + block, n_dim);
                        var i: usize = ii;
                        while (i < i_end) : (i += 1) {
                            var j: usize = jj;
                            while (j < j_end) : (j += 1) {
                                var sum_value: f32 = 0.0;
                                var kk: usize = 0;
                                const limit = k_dim - k_dim % vector_width;
                                var accumulator: Vec8 = @splat(0.0);
                                while (kk < limit) : (kk += vector_width) {
                                    const av: Vec8 = a_ptr.data[i * a_ptr.shape.strides[0] + kk..][0..vector_width].*;
                                    const bv: Vec8 = bt_ptr.data[j * bt_ptr.shape.strides[0] + kk..][0..vector_width].*;
                                    accumulator += av * bv;
                                }
                                sum_value += @reduce(.Add, accumulator);
                                while (kk < k_dim) : (kk += 1) {
                                    sum_value += a_ptr.data[i * a_ptr.shape.strides[0] + kk * a_ptr.shape.strides[1]] * bt_ptr.data[j * bt_ptr.shape.strides[0] + kk * bt_ptr.shape.strides[1]];
                                }
                                out_ptr.data[i * out_ptr.shape.strides[0] + j * out_ptr.shape.strides[1]] = sum_value;
                            }
                        }
                    }
                }
            }
        };
        const thread_count = @min(effectiveCpuCount(), @min(m, 8));
        if (thread_count <= 1) {
            Worker.run(&a_contiguous, &b_transposed, &result, 0, m, k, n);
        } else {
            var threads: [8]std.Thread = undefined;
            var active: usize = 0;
            const chunk = (m + thread_count - 1) / thread_count;
            var start: usize = 0;
            while (start < m and active < thread_count) : (start += chunk) {
                const end = @min(start + chunk, m);
                threads[active] = try std.Thread.spawn(.{}, Worker.run, .{ &a_contiguous, &b_transposed, &result, start, end, k, n });
                active += 1;
            }
            for (threads[0..active]) |thread| thread.join();
        }
        return result;
    }

    pub fn isClose(self: *const Tensor, other: *const Tensor, rtol: f32, atol: f32) !bool {
        if (!self.shape.equals(&other.shape)) return Error.ShapeMismatch;
        var a_iterator = TensorIterator.init(&self.shape);
        var b_iterator = TensorIterator.init(&other.shape);
        var i: usize = 0;
        while (i < self.shape.totalSize()) : (i += 1) {
            const av = self.data[a_iterator.offset];
            const bv = other.data[b_iterator.offset];
            if (@abs(av - bv) > atol + rtol * @abs(bv)) return false;
            _ = a_iterator.advance();
            _ = b_iterator.advance();
        }
        return true;
    }

    pub fn toInt(self: *const Tensor, allocator: Allocator) !Tensor {
        var result = try Tensor.init(allocator, self.shape.dims);
        var iterator = TensorIterator.init(&self.shape);
        var i: usize = 0;
        const max_precise_int: f32 = 16777216.0;
        while (i < self.shape.totalSize()) : (i += 1) {
            result.data[i] = @round(math.clamp(self.data[iterator.offset], -max_precise_int, max_precise_int));
            _ = iterator.advance();
        }
        return result;
    }

    pub fn toFixedFast(self: *const Tensor, allocator: Allocator) !Tensor {
        var result = try Tensor.init(allocator, self.shape.dims);
        var iterator = TensorIterator.init(&self.shape);
        var i: usize = 0;
        while (i < self.shape.totalSize()) : (i += 1) {
            result.data[i] = @floor(self.data[iterator.offset] * 4294967296.0) / 4294967296.0;
            _ = iterator.advance();
        }
        return result;
    }

    pub fn toFixed(self: *const Tensor, allocator: Allocator) !Tensor {
        return self.toFixedFast(allocator);
    }

    pub fn arange(allocator: Allocator, start: f32, end: f32, step: f32) !Tensor {
        if (step == 0.0) return Error.InvalidShape;
        if (step > 0 and start >= end) return Error.InvalidShape;
        if (step < 0 and start <= end) return Error.InvalidShape;
        const count_float = @ceil((end - start) / step);
        if (count_float <= 0.0 or !std.math.isFinite(count_float)) return Error.InvalidShape;
        const count: usize = @intFromFloat(count_float);
        if (count == 0) return Error.InvalidShape;
        var result = try Tensor.init(allocator, &.{count});
        var i: usize = 0;
        while (i < count) : (i += 1) result.data[i] = start + @as(f32, @floatFromInt(i)) * step;
        return result;
    }

    pub fn linspace(allocator: Allocator, start: f32, end: f32, count: usize) !Tensor {
        if (count == 0) return Error.InvalidShape;
        var result = try Tensor.init(allocator, &.{count});
        var i: usize = 0;
        while (i < count) : (i += 1) {
            result.data[i] = if (count == 1) start else start + (@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(count - 1))) * (end - start);
        }
        return result;
    }

    pub fn det(self: *const Tensor, allocator: Allocator) !f32 {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return Error.MustBeSquare;
        const n = self.shape.dims[0];
        var matrix = try self.copy(allocator);
        defer matrix.deinit();
        var determinant: f32 = 1.0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var pivot = i;
            var max_value = @abs(matrix.data[i * n + i]);
            var row: usize = i + 1;
            while (row < n) : (row += 1) {
                const value = @abs(matrix.data[row * n + i]);
                if (value > max_value) {
                    max_value = value;
                    pivot = row;
                }
            }
            if (max_value < 1e-30) return 0.0;
            if (pivot != i) {
                var col: usize = 0;
                while (col < n) : (col += 1) {
                    const temporary = matrix.data[i * n + col];
                    matrix.data[i * n + col] = matrix.data[pivot * n + col];
                    matrix.data[pivot * n + col] = temporary;
                }
                determinant = -determinant;
            }
            const pivot_value = matrix.data[i * n + i];
            determinant *= pivot_value;
            row = i + 1;
            while (row < n) : (row += 1) {
                const factor = matrix.data[row * n + i] / pivot_value;
                var col: usize = i;
                while (col < n) : (col += 1) matrix.data[row * n + col] -= factor * matrix.data[i * n + col];
            }
        }
        return determinant;
    }

    pub fn save(self: *const Tensor, writer: anytype) !void {
        const ndim: u64 = @intCast(self.shape.dims.len);
        try writer.writeInt(u64, ndim, .little);
        for (self.shape.dims) |d| try writer.writeInt(u64, @intCast(d), .little);
        for (self.data) |v| try writer.writeInt(u32, @bitCast(v), .little);
    }

    pub fn load(allocator: Allocator, reader: anytype) !Tensor {
        const ndim = try reader.readInt(u64, .little);
        if (ndim == 0 or ndim > 8) return Error.InvalidShape;
        var dims: [8]usize = undefined;
        var i: usize = 0;
        while (i < ndim) : (i += 1) {
            const d = try reader.readInt(u64, .little);
            if (d == 0) return Error.InvalidShape;
            dims[i] = @intCast(d);
        }
        var t = try Tensor.init(allocator, dims[0..ndim]);
        errdefer t.deinit();
        var j: usize = 0;
        while (j < t.data.len) : (j += 1) {
            t.data[j] = @bitCast(try reader.readInt(u32, .little));
        }
        return t;
    }

    pub fn eye(allocator: Allocator, dims: []const usize) !Tensor {
        if (dims.len != 2 or dims[0] != dims[1]) return Error.InvalidShape;
        var tensor = try init(allocator, dims);
        try tensor.fill(0.0);
        const n = dims[0];
        var i: usize = 0;
        while (i < n) : (i += 1) tensor.data[i * n + i] = 1.0;
        return tensor;
    }

    pub fn inverse(self: *const Tensor, allocator: Allocator) !Tensor {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return Error.MustBeSquare;
        const n = self.shape.dims[0];
        var augmented = try Tensor.init(allocator, &.{ n, 2 * n });
        defer augmented.deinit();
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) augmented.data[i * 2 * n + j] = self.data[i * self.shape.strides[0] + j * self.shape.strides[1]];
            augmented.data[i * 2 * n + i + n] = 1.0;
        }
        i = 0;
        while (i < n) : (i += 1) {
            var pivot = i;
            var max_value = @abs(augmented.data[i * 2 * n + i]);
            var row: usize = i + 1;
            while (row < n) : (row += 1) {
                const value = @abs(augmented.data[row * 2 * n + i]);
                if (value > max_value) {
                    max_value = value;
                    pivot = row;
                }
            }
            if (max_value < 1e-30) return Error.SingularMatrix;
            if (pivot != i) {
                var col: usize = 0;
                while (col < 2 * n) : (col += 1) {
                    const temporary = augmented.data[i * 2 * n + col];
                    augmented.data[i * 2 * n + col] = augmented.data[pivot * 2 * n + col];
                    augmented.data[pivot * 2 * n + col] = temporary;
                }
            }
            const pivot_value = augmented.data[i * 2 * n + i];
            var col: usize = 0;
            while (col < 2 * n) : (col += 1) augmented.data[i * 2 * n + col] /= pivot_value;
            row = 0;
            while (row < n) : (row += 1) {
                if (row != i) {
                    const factor = augmented.data[row * 2 * n + i];
                    col = 0;
                    while (col < 2 * n) : (col += 1) augmented.data[row * 2 * n + col] -= factor * augmented.data[i * 2 * n + col];
                }
            }
        }
        var result = try Tensor.init(allocator, &.{ n, n });
        i = 0;
        while (i < n) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) result.data[i * n + j] = augmented.data[i * 2 * n + j + n];
        }
        return result;
    }
};

test "Tensor init and basic operations" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.init(allocator, &.{ 2, 3 });
    defer tensor.deinit();
    try tensor.set(&.{ 0, 0 }, 1.0);
    try tensor.set(&.{ 1, 2 }, 6.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), try tensor.get(&.{ 0, 0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), try tensor.get(&.{ 1, 2 }), 1e-6);
}

test "Tensor operations" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.init(allocator, &.{ 2, 2 });
    defer tensor.deinit();
    try tensor.fill(2.0);
    try tensor.addScalar(3.0);
    try tensor.mulScalar(2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), try tensor.get(&.{ 0, 0 }), 1e-6);
}

test "Tensor matmul" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, &.{ 2, 3 });
    defer a.deinit();
    var b = try Tensor.init(allocator, &.{ 3, 2 });
    defer b.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;
    a.data[4] = 5.0;
    a.data[5] = 6.0;
    b.data[0] = 7.0;
    b.data[1] = 8.0;
    b.data[2] = 9.0;
    b.data[3] = 10.0;
    b.data[4] = 11.0;
    b.data[5] = 12.0;
    var c = try Tensor.matmul(&a, &b, allocator);
    defer c.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 58.0), try c.get(&.{ 0, 0 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), try c.get(&.{ 0, 1 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 139.0), try c.get(&.{ 1, 0 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 154.0), try c.get(&.{ 1, 1 }), 1e-5);
}

test "Tensor inverse and det" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.init(allocator, &.{ 2, 2 });
    defer tensor.deinit();
    tensor.data[0] = 4.0;
    tensor.data[1] = 7.0;
    tensor.data[2] = 2.0;
    tensor.data[3] = 6.0;
    const determinant = try tensor.det(allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), determinant, 1e-5);
    var inverse_tensor = try tensor.inverse(allocator);
    defer inverse_tensor.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), try inverse_tensor.get(&.{ 0, 0 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.7), try inverse_tensor.get(&.{ 0, 1 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.2), try inverse_tensor.get(&.{ 1, 0 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), try inverse_tensor.get(&.{ 1, 1 }), 1e-5);
}
