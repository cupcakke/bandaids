const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const core_types = @import("../core/types.zig");
const core_tensor = @import("../core/tensor.zig");
const core_memory = @import("../core/memory.zig");
const PRNG = core_types.PRNG;

var global_prng_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);

fn nextSeed() u64 {
    var prng = PRNG.init(global_prng_counter.fetchAdd(1, .monotonic));
    return prng.uint64();
}

fn shapesEqual(a: Shape, b: Shape) bool {
    if (a.dims.len != b.dims.len) return false;
    var i: usize = 0;
    while (i < a.dims.len) : (i += 1) {
        if (a.dims[i] != b.dims[i]) return false;
    }
    return true;
}

fn quantizeValue(value: f32, precision: Precision) f32 {
    if (!std.math.isFinite(value)) return value;
    return switch (precision) {
        .fp4 => blk: {
            const clamped = std.math.clamp(value, -6.0, 6.0);
            const abs_v = if (clamped < 0) -clamped else clamped;
            const sign: f32 = if (clamped < 0) -1.0 else 1.0;
            const best: f32 = if (abs_v < 0.25) 0.0
                else if (abs_v < 0.75) 0.5
                else if (abs_v < 1.25) 1.0
                else if (abs_v < 1.75) 1.5
                else if (abs_v < 2.5) 2.0
                else if (abs_v < 3.5) 3.0
                else if (abs_v < 5.0) 4.0
                else 6.0;
            break :blk sign * best;
        },
        .fp8 => blk: {
            const clamped = std.math.clamp(value, -448.0, 448.0);
            if (clamped == 0.0) break :blk 0.0;
            const sign: f32 = if (clamped < 0) -1.0 else 1.0;
            const abs_v = if (clamped < 0) -clamped else clamped;
            const exp_f = @floor(@log2(abs_v));
            const exp_clamped = std.math.clamp(exp_f, -9.0, 8.0);
            const step = std.math.pow(f32, 2.0, exp_clamped - 3.0);
            const quantized = @round(abs_v / step) * step;
            break :blk sign * std.math.clamp(quantized, 0.0, 448.0);
        },
        .fp16 => blk: {
            const clamped = std.math.clamp(value, -65504.0, 65504.0);
            if (clamped == 0.0) break :blk 0.0;
            const abs_v = if (clamped < 0) -clamped else clamped;
            const sign: f32 = if (clamped < 0) -1.0 else 1.0;
            const exp_f = @floor(@log2(abs_v));
            const exp_clamped = std.math.clamp(exp_f, -24.0, 15.0);
            const step = std.math.pow(f32, 2.0, exp_clamped - 10.0);
            break :blk sign * (@round(abs_v / step) * step);
        },
        .fp32, .fp64 => value,
    };
}

fn tensorFlagsToBits(flags: TensorFlags) u8 {
    var bits: u8 = 0;
    if (flags.in_tensor_memory) bits |= 0b001;
    if (flags.requires_grad) bits |= 0b010;
    if (flags.is_compressed) bits |= 0b100;
    return bits;
}

fn tensorFlagsFromBits(bits: u8) TensorFlags {
    return TensorFlags{
        .in_tensor_memory = (bits & 0b001) != 0,
        .requires_grad = (bits & 0b010) != 0,
        .is_compressed = (bits & 0b100) != 0,
    };
}

fn erfApprox(x: f32) f32 {
    const a1: f32 = 0.254829592;
    const a2: f32 = -0.284496736;
    const a3: f32 = 1.421413741;
    const a4: f32 = -1.453152027;
    const a5: f32 = 1.061405429;
    const p: f32 = 0.3275911;
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const abs_x = if (x < 0) -x else x;
    const t = 1.0 / (1.0 + p * abs_x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * @exp(-abs_x * abs_x);
    return sign * y;
}

pub const Precision = enum {
    fp4,
    fp8,
    fp16,
    fp32,
    fp64,
};

const Shape = struct {
    dims: []const usize,

    pub fn totalSize(self: Shape) usize {
        var size: usize = 1;
        for (self.dims) |dim| {
            if (dim != 0 and size > std.math.maxInt(usize) / dim) @panic("Shape.totalSize overflow");
            size *= dim;
        }
        return size;
    }
};

pub const TensorFlags = struct {
    in_tensor_memory: bool = false,
    requires_grad: bool = true,
    is_compressed: bool = false,
};

pub const Tensor = struct {
    data: []f32,
    shape: Shape,
    dtype: Precision = .fp32,
    flags: TensorFlags = .{},
    allocator: Allocator,

    pub fn init(allocator: Allocator, dims: []const usize) !Tensor {
        for (dims) |d| {
            if (d == 0) return error.InvalidShape;
        }
        const owned_dims = try allocator.dupe(usize, dims);
        errdefer allocator.free(owned_dims);

        const shape = Shape{ .dims = owned_dims };
        const size = shape.totalSize();

        const data = try allocator.alloc(f32, size);
        errdefer allocator.free(data);

        return Tensor{
            .data = data,
            .shape = shape,
            .allocator = allocator,
        };
    }

    pub fn zeros(allocator: Allocator, dims: []const usize) !Tensor {
        var tensor = try init(allocator, dims);
        tensor.fill(0.0);
        return tensor;
    }

    pub fn ones(allocator: Allocator, dims: []const usize) !Tensor {
        var tensor = try init(allocator, dims);
        tensor.fill(1.0);
        return tensor;
    }

    pub fn eye(allocator: Allocator, dims: []const usize) !Tensor {
        if (dims.len != 2 or dims[0] != dims[1]) return error.InvalidShape;
        var tensor = try init(allocator, dims);
        tensor.fill(0.0);
        const n = dims[0];
        var i: usize = 0;
        while (i < n) : (i += 1) {
            tensor.data[i * n + i] = 1.0;
        }
        return tensor;
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape.dims);
    }

    pub fn fill(self: *Tensor, value: f32) void {
        for (self.data) |*v| {
            v.* = value;
        }
    }

    pub fn fillRandomNormal(self: *Tensor, mean: f32, std_dev: f32) void {
        var prng = std.Random.DefaultPrng.init(nextSeed());
        const random = prng.random();

        var i: usize = 0;
        while (i + 1 < self.data.len) : (i += 2) {
            const rand_u = @max(random.float(f32), 1e-7);
            const rand_v = random.float(f32);
            const r = @sqrt(-2.0 * @log(rand_u));
            const theta = 2.0 * std.math.pi * rand_v;
            self.data[i] = mean + std_dev * r * @cos(theta);
            self.data[i + 1] = mean + std_dev * r * @sin(theta);
        }
        if (i < self.data.len) {
            const rand_u = @max(random.float(f32), 1e-7);
            const rand_v = random.float(f32);
            const z0 = @sqrt(-2.0 * @log(rand_u)) * @cos(2.0 * std.math.pi * rand_v);
            self.data[i] = mean + std_dev * z0;
        }
    }

    pub fn fillRademacher(self: *Tensor) void {
        var prng = std.Random.DefaultPrng.init(nextSeed());
        const random = prng.random();

        for (self.data) |*v| {
            v.* = if (random.float(f32) < 0.5) -1.0 else 1.0;
        }
    }

    pub fn clone(self: *const Tensor, allocator: Allocator) !Tensor {
        var new_tensor = try Tensor.init(allocator, self.shape.dims);
        @memcpy(new_tensor.data, self.data);
        new_tensor.dtype = self.dtype;
        new_tensor.flags = self.flags;
        new_tensor.flags.in_tensor_memory = false;
        return new_tensor;
    }

    pub fn copyFrom(self: *Tensor, other: *const Tensor) !void {
        if (!shapesEqual(self.shape, other.shape)) return error.ShapeMismatch;
        @memcpy(self.data, other.data);
        self.flags.requires_grad = other.flags.requires_grad;
        self.flags.is_compressed = other.flags.is_compressed;
        self.dtype = other.dtype;
    }

    pub fn copyFromWithCast(self: *Tensor, other: *const Tensor) !void {
        if (!shapesEqual(self.shape, other.shape)) return error.ShapeMismatch;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data[i] = quantizeValue(other.data[i], self.dtype);
        }
        self.flags.requires_grad = other.flags.requires_grad;
        self.flags.is_compressed = self.dtype == .fp4 or self.dtype == .fp8;
    }

    pub fn mulScalar(self: *Tensor, scalar: f32) void {
        for (self.data) |*v| {
            v.* *= scalar;
        }
    }

    pub fn add(self: *Tensor, other: *const Tensor) !void {
        if (!shapesEqual(self.shape, other.shape)) return error.ShapeMismatch;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data[i] += other.data[i];
        }
    }

    pub fn sub(self: *Tensor, other: *const Tensor) !void {
        if (!shapesEqual(self.shape, other.shape)) return error.ShapeMismatch;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data[i] -= other.data[i];
        }
    }

    pub fn normL2(self: *const Tensor) f32 {
        var sum: f64 = 0.0;
        for (self.data) |v| {
            if (std.math.isNan(v)) return std.math.nan(f32);
            if (!std.math.isFinite(v)) return std.math.inf(f32);
            sum += @as(f64, v) * @as(f64, v);
        }
        return @floatCast(@sqrt(sum));
    }

    pub fn spectralNorm(self: *const Tensor, allocator: Allocator, max_iter: usize, eps: f32) !f32 {
        if (self.shape.dims.len != 2) return error.InvalidShape;

        const m = self.shape.dims[0];
        const n = self.shape.dims[1];

        var v = try Tensor.init(allocator, &[_]usize{n});
        defer v.deinit();
        v.fillRandomNormal(0.0, 1.0);

        var u = try Tensor.init(allocator, &[_]usize{m});
        defer u.deinit();
        u.fill(0.0);

        const effective_iter = if (max_iter == 0) @as(usize, 1) else max_iter;
        var iter: usize = 0;
        while (iter < effective_iter) : (iter += 1) {
            u.fill(0.0);
            var i: usize = 0;
            while (i < m) : (i += 1) {
                var j: usize = 0;
                while (j < n) : (j += 1) {
                    u.data[i] += self.data[i * n + j] * v.data[j];
                }
            }

            const u_norm = u.normL2();
            if (std.math.isFinite(u_norm) and u_norm > eps) {
                u.mulScalar(1.0 / u_norm);
            }

            v.fill(0.0);
            i = 0;
            while (i < m) : (i += 1) {
                var j: usize = 0;
                while (j < n) : (j += 1) {
                    v.data[j] += self.data[i * n + j] * u.data[i];
                }
            }

            const v_norm = v.normL2();
            if (std.math.isFinite(v_norm) and v_norm > eps) {
                v.mulScalar(1.0 / v_norm);
            }
        }

        var sigma: f64 = 0.0;
        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                sigma += @as(f64, u.data[i]) * @as(f64, self.data[i * n + j]) * @as(f64, v.data[j]);
            }
        }

        const sigma_f32: f32 = @floatCast(sigma);
        return if (sigma_f32 >= 0) sigma_f32 else -sigma_f32;
    }

    pub fn matmul(self: *Tensor, A: *const Tensor, B: *const Tensor) !void {
        if (A.shape.dims.len != 2 or B.shape.dims.len != 2 or self.shape.dims.len != 2) return error.InvalidShape;

        const m = A.shape.dims[0];
        const k = A.shape.dims[1];
        const n = B.shape.dims[1];

        if (B.shape.dims[0] != k) return error.ShapeMismatch;
        if (self.shape.dims[0] != m or self.shape.dims[1] != n) return error.ShapeMismatch;
        if (self.data.len != m * n) return error.ShapeMismatch;

        if (self.data.ptr == A.data.ptr or self.data.ptr == B.data.ptr) {
            var temp = try Tensor.init(self.allocator, &[_]usize{ m, n });
            defer temp.deinit();
            try temp.matmul(A, B);
            try self.copyFrom(&temp);
            return;
        }

        self.fill(0.0);

        const b_transposed = try self.allocator.alignedAlloc(f32, @as(?u29, 32), n * k);
        defer self.allocator.free(b_transposed);
        var pi: usize = 0;
        while (pi < k) : (pi += 1) {
            var ji: usize = 0;
            while (ji < n) : (ji += 1) {
                b_transposed[ji * k + pi] = B.data[pi * n + ji];
            }
        }

        const SfdMatmulVecWidth = 8;
        const SfdMatmulVec8 = @Vector(SfdMatmulVecWidth, f32);
        const SfdMatmulBlock: usize = 32;

        const Worker = struct {
            fn run(a_ptr: []const f32, bt_ptr: []const f32, out_ptr: []f32, start: usize, end: usize, k_dim: usize, n_dim: usize) void {
                var ii: usize = start;
                while (ii < end) : (ii += SfdMatmulBlock) {
                    const i_end = @min(ii + SfdMatmulBlock, end);
                    var jj: usize = 0;
                    while (jj < n_dim) : (jj += SfdMatmulBlock) {
                        const j_end = @min(jj + SfdMatmulBlock, n_dim);
                        var i: usize = ii;
                        while (i < i_end) : (i += 1) {
                            var j: usize = jj;
                            while (j < j_end) : (j += 1) {
                                var accumulator: SfdMatmulVec8 = @splat(0.0);
                                var kk: usize = 0;
                                const limit = k_dim - k_dim % SfdMatmulVecWidth;
                                while (kk < limit) : (kk += SfdMatmulVecWidth) {
                                    const av: SfdMatmulVec8 = a_ptr[i * k_dim + kk ..][0..SfdMatmulVecWidth].*;
                                    const bv: SfdMatmulVec8 = bt_ptr[j * k_dim + kk ..][0..SfdMatmulVecWidth].*;
                                    accumulator += av * bv;
                                }
                                var sum_value: f32 = @reduce(.Add, accumulator);
                                while (kk < k_dim) : (kk += 1) {
                                    sum_value += a_ptr[i * k_dim + kk] * bt_ptr[j * k_dim + kk];
                                }
                                out_ptr[i * n_dim + j] = sum_value;
                            }
                        }
                    }
                }
            }
        };

        const thread_count = @min(core_tensor.effectiveCpuCount(), @min(m, 8));
        if (thread_count <= 1) {
            Worker.run(A.data, b_transposed, self.data, 0, m, k, n);
        } else {
            var threads: [8]std.Thread = undefined;
            var active: usize = 0;
            const chunk = (m + thread_count - 1) / thread_count;
            var start: usize = 0;
            while (start < m and active < thread_count) : (start += chunk) {
                const end = @min(start + chunk, m);
                threads[active] = try std.Thread.spawn(.{}, Worker.run, .{ A.data, b_transposed, self.data, start, end, k, n });
                active += 1;
            }
            for (threads[0..active]) |thread| thread.join();
        }
    }

    pub fn outerProduct(self: *const Tensor, allocator: Allocator, other: *const Tensor) !Tensor {
        const m = self.data.len;
        const n = other.data.len;

        var result = try Tensor.init(allocator, &[_]usize{ m, n });

        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                result.data[i * n + j] = self.data[i] * other.data[j];
            }
        }

        return result;
    }

    pub fn sizeBytes(self: *const Tensor) usize {
        return self.data.len * @sizeOf(f32);
    }

    pub fn convertToFP4(self: *Tensor) !void {
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data[i] = quantizeValue(self.data[i], .fp4);
        }
        self.dtype = .fp4;
        self.flags.is_compressed = true;
    }

    pub fn save(self: *const Tensor, writer: anytype) !void {
        try writer.writeInt(u32, 0x54464453, .little);
        try writer.writeInt(u8, @intFromEnum(self.dtype), .little);
        try writer.writeInt(u8, tensorFlagsToBits(self.flags), .little);
        try writer.writeInt(u64, @intCast(self.shape.dims.len), .little);
        for (self.shape.dims) |dim| {
            try writer.writeInt(u64, @intCast(dim), .little);
        }
        for (self.data) |val| {
            try writer.writeInt(u32, @as(u32, @bitCast(val)), .little);
        }
    }

    pub fn load(allocator: Allocator, reader: anytype) !Tensor {
        const magic = try reader.readInt(u32, .little);
        if (magic != 0x54464453) return error.InvalidTensorFormat;

        const dtype_raw = try reader.readInt(u8, .little);
        const flags_raw = try reader.readInt(u8, .little);
        const ndims_u64 = try reader.readInt(u64, .little);
        if (ndims_u64 > @as(u64, std.math.maxInt(usize))) return error.InvalidShape;
        const ndims: usize = @intCast(ndims_u64);
        var dims = try allocator.alloc(usize, ndims);
        errdefer allocator.free(dims);

        var i: usize = 0;
        while (i < ndims) : (i += 1) {
            const dim_u64 = try reader.readInt(u64, .little);
            if (dim_u64 > @as(u64, std.math.maxInt(usize))) return error.InvalidShape;
            dims[i] = @intCast(dim_u64);
        }

        var tensor = try Tensor.init(allocator, dims);
        errdefer tensor.deinit();

        allocator.free(dims);

        tensor.dtype = try std.meta.intToEnum(Precision, dtype_raw);
        tensor.flags = tensorFlagsFromBits(flags_raw);

        i = 0;
        while (i < tensor.data.len) : (i += 1) {
            const bits = try reader.readInt(u32, .little);
            tensor.data[i] = @as(f32, @bitCast(bits));
        }

        return tensor;
    }

    pub fn fromCoreTensor(ct: *const core_tensor.Tensor, allocator: Allocator) !Tensor {
        const t = try Tensor.init(allocator, ct.shape.dims);
        @memcpy(t.data, ct.data);
        return t;
    }

    pub fn toCoreTensor(self: *const Tensor, allocator: Allocator) !core_tensor.Tensor {
        const ct = try core_tensor.Tensor.init(allocator, self.shape.dims);
        @memcpy(ct.data, self.data);
        return ct;
    }

    pub fn initWithArena(arena: *core_memory.ArenaAllocator, dims: []const usize) !Tensor {
        return init(arena.allocator(), dims);
    }

    pub fn initWithPool(pool: *core_memory.PoolAllocator, dims: []const usize) !Tensor {
        return init(pool.allocator(), dims);
    }

    pub fn initWithSlab(slab: *core_memory.SlabAllocator, dims: []const usize) !Tensor {
        return init(slab.allocator(), dims);
    }

    pub fn initWithBuddy(buddy: *core_memory.BuddyAllocator, dims: []const usize) !Tensor {
        return init(buddy.allocator(), dims);
    }
};

pub const LossFn = *const fn (params: *const Tensor, context: ?*anyopaque) anyerror!f32;

pub const SFDConfig = struct {
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    clip_threshold: f32 = 1.0,
    fisher_max: f32 = 1e6,
    warmup_steps: usize = 10,
    finite_diff_eps: f32 = 1e-5,
    second_order_eps: f32 = 1e-4,
    use_external_fisher: bool = false,
};

pub const KFACBlock = struct {
    A_diag: Tensor,
    G_diag: Tensor,
    damping: f32,
    alpha: f32,
    update_freq: usize,
    last_update: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, input_dim: usize, output_dim: usize, damping: f32) !KFACBlock {
        return initWithAlpha(allocator, input_dim, output_dim, damping, 0.95);
    }

    pub fn initWithAlpha(allocator: Allocator, input_dim: usize, output_dim: usize, damping: f32, alpha: f32) !KFACBlock {
        const A_shape = [_]usize{ input_dim, input_dim };
        const G_shape = [_]usize{ output_dim, output_dim };

        var A = try Tensor.eye(allocator, &A_shape);
        errdefer A.deinit();

        var G = try Tensor.eye(allocator, &G_shape);
        errdefer G.deinit();

        return KFACBlock{
            .A_diag = A,
            .G_diag = G,
            .damping = damping,
            .alpha = alpha,
            .update_freq = 10,
            .last_update = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *KFACBlock) void {
        self.A_diag.deinit();
        self.G_diag.deinit();
    }

    pub fn updateStatistics(self: *KFACBlock, activations: *const Tensor, gradients: *const Tensor) !void {
        const a_dim = self.A_diag.shape.dims[0];
        const g_dim = self.G_diag.shape.dims[0];

        var row: usize = 0;
        while (row < a_dim) : (row += 1) {
            var col: usize = 0;
            while (col < a_dim) : (col += 1) {
                const idx = row * a_dim + col;
                const a_r: f32 = if (row < activations.data.len) activations.data[row] else 0.0;
                const a_c: f32 = if (col < activations.data.len) activations.data[col] else 0.0;
                const diag_term: f32 = if (row == col) self.damping else 0.0;
                const target = a_r * a_c + diag_term;
                self.A_diag.data[idx] = self.alpha * self.A_diag.data[idx] + (1.0 - self.alpha) * target;
            }
        }

        row = 0;
        while (row < g_dim) : (row += 1) {
            var col: usize = 0;
            while (col < g_dim) : (col += 1) {
                const idx = row * g_dim + col;
                const g_r: f32 = if (row < gradients.data.len) gradients.data[row] else 0.0;
                const g_c: f32 = if (col < gradients.data.len) gradients.data[col] else 0.0;
                const diag_term: f32 = if (row == col) self.damping else 0.0;
                const target = g_r * g_c + diag_term;
                self.G_diag.data[idx] = self.alpha * self.G_diag.data[idx] + (1.0 - self.alpha) * target;
            }
        }
    }

    pub fn preconditionGradient(self: *const KFACBlock, grad: *Tensor) !void {
        var A_inv_sqrt = try self.computeInverseSqrt(&self.A_diag);
        defer A_inv_sqrt.deinit();

        var G_inv_sqrt = try self.computeInverseSqrt(&self.G_diag);
        defer G_inv_sqrt.deinit();

        const g_dim = self.G_diag.shape.dims[0];
        const a_dim = self.A_diag.shape.dims[0];

        if (grad.shape.dims.len == 2 and grad.shape.dims[0] == g_dim and grad.shape.dims[1] == a_dim) {
            var original = try grad.clone(self.allocator);
            defer original.deinit();

            var g_scaled = try Tensor.init(self.allocator, &[_]usize{ g_dim, a_dim });
            errdefer g_scaled.deinit();
            try g_scaled.matmul(&G_inv_sqrt, &original);

            try grad.matmul(&g_scaled, &A_inv_sqrt);
            g_scaled.deinit();
            return;
        }

        var idx: usize = 0;
        while (idx < grad.data.len) : (idx += 1) {
            const left_idx = idx % g_dim;
            const right_idx = idx % a_dim;
            const left_scale = G_inv_sqrt.data[left_idx * g_dim + left_idx];
            const right_scale = A_inv_sqrt.data[right_idx * a_dim + right_idx];
            grad.data[idx] *= left_scale * right_scale;
        }
    }

    fn computeInverseSqrt(self: *const KFACBlock, M: *const Tensor) !Tensor {
        if (M.shape.dims.len != 2 or M.shape.dims[0] != M.shape.dims[1]) return error.InvalidShape;
        const n = M.shape.dims[0];

        var m_damped = try Tensor.init(self.allocator, M.shape.dims);
        errdefer m_damped.deinit();
        @memcpy(m_damped.data, M.data);
        var d: usize = 0;
        while (d < n) : (d += 1) {
            m_damped.data[d * n + d] += self.damping;
        }

        var frob_sq: f64 = 0.0;
        for (m_damped.data) |v| {
            frob_sq += @as(f64, v) * @as(f64, v);
        }
        const frob: f32 = @floatCast(@sqrt(frob_sq));
        if (frob < 1e-12) {
            const result = try Tensor.eye(self.allocator, M.shape.dims);
            return result;
        }

        var y = try Tensor.eye(self.allocator, M.shape.dims);
        errdefer y.deinit();
        const inv_frob = 1.0 / frob;
        y.mulScalar(inv_frob);

        var z = try Tensor.eye(self.allocator, M.shape.dims);
        errdefer z.deinit();
        z.mulScalar(frob);

        const max_iters: usize = 20;
        var iter: usize = 0;
        while (iter < max_iters) : (iter += 1) {
            var yz = try Tensor.init(self.allocator, M.shape.dims);
            errdefer yz.deinit();
            try yz.matmul(&y, &z);

            var myz = try Tensor.init(self.allocator, M.shape.dims);
            errdefer myz.deinit();
            try myz.matmul(&m_damped, &yz);

            var three_i_minus_myz = try Tensor.eye(self.allocator, M.shape.dims);
            errdefer three_i_minus_myz.deinit();
            three_i_minus_myz.mulScalar(3.0);
            var k: usize = 0;
            while (k < n * n) : (k += 1) {
                three_i_minus_myz.data[k] -= myz.data[k];
            }
            three_i_minus_myz.mulScalar(0.5);

            var y_new = try Tensor.init(self.allocator, M.shape.dims);
            errdefer y_new.deinit();
            try y_new.matmul(&three_i_minus_myz, &y);
            y.deinit();
            y = y_new;

            var zy = try Tensor.init(self.allocator, M.shape.dims);
            errdefer zy.deinit();
            try zy.matmul(&z, &y);

            var zy_m = try Tensor.init(self.allocator, M.shape.dims);
            errdefer zy_m.deinit();
            try zy_m.matmul(&zy, &m_damped);

            var three_i_minus_zym = try Tensor.eye(self.allocator, M.shape.dims);
            errdefer three_i_minus_zym.deinit();
            three_i_minus_zym.mulScalar(3.0);
            k = 0;
            while (k < n * n) : (k += 1) {
                three_i_minus_zym.data[k] -= zy_m.data[k];
            }
            three_i_minus_zym.mulScalar(0.5);

            var z_new = try Tensor.init(self.allocator, M.shape.dims);
            errdefer z_new.deinit();
            try z_new.matmul(&z, &three_i_minus_zym);
            z.deinit();
            z = z_new;

            myz.deinit();
            yz.deinit();
            zy.deinit();
            zy_m.deinit();
            three_i_minus_myz.deinit();
            three_i_minus_zym.deinit();

            var diff = try Tensor.init(self.allocator, M.shape.dims);
            errdefer diff.deinit();
            @memcpy(diff.data, y.data);
            var yy = try Tensor.init(self.allocator, M.shape.dims);
            errdefer yy.deinit();
            try yy.matmul(&y, &y);
            var yym = try Tensor.init(self.allocator, M.shape.dims);
            errdefer yym.deinit();
            try yym.matmul(&yy, &m_damped);
            var ii: usize = 0;
            var delta_sq: f64 = 0.0;
            while (ii < n * n) : (ii += 1) {
                diff.data[ii] -= yym.data[ii];
                delta_sq += @as(f64, diff.data[ii]) * @as(f64, diff.data[ii]);
            }
            diff.deinit();
            yy.deinit();
            yym.deinit();
            if (@sqrt(@as(f32, @floatCast(delta_sq))) < 1e-4) break;
        }

        m_damped.deinit();
        z.deinit();
        return y;
    }
};

pub const SpectralNormalizerConfig = struct {
    power_iterations: usize = 5,
    eps: f32 = 1e-12,
    max_singular_value: f32 = 1.0,
};

pub const SpectralNormalizer = struct {
    power_iterations: usize,
    eps: f32,
    max_singular_value: f32,

    pub fn init(power_iterations: usize) SpectralNormalizer {
        return SpectralNormalizer{
            .power_iterations = power_iterations,
            .eps = 1e-12,
            .max_singular_value = 1.0,
        };
    }

    pub fn initWithConfig(config: SpectralNormalizerConfig) SpectralNormalizer {
        return SpectralNormalizer{
            .power_iterations = config.power_iterations,
            .eps = config.eps,
            .max_singular_value = config.max_singular_value,
        };
    }

    pub fn normalizeWeights(self: *SpectralNormalizer, weights: *Tensor, allocator: Allocator) !void {
        const sigma = try weights.spectralNorm(allocator, self.power_iterations, self.eps);

        if (sigma > self.max_singular_value) {
            weights.mulScalar(self.max_singular_value / sigma);
        }
    }

    pub fn lipschitzRegularization(_: *const SpectralNormalizer, loss: f32, spectral_norms: []const f32, lambda: f32) f32 {
        var reg_term: f32 = 0.0;
        for (spectral_norms) |sigma| {
            const deviation = sigma - 1.0;
            reg_term += deviation * deviation;
        }

        return loss + lambda * reg_term;
    }
};

pub const GradientFlowConfig = struct {
    gradient_clip_norm: f32 = 1.0,
    use_normalized_gradient_flow: bool = true,
    spectral_power_iterations: usize = 5,
};

pub const GradientFlowController = struct {
    spectral_normalizer: SpectralNormalizer,
    gradient_clip_norm: f32,
    use_normalized_gradient_flow: bool,
    step_counter: usize,

    pub fn init() GradientFlowController {
        return GradientFlowController{
            .spectral_normalizer = SpectralNormalizer.init(5),
            .gradient_clip_norm = 1.0,
            .use_normalized_gradient_flow = true,
            .step_counter = 0,
        };
    }

    pub fn initWithConfig(config: GradientFlowConfig) GradientFlowController {
        return GradientFlowController{
            .spectral_normalizer = SpectralNormalizer.init(config.spectral_power_iterations),
            .gradient_clip_norm = config.gradient_clip_norm,
            .use_normalized_gradient_flow = config.use_normalized_gradient_flow,
            .step_counter = 0,
        };
    }

    pub fn stabilizeGradients(self: *GradientFlowController, gradients: []*Tensor, weights: []*Tensor, allocator: Allocator) !void {
        self.step_counter += 1;
        if (self.step_counter % 10 == 0) {
            for (weights) |w| {
                try self.spectral_normalizer.normalizeWeights(w, allocator);
            }
        }

        if (self.use_normalized_gradient_flow) {
            for (gradients) |grad| {
                if (grad.data.len == 0) continue;

                const norm = grad.normL2();
                if (!std.math.isFinite(norm) or norm < 1e-12) continue;

                if (norm > self.gradient_clip_norm) {
                    const scale = self.gradient_clip_norm / norm;
                    grad.mulScalar(scale);
                }
            }
        }
    }
};

pub const MARSConfig = struct {
    snapshot_freq: usize = 100,
    scale_factor: f32 = 1.0,
    momentum: f32 = 0.9,
};

pub const MARSVarianceReducer = struct {
    reference_gradients: []Tensor,
    snapshot_freq: usize,
    scale_factor: f32,
    momentum: f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, param_shapes: []const []const usize, config: MARSConfig) !MARSVarianceReducer {
        var ref_grads = try allocator.alloc(Tensor, param_shapes.len);
        errdefer allocator.free(ref_grads);

        var initialized: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < initialized) : (idx += 1) {
                ref_grads[idx].deinit();
            }
        }

        var i: usize = 0;
        while (i < param_shapes.len) : (i += 1) {
            const shape = param_shapes[i];
            ref_grads[i] = try Tensor.init(allocator, shape);
            ref_grads[i].fill(0.0);
            initialized += 1;
        }

        return MARSVarianceReducer{
            .reference_gradients = ref_grads,
            .snapshot_freq = config.snapshot_freq,
            .scale_factor = config.scale_factor,
            .momentum = config.momentum,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MARSVarianceReducer) void {
        for (self.reference_gradients) |*rg| {
            rg.deinit();
        }
        self.allocator.free(self.reference_gradients);
    }

    pub fn varianceReducedGradient(self: *MARSVarianceReducer, current_grad: *const Tensor, reference_grad: *const Tensor, param_idx: usize) !Tensor {
        if (param_idx >= self.reference_gradients.len) return error.InvalidParameterIndex;
        if (!shapesEqual(current_grad.shape, reference_grad.shape)) return error.ShapeMismatch;
        if (!shapesEqual(current_grad.shape, self.reference_gradients[param_idx].shape)) return error.ShapeMismatch;

        var vr_grad = try Tensor.init(self.allocator, current_grad.shape.dims);
        errdefer vr_grad.deinit();

        var i: usize = 0;
        while (i < vr_grad.data.len) : (i += 1) {
            const g_current = current_grad.data[i];
            const g_ref = self.reference_gradients[param_idx].data[i];
            const ref_grad_val = reference_grad.data[i];

            const variance_reduced = g_current - ref_grad_val + g_ref;
            vr_grad.data[i] = self.momentum * variance_reduced + (1.0 - self.momentum) * g_current;
            vr_grad.data[i] *= self.scale_factor;
        }

        return vr_grad;
    }

    pub fn updateReferenceGradients(self: *MARSVarianceReducer, full_batch_gradients: []const Tensor) !void {
        if (full_batch_gradients.len != self.reference_gradients.len) return error.GradientCountMismatch;
        var i: usize = 0;
        while (i < full_batch_gradients.len) : (i += 1) {
            const grad = full_batch_gradients[i];
            if (!shapesEqual(self.reference_gradients[i].shape, grad.shape)) return error.ShapeMismatch;
            @memcpy(self.reference_gradients[i].data, grad.data);
        }
    }
};

pub const ReversibleOptimizerState = struct {
    forward_cache_policy: CachePolicy,
    recompute_threshold: f32,
    available_memory_bytes: f32,
    jacobian_cache: std.AutoHashMap(usize, Tensor),
    allocator: Allocator,

    pub const CachePolicy = enum {
        cache_all,
        recompute_all,
        adaptive,
    };

    pub fn init(allocator: Allocator) ReversibleOptimizerState {
        return ReversibleOptimizerState{
            .forward_cache_policy = .adaptive,
            .recompute_threshold = 0.5,
            .available_memory_bytes = 1024.0 * 1024.0 * 1024.0,
            .jacobian_cache = std.AutoHashMap(usize, Tensor).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ReversibleOptimizerState) void {
        var iter = self.jacobian_cache.iterator();
        while (iter.next()) |entry| {
            var tensor = entry.value_ptr.*;
            tensor.deinit();
        }
        self.jacobian_cache.deinit();
    }

    pub fn shouldRecompute(self: *ReversibleOptimizerState, layer_idx: usize, computation_cost: f32, memory_cost: f32, available_memory: f32) bool {
        switch (self.forward_cache_policy) {
            .cache_all => return false,
            .recompute_all => return true,
            .adaptive => {
                if (self.jacobian_cache.contains(layer_idx)) return false;
                if (available_memory < memory_cost) return true;

                const recompute_cost = computation_cost;
                const cache_cost = memory_cost * self.recompute_threshold;

                return recompute_cost < cache_cost;
            },
        }
    }

    pub fn backwardPassReversible(self: *ReversibleOptimizerState, forward_outputs: []const Tensor, grad_output: *const Tensor) ![]Tensor {
        var grad_inputs = try self.allocator.alloc(Tensor, forward_outputs.len);
        errdefer self.allocator.free(grad_inputs);

        var initialized = try self.allocator.alloc(bool, forward_outputs.len);
        defer self.allocator.free(initialized);
        @memset(initialized, false);
        errdefer {
            var idx: usize = 0;
            while (idx < forward_outputs.len) : (idx += 1) {
                if (initialized[idx]) {
                    grad_inputs[idx].deinit();
                }
            }
        }

        var current_grad = try grad_output.clone(self.allocator);
        defer current_grad.deinit();

        var i: usize = forward_outputs.len;
        while (i > 0) : (i -= 1) {
            const layer_idx = i - 1;

            const mem_bytes: f32 = @floatFromInt(forward_outputs[layer_idx].sizeBytes());
            const cached_count_f: f32 = @floatFromInt(self.jacobian_cache.count());
            const cached_size: f32 = cached_count_f * mem_bytes;
            const available: f32 = @max(0.0, self.available_memory_bytes - cached_size);
            const comp_cost: f32 = mem_bytes / (1024.0 * 1024.0);

            const should_recomp = self.shouldRecompute(layer_idx, comp_cost, mem_bytes, available);

            if (should_recomp) {
                var reconstructed_input = try self.reverseLayer(&forward_outputs[layer_idx], layer_idx);
                defer reconstructed_input.deinit();

                grad_inputs[layer_idx] = try self.computeGradient(&reconstructed_input, &current_grad);
            } else {
                grad_inputs[layer_idx] = try current_grad.clone(self.allocator);
            }
            initialized[layer_idx] = true;

            try current_grad.copyFrom(&grad_inputs[layer_idx]);
        }

        return grad_inputs;
    }

    fn reverseLayer(self: *ReversibleOptimizerState, output: *const Tensor, layer_idx: usize) !Tensor {
        var x = try output.clone(self.allocator);
        errdefer x.deinit();

        const max_iter: usize = 10;
        var iter: usize = 0;
        while (iter < max_iter) : (iter += 1) {
            var g_x = try self.computeResidual(&x, layer_idx);
            defer g_x.deinit();

            try x.copyFrom(output);
            try x.sub(&g_x);

            const delta = g_x.normL2();
            if (delta < 1e-6) break;
        }

        return x;
    }

    fn computeResidual(self: *ReversibleOptimizerState, input: *const Tensor, layer_idx: usize) !Tensor {
        if (self.jacobian_cache.get(layer_idx)) |cached| {
            if (shapesEqual(input.shape, cached.shape)) {
                var residual = try input.clone(self.allocator);
                errdefer residual.deinit();
                try residual.sub(&cached);
                return residual;
            }
        }
        var residual = try input.clone(self.allocator);
        errdefer residual.deinit();
        var i: usize = 0;
        while (i < residual.data.len) : (i += 1) {
            const v = residual.data[i];
            residual.data[i] = v * v * (1.0 / (1.0 + @abs(v)));
        }
        return residual;
    }

    fn computeGradient(self: *ReversibleOptimizerState, input: *const Tensor, grad_output: *const Tensor) !Tensor {
        var grad_input = try grad_output.clone(self.allocator);
        errdefer grad_input.deinit();
        const input_norm = input.normL2();
        if (std.math.isFinite(input_norm) and input_norm > 1e-8) {
            var i: usize = 0;
            while (i < grad_input.data.len) : (i += 1) {
                const x_val = input.data[i];
                const abs_x = if (x_val < 0) -x_val else x_val;
                const scale = @min(1.0, abs_x / input_norm);
                grad_input.data[i] *= scale;
            }
        }
        return grad_input;
    }
};

pub const LRScheduleType = enum {
    cosine_annealing,
    cosine_annealing_with_warmup,
    polynomial_decay,
    exponential_decay,
    one_cycle,
    sophia_style,
};

pub const LRScheduler = struct {
    schedule_type: LRScheduleType,
    base_lr: f32,
    min_lr: f32,
    max_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, schedule_type: LRScheduleType, base_lr: f32, warmup_steps: usize, total_steps: usize) LRScheduler {
        return LRScheduler{
            .schedule_type = schedule_type,
            .base_lr = base_lr,
            .min_lr = base_lr * 0.01,
            .max_lr = base_lr * 10.0,
            .warmup_steps = warmup_steps,
            .total_steps = total_steps,
            .current_step = 0,
            .allocator = allocator,
        };
    }

    fn selectOMedian(hess_values: []f32) f32 {
        if (hess_values.len == 0) return 0.0;
        if (hess_values.len == 1) return hess_values[0];
        const target = hess_values.len / 2;
        var lo: usize = 0;
        var hi: usize = hess_values.len;
        while (lo < hi) {
            const pivot_val = hess_values[lo + (hi - lo) / 2];
            var lt: usize = lo;
            var eq: usize = lo;
            var gt: usize = hi;
            while (eq < gt) {
                if (hess_values[eq] < pivot_val) {
                    const tmp = hess_values[lt];
                    hess_values[lt] = hess_values[eq];
                    hess_values[eq] = tmp;
                    lt += 1;
                    eq += 1;
                } else if (hess_values[eq] == pivot_val) {
                    eq += 1;
                } else {
                    gt -= 1;
                    const tmp = hess_values[eq];
                    hess_values[eq] = hess_values[gt];
                    hess_values[gt] = tmp;
                }
            }
            if (target < lt) {
                hi = lt;
            } else if (target < gt) {
                return hess_values[target];
            } else {
                lo = gt;
            }
        }
        return hess_values[target];
    }

    pub fn getLearningRate(self: *LRScheduler, hessian_info: ?*const Tensor) !f32 {
        const decay_steps = if (self.total_steps > self.warmup_steps) self.total_steps - self.warmup_steps else 1;

        if (self.warmup_steps > 0 and self.current_step < self.warmup_steps) {
            const warmup_progress = @as(f32, @floatFromInt(self.current_step)) / @as(f32, @floatFromInt(self.warmup_steps));
            return self.base_lr * warmup_progress;
        }

        var lr = self.base_lr;
        switch (self.schedule_type) {
            .cosine_annealing, .cosine_annealing_with_warmup => {
                const progress = @as(f32, @floatFromInt(self.current_step - @min(self.current_step, self.warmup_steps))) / @as(f32, @floatFromInt(decay_steps));
                const p = @min(progress, 1.0);
                lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + @cos(std.math.pi * p));
            },
            .polynomial_decay => {
                const progress = @as(f32, @floatFromInt(self.current_step - @min(self.current_step, self.warmup_steps))) / @as(f32, @floatFromInt(decay_steps));
                const p = @min(progress, 1.0);
                const power: f32 = 2.0;
                lr = self.base_lr * std.math.pow(f32, @max(0.0, 1.0 - p), power);
            },
            .exponential_decay => {
                const decay_rate: f32 = 0.96;
                const decay_interval: f32 = 1000.0;
                const steps_since_warmup = @as(f32, @floatFromInt(self.current_step - @min(self.current_step, self.warmup_steps)));
                lr = self.base_lr * std.math.pow(f32, decay_rate, steps_since_warmup / decay_interval);
            },
            .one_cycle => {
                const mid_point = @max(self.warmup_steps +| 1, self.total_steps / 2);
                if (self.current_step < mid_point) {
                    const rise_steps = @max(@as(usize, 1), mid_point - self.warmup_steps);
                    const progress = @as(f32, @floatFromInt(self.current_step - @min(self.current_step, self.warmup_steps))) / @as(f32, @floatFromInt(rise_steps));
                    const p = @min(progress, 1.0);
                    lr = self.base_lr + (self.max_lr - self.base_lr) * p;
                } else {
                    const fall_steps = @max(@as(usize, 1), if (self.total_steps > mid_point) self.total_steps - mid_point else 1);
                    const progress = @as(f32, @floatFromInt(self.current_step - mid_point)) / @as(f32, @floatFromInt(fall_steps));
                    const p = @min(progress, 1.0);
                    lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1.0 + @cos(std.math.pi * p));
                }
            },
            .sophia_style => {
                if (hessian_info) |hess| {
                    if (hess.data.len == 0) {
                        lr = self.base_lr;
                    } else {
                        const hess_values = try self.allocator.alloc(f32, hess.data.len);
                        defer self.allocator.free(hess_values);
                        @memcpy(hess_values, hess.data);
                        const median_hess = selectOMedian(hess_values);
                        lr = self.base_lr / @sqrt(@max(median_hess, 1e-8));
                        lr = std.math.clamp(lr, self.min_lr, self.max_lr);
                    }
                } else {
                    lr = self.base_lr;
                }
            },
        }

        return std.math.clamp(lr, self.min_lr, self.max_lr);
    }

    pub fn step(self: *LRScheduler) void {
        self.current_step +|= 1;
    }
};

pub const MixedPrecisionConfig = struct {
    use_fp4: bool = true,
    use_fp8: bool = true,
    use_fp16: bool = true,
    master_weights_precision: Precision = .fp32,
    gradient_accumulation_steps: usize = 4,
    loss_scale: f32 = 1024.0,
    max_loss_scale: f32 = 65536.0,
    dynamic_loss_scaling: bool = true,
};

pub const DynamicLossScaler = struct {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    steps_since_last_overflow: usize,
    max_scale: f32,

    pub fn init(initial_scale: f32) DynamicLossScaler {
        return initWithMaxScale(initial_scale, 65536.0);
    }

    pub fn initWithMaxScale(initial_scale: f32, max_scale: f32) DynamicLossScaler {
        return DynamicLossScaler{
            .scale = initial_scale,
            .growth_factor = 2.0,
            .backoff_factor = 0.5,
            .growth_interval = 2000,
            .steps_since_last_overflow = 0,
            .max_scale = max_scale,
        };
    }

    pub fn update(self: *DynamicLossScaler, gradients: []const Tensor) void {
        var has_overflow = false;
        for (gradients) |grad| {
            for (grad.data) |g| {
                if (!std.math.isFinite(g)) {
                    has_overflow = true;
                    break;
                }
            }
            if (has_overflow) break;
        }

        if (has_overflow) {
            self.scale *= self.backoff_factor;
            self.steps_since_last_overflow = 0;
        } else {
            self.steps_since_last_overflow += 1;
            if (self.steps_since_last_overflow >= self.growth_interval) {
                self.scale *= self.growth_factor;
                self.steps_since_last_overflow = 0;
            }
        }

        self.scale = std.math.clamp(self.scale, 1.0, self.max_scale);
    }
};

pub const MixedPrecisionTrainer = struct {
    config: MixedPrecisionConfig,
    master_weights: []Tensor,
    working_weights: []Tensor,
    accumulated_gradients: []Tensor,
    accumulation_counter: usize,
    loss_scaler: DynamicLossScaler,
    allocator: Allocator,

    pub fn init(allocator: Allocator, weight_shapes: []const []const usize, config: MixedPrecisionConfig) !MixedPrecisionTrainer {
        var master_w = try allocator.alloc(Tensor, weight_shapes.len);
        errdefer allocator.free(master_w);

        var working_w = try allocator.alloc(Tensor, weight_shapes.len);
        errdefer allocator.free(working_w);

        var accum_g = try allocator.alloc(Tensor, weight_shapes.len);
        errdefer allocator.free(accum_g);

        var init_master: usize = 0;
        var init_working: usize = 0;
        var init_accum: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < init_master) : (idx += 1) {
                master_w[idx].deinit();
            }
            idx = 0;
            while (idx < init_working) : (idx += 1) {
                working_w[idx].deinit();
            }
            idx = 0;
            while (idx < init_accum) : (idx += 1) {
                accum_g[idx].deinit();
            }
        }

        var i: usize = 0;
        while (i < weight_shapes.len) : (i += 1) {
            const shape = weight_shapes[i];
            master_w[i] = try Tensor.init(allocator, shape);
            master_w[i].dtype = .fp32;
            master_w[i].fillRandomNormal(0.0, 0.02);
            init_master += 1;

            working_w[i] = try Tensor.init(allocator, shape);
            working_w[i].dtype = if (config.use_fp4) .fp4 else if (config.use_fp8) .fp8 else .fp16;
            try working_w[i].copyFromWithCast(&master_w[i]);
            init_working += 1;

            accum_g[i] = try Tensor.init(allocator, shape);
            accum_g[i].dtype = .fp32;
            accum_g[i].fill(0.0);
            init_accum += 1;
        }

        return MixedPrecisionTrainer{
            .config = config,
            .master_weights = master_w,
            .working_weights = working_w,
            .accumulated_gradients = accum_g,
            .accumulation_counter = 0,
            .loss_scaler = DynamicLossScaler.initWithMaxScale(config.loss_scale, config.max_loss_scale),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MixedPrecisionTrainer) void {
        for (self.master_weights) |*w| {
            w.deinit();
        }
        self.allocator.free(self.master_weights);

        for (self.working_weights) |*w| {
            w.deinit();
        }
        self.allocator.free(self.working_weights);

        for (self.accumulated_gradients) |*g| {
            g.deinit();
        }
        self.allocator.free(self.accumulated_gradients);
    }

    pub fn accumulateGradient(self: *MixedPrecisionTrainer, grads: []const Tensor) !void {
        if (grads.len != self.accumulated_gradients.len) return error.GradientCountMismatch;
        const inv_scale = 1.0 / @max(self.loss_scaler.scale, 1e-8);
        var i: usize = 0;
        while (i < grads.len) : (i += 1) {
            const src = grads[i];
            if (!shapesEqual(src.shape, self.accumulated_gradients[i].shape)) return error.ShapeMismatch;
            var j: usize = 0;
            while (j < src.data.len) : (j += 1) {
                self.accumulated_gradients[i].data[j] += src.data[j] * inv_scale;
            }
        }
        self.accumulation_counter += 1;
    }

    pub fn updateWeights(self: *MixedPrecisionTrainer, lr: f32) !void {
        const actual_steps = @max(@as(usize, 1), self.accumulation_counter);
        const scale = 1.0 / @as(f32, @floatFromInt(actual_steps));

        for (self.accumulated_gradients) |*grad| {
            grad.mulScalar(scale);
        }

        self.loss_scaler.update(self.accumulated_gradients);

        var i: usize = 0;
        while (i < self.master_weights.len) : (i += 1) {
            const master = &self.master_weights[i];
            var j: usize = 0;
            while (j < master.data.len) : (j += 1) {
                const w = &master.data[j];
                const grad_val = self.accumulated_gradients[i].data[j];
                if (std.math.isFinite(grad_val)) {
                    w.* -= lr * grad_val;
                }
            }
        }

        i = 0;
        while (i < self.working_weights.len) : (i += 1) {
            const working = &self.working_weights[i];
            try working.copyFromWithCast(&self.master_weights[i]);
        }

        for (self.accumulated_gradients) |*grad| {
            grad.fill(0.0);
        }
        self.accumulation_counter = 0;
    }
};

pub const B200OptimizationConfig = struct {
    use_fp4_tensor_cores: bool = true,
    use_tensor_memory: bool = true,
    nvlink_bandwidth_tbps: f32 = 1.8,
    hbm_bandwidth_tbps: f32 = 8.0,
    decompression_engine: bool = true,
    multi_instance_gpu: bool = false,
    l2_cache_size_mb: usize = 50,
    tmem_size_mb: usize = 32,
    tmem_access_freq_threshold: usize = 10,
};

pub const B200MemoryManager = struct {
    config: B200OptimizationConfig,
    tensor_memory_pool: []u8,
    tensor_memory_used: usize,
    prefetch_queue: ArrayList(usize),
    allocator: Allocator,

    pub fn init(allocator: Allocator, config: B200OptimizationConfig) !B200MemoryManager {
        const tmem_size = config.tmem_size_mb * 1024 * 1024;
        const tmem_pool = try allocator.alloc(u8, tmem_size);
        errdefer allocator.free(tmem_pool);

        var prefetch_q = ArrayList(usize).init(allocator);
        errdefer prefetch_q.deinit();

        return B200MemoryManager{
            .config = config,
            .tensor_memory_pool = tmem_pool,
            .tensor_memory_used = 0,
            .prefetch_queue = prefetch_q,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *B200MemoryManager) void {
        self.allocator.free(self.tensor_memory_pool);
        self.prefetch_queue.deinit();
    }

    pub fn optimizeMemoryAccess(self: *B200MemoryManager, tensors: []*Tensor, access_pattern: []const usize) !void {
        if (access_pattern.len < tensors.len) return error.InvalidAccessPattern;
        self.prefetch_queue.clearRetainingCapacity();

        var i: usize = 0;
        while (i < tensors.len) : (i += 1) {
            const tensor = tensors[i];
            const access_freq = access_pattern[i];

            if (access_freq > self.config.tmem_access_freq_threshold and self.config.use_tensor_memory) {
                self.moveToTensorMemory(tensor) catch |err| {
                    if (err != error.OutOfTensorMemory) return err;
                };
            }
        }

        var idx: usize = 0;
        while (idx < tensors.len) : (idx += 1) {
            const freq = access_pattern[idx];
            if (freq > 0) {
                try self.prefetch_queue.append(idx);
            }
        }

        if (self.config.decompression_engine) {
            for (tensors) |tensor| {
                try self.compressIfBeneficial(tensor);
            }
        }
    }

    fn moveToTensorMemory(self: *B200MemoryManager, tensor: *Tensor) !void {
        if (tensor.flags.in_tensor_memory) return;
        const tensor_size = tensor.data.len * @sizeOf(f32);
        if (self.tensor_memory_used + tensor_size > self.tensor_memory_pool.len) return error.OutOfTensorMemory;

        const start = self.tensor_memory_used;
        const end = start + tensor_size;
        @memcpy(self.tensor_memory_pool[start..end], std.mem.sliceAsBytes(tensor.data));
        self.tensor_memory_used = end;
        tensor.flags.in_tensor_memory = true;
    }

    fn compressIfBeneficial(self: *B200MemoryManager, tensor: *Tensor) !void {
        if (tensor.flags.requires_grad) return;
        if (tensor.dtype == .fp32 and self.config.use_fp4_tensor_cores) {
            try tensor.convertToFP4();
        }
    }
};

pub const HyperparameterSpace = struct {
    lr_min: f32 = 1e-6,
    lr_max: f32 = 1e-2,
    beta1_min: f32 = 0.85,
    beta1_max: f32 = 0.95,
    beta2_min: f32 = 0.99,
    beta2_max: f32 = 0.9999,
    weight_decay_min: f32 = 0.0,
    weight_decay_max: f32 = 0.1,
};

pub const HyperparamConfig = struct {
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
};

pub const Observation = struct {
    params: HyperparamConfig,
    score: f32,
};

pub const Prediction = struct {
    mean: f32,
    variance: f32,
};

pub const GaussianProcess = struct {
    observations: []Observation,
    kernel_variance: f32,
    length_scale: f32,
    noise_variance: f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, obs: []const Observation) !GaussianProcess {
        const owned = try allocator.dupe(Observation, obs);
        return GaussianProcess{
            .observations = owned,
            .kernel_variance = 1.0,
            .length_scale = 0.1,
            .noise_variance = 0.01,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GaussianProcess) void {
        self.allocator.free(self.observations);
    }

    pub fn expectedImprovement(self: *GaussianProcess, candidate: HyperparamConfig, best_score: f32) !f32 {
        const prediction = try self.predict(candidate);
        const mean = prediction.mean;
        const std_dev = @sqrt(prediction.variance);

        if (std_dev < 1e-8) {
            return 0.0;
        }

        const improvement = best_score - mean;
        const z = improvement / std_dev;

        const phi_z = 0.5 * (1.0 + erfApprox(z / @sqrt(2.0)));
        const pdf_z = @exp(-0.5 * z * z) / @sqrt(2.0 * std.math.pi);

        const ei = improvement * phi_z + std_dev * pdf_z;
        return @max(ei, 0.0);
    }

    fn predict(self: *GaussianProcess, config: HyperparamConfig) !Prediction {
        if (self.observations.len == 0) {
            return Prediction{
                .mean = 0.0,
                .variance = self.kernel_variance,
            };
        }

        const n = self.observations.len;
        var k_matrix = try self.allocator.alloc(f32, n * n);
        defer self.allocator.free(k_matrix);
        var y = try self.allocator.alloc(f32, n);
        defer self.allocator.free(y);
        var k_star = try self.allocator.alloc(f32, n);
        defer self.allocator.free(k_star);

        var i: usize = 0;
        while (i < n) : (i += 1) {
            y[i] = self.observations[i].score;
            k_star[i] = self.kernel(config, self.observations[i].params);

            var j: usize = 0;
            while (j < n) : (j += 1) {
                var value = self.kernel(self.observations[i].params, self.observations[j].params);
                if (i == j) value += self.noise_variance;
                k_matrix[i * n + j] = value;
            }
        }

        const alpha = try self.solveLinearSystem(k_matrix, y);
        defer self.allocator.free(alpha);
        const v = try self.solveLinearSystem(k_matrix, k_star);
        defer self.allocator.free(v);

        var mean: f32 = 0.0;
        var variance_reduction: f32 = 0.0;
        i = 0;
        while (i < n) : (i += 1) {
            mean += k_star[i] * alpha[i];
            variance_reduction += k_star[i] * v[i];
        }

        const prior_var = self.kernel(config, config) + self.noise_variance;
        const variance = @max(prior_var - variance_reduction, 1e-8);
        return Prediction{
            .mean = mean,
            .variance = variance,
        };
    }

    fn solveLinearSystem(self: *GaussianProcess, matrix: []const f32, rhs: []const f32) ![]f32 {
        const n = rhs.len;
        var a = try self.allocator.dupe(f32, matrix);
        errdefer self.allocator.free(a);
        var b = try self.allocator.dupe(f32, rhs);
        errdefer self.allocator.free(b);

        var col: usize = 0;
        while (col < n) : (col += 1) {
            var pivot_row = col;
            var pivot_value = blk: {
                const v = a[col * n + col];
                break :blk if (v >= 0) v else -v;
            };
            var row: usize = col + 1;
            while (row < n) : (row += 1) {
                const candidate = blk: {
                    const v = a[row * n + col];
                    break :blk if (v >= 0) v else -v;
                };
                if (candidate > pivot_value) {
                    pivot_value = candidate;
                    pivot_row = row;
                }
            }

            if (pivot_row != col) {
                var j: usize = 0;
                while (j < n) : (j += 1) {
                    const tmp = a[col * n + j];
                    a[col * n + j] = a[pivot_row * n + j];
                    a[pivot_row * n + j] = tmp;
                }
                const tmp_b = b[col];
                b[col] = b[pivot_row];
                b[pivot_row] = tmp_b;
            }

            if (pivot_value < 1e-8) {
                a[col * n + col] += 1e-8;
            }

            const pivot = a[col * n + col];
            var row2: usize = col + 1;
            while (row2 < n) : (row2 += 1) {
                const factor = a[row2 * n + col] / pivot;
                a[row2 * n + col] = 0.0;
                var j: usize = col + 1;
                while (j < n) : (j += 1) {
                    a[row2 * n + j] -= factor * a[col * n + j];
                }
                b[row2] -= factor * b[col];
            }
        }

        var solution = try self.allocator.alloc(f32, n);
        errdefer self.allocator.free(solution);
        var idx: usize = n;
        while (idx > 0) {
            idx -= 1;
            var sum = b[idx];
            var j: usize = idx + 1;
            while (j < n) : (j += 1) {
                sum -= a[idx * n + j] * solution[j];
            }
            const diag = a[idx * n + idx];
            const abs_diag = if (diag < 0) -diag else diag;
            if (abs_diag < 1e-12) {
                solution[idx] = 0.0;
            } else {
                solution[idx] = sum / diag;
            }
        }

        self.allocator.free(a);
        self.allocator.free(b);
        return solution;
    }

    fn kernel(self: *GaussianProcess, x1: HyperparamConfig, x2: HyperparamConfig) f32 {
        const diff_lr = x1.lr - x2.lr;
        const diff_beta1 = x1.beta1 - x2.beta1;
        const diff_beta2 = x1.beta2 - x2.beta2;
        const diff_wd = x1.weight_decay - x2.weight_decay;

        const dist_sq = diff_lr * diff_lr + diff_beta1 * diff_beta1 + diff_beta2 * diff_beta2 + diff_wd * diff_wd;

        return self.kernel_variance * @exp(-dist_sq / (2.0 * self.length_scale * self.length_scale));
    }
};

pub const BayesianOptimizer = struct {
    space: HyperparameterSpace,
    observations: ArrayList(Observation),
    best_params: HyperparamConfig,
    best_score: f32,
    num_candidates: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, space: HyperparameterSpace) !BayesianOptimizer {
        return BayesianOptimizer{
            .space = space,
            .observations = ArrayList(Observation).init(allocator),
            .best_params = HyperparamConfig{
                .lr = 0.001,
                .beta1 = 0.9,
                .beta2 = 0.999,
                .weight_decay = 0.01,
            },
            .best_score = std.math.inf(f32),
            .num_candidates = 100,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BayesianOptimizer) void {
        self.observations.deinit();
    }

    pub fn suggestNext(self: *BayesianOptimizer) !HyperparamConfig {
        if (self.observations.items.len < 5) {
            return self.sampleRandom();
        }

        var gp = try GaussianProcess.init(self.allocator, self.observations.items);
        defer gp.deinit();

        var best_ei: f32 = -std.math.inf(f32);
        var best_config: HyperparamConfig = try self.sampleRandom();

        var i: usize = 0;
        while (i < self.num_candidates) : (i += 1) {
            const candidate = try self.sampleRandom();
            const ei = try gp.expectedImprovement(candidate, self.best_score);

            if (ei > best_ei) {
                best_ei = ei;
                best_config = candidate;
            }
        }

        return best_config;
    }

    pub fn observe(self: *BayesianOptimizer, params: HyperparamConfig, score: f32) !void {
        try self.observations.append(Observation{ .params = params, .score = score });

        if (score < self.best_score) {
            self.best_score = score;
            self.best_params = params;
        }
    }

    fn sampleRandom(self: *BayesianOptimizer) !HyperparamConfig {
        var prng = std.Random.DefaultPrng.init(nextSeed());
        const random = prng.random();

        return HyperparamConfig{
            .lr = self.space.lr_min + random.float(f32) * (self.space.lr_max - self.space.lr_min),
            .beta1 = self.space.beta1_min + random.float(f32) * (self.space.beta1_max - self.space.beta1_min),
            .beta2 = self.space.beta2_min + random.float(f32) * (self.space.beta2_max - self.space.beta2_min),
            .weight_decay = self.space.weight_decay_min + random.float(f32) * (self.space.weight_decay_max - self.space.weight_decay_min),
        };
    }
};

pub const GPUMetrics = struct {
    utilization_percent: f32,
    memory_used_gb: f32,
    tensor_core_util: f32,
    nvlink_bandwidth_util: f32,
};

pub const B200Profiler = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) !B200Profiler {
        return B200Profiler{ .allocator = allocator };
    }

    pub fn captureGPUMetrics(self: *B200Profiler) !GPUMetrics {
        _ = self;
        return GPUMetrics{
            .utilization_percent = 0.0,
            .memory_used_gb = 0.0,
            .tensor_core_util = 0.0,
            .nvlink_bandwidth_util = 0.0,
        };
    }
};

pub const MetricsStore = struct {
    training_losses: ArrayList(f32),
    validation_losses: ArrayList(f32),
    learning_rates: ArrayList(f32),
    gradient_norms: ArrayList(f32),
    parameter_norms: ArrayList(f32),
    step_times_ms: ArrayList(f32),
    gpu_utilization: ArrayList(f32),
    memory_usage_gb: ArrayList(f32),
    tensor_core_utilization: ArrayList(f32),
    nvlink_bandwidth_utilization: ArrayList(f32),

    pub fn init(allocator: Allocator) MetricsStore {
        return MetricsStore{
            .training_losses = ArrayList(f32).init(allocator),
            .validation_losses = ArrayList(f32).init(allocator),
            .learning_rates = ArrayList(f32).init(allocator),
            .gradient_norms = ArrayList(f32).init(allocator),
            .parameter_norms = ArrayList(f32).init(allocator),
            .step_times_ms = ArrayList(f32).init(allocator),
            .gpu_utilization = ArrayList(f32).init(allocator),
            .memory_usage_gb = ArrayList(f32).init(allocator),
            .tensor_core_utilization = ArrayList(f32).init(allocator),
            .nvlink_bandwidth_utilization = ArrayList(f32).init(allocator),
        };
    }

    pub fn deinit(self: *MetricsStore) void {
        self.training_losses.deinit();
        self.validation_losses.deinit();
        self.learning_rates.deinit();
        self.gradient_norms.deinit();
        self.parameter_norms.deinit();
        self.step_times_ms.deinit();
        self.gpu_utilization.deinit();
        self.memory_usage_gb.deinit();
        self.tensor_core_utilization.deinit();
        self.nvlink_bandwidth_utilization.deinit();
    }
};

pub const Report = struct {
    average_loss: f32,
    average_step_time_ms: f32,
    throughput_steps_per_sec: f32,
    average_gpu_utilization: f32,
    average_memory_usage_gb: f32,
    average_tensor_core_utilization: f32,
    average_nvlink_utilization: f32,
    total_steps: usize,
};

pub const PerformanceMonitor = struct {
    metrics: MetricsStore,
    profiler: B200Profiler,
    telemetry_enabled: bool,
    allocator: Allocator,

    pub fn init(allocator: Allocator, enable_telemetry: bool) !PerformanceMonitor {
        return PerformanceMonitor{
            .metrics = MetricsStore.init(allocator),
            .profiler = try B200Profiler.init(allocator),
            .telemetry_enabled = enable_telemetry,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PerformanceMonitor) void {
        self.metrics.deinit();
    }

    pub fn recordStep(self: *PerformanceMonitor, loss: f32, lr: f32, grad_norm: f32, param_norm: f32, step_time_ms: f32) !void {
        if (!self.telemetry_enabled) return;
        try self.metrics.training_losses.append(loss);
        try self.metrics.learning_rates.append(lr);
        try self.metrics.gradient_norms.append(grad_norm);
        try self.metrics.parameter_norms.append(param_norm);
        try self.metrics.step_times_ms.append(step_time_ms);

        const gpu_metrics = try self.profiler.captureGPUMetrics();
        try self.metrics.gpu_utilization.append(gpu_metrics.utilization_percent);
        try self.metrics.memory_usage_gb.append(gpu_metrics.memory_used_gb);
        try self.metrics.tensor_core_utilization.append(gpu_metrics.tensor_core_util);
        try self.metrics.nvlink_bandwidth_utilization.append(gpu_metrics.nvlink_bandwidth_util);
    }

    pub fn generateReport(self: *PerformanceMonitor) !Report {
        const avg_loss = self.computeMean(self.metrics.training_losses.items);
        const avg_step_time = self.computeMean(self.metrics.step_times_ms.items);
        const avg_gpu_util = self.computeMean(self.metrics.gpu_utilization.items);
        const avg_memory = self.computeMean(self.metrics.memory_usage_gb.items);

        const total_steps = self.metrics.training_losses.items.len;
        const total_time_sec = self.computeSum(self.metrics.step_times_ms.items) / 1000.0;
        const throughput_steps_per_sec = if (total_time_sec > 0.0) @as(f32, @floatFromInt(total_steps)) / total_time_sec else 0.0;

        const avg_tensor_core_util = self.computeMean(self.metrics.tensor_core_utilization.items);
        const avg_nvlink_util = self.computeMean(self.metrics.nvlink_bandwidth_utilization.items);

        return Report{
            .average_loss = avg_loss,
            .average_step_time_ms = avg_step_time,
            .throughput_steps_per_sec = throughput_steps_per_sec,
            .average_gpu_utilization = avg_gpu_util,
            .average_memory_usage_gb = avg_memory,
            .average_tensor_core_utilization = avg_tensor_core_util,
            .average_nvlink_utilization = avg_nvlink_util,
            .total_steps = total_steps,
        };
    }

    fn computeMean(self: *PerformanceMonitor, values: []const f32) f32 {
        _ = self;
        if (values.len == 0) return 0.0;
        var sum: f64 = 0.0;
        for (values) |v| {
            sum += @as(f64, v);
        }
        return @floatCast(sum / @as(f64, @floatFromInt(values.len)));
    }

    fn computeSum(self: *PerformanceMonitor, values: []const f32) f32 {
        _ = self;
        var sum: f64 = 0.0;
        for (values) |v| {
            sum += @as(f64, v);
        }
        return @floatCast(sum);
    }
};

pub const SFD = struct {
    fisher_diag: Tensor,
    momentum_buffer: Tensor,
    velocity_buffer: Tensor,
    beta1: f32,
    beta2: f32,
    eps: f32,
    clip_threshold: f32,
    fisher_max: f32,
    warmup_steps: usize,
    step_count: usize,
    allocator: Allocator,
    param_size: usize,
    initialized: bool,
    use_external_fisher: bool,

    pub fn init(allocator: Allocator, param_size: usize) !SFD {
        return initWithConfig(allocator, param_size, .{});
    }

    pub fn initWithConfig(allocator: Allocator, param_size: usize, config: SFDConfig) !SFD {
        if (param_size == 0) return error.InvalidParamSize;
        if (config.beta1 <= 0.0 or config.beta1 >= 1.0) return error.InvalidBeta1;
        if (config.beta2 <= 0.0 or config.beta2 >= 1.0) return error.InvalidBeta2;
        if (config.eps <= 0.0) return error.InvalidEpsilon;
        if (config.clip_threshold <= 0.0) return error.InvalidClipThreshold;
        if (!std.math.isFinite(config.fisher_max) or config.fisher_max <= 0.0) return error.InvalidFisherMax;

        const shape = [_]usize{param_size};

        var diag = try Tensor.init(allocator, &shape);
        errdefer diag.deinit();
        diag.fill(1.0);

        var momentum = try Tensor.init(allocator, &shape);
        errdefer momentum.deinit();
        momentum.fill(0.0);

        var velocity = try Tensor.init(allocator, &shape);
        errdefer velocity.deinit();
        velocity.fill(0.0);

        return SFD{
            .fisher_diag = diag,
            .momentum_buffer = momentum,
            .velocity_buffer = velocity,
            .beta1 = config.beta1,
            .beta2 = config.beta2,
            .eps = config.eps,
            .clip_threshold = config.clip_threshold,
            .fisher_max = config.fisher_max,
            .warmup_steps = config.warmup_steps,
            .step_count = 0,
            .allocator = allocator,
            .param_size = param_size,
            .initialized = true,
            .use_external_fisher = config.use_external_fisher,
        };
    }

    pub fn initWithArena(arena: *core_memory.ArenaAllocator, param_size: usize) !SFD {
        return initWithConfig(arena.allocator(), param_size, .{});
    }

    pub fn initWithPool(pool: *core_memory.PoolAllocator, param_size: usize) !SFD {
        return initWithConfig(pool.allocator(), param_size, .{});
    }

    pub fn initWithBuddy(buddy: *core_memory.BuddyAllocator, param_size: usize) !SFD {
        return initWithConfig(buddy.allocator(), param_size, .{});
    }

    pub fn deinit(self: *SFD) void {
        if (!self.initialized) return;
        self.fisher_diag.deinit();
        self.momentum_buffer.deinit();
        self.velocity_buffer.deinit();
        self.initialized = false;
    }

    pub fn update(self: *SFD, gradients: *const Tensor, params: *Tensor, lr: f32) !void {
        if (!self.initialized) return error.NotInitialized;
        if (!std.math.isFinite(lr) or lr < 0.0) return error.InvalidLearningRate;

        const grad_count = gradients.data.len;
        const param_count = params.data.len;

        if (grad_count != param_count) return error.ShapeMismatch;
        if (grad_count != self.param_size) return error.ShapeMismatch;

        self.step_count +|= 1;
        const step_f = @as(f32, @floatFromInt(self.step_count));

        const grad_data = gradients.data;
        const param_data = params.data;
        const momentum_data = self.momentum_buffer.data;
        const velocity_data = self.velocity_buffer.data;
        const fisher_data = self.fisher_diag.data;

        const warmup_steps_f = @as(f32, @floatFromInt(self.warmup_steps));
        const warmup_factor: f32 = if (self.step_count <= self.warmup_steps and self.warmup_steps > 0) step_f / warmup_steps_f else 1.0;

        const beta1_power = std.math.pow(f32, self.beta1, step_f);
        const beta2_power = std.math.pow(f32, self.beta2, step_f);
        const m_correction = 1.0 - beta1_power;
        const v_correction = 1.0 - beta2_power;

        var i: usize = 0;
        while (i < self.param_size) : (i += 1) {
            const g = grad_data[i];

            if (!std.math.isFinite(g)) continue;

            momentum_data[i] = self.beta1 * momentum_data[i] + (1.0 - self.beta1) * g;
            velocity_data[i] = self.beta2 * velocity_data[i] + (1.0 - self.beta2) * g * g;

            var m_hat = momentum_data[i];
            var v_hat = velocity_data[i];

            if (m_correction > 1e-10) {
                m_hat = momentum_data[i] / m_correction;
            }
            if (v_correction > 1e-10) {
                v_hat = velocity_data[i] / v_correction;
            }

            const sqrt_v = std.math.sqrt(@max(0.0, v_hat));
            const adaptive_lr = lr * warmup_factor / (sqrt_v + self.eps);

            if (!self.use_external_fisher) {
                fisher_data[i] = self.beta2 * fisher_data[i] + (1.0 - self.beta2) * g * g;
                fisher_data[i] = @min(fisher_data[i], self.fisher_max);
                if (!std.math.isFinite(fisher_data[i])) {
                    fisher_data[i] = 1.0;
                }
            }

            const sqrt_fisher = std.math.sqrt(@max(0.0, fisher_data[i]));
            var update_val = m_hat * adaptive_lr / (sqrt_fisher + self.eps);

            update_val = std.math.clamp(update_val, -self.clip_threshold, self.clip_threshold);

            if (std.math.isFinite(update_val)) {
                param_data[i] -= update_val;
            }
        }
    }

    pub fn correctEigenvalues(self: *SFD, step_size: f32) !void {
        const fisher_data = self.fisher_diag.data;
        const velocity_data = self.velocity_buffer.data;
        const blend = std.math.clamp(step_size, 0.0, 1.0);

        var i: usize = 0;
        while (i < self.param_size) : (i += 1) {
            const adam_second_moment = @sqrt(velocity_data[i] + self.eps);
            const shampoo_eigenval = @sqrt(fisher_data[i] + self.eps);
            const correction_factor = adam_second_moment / (shampoo_eigenval + self.eps);
            const corrected = std.math.clamp(fisher_data[i] * correction_factor * correction_factor, 1e-8, self.fisher_max);
            fisher_data[i] = fisher_data[i] * (1.0 - blend) + corrected * blend;
        }
    }

    pub fn adaptiveLR(self: *const SFD, grad_norm: f32, param_norm: f32) f32 {
        if (!std.math.isFinite(grad_norm) or grad_norm < 0.0) return 1.0;
        if (!std.math.isFinite(param_norm) or param_norm < 0.0) return 1.0;

        const denom = param_norm + self.eps;
        const ratio = grad_norm / denom;
        const inner = ratio + self.eps;

        if (inner <= 0.0) return 1.0;

        const result = 1.0 / std.math.sqrt(inner);
        return if (std.math.isFinite(result)) result else 1.0;
    }

    pub fn spectralClip(self: *SFD, tensor: *Tensor, max_eig: f32) !void {
        return self.spectralClipWithIters(tensor, max_eig, 100);
    }

    pub fn spectralClipWithIters(self: *SFD, tensor: *Tensor, max_eig: f32, power_iter: usize) !void {
        if (!self.initialized) return error.NotInitialized;
        if (!std.math.isFinite(max_eig) or max_eig <= 0.0) return error.InvalidMaxEig;

        const current_max_ev = try tensor.spectralNorm(self.allocator, power_iter, 1e-6);
        if (!std.math.isFinite(current_max_ev) or current_max_ev <= 0.0) return;

        if (current_max_ev > max_eig) {
            const scale = max_eig / current_max_ev;
            if (std.math.isFinite(scale) and scale > 0.0) {
                tensor.mulScalar(scale);
            }
        }
    }

    pub fn accumulateFisher(self: *SFD, grads: []const Tensor) !void {
        if (!self.initialized) return error.NotInitialized;
        if (grads.len == 0) return;

        const fisher_data = self.fisher_diag.data;

        for (grads) |grad| {
            const g_data = grad.data;
            const count = @min(fisher_data.len, g_data.len);

            var j: usize = 0;
            while (j < count) : (j += 1) {
                const g = g_data[j];
                if (std.math.isFinite(g)) {
                    fisher_data[j] += g * g;
                    fisher_data[j] = @min(fisher_data[j], self.fisher_max);
                }
            }
        }
    }

    pub fn resetFisher(self: *SFD) void {
        if (!self.initialized) return;

        for (self.fisher_diag.data) |*val| {
            val.* = 1.0;
        }
    }

    pub fn clipGradNorm(self: *SFD, grads: []*Tensor, max_norm: f32) f32 {
        if (!std.math.isFinite(max_norm) or max_norm <= 0.0) return 0.0;

        var total_norm_sq: f64 = 0.0;

        for (grads) |grad| {
            const norm = grad.normL2();
            if (std.math.isFinite(norm)) {
                total_norm_sq += @as(f64, norm) * @as(f64, norm);
            }
        }

        const total_norm: f32 = @floatCast(std.math.sqrt(total_norm_sq));

        if (total_norm > max_norm) {
            const scale = max_norm / (total_norm + self.eps);
            if (std.math.isFinite(scale) and scale > 0.0) {
                for (grads) |grad| {
                    grad.mulScalar(scale);
                }
            }
        }

        return total_norm;
    }

    pub fn ampSchedule(_: *const SFD, step: usize, warmup: usize, total: usize) f32 {
        if (warmup == 0) return 1.0;
        if (total <= warmup) return 1.0;

        if (step < warmup) {
            return @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(warmup));
        }

        const progress_num = step - warmup;
        const progress_denom = total - warmup;

        if (progress_denom == 0) return 0.5;

        var progress = @as(f32, @floatFromInt(progress_num)) / @as(f32, @floatFromInt(progress_denom));
        progress = @min(progress, 1.0);

        return 0.5 * (1.0 + std.math.cos(std.math.pi * progress));
    }

    pub fn saveState(self: *const SFD, path: []const u8) !void {
        if (!self.initialized) return error.NotInitialized;

        var file = try std.fs.cwd().createFile(path, .{ .mode = 0o600 });
        defer file.close();

        var buffered = std.io.bufferedWriter(file.writer());
        var writer = buffered.writer();

        try writer.writeInt(u32, 0x53464431, .little);
        try writer.writeInt(u32, @as(u32, @bitCast(self.beta1)), .little);
        try writer.writeInt(u32, @as(u32, @bitCast(self.beta2)), .little);
        try writer.writeInt(u32, @as(u32, @bitCast(self.eps)), .little);
        try writer.writeInt(u32, @as(u32, @bitCast(self.clip_threshold)), .little);
        try writer.writeInt(u32, @as(u32, @bitCast(self.fisher_max)), .little);
        try writer.writeInt(u64, @intCast(self.warmup_steps), .little);
        try writer.writeInt(u64, @intCast(self.param_size), .little);
        try writer.writeInt(u64, @intCast(self.step_count), .little);
        try self.fisher_diag.save(writer);
        try self.momentum_buffer.save(writer);
        try self.velocity_buffer.save(writer);

        try buffered.flush();
    }

    pub fn loadState(self: *SFD, path: []const u8) !void {
        if (!self.initialized) return error.NotInitialized;

        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        defer file.close();

        var buffered = std.io.bufferedReader(file.reader());
        var reader = buffered.reader();

        const magic = try reader.readInt(u32, .little);
        if (magic != 0x53464431) return error.InvalidStateFormat;

        const beta1 = @as(f32, @bitCast(try reader.readInt(u32, .little)));
        const beta2 = @as(f32, @bitCast(try reader.readInt(u32, .little)));
        const eps = @as(f32, @bitCast(try reader.readInt(u32, .little)));
        const clip_threshold = @as(f32, @bitCast(try reader.readInt(u32, .little)));
        const fisher_max = @as(f32, @bitCast(try reader.readInt(u32, .little)));
        const warmup_steps_u64 = try reader.readInt(u64, .little);
        const param_size_u64 = try reader.readInt(u64, .little);
        const step_count_u64 = try reader.readInt(u64, .little);

        if (beta1 <= 0.0 or beta1 >= 1.0) return error.InvalidStateFormat;
        if (beta2 <= 0.0 or beta2 >= 1.0) return error.InvalidStateFormat;
        if (eps <= 0.0 or !std.math.isFinite(eps)) return error.InvalidStateFormat;
        if (clip_threshold <= 0.0 or !std.math.isFinite(clip_threshold)) return error.InvalidStateFormat;
        if (fisher_max <= 0.0 or !std.math.isFinite(fisher_max)) return error.InvalidStateFormat;

        if (warmup_steps_u64 > @as(u64, std.math.maxInt(usize)) or param_size_u64 > @as(u64, std.math.maxInt(usize)) or step_count_u64 > @as(u64, std.math.maxInt(usize))) return error.InvalidStateFormat;
        if (@as(usize, @intCast(param_size_u64)) != self.param_size) return error.ShapeMismatch;

        var loaded_fisher = try Tensor.load(self.allocator, reader);
        errdefer loaded_fisher.deinit();
        var loaded_momentum = try Tensor.load(self.allocator, reader);
        errdefer loaded_momentum.deinit();
        var loaded_velocity = try Tensor.load(self.allocator, reader);
        errdefer loaded_velocity.deinit();

        if (loaded_fisher.data.len != self.param_size or loaded_momentum.data.len != self.param_size or loaded_velocity.data.len != self.param_size) return error.ShapeMismatch;

        self.fisher_diag.deinit();
        self.momentum_buffer.deinit();
        self.velocity_buffer.deinit();

        self.fisher_diag = loaded_fisher;
        self.momentum_buffer = loaded_momentum;
        self.velocity_buffer = loaded_velocity;
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.eps = eps;
        self.clip_threshold = clip_threshold;
        self.fisher_max = fisher_max;
        self.warmup_steps = @intCast(warmup_steps_u64);
        self.step_count = @intCast(step_count_u64);
    }

    pub fn warmStart(self: *SFD, prev_diag: *const Tensor) void {
        if (!self.initialized) return;

        const fisher_data = self.fisher_diag.data;
        const prev_data = prev_diag.data;
        const count = @min(fisher_data.len, prev_data.len);

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const prev = prev_data[i];
            if (!std.math.isFinite(prev) or prev < 0.0) continue;
            const combined = (fisher_data[i] + prev) * 0.5;
            fisher_data[i] = @min(combined, self.fisher_max);
        }
    }

    pub fn varianceReduction(self: *SFD, noise_grads: []const Tensor) !void {
        if (!self.initialized) return error.NotInitialized;
        if (noise_grads.len == 0) return error.EmptyGrads;

        const shape = self.fisher_diag.shape.dims;
        var mean_grad = try Tensor.zeros(self.allocator, shape);
        defer mean_grad.deinit();
        var second_moment = try Tensor.zeros(self.allocator, shape);
        defer second_moment.deinit();

        for (noise_grads) |ng| {
            const count = @min(mean_grad.data.len, ng.data.len);

            var j: usize = 0;
            while (j < count) : (j += 1) {
                const g = ng.data[j];
                if (std.math.isFinite(g)) {
                    mean_grad.data[j] += g;
                    second_moment.data[j] += g * g;
                }
            }
        }

        const divisor = @as(f32, @floatFromInt(noise_grads.len));
        var i: usize = 0;
        while (i < mean_grad.data.len) : (i += 1) {
            mean_grad.data[i] /= divisor;
            second_moment.data[i] /= divisor;
            const variance = @max(0.0, second_moment.data[i] - mean_grad.data[i] * mean_grad.data[i]);
            self.fisher_diag.data[i] = @max(1e-8, self.fisher_diag.data[i] - variance);
        }
    }
};

pub const SophiaSOAPConfig = struct {
    rho: f32 = 0.04,
    gamma: f32 = 0.01,
    hessian_update_freq: usize = 10,
    use_gauss_newton: bool = true,
    kfac_damping: f32 = 0.001,
    hessian_ema_alpha: f32 = 0.9,
};

pub const SophiaSOAPOptimizer = struct {
    sfd: SFD,
    kfac_blocks: []KFACBlock,
    hessian_diag: Tensor,
    hutchinson_vector: Tensor,
    config: SophiaSOAPConfig,
    gradient_flow: GradientFlowController,
    variance_reducer: MARSVarianceReducer,
    reversible_state: ReversibleOptimizerState,
    allocator: Allocator,

    pub fn init(allocator: Allocator, param_size: usize, layer_dims: []const [2]usize, sophia_config: SophiaSOAPConfig) !SophiaSOAPOptimizer {
        var sfd = try SFD.init(allocator, param_size);
        errdefer sfd.deinit();

        var kfac_blocks = try allocator.alloc(KFACBlock, layer_dims.len);
        errdefer allocator.free(kfac_blocks);

        var initialized_blocks: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < initialized_blocks) : (idx += 1) {
                kfac_blocks[idx].deinit();
            }
        }

        var i: usize = 0;
        while (i < layer_dims.len) : (i += 1) {
            const dims = layer_dims[i];
            kfac_blocks[i] = try KFACBlock.init(allocator, dims[0], dims[1], sophia_config.kfac_damping);
            initialized_blocks += 1;
        }

        const shape = [_]usize{param_size};
        var hess_diag = try Tensor.init(allocator, &shape);
        errdefer hess_diag.deinit();
        hess_diag.fill(1.0);

        var hutch_vec = try Tensor.init(allocator, &shape);
        errdefer hutch_vec.deinit();
        hutch_vec.fillRademacher();

        var param_shapes = try allocator.alloc([]const usize, 1);
        defer allocator.free(param_shapes);
        param_shapes[0] = &shape;

        var vr = try MARSVarianceReducer.init(allocator, param_shapes, .{});
        errdefer vr.deinit();

        return SophiaSOAPOptimizer{
            .sfd = sfd,
            .kfac_blocks = kfac_blocks,
            .hessian_diag = hess_diag,
            .hutchinson_vector = hutch_vec,
            .config = sophia_config,
            .gradient_flow = GradientFlowController.init(),
            .variance_reducer = vr,
            .reversible_state = ReversibleOptimizerState.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SophiaSOAPOptimizer) void {
        self.sfd.deinit();
        for (self.kfac_blocks) |*block| {
            block.deinit();
        }
        self.allocator.free(self.kfac_blocks);
        self.hessian_diag.deinit();
        self.hutchinson_vector.deinit();
        self.variance_reducer.deinit();
        self.reversible_state.deinit();
    }

    pub fn update(self: *SophiaSOAPOptimizer, gradients: *const Tensor, params: *Tensor, activations: []const Tensor, lr: f32) !void {
        var original_grad = try gradients.clone(self.allocator);
        defer original_grad.deinit();

        var hybrid_grad = try gradients.clone(self.allocator);
        defer hybrid_grad.deinit();

        var i: usize = 0;
        while (i < self.kfac_blocks.len) : (i += 1) {
            const block = &self.kfac_blocks[i];
            if (i < activations.len) {
                try block.updateStatistics(&activations[i], &original_grad);

                if (self.sfd.step_count % block.update_freq == 0) {
                    try block.preconditionGradient(&hybrid_grad);
                }
            }
        }

        if (self.config.hessian_update_freq > 0 and self.sfd.step_count % self.config.hessian_update_freq == 0) {
            try self.updateHessianDiagonal(params, &original_grad);
        }

        {
            var pi: usize = 0;
            while (pi < params.data.len) : (pi += 1) {
                const g = hybrid_grad.data[pi];
                const h = self.hessian_diag.data[pi];

                if (self.config.use_gauss_newton) {
                    const gn_h = @max(h, self.config.gamma);
                    params.data[pi] -= lr * g / gn_h;
                } else {
                    const abs_h = if (h < 0) -h else h;
                    const h_clipped = @max(abs_h, self.config.gamma);
                    params.data[pi] -= lr * g / h_clipped;
                }
            }
        }

        try self.sfd.update(&hybrid_grad, params, lr * 0.5);
        try self.sfd.correctEigenvalues(lr);
    }

    fn updateHessianDiagonal(self: *SophiaSOAPOptimizer, params: *const Tensor, grad: *const Tensor) !void {
        _ = params;
        self.hutchinson_vector.fillRademacher();

        const alpha: f32 = self.config.hessian_ema_alpha;

        var i: usize = 0;
        while (i < self.hessian_diag.data.len and i < grad.data.len) : (i += 1) {
            const h = &self.hessian_diag.data[i];
            const g = grad.data[i];
            const direction = self.hutchinson_vector.data[i];
            const curvature = g * direction;
            h.* = alpha * h.* + (1.0 - alpha) * curvature;

            if (self.config.use_gauss_newton) {
                h.* = @max(h.*, 1e-6);
            }
        }
    }
};

test "SFD init and deinit" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 4);
    defer sfd.deinit();

    try std.testing.expect(sfd.initialized);
    try std.testing.expectEqual(@as(usize, 4), sfd.param_size);
}

test "SFD update" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 4);
    defer sfd.deinit();

    const shape = [_]usize{4};
    var grads = try Tensor.init(gpa, &shape);
    defer grads.deinit();

    grads.data[0] = 1.0;
    grads.data[1] = 2.0;
    grads.data[2] = 3.0;
    grads.data[3] = 4.0;

    var params = try Tensor.init(gpa, &shape);
    defer params.deinit();

    params.fill(0.0);

    try sfd.update(&grads, &params, 0.1);
    for (params.data) |v| {
        try std.testing.expect(v < 0);
    }
}

test "Tensor clone" {
    const gpa = std.testing.allocator;
    const shape = [_]usize{ 2, 3 };
    var t1 = try Tensor.init(gpa, &shape);
    defer t1.deinit();

    t1.fill(5.0);

    var t2 = try t1.clone(gpa);
    defer t2.deinit();

    try std.testing.expectEqual(t1.data[0], t2.data[0]);
    try std.testing.expectEqual(false, t2.flags.in_tensor_memory);
}

test "KFACBlock init" {
    const gpa = std.testing.allocator;
    var block = try KFACBlock.init(gpa, 4, 4, 0.001);
    defer block.deinit();

    try std.testing.expectEqual(@as(f32, 0.001), block.damping);
    try std.testing.expectEqual(@as(f32, 0.95), block.alpha);
}

test "SpectralNormalizer normalizeWeights" {
    const gpa = std.testing.allocator;
    var normalizer = SpectralNormalizer.init(10);
    const shape = [_]usize{ 4, 4 };
    var w = try Tensor.init(gpa, &shape);
    defer w.deinit();
    w.fillRandomNormal(0.0, 2.0);
    try normalizer.normalizeWeights(&w, gpa);
    const sigma = try w.spectralNorm(gpa, 20, 1e-6);
    try std.testing.expect(sigma <= normalizer.max_singular_value + 1e-3);
}

test "LRScheduler cosine annealing full schedule" {
    const gpa = std.testing.allocator;
    var scheduler = LRScheduler.init(gpa, .cosine_annealing, 0.1, 10, 100);
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const lr = try scheduler.getLearningRate(null);
        scheduler.step();
        try std.testing.expect(lr >= 0.0 and lr <= scheduler.base_lr + 1e-6);
    }
}

test "BayesianOptimizer init" {
    const gpa = std.testing.allocator;
    var opt = try BayesianOptimizer.init(gpa, .{});
    defer opt.deinit();

    const config = try opt.suggestNext();
    try std.testing.expect(config.lr >= 1e-6 and config.lr <= 1e-2);
}

test "BayesianOptimizer GP suggestion" {
    const gpa = std.testing.allocator;
    var opt = try BayesianOptimizer.init(gpa, .{});
    defer opt.deinit();
    var i: usize = 0;
    while (i < 6) : (i += 1) {
        try opt.observe(.{ .lr = 0.001, .beta1 = 0.9, .beta2 = 0.999, .weight_decay = 0.01 }, @as(f32, @floatFromInt(i)) * 0.1);
    }
    const config = try opt.suggestNext();
    try std.testing.expect(config.lr >= 1e-6 and config.lr <= 1e-2);
}

test "SophiaSOAPOptimizer init" {
    const gpa = std.testing.allocator;
    const layer_dims = [_][2]usize{[_]usize{ 4, 4 }};
    var opt = try SophiaSOAPOptimizer.init(gpa, 16, &layer_dims, .{});
    defer opt.deinit();

    try std.testing.expect(opt.sfd.initialized);
}

test "DynamicLossScaler custom max scale" {
    const scaler = DynamicLossScaler.initWithMaxScale(1024.0, 32768.0);
    try std.testing.expectEqual(@as(f32, 32768.0), scaler.max_scale);
}

test "MARSVarianceReducer config" {
    const gpa = std.testing.allocator;
    const shape = [_]usize{4};
    const shapes = [_][]const usize{&shape};
    var mars = try MARSVarianceReducer.init(gpa, &shapes, .{ .momentum = 0.5, .scale_factor = 2.0, .snapshot_freq = 50 });
    defer mars.deinit();
    try std.testing.expectEqual(@as(f32, 0.5), mars.momentum);
    try std.testing.expectEqual(@as(f32, 2.0), mars.scale_factor);
    try std.testing.expectEqual(@as(usize, 50), mars.snapshot_freq);
}

