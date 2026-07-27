const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../core/tensor.zig").Tensor;
const memory = @import("../core/memory.zig");
const accel = @import("../hw/accel/accel_interface.zig");
const OFTB = @import("oftb.zig").OFTB;
const types = @import("../core/types.zig");
const Thread = std.Thread;

const LAYER_TARGET_SPECTRAL_NORM: f32 = 0.9;
const LAYER_SPECTRAL_POWER_ITERATIONS: usize = 30;

fn spectralNormPowerIteration(allocator: Allocator, data: []const f32, rows: usize, cols: usize, iterations: usize, seed: u64) !f32 {
    const u = try allocator.alloc(f32, rows);
    defer allocator.free(u);
    const v = try allocator.alloc(f32, cols);
    defer allocator.free(v);

    var prng = types.PRNG.init(seed);
    for (v) |*x| x.* = prng.float() * 2.0 - 1.0;

    var iter: usize = 0;
    while (iter < iterations) : (iter += 1) {
        for (u) |*x| x.* = 0.0;
        var i: usize = 0;
        while (i < rows) : (i += 1) {
            const row = data[i * cols .. i * cols + cols];
            var sum: f32 = 0.0;
            var j: usize = 0;
            while (j < cols) : (j += 1) sum += row[j] * v[j];
            u[i] = sum;
        }

        var u_norm_sq: f32 = 0.0;
        for (u) |x| u_norm_sq += x * x;
        const u_norm = @sqrt(u_norm_sq);
        if (u_norm > 1e-12) {
            for (u) |*x| x.* /= u_norm;
        }

        for (v) |*x| x.* = 0.0;
        i = 0;
        while (i < rows) : (i += 1) {
            const row = data[i * cols .. i * cols + cols];
            const ui = u[i];
            var j: usize = 0;
            while (j < cols) : (j += 1) v[j] += row[j] * ui;
        }

        var v_norm_sq: f32 = 0.0;
        for (v) |x| v_norm_sq += x * x;
        const v_norm = @sqrt(v_norm_sq);
        if (v_norm > 1e-12) {
            for (v) |*x| x.* /= v_norm;
        }
    }

    var sigma: f64 = 0.0;
    var i: usize = 0;
    while (i < rows) : (i += 1) {
        const row = data[i * cols .. i * cols + cols];
        var partial: f64 = 0.0;
        var j: usize = 0;
        while (j < cols) : (j += 1) partial += @as(f64, row[j]) * @as(f64, v[j]);
        sigma += @as(f64, u[i]) * partial;
    }

    const sigma_f32: f32 = @floatCast(sigma);
    return if (sigma_f32 >= 0) sigma_f32 else -sigma_f32;
}

fn constrainSpectralNorm(allocator: Allocator, weight: *Tensor, rows: usize, cols: usize, target: f32, seed: u64) !void {
    const norm = try spectralNormPowerIteration(allocator, weight.data, rows, cols, LAYER_SPECTRAL_POWER_ITERATIONS, seed);
    if (norm > target and norm > 1e-12) {
        const factor = target / norm;
        for (weight.data) |*x| x.* *= factor;
    }
}

pub const RSFLayerConfig = struct {
    clip_min: f32 = -5.0,
    clip_max: f32 = 5.0,
    seed_offset: u64 = 0,
    grad_mean: bool = true,
};

pub const RSFConfig = struct {
    clip_min: f32 = -5.0,
    clip_max: f32 = 5.0,
    grad_mean: bool = true,
    max_dim: usize = 1 << 20,
    max_layers: usize = 1 << 20,
};

const SAVE_VERSION: u32 = 5;

var scratch_gpa_backing = std.heap.GeneralPurposeAllocator(.{}){};

fn scratchAllocator() Allocator {
    return scratch_gpa_backing.allocator();
}

fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch return error.Overflow;
}

fn checkedMulU64(a: u64, b: u64) !u64 {
    return std.math.mul(u64, a, b) catch return error.Overflow;
}

fn checkedAddU64(a: u64, b: u64) !u64 {
    return std.math.add(u64, a, b) catch return error.Overflow;
}

fn checkedCastU64ToUsize(v: u64) !usize {
    if (v > std.math.maxInt(usize)) return error.TooLarge;
    return @intCast(v);
}

fn validateClipRange(clip_min: f32, clip_max: f32) !void {
    if (!std.math.isFinite(clip_min) or !std.math.isFinite(clip_max)) return error.NonFinite;
    if (!(clip_min < clip_max)) return error.InvalidConfig;
    if (clip_max > 20.0 or clip_min < -20.0) return error.InvalidConfig;
}

fn validateComparisonTolerances(abs_tol: f32, rel_tol: f32) !void {
    if (!std.math.isFinite(abs_tol) or !std.math.isFinite(rel_tol)) return error.InvalidTolerance;
    if (abs_tol < 0.0 or rel_tol < 0.0) return error.InvalidTolerance;
}

fn validateTensor2D(t: *const Tensor) !void {
    if (t.shape.dims.len != 2) return error.ShapeMismatch;
    const expected = try checkedMul(t.shape.dims[0], t.shape.dims[1]);
    if (t.data.len != expected) return error.DataLengthMismatch;
}

fn validateTensor2DShape(t: *const Tensor, rows: usize, cols: usize) !void {
    if (t.shape.dims.len != 2 or t.shape.dims[0] != rows or t.shape.dims[1] != cols) return error.ShapeMismatch;
    const expected = try checkedMul(rows, cols);
    if (t.data.len != expected) return error.DataLengthMismatch;
}

fn tensorHasShape(t: *const Tensor, rows: usize, cols: usize) bool {
    return t.shape.dims.len == 2 and t.shape.dims[0] == rows and t.shape.dims[1] == cols;
}

fn tensorsSameShape(a: *const Tensor, b: *const Tensor) bool {
    return a.shape.dims.len == 2 and b.shape.dims.len == 2 and a.shape.dims[0] == b.shape.dims[0] and a.shape.dims[1] == b.shape.dims[1];
}

fn ensureFiniteSlice(data: []const f32) !void {
    for (data) |v| {
        if (!std.math.isFinite(v)) return error.NonFinite;
    }
}

fn zeroTensor(t: *Tensor) void {
    @memset(t.data, @as(f32, 0.0));
}

fn tensorsOverlap(a: *const Tensor, b: *const Tensor) bool {
    if (a.data.len == 0 or b.data.len == 0) return false;
    const a_start: usize = @intFromPtr(a.data.ptr);
    const b_start: usize = @intFromPtr(b.data.ptr);
    const a_bytes = std.math.mul(usize, a.data.len, @sizeOf(f32)) catch return true;
    const b_bytes = std.math.mul(usize, b.data.len, @sizeOf(f32)) catch return true;
    const a_end = std.math.add(usize, a_start, a_bytes) catch return true;
    const b_end = std.math.add(usize, b_start, b_bytes) catch return true;
    return a_start < b_end and b_start < a_end;
}

fn tensorClone(allocator: Allocator, src: *const Tensor) !Tensor {
    try validateTensor2D(src);
    var dst = try Tensor.init(allocator, &.{ src.shape.dims[0], src.shape.dims[1] });
    errdefer dst.deinit();
    @memcpy(dst.data, src.data);
    return dst;
}

fn tensorAllCloseEq(a: *const Tensor, b: *const Tensor, abs_tol: f32, rel_tol: f32) !bool {
    try validateComparisonTolerances(abs_tol, rel_tol);
    try validateTensor2D(a);
    try validateTensor2D(b);
    if (!tensorsSameShape(a, b)) return false;
    var i: usize = 0;
    while (i < a.data.len) : (i += 1) {
        const av = a.data[i];
        const bv = b.data[i];
        if (!std.math.isFinite(av) or !std.math.isFinite(bv)) return false;
        const diff = @abs(av - bv);
        const scale = @max(@abs(av), @abs(bv));
        if (diff > abs_tol + rel_tol * scale) return false;
    }
    return true;
}

fn validateModelConfigValues(dim: usize, num_layers: usize, cfg: RSFConfig) !void {
    if (dim == 0) return error.InvalidDimension;
    if (num_layers == 0) return error.InvalidLayerCount;
    try validateClipRange(cfg.clip_min, cfg.clip_max);
    if (cfg.max_dim == 0 or cfg.max_layers == 0) return error.InvalidConfig;
    if (dim > cfg.max_dim or num_layers > cfg.max_layers) return error.InvalidConfig;
}

const LayerCore = struct {
    s_weight: Tensor,
    t_weight: Tensor,
    s_weight_grad: ?Tensor,
    t_weight_grad: ?Tensor,
    dim: usize,
    allocator: Allocator,
    clip_min: f32,
    clip_max: f32,
    grad_mean: bool,
    rwlock: Thread.RwLock,

    fn initOwned(allocator: Allocator, dim: usize, config: RSFLayerConfig) !LayerCore {
        if (dim == 0) return error.InvalidDimension;
        try validateClipRange(config.clip_min, config.clip_max);

        _ = try checkedMul(dim, dim + 1);

        const fan_in: f32 = @floatFromInt(dim);
        const fan_out: f32 = @floatFromInt(dim);
        const fan_sum = fan_in + fan_out;
        if (!(fan_sum > 0.0)) return error.InvalidDimension;

        const xavier_bound: f32 = @sqrt(6.0 / fan_sum);
        const weight_shape = [_]usize{ dim, dim + 1 };

        const seed1 = try checkedAddU64(42, config.seed_offset);
        const seed2 = try checkedAddU64(43, config.seed_offset);

        var s_w = try Tensor.randomUniform(allocator, &weight_shape, -xavier_bound, xavier_bound, seed1);
        errdefer s_w.deinit();

        var t_w = try Tensor.randomUniform(allocator, &weight_shape, -xavier_bound, xavier_bound, seed2);
        errdefer t_w.deinit();

        for (0..dim) |d| {
            s_w.data[d * (dim + 1) + dim] = 0.0;
            t_w.data[d * (dim + 1) + dim] = 0.0;
        }

        try constrainSpectralNorm(allocator, &s_w, dim, dim + 1, LAYER_TARGET_SPECTRAL_NORM, checkedAddU64(seed1, 9_000_000) catch seed1);
        try constrainSpectralNorm(allocator, &t_w, dim, dim + 1, LAYER_TARGET_SPECTRAL_NORM, checkedAddU64(seed2, 9_000_000) catch seed2);

        return LayerCore{
            .s_weight = s_w,
            .t_weight = t_w,
            .s_weight_grad = null,
            .t_weight_grad = null,
            .dim = dim,
            .allocator = allocator,
            .clip_min = config.clip_min,
            .clip_max = config.clip_max,
            .grad_mean = config.grad_mean,
            .rwlock = .{},
        };
    }

    fn deinitOwned(self: *LayerCore) void {
        self.s_weight.deinit();
        self.t_weight.deinit();
        if (self.s_weight_grad) |*g| g.deinit();
        if (self.t_weight_grad) |*g| g.deinit();
        self.s_weight_grad = null;
        self.t_weight_grad = null;
    }

    pub fn ensureGradients(self: *LayerCore) !void {
        const need_swg = self.s_weight_grad == null;
        const need_twg = self.t_weight_grad == null;

        if (!(need_swg or need_twg)) return;

        const weight_shape = [_]usize{ self.dim, self.dim + 1 };

        var swg_new: ?Tensor = null;
        var twg_new: ?Tensor = null;

        errdefer {
            if (swg_new) |*t| t.deinit();
            if (twg_new) |*t| t.deinit();
        }

        if (need_swg) swg_new = try Tensor.zeros(self.allocator, &weight_shape);
        if (need_twg) twg_new = try Tensor.zeros(self.allocator, &weight_shape);

        if (swg_new) |t| self.s_weight_grad = t;
        if (twg_new) |t| self.t_weight_grad = t;

        swg_new = null;
        twg_new = null;
    }

    fn zeroGradients(self: *LayerCore) void {
        if (self.s_weight_grad) |*g| zeroTensor(g);
        if (self.t_weight_grad) |*g| zeroTensor(g);
    }

    fn validatePair(self: *const LayerCore, a: *const Tensor, b: *const Tensor) !usize {
        try validateTensor2D(a);
        try validateTensor2D(b);
        if (a.shape.dims[1] != self.dim or b.shape.dims[1] != self.dim) return error.ShapeMismatch;
        if (a.shape.dims[0] != b.shape.dims[0]) return error.ShapeMismatch;
        const batch_size = a.shape.dims[0];
        if (batch_size == 0) return error.InvalidBatchSize;
        _ = try checkedMul(batch_size, self.dim);
        return batch_size;
    }

    fn computeTranslationRow(self: *const LayerCore, input_row: []const f32, out_row: []f32) void {
        const dim = self.dim;
        const dim1 = dim + 1;
        var d: usize = 0;
        while (d < dim) : (d += 1) {
            var sum: f32 = self.t_weight.data[d * dim1 + dim];
            const w_row = self.t_weight.data[d * dim1 .. d * dim1 + dim];
            var j: usize = 0;
            while (j < dim) : (j += 1) sum += w_row[j] * input_row[j];
            out_row[d] = sum;
        }
    }

    fn computeScaleRow(self: *const LayerCore, input_row: []const f32, out_row: []f32) void {
        const dim = self.dim;
        const dim1 = dim + 1;
        var d: usize = 0;
        while (d < dim) : (d += 1) {
            var sum: f32 = self.s_weight.data[d * dim1 + dim];
            const w_row = self.s_weight.data[d * dim1 .. d * dim1 + dim];
            var j: usize = 0;
            while (j < dim) : (j += 1) sum += w_row[j] * input_row[j];
            const clipped = if (sum < self.clip_min) self.clip_min else if (sum > self.clip_max) self.clip_max else sum;
            out_row[d] = @exp(clipped);
        }
    }

    fn forwardInPlace(self: *const LayerCore, x1: *Tensor, x2: *Tensor, scale: []f32, trans: []f32) !void {
        if (scale.len < self.dim or trans.len < self.dim) return error.DataLengthMismatch;
        if (tensorsOverlap(x1, x2)) return error.AliasedBuffers;
        const batch_size = try self.validatePair(x1, x2);

        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const x1_row = x1.data[b * self.dim .. b * self.dim + self.dim];
            const x2_row = x2.data[b * self.dim .. b * self.dim + self.dim];

            self.computeScaleRow(x2_row, scale);

            var i: usize = 0;
            while (i < self.dim) : (i += 1) x1_row[i] *= scale[i];

            self.computeTranslationRow(x1_row, trans);

            i = 0;
            while (i < self.dim) : (i += 1) x2_row[i] += trans[i];
        }
    }

    fn inverseInPlace(self: *const LayerCore, y1: *Tensor, y2: *Tensor, scale: []f32, trans: []f32) !void {
        if (scale.len < self.dim or trans.len < self.dim) return error.DataLengthMismatch;
        if (tensorsOverlap(y1, y2)) return error.AliasedBuffers;
        const batch_size = try self.validatePair(y1, y2);

        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const y1_row = y1.data[b * self.dim .. b * self.dim + self.dim];
            const y2_row = y2.data[b * self.dim .. b * self.dim + self.dim];

            self.computeTranslationRow(y1_row, trans);

            var i: usize = 0;
            while (i < self.dim) : (i += 1) y2_row[i] -= trans[i];

            self.computeScaleRow(y2_row, scale);

            i = 0;
            while (i < self.dim) : (i += 1) y1_row[i] /= scale[i];
        }
    }

    fn backwardFromOutputsRow(
        self: *LayerCore,
        y1_row: []const f32,
        y2_row: []const f32,
        dy1_row: []const f32,
        dy2_row: []const f32,
        x1_row_out: []f32,
        x2_row_out: []f32,
        dx1_row_out: []f32,
        dx2_row_out: []f32,
        dy1_total: []f32,
        ds: []f32,
        grad_scale: f32,
    ) !void {
        const dim = self.dim;
        const dim1 = dim + 1;
        if (!std.math.isFinite(grad_scale)) return error.NonFinite;
        if (y1_row.len != dim or y2_row.len != dim) return error.ShapeMismatch;
        if (dy1_row.len != dim or dy2_row.len != dim) return error.ShapeMismatch;
        if (x1_row_out.len != dim or x2_row_out.len != dim) return error.ShapeMismatch;
        if (dx1_row_out.len != dim or dx2_row_out.len != dim) return error.ShapeMismatch;
        if (dy1_total.len != dim or ds.len != dim) return error.DataLengthMismatch;

        @memcpy(dy1_total, dy1_row);
        {
            var d: usize = 0;
            while (d < dim) : (d += 1) {
                const dy2_val = dy2_row[d];
                const t_row = self.t_weight.data[d * dim1 .. d * dim1 + dim];
                var j: usize = 0;
                while (j < dim) : (j += 1) dy1_total[j] += t_row[j] * dy2_val;
            }
        }

        if (self.t_weight_grad) |*twg| {
            var d: usize = 0;
            while (d < dim) : (d += 1) {
                const dyv = dy2_row[d] * grad_scale;
                var j: usize = 0;
                while (j < dim) : (j += 1) twg.data[d * dim1 + j] += dyv * y1_row[j];
                twg.data[d * dim1 + dim] += dyv;
            }
        }

        {
            var d: usize = 0;
            while (d < dim) : (d += 1) {
                var trans_sum: f32 = self.t_weight.data[d * dim1 + dim];
                const t_row = self.t_weight.data[d * dim1 .. d * dim1 + dim];
                var j: usize = 0;
                while (j < dim) : (j += 1) trans_sum += t_row[j] * y1_row[j];
                x2_row_out[d] = y2_row[d] - trans_sum;
            }
        }

        {
            var d2: usize = 0;
            while (d2 < dim) : (d2 += 1) {
                var pre_sum: f32 = self.s_weight.data[d2 * dim1 + dim];
                const s_row = self.s_weight.data[d2 * dim1 .. d2 * dim1 + dim];
                var j2: usize = 0;
                while (j2 < dim) : (j2 += 1) pre_sum += s_row[j2] * x2_row_out[j2];

                const clipped = if (pre_sum < self.clip_min) self.clip_min else if (pre_sum > self.clip_max) self.clip_max else pre_sum;
                const scale = @exp(clipped);

                x1_row_out[d2] = y1_row[d2] / scale;
                dx1_row_out[d2] = dy1_total[d2] * scale;
                ds[d2] = if (pre_sum < self.clip_min or pre_sum > self.clip_max) 0.0 else dy1_total[d2] * y1_row[d2];
            }
        }

        if (self.s_weight_grad) |*swg| {
            var d3: usize = 0;
            while (d3 < dim) : (d3 += 1) {
                const dsv = ds[d3] * grad_scale;
                var j3: usize = 0;
                while (j3 < dim) : (j3 += 1) swg.data[d3 * dim1 + j3] += dsv * x2_row_out[j3];
                swg.data[d3 * dim1 + dim] += dsv;
            }
        }

        @memcpy(dx2_row_out, dy2_row);
        {
            var d5: usize = 0;
            while (d5 < dim) : (d5 += 1) {
                const ds_val = ds[d5];
                const s_row = self.s_weight.data[d5 * dim1 .. d5 * dim1 + dim];
                var j4: usize = 0;
                while (j4 < dim) : (j4 += 1) dx2_row_out[j4] += s_row[j4] * ds_val;
            }
        }
    }
};

const LayerRegistryEntry = struct {
    core: *LayerCore,
    active_ops: usize,
    destroyed: bool,
};

fn maybeShrinkRegistry(comptime EntryType: type, registry: *std.AutoHashMap(u64, EntryType)) void {
    if (registry.count() == 0) {
        registry.deinit();
        registry.* = std.AutoHashMap(u64, EntryType).init(std.heap.page_allocator);
    }
}

fn registerRegistryCore(
    comptime CoreType: type,
    comptime EntryType: type,
    mutex: *Thread.Mutex,
    registry: *std.AutoHashMap(u64, EntryType),
    next_id: *std.atomic.Value(u64),
    core: *CoreType,
) !u64 {
    mutex.lock();
    defer mutex.unlock();
    var id: u64 = 0;
    while (id == 0 or registry.contains(id)) {
        id = next_id.fetchAdd(1, .monotonic);
    }
    try registry.put(id, .{ .core = core, .active_ops = 0, .destroyed = false });
    return id;
}

fn acquireRegistryCore(
    comptime CoreType: type,
    comptime EntryType: type,
    mutex: *Thread.Mutex,
    registry: *std.AutoHashMap(u64, EntryType),
    id: u64,
) !*CoreType {
    if (id == 0) return error.NotInitialized;
    mutex.lock();
    defer mutex.unlock();
    const entry = registry.getPtr(id) orelse return error.NotInitialized;
    if (entry.destroyed) return error.NotInitialized;
    if (entry.active_ops == std.math.maxInt(usize)) return error.TooManyActiveOperations;
    entry.active_ops += 1;
    return entry.core;
}

fn releaseRegistryCore(
    comptime CoreType: type,
    comptime EntryType: type,
    mutex: *Thread.Mutex,
    registry: *std.AutoHashMap(u64, EntryType),
    id: u64,
    destroy_fn: *const fn (*CoreType) void,
) void {
    if (id == 0) return;
    var core_to_destroy: ?*CoreType = null;
    mutex.lock();
    if (registry.getPtr(id)) |entry| {
        if (entry.active_ops > 0) entry.active_ops -= 1;
        if (entry.destroyed and entry.active_ops == 0) {
            if (registry.fetchRemove(id)) |kv| {
                core_to_destroy = kv.value.core;
                maybeShrinkRegistry(EntryType, registry);
            }
        }
    }
    mutex.unlock();
    if (core_to_destroy) |core| destroy_fn(core);
}

fn requestDestroyRegistryCore(
    comptime CoreType: type,
    comptime EntryType: type,
    mutex: *Thread.Mutex,
    registry: *std.AutoHashMap(u64, EntryType),
    id: u64,
    destroy_fn: *const fn (*CoreType) void,
) void {
    if (id == 0) return;
    var core_to_destroy: ?*CoreType = null;
    mutex.lock();
    if (registry.getPtr(id)) |entry| {
        entry.destroyed = true;
        if (entry.active_ops == 0) {
            if (registry.fetchRemove(id)) |kv| {
                core_to_destroy = kv.value.core;
                maybeShrinkRegistry(EntryType, registry);
            }
        }
    }
    mutex.unlock();
    if (core_to_destroy) |core| destroy_fn(core);
}

var g_layer_registry_mutex: Thread.Mutex = .{};
var g_layer_registry = std.AutoHashMap(u64, LayerRegistryEntry).init(std.heap.page_allocator);
var g_layer_next_id = std.atomic.Value(u64).init(1);

fn destroyLayerCore(core: *LayerCore) void {
    const allocator = core.allocator;
    core.deinitOwned();
    allocator.destroy(core);
}

fn registerLayerCore(core: *LayerCore) !u64 {
    return registerRegistryCore(LayerCore, LayerRegistryEntry, &g_layer_registry_mutex, &g_layer_registry, &g_layer_next_id, core);
}

fn acquireLayerCore(id: u64) !*LayerCore {
    return acquireRegistryCore(LayerCore, LayerRegistryEntry, &g_layer_registry_mutex, &g_layer_registry, id);
}

fn releaseLayerCore(id: u64) void {
    releaseRegistryCore(LayerCore, LayerRegistryEntry, &g_layer_registry_mutex, &g_layer_registry, id, destroyLayerCore);
}

fn requestDestroyLayerCore(id: u64) void {
    requestDestroyRegistryCore(LayerCore, LayerRegistryEntry, &g_layer_registry_mutex, &g_layer_registry, id, destroyLayerCore);
}

pub const RSFLayer = struct {
    id: u64 = 0,

    pub fn init(allocator: Allocator, dim: usize) !RSFLayer {
        return initWithConfig(allocator, dim, .{});
    }

    pub fn initWithArena(arena: *memory.ArenaAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(arena.allocator(), dim, config);
    }

    pub fn initWithPool(pool: *memory.PoolAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(pool.allocator(), dim, config);
    }

    pub fn initWithSlab(slab: *memory.SlabAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(slab.allocator(), dim, config);
    }

    pub fn initWithBuddy(buddy: *memory.BuddyAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(buddy.allocator(), dim, config);
    }

    pub fn initWithConfig(allocator: Allocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        const core = try allocator.create(LayerCore);
        errdefer allocator.destroy(core);

        core.* = try LayerCore.initOwned(allocator, dim, config);
        errdefer core.deinitOwned();

        const id = try registerLayerCore(core);
        return RSFLayer{ .id = id };
    }

    pub fn ensureGradients(self: *RSFLayer) !void {
        const id = try bindLayerHandle(self);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try core.ensureGradients();
    }

    pub fn deinit(self: *RSFLayer) void {
        const id = self.id;
        if (id == 0) return;
        const should_destroy = shouldDestroyLayerHandle(self);
        self.id = 0;
        if (should_destroy) requestDestroyLayerCore(id);
    }

    pub fn zeroGradients(self: *RSFLayer) !void {
        const id = try bindLayerHandle(self);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        core.zeroGradients();
    }

    pub fn forward(self: *const RSFLayer, x1: *Tensor, x2: *Tensor) !void {
        const id = try bindLayerHandle(self);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const allocator = scratchAllocator();
        const scale = try allocator.alloc(f32, core.dim);
        defer allocator.free(scale);
        const trans = try allocator.alloc(f32, core.dim);
        defer allocator.free(trans);
        try core.forwardInPlace(x1, x2, scale, trans);
    }

    pub fn inverse(self: *const RSFLayer, y1: *Tensor, y2: *Tensor) !void {
        const id = try bindLayerHandle(self);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        const allocator = scratchAllocator();
        const scale = try allocator.alloc(f32, core.dim);
        defer allocator.free(scale);
        const trans = try allocator.alloc(f32, core.dim);
        defer allocator.free(trans);
        try core.inverseInPlace(y1, y2, scale, trans);
    }

    pub fn verifyInvertible(self: *const RSFLayer, x1: *const Tensor, x2: *const Tensor, abs_tol: f32, rel_tol: f32) !bool {
        try validateComparisonTolerances(abs_tol, rel_tol);
        const id = try bindLayerHandle(self);
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();

        const allocator = scratchAllocator();
        var fx1 = try tensorClone(allocator, x1);
        defer fx1.deinit();
        var fx2 = try tensorClone(allocator, x2);
        defer fx2.deinit();

        const scale = try allocator.alloc(f32, core.dim);
        defer allocator.free(scale);
        const trans = try allocator.alloc(f32, core.dim);
        defer allocator.free(trans);
        try core.forwardInPlace(&fx1, &fx2, scale, trans);
        try core.inverseInPlace(&fx1, &fx2, scale, trans);

        const ok1 = try tensorAllCloseEq(x1, &fx1, abs_tol, rel_tol);
        if (!ok1) return false;
        const ok2 = try tensorAllCloseEq(x2, &fx2, abs_tol, rel_tol);
        return ok2;
    }
};

var g_layer_handle_mutex: Thread.Mutex = .{};
var g_layer_handle_owner = std.AutoHashMap(u64, usize).init(std.heap.page_allocator);

fn bindLayerHandle(self: *const RSFLayer) !u64 {
    const id = self.id;
    if (id == 0) return error.NotInitialized;
    const self_addr: usize = @intFromPtr(self);
    g_layer_handle_mutex.lock();
    defer g_layer_handle_mutex.unlock();
    if (g_layer_handle_owner.get(id)) |owner_addr| {
        if (owner_addr != self_addr) return error.HandleCopied;
    } else {
        try g_layer_handle_owner.put(id, self_addr);
    }
    return id;
}

fn maybeShrinkHandleMap(map: *std.AutoHashMap(u64, usize)) void {
    if (map.count() == 0) {
        map.deinit();
        map.* = std.AutoHashMap(u64, usize).init(std.heap.page_allocator);
    }
}

fn shouldDestroyLayerHandle(self: *RSFLayer) bool {
    const id = self.id;
    if (id == 0) return false;
    const self_addr: usize = @intFromPtr(self);
    g_layer_handle_mutex.lock();
    defer g_layer_handle_mutex.unlock();
    if (g_layer_handle_owner.get(id)) |owner_addr| {
        if (owner_addr == self_addr) {
            _ = g_layer_handle_owner.remove(id);
            maybeShrinkHandleMap(&g_layer_handle_owner);
            return true;
        }
        return false;
    }
    return true;
}

const RSFCore = struct {
    allocator: Allocator,
    dim: usize,
    num_layers: usize,
    layers: []LayerCore,
    cfg: RSFConfig,
    rwlock: Thread.RwLock,
    gpu_accel: ?accel.RSFAccelerator,
    gpu_available: std.atomic.Value(u8),
    gpu_weight_version: u64,
    cpu_weight_version: u64,
    f16_buf: ?[]f16,
    oftb: OFTB,
};

const ModelRegistryEntry = struct {
    core: *RSFCore,
    active_ops: usize,
    destroyed: bool,
};

var g_model_registry_mutex: Thread.Mutex = .{};
var g_model_registry = std.AutoHashMap(u64, ModelRegistryEntry).init(std.heap.page_allocator);
var g_model_next_id = std.atomic.Value(u64).init(1);

fn destroyModelCore(core: *RSFCore) void {
    if (core.gpu_accel) |*ga| {
        ga.deinit();
        core.gpu_accel = null;
    }
    if (core.f16_buf) |buf| {
        core.allocator.free(buf);
        core.f16_buf = null;
    }
    core.gpu_available.store(0, .monotonic);

    const allocator = core.allocator;
    for (core.layers) |*layer| layer.deinitOwned();
    allocator.free(core.layers);
    allocator.destroy(core);
}

fn registerModelCore(core: *RSFCore) !u64 {
    return registerRegistryCore(RSFCore, ModelRegistryEntry, &g_model_registry_mutex, &g_model_registry, &g_model_next_id, core);
}

fn acquireModelCore(id: u64) !*RSFCore {
    return acquireRegistryCore(RSFCore, ModelRegistryEntry, &g_model_registry_mutex, &g_model_registry, id);
}

fn releaseModelCore(id: u64) void {
    releaseRegistryCore(RSFCore, ModelRegistryEntry, &g_model_registry_mutex, &g_model_registry, id, destroyModelCore);
}

fn requestDestroyModelCore(id: u64) void {
    requestDestroyRegistryCore(RSFCore, ModelRegistryEntry, &g_model_registry_mutex, &g_model_registry, id, destroyModelCore);
}

fn checkedModelLayerCount(core: *const RSFCore) !usize {
    if (core.num_layers != core.layers.len) return error.InvalidModelState;
    if (core.layers.len == 0) return error.InvalidLayerCount;
    return core.layers.len;
}

fn validateModelMetadata(core: *const RSFCore) !void {
    const layer_count = try checkedModelLayerCount(core);
    try validateModelConfigValues(core.dim, layer_count, core.cfg);

    var i: usize = 0;
    while (i < layer_count) : (i += 1) {
        const layer = &core.layers[i];
        if (layer.dim != core.dim) return error.InvalidModelState;
        if (layer.clip_min != core.cfg.clip_min or layer.clip_max != core.cfg.clip_max or layer.grad_mean != core.cfg.grad_mean) return error.InvalidConfig;
        try validateTensor2DShape(&layer.s_weight, core.dim, core.dim + 1);
        try validateTensor2DShape(&layer.t_weight, core.dim, core.dim + 1);
    }
}

fn forwardOnCore(core: *const RSFCore, x: *Tensor) !void {
    try validateTensor2D(x);
    const layer_count = try checkedModelLayerCount(core);

    const dim2 = try checkedMul(core.dim, 2);
    if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = x.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;

    const allocator = scratchAllocator();

    const scale = try allocator.alloc(f32, core.dim);
    defer allocator.free(scale);

    const trans = try allocator.alloc(f32, core.dim);
    defer allocator.free(trans);

    const oftb_scale: f32 = @floatCast(OFTB.FRACTAL_SCALE);

    var l: usize = 0;
    while (l < layer_count) : (l += 1) {
        const layer = &core.layers[l];
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const row = x.data[b * dim2 .. b * dim2 + dim2];
            const x1_row = row[0..core.dim];
            const x2_row = row[core.dim..dim2];

            layer.computeScaleRow(x2_row, scale);

            var i: usize = 0;
            while (i < core.dim) : (i += 1) x1_row[i] *= scale[i];

            layer.computeTranslationRow(x1_row, trans);

            i = 0;
            while (i < core.dim) : (i += 1) x2_row[i] += trans[i];

            i = 0;
            while (i < core.dim) : (i += 1) {
                const a = x1_row[i];
                const b_val = x2_row[i];
                x1_row[i] = (a - b_val) * oftb_scale;
                x2_row[i] = (a + b_val) * oftb_scale;
            }
        }
    }
}

fn inverseOnCore(core: *const RSFCore, y: *Tensor) !void {
    try validateTensor2D(y);
    const layer_count = try checkedModelLayerCount(core);

    const dim2 = try checkedMul(core.dim, 2);
    if (y.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = y.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;

    const allocator = scratchAllocator();

    const trans = try allocator.alloc(f32, core.dim);
    defer allocator.free(trans);

    const scale = try allocator.alloc(f32, core.dim);
    defer allocator.free(scale);

    const oftb_scale: f32 = @floatCast(OFTB.FRACTAL_SCALE);

    var idx = layer_count;
    while (idx > 0) : (idx -= 1) {
        const layer = &core.layers[idx - 1];
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const row = y.data[b * dim2 .. b * dim2 + dim2];
            const y1_row = row[0..core.dim];
            const y2_row = row[core.dim..dim2];

            var i: usize = 0;
            while (i < core.dim) : (i += 1) {
                const a = y1_row[i];
                const b_val = y2_row[i];
                y1_row[i] = (a + b_val) * oftb_scale;
                y2_row[i] = (b_val - a) * oftb_scale;
            }

            layer.computeTranslationRow(y1_row, trans);

            i = 0;
            while (i < core.dim) : (i += 1) y2_row[i] -= trans[i];

            layer.computeScaleRow(y2_row, scale);

            i = 0;
            while (i < core.dim) : (i += 1) y1_row[i] /= scale[i];
        }
    }
}

fn backwardOnCore(core: *RSFCore, grad_output: *const Tensor, input: *const Tensor, output: *const Tensor, grad_input_out: *Tensor) !void {
    try validateTensor2D(grad_output);
    try validateTensor2D(input);
    try validateTensor2D(output);
    try validateTensor2D(grad_input_out);

    const layer_count = try checkedModelLayerCount(core);
    const dim = core.dim;
    const dim2 = try checkedMul(dim, 2);

    if (input.shape.dims[1] != dim2) return error.ShapeMismatch;
    if (!tensorsSameShape(grad_output, input)) return error.ShapeMismatch;
    if (!tensorsSameShape(output, input)) return error.ShapeMismatch;
    if (!tensorsSameShape(grad_input_out, input)) return error.ShapeMismatch;

    const batch_size = input.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;

    var li: usize = 0;
    while (li < layer_count) : (li += 1) try core.layers[li].ensureGradients();

    const grad_scale: f32 = blk: {
        if (!core.cfg.grad_mean) break :blk 1.0;
        const s = 1.0 / @as(f32, @floatFromInt(batch_size));
        break :blk if (std.math.isFinite(s)) s else 1.0;
    };

    const allocator = scratchAllocator();

    const y1_row = try allocator.alloc(f32, dim);
    defer allocator.free(y1_row);
    const y2_row = try allocator.alloc(f32, dim);
    defer allocator.free(y2_row);
    const dy1_row = try allocator.alloc(f32, dim);
    defer allocator.free(dy1_row);
    const dy2_row = try allocator.alloc(f32, dim);
    defer allocator.free(dy2_row);
    const x1_row = try allocator.alloc(f32, dim);
    defer allocator.free(x1_row);
    const x2_row = try allocator.alloc(f32, dim);
    defer allocator.free(x2_row);
    const dx1_row = try allocator.alloc(f32, dim);
    defer allocator.free(dx1_row);
    const dx2_row = try allocator.alloc(f32, dim);
    defer allocator.free(dx2_row);
    const dy1_total = try allocator.alloc(f32, dim);
    defer allocator.free(dy1_total);
    const ds = try allocator.alloc(f32, dim);
    defer allocator.free(ds);

    const oftb_scale: f32 = @floatCast(OFTB.FRACTAL_SCALE);

    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        @memcpy(y1_row, output.data[b * dim2 .. b * dim2 + dim]);
        @memcpy(y2_row, output.data[b * dim2 + dim .. b * dim2 + dim2]);

        @memcpy(dy1_row, grad_output.data[b * dim2 .. b * dim2 + dim]);
        @memcpy(dy2_row, grad_output.data[b * dim2 + dim .. b * dim2 + dim2]);

        var idx = layer_count;
        while (idx > 0) : (idx -= 1) {
            {
                var i: usize = 0;
                while (i < dim) : (i += 1) {
                    const a = y1_row[i];
                    const b_val = y2_row[i];
                    y1_row[i] = (a + b_val) * oftb_scale;
                    y2_row[i] = (b_val - a) * oftb_scale;
                }
            }
            {
                var i: usize = 0;
                while (i < dim) : (i += 1) {
                    const a = dy1_row[i];
                    const b_val = dy2_row[i];
                    dy1_row[i] = (a + b_val) * oftb_scale;
                    dy2_row[i] = (b_val - a) * oftb_scale;
                }
            }

            try core.layers[idx - 1].backwardFromOutputsRow(
                y1_row,
                y2_row,
                dy1_row,
                dy2_row,
                x1_row,
                x2_row,
                dx1_row,
                dx2_row,
                dy1_total,
                ds,
                grad_scale,
            );
            @memcpy(y1_row, x1_row);
            @memcpy(y2_row, x2_row);
            @memcpy(dy1_row, dx1_row);
            @memcpy(dy2_row, dx2_row);
        }

        @memcpy(grad_input_out.data[b * dim2 .. b * dim2 + dim], dy1_row);
        @memcpy(grad_input_out.data[b * dim2 + dim .. b * dim2 + dim2], dy2_row);
    }
}

fn layerGPUCompatible(layer: *const LayerCore, cfg: *const RSFConfig, dim: usize) bool {
    if (layer.dim != dim) return false;
    if (layer.clip_min != cfg.clip_min or layer.clip_max != cfg.clip_max or layer.grad_mean != cfg.grad_mean) return false;
    if (layer.clip_min != -5.0 or layer.clip_max != 5.0) return false;
    return true;
}

fn modelGPUCompatible(core: *const RSFCore) bool {
    if (comptime !accel.gpu_enabled) return false;
    if (core.layers.len == 0) return false;
    for (core.layers) |*layer| {
        if (!layerGPUCompatible(layer, &core.cfg, core.dim)) return false;
    }
    return true;
}

fn disableGPU(core: *RSFCore) void {
    core.gpu_available.store(0, .monotonic);
    if (core.gpu_accel) |*ga| {
        ga.deinit();
        core.gpu_accel = null;
    }
    if (core.f16_buf) |buf| {
        core.allocator.free(buf);
        core.f16_buf = null;
    }
    core.gpu_weight_version = 0;
}

fn validateF16Convertible(data: []const f32) !void {
    const max_f16: f32 = @floatCast(std.math.floatMax(f16));
    for (data) |v| {
        if (!std.math.isFinite(v)) return error.NonFinite;
        if (@abs(v) > max_f16) return error.NumericFailure;
    }
}

fn uploadLayerToAccel(core: *RSFCore, layer: *const LayerCore, ga: *accel.RSFAccelerator, f16_buf: []f16) !void {
    const dim_sq = try checkedMul(core.dim, core.dim + 1);
    if (f16_buf.len < dim_sq) return error.DataLengthMismatch;

    try validateTensor2DShape(&layer.s_weight, core.dim, core.dim + 1);
    try validateTensor2DShape(&layer.t_weight, core.dim, core.dim + 1);
    try validateF16Convertible(layer.s_weight.data);
    try validateF16Convertible(layer.t_weight.data);

    var i: usize = 0;
    while (i < dim_sq) : (i += 1) f16_buf[i] = @floatCast(layer.s_weight.data[i]);
    try ga.setLayerWeightsS(0, f16_buf[0..dim_sq], core.dim, core.dim + 1);

    i = 0;
    while (i < dim_sq) : (i += 1) f16_buf[i] = @floatCast(layer.t_weight.data[i]);
    try ga.setLayerWeightsT(0, f16_buf[0..dim_sq], core.dim, core.dim + 1);
}

fn syncAllLayersGPU(core: *RSFCore) !void {
    var success = false;
    defer if (!success) disableGPU(core);

    if (comptime !accel.gpu_enabled) return error.GPUUnsupportedConfiguration;
    try validateModelMetadata(core);
    if (!modelGPUCompatible(core)) return error.GPUUnsupportedConfiguration;

    const dim_sq = try checkedMul(core.dim, core.dim + 1);

    for (core.layers) |*layer| {
        try ensureFiniteSlice(layer.s_weight.data);
        try ensureFiniteSlice(layer.t_weight.data);
        try validateF16Convertible(layer.s_weight.data);
        try validateF16Convertible(layer.t_weight.data);
    }

    const local_f16 = try core.allocator.alloc(f16, dim_sq);
    var local_f16_owned = true;
    errdefer if (local_f16_owned) core.allocator.free(local_f16);

    var staged_accel = accel.RSFAccelerator.initMultiLayer(core.dim, 1, core.allocator) catch return error.NoGPUAvailable;
    var staged_owned = true;
    errdefer if (staged_owned) staged_accel.deinit();

    try staged_accel.setClipRange(@floatCast(core.cfg.clip_min), @floatCast(core.cfg.clip_max));

    for (core.layers) |*layer| {
        try uploadLayerToAccel(core, layer, &staged_accel, local_f16);
    }

    if (core.gpu_accel) |*ga| ga.deinit();
    if (core.f16_buf) |buf| core.allocator.free(buf);

    core.gpu_accel = staged_accel;
    staged_owned = false;
    core.f16_buf = local_f16;
    local_f16_owned = false;
    core.gpu_weight_version = core.cpu_weight_version;
    core.gpu_available.store(1, .monotonic);
    success = true;
}

fn tryForwardGPU(core: *RSFCore, x: *Tensor) !bool {
    if (comptime !accel.gpu_enabled) return false;
    if (!modelGPUCompatible(core)) {
        disableGPU(core);
        return false;
    }
    if (core.gpu_available.load(.monotonic) == 0) return false;
    if (core.gpu_weight_version != core.cpu_weight_version) return false;

    if (core.gpu_accel) |*ga| {
        const allocator = scratchAllocator();

        if (core.layers.len == 1) {
            if (ga.forwardFromTensor(x, allocator)) |result| {
                var gpu_result = result;
                if (!tensorHasShape(&gpu_result, x.shape.dims[0], x.shape.dims[1]) or gpu_result.data.len != x.data.len) {
                    gpu_result.deinit();
                    disableGPU(core);
                    return false;
                }
                x.deinit();
                x.* = gpu_result;
                return true;
            } else |_| {
                disableGPU(core);
                return false;
            }
        }

        if (core.f16_buf == null) {
            disableGPU(core);
            return false;
        }
        const f16_buf = core.f16_buf.?;
        const dim_sq = checkedMul(core.dim, core.dim + 1) catch {
            disableGPU(core);
            return false;
        };
        if (f16_buf.len < dim_sq) {
            disableGPU(core);
            return false;
        }

        var working = tensorClone(allocator, x) catch {
            disableGPU(core);
            return false;
        };

        for (core.layers) |*layer| {
            uploadLayerToAccel(core, layer, ga, f16_buf) catch {
                working.deinit();
                disableGPU(core);
                return false;
            };

            if (ga.forwardFromTensor(&working, allocator)) |result| {
                var gpu_result = result;
                if (!tensorHasShape(&gpu_result, working.shape.dims[0], working.shape.dims[1]) or gpu_result.data.len != working.data.len) {
                    gpu_result.deinit();
                    working.deinit();
                    disableGPU(core);
                    return false;
                }
                working.deinit();
                working = gpu_result;
            } else |_| {
                working.deinit();
                disableGPU(core);
                return false;
            }
        }
        x.deinit();
        x.* = working;
        return true;
    }

    return false;
}

const SavedLayerSnapshot = struct {
    clip_min: f32,
    clip_max: f32,
    grad_mean: bool,
    s_weight: Tensor,
    t_weight: Tensor,
};

const SavedModelSnapshot = struct {
    allocator: Allocator,
    dim: usize,
    num_layers: usize,
    cfg: RSFConfig,
    layers: []SavedLayerSnapshot,

    fn deinit(self: *SavedModelSnapshot) void {
        const layers = self.layers;
        self.layers = layers[0..0];
        for (layers) |*layer| {
            layer.s_weight.deinit();
            layer.t_weight.deinit();
        }
        if (layers.len != 0) self.allocator.free(layers);
    }
};

fn snapshotModelForSave(allocator: Allocator, core: *const RSFCore) !SavedModelSnapshot {
    try validateModelMetadata(core);
    const layer_count = core.layers.len;

    const layers = try allocator.alloc(SavedLayerSnapshot, layer_count);
    errdefer allocator.free(layers);

    var initialized: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < initialized) : (i += 1) {
            layers[i].s_weight.deinit();
            layers[i].t_weight.deinit();
        }
    }

    var i: usize = 0;
    while (i < layer_count) : (i += 1) {
        const layer = &core.layers[i];
        try validateClipRange(layer.clip_min, layer.clip_max);
        try ensureFiniteSlice(layer.s_weight.data);
        try ensureFiniteSlice(layer.t_weight.data);

        var sw = try tensorClone(allocator, &layer.s_weight);
        errdefer sw.deinit();
        const tw = try tensorClone(allocator, &layer.t_weight);

        layers[i] = .{
            .clip_min = layer.clip_min,
            .clip_max = layer.clip_max,
            .grad_mean = layer.grad_mean,
            .s_weight = sw,
            .t_weight = tw,
        };
        initialized += 1;
    }

    return .{
        .allocator = allocator,
        .dim = core.dim,
        .num_layers = core.num_layers,
        .cfg = core.cfg,
        .layers = layers,
    };
}

pub const RSF = struct {
    id: u64 = 0,
    ctrl: ?*RSFCore = null,

    pub fn init(allocator: Allocator, dim: usize, num_layers: usize) !RSF {
        return initWithConfig(allocator, dim, num_layers, .{});
    }

    pub fn initWithConfig(allocator: Allocator, dim: usize, num_layers: usize, cfg: RSFConfig) !RSF {
        try validateModelConfigValues(dim, num_layers, cfg);

        _ = try checkedMul(dim, dim + 1);
        _ = try checkedMul(dim, 2);

        const core = try allocator.create(RSFCore);
        errdefer allocator.destroy(core);

        core.* = .{
            .allocator = allocator,
            .dim = dim,
            .num_layers = num_layers,
            .layers = try allocator.alloc(LayerCore, num_layers),
            .cfg = cfg,
            .rwlock = .{},
            .gpu_accel = null,
            .gpu_available = std.atomic.Value(u8).init(0),
            .gpu_weight_version = 0,
            .cpu_weight_version = 1,
            .f16_buf = null,
            .oftb = OFTB.init(dim),
        };
        errdefer {
            if (core.gpu_accel) |*ga| {
                ga.deinit();
                core.gpu_accel = null;
            }
            if (core.f16_buf) |buf| {
                allocator.free(buf);
                core.f16_buf = null;
            }
            core.gpu_available.store(0, .monotonic);
        }
        errdefer allocator.free(core.layers);

        var initialized: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < initialized) : (j += 1) core.layers[j].deinitOwned();
        }

        var l: usize = 0;
        while (l < num_layers) : (l += 1) {
            const seed_base = try checkedMulU64(@as(u64, @intCast(l)), 10007);
            const layer_cfg = RSFLayerConfig{
                .clip_min = cfg.clip_min,
                .clip_max = cfg.clip_max,
                .seed_offset = seed_base,
                .grad_mean = cfg.grad_mean,
            };
            core.layers[l] = try LayerCore.initOwned(allocator, dim, layer_cfg);
            initialized += 1;
        }

        try validateModelMetadata(core);

        if (modelGPUCompatible(core)) {
            syncAllLayersGPU(core) catch disableGPU(core);
        }

        const id = try registerModelCore(core);
        return RSF{ .id = id, .ctrl = core };
    }

    pub fn deinit(self: *RSF) void {
        const id = self.id;
        if (id == 0) return;
        const should_destroy = shouldDestroyModelHandle(self);
        self.id = 0;
        self.ctrl = null;
        if (should_destroy) requestDestroyModelCore(id);
    }

    pub fn isGPUAvailable(self: *const RSF) bool {
        const id = bindModelHandle(self) catch return false;
        const core = acquireModelCore(id) catch return false;
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        return modelGPUCompatible(core) and core.gpu_available.load(.monotonic) != 0 and core.gpu_weight_version == core.cpu_weight_version and core.gpu_accel != null;
    }

    pub fn syncWeightsToGPU(self: *RSF) !void {
        const id = try bindModelHandle(self);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try syncAllLayersGPU(core);
    }

    pub fn zeroGradients(self: *RSF) !void {
        const id = try bindModelHandle(self);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        for (core.layers) |*layer| layer.zeroGradients();
    }

    pub fn forwardCPU(self: *RSF, x: *Tensor) !void {
        const id = try bindModelHandle(self);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try forwardOnCore(core, x);
    }

    pub fn forward(self: *RSF, x: *Tensor) !void {
        const id = try bindModelHandle(self);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);

        try validateTensor2D(x);
        const dim2 = try checkedMul(core.dim, 2);
        if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
        if (x.shape.dims[0] == 0) return error.InvalidBatchSize;

        if (comptime accel.gpu_enabled) {
            const needs_write = core.gpu_available.load(.monotonic) != 0 or modelGPUCompatible(core);
            if (needs_write) {
                var gpu_succeeded = false;
                {
                    core.rwlock.lock();
                    defer core.rwlock.unlock();

                    if (modelGPUCompatible(core)) {
                        if (try tryForwardGPU(core, x)) {
                            gpu_succeeded = true;
                        } else {
                            syncAllLayersGPU(core) catch {};
                            if (try tryForwardGPU(core, x)) {
                                gpu_succeeded = true;
                            }
                        }
                    } else if (core.gpu_available.load(.monotonic) != 0 or core.gpu_accel != null or core.f16_buf != null or core.gpu_weight_version != 0) {
                        disableGPU(core);
                    }
                }
                if (!gpu_succeeded) {
                    core.rwlock.lockShared();
                    defer core.rwlock.unlockShared();
                    try forwardOnCore(core, x);
                }
            } else {
                core.rwlock.lockShared();
                defer core.rwlock.unlockShared();
                try forwardOnCore(core, x);
            }
        } else {
            core.rwlock.lockShared();
            defer core.rwlock.unlockShared();
            try forwardOnCore(core, x);
        }
    }

    pub fn inverse(self: *RSF, y: *Tensor) !void {
        const id = try bindModelHandle(self);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try inverseOnCore(core, y);
    }

    pub fn backward(self: *RSF, grad_output: *const Tensor, input: *const Tensor, output: *const Tensor, grad_input_out: *Tensor) !void {
        const id = try bindModelHandle(self);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try backwardOnCore(core, grad_output, input, output, grad_input_out);
    }

    pub fn notifyWeightsChanged(self: *RSF) !void {
        const id = try bindModelHandle(self);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        core.cpu_weight_version +%= 1;
    }

    pub fn verifyInvertible(self: *RSF, x: *const Tensor, abs_tol: f32, rel_tol: f32) !bool {
        try validateComparisonTolerances(abs_tol, rel_tol);
        const id = try bindModelHandle(self);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();

        const allocator = scratchAllocator();
        var y = try tensorClone(allocator, x);
        defer y.deinit();
        try forwardOnCore(core, &y);
        try inverseOnCore(core, &y);
        return tensorAllCloseEq(x, &y, abs_tol, rel_tol);
    }

    pub fn save(self: *const RSF, path: []const u8) !void {
        const id = try bindModelHandle(self);
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);

        const allocator = scratchAllocator();

        core.rwlock.lockShared();
        var snapshot = snapshotModelForSave(allocator, core) catch |err| {
            core.rwlock.unlockShared();
            return err;
        };
        core.rwlock.unlockShared();
        defer snapshot.deinit();

        try writeSnapshotVersion5ToPath(&snapshot, path, allocator);
    }

    pub fn load(allocator: Allocator, path: []const u8) !RSF {
        return loadWithConfig(allocator, path, null);
    }

    pub fn loadWithConfig(allocator: Allocator, path: []const u8, policy: ?RSFConfig) !RSF {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var buffered = std.io.bufferedReader(file.reader());
        const r = buffered.reader();

        var magic: [4]u8 = undefined;
        try r.readNoEof(&magic);
        if (!std.mem.eql(u8, &magic, "RSF0")) return error.BadFileFormat;

        const version = try r.readInt(u32, .little);
        if (version != SAVE_VERSION) return error.UnsupportedVersion;

        const num_layers_u64 = try r.readInt(u64, .little);
        const dim_u64 = try r.readInt(u64, .little);
        if (num_layers_u64 == 0) return error.InvalidLayerCount;
        if (dim_u64 == 0) return error.InvalidDimension;

        const policy_max_dim: usize = if (policy) |p| p.max_dim else (1 << 20);
        const policy_max_layers: usize = if (policy) |p| p.max_layers else (1 << 20);
        if (num_layers_u64 > @as(u64, @intCast(policy_max_layers)) or dim_u64 > @as(u64, @intCast(policy_max_dim))) return error.TooLarge;

        const num_layers = try checkedCastU64ToUsize(num_layers_u64);
        const dim = try checkedCastU64ToUsize(dim_u64);
        _ = try checkedMul(dim, dim + 1);
        _ = try checkedMul(dim, 2);

        var hasher = std.hash.Crc32.init();
        hasher.update("RSF0");
        crcUpdateU32LE(&hasher, version);
        crcUpdateU64LE(&hasher, num_layers_u64);
        crcUpdateU64LE(&hasher, dim_u64);

        const clip_min_bits = try r.readInt(u32, .little);
        const clip_max_bits = try r.readInt(u32, .little);
        const clip_min: f32 = @bitCast(clip_min_bits);
        const clip_max: f32 = @bitCast(clip_max_bits);
        const grad_mean = try readEncodedBool(r);
        try validateClipRange(clip_min, clip_max);

        crcUpdateU32LE(&hasher, clip_min_bits);
        crcUpdateU32LE(&hasher, clip_max_bits);
        crcUpdateU8(&hasher, if (grad_mean) @as(u8, 1) else @as(u8, 0));

        const saved_max_dim_u64 = try r.readInt(u64, .little);
        const saved_max_layers_u64 = try r.readInt(u64, .little);
        crcUpdateU64LE(&hasher, saved_max_dim_u64);
        crcUpdateU64LE(&hasher, saved_max_layers_u64);

        if (saved_max_dim_u64 == 0 or saved_max_layers_u64 == 0) return error.InvalidConfig;
        if (saved_max_dim_u64 < dim_u64 or saved_max_layers_u64 < num_layers_u64) return error.InvalidConfig;

        const loaded_cfg = RSFConfig{
            .clip_min = clip_min,
            .clip_max = clip_max,
            .grad_mean = grad_mean,
            .max_dim = try checkedCastU64ToUsize(saved_max_dim_u64),
            .max_layers = try checkedCastU64ToUsize(saved_max_layers_u64),
        };
        try validateModelConfigValues(dim, num_layers, loaded_cfg);

        const core = try allocator.create(RSFCore);
        errdefer allocator.destroy(core);

        core.* = .{
            .allocator = allocator,
            .dim = dim,
            .num_layers = num_layers,
            .layers = try allocator.alloc(LayerCore, num_layers),
            .cfg = loaded_cfg,
            .rwlock = .{},
            .gpu_accel = null,
            .gpu_available = std.atomic.Value(u8).init(0),
            .gpu_weight_version = 0,
            .cpu_weight_version = 1,
            .f16_buf = null,
            .oftb = OFTB.init(dim),
        };
        errdefer {
            if (core.gpu_accel) |*ga| {
                ga.deinit();
                core.gpu_accel = null;
            }
            if (core.f16_buf) |buf| {
                allocator.free(buf);
                core.f16_buf = null;
            }
            core.gpu_available.store(0, .monotonic);
        }
        errdefer allocator.free(core.layers);

        var initialized: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < initialized) : (j += 1) core.layers[j].deinitOwned();
        }

        var i: usize = 0;
        while (i < num_layers) : (i += 1) {
            const layer_clip_min_bits = try r.readInt(u32, .little);
            const layer_clip_max_bits = try r.readInt(u32, .little);
            const layer_clip_min: f32 = @bitCast(layer_clip_min_bits);
            const layer_clip_max: f32 = @bitCast(layer_clip_max_bits);
            const layer_grad_mean = try readEncodedBool(r);

            try validateClipRange(layer_clip_min, layer_clip_max);
            if (layer_clip_min != clip_min or layer_clip_max != clip_max or layer_grad_mean != grad_mean) return error.InvalidConfig;

            crcUpdateU32LE(&hasher, layer_clip_min_bits);
            crcUpdateU32LE(&hasher, layer_clip_max_bits);
            crcUpdateU8(&hasher, if (layer_grad_mean) @as(u8, 1) else @as(u8, 0));

            var s_w_new = try readTensorData(allocator, r, dim, dim + 1);
            errdefer s_w_new.deinit();
            var t_w_new = try readTensorData(allocator, r, dim, dim + 1);
            errdefer t_w_new.deinit();

            try validateTensor2DShape(&s_w_new, dim, dim + 1);
            try validateTensor2DShape(&t_w_new, dim, dim + 1);
            try ensureFiniteSlice(s_w_new.data);
            try ensureFiniteSlice(t_w_new.data);

            hashTensorDataVersion4(&hasher, &s_w_new);
            hashTensorDataVersion4(&hasher, &t_w_new);

            core.layers[i] = .{
                .s_weight = s_w_new,
                .t_weight = t_w_new,
                .s_weight_grad = null,
                .t_weight_grad = null,
                .dim = dim,
                .allocator = allocator,
                .clip_min = layer_clip_min,
                .clip_max = layer_clip_max,
                .grad_mean = layer_grad_mean,
                .rwlock = .{},
            };
            initialized += 1;
        }

        const stored_crc = try r.readInt(u32, .little);
        if (stored_crc != hasher.final()) return error.ChecksumMismatch;

        var eof_buf: [1]u8 = undefined;
        if ((try r.read(&eof_buf)) != 0) return error.TrailingData;

        try validateModelMetadata(core);

        if (modelGPUCompatible(core)) {
            syncAllLayersGPU(core) catch disableGPU(core);
        }

        const id = try registerModelCore(core);
        return RSF{ .id = id, .ctrl = core };
    }

    pub fn saveLoadRoundtrip(allocator: Allocator, self: *const RSF, path: []const u8, abs_tol: f32, rel_tol: f32) !bool {
        try validateComparisonTolerances(abs_tol, rel_tol);
        try self.save(path);
        var loaded = try RSF.load(allocator, path);
        defer loaded.deinit();

        const id1 = try bindModelHandle(self);
        const core1 = try acquireModelCore(id1);
        defer releaseModelCore(id1);

        const id2 = try bindModelHandle(&loaded);
        const core2 = try acquireModelCore(id2);
        defer releaseModelCore(id2);

        core1.rwlock.lockShared();
        defer core1.rwlock.unlockShared();
        core2.rwlock.lockShared();
        defer core2.rwlock.unlockShared();

        const layer_count1 = try checkedModelLayerCount(core1);
        const layer_count2 = try checkedModelLayerCount(core2);
        if (core1.dim != core2.dim) return false;
        if (layer_count1 != layer_count2) return false;
        if (core1.cfg.clip_min != core2.cfg.clip_min or core1.cfg.clip_max != core2.cfg.clip_max or core1.cfg.grad_mean != core2.cfg.grad_mean) return false;
        if (core1.cfg.max_dim != core2.cfg.max_dim or core1.cfg.max_layers != core2.cfg.max_layers) return false;

        var i: usize = 0;
        while (i < layer_count1) : (i += 1) {
            if (!(try tensorAllCloseEq(&core1.layers[i].s_weight, &core2.layers[i].s_weight, abs_tol, rel_tol))) return false;
            if (!(try tensorAllCloseEq(&core1.layers[i].t_weight, &core2.layers[i].t_weight, abs_tol, rel_tol))) return false;
            if (core1.layers[i].clip_min != core2.layers[i].clip_min or core1.layers[i].clip_max != core2.layers[i].clip_max or core1.layers[i].grad_mean != core2.layers[i].grad_mean) return false;
        }

        return true;
    }
};

var g_model_handle_mutex: Thread.Mutex = .{};
var g_model_handle_owner = std.AutoHashMap(u64, usize).init(std.heap.page_allocator);

fn bindModelHandle(self: *const RSF) !u64 {
    const id = self.id;
    if (id == 0) return error.NotInitialized;
    const self_addr: usize = @intFromPtr(self);
    g_model_handle_mutex.lock();
    defer g_model_handle_mutex.unlock();
    if (g_model_handle_owner.get(id)) |owner_addr| {
        if (owner_addr != self_addr) return error.HandleCopied;
    } else {
        try g_model_handle_owner.put(id, self_addr);
    }
    return id;
}

fn shouldDestroyModelHandle(self: *RSF) bool {
    const id = self.id;
    if (id == 0) return false;
    const self_addr: usize = @intFromPtr(self);
    g_model_handle_mutex.lock();
    defer g_model_handle_mutex.unlock();
    if (g_model_handle_owner.get(id)) |owner_addr| {
        if (owner_addr == self_addr) {
            _ = g_model_handle_owner.remove(id);
            maybeShrinkHandleMap(&g_model_handle_owner);
            return true;
        }
        return false;
    }
    return true;
}

fn crcUpdateU32LE(hasher: *std.hash.Crc32, v: u32) void {
    const le = std.mem.nativeToLittle(u32, v);
    hasher.update(std.mem.asBytes(&le));
}

fn crcUpdateU64LE(hasher: *std.hash.Crc32, v: u64) void {
    const le = std.mem.nativeToLittle(u64, v);
    hasher.update(std.mem.asBytes(&le));
}

fn crcUpdateU8(hasher: *std.hash.Crc32, v: u8) void {
    hasher.update(&.{v});
}

fn writeTensorDataVersion4(w: anytype, hasher: *std.hash.Crc32, t: *const Tensor) !void {
    try validateTensor2D(t);
    try ensureFiniteSlice(t.data);
    const rows = t.shape.dims[0];
    const cols = t.shape.dims[1];
    try w.writeInt(u64, 2, .little);
    crcUpdateU64LE(hasher, 2);
    try w.writeInt(u64, @intCast(rows), .little);
    try w.writeInt(u64, @intCast(cols), .little);
    crcUpdateU64LE(hasher, @intCast(rows));
    crcUpdateU64LE(hasher, @intCast(cols));
    for (t.data) |v| {
        const bits = @as(u32, @bitCast(v));
        try w.writeInt(u32, bits, .little);
        crcUpdateU32LE(hasher, bits);
    }
}

fn hashTensorDataVersion4(hasher: *std.hash.Crc32, t: *const Tensor) void {
    crcUpdateU64LE(hasher, 2);
    crcUpdateU64LE(hasher, @intCast(t.shape.dims[0]));
    crcUpdateU64LE(hasher, @intCast(t.shape.dims[1]));
    for (t.data) |v| crcUpdateU32LE(hasher, @as(u32, @bitCast(v)));
}

fn readEncodedBool(r: anytype) !bool {
    const b = try r.readByte();
    return switch (b) {
        0 => false,
        1 => true,
        else => error.BadFileFormat,
    };
}

fn readTensorData(allocator: Allocator, r: anytype, expected_rows: usize, expected_cols: usize) !Tensor {
    if ((try r.readInt(u64, .little)) != 2) return error.BadFileFormat;
    const d0_u64 = try r.readInt(u64, .little);
    const d1_u64 = try r.readInt(u64, .little);
    const expected_rows_u64: u64 = @intCast(expected_rows);
    const expected_cols_u64: u64 = @intCast(expected_cols);
    if (d0_u64 != expected_rows_u64 or d1_u64 != expected_cols_u64) return error.ShapeMismatch;
    const expected = try checkedMul(expected_rows, expected_cols);
    var t = try Tensor.init(allocator, &.{ expected_rows, expected_cols });
    errdefer t.deinit();
    var i: usize = 0;
    while (i < expected) : (i += 1) {
        const bits = try r.readInt(u32, .little);
        t.data[i] = @as(f32, @bitCast(bits));
    }
    return t;
}

const TempFile = struct {
    file: std.fs.File,
    tmp_name: []u8,
};

fn hexEncodeLower(dst: []u8, src: []const u8) []u8 {
    const alphabet = "0123456789abcdef";
    var i: usize = 0;
    while (i < src.len) : (i += 1) {
        const hi: usize = @intCast((src[i] >> 4) & 0x0f);
        const lo: usize = @intCast(src[i] & 0x0f);
        dst[i * 2] = alphabet[hi];
        dst[i * 2 + 1] = alphabet[lo];
    }
    return dst[0 .. src.len * 2];
}

fn createUniqueTempFile(dir: *std.fs.Dir, allocator: Allocator, base_name: []const u8) !TempFile {
    var attempt: usize = 0;
    while (attempt < 64) : (attempt += 1) {
        var rnd: [16]u8 = undefined;
        std.crypto.random.bytes(&rnd);
        var hex_buf: [32]u8 = undefined;
        const hex = hexEncodeLower(&hex_buf, &rnd);
        const tmp_name = try std.fmt.allocPrint(allocator, ".{s}.tmp.{s}", .{ base_name, hex });
        errdefer allocator.free(tmp_name);
        const file = dir.createFile(tmp_name, .{ .exclusive = true, .mode = 0o600 }) catch |e| switch (e) {
            error.PathAlreadyExists => {
                allocator.free(tmp_name);
                continue;
            },
            else => return e,
        };
        return .{ .file = file, .tmp_name = tmp_name };
    }
    return error.TempFileCollision;
}

fn writeSnapshotVersion5ToPath(snapshot: *const SavedModelSnapshot, path: []const u8, allocator: Allocator) !void {
    if (snapshot.num_layers != snapshot.layers.len) return error.InvalidModelState;
    try validateModelConfigValues(snapshot.dim, snapshot.num_layers, snapshot.cfg);
    if (path.len == 0) return error.InvalidPath;

    const parent_path = if (std.fs.path.dirname(path)) |p| p else ".";
    const base_name = std.fs.path.basename(path);
    if (base_name.len == 0) return error.InvalidPath;
    var parent_dir = if (std.fs.path.isAbsolute(parent_path)) try std.fs.openDirAbsolute(parent_path, .{}) else try std.fs.cwd().openDir(parent_path, .{});
    defer parent_dir.close();

    const temp = try createUniqueTempFile(&parent_dir, allocator, base_name);
    defer allocator.free(temp.tmp_name);

    const file = temp.file;
    var file_open = true;
    var tmp_exists = true;
    errdefer {
        if (file_open) file.close();
        if (tmp_exists) parent_dir.deleteFile(temp.tmp_name) catch {};
    }

    var buffered = std.io.bufferedWriter(file.writer());
    const w = buffered.writer();
    var hasher = std.hash.Crc32.init();

    try w.writeAll("RSF0");
    hasher.update("RSF0");
    try w.writeInt(u32, SAVE_VERSION, .little);
    crcUpdateU32LE(&hasher, SAVE_VERSION);
    try w.writeInt(u64, @intCast(snapshot.num_layers), .little);
    crcUpdateU64LE(&hasher, @intCast(snapshot.num_layers));
    try w.writeInt(u64, @intCast(snapshot.dim), .little);
    crcUpdateU64LE(&hasher, @intCast(snapshot.dim));

    const clip_min_bits = @as(u32, @bitCast(snapshot.cfg.clip_min));
    const clip_max_bits = @as(u32, @bitCast(snapshot.cfg.clip_max));
    try w.writeInt(u32, clip_min_bits, .little);
    try w.writeInt(u32, clip_max_bits, .little);
    crcUpdateU32LE(&hasher, clip_min_bits);
    crcUpdateU32LE(&hasher, clip_max_bits);

    const gm_byte: u8 = if (snapshot.cfg.grad_mean) 1 else 0;
    try w.writeByte(gm_byte);
    crcUpdateU8(&hasher, gm_byte);

    try w.writeInt(u64, @intCast(snapshot.cfg.max_dim), .little);
    try w.writeInt(u64, @intCast(snapshot.cfg.max_layers), .little);
    crcUpdateU64LE(&hasher, @intCast(snapshot.cfg.max_dim));
    crcUpdateU64LE(&hasher, @intCast(snapshot.cfg.max_layers));

    var i: usize = 0;
    while (i < snapshot.layers.len) : (i += 1) {
        const layer = &snapshot.layers[i];
        if (layer.clip_min != snapshot.cfg.clip_min or layer.clip_max != snapshot.cfg.clip_max or layer.grad_mean != snapshot.cfg.grad_mean) return error.InvalidConfig;

        const lmin_bits = @as(u32, @bitCast(layer.clip_min));
        const lmax_bits = @as(u32, @bitCast(layer.clip_max));
        try w.writeInt(u32, lmin_bits, .little);
        try w.writeInt(u32, lmax_bits, .little);
        crcUpdateU32LE(&hasher, lmin_bits);
        crcUpdateU32LE(&hasher, lmax_bits);

        const lgm: u8 = if (layer.grad_mean) 1 else 0;
        try w.writeByte(lgm);
        crcUpdateU8(&hasher, lgm);

        try writeTensorDataVersion4(w, &hasher, &layer.s_weight);
        try writeTensorDataVersion4(w, &hasher, &layer.t_weight);
    }

    try w.writeInt(u32, hasher.final(), .little);
    try buffered.flush();
    try file.sync();
    file.close();
    file_open = false;

    try parent_dir.rename(temp.tmp_name, base_name);
    tmp_exists = false;
}

test "RSF forward then inverse returns input within 1e-4 tolerance" {
    const allocator = std.testing.allocator;

    var rsf = try RSFLayer.init(allocator, 32);
    defer rsf.deinit();

    var x1 = try Tensor.randomUniform(allocator, &[_]usize{ 4, 32 }, -1.0, 1.0, 99);
    defer x1.deinit();
    var x2 = try Tensor.randomUniform(allocator, &[_]usize{ 4, 32 }, -1.0, 1.0, 100);
    defer x2.deinit();

    var c1 = try tensorClone(allocator, &x1);
    defer c1.deinit();
    var c2 = try tensorClone(allocator, &x2);
    defer c2.deinit();

    try rsf.forward(&x1, &x2);
    try rsf.inverse(&x1, &x2);

    var idx: usize = 0;
    while (idx < x1.data.len) : (idx += 1) {
        try std.testing.expectApproxEqAbs(x1.data[idx], c1.data[idx], 1e-4);
    }
    idx = 0;
    while (idx < x2.data.len) : (idx += 1) {
        try std.testing.expectApproxEqAbs(x2.data[idx], c2.data[idx], 1e-4);
    }
}

test "RSF with OFTB forward then inverse returns input within 1e-4 tolerance" {
    const allocator = std.testing.allocator;

    const dim: usize = 16;
    const num_layers: usize = 4;

    var rsf = try RSF.init(allocator, dim, num_layers);
    defer rsf.deinit();

    var input = try Tensor.randomUniform(allocator, &[_]usize{ 2, dim * 2 }, -0.5, 0.5, 123);
    defer input.deinit();

    var original = try tensorClone(allocator, &input);
    defer original.deinit();

    try rsf.forwardCPU(&input);
    try rsf.inverse(&input);

    var idx: usize = 0;
    while (idx < input.data.len) : (idx += 1) {
        try std.testing.expectApproxEqAbs(input.data[idx], original.data[idx], 1e-4);
    }
}
