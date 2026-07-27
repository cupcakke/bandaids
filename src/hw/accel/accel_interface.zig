const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const futhark = @import("futhark_bindings.zig");
const core_tensor = @import("../../core/tensor.zig");
const core_memory = @import("../../core/memory.zig");

pub const gpu_enabled: bool = @import("build_options").gpu_acceleration;

pub const AccelError = error{
    FutharkConfigFailed,
    FutharkContextFailed,
    FutharkSyncFailed,
    FutharkArrayNewFailed,
    FutharkValuesFailed,
    FutharkForwardFailed,
    FutharkTrainingStepFailed,
    FutharkScaleWeightsFailed,
    FutharkShapeFailed,
    FutharkComputeLossFailed,
    FutharkBackwardFailed,
    FutharkSFDUpdateFailed,
    CudaHostAllocFailed,
    CudaFreeFailed,
    NullPointer,
    InvalidDimensions,
    AllocationFailed,
    PartialRowCleanup,
};

pub const WeightKind = enum {
    weights_s,
    weights_t,
    velocity_s,
    velocity_t,
};

pub const FutharkContext = struct {
    ctx: ?*futhark.struct_futhark_context,
    cfg: ?*futhark.struct_futhark_context_config,

    const Self = @This();

    pub fn init() AccelError!Self {
        const cfg = futhark.futhark_context_config_new();
        if (cfg == null) return AccelError.FutharkConfigFailed;

        if (gpu_enabled) {
            futhark.gpu_stub.set_device(cfg, "");
            futhark.gpu_stub.set_default_group_size(cfg, 256);
            futhark.gpu_stub.set_default_num_groups(cfg, 128);
            futhark.gpu_stub.set_default_tile_size(cfg, 32);

            // Persist compiled NVRTC kernels across Modal container runs.
            // First run: NVRTC compiles and writes cache (slow).
            // Subsequent runs: reads from cache, skips compilation (fast).
            if (std.posix.getenv("JAIDE_FUTHARK_CACHE")) |cache_path| {
                std.debug.print("[FutharkContext] GPU kernel cache: {s}\n", .{cache_path});
                futhark.gpu_stub.set_cache_file(cfg, @as([*:0]const u8, @ptrCast(cache_path.ptr)));
            } else {
                std.debug.print("[FutharkContext] WARN: JAIDE_FUTHARK_CACHE not set — NVRTC will recompile on every container start\n", .{});
            }
        }

        const ctx = futhark.futhark_context_new(cfg);
        if (ctx == null) {
            futhark.futhark_context_config_free(cfg);
            return AccelError.FutharkContextFailed;
        }

        if (futhark.futhark_context_sync(ctx) != 0) {
            futhark.futhark_context_free(ctx);
            futhark.futhark_context_config_free(cfg);
            return AccelError.FutharkSyncFailed;
        }

        return Self{ .ctx = ctx, .cfg = cfg };
    }

    pub fn deinit(self: *Self) void {
        if (self.ctx) |ctx| {
            const err_str = futhark.futhark_context_get_error(ctx);
            if (err_str != null) {
                _ = futhark.futhark_context_clear_caches(ctx);
            }
            futhark.futhark_context_free(ctx);
            self.ctx = null;
        }
        if (self.cfg) |cfg| {
            futhark.futhark_context_config_free(cfg);
            self.cfg = null;
        }
    }

    pub fn sync(self: *Self) AccelError!void {
        if (self.ctx == null) return AccelError.NullPointer;
        if (futhark.futhark_context_sync(self.ctx) != 0) {
            return AccelError.FutharkSyncFailed;
        }
    }

    pub fn getDataPointer(self: *Self, array: *FutharkArray2DF16) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;

        const raw_ptr = futhark.futhark_values_raw_f16_2d(self.ctx, array.arr);
        if (raw_ptr == null) {
            return AccelError.NullPointer;
        }

        return raw_ptr.?;
    }

    pub fn getDataPointer1D(self: *Self, array: *FutharkArray1DF16) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;

        const raw_ptr = futhark.futhark_values_raw_f16_1d(self.ctx, array.arr);
        if (raw_ptr == null) {
            return AccelError.NullPointer;
        }

        return raw_ptr.?;
    }
};

fn get1DDevicePtr(ctx: *FutharkContext, array: *FutharkArray1DF16) AccelError!*anyopaque {
    return ctx.getDataPointer1D(array);
}

pub const PinnedMemory = struct {
    ptr: ?*anyopaque,
    size: usize,
    fallback_slice: ?[]align(64) u8,

    const Self = @This();

    pub fn alloc(size: usize) AccelError!Self {
        if (size == 0) {
            return Self{ .ptr = null, .size = 0, .fallback_slice = null };
        }

        if (gpu_enabled) {
            var ptr: ?*anyopaque = null;
            const err = cuda.cudaHostAlloc(&ptr, size, cuda.cudaHostAllocDefault);
            if (err != cuda.cudaSuccess) {
                return AccelError.CudaHostAllocFailed;
            }
            return Self{
                .ptr = ptr,
                .size = size,
                .fallback_slice = null,
            };
        }

        const slice = std.heap.page_allocator.alignedAlloc(u8, 64, size) catch return AccelError.CudaHostAllocFailed;
        return Self{
            .ptr = @ptrCast(slice.ptr),
            .size = size,
            .fallback_slice = slice,
        };
    }

    pub fn free(self: *Self) void {
        if (self.fallback_slice) |slice| {
            std.heap.page_allocator.free(slice);
            self.fallback_slice = null;
            self.ptr = null;
            self.size = 0;
            return;
        }
        if (self.ptr) |p| {
            if (gpu_enabled) {
                _ = cuda.cudaFreeHost(p);
            }
            self.ptr = null;
            self.size = 0;
        }
    }

    pub fn asSlice(self: *Self, comptime T: type) ?[]T {
        if (self.ptr == null) return null;
        const count = self.size / @sizeOf(T);
        const aligned: [*]T = @ptrCast(@alignCast(self.ptr.?));
        return aligned[0..count];
    }
};

pub const FutharkArray1DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_1d,
    len: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, length: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (length == 0) return AccelError.InvalidDimensions;
        if (flat_data.len != length) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_1d(
            ctx.ctx,
            @ptrCast(flat_data.ptr),
            @intCast(length),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .len = length };
    }

    pub fn newZeros(ctx: *FutharkContext, length: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (length == 0) return AccelError.InvalidDimensions;

        const zeros = allocator.alloc(f16, length) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_1d(
            ctx.ctx,
            @ptrCast(zeros.ptr),
            @intCast(length),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .len = length };
    }

    pub fn values1D(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;

        const buf = allocator.alloc(f16, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);

        const result = futhark.futhark_values_f16_1d(ctx.ctx, self.arr, @ptrCast(buf.ptr));
        if (result != 0) {
            allocator.free(buf);
            return AccelError.FutharkValuesFailed;
        }

        const sync_result = futhark.futhark_context_sync(ctx.ctx);
        if (sync_result != 0) {
            allocator.free(buf);
            return AccelError.FutharkSyncFailed;
        }

        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const FutharkArray2DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn new(ctx: *FutharkContext, data: []const []const f16) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;

        const rows = data.len;
        const cols = data[0].len;
        if (cols == 0) return AccelError.InvalidDimensions;

        for (data) |row| {
            if (row.len != cols) return AccelError.InvalidDimensions;
        }

        const total = rows * cols;
        var flat_data = std.ArrayList(f16).init(std.heap.page_allocator);
        defer flat_data.deinit();

        flat_data.ensureTotalCapacity(total) catch return AccelError.AllocationFailed;

        for (data) |row| {
            flat_data.appendSlice(row) catch return AccelError.AllocationFailed;
        }

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.items.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (flat_data.len != rows * cols) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;

        const total = rows * cols;
        const zeros = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(zeros.ptr),
            @intCast(rows),
            @intCast(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn values(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![][]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;

        const rows = self.rows;
        const cols = self.cols;

        if (rows == 0 or cols == 0) {
            return allocator.alloc([]f16, 0) catch return AccelError.AllocationFailed;
        }

        const flat = allocator.alloc(f16, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(flat);

        if (futhark.futhark_values_f16_2d(ctx.ctx, self.arr, @ptrCast(flat.ptr)) != 0) {
            return AccelError.FutharkValuesFailed;
        }

        const result = allocator.alloc([]f16, rows) catch return AccelError.AllocationFailed;
        var i: usize = 0;
        while (i < rows) : (i += 1) {
            result[i] = allocator.alloc(f16, cols) catch {
                var j: usize = 0;
                while (j < i) : (j += 1) {
                    allocator.free(result[j]);
                }
                allocator.free(result);
                return AccelError.PartialRowCleanup;
            };
            @memcpy(result[i], flat[i * cols .. (i + 1) * cols]);
        }

        return result;
    }
};

pub const FutharkArray3DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_3d,
    dim0: usize,
    dim1: usize,
    dim2: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, flat: []const f16, d0: usize, d1: usize, d2: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        if (flat.len != d0 * d1 * d2) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_3d(
            ctx.ctx,
            @ptrCast(flat.ptr),
            @intCast(d0),
            @intCast(d1),
            @intCast(d2),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .dim0 = d0, .dim1 = d1, .dim2 = d2 };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_3d(ctx.ctx, arr);
            self.arr = null;
            self.dim0 = 0;
            self.dim1 = 0;
            self.dim2 = 0;
        }
    }
};

pub const FutharkArray2DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = tensor.shape.dims[0];
        const cols = tensor.shape.dims[1];
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, tensor.data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, data: []const f32, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, data.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const zeros = allocator.alloc(f32, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, zeros.ptr, @intCast(rows), @intCast(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            futhark.futhark_free_f32_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{ self.rows, self.cols };
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_2d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        return tensor;
    }
};

pub const FutharkArray1DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_1d,
    len: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 1) return AccelError.InvalidDimensions;
        const n = tensor.shape.dims[0];
        if (n == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, tensor.data.ptr, @intCast(n));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = n };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            futhark.futhark_free_f32_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{self.len};
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        return tensor;
    }

    pub fn newFromSlice(ctx: *FutharkContext, data: []const f32) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, data.ptr, @intCast(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn valuesSlice(self: *FutharkArray1DF32, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f32 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f32, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }
};

pub const RSFLayer = struct {
    weights_s: FutharkArray2DF16,
    weights_t: FutharkArray2DF16,
    velocity_s: FutharkArray2DF16,
    velocity_t: FutharkArray2DF16,

    pub fn free(self: *RSFLayer, ctx: *FutharkContext) void {
        self.velocity_t.free(ctx);
        self.velocity_s.free(ctx);
        self.weights_t.free(ctx);
        self.weights_s.free(ctx);
    }
};

pub const TrainingStepResult = struct {
    loss: f16,
    input_gradient: FutharkArray3DF16,
};

pub const RSFAccelerator = struct {
    ctx: FutharkContext,
    layers: []RSFLayer,
    layers_owner: std.mem.Allocator,
    model_dim: usize,
    num_layers: usize,
    clip_min: f16,
    clip_max: f16,
    initialized: bool,

    const Self = @This();

    pub fn init(model_dim: usize) AccelError!Self {
        return initMultiLayer(model_dim, 1, std.heap.page_allocator);
    }

    pub fn initMultiLayer(model_dim: usize, num_layers: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (model_dim == 0) return AccelError.InvalidDimensions;
        if (model_dim % 2 != 0) return AccelError.InvalidDimensions;
        if (num_layers == 0) return AccelError.InvalidDimensions;
        const half: usize = model_dim / 2;

        var ctx = try FutharkContext.init();
        errdefer ctx.deinit();

        const base_seed: u64 = 0x4A41494445204E4F;
        const init_stddev: f32 = 0.25 / @sqrt(@as(f32, @floatFromInt(half)));

        var layers = allocator.alloc(RSFLayer, num_layers) catch return AccelError.AllocationFailed;
        errdefer allocator.free(layers);

        const total: usize = half * (half + 1);
        const ws_buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(ws_buf);
        const wt_buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(wt_buf);

        var layers_built: usize = 0;
        errdefer {
            var idx: usize = 0;
            while (idx < layers_built) : (idx += 1) {
                layers[idx].free(&ctx);
            }
        }

        var layer_idx: usize = 0;
        while (layer_idx < num_layers) : (layer_idx += 1) {
            const layer_seed: u64 = base_seed +% (@as(u64, @intCast(layer_idx)) *% 0x9E3779B97F4A7C15);
            var rng = std.Random.DefaultPrng.init(layer_seed);
            const rnd = rng.random();
            for (ws_buf) |*v| {
                const r = rnd.floatNorm(f32) * init_stddev;
                v.* = @floatCast(r);
            }
            for (wt_buf) |*v| {
                const r = rnd.floatNorm(f32) * init_stddev;
                v.* = @floatCast(r);
            }
            {
                var d: usize = 0;
                while (d < half) : (d += 1) {
                    ws_buf[d * (half + 1) + half] = @as(f16, 0.0);
                    wt_buf[d * (half + 1) + half] = @as(f16, 0.0);
                }
            }

            const weights_s = try FutharkArray2DF16.newFromFlat(&ctx, ws_buf, half, half + 1);
            const weights_t = try FutharkArray2DF16.newFromFlat(&ctx, wt_buf, half, half + 1);
            const velocity_s = try FutharkArray2DF16.newZeros(&ctx, half, half + 1, allocator);
            const velocity_t = try FutharkArray2DF16.newZeros(&ctx, half, half + 1, allocator);

            layers[layer_idx] = .{
                .weights_s = weights_s,
                .weights_t = weights_t,
                .velocity_s = velocity_s,
                .velocity_t = velocity_t,
            };
            layers_built += 1;
        }

        return Self{
            .ctx = ctx,
            .layers = layers,
            .layers_owner = allocator,
            .model_dim = model_dim,
            .num_layers = num_layers,
            .clip_min = @as(f16, -5.0),
            .clip_max = @as(f16, 5.0),
            .initialized = true,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;

        var i: usize = self.layers.len;
        while (i > 0) {
            i -= 1;
            self.layers[i].free(&self.ctx);
        }
        self.layers_owner.free(self.layers);
        self.ctx.deinit();
        self.initialized = false;
    }

    pub fn forward(self: *Self, input: *FutharkArray2DF16) AccelError!FutharkArray2DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (input.arr == null) return AccelError.NullPointer;
        if (self.layers.len == 0) return AccelError.NullPointer;

        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        var current_arr: ?*futhark.struct_futhark_f16_2d = input.arr;
        const rows = input.rows;
        const cols = input.cols;

        var li: usize = 0;
        while (li < self.layers.len) : (li += 1) {
            const layer = &self.layers[li];
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;

            var next_arr: ?*futhark.struct_futhark_f16_2d = null;
            const result = futhark.futhark_entry_rsf_forward(
                self.ctx.ctx,
                &next_arr,
                current_arr,
                layer.weights_s.arr,
                layer.weights_t.arr,
                clip_min_bits,
                clip_max_bits,
            );
            if (result != 0) {
                if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
                return AccelError.FutharkForwardFailed;
            }
            if (next_arr == null) {
                if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
                return AccelError.NullPointer;
            }

            if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
            current_arr = next_arr;
        }

        return FutharkArray2DF16{ .arr = current_arr, .rows = rows, .cols = cols };
    }

    pub fn trainingStep(
        self: *Self,
        inputs: *FutharkArray3DF16,
        targets: *FutharkArray3DF16,
        sequence_lengths: []const usize,
        learning_rate: f16,
        momentum: f16,
    ) AccelError!TrainingStepResult {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (inputs.arr == null or targets.arr == null) return AccelError.NullPointer;
        if (self.layers.len == 0) return AccelError.NullPointer;
        if (inputs.dim0 != targets.dim0 or inputs.dim1 != targets.dim1 or inputs.dim2 != targets.dim2) return AccelError.InvalidDimensions;
        if (inputs.dim2 != self.model_dim or sequence_lengths.len != inputs.dim0) return AccelError.InvalidDimensions;

        const lengths_i64 = std.heap.page_allocator.alloc(i64, sequence_lengths.len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(lengths_i64);
        for (sequence_lengths, 0..) |length, index| {
            if (length > inputs.dim1) return AccelError.InvalidDimensions;
            lengths_i64[index] = @intCast(length);
        }
        var lengths_array = try FutharkArray1DI64.newFromSlice(&self.ctx, lengths_i64);
        defer lengths_array.free(&self.ctx);

        const lr_bits: u16 = @bitCast(learning_rate);
        const momentum_bits: u16 = @bitCast(momentum);
        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);
        const n_layers = self.layers.len;

        var current_act: ?*futhark.struct_futhark_f16_3d = inputs.arr;
        var current_act_owned = false;
        var grad_out: ?*futhark.struct_futhark_f16_3d = null;

        errdefer {
            if (grad_out) |gradient| {
                _ = futhark.futhark_free_f16_3d(self.ctx.ctx, gradient);
                grad_out = null;
            }
            if (current_act_owned) {
                if (current_act) |activation| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, activation);
                    current_act = null;
                }
                current_act_owned = false;
            }
        }

        var layer_index: usize = 0;
        while (layer_index < n_layers) : (layer_index += 1) {
            const layer = &self.layers[layer_index];
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;

            var rsf_output: ?*futhark.struct_futhark_f16_3d = null;
            const forward_result = futhark.futhark_entry_batch_forward(
                self.ctx.ctx,
                &rsf_output,
                current_act,
                layer.weights_s.arr,
                layer.weights_t.arr,
                clip_min_bits,
                clip_max_bits,
            );
            if (forward_result != 0 or rsf_output == null) {
                const error_string = futhark.futhark_context_get_error(self.ctx.ctx);
                if (error_string) |message| std.debug.print("[Futhark batch_forward L{d} error] {s}\n", .{ layer_index, std.mem.span(message) });
                if (rsf_output) |output| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, output);
                return AccelError.FutharkForwardFailed;
            }

            if (current_act_owned) {
                if (current_act) |activation| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, activation);
            }
            current_act = rsf_output;
            current_act_owned = true;

            var oftb_output: ?*futhark.struct_futhark_f16_3d = null;
            const oftb_result = futhark.futhark_entry_batch_oftb_forward(self.ctx.ctx, &oftb_output, current_act);
            if (current_act) |activation| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, activation);
            current_act = null;
            current_act_owned = false;
            if (oftb_result != 0 or oftb_output == null) {
                if (oftb_output) |output| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, output);
                return AccelError.FutharkForwardFailed;
            }
            current_act = oftb_output;
            current_act_owned = true;
        }

        const final_activation = current_act orelse return AccelError.NullPointer;
        var loss_bits: u16 = 0;
        const loss_result = futhark.futhark_entry_batch_compute_loss_masked(
            self.ctx.ctx,
            &loss_bits,
            final_activation,
            targets.arr,
            lengths_array.arr,
        );
        if (loss_result != 0) {
            const error_string = futhark.futhark_context_get_error(self.ctx.ctx);
            if (error_string) |message| std.debug.print("[Futhark batch_compute_loss_masked error] {s}\n", .{std.mem.span(message)});
            return AccelError.FutharkComputeLossFailed;
        }
        const loss: f16 = @bitCast(loss_bits);

        const gradient_seed_result = futhark.futhark_entry_compute_initial_grad_l2_masked(
            self.ctx.ctx,
            &grad_out,
            final_activation,
            targets.arr,
            lengths_array.arr,
        );
        if (gradient_seed_result != 0 or grad_out == null) {
            const error_string = futhark.futhark_context_get_error(self.ctx.ctx);
            if (error_string) |message| std.debug.print("[Futhark compute_initial_grad_l2_masked error] {s}\n", .{std.mem.span(message)});
            return AccelError.FutharkBackwardFailed;
        }
        try self.ctx.sync();

        var backward_layer = n_layers;
        while (backward_layer > 0) {
            backward_layer -= 1;
            const layer = &self.layers[backward_layer];
            if (current_act == null or grad_out == null) return AccelError.NullPointer;
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;

            var rsf_output_reconstructed: ?*futhark.struct_futhark_f16_3d = null;
            const oftb_inverse_result = futhark.futhark_entry_batch_oftb_backward(
                self.ctx.ctx,
                &rsf_output_reconstructed,
                current_act,
            );
            if (oftb_inverse_result != 0 or rsf_output_reconstructed == null) {
                std.debug.print("[Futhark batch_oftb_backward L{d} error] rc={d}\n", .{ backward_layer, oftb_inverse_result });
                if (rsf_output_reconstructed) |output| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, output);
                return AccelError.FutharkBackwardFailed;
            }

            if (current_act) |activation| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, activation);
            current_act = null;
            current_act_owned = false;

            var layer_input_reconstructed: ?*futhark.struct_futhark_f16_3d = null;
            const rsf_inverse_result = futhark.futhark_entry_batch_rsf_inverse(
                self.ctx.ctx,
                &layer_input_reconstructed,
                rsf_output_reconstructed,
                layer.weights_s.arr,
                layer.weights_t.arr,
                clip_min_bits,
                clip_max_bits,
            );
            if (rsf_output_reconstructed) |output| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, output);
            rsf_output_reconstructed = null;
            if (rsf_inverse_result != 0 or layer_input_reconstructed == null) {
                const error_string = futhark.futhark_context_get_error(self.ctx.ctx);
                if (error_string) |message| std.debug.print("[Futhark batch_rsf_inverse L{d} error] {s}\n", .{ backward_layer, std.mem.span(message) });
                if (layer_input_reconstructed) |input| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, input);
                return AccelError.FutharkBackwardFailed;
            }

            var transformed_gradient: ?*futhark.struct_futhark_f16_3d = null;
            const oftb_backward_result = futhark.futhark_entry_batch_oftb_backward(
                self.ctx.ctx,
                &transformed_gradient,
                grad_out,
            );
            if (grad_out) |gradient| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, gradient);
            grad_out = null;
            if (oftb_backward_result != 0 or transformed_gradient == null) {
                std.debug.print("[Futhark oftb_backward grad L{d} error] rc={d}\n", .{ backward_layer, oftb_backward_result });
                if (transformed_gradient) |gradient| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, gradient);
                if (layer_input_reconstructed) |input| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, input);
                return AccelError.FutharkBackwardFailed;
            }

            var gradient_tuple: ?*futhark.struct_futhark_opaque_tup3_grad_full = null;
            const gradient_result = futhark.futhark_entry_batch_gradients_full(
                self.ctx.ctx,
                &gradient_tuple,
                layer_input_reconstructed,
                transformed_gradient,
                layer.weights_s.arr,
                layer.weights_t.arr,
                clip_min_bits,
                clip_max_bits,
            );
            if (transformed_gradient) |gradient| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, gradient);
            transformed_gradient = null;
            if (gradient_result != 0 or gradient_tuple == null) {
                const error_string = futhark.futhark_context_get_error(self.ctx.ctx);
                if (error_string) |message| std.debug.print("[Futhark batch_gradients_full L{d} error] {s}\n", .{ backward_layer, std.mem.span(message) });
                if (layer_input_reconstructed) |input| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, input);
                return AccelError.FutharkBackwardFailed;
            }
            var gradient_weights_s: ?*futhark.struct_futhark_f16_2d = null;
            var gradient_weights_t: ?*futhark.struct_futhark_f16_2d = null;
            var gradient_input: ?*futhark.struct_futhark_f16_3d = null;
            const projection_s = futhark.futhark_project_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16_0(self.ctx.ctx, &gradient_weights_s, gradient_tuple);
            const projection_t = futhark.futhark_project_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16_1(self.ctx.ctx, &gradient_weights_t, gradient_tuple);
            const projection_input = futhark.futhark_project_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16_2(self.ctx.ctx, &gradient_input, gradient_tuple);
            _ = futhark.futhark_free_opaque_tup3_arr2d_f16_arr2d_f16_arr3d_f16(self.ctx.ctx, gradient_tuple);
            if (projection_s != 0 or projection_t != 0 or projection_input != 0 or gradient_weights_s == null or gradient_weights_t == null or gradient_input == null) {
                if (gradient_weights_s) |gradient| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, gradient);
                if (gradient_weights_t) |gradient| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, gradient);
                if (gradient_input) |gradient| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, gradient);
                if (layer_input_reconstructed) |input| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, input);
                return AccelError.FutharkBackwardFailed;
            }

            errdefer {
                if (gradient_weights_s) |gradient| {
                    _ = futhark.futhark_free_f16_2d(self.ctx.ctx, gradient);
                    gradient_weights_s = null;
                }
                if (gradient_weights_t) |gradient| {
                    _ = futhark.futhark_free_f16_2d(self.ctx.ctx, gradient);
                    gradient_weights_t = null;
                }
                if (gradient_input) |gradient| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, gradient);
                    gradient_input = null;
                }
                if (layer_input_reconstructed) |input| {
                    _ = futhark.futhark_free_f16_3d(self.ctx.ctx, input);
                    layer_input_reconstructed = null;
                }
            }

            try sfdUpdateMat(self, &layer.weights_s, &layer.velocity_s, gradient_weights_s, lr_bits, momentum_bits);
            try sfdUpdateMat(self, &layer.weights_t, &layer.velocity_t, gradient_weights_t, lr_bits, momentum_bits);
            _ = futhark.futhark_free_f16_2d(self.ctx.ctx, gradient_weights_s);
            gradient_weights_s = null;
            _ = futhark.futhark_free_f16_2d(self.ctx.ctx, gradient_weights_t);
            gradient_weights_t = null;

            grad_out = gradient_input;
            gradient_input = null;
            current_act = layer_input_reconstructed;
            current_act_owned = true;
            layer_input_reconstructed = null;
        }

        if (current_act_owned) {
            if (current_act) |activation| {
                _ = futhark.futhark_free_f16_3d(self.ctx.ctx, activation);
                current_act = null;
            }
            current_act_owned = false;
        }

        const input_gradient_pointer = grad_out orelse return AccelError.FutharkBackwardFailed;
        grad_out = null;
        return TrainingStepResult{
            .loss = loss,
            .input_gradient = FutharkArray3DF16{
                .arr = input_gradient_pointer,
                .dim0 = inputs.dim0,
                .dim1 = inputs.dim1,
                .dim2 = inputs.dim2,
            },
        };
    }

    fn sfdUpdateMat(
        self: *Self,
        weights: *FutharkArray2DF16,
        velocity: *FutharkArray2DF16,
        gradients: ?*futhark.struct_futhark_f16_2d,
        lr_bits: u16,
        momentum_bits: u16,
    ) AccelError!void {
        if (weights.arr == null or velocity.arr == null) return AccelError.NullPointer;
        var out_tup: ?*futhark.struct_futhark_opaque_tup2_2d = null;
        const rc = futhark.futhark_entry_sfd_update_mat(
            self.ctx.ctx,
            &out_tup,
            weights.arr,
            gradients,
            lr_bits,
            momentum_bits,
            velocity.arr,
        );
        if (rc != 0 or out_tup == null) {
            const err_str = futhark.futhark_context_get_error(self.ctx.ctx);
            if (err_str) |s| std.debug.print("[Futhark sfd_update_mat error] {s}\n", .{std.mem.span(s)});
            return AccelError.FutharkSFDUpdateFailed;
        }
        var new_w: ?*futhark.struct_futhark_f16_2d = null;
        var new_v: ?*futhark.struct_futhark_f16_2d = null;
        _ = futhark.futhark_project_opaque_tup2_arr2d_f16_arr2d_f16_0(self.ctx.ctx, &new_w, out_tup);
        _ = futhark.futhark_project_opaque_tup2_arr2d_f16_arr2d_f16_1(self.ctx.ctx, &new_v, out_tup);
        _ = futhark.futhark_free_opaque_tup2_arr2d_f16_arr2d_f16(self.ctx.ctx, out_tup);
        if (new_w == null or new_v == null) return AccelError.FutharkSFDUpdateFailed;
        const old_w = weights.arr;
        const old_v = velocity.arr;
        weights.arr = new_w;
        velocity.arr = new_v;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_w);
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_v);
    }

    pub fn scaleWeights(self: *Self, scale_factor: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (scale_factor == @as(f16, 0.0)) return AccelError.InvalidDimensions;

        const scale_bits: u16 = @bitCast(scale_factor);
        for (self.layers) |*layer| {
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;

            var new_ws: ?*futhark.struct_futhark_f16_2d = null;
            const result_s = futhark.futhark_entry_scale_weights_inplace(
                self.ctx.ctx,
                &new_ws,
                layer.weights_s.arr,
                scale_bits,
            );
            if (result_s != 0) return AccelError.FutharkScaleWeightsFailed;
            if (new_ws != null) {
                const old = layer.weights_s.arr;
                layer.weights_s.arr = new_ws;
                _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
            }

            var new_wt: ?*futhark.struct_futhark_f16_2d = null;
            const result_t = futhark.futhark_entry_scale_weights_inplace(
                self.ctx.ctx,
                &new_wt,
                layer.weights_t.arr,
                scale_bits,
            );
            if (result_t != 0) return AccelError.FutharkScaleWeightsFailed;
            if (new_wt != null) {
                const old = layer.weights_t.arr;
                layer.weights_t.arr = new_wt;
                _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
            }
        }
    }

    pub fn sync(self: *Self) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        return self.ctx.sync();
    }

    pub fn numLayers(self: *const Self) usize {
        return self.num_layers;
    }

    pub fn layerPtr(self: *Self, layer_idx: usize) AccelError!*RSFLayer {
        if (!self.initialized) return AccelError.NullPointer;
        if (layer_idx >= self.layers.len) return AccelError.InvalidDimensions;
        return &self.layers[layer_idx];
    }

    pub fn setLayerWeightsS(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.weights_s.free(&self.ctx);
        layer.weights_s = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setLayerWeightsT(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.weights_t.free(&self.ctx);
        layer.weights_t = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setLayerVelocityS(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.velocity_s.free(&self.ctx);
        layer.velocity_s = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn setLayerVelocityT(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const layer = try self.layerPtr(layer_idx);
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != rows * cols) return AccelError.InvalidDimensions;
        layer.velocity_t.free(&self.ctx);
        layer.velocity_t = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
    }

    pub fn readLayerWeightsFlat(self: *Self, layer_idx: usize, kind: WeightKind, allocator: std.mem.Allocator) AccelError![]f16 {
        const layer = try self.layerPtr(layer_idx);
        return switch (kind) {
            .weights_s => readMatFlat(self, &layer.weights_s, allocator),
            .weights_t => readMatFlat(self, &layer.weights_t, allocator),
            .velocity_s => readMatFlat(self, &layer.velocity_s, allocator),
            .velocity_t => readMatFlat(self, &layer.velocity_t, allocator),
        };
    }

    pub fn getLayerDevicePtr(
        self: *Self,
        layer_idx: usize,
        kind: WeightKind,
    ) AccelError!struct { ptr: *anyopaque, count: usize } {
        if (!self.initialized) return AccelError.NullPointer;
        const layer = try self.layerPtr(layer_idx);
        const half = self.model_dim / 2;
        return switch (kind) {
            .weights_s => .{ .ptr = try self.ctx.getDataPointer(&layer.weights_s), .count = half * (half + 1) },
            .weights_t => .{ .ptr = try self.ctx.getDataPointer(&layer.weights_t), .count = half * (half + 1) },
            .velocity_s => .{ .ptr = try self.ctx.getDataPointer(&layer.velocity_s), .count = half * (half + 1) },
            .velocity_t => .{ .ptr = try self.ctx.getDataPointer(&layer.velocity_t), .count = half * (half + 1) },
        };
    }

    pub fn scaleLayerMatrix(self: *Self, layer_idx: usize, kind: WeightKind, scale_factor: f16) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        const scale_f32: f32 = @floatCast(scale_factor);
        if (!std.math.isFinite(scale_f32) or scale_f32 < 0.0) return AccelError.InvalidDimensions;
        const layer = try self.layerPtr(layer_idx);
        const matrix: *FutharkArray2DF16 = switch (kind) {
            .weights_s => &layer.weights_s,
            .weights_t => &layer.weights_t,
            .velocity_s => &layer.velocity_s,
            .velocity_t => &layer.velocity_t,
        };
        if (matrix.arr == null) return AccelError.NullPointer;
        var scaled: ?*futhark.struct_futhark_f16_2d = null;
        const result = futhark.futhark_entry_scale_matrix_f16(
            self.ctx.ctx,
            &scaled,
            matrix.arr,
            @bitCast(scale_factor),
        );
        if (result != 0 or scaled == null) {
            if (scaled) |value| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, value);
            return AccelError.FutharkScaleWeightsFailed;
        }
        const old = matrix.arr;
        matrix.arr = scaled;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
    }

    fn readMatFlat(self: *Self, mat: *FutharkArray2DF16, allocator: std.mem.Allocator) AccelError![]f16 {
        const rows = try mat.values(&self.ctx, allocator);
        defer {
            for (rows) |row| allocator.free(row);
            allocator.free(rows);
        }
        const half = self.model_dim / 2;
        const cols = half + 1;
        if (rows.len != half) return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, half, cols) catch return AccelError.AllocationFailed;
        var flat = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        var idx: usize = 0;
        for (rows) |row| {
            if (row.len != cols) return AccelError.InvalidDimensions;
            for (row) |v| {
                flat[idx] = v;
                idx += 1;
            }
        }
        return flat;
    }

    pub fn setClipRange(self: *Self, clip_min_val: f16, clip_max_val: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        const minimum: f32 = @floatCast(clip_min_val);
        const maximum: f32 = @floatCast(clip_max_val);
        if (!std.math.isFinite(minimum) or !std.math.isFinite(maximum)) return AccelError.InvalidDimensions;
        if (minimum >= maximum or minimum < -20.0 or maximum > 20.0) return AccelError.InvalidDimensions;
        if (gpu_enabled and (minimum != -5.0 or maximum != 5.0)) return AccelError.InvalidDimensions;
        self.clip_min = clip_min_val;
        self.clip_max = clip_max_val;
    }

    pub fn forwardFromTensor(self: *Self, input: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (!self.initialized) return AccelError.NullPointer;
        if (input.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = input.shape.dims[0];
        const cols = input.shape.dims[1];
        const f16_data = allocator.alloc(f16, rows * cols) catch return AccelError.AllocationFailed;
        defer allocator.free(f16_data);
        {
            var i: usize = 0;
            while (i < input.data.len) : (i += 1) {
                const v = input.data[i];
                f16_data[i] = @floatCast(v);
            }
        }
        var f16_input = try FutharkArray2DF16.newFromFlat(&self.ctx, f16_data, rows, cols);
        defer f16_input.free(&self.ctx);
        var output = try self.forward(&f16_input);
        defer output.free(&self.ctx);
        const shape = [_]usize{ output.rows, output.cols };
        var result = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        const out_f16 = allocator.alloc(f16, output.rows * output.cols) catch {
            result.deinit();
            return AccelError.AllocationFailed;
        };
        defer allocator.free(out_f16);
        if (futhark.futhark_values_f16_2d(self.ctx.ctx, output.arr, @ptrCast(out_f16.ptr)) != 0) {
            result.deinit();
            return AccelError.FutharkValuesFailed;
        }
        {
            var i: usize = 0;
            while (i < out_f16.len) : (i += 1) {
                const v = out_f16[i];
                result.data[i] = @floatCast(v);
            }
        }
        return result;
    }
};

pub const GPUOps = struct {
    ctx: FutharkContext,

    const Self = @This();

    pub fn init() AccelError!Self {
        return Self{ .ctx = try FutharkContext.init() };
    }

    pub fn deinit(self: *Self) void {
        self.ctx.deinit();
    }

    pub fn matmul(self: *Self, a: *const core_tensor.Tensor, b: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        var fa = try FutharkArray2DF32.fromTensor(&self.ctx, a);
        defer fa.free(&self.ctx);
        var fb = try FutharkArray2DF32.fromTensor(&self.ctx, b);
        defer fb.free(&self.ctx);

        var out_arr: ?*futhark.struct_futhark_f32_2d = null;
        if (futhark.futhark_entry_matmul(self.ctx.ctx, &out_arr, fa.arr, fb.arr) != 0) {
            return AccelError.FutharkForwardFailed;
        }
        if (out_arr == null) return AccelError.NullPointer;

        var result = FutharkArray2DF32{ .arr = out_arr, .rows = a.shape.dims[0], .cols = b.shape.dims[1] };
        defer result.free(&self.ctx);
        return result.toTensor(&self.ctx, allocator);
    }
};

pub const FutharkArray1DI64 = struct {
    arr: ?*futhark.struct_futhark_i64_1d,
    len: usize,

    const Self = @This();

    pub fn newFromSlice(ctx: *FutharkContext, data: []const i64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_i64_1d(ctx.ctx, data.ptr, @intCast(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn valuesSlice(self: *FutharkArray1DI64, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]i64 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(i64, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_i64_1d(ctx.ctx, self.arr, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            futhark.futhark_free_i64_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const EmbeddingAccelerator = struct {
    ctx: *FutharkContext,
    weight: FutharkArray2DF16,
    grad_weight: FutharkArray2DF16,
    vocab_size: usize,
    dim: usize,
    initialized: bool,

    const Self = @This();

    pub fn init(ctx: *FutharkContext, vocab_size: usize, dim: usize, seed: u64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (vocab_size == 0 or dim == 0) return AccelError.InvalidDimensions;

        var rng = std.Random.DefaultPrng.init(seed);
        const rnd = rng.random();
        const total = vocab_size * dim;
        const weight_data = std.heap.page_allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(weight_data);
        for (weight_data) |*v| {
            v.* = @floatCast((rnd.float(f32) - 0.5) * 0.02);
        }

        var weight = try FutharkArray2DF16.newFromFlat(ctx, weight_data, vocab_size, dim);
        errdefer weight.free(ctx);
        const grad_weight = try FutharkArray2DF16.newZeros(ctx, vocab_size, dim, std.heap.page_allocator);

        return Self{
            .ctx = ctx,
            .weight = weight,
            .grad_weight = grad_weight,
            .vocab_size = vocab_size,
            .dim = dim,
            .initialized = true,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;
        self.grad_weight.free(self.ctx);
        self.weight.free(self.ctx);
        self.initialized = false;
    }

    pub fn forwardPadded(
        self: *Self,
        tokens: []const u32,
        sequence_lengths: []const usize,
        sequence_length: usize,
    ) AccelError!FutharkArray3DF16 {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        if (sequence_lengths.len == 0 or sequence_length == 0) return AccelError.InvalidDimensions;
        const expected_tokens = std.math.mul(usize, sequence_lengths.len, sequence_length) catch return AccelError.InvalidDimensions;
        if (tokens.len != expected_tokens) return AccelError.InvalidDimensions;

        const token_i64s = std.heap.page_allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(token_i64s);
        for (tokens, 0..) |token, index| {
            if (@as(usize, token) >= self.vocab_size) return AccelError.InvalidDimensions;
            token_i64s[index] = @intCast(token);
        }

        const lengths_i64 = std.heap.page_allocator.alloc(i64, sequence_lengths.len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(lengths_i64);
        for (sequence_lengths, 0..) |length, index| {
            if (length > sequence_length) return AccelError.InvalidDimensions;
            lengths_i64[index] = @intCast(length);
        }

        const positions_i64 = std.heap.page_allocator.alloc(i64, sequence_length) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(positions_i64);
        for (positions_i64, 0..) |*position, index| {
            position.* = @intCast(index);
        }

        var token_array = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer token_array.free(self.ctx);
        var length_array = try FutharkArray1DI64.newFromSlice(self.ctx, lengths_i64);
        defer length_array.free(self.ctx);
        var position_array = try FutharkArray1DI64.newFromSlice(self.ctx, positions_i64);
        defer position_array.free(self.ctx);

        var output: ?*futhark.struct_futhark_f16_3d = null;
        const result = futhark.futhark_entry_embedding_forward_padded(
            self.ctx.ctx,
            &output,
            token_array.arr,
            length_array.arr,
            position_array.arr,
            self.weight.arr,
        );
        if (result != 0 or output == null) {
            if (output) |value| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, value);
            return AccelError.FutharkForwardFailed;
        }
        return FutharkArray3DF16{
            .arr = output,
            .dim0 = sequence_lengths.len,
            .dim1 = sequence_length,
            .dim2 = self.dim,
        };
    }

    pub fn backwardPaddedAccumulate(
        self: *Self,
        tokens: []const u32,
        sequence_lengths: []const usize,
        gradient_output: *FutharkArray3DF16,
        clip_norm: f32,
    ) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        if (!std.math.isFinite(clip_norm) or clip_norm < 0.0) return AccelError.InvalidDimensions;
        if (gradient_output.arr == null or gradient_output.dim2 != self.dim) return AccelError.InvalidDimensions;
        if (sequence_lengths.len != gradient_output.dim0) return AccelError.InvalidDimensions;
        const expected_tokens = std.math.mul(usize, gradient_output.dim0, gradient_output.dim1) catch return AccelError.InvalidDimensions;
        if (tokens.len != expected_tokens) return AccelError.InvalidDimensions;

        const token_i64s = std.heap.page_allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(token_i64s);
        for (tokens, 0..) |token, index| {
            if (@as(usize, token) >= self.vocab_size) return AccelError.InvalidDimensions;
            token_i64s[index] = @intCast(token);
        }

        const lengths_i64 = std.heap.page_allocator.alloc(i64, sequence_lengths.len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(lengths_i64);
        for (sequence_lengths, 0..) |length, index| {
            if (length > gradient_output.dim1) return AccelError.InvalidDimensions;
            lengths_i64[index] = @intCast(length);
        }

        var token_array = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer token_array.free(self.ctx);
        var length_array = try FutharkArray1DI64.newFromSlice(self.ctx, lengths_i64);
        defer length_array.free(self.ctx);

        var new_gradient: ?*futhark.struct_futhark_f16_2d = null;
        const result = futhark.futhark_entry_embedding_backward_padded(
            self.ctx.ctx,
            &new_gradient,
            token_array.arr,
            length_array.arr,
            gradient_output.arr,
            self.grad_weight.arr,
            clip_norm,
        );
        if (result != 0 or new_gradient == null) {
            if (new_gradient) |value| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, value);
            return AccelError.FutharkBackwardFailed;
        }
        const old_gradient = self.grad_weight.arr;
        self.grad_weight.arr = new_gradient;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_gradient);
    }

    pub fn getGradientDevicePtr(self: *Self) AccelError!struct { ptr: *anyopaque, count: usize } {
        if (!self.initialized or self.grad_weight.arr == null) return AccelError.NullPointer;
        const count = std.math.mul(usize, self.vocab_size, self.dim) catch return AccelError.InvalidDimensions;
        return .{
            .ptr = try self.ctx.getDataPointer(&self.grad_weight),
            .count = count,
        };
    }

    pub fn scaleGradient(self: *Self, scale_factor: f16) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null or self.grad_weight.arr == null) return AccelError.NullPointer;
        const scale_f32: f32 = @floatCast(scale_factor);
        if (!std.math.isFinite(scale_f32) or scale_f32 < 0.0) return AccelError.InvalidDimensions;
        var scaled: ?*futhark.struct_futhark_f16_2d = null;
        const result = futhark.futhark_entry_scale_matrix_f16(
            self.ctx.ctx,
            &scaled,
            self.grad_weight.arr,
            @bitCast(scale_factor),
        );
        if (result != 0 or scaled == null) {
            if (scaled) |value| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, value);
            return AccelError.FutharkScaleWeightsFailed;
        }
        const old = self.grad_weight.arr;
        self.grad_weight.arr = scaled;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
    }

    pub fn forward(self: *Self, tokens: []const u32) AccelError!FutharkArray2DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;

        const token_i64s = std.heap.page_allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(token_i64s);
        for (token_i64s, 0..) |*t, i| {
            t.* = @intCast(@min(tokens[i], @as(u32, @intCast(self.vocab_size - 1))));
        }

        var tok_arr = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer tok_arr.free(self.ctx);

        var out: ?*futhark.struct_futhark_f16_2d = null;
        const rc = futhark.futhark_entry_embedding_forward(
            self.ctx.ctx,
            &out,
            tok_arr.arr,
            self.weight.arr,
        );

        if (rc != 0 or out == null) return AccelError.FutharkForwardFailed;
        return FutharkArray2DF16{ .arr = out, .rows = tokens.len, .cols = self.dim };
    }

    pub fn backwardAndUpdate(self: *Self, tokens: []const u32, grad_output: *FutharkArray2DF16, lr: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;

        const token_i64s = std.heap.page_allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(token_i64s);
        for (token_i64s, 0..) |*t, i| {
            t.* = @intCast(@min(tokens[i], @as(u32, @intCast(self.vocab_size - 1))));
        }

        var tok_arr = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer tok_arr.free(self.ctx);

        var new_grad: ?*futhark.struct_futhark_f16_2d = null;
        const bwd_rc = futhark.futhark_entry_embedding_backward(
            self.ctx.ctx,
            &new_grad,
            tok_arr.arr,
            grad_output.arr,
            self.grad_weight.arr,
        );
        if (bwd_rc != 0 or new_grad == null) return AccelError.FutharkBackwardFailed;

        const old_grad = self.grad_weight.arr;
        self.grad_weight.arr = new_grad;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_grad);

        var new_weight: ?*futhark.struct_futhark_f16_2d = null;
        const lr_bits: u16 = @bitCast(lr);
        const upd_rc = futhark.futhark_entry_embedding_update(
            self.ctx.ctx,
            &new_weight,
            self.weight.arr,
            self.grad_weight.arr,
            lr_bits,
        );
        if (upd_rc != 0 or new_weight == null) return AccelError.FutharkSFDUpdateFailed;

        const old_weight = self.weight.arr;
        self.weight.arr = new_weight;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_weight);

        const zeroed_grad = try FutharkArray2DF16.newZeros(self.ctx, self.vocab_size, self.dim, std.heap.page_allocator);
        const old_g = self.grad_weight.arr;
        self.grad_weight.arr = zeroed_grad.arr;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_g);
    }

    pub fn initWithWeights(ctx: *FutharkContext, vocab_size: usize, dim: usize, weight_f16: []const f16) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (vocab_size == 0 or dim == 0) return AccelError.InvalidDimensions;
        if (weight_f16.len != vocab_size * dim) return AccelError.InvalidDimensions;
        var weight = try FutharkArray2DF16.newFromFlat(ctx, weight_f16, vocab_size, dim);
        errdefer weight.free(ctx);
        const grad_weight = try FutharkArray2DF16.newZeros(ctx, vocab_size, dim, std.heap.page_allocator);
        return Self{
            .ctx = ctx,
            .weight = weight,
            .grad_weight = grad_weight,
            .vocab_size = vocab_size,
            .dim = dim,
            .initialized = true,
        };
    }

    pub fn backwardAccumulate(self: *Self, tokens: []const u32, grad_output: *FutharkArray2DF16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        const token_i64s = std.heap.page_allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(token_i64s);
        for (token_i64s, 0..) |*t, i| {
            t.* = @intCast(@min(tokens[i], @as(u32, @intCast(self.vocab_size - 1))));
        }
        var tok_arr = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer tok_arr.free(self.ctx);
        var new_grad: ?*futhark.struct_futhark_f16_2d = null;
        const rc = futhark.futhark_entry_embedding_backward(
            self.ctx.ctx,
            &new_grad,
            tok_arr.arr,
            grad_output.arr,
            self.grad_weight.arr,
        );
        if (rc != 0 or new_grad == null) return AccelError.FutharkBackwardFailed;
        const old_grad = self.grad_weight.arr;
        self.grad_weight.arr = new_grad;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_grad);
    }

    pub fn applyUpdate(self: *Self, lr: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        var new_weight: ?*futhark.struct_futhark_f16_2d = null;
        const lr_bits: u16 = @bitCast(lr);
        const rc = futhark.futhark_entry_embedding_update(
            self.ctx.ctx,
            &new_weight,
            self.weight.arr,
            self.grad_weight.arr,
            lr_bits,
        );
        if (rc != 0 or new_weight == null) return AccelError.FutharkSFDUpdateFailed;
        const old_weight = self.weight.arr;
        self.weight.arr = new_weight;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_weight);
        const zeroed_grad = try FutharkArray2DF16.newZeros(self.ctx, self.vocab_size, self.dim, std.heap.page_allocator);
        const old_g = self.grad_weight.arr;
        self.grad_weight.arr = zeroed_grad.arr;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_g);
    }

    pub fn readGradFlat(self: *Self, allocator: std.mem.Allocator) AccelError![]f16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.grad_weight.arr == null) return AccelError.NullPointer;
        const total = self.vocab_size * self.dim;
        if (total == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f16_2d(self.ctx.ctx, self.grad_weight.arr, @ptrCast(buf.ptr)) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(self.ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn setGradFromFlat(self: *Self, data: []const f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (data.len != self.vocab_size * self.dim) return AccelError.InvalidDimensions;
        const new_grad = try FutharkArray2DF16.newFromFlat(self.ctx, data, self.vocab_size, self.dim);
        const old = self.grad_weight.arr;
        self.grad_weight.arr = new_grad.arr;
        self.grad_weight.rows = new_grad.rows;
        self.grad_weight.cols = new_grad.cols;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
    }

    pub fn spectralNormalize(
        self: *Self,
        u: *FutharkArray1DF32,
        v: *FutharkArray1DF32,
        power_iters: usize,
    ) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (self.weight.arr == null or u.arr == null or v.arr == null) return AccelError.NullPointer;
        var out_tup: ?*futhark.struct_futhark_opaque_tup3_spectral = null;
        const rc = futhark.futhark_entry_embedding_spectral_normalize(
            self.ctx.ctx,
            &out_tup,
            self.weight.arr,
            u.arr,
            v.arr,
            @intCast(power_iters),
        );
        if (rc != 0 or out_tup == null) return AccelError.FutharkForwardFailed;
        var new_weight: ?*futhark.struct_futhark_f16_2d = null;
        var new_u: ?*futhark.struct_futhark_f32_1d = null;
        var new_v: ?*futhark.struct_futhark_f32_1d = null;
        const p0 = futhark.futhark_project_opaque_tup3_spectral_0(self.ctx.ctx, &new_weight, out_tup);
        const p1 = futhark.futhark_project_opaque_tup3_spectral_1(self.ctx.ctx, &new_u, out_tup);
        const p2 = futhark.futhark_project_opaque_tup3_spectral_2(self.ctx.ctx, &new_v, out_tup);
        _ = futhark.futhark_free_opaque_tup3_spectral(self.ctx.ctx, out_tup);
        if (p0 != 0 or new_weight == null or p1 != 0 or new_u == null or p2 != 0 or new_v == null) {
            if (new_weight) |w| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, w);
            if (new_u) |nu| futhark.futhark_free_f32_1d(self.ctx.ctx, nu);
            if (new_v) |nv| futhark.futhark_free_f32_1d(self.ctx.ctx, nv);
            return AccelError.FutharkForwardFailed;
        }
        const old_w = self.weight.arr;
        self.weight.arr = new_weight;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_w);
        const old_u = u.arr;
        u.arr = new_u;
        futhark.futhark_free_f32_1d(self.ctx.ctx, old_u);
        const old_v = v.arr;
        v.arr = new_v;
        futhark.futhark_free_f32_1d(self.ctx.ctx, old_v);
    }

    pub fn readWeightFlat(self: *Self, allocator: std.mem.Allocator) AccelError![]f16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.weight.arr == null) return AccelError.NullPointer;
        const total = self.vocab_size * self.dim;
        if (total == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f16_2d(self.ctx.ctx, self.weight.arr, @ptrCast(buf.ptr)) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(self.ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn resetGrad(self: *Self) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        const zeroed = try FutharkArray2DF16.newZeros(self.ctx, self.vocab_size, self.dim, std.heap.page_allocator);
        const old = self.grad_weight.arr;
        self.grad_weight.arr = zeroed.arr;
        self.grad_weight.rows = zeroed.rows;
        self.grad_weight.cols = zeroed.cols;
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old);
    }
};

pub const FutharkArray1DU64 = struct {
    arr: ?*futhark.struct_futhark_u64_1d,
    len: usize,

    const Self = @This();

    pub fn newFromSlice(ctx: *FutharkContext, data: []const u64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_u64_1d(ctx.ctx, data.ptr, @intCast(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn valuesSlice(self: *FutharkArray1DU64, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]u64 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(u64, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_u64_1d(ctx.ctx, self.arr, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            futhark.futhark_free_u64_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const GraphBatchEncodeResult = struct {
    hashes: []u64,
    re_a: []f32,
    im_a: []f32,
    re_b: []f32,
    im_b: []f32,
    edge_srcs: []i64,
    edge_tgts: []i64,
    node_count: usize,
    edge_count: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GraphBatchEncodeResult) void {
        self.allocator.free(self.hashes);
        self.allocator.free(self.re_a);
        self.allocator.free(self.im_a);
        self.allocator.free(self.re_b);
        self.allocator.free(self.im_b);
        self.allocator.free(self.edge_srcs);
        self.allocator.free(self.edge_tgts);
    }
};

const GRAPH_ENCODE_CHUNK_SIZE: usize = 4096;

pub fn batchEncodeGraph(
    ctx: *FutharkContext,
    hashes: []const u64,
    seed: u64,
    allocator: std.mem.Allocator,
) AccelError!GraphBatchEncodeResult {
    if (ctx.ctx == null) return AccelError.NullPointer;
    if (hashes.len == 0) return AccelError.InvalidDimensions;

    var acc_hashes = std.ArrayList(u64).init(allocator);
    errdefer acc_hashes.deinit();
    var acc_re_a = std.ArrayList(f32).init(allocator);
    errdefer acc_re_a.deinit();
    var acc_im_a = std.ArrayList(f32).init(allocator);
    errdefer acc_im_a.deinit();
    var acc_re_b = std.ArrayList(f32).init(allocator);
    errdefer acc_re_b.deinit();
    var acc_im_b = std.ArrayList(f32).init(allocator);
    errdefer acc_im_b.deinit();
    var acc_edge_srcs = std.ArrayList(i64).init(allocator);
    errdefer acc_edge_srcs.deinit();
    var acc_edge_tgts = std.ArrayList(i64).init(allocator);
    errdefer acc_edge_tgts.deinit();

    acc_hashes.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_re_a.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_im_a.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_re_b.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_im_b.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_edge_srcs.ensureTotalCapacity(hashes.len * 3) catch return AccelError.AllocationFailed;
    acc_edge_tgts.ensureTotalCapacity(hashes.len * 3) catch return AccelError.AllocationFailed;

    var offset: usize = 0;
    while (offset < hashes.len) {
        const chunk_end = @min(offset + GRAPH_ENCODE_CHUNK_SIZE, hashes.len);
        const chunk = hashes[offset..chunk_end];
        const chunk_n = chunk.len;
        const chunk_ne = chunk_n * 3;

        var in_chunk = FutharkArray1DU64.newFromSlice(ctx, chunk) catch |err| {
            std.debug.print("[batchEncodeGraph] chunk offset={d} upload failed: {}\n", .{ offset, err });
            return err;
        };
        defer in_chunk.free(ctx);

        var out_tup: ?*futhark.struct_futhark_opaque_tup7_graph_encode = null;
        defer {
            if (out_tup) |p| {
                _ = futhark.futhark_free_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64(ctx.ctx, p);
            }
        }

        var out_ids: ?*futhark.struct_futhark_u64_1d = null;
        var out_re_a: ?*futhark.struct_futhark_f32_1d = null;
        var out_im_a: ?*futhark.struct_futhark_f32_1d = null;
        var out_re_b: ?*futhark.struct_futhark_f32_1d = null;
        var out_im_b: ?*futhark.struct_futhark_f32_1d = null;
        var out_edge_srcs: ?*futhark.struct_futhark_i64_1d = null;
        var out_edge_tgts: ?*futhark.struct_futhark_i64_1d = null;

        defer {
            if (out_ids) |p| futhark.futhark_free_u64_1d(ctx.ctx, p);
            if (out_re_a) |p| futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_im_a) |p| futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_re_b) |p| futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_im_b) |p| futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_edge_srcs) |p| futhark.futhark_free_i64_1d(ctx.ctx, p);
            if (out_edge_tgts) |p| futhark.futhark_free_i64_1d(ctx.ctx, p);
        }

        const rc = futhark.futhark_entry_graph_batch_encode(
            ctx.ctx,
            &out_tup,
            in_chunk.arr,
            seed,
        );

        if (rc != 0) {
            const ctx_err = futhark.futhark_context_get_error(ctx.ctx);
            if (ctx_err) |msg| {
                std.debug.print("[batchEncodeGraph] Futhark entry error at offset={d} n={d}: {s}\n", .{ offset, chunk_n, std.mem.span(msg) });
            } else {
                std.debug.print("[batchEncodeGraph] Futhark entry failed at offset={d} n={d}: rc={d}\n", .{ offset, chunk_n, rc });
            }
            return AccelError.FutharkForwardFailed;
        }

        const sync_rc = futhark.futhark_context_sync(ctx.ctx);
        if (sync_rc != 0) {
            const ctx_err = futhark.futhark_context_get_error(ctx.ctx);
            if (ctx_err) |msg| {
                std.debug.print("[batchEncodeGraph] Futhark sync error at offset={d} n={d}: {s}\n", .{ offset, chunk_n, std.mem.span(msg) });
            } else {
                std.debug.print("[batchEncodeGraph] futhark_context_sync failed at offset={d} n={d}: rc={d}\n", .{ offset, chunk_n, sync_rc });
            }
            return AccelError.FutharkSyncFailed;
        }

        const tup = out_tup orelse {
            std.debug.print("[batchEncodeGraph] out_tup null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        };
        const proj0 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_0(ctx.ctx, &out_ids, tup);
        const proj1 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_1(ctx.ctx, &out_re_a, tup);
        const proj2 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_2(ctx.ctx, &out_im_a, tup);
        const proj3 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_3(ctx.ctx, &out_re_b, tup);
        const proj4 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_4(ctx.ctx, &out_im_b, tup);
        const proj5 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_5(ctx.ctx, &out_edge_srcs, tup);
        const proj6 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_6(ctx.ctx, &out_edge_tgts, tup);
        if (proj0 != 0 or proj1 != 0 or proj2 != 0 or proj3 != 0 or proj4 != 0 or proj5 != 0 or proj6 != 0) {
            std.debug.print("[batchEncodeGraph] projection failed at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.FutharkForwardFailed;
        }

        if (out_ids == null) {
            std.debug.print("[batchEncodeGraph] out_ids null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_re_a == null) {
            std.debug.print("[batchEncodeGraph] out_re_a null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_im_a == null) {
            std.debug.print("[batchEncodeGraph] out_im_a null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_re_b == null) {
            std.debug.print("[batchEncodeGraph] out_re_b null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_im_b == null) {
            std.debug.print("[batchEncodeGraph] out_im_b null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_edge_srcs == null) {
            std.debug.print("[batchEncodeGraph] out_edge_srcs null at offset={d} ne={d}\n", .{ offset, chunk_ne });
            return AccelError.NullPointer;
        }
        if (out_edge_tgts == null) {
            std.debug.print("[batchEncodeGraph] out_edge_tgts null at offset={d} ne={d}\n", .{ offset, chunk_ne });
            return AccelError.NullPointer;
        }

        {
            const buf = allocator.alloc(u64, chunk_n) catch return AccelError.AllocationFailed;
            defer allocator.free(buf);
            if (futhark.futhark_values_u64_1d(ctx.ctx, out_ids, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
            if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
            acc_hashes.appendSlice(buf) catch return AccelError.AllocationFailed;
        }

        {
            const buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
            defer allocator.free(buf);
            if (futhark.futhark_values_f32_1d(ctx.ctx, out_re_a, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
            if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
            acc_re_a.appendSlice(buf) catch return AccelError.AllocationFailed;
        }

        {
            const buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
            defer allocator.free(buf);
            if (futhark.futhark_values_f32_1d(ctx.ctx, out_im_a, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
            if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
            acc_im_a.appendSlice(buf) catch return AccelError.AllocationFailed;
        }

        {
            const buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
            defer allocator.free(buf);
            if (futhark.futhark_values_f32_1d(ctx.ctx, out_re_b, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
            if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
            acc_re_b.appendSlice(buf) catch return AccelError.AllocationFailed;
        }

        {
            const buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
            defer allocator.free(buf);
            if (futhark.futhark_values_f32_1d(ctx.ctx, out_im_b, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
            if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
            acc_im_b.appendSlice(buf) catch return AccelError.AllocationFailed;
        }

        {
            const buf = allocator.alloc(i64, chunk_ne) catch return AccelError.AllocationFailed;
            defer allocator.free(buf);
            if (futhark.futhark_values_i64_1d(ctx.ctx, out_edge_srcs, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
            if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
            for (buf) |v| {
                const adjusted: i64 = if (v >= 0) v + @as(i64, @intCast(offset)) else v;
                acc_edge_srcs.append(adjusted) catch return AccelError.AllocationFailed;
            }
        }

        {
            const buf = allocator.alloc(i64, chunk_ne) catch return AccelError.AllocationFailed;
            defer allocator.free(buf);
            if (futhark.futhark_values_i64_1d(ctx.ctx, out_edge_tgts, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
            if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
            for (buf) |v| {
                const adjusted: i64 = if (v >= 0) v + @as(i64, @intCast(offset)) else v;
                acc_edge_tgts.append(adjusted) catch return AccelError.AllocationFailed;
            }
        }

        offset = chunk_end;
    }

    const total_n = acc_hashes.items.len;
    const total_ne = acc_edge_srcs.items.len;

    const owned_hashes = acc_hashes.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_hashes);
    const owned_re_a = acc_re_a.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_re_a);
    const owned_im_a = acc_im_a.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_im_a);
    const owned_re_b = acc_re_b.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_re_b);
    const owned_im_b = acc_im_b.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_im_b);
    const owned_edge_srcs = acc_edge_srcs.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_edge_srcs);
    const owned_edge_tgts = acc_edge_tgts.toOwnedSlice() catch return AccelError.AllocationFailed;

    return GraphBatchEncodeResult{
        .hashes = owned_hashes,
        .re_a = owned_re_a,
        .im_a = owned_im_a,
        .re_b = owned_re_b,
        .im_b = owned_im_b,
        .edge_srcs = owned_edge_srcs,
        .edge_tgts = owned_edge_tgts,
        .node_count = total_n,
        .edge_count = total_ne,
        .allocator = allocator,
    };
}
