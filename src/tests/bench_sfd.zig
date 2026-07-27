const std = @import("std");
const deps = @import("deps");
const core = deps.core_tensor;
const Tensor = deps.sfd.Tensor;
const SpectralNormalizer = deps.sfd.SpectralNormalizer;

// FP4 quantization constants
const QUANT_N: usize = 1 << 20; // 1M values
const QUANT_ITERS: usize = 100;

// Spectral normalizer constants
const WEIGHT_DIM: usize = 512;
const SPECTRAL_ITERS: usize = 50;
const POWER_ITERS_FULL: usize = 20;
const POWER_ITERS_SPARSE: usize = 5;

// FP4 quantization logic (replicated from sfd.zig since quantizeValue is private)
fn quantizeFP4(value: f32) f32 {
    if (!std.math.isFinite(value)) return value;
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
    return sign * best;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) {
            std.debug.print("RESULT: FAIL (memory leak detected)\n", .{});
        }
    }
    const allocator = gpa.allocator();

    const eff = core.effectiveCpuCount();
    const src = core.cgroupSource();
    std.debug.print("[env] effective_cpu={d} cgroup_source={s}\n", .{ eff, src });

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("BENCHMARK: SFD Optimizations (FP4 quantization + SpectralNorm)\n", .{});
    std.debug.print("================================================================================\n", .{});

    // --- FP4 quantization benchmark ---
    {
        std.debug.print("Config: quant_n={d}, iters={d}\n", .{ QUANT_N, QUANT_ITERS });
        std.debug.print("--------------------------------------------------------------------------------\n", .{});

        const input = try allocator.alloc(f32, QUANT_N);
        defer allocator.free(input);
        const output = try allocator.alloc(f32, QUANT_N);
        defer allocator.free(output);

        // Fill with linearly spaced values from -6.0 to 6.0
        var i: usize = 0;
        while (i < QUANT_N) : (i += 1) {
            const t_val: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(QUANT_N - 1));
            input[i] = -6.0 + 12.0 * t_val;
        }

        // Timed
        var timer = try std.time.Timer.start();
        var iter: usize = 0;
        while (iter < QUANT_ITERS) : (iter += 1) {
            for (input, 0..) |v, idx| {
                output[idx] = quantizeFP4(v);
            }
        }
        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const total_elements = @as(f64, @floatFromInt(QUANT_N)) * @as(f64, @floatFromInt(QUANT_ITERS));
        const ns_per_elem = @as(f64, @floatFromInt(elapsed_ns)) / total_elements;
        const elapsed_secs = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
        const elems_per_sec = if (elapsed_secs > 0) total_elements / elapsed_secs else 0;

        std.debug.print("[FP4 quantization]\n", .{});
        std.debug.print("  Total time:        {d:.2} ms\n", .{elapsed_ms});
        std.debug.print("  ns per element:    {d:.2} ns\n", .{ns_per_elem});
        std.debug.print("  Throughput:        {d:.2} elements/sec\n", .{elems_per_sec});
        std.debug.print("--------------------------------------------------------------------------------\n", .{});
    }

    // --- SpectralNormalizer benchmark ---
    {
        std.debug.print("Config: weight_dim={d}, iters={d}, power_iters_full={d}, power_iters_sparse={d}\n", .{ WEIGHT_DIM, SPECTRAL_ITERS, POWER_ITERS_FULL, POWER_ITERS_SPARSE });
        std.debug.print("--------------------------------------------------------------------------------\n", .{});

        const dims = [_]usize{ WEIGHT_DIM, WEIGHT_DIM };

        // Fill with deterministic pseudo-random values
        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();

        // Save initial values for reset
        const initial_data = try allocator.alloc(f32, WEIGHT_DIM * WEIGHT_DIM);
        defer allocator.free(initial_data);
        for (initial_data) |*v| {
            v.* = random.float(f32) * 2.0 - 1.0;
        }

        // Benchmark with full power iterations
        var weights_full = try Tensor.init(allocator, &dims);
        defer weights_full.deinit();

        var normalizer_full = SpectralNormalizer.init(POWER_ITERS_FULL);

        var full_timer = try std.time.Timer.start();
        var iter: usize = 0;
        while (iter < SPECTRAL_ITERS) : (iter += 1) {
            @memcpy(weights_full.data, initial_data);
            try normalizer_full.normalizeWeights(&weights_full, allocator);
        }
        const full_ns = full_timer.read();
        const full_ms = @as(f64, @floatFromInt(full_ns)) / 1_000_000.0;
        const full_per_iter = full_ms / @as(f64, @floatFromInt(SPECTRAL_ITERS));

        std.debug.print("[SpectralNorm power_iterations={d}]\n", .{POWER_ITERS_FULL});
        std.debug.print("  Total time:        {d:.2} ms\n", .{full_ms});
        std.debug.print("  Per iteration:     {d:.2} ms\n", .{full_per_iter});
        std.debug.print("--------------------------------------------------------------------------------\n", .{});

        // Benchmark with sparse power iterations
        var weights_sparse = try Tensor.init(allocator, &dims);
        defer weights_sparse.deinit();

        var normalizer_sparse = SpectralNormalizer.init(POWER_ITERS_SPARSE);

        var sparse_timer = try std.time.Timer.start();
        iter = 0;
        while (iter < SPECTRAL_ITERS) : (iter += 1) {
            @memcpy(weights_sparse.data, initial_data);
            try normalizer_sparse.normalizeWeights(&weights_sparse, allocator);
        }
        const sparse_ns = sparse_timer.read();
        const sparse_ms = @as(f64, @floatFromInt(sparse_ns)) / 1_000_000.0;
        const sparse_per_iter = sparse_ms / @as(f64, @floatFromInt(SPECTRAL_ITERS));

        std.debug.print("[SpectralNorm power_iterations={d}]\n", .{POWER_ITERS_SPARSE});
        std.debug.print("  Total time:        {d:.2} ms\n", .{sparse_ms});
        std.debug.print("  Per iteration:     {d:.2} ms\n", .{sparse_per_iter});
        std.debug.print("--------------------------------------------------------------------------------\n", .{});

        const speedup = if (sparse_per_iter > 0) full_per_iter / sparse_per_iter else 0;
        std.debug.print("Speedup ratio (full/sparse): {d:.2}x\n", .{speedup});
        std.debug.print("--------------------------------------------------------------------------------\n", .{});
    }

    std.debug.print("RESULT: PASS\n", .{});
}

