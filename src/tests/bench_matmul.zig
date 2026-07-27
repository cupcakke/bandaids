const std = @import("std");
const deps = @import("deps");
const core = deps.core_tensor;
const Tensor = deps.core_tensor.Tensor;

const SIZES = [_]usize{ 128, 256, 512, 1024 };
const BENCH_ITERS: usize = 100;
const BENCH_WARMUP: usize = 10;

const Result = struct {
    n: usize,
    total_ms: f64,
    per_iter_ms: f64,
    gflops: f64,
};

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
    std.debug.print("BENCHMARK: Matrix Multiplication (cache-friendly i-p-j)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Config: sizes={d},{d},{d},{d}, iters={d}\n", .{ SIZES[0], SIZES[1], SIZES[2], SIZES[3], BENCH_ITERS });
    std.debug.print("Warmup: {d} iterations (not timed)\n", .{BENCH_WARMUP});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    var results: [SIZES.len]Result = undefined;

    for (SIZES, 0..) |n, idx| {
        const dims = [_]usize{ n, n };

        var a = try Tensor.init(allocator, &dims);
        defer a.deinit();
        var b = try Tensor.init(allocator, &dims);
        defer b.deinit();

        const fill_val: f32 = 1.0 / @as(f32, @floatFromInt(n));
        try a.fill(fill_val);
        try b.fill(fill_val);

        var w: usize = 0;
        while (w < BENCH_WARMUP) : (w += 1) {
            var r = try Tensor.matmul(&a, &b, allocator);
            r.deinit();
        }

        var timer = try std.time.Timer.start();
        var iter: usize = 0;
        while (iter < BENCH_ITERS) : (iter += 1) {
            var r = try Tensor.matmul(&a, &b, allocator);
            r.deinit();
        }
        const elapsed_ns = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const per_iter_ms = elapsed_ms / @as(f64, @floatFromInt(BENCH_ITERS));
        const n_f64 = @as(f64, @floatFromInt(n));
        const flops = 2.0 * n_f64 * n_f64 * n_f64 * @as(f64, @floatFromInt(BENCH_ITERS));
        const elapsed_secs = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
        const gflops = if (elapsed_secs > 0) flops / elapsed_secs / 1_000_000_000.0 else 0;

        results[idx] = .{
            .n = n,
            .total_ms = elapsed_ms,
            .per_iter_ms = per_iter_ms,
            .gflops = gflops,
        };

        std.debug.print("[matmul {d}x{d}]\n", .{ n, n });
        std.debug.print("  Total time:        {d:.2} ms\n", .{elapsed_ms});
        std.debug.print("  Per iteration:     {d:.2} ms\n", .{per_iter_ms});
        std.debug.print("  Throughput:        {d:.4} GFLOPS\n", .{gflops});
        std.debug.print("--------------------------------------------------------------------------------\n", .{});
    }

    std.debug.print("\nSummary:\n", .{});
    std.debug.print("  {s:>8}  {s:>12}  {s:>10}\n", .{ "Size", "ms/iter", "GFLOPS" });
    for (results) |r| {
        std.debug.print("  {d:>6}x{d:<6}{d:>10.2}  {d:>10.4}\n", .{ r.n, r.n, r.per_iter_ms, r.gflops });
    }
    std.debug.print("--------------------------------------------------------------------------------\n", .{});
    std.debug.print("RESULT: PASS\n", .{});
}
