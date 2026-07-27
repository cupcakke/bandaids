const std = @import("std");
const deps = @import("deps");
const core = deps.core_tensor;
const RSF = deps.rsf.RSF;
const Tensor = deps.core_tensor.Tensor;

const BENCH_DIM: usize = 512;
const BENCH_LAYERS: usize = 12;
const BENCH_BATCH: usize = 64;
const BENCH_WARMUP: usize = 20;
const BENCH_ITERS: usize = 200;

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
    std.debug.print("BENCHMARK: RSF Forward/Backward Pass\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Config: dim={d}, layers={d}, batch={d}, iters={d}\n", .{ BENCH_DIM, BENCH_LAYERS, BENCH_BATCH, BENCH_ITERS });
    std.debug.print("Warmup: {d} iterations (not timed)\n", .{BENCH_WARMUP});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    var model = try RSF.init(allocator, BENCH_DIM, BENCH_LAYERS);
    defer model.deinit();

    const dim2 = BENCH_DIM * 2;
    const input_shape = [_]usize{ BENCH_BATCH, dim2 };

    var x = try Tensor.init(allocator, &input_shape);
    defer x.deinit();
    @memset(x.data[0..x.shape.totalSize()], 0.1);

    var y = try Tensor.init(allocator, &input_shape);
    defer y.deinit();

    var grad_output = try Tensor.init(allocator, &input_shape);
    defer grad_output.deinit();
    @memset(grad_output.data[0..grad_output.shape.totalSize()], 0.01);

    var grad_input = try Tensor.init(allocator, &input_shape);
    defer grad_input.deinit();

    // --- Forward benchmark ---
    // Warmup
    var w: usize = 0;
    while (w < BENCH_WARMUP) : (w += 1) {
        @memcpy(y.data[0..y.shape.totalSize()], x.data[0..x.shape.totalSize()]);
        try model.forward(&y);
    }

    // Timed
    var fwd_timer = try std.time.Timer.start();
    var iter: usize = 0;
    while (iter < BENCH_ITERS) : (iter += 1) {
        @memcpy(y.data[0..y.shape.totalSize()], x.data[0..x.shape.totalSize()]);
        try model.forward(&y);
    }
    const fwd_ns = fwd_timer.read();
    const fwd_ms = @as(f64, @floatFromInt(fwd_ns)) / 1_000_000.0;
    const fwd_per_iter = fwd_ms / @as(f64, @floatFromInt(BENCH_ITERS));
    const fwd_secs = fwd_ms / 1000.0;
    const total_elements: f64 = @as(f64, @floatFromInt(BENCH_BATCH)) * @as(f64, @floatFromInt(BENCH_ITERS)) * @as(f64, @floatFromInt(dim2));
    const fwd_throughput = if (fwd_secs > 0) total_elements / fwd_secs else 0;

    std.debug.print("[forward]\n", .{});
    std.debug.print("  Total time:        {d:.2} ms\n", .{fwd_ms});
    std.debug.print("  Per iteration:     {d:.2} ms\n", .{fwd_per_iter});
    std.debug.print("  Throughput:        {d:.2} elements/sec\n", .{fwd_throughput});
    std.debug.print("  ms per batch:      {d:.2} ms\n", .{fwd_per_iter});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    // --- Backward benchmark ---
    // Prepare valid forward output
    @memcpy(y.data[0..y.shape.totalSize()], x.data[0..x.shape.totalSize()]);
    try model.forward(&y);

    // Warmup
    w = 0;
    while (w < BENCH_WARMUP) : (w += 1) {
        try model.backward(&grad_output, &x, &y, &grad_input);
    }

    // Timed
    var bwd_timer = try std.time.Timer.start();
    iter = 0;
    while (iter < BENCH_ITERS) : (iter += 1) {
        try model.backward(&grad_output, &x, &y, &grad_input);
    }
    const bwd_ns = bwd_timer.read();
    const bwd_ms = @as(f64, @floatFromInt(bwd_ns)) / 1_000_000.0;
    const bwd_per_iter = bwd_ms / @as(f64, @floatFromInt(BENCH_ITERS));
    const bwd_secs = bwd_ms / 1000.0;
    const bwd_throughput = if (bwd_secs > 0) total_elements / bwd_secs else 0;

    std.debug.print("[backward]\n", .{});
    std.debug.print("  Total time:        {d:.2} ms\n", .{bwd_ms});
    std.debug.print("  Per iteration:     {d:.2} ms\n", .{bwd_per_iter});
    std.debug.print("  Throughput:        {d:.2} elements/sec\n", .{bwd_throughput});
    std.debug.print("  ms per batch:      {d:.2} ms\n", .{bwd_per_iter});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    const ratio = if (fwd_per_iter > 0) bwd_per_iter / fwd_per_iter else 0;
    std.debug.print("backward/forward time ratio: {d:.2}x\n", .{ratio});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    const invertible = try model.verifyInvertible(&x, 1e-4, 1e-4);
    if (invertible) {
        std.debug.print("RESULT: PASS\n", .{});
    } else {
        std.debug.print("RESULT: FAIL (invertibility check failed)\n", .{});
    }
}

