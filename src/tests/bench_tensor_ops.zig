const std = @import("std");
const deps = @import("deps");
const core = deps.core_tensor;
const Tensor = deps.core_tensor.Tensor;

const TENSOR_SIZE: usize = 1 << 22; // 4M elements
const BENCH_ITERS: usize = 500;
const BENCH_WARMUP: usize = 50;

const BenchResult = struct {
    name: []const u8,
    total_ms: f64,
    ns_per_element: f64,
    gb_per_sec: f64,
};

fn benchFill(allocator: std.mem.Allocator) !BenchResult {
    const shape = [_]usize{TENSOR_SIZE};
    var t = try Tensor.init(allocator, &shape);
    defer t.deinit();

    // Warmup
    var w: usize = 0;
    while (w < BENCH_WARMUP) : (w += 1) {
        try t.fill(1.0);
    }

    // Timed
    var timer = try std.time.Timer.start();
    var iter: usize = 0;
    while (iter < BENCH_ITERS) : (iter += 1) {
        try t.fill(1.0);
    }
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const total_ops = @as(f64, @floatFromInt(TENSOR_SIZE)) * @as(f64, @floatFromInt(BENCH_ITERS));
    const ns_per_elem = @as(f64, @floatFromInt(elapsed_ns)) / total_ops;
    const bytes = @as(f64, @floatFromInt(TENSOR_SIZE)) * @as(f64, @sizeOf(f32)) * @as(f64, @floatFromInt(BENCH_ITERS));
    const elapsed_secs = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const gb_s = if (elapsed_secs > 0) bytes / elapsed_secs / 1_000_000_000.0 else 0;

    return .{ .name = "fill", .total_ms = elapsed_ms, .ns_per_element = ns_per_elem, .gb_per_sec = gb_s };
}

fn benchAdd(allocator: std.mem.Allocator) !BenchResult {
    const shape = [_]usize{TENSOR_SIZE};
    var a = try Tensor.init(allocator, &shape);
    defer a.deinit();
    var b = try Tensor.init(allocator, &shape);
    defer b.deinit();

    try a.fill(1.0);
    try b.fill(0.001);

    // Warmup
    var w: usize = 0;
    while (w < BENCH_WARMUP) : (w += 1) {
        try a.add(&b);
    }

    // Timed
    var timer = try std.time.Timer.start();
    var iter: usize = 0;
    while (iter < BENCH_ITERS) : (iter += 1) {
        try a.add(&b);
    }
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const total_ops = @as(f64, @floatFromInt(TENSOR_SIZE)) * @as(f64, @floatFromInt(BENCH_ITERS));
    const ns_per_elem = @as(f64, @floatFromInt(elapsed_ns)) / total_ops;
    const bytes = @as(f64, @floatFromInt(TENSOR_SIZE)) * @as(f64, @sizeOf(f32)) * @as(f64, @floatFromInt(BENCH_ITERS));
    const elapsed_secs = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const gb_s = if (elapsed_secs > 0) bytes / elapsed_secs / 1_000_000_000.0 else 0;

    return .{ .name = "add", .total_ms = elapsed_ms, .ns_per_element = ns_per_elem, .gb_per_sec = gb_s };
}

fn benchMul(allocator: std.mem.Allocator) !BenchResult {
    const shape = [_]usize{TENSOR_SIZE};
    var a = try Tensor.init(allocator, &shape);
    defer a.deinit();
    var b = try Tensor.init(allocator, &shape);
    defer b.deinit();

    try a.fill(1.0);
    try b.fill(1.000001);

    // Warmup
    var w: usize = 0;
    while (w < BENCH_WARMUP) : (w += 1) {
        try a.mul(&b);
    }

    // Timed
    var timer = try std.time.Timer.start();
    var iter: usize = 0;
    while (iter < BENCH_ITERS) : (iter += 1) {
        try a.mul(&b);
    }
    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const total_ops = @as(f64, @floatFromInt(TENSOR_SIZE)) * @as(f64, @floatFromInt(BENCH_ITERS));
    const ns_per_elem = @as(f64, @floatFromInt(elapsed_ns)) / total_ops;
    const bytes = @as(f64, @floatFromInt(TENSOR_SIZE)) * @as(f64, @sizeOf(f32)) * @as(f64, @floatFromInt(BENCH_ITERS));
    const elapsed_secs = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
    const gb_s = if (elapsed_secs > 0) bytes / elapsed_secs / 1_000_000_000.0 else 0;

    return .{ .name = "mul", .total_ms = elapsed_ms, .ns_per_element = ns_per_elem, .gb_per_sec = gb_s };
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
    std.debug.print("BENCHMARK: Tensor Element-wise Operations (contiguous fast path)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Config: tensor_size={d} ({d} MB), iters={d}\n", .{ TENSOR_SIZE, TENSOR_SIZE * @sizeOf(f32) / (1024 * 1024), BENCH_ITERS });
    std.debug.print("Warmup: {d} iterations (not timed)\n", .{BENCH_WARMUP});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    const fill_result = try benchFill(allocator);
    std.debug.print("[{s}]\n", .{fill_result.name});
    std.debug.print("  Total time:        {d:.2} ms\n", .{fill_result.total_ms});
    std.debug.print("  ns per element:    {d:.2} ns\n", .{fill_result.ns_per_element});
    std.debug.print("  Bandwidth:         {d:.2} GB/s\n", .{fill_result.gb_per_sec});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    const add_result = try benchAdd(allocator);
    std.debug.print("[{s}]\n", .{add_result.name});
    std.debug.print("  Total time:        {d:.2} ms\n", .{add_result.total_ms});
    std.debug.print("  ns per element:    {d:.2} ns\n", .{add_result.ns_per_element});
    std.debug.print("  Bandwidth:         {d:.2} GB/s\n", .{add_result.gb_per_sec});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    const mul_result = try benchMul(allocator);
    std.debug.print("[{s}]\n", .{mul_result.name});
    std.debug.print("  Total time:        {d:.2} ms\n", .{mul_result.total_ms});
    std.debug.print("  ns per element:    {d:.2} ns\n", .{mul_result.ns_per_element});
    std.debug.print("  Bandwidth:         {d:.2} GB/s\n", .{mul_result.gb_per_sec});
    std.debug.print("--------------------------------------------------------------------------------\n", .{});

    std.debug.print("RESULT: PASS\n", .{});
}

