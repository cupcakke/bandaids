const std = @import("std");
const Tensor = @import("../core/tensor.zig").Tensor;
const Thread = std.Thread;
const Allocator = std.mem.Allocator;

const TestConfig = struct {
    num_threads: usize,
    ops_per_thread: usize,
    num_tensors: usize,
};

const ThreadContext = struct {
    tensors: []Tensor,
    thread_id: usize,
    barrier: *std.atomic.Value(usize),
    total_threads: usize,
    ops_per_thread: usize,
};

fn threadWorker(ctx: ThreadContext) void {
    const seed: u64 = @as(u64, ctx.thread_id) *% 12345 +% 1;
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    _ = ctx.barrier.fetchAdd(1, .seq_cst);
    while (ctx.barrier.load(.seq_cst) < ctx.total_threads) {
        std.Thread.yield() catch {};
    }

    const num_tensors = ctx.tensors.len;
    if (num_tensors == 0) return;

    var ops: usize = 0;
    while (ops < ctx.ops_per_thread) : (ops += 1) {
        const tensor_idx = rand.intRangeAtMost(usize, 0, num_tensors - 1);
        const op_type = rand.intRangeAtMost(u8, 0, 99);

        if (op_type < 50) {
            ctx.tensors[tensor_idx].retain();
            if (rand.boolean()) {
                std.Thread.yield() catch {};
            }
            ctx.tensors[tensor_idx].release();
        } else if (op_type < 75) {
            ctx.tensors[tensor_idx].retain();
            ctx.tensors[tensor_idx].retain();
            if (rand.boolean()) {
                std.Thread.yield() catch {};
            }
            ctx.tensors[tensor_idx].release();
            ctx.tensors[tensor_idx].release();
        } else if (op_type < 90) {
            var other_idx = rand.intRangeAtMost(usize, 0, num_tensors - 1);
            if (num_tensors > 1) {
                while (other_idx == tensor_idx) {
                    other_idx = rand.intRangeAtMost(usize, 0, num_tensors - 1);
                }
            }
            ctx.tensors[tensor_idx].retain();
            ctx.tensors[other_idx].retain();
            if (rand.boolean()) {
                std.Thread.yield() catch {};
            }
            ctx.tensors[tensor_idx].release();
            ctx.tensors[other_idx].release();
        } else {
            var local_retains: usize = 0;
            while (local_retains < 5) : (local_retains += 1) {
                ctx.tensors[tensor_idx].retain();
            }
            if (rand.boolean()) {
                std.Thread.yield() catch {};
            }
            local_retains = 0;
            while (local_retains < 5) : (local_retains += 1) {
                ctx.tensors[tensor_idx].release();
            }
        }

        if (ops % 1000 == 0 and ctx.thread_id == 0) {
            std.debug.print("Thread {d}: {d}/{d} operations completed\n", .{ ctx.thread_id, ops, ctx.ops_per_thread });
        }
    }
}

fn getRefcount(tensor: *const Tensor) usize {
    return @atomicLoad(usize, tensor.refcount, .seq_cst);
}

fn runStressTest(allocator: Allocator, config: TestConfig) !void {
    if (config.num_tensors == 0) return error.InvalidConfiguration;
    if (config.num_threads == 0) return error.InvalidConfiguration;

    var tensors = try allocator.alloc(Tensor, config.num_tensors);
    var initialized_count: usize = 0;
    errdefer {
        var j: usize = 0;
        while (j < initialized_count) : (j += 1) {
            tensors[j].deinit();
        }
        allocator.free(tensors);
    }

    const tensor_shape = [_]usize{ 64, 64 };
    var i: usize = 0;
    while (i < config.num_tensors) : (i += 1) {
        tensors[i] = try Tensor.init(allocator, &tensor_shape);
        initialized_count += 1;
    }

    i = 0;
    while (i < config.num_tensors) : (i += 1) {
        const refcount = getRefcount(&tensors[i]);
        if (refcount != 1) {
            std.debug.print("ERROR: Tensor {d} initial refcount is {d}, expected 1\n", .{ i, refcount });
            return error.InvalidInitialRefcount;
        }
    }

    var barrier = std.atomic.Value(usize).init(0);

    var threads = try allocator.alloc(Thread, config.num_threads);
    defer allocator.free(threads);

    var spawned_count: usize = 0;
    errdefer {
        barrier.store(config.num_threads, .seq_cst);
        var j: usize = 0;
        while (j < spawned_count) : (j += 1) {
            threads[j].join();
        }
    }

    var timer = try std.time.Timer.start();

    var t: usize = 0;
    while (t < config.num_threads) : (t += 1) {
        const ctx = ThreadContext{
            .tensors = tensors,
            .thread_id = t,
            .barrier = &barrier,
            .total_threads = config.num_threads,
            .ops_per_thread = config.ops_per_thread,
        };
        threads[t] = try Thread.spawn(.{}, threadWorker, .{ctx});
        spawned_count += 1;
    }

    t = 0;
    while (t < config.num_threads) : (t += 1) {
        threads[t].join();
    }
    spawned_count = 0;

    const elapsed_ns = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const total_ops = config.num_threads * config.ops_per_thread;

    var ops_per_sec: f64 = 0.0;
    var avg_time_per_op: f64 = 0.0;
    if (elapsed_ns > 0) {
        ops_per_sec = @as(f64, @floatFromInt(total_ops)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0);
        avg_time_per_op = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(total_ops));
    }

    std.debug.print("\nTotal time: {d:.2} ms\n", .{elapsed_ms});
    std.debug.print("Total operations: {d}\n", .{total_ops});
    std.debug.print("Throughput: {d:.2} ops/sec\n", .{ops_per_sec});
    std.debug.print("Average time per operation: {d:.2} ns\n", .{avg_time_per_op});

    var all_correct = true;
    i = 0;
    while (i < config.num_tensors) : (i += 1) {
        const refcount = getRefcount(&tensors[i]);
        if (refcount != 1) {
            std.debug.print("Tensor {d}: refcount = {d} (expected 1)\n", .{ i, refcount });
            all_correct = false;
        }
    }

    if (!all_correct) {
        return error.RefcountMismatch;
    }

    i = 0;
    while (i < config.num_tensors) : (i += 1) {
        tensors[i].deinit();
    }
    allocator.free(tensors);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{
        .thread_safe = true,
    }){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) {
            std.debug.print("\n[LEAK DETECTED] Memory leaked!\n", .{});
            std.process.exit(1);
        }
    }
    const allocator = gpa.allocator();

    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("TENSOR REFCOUNT STRESS TEST\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});

    const config = TestConfig{
        .num_threads = 12,
        .ops_per_thread = 15000,
        .num_tensors = 8,
    };

    std.debug.print("Threads: {d}\n", .{config.num_threads});
    std.debug.print("Operations per thread: {d}\n", .{config.ops_per_thread});
    std.debug.print("Shared tensors: {d}\n", .{config.num_tensors});
    std.debug.print("Total operations: {d}\n", .{config.num_threads * config.ops_per_thread});
    std.debug.print("-" ** 80 ++ "\n", .{});

    try runStressTest(allocator, config);

    std.debug.print("\n[SUCCESS] Stress test passed! No memory leaks, no refcount errors.\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
}

test "single tensor retain release" {
    const allocator = std.testing.allocator;

    const tensor_shape = [_]usize{ 8, 8 };
    var tensor = try Tensor.init(allocator, &tensor_shape);
    defer tensor.deinit();

    try std.testing.expectEqual(@as(usize, 1), getRefcount(&tensor));

    tensor.retain();
    try std.testing.expectEqual(@as(usize, 2), getRefcount(&tensor));

    tensor.release();
    try std.testing.expectEqual(@as(usize, 1), getRefcount(&tensor));
}

test "single tensor multiple retains" {
    const allocator = std.testing.allocator;

    const tensor_shape = [_]usize{ 4, 4 };
    var tensor = try Tensor.init(allocator, &tensor_shape);
    defer tensor.deinit();

    var i: usize = 0;
    while (i < 100) : (i += 1) {
        tensor.retain();
    }
    try std.testing.expectEqual(@as(usize, 101), getRefcount(&tensor));

    i = 0;
    while (i < 100) : (i += 1) {
        tensor.release();
    }
    try std.testing.expectEqual(@as(usize, 1), getRefcount(&tensor));
}

test "concurrent tensor stress small scale" {
    const allocator = std.testing.allocator;

    const config = TestConfig{
        .num_threads = 4,
        .ops_per_thread = 500,
        .num_tensors = 2,
    };

    var tensors = try allocator.alloc(Tensor, config.num_tensors);
    var initialized_count: usize = 0;
    errdefer {
        var j: usize = 0;
        while (j < initialized_count) : (j += 1) {
            tensors[j].deinit();
        }
        allocator.free(tensors);
    }

    const tensor_shape = [_]usize{ 16, 16 };
    var i: usize = 0;
    while (i < config.num_tensors) : (i += 1) {
        tensors[i] = try Tensor.init(allocator, &tensor_shape);
        initialized_count += 1;
    }

    var barrier = std.atomic.Value(usize).init(0);
    var threads = try allocator.alloc(Thread, config.num_threads);
    defer allocator.free(threads);

    var spawned_count: usize = 0;
    errdefer {
        barrier.store(config.num_threads, .seq_cst);
        var j: usize = 0;
        while (j < spawned_count) : (j += 1) {
            threads[j].join();
        }
    }

    var t: usize = 0;
    while (t < config.num_threads) : (t += 1) {
        const ctx = ThreadContext{
            .tensors = tensors,
            .thread_id = t,
            .barrier = &barrier,
            .total_threads = config.num_threads,
            .ops_per_thread = config.ops_per_thread,
        };
        threads[t] = try Thread.spawn(.{}, threadWorker, .{ctx});
        spawned_count += 1;
    }

    t = 0;
    while (t < config.num_threads) : (t += 1) {
        threads[t].join();
    }

    i = 0;
    while (i < config.num_tensors) : (i += 1) {
        const refcount = getRefcount(&tensors[i]);
        try std.testing.expectEqual(@as(usize, 1), refcount);
        tensors[i].deinit();
    }
    allocator.free(tensors);
}

