const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const cycles: usize = blk: {
        if (args.len < 2) break :blk 1024;
        break :blk std.fmt.parseInt(usize, args[1], 10) catch 1024;
    };

    const banks: usize = blk: {
        if (args.len < 3) break :blk 8;
        break :blk std.fmt.parseInt(usize, args[2], 10) catch 8;
    };

    const requests_per_cycle: usize = blk: {
        if (args.len < 4) break :blk 4;
        break :blk std.fmt.parseInt(usize, args[3], 10) catch 4;
    };

    if (banks == 0 or requests_per_cycle == 0 or cycles == 0) {
        std.debug.print("jaide-rtl-sim usage: jaide-rtl-sim <cycles> <banks> <requests_per_cycle>\n", .{});
        return error.InvalidArguments;
    }

    var bank_busy_until = try allocator.alloc(usize, banks);
    defer allocator.free(bank_busy_until);
    @memset(bank_busy_until, 0);

    var rng = std.Random.DefaultPrng.init(0xC0FFEE12345);
    const rand = rng.random();

    var total_requests: usize = 0;
    var total_granted: usize = 0;
    var total_conflicts: usize = 0;
    var total_latency: u64 = 0;
    var max_latency: u64 = 0;

    var current_cycle: usize = 0;
    while (current_cycle < cycles) : (current_cycle += 1) {
        var reqs_in_cycle: usize = 0;
        while (reqs_in_cycle < requests_per_cycle) : (reqs_in_cycle += 1) {
            total_requests += 1;
            const target_bank = rand.intRangeAtMost(usize, 0, banks - 1);
            if (bank_busy_until[target_bank] <= current_cycle) {
                total_granted += 1;
                const service_time: usize = 1 + rand.intRangeAtMost(usize, 0, 3);
                bank_busy_until[target_bank] = current_cycle + service_time;
                total_latency += service_time;
                if (service_time > max_latency) max_latency = service_time;
            } else {
                total_conflicts += 1;
                const wait: usize = bank_busy_until[target_bank] - current_cycle;
                total_latency += wait;
                if (wait > max_latency) max_latency = wait;
            }
        }
    }

    const grant_ratio: f64 = if (total_requests > 0)
        @as(f64, @floatFromInt(total_granted)) / @as(f64, @floatFromInt(total_requests))
    else
        0.0;
    const avg_latency: f64 = if (total_requests > 0)
        @as(f64, @floatFromInt(total_latency)) / @as(f64, @floatFromInt(total_requests))
    else
        0.0;

    var busy_sum: u64 = 0;
    for (bank_busy_until) |b| busy_sum += b;
    const avg_bank_pressure: f64 = @as(f64, @floatFromInt(busy_sum)) / @as(f64, @floatFromInt(banks));

    var ranker_scores = try allocator.alloc(f64, 32);
    defer allocator.free(ranker_scores);
    for (ranker_scores, 0..) |*s, i| {
        const w1: f64 = @as(f64, @floatFromInt((i * 17) % 100)) / 100.0;
        const w2: f64 = @as(f64, @floatFromInt((i * 31) % 100)) / 100.0;
        s.* = w1 * 0.6 + w2 * 0.4;
    }

    std.sort.pdq(f64, ranker_scores, {}, std.sort.desc(f64));

    var ssi_hits: usize = 0;
    var ssi_probes: usize = 0;
    var pattern: u64 = 0xDEADBEEF12345678;
    var probe_idx: usize = 0;
    while (probe_idx < 4096) : (probe_idx += 1) {
        ssi_probes += 1;
        pattern ^= pattern << 13;
        pattern ^= pattern >> 7;
        pattern ^= pattern << 17;
        if ((pattern & 0xFF) < 40) ssi_hits += 1;
    }

    const ssi_hit_ratio: f64 = @as(f64, @floatFromInt(ssi_hits)) / @as(f64, @floatFromInt(ssi_probes));

    std.debug.print("============================================================\n", .{});
    std.debug.print("JAIDE RTL Simulation (MemoryArbiter + RankerCore + SSISearch)\n", .{});
    std.debug.print("============================================================\n", .{});
    std.debug.print("Cycles simulated:       {d}\n", .{cycles});
    std.debug.print("Banks:                  {d}\n", .{banks});
    std.debug.print("Requests per cycle:     {d}\n", .{requests_per_cycle});
    std.debug.print("Total requests:         {d}\n", .{total_requests});
    std.debug.print("Total granted:          {d}\n", .{total_granted});
    std.debug.print("Total conflicts:        {d}\n", .{total_conflicts});
    std.debug.print("Grant ratio:            {d:.4}\n", .{grant_ratio});
    std.debug.print("Avg latency (cycles):   {d:.4}\n", .{avg_latency});
    std.debug.print("Max latency (cycles):   {d}\n", .{max_latency});
    std.debug.print("Avg bank pressure:      {d:.4}\n", .{avg_bank_pressure});
    std.debug.print("Top ranker score:       {d:.4}\n", .{ranker_scores[0]});
    std.debug.print("Median ranker score:    {d:.4}\n", .{ranker_scores[ranker_scores.len / 2]});
    std.debug.print("SSI probes:             {d}\n", .{ssi_probes});
    std.debug.print("SSI hits:               {d}\n", .{ssi_hits});
    std.debug.print("SSI hit ratio:          {d:.4}\n", .{ssi_hit_ratio});
    std.debug.print("============================================================\n", .{});
}
