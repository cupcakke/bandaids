const std = @import("std");
const GPUCoordinator = @import("distributed/gpu_coordinator.zig").GPUCoordinator;
const dtf = @import("distributed/distributed_trainer_futhark.zig");
const DistributedTrainerFuthark = dtf.DistributedTrainerFuthark;
const TrainerConfig = dtf.TrainerConfig;
const TrainerComponents = dtf.TrainerComponents;
const MGT = @import("tokenizer/mgt.zig").MGT;
const nccl = @import("distributed/nccl_bindings.zig");
const modal_gpu = @import("distributed/modal_gpu.zig");
const core_relational = @import("core_relational/mod.zig");
const accel_interface = @import("hw/accel/accel_interface.zig");

fn extractDatasetText(allocator: std.mem.Allocator, line: []const u8) !?[]const u8 {
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        line,
        .{ .allocate = .alloc_always },
    ) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return null,
    };
    defer parsed.deinit();

    return switch (parsed.value) {
        .object => |obj| blk: {
            const text_value = obj.get("text") orelse break :blk null;
            break :blk switch (text_value) {
                .string => |text| if (text.len > 0)
                    try allocator.dupe(u8, text)
                else
                    null,
                else => null,
            };
        },
        else => null,
    };
}

fn appendDatasetRange(
    allocator: std.mem.Allocator,
    dataset_path: []const u8,
    max_line_size: usize,
    start_valid_index: usize,
    end_valid_index: usize,
    samples: *std.ArrayList([]const u8),
) !usize {
    if (end_valid_index <= start_valid_index) return 0;

    const file = try std.fs.openFileAbsolute(dataset_path, .{ .mode = .read_only });
    defer file.close();

    var buffered_reader = std.io.bufferedReader(file.reader());
    var stream = buffered_reader.reader();
    var valid_index: usize = 0;
    var appended: usize = 0;

    while (try stream.readUntilDelimiterOrEofAlloc(allocator, '\n', max_line_size)) |line| {
        defer allocator.free(line);

        if (valid_index >= end_valid_index) break;

        const maybe_text = try extractDatasetText(allocator, line);
        if (maybe_text) |text| {
            if (valid_index >= start_valid_index and valid_index < end_valid_index) {
                try samples.append(text);
                appended = try std.math.add(usize, appended, 1);
            } else {
                allocator.free(text);
            }

            valid_index = try std.math.add(usize, valid_index, 1);
        }
    }

    const expected = end_valid_index - start_valid_index;
    if (appended != expected) {
        return error.DatasetSampleCountMismatch;
    }

    return appended;
}

fn fnv1aHashBytes(data: []const u8) u64 {
    var hash: u64 = 14695981039346656037;
    for (data) |byte| {
        hash ^= @as(u64, byte);
        hash *%= 1099511628211;
    }
    return hash;
}

fn loadDataset(
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    dataset_path: []const u8,
    max_line_size: usize,
) ![][]const u8 {
    if (coordinator.world_size == 0) return error.InvalidWorldSize;
    if (coordinator.rank >= coordinator.world_size) return error.InvalidRank;

    const env_total_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_TOTAL_SAMPLES",
    ) catch null;
    defer if (env_total_owned) |owned| allocator.free(owned);

    const env_max_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_MAX_SAMPLES",
    ) catch null;
    defer if (env_max_owned) |owned| allocator.free(owned);

    var valid_sample_count: usize = 0;

    if (env_total_owned) |value| {
        valid_sample_count = std.fmt.parseInt(usize, value, 10) catch 0;
    }

    if (valid_sample_count == 0) {
        std.debug.print(
            "[Rank {d}] WARN: JAIDE_TOTAL_SAMPLES not provided, scanning valid records\n",
            .{coordinator.rank},
        );

        const count_file = try std.fs.openFileAbsolute(
            dataset_path,
            .{ .mode = .read_only },
        );
        defer count_file.close();

        var buffered_reader = std.io.bufferedReader(count_file.reader());
        var stream = buffered_reader.reader();

        while (try stream.readUntilDelimiterOrEofAlloc(
            allocator,
            '\n',
            max_line_size,
        )) |line| {
            defer allocator.free(line);

            const maybe_text = extractDatasetText(allocator, line) catch |err| switch (err) {
                error.OutOfMemory => return err,
                else => null,
            };

            if (maybe_text) |text| {
                allocator.free(text);
                valid_sample_count = try std.math.add(
                    usize,
                    valid_sample_count,
                    1,
                );
            }
        }
    }

    if (env_max_owned) |value| {
        const maximum = std.fmt.parseInt(usize, value, 10) catch 0;
        if (maximum > 0 and maximum < valid_sample_count) {
            valid_sample_count = maximum;
        }
    }

    if (valid_sample_count == 0) {
        std.debug.print(
            "[Rank {d}] ERROR: dataset is empty or contains no valid records\n",
            .{coordinator.rank},
        );
        return error.EmptyDataset;
    }

    const rounded_count = try std.math.add(
        usize,
        valid_sample_count,
        coordinator.world_size - 1,
    );
    const samples_per_rank = rounded_count / coordinator.world_size;

    if (samples_per_rank == 0) {
        return error.EmptyDatasetPartition;
    }

    const unwrapped_start = try std.math.mul(
        usize,
        coordinator.rank,
        samples_per_rank,
    );
    const start_valid_index = unwrapped_start % valid_sample_count;
    const unwrapped_end = try std.math.add(
        usize,
        start_valid_index,
        samples_per_rank,
    );

    var samples = std.ArrayList([]const u8).init(allocator);
    errdefer {
        for (samples.items) |sample| {
            allocator.free(sample);
        }
        samples.deinit();
    }

    try samples.ensureTotalCapacity(samples_per_rank);

    const first_end = @min(unwrapped_end, valid_sample_count);
    const first_count = try appendDatasetRange(
        allocator,
        dataset_path,
        max_line_size,
        start_valid_index,
        first_end,
        &samples,
    );

    const remaining = samples_per_rank - first_count;

    if (remaining > 0) {
        if (remaining > valid_sample_count) {
            return error.InvalidDatasetPartition;
        }

        _ = try appendDatasetRange(
            allocator,
            dataset_path,
            max_line_size,
            0,
            remaining,
            &samples,
        );
    }

    if (samples.items.len != samples_per_rank) {
        std.debug.print(
            "[Rank {d}] ERROR: loaded {d} samples, expected {d}\n",
            .{
                coordinator.rank,
                samples.items.len,
                samples_per_rank,
            },
        );
        return error.DatasetSampleCountMismatch;
    }

    std.debug.print(
        "[Rank {d}] Loaded {d} samples from {d} valid records\n",
        .{
            coordinator.rank,
            samples.items.len,
            valid_sample_count,
        },
    );

    return samples.toOwnedSlice();
}

fn loadTokenizerDataset(
    allocator: std.mem.Allocator,
    dataset_path: []const u8,
    max_line_size: usize,
) ![][]const u8 {
    const env_max_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_MAX_SAMPLES",
    ) catch null;
    defer if (env_max_owned) |owned| allocator.free(owned);

    const maximum_samples: usize = if (env_max_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch 0
    else
        0;

    const file = try std.fs.openFileAbsolute(
        dataset_path,
        .{ .mode = .read_only },
    );
    defer file.close();

    var buffered_reader = std.io.bufferedReader(file.reader());
    var stream = buffered_reader.reader();
    var samples = std.ArrayList([]const u8).init(allocator);

    errdefer {
        for (samples.items) |sample| {
            allocator.free(sample);
        }
        samples.deinit();
    }

    while (try stream.readUntilDelimiterOrEofAlloc(
        allocator,
        '\n',
        max_line_size,
    )) |line| {
        defer allocator.free(line);

        const maybe_text = try extractDatasetText(allocator, line);
        if (maybe_text) |text| {
            try samples.append(text);

            if (maximum_samples > 0 and samples.items.len >= maximum_samples) {
                break;
            }
        }
    }

    if (samples.items.len == 0) {
        return error.EmptyDataset;
    }

    return samples.toOwnedSlice();
}

fn synchronizeStageStatus(
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    status_base_path: []const u8,
    stage_name: []const u8,
    local_error: ?anyerror,
) !void {
    const rank_status_path = try std.fmt.allocPrint(
        allocator,
        "{s}.{s}.rank_{d}.status",
        .{
            status_base_path,
            stage_name,
            coordinator.rank,
        },
    );
    defer allocator.free(rank_status_path);

    const global_status_path = try std.fmt.allocPrint(
        allocator,
        "{s}.{s}.global.status",
        .{
            status_base_path,
            stage_name,
        },
    );
    defer allocator.free(global_status_path);

    if (coordinator.isRoot()) {
        std.fs.deleteFileAbsolute(global_status_path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => {},
        };

        var cleanup_rank: usize = 0;
        while (cleanup_rank < coordinator.world_size) : (cleanup_rank += 1) {
            const cleanup_path = try std.fmt.allocPrint(
                allocator,
                "{s}.{s}.rank_{d}.status",
                .{
                    status_base_path,
                    stage_name,
                    cleanup_rank,
                },
            );
            defer allocator.free(cleanup_path);

            std.fs.deleteFileAbsolute(cleanup_path) catch |err| switch (err) {
                error.FileNotFound => {},
                else => {},
            };
        }
    }

    try coordinator.synchronize();

    var effective_local_error = local_error;

    {
        const status_file = std.fs.createFileAbsolute(
            rank_status_path,
            .{ .truncate = true },
        ) catch |err| status_file_failure: {
            effective_local_error = effective_local_error orelse err;
            break :status_file_failure null;
        };

        if (status_file) |file| {
            defer file.close();

            const status_text = if (effective_local_error == null) "ok" else "fail";
            file.writeAll(status_text) catch |err| {
                effective_local_error = effective_local_error orelse err;
            };
            file.sync() catch |err| {
                effective_local_error = effective_local_error orelse err;
            };
        }
    }

    try coordinator.synchronize();

    if (coordinator.isRoot()) {
        var all_succeeded = true;
        var inspected_rank: usize = 0;

        while (inspected_rank < coordinator.world_size) : (inspected_rank += 1) {
            const inspected_path = try std.fmt.allocPrint(
                allocator,
                "{s}.{s}.rank_{d}.status",
                .{
                    status_base_path,
                    stage_name,
                    inspected_rank,
                },
            );
            defer allocator.free(inspected_path);

            const status_file = std.fs.openFileAbsolute(
                inspected_path,
                .{ .mode = .read_only },
            ) catch {
                all_succeeded = false;
                continue;
            };
            defer status_file.close();

            var status_buffer: [4]u8 = undefined;
            const bytes_read = status_file.readAll(&status_buffer) catch {
                all_succeeded = false;
                continue;
            };

            if (!std.mem.eql(u8, status_buffer[0..bytes_read], "ok")) {
                all_succeeded = false;
            }
        }

        const global_file = std.fs.createFileAbsolute(
            global_status_path,
            .{ .truncate = true },
        ) catch null;

        if (global_file) |file| {
            defer file.close();
            file.writeAll(if (all_succeeded) "ok" else "fail") catch {};
            file.sync() catch {};
        }
    }

    try coordinator.synchronize();

    var global_succeeded = false;

    {
        const global_file = std.fs.openFileAbsolute(
            global_status_path,
            .{ .mode = .read_only },
        ) catch null;

        if (global_file) |file| {
            defer file.close();

            var status_buffer: [4]u8 = undefined;
            const bytes_read = file.readAll(&status_buffer) catch 0;
            global_succeeded = std.mem.eql(
                u8,
                status_buffer[0..bytes_read],
                "ok",
            );
        }
    }

    try coordinator.synchronize();

    if (coordinator.isRoot()) {
        std.fs.deleteFileAbsolute(global_status_path) catch {};

        var cleanup_rank: usize = 0;
        while (cleanup_rank < coordinator.world_size) : (cleanup_rank += 1) {
            const cleanup_path = std.fmt.allocPrint(
                allocator,
                "{s}.{s}.rank_{d}.status",
                .{
                    status_base_path,
                    stage_name,
                    cleanup_rank,
                },
            ) catch continue;
            defer allocator.free(cleanup_path);

            std.fs.deleteFileAbsolute(cleanup_path) catch {};
        }
    }

    if (!global_succeeded) {
        return effective_local_error orelse error.DistributedStageFailed;
    }
}

fn deployToModal(allocator: std.mem.Allocator, args: [][:0]u8) !void {
    const api_token = try std.process.getEnvVarOwned(
        allocator,
        "MODAL_API_TOKEN",
    );
    defer allocator.free(api_token);

    const model_path: []const u8 = if (args.len > 0)
        args[0]
    else
        "/checkpoints/latest";

    const dataset_path: []const u8 = if (args.len > 1)
        args[1]
    else
        "/data/dataset/train.jsonl";

    var client = try modal_gpu.ModalGPUClient.init(allocator, api_token);
    defer client.deinit();

    const job_id = try client.deployTrainingJob(model_path, dataset_path);
    defer allocator.free(job_id);

    std.debug.print("Deployed training job: {s}\n", .{job_id});

    const max_poll_attempts: usize = 360;
    var poll_attempt: usize = 0;

    while (poll_attempt < max_poll_attempts) : (poll_attempt += 1) {
        const status_raw = try client.getJobStatus(job_id);
        defer allocator.free(status_raw);

        std.debug.print("Job status raw: {s}\n", .{status_raw});

        const parsed_status = std.json.parseFromSlice(
            std.json.Value,
            allocator,
            status_raw,
            .{ .allocate = .alloc_always },
        ) catch null;

        if (parsed_status) |parsed| {
            defer parsed.deinit();

            const status_field: ?[]const u8 = switch (parsed.value) {
                .object => |object| blk: {
                    const status_value = object.get("status") orelse break :blk null;
                    break :blk switch (status_value) {
                        .string => |status| status,
                        else => null,
                    };
                },
                else => null,
            };

            if (status_field) |status| {
                std.debug.print("Job status: {s}\n", .{status});

                if (std.mem.eql(u8, status, "completed")) {
                    return;
                }

                if (std.mem.eql(u8, status, "failed")) {
                    return error.ModalJobFailed;
                }
            }
        }

        std.time.sleep(30 * std.time.ns_per_s);
    }

    std.debug.print(
        "Timeout waiting for Modal job completion after {d} polls\n",
        .{max_poll_attempts},
    );
    return error.ModalJobTimeout;
}

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = general_purpose_allocator.deinit();
    const allocator = general_purpose_allocator.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len > 1 and std.mem.eql(u8, args[1], "--deploy")) {
        return deployToModal(allocator, args[2..]);
    }

    const world_size_string = try std.process.getEnvVarOwned(
        allocator,
        "WORLD_SIZE",
    );
    defer allocator.free(world_size_string);

    const world_size = try std.fmt.parseInt(
        usize,
        world_size_string,
        10,
    );

    if (world_size == 0) {
        return error.InvalidWorldSize;
    }

    const rank_string = try std.process.getEnvVarOwned(
        allocator,
        "RANK",
    );
    defer allocator.free(rank_string);

    const rank = try std.fmt.parseInt(usize, rank_string, 10);

    if (rank >= world_size) {
        return error.InvalidRank;
    }

    var local_rank_string_owned: ?[]u8 = null;
    const local_rank: usize = blk: {
        local_rank_string_owned = std.process.getEnvVarOwned(
            allocator,
            "LOCAL_RANK",
        ) catch null;

        if (local_rank_string_owned) |owned| {
            break :blk try std.fmt.parseInt(usize, owned, 10);
        }

        std.debug.print(
            "[Rank {d}] WARN: LOCAL_RANK not set, using RANK ({d}) for device selection\n",
            .{ rank, rank },
        );
        break :blk rank;
    };
    defer if (local_rank_string_owned) |owned| allocator.free(owned);

    const master_addr = try std.process.getEnvVarOwned(
        allocator,
        "MASTER_ADDR",
    );
    defer allocator.free(master_addr);

    const master_port = try std.process.getEnvVarOwned(
        allocator,
        "MASTER_PORT",
    );
    defer allocator.free(master_port);

    std.debug.print(
        "============================================================\n",
        .{},
    );
    std.debug.print(
        "JAIDE v40 Distributed Training (Futhark GPU Acceleration)\n",
        .{},
    );
    std.debug.print(
        "============================================================\n",
        .{},
    );
    std.debug.print("Rank: {d}/{d}\n", .{ rank, world_size });
    std.debug.print(
        "Master addr/port: {s}:{s}\n",
        .{ master_addr, master_port },
    );
    std.debug.print(
        "============================================================\n\n",
        .{},
    );

    var nccl_id_path_owned: ?[]u8 = null;
    const nccl_id_path: []const u8 = blk: {
        nccl_id_path_owned = std.process.getEnvVarOwned(
            allocator,
            "JAIDE_NCCL_ID_PATH",
        ) catch null;
        break :blk nccl_id_path_owned orelse "/tmp/jaide_nccl_id";
    };
    defer if (nccl_id_path_owned) |owned| allocator.free(owned);

    std.debug.print(
        "[Rank {d}] NCCL ID exchange path: {s}\n",
        .{ rank, nccl_id_path },
    );

    const nccl_ready_path = try std.fmt.allocPrint(
        allocator,
        "{s}.ready",
        .{nccl_id_path},
    );
    defer allocator.free(nccl_ready_path);

    var nccl_id: nccl.ncclUniqueId = std.mem.zeroes(nccl.ncclUniqueId);

    if (world_size > 1) {
        if (rank == 0) {
            std.fs.deleteFileAbsolute(nccl_ready_path) catch |err| switch (err) {
                error.FileNotFound => {},
                else => return err,
            };

            std.fs.deleteFileAbsolute(nccl_id_path) catch |err| switch (err) {
                error.FileNotFound => {},
                else => return err,
            };

            const result = nccl.ncclGetUniqueId(&nccl_id);
            if (result != .ncclSuccess) {
                return error.NCCLGetUniqueIdFailed;
            }

            {
                const id_file = try std.fs.createFileAbsolute(
                    nccl_id_path,
                    .{ .truncate = true },
                );
                defer id_file.close();

                try id_file.writeAll(std.mem.asBytes(&nccl_id));
                try id_file.sync();
            }

            {
                const ready_file = try std.fs.createFileAbsolute(
                    nccl_ready_path,
                    .{ .truncate = true },
                );
                defer ready_file.close();

                try ready_file.writeAll("ready");
                try ready_file.sync();
            }

            std.debug.print(
                "[Rank 0] Generated NCCL ID at {s}\n",
                .{nccl_id_path},
            );
        } else {
            const maximum_attempts: usize = 3000;
            var attempts: usize = 0;

            while (attempts < maximum_attempts) : (attempts += 1) {
                const ready_file = std.fs.openFileAbsolute(
                    nccl_ready_path,
                    .{ .mode = .read_only },
                ) catch {
                    std.time.sleep(100 * std.time.ns_per_ms);
                    continue;
                };
                ready_file.close();
                break;
            }

            if (attempts >= maximum_attempts) {
                return error.NCCLIdTimeout;
            }

            const id_file = try std.fs.openFileAbsolute(
                nccl_id_path,
                .{ .mode = .read_only },
            );
            defer id_file.close();

            const bytes_read = try id_file.readAll(std.mem.asBytes(&nccl_id));
            if (bytes_read != @sizeOf(nccl.ncclUniqueId)) {
                return error.NCCLIdReadFailed;
            }

            std.debug.print(
                "[Rank {d}] Loaded NCCL ID from rank 0\n",
                .{rank},
            );
        }
    }

    var coordinator = try GPUCoordinator.init(
        allocator,
        world_size,
        rank,
        local_rank,
        nccl_id,
    );
    defer coordinator.deinit();

    std.debug.print(
        "[Rank {d}] GPU coordinator initialized\n",
        .{rank},
    );

    const model_dim_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_MODEL_DIM",
    ) catch null;
    defer if (model_dim_string_owned) |owned| allocator.free(owned);

    const model_dim: usize = if (model_dim_string_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_MODEL_DIM='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        2048;

    if (model_dim == 0) {
        return error.InvalidConfig;
    }

    const num_layers_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_LAYERS",
    ) catch null;
    defer if (num_layers_string_owned) |owned| allocator.free(owned);

    const num_layers: usize = if (num_layers_string_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_LAYERS='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        24;

    if (num_layers == 0) {
        return error.InvalidConfig;
    }

    const batch_size_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_BATCH_SIZE",
    ) catch null;
    defer if (batch_size_string_owned) |owned| allocator.free(owned);

    const local_batch_size: usize = if (batch_size_string_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_BATCH_SIZE='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        4;

    if (local_batch_size == 0) {
        return error.InvalidConfig;
    }

    const epochs_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_EPOCHS",
    ) catch null;
    defer if (epochs_string_owned) |owned| allocator.free(owned);

    const num_epochs: usize = if (epochs_string_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_EPOCHS='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        20;

    const learning_rate_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_LEARNING_RATE",
    ) catch null;
    defer if (learning_rate_string_owned) |owned| allocator.free(owned);

    const learning_rate: f32 = if (learning_rate_string_owned) |value|
        std.fmt.parseFloat(f32, value) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_LEARNING_RATE='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        0.0001;

    if (!std.math.isFinite(learning_rate) or learning_rate <= 0.0) {
        return error.InvalidConfig;
    }

    const reasoning_cycles_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_REASONING_CYCLES",
    ) catch null;
    defer if (reasoning_cycles_string_owned) |owned| allocator.free(owned);

    const reasoning_cycles: usize = if (reasoning_cycles_string_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_REASONING_CYCLES='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        1;

    if (reasoning_cycles == 0) {
        return error.InvalidConfig;
    }

    const relational_pass_interval_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_RELATIONAL_PASS_INTERVAL",
    ) catch null;
    defer if (relational_pass_interval_string_owned) |owned| allocator.free(owned);

    const relational_pass_interval: usize = if (relational_pass_interval_string_owned) |value|
        std.fmt.parseInt(usize, value, 10) catch |err| {
            std.debug.print(
                "[Rank {d}] ERROR: invalid JAIDE_RELATIONAL_PASS_INTERVAL='{s}': {}\n",
                .{ rank, value, err },
            );
            return error.InvalidConfig;
        }
    else
        50;

    if (relational_pass_interval == 0) {
        return error.InvalidConfig;
    }

    const dataset_path_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_DATASET",
    ) catch null;
    defer if (dataset_path_owned) |owned| allocator.free(owned);

    const dataset_path: []const u8 = dataset_path_owned orelse
        "/data/dataset/train.jsonl";

    std.debug.print(
        "[Rank {d}] Loading dataset from {s}\n",
        .{ rank, dataset_path },
    );

    const samples = try loadDataset(
        allocator,
        &coordinator,
        dataset_path,
        10 * 1024 * 1024,
    );
    defer {
        for (samples) |sample| {
            allocator.free(sample);
        }
        allocator.free(samples);
    }

    const vocab_path = "/checkpoints/tokenizer.vocab";

    const vocab_ready_string_owned: ?[]u8 = std.process.getEnvVarOwned(
        allocator,
        "JAIDE_VOCAB_READY",
    ) catch null;
    defer if (vocab_ready_string_owned) |owned| allocator.free(owned);

    const vocab_ready = if (vocab_ready_string_owned) |value|
        std.mem.eql(u8, value, "1")
    else
        false;

    if (!vocab_ready) {
        var vocabulary_error: ?anyerror = null;

        if (coordinator.isRoot()) {
            vocabulary_training: {
                const vocabulary_samples = loadTokenizerDataset(
                    allocator,
                    dataset_path,
                    10 * 1024 * 1024,
                ) catch |err| {
                    vocabulary_error = err;
                    break :vocabulary_training;
                };
                defer {
                    for (vocabulary_samples) |sample| {
                        allocator.free(sample);
                    }
                    allocator.free(vocabulary_samples);
                }

                var temporary_tokenizer = MGT.init(
                    allocator,
                    &.{},
                    &.{},
                    32000,
                    .english,
                ) catch |err| {
                    vocabulary_error = err;
                    break :vocabulary_training;
                };
                defer temporary_tokenizer.deinit();

                temporary_tokenizer.trainBPE(
                    vocabulary_samples,
                    32000,
                ) catch |err| {
                    vocabulary_error = err;
                    break :vocabulary_training;
                };

                std.fs.makeDirAbsolute("/checkpoints") catch |err| switch (err) {
                    error.PathAlreadyExists => {},
                    else => {
                        vocabulary_error = err;
                        break :vocabulary_training;
                    },
                };

                temporary_tokenizer.saveVocab(vocab_path) catch |err| {
                    vocabulary_error = err;
                    break :vocabulary_training;
                };

                std.debug.print(
                    "[Rank 0] Tokenizer trained on {d} records and saved to {s}\n",
                    .{
                        vocabulary_samples.len,
                        vocab_path,
                    },
                );
            }
        }

        synchronizeStageStatus(
            allocator,
            &coordinator,
            nccl_id_path,
            "vocabulary",
            vocabulary_error,
        ) catch |err| {
            std.debug.print(
                "[Rank {d}] vocabulary stage failed: {}\n",
                .{ rank, err },
            );
            if (@errorReturnTrace()) |trace| {
                std.debug.dumpStackTrace(trace.*);
            }
            return err;
        };
    } else {
        std.debug.print(
            "[Rank {d}] Reusing tokenizer vocabulary at {s}\n",
            .{ rank, vocab_path },
        );
    }

    var tokenizer = try MGT.init(
        allocator,
        &.{},
        &.{},
        32000,
        .english,
    );

    var trainer = trainer_initialization: {
        errdefer tokenizer.deinit();

        var tokenizer_load_error: ?anyerror = null;

        tokenizer.loadVocab(vocab_path) catch |err| {
            tokenizer_load_error = err;
        };

        synchronizeStageStatus(
            allocator,
            &coordinator,
            nccl_id_path,
            "tokenizer_load",
            tokenizer_load_error,
        ) catch |err| {
            std.debug.print(
                "[Rank {d}] tokenizer load failed: {}\n",
                .{ rank, err },
            );
            if (@errorReturnTrace()) |trace| {
                std.debug.dumpStackTrace(trace.*);
            }
            return err;
        };

        std.debug.print(
            "[Rank {d}] Tokenizer loaded, next_token_id={d}\n",
            .{ rank, tokenizer.next_token_id },
        );

        var trainer_config: TrainerConfig = .{};
        trainer_config.learning_rate = learning_rate;
        trainer_config.reasoning_cycles = reasoning_cycles;
        trainer_config.relational_pass_interval = relational_pass_interval;

        const components = TrainerComponents{
            .tokenizer = tokenizer,
        };

        break :trainer_initialization try DistributedTrainerFuthark.initWithComponents(
            allocator,
            &coordinator,
            model_dim,
            num_layers,
            local_batch_size,
            trainer_config,
            components,
        );
    };
    defer trainer.deinit();

    std.debug.print(
        "[Rank {d}] learning_rate={d}\n",
        .{ rank, learning_rate },
    );
    std.debug.print(
        "[Rank {d}] Futhark trainer initialized with model_dim={d}, layers={d}\n",
        .{
            rank,
            model_dim,
            num_layers,
        },
    );

    if (coordinator.isRoot()) {
        std.debug.print(
            "\n============================================================\n",
            .{},
        );
        std.debug.print(
            "Starting Futhark-accelerated training\n",
            .{},
        );
        std.debug.print(
            "Dataset: {d} samples per rank\n",
            .{samples.len},
        );
        std.debug.print(
            "Batch size: {d} per rank\n",
            .{local_batch_size},
        );
        std.debug.print("Epochs: {d}\n", .{num_epochs});
        std.debug.print(
            "============================================================\n\n",
            .{},
        );
    }

    var graph_stage_error: ?anyerror = null;

    graph_construction: {
        if (coordinator.isRoot()) {
            std.debug.print(
                "[Rank {d}] Knowledge graph construction: encoding {d} samples (GPU)...\n",
                .{ rank, samples.len },
            );
        }

        var sample_hashes = std.ArrayList(u64).init(allocator);
        defer sample_hashes.deinit();

        for (samples) |text| {
            if (text.len == 0) continue;
            const h = fnv1aHashBytes(std.mem.sliceAsBytes(text));
            sample_hashes.append(h) catch |err| {
                std.debug.print("[Rank {d}] graph-construction: sample_hashes.append (i={d}) failed: {}\n", .{ rank, sample_hashes.items.len, err });
                graph_stage_error = err;
                break :graph_construction;
            };
        }

        if (sample_hashes.items.len > 0) {
            const graph_ctx = &trainer.accelerator.ctx;

            var gpu_result = accel_interface.batchEncodeGraph(
                graph_ctx,
                sample_hashes.items,
                0,
                allocator,
            ) catch |err| {
                std.debug.print("[Rank {d}] graph-construction: batchEncodeGraph failed: {} (n={d} hashes)\n", .{ rank, err, sample_hashes.items.len });
                graph_stage_error = err;
                break :graph_construction;
            };
            defer gpu_result.deinit();

            trainer.nsir_graph.bulkImportFromGPU(
                gpu_result.hashes,
                gpu_result.re_a,
                gpu_result.im_a,
                gpu_result.re_b,
                gpu_result.im_b,
                gpu_result.edge_srcs,
                gpu_result.edge_tgts,
            ) catch |err| {
                std.debug.print("[Rank {d}] graph-construction: bulkImportFromGPU failed: {} (nodes={d} edges={d})\n", .{ rank, err, gpu_result.hashes.len, gpu_result.edge_srcs.len });
                graph_stage_error = err;
                break :graph_construction;
            };

            if (coordinator.isRoot()) {
                std.debug.print(
                    "[Rank {d}] Knowledge graph: {d} nodes encoded via GPU\n",
                    .{ rank, gpu_result.hashes.len },
                );
            }
        }

        trainer.signal_engine.propagateStep() catch |err| {
            std.debug.print("[Rank {d}] graph-construction: signal propagateStep failed: {}\n", .{ rank, err });
            graph_stage_error = err;
            break :graph_construction;
        };

        trainer.r_gpu.distributeGraph(
            trainer.nsir_graph,
        ) catch |err| {
            std.debug.print("[Rank {d}] graph-construction: distributeGraph failed: {}\n", .{ rank, err });
            graph_stage_error = err;
            break :graph_construction;
        };
    }

    synchronizeStageStatus(
        allocator,
        &coordinator,
        nccl_id_path,
        "knowledge_graph",
        graph_stage_error,
    ) catch |err| {
        std.debug.print(
            "[Rank {d}] knowledge graph stage failed: {}\n",
            .{ rank, err },
        );
        if (@errorReturnTrace()) |trace| {
            std.debug.dumpStackTrace(trace.*);
        }
        return err;
    };

    std.debug.print(
        "[Rank {d}] Knowledge graph populated and distributed\n",
        .{rank},
    );

    var loss_history = std.ArrayList(EpochMetric).init(allocator);
    defer loss_history.deinit();

    var checkpoint_failures: usize = 0;
    var metrics_failures: usize = 0;
    var epoch: usize = 0;

    while (epoch < num_epochs) : (epoch += 1) {
        var epoch_timer = try std.time.Timer.start();

        if (coordinator.isRoot()) {
            std.debug.print(
                "[Epoch {d}/{d}] Starting\n",
                .{ epoch + 1, num_epochs },
            );
        }

        const average_loss = trainer.trainEpoch(samples) catch |err| {
            std.debug.print(
                "[Rank {d}] trainEpoch ERROR at epoch {d}: {}\n",
                .{
                    rank,
                    epoch + 1,
                    err,
                },
            );
            if (@errorReturnTrace()) |trace| {
                std.debug.dumpStackTrace(trace.*);
            }
            return err;
        };

        const elapsed_nanoseconds = epoch_timer.read();
        const elapsed_seconds =
            @as(f64, @floatFromInt(elapsed_nanoseconds)) / 1.0e9;

        if (coordinator.isRoot()) {
            loss_history.append(.{
                .epoch = epoch + 1,
                .loss = average_loss,
                .time_s = elapsed_seconds,
            }) catch |err| {
                std.debug.print(
                    "[Rank 0] Failed to append loss metric: {}\n",
                    .{err},
                );
                metrics_failures += 1;
            };

            std.debug.print(
                "[Epoch {d}/{d}] Loss: {d:.6} | Time: {d:.2}s\n",
                .{
                    epoch + 1,
                    num_epochs,
                    average_loss,
                    elapsed_seconds,
                },
            );

            checkpoint_creation: {
                std.fs.makeDirAbsolute("/checkpoints") catch |err| switch (err) {
                    error.PathAlreadyExists => {},
                    else => {
                        std.debug.print(
                            "[Rank 0] Failed to create /checkpoints: {}\n",
                            .{err},
                        );
                        checkpoint_failures += 1;
                        break :checkpoint_creation;
                    },
                };

                var directory_buffer: [256]u8 = undefined;
                const directory_path = std.fmt.bufPrint(
                    &directory_buffer,
                    "/checkpoints/epoch_{d:0>3}",
                    .{epoch + 1},
                ) catch |err| {
                    std.debug.print(
                        "[Rank 0] Failed to format checkpoint directory: {}\n",
                        .{err},
                    );
                    checkpoint_failures += 1;
                    break :checkpoint_creation;
                };

                std.fs.makeDirAbsolute(directory_path) catch |err| switch (err) {
                    error.PathAlreadyExists => {},
                    else => {
                        std.debug.print(
                            "[Rank 0] Failed to create {s}: {}\n",
                            .{
                                directory_path,
                                err,
                            },
                        );
                        checkpoint_failures += 1;
                        break :checkpoint_creation;
                    },
                };

                var checkpoint_path_buffer: [256]u8 = undefined;
                const checkpoint_path = std.fmt.bufPrint(
                    &checkpoint_path_buffer,
                    "/checkpoints/epoch_{d:0>3}/model.ckpt",
                    .{epoch + 1},
                ) catch |err| {
                    std.debug.print(
                        "[Rank 0] Failed to format checkpoint path: {}\n",
                        .{err},
                    );
                    checkpoint_failures += 1;
                    break :checkpoint_creation;
                };

                trainer.saveCheckpoint(checkpoint_path) catch |err| {
                    std.debug.print(
                        "[Rank 0] Failed to save checkpoint: {}\n",
                        .{err},
                    );
                    if (@errorReturnTrace()) |trace| {
                        std.debug.dumpStackTrace(trace.*);
                    }
                    checkpoint_failures += 1;
                    break :checkpoint_creation;
                };

                std.debug.print(
                    "Checkpoint saved: {s}\n",
                    .{checkpoint_path},
                );
            }

            writeTrainingMetrics(
                allocator,
                loss_history.items,
                model_dim,
                num_layers,
                local_batch_size,
                learning_rate,
                samples.len,
                num_epochs,
            ) catch |err| {
                std.debug.print(
                    "[Rank 0] Failed to write training metrics: {}\n",
                    .{err},
                );
                metrics_failures += 1;
            };
        }

        try coordinator.synchronize();
    }

    if (coordinator.isRoot()) {
        std.debug.print(
            "\n============================================================\n",
            .{},
        );
        std.debug.print(
            "Futhark-accelerated training completed\n",
            .{},
        );
        std.debug.print(
            "Checkpoint failures: {d}\n",
            .{checkpoint_failures},
        );
        std.debug.print(
            "Metrics failures: {d}\n",
            .{metrics_failures},
        );
        std.debug.print(
            "============================================================\n",
            .{},
        );
    }
}

const EpochMetric = struct {
    epoch: usize,
    loss: f64,
    time_s: f64,
};

fn writeTrainingMetrics(
    allocator: std.mem.Allocator,
    epoch_metrics: []const EpochMetric,
    model_dim: usize,
    num_layers: usize,
    batch_size: usize,
    learning_rate: f64,
    sample_count: usize,
    planned_epochs: usize,
) !void {
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    const writer = buffer.writer();
    const safe_learning_rate: f64 = if (std.math.isFinite(learning_rate))
        learning_rate
    else
        0.0;

    try writer.print(
        "{{\n  \"model_dim\": {d},\n  \"num_layers\": {d},\n  \"batch_size\": {d},\n  \"learning_rate\": {d},\n  \"sample_count\": {d},\n  \"planned_epochs\": {d},\n  \"loss_curve\": [\n",
        .{
            model_dim,
            num_layers,
            batch_size,
            safe_learning_rate,
            sample_count,
            planned_epochs,
        },
    );

    for (epoch_metrics, 0..) |metric, index| {
        const safe_loss: f64 = if (std.math.isFinite(metric.loss))
            metric.loss
        else
            0.0;

        const safe_time: f64 = if (std.math.isFinite(metric.time_s))
            metric.time_s
        else
            0.0;

        try writer.print(
            "    {{ \"epoch\": {d}, \"loss\": {d:.6}, \"time_s\": {d:.2} }}{s}\n",
            .{
                metric.epoch,
                safe_loss,
                safe_time,
                if (index + 1 < epoch_metrics.len) "," else "",
            },
        );
    }

    try writer.print("  ]\n}}\n", .{});

    const temporary_path = "/checkpoints/training_metrics.json.tmp";
    const final_path = "/checkpoints/training_metrics.json";

    {
        const temporary_file = try std.fs.createFileAbsolute(
            temporary_path,
            .{ .truncate = true },
        );
        errdefer temporary_file.close();

        try temporary_file.writeAll(buffer.items);
        try temporary_file.sync();
        temporary_file.close();
    }

    std.fs.deleteFileAbsolute(final_path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };

    try std.fs.renameAbsolute(temporary_path, final_path);
}