const std = @import("std");
const nsir = @import("nsir_core.zig");
const ArrayList = std.ArrayListUnmanaged;
const Allocator = std.mem.Allocator;
const AutoHashMap = std.AutoHashMap;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const ChaosCoreConfig = struct {
    pub const BLOCK_ID_SIZE: usize = 16;
    pub const CONTENT_HASH_SIZE: usize = 16;
    pub const SHA256_DIGEST_SIZE: usize = 32;
    pub const ENTROPY_BITS_SHIFT: u5 = 24;
    pub const DEFAULT_BLOCK_SIZE: usize = 1048576;
    pub const OPTIMIZATION_THRESHOLD: f64 = 0.6;
    pub const LOAD_HIGH_THRESHOLD: f64 = 1.3;
    pub const LOAD_LOW_THRESHOLD: f64 = 0.7;
    pub const BALANCE_INTERVAL_CYCLES: usize = 100;
};

pub const SelfSimilarRelationalGraph = nsir.SelfSimilarRelationalGraph;
pub const Node = nsir.Node;
pub const Edge = nsir.Edge;
pub const EdgeQuality = nsir.EdgeQuality;

fn writeU64Little(out: []u8, v: u64) void {
    var x = v;
    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        out[i] = @as(u8, @truncate(x));
        x >>= 8;
    }
}

fn writeI128Little(out: []u8, v: i128) void {
    const ux: u128 = @bitCast(v);
    var x = ux;
    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        out[i] = @as(u8, @truncate(x));
        x >>= 8;
    }
}

fn bytesToHexLower(out: []u8, in: []const u8) void {
    const table = "0123456789abcdef";
    var i: usize = 0;
    while (i < in.len) : (i += 1) {
        const b = in[i];
        out[i * 2] = table[(b >> 4) & 0xF];
        out[i * 2 + 1] = table[b & 0xF];
    }
}

pub const MemoryBlockState = enum(u8) {
    free = 0,
    allocated = 1,
    entangled = 2,
    migrating = 3,

    pub fn toString(self: MemoryBlockState) []const u8 {
        return switch (self) {
            .free => "free",
            .allocated => "allocated",
            .entangled => "entangled",
            .migrating => "migrating",
        };
    }

    pub fn fromString(s: []const u8) ?MemoryBlockState {
        if (std.mem.eql(u8, s, "free")) return .free;
        if (std.mem.eql(u8, s, "allocated")) return .allocated;
        if (std.mem.eql(u8, s, "entangled")) return .entangled;
        if (std.mem.eql(u8, s, "migrating")) return .migrating;
        return null;
    }
};

pub const BlockIdContext = struct {
    pub fn hash(_: @This(), key: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&key));
        return hasher.final();
    }

    pub fn eql(_: @This(), a: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, b: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) bool {
        return std.mem.eql(u8, a[0..], b[0..]);
    }
};

pub const BlockIdSet = std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, void, BlockIdContext, std.hash_map.default_max_load_percentage);

pub const MemoryBlock = struct {
    block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
    content_hash: [ChaosCoreConfig.CONTENT_HASH_SIZE]u8,
    data: []u8,
    state: MemoryBlockState,
    affinity_core: ?usize,
    access_count: usize,
    last_access_time: i128,
    entangled_blocks: BlockIdSet,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
        content_hash: [ChaosCoreConfig.CONTENT_HASH_SIZE]u8,
        data: []const u8,
        preferred_core: ?usize,
    ) !MemoryBlock {
        return MemoryBlock{
            .block_id = block_id,
            .content_hash = content_hash,
            .data = try allocator.dupe(u8, data),
            .state = .allocated,
            .affinity_core = preferred_core,
            .access_count = 1,
            .last_access_time = std.time.nanoTimestamp(),
            .entangled_blocks = BlockIdSet.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MemoryBlock) void {
        self.allocator.free(self.data);
        self.entangled_blocks.deinit();
    }

    pub fn updateAccessTime(self: *MemoryBlock) void {
        self.access_count += 1;
        self.last_access_time = std.time.nanoTimestamp();
    }

    pub fn addEntangledBlock(self: *MemoryBlock, other_block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) !void {
        try self.entangled_blocks.put(other_block_id, {});
    }

    pub fn removeEntangledBlock(self: *MemoryBlock, other_block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) bool {
        return self.entangled_blocks.remove(other_block_id);
    }

    pub fn hasEntanglement(self: *const MemoryBlock, other_block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) bool {
        return self.entangled_blocks.contains(other_block_id);
    }

    pub fn entangledCount(self: *const MemoryBlock) usize {
        return self.entangled_blocks.count();
    }

    pub fn clone(self: *const MemoryBlock, allocator: Allocator) !MemoryBlock {
        var new_block = MemoryBlock{
            .block_id = self.block_id,
            .content_hash = self.content_hash,
            .data = try allocator.dupe(u8, self.data),
            .state = self.state,
            .affinity_core = self.affinity_core,
            .access_count = self.access_count,
            .last_access_time = self.last_access_time,
            .entangled_blocks = BlockIdSet.init(allocator),
            .allocator = allocator,
        };
        var iter = self.entangled_blocks.iterator();
        while (iter.next()) |entry| {
            try new_block.entangled_blocks.put(entry.key_ptr.*, {});
        }
        return new_block;
    }
};

pub const TaskDescriptor = struct {
    task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8,
    priority: i32,
    data_dependencies: ArrayList([ChaosCoreConfig.BLOCK_ID_SIZE]u8),
    estimated_cycles: usize,
    assigned_core: ?usize,
    completion_status: bool,
    start_time: i128,
    end_time: i128,
    inference_priority: f64,
    inference_context_id: ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8,
        priority: i32,
        estimated_cycles: usize,
    ) TaskDescriptor {
        return TaskDescriptor{
            .task_id = task_id,
            .priority = priority,
            .data_dependencies = .{},
            .estimated_cycles = estimated_cycles,
            .assigned_core = null,
            .completion_status = false,
            .start_time = 0,
            .end_time = 0,
            .inference_priority = 0.0,
            .inference_context_id = null,
            .allocator = allocator,
        };
    }

    pub fn initWithInferenceContext(
        allocator: Allocator,
        task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8,
        priority: i32,
        estimated_cycles: usize,
        inference_context_id: ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8,
    ) TaskDescriptor {
        const abs_priority: i32 = if (priority < 0) -priority else priority;
        return TaskDescriptor{
            .task_id = task_id,
            .priority = priority,
            .data_dependencies = .{},
            .estimated_cycles = estimated_cycles,
            .assigned_core = null,
            .completion_status = false,
            .start_time = 0,
            .end_time = 0,
            .inference_priority = @as(f64, @floatFromInt(@max(priority, 0))) / @as(f64, @floatFromInt(@max(abs_priority, 1))),
            .inference_context_id = inference_context_id,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TaskDescriptor) void {
        self.data_dependencies.deinit(self.allocator);
    }

    pub fn addDependency(self: *TaskDescriptor, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) !void {
        try self.data_dependencies.append(self.allocator, block_id);
    }

    pub fn getDuration(self: *const TaskDescriptor) i128 {
        if (self.completion_status and self.end_time > self.start_time) {
            return self.end_time - self.start_time;
        }
        return 0;
    }

    pub fn setInferenceContext(self: *TaskDescriptor, context_id: ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8) void {
        self.inference_context_id = context_id;
    }

    pub fn hasInferenceContext(self: *const TaskDescriptor) bool {
        return self.inference_context_id != null;
    }

    pub fn cloneWithAllocator(self: *const TaskDescriptor, new_allocator: Allocator) !TaskDescriptor {
        var new_task = TaskDescriptor{
            .task_id = self.task_id,
            .priority = self.priority,
            .data_dependencies = .{},
            .estimated_cycles = self.estimated_cycles,
            .assigned_core = self.assigned_core,
            .completion_status = self.completion_status,
            .start_time = self.start_time,
            .end_time = self.end_time,
            .inference_priority = self.inference_priority,
            .inference_context_id = self.inference_context_id,
            .allocator = new_allocator,
        };
        for (self.data_dependencies.items) |dep| {
            try new_task.data_dependencies.append(new_allocator, dep);
        }
        return new_task;
    }

    pub fn clone(self: *const TaskDescriptor, allocator: Allocator) !TaskDescriptor {
        return self.cloneWithAllocator(allocator);
    }
};

pub const StorageStatistics = struct {
    total_blocks: usize,
    entangled_blocks: usize,
    used_capacity: usize,
    total_capacity: usize,
    utilization: f64,
    unique_contents: usize,
    cores_with_affinity: usize,
};

pub const SchedulerStatistics = struct {
    pending_tasks: usize,
    active_tasks: usize,
    completed_tasks: usize,
    average_completion_time: f64,
};

pub const KernelStatistics = struct {
    cycle_count: usize,
    migration_count: usize,
    storage: StorageStatistics,
    scheduler: SchedulerStatistics,
};

pub const CoreState = enum(u8) {
    idle = 0,
    active = 1,
    power_gated = 2,
    stalled = 3,
};

pub const ProcessingCore = struct {
    core_id: usize,
    state: CoreState,
    cycles_active: usize,
    cycles_idle: usize,

    pub fn init(core_id: usize) ProcessingCore {
        return ProcessingCore{
            .core_id = core_id,
            .state = .idle,
            .cycles_active = 0,
            .cycles_idle = 0,
        };
    }

    pub fn tick(self: *ProcessingCore, is_active: bool) void {
        if (is_active) {
            self.cycles_active += 1;
            self.state = .active;
        } else {
            self.cycles_idle += 1;
            if (self.state != .power_gated) {
                self.state = .idle;
            }
        }
    }

    pub fn getWorkload(self: *const ProcessingCore) f64 {
        const total = self.cycles_active + self.cycles_idle;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.cycles_active)) / @as(f64, @floatFromInt(total));
    }
};

pub const ContentAddressableStorage = struct {
    storage: std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, MemoryBlock, BlockIdContext, std.hash_map.default_max_load_percentage),
    content_index: std.HashMap([ChaosCoreConfig.CONTENT_HASH_SIZE]u8, BlockIdSet, BlockIdContext, std.hash_map.default_max_load_percentage),
    affinity_map: AutoHashMap(usize, BlockIdSet),
    total_capacity: usize,
    used_capacity: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, total_capacity: usize) ContentAddressableStorage {
        return ContentAddressableStorage{
            .storage = std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, MemoryBlock, BlockIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .content_index = std.HashMap([ChaosCoreConfig.CONTENT_HASH_SIZE]u8, BlockIdSet, BlockIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .affinity_map = AutoHashMap(usize, BlockIdSet).init(allocator),
            .total_capacity = total_capacity,
            .used_capacity = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ContentAddressableStorage) void {
        var storage_iter = self.storage.iterator();
        while (storage_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.storage.deinit();

        var content_iter = self.content_index.iterator();
        while (content_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.content_index.deinit();

        var affinity_iter = self.affinity_map.iterator();
        while (affinity_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.affinity_map.deinit();
    }

    fn computeContentHash(data: []const u8) [ChaosCoreConfig.CONTENT_HASH_SIZE]u8 {
        var hash_out: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(data, &hash_out, .{});
        var result: [ChaosCoreConfig.CONTENT_HASH_SIZE]u8 = undefined;
        std.mem.copyForwards(u8, result[0..], hash_out[0..ChaosCoreConfig.CONTENT_HASH_SIZE]);
        return result;
    }

    fn computeBlockId(content_hash: [ChaosCoreConfig.CONTENT_HASH_SIZE]u8, timestamp: i128) [ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
        var buffer: [ChaosCoreConfig.CONTENT_HASH_SIZE + 16]u8 = undefined;
        std.mem.copyForwards(u8, buffer[0..ChaosCoreConfig.CONTENT_HASH_SIZE], content_hash[0..]);
        writeI128Little(buffer[ChaosCoreConfig.CONTENT_HASH_SIZE .. ChaosCoreConfig.CONTENT_HASH_SIZE + 16], timestamp);
        var hash_out: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(buffer[0..], &hash_out, .{});
        var result: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
        std.mem.copyForwards(u8, result[0..], hash_out[0..ChaosCoreConfig.BLOCK_ID_SIZE]);
        return result;
    }

    fn unlinkEntanglements(self: *ContentAddressableStorage, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, entangled: *const BlockIdSet) void {
        var it = entangled.iterator();
        while (it.next()) |e| {
            const other_id = e.key_ptr.*;
            if (self.storage.getPtr(other_id)) |other_block| {
                _ = other_block.removeEntangledBlock(block_id);
                if (other_block.entangledCount() == 0 and other_block.state == .entangled) {
                    other_block.state = .allocated;
                }
            }
        }
    }

    pub fn store(self: *ContentAddressableStorage, data: []const u8, preferred_core: ?usize) ![ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
        const content_hash = computeContentHash(data);

        if (self.content_index.getPtr(content_hash)) |existing_blocks| {
            var stale: ArrayList([ChaosCoreConfig.BLOCK_ID_SIZE]u8) = .{};
            defer stale.deinit(self.allocator);

            var iter = existing_blocks.iterator();
            while (iter.next()) |entry| {
                const block_id = entry.key_ptr.*;
                if (self.storage.getPtr(block_id)) |block| {
                    block.updateAccessTime();
                    return block_id;
                } else {
                    try stale.append(self.allocator, block_id);
                }
            }

            for (stale.items) |sid| {
                _ = existing_blocks.remove(sid);
            }
        }

        const timestamp = std.time.nanoTimestamp();
        const block_id = computeBlockId(content_hash, timestamp);
        const data_size = data.len;

        if (data_size > self.total_capacity) return error.OutOfMemory;

        if (self.used_capacity + data_size > self.total_capacity) {
            const need = (self.used_capacity + data_size) - self.total_capacity;
            try self.evictLeastUsed(need);
            if (self.used_capacity + data_size > self.total_capacity) return error.OutOfMemory;
        }

        var block = try MemoryBlock.init(self.allocator, block_id, content_hash, data, preferred_core);
        var inserted = false;
        errdefer {
            if (!inserted) {
                block.deinit();
            }
        }

        try self.storage.put(block_id, block);
        inserted = true;

        errdefer {
            if (self.storage.fetchRemove(block_id)) |removed| {
                var b = removed.value;
                self.unlinkEntanglements(block_id, &b.entangled_blocks);
                b.deinit();
            }
        }

        var ci = try self.content_index.getOrPut(content_hash);
        if (!ci.found_existing) {
            ci.value_ptr.* = BlockIdSet.init(self.allocator);
        }
        try ci.value_ptr.put(block_id, {});

        errdefer {
            if (self.content_index.getPtr(content_hash)) |set| {
                _ = set.remove(block_id);
                if (set.count() == 0) {
                    if (self.content_index.fetchRemove(content_hash)) |removed_set| {
                        var s = removed_set.value;
                        s.deinit();
                    }
                }
            }
        }

        if (preferred_core) |core_id| {
            var am = try self.affinity_map.getOrPut(core_id);
            if (!am.found_existing) {
                am.value_ptr.* = BlockIdSet.init(self.allocator);
            }
            try am.value_ptr.put(block_id, {});

            errdefer {
                if (self.affinity_map.getPtr(core_id)) |set| {
                    _ = set.remove(block_id);
                    if (set.count() == 0) {
                        if (self.affinity_map.fetchRemove(core_id)) |removed_set| {
                            var s = removed_set.value;
                            s.deinit();
                        }
                    }
                }
            }
        }

        self.used_capacity += data_size;
        return block_id;
    }

    pub fn retrieve(self: *ContentAddressableStorage, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) ?[]const u8 {
        if (self.storage.getPtr(block_id)) |block| {
            block.updateAccessTime();
            return block.data;
        }
        return null;
    }

    pub fn retrieveByContent(self: *ContentAddressableStorage, data: []const u8) ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
        const content_hash = computeContentHash(data);
        if (self.content_index.getPtr(content_hash)) |block_set| {
            var iter = block_set.iterator();
            while (iter.next()) |entry| {
                const id = entry.key_ptr.*;
                if (self.storage.contains(id)) return id;
            }
        }
        return null;
    }

    pub fn entangleBlocks(self: *ContentAddressableStorage, block_id1: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, block_id2: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) !bool {
        const block1_ptr = self.storage.getPtr(block_id1);
        const block2_ptr = self.storage.getPtr(block_id2);

        if (block1_ptr == null or block2_ptr == null) {
            return false;
        }

        if (std.mem.eql(u8, block_id1[0..], block_id2[0..])) return false;

        try block1_ptr.?.addEntangledBlock(block_id2);
        try block2_ptr.?.addEntangledBlock(block_id1);
        block1_ptr.?.state = .entangled;
        block2_ptr.?.state = .entangled;
        return true;
    }

    pub fn findNearestCore(
        self: *ContentAddressableStorage,
        block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
        cores: *AutoHashMap(usize, ProcessingCore),
    ) ?usize {
        const block_ptr = self.storage.getPtr(block_id);
        if (block_ptr == null) {
            return null;
        }
        const block = block_ptr.?;

        if (block.affinity_core) |core| {
            if (cores.get(core)) |core_info| {
                if (core_info.state != .power_gated) {
                    return @as(?usize, core);
                }
            }
        }

        var min_distance: f64 = std.math.inf(f64);
        var nearest_core: ?usize = null;

        var core_iter = cores.iterator();
        while (core_iter.next()) |core_entry| {
            const core_id = core_entry.key_ptr.*;
            const core_info = core_entry.value_ptr.*;

            if (core_info.state == .power_gated) {
                continue;
            }

            var entangled_count: usize = 0;
            if (self.affinity_map.getPtr(core_id)) |core_blocks| {
                var entangle_iter = block.entangled_blocks.iterator();
                while (entangle_iter.next()) |ent_entry| {
                    if (core_blocks.contains(ent_entry.key_ptr.*)) {
                        entangled_count += 1;
                    }
                }
            }

            const distance = 1.0 / (1.0 + @as(f64, @floatFromInt(entangled_count)));
            if (distance < min_distance) {
                min_distance = distance;
                nearest_core = core_id;
            }
        }

        return nearest_core;
    }

    pub fn migrateBlock(self: *ContentAddressableStorage, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, target_core: usize) !bool {
        const block_ptr = self.storage.getPtr(block_id);
        if (block_ptr == null) {
            return false;
        }
        const block = block_ptr.?;

        if (block.affinity_core) |old_core| {
            if (self.affinity_map.getPtr(old_core)) |old_set| {
                _ = old_set.remove(block_id);
                if (old_set.count() == 0) {
                    if (self.affinity_map.fetchRemove(old_core)) |removed_set| {
                        var s = removed_set.value;
                        s.deinit();
                    }
                }
            }
        }

        block.state = .migrating;
        block.affinity_core = @as(?usize, target_core);

        var result = try self.affinity_map.getOrPut(target_core);
        if (!result.found_existing) {
            result.value_ptr.* = BlockIdSet.init(self.allocator);
        }
        try result.value_ptr.put(block_id, {});

        block.state = .allocated;
        return true;
    }

    const EvictionEntry = struct {
        id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
        access_count: usize,
        last_access: i128,
        size: usize,
        entangled: bool,

        fn lessThan(_: void, a: @This(), b: @This()) bool {
            if (a.entangled != b.entangled) return !a.entangled and b.entangled;
            if (a.access_count != b.access_count) return a.access_count < b.access_count;
            return a.last_access < b.last_access;
        }
    };

    pub fn evictLeastUsed(self: *ContentAddressableStorage, required_free: usize) !void {
        if (required_free == 0) return;

        var blocks_to_sort: ArrayList(EvictionEntry) = .{};
        defer blocks_to_sort.deinit(self.allocator);

        var iter = self.storage.iterator();
        while (iter.next()) |entry| {
            const block = entry.value_ptr.*;
            try blocks_to_sort.append(self.allocator, .{
                .id = block.block_id,
                .access_count = block.access_count,
                .last_access = block.last_access_time,
                .size = block.data.len,
                .entangled = block.state == .entangled,
            });
        }

        std.mem.sort(EvictionEntry, blocks_to_sort.items, {}, EvictionEntry.lessThan);

        var freed: usize = 0;
        var pass: usize = 0;
        while (freed < required_free) : (pass += 1) {
            if (pass > 1) return error.OutOfMemory;

            for (blocks_to_sort.items) |block_info| {
                if (freed >= required_free) break;
                if (pass == 0 and block_info.entangled) continue;
                if (self.removeBlock(block_info.id)) |removed_size| {
                    freed += removed_size;
                }
            }
        }
    }

    pub fn removeBlock(self: *ContentAddressableStorage, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) ?usize {
        if (self.storage.fetchRemove(block_id)) |removed| {
            var block = removed.value;

            self.unlinkEntanglements(block_id, &block.entangled_blocks);

            if (self.content_index.getPtr(block.content_hash)) |block_set| {
                _ = block_set.remove(block_id);
                if (block_set.count() == 0) {
                    if (self.content_index.fetchRemove(block.content_hash)) |rs| {
                        var set = rs.value;
                        set.deinit();
                    }
                }
            }

            if (block.affinity_core) |core_id| {
                if (self.affinity_map.getPtr(core_id)) |affinity_set| {
                    _ = affinity_set.remove(block_id);
                    if (affinity_set.count() == 0) {
                        if (self.affinity_map.fetchRemove(core_id)) |removed_set| {
                            var s = removed_set.value;
                            s.deinit();
                        }
                    }
                }
            }

            if (self.used_capacity >= block.data.len) {
                self.used_capacity -= block.data.len;
            } else {
                self.used_capacity = 0;
            }

            const sz = block.data.len;
            block.deinit();
            return sz;
        }
        return null;
    }

    pub fn getStatistics(self: *const ContentAddressableStorage) StorageStatistics {
        var entangled_count: usize = 0;
        var iter = self.storage.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.*.state == .entangled) {
                entangled_count += 1;
            }
        }

        const utilization: f64 = if (self.total_capacity > 0)
            @as(f64, @floatFromInt(self.used_capacity)) / @as(f64, @floatFromInt(self.total_capacity))
        else
            0.0;

        return StorageStatistics{
            .total_blocks = self.storage.count(),
            .entangled_blocks = entangled_count,
            .used_capacity = self.used_capacity,
            .total_capacity = self.total_capacity,
            .utilization = utilization,
            .unique_contents = self.content_index.count(),
            .cores_with_affinity = self.affinity_map.count(),
        };
    }

    pub fn getBlock(self: *ContentAddressableStorage, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) ?*MemoryBlock {
        return self.storage.getPtr(block_id);
    }

    pub fn containsBlock(self: *const ContentAddressableStorage, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) bool {
        return self.storage.contains(block_id);
    }
};

const TaskIdContext = struct {
    pub fn hash(_: @This(), key: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&key));
        return hasher.final();
    }

    pub fn eql(_: @This(), a: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8, b: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) bool {
        return std.mem.eql(u8, a[0..], b[0..]);
    }
};

pub const DynamicTaskScheduler = struct {
    pending_tasks: std.PriorityQueue(TaskDescriptor, void, compareTasks),
    active_tasks: std.HashMap([ChaosCoreConfig.SHA256_DIGEST_SIZE]u8, TaskDescriptor, TaskIdContext, std.hash_map.default_max_load_percentage),
    completed_tasks: ArrayList(TaskDescriptor),
    task_counter: u64,
    allocator: Allocator,

    fn compareTasks(_: void, a: TaskDescriptor, b: TaskDescriptor) std.math.Order {
        if (a.priority > b.priority) return .lt;
        if (a.priority < b.priority) return .gt;
        const oa = std.mem.asBytes(&a.task_id);
        const ob = std.mem.asBytes(&b.task_id);
        const ord = std.mem.order(u8, oa, ob);
        return switch (ord) {
            .lt => .lt,
            .eq => .eq,
            .gt => .gt,
        };
    }

    pub fn init(allocator: Allocator) DynamicTaskScheduler {
        return DynamicTaskScheduler{
            .pending_tasks = std.PriorityQueue(TaskDescriptor, void, compareTasks).init(allocator, {}),
            .active_tasks = std.HashMap([ChaosCoreConfig.SHA256_DIGEST_SIZE]u8, TaskDescriptor, TaskIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .completed_tasks = .{},
            .task_counter = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DynamicTaskScheduler) void {
        while (self.pending_tasks.removeOrNull()) |task_val| {
            var task = task_val;
            task.deinit();
        }
        self.pending_tasks.deinit();

        var active_iter = self.active_tasks.iterator();
        while (active_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.active_tasks.deinit();

        for (self.completed_tasks.items) |*task| {
            task.deinit();
        }
        self.completed_tasks.deinit(self.allocator);
    }

    fn generateTaskId(self: *DynamicTaskScheduler) [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 {
        var buffer: [8 + 16]u8 = undefined;
        writeU64Little(buffer[0..8], self.task_counter);
        writeI128Little(buffer[8..24], std.time.nanoTimestamp());
        var hash_out: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
        Sha256.hash(buffer[0..], &hash_out, .{});
        self.task_counter += 1;
        return hash_out;
    }

    pub fn submitTask(
        self: *DynamicTaskScheduler,
        priority: i32,
        data_dependencies: []const [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
        estimated_cycles: usize,
    ) ![ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 {
        const task_id = self.generateTaskId();
        var task = TaskDescriptor.init(self.allocator, task_id, priority, estimated_cycles);
        errdefer task.deinit();

        for (data_dependencies) |dep| {
            try task.addDependency(dep);
        }

        try self.pending_tasks.add(task);
        return task_id;
    }

    pub fn scheduleTask(
        self: *DynamicTaskScheduler,
        cores: *AutoHashMap(usize, ProcessingCore),
        storage: *ContentAddressableStorage,
    ) !?struct { task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8, core_id: usize } {
        const task_opt = self.pending_tasks.removeOrNull();
        if (task_opt == null) {
            return null;
        }
        var task = task_opt.?;

        var core_scores = AutoHashMap(usize, f64).init(self.allocator);
        defer core_scores.deinit();

        var core_iter = cores.iterator();
        while (core_iter.next()) |core_entry| {
            const core_id = core_entry.key_ptr.*;
            const core_info = core_entry.value_ptr.*;

            if (core_info.state == .power_gated) continue;

            var score: f64 = 0.0;
            for (task.data_dependencies.items) |dep_id| {
                if (storage.containsBlock(dep_id)) {
                    const nearest = storage.findNearestCore(dep_id, cores);
                    if (nearest) |nearest_core| {
                        if (nearest_core == core_id) {
                            score += 10.0;
                        } else {
                            score += 1.0;
                        }
                    }
                }
            }

            const workload = core_info.getWorkload();
            score += (1.0 - workload) * 5.0;

            try core_scores.put(core_id, score);
        }

        if (core_scores.count() == 0) {
            try self.pending_tasks.add(task);
            return null;
        }

        var best_core: usize = 0;
        var best_score: f64 = -std.math.inf(f64);
        var score_iter = core_scores.iterator();
        while (score_iter.next()) |entry| {
            const s = entry.value_ptr.*;
            if (s > best_score) {
                best_score = s;
                best_core = entry.key_ptr.*;
            }
        }

        task.assigned_core = @as(?usize, best_core);
        task.start_time = std.time.nanoTimestamp();

        const task_id = task.task_id;
        try self.active_tasks.put(task_id, task);

        return .{ .task_id = task_id, .core_id = best_core };
    }

    pub fn completeTask(self: *DynamicTaskScheduler, task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) !bool {
        if (self.active_tasks.fetchRemove(task_id)) |removed| {
            var task = removed.value;
            task.completion_status = true;
            task.end_time = std.time.nanoTimestamp();
            try self.completed_tasks.append(self.allocator, task);
            return true;
        }
        return false;
    }

    pub fn getStatistics(self: *const DynamicTaskScheduler) SchedulerStatistics {
        var avg_completion_time: f64 = 0.0;
        if (self.completed_tasks.items.len > 0) {
            var total_time: i128 = 0;
            for (self.completed_tasks.items) |task| {
                total_time += task.getDuration();
            }
            avg_completion_time = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(self.completed_tasks.items.len));
        }

        return SchedulerStatistics{
            .pending_tasks = self.pending_tasks.count(),
            .active_tasks = self.active_tasks.count(),
            .completed_tasks = self.completed_tasks.items.len,
            .average_completion_time = avg_completion_time,
        };
    }

    pub fn getActiveTask(self: *DynamicTaskScheduler, task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) ?*TaskDescriptor {
        return self.active_tasks.getPtr(task_id);
    }
};

const FlowEdgeKey = struct {
    source: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
    target: [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
};

const FlowEdgeKeyContext = struct {
    pub fn hash(_: @This(), key: FlowEdgeKey) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&key.source));
        hasher.update(std.mem.asBytes(&key.target));
        return hasher.final();
    }

    pub fn eql(_: @This(), a: FlowEdgeKey, b: FlowEdgeKey) bool {
        return std.mem.eql(u8, a.source[0..], b.source[0..]) and std.mem.eql(u8, a.target[0..], b.target[0..]);
    }
};

pub const AccessRecord = struct {
    timestamp: i128,
    core_id: usize,
};

pub const DataFlowAnalyzer = struct {
    flow_graph: std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, BlockIdSet, BlockIdContext, std.hash_map.default_max_load_percentage),
    flow_weights: std.HashMap(FlowEdgeKey, f64, FlowEdgeKeyContext, std.hash_map.default_max_load_percentage),
    access_patterns: std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, ArrayList(AccessRecord), BlockIdContext, std.hash_map.default_max_load_percentage),
    allocator: Allocator,

    pub fn init(allocator: Allocator) DataFlowAnalyzer {
        return DataFlowAnalyzer{
            .flow_graph = std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, BlockIdSet, BlockIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .flow_weights = std.HashMap(FlowEdgeKey, f64, FlowEdgeKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
            .access_patterns = std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, ArrayList(AccessRecord), BlockIdContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DataFlowAnalyzer) void {
        var flow_iter = self.flow_graph.iterator();
        while (flow_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.flow_graph.deinit();

        self.flow_weights.deinit();

        var access_iter = self.access_patterns.iterator();
        while (access_iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.access_patterns.deinit();
    }

    pub fn recordAccess(self: *DataFlowAnalyzer, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, core_id: usize) !void {
        var result = try self.access_patterns.getOrPut(block_id);
        if (!result.found_existing) {
            result.value_ptr.* = .{};
        }
        try result.value_ptr.append(self.allocator, .{
            .timestamp = std.time.nanoTimestamp(),
            .core_id = core_id,
        });
    }

    pub fn analyzeFlow(self: *const DataFlowAnalyzer, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, allocator: Allocator) !AutoHashMap(usize, f64) {
        var core_affinities = AutoHashMap(usize, f64).init(allocator);

        const patterns_ptr = self.access_patterns.getPtr(block_id);
        if (patterns_ptr == null) {
            return core_affinities;
        }
        const accesses = patterns_ptr.?.items;

        if (accesses.len < 2) {
            return core_affinities;
        }

        var core_frequencies = AutoHashMap(usize, usize).init(allocator);
        defer core_frequencies.deinit();

        for (accesses) |access| {
            const entry = try core_frequencies.getOrPut(access.core_id);
            if (!entry.found_existing) {
                entry.value_ptr.* = 0;
            }
            entry.value_ptr.* += 1;
        }

        const total_accesses = @as(f64, @floatFromInt(accesses.len));
        var freq_iter = core_frequencies.iterator();
        while (freq_iter.next()) |entry| {
            const affinity = @as(f64, @floatFromInt(entry.value_ptr.*)) / total_accesses;
            try core_affinities.put(entry.key_ptr.*, affinity);
        }

        return core_affinities;
    }

    pub fn buildFlowGraph(self: *DataFlowAnalyzer, tasks: []const TaskDescriptor) !void {
        for (tasks) |task| {
            const deps = task.data_dependencies.items;
            var i: usize = 0;
            while (i < deps.len) : (i += 1) {
                const dep1 = deps[i];
                for (deps[i + 1 ..]) |dep2| {
                    var result1 = try self.flow_graph.getOrPut(dep1);
                    if (!result1.found_existing) {
                        result1.value_ptr.* = BlockIdSet.init(self.allocator);
                    }
                    try result1.value_ptr.put(dep2, {});

                    var result2 = try self.flow_graph.getOrPut(dep2);
                    if (!result2.found_existing) {
                        result2.value_ptr.* = BlockIdSet.init(self.allocator);
                    }
                    try result2.value_ptr.put(dep1, {});

                    var sorted_key: FlowEdgeKey = undefined;
                    if (std.mem.order(u8, dep1[0..], dep2[0..]) == .lt) {
                        sorted_key = .{ .source = dep1, .target = dep2 };
                    } else {
                        sorted_key = .{ .source = dep2, .target = dep1 };
                    }

                    const weight_entry = try self.flow_weights.getOrPut(sorted_key);
                    if (!weight_entry.found_existing) {
                        weight_entry.value_ptr.* = 0.0;
                    }
                    weight_entry.value_ptr.* += 1.0;
                }
            }
        }
    }

    pub fn getCorrelatedBlocks(self: *const DataFlowAnalyzer, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, threshold: f64, allocator: Allocator) !BlockIdSet {
        var correlated = BlockIdSet.init(allocator);

        const neighbors_ptr = self.flow_graph.getPtr(block_id);
        if (neighbors_ptr == null) {
            return correlated;
        }

        var neighbor_iter = neighbors_ptr.?.iterator();
        while (neighbor_iter.next()) |neighbor_entry| {
            const neighbor_id = neighbor_entry.key_ptr.*;

            var sorted_key: FlowEdgeKey = undefined;
            if (std.mem.order(u8, block_id[0..], neighbor_id[0..]) == .lt) {
                sorted_key = .{ .source = block_id, .target = neighbor_id };
            } else {
                sorted_key = .{ .source = neighbor_id, .target = block_id };
            }

            if (self.flow_weights.get(sorted_key)) |weight| {
                if (weight >= threshold) {
                    try correlated.put(neighbor_id, {});
                }
            }
        }

        return correlated;
    }
};

pub const ChaosCoreKernel = struct {
    storage: ContentAddressableStorage,
    scheduler: DynamicTaskScheduler,
    flow_analyzer: DataFlowAnalyzer,
    cores: AutoHashMap(usize, ProcessingCore),
    cycle_count: usize,
    migration_count: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) ChaosCoreKernel {
        return ChaosCoreKernel{
            .storage = ContentAddressableStorage.init(allocator, ChaosCoreConfig.DEFAULT_BLOCK_SIZE),
            .scheduler = DynamicTaskScheduler.init(allocator),
            .flow_analyzer = DataFlowAnalyzer.init(allocator),
            .cores = AutoHashMap(usize, ProcessingCore).init(allocator),
            .cycle_count = 0,
            .migration_count = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ChaosCoreKernel) void {
        self.storage.deinit();
        self.scheduler.deinit();
        self.flow_analyzer.deinit();
        self.cores.deinit();
    }

    pub fn addCore(self: *ChaosCoreKernel, core_id: usize) !void {
        try self.cores.put(core_id, ProcessingCore.init(core_id));
    }

    pub fn removeCore(self: *ChaosCoreKernel, core_id: usize) bool {
        if (self.cores.fetchRemove(core_id)) |_| {
            if (self.storage.affinity_map.fetchRemove(core_id)) |entry| {
                var set = entry.value;
                set.deinit();
            }
            return true;
        }
        return false;
    }

    pub fn setCoreState(self: *ChaosCoreKernel, core_id: usize, state: CoreState) bool {
        if (self.cores.getPtr(core_id)) |core| {
            core.state = state;
            return true;
        }
        return false;
    }

    pub fn allocateMemory(self: *ChaosCoreKernel, data: []const u8, preferred_core: ?usize) ![ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
        return self.storage.store(data, preferred_core);
    }

    pub fn readMemory(self: *ChaosCoreKernel, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) ?[]const u8 {
        return self.storage.retrieve(block_id);
    }

    pub fn createTask(
        self: *ChaosCoreKernel,
        priority: i32,
        data_dependencies: []const [ChaosCoreConfig.BLOCK_ID_SIZE]u8,
        estimated_cycles: usize,
    ) ![ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 {
        const task_id = try self.scheduler.submitTask(priority, data_dependencies, estimated_cycles);

        var tmp = TaskDescriptor.init(self.allocator, task_id, priority, estimated_cycles);
        defer tmp.deinit();
        for (data_dependencies) |dep| {
            try tmp.addDependency(dep);
        }
        var slice = [_]TaskDescriptor{tmp};
        try self.flow_analyzer.buildFlowGraph(slice[0..]);

        return task_id;
    }

    pub fn finishTask(self: *ChaosCoreKernel, task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) !bool {
        return self.scheduler.completeTask(task_id);
    }

    pub fn entangleData(self: *ChaosCoreKernel, block_id1: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, block_id2: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) !bool {
        const success = try self.storage.entangleBlocks(block_id1, block_id2);
        if (success) {
            var correlated = try self.flow_analyzer.getCorrelatedBlocks(block_id1, 0.5, self.allocator);
            defer correlated.deinit();

            var corr_iter = correlated.iterator();
            while (corr_iter.next()) |corr_entry| {
                const correlated_id = corr_entry.key_ptr.*;
                if (!std.mem.eql(u8, correlated_id[0..], block_id2[0..])) {
                    _ = try self.storage.entangleBlocks(block_id2, correlated_id);
                }
            }
        }
        return success;
    }

    pub fn getKernelStatistics(self: *const ChaosCoreKernel) KernelStatistics {
        return KernelStatistics{
            .cycle_count = self.cycle_count,
            .migration_count = self.migration_count,
            .storage = self.storage.getStatistics(),
            .scheduler = self.scheduler.getStatistics(),
        };
    }

    pub fn executeCycle(self: *ChaosCoreKernel) !void {
        self.cycle_count += 1;

        const scheduled = try self.scheduler.scheduleTask(&self.cores, &self.storage);
        var scheduled_core_id: ?usize = null;

        if (scheduled) |sched_info| {
            scheduled_core_id = sched_info.core_id;
            if (self.scheduler.getActiveTask(sched_info.task_id)) |task| {
                for (task.data_dependencies.items) |dep_id| {
                    try self.flow_analyzer.recordAccess(dep_id, sched_info.core_id);
                }
            }
        }

        var core_iter = self.cores.iterator();
        while (core_iter.next()) |core_entry| {
            const core = core_entry.value_ptr;
            const is_active = if (scheduled_core_id) |cid| core.core_id == cid else false;
            core.tick(is_active);
        }

        try self.optimizeDataPlacement();

        if (self.cycle_count % ChaosCoreConfig.BALANCE_INTERVAL_CYCLES == 0) {
            try self.balanceLoad();
        }
    }

    fn optimizeDataPlacement(self: *ChaosCoreKernel) !void {
        const Migration = struct { id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, target: usize };
        var blocks_to_migrate: ArrayList(Migration) = .{};
        defer blocks_to_migrate.deinit(self.allocator);

        var storage_iter = self.storage.storage.iterator();
        while (storage_iter.next()) |entry| {
            const block = entry.value_ptr.*;
            if (block.state == .migrating) continue;

            var affinities = try self.flow_analyzer.analyzeFlow(block.block_id, self.allocator);
            defer affinities.deinit();

            if (affinities.count() == 0) continue;

            var best_core: usize = 0;
            var best_affinity: f64 = -1.0;
            var affinity_iter = affinities.iterator();
            while (affinity_iter.next()) |aff_entry| {
                const a = aff_entry.value_ptr.*;
                if (a > best_affinity) {
                    best_affinity = a;
                    best_core = aff_entry.key_ptr.*;
                }
            }

            const current_core = block.affinity_core orelse continue;
            if (current_core != best_core and best_affinity > ChaosCoreConfig.OPTIMIZATION_THRESHOLD) {
                try blocks_to_migrate.append(self.allocator, .{ .id = block.block_id, .target = best_core });
            }
        }

        for (blocks_to_migrate.items) |migration| {
            if (try self.storage.migrateBlock(migration.id, migration.target)) {
                self.migration_count += 1;
            }
        }
    }

    const CoreLoadEntry = struct {
        id: usize,
        load: f64,

        fn lessThan(_: void, a: @This(), b: @This()) bool {
            return a.load < b.load;
        }
    };

    fn balanceLoad(self: *ChaosCoreKernel) !void {
        var core_loads = AutoHashMap(usize, f64).init(self.allocator);
        defer core_loads.deinit();

        var total_load: f64 = 0.0;
        var active_core_count: usize = 0;

        var core_iter = self.cores.iterator();
        while (core_iter.next()) |entry| {
            const core = entry.value_ptr.*;
            if (core.state == .power_gated) continue;

            const load = core.getWorkload();
            try core_loads.put(entry.key_ptr.*, load);
            total_load += load;
            active_core_count += 1;
        }

        if (active_core_count == 0) return;

        const avg_load = total_load / @as(f64, @floatFromInt(active_core_count));

        var overloaded: ArrayList(usize) = .{};
        defer overloaded.deinit(self.allocator);
        var underloaded: ArrayList(CoreLoadEntry) = .{};
        defer underloaded.deinit(self.allocator);

        var load_iter = core_loads.iterator();
        while (load_iter.next()) |entry| {
            const cid = entry.key_ptr.*;
            const l = entry.value_ptr.*;
            if (l > avg_load * ChaosCoreConfig.LOAD_HIGH_THRESHOLD) {
                try overloaded.append(self.allocator, cid);
            } else if (l < avg_load * ChaosCoreConfig.LOAD_LOW_THRESHOLD) {
                try underloaded.append(self.allocator, .{ .id = cid, .load = l });
            }
        }

        if (overloaded.items.len == 0 or underloaded.items.len == 0) return;

        std.mem.sort(CoreLoadEntry, underloaded.items, {}, CoreLoadEntry.lessThan);

        var target_idx: usize = 0;
        for (overloaded.items) |over_core| {
            if (underloaded.items.len == 0) break;
            if (self.storage.affinity_map.getPtr(over_core)) |block_set| {
                if (block_set.count() == 0) continue;

                var blocks_list: ArrayList([ChaosCoreConfig.BLOCK_ID_SIZE]u8) = .{};
                defer blocks_list.deinit(self.allocator);

                var block_iter = block_set.iterator();
                while (block_iter.next()) |b_entry| {
                    try blocks_list.append(self.allocator, b_entry.key_ptr.*);
                }

                if (blocks_list.items.len == 0) continue;

                const migrate_count = @max(@as(usize, 1), blocks_list.items.len / 4);
                const capped = @min(migrate_count, blocks_list.items.len);

                var i: usize = 0;
                while (i < capped) : (i += 1) {
                    if (underloaded.items.len == 0) break;
                    const target_core = underloaded.items[target_idx % underloaded.items.len].id;
                    target_idx += 1;
                    if (target_core == over_core) continue;

                    const block_id = blocks_list.items[i];
                    if (try self.storage.migrateBlock(block_id, target_core)) {
                        self.migration_count += 1;
                    }
                }
            }
        }
    }

    pub fn selfOrganize(self: *ChaosCoreKernel) !void {
        try self.optimizeDataPlacement();
        try self.balanceLoad();
    }

    pub fn executeGraphOnKernel(self: *ChaosCoreKernel, graph: *SelfSimilarRelationalGraph) !std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, []const u8, BlockIdContext, std.hash_map.default_max_load_percentage) {
        var node_to_block = std.HashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8, []const u8, BlockIdContext, std.hash_map.default_max_load_percentage).init(self.allocator);
        var block_to_node = std.StringHashMap([ChaosCoreConfig.BLOCK_ID_SIZE]u8).init(self.allocator);
        defer block_to_node.deinit();

        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node_id = entry.key_ptr.*;
            const node = entry.value_ptr.*;
            const block_id = try self.allocateMemory(node.data, null);
            try node_to_block.put(block_id, node_id);
            try block_to_node.put(node_id, block_id);
        }

        var edge_iter = graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            const edge_key = entry.key_ptr.*;

            const source_block = block_to_node.get(edge_key.source);
            const target_block = block_to_node.get(edge_key.target);

            if (source_block != null and target_block != null) {
                _ = try self.entangleData(source_block.?, target_block.?);
            }
        }

        try self.selfOrganize();
        return node_to_block;
    }

    pub fn queryByContent(self: *ChaosCoreKernel, data: []const u8) ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
        return self.storage.retrieveByContent(data);
    }

    pub fn getBlockLocation(self: *ChaosCoreKernel, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) ?usize {
        if (self.storage.getBlock(block_id)) |block| {
            return block.affinity_core;
        }
        return null;
    }

    pub fn formatBlockId(block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) [ChaosCoreConfig.BLOCK_ID_SIZE * 2]u8 {
        var result: [ChaosCoreConfig.BLOCK_ID_SIZE * 2]u8 = undefined;
        bytesToHexLower(result[0..], block_id[0..]);
        return result;
    }

    pub fn formatTaskId(task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) [ChaosCoreConfig.SHA256_DIGEST_SIZE * 2]u8 {
        var result: [ChaosCoreConfig.SHA256_DIGEST_SIZE * 2]u8 = undefined;
        bytesToHexLower(result[0..], task_id[0..]);
        return result;
    }

    pub fn createInferenceTask(self: *ChaosCoreKernel, priority: i32, data_dependencies: []const [ChaosCoreConfig.BLOCK_ID_SIZE]u8, estimated_cycles: usize, inference_context_id: ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8) ![ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 {
        const task_id = try self.createTask(priority, data_dependencies, estimated_cycles);
        if (self.scheduler.getActiveTask(task_id)) |task| {
            task.inference_priority = @as(f64, @floatFromInt(@max(priority, 0))) / @as(f64, @floatFromInt(@max(@max(priority, 1), 1)));
            task.inference_context_id = inference_context_id;
        }
        return task_id;
    }

    pub fn getInferenceTaskResult(self: *ChaosCoreKernel, task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) ?struct { completed: bool, duration: i128 } {
        if (self.scheduler.active_tasks.get(task_id)) |task| {
            return .{ .completed = task.completion_status, .duration = task.getDuration() };
        }
        return null;
    }

    pub const InferenceHooks = struct {
        kernel: *ChaosCoreKernel,

        pub fn init(kernel: *ChaosCoreKernel) InferenceHooks {
            return InferenceHooks{ .kernel = kernel };
        }

        pub fn submitInferenceTask(self: *InferenceHooks, priority: i32, deps: []const [ChaosCoreConfig.BLOCK_ID_SIZE]u8, cycles: usize, context_id: ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8) ![ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 {
            return self.kernel.createInferenceTask(priority, deps, cycles, context_id);
        }

        pub fn getTaskResult(self: *const InferenceHooks, task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) ?struct { completed: bool, duration: i128 } {
            return self.kernel.getInferenceTaskResult(task_id);
        }

        pub fn storeData(self: *InferenceHooks, data: []const u8, core: ?usize) ![ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
            return self.kernel.allocateMemory(data, core);
        }

        pub fn readData(self: *const InferenceHooks, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) ?[]const u8 {
            return self.kernel.readMemory(block_id);
        }

        pub fn runCycle(self: *InferenceHooks) !void {
            return self.kernel.executeCycle();
        }

        pub fn getStats(self: *const InferenceHooks) KernelStatistics {
            return self.kernel.getKernelStatistics();
        }

        pub fn entangleDataBlocks(self: *InferenceHooks, block_id1: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, block_id2: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) !bool {
            return self.kernel.entangleData(block_id1, block_id2);
        }

        pub fn findBlockCore(self: *const InferenceHooks, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) ?usize {
            return self.kernel.getBlockLocation(block_id);
        }

        pub fn selfOrganize(self: *InferenceHooks) !void {
            return self.kernel.selfOrganize();
        }

        pub fn lookupByContent(self: *InferenceHooks, data: []const u8) ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8 {
            return self.kernel.queryByContent(data);
        }

        pub fn finishInferenceTask(self: *InferenceHooks, task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8) !bool {
            return self.kernel.finishTask(task_id);
        }

        pub fn allocateAndEntangle(self: *InferenceHooks, data1: []const u8, data2: []const u8, core: ?usize) !struct { block1: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, block2: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 } {
            const block1 = try self.kernel.allocateMemory(data1, core);
            const block2 = try self.kernel.allocateMemory(data2, core);
            _ = try self.kernel.entangleData(block1, block2);
            return .{ .block1 = block1, .block2 = block2 };
        }
    };

    pub fn createInferenceHooks(self: *ChaosCoreKernel) InferenceHooks {
        return InferenceHooks.init(self);
    }

    pub fn inferenceStoreAndSchedule(self: *ChaosCoreKernel, data: []const u8, priority: i32, estimated_cycles: usize, context_id: ?[ChaosCoreConfig.BLOCK_ID_SIZE]u8) !struct { block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8, task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 } {
        const block_id = try self.allocateMemory(data, null);
        var deps = [_][ChaosCoreConfig.BLOCK_ID_SIZE]u8{block_id};
        const task_id = try self.createInferenceTask(priority, deps[0..], estimated_cycles, context_id);
        return .{ .block_id = block_id, .task_id = task_id };
    }

    pub fn inferenceReadBlock(self: *ChaosCoreKernel, block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8) ?[]const u8 {
        return self.readMemory(block_id);
    }

    pub fn inferenceCycleAndReport(self: *ChaosCoreKernel) !KernelStatistics {
        try self.executeCycle();
        return self.getKernelStatistics();
    }
};

test "MemoryBlockState toString and fromString" {
    const testing = std.testing;

    try testing.expectEqualStrings("free", MemoryBlockState.free.toString());
    try testing.expectEqualStrings("allocated", MemoryBlockState.allocated.toString());
    try testing.expectEqualStrings("entangled", MemoryBlockState.entangled.toString());
    try testing.expectEqualStrings("migrating", MemoryBlockState.migrating.toString());

    try testing.expectEqual(MemoryBlockState.free, MemoryBlockState.fromString("free").?);
    try testing.expectEqual(MemoryBlockState.allocated, MemoryBlockState.fromString("allocated").?);
    try testing.expect(MemoryBlockState.fromString("invalid") == null);
}

test "MemoryBlock initialization and access" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&block_id, 0xAB);
    var content_hash: [ChaosCoreConfig.CONTENT_HASH_SIZE]u8 = undefined;
    @memset(&content_hash, 0xCD);

    var block = try MemoryBlock.init(allocator, block_id, content_hash, "test data", @as(?usize, 0));
    defer block.deinit();

    try testing.expectEqualStrings("test data", block.data);
    try testing.expectEqual(@as(usize, 9), block.data.len);
    try testing.expectEqual(MemoryBlockState.allocated, block.state);
    try testing.expectEqual(@as(?usize, 0), block.affinity_core);
    try testing.expectEqual(@as(usize, 1), block.access_count);
}

test "MemoryBlock entanglement" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var block_id1: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&block_id1, 0x11);
    var block_id2: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&block_id2, 0x22);
    var content_hash: [ChaosCoreConfig.CONTENT_HASH_SIZE]u8 = undefined;
    @memset(&content_hash, 0x00);

    var block = try MemoryBlock.init(allocator, block_id1, content_hash, "data", null);
    defer block.deinit();

    try block.addEntangledBlock(block_id2);
    try testing.expect(block.hasEntanglement(block_id2));
    try testing.expectEqual(@as(usize, 1), block.entangledCount());
}

test "TaskDescriptor initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
    @memset(&task_id, 0xFF);

    var task = TaskDescriptor.init(allocator, task_id, 5, 1000);
    defer task.deinit();

    try testing.expectEqual(@as(i32, 5), task.priority);
    try testing.expectEqual(@as(usize, 1000), task.estimated_cycles);
    try testing.expect(!task.completion_status);
}

test "ContentAddressableStorage store and retrieve" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var storage = ContentAddressableStorage.init(allocator, 1024 * 1024);
    defer storage.deinit();

    const block_id = try storage.store("hello world", @as(?usize, 0));
    const retrieved = storage.retrieve(block_id);

    try testing.expect(retrieved != null);
    try testing.expectEqualStrings("hello world", retrieved.?);
}

test "ContentAddressableStorage content deduplication" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var storage = ContentAddressableStorage.init(allocator, 1024 * 1024);
    defer storage.deinit();

    const block_id1 = try storage.store("duplicate", null);
    const block_id2 = try storage.store("duplicate", null);

    try testing.expect(std.mem.eql(u8, block_id1[0..], block_id2[0..]));
    try testing.expectEqual(@as(usize, 1), storage.storage.count());
}

test "ContentAddressableStorage entanglement" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var storage = ContentAddressableStorage.init(allocator, 1024 * 1024);
    defer storage.deinit();

    const block_id1 = try storage.store("data1", null);
    const block_id2 = try storage.store("data2", null);

    const success = try storage.entangleBlocks(block_id1, block_id2);
    try testing.expect(success);

    const block1 = storage.getBlock(block_id1).?;
    const block2 = storage.getBlock(block_id2).?;

    try testing.expectEqual(MemoryBlockState.entangled, block1.state);
    try testing.expectEqual(MemoryBlockState.entangled, block2.state);
    try testing.expect(block1.hasEntanglement(block_id2));
    try testing.expect(block2.hasEntanglement(block_id1));
}

test "ContentAddressableStorage statistics" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var storage = ContentAddressableStorage.init(allocator, 1024 * 1024);
    defer storage.deinit();

    _ = try storage.store("test1", @as(?usize, 0));
    _ = try storage.store("test2", @as(?usize, 1));

    const stats = storage.getStatistics();
    try testing.expectEqual(@as(usize, 2), stats.total_blocks);
    try testing.expectEqual(@as(usize, 2), stats.unique_contents);
}

test "DynamicTaskScheduler submit and complete" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var scheduler = DynamicTaskScheduler.init(allocator);
    defer scheduler.deinit();

    const deps: [0][ChaosCoreConfig.BLOCK_ID_SIZE]u8 = .{};
    const task_id = try scheduler.submitTask(1, deps[0..], 100);

    try testing.expectEqual(@as(usize, 1), scheduler.pending_tasks.count());

    var cores = AutoHashMap(usize, ProcessingCore).init(allocator);
    defer cores.deinit();
    try cores.put(0, ProcessingCore.init(0));

    var storage = ContentAddressableStorage.init(allocator, 1024 * 1024);
    defer storage.deinit();

    const scheduled = try scheduler.scheduleTask(&cores, &storage);
    try testing.expect(scheduled != null);
    try testing.expect(std.mem.eql(u8, scheduled.?.task_id[0..], task_id[0..]));

    const completed = try scheduler.completeTask(task_id);
    try testing.expect(completed);
    try testing.expectEqual(@as(usize, 1), scheduler.completed_tasks.items.len);
}

test "DynamicTaskScheduler statistics" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var scheduler = DynamicTaskScheduler.init(allocator);
    defer scheduler.deinit();

    const stats = scheduler.getStatistics();
    try testing.expectEqual(@as(usize, 0), stats.pending_tasks);
    try testing.expectEqual(@as(usize, 0), stats.active_tasks);
    try testing.expectEqual(@as(usize, 0), stats.completed_tasks);
}

test "DataFlowAnalyzer record and analyze" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var analyzer = DataFlowAnalyzer.init(allocator);
    defer analyzer.deinit();

    var block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&block_id, 0x12);

    try analyzer.recordAccess(block_id, 0);
    try analyzer.recordAccess(block_id, 0);
    try analyzer.recordAccess(block_id, 1);

    var affinities = try analyzer.analyzeFlow(block_id, allocator);
    defer affinities.deinit();

    try testing.expect(affinities.count() == 2);

    const core0_affinity = affinities.get(0).?;
    try testing.expect(core0_affinity > 0.6);
}

test "DataFlowAnalyzer flow graph" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var analyzer = DataFlowAnalyzer.init(allocator);
    defer analyzer.deinit();

    var task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
    @memset(&task_id, 0xFF);

    var task = TaskDescriptor.init(allocator, task_id, 1, 100);
    defer task.deinit();

    var dep1: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&dep1, 0x01);
    var dep2: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&dep2, 0x02);

    try task.addDependency(dep1);
    try task.addDependency(dep2);

    var tasks = [_]TaskDescriptor{task};
    try analyzer.buildFlowGraph(tasks[0..]);

    try testing.expect(analyzer.flow_graph.count() > 0);
}

test "ChaosCoreKernel initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    try kernel.addCore(0);
    try kernel.addCore(1);

    const stats = kernel.getKernelStatistics();
    try testing.expectEqual(@as(usize, 0), stats.cycle_count);
    try testing.expectEqual(@as(usize, 0), stats.migration_count);
}

test "ChaosCoreKernel memory operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    try kernel.addCore(0);

    const block_id = try kernel.allocateMemory("test data", @as(?usize, 0));
    const data = kernel.readMemory(block_id);

    try testing.expect(data != null);
    try testing.expectEqualStrings("test data", data.?);

    const location = kernel.getBlockLocation(block_id);
    try testing.expectEqual(@as(?usize, 0), location);
}

test "ChaosCoreKernel task management" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    try kernel.addCore(0);

    const block_id = try kernel.allocateMemory("dependency", @as(?usize, 0));
    var deps = [_][ChaosCoreConfig.BLOCK_ID_SIZE]u8{block_id};

    _ = try kernel.createTask(1, deps[0..], 500);

    const stats = kernel.getKernelStatistics();
    try testing.expectEqual(@as(usize, 1), stats.scheduler.pending_tasks);
}

test "ChaosCoreKernel entanglement" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    const block_id1 = try kernel.allocateMemory("data1", null);
    const block_id2 = try kernel.allocateMemory("data2", null);

    const success = try kernel.entangleData(block_id1, block_id2);
    try testing.expect(success);

    const stats = kernel.getKernelStatistics();
    try testing.expectEqual(@as(usize, 2), stats.storage.entangled_blocks);
}

test "ChaosCoreKernel query by content" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    const block_id = try kernel.allocateMemory("unique content", null);
    const found_id = kernel.queryByContent("unique content");

    try testing.expect(found_id != null);
    try testing.expect(std.mem.eql(u8, block_id[0..], found_id.?[0..]));
}

test "ChaosCoreKernel cycle execution" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    try kernel.addCore(0);
    try kernel.addCore(1);

    try kernel.executeCycle();
    try kernel.executeCycle();

    const stats = kernel.getKernelStatistics();
    try testing.expectEqual(@as(usize, 2), stats.cycle_count);
}

test "ProcessingCore workload calculation" {
    const testing = std.testing;

    var core = ProcessingCore.init(0);
    core.cycles_active = 75;
    core.cycles_idle = 25;

    const workload = core.getWorkload();
    try testing.expectApproxEqAbs(@as(f64, 0.75), workload, 0.001);
}

test "ChaosCoreKernel self organize" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    try kernel.addCore(0);
    try kernel.addCore(1);

    _ = try kernel.allocateMemory("block1", @as(?usize, 0));
    _ = try kernel.allocateMemory("block2", @as(?usize, 1));
    _ = try kernel.allocateMemory("block3", @as(?usize, 0));

    try kernel.selfOrganize();

    const stats = kernel.getKernelStatistics();
    try testing.expectEqual(@as(usize, 3), stats.storage.total_blocks);
}

test "ContentAddressableStorage migration" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var storage = ContentAddressableStorage.init(allocator, 1024 * 1024);
    defer storage.deinit();

    const block_id = try storage.store("migrate me", @as(?usize, 0));
    const success = try storage.migrateBlock(block_id, 1);

    try testing.expect(success);

    const block = storage.getBlock(block_id).?;
    try testing.expectEqual(@as(?usize, 1), block.affinity_core);
}

test "ChaosCoreKernel format helpers" {
    const testing = std.testing;

    var block_id: [ChaosCoreConfig.BLOCK_ID_SIZE]u8 = undefined;
    @memset(&block_id, 0xAB);

    const formatted = ChaosCoreKernel.formatBlockId(block_id);
    try testing.expectEqual(@as(usize, 32), formatted.len);

    var task_id: [ChaosCoreConfig.SHA256_DIGEST_SIZE]u8 = undefined;
    @memset(&task_id, 0xCD);

    const formatted_task = ChaosCoreKernel.formatTaskId(task_id);
    try testing.expectEqual(@as(usize, 64), formatted_task.len);
}

