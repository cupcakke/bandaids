const std = @import("std");
const builtin = @import("builtin");

const mem = std.mem;
const Allocator = mem.Allocator;

const Mutex = std.Thread.Mutex;
const CondVar = std.Thread.Condition;
const Semaphore = std.Thread.Semaphore;

const ArithmeticError = error{ Overflow, InvalidAlignment };
const EmptyError = error{Empty};

const min_page_align: usize = switch (builtin.os.tag) {
    .macos => if (builtin.cpu.arch == .aarch64) 16384 else 4096,
    else => 4096,
};

pub const MemoryConfig = struct {
    pub const PAGE_SIZE: usize = min_page_align;
    pub const CACHE_LINE_SIZE: usize = 128;
};

pub const PageSize: usize = MemoryConfig.PAGE_SIZE;

fn emptySlice(comptime T: type) []T {
    const addr: usize = @max(@alignOf(T), 1);
    const p: [*]T = @ptrFromInt(addr);
    return p[0..0];
}

fn emptyAlignedU8Slice() []align(PageSize) u8 {
    const p: [*]align(PageSize) u8 = @ptrFromInt(PageSize);
    return p[0..0];
}

fn emptyU8Slice() []u8 {
    return emptySlice(u8);
}

fn isPow2(x: usize) bool {
    return x != 0 and (x & (x - 1)) == 0;
}

fn alignForwardChecked(addr: usize, alignment: usize) ArithmeticError!usize {
    if (!isPow2(alignment)) return error.InvalidAlignment;
    return mem.alignForward(usize, addr, alignment);
}

fn addChecked(a: usize, b: usize) ArithmeticError!usize {
    const r = @addWithOverflow(a, b);
    if (r[1] != 0) return error.Overflow;
    return r[0];
}

fn mulChecked(a: usize, b: usize) ArithmeticError!usize {
    const r = @mulWithOverflow(a, b);
    if (r[1] != 0) return error.Overflow;
    return r[0];
}

fn saturatingSub(a: usize, b: usize) usize {
    return if (a >= b) a - b else 0;
}

fn runtimeAlignedAlloc(allocator: Allocator, comptime T: type, n: usize, alignment: usize) ![]T {
    if (!isPow2(alignment)) return error.InvalidAlignment;
    if (n == 0) return emptySlice(T);
    const byte_count = try mulChecked(n, @sizeOf(T));
    const log2a: u8 = @intCast(std.math.log2_int(usize, alignment));
    const raw = allocator.rawAlloc(byte_count, log2a, @returnAddress()) orelse return error.OutOfMemory;
    const typed: [*]T = @ptrCast(@alignCast(raw));
    return typed[0..n];
}

pub const Arena = struct {
    buffer: []align(PageSize) u8,
    offset: usize,
    allocator: Allocator,
    mutex: Mutex,

    pub fn init(allocator: Allocator, size: usize) !Arena {
        if (size == 0) return error.InvalidSize;
        const aligned_size = mem.alignForward(usize, size, PageSize);
        const buffer = try allocator.alignedAlloc(u8, PageSize, aligned_size);
        return .{
            .buffer = buffer,
            .offset = 0,
            .allocator = allocator,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *Arena) void {
        const buf = self.buffer;
        self.buffer = emptyAlignedU8Slice();
        self.offset = 0;
        if (buf.len != 0) self.allocator.free(buf);
    }

    pub fn secureDeinit(self: *Arena) void {
        self.secureResetInternal();
        const buf = self.buffer;
        self.buffer = emptyAlignedU8Slice();
        self.offset = 0;
        if (buf.len != 0) self.allocator.free(buf);
    }

    pub fn alloc(self: *Arena, size: usize, alignment: usize) ?[]u8 {
        if (size == 0) return emptyU8Slice();
        if (!isPow2(alignment)) return null;

        self.mutex.lock();
        defer self.mutex.unlock();

        const aligned_offset = mem.alignForward(usize, self.offset, alignment);
        const end = addChecked(aligned_offset, size) catch return null;
        if (end > self.buffer.len) return null;

        const out = self.buffer[aligned_offset..end];
        self.offset = end;
        return out;
    }

    pub fn allocBytes(self: *Arena, size: usize) ?[]u8 {
        return self.alloc(size, @alignOf(usize));
    }

    fn secureResetInternal(self: *Arena) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.offset > 0) secureZeroMemory(self.buffer.ptr, self.offset);
        self.offset = 0;
    }

    pub fn reset(self: *Arena) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.offset = 0;
    }

    pub fn secureReset(self: *Arena) void {
        self.secureResetInternal();
    }

    pub fn allocated(self: *Arena) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.offset;
    }

    pub fn remaining(self: *Arena) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.offset > self.buffer.len) return 0;
        return self.buffer.len - self.offset;
    }
};

pub const ArenaAllocator = struct {
    parent_allocator: Allocator,
    buffers: std.ArrayList([]u8),
    current_buffer: []u8,
    pos: usize,
    buffer_size: usize,
    mutex: Mutex,
    secure_zero: bool,

    pub fn init(parent_allocator: Allocator, buffer_size: usize) ArenaAllocator {
        return .{
            .parent_allocator = parent_allocator,
            .buffers = std.ArrayList([]u8).init(parent_allocator),
            .current_buffer = emptyU8Slice(),
            .pos = 0,
            .buffer_size = if (buffer_size == 0) 4096 else buffer_size,
            .mutex = .{},
            .secure_zero = false,
        };
    }

    pub fn enableSecureZero(self: *ArenaAllocator, enable: bool) void {
        self.secure_zero = enable;
    }

    pub fn deinit(self: *ArenaAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.buffers.items) |buf| {
            self.parent_allocator.free(buf);
        }
        self.buffers.deinit();
        self.current_buffer = emptyU8Slice();
        self.pos = 0;
    }

    pub fn secureDeinit(self: *ArenaAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.buffers.items) |buf| {
            secureZeroMemory(buf.ptr, buf.len);
            self.parent_allocator.free(buf);
        }
        self.buffers.deinit();
        self.current_buffer = emptyU8Slice();
        self.pos = 0;
    }

    pub fn allocator(self: *ArenaAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = arenaAlloc,
                .resize = arenaResize,
                .remap = arenaRemap,
                .free = arenaFree,
            },
        };
    }

    fn ensureBuffer(self: *ArenaAllocator, len: usize, alignment: usize) ?void {
        const need = addChecked(len, alignment - 1) catch return null;
        const new_size = if (self.buffer_size > need) self.buffer_size else need;
        const new_buf = self.parent_allocator.alloc(u8, new_size) catch return null;
        self.buffers.append(new_buf) catch {
            self.parent_allocator.free(new_buf);
            return null;
        };
        self.current_buffer = new_buf;
        self.pos = 0;
    }

    fn alignedPos(self: *ArenaAllocator, alignment: usize) ?usize {
        const base = @intFromPtr(self.current_buffer.ptr);
        const cur = addChecked(base, self.pos) catch return null;
        const aligned = mem.alignForward(usize, cur, alignment);
        return aligned - base;
    }

    fn arenaAlloc(ctx: *anyopaque, len: usize, log2_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *ArenaAllocator = @ptrCast(@alignCast(ctx));
        if (len == 0) return emptyU8Slice().ptr;

        const align_bytes: usize = log2_align.toByteUnits();

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.current_buffer.len == 0) {
            self.ensureBuffer(len, align_bytes) orelse return null;
        }

        var aligned_pos = self.alignedPos(align_bytes) orelse return null;
        var end = addChecked(aligned_pos, len) catch return null;
        if (end > self.current_buffer.len) {
            self.ensureBuffer(len, align_bytes) orelse return null;
            aligned_pos = self.alignedPos(align_bytes) orelse return null;
            end = addChecked(aligned_pos, len) catch return null;
            if (end > self.current_buffer.len) return null;
        }

        const p = self.current_buffer.ptr + aligned_pos;
        self.pos = end;
        return p;
    }

    fn arenaResize(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ret_addr;
        const self: *ArenaAllocator = @ptrCast(@alignCast(ctx));
        const align_bytes: usize = log2_align.toByteUnits();

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.current_buffer.len == 0) return false;
        const base = @intFromPtr(self.current_buffer.ptr);
        const buf_addr = @intFromPtr(buf.ptr);
        const expected_addr = mem.alignForward(usize, buf_addr, align_bytes);
        if (expected_addr != buf_addr) return false;
        const buf_end = addChecked(buf_addr, buf.len) catch return false;
        const cur_end = addChecked(base, self.pos) catch return false;
        if (buf_end != cur_end) return false;

        if (new_len >= buf.len) {
            const addl = new_len - buf.len;
            const new_pos = addChecked(self.pos, addl) catch return false;
            if (new_pos > self.current_buffer.len) return false;
            self.pos = new_pos;
            return true;
        }

        const shrink = buf.len - new_len;
        if (shrink > self.pos) return false;
        self.pos -= shrink;
        return true;
    }


    fn arenaRemap(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        if (arenaResize(ctx, buf, log2_align, new_len, ret_addr)) {
            return buf.ptr;
        }
        return null;
    }

    fn arenaFree(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, ret_addr: usize) void {
        _ = ctx;
        _ = buf;
        _ = log2_align;
        _ = ret_addr;
    }
};

pub const SlabAllocator = struct {
    slabs: []Slab,
    next_id: usize,
    backing_allocator: Allocator,
    slab_size: usize,
    block_size: usize,
    allocations: std.AutoHashMap(usize, AllocationMeta),
    mutex: Mutex,

    const AllocationMeta = struct {
        slab_index: usize,
        start_block: usize,
        blocks: usize,
        size: usize,
    };

    const Slab = struct {
        data: []u8,
        bitmap: []u64,
        num_blocks: usize,

        fn isBlockFree(self: *const Slab, block_idx: usize) bool {
            const word_idx = block_idx / 64;
            const bit_idx: u6 = @intCast(block_idx % 64);
            return (self.bitmap[word_idx] & (@as(u64, 1) << bit_idx)) == 0;
        }

        fn setBlockUsed(self: *Slab, block_idx: usize) void {
            const word_idx = block_idx / 64;
            const bit_idx: u6 = @intCast(block_idx % 64);
            self.bitmap[word_idx] |= (@as(u64, 1) << bit_idx);
        }

        fn setBlockFree(self: *Slab, block_idx: usize) void {
            const word_idx = block_idx / 64;
            const bit_idx: u6 = @intCast(block_idx % 64);
            self.bitmap[word_idx] &= ~(@as(u64, 1) << bit_idx);
        }

        fn setRange(self: *Slab, start_block: usize, blocks: usize, used: bool) void {
            var i: usize = start_block;
            const end = start_block + blocks;
            while (i < end) : (i += 1) {
                if (used) self.setBlockUsed(i) else self.setBlockFree(i);
            }
        }
    };

    pub fn init(parent_allocator: Allocator, slab_size: usize, num_slabs: usize, block_size: usize) !SlabAllocator {
        if (slab_size == 0) return error.InvalidSize;
        if (num_slabs == 0) return error.InvalidSlabCount;
        if (block_size == 0 or !isPow2(block_size)) return error.InvalidBlockSize;
        if (slab_size < block_size or slab_size % block_size != 0) return error.InvalidSize;

        var slabs = try parent_allocator.alloc(Slab, num_slabs);
        var initialized: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < initialized) : (i += 1) {
                parent_allocator.free(slabs[i].bitmap);
                parent_allocator.free(slabs[i].data);
            }
            parent_allocator.free(slabs);
        }

        const num_blocks = slab_size / block_size;
        const bitmap_words = (num_blocks + 63) / 64;
        while (initialized < num_slabs) : (initialized += 1) {
            slabs[initialized].data = try parent_allocator.alloc(u8, slab_size);
            slabs[initialized].bitmap = parent_allocator.alloc(u64, bitmap_words) catch |err| {
                parent_allocator.free(slabs[initialized].data);
                return err;
            };
            @memset(slabs[initialized].data, 0);
            @memset(slabs[initialized].bitmap, 0);
            slabs[initialized].num_blocks = num_blocks;
        }

        return .{
            .slabs = slabs,
            .next_id = 0,
            .backing_allocator = parent_allocator,
            .slab_size = slab_size,
            .block_size = block_size,
            .allocations = std.AutoHashMap(usize, AllocationMeta).init(parent_allocator),
            .mutex = .{},
        };
    }

    pub fn deinit(self: *SlabAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.allocations.deinit();
        const slabs = self.slabs;
        self.slabs = emptySlice(Slab);
        for (slabs) |slab| {
            self.backing_allocator.free(slab.bitmap);
            self.backing_allocator.free(slab.data);
        }
        if (slabs.len != 0) self.backing_allocator.free(slabs);
        self.next_id = 0;
    }

    pub fn secureDeinit(self: *SlabAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.allocations.deinit();
        const slabs = self.slabs;
        self.slabs = emptySlice(Slab);
        for (slabs) |slab| {
            secureZeroMemory(slab.data.ptr, slab.data.len);
            self.backing_allocator.free(slab.bitmap);
            self.backing_allocator.free(slab.data);
        }
        if (slabs.len != 0) self.backing_allocator.free(slabs);
        self.next_id = 0;
    }

    pub fn alloc(self: *SlabAllocator, size: usize) ?[]u8 {
        if (size == 0) return emptyU8Slice();
        if (size > self.slab_size) return null;

        self.mutex.lock();
        defer self.mutex.unlock();

        const blocks_needed = (size + self.block_size - 1) / self.block_size;
        var search_count: usize = 0;
        while (search_count < self.slabs.len) : (search_count += 1) {
            const slab_idx = (self.next_id + search_count) % self.slabs.len;
            var slab = &self.slabs[slab_idx];
            var consecutive: usize = 0;
            var start_idx: usize = 0;

            var i: usize = 0;
            while (i < slab.num_blocks) : (i += 1) {
                if (slab.isBlockFree(i)) {
                    if (consecutive == 0) start_idx = i;
                    consecutive += 1;
                    if (consecutive == blocks_needed) {
                        const offset = start_idx * self.block_size;
                        const block_span = blocks_needed * self.block_size;
                        const out = slab.data[offset .. offset + size];
                        slab.setRange(start_idx, blocks_needed, true);
                        self.allocations.put(@intFromPtr(out.ptr), .{
                            .slab_index = slab_idx,
                            .start_block = start_idx,
                            .blocks = blocks_needed,
                            .size = size,
                        }) catch {
                            slab.setRange(start_idx, blocks_needed, false);
                            return null;
                        };
                        @memset(slab.data[offset .. offset + block_span], 0);
                        self.next_id = (slab_idx + 1) % self.slabs.len;
                        return out;
                    }
                } else {
                    consecutive = 0;
                }
            }
        }
        return null;
    }

    pub fn allocator(self: *SlabAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = slabVtableAlloc,
                .resize = slabVtableResize,
                .remap = slabVtableRemap,
                .free = slabVtableFree,
            },
        };
    }

    fn slabVtableAlloc(ctx: *anyopaque, len: usize, log2_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *SlabAllocator = @ptrCast(@alignCast(ctx));
        const align_bytes = log2_align.toByteUnits();
        if (align_bytes > self.block_size) return null;
        const slice = self.alloc(len) orelse return null;
        if (!mem.isAligned(@intFromPtr(slice.ptr), align_bytes)) return null;
        return slice.ptr;
    }

    fn slabVtableResize(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = log2_align;
        _ = new_len;
        _ = ret_addr;
        return false;
    }


    fn slabVtableRemap(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        if (slabVtableResize(ctx, buf, log2_align, new_len, ret_addr)) {
            return buf.ptr;
        }
        return null;
    }

    fn slabVtableFree(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, ret_addr: usize) void {
        _ = log2_align;
        _ = ret_addr;
        const self: *SlabAllocator = @ptrCast(@alignCast(ctx));
        self.free(buf) catch {};
    }

    pub fn free(self: *SlabAllocator, ptr: []u8) !void {
        if (ptr.len == 0) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        const meta = self.allocations.get(@intFromPtr(ptr.ptr)) orelse return error.InvalidPointer;
        if (ptr.len != meta.size) return error.InvalidPointer;
        _ = self.allocations.fetchRemove(@intFromPtr(ptr.ptr));
        var slab = &self.slabs[meta.slab_index];
        slab.setRange(meta.start_block, meta.blocks, false);
    }

    pub fn secureFree(self: *SlabAllocator, ptr: []u8) !void {
        if (ptr.len == 0) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        const meta = self.allocations.get(@intFromPtr(ptr.ptr)) orelse return error.InvalidPointer;
        if (ptr.len != meta.size) return error.InvalidPointer;
        _ = self.allocations.fetchRemove(@intFromPtr(ptr.ptr));
        var slab = &self.slabs[meta.slab_index];
        const offset = meta.start_block * self.block_size;
        secureZeroMemory(slab.data.ptr + offset, meta.size);
        slab.setRange(meta.start_block, meta.blocks, false);
    }
};

pub const PoolAllocator = struct {
    pools: []Pool,
    backing_allocator: Allocator,
    allocations: std.AutoHashMap(usize, AllocationMeta),
    mutex: Mutex,

    const AllocationMeta = struct {
        pool_index: usize,
        block_index: usize,
        size: usize,
    };

    const Pool = struct {
        buffer: []align(@alignOf(?usize)) u8,
        block_size: usize,
        num_blocks: usize,
        free_list_head: ?usize,
        used: usize,
        allocated_bitmap: []u64,

        fn isBlockAllocated(self: *const Pool, idx: usize) bool {
            const word_idx = idx / 64;
            const bit_idx: u6 = @intCast(idx % 64);
            return (self.allocated_bitmap[word_idx] & (@as(u64, 1) << bit_idx)) != 0;
        }

        fn setBlockAllocated(self: *Pool, idx: usize) void {
            const word_idx = idx / 64;
            const bit_idx: u6 = @intCast(idx % 64);
            self.allocated_bitmap[word_idx] |= (@as(u64, 1) << bit_idx);
        }

        fn clearBlockAllocated(self: *Pool, idx: usize) void {
            const word_idx = idx / 64;
            const bit_idx: u6 = @intCast(idx % 64);
            self.allocated_bitmap[word_idx] &= ~(@as(u64, 1) << bit_idx);
        }

        fn ptrForIndex(self: *Pool, idx: usize) [*]u8 {
            return self.buffer.ptr + idx * self.block_size;
        }

        fn nextPtr(self: *Pool, idx: usize) *?usize {
            const base = self.ptrForIndex(idx);
            const single: *u8 = &base[0];
            return @ptrCast(@alignCast(single));
        }

        fn initFreeList(self: *Pool) void {
            if (self.num_blocks == 0) {
                self.free_list_head = null;
                return;
            }
            var i: usize = 0;
            while (i < self.num_blocks) : (i += 1) {
                const next = self.nextPtr(i);
                next.* = if (i + 1 < self.num_blocks) i + 1 else null;
            }
            self.free_list_head = 0;
            self.used = 0;
        }
    };

    pub fn init(parent_allocator: Allocator, block_size: usize, num_blocks: usize, num_pools: usize) !PoolAllocator {
        if (block_size == 0) return error.InvalidBlockSize;
        if (num_blocks == 0) return error.InvalidBlockCount;
        if (num_pools == 0) return error.InvalidPoolCount;

        const actual_block_size = mem.alignForward(usize, @max(block_size, @sizeOf(?usize)), @alignOf(?usize));
        var pools = try parent_allocator.alloc(Pool, num_pools);
        var initialized: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < initialized) : (i += 1) {
                parent_allocator.free(pools[i].allocated_bitmap);
                parent_allocator.free(pools[i].buffer);
            }
            parent_allocator.free(pools);
        }

        const bitmap_words = (num_blocks + 63) / 64;
        while (initialized < num_pools) : (initialized += 1) {
            const total = try mulChecked(actual_block_size, num_blocks);
            pools[initialized].buffer = try parent_allocator.alignedAlloc(u8, @alignOf(?usize), total);
            pools[initialized].allocated_bitmap = parent_allocator.alloc(u64, bitmap_words) catch |err| {
                parent_allocator.free(pools[initialized].buffer);
                return err;
            };
            @memset(pools[initialized].buffer, 0);
            @memset(pools[initialized].allocated_bitmap, 0);
            pools[initialized].block_size = actual_block_size;
            pools[initialized].num_blocks = num_blocks;
            pools[initialized].free_list_head = null;
            pools[initialized].used = 0;
            pools[initialized].initFreeList();
        }

        return .{
            .pools = pools,
            .backing_allocator = parent_allocator,
            .allocations = std.AutoHashMap(usize, AllocationMeta).init(parent_allocator),
            .mutex = .{},
        };
    }

    pub fn deinit(self: *PoolAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.allocations.deinit();
        const pools = self.pools;
        self.pools = emptySlice(Pool);
        for (pools) |pool| {
            self.backing_allocator.free(pool.allocated_bitmap);
            self.backing_allocator.free(pool.buffer);
        }
        if (pools.len != 0) self.backing_allocator.free(pools);
    }

    pub fn secureDeinit(self: *PoolAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.allocations.deinit();
        const pools = self.pools;
        self.pools = emptySlice(Pool);
        for (pools) |pool| {
            secureZeroMemory(pool.buffer.ptr, pool.buffer.len);
            self.backing_allocator.free(pool.allocated_bitmap);
            self.backing_allocator.free(pool.buffer);
        }
        if (pools.len != 0) self.backing_allocator.free(pools);
    }

    pub fn alloc(self: *PoolAllocator, size: usize) ?[]u8 {
        if (size == 0) return emptyU8Slice();

        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.pools, 0..) |*pool, pool_index| {
            if (size > pool.block_size) continue;
            const head_idx = pool.free_list_head orelse continue;

            pool.setBlockAllocated(head_idx);
            const next = pool.nextPtr(head_idx);
            pool.free_list_head = next.*;
            pool.used += 1;

            const full = pool.buffer[head_idx * pool.block_size .. (head_idx + 1) * pool.block_size];
            @memset(full, 0);
            const out = full[0..size];
            self.allocations.put(@intFromPtr(out.ptr), .{
                .pool_index = pool_index,
                .block_index = head_idx,
                .size = size,
            }) catch {
                pool.clearBlockAllocated(head_idx);
                next.* = pool.free_list_head;
                pool.free_list_head = head_idx;
                pool.used -= 1;
                return null;
            };
            return out;
        }

        return null;
    }

    pub fn allocator(self: *PoolAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = poolVtableAlloc,
                .resize = poolVtableResize,
                .remap = poolVtableRemap,
                .free = poolVtableFree,
            },
        };
    }

    fn poolVtableAlloc(ctx: *anyopaque, len: usize, log2_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *PoolAllocator = @ptrCast(@alignCast(ctx));
        const align_bytes = log2_align.toByteUnits();
        if (align_bytes > @alignOf(?usize)) return null;
        const slice = self.alloc(len) orelse return null;
        if (!mem.isAligned(@intFromPtr(slice.ptr), align_bytes)) return null;
        return slice.ptr;
    }

    fn poolVtableResize(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = log2_align;
        _ = new_len;
        _ = ret_addr;
        return false;
    }


    fn poolVtableRemap(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        if (poolVtableResize(ctx, buf, log2_align, new_len, ret_addr)) {
            return buf.ptr;
        }
        return null;
    }

    fn poolVtableFree(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, ret_addr: usize) void {
        _ = log2_align;
        _ = ret_addr;
        const self: *PoolAllocator = @ptrCast(@alignCast(ctx));
        self.free(buf) catch {};
    }

    pub fn free(self: *PoolAllocator, ptr: []u8) !void {
        if (ptr.len == 0) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        const meta = self.allocations.get(@intFromPtr(ptr.ptr)) orelse return error.InvalidPointer;
        if (ptr.len != meta.size) return error.InvalidPointer;
        var pool = &self.pools[meta.pool_index];
        if (!pool.isBlockAllocated(meta.block_index)) return error.DoubleFree;

        _ = self.allocations.fetchRemove(@intFromPtr(ptr.ptr));
        pool.clearBlockAllocated(meta.block_index);
        const next = pool.nextPtr(meta.block_index);
        next.* = pool.free_list_head;
        pool.free_list_head = meta.block_index;
        pool.used -= 1;
    }

    pub fn secureFree(self: *PoolAllocator, ptr: []u8) !void {
        if (ptr.len == 0) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        const meta = self.allocations.get(@intFromPtr(ptr.ptr)) orelse return error.InvalidPointer;
        if (ptr.len != meta.size) return error.InvalidPointer;
        var pool = &self.pools[meta.pool_index];
        if (!pool.isBlockAllocated(meta.block_index)) return error.DoubleFree;

        _ = self.allocations.fetchRemove(@intFromPtr(ptr.ptr));
        pool.clearBlockAllocated(meta.block_index);
        const full = pool.buffer[meta.block_index * pool.block_size .. (meta.block_index + 1) * pool.block_size];
        secureZeroMemory(full.ptr, full.len);
        const next = pool.nextPtr(meta.block_index);
        next.* = pool.free_list_head;
        pool.free_list_head = meta.block_index;
        pool.used -= 1;
    }
};

pub const BuddyAllocator = struct {
    backing_allocator: Allocator,
    memory: []align(PageSize) u8,
    tree: []State,
    max_order: u32,
    min_order: u32,
    size_map: std.AutoHashMap(usize, AllocationMeta),
    mutex: Mutex,

    const AllocationMeta = struct {
        order: u32,
        size: usize,
    };

    const State = enum(u8) {
        free,
        split,
        full,
    };

    pub fn init(parent_allocator: Allocator, size: usize, min_order: u32) !BuddyAllocator {
        if (size == 0) return error.InvalidSize;
        if (min_order >= @bitSizeOf(usize)) return error.InvalidSize;
        const min_block = @as(usize, 1) << @intCast(min_order);
        if (size < min_block) return error.SizeTooSmall;

        const capacity = std.math.ceilPowerOfTwo(usize, size) catch return error.InvalidSize;
        const max_order: u32 = @intCast(std.math.log2_int(usize, capacity));
        if (max_order < min_order) return error.SizeTooSmall;
        const leaf_count = capacity / min_block;
        if (leaf_count == 0) return error.InvalidSize;
        const tree_nodes = try subTwoTimesMinusOne(leaf_count);

        const tree = try parent_allocator.alloc(State, tree_nodes);
        @memset(tree, .free);
        errdefer parent_allocator.free(tree);

        const memory = try parent_allocator.alignedAlloc(u8, PageSize, capacity);
        errdefer parent_allocator.free(memory);

        return .{
            .backing_allocator = parent_allocator,
            .memory = memory,
            .tree = tree,
            .max_order = max_order,
            .min_order = min_order,
            .size_map = std.AutoHashMap(usize, AllocationMeta).init(parent_allocator),
            .mutex = .{},
        };
    }

    fn subTwoTimesMinusOne(x: usize) ArithmeticError!usize {
        const twice = try mulChecked(x, 2);
        if (twice == 0) return error.Overflow;
        return twice - 1;
    }

    pub fn deinit(self: *BuddyAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.size_map.deinit();
        const memory = self.memory;
        const tree = self.tree;
        self.memory = emptyAlignedU8Slice();
        self.tree = emptySlice(State);
        if (memory.len != 0) self.backing_allocator.free(memory);
        if (tree.len != 0) self.backing_allocator.free(tree);
    }

    pub fn secureDeinit(self: *BuddyAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.size_map.deinit();
        const memory = self.memory;
        const tree = self.tree;
        self.memory = emptyAlignedU8Slice();
        self.tree = emptySlice(State);
        if (memory.len != 0) {
            const bytes_ptr: [*]u8 = memory.ptr;
            secureZeroMemory(bytes_ptr, memory.len);
            self.backing_allocator.free(memory);
        }
        if (tree.len != 0) self.backing_allocator.free(tree);
    }

    fn leftChild(idx: usize) usize {
        return idx * 2 + 1;
    }

    fn rightChild(idx: usize) usize {
        return idx * 2 + 2;
    }

    fn parent(idx: usize) usize {
        return (idx - 1) / 2;
    }

    fn levelStart(level: u32) usize {
        return (@as(usize, 1) << @intCast(level)) - 1;
    }

    fn updateUp(self: *BuddyAllocator, idx: usize) void {
        const l = leftChild(idx);
        const r = rightChild(idx);
        if (r >= self.tree.len) return;
        const ls = self.tree[l];
        const rs = self.tree[r];
        self.tree[idx] = if (ls == .full and rs == .full)
            .full
        else if (ls == .free and rs == .free)
            .free
        else
            .split;
    }

    fn allocRec(self: *BuddyAllocator, idx: usize, cur_order: u32, want_order: u32) ?usize {
        if (idx >= self.tree.len) return null;
        const st = self.tree[idx];
        if (st == .full) return null;

        if (cur_order == want_order) {
            if (st != .free) return null;
            self.tree[idx] = .full;
            return idx;
        }

        const l = leftChild(idx);
        const r = rightChild(idx);
        if (r >= self.tree.len) return null;

        if (st == .free) {
            self.tree[idx] = .split;
            self.tree[l] = .free;
            self.tree[r] = .free;
        } else if (st != .split) {
            return null;
        }

        if (self.allocRec(l, cur_order - 1, want_order)) |found| {
            self.updateUp(idx);
            return found;
        }
        if (self.allocRec(r, cur_order - 1, want_order)) |found| {
            self.updateUp(idx);
            return found;
        }
        self.updateUp(idx);
        return null;
    }

    fn ptrFromIndex(self: *BuddyAllocator, idx: usize, order: u32) []align(PageSize) u8 {
        const level = self.max_order - order;
        const start = levelStart(level);
        const offset_in_level = idx - start;
        const block_size = @as(usize, 1) << @intCast(order);
        const byte_offset = offset_in_level * block_size;
        const p: [*]align(PageSize) u8 = @alignCast(self.memory.ptr + byte_offset);
        return p[0..block_size];
    }

    fn freeIndex(self: *BuddyAllocator, idx: usize, order: u32) void {
        _ = order;
        if (idx >= self.tree.len) return;
        self.tree[idx] = .free;
        var cur = idx;
        while (cur != 0) {
            const p = parent(cur);
            self.updateUp(p);
            cur = p;
        }
    }

    fn allocAlignedInternal(self: *BuddyAllocator, size: usize, alignment: usize) ![]u8 {
        if (size == 0) return error.InvalidSize;
        if (!isPow2(alignment)) return error.InvalidAlignment;

        self.mutex.lock();
        defer self.mutex.unlock();

        const min_block_size = @as(usize, 1) << @intCast(self.min_order);
        const base_needed = @max(size, min_block_size);
        var want_order: u32 = @intCast(std.math.log2_int_ceil(usize, base_needed));
        if (want_order < self.min_order) want_order = self.min_order;

        var order = want_order;
        while (order <= self.max_order) {
            const block_size = @as(usize, 1) << @intCast(order);
            if (block_size < alignment) {
                order += 1;
                continue;
            }
            const found = self.allocRec(0, self.max_order, order) orelse {
                order += 1;
                continue;
            };
            const block = self.ptrFromIndex(found, order);
            const out = block[0..size];
            self.size_map.put(@intFromPtr(out.ptr), .{ .order = order, .size = size }) catch {
                self.freeIndex(found, order);
                return error.OutOfMemory;
            };
            @memset(block, 0);
            return out;
        }
        return error.OutOfMemory;
    }

    pub fn allocator(self: *BuddyAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = buddyVtableAlloc,
                .resize = buddyVtableResize,
                .remap = buddyVtableRemap,
                .free = buddyVtableFree,
            },
        };
    }

    fn buddyVtableAlloc(ctx: *anyopaque, len: usize, log2_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *BuddyAllocator = @ptrCast(@alignCast(ctx));
        const align_bytes = log2_align.toByteUnits();
        const slice = self.allocAlignedInternal(len, align_bytes) catch return null;
        return slice.ptr;
    }

    fn buddyVtableResize(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = log2_align;
        _ = new_len;
        _ = ret_addr;
        return false;
    }


    fn buddyVtableRemap(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        if (buddyVtableResize(ctx, buf, log2_align, new_len, ret_addr)) {
            return buf.ptr;
        }
        return null;
    }

    fn buddyVtableFree(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, ret_addr: usize) void {
        _ = log2_align;
        _ = ret_addr;
        const self: *BuddyAllocator = @ptrCast(@alignCast(ctx));
        self.free(buf) catch {};
    }

    pub fn alloc(self: *BuddyAllocator, size: usize) ![]u8 {
        return self.allocAlignedInternal(size, 1);
    }

    pub fn free(self: *BuddyAllocator, ptr: []u8) !void {
        if (ptr.len == 0) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        const meta = self.size_map.get(@intFromPtr(ptr.ptr)) orelse return error.InvalidPointer;
        if (ptr.len != meta.size) return error.InvalidPointer;

        const ptr_addr = @intFromPtr(ptr.ptr);
        const base = @intFromPtr(self.memory.ptr);
        if (ptr_addr < base or ptr_addr >= base + self.memory.len) return error.InvalidPointer;

        const block_size = @as(usize, 1) << @intCast(meta.order);
        const offset = ptr_addr - base;
        if (offset % block_size != 0) return error.InvalidPointer;

        _ = self.size_map.fetchRemove(@intFromPtr(ptr.ptr));

        const level = self.max_order - meta.order;
        const start = levelStart(level);
        const offset_in_level = offset / block_size;
        const idx = start + offset_in_level;
        self.freeIndex(idx, meta.order);
    }

    pub fn secureFree(self: *BuddyAllocator, ptr: []u8) !void {
        if (ptr.len == 0) return;

        self.mutex.lock();
        defer self.mutex.unlock();

        const meta = self.size_map.get(@intFromPtr(ptr.ptr)) orelse return error.InvalidPointer;
        if (ptr.len != meta.size) return error.InvalidPointer;

        const ptr_addr = @intFromPtr(ptr.ptr);
        const base = @intFromPtr(self.memory.ptr);
        if (ptr_addr < base or ptr_addr >= base + self.memory.len) return error.InvalidPointer;

        const block_size = @as(usize, 1) << @intCast(meta.order);
        const offset = ptr_addr - base;
        if (offset % block_size != 0) return error.InvalidPointer;

        _ = self.size_map.fetchRemove(@intFromPtr(ptr.ptr));

        secureZeroMemory(ptr.ptr, ptr.len);

        const level = self.max_order - meta.order;
        const start = levelStart(level);
        const offset_in_level = offset / block_size;
        const idx = start + offset_in_level;
        self.freeIndex(idx, meta.order);
    }
};

pub const MutexQueue = struct {
    head: usize,
    tail: usize,
    mask: usize,
    buffer: []?*anyopaque,
    allocator: Allocator,
    capacity: usize,
    mutex: Mutex,

    pub fn init(allocator: Allocator, capacity: usize) !MutexQueue {
        if (capacity < 2 or !isPow2(capacity)) return error.InvalidSize;
        const buf = try allocator.alloc(?*anyopaque, capacity);
        @memset(buf, null);
        return .{
            .head = 0,
            .tail = 0,
            .mask = capacity - 1,
            .buffer = buf,
            .allocator = allocator,
            .capacity = capacity,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *MutexQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const buf = self.buffer;
        self.buffer = emptySlice(?*anyopaque);
        self.capacity = 0;
        self.mask = 0;
        if (buf.len != 0) self.allocator.free(buf);
    }

    pub fn enqueue(self: *MutexQueue, item: *anyopaque) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        const next_tail = (self.tail + 1) & self.mask;
        if (next_tail == self.head) return false;
        self.buffer[self.tail] = item;
        self.tail = next_tail;
        return true;
    }

    pub fn dequeue(self: *MutexQueue) ?*anyopaque {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.head == self.tail) return null;
        const item = self.buffer[self.head];
        self.buffer[self.head] = null;
        self.head = (self.head + 1) & self.mask;
        return item;
    }
};

pub const LockFreeQueue = struct {
    head: usize,
    tail: usize,
    mask: usize,
    buffer: []Cell,
    allocator: Allocator,

    const Cell = struct {
        sequence: usize,
        data: usize,
    };

    pub fn init(allocator: Allocator, capacity: usize) !LockFreeQueue {
        if (capacity < 2 or !isPow2(capacity)) return error.InvalidSize;
        const buffer = try allocator.alloc(Cell, capacity);
        for (buffer, 0..) |*cell, i| {
            cell.sequence = i;
            cell.data = 0;
        }
        return .{
            .head = 0,
            .tail = 0,
            .mask = capacity - 1,
            .buffer = buffer,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LockFreeQueue) void {
        const buf = self.buffer;
        self.buffer = emptySlice(Cell);
        self.head = 0;
        self.tail = 0;
        self.mask = 0;
        if (buf.len != 0) self.allocator.free(buf);
    }

    pub fn enqueue(self: *LockFreeQueue, item: *anyopaque) bool {
        var pos = @atomicLoad(usize, &self.tail, .monotonic);
        while (true) {
            const cell = &self.buffer[pos & self.mask];
            const seq = @atomicLoad(usize, &cell.sequence, .acquire);
            const diff: isize = @as(isize, @bitCast(seq)) -% @as(isize, @bitCast(pos));
            if (diff == 0) {
                cell.data = @intFromPtr(item);
                if (@cmpxchgWeak(usize, &self.tail, pos, pos + 1, .monotonic, .monotonic) == null) {
                    @atomicStore(usize, &cell.sequence, pos + 1, .release);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = @atomicLoad(usize, &self.tail, .monotonic);
            }
        }
    }

    pub fn dequeue(self: *LockFreeQueue) ?*anyopaque {
        var pos = @atomicLoad(usize, &self.head, .monotonic);
        while (true) {
            const cell = &self.buffer[pos & self.mask];
            const seq = @atomicLoad(usize, &cell.sequence, .acquire);
            const diff: isize = @as(isize, @bitCast(seq)) -% @as(isize, @bitCast(pos + 1));
            if (diff == 0) {
                if (@cmpxchgWeak(usize, &self.head, pos, pos + 1, .monotonic, .monotonic) == null) {
                    const data = cell.data;
                    @atomicStore(usize, &cell.sequence, pos + self.mask + 1, .release);
                    return @ptrFromInt(data);
                }
            } else if (diff < 0) {
                return null;
            } else {
                pos = @atomicLoad(usize, &self.head, .monotonic);
            }
        }
    }
};

pub const MutexStack = struct {
    mutex: Mutex,
    top: ?*Node,
    allocator: Allocator,

    const Node = struct {
        value: *anyopaque,
        next: ?*Node,
    };

    pub fn init(allocator: Allocator) MutexStack {
        return .{
            .mutex = .{},
            .top = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MutexStack) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        var cur = self.top;
        self.top = null;
        while (cur) |n| {
            const next = n.next;
            n.next = null;
            self.allocator.destroy(n);
            cur = next;
        }
    }

    pub fn push(self: *MutexStack, value: *anyopaque) !void {
        const node = try self.allocator.create(Node);
        node.* = .{ .value = value, .next = null };
        self.mutex.lock();
        defer self.mutex.unlock();
        node.next = self.top;
        self.top = node;
    }

    pub fn pop(self: *MutexStack) ?*anyopaque {
        self.mutex.lock();
        defer self.mutex.unlock();
        const node = self.top orelse return null;
        self.top = node.next;
        node.next = null;
        const value = node.value;
        self.allocator.destroy(node);
        return value;
    }
};

pub const LockFreeStack = struct {
    top: usize,
    allocator: Allocator,

    const Node = struct {
        value: *anyopaque,
        next: usize,
    };

    pub fn init(allocator: Allocator) LockFreeStack {
        return .{ .top = 0, .allocator = allocator };
    }

    pub fn deinit(self: *LockFreeStack) void {
        var cur = @atomicLoad(usize, &self.top, .acquire);
        @atomicStore(usize, &self.top, 0, .release);
        while (cur != 0) {
            const node: *Node = @ptrFromInt(cur);
            cur = node.next;
            self.allocator.destroy(node);
        }
    }

    pub fn push(self: *LockFreeStack, value: *anyopaque) !void {
        const node = try self.allocator.create(Node);
        while (true) {
            const old = @atomicLoad(usize, &self.top, .acquire);
            node.* = .{ .value = value, .next = old };
            if (@cmpxchgWeak(usize, &self.top, old, @intFromPtr(node), .acq_rel, .acquire) == null) return;
        }
    }

    pub fn pop(self: *LockFreeStack) ?*anyopaque {
        while (true) {
            const old = @atomicLoad(usize, &self.top, .acquire);
            if (old == 0) return null;
            const node: *Node = @ptrFromInt(old);
            if (@cmpxchgWeak(usize, &self.top, old, node.next, .acq_rel, .acquire) == null) {
                const value = node.value;
                self.allocator.destroy(node);
                return value;
            }
        }
    }
};

pub const PageAllocator = struct {
    pages: []align(PageSize) u8,
    allocator: Allocator,
    page_size: usize,
    bitmap: []u64,
    mutex: Mutex,

    pub fn init(allocator: Allocator, num_pages: usize) !PageAllocator {
        if (num_pages == 0) return error.InvalidSize;
        const total = try mulChecked(num_pages, PageSize);
        const pages = try allocator.alignedAlloc(u8, PageSize, total);
        const bitmap_words = (num_pages + 63) / 64;
        const bitmap = try allocator.alloc(u64, bitmap_words);
        @memset(bitmap, 0);
        return .{
            .pages = pages,
            .allocator = allocator,
            .page_size = PageSize,
            .bitmap = bitmap,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *PageAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const pages = self.pages;
        const bitmap = self.bitmap;
        self.pages = emptyAlignedU8Slice();
        self.bitmap = emptySlice(u64);
        if (pages.len != 0) self.allocator.free(pages);
        if (bitmap.len != 0) self.allocator.free(bitmap);
    }

    pub fn secureDeinit(self: *PageAllocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const pages = self.pages;
        const bitmap = self.bitmap;
        self.pages = emptyAlignedU8Slice();
        self.bitmap = emptySlice(u64);
        if (pages.len != 0) {
            const bytes_ptr: [*]u8 = pages.ptr;
            secureZeroMemory(bytes_ptr, pages.len);
            self.allocator.free(pages);
        }
        if (bitmap.len != 0) self.allocator.free(bitmap);
    }

    fn isPageFree(self: *const PageAllocator, page_idx: usize) bool {
        const word_idx = page_idx / 64;
        const bit_idx: u6 = @intCast(page_idx % 64);
        return (self.bitmap[word_idx] & (@as(u64, 1) << bit_idx)) == 0;
    }

    fn setPageUsed(self: *PageAllocator, page_idx: usize) void {
        const word_idx = page_idx / 64;
        const bit_idx: u6 = @intCast(page_idx % 64);
        self.bitmap[word_idx] |= (@as(u64, 1) << bit_idx);
    }

    fn setPageFree(self: *PageAllocator, page_idx: usize) void {
        const word_idx = page_idx / 64;
        const bit_idx: u6 = @intCast(page_idx % 64);
        self.bitmap[word_idx] &= ~(@as(u64, 1) << bit_idx);
    }

    pub fn allocPages(self: *PageAllocator, num_pages: usize) ?[]u8 {
        if (num_pages == 0) return emptyU8Slice();
        self.mutex.lock();
        defer self.mutex.unlock();

        const total_pages = self.pages.len / self.page_size;
        if (num_pages > total_pages) return null;
        var consecutive: usize = 0;
        var start_page: usize = 0;
        var i: usize = 0;
        while (i < total_pages) : (i += 1) {
            if (self.isPageFree(i)) {
                if (consecutive == 0) start_page = i;
                consecutive += 1;
                if (consecutive == num_pages) {
                    const offset = start_page * self.page_size;
                    const size = num_pages * self.page_size;
                    var j: usize = start_page;
                    while (j < start_page + num_pages) : (j += 1) self.setPageUsed(j);
                    @memset(self.pages[offset .. offset + size], 0);
                    return self.pages[offset .. offset + size];
                }
            } else {
                consecutive = 0;
            }
        }
        return null;
    }

    pub fn freePages(self: *PageAllocator, ptr: []u8) !void {
        if (ptr.len == 0) return;
        if (ptr.len % self.page_size != 0) return error.InvalidPointer;

        self.mutex.lock();
        defer self.mutex.unlock();

        const pages_start = @intFromPtr(self.pages.ptr);
        const pages_end = pages_start + self.pages.len;
        const ptr_addr = @intFromPtr(ptr.ptr);
        if (ptr_addr < pages_start or ptr_addr >= pages_end) return error.InvalidPointer;
        const offset = ptr_addr - pages_start;
        if (offset % self.page_size != 0) return error.InvalidPointer;

        const start_page = offset / self.page_size;
        const num_pages = ptr.len / self.page_size;
        if (start_page + num_pages > self.pages.len / self.page_size) return error.InvalidPointer;
        var i: usize = start_page;
        while (i < start_page + num_pages) : (i += 1) {
            if (self.isPageFree(i)) return error.DoubleFree;
        }
        i = start_page;
        while (i < start_page + num_pages) : (i += 1) self.setPageFree(i);
    }

    pub fn secureFreePages(self: *PageAllocator, ptr: []u8) !void {
        if (ptr.len == 0) return;
        if (ptr.len % self.page_size != 0) return error.InvalidPointer;

        self.mutex.lock();
        defer self.mutex.unlock();

        const pages_start = @intFromPtr(self.pages.ptr);
        const pages_end = pages_start + self.pages.len;
        const ptr_addr = @intFromPtr(ptr.ptr);
        if (ptr_addr < pages_start or ptr_addr >= pages_end) return error.InvalidPointer;
        const offset = ptr_addr - pages_start;
        if (offset % self.page_size != 0) return error.InvalidPointer;

        const start_page = offset / self.page_size;
        const num_pages = ptr.len / self.page_size;
        if (start_page + num_pages > self.pages.len / self.page_size) return error.InvalidPointer;
        var i: usize = start_page;
        while (i < start_page + num_pages) : (i += 1) {
            if (self.isPageFree(i)) return error.DoubleFree;
        }
        secureZeroMemory(ptr.ptr, ptr.len);
        i = start_page;
        while (i < start_page + num_pages) : (i += 1) self.setPageFree(i);
    }

    pub fn mapPage(self: *PageAllocator, page_idx: usize) ?[]align(PageSize) u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const total_pages = self.pages.len / self.page_size;
        if (page_idx >= total_pages) return null;
        const offset = page_idx * self.page_size;
        const p: [*]align(PageSize) u8 = @alignCast(self.pages.ptr + offset);
        return p[0..self.page_size];
    }
};

pub const ZeroCopySlice = struct {
    ptr: [*]const u8,
    len: usize,

    pub fn init(ptr: [*]const u8, len: usize) ZeroCopySlice {
        return .{ .ptr = ptr, .len = len };
    }

    pub fn slice(self: *const ZeroCopySlice, start: usize, end: usize) ZeroCopySlice {
        const s = @min(start, self.len);
        const e = @min(end, self.len);
        if (s >= e) return .{ .ptr = self.ptr + s, .len = 0 };
        return .{ .ptr = self.ptr + s, .len = e - s };
    }

    pub fn copyTo(self: *const ZeroCopySlice, allocator: Allocator) ![]u8 {
        if (self.len == 0) return emptyU8Slice();
        const buf = try allocator.alloc(u8, self.len);
        @memcpy(buf, self.asBytes());
        return buf;
    }

    pub fn asBytes(self: *const ZeroCopySlice) []const u8 {
        return self.ptr[0..self.len];
    }
};

pub const ResizeBuffer = struct {
    buffer: []u8,
    len: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) ResizeBuffer {
        return .{ .buffer = emptyU8Slice(), .len = 0, .allocator = allocator };
    }

    pub fn deinit(self: *ResizeBuffer) void {
        if (self.buffer.len != 0) {
            self.allocator.free(self.buffer);
        }
        self.buffer = emptyU8Slice();
        self.len = 0;
    }

    pub fn secureDeinit(self: *ResizeBuffer) void {
        if (self.buffer.len != 0) {
            secureZeroMemory(self.buffer.ptr, self.buffer.len);
            self.allocator.free(self.buffer);
        }
        self.buffer = emptyU8Slice();
        self.len = 0;
    }

    pub fn append(self: *ResizeBuffer, data: []const u8) !void {
        const new_len = try addChecked(self.len, data.len);
        if (new_len > self.buffer.len) {
            const new_cap = std.math.ceilPowerOfTwo(usize, @max(new_len, 16)) catch return error.OutOfMemory;
            if (self.buffer.len == 0) {
                self.buffer = try self.allocator.alloc(u8, new_cap);
                @memset(self.buffer, 0);
            } else {
                const old_len = self.buffer.len;
                const resized = self.allocator.realloc(self.buffer, new_cap) catch return error.OutOfMemory;
                self.buffer = resized;
                if (new_cap > old_len) @memset(self.buffer[old_len..new_cap], 0);
            }
        }
        mem.copyForwards(u8, self.buffer[self.len..new_len], data);
        self.len = new_len;
    }

    pub fn clear(self: *ResizeBuffer) void {
        self.len = 0;
    }

    pub fn secureClear(self: *ResizeBuffer) void {
        if (self.len != 0) secureZeroMemory(self.buffer.ptr, self.len);
        self.len = 0;
    }

    pub fn toOwnedSlice(self: *ResizeBuffer) ![]u8 {
        if (self.len == 0) {
            self.deinit();
            return emptyU8Slice();
        }
        const out = try self.allocator.alloc(u8, self.len);
        errdefer self.allocator.free(out);
        mem.copyForwards(u8, out, self.buffer[0..self.len]);
        self.deinit();
        return out;
    }
};

pub fn zeroCopyTransfer(src: []const u8, dest: []u8) void {
    const n = @min(src.len, dest.len);
    if (n == 0) return;
    mem.copyForwards(u8, dest[0..n], src[0..n]);
}

pub fn alignedAlloc(allocator: Allocator, comptime T: type, n: usize, alignment: usize) ![]T {
    if (n == 0) return emptySlice(T);
    return runtimeAlignedAlloc(allocator, T, n, alignment);
}

pub fn cacheAlignedAlloc(allocator: Allocator, size: usize, cache_line_size: usize) ![]u8 {
    if (size == 0) return emptyU8Slice();
    return alignedAlloc(allocator, u8, size, cache_line_size);
}

pub fn sliceMemory(base: [*]u8, offset: usize, size: usize, buffer_size: usize) ![]u8 {
    if (offset > buffer_size) return error.OffsetOutOfBounds;
    const end = @addWithOverflow(offset, size);
    if (end[1] != 0) return error.SliceOverflow;
    if (end[0] > buffer_size) return error.SliceOutOfBounds;
    return base[offset..end[0]];
}

pub fn zeroInitMemory(ptr: [*]u8, size: usize) void {
    if (size == 0) return;
    @memset(ptr[0..size], 0);
}

pub fn secureZeroMemory(ptr: [*]u8, size: usize) void {
    if (size == 0) return;
    const p: [*]volatile u8 = @ptrCast(ptr);
    var i: usize = 0;
    while (i < size) : (i += 1) p[i] = 0;
    _ = p[size - 1];
}

pub fn constantTimeCompare(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    if (a.len == 0) return true;
    const pa: [*]const volatile u8 = @ptrCast(a.ptr);
    const pb: [*]const volatile u8 = @ptrCast(b.ptr);
    var diff: u8 = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 1) diff |= pa[i] ^ pb[i];
    _ = @atomicLoad(u8, &diff, .seq_cst);
    return diff == 0;
}

pub fn compareMemory(a: []const u8, b: []const u8) bool {
    return constantTimeCompare(a, b);
}

pub fn hashMemory(data: []const u8) u64 {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(data);
    return hasher.final();
}

pub fn alignForward(addr: usize, alignment: usize) !usize {
    return alignForwardChecked(addr, alignment);
}

pub fn alignBackward(addr: usize, alignment: usize) !usize {
    if (!isPow2(alignment)) return error.InvalidAlignment;
    return mem.alignBackward(usize, addr, alignment);
}

pub fn isAligned(addr: usize, alignment: usize) bool {
    return mem.isAligned(addr, alignment);
}

pub fn pageAlignedSize(size: usize) usize {
    return mem.alignForward(usize, size, PageSize);
}

pub fn memoryBarrier() void {
    _ = @atomicLoad(u8, &fence_dummy, .seq_cst);
}

pub fn readMemoryFence() void {
    _ = @atomicLoad(u8, &fence_dummy, .acquire);
}

pub fn writeMemoryFence() void {
    @atomicStore(u8, &fence_dummy, 0, .release);
}

var fence_dummy: u8 = 0;

pub fn compareExchangeMemory(ptr: *u64, expected: u64, desired: u64) bool {
    return @cmpxchgStrong(u64, ptr, expected, desired, .seq_cst, .seq_cst) == null;
}

pub fn atomicLoad(ptr: *u64) u64 {
    return @atomicLoad(u64, ptr, .seq_cst);
}

pub fn atomicStore(ptr: *u64, value: u64) void {
    @atomicStore(u64, ptr, value, .seq_cst);
}

pub fn atomicAdd(ptr: *u64, delta: u64) u64 {
    return @atomicRmw(u64, ptr, .Add, delta, .seq_cst);
}

pub fn atomicSub(ptr: *u64, delta: u64) u64 {
    return @atomicRmw(u64, ptr, .Sub, delta, .seq_cst);
}

pub fn atomicAnd(ptr: *u64, mask: u64) u64 {
    return @atomicRmw(u64, ptr, .And, mask, .seq_cst);
}

pub fn atomicOr(ptr: *u64, mask: u64) u64 {
    return @atomicRmw(u64, ptr, .Or, mask, .seq_cst);
}

pub fn atomicXor(ptr: *u64, mask: u64) u64 {
    return @atomicRmw(u64, ptr, .Xor, mask, .seq_cst);
}

pub fn atomicInc(ptr: *u64) u64 {
    return atomicAdd(ptr, 1);
}

pub fn atomicDec(ptr: *u64) u64 {
    return atomicSub(ptr, 1);
}

pub fn memoryEfficientCopy(src: []const u8, dest: []u8) !void {
    if (dest.len < src.len) return error.DestinationTooSmall;
    @memcpy(dest[0..src.len], src);
}

pub fn secureErase(ptr: [*]u8, size: usize) void {
    if (size == 0) return;
    const p: [*]volatile u8 = @ptrCast(ptr);
    var i: usize = 0;
    while (i < size) : (i += 1) p[i] = 0x55;
    i = 0;
    while (i < size) : (i += 1) p[i] = 0xAA;
    i = 0;
    while (i < size) : (i += 1) p[i] = 0x00;
    _ = p[size - 1];
}

pub fn duplicateMemory(allocator: Allocator, data: []const u8) ![]u8 {
    if (data.len == 0) return emptyU8Slice();
    const dup = try allocator.alloc(u8, data.len);
    @memcpy(dup, data);
    return dup;
}

pub fn concatenateMemory(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    const total = try addChecked(a.len, b.len);
    if (total == 0) return emptyU8Slice();
    const cat = try allocator.alloc(u8, total);
    @memcpy(cat[0..a.len], a);
    @memcpy(cat[a.len..total], b);
    return cat;
}

pub fn searchMemory(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0) return null;
    return mem.indexOf(u8, haystack, needle);
}

pub fn replaceMemory(data: []u8, old: u8, new: u8) void {
    for (data) |*c| {
        if (c.* == old) c.* = new;
    }
}

pub fn reverseMemory(data: []u8) void {
    mem.reverse(u8, data);
}

pub fn rotateMemory(data: []u8, shift: usize) void {
    mem.rotate(u8, data, shift);
}

pub fn countMemory(data: []const u8, value: u8) usize {
    var count: usize = 0;
    for (data) |c| {
        if (c == value) count += 1;
    }
    return count;
}

pub fn sumMemory(data: []const u8) u64 {
    var sum: u64 = 0;
    for (data) |c| sum +%= c;
    return sum;
}

pub fn productMemory(data: []const u8) u64 {
    var prod: u64 = 1;
    for (data) |c| prod *%= c;
    return prod;
}

pub fn minMemory(data: []const u8) EmptyError!u8 {
    if (data.len == 0) return error.Empty;
    var m = data[0];
    for (data[1..]) |c| {
        if (c < m) m = c;
    }
    return m;
}

pub fn maxMemory(data: []const u8) EmptyError!u8 {
    if (data.len == 0) return error.Empty;
    var m = data[0];
    for (data[1..]) |c| {
        if (c > m) m = c;
    }
    return m;
}

pub fn sortMemory(data: []u8) void {
    std.sort.heap(u8, data, {}, std.sort.asc(u8));
}

pub fn shuffleMemory(data: []u8, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    prng.random().shuffle(u8, data);
}

pub fn uniqueMemory(allocator: Allocator, data: []const u8) ![]u8 {
    var seen: [256]bool = [_]bool{false} ** 256;
    var count: usize = 0;
    for (data) |c| {
        if (!seen[c]) {
            seen[c] = true;
            count += 1;
        }
    }
    if (count == 0) return emptyU8Slice();
    const out = try allocator.alloc(u8, count);
    var i: usize = 0;
    var b: usize = 0;
    while (b < 256) : (b += 1) {
        if (seen[b]) {
            out[i] = @intCast(b);
            i += 1;
        }
    }
    return out;
}

pub fn intersectMemory(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    var set_a: [256]bool = [_]bool{false} ** 256;
    var added: [256]bool = [_]bool{false} ** 256;
    for (a) |c| set_a[c] = true;
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();
    for (b) |c| {
        if (set_a[c] and !added[c]) {
            added[c] = true;
            try list.append(c);
        }
    }
    return try list.toOwnedSlice(allocator);
}

pub fn unionMemory(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    var seen: [256]bool = [_]bool{false} ** 256;
    for (a) |c| seen[c] = true;
    for (b) |c| seen[c] = true;
    var count: usize = 0;
    for (seen) |v| {
        if (v) count += 1;
    }
    if (count == 0) return emptyU8Slice();
    const out = try allocator.alloc(u8, count);
    var i: usize = 0;
    var idx: usize = 0;
    while (idx < 256) : (idx += 1) {
        if (seen[idx]) {
            out[i] = @intCast(idx);
            i += 1;
        }
    }
    return out;
}

pub fn differenceMemory(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    var set_b: [256]bool = [_]bool{false} ** 256;
    var added: [256]bool = [_]bool{false} ** 256;
    for (b) |c| set_b[c] = true;
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();
    for (a) |c| {
        if (!set_b[c] and !added[c]) {
            added[c] = true;
            try list.append(c);
        }
    }
    return try list.toOwnedSlice(allocator);
}

pub fn isSubsetMemory(allocator: Allocator, a: []const u8, b: []const u8) !bool {
    _ = allocator;
    var set_b: [256]bool = [_]bool{false} ** 256;
    for (b) |c| set_b[c] = true;
    for (a) |c| if (!set_b[c]) return false;
    return true;
}

pub fn isSupersetMemory(allocator: Allocator, a: []const u8, b: []const u8) !bool {
    return isSubsetMemory(allocator, b, a);
}

pub fn isDisjointMemory(allocator: Allocator, a: []const u8, b: []const u8) !bool {
    _ = allocator;
    var set_a: [256]bool = [_]bool{false} ** 256;
    for (a) |c| set_a[c] = true;
    for (b) |c| if (set_a[c]) return false;
    return true;
}

pub const MemoryStats = struct {
    allocated: usize,
    freed: usize,
    peak: usize,
};

var global_memory_stats: MemoryStats = .{ .allocated = 0, .freed = 0, .peak = 0 };
var global_memory_stats_mutex: Mutex = .{};

pub fn trackAllocation(size: usize) void {
    global_memory_stats_mutex.lock();
    defer global_memory_stats_mutex.unlock();
    const new_alloc = std.math.add(usize, global_memory_stats.allocated, size) catch std.math.maxInt(usize);
    global_memory_stats.allocated = new_alloc;
    const current = saturatingSub(global_memory_stats.allocated, global_memory_stats.freed);
    if (current > global_memory_stats.peak) global_memory_stats.peak = current;
}

pub fn trackFree(size: usize) void {
    global_memory_stats_mutex.lock();
    defer global_memory_stats_mutex.unlock();
    global_memory_stats.freed = std.math.add(usize, global_memory_stats.freed, size) catch std.math.maxInt(usize);
}

pub fn getMemoryStats() MemoryStats {
    global_memory_stats_mutex.lock();
    defer global_memory_stats_mutex.unlock();
    return global_memory_stats;
}

pub fn resetMemoryStats() void {
    global_memory_stats_mutex.lock();
    defer global_memory_stats_mutex.unlock();
    global_memory_stats = .{ .allocated = 0, .freed = 0, .peak = 0 };
}

pub fn cleanupGlobalState() void {
    global_memory_stats_mutex.lock();
    defer global_memory_stats_mutex.unlock();
    global_memory_stats = .{ .allocated = 0, .freed = 0, .peak = 0 };
}

pub fn memoryFootprint() usize {
    const s = getMemoryStats();
    return saturatingSub(s.allocated, s.freed);
}

pub fn memoryPressure() f32 {
    const s = getMemoryStats();
    const cur = saturatingSub(s.allocated, s.freed);
    if (s.peak == 0) return 0.0;
    const ratio = @as(f32, @floatFromInt(cur)) / @as(f32, @floatFromInt(s.peak));
    return if (ratio > 1.0) 1.0 else ratio;
}

pub const TrackingAllocator = struct {
    parent: Allocator,

    pub fn init(parent: Allocator) TrackingAllocator {
        return .{ .parent = parent };
    }

    pub fn allocator(self: *TrackingAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = trackingAlloc,
                .resize = trackingResize,
                .remap = trackingRemap,
                .free = trackingFree,
            },
        };
    }

    fn trackingAlloc(ctx: *anyopaque, len: usize, log2_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const ptr = self.parent.vtable.alloc(self.parent.ptr, len, log2_align, ret_addr);
        if (ptr != null) trackAllocation(len);
        return ptr;
    }

    fn trackingResize(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const old_len = buf.len;
        const ok = self.parent.vtable.resize(self.parent.ptr, buf, log2_align, new_len, ret_addr);
        if (ok) {
            if (new_len > old_len) trackAllocation(new_len - old_len) else trackFree(old_len - new_len);
        }
        return ok;
    }


    fn trackingRemap(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        if (trackingResize(ctx, buf, log2_align, new_len, ret_addr)) {
            return buf.ptr;
        }
        return null;
    }

    fn trackingFree(ctx: *anyopaque, buf: []u8, log2_align: std.mem.Alignment, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        trackFree(buf.len);
        self.parent.vtable.free(self.parent.ptr, buf, log2_align, ret_addr);
    }
};

pub const ReadWriteLock = struct {
    readers: usize,
    writer: bool,
    waiting_writers: usize,
    mutex: Mutex,
    cond: CondVar,

    pub fn init() ReadWriteLock {
        return .{
            .readers = 0,
            .writer = false,
            .waiting_writers = 0,
            .mutex = .{},
            .cond = .{},
        };
    }

    pub fn readLock(self: *ReadWriteLock) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.writer or self.waiting_writers != 0) self.cond.wait(&self.mutex);
        self.readers += 1;
    }

    pub fn readUnlock(self: *ReadWriteLock) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        std.debug.assert(self.readers > 0);
        self.readers -= 1;
        if (self.readers == 0) self.cond.broadcast();
    }

    pub fn writeLock(self: *ReadWriteLock) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.waiting_writers += 1;
        defer self.waiting_writers -= 1;
        while (self.writer or self.readers != 0) self.cond.wait(&self.mutex);
        self.writer = true;
    }

    pub fn writeUnlock(self: *ReadWriteLock) void {
        self.mutex.lock();
        self.writer = false;
        self.cond.broadcast();
        self.mutex.unlock();
    }
};

pub fn atomicFlagTestAndSet(flag: *bool) bool {
    return @atomicRmw(bool, flag, .Xchg, true, .seq_cst);
}

pub fn atomicFlagClear(flag: *bool) void {
    @atomicStore(bool, flag, false, .seq_cst);
}

pub fn spinLockAcquire(lock: *u64) void {
    while (@cmpxchgStrong(u64, lock, @as(u64, 0), @as(u64, 1), .acquire, .monotonic) != null) {
        std.atomic.spinLoopHint();
    }
}

pub fn spinLockRelease(lock: *u64) void {
    @atomicStore(u64, lock, @as(u64, 0), .release);
}

pub fn memoryPatternFill(ptr: [*]u8, size: usize, pattern: []const u8) !void {
    if (size == 0) return;
    if (pattern.len == 0) return error.InvalidPattern;
    var i: usize = 0;
    while (i < size) : (i += pattern.len) {
        const copy_len = @min(pattern.len, size - i);
        @memcpy(ptr[i .. i + copy_len], pattern[0..copy_len]);
    }
}

pub fn memoryPatternVerify(ptr: [*]const u8, size: usize, pattern: []const u8) !bool {
    if (size == 0) return true;
    if (pattern.len == 0) return error.InvalidPattern;
    var i: usize = 0;
    while (i < size) : (i += pattern.len) {
        const check_len = @min(pattern.len, size - i);
        if (!mem.eql(u8, ptr[i .. i + check_len], pattern[0..check_len])) return false;
    }
    return true;
}

pub fn virtualMemoryMap(addr: ?*anyopaque, size: usize, prot: u32, flags: u32) !*anyopaque {
    if (builtin.os.tag == .windows) return error.Unsupported;
    if (size == 0) return error.InvalidSize;
    const hint: ?[*]align(PageSize) u8 = if (addr) |a| @ptrFromInt(mem.alignBackward(usize, @intFromPtr(a), PageSize)) else null;
    const map_flags: std.posix.system.MAP = @bitCast(flags);
    const mapped = try std.posix.mmap(hint, size, prot, map_flags, -1, 0);
    return @ptrCast(mapped.ptr);
}

pub fn virtualMemoryUnmap(addr: *anyopaque, size: usize) !void {
    if (builtin.os.tag == .windows) return error.Unsupported;
    if (size == 0) return;
    const base_addr = @intFromPtr(addr);
    const aligned_addr = mem.alignBackward(usize, base_addr, PageSize);
    const delta = base_addr - aligned_addr;
    const span = try addChecked(size, delta);
    const aligned_size = mem.alignForward(usize, span, PageSize);
    const p: [*]align(PageSize) u8 = @ptrFromInt(aligned_addr);
    std.posix.munmap(p[0..aligned_size]);
}

pub fn protectMemory(addr: *anyopaque, size: usize, prot: u32) !void {
    if (builtin.os.tag == .windows) return error.Unsupported;
    if (size == 0) return error.InvalidSize;
    const base_addr = @intFromPtr(addr);
    const aligned_addr = mem.alignBackward(usize, base_addr, PageSize);
    const delta = base_addr - aligned_addr;
    const span = try addChecked(size, delta);
    const aligned_size = mem.alignForward(usize, span, PageSize);
    const p: [*]align(PageSize) u8 = @ptrFromInt(aligned_addr);
    try std.posix.mprotect(p[0..aligned_size], prot);
}

pub fn lockMemory(addr: *anyopaque, size: usize) !void {
    if (builtin.os.tag == .windows) return error.Unsupported;
    if (size == 0) return error.InvalidSize;
    const base_addr = @intFromPtr(addr);
    const aligned_addr = mem.alignBackward(usize, base_addr, PageSize);
    const delta = base_addr - aligned_addr;
    const span = try addChecked(size, delta);
    const aligned_size = mem.alignForward(usize, span, PageSize);
    const p: [*]align(PageSize) u8 = @ptrFromInt(aligned_addr);
    try std.posix.mlock(p[0..aligned_size]);
}

pub fn unlockMemory(addr: *anyopaque, size: usize) !void {
    if (builtin.os.tag == .windows) return error.Unsupported;
    if (size == 0) return;
    const base_addr = @intFromPtr(addr);
    const aligned_addr = mem.alignBackward(usize, base_addr, PageSize);
    const delta = base_addr - aligned_addr;
    const span = try addChecked(size, delta);
    const aligned_size = mem.alignForward(usize, span, PageSize);
    const p: [*]align(PageSize) u8 = @ptrFromInt(aligned_addr);
    try std.posix.munlock(p[0..aligned_size]);
}

pub fn adviseMemory(addr: *anyopaque, size: usize, advice: u32) !void {
    if (builtin.os.tag == .windows) return error.Unsupported;
    if (size == 0) return error.InvalidSize;
    const base_addr = @intFromPtr(addr);
    const aligned_addr = mem.alignBackward(usize, base_addr, PageSize);
    const delta = base_addr - aligned_addr;
    const span = try addChecked(size, delta);
    const aligned_size = mem.alignForward(usize, span, PageSize);
    const p: [*]align(PageSize) u8 = @ptrFromInt(aligned_addr);
    try std.posix.madvise(p, aligned_size, advice);
}

pub fn prefetchMemory(addr: *const anyopaque, size: usize) void {
    const cache_line = MemoryConfig.CACHE_LINE_SIZE;
    const p: [*]const u8 = @ptrCast(addr);
    var i: usize = 0;
    while (i < size) : (i += cache_line) {
        @prefetch(p + i, .{ .rw = .read, .locality = 1, .cache = .data });
    }
}

pub fn trimExcessCapacity(allocator: Allocator, buf: []u8, used: usize) ![]u8 {
    if (used > buf.len) return error.InvalidSize;
    if (used == buf.len) return buf;
    if (used == 0) {
        allocator.free(buf);
        return emptyU8Slice();
    }
    const out = try allocator.alloc(u8, used);
    @memcpy(out, buf[0..used]);
    allocator.free(buf);
    return out;
}

pub fn splitMemory(allocator: Allocator, data: []const u8, delim: u8) ![][]const u8 {
    var parts = std.ArrayList([]const u8).init(allocator);
    defer parts.deinit();
    var start: usize = 0;
    var i: usize = 0;
    while (i < data.len) : (i += 1) {
        if (data[i] == delim) {
            try parts.append(data[start..i]);
            start = i + 1;
        }
    }
    try parts.append(data[start..]);
    return try parts.toOwnedSlice(allocator);
}

pub fn branchlessSelect(cond: bool, true_val: usize, false_val: usize) usize {
    const mask: usize = ~@as(usize, 0) *% @as(usize, @intFromBool(cond));
    return (true_val & mask) | (false_val & ~mask);
}

pub fn criticalSectionEnter(mutex: *Mutex) void {
    mutex.lock();
}

pub fn criticalSectionExit(mutex: *Mutex) void {
    mutex.unlock();
}

pub fn waitOnCondition(cond: *CondVar, mutex: *Mutex, predicate: *const fn () bool) void {
    while (!predicate()) cond.wait(mutex);
}

pub fn signalCondition(cond: *CondVar) void {
    cond.signal();
}

pub fn broadcastCondition(cond: *CondVar) void {
    cond.broadcast();
}

pub fn semaphoreWait(sem: *Semaphore) void {
    sem.wait();
}

pub fn semaphorePost(sem: *Semaphore) void {
    sem.post();
}

pub fn compressMemory(data: []const u8, allocator: Allocator) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    defer out.deinit();
    var i: usize = 0;
    while (i < data.len) {
        const value = data[i];
        var run: usize = 1;
        while (i + run < data.len and data[i + run] == value and run < 255) : (run += 1) {}
        try out.append(@intCast(run));
        try out.append(value);
        i += run;
    }
    return try out.toOwnedSlice(allocator);
}

pub fn decompressMemory(data: []const u8, allocator: Allocator) ![]u8 {
    if (data.len % 2 != 0) return error.InvalidData;
    var out = std.ArrayList(u8).init(allocator);
    defer out.deinit();
    var i: usize = 0;
    while (i < data.len) : (i += 2) {
        const run = data[i];
        const value = data[i + 1];
        try out.appendNTimes(allocator, value, run);
    }
    return try out.toOwnedSlice(allocator);
}

const AEAD = std.crypto.aead.chacha_poly.ChaCha20Poly1305;

pub const EncryptedBlob = struct {
    nonce: [AEAD.nonce_length]u8,
    tag: [AEAD.tag_length]u8,
    ciphertext: []u8,
    allocator: Allocator,

    pub fn deinit(self: *EncryptedBlob) void {
        secureZeroMemory(@as([*]u8, @ptrCast(&self.nonce)), self.nonce.len);
        secureZeroMemory(@as([*]u8, @ptrCast(&self.tag)), self.tag.len);
        if (self.ciphertext.len != 0) {
            secureZeroMemory(self.ciphertext.ptr, self.ciphertext.len);
            self.allocator.free(self.ciphertext);
        }
        self.ciphertext = emptyU8Slice();
    }
};

pub fn encryptMemory(allocator: Allocator, plaintext: []const u8, key: [AEAD.key_length]u8) !EncryptedBlob {
    var nonce: [AEAD.nonce_length]u8 = undefined;
    std.crypto.random.bytes(&nonce);
    const ciphertext = if (plaintext.len == 0) emptyU8Slice() else try allocator.alloc(u8, plaintext.len);
    var tag: [AEAD.tag_length]u8 = undefined;
    AEAD.encrypt(ciphertext, &tag, plaintext, &[_]u8{}, nonce, key);
    return .{ .nonce = nonce, .tag = tag, .ciphertext = ciphertext, .allocator = allocator };
}

pub fn decryptMemory(allocator: Allocator, blob: EncryptedBlob, key: [AEAD.key_length]u8) ![]u8 {
    const out = if (blob.ciphertext.len == 0) emptyU8Slice() else try allocator.alloc(u8, blob.ciphertext.len);
    errdefer if (out.len != 0) allocator.free(out);
    AEAD.decrypt(out, blob.ciphertext, blob.tag, &[_]u8{}, blob.nonce, key) catch return error.AuthenticationFailed;
    return out;
}

pub const CompressedStorage = struct {
    compressed: []u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, data: []const u8) !CompressedStorage {
        return .{ .compressed = try compressMemory(data, allocator), .allocator = allocator };
    }

    pub fn deinit(self: *CompressedStorage) void {
        if (self.compressed.len != 0) {
            secureZeroMemory(self.compressed.ptr, self.compressed.len);
            self.allocator.free(self.compressed);
        }
        self.compressed = emptyU8Slice();
    }

    pub fn decompress(self: *const CompressedStorage) ![]u8 {
        return try decompressMemory(self.compressed, self.allocator);
    }
};

pub const EncryptedStorage = struct {
    encrypted: EncryptedBlob,
    key: [AEAD.key_length]u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, data: []const u8, key: [AEAD.key_length]u8) !EncryptedStorage {
        return .{ .encrypted = try encryptMemory(allocator, data, key), .key = key, .allocator = allocator };
    }

    pub fn deinit(self: *EncryptedStorage) void {
        self.encrypted.deinit();
        secureZeroMemory(@as([*]u8, @ptrCast(&self.key)), self.key.len);
    }

    pub fn decrypt(self: *const EncryptedStorage) ![]u8 {
        return try decryptMemory(self.allocator, self.encrypted, self.key);
    }
};

pub fn memoryAlign(ptr: *anyopaque, alignment: usize) !*anyopaque {
    if (!isPow2(alignment)) return error.InvalidAlignment;
    return @ptrFromInt(mem.alignForward(usize, @intFromPtr(ptr), alignment));
}

pub fn isMemoryOverlap(a_start: *const anyopaque, a_size: usize, b_start: *const anyopaque, b_size: usize) !bool {
    const a_addr = @intFromPtr(a_start);
    const b_addr = @intFromPtr(b_start);
    const a_end = try addChecked(a_addr, a_size);
    const b_end = try addChecked(b_addr, b_size);
    return a_addr < b_end and b_addr < a_end;
}

pub fn copyNonOverlapping(dest: []u8, src: []const u8) !void {
    if (dest.len != src.len) return error.SizeMismatch;
    if (try isMemoryOverlap(dest.ptr, dest.len, src.ptr, src.len)) return error.Overlap;
    @memcpy(dest, src);
}

pub fn moveMemory(dest: []u8, src: []const u8) !void {
    if (dest.len != src.len) return error.SizeMismatch;
    if (@intFromPtr(dest.ptr) <= @intFromPtr(src.ptr)) {
        mem.copyForwards(u8, dest, src);
    } else {
        mem.copyBackwards(u8, dest, src);
    }
}

pub const MemoryPool = PoolAllocator;
pub const MemoryArena = Arena;
pub const MemorySlab = SlabAllocator;
pub const MemoryBuddy = BuddyAllocator;
pub const MemoryLockFreeQueue = LockFreeQueue;
pub const MemoryLockFreeStack = LockFreeStack;

const testing = std.testing;

test "Arena allocation" {
    var arena = try Arena.init(testing.allocator, 1024);
    defer arena.deinit();
    const ptr1 = arena.alloc(128, 8) orelse return error.OutOfMemory;
    const ptr2 = arena.alloc(64, 4) orelse return error.OutOfMemory;
    try testing.expectEqual(@as(usize, 128), ptr1.len);
    try testing.expectEqual(@as(usize, 64), ptr2.len);
}

test "SlabAllocator" {
    var slab = try SlabAllocator.init(testing.allocator, 256, 4, 64);
    defer slab.deinit();
    const ptr1 = slab.alloc(100) orelse return error.OutOfMemory;
    const ptr2 = slab.alloc(150) orelse return error.OutOfMemory;
    try testing.expectEqual(@as(usize, 100), ptr1.len);
    try testing.expectEqual(@as(usize, 150), ptr2.len);
    try slab.free(ptr1);
    try slab.free(ptr2);
}

test "PoolAllocator" {
    var pool = try PoolAllocator.init(testing.allocator, 64, 16, 2);
    defer pool.deinit();
    const ptr1 = pool.alloc(64) orelse return error.OutOfMemory;
    const ptr2 = pool.alloc(64) orelse return error.OutOfMemory;
    try testing.expectEqual(@as(usize, 64), ptr1.len);
    try testing.expectEqual(@as(usize, 64), ptr2.len);
    try pool.free(ptr1);
    try pool.free(ptr2);
}

test "PageAllocator" {
    var page_alloc = try PageAllocator.init(testing.allocator, 4);
    defer page_alloc.deinit();
    const pages = page_alloc.allocPages(2) orelse return error.OutOfMemory;
    try testing.expectEqual(@as(usize, 2 * PageSize), pages.len);
    try page_alloc.freePages(pages);
}

test "ZeroCopySlice" {
    const data = "hello world";
    const zcs = ZeroCopySlice.init(@as([*]const u8, @ptrCast(data.ptr)), data.len);
    const slice = zcs.slice(0, 5);
    try testing.expectEqualStrings("hello", slice.asBytes());
}

test "ResizeBuffer" {
    var buf = ResizeBuffer.init(testing.allocator);
    defer buf.deinit();
    try buf.append("hello");
    try buf.append(" world");
    const owned = try buf.toOwnedSlice();
    defer if (owned.len != 0) testing.allocator.free(owned);
    try testing.expectEqualStrings("hello world", owned);
}

test "ArenaAllocator basic allocation" {
    var arena = ArenaAllocator.init(testing.allocator, 1024);
    defer arena.deinit();
    const alloc = arena.allocator();
    const slice1 = try alloc.alloc(u8, 100);
    const slice2 = try alloc.alloc(u8, 100);
    @memset(slice1, 42);
    @memset(slice2, 84);
    try testing.expectEqual(@as(u8, 42), slice1[0]);
    try testing.expectEqual(@as(u8, 84), slice2[0]);
}

test "zero copy transfer" {
    var src = [_]u8{ 1, 2, 3, 4, 5 };
    var dest: [5]u8 = undefined;
    zeroCopyTransfer(&src, &dest);
    try testing.expectEqualSlices(u8, &src, &dest);
}

test "memory hashing" {
    const data1 = "hello world";
    const data2 = "hello world";
    const data3 = "hello world!";
    const hash1 = hashMemory(data1);
    const hash2 = hashMemory(data2);
    const hash3 = hashMemory(data3);
    try testing.expectEqual(hash1, hash2);
    try testing.expect(hash1 != hash3);
}

test "memory comparison constant time" {
    const data1 = "test";
    const data2 = "test";
    const data3 = "best";
    try testing.expect(compareMemory(data1, data2));
    try testing.expect(!compareMemory(data1, data3));
}

test "search memory" {
    const haystack = "hello world, hello universe";
    const needle = "world";
    const pos = searchMemory(haystack, needle);
    try testing.expect(pos != null);
    try testing.expectEqual(@as(usize, 6), pos.?);
}

test "count memory" {
    const data = "hello world";
    const count = countMemory(data, 'l');
    try testing.expectEqual(@as(usize, 3), count);
}

test "unique memory" {
    const data = "aabbccddaa";
    const uniq = try uniqueMemory(testing.allocator, data);
    defer if (uniq.len != 0) testing.allocator.free(uniq);
    try testing.expectEqual(@as(usize, 4), uniq.len);
}

test "atomic operations" {
    var value: u64 = 0;
    const prev = atomicAdd(&value, 5);
    try testing.expectEqual(@as(u64, 0), prev);
    try testing.expectEqual(@as(u64, 5), atomicLoad(&value));
    atomicStore(&value, 10);
    try testing.expectEqual(@as(u64, 10), atomicLoad(&value));
    _ = atomicInc(&value);
    try testing.expectEqual(@as(u64, 11), atomicLoad(&value));
}

test "ReadWriteLock" {
    var rwlock = ReadWriteLock.init();
    rwlock.readLock();
    rwlock.readUnlock();
    rwlock.writeLock();
    rwlock.writeUnlock();
}

test "BuddyAllocator" {
    var buddy = try BuddyAllocator.init(testing.allocator, 4096, 6);
    defer buddy.deinit();
    const ptr1 = try buddy.alloc(128);
    try testing.expectEqual(@as(usize, 128), ptr1.len);
    try buddy.free(ptr1);
}

test "LockFreeQueue" {
    var queue = try LockFreeQueue.init(testing.allocator, 16);
    defer queue.deinit();
    var item: usize = 42;
    try testing.expect(queue.enqueue(@as(*anyopaque, @ptrCast(&item))));
    const retrieved = queue.dequeue();
    try testing.expect(retrieved != null);
    try testing.expectEqual(@intFromPtr(@as(*anyopaque, @ptrCast(&item))), @intFromPtr(retrieved.?));
}

test "LockFreeStack" {
    var stack = LockFreeStack.init(testing.allocator);
    defer stack.deinit();
    var item: usize = 42;
    try stack.push(@as(*anyopaque, @ptrCast(&item)));
    const retrieved = stack.pop();
    try testing.expect(retrieved != null);
    try testing.expectEqual(@intFromPtr(@as(*anyopaque, @ptrCast(&item))), @intFromPtr(retrieved.?));
}

test "memory stats tracking" {
    resetMemoryStats();
    trackAllocation(100);
    trackAllocation(200);
    trackFree(50);
    const stats = getMemoryStats();
    try testing.expectEqual(@as(usize, 300), stats.allocated);
    try testing.expectEqual(@as(usize, 50), stats.freed);
    try testing.expectEqual(@as(usize, 250), memoryFootprint());
}
