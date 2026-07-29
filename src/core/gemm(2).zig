const std = @import("std");
const builtin = @import("builtin");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const types = @import("types.zig");
const Error = types.Error;
const Fixed32_32 = types.Fixed32_32;
const memory = @import("memory.zig");

const alignment = 32;
const vector_width = 8;
const Vec8 = @Vector(vector_width, f32);

pub const NC: usize = 4096;
pub const KC: usize = 256;
pub const MC: usize = 256;
pub const NR: usize = 8;
pub const MR: usize = 8;
pub const huge_page_size: usize = 2 * 1024 * 1024;
pub const hugePageSetupCommand = "echo 2000 > /proc/sys/vm/nr_hugepages";
pub const buildCommand = "zig build-exe -OReleaseFast -mcpu=native -fno-strip -femit-bin=gemm_bench src/gemm.zig";

const max_worker_cores: usize = 1024;
const map_private: u32 = 0x00000002;
const map_anonymous: u32 = 0x00000020;
const map_hugetlb: u32 = 0x00040000;
const map_huge_2mb: u32 = 21 << 26;

var global_huge_attempts: usize = 0;
var global_huge_successes: usize = 0;
var global_huge_fallbacks: usize = 0;

pub const HugePageStats = struct {
    attempts: usize,
    successes: usize,
    fallbacks: usize,
};

pub fn hugePageStats() HugePageStats {
    return .{
        .attempts = @atomicLoad(usize, &global_huge_attempts, .acquire),
        .successes = @atomicLoad(usize, &global_huge_successes, .acquire),
        .fallbacks = @atomicLoad(usize, &global_huge_fallbacks, .acquire),
    };
}

fn roundUpToHugePage(len: usize) !usize {
    const sum = @addWithOverflow(len, huge_page_size - 1);
    if (sum[1] != 0) return Error.Overflow;
    return sum[0] & ~(huge_page_size - 1);
}

const HugeMapping = struct {
    address: usize,
    mapped_len: usize,
    next: ?*HugeMapping,
};

pub const HugePageAllocator = struct {
    parent: Allocator,
    mutex: std.Thread.Mutex = .{},
    parent_mutex: std.Thread.Mutex = .{},
    mappings: ?*HugeMapping = null,

    pub fn init(parent: ?Allocator) HugePageAllocator {
        return .{ .parent = parent orelse std.heap.page_allocator };
    }

    pub fn create(parent: ?Allocator) !*HugePageAllocator {
        const actual_parent = parent orelse std.heap.page_allocator;
        const self = try actual_parent.create(HugePageAllocator);
        self.* = HugePageAllocator.init(actual_parent);
        return self;
    }

    pub fn allocator(self: *HugePageAllocator) Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    pub fn deinit(self: *HugePageAllocator) void {
        while (true) {
            self.mutex.lock();
            const mapping = self.mappings orelse {
                self.mutex.unlock();
                break;
            };
            self.mappings = mapping.next;
            self.mutex.unlock();
            const ptr: [*]align(mem.page_size) const u8 = @ptrFromInt(mapping.address);
            std.posix.munmap(ptr[0..mapping.mapped_len]);
            self.parent_mutex.lock();
            self.parent.destroy(mapping);
            self.parent_mutex.unlock();
        }
    }

    pub fn isHugePointer(self: *HugePageAllocator, pointer: *const anyopaque) bool {
        const address = @intFromPtr(pointer);
        self.mutex.lock();
        defer self.mutex.unlock();
        var current = self.mappings;
        while (current) |mapping| : (current = mapping.next) {
            if (mapping.address == address) return true;
        }
        return false;
    }

    fn createMapping(self: *HugePageAllocator) !*HugeMapping {
        self.parent_mutex.lock();
        defer self.parent_mutex.unlock();
        return self.parent.create(HugeMapping);
    }

    fn destroyMapping(self: *HugePageAllocator, mapping: *HugeMapping) void {
        self.parent_mutex.lock();
        self.parent.destroy(mapping);
        self.parent_mutex.unlock();
    }

    fn registerMapping(self: *HugePageAllocator, mapping: *HugeMapping, address: usize, mapped_len: usize) void {
        self.mutex.lock();
        mapping.* = .{
            .address = address,
            .mapped_len = mapped_len,
            .next = self.mappings,
        };
        self.mappings = mapping;
        self.mutex.unlock();
    }

    fn takeMapping(self: *HugePageAllocator, address: usize) ?HugeMapping {
        self.mutex.lock();
        var link = &self.mappings;
        while (link.*) |mapping| {
            if (mapping.address == address) {
                link.* = mapping.next;
                const value = mapping.*;
                self.mutex.unlock();
                self.destroyMapping(mapping);
                return value;
            }
            link = &mapping.next;
        }
        self.mutex.unlock();
        return null;
    }

    fn hasMapping(self: *HugePageAllocator, address: usize) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        var current = self.mappings;
        while (current) |mapping| : (current = mapping.next) {
            if (mapping.address == address) return true;
        }
        return false;
    }

    fn mapHuge(self: *HugePageAllocator, mapped_len: usize) ![*]u8 {
        if (comptime builtin.os.tag != .linux) return error.UnsupportedOperatingSystem;
        const mapping = self.createMapping() catch return error.HugePageMetadataOutOfMemory;
        errdefer self.destroyMapping(mapping);
        _ = @atomicRmw(usize, &global_huge_attempts, .Add, 1, .monotonic);
        const flags: std.posix.MAP = @bitCast(map_private | map_anonymous | map_hugetlb | map_huge_2mb);
        const mapped = try std.posix.mmap(null, mapped_len, std.posix.PROT.READ | std.posix.PROT.WRITE, flags, -1, 0);
        if (!mem.isAligned(@intFromPtr(mapped.ptr), huge_page_size)) {
            std.posix.munmap(mapped);
            return error.HugePageAlignmentFailure;
        }
        self.registerMapping(mapping, @intFromPtr(mapped.ptr), mapped.len);
        _ = @atomicRmw(usize, &global_huge_successes, .Add, 1, .monotonic);
        return mapped.ptr;
    }

    fn parentAlloc(self: *HugePageAllocator, len: usize, log2_alignment: u8, ret_addr: usize) ?[*]u8 {
        self.parent_mutex.lock();
        defer self.parent_mutex.unlock();
        return self.parent.rawAlloc(len, log2_alignment, ret_addr);
    }

    fn parentResize(self: *HugePageAllocator, buffer: []u8, log2_alignment: u8, new_len: usize, ret_addr: usize) bool {
        self.parent_mutex.lock();
        defer self.parent_mutex.unlock();
        return self.parent.rawResize(buffer, log2_alignment, new_len, ret_addr);
    }

    fn parentFree(self: *HugePageAllocator, buffer: []u8, log2_alignment: u8, ret_addr: usize) void {
        self.parent_mutex.lock();
        self.parent.rawFree(buffer, log2_alignment, ret_addr);
        self.parent_mutex.unlock();
    }

    fn allocFn(context: *anyopaque, len: usize, log2_alignment: u8, ret_addr: usize) ?[*]u8 {
        const self: *HugePageAllocator = @ptrCast(@alignCast(context));
        if (len < huge_page_size) return self.parentAlloc(len, log2_alignment, ret_addr);
        if (len % huge_page_size != 0 or log2_alignment > 21) return null;
        return self.mapHuge(len) catch |err| switch (err) {
            error.OutOfMemory => blk: {
                _ = @atomicRmw(usize, &global_huge_fallbacks, .Add, 1, .monotonic);
                break :blk self.parentAlloc(len, log2_alignment, ret_addr);
            },
            else => null,
        };
    }

    fn resizeFn(context: *anyopaque, buffer: []u8, log2_alignment: u8, new_len: usize, ret_addr: usize) bool {
        const self: *HugePageAllocator = @ptrCast(@alignCast(context));
        if (self.hasMapping(@intFromPtr(buffer.ptr))) return new_len == buffer.len;
        if (new_len >= huge_page_size) return false;
        return self.parentResize(buffer, log2_alignment, new_len, ret_addr);
    }

    fn freeFn(context: *anyopaque, buffer: []u8, log2_alignment: u8, ret_addr: usize) void {
        const self: *HugePageAllocator = @ptrCast(@alignCast(context));
        if (self.takeMapping(@intFromPtr(buffer.ptr))) |mapping| {
            const ptr: [*]align(mem.page_size) const u8 = @ptrCast(@alignCast(buffer.ptr));
            std.posix.munmap(ptr[0..mapping.mapped_len]);
            return;
        }
        self.parentFree(buffer, log2_alignment, ret_addr);
    }

    const vtable = Allocator.VTable{
        .alloc = allocFn,
        .resize = resizeFn,
        .free = freeFn,
    };
};

fn parseCpuList(text: []const u8, output: []usize) usize {
    var count: usize = 0;
    var groups = mem.splitScalar(u8, text, ',');
    while (groups.next()) |raw_group| {
        const group = mem.trim(u8, raw_group, " \n\r\t");
        if (group.len == 0) continue;
        if (mem.indexOfScalar(u8, group, '-')) |separator| {
            const first = std.fmt.parseInt(usize, group[0..separator], 10) catch continue;
            const last = std.fmt.parseInt(usize, group[separator + 1 ..], 10) catch continue;
            if (last < first) continue;
            var cpu = first;
            while (cpu <= last and count < output.len) : (cpu += 1) {
                output[count] = cpu;
                count += 1;
            }
        } else if (count < output.len) {
            output[count] = std.fmt.parseInt(usize, group, 10) catch continue;
            count += 1;
        }
    }
    return count;
}

fn loadAllowedCpuIds(output: []usize) usize {
    if (builtin.os.tag == .linux) {
        var buffer: [8192]u8 = undefined;
        if (readSmallFile("/sys/fs/cgroup/cpuset.cpus.effective", &buffer)) |text| {
            const count = parseCpuList(text, output);
            if (count != 0) return count;
        }
        if (readSmallFile("/sys/fs/cgroup/cpuset/cpuset.cpus", &buffer)) |text| {
            const count = parseCpuList(text, output);
            if (count != 0) return count;
        }
    }
    const host_count = std.Thread.getCpuCount() catch 1;
    const count = @min(host_count, output.len);
    for (output[0..count], 0..) |*slot, index| slot.* = index;
    return count;
}

fn readTopologyValue(cpu_id: usize, name: []const u8) ?usize {
    var path_buffer: [160]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buffer, "/sys/devices/system/cpu/cpu{d}/topology/{s}", .{ cpu_id, name }) catch return null;
    var value_buffer: [64]u8 = undefined;
    const value = readSmallFile(path, &value_buffer) orelse return null;
    return std.fmt.parseInt(usize, value, 10) catch null;
}

fn fillEffectiveCoreIds(output: []usize) usize {
    var allowed: [max_worker_cores]usize = undefined;
    const allowed_count = loadAllowedCpuIds(&allowed);
    var packages: [max_worker_cores]usize = undefined;
    var cores: [max_worker_cores]usize = undefined;
    var physical_count: usize = 0;
    for (allowed[0..allowed_count]) |cpu_id| {
        const package_id = readTopologyValue(cpu_id, "physical_package_id") orelse 0;
        const core_id = readTopologyValue(cpu_id, "core_id") orelse cpu_id;
        var duplicate = false;
        var index: usize = 0;
        while (index < physical_count) : (index += 1) {
            if (packages[index] == package_id and cores[index] == core_id) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate and physical_count < output.len) {
            packages[physical_count] = package_id;
            cores[physical_count] = core_id;
            output[physical_count] = cpu_id;
            physical_count += 1;
        }
    }
    if (physical_count == 0) {
        const fallback_count = @min(allowed_count, output.len);
        if (fallback_count != 0) {
            @memcpy(output[0..fallback_count], allowed[0..fallback_count]);
            physical_count = fallback_count;
        } else {
            output[0] = 0;
            physical_count = 1;
        }
    }
    var quota_limit = physical_count;
    if (cgroupV2CpuCount()) |count| quota_limit = @min(quota_limit, count);
    if (cgroupV1CpuCount()) |count| quota_limit = @min(quota_limit, count);
    return @max(@min(quota_limit, output.len), 1);
}

pub fn pinThreadToCore(core_id: usize) !void {
    if (comptime builtin.os.tag != .linux) return error.UnsupportedOperatingSystem;
    var cpu_set = [_]u64{0} ** 16;
    const word_index = core_id / 64;
    if (word_index >= cpu_set.len) return error.InvalidCoreId;
    cpu_set[word_index] = @as(u64, 1) << @intCast(core_id % 64);
    const result = std.os.linux.syscall3(.sched_setaffinity, 0, @sizeOf(@TypeOf(cpu_set)), @intFromPtr(&cpu_set));
    const signed_result: isize = @bitCast(result);
    if (signed_result < 0 and signed_result >= -4095) return error.ThreadPinFailed;
}

pub const GemmReport = struct {
    cores_used: usize = 0,
    huge_attempts: usize = 0,
    huge_successes: usize = 0,
    huge_fallbacks: usize = 0,
    core_ids: [max_worker_cores]usize = [_]usize{0} ** max_worker_cores,
    pinned: [max_worker_cores]bool = [_]bool{false} ** max_worker_cores,
};

var report_mutex: std.Thread.Mutex = .{};
var last_gemm_report: GemmReport = .{};

pub fn getLastGemmReport() GemmReport {
    report_mutex.lock();
    defer report_mutex.unlock();
    return last_gemm_report;
}

fn storeGemmReport(report: GemmReport) void {
    report_mutex.lock();
    defer report_mutex.unlock();
    last_gemm_report = report;
}

pub fn packA(a: []const f32, lda: usize, mc: usize, kc: usize, packed: []align(32) f32) void {
    @setRuntimeSafety(false);
    var row: usize = 0;
    while (row < mc) : (row += 1) {
        const source_base = row * lda;
        const destination_base = row * kc;
        var depth: usize = 0;
        const vector_limit = kc - kc % vector_width;
        while (depth < vector_limit) : (depth += vector_width) {
            const values: Vec8 = a[source_base + depth ..][0..vector_width].*;
            packed[destination_base + depth ..][0..vector_width].* = values;
        }
        while (depth < kc) : (depth += 1) packed[destination_base + depth] = a[source_base + depth];
    }
}

pub fn packB(b: []const f32, ldb: usize, kc: usize, nc: usize, packed: []align(32) f32) void {
    @setRuntimeSafety(false);
    const padded_nc = mem.alignForward(usize, nc, NR);
    var column_block: usize = 0;
    while (column_block < padded_nc) : (column_block += NR) {
        var depth: usize = 0;
        while (depth < kc) : (depth += 1) {
            const destination = column_block * kc + depth * NR;
            if (column_block + NR <= nc) {
                const values: Vec8 = b[depth * ldb + column_block ..][0..NR].*;
                packed[destination..][0..NR].* = values;
            } else {
                var column: usize = 0;
                while (column < NR) : (column += 1) {
                    const source_column = column_block + column;
                    packed[destination + column] = if (source_column < nc) b[depth * ldb + source_column] else 0.0;
                }
            }
        }
    }
}

noinline fn microKernel(
    _: [*]align(32) const f32,
    _: [*]align(32) const f32,
    _: [*]align(32) f32,
    _: usize,
) callconv(.Naked) void {
    asm volatile (
        \\.intel_syntax noprefix
        \\vmovaps ymm0, ymmword ptr [rdx]
        \\vmovaps ymm1, ymmword ptr [rdx + 32]
        \\vmovaps ymm2, ymmword ptr [rdx + 64]
        \\vmovaps ymm3, ymmword ptr [rdx + 96]
        \\vmovaps ymm4, ymmword ptr [rdx + 128]
        \\vmovaps ymm5, ymmword ptr [rdx + 160]
        \\vmovaps ymm6, ymmword ptr [rdx + 192]
        \\vmovaps ymm7, ymmword ptr [rdx + 224]
        \\lea r10, [rcx * 4]
        \\mov rax, rcx
        \\shr rax, 2
        \\test rax, rax
        \\jz .Lgemm_tail
        \\.Lgemm_k4:
        \\vmovaps ymm9, ymmword ptr [rsi]
        \\mov r11, rdi
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm0, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm1, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm2, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm3, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm4, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm5, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm6, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm7, ymm8, ymm9
        \\add rdi, 4
        \\add rsi, 32
        \\vmovaps ymm9, ymmword ptr [rsi]
        \\mov r11, rdi
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm0, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm1, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm2, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm3, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm4, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm5, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm6, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm7, ymm8, ymm9
        \\add rdi, 4
        \\add rsi, 32
        \\vmovaps ymm9, ymmword ptr [rsi]
        \\mov r11, rdi
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm0, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm1, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm2, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm3, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm4, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm5, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm6, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm7, ymm8, ymm9
        \\add rdi, 4
        \\add rsi, 32
        \\vmovaps ymm9, ymmword ptr [rsi]
        \\mov r11, rdi
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm0, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm1, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm2, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm3, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm4, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm5, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm6, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm7, ymm8, ymm9
        \\add rdi, 4
        \\add rsi, 32
        \\dec rax
        \\jnz .Lgemm_k4
        \\.Lgemm_tail:
        \\and rcx, 3
        \\jz .Lgemm_store
        \\.Lgemm_k1:
        \\vmovaps ymm9, ymmword ptr [rsi]
        \\mov r11, rdi
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm0, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm1, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm2, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm3, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm4, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm5, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm6, ymm8, ymm9
        \\add r11, r10
        \\vbroadcastss ymm8, dword ptr [r11]
        \\vfmadd231ps ymm7, ymm8, ymm9
        \\add rdi, 4
        \\add rsi, 32
        \\dec rcx
        \\jnz .Lgemm_k1
        \\.Lgemm_store:
        \\vmovaps ymmword ptr [rdx], ymm0
        \\vmovaps ymmword ptr [rdx + 32], ymm1
        \\vmovaps ymmword ptr [rdx + 64], ymm2
        \\vmovaps ymmword ptr [rdx + 96], ymm3
        \\vmovaps ymmword ptr [rdx + 128], ymm4
        \\vmovaps ymmword ptr [rdx + 160], ymm5
        \\vmovaps ymmword ptr [rdx + 192], ymm6
        \\vmovaps ymmword ptr [rdx + 224], ymm7
        \\vzeroupper
        \\ret
        \\.att_syntax prefix
    );
}

const GemmContext = struct {
    a: []const f32,
    b: []const f32,
    c: []align(32) f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    worker_count: usize,
    core_ids: []const usize,
    pinned: []bool,
    workspace_allocator: Allocator,
    failure: u8 = 0,
};

fn setWorkerFailure(context: *GemmContext, code: u8) void {
    _ = @cmpxchgStrong(u8, &context.failure, 0, code, .acq_rel, .acquire);
}

fn microKernelEdge(
    packed_a: []const f32,
    packed_b: []const f32,
    c: []f32,
    ldc: usize,
    global_row: usize,
    global_column: usize,
    local_row: usize,
    mr: usize,
    nr: usize,
    kc: usize,
) void {
    @setRuntimeSafety(false);
    var row: usize = 0;
    while (row < mr) : (row += 1) {
        var column: usize = 0;
        while (column < nr) : (column += 1) {
            const c_index = (global_row + row) * ldc + global_column + column;
            var accumulator = c[c_index];
            var depth: usize = 0;
            while (depth < kc) : (depth += 1) {
                accumulator += packed_a[(local_row + row) * kc + depth] * packed_b[depth * NR + column];
            }
            c[c_index] = accumulator;
        }
    }
}

fn workerMain(context: *GemmContext, worker_id: usize) void {
    pinThreadToCore(context.core_ids[worker_id]) catch {
        context.pinned[worker_id] = false;
        setWorkerFailure(context, 1);
        return;
    };
    context.pinned[worker_id] = true;
    const total_ic_blocks = (context.m + MC - 1) / MC;
    const first_block = total_ic_blocks * worker_id / context.worker_count;
    const last_block = total_ic_blocks * (worker_id + 1) / context.worker_count;
    if (first_block == last_block) return;
    const packed_a_storage = context.workspace_allocator.alignedAlloc(f32, @as(?u29, alignment), huge_page_size / @sizeOf(f32)) catch {
        setWorkerFailure(context, 2);
        return;
    };
    defer context.workspace_allocator.free(packed_a_storage);
    const packed_b_storage = context.workspace_allocator.alignedAlloc(f32, @as(?u29, alignment), KC * NC) catch {
        setWorkerFailure(context, 2);
        return;
    };
    defer context.workspace_allocator.free(packed_b_storage);
    @memset(packed_a_storage, 0.0);
    @memset(packed_b_storage, 0.0);
    const first_row = first_block * MC;
    const last_row = @min(last_block * MC, context.m);
    var row = first_row;
    while (row < last_row) : (row += 1) {
        @memset(context.c[row * context.ldc ..][0..context.n], 0.0);
    }
    var jc: usize = 0;
    while (jc < context.n) : (jc += NC) {
        const nc = @min(NC, context.n - jc);
        const padded_nc = mem.alignForward(usize, nc, NR);
        var pc: usize = 0;
        while (pc < context.k) : (pc += KC) {
            if (@atomicLoad(u8, &context.failure, .acquire) != 0) return;
            const kc = @min(KC, context.k - pc);
            const packed_b_len = padded_nc * kc;
            packB(context.b[pc * context.ldb + jc ..], context.ldb, kc, nc, packed_b_storage[0..packed_b_len]);
            var block_index = first_block;
            while (block_index < last_block) : (block_index += 1) {
                const ic = block_index * MC;
                if (ic >= context.m) break;
                const mc = @min(MC, context.m - ic);
                const packed_a_len = mc * kc;
                packA(context.a[ic * context.lda + pc ..], context.lda, mc, kc, packed_a_storage[0..packed_a_len]);
                var jr: usize = 0;
                while (jr < nc) : (jr += NR) {
                    const nr = @min(NR, nc - jr);
                    const b_offset = jr * kc;
                    const b_pointer: [*]align(32) const f32 = @ptrCast(@alignCast(packed_b_storage.ptr + b_offset));
                    var ir: usize = 0;
                    while (ir < mc) : (ir += MR) {
                        const mr = @min(MR, mc - ir);
                        if (mr == MR and nr == NR) {
                            var tile: [MR * NR]f32 align(32) = undefined;
                            var tile_row: usize = 0;
                            while (tile_row < MR) : (tile_row += 1) {
                                const c_offset = (ic + ir + tile_row) * context.ldc + jc + jr;
                                const values: Vec8 = context.c[c_offset..][0..NR].*;
                                tile[tile_row * NR ..][0..NR].* = values;
                            }
                            const a_pointer: [*]align(32) const f32 = @ptrCast(@alignCast(packed_a_storage.ptr + ir * kc));
                            const tile_pointer: [*]align(32) f32 = @ptrCast(&tile);
                            microKernel(a_pointer, b_pointer, tile_pointer, kc);
                            tile_row = 0;
                            while (tile_row < MR) : (tile_row += 1) {
                                const c_offset = (ic + ir + tile_row) * context.ldc + jc + jr;
                                const values: Vec8 = tile[tile_row * NR ..][0..NR].*;
                                context.c[c_offset..][0..NR].* = values;
                            }
                        } else {
                            microKernelEdge(
                                packed_a_storage[0..packed_a_len],
                                packed_b_storage[b_offset .. b_offset + kc * NR],
                                context.c,
                                context.ldc,
                                ic + ir,
                                jc + jr,
                                ir,
                                mr,
                                nr,
                                kc,
                            );
                        }
                    }
                }
            }
        }
    }
}

fn runHighPerformanceGemm(
    a: []const f32,
    b: []const f32,
    c: []align(32) f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    workspace_allocator: Allocator,
    parent_allocator: Allocator,
    stats_before: HugePageStats,
) !void {
    var core_ids_storage: [max_worker_cores]usize = undefined;
    const worker_count = fillEffectiveCoreIds(&core_ids_storage);
    const core_ids = try parent_allocator.dupe(usize, core_ids_storage[0..worker_count]);
    defer parent_allocator.free(core_ids);
    const pinned = try parent_allocator.alloc(bool, worker_count);
    defer parent_allocator.free(pinned);
    @memset(pinned, false);
    const threads = try parent_allocator.alloc(std.Thread, worker_count);
    defer parent_allocator.free(threads);
    var context = GemmContext{
        .a = a,
        .b = b,
        .c = c,
        .m = m,
        .n = n,
        .k = k,
        .lda = lda,
        .ldb = ldb,
        .ldc = ldc,
        .worker_count = worker_count,
        .core_ids = core_ids,
        .pinned = pinned,
        .workspace_allocator = workspace_allocator,
    };
    var started: usize = 0;
    errdefer {
        for (threads[0..started]) |thread| thread.join();
    }
    while (started < worker_count) : (started += 1) {
        threads[started] = try std.Thread.spawn(.{}, workerMain, .{ &context, started });
    }
    for (threads) |thread| thread.join();
    const stats_after = hugePageStats();
    var report = GemmReport{
        .cores_used = worker_count,
        .huge_attempts = stats_after.attempts - stats_before.attempts,
        .huge_successes = stats_after.successes - stats_before.successes,
        .huge_fallbacks = stats_after.fallbacks - stats_before.fallbacks,
    };
    for (0..worker_count) |index| {
        report.core_ids[index] = core_ids[index];
        report.pinned[index] = pinned[index];
    }
    storeGemmReport(report);
    switch (@atomicLoad(u8, &context.failure, .acquire)) {
        0 => return,
        1 => return error.ThreadPinFailed,
        2 => return error.OutOfMemory,
        else => return error.GemmWorkerFailure,
    }
}

fn scalarBlockedMatmul(a: *const Tensor, b: *const Tensor, allocator: Allocator) !Tensor {
    const m = a.shape.dims[0];
    const k = a.shape.dims[1];
    const n = b.shape.dims[1];
    var result = try Tensor.init(allocator, &.{ m, n });
    errdefer result.deinit();
    const block: usize = 32;
    var ii: usize = 0;
    while (ii < m) : (ii += block) {
        const i_end = @min(ii + block, m);
        var kk: usize = 0;
        while (kk < k) : (kk += block) {
            const k_end = @min(kk + block, k);
            var jj: usize = 0;
            while (jj < n) : (jj += block) {
                const j_end = @min(jj + block, n);
                var i = ii;
                while (i < i_end) : (i += 1) {
                    var depth = kk;
                    while (depth < k_end) : (depth += 1) {
                        const a_value = a.data[i * a.shape.strides[0] + depth * a.shape.strides[1]];
                        var j = jj;
                        while (j < j_end) : (j += 1) {
                            result.data[i * result.shape.strides[0] + j] += a_value * b.data[depth * b.shape.strides[0] + j * b.shape.strides[1]];
                        }
                    }
                }
            }
        }
    }
    return result;
}

fn highPerformanceAvailable() bool {
    if (builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) return false;
    return std.Target.x86.featureSetHas(builtin.cpu.features, .avx2) and std.Target.x86.featureSetHas(builtin.cpu.features, .fma);
}

fn highPerformanceMatmul(a: *const Tensor, b: *const Tensor, allocator: Allocator) !Tensor {
    const m = a.shape.dims[0];
    const k = a.shape.dims[1];
    const n = b.shape.dims[1];
    const stats_before = hugePageStats();
    var staged_a: Tensor = undefined;
    var has_staged_a = false;
    defer if (has_staged_a) staged_a.deinit();
    var staged_b: Tensor = undefined;
    var has_staged_b = false;
    defer if (has_staged_b) staged_b.deinit();
    const effective_a: *const Tensor = if (a.isHugeBacked()) a else blk: {
        staged_a = try a.copyHuge(allocator);
        has_staged_a = true;
        break :blk &staged_a;
    };
    const effective_b: *const Tensor = if (b.isHugeBacked()) b else blk: {
        staged_b = try b.copyHuge(allocator);
        has_staged_b = true;
        break :blk &staged_b;
    };
    var result = try Tensor.initHugeUninitialized(allocator, &.{ m, n });
    errdefer result.deinit();
    try runHighPerformanceGemm(
        effective_a.data,
        effective_b.data,
        result.data,
        m,
        n,
        k,
        effective_a.shape.strides[0],
        effective_b.shape.strides[0],
        result.shape.strides[0],
        result.allocator,
        allocator,
        stats_before,
    );
    return result;
}

fn readSmallFile(path: []const u8, buf: []u8) ?[]const u8 {
    const file = std.fs.openFileAbsolute(path, .{}) catch return null;
    defer file.close();
    const n = file.read(buf) catch return null;
    if (n == 0) return null;
    return mem.trim(u8, buf[0..n], " \n\t\r");
}

fn cgroupV2CpuCount() ?usize {
    var buf: [128]u8 = undefined;
    const content = readSmallFile("/sys/fs/cgroup/cpu.max", &buf) orelse return null;
    var it = mem.splitScalar(u8, content, ' ');
    const quota_str = it.next() orelse return null;
    const period_str = it.next() orelse return null;
    if (mem.eql(u8, quota_str, "max")) return null;
    const quota = std.fmt.parseInt(i64, quota_str, 10) catch return null;
    const period = std.fmt.parseInt(i64, period_str, 10) catch return null;
    if (quota <= 0 or period <= 0) return null;
    const cpus = @divTrunc(quota, period);
    if (cpus < 1) return 1;
    return @intCast(cpus);
}

fn cgroupV1CpuCount() ?usize {
    var qbuf: [64]u8 = undefined;
    const quota_str = readSmallFile("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", &qbuf) orelse return null;
    const quota = std.fmt.parseInt(i64, quota_str, 10) catch return null;
    if (quota <= 0) return null;
    var pbuf: [64]u8 = undefined;
    const period_str = readSmallFile("/sys/fs/cgroup/cpu/cpu.cfs_period_us", &pbuf) orelse return null;
    const period = std.fmt.parseInt(i64, period_str, 10) catch return null;
    if (period <= 0) return null;
    const cpus = @divTrunc(quota, period);
    if (cpus < 1) return 1;
    return @intCast(cpus);
}

pub fn cgroupSource() []const u8 {
    if (builtin.os.tag != .linux) return "fallback";
    if (cgroupV2CpuCount() != null) return "cgroup_v2";
    if (cgroupV1CpuCount() != null) return "cgroup_v1";
    return "fallback";
}

pub fn effectiveCpuCount() usize {
    var core_ids: [max_worker_cores]usize = undefined;
    return fillEffectiveCoreIds(&core_ids);
}

pub const TensorIterator = struct {
    shape: *const Shape,
    indices: [8]usize,
    offset: usize,
    done: bool,

    pub fn init(shape: *const Shape) TensorIterator {
        return .{
            .shape = shape,
            .indices = [_]usize{0} ** 8,
            .offset = 0,
            .done = false,
        };
    }

    pub fn advance(self: *TensorIterator) bool {
        if (self.done) return false;
        if (self.shape.dims.len == 0) {
            self.done = true;
            return false;
        }
        var axis: usize = self.shape.dims.len;
        while (axis > 0) {
            axis -= 1;
            self.indices[axis] += 1;
            self.offset += self.shape.strides[axis];
            if (self.indices[axis] < self.shape.dims[axis]) return true;
            self.offset -= self.shape.dims[axis] * self.shape.strides[axis];
            self.indices[axis] = 0;
        }
        self.done = true;
        return false;
    }
};

pub const Shape = struct {
    dims: []usize,
    strides: []usize,
    total_size: usize,
    freed: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    pub fn init(allocator: Allocator, dims_in: []const usize) !Shape {
        if (dims_in.len == 0 or dims_in.len > 8) return Error.InvalidShape;
        var total: usize = 1;
        for (dims_in) |dim| {
            if (dim == 0) return Error.InvalidShape;
            const result = @mulWithOverflow(total, dim);
            if (result[1] != 0) return Error.Overflow;
            total = result[0];
        }
        const dims = try allocator.alloc(usize, dims_in.len);
        errdefer allocator.free(dims);
        const strides = try allocator.alloc(usize, dims_in.len);
        errdefer allocator.free(strides);
        @memcpy(dims, dims_in);
        var stride: usize = 1;
        var i: usize = dims_in.len;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            const result = @mulWithOverflow(stride, dims[i]);
            if (result[1] != 0) return Error.Overflow;
            stride = result[0];
        }
        return .{ .dims = dims, .strides = strides, .total_size = total };
    }

    pub fn initWithStrides(allocator: Allocator, dims_in: []const usize, strides_in: []const usize) !Shape {
        if (dims_in.len == 0 or dims_in.len > 8 or dims_in.len != strides_in.len) return Error.InvalidShape;
        var total: usize = 1;
        for (dims_in) |dim| {
            if (dim == 0) return Error.InvalidShape;
            const result = @mulWithOverflow(total, dim);
            if (result[1] != 0) return Error.Overflow;
            total = result[0];
        }
        const dims = try allocator.dupe(usize, dims_in);
        errdefer allocator.free(dims);
        const strides = try allocator.dupe(usize, strides_in);
        errdefer allocator.free(strides);
        return .{ .dims = dims, .strides = strides, .total_size = total };
    }

    pub fn deinit(self: *Shape, allocator: Allocator) void {

        if (self.freed.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) {
            return;
        }
        allocator.free(self.dims);
        allocator.free(self.strides);
    }

    pub fn copy(self: *const Shape, allocator: Allocator) !Shape {
        return Shape.initWithStrides(allocator, self.dims, self.strides);
    }

    pub fn totalSize(self: *const Shape) usize {
        return self.total_size;
    }

    pub fn equals(self: *const Shape, other: *const Shape) bool {
        return mem.eql(usize, self.dims, other.dims) and mem.eql(usize, self.strides, other.strides);
    }

    pub fn isContiguous(self: *const Shape) bool {
        var expected: usize = 1;
        var i: usize = self.dims.len;
        while (i > 0) {
            i -= 1;
            if (self.strides[i] != expected) return false;
            expected *= self.dims[i];
        }
        return true;
    }

    pub fn broadcastCompatible(self: *const Shape, target: *const Shape) bool {
        if (target.dims.len < self.dims.len) return false;
        const offset = target.dims.len - self.dims.len;
        var i: usize = 0;
        while (i < self.dims.len) : (i += 1) {
            const source_dim = self.dims[i];
            const target_dim = target.dims[offset + i];
            if (source_dim != target_dim and source_dim != 1) return false;
        }
        return true;
    }
};

pub fn MatmulComptime(comptime M: usize, comptime K: usize, comptime N: usize) type {
    return struct {
        pub fn execute(a: *const Tensor, b: *const Tensor, out: *Tensor) void {
            comptime var i: usize = 0;
            inline while (i < M) : (i += 1) {
                comptime var j: usize = 0;
                inline while (j < N) : (j += 1) {
                    var sum_value: f32 = 0.0;
                    comptime var k: usize = 0;
                    inline while (k < K) : (k += 1) {
                        sum_value += a.data[i * a.shape.strides[0] + k * a.shape.strides[1]] * b.data[k * b.shape.strides[0] + j * b.shape.strides[1]];
                    }
                    out.data[i * out.shape.strides[0] + j * out.shape.strides[1]] = sum_value;
                }
            }
        }
    };
}

pub const Tensor = struct {
    data: []align(32) f32,
    base_data: []align(32) f32,
    shape: Shape,
    allocator: Allocator,
    refcount: *usize,
    cow: *bool,
    huge_allocator_owner: ?*HugePageAllocator,

    pub fn init(allocator: Allocator, dims: []const usize) !Tensor {
        var shape = try Shape.init(allocator, dims);
        errdefer shape.deinit(allocator);
        const data = try allocator.alignedAlloc(f32, @as(?u29, alignment), shape.totalSize());
        errdefer allocator.free(data);
        @memset(data, 0.0);
        const refcount = try allocator.create(usize);
        errdefer allocator.destroy(refcount);
        refcount.* = 1;
        const cow = try allocator.create(bool);
        errdefer allocator.destroy(cow);
        cow.* = false;
        return .{ .data = data, .base_data = data, .shape = shape, .allocator = allocator, .refcount = refcount, .cow = cow, .huge_allocator_owner = null };
    }

    fn initHugeUninitialized(parent_allocator: Allocator, dims: []const usize) !Tensor {
        const owner = try HugePageAllocator.create(parent_allocator);
        errdefer {
            const parent = owner.parent;
            owner.deinit();
            parent.destroy(owner);
        }
        const allocator = owner.allocator();
        var shape = try Shape.init(allocator, dims);
        errdefer shape.deinit(allocator);
        const element_count = shape.totalSize();
        const byte_count_result = @mulWithOverflow(element_count, @sizeOf(f32));
        if (byte_count_result[1] != 0) return Error.Overflow;
        const allocation_bytes = try roundUpToHugePage(byte_count_result[0]);
        const allocation_elements = allocation_bytes / @sizeOf(f32);
        const base_data = try allocator.alignedAlloc(f32, @as(?u29, alignment), allocation_elements);
        errdefer allocator.free(base_data);
        const data = base_data[0..element_count];
        const refcount = try allocator.create(usize);
        errdefer allocator.destroy(refcount);
        refcount.* = 1;
        const cow = try allocator.create(bool);
        errdefer allocator.destroy(cow);
        cow.* = false;
        return .{
            .data = data,
            .base_data = base_data,
            .shape = shape,
            .allocator = allocator,
            .refcount = refcount,
            .cow = cow,
            .huge_allocator_owner = owner,
        };
    }

    pub fn initHuge(parent_allocator: ?Allocator, dims: []const usize) !Tensor {
        const tensor = try Tensor.initHugeUninitialized(parent_allocator orelse std.heap.page_allocator, dims);
        @memset(tensor.data, 0.0);
        return tensor;
    }

    pub fn copyHuge(self: *const Tensor, parent_allocator: Allocator) !Tensor {
        var result = try Tensor.initHugeUninitialized(parent_allocator, self.shape.dims);
        errdefer result.deinit();
        const total = self.shape.totalSize();
        if (self.shape.isContiguous()) {
            @memcpy(result.data[0..total], self.data[0..total]);
        } else {
            var iterator = TensorIterator.init(&self.shape);
            var index: usize = 0;
            while (index < total) : (index += 1) {
                result.data[index] = self.data[iterator.offset];
                _ = iterator.advance();
            }
        }
        return result;
    }

    pub fn isHugeBacked(self: *const Tensor) bool {
        const owner = self.huge_allocator_owner orelse return false;
        return owner.isHugePointer(@ptrCast(self.base_data.ptr));
    }

    pub fn initWithArena(arena: *memory.ArenaAllocator, dims: []const usize) !Tensor {
        return init(arena.allocator(), dims);
    }

    pub fn initWithPool(pool: *memory.PoolAllocator, dims: []const usize) !Tensor {
        return init(pool.allocator(), dims);
    }

    pub fn initWithSlab(slab: *memory.SlabAllocator, dims: []const usize) !Tensor {
        return init(slab.allocator(), dims);
    }

    pub fn initWithBuddy(buddy: *memory.BuddyAllocator, dims: []const usize) !Tensor {
        return init(buddy.allocator(), dims);
    }

    pub fn retain(self: *Tensor) void {
        _ = @atomicRmw(usize, self.refcount, .Add, 1, .acq_rel);
        self.cow.* = true;
    }

    pub fn release(self: *Tensor) void {
        const allocator = self.allocator;
        const owner = self.huge_allocator_owner;
        self.shape.deinit(allocator);
        const old = @atomicRmw(usize, self.refcount, .Sub, 1, .acq_rel);
        if (old == 1) {
            allocator.free(self.base_data);
            allocator.destroy(self.refcount);
            allocator.destroy(self.cow);
            if (owner) |huge_owner| {
                const parent = huge_owner.parent;
                huge_owner.deinit();
                parent.destroy(huge_owner);
            }
            self.* = undefined;
        }
    }

    pub fn deinit(self: *Tensor) void {
        self.release();
    }

    fn flatIndex(self: *const Tensor, indices: []const usize) !usize {
        if (indices.len != self.shape.dims.len) return Error.InvalidAxis;
        var offset: usize = 0;
        for (indices, 0..) |index, axis| {
            if (index >= self.shape.dims[axis]) return Error.OutOfBounds;
            offset += index * self.shape.strides[axis];
        }
        return offset;
    }

    fn ensureWritable(self: *Tensor) !void {
        if (@atomicLoad(usize, self.refcount, .acquire) == 1) {
            self.cow.* = false;
            return;
        }
        const old_allocator = self.allocator;
        const old_owner = self.huge_allocator_owner;
        const new_allocator = if (old_owner) |owner| owner.parent else old_allocator;
        const total = self.shape.totalSize();
        const new_data = try new_allocator.alignedAlloc(f32, @as(?u29, alignment), total);
        errdefer new_allocator.free(new_data);
        if (self.shape.isContiguous()) {
            @memcpy(new_data, self.data[0..total]);
        } else {
            var iterator = TensorIterator.init(&self.shape);
            var index: usize = 0;
            while (index < total) : (index += 1) {
                new_data[index] = self.data[iterator.offset];
                _ = iterator.advance();
            }
        }
        const new_refcount = try new_allocator.create(usize);
        errdefer new_allocator.destroy(new_refcount);
        new_refcount.* = 1;
        const new_cow = try new_allocator.create(bool);
        errdefer new_allocator.destroy(new_cow);
        new_cow.* = false;
        const old_base_data = self.base_data;
        const old_refcount = self.refcount;
        const old_cow = self.cow;
        const old_count = @atomicRmw(usize, old_refcount, .Sub, 1, .acq_rel);
        self.data = new_data;
        self.base_data = new_data;
        self.allocator = new_allocator;
        self.refcount = new_refcount;
        self.cow = new_cow;
        self.huge_allocator_owner = null;
        if (old_count == 1) {
            old_allocator.free(old_base_data);
            old_allocator.destroy(old_refcount);
            old_allocator.destroy(old_cow);
            if (old_owner) |owner| {
                const parent = owner.parent;
                owner.deinit();
                parent.destroy(owner);
            }
        }
    }

    pub fn copy(self: *const Tensor, allocator: Allocator) !Tensor {
        var result = try Tensor.init(allocator, self.shape.dims);
        errdefer result.deinit();
        const total = self.shape.totalSize();
        const contiguous = self.shape.isContiguous();
        if (contiguous) {
            @memcpy(result.data[0..total], self.data[0..total]);
        } else {
            var iterator = TensorIterator.init(&self.shape);
            var i: usize = 0;
            while (i < total) : (i += 1) {
                result.data[i] = self.data[iterator.offset];
                _ = iterator.advance();
            }
        }
        return result;
    }

    pub fn get(self: *const Tensor, indices: []const usize) !f32 {
        return self.data[try self.flatIndex(indices)];
    }

    pub fn set(self: *Tensor, indices: []const usize, value: f32) !void {
        try self.ensureWritable();
        self.data[try self.flatIndex(indices)] = value;
    }

    pub fn fill(self: *Tensor, value: f32) !void {
        try self.ensureWritable();
        const total = self.shape.totalSize();
        const contiguous = self.shape.isContiguous();
        if (contiguous) {
            @memset(self.data[0..total], value);
            return;
        }
        var iterator = TensorIterator.init(&self.shape);
        var i: usize = 0;
        while (i < total) : (i += 1) {
            self.data[iterator.offset] = value;
            _ = iterator.advance();
        }
    }

    fn binaryFast(self: *Tensor, other: *const Tensor, comptime op: enum { add, sub, mul, div }) !void {
        if (!self.shape.equals(&other.shape)) return Error.ShapeMismatch;
        try self.ensureWritable();
        const total = self.shape.totalSize();
        const self_contiguous = self.shape.isContiguous();
        const other_contiguous = other.shape.isContiguous();
        if (self_contiguous and other_contiguous) {
            if (op == .div) {
                var i: usize = 0;
                while (i < total) : (i += 1) {
                    if (other.data[i] == 0.0) return Error.DivideByZero;
                }
                i = 0;
                const limit = total - total % vector_width;
                while (i < limit) : (i += vector_width) {
                    const a: Vec8 = self.data[i..][0..vector_width].*;
                    const b: Vec8 = other.data[i..][0..vector_width].*;
                    self.data[i..][0..vector_width].* = a / b;
                }
                while (i < total) : (i += 1) {
                    self.data[i] /= other.data[i];
                }
            } else {
                var i: usize = 0;
                const limit = total - total % vector_width;
                while (i < limit) : (i += vector_width) {
                    const a: Vec8 = self.data[i..][0..vector_width].*;
                    const b: Vec8 = other.data[i..][0..vector_width].*;
                    self.data[i..][0..vector_width].* = switch (op) {
                        .add => a + b,
                        .sub => a - b,
                        .mul => a * b,
                        .div => unreachable,
                    };
                }
                while (i < total) : (i += 1) {
                    switch (op) {
                        .add => self.data[i] += other.data[i],
                        .sub => self.data[i] -= other.data[i],
                        .mul => self.data[i] *= other.data[i],
                        .div => unreachable,
                    }
                }
            }
            return;
        }
        if (op == .div) {
            var check_iterator = TensorIterator.init(&other.shape);
            var ci: usize = 0;
            while (ci < total) : (ci += 1) {
                if (other.data[check_iterator.offset] == 0.0) return Error.DivideByZero;
                _ = check_iterator.advance();
            }
        }
        var self_iterator = TensorIterator.init(&self.shape);
        var other_iterator = TensorIterator.init(&other.shape);
        var i: usize = 0;
        while (i < total) : (i += 1) {
            switch (op) {
                .add => self.data[self_iterator.offset] += other.data[other_iterator.offset],
                .sub => self.data[self_iterator.offset] -= other.data[other_iterator.offset],
                .mul => self.data[self_iterator.offset] *= other.data[other_iterator.offset],
                .div => self.data[self_iterator.offset] /= other.data[other_iterator.offset],
            }
            _ = self_iterator.advance();
            _ = other_iterator.advance();
        }
    }

    fn scalarFast(self: *Tensor, scalar: f32, comptime op: enum { add, sub, mul, div }) !void {
        if (op == .div and scalar == 0.0) return Error.DivideByZero;
        try self.ensureWritable();
        const total = self.shape.totalSize();
        const contiguous = self.shape.isContiguous();
        if (contiguous) {
            const scalar_vector: Vec8 = @splat(scalar);
            var i: usize = 0;
            const limit = total - total % vector_width;
            while (i < limit) : (i += vector_width) {
                const a: Vec8 = self.data[i..][0..vector_width].*;
                self.data[i..][0..vector_width].* = switch (op) {
                    .add => a + scalar_vector,
                    .sub => a - scalar_vector,
                    .mul => a * scalar_vector,
                    .div => a / scalar_vector,
                };
            }
            while (i < total) : (i += 1) {
                switch (op) {
                    .add => self.data[i] += scalar,
                    .sub => self.data[i] -= scalar,
                    .mul => self.data[i] *= scalar,
                    .div => self.data[i] /= scalar,
                }
            }
            return;
        }
        var iterator = TensorIterator.init(&self.shape);
        var i: usize = 0;
        while (i < total) : (i += 1) {
            switch (op) {
                .add => self.data[iterator.offset] += scalar,
                .sub => self.data[iterator.offset] -= scalar,
                .mul => self.data[iterator.offset] *= scalar,
                .div => self.data[iterator.offset] /= scalar,
            }
            _ = iterator.advance();
        }
    }

    pub fn addFast(self: *Tensor, other: *const Tensor) !void {
        return self.binaryFast(other, .add);
    }

    pub fn subFast(self: *Tensor, other: *const Tensor) !void {
        return self.binaryFast(other, .sub);
    }

    pub fn mulFast(self: *Tensor, other: *const Tensor) !void {
        return self.binaryFast(other, .mul);
    }

    pub fn divFast(self: *Tensor, other: *const Tensor) !void {
        return self.binaryFast(other, .div);
    }

    pub fn add(self: *Tensor, other: *const Tensor) !void {
        return self.addFast(other);
    }

    pub fn sub(self: *Tensor, other: *const Tensor) !void {
        return self.subFast(other);
    }

    pub fn mul(self: *Tensor, other: *const Tensor) !void {
        return self.mulFast(other);
    }

    pub fn div(self: *Tensor, other: *const Tensor) !void {
        return self.divFast(other);
    }

    pub fn addScalarFast(self: *Tensor, scalar: f32) !void {
        return self.scalarFast(scalar, .add);
    }

    pub fn subScalarFast(self: *Tensor, scalar: f32) !void {
        return self.scalarFast(scalar, .sub);
    }

    pub fn mulScalarFast(self: *Tensor, scalar: f32) !void {
        return self.scalarFast(scalar, .mul);
    }

    pub fn divScalarFast(self: *Tensor, scalar: f32) !void {
        return self.scalarFast(scalar, .div);
    }

    pub fn addScalar(self: *Tensor, scalar: f32) !void {
        return self.addScalarFast(scalar);
    }

    pub fn subScalar(self: *Tensor, scalar: f32) !void {
        return self.subScalarFast(scalar);
    }

    pub fn mulScalar(self: *Tensor, scalar: f32) !void {
        return self.mulScalarFast(scalar);
    }

    pub fn divScalar(self: *Tensor, scalar: f32) !void {
        return self.divScalarFast(scalar);
    }

    fn unaryFast(self: *Tensor, comptime op: enum { exp, log, sin, cos, tan, sqrt, abs }) !void {
        try self.ensureWritable();
        var iterator = TensorIterator.init(&self.shape);
        const total = self.shape.totalSize();
        const contiguous = self.shape.isContiguous();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            const offset = if (contiguous) i else iterator.offset;
            self.data[offset] = switch (op) {
                .exp => @exp(self.data[offset]),
                .log => if (self.data[offset] <= 0.0) -math.inf(f32) else @log(self.data[offset]),
                .sin => @sin(self.data[offset]),
                .cos => @cos(self.data[offset]),
                .tan => @tan(self.data[offset]),
                .sqrt => if (self.data[offset] < 0.0) math.nan(f32) else @sqrt(self.data[offset]),
                .abs => @abs(self.data[offset]),
            };
            if (!contiguous) _ = iterator.advance();
        }
    }

    pub fn expFast(self: *Tensor) !void {
        return self.unaryFast(.exp);
    }

    pub fn logFast(self: *Tensor) !void {
        return self.unaryFast(.log);
    }

    pub fn sinFast(self: *Tensor) !void {
        return self.unaryFast(.sin);
    }

    pub fn cosFast(self: *Tensor) !void {
        return self.unaryFast(.cos);
    }

    pub fn tanFast(self: *Tensor) !void {
        return self.unaryFast(.tan);
    }

    pub fn sqrtFast(self: *Tensor) !void {
        return self.unaryFast(.sqrt);
    }

    pub fn absFast(self: *Tensor) !void {
        return self.unaryFast(.abs);
    }

    pub fn exp(self: *Tensor) !void {
        return self.expFast();
    }

    pub fn log(self: *Tensor) !void {
        return self.logFast();
    }

    pub fn sin(self: *Tensor) !void {
        return self.sinFast();
    }

    pub fn cos(self: *Tensor) !void {
        return self.cosFast();
    }

    pub fn tan(self: *Tensor) !void {
        return self.tanFast();
    }

    pub fn sqrt(self: *Tensor) !void {
        return self.sqrtFast();
    }

    pub fn abs(self: *Tensor) !void {
        return self.absFast();
    }

    pub fn powFast(self: *Tensor, exponent: f32) !void {
        try self.ensureWritable();
        var iterator = TensorIterator.init(&self.shape);
        const total = self.shape.totalSize();
        const contiguous = self.shape.isContiguous();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            const offset = if (contiguous) i else iterator.offset;
            self.data[offset] = math.pow(f32, self.data[offset], exponent);
            if (!contiguous) _ = iterator.advance();
        }
    }

    pub fn pow(self: *Tensor, exponent: f32) !void {
        return self.powFast(exponent);
    }

    pub fn clipFast(self: *Tensor, min_value: f32, max_value: f32) !void {
        try self.ensureWritable();
        var iterator = TensorIterator.init(&self.shape);
        const total = self.shape.totalSize();
        const contiguous = self.shape.isContiguous();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            const offset = if (contiguous) i else iterator.offset;
            self.data[offset] = math.clamp(self.data[offset], min_value, max_value);
            if (!contiguous) _ = iterator.advance();
        }
    }

    pub fn clip(self: *Tensor, min_value: f32, max_value: f32) !void {
        return self.clipFast(min_value, max_value);
    }

    pub fn reshape(self: *Tensor, new_dims: []const usize) !void {
        if (!self.shape.isContiguous()) return Error.InvalidShape;
        var new_shape = try Shape.init(self.allocator, new_dims);
        errdefer new_shape.deinit(self.allocator);
        if (new_shape.totalSize() != self.shape.totalSize()) return Error.InvalidShape;
        var old_shape = self.shape;
        self.shape = new_shape;
        old_shape.deinit(self.allocator);
    }

    pub fn view(self: *Tensor, new_dims: []const usize) !Tensor {
        if (!self.shape.isContiguous()) return Error.InvalidShape;
        var new_shape = try Shape.init(self.allocator, new_dims);
        errdefer new_shape.deinit(self.allocator);
        if (new_shape.totalSize() != self.shape.totalSize()) return Error.InvalidShape;
        self.retain();
        return .{ .data = self.data, .base_data = self.base_data, .shape = new_shape, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow, .huge_allocator_owner = self.huge_allocator_owner };
    }

    pub fn newView(self: *Tensor, shape: Shape) !Tensor {
        if (shape.totalSize() != self.shape.totalSize()) return Error.InvalidShape;
        self.retain();
        return .{ .data = self.data, .base_data = self.base_data, .shape = shape, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow, .huge_allocator_owner = self.huge_allocator_owner };
    }

    pub fn slice(self: *Tensor, starts: []const usize, ends: []const usize) !Tensor {
        if (starts.len != self.shape.dims.len or ends.len != self.shape.dims.len) return Error.InvalidAxis;
        var new_dims_stack: [8]usize = undefined;
        var new_strides_stack: [8]usize = undefined;
        var offset: usize = 0;
        for (starts, 0..) |start, axis| {
            if (start > ends[axis] or ends[axis] > self.shape.dims[axis] or ends[axis] == start) return Error.OutOfBounds;
            new_dims_stack[axis] = ends[axis] - start;
            new_strides_stack[axis] = self.shape.strides[axis];
            offset += start * self.shape.strides[axis];
        }
        var result = try Tensor.init(self.allocator, new_dims_stack[0..starts.len]);
        errdefer result.deinit();
        var src_shape = try Shape.initWithStrides(self.allocator, new_dims_stack[0..starts.len], new_strides_stack[0..starts.len]);
        defer src_shape.deinit(self.allocator);
        var src_iterator = TensorIterator.init(&src_shape);
        const total = src_shape.totalSize();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            result.data[i] = self.data[offset + src_iterator.offset];
            _ = src_iterator.advance();
        }
        return result;
    }

    pub fn transpose(self: *Tensor, axes: []const usize) !Tensor {
        if (axes.len != self.shape.dims.len) return Error.InvalidAxis;
        var seen = [_]bool{false} ** 8;
        var dims_stack: [8]usize = undefined;
        var strides_stack: [8]usize = undefined;
        for (axes, 0..) |axis, i| {
            if (axis >= axes.len or seen[axis]) return Error.InvalidAxis;
            seen[axis] = true;
            dims_stack[i] = self.shape.dims[axis];
            strides_stack[i] = self.shape.strides[axis];
        }
        var new_shape = try Shape.initWithStrides(self.allocator, dims_stack[0..axes.len], strides_stack[0..axes.len]);
        errdefer new_shape.deinit(self.allocator);
        self.retain();
        return .{ .data = self.data, .base_data = self.base_data, .shape = new_shape, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow, .huge_allocator_owner = self.huge_allocator_owner };
    }

    pub fn broadcast(self: *Tensor, target_dims: []const usize) !Tensor {
        if (target_dims.len < self.shape.dims.len or target_dims.len > 8) return Error.ShapeMismatch;
        var strides_stack: [8]usize = [_]usize{0} ** 8;
        const offset = target_dims.len - self.shape.dims.len;
        var axis: usize = 0;
        while (axis < target_dims.len) : (axis += 1) {
            if (axis < offset) {
                strides_stack[axis] = 0;
            } else {
                const source_axis = axis - offset;
                const source_dim = self.shape.dims[source_axis];
                const target_dim = target_dims[axis];
                if (source_dim != target_dim and source_dim != 1) return Error.ShapeMismatch;
                strides_stack[axis] = if (source_dim == 1 and target_dim > 1) 0 else self.shape.strides[source_axis];
            }
        }
        var new_shape = try Shape.initWithStrides(self.allocator, target_dims, strides_stack[0..target_dims.len]);
        errdefer new_shape.deinit(self.allocator);
        self.retain();
        return .{ .data = self.data, .base_data = self.base_data, .shape = new_shape, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow, .huge_allocator_owner = self.huge_allocator_owner };
    }

    pub fn unsqueeze(self: *Tensor, axis: usize) !Tensor {
        if (axis > self.shape.dims.len or self.shape.dims.len == 8) return Error.InvalidAxis;
        var dims_stack: [8]usize = undefined;
        var strides_stack: [8]usize = undefined;
        var source_axis: usize = 0;
        var target_axis: usize = 0;
        while (target_axis < self.shape.dims.len + 1) : (target_axis += 1) {
            if (target_axis == axis) {
                dims_stack[target_axis] = 1;
                strides_stack[target_axis] = if (source_axis < self.shape.strides.len) self.shape.strides[source_axis] else 1;
            } else {
                dims_stack[target_axis] = self.shape.dims[source_axis];
                strides_stack[target_axis] = self.shape.strides[source_axis];
                source_axis += 1;
            }
        }
        var new_shape = try Shape.initWithStrides(self.allocator, dims_stack[0 .. self.shape.dims.len + 1], strides_stack[0 .. self.shape.dims.len + 1]);
        errdefer new_shape.deinit(self.allocator);
        self.retain();
        return .{ .data = self.data, .base_data = self.base_data, .shape = new_shape, .allocator = self.allocator, .refcount = self.refcount, .cow = self.cow, .huge_allocator_owner = self.huge_allocator_owner };
    }

    pub fn zeros(allocator: Allocator, dims: []const usize) !Tensor {
        return Tensor.init(allocator, dims);
    }

    pub fn ones(allocator: Allocator, dims: []const usize) !Tensor {
        var tensor = try Tensor.init(allocator, dims);
        try tensor.fill(1.0);
        return tensor;
    }

    pub fn full(allocator: Allocator, dims: []const usize, value: f32) !Tensor {
        var tensor = try Tensor.init(allocator, dims);
        try tensor.fill(value);
        return tensor;
    }

    pub fn randomUniform(allocator: Allocator, dims: []const usize, min_value: f32, max_value: f32, seed: u64) !Tensor {
        var prng = types.PRNG.init(seed);
        var tensor = try Tensor.init(allocator, dims);
        const total = tensor.shape.totalSize();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            tensor.data[i] = prng.float() * (max_value - min_value) + min_value;
        }
        return tensor;
    }

    pub fn randomNormal(allocator: Allocator, dims: []const usize, mean_value: f32, stddev_value: f32, seed: u64) !Tensor {
        var prng = types.PRNG.init(seed);
        var tensor = try Tensor.init(allocator, dims);
        const total = tensor.shape.totalSize();
        var i: usize = 0;
        while (i < total) : (i += 1) {
            const u = 1.0 - prng.float();
            const v = 1.0 - prng.float();
            tensor.data[i] = mean_value + stddev_value * (@sqrt(-2.0 * @log(u)) * @cos(2.0 * math.pi * v));
        }
        return tensor;
    }

    pub fn identity(allocator: Allocator, n: usize) !Tensor {
        if (n == 0) return Error.InvalidShape;
        var tensor = try Tensor.init(allocator, &.{ n, n });
        var i: usize = 0;
        while (i < n) : (i += 1) tensor.data[i * n + i] = 1.0;
        return tensor;
    }

    pub fn sum(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        var dims_stack: [8]usize = undefined;
        const result_rank = if (self.shape.dims.len == 1) 1 else self.shape.dims.len - 1;
        if (self.shape.dims.len == 1) {
            dims_stack[0] = 1;
        } else {
            var j: usize = 0;
            for (self.shape.dims, 0..) |dim, i| {
                if (i != axis) {
                    dims_stack[j] = dim;
                    j += 1;
                }
            }
        }
        var result = try Tensor.init(allocator, dims_stack[0..result_rank]);
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            var result_offset: usize = 0;
            var result_axis: usize = 0;
            for (0..self.shape.dims.len) |input_axis| {
                if (input_axis != axis) {
                    result_offset += iterator.indices[input_axis] * result.shape.strides[result_axis];
                    result_axis += 1;
                }
            }
            result.data[result_offset] += self.data[iterator.offset];
            _ = iterator.advance();
        }
        return result;
    }

    pub fn mean(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var result = try self.sum(allocator, axis);
        try result.divScalar(@floatFromInt(self.shape.dims[axis]));
        return result;
    }

    pub fn max(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        var dims_stack: [8]usize = undefined;
        const result_rank = if (self.shape.dims.len == 1) 1 else self.shape.dims.len - 1;
        if (self.shape.dims.len == 1) {
            dims_stack[0] = 1;
        } else {
            var j: usize = 0;
            for (self.shape.dims, 0..) |dim, i| {
                if (i != axis) {
                    dims_stack[j] = dim;
                    j += 1;
                }
            }
        }
        var result = try Tensor.init(allocator, dims_stack[0..result_rank]);
        try result.fill(-math.inf(f32));
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            var result_offset: usize = 0;
            var result_axis: usize = 0;
            for (0..self.shape.dims.len) |input_axis| {
                if (input_axis != axis) {
                    result_offset += iterator.indices[input_axis] * result.shape.strides[result_axis];
                    result_axis += 1;
                }
            }
            result.data[result_offset] = @max(result.data[result_offset], self.data[iterator.offset]);
            _ = iterator.advance();
        }
        return result;
    }

    pub fn min(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.shape.dims.len) return Error.InvalidAxis;
        var dims_stack: [8]usize = undefined;
        const result_rank = if (self.shape.dims.len == 1) 1 else self.shape.dims.len - 1;
        if (self.shape.dims.len == 1) {
            dims_stack[0] = 1;
        } else {
            var j: usize = 0;
            for (self.shape.dims, 0..) |dim, i| {
                if (i != axis) {
                    dims_stack[j] = dim;
                    j += 1;
                }
            }
        }
        var result = try Tensor.init(allocator, dims_stack[0..result_rank]);
        try result.fill(math.inf(f32));
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            var result_offset: usize = 0;
            var result_axis: usize = 0;
            for (0..self.shape.dims.len) |input_axis| {
                if (input_axis != axis) {
                    result_offset += iterator.indices[input_axis] * result.shape.strides[result_axis];
                    result_axis += 1;
                }
            }
            result.data[result_offset] = @min(result.data[result_offset], self.data[iterator.offset]);
            _ = iterator.advance();
        }
        return result;
    }

    pub fn variancePopulation(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var mean_tensor = try self.mean(allocator, axis);
        defer mean_tensor.deinit();
        var result = try Tensor.init(allocator, mean_tensor.shape.dims);
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            var result_offset: usize = 0;
            var result_axis: usize = 0;
            for (0..self.shape.dims.len) |input_axis| {
                if (input_axis != axis) {
                    result_offset += iterator.indices[input_axis] * result.shape.strides[result_axis];
                    result_axis += 1;
                }
            }
            const difference = self.data[iterator.offset] - mean_tensor.data[result_offset];
            result.data[result_offset] += difference * difference;
            _ = iterator.advance();
        }
        try result.divScalar(@floatFromInt(self.shape.dims[axis]));
        return result;
    }

    pub fn varianceSample(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var mean_tensor = try self.mean(allocator, axis);
        defer mean_tensor.deinit();
        var result = try Tensor.init(allocator, mean_tensor.shape.dims);
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            var result_offset: usize = 0;
            var result_axis: usize = 0;
            for (0..self.shape.dims.len) |input_axis| {
                if (input_axis != axis) {
                    result_offset += iterator.indices[input_axis] * result.shape.strides[result_axis];
                    result_axis += 1;
                }
            }
            const difference = self.data[iterator.offset] - mean_tensor.data[result_offset];
            result.data[result_offset] += difference * difference;
            _ = iterator.advance();
        }
        const n = self.shape.dims[axis];
        if (n > 1) {
            try result.divScalar(@floatFromInt(n - 1));
        }
        return result;
    }

    pub fn stddev(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var result = try self.variancePopulation(allocator, axis);
        try result.sqrt();
        return result;
    }

    pub fn stddevSample(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var result = try self.varianceSample(allocator, axis);
        try result.sqrt();
        return result;
    }

    pub fn normL2(self: *const Tensor) !f32 {
        var result: f32 = 0.0;
        const contiguous = self.shape.isContiguous();
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            const value = if (contiguous) self.data[count] else self.data[iterator.offset];
            result += value * value;
            if (!contiguous) _ = iterator.advance();
        }
        return @sqrt(result);
    }

    pub fn norm(self: *const Tensor, order: f32) !f32 {
        if (order <= 0.0) return Error.InvalidShape;
        var result: f32 = 0.0;
        const contiguous = self.shape.isContiguous();
        var iterator = TensorIterator.init(&self.shape);
        var count: usize = 0;
        while (count < self.shape.totalSize()) : (count += 1) {
            const value = if (contiguous) self.data[count] else self.data[iterator.offset];
            result += math.pow(f32, @abs(value), order);
            if (!contiguous) _ = iterator.advance();
        }
        return math.pow(f32, result, 1.0 / order);
    }

    pub fn dot(self: *const Tensor, other: *const Tensor) !f32 {
        if (self.shape.dims.len != 1 or other.shape.dims.len != 1 or self.shape.dims[0] != other.shape.dims[0]) return Error.ShapeMismatch;
        var result: f32 = 0.0;
        const n = self.shape.dims[0];
        var i: usize = 0;
        while (i < n) : (i += 1) result += self.data[i * self.shape.strides[0]] * other.data[i * other.shape.strides[0]];
        return result;
    }

    pub fn outer(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
        if (a.shape.dims.len != 1 or b.shape.dims.len != 1) return Error.ShapeMismatch;
        var result = try Tensor.init(allocator, &.{ a.shape.dims[0], b.shape.dims[0] });
        var i: usize = 0;
        while (i < a.shape.dims[0]) : (i += 1) {
            var j: usize = 0;
            while (j < b.shape.dims[0]) : (j += 1) result.data[i * result.shape.strides[0] + j * result.shape.strides[1]] = a.data[i * a.shape.strides[0]] * b.data[j * b.shape.strides[0]];
        }
        return result;
    }

    pub fn trace(self: *const Tensor) !f32 {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return Error.MustBeSquare;
        var result: f32 = 0.0;
        var i: usize = 0;
        while (i < self.shape.dims[0]) : (i += 1) result += self.data[i * self.shape.strides[0] + i * self.shape.strides[1]];
        return result;
    }

    pub fn matmul(a: *const Tensor, b: *const Tensor, allocator: Allocator) !Tensor {
        if (a.shape.dims.len != 2 or b.shape.dims.len != 2 or a.shape.dims[1] != b.shape.dims[0]) return Error.ShapeMismatch;
        const m = a.shape.dims[0];
        const k = a.shape.dims[1];
        const n = b.shape.dims[1];
        if (highPerformanceAvailable() and a.shape.isContiguous() and b.shape.isContiguous() and m >= 256 and n >= 256 and k >= 256) {
            return highPerformanceMatmul(a, b, allocator);
        }
        return scalarBlockedMatmul(a, b, allocator);
    }

    pub fn isClose(self: *const Tensor, other: *const Tensor, rtol: f32, atol: f32) !bool {
        if (!self.shape.equals(&other.shape)) return Error.ShapeMismatch;
        var a_iterator = TensorIterator.init(&self.shape);
        var b_iterator = TensorIterator.init(&other.shape);
        var i: usize = 0;
        while (i < self.shape.totalSize()) : (i += 1) {
            const av = self.data[a_iterator.offset];
            const bv = other.data[b_iterator.offset];
            if (@abs(av - bv) > atol + rtol * @abs(bv)) return false;
            _ = a_iterator.advance();
            _ = b_iterator.advance();
        }
        return true;
    }

    pub fn toInt(self: *const Tensor, allocator: Allocator) !Tensor {
        var result = try Tensor.init(allocator, self.shape.dims);
        var iterator = TensorIterator.init(&self.shape);
        var i: usize = 0;
        const max_precise_int: f32 = 16777216.0;
        while (i < self.shape.totalSize()) : (i += 1) {
            result.data[i] = @round(math.clamp(self.data[iterator.offset], -max_precise_int, max_precise_int));
            _ = iterator.advance();
        }
        return result;
    }

    pub fn toFixedFast(self: *const Tensor, allocator: Allocator) !Tensor {
        var result = try Tensor.init(allocator, self.shape.dims);
        var iterator = TensorIterator.init(&self.shape);
        var i: usize = 0;
        while (i < self.shape.totalSize()) : (i += 1) {
            result.data[i] = @floor(self.data[iterator.offset] * 4294967296.0) / 4294967296.0;
            _ = iterator.advance();
        }
        return result;
    }

    pub fn toFixed(self: *const Tensor, allocator: Allocator) !Tensor {
        return self.toFixedFast(allocator);
    }

    pub fn arange(allocator: Allocator, start: f32, end: f32, step: f32) !Tensor {
        if (step == 0.0) return Error.InvalidShape;
        if (step > 0 and start >= end) return Error.InvalidShape;
        if (step < 0 and start <= end) return Error.InvalidShape;
        const count_float = @ceil((end - start) / step);
        if (count_float <= 0.0 or !std.math.isFinite(count_float)) return Error.InvalidShape;
        const count: usize = @intFromFloat(count_float);
        if (count == 0) return Error.InvalidShape;
        var result = try Tensor.init(allocator, &.{count});
        var i: usize = 0;
        while (i < count) : (i += 1) result.data[i] = start + @as(f32, @floatFromInt(i)) * step;
        return result;
    }

    pub fn linspace(allocator: Allocator, start: f32, end: f32, count: usize) !Tensor {
        if (count == 0) return Error.InvalidShape;
        var result = try Tensor.init(allocator, &.{count});
        var i: usize = 0;
        while (i < count) : (i += 1) {
            result.data[i] = if (count == 1) start else start + (@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(count - 1))) * (end - start);
        }
        return result;
    }

    pub fn det(self: *const Tensor, allocator: Allocator) !f32 {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return Error.MustBeSquare;
        const n = self.shape.dims[0];
        var matrix = try self.copy(allocator);
        defer matrix.deinit();
        var determinant: f32 = 1.0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var pivot = i;
            var max_value = @abs(matrix.data[i * n + i]);
            var row: usize = i + 1;
            while (row < n) : (row += 1) {
                const value = @abs(matrix.data[row * n + i]);
                if (value > max_value) {
                    max_value = value;
                    pivot = row;
                }
            }
            if (max_value < 1e-30) return 0.0;
            if (pivot != i) {
                var col: usize = 0;
                while (col < n) : (col += 1) {
                    const temporary = matrix.data[i * n + col];
                    matrix.data[i * n + col] = matrix.data[pivot * n + col];
                    matrix.data[pivot * n + col] = temporary;
                }
                determinant = -determinant;
            }
            const pivot_value = matrix.data[i * n + i];
            determinant *= pivot_value;
            row = i + 1;
            while (row < n) : (row += 1) {
                const factor = matrix.data[row * n + i] / pivot_value;
                var col: usize = i;
                while (col < n) : (col += 1) matrix.data[row * n + col] -= factor * matrix.data[i * n + col];
            }
        }
        return determinant;
    }

    pub fn save(self: *const Tensor, writer: anytype) !void {
        const ndim: u64 = @intCast(self.shape.dims.len);
        try writer.writeInt(u64, ndim, .little);
        for (self.shape.dims) |d| try writer.writeInt(u64, @intCast(d), .little);
        for (self.data) |v| try writer.writeInt(u32, @bitCast(v), .little);
    }

    pub fn load(allocator: Allocator, reader: anytype) !Tensor {
        const ndim = try reader.readInt(u64, .little);
        if (ndim == 0 or ndim > 8) return Error.InvalidShape;
        var dims: [8]usize = undefined;
        var i: usize = 0;
        while (i < ndim) : (i += 1) {
            const d = try reader.readInt(u64, .little);
            if (d == 0) return Error.InvalidShape;
            dims[i] = @intCast(d);
        }
        var t = try Tensor.init(allocator, dims[0..ndim]);
        errdefer t.deinit();
        var j: usize = 0;
        while (j < t.data.len) : (j += 1) {
            t.data[j] = @bitCast(try reader.readInt(u32, .little));
        }
        return t;
    }

    pub fn eye(allocator: Allocator, dims: []const usize) !Tensor {
        if (dims.len != 2 or dims[0] != dims[1]) return Error.InvalidShape;
        var tensor = try init(allocator, dims);
        try tensor.fill(0.0);
        const n = dims[0];
        var i: usize = 0;
        while (i < n) : (i += 1) tensor.data[i * n + i] = 1.0;
        return tensor;
    }

    pub fn inverse(self: *const Tensor, allocator: Allocator) !Tensor {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return Error.MustBeSquare;
        const n = self.shape.dims[0];
        var augmented = try Tensor.init(allocator, &.{ n, 2 * n });
        defer augmented.deinit();
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) augmented.data[i * 2 * n + j] = self.data[i * self.shape.strides[0] + j * self.shape.strides[1]];
            augmented.data[i * 2 * n + i + n] = 1.0;
        }
        i = 0;
        while (i < n) : (i += 1) {
            var pivot = i;
            var max_value = @abs(augmented.data[i * 2 * n + i]);
            var row: usize = i + 1;
            while (row < n) : (row += 1) {
                const value = @abs(augmented.data[row * 2 * n + i]);
                if (value > max_value) {
                    max_value = value;
                    pivot = row;
                }
            }
            if (max_value < 1e-30) return Error.SingularMatrix;
            if (pivot != i) {
                var col: usize = 0;
                while (col < 2 * n) : (col += 1) {
                    const temporary = augmented.data[i * 2 * n + col];
                    augmented.data[i * 2 * n + col] = augmented.data[pivot * 2 * n + col];
                    augmented.data[pivot * 2 * n + col] = temporary;
                }
            }
            const pivot_value = augmented.data[i * 2 * n + i];
            var col: usize = 0;
            while (col < 2 * n) : (col += 1) augmented.data[i * 2 * n + col] /= pivot_value;
            row = 0;
            while (row < n) : (row += 1) {
                if (row != i) {
                    const factor = augmented.data[row * 2 * n + i];
                    col = 0;
                    while (col < 2 * n) : (col += 1) augmented.data[row * 2 * n + col] -= factor * augmented.data[i * 2 * n + col];
                }
            }
        }
        var result = try Tensor.init(allocator, &.{ n, n });
        i = 0;
        while (i < n) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) result.data[i * n + j] = augmented.data[i * 2 * n + j + n];
        }
        return result;
    }
};


fn fillDeterministic(tensor: *Tensor, seed: u64) void {
    var generator = types.PRNG.init(seed);
    var index: usize = 0;
    while (index < tensor.data.len) : (index += 1) {
        tensor.data[index] = generator.float() - 0.5;
    }
}

fn verifyMatmulResult(a: *const Tensor, b: *const Tensor, c: *const Tensor, relative_tolerance: f32) bool {
    const m = a.shape.dims[0];
    const k = a.shape.dims[1];
    const n = b.shape.dims[1];
    const total_outputs = m * n;
    const sample_count = @min(total_outputs, 4096);
    var sample: usize = 0;
    while (sample < sample_count) : (sample += 1) {
        const mixed = @as(u64, @intCast(sample)) *% 0x9e3779b97f4a7c15 +% 0xbf58476d1ce4e5b9;
        const row = @as(usize, @intCast(mixed % @as(u64, @intCast(m))));
        const column = @as(usize, @intCast((mixed >> 17) % @as(u64, @intCast(n))));
        var reference: f32 = 0.0;
        var depth: usize = 0;
        while (depth < k) : (depth += 1) {
            reference += a.data[row * a.shape.strides[0] + depth * a.shape.strides[1]] * b.data[depth * b.shape.strides[0] + column * b.shape.strides[1]];
        }
        const actual = c.data[row * c.shape.strides[0] + column * c.shape.strides[1]];
        const scale = @max(@abs(reference), 1.0);
        if (@abs(actual - reference) > relative_tolerance * scale) return false;
    }
    return true;
}

pub fn benchmarkGemm(parent_allocator: Allocator, m: usize, n: usize, k: usize, minimum_duration_ns: u64) !void {
    if (m == 0 or n == 0 or k == 0) return Error.InvalidShape;
    var a = try Tensor.initHuge(parent_allocator, &.{ m, k });
    defer a.deinit();
    var b = try Tensor.initHuge(parent_allocator, &.{ k, n });
    defer b.deinit();
    fillDeterministic(&a, 0x123456789abcdef0);
    fillDeterministic(&b, 0xfedcba9876543210);
    var warmup = try Tensor.matmul(&a, &b, parent_allocator);
    defer warmup.deinit();
    if (!verifyMatmulResult(&a, &b, &warmup, 1.0e-4)) return error.VerificationFailed;
    var last_result: Tensor = undefined;
    var has_last_result = false;
    defer if (has_last_result) last_result.deinit();
    var timer = try std.time.Timer.start();
    var iterations: usize = 0;
    var elapsed: u64 = 0;
    while (elapsed < minimum_duration_ns) {
        if (has_last_result) {
            last_result.deinit();
            has_last_result = false;
        }
        last_result = try Tensor.matmul(&a, &b, parent_allocator);
        has_last_result = true;
        iterations += 1;
        elapsed = timer.read();
    }
    if (!verifyMatmulResult(&a, &b, &last_result, 1.0e-4)) return error.VerificationFailed;
    const operations = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k)) * @as(f64, @floatFromInt(iterations));
    const seconds = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(std.time.ns_per_s));
    const gflops = operations / seconds / 1.0e9;
    const stats = hugePageStats();
    const success_rate = if (stats.attempts == 0) 0.0 else 100.0 * @as(f64, @floatFromInt(stats.successes)) / @as(f64, @floatFromInt(stats.attempts));
    const report = getLastGemmReport();
    const stdout = std.io.getStdOut().writer();
    try stdout.print("M={d} N={d} K={d} iterations={d} seconds={d:.6} GFLOPS={d:.3}\n", .{ m, n, k, iterations, seconds, gflops });
    try stdout.print("verification=passed tolerance=1e-4 samples={d}\n", .{@min(m * n, 4096)});
    try stdout.print("cores_used={d} huge_attempts={d} huge_successes={d} huge_fallbacks={d} huge_success_rate={d:.2}%\n", .{ report.cores_used, stats.attempts, stats.successes, stats.fallbacks, success_rate });
    var worker: usize = 0;
    while (worker < report.cores_used) : (worker += 1) {
        try stdout.print("worker={d} core={d} pinned={}\n", .{ worker, report.core_ids[worker], report.pinned[worker] });
    }
}

pub fn main() !void {
    var arguments = std.process.args();
    _ = arguments.next();
    const m = if (arguments.next()) |value| try std.fmt.parseInt(usize, value, 10) else 4096;
    const n = if (arguments.next()) |value| try std.fmt.parseInt(usize, value, 10) else m;
    const k = if (arguments.next()) |value| try std.fmt.parseInt(usize, value, 10) else m;
    try benchmarkGemm(std.heap.page_allocator, m, n, k, 5 * std.time.ns_per_s);
}

test "Tensor init and basic operations" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.init(allocator, &.{ 2, 3 });
    defer tensor.deinit();
    try tensor.set(&.{ 0, 0 }, 1.0);
    try tensor.set(&.{ 1, 2 }, 6.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), try tensor.get(&.{ 0, 0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), try tensor.get(&.{ 1, 2 }), 1e-6);
}

test "Tensor operations" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.init(allocator, &.{ 2, 2 });
    defer tensor.deinit();
    try tensor.fill(2.0);
    try tensor.addScalar(3.0);
    try tensor.mulScalar(2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), try tensor.get(&.{ 0, 0 }), 1e-6);
}

test "Tensor matmul" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, &.{ 2, 3 });
    defer a.deinit();
    var b = try Tensor.init(allocator, &.{ 3, 2 });
    defer b.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;
    a.data[4] = 5.0;
    a.data[5] = 6.0;
    b.data[0] = 7.0;
    b.data[1] = 8.0;
    b.data[2] = 9.0;
    b.data[3] = 10.0;
    b.data[4] = 11.0;
    b.data[5] = 12.0;
    var c = try Tensor.matmul(&a, &b, allocator);
    defer c.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 58.0), try c.get(&.{ 0, 0 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), try c.get(&.{ 0, 1 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 139.0), try c.get(&.{ 1, 0 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 154.0), try c.get(&.{ 1, 1 }), 1e-5);
}

test "Tensor inverse and det" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.init(allocator, &.{ 2, 2 });
    defer tensor.deinit();
    tensor.data[0] = 4.0;
    tensor.data[1] = 7.0;
    tensor.data[2] = 2.0;
    tensor.data[3] = 6.0;
    const determinant = try tensor.det(allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), determinant, 1e-5);
    var inverse_tensor = try tensor.inverse(allocator);
    defer inverse_tensor.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), try inverse_tensor.get(&.{ 0, 0 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.7), try inverse_tensor.get(&.{ 0, 1 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.2), try inverse_tensor.get(&.{ 1, 0 }), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), try inverse_tensor.get(&.{ 1, 1 }), 1e-5);
}
