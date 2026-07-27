const std = @import("std");
const fs = std.fs;
const mem = std.mem;
const math = std.math;
const builtin = @import("builtin");
const Allocator = mem.Allocator;

pub const IoConfig = struct {
    pub const BUFFER_SIZE: usize = 8192;
    pub const LARGE_CHUNK_SIZE: usize = 65536;
    pub const MAX_READ_BYTES: usize = 100 * 1024 * 1024;
    pub const MAX_FILE_SIZE: usize = 1024 * 1024 * 1024;
    pub const PAGE_SIZE: usize = mem.page_size;
    pub const TRUNCATE_SHIFT: u6 = 32;
    pub const SECURE_FILE_MODE: u9 = 0o600;
    pub const MAX_PATH_LEN: usize = 4096;
};

pub const IoError = error{
    InvalidFileSize,
    FileTooLarge,
    FileIsEmpty,
    BufferNotMapped,
    OutOfBounds,
    MaxBytesExceeded,
    UnexpectedEndOfFile,
    FileNotFound,
    AccessDenied,
    InvalidPath,
    NotADirectory,
    ReadOnlyFile,
    Overflow,
    PathTooLong,
    InvalidBufferSize,
    ResourceReleased,
};

fn generateRuntimeSeed() u64 {
    var entropy_buf: [32]u8 = undefined;
    std.crypto.random.bytes(&entropy_buf);
    var hasher = std.crypto.hash.blake2.Blake2b256.init(.{});
    hasher.update(&entropy_buf);
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    const result = mem.readInt(u64, digest[0..8], .little);
    secureZeroBytes(&digest);
    secureZeroBytes(&entropy_buf);
    return result;
}

fn secureZeroBytes(buf: []u8) void {
    const p: [*]volatile u8 = @ptrCast(buf.ptr);
    var i: usize = 0;
    while (i < buf.len) : (i += 1) {
        p[i] = 0;
    }
}

fn mixHash(h: u64) u64 {
    const prime1: u64 = 0xff51afd7ed558ccd;
    const prime2: u64 = 0xc4ceb9fe1a85ec53;
    var mixed = h ^ (h >> 33);
    mixed *%= prime1;
    mixed ^= mixed >> 29;
    mixed *%= prime2;
    return mixed ^ (mixed >> 32);
}

fn addChecked(a: usize, b: usize) !usize {
    const r = @addWithOverflow(a, b);
    if (r[1] != 0) return IoError.Overflow;
    return r[0];
}

pub const MMAP = struct {
    file: fs.File,
    buffer: ?[]align(mem.page_size) u8,
    allocator: Allocator,
    is_writable: bool,
    actual_size: usize,
    mutex: std.Thread.Mutex,
    released: bool,
    last_read: ?[]u8,

    pub fn open(allocator: Allocator, path: []const u8, mode: fs.File.OpenFlags) !MMAP {
        const file = try fs.cwd().openFile(path, mode);
        errdefer file.close();
        return openFromFile(allocator, file, mode);
    }

    pub fn openWithDir(allocator: Allocator, dir: fs.Dir, path: []const u8, mode: fs.File.OpenFlags) !MMAP {
        const file = try dir.openFile(path, mode);
        errdefer file.close();
        return openFromFile(allocator, file, mode);
    }

    fn openFromFile(allocator: Allocator, file: fs.File, mode: fs.File.OpenFlags) !MMAP {
        const stat = try file.stat();
        const size_u64: u64 = stat.size;
        if (math.maxInt(usize) < math.maxInt(u64)) {
            if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
        }
        if (size_u64 > IoConfig.MAX_FILE_SIZE) return error.FileTooLarge;
        var file_size: usize = @intCast(size_u64);

        const is_writable = mode.mode == .read_write or mode.mode == .write_only;

        if (file_size == 0) {
            if (is_writable) {
                try file.setEndPos(IoConfig.PAGE_SIZE);
                file_size = IoConfig.PAGE_SIZE;
            } else {
                return IoError.FileIsEmpty;
            }
        }

        var prot_flags: u32 = std.posix.PROT.READ;
        if (is_writable) {
            prot_flags |= std.posix.PROT.WRITE;
        }

        const aligned_size = mem.alignForward(usize, file_size, IoConfig.PAGE_SIZE);
        const map_flags: u32 = if (is_writable) std.posix.MAP.SHARED else std.posix.MAP.PRIVATE;

        const buffer = try std.posix.mmap(
            null,
            aligned_size,
            prot_flags,
            map_flags,
            file.handle,
            0
        );

        return .{
            .file = file,
            .buffer = buffer,
            .allocator = allocator,
            .is_writable = is_writable,
            .actual_size = file_size,
            .mutex = .{},
            .released = false,
            .last_read = null,
        };
    }

    pub fn close(self: *MMAP) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.released) return;
        self.released = true;
        if (self.last_read) |lr| {
            self.allocator.free(lr);
            self.last_read = null;
        }
        if (self.buffer) |buf| {
            std.posix.munmap(buf);
            self.buffer = null;
        }
        self.file.close();
    }

    pub fn deinit(self: *MMAP) void {
        self.close();
    }

    pub fn read(self: *MMAP, offset: usize, len: usize) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.released) return IoError.ResourceReleased;
        const buf = self.buffer orelse return IoError.BufferNotMapped;
        if (offset > self.actual_size) return IoError.OutOfBounds;
        const end = try addChecked(offset, len);
        const read_end = @min(end, self.actual_size);
        const actual_len = read_end - offset;
        if (self.last_read) |lr| {
            self.allocator.free(lr);
            self.last_read = null;
        }
        const result = try self.allocator.alloc(u8, actual_len);
        @memcpy(result, buf[offset..read_end]);
        self.last_read = result;
        return result;
    }

    pub fn copyRead(self: *MMAP, allocator: Allocator, offset: usize, len: usize) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.released) return IoError.ResourceReleased;
        const buf = self.buffer orelse return IoError.BufferNotMapped;
        if (offset > self.actual_size) return IoError.OutOfBounds;
        const end = try addChecked(offset, len);
        const read_end = @min(end, self.actual_size);
        const result = try allocator.alloc(u8, read_end - offset);
        @memcpy(result, buf[offset..read_end]);
        return result;
    }

    pub const SyncMode = enum {
        sync,
        nosync,
    };

    pub fn write(self: *MMAP, offset: usize, data: []const u8, sync_mode: SyncMode) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.released) return IoError.ResourceReleased;
        if (!self.is_writable) return IoError.ReadOnlyFile;
        const buf = self.buffer orelse return IoError.BufferNotMapped;
        if (offset > self.actual_size) return IoError.OutOfBounds;
        const end = try addChecked(offset, data.len);
        if (end > self.actual_size) return IoError.OutOfBounds;
        @memcpy(buf[offset..end], data);
        if (sync_mode == .sync) {
            try std.posix.msync(buf, std.posix.MSF.SYNC);
        }
    }

    pub fn append(self: *MMAP, data: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.released) return IoError.ResourceReleased;
        if (!self.is_writable) return IoError.ReadOnlyFile;
        const buf = self.buffer orelse return IoError.BufferNotMapped;

        const current_size = self.actual_size;
        const new_size = try addChecked(current_size, data.len);
        if (new_size > IoConfig.MAX_FILE_SIZE) return error.FileTooLarge;

        try self.file.setEndPos(new_size);
        errdefer self.file.setEndPos(current_size) catch {};

        try self.file.pwriteAll(data, current_size);

        const aligned_size = mem.alignForward(usize, new_size, IoConfig.PAGE_SIZE);

        var prot_flags: u32 = std.posix.PROT.READ;
        if (self.is_writable) prot_flags |= std.posix.PROT.WRITE;
        const map_flags: u32 = if (self.is_writable) std.posix.MAP.SHARED else std.posix.MAP.PRIVATE;

        const new_buf = std.posix.mmap(
            null,
            aligned_size,
            prot_flags,
            map_flags,
            self.file.handle,
            0
        ) catch {
            return IoError.BufferNotMapped;
        };

        std.posix.munmap(buf);
        self.buffer = new_buf;
        self.actual_size = new_size;
    }

    pub fn sync(self: *MMAP) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.released) return IoError.ResourceReleased;
        const buf = self.buffer orelse return IoError.BufferNotMapped;
        try std.posix.msync(buf, std.posix.MSF.SYNC);
    }

    pub fn size(self: *MMAP) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.actual_size;
    }
};

pub const DurableWriter = struct {
    file: fs.File,
    buffer: [IoConfig.BUFFER_SIZE]u8,
    pos: usize,
    enable_sync: bool,
    mutex: std.Thread.Mutex,

    pub fn init(path: []const u8, enable_sync: bool) !DurableWriter {
        const file = try fs.cwd().createFile(path, .{ .truncate = true, .mode = IoConfig.SECURE_FILE_MODE });
        return .{
            .file = file,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .enable_sync = enable_sync,
            .mutex = .{},
        };
    }

    pub fn initWithDir(dir: fs.Dir, path: []const u8, enable_sync: bool) !DurableWriter {
        const file = try dir.createFile(path, .{ .truncate = true, .mode = IoConfig.SECURE_FILE_MODE });
        return .{
            .file = file,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .enable_sync = enable_sync,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *DurableWriter) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.flushInternal() catch {};
        if (self.enable_sync) {
            self.file.sync() catch {};
        }
        self.file.close();
    }

    pub fn write(self: *DurableWriter, data: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        var remaining = data;
        while (remaining.len > 0) {
            const space = self.buffer.len - self.pos;
            if (space == 0) {
                try self.flushInternal();
                continue;
            }
            const to_copy = @min(remaining.len, space);
            @memcpy(self.buffer[self.pos..self.pos + to_copy], remaining[0..to_copy]);
            self.pos += to_copy;
            remaining = remaining[to_copy..];
        }
    }

    pub fn flush(self: *DurableWriter) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.flushInternal();
    }

    fn flushInternal(self: *DurableWriter) !void {
        if (self.pos > 0) {
            try self.file.writeAll(self.buffer[0..self.pos]);
            self.pos = 0;
        }
    }

    pub fn writeAll(self: *DurableWriter, data: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.flushInternal();
        var remaining = data;
        while (remaining.len > 0) {
            const space = self.buffer.len - self.pos;
            if (space == 0) {
                try self.flushInternal();
                continue;
            }
            const to_copy = @min(remaining.len, space);
            @memcpy(self.buffer[self.pos..self.pos + to_copy], remaining[0..to_copy]);
            self.pos += to_copy;
            remaining = remaining[to_copy..];
        }
        try self.flushInternal();
    }
};

pub const BufferedReader = struct {
    file: fs.File,
    buffer: [IoConfig.BUFFER_SIZE]u8,
    pos: usize,
    limit: usize,
    max_read_bytes: usize,
    total_read: usize,
    mutex: std.Thread.Mutex,

    pub fn init(path: []const u8) !BufferedReader {
        return initWithMaxBytes(path, IoConfig.MAX_READ_BYTES);
    }

    pub fn initWithDir(dir: fs.Dir, path: []const u8) !BufferedReader {
        return initWithDirAndMaxBytes(dir, path, IoConfig.MAX_READ_BYTES);
    }

    pub fn initWithDirAndMaxBytes(dir: fs.Dir, path: []const u8, max_bytes: usize) !BufferedReader {
        const file = try dir.openFile(path, .{});
        return .{
            .file = file,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .limit = 0,
            .max_read_bytes = max_bytes,
            .total_read = 0,
            .mutex = .{},
        };
    }

    pub fn initWithMaxBytes(path: []const u8, max_bytes: usize) !BufferedReader {
        const file = try fs.cwd().openFile(path, .{});
        return .{
            .file = file,
            .buffer = mem.zeroes([IoConfig.BUFFER_SIZE]u8),
            .pos = 0,
            .limit = 0,
            .max_read_bytes = max_bytes,
            .total_read = 0,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *BufferedReader) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.file.close();
    }

    fn fillBuffer(self: *BufferedReader) !bool {
        if (self.total_read >= self.max_read_bytes) return IoError.MaxBytesExceeded;
        const remaining_allowed = self.max_read_bytes - self.total_read;
        const to_read = @min(self.buffer.len, remaining_allowed);
        const n = try self.file.read(self.buffer[0..to_read]);
        self.limit = n;
        self.pos = 0;
        self.total_read += n;
        return n > 0;
    }

    pub fn read(self: *BufferedReader, buf: []u8) !usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        var total: usize = 0;
        while (total < buf.len) {
            if (self.pos < self.limit) {
                const avail = @min(self.limit - self.pos, buf.len - total);
                @memcpy(buf[total..total + avail], self.buffer[self.pos..self.pos + avail]);
                self.pos += avail;
                total += avail;
            } else {
                if (!try self.fillBuffer()) break;
            }
        }
        return total;
    }

    pub fn readUntil(self: *BufferedReader, delim: u8, allocator: Allocator) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();

        while (list.items.len < self.max_read_bytes) {
            if (self.pos < self.limit) {
                const chunk = self.buffer[self.pos..self.limit];
                if (mem.indexOfScalar(u8, chunk, delim)) |idx| {
                    try list.appendSlice(chunk[0..idx]);
                    self.pos += idx + 1;
                    return list.toOwnedSlice();
                } else {
                    try list.appendSlice(chunk);
                    self.pos = self.limit;
                }
            } else {
                if (!try self.fillBuffer()) return list.toOwnedSlice();
            }
        }
        return IoError.MaxBytesExceeded;
    }

    pub fn readLine(self: *BufferedReader, allocator: Allocator) !?[]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();

        while (list.items.len < self.max_read_bytes) {
            if (self.pos < self.limit) {
                const chunk = self.buffer[self.pos..self.limit];
                if (mem.indexOfScalar(u8, chunk, '\n')) |idx| {
                    try list.appendSlice(chunk[0..idx]);
                    self.pos += idx + 1;
                    if (list.items.len > 0 and list.items[list.items.len - 1] == '\r') {
                        _ = list.pop();
                    }
                    return list.toOwnedSlice();
                } else {
                    try list.appendSlice(chunk);
                    self.pos = self.limit;
                }
            } else {
                if (!try self.fillBuffer()) {
                    if (list.items.len == 0) return null;
                    if (list.items.len > 0 and list.items[list.items.len - 1] == '\r') {
                        _ = list.pop();
                    }
                    return list.toOwnedSlice();
                }
            }
        }
        return IoError.MaxBytesExceeded;
    }

    pub fn peek(self: *BufferedReader) !?u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.pos < self.limit) return self.buffer[self.pos];
        if (!try self.fillBuffer()) return null;
        if (self.pos < self.limit) return self.buffer[self.pos];
        return null;
    }
};

pub const BufferedWriter = struct {
    file: fs.File,
    buffer: []u8,
    pos: usize,
    allocator: Allocator,
    mutex: std.Thread.Mutex,

    pub fn init(allocator: Allocator, path: []const u8, buffer_size: usize) !BufferedWriter {
        if (buffer_size == 0) return IoError.InvalidBufferSize;
        const file = try fs.cwd().createFile(path, .{ .truncate = true, .mode = IoConfig.SECURE_FILE_MODE });
        errdefer file.close();
        const buffer = try allocator.alloc(u8, buffer_size);
        errdefer allocator.free(buffer);
        return .{
            .file = file,
            .buffer = buffer,
            .pos = 0,
            .allocator = allocator,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *BufferedWriter) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.flushInternal() catch {};
        self.allocator.free(self.buffer);
        self.file.close();
    }

    pub fn writeByte(self: *BufferedWriter, byte: u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.pos >= self.buffer.len) {
            try self.flushInternal();
        }
        self.buffer[self.pos] = byte;
        self.pos += 1;
    }

    pub fn writeBytes(self: *BufferedWriter, data: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        var remaining = data;
        while (remaining.len > 0) {
            if (self.pos >= self.buffer.len) {
                try self.flushInternal();
            }
            const available = self.buffer.len - self.pos;
            const to_write = @min(available, remaining.len);
            @memcpy(self.buffer[self.pos..self.pos + to_write], remaining[0..to_write]);
            self.pos += to_write;
            remaining = remaining[to_write..];
        }
    }

    pub fn flush(self: *BufferedWriter) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.flushInternal();
    }

    fn flushInternal(self: *BufferedWriter) !void {
        if (self.pos > 0) {
            try self.file.writeAll(self.buffer[0..self.pos]);
            self.pos = 0;
        }
    }

    pub fn sync(self: *BufferedWriter) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.flushInternal();
        try self.file.sync();
    }
};

pub fn stableHash(data: []const u8, seed: u64) u64 {
    var hasher = std.hash.Wyhash.init(seed);
    hasher.update(data);
    return mixHash(hasher.final());
}

var g_hash_seed_initialized = std.atomic.Value(bool).init(false);
var g_hash_seed = std.atomic.Value(u64).init(0);
var g_hash_seed_mutex: std.Thread.Mutex = .{};

fn getHashSeed() u64 {
    if (g_hash_seed_initialized.load(.acquire)) {
        return g_hash_seed.load(.acquire);
    }
    g_hash_seed_mutex.lock();
    defer g_hash_seed_mutex.unlock();
    if (!g_hash_seed_initialized.load(.acquire)) {
        g_hash_seed.store(generateRuntimeSeed(), .release);
        g_hash_seed_initialized.store(true, .release);
    }
    return g_hash_seed.load(.acquire);
}

pub fn hash64(data: []const u8) u64 {
    const seed = getHashSeed();
    var hasher = std.hash.Wyhash.init(seed);
    hasher.update(data);
    return mixHash(hasher.final());
}

pub fn hash32(data: []const u8) u32 {
    const h64 = hash64(data);
    const mixed = h64 ^ (h64 >> IoConfig.TRUNCATE_SHIFT);
    return @truncate(mixed);
}

pub fn pathExists(path: []const u8) !bool {
    _ = fs.cwd().statFile(path) catch |err| {
        if (err == error.FileNotFound) return false;
        return err;
    };
    return true;
}

pub fn createDirRecursive(path: []const u8) !void {
    if (path.len == 0) return IoError.InvalidPath;
    try fs.cwd().makePath(path);
}

pub fn readFile(allocator: Allocator, path: []const u8) ![]u8 {
    return readFileLimited(allocator, path, IoConfig.MAX_FILE_SIZE);
}

pub fn readFileWithDir(allocator: Allocator, dir: fs.Dir, path: []const u8) ![]u8 {
    const file = try dir.openFile(path, .{});
    defer file.close();
    return file.readToEndAlloc(allocator, IoConfig.MAX_FILE_SIZE);
}

pub fn readFileLimited(allocator: Allocator, path: []const u8, max_size: usize) ![]u8 {
    const file = try fs.cwd().openFile(path, .{});
    defer file.close();
    return file.readToEndAlloc(allocator, max_size);
}

pub const WriteFileOptions = struct {
    create_backup: bool = false,
    sync_after_write: bool = false,
};

pub fn writeFile(path: []const u8, data: []const u8) !void {
    return writeFileWithOptions(path, data, .{});
}

pub fn writeFileWithOptions(path: []const u8, data: []const u8, options: WriteFileOptions) !void {
    if (options.create_backup) {
        const exists = fs.cwd().statFile(path) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
        if (exists != null) {
            var backup_buf: [IoConfig.MAX_PATH_LEN]u8 = undefined;
            const backup_path = std.fmt.bufPrint(&backup_buf, "{s}.bak", .{path}) catch return IoError.PathTooLong;
            try fs.cwd().copyFile(path, fs.cwd(), backup_path, .{});
        }
    }
    const file = try fs.cwd().createFile(path, .{ .truncate = true, .mode = IoConfig.SECURE_FILE_MODE });
    defer file.close();
    try file.writeAll(data);
    if (options.sync_after_write) try file.sync();
}

pub fn appendFile(path: []const u8, data: []const u8) !void {
    var path_c: [IoConfig.MAX_PATH_LEN]u8 = undefined;
    if (path.len >= IoConfig.MAX_PATH_LEN) return IoError.PathTooLong;
    @memcpy(path_c[0..path.len], path);
    path_c[path.len] = 0;
    const fd = try std.posix.openat(fs.cwd().fd, &path_c, std.posix.O.WRONLY | std.posix.O.APPEND | std.posix.O.CREAT, IoConfig.SECURE_FILE_MODE);
    const file = fs.File{ .handle = fd };
    defer file.close();
    try file.writeAll(data);
}

pub fn deleteFile(path: []const u8) !void {
    try fs.cwd().deleteFile(path);
}

pub const CopyProgress = struct {
    bytes_copied: u64,
    total_bytes: u64,
};

pub fn copyFile(allocator: Allocator, src: []const u8, dst: []const u8) !void {
    return copyFileWithProgress(allocator, src, dst, null);
}

pub fn copyFileWithProgress(
    allocator: Allocator,
    src: []const u8,
    dst: []const u8,
    progress_callback: ?*const fn(CopyProgress) void
) !void {
    const src_file = try fs.cwd().openFile(src, .{});
    defer src_file.close();

    const dst_file = try fs.cwd().createFile(dst, .{ .truncate = true, .mode = IoConfig.SECURE_FILE_MODE });
    var dst_closed = false;
    errdefer {
        if (!dst_closed) {
            dst_file.close();
            dst_closed = true;
        }
        fs.cwd().deleteFile(dst) catch {};
    }
    defer {
        if (!dst_closed) {
            dst_file.close();
        }
    }

    const stat = try src_file.stat();
    const total_size: u64 = stat.size;

    const buffer = try allocator.alloc(u8, IoConfig.LARGE_CHUNK_SIZE);
    defer allocator.free(buffer);

    var bytes_copied: u64 = 0;
    while (true) {
        const n = try src_file.read(buffer);
        if (n == 0) break;
        try dst_file.writeAll(buffer[0..n]);
        bytes_copied += n;
        if (progress_callback) |cb| {
            cb(.{ .bytes_copied = bytes_copied, .total_bytes = total_size });
        }
    }

    try dst_file.sync();
}

pub fn moveFile(allocator: Allocator, old: []const u8, new: []const u8) !void {
    fs.cwd().rename(old, new) catch |err| {
        if (err == error.RenameAcrossMountPoints) {
            try copyFile(allocator, old, new);
            try fs.cwd().deleteFile(old);
            return;
        }
        return err;
    };
}

pub fn listDir(allocator: Allocator, path: []const u8) ![][]u8 {
    var dir = try fs.cwd().openDir(path, .{ .iterate = true });
    defer dir.close();

    var list = std.ArrayList([]u8).init(allocator);
    errdefer {
        for (list.items) |item| allocator.free(item);
        list.deinit();
    }

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        const name = try allocator.dupe(u8, entry.name);
        errdefer allocator.free(name);
        try list.append(name);
    }
    return list.toOwnedSlice();
}

pub fn removeDir(path: []const u8) !void {
    try fs.cwd().deleteDir(path);
}

pub fn getFileSize(path: []const u8) !usize {
    const stat = try fs.cwd().statFile(path);
    const size_u64: u64 = stat.size;
    if (math.maxInt(usize) < math.maxInt(u64)) {
        if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
    }
    return @intCast(size_u64);
}

pub fn isDir(path: []const u8) !bool {
    const stat_result = fs.cwd().statFile(path) catch |err| {
        if (err == error.FileNotFound) return false;
        return err;
    };
    return stat_result.kind == .directory;
}

pub fn isFile(path: []const u8) !bool {
    const stat_result = fs.cwd().statFile(path) catch |err| {
        if (err == error.FileNotFound) return false;
        return err;
    };
    return stat_result.kind == .file;
}

pub inline fn toLittleEndian(T: type, value: T) T {
    comptime {
        const info = @typeInfo(T);
        if (info != .int) @compileError("toLittleEndian requires integer type");
    }
    return switch (comptime builtin.target.cpu.arch.endian()) {
        .little => value,
        .big => @byteSwap(value),
    };
}

pub inline fn fromLittleEndian(T: type, bytes: *const [@sizeOf(T)]u8) T {
    comptime {
        const info = @typeInfo(T);
        if (info != .int) @compileError("fromLittleEndian requires integer type");
    }
    return mem.readInt(T, bytes, .little);
}

pub inline fn toBigEndian(T: type, value: T) T {
    comptime {
        const info = @typeInfo(T);
        if (info != .int) @compileError("toBigEndian requires integer type");
    }
    return switch (comptime builtin.target.cpu.arch.endian()) {
        .little => @byteSwap(value),
        .big => value,
    };
}

pub inline fn fromBigEndian(T: type, bytes: *const [@sizeOf(T)]u8) T {
    comptime {
        const info = @typeInfo(T);
        if (info != .int) @compileError("fromBigEndian requires integer type");
    }
    return mem.readInt(T, bytes, .big);
}

var sequential_write_mutex: std.Thread.Mutex = .{};

pub fn sequentialWrite(allocator: Allocator, path: []const u8, data: []const []const u8) !void {
    sequential_write_mutex.lock();
    defer sequential_write_mutex.unlock();
    var writer = try BufferedWriter.init(allocator, path, IoConfig.LARGE_CHUNK_SIZE);
    defer writer.deinit();

    for (data) |chunk| {
        try writer.writeBytes(chunk);
    }
    try writer.sync();
}

pub fn sequentialRead(allocator: Allocator, path: []const u8, chunk_callback: *const fn([]const u8) anyerror!void) !void {
    const file = try fs.cwd().openFile(path, .{});
    defer file.close();

    const buffer = try allocator.alloc(u8, IoConfig.LARGE_CHUNK_SIZE);
    defer allocator.free(buffer);

    while (true) {
        const n = try file.read(buffer);
        if (n == 0) break;
        try chunk_callback(buffer[0..n]);
    }
}

pub fn atomicWrite(path: []const u8, data: []const u8) !void {
    var temp_buf: [IoConfig.MAX_PATH_LEN + 32]u8 = undefined;
    const temp_path = std.fmt.bufPrint(&temp_buf, "{s}.tmp.{x}", .{path, std.crypto.random.int(u64)}) catch return IoError.PathTooLong;

    const file = try fs.cwd().createFile(temp_path, .{ .mode = IoConfig.SECURE_FILE_MODE });
    var file_closed = false;
    errdefer {
        if (!file_closed) file.close();
        fs.cwd().deleteFile(temp_path) catch {};
    }

    try file.writeAll(data);
    try file.sync();
    file.close();
    file_closed = true;

    try fs.cwd().rename(temp_path, path);
}

pub fn compareFiles(allocator: Allocator, path1: []const u8, path2: []const u8) !bool {
    const file1 = try fs.cwd().openFile(path1, .{});
    defer file1.close();

    const file2 = try fs.cwd().openFile(path2, .{});
    defer file2.close();

    const stat1 = try file1.stat();
    const stat2 = try file2.stat();
    if (stat1.size != stat2.size) return false;

    const buf1 = try allocator.alloc(u8, IoConfig.LARGE_CHUNK_SIZE);
    defer allocator.free(buf1);
    const buf2 = try allocator.alloc(u8, IoConfig.LARGE_CHUNK_SIZE);
    defer allocator.free(buf2);

    while (true) {
        var total1: usize = 0;
        while (total1 < buf1.len) {
            const n = try file1.read(buf1[total1..]);
            if (n == 0) break;
            total1 += n;
        }
        var total2: usize = 0;
        while (total2 < buf2.len) {
            const n = try file2.read(buf2[total2..]);
            if (n == 0) break;
            total2 += n;
        }
        if (total1 != total2) return false;
        if (total1 == 0) break;
        if (!mem.eql(u8, buf1[0..total1], buf2[0..total2])) return false;
    }

    return true;
}

test "MMAP open and close" {
    const gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const file = try tmp_dir.dir.createFile("test_mmap.bin", .{});
    try file.writeAll("test data for mmap");
    file.close();

    var mmap = try MMAP.openWithDir(gpa, tmp_dir.dir, "test_mmap.bin", .{ .mode = .read_only });
    defer mmap.close();

    const content = try mmap.read(0, 9);
    defer gpa.free(content);
    try std.testing.expectEqualStrings("test data", content);
}

test "DurableWriter with sync" {
    const gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    var writer = try DurableWriter.initWithDir(tmp_dir.dir, "test_durable.txt", false);
    try writer.writeAll("hello world");
    writer.deinit();

    const content = try readFileWithDir(gpa, tmp_dir.dir, "test_durable.txt");
    defer gpa.free(content);
    try std.testing.expectEqualStrings("hello world", content);
}

test "BufferedReader zero init" {
    const gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const file = try tmp_dir.dir.createFile("test_buffered.txt", .{});
    try file.writeAll("line1\nline2\nline3");
    file.close();

    var reader = try BufferedReader.initWithDir(tmp_dir.dir, "test_buffered.txt");
    defer reader.deinit();

    const line1 = try reader.readUntil('\n', gpa);
    defer gpa.free(line1);
    try std.testing.expectEqualStrings("line1", line1);

    const line2 = try reader.readUntil('\n', gpa);
    defer gpa.free(line2);
    try std.testing.expectEqualStrings("line2", line2);

    const line3 = try reader.readUntil('\n', gpa);
    defer gpa.free(line3);
    try std.testing.expectEqualStrings("line3", line3);
}

test "BufferedReader readLine strips carriage return" {
    const gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const file = try tmp_dir.dir.createFile("test_crlf.txt", .{});
    try file.writeAll("line1\r\nline2\r\nline3");
    file.close();

    var reader = try BufferedReader.initWithDir(tmp_dir.dir, "test_crlf.txt");
    defer reader.deinit();

    const line1 = (try reader.readLine(gpa)).?;
    defer gpa.free(line1);
    try std.testing.expectEqualStrings("line1", line1);

    const line2 = (try reader.readLine(gpa)).?;
    defer gpa.free(line2);
    try std.testing.expectEqualStrings("line2", line2);

    const line3 = (try reader.readLine(gpa)).?;
    defer gpa.free(line3);
    try std.testing.expectEqualStrings("line3", line3);
}

test "Stable hash mixing" {
    const data = "test";
    const seed: u64 = 12345;
    const hash1 = stableHash(data, seed);
    const hash2 = stableHash(data, seed);

    try std.testing.expectEqual(hash1, hash2);
}

test "Atomic write" {
    const gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    var path_buf: [IoConfig.MAX_PATH_LEN]u8 = undefined;
    const full_path = try tmp_dir.dir.realpath(".", &path_buf);
    const test_path = try std.fmt.allocPrint(gpa, "{s}/test_atomic.txt", .{full_path});
    defer gpa.free(test_path);

    try atomicWrite(test_path, "data");
    defer fs.cwd().deleteFile(test_path) catch {};
    const content = try readFile(gpa, test_path);
    defer gpa.free(content);
    try std.testing.expectEqualStrings("data", content);
}

test "MMAP deinit prevents double close" {
    const gpa = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const file = try tmp_dir.dir.createFile("test_mmap_deinit.bin", .{});
    try file.writeAll("test deinit safety");
    file.close();

    var mmap = try MMAP.openWithDir(gpa, tmp_dir.dir, "test_mmap_deinit.bin", .{ .mode = .read_only });
    mmap.deinit();
    mmap.deinit();
}
