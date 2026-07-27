const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;

pub const LearnedEmbedding = struct {
    weight: Tensor,
    grad: Tensor,
    velocity: Tensor,
    vocab_size: usize,
    dim: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, v_size: usize, d: usize, seed: u64) !LearnedEmbedding {
        var w = try Tensor.init(allocator, &.{ v_size, d });
        errdefer w.deinit();
        var g = try Tensor.init(allocator, &.{ v_size, d });
        errdefer g.deinit();
        var v = try Tensor.init(allocator, &.{ v_size, d });
        errdefer v.deinit();
        @memset(g.data, 0.0);
        @memset(v.data, 0.0);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        var i: usize = 0;
        while (i < w.data.len) : (i += 1) {
            w.data[i] = (random.float(f32) - 0.5) * 0.02;
        }
        return LearnedEmbedding{
            .weight = w,
            .grad = g,
            .velocity = v,
            .vocab_size = v_size,
            .dim = d,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LearnedEmbedding) void {
        self.weight.deinit();
        self.grad.deinit();
        self.velocity.deinit();
    }

    pub fn forward(self: *LearnedEmbedding, allocator: Allocator, tokens: []const u32, max_seq_len: usize) !Tensor {
        const seq_len = @min(tokens.len, if (max_seq_len > 0) max_seq_len else tokens.len);
        if (seq_len == 0) return error.EmptyTokens;
        var out = try Tensor.init(allocator, &.{ seq_len, self.dim });
        @memset(out.data, 0.0);
        var r: usize = 0;
        while (r < seq_len) : (r += 1) {
            const t = @min(@as(usize, tokens[r]), self.vocab_size - 1);
            var c: usize = 0;
            while (c < self.dim) : (c += 1) {
                const w_idx = t * self.dim + c;
                const out_idx = r * self.dim + c;
                if (w_idx < self.weight.data.len and out_idx < out.data.len) {
                    out.data[out_idx] = self.weight.data[w_idx];
                }
            }
        }
        return out;
    }

    pub fn backward(self: *LearnedEmbedding, tokens: []const u32, out_grad: []const f32, max_seq_len: usize) void {
        const seq_len = @min(tokens.len, if (max_seq_len > 0) max_seq_len else tokens.len);
        if (seq_len == 0) return;
        var r: usize = 0;
        while (r < seq_len) : (r += 1) {
            const t = @min(@as(usize, tokens[r]), self.vocab_size - 1);
            var c: usize = 0;
            while (c < self.dim) : (c += 1) {
                const g_idx = t * self.dim + c;
                const grad_idx = r * self.dim + c;
                if (g_idx < self.grad.data.len and grad_idx < out_grad.len) {
                    self.grad.data[g_idx] += out_grad[grad_idx];
                }
            }
        }
    }

    pub fn zeroGrad(self: *LearnedEmbedding) void {
        @memset(self.grad.data, 0.0);
    }

    pub fn applyGradients(self: *LearnedEmbedding, lr: f32, momentum: f32) void {
        var i: usize = 0;
        while (i < self.weight.data.len) : (i += 1) {
            self.velocity.data[i] = momentum * self.velocity.data[i] + self.grad.data[i];
            self.weight.data[i] -= lr * self.velocity.data[i];
        }
    }

    pub fn paramCount(self: *const LearnedEmbedding) usize {
        return self.vocab_size * self.dim;
    }

    pub fn flattenParams(self: *const LearnedEmbedding, dst: []f32) void {
        const count = @min(dst.len, self.weight.data.len);
        @memcpy(dst[0..count], self.weight.data[0..count]);
    }

    pub fn flattenGrads(self: *const LearnedEmbedding, dst: []f32) void {
        const count = @min(dst.len, self.grad.data.len);
        @memcpy(dst[0..count], self.grad.data[0..count]);
    }

    pub fn scatterParams(self: *LearnedEmbedding, src: []const f32) void {
        const count = @min(src.len, self.weight.data.len);
        @memcpy(self.weight.data[0..count], src[0..count]);
    }

    pub fn save(self: *const LearnedEmbedding, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        var buf_writer = std.io.bufferedWriter(file.writer());
        const writer = buf_writer.writer();
        try writer.writeInt(u32, 0x4A454D42, .little);
        try writer.writeInt(u32, 1, .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.vocab_size)), .little);
        try writer.writeInt(u64, @as(u64, @intCast(self.dim)), .little);
        for (self.weight.data) |w| {
            try writer.writeInt(u32, @as(u32, @bitCast(w)), .little);
        }
        try buf_writer.flush();
    }

    pub fn load(allocator: Allocator, path: []const u8) !LearnedEmbedding {
        const file = std.fs.cwd().openFile(path, .{}) catch return error.FileNotFound;
        defer file.close();
        var buf_reader = std.io.bufferedReader(file.reader());
        const reader = buf_reader.reader();
        const magic = try reader.readInt(u32, .little);
        if (magic != 0x4A454D42) return error.InvalidFormat;
        const version = try reader.readInt(u32, .little);
        if (version != 1) return error.IncompatibleVersion;
        const v_size = @as(usize, @intCast(try reader.readInt(u64, .little)));
        const d = @as(usize, @intCast(try reader.readInt(u64, .little)));
        var w = try Tensor.init(allocator, &.{ v_size, d });
        errdefer w.deinit();
        var g = try Tensor.init(allocator, &.{ v_size, d });
        errdefer g.deinit();
        var v = try Tensor.init(allocator, &.{ v_size, d });
        errdefer v.deinit();
        @memset(g.data, 0.0);
        @memset(v.data, 0.0);
        var i: usize = 0;
        while (i < w.data.len) : (i += 1) {
            w.data[i] = @bitCast(try reader.readInt(u32, .little));
        }
        return LearnedEmbedding{
            .weight = w,
            .grad = g,
            .velocity = v,
            .vocab_size = v_size,
            .dim = d,
            .allocator = allocator,
        };
    }
};

test "LearnedEmbedding weights update with non-zero gradient" {
    const allocator = std.testing.allocator;

    var emb = try LearnedEmbedding.init(allocator, 100, 16, 42);
    defer emb.deinit();

    const initial_norm: f32 = blk: {
        var sum: f32 = 0.0;
        for (emb.weight.data) |w| sum += w * w;
        break :blk @sqrt(sum);
    };

    const tokens = [_]u32{ 5, 10, 15 };
    var out = try emb.forward(allocator, &tokens, 8);
    defer out.deinit();

    const grad_len = emb.dim * tokens.len;
    const grad_data = try allocator.alloc(f32, grad_len);
    defer allocator.free(grad_data);
    var gi: usize = 0;
    while (gi < grad_len) : (gi += 1) {
        grad_data[gi] = 0.1 * @as(f32, @floatFromInt(gi + 1));
    }

    emb.zeroGrad();
    emb.backward(&tokens, grad_data, 8);
    emb.applyGradients(0.01, 0.9);

    const post_norm: f32 = blk: {
        var sum: f32 = 0.0;
        for (emb.weight.data) |w| sum += w * w;
        break :blk @sqrt(sum);
    };

    try std.testing.expect(@abs(initial_norm - post_norm) > 1e-6);
}
