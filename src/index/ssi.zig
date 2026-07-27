const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const types = @import("../core/types.zig");
const Tensor = @import("../core/tensor.zig").Tensor;
const Error = types.Error;

pub const SSI = struct {
    root: ?*Node,
    allocator: Allocator,
    height: usize = 0,
    size: usize = 0,
    max_height: usize = 6,

    const bucket_width: usize = 6;
    const bucket_count: usize = 1 << bucket_width;
    const tensor_width: usize = 134;

    const Segment = struct {
        tokens: []u32,
        position: u64,
        score: f32,
        anchor_hash: u64,

        pub fn init(allocator: Allocator, tokens: []const u32, position: u64, score: f32, anchor_hash: u64) !Segment {
            return .{
                .tokens = try allocator.dupe(u32, tokens),
                .position = position,
                .score = score,
                .anchor_hash = anchor_hash,
            };
        }

        pub fn deinit(self: *Segment, allocator: Allocator) void {
            allocator.free(self.tokens);
            self.tokens = &.{};
        }

        pub fn tokenHash(self: *const Segment) u64 {
            return hashTokens(self.tokens);
        }

        pub fn fullHash(self: *const Segment) u64 {
            var state: u64 = 0;
            state = mixHash(state, self.position);
            state = mixHash(state, @as(u64, scoreBits(self.score)));
            state = mixHash(state, self.anchor_hash);
            state = mixHash(state, @as(u64, @intCast(self.tokens.len)));
            for (self.tokens) |tok| {
                state = mixHash(state, tok);
            }
            return state;
        }
    };

    const CollisionNode = struct {
        seg: Segment,
        next: ?*CollisionNode,
    };

    const Node = struct {
        hash: u64,
        children: ?[]?*Node,
        segment: ?Segment,
        collision_chain: ?*CollisionNode,
        height: usize,
        is_leaf: bool,

        pub fn init(allocator: Allocator, height: usize) !Node {
            var children: ?[]?*Node = null;
            if (height > 0) {
                const allocated = try allocator.alloc(?*Node, bucket_count);
                @memset(allocated, null);
                children = allocated;
            }
            return .{
                .hash = 0,
                .children = children,
                .segment = null,
                .collision_chain = null,
                .height = height,
                .is_leaf = height == 0,
            };
        }

        pub fn deinit(self: *Node, allocator: Allocator) void {
            if (self.segment) |*seg| {
                seg.deinit(allocator);
                self.segment = null;
            }
            var chain = self.collision_chain;
            while (chain) |c| {
                const next = c.next;
                c.seg.deinit(allocator);
                allocator.destroy(c);
                chain = next;
            }
            self.collision_chain = null;
            if (self.children) |children| {
                allocator.free(children);
                self.children = null;
            }
        }
    };

    pub fn init(allocator: Allocator) SSI {
        return .{
            .root = null,
            .allocator = allocator,
            .height = 0,
            .size = 0,
            .max_height = bucket_width,
        };
    }

    fn mixHash(state: u64, value: u64) u64 {
        return state *% 0x9E3779B185EBCA87 +% value +% 0x517CC1B727220A95;
    }

    fn scoreBits(value: f32) u32 {
        return @as(u32, @bitCast(value));
    }

    fn hashTokens(tokens: []const u32) u64 {
        var state: u64 = 0;
        state = mixHash(state, @as(u64, @intCast(tokens.len)));
        for (tokens) |tok| {
            state = mixHash(state, tok);
        }
        return state;
    }

    fn computeAnchorHash(tokens: []const u32, position: u64) u64 {
        var state: u64 = position;
        state = mixHash(state, @as(u64, @intCast(tokens.len)));
        for (tokens) |tok| {
            state = mixHash(state, tok);
        }
        return state;
    }

    fn bucketIndex(position: u64) usize {
        var h = position *% 0x9E3779B185EBCA87;
        h = (h ^ (h >> 30)) *% 0xbf58476d1ce4e5b9;
        h = (h ^ (h >> 27)) *% 0x94d049bb133111eb;
        h = h ^ (h >> 31);
        return @as(usize, @intCast(h & (bucket_count - 1)));
    }

    fn low32(value: u64) u32 {
        return @as(u32, @intCast(value & 0xFFFF_FFFF));
    }

    fn high32(value: u64) u32 {
        return @as(u32, @intCast(value >> 32));
    }

    fn joinU64(lo: u32, hi: u32) u64 {
        return (@as(u64, hi) << 32) | @as(u64, lo);
    }

    fn bitsToFloat(bits: u32) f32 {
        return @as(f32, @bitCast(bits));
    }

    fn floatToBits(value: f32) u32 {
        return @as(u32, @bitCast(value));
    }

    fn safeFloat(v: f32) f32 {
        if (std.math.isNan(v) or std.math.isInf(v)) return 0.0;
        return std.math.clamp(v, -3.4e38, 3.4e38);
    }

    fn recursiveDeinit(node: *Node, allocator: Allocator) void {
        if (node.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |child| {
                    recursiveDeinit(child, allocator);
                }
            }
        }
        node.deinit(allocator);
        allocator.destroy(node);
    }

    pub fn deinit(self: *SSI) void {
        if (self.root) |root| {
            recursiveDeinit(root, self.allocator);
        }
        self.root = null;
        self.height = 0;
        self.size = 0;
    }

    fn computeLeafHash(node: *const Node) u64 {
        var acc: u64 = 0;
        if (node.segment) |seg| {
            acc +%= seg.fullHash();
        }
        var chain = node.collision_chain;
        while (chain) |c| {
            acc +%= c.seg.fullHash();
            chain = c.next;
        }
        return acc;
    }

    fn computeBranchHash(node: *const Node) u64 {
        var acc: u64 = 0;
        if (node.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |child| {
                    acc +%= child.hash;
                }
            }
        }
        return acc;
    }

    fn refreshHash(node: *Node) void {
        node.hash = if (node.is_leaf) computeLeafHash(node) else computeBranchHash(node);
    }

    fn ensureRoot(self: *SSI) !*Node {
        if (self.root == null) {
            const root = try self.allocator.create(Node);
            root.* = try Node.init(self.allocator, bucket_width);
            root.is_leaf = false;
            root.height = bucket_width;
            refreshHash(root);
            self.root = root;
            self.height = bucket_width;
        }
        return self.root.?;
    }

    fn insertIntoLeaf(self: *SSI, leaf: *Node, tokens: []const u32, position: u64, score: f32, anchor_hash: u64) !bool {
        if (!leaf.is_leaf or leaf.height != 0) {
            return error.InvalidNodeState;
        }
        if (leaf.segment == null) {
            leaf.segment = try Segment.init(self.allocator, tokens, position, score, anchor_hash);
            refreshHash(leaf);
            return true;
        }
        if (leaf.segment.?.position == position) {
            var old = leaf.segment.?;
            old.deinit(self.allocator);
            leaf.segment = try Segment.init(self.allocator, tokens, position, score, anchor_hash);
            refreshHash(leaf);
            return false;
        }
        var chain = leaf.collision_chain;
        while (chain) |c| {
            if (c.seg.position == position) {
                c.seg.deinit(self.allocator);
                c.seg = try Segment.init(self.allocator, tokens, position, score, anchor_hash);
                refreshHash(leaf);
                return false;
            }
            chain = c.next;
        }
        const collision = try self.allocator.create(CollisionNode);
        collision.* = .{
            .seg = try Segment.init(self.allocator, tokens, position, score, anchor_hash),
            .next = leaf.collision_chain,
        };
        leaf.collision_chain = collision;
        refreshHash(leaf);
        return true;
    }

    fn addSequenceWithMetadata(self: *SSI, tokens: []const u32, position: u64, score: f32, anchor_hash: u64) !void {
        const root = try self.ensureRoot();
        const idx = bucketIndex(position);
        if (root.children.?[idx] == null) {
            const leaf = try self.allocator.create(Node);
            leaf.* = try Node.init(self.allocator, 0);
            root.children.?[idx] = leaf;
        }
        const leaf = root.children.?[idx].?;
        const inserted_new = try self.insertIntoLeaf(leaf, tokens, position, score, anchor_hash);
        refreshHash(root);
        if (inserted_new) {
            self.size += 1;
        }
    }

    fn copyInto(self: *const SSI, target: *SSI) !void {
        if (self.root == null) {
            return;
        }
        const root = self.root.?;
        if (root.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |leaf| {
                    if (leaf.segment) |seg| {
                        try target.addSequenceWithMetadata(seg.tokens, seg.position, seg.score, seg.anchor_hash);
                    }
                    var chain = leaf.collision_chain;
                    while (chain) |c| {
                        try target.addSequenceWithMetadata(c.seg.tokens, c.seg.position, c.seg.score, c.seg.anchor_hash);
                        chain = c.next;
                    }
                }
            }
        }
    }

    pub fn addSequence(self: *SSI, tokens: []const u32, position: u64, is_anchor: bool) !void {
        const anchor_hash = if (is_anchor) computeAnchorHash(tokens, position) else 0;
        try self.addSequenceWithMetadata(tokens, position, 0.0, anchor_hash);
        if (self.size > 0) {
            const load_factor = @as(f64, @floatFromInt(self.size)) / @as(f64, @floatFromInt(bucket_count));
            if (load_factor > 8.0) {
                try self.compact();
            }
        }
    }

    pub fn retrieveTopK(self: *const SSI, query_tokens: []const u32, k: usize, allocator: Allocator) ![]types.RankedSegment {
        if (k == 0) {
            return allocator.alloc(types.RankedSegment, 0);
        }
        var heap = std.PriorityQueue(types.RankedSegment, void, struct {
            pub fn lessThan(_: void, a: types.RankedSegment, b: types.RankedSegment) std.math.Order {
                return std.math.order(a.score, b.score);
            }
        }.lessThan).init(allocator, {});
        errdefer {
            while (heap.removeOrNull()) |item| {
                var mut = item;
                mut.deinit(allocator);
            }
            heap.deinit();
        }
        defer heap.deinit();
        const query_hash = hashTokens(query_tokens);
        try self.traverse(self.root, query_hash, &heap, k, allocator);
        const result_len = @min(k, heap.count());
        var top_n = try allocator.alloc(types.RankedSegment, result_len);
        var index = result_len;
        while (heap.removeOrNull()) |item| {
            index -= 1;
            top_n[index] = item;
        }
        return top_n;
    }

    fn traverse(self: *const SSI, node: ?*Node, query_hash: u64, heap: anytype, k: usize, allocator: Allocator) !void {
        if (node == null) {
            return;
        }
        const current = node.?;
        if (current.is_leaf) {
            if (current.segment) |seg| {
                try addSegmentToHeap(seg, query_hash, heap, k, allocator);
            }
            var chain = current.collision_chain;
            while (chain) |c| {
                try addSegmentToHeap(c.seg, query_hash, heap, k, allocator);
                chain = c.next;
            }
            return;
        }
        if (current.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |child| {
                    try traverse(self, child, query_hash, heap, k, allocator);
                }
            }
        }
    }

    fn addSegmentToHeap(seg: Segment, query_hash: u64, heap: anytype, k: usize, allocator: Allocator) !void {
        const similarity = computeSimilarity(query_hash, seg.tokenHash());
        if (heap.count() >= k) {
            if (heap.peek()) |top| {
                if (similarity <= top.score) {
                    return;
                }
            }
        }
        const ranked = types.RankedSegment{
            .tokens = try allocator.dupe(u32, seg.tokens),
            .score = similarity,
            .position = seg.position,
            .anchor = seg.anchor_hash != 0,
        };
        errdefer allocator.free(ranked.tokens);
        if (heap.count() < k) {
            try heap.add(ranked);
            return;
        }
        try heap.add(ranked);
        var removed = heap.remove();
        removed.deinit(allocator);
    }

    fn computeSimilarity(h1: u64, h2: u64) f32 {
        const pc1 = @popCount(h1);
        const pc2 = @popCount(h2);
        if (pc1 == 0 and pc2 == 0) return 1.0;
        if (pc1 == 0 or pc2 == 0) return 0.0;
        const intersection = @popCount(h1 & h2);
        const denom = @sqrt(@as(f32, @floatFromInt(pc1)) * @as(f32, @floatFromInt(pc2)));
        return @as(f32, @floatFromInt(intersection)) / denom;
    }

    pub fn compact(self: *SSI) !void {
        if (self.size < 1000) {
            return;
        }
        var rebuilt = SSI.init(self.allocator);
        rebuilt.max_height = self.max_height;
        errdefer rebuilt.deinit();
        try self.copyInto(&rebuilt);
        self.deinit();
        self.* = rebuilt;
    }

    pub fn updateScore(self: *SSI, position: u64, new_score: f32) !void {
        const root = self.root orelse return Error.OutOfBounds;
        const child = root.children.?[bucketIndex(position)] orelse return Error.OutOfBounds;
        if (child.segment) |*seg| {
            if (seg.position == position) {
                seg.score = new_score;
                refreshHash(child);
                refreshHash(root);
                return;
            }
        }
        var chain = child.collision_chain;
        while (chain) |c| {
            if (c.seg.position == position) {
                c.seg.score = new_score;
                refreshHash(child);
                refreshHash(root);
                return;
            }
            chain = c.next;
        }
        return Error.OutOfBounds;
    }

    pub fn getSegment(self: *const SSI, position: u64) ?Segment {
        const root = self.root orelse return null;
        const child = root.children.?[bucketIndex(position)] orelse return null;
        if (child.segment) |seg| {
            if (seg.position == position) {
                return seg;
            }
        }
        var chain = child.collision_chain;
        while (chain) |c| {
            if (c.seg.position == position) {
                return c.seg;
            }
            chain = c.next;
        }
        return null;
    }

    fn countSegments(self: *const SSI) usize {
        const root = self.root orelse return 0;
        var count: usize = 0;
        if (root.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |leaf| {
                    if (leaf.segment != null) {
                        count += 1;
                    }
                    var chain = leaf.collision_chain;
                    while (chain) |c| {
                        count += 1;
                        chain = c.next;
                    }
                }
            }
        }
        return count;
    }

    fn writeBoolFlag(writer: anytype, value: bool) !void {
        try writer.writeInt(u8, if (value) 1 else 0, .little);
    }

    fn readBoolFlag(reader: anytype) !bool {
        return (try reader.readInt(u8, .little)) != 0;
    }

    fn writeSegment(writer: anytype, seg: Segment) !void {
        try writer.writeInt(u64, seg.position, .little);
        try writer.writeInt(u32, floatToBits(seg.score), .little);
        try writer.writeInt(u64, seg.anchor_hash, .little);
        try writer.writeInt(u64, @as(u64, seg.tokens.len), .little);
        for (seg.tokens) |tok| {
            try writer.writeInt(u32, tok, .little);
        }
    }

    fn readSegment(allocator: Allocator, reader: anytype) !Segment {
        const position = try reader.readInt(u64, .little);
        const score = bitsToFloat(try reader.readInt(u32, .little));
        const anchor_hash = try reader.readInt(u64, .little);
        const token_len_raw = try reader.readInt(u64, .little);
        if (token_len_raw > std.math.maxInt(usize)) return error.InvalidData;
        const token_len: usize = @intCast(token_len_raw);
        const tokens = try allocator.alloc(u32, token_len);
        errdefer allocator.free(tokens);
        for (tokens) |*tok| {
            tok.* = try reader.readInt(u32, .little);
        }
        return .{
            .tokens = tokens,
            .position = position,
            .score = score,
            .anchor_hash = anchor_hash,
        };
    }

    fn serializeNode(node: *const Node, writer: anytype) !void {
        try writeBoolFlag(writer, node.is_leaf);
        try writer.writeInt(u64, @as(u64, node.height), .little);
        try writer.writeInt(u64, node.hash, .little);
        if (node.is_leaf) {
            try writeBoolFlag(writer, node.segment != null);
            if (node.segment) |seg| {
                try writeSegment(writer, seg);
            }
            var chain_len: usize = 0;
            var chain = node.collision_chain;
            while (chain) |c| {
                chain_len += 1;
                chain = c.next;
            }
            try writer.writeInt(u64, @as(u64, chain_len), .little);
            chain = node.collision_chain;
            while (chain) |c| {
                try writeSegment(writer, c.seg);
                chain = c.next;
            }
            return;
        }
        const children = node.children orelse return error.InvalidNodeState;
        try writer.writeInt(u64, @as(u64, children.len), .little);
        for (children) |maybe_child| {
            try writeBoolFlag(writer, maybe_child != null);
            if (maybe_child) |child| {
                try serializeNode(child, writer);
            }
        }
    }

    fn deserializeNode(allocator: Allocator, reader: anytype) !*Node {
        const is_leaf = try readBoolFlag(reader);
        const height_raw = try reader.readInt(u64, .little);
        if (height_raw > std.math.maxInt(usize)) return error.InvalidData;
        const height: usize = @intCast(height_raw);
        const stored_hash = try reader.readInt(u64, .little);
        const node = try allocator.create(Node);
        var cleanup = true;
        errdefer {
            if (cleanup) {
                recursiveDeinit(node, allocator);
            }
        }
        node.* = try Node.init(allocator, if (is_leaf) 0 else height);
        if (node.is_leaf != is_leaf) {
            return error.InvalidData;
        }
        if (is_leaf) {
            const has_segment = try readBoolFlag(reader);
            if (has_segment) {
                node.segment = try readSegment(allocator, reader);
            }
            const chain_len_raw = try reader.readInt(u64, .little);
            if (chain_len_raw > std.math.maxInt(usize)) return error.InvalidData;
            const chain_len: usize = @intCast(chain_len_raw);
            var head: ?*CollisionNode = null;
            var tail: ?*CollisionNode = null;
            var index: usize = 0;
            while (index < chain_len) : (index += 1) {
                const collision = try allocator.create(CollisionNode);
                collision.* = .{
                    .seg = try readSegment(allocator, reader),
                    .next = null,
                };
                if (head == null) {
                    head = collision;
                    tail = collision;
                } else {
                    tail.?.next = collision;
                    tail = collision;
                }
            }
            node.collision_chain = head;
        } else {
            const children_len_raw = try reader.readInt(u64, .little);
            if (children_len_raw > std.math.maxInt(usize)) return error.InvalidData;
            const children_len: usize = @intCast(children_len_raw);
            if (children_len != bucket_count) {
                return error.InvalidData;
            }
            for (0..children_len) |i| {
                const has_child = try readBoolFlag(reader);
                if (has_child) {
                    node.children.?[i] = try deserializeNode(allocator, reader);
                }
            }
        }
        refreshHash(node);
        if (node.hash != stored_hash) {
            return error.InvalidData;
        }
        cleanup = false;
        return node;
    }

    pub fn serialize(self: *SSI, writer: anytype) !void {
        try writer.writeInt(u64, @as(u64, self.max_height), .little);
        try writer.writeInt(u64, @as(u64, self.height), .little);
        try writer.writeInt(u64, @as(u64, self.size), .little);
        try writeBoolFlag(writer, self.root != null);
        if (self.root) |root| {
            try serializeNode(root, writer);
        }
    }

    pub fn deserialize(allocator: Allocator, reader: anytype) !SSI {
        var ssi = SSI.init(allocator);
        const max_height_raw = try reader.readInt(u64, .little);
        const height_raw = try reader.readInt(u64, .little);
        const size_raw = try reader.readInt(u64, .little);
        if (max_height_raw > std.math.maxInt(usize) or height_raw > std.math.maxInt(usize) or size_raw > std.math.maxInt(usize)) {
            return error.InvalidData;
        }
        ssi.max_height = @intCast(max_height_raw);
        ssi.height = @intCast(height_raw);
        ssi.size = @intCast(size_raw);
        const has_root = try readBoolFlag(reader);
        if (has_root) {
            ssi.root = try deserializeNode(allocator, reader);
        }
        if (ssi.countSegments() != ssi.size) {
            ssi.deinit();
            return error.InvalidData;
        }
        return ssi;
    }

    pub fn exportToTensor(self: *SSI, allocator: Allocator) !Tensor {
        const segment_count = self.countSegments();
        const rows = if (segment_count == 0) 1 else segment_count;
        var tensor = try Tensor.init(allocator, &.{ rows, tensor_width });
        @memset(tensor.data, 0);
        const root = self.root orelse return tensor;
        var row: usize = 0;
        if (root.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |leaf| {
                    if (leaf.segment) |seg| {
                        encodeSegmentRow(&tensor, row, seg);
                        row += 1;
                    }
                    var chain = leaf.collision_chain;
                    while (chain) |c| {
                        encodeSegmentRow(&tensor, row, c.seg);
                        row += 1;
                        chain = c.next;
                    }
                }
            }
        }
        return tensor;
    }

    fn encodeSegmentRow(tensor: *Tensor, row: usize, seg: Segment) void {
        const offset = row * tensor_width;
        tensor.data[offset + 0] = safeFloat(@as(f32, @floatFromInt(seg.tokens.len)));
        tensor.data[offset + 1] = safeFloat(bitsToFloat(low32(seg.position)));
        tensor.data[offset + 2] = safeFloat(bitsToFloat(high32(seg.position)));
        tensor.data[offset + 3] = safeFloat(seg.score);
        tensor.data[offset + 4] = safeFloat(bitsToFloat(low32(seg.anchor_hash)));
        tensor.data[offset + 5] = safeFloat(bitsToFloat(high32(seg.anchor_hash)));
        var i: usize = 0;
        while (i < seg.tokens.len and i < 128) : (i += 1) {
            tensor.data[offset + 6 + i] = safeFloat(std.math.clamp(bitsToFloat(seg.tokens[i]), -3.4e38, 3.4e38));
        }
    }

    pub fn importFromTensor(self: *SSI, tensor: *const Tensor) !void {
        self.deinit();
        if (tensor.shape.dims.len < 2) {
            return;
        }
        if (tensor.shape.dims[1] < tensor_width) {
            return error.InvalidData;
        }
        const rows = tensor.shape.dims[0];
        var tokens_buffer: [128]u32 = undefined;
        var row: usize = 0;
        while (row < rows) : (row += 1) {
            const offset = row * tensor_width;
            if (offset + tensor_width > tensor.data.len) {
                break;
            }
            const token_len_float = tensor.data[offset + 0];
            if (!(token_len_float >= 0) or std.math.isInf(token_len_float)) {
                continue;
            }
            const token_len_raw: usize = @intFromFloat(token_len_float);
            const token_len = @min(token_len_raw, 128);
            const position = joinU64(floatToBits(tensor.data[offset + 1]), floatToBits(tensor.data[offset + 2]));
            const score = tensor.data[offset + 3];
            const anchor_hash = joinU64(floatToBits(tensor.data[offset + 4]), floatToBits(tensor.data[offset + 5]));
            var i: usize = 0;
            while (i < token_len) : (i += 1) {
                tokens_buffer[i] = floatToBits(tensor.data[offset + 6 + i]);
            }
            try self.addSequenceWithMetadata(tokens_buffer[0..token_len], position, score, anchor_hash);
        }
    }

    pub fn merge(self: *SSI, other: *const SSI) !void {
        try other.copyInto(self);
    }

    pub fn split(self: *SSI, threshold: f32) !SSI {
        var result = SSI.init(self.allocator);
        result.max_height = self.max_height;
        if (self.root == null) {
            return result;
        }
        const root = self.root.?;
        if (root.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |leaf| {
                    if (leaf.segment) |seg| {
                        if (seg.score > threshold) {
                            try result.addSequenceWithMetadata(seg.tokens, seg.position, seg.score, seg.anchor_hash);
                        }
                    }
                    var chain = leaf.collision_chain;
                    while (chain) |c| {
                        if (c.seg.score > threshold) {
                            try result.addSequenceWithMetadata(c.seg.tokens, c.seg.position, c.seg.score, c.seg.anchor_hash);
                        }
                        chain = c.next;
                    }
                }
            }
        }
        return result;
    }

    pub fn balance(self: *SSI) void {
        if (self.root == null) {
            return;
        }
        var rebuilt = SSI.init(self.allocator);
        rebuilt.max_height = self.max_height;
        self.copyInto(&rebuilt) catch {
            rebuilt.deinit();
            return;
        };
        if (rebuilt.countSegments() != self.size) {
            rebuilt.deinit();
            return;
        }
        const old_root = self.root;
        const old_height = self.height;
        const old_size = self.size;
        self.root = rebuilt.root;
        self.height = rebuilt.height;
        self.size = rebuilt.size;
        rebuilt.root = old_root;
        rebuilt.height = old_height;
        rebuilt.size = old_size;
        rebuilt.deinit();
    }

    pub fn stats(self: *const SSI) struct { nodes: usize, leaves: usize, depth: usize } {
        var nodes: usize = 0;
        var leaves: usize = 0;
        var depth: usize = 0;
        const root = self.root orelse return .{ .nodes = 0, .leaves = 0, .depth = 0 };
        var stack = std.ArrayList(struct { node: *const Node, d: usize }).init(self.allocator);
        defer stack.deinit();
        stack.append(.{ .node = root, .d = 0 }) catch return .{ .nodes = nodes, .leaves = leaves, .depth = depth };
        while (stack.pop()) |entry| {
            nodes += 1;
            if (entry.node.is_leaf) {
                leaves += 1;
            }
            if (entry.d > depth) {
                depth = entry.d;
            }
            if (entry.node.children) |children| {
                for (children) |maybe_child| {
                    if (maybe_child) |child| {
                        stack.append(.{ .node = child, .d = entry.d + 1 }) catch {};
                    }
                }
            }
        }
        return .{ .nodes = nodes, .leaves = leaves, .depth = depth };
    }

    fn validateLeaf(node: *const Node) bool {
        if (!node.is_leaf) {
            return false;
        }
        if (node.height != 0) {
            return false;
        }
        if (node.children != null) {
            return false;
        }
        if (node.segment == null and node.collision_chain == null) {
            return true;
        }
        return computeLeafHash(node) == node.hash;
    }

    fn validateNode(node: *const Node, position_set: anytype) !bool {
        if (node.is_leaf) {
            if (!validateLeaf(node)) {
                return false;
            }
            if (node.segment) |seg| {
                if (position_set.contains(seg.position)) return false;
                try position_set.put(seg.position, {});
            }
            var chain = node.collision_chain;
            while (chain) |c| {
                if (position_set.contains(c.seg.position)) return false;
                try position_set.put(c.seg.position, {});
                chain = c.next;
            }
            return true;
        }
        if (node.height != bucket_width) {
            return false;
        }
        const children = node.children orelse return false;
        if (children.len != bucket_count) {
            return false;
        }
        var acc: u64 = 0;
        for (children) |maybe_child| {
            if (maybe_child) |child| {
                if (!(try validateNode(child, position_set))) {
                    return false;
                }
                acc +%= child.hash;
            }
        }
        return acc == node.hash;
    }

    pub fn validate(self: *SSI) bool {
        const root = self.root orelse return self.size == 0;
        if (self.height != bucket_width) {
            return false;
        }
        var position_set = std.AutoHashMap(u64, void).init(self.allocator);
        defer position_set.deinit();
        const valid = validateNode(root, &position_set) catch return false;
        if (!valid) return false;
        const counted = position_set.count();
        if (counted != self.size) {
            return false;
        }
        return true;
    }
};
