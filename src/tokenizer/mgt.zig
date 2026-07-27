const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const testing = std.testing;
const core_tensor = @import("../core/tensor.zig");
const core_memory = @import("../core/memory.zig");

pub const MGT = struct {
    token_to_id: std.StringHashMap(u32),
    id_to_token: std.AutoHashMap(u32, []const u8),
    prefixes: std.StringHashMap(u32),
    suffixes: std.StringHashMap(u32),
    roots: std.StringHashMap(u32),
    bpe_pairs: std.StringHashMap(BPEMerge),
    anchors: std.StringHashMap(u64),
    allocated_strings: std.ArrayList([]u8),
    allocator: Allocator,
    next_token_id: u32,
    language: Language,
    max_vocab_size: ?u32,
    sorted_prefix_keys: std.ArrayList([]const u8),
    sorted_suffix_keys: std.ArrayList([]const u8),
    max_prefix_len: usize,
    max_suffix_len: usize,
    byte_token_ids: [256]?u32,
    byte_token_values: std.AutoHashMap(u32, u8),

    pub const Language = enum {
        english,
        hungarian,
    };

    const BPEMerge = struct {
        token_id: u32,
        priority: u32,
    };

    const TokenPairKey = struct {
        first: u32,
        second: u32,
    };

    const PairFreq = struct {
        key: TokenPairKey,
        freq: u64,
    };

    const BpeSequence = struct {
        storage: []u32,
        len: usize,

        fn items(self: *const BpeSequence) []const u32 {
            return self.storage[0..self.len];
        }

        fn mutableItems(self: *BpeSequence) []u32 {
            return self.storage[0..self.len];
        }
    };

    const BpeScanWorkerCtx = struct {
        sequences: []BpeSequence,
        pair_freqs: std.AutoHashMap(TokenPairKey, u64),
        err: ?anyerror,
    };

    const BpeRebuildWorkerCtx = struct {
        sequences: []BpeSequence,
        best_first: u32,
        best_second: u32,
        merge_id: u32,
    };

    const SPECIAL_TOKENS = struct {
        const PAD: u32 = 0;
        const UNK: u32 = 1;
        const BOS: u32 = 2;
        const EOS: u32 = 3;
    };

    pub fn init(allocator: Allocator, vocab: []const []const u8, anchors: []const []const u8, max_vocab_size: ?u32, language: Language) !MGT {
        if (max_vocab_size) |max| {
            if (max < 4) return error.VocabularyTooSmall;
        }

        var mgt = initEmpty(allocator, max_vocab_size, language);
        errdefer mgt.deinit();

        _ = try mgt.addToken("[PAD]");
        _ = try mgt.addToken("[UNK]");
        _ = try mgt.addToken("[BOS]");
        _ = try mgt.addToken("[EOS]");

        for (vocab) |word| {
            if (!mgt.canAddToken()) break;
            _ = try mgt.addToken(word);
        }

        try mgt.initMorphemes();

        for (anchors) |anch| {
            if (!mgt.canAddToken() and !mgt.token_to_id.contains(anch)) break;
            const tid = mgt.token_to_id.get(anch) orelse try mgt.addToken(anch);
            const key = mgt.id_to_token.get(tid) orelse return error.InvalidData;
            try mgt.anchors.put(key, @as(u64, tid));
        }

        return mgt;
    }

    pub fn initWithArena(arena: *core_memory.ArenaAllocator, vocab: []const []const u8, anchors_list: []const []const u8, max_vocab_size: ?u32, language: Language) !MGT {
        return init(arena.allocator(), vocab, anchors_list, max_vocab_size, language);
    }

    pub fn initWithPool(pool: *core_memory.PoolAllocator, vocab: []const []const u8, anchors_list: []const []const u8, max_vocab_size: ?u32, language: Language) !MGT {
        return init(pool.allocator(), vocab, anchors_list, max_vocab_size, language);
    }

    pub fn initWithBuddy(buddy: *core_memory.BuddyAllocator, vocab: []const []const u8, anchors_list: []const []const u8, max_vocab_size: ?u32, language: Language) !MGT {
        return init(buddy.allocator(), vocab, anchors_list, max_vocab_size, language);
    }

    fn initEmpty(allocator: Allocator, max_vocab_size: ?u32, language: Language) MGT {
        return .{
            .token_to_id = std.StringHashMap(u32).init(allocator),
            .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
            .prefixes = std.StringHashMap(u32).init(allocator),
            .suffixes = std.StringHashMap(u32).init(allocator),
            .roots = std.StringHashMap(u32).init(allocator),
            .bpe_pairs = std.StringHashMap(BPEMerge).init(allocator),
            .anchors = std.StringHashMap(u64).init(allocator),
            .allocated_strings = std.ArrayList([]u8).init(allocator),
            .allocator = allocator,
            .next_token_id = 0,
            .language = language,
            .max_vocab_size = max_vocab_size,
            .sorted_prefix_keys = std.ArrayList([]const u8).init(allocator),
            .sorted_suffix_keys = std.ArrayList([]const u8).init(allocator),
            .max_prefix_len = 0,
            .max_suffix_len = 0,
            .byte_token_ids = [_]?u32{null} ** 256,
            .byte_token_values = std.AutoHashMap(u32, u8).init(allocator),
        };
    }

    fn canAddToken(self: *const MGT) bool {
        if (self.next_token_id == std.math.maxInt(u32)) return false;
        if (self.max_vocab_size) |max| {
            return self.token_to_id.count() < @as(usize, max);
        }
        return true;
    }

    fn reset(self: *MGT) void {
        const allocator = self.allocator;
        const mvs = self.max_vocab_size;
        const lang = self.language;
        self.deinit();
        self.* = initEmpty(allocator, mvs, lang);
    }

    fn initMorphemes(self: *MGT) !void {
        const english_prefix_list = [_][]const u8{
            "un",  "re",   "pre",   "dis",   "mis",  "over", "under", "out",
            "sub", "inter", "fore",  "de",    "trans", "super", "semi", "anti",
            "mid", "non",   "ex",    "post",  "pro",  "co",    "en",   "em",
        };

        const hungarian_prefix_list = [_][]const u8{
            "meg", "el", "fel", "le", "be", "ki", "rá", "át", "szét", "vissza",
            "ide", "oda", "alá", "fölé", "közé", "egy", "össze", "tul", "hozzá", "körül",
            "alig", "éppen", "majd", "csak", "is", "leg", "legesleg",
        };

        const english_suffix_list = [_][]const u8{
            "ing", "ed",  "er",   "est",  "ly",   "tion", "sion", "ness",
            "ment", "ful", "less", "ous",  "ive",  "able", "ible", "al",
            "ial", "y",   "s",    "es",   "en",   "ize",  "ise",  "ate",
        };

        const hungarian_suffix_list = [_][]const u8{
            "ság", "ség", "ságú", "ségű", "é", "je", "ja", "ban", "ben",
            "ba", "be", "ból", "ből", "hoz", "hez", "höz", "tól", "től",
            "nak", "nek", "val", "vel", "ért", "ul", "ül", "ként", "án",
            "én", "ig", "at", "et", "tat", "tet", "ott", "ett", "atlan",
            "etlen", "talan", "telen", "ál", "él", "oz", "ez", "öd", "ed",
            "gyet", "get", "j", "unk", "jatok", "játok", "i", "ni", "nként",
            "kor", "ra", "re",
        };

        const prefix_list: []const []const u8 = switch (self.language) {
            .english => english_prefix_list[0..],
            .hungarian => hungarian_prefix_list[0..],
        };

        const suffix_list: []const []const u8 = switch (self.language) {
            .english => english_suffix_list[0..],
            .hungarian => hungarian_suffix_list[0..],
        };

        for (prefix_list) |prefix| {
            if (!self.canAddToken() and !self.token_to_id.contains(prefix)) break;
            const id = self.token_to_id.get(prefix) orelse try self.addToken(prefix);
            const key = self.id_to_token.get(id) orelse return error.InvalidData;
            try self.prefixes.put(key, id);
        }

        for (suffix_list) |suffix| {
            if (!self.canAddToken() and !self.token_to_id.contains(suffix)) break;
            const id = self.token_to_id.get(suffix) orelse try self.addToken(suffix);
            const key = self.id_to_token.get(id) orelse return error.InvalidData;
            try self.suffixes.put(key, id);
        }

        try self.rebuildSortedMorphemes();
    }

    fn rebuildSortedMorphemes(self: *MGT) !void {
        self.sorted_prefix_keys.clearRetainingCapacity();
        var pit = self.prefixes.iterator();
        while (pit.next()) |entry| {
            try self.sorted_prefix_keys.append(entry.key_ptr.*);
        }

        std.mem.sort([]const u8, self.sorted_prefix_keys.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan);

        self.max_prefix_len = 0;
        for (self.sorted_prefix_keys.items) |key| {
            self.max_prefix_len = @max(self.max_prefix_len, key.len);
        }

        self.sorted_suffix_keys.clearRetainingCapacity();
        var sit = self.suffixes.iterator();
        while (sit.next()) |entry| {
            try self.sorted_suffix_keys.append(entry.key_ptr.*);
        }

        std.mem.sort([]const u8, self.sorted_suffix_keys.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan);

        self.max_suffix_len = 0;
        for (self.sorted_suffix_keys.items) |key| {
            self.max_suffix_len = @max(self.max_suffix_len, key.len);
        }
    }

    pub fn deinit(self: *MGT) void {
        self.token_to_id.deinit();
        self.id_to_token.deinit();
        self.prefixes.deinit();
        self.suffixes.deinit();
        self.roots.deinit();
        self.bpe_pairs.deinit();
        self.anchors.deinit();
        self.sorted_prefix_keys.deinit();
        self.sorted_suffix_keys.deinit();
        self.byte_token_values.deinit();

        for (self.allocated_strings.items) |str| {
            self.allocator.free(str);
        }

        self.allocated_strings.deinit();
    }

    fn isWhitespace(c: u8) bool {
        return c == ' ' or c == '\n' or c == '\t' or c == '\r';
    }

    fn isPunctuation(c: u8) bool {
        return c == '.' or c == ',' or c == '!' or c == '?' or c == ';' or
            c == ':' or c == '"' or c == '\'' or c == '(' or c == ')' or
            c == '{' or c == '}';
    }

    fn isKnownSpecialTokenStart(self: *const MGT, text: []const u8, pos: usize) bool {
        if (pos >= text.len or text[pos] != '[') return false;

        const specials = [_][]const u8{ "[PAD]", "[UNK]", "[BOS]", "[EOS]" };
        for (specials) |special| {
            if (special.len <= text.len - pos and
                mem.eql(u8, text[pos .. pos + special.len], special) and
                self.token_to_id.contains(special))
            {
                return true;
            }
        }

        return false;
    }

    fn getKnownSpecialTokenLen(self: *const MGT, text: []const u8, pos: usize) ?usize {
        if (pos >= text.len or text[pos] != '[') return null;

        const specials = [_][]const u8{ "[PAD]", "[UNK]", "[BOS]", "[EOS]" };
        for (specials) |special| {
            if (special.len <= text.len - pos and
                mem.eql(u8, text[pos .. pos + special.len], special) and
                self.token_to_id.contains(special))
            {
                return special.len;
            }
        }

        return null;
    }

    fn utf8CharLen(first_byte: u8) u8 {
        if (first_byte & 0x80 == 0) return 1;
        if (first_byte & 0xE0 == 0xC0) return 2;
        if (first_byte & 0xF0 == 0xE0) return 3;
        if (first_byte & 0xF8 == 0xF0) return 4;
        return 1;
    }

    fn safeUtf8SequenceLenAt(text: []const u8, pos: usize) usize {
        if (pos >= text.len) return 0;

        const sequence_len: usize = utf8CharLen(text[pos]);
        if (sequence_len > text.len - pos) return 1;
        if (sequence_len == 1) return 1;

        var i: usize = 1;
        while (i < sequence_len) : (i += 1) {
            if ((text[pos + i] & 0xC0) != 0x80) return 1;
        }

        return sequence_len;
    }

    fn emitToken(self: *const MGT, tid: u32, byte_pos: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        try out_tokens.append(tid);

        if (out_anchors) |anchors_out| {
            if (self.id_to_token.get(tid)) |token_str| {
                if (self.anchors.contains(token_str)) {
                    try anchors_out.append(byte_pos);
                }
            }
        }
    }

    fn appendUnknownForSlice(self: *const MGT, slice: []const u8, byte_pos: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        const tid = self.unknownReplacement(slice);
        try self.emitToken(tid, byte_pos, out_tokens, out_anchors);
    }

    fn appendBPEOrUnknown(self: *MGT, slice: []const u8, byte_pos: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        const tokens = try self.encodeBPE(slice);
        defer self.allocator.free(tokens);

        if (tokens.len == 0) {
            try self.appendUnknownForSlice(slice, byte_pos, out_tokens, out_anchors);
            return;
        }

        for (tokens) |tid| {
            try self.emitToken(tid, byte_pos, out_tokens, out_anchors);
        }
    }

    fn encodeInternal(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        var i: usize = 0;

        while (i < text.len) {
            if (self.getKnownSpecialTokenLen(text, i)) |special_len| {
                const special_token = text[i .. i + special_len];
                const tid = self.token_to_id.get(special_token) orelse return error.InvalidData;
                try self.emitToken(tid, i, out_tokens, out_anchors);
                i += special_len;
                continue;
            }

            if (isWhitespace(text[i])) {
                const whitespace = text[i .. i + 1];

                if (self.token_to_id.get(whitespace)) |tid| {
                    try self.emitToken(tid, i, out_tokens, out_anchors);
                } else if (text[i] == ' ') {
                    if (self.token_to_id.get(" ")) |space_tid| {
                        try self.emitToken(space_tid, i, out_tokens, out_anchors);
                    } else {
                        try self.appendUnknownForSlice(whitespace, i, out_tokens, out_anchors);
                    }
                } else {
                    try self.appendUnknownForSlice(whitespace, i, out_tokens, out_anchors);
                }

                i += 1;
                continue;
            }

            if (isPunctuation(text[i])) {
                const punctuation = text[i .. i + 1];

                if (self.token_to_id.get(punctuation)) |tid| {
                    try self.emitToken(tid, i, out_tokens, out_anchors);
                } else {
                    try self.appendBPEOrUnknown(punctuation, i, out_tokens, out_anchors);
                }

                i += 1;
                continue;
            }

            var word_end = i;
            while (word_end < text.len) {
                if (self.isKnownSpecialTokenStart(text, word_end)) break;

                const c = text[word_end];
                if (isWhitespace(c) or isPunctuation(c)) break;

                const char_len = safeUtf8SequenceLenAt(text, word_end);
                if (char_len == 0) break;
                word_end += char_len;
            }

            if (word_end == i) {
                const char_len = safeUtf8SequenceLenAt(text, i);
                if (char_len == 0 or char_len > text.len - i) return error.InvalidData;
                try self.appendBPEOrUnknown(text[i .. i + char_len], i, out_tokens, out_anchors);
                i += char_len;
                continue;
            }

            const word = text[i..word_end];

            if (self.token_to_id.get(word)) |tid| {
                try self.emitToken(tid, i, out_tokens, out_anchors);
            } else if (try self.morphDecompose(word, i, out_tokens, out_anchors)) {
            } else {
                try self.subwordSplitInto(word, i, out_tokens, out_anchors);
            }

            i = word_end;
        }
    }

    pub fn encode(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32)) !void {
        try self.encodeInternal(text, out_tokens, null);
    }

    fn binarySearchString(sorted: []const []const u8, target: []const u8) bool {
        var lo: usize = 0;
        var hi: usize = sorted.len;

        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;

            if (std.mem.lessThan(u8, sorted[mid], target)) {
                lo = mid + 1;
            } else if (std.mem.lessThan(u8, target, sorted[mid])) {
                hi = mid;
            } else {
                return true;
            }
        }

        return false;
    }

    fn findLongestPrefix(self: *MGT, word: []const u8) ?struct { prefix: []const u8, len: usize } {
        if (word.len < 2) return null;

        var test_len = @min(self.max_prefix_len, word.len - 1);
        while (test_len > 0) : (test_len -= 1) {
            const candidate = word[0..test_len];

            if (self.prefixes.get(candidate)) |id| {
                const key = self.id_to_token.get(id) orelse return null;
                return .{ .prefix = key, .len = test_len };
            }
        }

        return null;
    }

    fn findLongestSuffix(self: *MGT, word: []const u8) ?struct { suffix: []const u8, len: usize } {
        if (word.len < 2) return null;

        var test_len = @min(self.max_suffix_len, word.len - 1);
        while (test_len > 0) : (test_len -= 1) {
            const candidate = word[word.len - test_len ..];

            if (self.suffixes.get(candidate)) |id| {
                const key = self.id_to_token.get(id) orelse return null;
                return .{ .suffix = key, .len = test_len };
            }
        }

        return null;
    }

    fn morphDecompose(self: *MGT, word: []const u8, word_start: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !bool {
        if (word.len < 4) return false;

        const prefix_result = self.findLongestPrefix(word);
        const suffix_result = self.findLongestSuffix(word);

        const prefix_len = if (prefix_result) |prefix| prefix.len else 0;
        const suffix_len = if (suffix_result) |suffix| suffix.len else 0;

        if (prefix_len == 0 and suffix_len == 0) return false;
        if (prefix_len > word.len or suffix_len > word.len - prefix_len) return false;

        const root_start = prefix_len;
        const root_end = word.len - suffix_len;

        if (root_end <= root_start or root_end - root_start < 2) return false;

        const root = word[root_start..root_end];
        const root_tid = self.token_to_id.get(root) orelse return false;

        if (prefix_result) |prefix| {
            const tid = self.token_to_id.get(prefix.prefix) orelse return false;
            try self.emitToken(tid, word_start, out_tokens, out_anchors);
        }

        try self.emitToken(root_tid, word_start + root_start, out_tokens, out_anchors);

        if (suffix_result) |suffix| {
            const tid = self.token_to_id.get(suffix.suffix) orelse return false;
            try self.emitToken(tid, word_start + word.len - suffix.len, out_tokens, out_anchors);
        }

        return true;
    }

    fn addByteToken(self: *MGT, byte: u8) !u32 {
        if (self.byte_token_ids[byte]) |existing| return existing;

        var buffer: [16]u8 = undefined;
        const byte_string = try std.fmt.bufPrint(&buffer, "<{x:0>2}>", .{byte});
        const id = try self.addToken(byte_string);

        try self.byte_token_values.put(id, byte);
        self.byte_token_ids[byte] = id;

        return id;
    }

    fn addToken(self: *MGT, token: []const u8) !u32 {
        if (self.token_to_id.get(token)) |existing| {
            return existing;
        }

        if (!self.canAddToken()) {
            return error.VocabularyFull;
        }

        const id = self.next_token_id;
        const next_id = std.math.add(u32, id, 1) catch return error.TokenIdOverflow;
        const token_copy = try self.allocator.dupe(u8, token);
        errdefer self.allocator.free(token_copy);

        try self.token_to_id.put(token_copy, id);
        errdefer _ = self.token_to_id.remove(token_copy);

        try self.id_to_token.put(id, token_copy);
        errdefer _ = self.id_to_token.remove(id);

        try self.allocated_strings.append(token_copy);
        self.next_token_id = next_id;

        return id;
    }

    fn adoptTokenWithId(self: *MGT, token: []u8, id: u32) !void {
        if (id == std.math.maxInt(u32)) return error.TokenIdOverflow;
        if (self.token_to_id.contains(token) or self.id_to_token.contains(id)) {
            return error.InvalidData;
        }

        errdefer self.allocator.free(token);

        try self.token_to_id.put(token, id);
        errdefer _ = self.token_to_id.remove(token);

        try self.id_to_token.put(id, token);
        errdefer _ = self.id_to_token.remove(id);

        try self.allocated_strings.append(token);

        if (id >= self.next_token_id) {
            self.next_token_id = std.math.add(u32, id, 1) catch return error.TokenIdOverflow;
        }
    }

    fn getCanonicalTokenForLoad(self: *MGT, raw: []u8, id: u32) ![]const u8 {
        if (self.id_to_token.get(id)) |canonical| {
            if (!mem.eql(u8, canonical, raw)) return error.InvalidData;
            self.allocator.free(raw);
            return canonical;
        }

        if (self.token_to_id.get(raw)) |existing_id| {
            if (existing_id != id) return error.InvalidData;
            const canonical = self.id_to_token.get(existing_id) orelse return error.InvalidData;
            self.allocator.free(raw);
            return canonical;
        }

        try self.adoptTokenWithId(raw, id);
        return self.id_to_token.get(id) orelse return error.InvalidData;
    }

    fn encodeBPE(self: *MGT, text: []const u8) ![]u32 {
        if (text.len == 0) return self.allocator.alloc(u32, 0);

        var current = std.ArrayList(u32).init(self.allocator);
        defer current.deinit();

        for (text) |byte| {
            const tid = if (self.byte_token_ids[byte]) |existing|
                existing
            else
                try self.addByteToken(byte);

            try current.append(tid);
        }

        var pair_cache = std.StringHashMap(BPEMerge).init(self.allocator);
        defer {
            var iterator = pair_cache.iterator();
            while (iterator.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            pair_cache.deinit();
        }

        while (current.items.len > 1) {
            var best_priority: u32 = std.math.maxInt(u32);
            var best_left: u32 = 0;
            var best_right: u32 = 0;
            var best_merge_id: u32 = 0;
            var found = false;

            var i: usize = 0;
            while (i < current.items.len - 1) : (i += 1) {
                const left_id = current.items[i];
                const right_id = current.items[i + 1];

                var cache_key_buffer: [64]u8 = undefined;
                const cache_key = try std.fmt.bufPrint(&cache_key_buffer, "{d}_{d}", .{ left_id, right_id });

                if (pair_cache.get(cache_key)) |merge| {
                    if (merge.priority < best_priority) {
                        best_priority = merge.priority;
                        best_left = left_id;
                        best_right = right_id;
                        best_merge_id = merge.token_id;
                        found = true;
                    }
                    continue;
                }

                const left = self.id_to_token.get(left_id) orelse return error.InvalidData;
                const right = self.id_to_token.get(right_id) orelse return error.InvalidData;
                const pair = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ left, right });
                defer self.allocator.free(pair);

                const merge = self.bpe_pairs.get(pair) orelse BPEMerge{
                    .token_id = 0,
                    .priority = std.math.maxInt(u32),
                };

                const owned_key = try std.fmt.allocPrint(self.allocator, "{d}_{d}", .{ left_id, right_id });
                errdefer self.allocator.free(owned_key);
                try pair_cache.put(owned_key, merge);

                if (merge.priority < best_priority) {
                    best_priority = merge.priority;
                    best_left = left_id;
                    best_right = right_id;
                    best_merge_id = merge.token_id;
                    found = true;
                }
            }

            if (!found) break;

            var write: usize = 0;
            var read: usize = 0;

            while (read < current.items.len) {
                if (read < current.items.len - 1 and
                    current.items[read] == best_left and
                    current.items[read + 1] == best_right)
                {
                    current.items[write] = best_merge_id;
                    write += 1;
                    read += 2;
                } else {
                    if (write != read) {
                        current.items[write] = current.items[read];
                    }
                    write += 1;
                    read += 1;
                }
            }

            current.shrinkRetainingCapacity(write);

            var clear_iterator = pair_cache.iterator();
            while (clear_iterator.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            pair_cache.clearRetainingCapacity();
        }

        return current.toOwnedSlice();
    }

    fn validateAllocationCount(comptime T: type, count: usize) !void {
        const bytes = std.math.mul(usize, count, @sizeOf(T)) catch return error.InputTooLarge;
        _ = std.math.add(usize, bytes, 65535) catch return error.InputTooLarge;
    }

    fn validateCorpus(corpus: []const []const u8) !void {
        try validateAllocationCount(BpeSequence, corpus.len);

        var total_tokens: usize = 0;
        for (corpus) |text| {
            try validateAllocationCount(u32, text.len);
            total_tokens = std.math.add(usize, total_tokens, text.len) catch return error.CorpusTooLarge;
        }

        try validateAllocationCount(u32, total_tokens);
    }

    fn bpeScanWorkerFn(ctx: *BpeScanWorkerCtx) void {
        for (ctx.sequences) |*sequence| {
            const seq = sequence.items();
            if (seq.len < 2) continue;

            var i: usize = 0;
            while (i < seq.len - 1) : (i += 1) {
                const key = TokenPairKey{
                    .first = seq[i],
                    .second = seq[i + 1],
                };

                const entry = ctx.pair_freqs.getOrPut(key) catch |err| {
                    ctx.err = err;
                    return;
                };

                if (entry.found_existing) {
                    entry.value_ptr.* = std.math.add(u64, entry.value_ptr.*, 1) catch {
                        ctx.err = error.FrequencyOverflow;
                        return;
                    };
                } else {
                    entry.value_ptr.* = 1;
                }
            }
        }
    }

    fn bpeRebuildWorkerFn(ctx: *BpeRebuildWorkerCtx) void {
        for (ctx.sequences) |*sequence| {
            var seq = sequence.mutableItems();
            if (seq.len < 2) continue;

            var write: usize = 0;
            var read: usize = 0;

            while (read < seq.len) {
                if (read < seq.len - 1 and
                    seq[read] == ctx.best_first and
                    seq[read + 1] == ctx.best_second)
                {
                    seq[write] = ctx.merge_id;
                    write += 1;
                    read += 2;
                } else {
                    if (write != read) {
                        seq[write] = seq[read];
                    }
                    write += 1;
                    read += 1;
                }
            }

            sequence.len = write;
        }
    }

    fn scanBpePairsParallel(sequences: []BpeSequence, requested_workers: usize, transient_allocator: Allocator, destination: *std.AutoHashMap(TokenPairKey, u64)) !void {
        if (sequences.len == 0) return;

        const worker_count = @min(@max(@as(usize, 1), requested_workers), sequences.len);
        const contexts = try transient_allocator.alloc(BpeScanWorkerCtx, worker_count);
        defer transient_allocator.free(contexts);

        const threads = try transient_allocator.alloc(std.Thread, worker_count);
        defer transient_allocator.free(threads);

        const base_chunk = sequences.len / worker_count;
        const remainder = sequences.len % worker_count;

        var offset: usize = 0;
        for (contexts, 0..) |*ctx, worker_index| {
            const chunk = base_chunk + @as(usize, if (worker_index < remainder) 1 else 0);
            ctx.* = .{
                .sequences = sequences[offset .. offset + chunk],
                .pair_freqs = std.AutoHashMap(TokenPairKey, u64).init(transient_allocator),
                .err = null,
            };
            offset += chunk;
        }

        defer {
            for (contexts) |*ctx| {
                ctx.pair_freqs.deinit();
            }
        }

        var spawned: usize = 0;
        while (spawned < worker_count) {
            threads[spawned] = std.Thread.spawn(.{}, bpeScanWorkerFn, .{&contexts[spawned]}) catch |err| {
                for (threads[0..spawned]) |thread| {
                    thread.join();
                }
                return err;
            };
            spawned += 1;
        }

        for (threads[0..spawned]) |thread| {
            thread.join();
        }

        for (contexts) |*ctx| {
            if (ctx.err) |err| return err;
        }

        for (contexts) |*ctx| {
            var iterator = ctx.pair_freqs.iterator();
            while (iterator.next()) |entry| {
                const global_entry = try destination.getOrPut(entry.key_ptr.*);

                if (global_entry.found_existing) {
                    global_entry.value_ptr.* = std.math.add(
                        u64,
                        global_entry.value_ptr.*,
                        entry.value_ptr.*,
                    ) catch return error.FrequencyOverflow;
                } else {
                    global_entry.value_ptr.* = entry.value_ptr.*;
                }
            }
        }
    }

    fn rebuildBpeSequencesParallel(sequences: []BpeSequence, requested_workers: usize, transient_allocator: Allocator, best_key: TokenPairKey, merge_id: u32) !void {
        if (sequences.len == 0) return;

        const worker_count = @min(@max(@as(usize, 1), requested_workers), sequences.len);
        const contexts = try transient_allocator.alloc(BpeRebuildWorkerCtx, worker_count);
        defer transient_allocator.free(contexts);

        const threads = try transient_allocator.alloc(std.Thread, worker_count);
        defer transient_allocator.free(threads);

        const base_chunk = sequences.len / worker_count;
        const remainder = sequences.len % worker_count;

        var offset: usize = 0;
        for (contexts, 0..) |*ctx, worker_index| {
            const chunk = base_chunk + @as(usize, if (worker_index < remainder) 1 else 0);
            ctx.* = .{
                .sequences = sequences[offset .. offset + chunk],
                .best_first = best_key.first,
                .best_second = best_key.second,
                .merge_id = merge_id,
            };
            offset += chunk;
        }

        var spawned: usize = 0;
        while (spawned < worker_count) {
            threads[spawned] = std.Thread.spawn(.{}, bpeRebuildWorkerFn, .{&contexts[spawned]}) catch |err| {
                for (threads[0..spawned]) |thread| {
                    thread.join();
                }
                return err;
            };
            spawned += 1;
        }

        for (threads[0..spawned]) |thread| {
            thread.join();
        }
    }

    fn initialBpePriority(self: *const MGT) !u32 {
        var next_priority: u32 = 0;
        var iterator = self.bpe_pairs.iterator();

        while (iterator.next()) |entry| {
            const priority = entry.value_ptr.priority;
            if (priority >= next_priority) {
                next_priority = std.math.add(u32, priority, 1) catch return error.PriorityOverflow;
            }
        }

        return next_priority;
    }

    pub fn trainBPE(self: *MGT, corpus: []const []const u8, target_vocab_size: u32) !void {
        const configured_target = if (self.max_vocab_size) |limit|
            @min(target_vocab_size, limit)
        else
            target_vocab_size;

        if (self.vocabSize() >= @as(usize, configured_target)) return;

        try validateCorpus(corpus);

        const transient_allocator = std.heap.page_allocator;
        const sequences = try transient_allocator.alloc(BpeSequence, corpus.len);
        var sequence_count: usize = 0;

        defer {
            for (sequences[0..sequence_count]) |sequence| {
                transient_allocator.free(sequence.storage);
            }
            transient_allocator.free(sequences);
        }

        for (corpus) |text| {
            if (text.len == 0) continue;

            try validateAllocationCount(u32, text.len);
            const storage = try transient_allocator.alloc(u32, text.len);
            errdefer transient_allocator.free(storage);

            for (text, 0..) |byte, index| {
                storage[index] = try self.addByteToken(byte);
            }

            sequences[sequence_count] = .{
                .storage = storage,
                .len = storage.len,
            };
            sequence_count += 1;
        }

        if (sequence_count == 0) return;
        if (self.vocabSize() >= @as(usize, configured_target)) return;

        const cpu_count = std.Thread.getCpuCount() catch 1;
        const worker_count = @min(@max(@as(usize, 1), cpu_count), sequence_count);

        var pair_freqs = std.AutoHashMap(TokenPairKey, u64).init(transient_allocator);
        defer pair_freqs.deinit();

        try scanBpePairsParallel(
            sequences[0..sequence_count],
            worker_count,
            transient_allocator,
            &pair_freqs,
        );

        var merge_priority = try self.initialBpePriority();

        while (self.vocabSize() < @as(usize, configured_target)) {
            var best: ?PairFreq = null;
            var iterator = pair_freqs.iterator();

            while (iterator.next()) |entry| {
                const candidate = PairFreq{
                    .key = entry.key_ptr.*,
                    .freq = entry.value_ptr.*,
                };

                if (candidate.freq < 2) continue;

                if (best == null or
                    candidate.freq > best.?.freq or
                    (candidate.freq == best.?.freq and candidate.key.first < best.?.key.first) or
                    (candidate.freq == best.?.freq and
                        candidate.key.first == best.?.key.first and
                        candidate.key.second < best.?.key.second))
                {
                    best = candidate;
                }
            }

            const selected = best orelse break;
            const first_string = self.id_to_token.get(selected.key.first) orelse return error.InvalidData;
            const second_string = self.id_to_token.get(selected.key.second) orelse return error.InvalidData;
            const merged_text = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ first_string, second_string });
            defer self.allocator.free(merged_text);

            const vocabulary_size_before = self.vocabSize();
            const merge_token_id = try self.addToken(merged_text);
            const canonical = self.id_to_token.get(merge_token_id) orelse return error.InvalidData;

            try self.bpe_pairs.put(canonical, .{
                .token_id = merge_token_id,
                .priority = merge_priority,
            });

            try rebuildBpeSequencesParallel(
                sequences[0..sequence_count],
                worker_count,
                transient_allocator,
                selected.key,
                merge_token_id,
            );

            pair_freqs.clearRetainingCapacity();

            try scanBpePairsParallel(
                sequences[0..sequence_count],
                worker_count,
                transient_allocator,
                &pair_freqs,
            );

            if (merge_priority == std.math.maxInt(u32)) {
                if (self.vocabSize() < @as(usize, configured_target)) {
                    return error.PriorityOverflow;
                }
            } else {
                merge_priority += 1;
            }

            if (self.vocabSize() == vocabulary_size_before and pair_freqs.count() == 0) {
                break;
            }
        }
    }

    pub fn decode(self: *MGT, tokens: []const u32, out_text: *std.ArrayList(u8)) !void {
        for (tokens) |token| {
            if (self.byte_token_values.get(token)) |byte| {
                try out_text.append(byte);
            } else if (self.id_to_token.get(token)) |token_string| {
                try out_text.appendSlice(token_string);
            } else {
                const unknown = self.id_to_token.get(SPECIAL_TOKENS.UNK) orelse "[UNK]";
                try out_text.appendSlice(unknown);
            }
        }
    }

    pub fn longestMatch(self: *MGT, text: []const u8, start: usize) usize {
        if (start >= text.len) return 0;

        var max_len: usize = 0;
        var end = start;

        while (end < text.len) {
            const step = safeUtf8SequenceLenAt(text, end);
            if (step == 0 or step > text.len - end) break;

            end += step;
            const substring = text[start..end];

            if (self.token_to_id.contains(substring)) {
                max_len = end - start;
            }
        }

        return max_len;
    }

    pub fn vocabSize(self: *const MGT) usize {
        return self.token_to_id.count();
    }

    pub fn addVocabWord(self: *MGT, word: []const u8, is_anchor: bool) !void {
        if (!self.canAddToken() and !self.token_to_id.contains(word)) {
            return error.VocabularyFull;
        }

        const id = try self.addToken(word);

        if (is_anchor) {
            const key = self.id_to_token.get(id) orelse return error.InvalidData;
            try self.anchors.put(key, @as(u64, id));
        }
    }

    pub fn removeVocabWord(self: *MGT, word: []const u8) void {
        if (mem.eql(u8, word, "[PAD]") or
            mem.eql(u8, word, "[UNK]") or
            mem.eql(u8, word, "[BOS]") or
            mem.eql(u8, word, "[EOS]"))
        {
            return;
        }

        if (self.token_to_id.get(word)) |id| {
            if (self.id_to_token.get(id)) |allocated_pointer| {
                _ = self.token_to_id.remove(word);
                _ = self.id_to_token.remove(id);
                _ = self.anchors.remove(word);
                _ = self.prefixes.remove(word);
                _ = self.suffixes.remove(word);
                _ = self.roots.remove(word);

                if (self.byte_token_values.fetchRemove(id)) |entry| {
                    self.byte_token_ids[entry.value] = null;
                }

                var bpe_remove = std.ArrayList([]const u8).init(self.allocator);
                defer bpe_remove.deinit();

                var bpe_iterator = self.bpe_pairs.iterator();
                while (bpe_iterator.next()) |entry| {
                    const key = entry.key_ptr.*;
                    const merge = entry.value_ptr.*;

                    if (mem.eql(u8, key, allocated_pointer) or merge.token_id == id) {
                        bpe_remove.append(key) catch return;
                    }
                }

                for (bpe_remove.items) |key| {
                    _ = self.bpe_pairs.remove(key);
                }

                var index: usize = 0;
                while (index < self.allocated_strings.items.len) : (index += 1) {
                    const str = self.allocated_strings.items[index];

                    if (str.ptr == allocated_pointer.ptr and str.len == allocated_pointer.len) {
                        self.allocator.free(str);
                        _ = self.allocated_strings.swapRemove(index);
                        break;
                    }
                }
            }
        }

        self.rebuildSortedMorphemes() catch {};
    }

    pub fn tokenizeWithAnchors(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32), out_anchors: *std.ArrayList(usize)) !void {
        try self.encodeInternal(text, out_tokens, out_anchors);
    }

    pub fn detokenize(self: *MGT, tokens: []const u32) ![]u8 {
        return self.detokenizeAlloc(tokens, self.allocator);
    }

    fn detokenizeAlloc(self: *MGT, tokens: []const u32, allocator: Allocator) ![]u8 {
        var text = std.ArrayList(u8).init(allocator);
        defer text.deinit();

        try self.decode(tokens, &text);
        return text.toOwnedSlice();
    }

    pub fn encodeBatch(self: *MGT, texts: []const []const u8, allocator: Allocator) ![][]u32 {
        const results = try allocator.alloc([]u32, texts.len);
        errdefer allocator.free(results);

        var initialized: usize = 0;
        errdefer {
            for (results[0..initialized]) |result| {
                allocator.free(result);
            }
        }

        for (texts) |text| {
            var tokens = std.ArrayList(u32).init(allocator);
            defer tokens.deinit();

            try self.encode(text, &tokens);
            results[initialized] = try tokens.toOwnedSlice();
            initialized += 1;
        }

        return results;
    }

    pub fn batchDetokenize(self: *MGT, token_lists: []const []const u32, allocator: Allocator) ![][]u8 {
        const results = try allocator.alloc([]u8, token_lists.len);
        errdefer allocator.free(results);

        var initialized: usize = 0;
        errdefer {
            for (results[0..initialized]) |result| {
                allocator.free(result);
            }
        }

        for (token_lists) |tokens| {
            results[initialized] = try self.detokenizeAlloc(tokens, allocator);
            initialized += 1;
        }

        return results;
    }

    fn usizeToU32(value: usize) !u32 {
        if (value > std.math.maxInt(u32)) return error.DataTooLarge;
        return @intCast(value);
    }

    fn writeStringMapSorted(map: std.StringHashMap(u32), writer: anytype, allocator: Allocator) !void {
        const Item = struct {
            key: []const u8,
            value: u32,
        };

        const Context = struct {
            fn lessThan(_: @This(), a: Item, b: Item) bool {
                if (a.value != b.value) return a.value < b.value;
                return std.mem.lessThan(u8, a.key, b.key);
            }
        };

        var items = std.ArrayList(Item).init(allocator);
        defer items.deinit();

        var iterator = map.iterator();
        while (iterator.next()) |entry| {
            try items.append(.{
                .key = entry.key_ptr.*,
                .value = entry.value_ptr.*,
            });
        }

        std.mem.sort(Item, items.items, Context{}, Context.lessThan);
        try writer.writeInt(u32, try usizeToU32(items.items.len), .little);

        for (items.items) |item| {
            try writer.writeInt(u32, try usizeToU32(item.key.len), .little);
            try writer.writeAll(item.key);
            try writer.writeInt(u32, item.value, .little);
        }
    }

    pub fn saveVocab(self: *MGT, path: []const u8) !void {
        var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer file.close();

        var writer = file.writer();

        const TokenItem = struct {
            id: u32,
            token: []const u8,
        };

        const TokenContext = struct {
            fn lessThan(_: @This(), a: TokenItem, b: TokenItem) bool {
                if (a.id != b.id) return a.id < b.id;
                return std.mem.lessThan(u8, a.token, b.token);
            }
        };

        var token_items = std.ArrayList(TokenItem).init(self.allocator);
        defer token_items.deinit();

        var token_iterator = self.id_to_token.iterator();
        while (token_iterator.next()) |entry| {
            try token_items.append(.{
                .id = entry.key_ptr.*,
                .token = entry.value_ptr.*,
            });
        }

        std.mem.sort(TokenItem, token_items.items, TokenContext{}, TokenContext.lessThan);
        try writer.writeInt(u32, try usizeToU32(token_items.items.len), .little);

        for (token_items.items) |item| {
            try writer.writeInt(u32, try usizeToU32(item.token.len), .little);
            try writer.writeAll(item.token);
            try writer.writeInt(u32, item.id, .little);
        }

        const BpeItem = struct {
            key: []const u8,
            merge: BPEMerge,
        };

        const BpeContext = struct {
            fn lessThan(_: @This(), a: BpeItem, b: BpeItem) bool {
                if (a.merge.priority != b.merge.priority) {
                    return a.merge.priority < b.merge.priority;
                }
                return std.mem.lessThan(u8, a.key, b.key);
            }
        };

        var bpe_items = std.ArrayList(BpeItem).init(self.allocator);
        defer bpe_items.deinit();

        var bpe_iterator = self.bpe_pairs.iterator();
        while (bpe_iterator.next()) |entry| {
            try bpe_items.append(.{
                .key = entry.key_ptr.*,
                .merge = entry.value_ptr.*,
            });
        }

        std.mem.sort(BpeItem, bpe_items.items, BpeContext{}, BpeContext.lessThan);
        try writer.writeInt(u32, try usizeToU32(bpe_items.items.len), .little);

        for (bpe_items.items) |item| {
            try writer.writeInt(u32, try usizeToU32(item.key.len), .little);
            try writer.writeAll(item.key);
            try writer.writeInt(u32, item.merge.token_id, .little);
            try writer.writeInt(u32, item.merge.priority, .little);
        }

        try writeStringMapSorted(self.prefixes, writer, self.allocator);
        try writeStringMapSorted(self.suffixes, writer, self.allocator);
        try writeStringMapSorted(self.roots, writer, self.allocator);

        const AnchorItem = struct {
            key: []const u8,
            value: u64,
        };

        const AnchorContext = struct {
            fn lessThan(_: @This(), a: AnchorItem, b: AnchorItem) bool {
                if (a.value != b.value) return a.value < b.value;
                return std.mem.lessThan(u8, a.key, b.key);
            }
        };

        var anchor_items = std.ArrayList(AnchorItem).init(self.allocator);
        defer anchor_items.deinit();

        var anchor_iterator = self.anchors.iterator();
        while (anchor_iterator.next()) |entry| {
            try anchor_items.append(.{
                .key = entry.key_ptr.*,
                .value = entry.value_ptr.*,
            });
        }

        std.mem.sort(AnchorItem, anchor_items.items, AnchorContext{}, AnchorContext.lessThan);
        try writer.writeInt(u32, try usizeToU32(anchor_items.items.len), .little);

        for (anchor_items.items) |item| {
            try writer.writeInt(u32, try usizeToU32(item.key.len), .little);
            try writer.writeAll(item.key);
            try writer.writeInt(u64, item.value, .little);
        }
    }

    pub fn loadVocab(self: *MGT, path: []const u8) !void {
        self.reset();

        var file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var reader = file.reader();
        const token_count = try reader.readInt(u32, .little);

        if (self.max_vocab_size) |limit| {
            if (token_count > limit) return error.VocabularyFull;
        }

        var token_index: u32 = 0;
        while (token_index < token_count) : (token_index += 1) {
            const word_len = try reader.readInt(u32, .little);
            const word_buffer = try self.allocator.alloc(u8, @as(usize, word_len));
            var word_owned = true;
            errdefer if (word_owned) self.allocator.free(word_buffer);

            try reader.readNoEof(word_buffer);
            const id = try reader.readInt(u32, .little);

            if (id == std.math.maxInt(u32) or
                self.token_to_id.contains(word_buffer) or
                self.id_to_token.contains(id))
            {
                return error.InvalidData;
            }

            try self.adoptTokenWithId(word_buffer, id);
            word_owned = false;
        }

        const bpe_count = try reader.readInt(u32, .little);
        var bpe_index: u32 = 0;

        while (bpe_index < bpe_count) : (bpe_index += 1) {
            const key_len = try reader.readInt(u32, .little);
            const key_buffer = try self.allocator.alloc(u8, @as(usize, key_len));
            var key_owned = true;
            errdefer if (key_owned) self.allocator.free(key_buffer);

            try reader.readNoEof(key_buffer);

            const token_id = try reader.readInt(u32, .little);
            const priority = try reader.readInt(u32, .little);
            const canonical = try self.getCanonicalTokenForLoad(key_buffer, token_id);
            key_owned = false;

            try self.bpe_pairs.put(canonical, .{
                .token_id = token_id,
                .priority = priority,
            });
        }

        const ReadStringMap = struct {
            fn read(self_mgt: *MGT, map: *std.StringHashMap(u32), input_reader: anytype) !void {
                const count = try input_reader.readInt(u32, .little);
                var index: u32 = 0;

                while (index < count) : (index += 1) {
                    const length = try input_reader.readInt(u32, .little);
                    const buffer = try self_mgt.allocator.alloc(u8, @as(usize, length));
                    var buffer_owned = true;
                    errdefer if (buffer_owned) self_mgt.allocator.free(buffer);

                    try input_reader.readNoEof(buffer);

                    const id = try input_reader.readInt(u32, .little);
                    const canonical = try self_mgt.getCanonicalTokenForLoad(buffer, id);
                    buffer_owned = false;

                    try map.put(canonical, id);
                }
            }
        };

        try ReadStringMap.read(self, &self.prefixes, reader);
        try ReadStringMap.read(self, &self.suffixes, reader);
        try ReadStringMap.read(self, &self.roots, reader);

        const anchor_count = try reader.readInt(u32, .little);
        var anchor_index: u32 = 0;

        while (anchor_index < anchor_count) : (anchor_index += 1) {
            const key_len = try reader.readInt(u32, .little);
            const key_buffer = try self.allocator.alloc(u8, @as(usize, key_len));
            var key_owned = true;
            errdefer if (key_owned) self.allocator.free(key_buffer);

            try reader.readNoEof(key_buffer);

            const value = try reader.readInt(u64, .little);
            if (value > std.math.maxInt(u32)) return error.InvalidData;

            const canonical = try self.getCanonicalTokenForLoad(key_buffer, @intCast(value));
            key_owned = false;

            try self.anchors.put(canonical, value);
        }

        try self.rebuildSortedMorphemes();
        try self.rebuildByteTokenLookup();
    }

    fn rebuildByteTokenLookup(self: *MGT) !void {
        self.byte_token_ids = [_]?u32{null} ** 256;
        self.byte_token_values.clearRetainingCapacity();

        var iterator = self.id_to_token.iterator();
        while (iterator.next()) |entry| {
            const token_string = entry.value_ptr.*;
            const id = entry.key_ptr.*;

            if (token_string.len == 4 and
                token_string[0] == '<' and
                token_string[3] == '>')
            {
                const hex = token_string[1..3];

                if (std.fmt.parseInt(u8, hex, 16)) |byte_value| {
                    if (self.byte_token_ids[byte_value] != null) {
                        return error.InvalidData;
                    }

                    self.byte_token_ids[byte_value] = id;
                    try self.byte_token_values.put(id, byte_value);
                } else |_| {}
            }
        }
    }

    pub fn unknownReplacement(self: *const MGT, context: []const u8) u32 {
        _ = self;
        _ = context;
        return SPECIAL_TOKENS.UNK;
    }

    fn subwordSplitInto(self: *MGT, word: []const u8, word_start: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        var i: usize = 0;

        while (i < word.len) {
            const match_len = self.longestMatch(word, i);

            if (match_len > 0 and match_len <= word.len - i) {
                const found_word = word[i .. i + match_len];

                if (self.token_to_id.get(found_word)) |tid| {
                    try self.emitToken(tid, word_start + i, out_tokens, out_anchors);
                    i += match_len;
                    continue;
                }
            }

            const char_len = safeUtf8SequenceLenAt(word, i);
            if (char_len == 0 or char_len > word.len - i) return error.InvalidData;

            const piece = word[i .. i + char_len];
            try self.appendBPEOrUnknown(piece, word_start + i, out_tokens, out_anchors);
            i += char_len;
        }
    }

    pub fn subwordSplit(self: *MGT, word: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).init(self.allocator);
        defer tokens.deinit();

        try self.subwordSplitInto(word, 0, &tokens, null);
        return tokens.toOwnedSlice();
    }

    pub fn mergeSubwords(self: *MGT, subwords: []const []const u32) ![]u32 {
        var merged = std.ArrayList(u32).init(self.allocator);
        defer merged.deinit();

        var total_length: usize = 0;
        for (subwords) |subword| {
            total_length = std.math.add(usize, total_length, subword.len) catch return error.InputTooLarge;
        }

        try merged.ensureTotalCapacity(total_length);

        for (subwords) |subword| {
            try merged.appendSlice(subword);
        }

        return merged.toOwnedSlice();
    }

    pub fn validateTokens(self: *MGT, tokens: []const u32) bool {
        for (tokens) |token| {
            if (!self.id_to_token.contains(token)) return false;
        }

        return true;
    }

    pub fn coverage(self: *MGT, corpus: []const u8) f32 {
        if (corpus.len == 0) return 0.0;

        var covered: usize = 0;
        var i: usize = 0;

        while (i < corpus.len) {
            if (self.getKnownSpecialTokenLen(corpus, i)) |special_len| {
                covered = std.math.add(usize, covered, special_len) catch return 0.0;
                i += special_len;
                continue;
            }

            if (isWhitespace(corpus[i]) or isPunctuation(corpus[i])) {
                const slice = corpus[i .. i + 1];

                if (self.token_to_id.contains(slice) or
                    (corpus[i] == ' ' and self.token_to_id.contains(" ")))
                {
                    covered = std.math.add(usize, covered, 1) catch return 0.0;
                }

                i += 1;
                continue;
            }

            var word_end = i;
            while (word_end < corpus.len) {
                if (self.isKnownSpecialTokenStart(corpus, word_end)) break;

                const c = corpus[word_end];
                if (isWhitespace(c) or isPunctuation(c)) break;

                const char_len = safeUtf8SequenceLenAt(corpus, word_end);
                if (char_len == 0 or char_len > corpus.len - word_end) break;
                word_end += char_len;
            }

            if (word_end == i) {
                const char_len = safeUtf8SequenceLenAt(corpus, i);
                if (char_len == 0 or char_len > corpus.len - i) break;

                const maybe_bpe = self.encodeBPE(corpus[i .. i + char_len]) catch null;
                if (maybe_bpe) |bpe| {
                    defer self.allocator.free(bpe);

                    var all_unknown = true;
                    for (bpe) |tid| {
                        if (tid != SPECIAL_TOKENS.UNK) {
                            all_unknown = false;
                            break;
                        }
                    }

                    if (!all_unknown) {
                        covered = std.math.add(usize, covered, char_len) catch return 0.0;
                    }
                }

                i += char_len;
                continue;
            }

            const word = corpus[i..word_end];

            if (self.token_to_id.contains(word)) {
                covered = std.math.add(usize, covered, word.len) catch return 0.0;
            } else {
                var temporary = std.ArrayList(u32).init(self.allocator);
                defer temporary.deinit();

                if (self.morphDecompose(word, i, &temporary, null) catch false) {
                    covered = std.math.add(usize, covered, word.len) catch return 0.0;
                } else {
                    const maybe_subwords = self.subwordSplit(word) catch null;

                    if (maybe_subwords) |subwords| {
                        defer self.allocator.free(subwords);

                        var all_unknown = true;
                        for (subwords) |tid| {
                            if (tid != SPECIAL_TOKENS.UNK) {
                                all_unknown = false;
                                break;
                            }
                        }

                        if (!all_unknown) {
                            covered = std.math.add(usize, covered, word.len) catch return 0.0;
                        }
                    }
                }
            }

            i = word_end;
        }

        return @as(f32, @floatFromInt(covered)) /
            @as(f32, @floatFromInt(corpus.len));
    }

    pub fn encodeToTensor(self: *MGT, text: []const u8, allocator: Allocator) !core_tensor.Tensor {
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        try self.encode(text, &tokens);

        const shape = [_]usize{tokens.items.len};
        var tensor = try core_tensor.Tensor.init(allocator, &shape);

        for (tokens.items, 0..) |token, index| {
            tensor.data[index] = @floatFromInt(token);
        }

        return tensor;
    }

    pub fn encodeBatchToTensor(self: *MGT, texts: []const []const u8, allocator: Allocator) !core_tensor.Tensor {
        var max_len: usize = 0;
        var per_row = std.ArrayList([]u32).init(allocator);

        defer {
            for (per_row.items) |row| {
                allocator.free(row);
            }
            per_row.deinit();
        }

        for (texts) |text| {
            var tokens = std.ArrayList(u32).init(allocator);
            defer tokens.deinit();

            try self.encode(text, &tokens);

            const owned = try tokens.toOwnedSlice();
            errdefer allocator.free(owned);

            try per_row.append(owned);
            max_len = @max(max_len, owned.len);
        }

        if (max_len == 0) max_len = 1;
        _ = std.math.mul(usize, texts.len, max_len) catch return error.InputTooLarge;

        const shape = [_]usize{ texts.len, max_len };
        var tensor = try core_tensor.Tensor.init(allocator, &shape);
        @memset(tensor.data, @as(@TypeOf(tensor.data[0]), 0));

        for (per_row.items, 0..) |row, row_index| {
            for (row, 0..) |token, column_index| {
                const row_offset = std.math.mul(usize, row_index, max_len) catch return error.InputTooLarge;
                const tensor_index = std.math.add(usize, row_offset, column_index) catch return error.InputTooLarge;
                tensor.data[tensor_index] = @floatFromInt(token);
            }
        }

        return tensor;
    }

    pub fn decodeFromTensor(self: *MGT, tensor: *const core_tensor.Tensor, allocator: Allocator) ![]u8 {
        const tokens = try allocator.alloc(u32, tensor.data.len);
        defer allocator.free(tokens);

        for (tensor.data, 0..) |value, index| {
            if (std.math.isNan(value) or
                std.math.isInf(value) or
                value < 0.0 or
                value > @as(@TypeOf(value), @floatFromInt(std.math.maxInt(u32))))
            {
                tokens[index] = SPECIAL_TOKENS.UNK;
            } else {
                tokens[index] = @intFromFloat(value);

                if (!self.id_to_token.contains(tokens[index])) {
                    tokens[index] = SPECIAL_TOKENS.UNK;
                }
            }
        }

        return self.detokenizeAlloc(tokens, allocator);
    }
};

test "MGT encode decode" {
    const gpa = testing.allocator;
    const vocab = &.{ "hello", "world", " " };
    const anchors = &.{"hello"};

    var mgt = try MGT.init(gpa, vocab, anchors, null, .english);
    defer mgt.deinit();

    var tokens = std.ArrayList(u32).init(gpa);
    defer tokens.deinit();

    try mgt.encode("hello world", &tokens);
    try testing.expect(tokens.items.len >= 3);

    var text = std.ArrayList(u8).init(gpa);
    defer text.deinit();

    try mgt.decode(tokens.items, &text);
    try testing.expectEqualStrings("hello world", text.items);
}

test "MGT add remove vocab" {
    const gpa = testing.allocator;

    var mgt = try MGT.init(gpa, &.{}, &.{}, null, .english);
    defer mgt.deinit();

    try mgt.addVocabWord("test", true);
    try testing.expect(mgt.anchors.contains("test"));

    mgt.removeVocabWord("test");

    try testing.expect(!mgt.anchors.contains("test"));
    try testing.expect(!mgt.token_to_id.contains("test"));
}

test "MGT longest match" {
    const gpa = testing.allocator;

    var mgt = try MGT.init(gpa, &.{ "hello", "hell" }, &.{}, null, .english);
    defer mgt.deinit();

    const len = mgt.longestMatch("hello", 0);
    try testing.expectEqual(@as(usize, 5), len);
}

test "MGT batch encode" {
    const gpa = testing.allocator;

    var mgt = try MGT.init(gpa, &.{ "a", "b" }, &.{}, null, .english);
    defer mgt.deinit();

    const texts = &.{ "a", "b" };
    const batches = try mgt.encodeBatch(texts, gpa);

    defer {
        for (batches) |batch| {
            gpa.free(batch);
        }
        gpa.free(batches);
    }

    try testing.expectEqual(@as(usize, 2), batches.len);
    try testing.expectEqual(@as(usize, 1), batches[0].len);
    try testing.expectEqual(@as(usize, 1), batches[1].len);
}

test "MGT subword split" {
    const gpa = testing.allocator;

    var mgt = try MGT.init(gpa, &.{ "hel", "lo" }, &.{}, null, .english);
    defer mgt.deinit();

    const subwords = try mgt.subwordSplit("hello");
    defer gpa.free(subwords);

    try testing.expectEqual(@as(usize, 2), subwords.len);
    try testing.expect(mgt.validateTokens(subwords));
}

test "MGT coverage" {
    const gpa = testing.allocator;

    var mgt = try MGT.init(gpa, &.{ "hello", "world", " " }, &.{}, null, .english);
    defer mgt.deinit();

    const result = mgt.coverage("hello world");
    try testing.expect(result > 0.99);
}

test "MGT BPE training" {
    const gpa = testing.allocator;
    const corpus = &.{
        "lower",
        "lowest",
        "newer",
        "wider",
        "lower",
        "lowest",
    };

    var mgt = try MGT.init(gpa, &.{}, &.{}, 512, .english);
    defer mgt.deinit();

    try mgt.trainBPE(corpus, 320);

    var encoded = std.ArrayList(u32).init(gpa);
    defer encoded.deinit();

    try mgt.encode("lower", &encoded);
    try testing.expect(encoded.items.len > 0);
    try testing.expect(mgt.validateTokens(encoded.items));

    var decoded = std.ArrayList(u8).init(gpa);
    defer decoded.deinit();

    try mgt.decode(encoded.items, &decoded);
    try testing.expectEqualStrings("lower", decoded.items);
}

test "MGT empty BPE corpus" {
    const gpa = testing.allocator;

    var mgt = try MGT.init(gpa, &.{}, &.{}, 512, .english);
    defer mgt.deinit();

    try mgt.trainBPE(&.{}, 320);
    try testing.expect(mgt.vocabSize() >= 4);
}

test "MGT BPE target vocabulary size" {
    const gpa = testing.allocator;
    const corpus = &.{
        "aaaaaaaa",
        "aaaaaaaa",
        "abababab",
        "abababab",
    };

    var mgt = try MGT.init(gpa, &.{}, &.{}, 512, .english);
    defer mgt.deinit();

    try mgt.trainBPE(corpus, 300);
    try testing.expect(mgt.vocabSize() <= 300);
}