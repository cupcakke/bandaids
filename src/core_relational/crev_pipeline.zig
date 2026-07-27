const std = @import("std");
const nsir_core = @import("nsir_core.zig");
const chaos_core = @import("chaos_core.zig");

const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Complex = std.math.Complex;

pub const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
pub const Node = nsir_core.Node;
pub const Edge = nsir_core.Edge;
pub const EdgeQuality = nsir_core.EdgeQuality;
pub const ChaosCoreKernel = chaos_core.ChaosCoreKernel;

fn clamp01(x: f64) f64 {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

fn hashTripletFields(subject: []const u8, relation: []const u8, object: []const u8, confidence: f64, extraction_time: i128) [32]u8 {
    var h = Sha256.init(.{});
    h.update(subject);
    h.update(&[_]u8{0});
    h.update(relation);
    h.update(&[_]u8{0});
    h.update(object);
    h.update(&[_]u8{0});
    const conf_bits: u64 = @bitCast(confidence);
    var conf_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &conf_le, conf_bits, .little);
    h.update(&conf_le);
    var time_le: [16]u8 = undefined;
    std.mem.writeInt(i128, &time_le, extraction_time, .little);
    h.update(&time_le);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

fn hashTripletIdentity(subject: []const u8, relation: []const u8, object: []const u8) [32]u8 {
    var h = Sha256.init(.{});
    h.update(subject);
    h.update(&[_]u8{0});
    h.update(relation);
    h.update(&[_]u8{0});
    h.update(object);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

pub const ExtractionStage = enum(u8) {
    tokenization = 0,
    triplet_extraction = 1,
    validation = 2,
    integration = 3,
    indexing = 4,

    pub fn toString(self: ExtractionStage) []const u8 {
        return switch (self) {
            .tokenization => "tokenization",
            .triplet_extraction => "triplet_extraction",
            .validation => "validation",
            .integration => "integration",
            .indexing => "indexing",
        };
    }

    pub fn fromString(s: []const u8) ?ExtractionStage {
        const t = std.mem.trim(u8, s, " \t\r\n");
        if (std.ascii.eqlIgnoreCase(t, "tokenization")) return .tokenization;
        if (std.ascii.eqlIgnoreCase(t, "triplet_extraction")) return .triplet_extraction;
        if (std.ascii.eqlIgnoreCase(t, "validation")) return .validation;
        if (std.ascii.eqlIgnoreCase(t, "integration")) return .integration;
        if (std.ascii.eqlIgnoreCase(t, "indexing")) return .indexing;
        return null;
    }

    pub fn next(self: ExtractionStage) ?ExtractionStage {
        return switch (self) {
            .tokenization => .triplet_extraction,
            .triplet_extraction => .validation,
            .validation => .integration,
            .integration => .indexing,
            .indexing => null,
        };
    }
};

const WordToken = struct {
    text: []const u8,
    start: usize,
    end: usize,
};

const MorphemeMatch = struct {
    match_start: usize,
    match_end: usize,
};

fn tokenizeIntoWords(text: []const u8, tokens: *ArrayList(WordToken)) !void {
    var i: usize = 0;
    while (i < text.len) {
        while (i < text.len and (text[i] == ' ' or text[i] == '\t' or text[i] == '\n' or text[i] == '\r' or text[i] == ',' or text[i] == ';' or text[i] == ':')) {
            i += 1;
        }
        if (i >= text.len) break;
        const start = i;
        while (i < text.len and text[i] != ' ' and text[i] != '\t' and text[i] != '\n' and text[i] != '\r' and text[i] != ',' and text[i] != ';' and text[i] != ':') {
            i += 1;
        }
        if (start < i) {
            try tokens.append(.{ .text = text[start..i], .start = start, .end = i });
        }
    }
}

fn stemWord(word: []const u8) []const u8 {
    if (word.len > 5 and std.mem.endsWith(u8, word, "ting")) {
        return word[0 .. word.len - 4];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ing")) {
        if (word.len > 5 and word[word.len - 4] == word[word.len - 5]) {
            return word[0 .. word.len - 4];
        }
        return word[0 .. word.len - 3];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ated")) {
        return word[0 .. word.len - 4];
    }
    if (word.len > 3 and std.mem.endsWith(u8, word, "ed")) {
        if (word.len > 4 and word[word.len - 3] == word[word.len - 4]) {
            return word[0 .. word.len - 3];
        }
        return word[0 .. word.len - 2];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ies")) {
        return word[0 .. word.len - 3];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ches") or (word.len > 4 and std.mem.endsWith(u8, word, "shes")) or (word.len > 4 and std.mem.endsWith(u8, word, "sses"))) {
        return word[0 .. word.len - 2];
    }
    if (word.len > 3 and std.mem.endsWith(u8, word, "es")) {
        return word[0 .. word.len - 2];
    }
    if (word.len > 2 and std.mem.endsWith(u8, word, "s") and !std.mem.endsWith(u8, word, "ss") and !std.mem.endsWith(u8, word, "us")) {
        return word[0 .. word.len - 1];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ally")) {
        return word[0 .. word.len - 4];
    }
    if (word.len > 3 and std.mem.endsWith(u8, word, "ly")) {
        return word[0 .. word.len - 2];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ment")) {
        return word[0 .. word.len - 4];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ness")) {
        return word[0 .. word.len - 4];
    }
    if (word.len > 3 and std.mem.endsWith(u8, word, "er") and !std.mem.endsWith(u8, word, "ver") and !std.mem.endsWith(u8, word, "her")) {
        if (word.len > 4 and word[word.len - 3] == word[word.len - 4]) {
            return word[0 .. word.len - 3];
        }
        return word[0 .. word.len - 2];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "est")) {
        return word[0 .. word.len - 3];
    }
    return word;
}

fn wordsMatchStem(a: []const u8, b: []const u8) bool {
    if (std.mem.eql(u8, a, b)) return true;
    var a_buf: [256]u8 = undefined;
    var b_buf: [256]u8 = undefined;
    if (a.len >= a_buf.len or b.len >= b_buf.len) return std.mem.eql(u8, a, b);
    const a_lower = std.ascii.lowerString(&a_buf, a);
    const b_lower = std.ascii.lowerString(&b_buf, b);
    if (std.mem.eql(u8, a_lower, b_lower)) return true;
    const a_stem = stemWord(a_lower);
    const b_stem = stemWord(b_lower);
    if (a_stem.len > 0 and b_stem.len > 0 and std.mem.eql(u8, a_stem, b_stem)) return true;
    return false;
}

fn matchPatternMorphemeAware(sentence: []const u8, pattern: []const u8, allocator: Allocator) ?MorphemeMatch {
    var sent_tokens = ArrayList(WordToken).init(allocator);
    defer sent_tokens.deinit();
    tokenizeIntoWords(sentence, &sent_tokens) catch return null;

    var pat_tokens = ArrayList(WordToken).init(allocator);
    defer pat_tokens.deinit();
    tokenizeIntoWords(pattern, &pat_tokens) catch return null;

    if (pat_tokens.items.len == 0 or sent_tokens.items.len == 0) return null;
    if (pat_tokens.items.len > sent_tokens.items.len) return null;

    var si: usize = 0;
    while (si <= sent_tokens.items.len - pat_tokens.items.len) {
        var all_match = true;
        for (pat_tokens.items, 0..) |pat_tok, pi| {
            if (!wordsMatchStem(sent_tokens.items[si + pi].text, pat_tok.text)) {
                all_match = false;
                break;
            }
        }
        if (all_match) {
            const last_idx = si + pat_tokens.items.len - 1;
            return MorphemeMatch{
                .match_start = sent_tokens.items[si].start,
                .match_end = sent_tokens.items[last_idx].end,
            };
        }
        si += 1;
    }
    return null;
}

pub const RelationalTriplet = struct {
    subject: []u8,
    relation: []u8,
    object: []u8,
    confidence: f64,
    source_hash: [32]u8,
    extraction_time: i128,
    allocator: Allocator,
    metadata: StringHashMap([]u8),

    pub fn init(
        allocator: Allocator,
        subject: []const u8,
        relation: []const u8,
        object: []const u8,
        confidence_in: f64,
    ) !RelationalTriplet {
        const now = @as(i64, @truncate(std.time.nanoTimestamp()));

        const s = try allocator.dupe(u8, subject);
        errdefer allocator.free(s);
        const r = try allocator.dupe(u8, relation);
        errdefer allocator.free(r);
        const o = try allocator.dupe(u8, object);
        errdefer allocator.free(o);

        return RelationalTriplet{
            .subject = s,
            .relation = r,
            .object = o,
            .confidence = clamp01(confidence_in),
            .source_hash = hashTripletIdentity(subject, relation, object),
            .extraction_time = now,
            .allocator = allocator,
            .metadata = StringHashMap([]u8).init(allocator),
        };
    }

    pub fn initWithHash(
        allocator: Allocator,
        subject: []const u8,
        relation: []const u8,
        object: []const u8,
        confidence_in: f64,
        source_hash: [32]u8,
        extraction_time: i128,
    ) !RelationalTriplet {
        const s = try allocator.dupe(u8, subject);
        errdefer allocator.free(s);
        const r = try allocator.dupe(u8, relation);
        errdefer allocator.free(r);
        const o = try allocator.dupe(u8, object);
        errdefer allocator.free(o);

        return RelationalTriplet{
            .subject = s,
            .relation = r,
            .object = o,
            .confidence = clamp01(confidence_in),
            .source_hash = source_hash,
            .extraction_time = extraction_time,
            .allocator = allocator,
            .metadata = StringHashMap([]u8).init(allocator),
        };
    }

    pub fn deinit(self: *RelationalTriplet) void {
        self.allocator.free(self.subject);
        self.allocator.free(self.relation);
        self.allocator.free(self.object);

        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();

        self.subject = &[_]u8{};
        self.relation = &[_]u8{};
        self.object = &[_]u8{};
    }

    pub fn clone(self: *const RelationalTriplet, allocator: Allocator) !RelationalTriplet {
        const s = try allocator.dupe(u8, self.subject);
        errdefer allocator.free(s);
        const r = try allocator.dupe(u8, self.relation);
        errdefer allocator.free(r);
        const o = try allocator.dupe(u8, self.object);
        errdefer allocator.free(o);

        var t = RelationalTriplet{
            .subject = s,
            .relation = r,
            .object = o,
            .confidence = self.confidence,
            .source_hash = self.source_hash,
            .extraction_time = self.extraction_time,
            .allocator = allocator,
            .metadata = StringHashMap([]u8).init(allocator),
        };
        errdefer t.deinit();

        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            const k = try allocator.dupe(u8, entry.key_ptr.*);
            errdefer allocator.free(k);
            const v = try allocator.dupe(u8, entry.value_ptr.*);
            errdefer allocator.free(v);
            try t.metadata.put(k, v);
        }
        return t;
    }

    pub fn computeHash(self: *const RelationalTriplet) [32]u8 {
        return hashTripletFields(self.subject, self.relation, self.object, self.confidence, self.extraction_time);
    }

    pub fn setMetadata(self: *RelationalTriplet, key: []const u8, value: []const u8) !void {
        const v_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(v_copy);

        if (self.metadata.getPtr(key)) |existing_v| {
            self.allocator.free(existing_v.*);
            existing_v.* = v_copy;
            return;
        }

        const k_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(k_copy);

        try self.metadata.put(k_copy, v_copy);
    }

    pub fn getMetadata(self: *const RelationalTriplet, key: []const u8) ?[]const u8 {
        if (self.metadata.get(key)) |v| return v;
        return null;
    }

    pub fn equals(self: *const RelationalTriplet, other: *const RelationalTriplet) bool {
        return std.mem.eql(u8, self.subject, other.subject) and
            std.mem.eql(u8, self.relation, other.relation) and
            std.mem.eql(u8, self.object, other.object);
    }

    pub fn hashEquals(self: *const RelationalTriplet, other: *const RelationalTriplet) bool {
        return std.mem.eql(u8, self.source_hash[0..], other.source_hash[0..]);
    }

    pub fn toGraphElements(self: *const RelationalTriplet, allocator: Allocator) !struct {
        subject_node: Node,
        object_node: Node,
        edge: Edge,
    } {
        var subject_id_hash: [32]u8 = undefined;
        Sha256.hash(self.subject, &subject_id_hash, .{});
        var subject_id: [16]u8 = undefined;
        @memcpy(subject_id[0..], subject_id_hash[0..16]);

        var object_id_hash: [32]u8 = undefined;
        Sha256.hash(self.object, &object_id_hash, .{});
        var object_id: [16]u8 = undefined;
        @memcpy(object_id[0..], object_id_hash[0..16]);

        var subject_id_str: [32]u8 = undefined;
        _ = try std.fmt.bufPrint(subject_id_str[0..], "{s}", .{std.fmt.fmtSliceHexLower(subject_id[0..])});

        var object_id_str: [32]u8 = undefined;
        _ = try std.fmt.bufPrint(object_id_str[0..], "{s}", .{std.fmt.fmtSliceHexLower(object_id[0..])});

        const c = clamp01(self.confidence);
        const imag_sq = 1.0 - c * c;
        const imag = @sqrt(@max(0.0, imag_sq));
        const quantum_state = Complex(f64).init(c, imag);

        const period_ns: i128 = 360 * 1_000_000_000;
        const mod_ns: i128 = @mod(self.extraction_time, period_ns);
        const phase = @as(f64, @floatFromInt(mod_ns)) / @as(f64, @floatFromInt(period_ns)) * std.math.pi * 2.0;

        var subject_node = try Node.initWithComplex(
            allocator,
            subject_id_str[0..],
            self.subject,
            quantum_state,
            phase,
        );
        errdefer subject_node.deinit();
        try subject_node.setMetadata("type", "entity");
        try subject_node.setMetadata("role", "subject");

        var object_node = try Node.initWithComplex(
            allocator,
            object_id_str[0..],
            self.object,
            quantum_state,
            phase,
        );
        errdefer object_node.deinit();
        try object_node.setMetadata("type", "entity");
        try object_node.setMetadata("role", "object");

        var edge = try Edge.initWithComplex(
            allocator,
            subject_id_str[0..],
            object_id_str[0..],
            .coherent,
            c,
            quantum_state,
            1.0,
        );
        errdefer edge.deinit();
        try edge.setMetadata("relation", self.relation);

        var conf_buf: [64]u8 = undefined;
        const conf_str = try std.fmt.bufPrint(conf_buf[0..], "{d:.6}", .{c});
        try edge.setMetadata("confidence", conf_str);

        return .{
            .subject_node = subject_node,
            .object_node = object_node,
            .edge = edge,
        };
    }
};

pub const ValidationResult = struct {
    triplet: *RelationalTriplet,
    is_valid: bool,
    confidence_adjusted: f64,
    validation_method: []const u8,
    conflicts: ArrayList(*RelationalTriplet),
    anomaly_score: f64,
    validation_time: i128,
    allocator: Allocator,

    pub fn init(allocator: Allocator, triplet: *RelationalTriplet) ValidationResult {
        return ValidationResult{
            .triplet = triplet,
            .is_valid = true,
            .confidence_adjusted = triplet.confidence,
            .validation_method = "",
            .conflicts = ArrayList(*RelationalTriplet).init(allocator),
            .anomaly_score = 0.0,
            .validation_time = @as(i64, @truncate(std.time.nanoTimestamp())),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ValidationResult) void {
        self.conflicts.deinit();
    }

    pub fn addConflict(self: *ValidationResult, conflict: *RelationalTriplet) !void {
        try self.conflicts.append(conflict);
    }

    pub fn hasConflicts(self: *const ValidationResult) bool {
        return self.conflicts.items.len > 0;
    }

    pub fn conflictCount(self: *const ValidationResult) usize {
        return self.conflicts.items.len;
    }

    pub fn setValidationMethod(self: *ValidationResult, method: []const u8) void {
        self.validation_method = method;
    }
};

pub const KnowledgeGraphIndex = struct {
    subject_index: StringHashMap(ArrayList(*RelationalTriplet)),
    relation_index: StringHashMap(ArrayList(*RelationalTriplet)),
    object_index: StringHashMap(ArrayList(*RelationalTriplet)),
    all_triplets: ArrayList(*RelationalTriplet),
    allocator: Allocator,

    pub fn init(allocator: Allocator) KnowledgeGraphIndex {
        return KnowledgeGraphIndex{
            .subject_index = StringHashMap(ArrayList(*RelationalTriplet)).init(allocator),
            .relation_index = StringHashMap(ArrayList(*RelationalTriplet)).init(allocator),
            .object_index = StringHashMap(ArrayList(*RelationalTriplet)).init(allocator),
            .all_triplets = ArrayList(*RelationalTriplet).init(allocator),
            .allocator = allocator,
        };
    }

    fn deinitIndexMap(self: *KnowledgeGraphIndex, map: *StringHashMap(ArrayList(*RelationalTriplet))) void {
        var it = map.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        map.deinit();
    }

    pub fn deinit(self: *KnowledgeGraphIndex) void {
        self.deinitIndexMap(&self.subject_index);
        self.deinitIndexMap(&self.relation_index);
        self.deinitIndexMap(&self.object_index);

        for (self.all_triplets.items) |triplet| {
            triplet.deinit();
            self.allocator.destroy(triplet);
        }
        self.all_triplets.deinit();
    }

    fn indexIntoMap(
        self: *KnowledgeGraphIndex,
        map: *StringHashMap(ArrayList(*RelationalTriplet)),
        key: []const u8,
        triplet: *RelationalTriplet,
    ) !void {
        const gop = try map.getOrPut(key);
        if (!gop.found_existing) {
            const key_copy = try self.allocator.dupe(u8, key);
            errdefer self.allocator.free(key_copy);
            gop.key_ptr.* = key_copy;
            gop.value_ptr.* = ArrayList(*RelationalTriplet).init(self.allocator);
        }
        try gop.value_ptr.*.append(triplet);
    }

    pub fn index(self: *KnowledgeGraphIndex, triplet: *RelationalTriplet) !void {
        try self.all_triplets.append(triplet);
        errdefer {
            _ = self.all_triplets.pop();
        }

        try self.indexIntoMap(&self.subject_index, triplet.subject, triplet);
        try self.indexIntoMap(&self.relation_index, triplet.relation, triplet);
        try self.indexIntoMap(&self.object_index, triplet.object, triplet);
    }

    pub fn query(
        self: *KnowledgeGraphIndex,
        subject: ?[]const u8,
        relation: ?[]const u8,
        object: ?[]const u8,
        allocator: Allocator,
    ) !ArrayList(*RelationalTriplet) {
        var results = ArrayList(*RelationalTriplet).init(allocator);

        if (subject == null and relation == null and object == null) {
            try results.appendSlice(self.all_triplets.items);
            return results;
        }

        var best: ?*ArrayList(*RelationalTriplet) = null;
        var best_len: usize = std.math.maxInt(usize);

        if (subject) |s| {
            if (self.subject_index.getPtr(s)) |list| {
                if (list.items.len < best_len) {
                    best = list;
                    best_len = list.items.len;
                }
            } else {
                return results;
            }
        }

        if (relation) |r| {
            if (self.relation_index.getPtr(r)) |list| {
                if (list.items.len < best_len) {
                    best = list;
                    best_len = list.items.len;
                }
            } else {
                return results;
            }
        }

        if (object) |o| {
            if (self.object_index.getPtr(o)) |list| {
                if (list.items.len < best_len) {
                    best = list;
                    best_len = list.items.len;
                }
            } else {
                return results;
            }
        }

        const cands = best orelse return results;

        for (cands.items) |t| {
            if (subject) |s| {
                if (!std.mem.eql(u8, t.subject, s)) continue;
            }
            if (relation) |r| {
                if (!std.mem.eql(u8, t.relation, r)) continue;
            }
            if (object) |o| {
                if (!std.mem.eql(u8, t.object, o)) continue;
            }
            try results.append(t);
        }

        return results;
    }

    fn queryMorphemeAware(
        self: *KnowledgeGraphIndex,
        subject: ?[]const u8,
        relation: ?[]const u8,
        object: ?[]const u8,
        allocator: Allocator,
    ) !ArrayList(*RelationalTriplet) {
        var results = ArrayList(*RelationalTriplet).init(allocator);

        for (self.all_triplets.items) |t| {
            var subj_match = true;
            var rel_match = true;
            var obj_match = true;

            if (subject) |s| {
                var s_buf: [256]u8 = undefined;
                var t_buf: [256]u8 = undefined;
                if (s.len < s_buf.len and t.subject.len < t_buf.len) {
                    const s_lower = std.ascii.lowerString(&s_buf, s);
                    const t_lower = std.ascii.lowerString(&t_buf, t.subject);
                    const s_stem = stemWord(s_lower);
                    const t_stem = stemWord(t_lower);
                    subj_match = std.mem.eql(u8, s_stem, t_stem) or std.mem.eql(u8, s, t.subject);
                } else {
                    subj_match = std.mem.eql(u8, s, t.subject);
                }
            }
            if (relation) |r| {
                var r_buf: [256]u8 = undefined;
                var t_buf: [256]u8 = undefined;
                if (r.len < r_buf.len and t.relation.len < t_buf.len) {
                    const r_lower = std.ascii.lowerString(&r_buf, r);
                    const t_lower = std.ascii.lowerString(&t_buf, t.relation);
                    const r_stem = stemWord(r_lower);
                    const t_stem = stemWord(t_lower);
                    rel_match = std.mem.eql(u8, r_stem, t_stem) or std.mem.eql(u8, r, t.relation);
                } else {
                    rel_match = std.mem.eql(u8, r, t.relation);
                }
            }
            if (object) |o| {
                var o_buf: [256]u8 = undefined;
                var t_buf: [256]u8 = undefined;
                if (o.len < o_buf.len and t.object.len < t_buf.len) {
                    const o_lower = std.ascii.lowerString(&o_buf, o);
                    const t_lower = std.ascii.lowerString(&t_buf, t.object);
                    const o_stem = stemWord(o_lower);
                    const t_stem = stemWord(t_lower);
                    obj_match = std.mem.eql(u8, o_stem, t_stem) or std.mem.eql(u8, o, t.object);
                } else {
                    obj_match = std.mem.eql(u8, o, t.object);
                }
            }

            if (subj_match and rel_match and obj_match) {
                try results.append(t);
            }
        }

        return results;
    }

    pub fn queryBySubject(self: *KnowledgeGraphIndex, subject: []const u8) []*RelationalTriplet {
        if (self.subject_index.getPtr(subject)) |list| return list.items;
        return &[_]*RelationalTriplet{};
    }

    pub fn queryByRelation(self: *KnowledgeGraphIndex, relation: []const u8) []*RelationalTriplet {
        if (self.relation_index.getPtr(relation)) |list| return list.items;
        return &[_]*RelationalTriplet{};
    }

    pub fn queryByObject(self: *KnowledgeGraphIndex, object: []const u8) []*RelationalTriplet {
        if (self.object_index.getPtr(object)) |list| return list.items;
        return &[_]*RelationalTriplet{};
    }

    fn removeFromList(list: *ArrayList(*RelationalTriplet), triplet: *RelationalTriplet) bool {
        var removed_any = false;
        var i: usize = 0;
        while (i < list.items.len) {
            if (list.items[i] == triplet) {
                _ = list.orderedRemove(i);
                removed_any = true;
            } else {
                i += 1;
            }
        }
        return removed_any;
    }

    fn removeFromMap(
        self: *KnowledgeGraphIndex,
        map: *StringHashMap(ArrayList(*RelationalTriplet)),
        key: []const u8,
        triplet: *RelationalTriplet,
    ) bool {
        if (map.getPtr(key)) |list| {
            const removed = removeFromList(list, triplet);
            if (removed and list.items.len == 0) {
                if (map.fetchRemove(key)) |kv| {
                    self.allocator.free(kv.key);
                    kv.value.deinit();
                }
            }
            return removed;
        }
        return false;
    }

    pub fn remove(self: *KnowledgeGraphIndex, triplet: *RelationalTriplet) bool {
        var removed_any = false;

        removed_any = self.removeFromMap(&self.subject_index, triplet.subject, triplet) or removed_any;
        removed_any = self.removeFromMap(&self.relation_index, triplet.relation, triplet) or removed_any;
        removed_any = self.removeFromMap(&self.object_index, triplet.object, triplet) or removed_any;

        var i: usize = 0;
        while (i < self.all_triplets.items.len) {
            if (self.all_triplets.items[i] == triplet) {
                _ = self.all_triplets.orderedRemove(i);
                removed_any = true;
            } else {
                i += 1;
            }
        }

        return removed_any;
    }

    pub fn count(self: *const KnowledgeGraphIndex) usize {
        return self.all_triplets.items.len;
    }

    pub fn getUniqueSubjects(self: *const KnowledgeGraphIndex) usize {
        return self.subject_index.count();
    }

    pub fn getUniqueRelations(self: *const KnowledgeGraphIndex) usize {
        return self.relation_index.count();
    }

    pub fn getUniqueObjects(self: *const KnowledgeGraphIndex) usize {
        return self.object_index.count();
    }
};

pub const StreamBuffer = struct {
    buffer: []?*RelationalTriplet,
    capacity: usize,
    head: usize,
    tail: usize,
    size: usize,
    allocator: Allocator,
    overflow_count: usize,
    total_pushed: usize,
    total_popped: usize,

    pub fn init(allocator: Allocator, capacity: usize) !StreamBuffer {
        const buf = try allocator.alloc(?*RelationalTriplet, capacity);
        @memset(buf, null);
        return StreamBuffer{
            .buffer = buf,
            .capacity = capacity,
            .head = 0,
            .tail = 0,
            .size = 0,
            .allocator = allocator,
            .overflow_count = 0,
            .total_pushed = 0,
            .total_popped = 0,
        };
    }

    pub fn deinit(self: *StreamBuffer) void {
        self.allocator.free(self.buffer);
        self.buffer = &[_]?*RelationalTriplet{};
        self.capacity = 0;
        self.head = 0;
        self.tail = 0;
        self.size = 0;
    }

    pub fn push(self: *StreamBuffer, triplet: *RelationalTriplet) bool {
        if (self.capacity == 0) {
            self.overflow_count += 1;
            return false;
        }
        if (self.isFull()) {
            self.overflow_count += 1;
            return false;
        }
        self.buffer[self.tail] = triplet;
        self.tail = (self.tail + 1) % self.capacity;
        self.size += 1;
        self.total_pushed += 1;
        return true;
    }

    pub fn pop(self: *StreamBuffer) ?*RelationalTriplet {
        if (self.isEmpty()) return null;
        const t = self.buffer[self.head];
        self.buffer[self.head] = null;
        self.head = (self.head + 1) % self.capacity;
        self.size -= 1;
        self.total_popped += 1;
        return t;
    }

    pub fn peek(self: *const StreamBuffer) ?*RelationalTriplet {
        if (self.isEmpty()) return null;
        return self.buffer[self.head];
    }

    pub fn peekAt(self: *const StreamBuffer, offset: usize) ?*RelationalTriplet {
        if (offset >= self.size) return null;
        if (self.capacity == 0) return null;
        const idx = (self.head + offset) % self.capacity;
        return self.buffer[idx];
    }

    pub fn isFull(self: *const StreamBuffer) bool {
        return self.capacity != 0 and self.size >= self.capacity;
    }

    pub fn isEmpty(self: *const StreamBuffer) bool {
        return self.size == 0;
    }

    pub fn getSize(self: *const StreamBuffer) usize {
        return self.size;
    }

    pub fn getCapacity(self: *const StreamBuffer) usize {
        return self.capacity;
    }

    pub fn clear(self: *StreamBuffer) void {
        if (self.capacity != 0) {
            @memset(self.buffer, null);
        }
        self.head = 0;
        self.tail = 0;
        self.size = 0;
    }

    pub fn getUtilization(self: *const StreamBuffer) f64 {
        if (self.capacity == 0) return 0.0;
        return @as(f64, @floatFromInt(self.size)) / @as(f64, @floatFromInt(self.capacity));
    }
};

pub const PipelineResult = struct {
    triplets_extracted: usize,
    triplets_validated: usize,
    triplets_integrated: usize,
    conflicts_resolved: usize,
    processing_time_ns: i128,
    stage: ExtractionStage,
    success: bool,
    error_message: ?[]const u8,

    pub fn init() PipelineResult {
        return PipelineResult{
            .triplets_extracted = 0,
            .triplets_validated = 0,
            .triplets_integrated = 0,
            .conflicts_resolved = 0,
            .processing_time_ns = 0,
            .stage = .tokenization,
            .success = true,
            .error_message = null,
        };
    }

    pub fn merge(self: *PipelineResult, other: PipelineResult) void {
        self.triplets_extracted += other.triplets_extracted;
        self.triplets_validated += other.triplets_validated;
        self.triplets_integrated += other.triplets_integrated;
        self.conflicts_resolved += other.conflicts_resolved;
        self.processing_time_ns += other.processing_time_ns;
        self.success = self.success and other.success;
        if (self.error_message == null) self.error_message = other.error_message;
    }
};

pub const PipelineStatistics = struct {
    total_extractions: usize,
    total_validations: usize,
    total_integrations: usize,
    average_confidence: f64,
    conflict_rate: f64,
    throughput: f64,
    buffer_utilization: f64,
    unique_subjects: usize,
    unique_relations: usize,
    unique_objects: usize,
    uptime_ms: i64,

    pub fn init() PipelineStatistics {
        return PipelineStatistics{
            .total_extractions = 0,
            .total_validations = 0,
            .total_integrations = 0,
            .average_confidence = 0.0,
            .conflict_rate = 0.0,
            .throughput = 0.0,
            .buffer_utilization = 0.0,
            .unique_subjects = 0,
            .unique_relations = 0,
            .unique_objects = 0,
            .uptime_ms = 0,
        };
    }
};

pub const RelationPattern = struct {
    pattern: []const u8,
    relation_type: []const u8,
    weight: f64,
};

pub const TokenizerConfig = struct {
    min_entity_length: usize,
    max_entity_length: usize,
    min_confidence_threshold: f64,
    enable_coreference: bool,
    language: []const u8,

    pub fn default() TokenizerConfig {
        return TokenizerConfig{
            .min_entity_length = 2,
            .max_entity_length = 100,
            .min_confidence_threshold = 0.3,
            .enable_coreference = true,
            .language = "en",
        };
    }
};

pub const InferenceHook = struct {
    pre_process: ?*const fn (*anyopaque, []const u8) void,
    post_process: ?*const fn (*anyopaque, *PipelineResult) void,
    pre_query: ?*const fn (*anyopaque, ?[]const u8, ?[]const u8, ?[]const u8) void,
    post_query: ?*const fn (*anyopaque, usize) void,
    context: *anyopaque,
};

pub const CREVPipeline = struct {
    kernel: *ChaosCoreKernel,
    triplet_buffer: StreamBuffer,
    knowledge_index: KnowledgeGraphIndex,
    validation_threshold: f64,
    extraction_count: usize,
    validation_count: usize,
    integration_count: usize,
    conflict_count: usize,
    allocator: Allocator,
    start_time: i128,
    total_confidence_sum: f64,
    relation_patterns: ArrayList(RelationPattern),
    tokenizer_config: TokenizerConfig,
    relation_statistics: StringHashMap(RelationStatistics),
    entity_statistics: StringHashMap(EntityStatistics),
    is_running: bool,
    inference_hooks: ArrayList(InferenceHook),

    pub const RelationStatistics = struct {
        count: usize,
        total_confidence: f64,
        m2: f64,
        avg_confidence: f64,

        pub fn init() RelationStatistics {
            return RelationStatistics{
                .count = 0,
                .total_confidence = 0.0,
                .m2 = 0.0,
                .avg_confidence = 0.0,
            };
        }

        pub fn update(self: *RelationStatistics, confidence_in: f64) void {
            const x = clamp01(confidence_in);
            self.count += 1;
            self.total_confidence += x;
            const delta = x - self.avg_confidence;
            self.avg_confidence += delta / @as(f64, @floatFromInt(self.count));
            const delta2 = x - self.avg_confidence;
            self.m2 += delta * delta2;
        }

        pub fn getVariance(self: *const RelationStatistics) f64 {
            if (self.count < 2) return 0.0;
            const v = self.m2 / @as(f64, @floatFromInt(self.count - 1));
            return @max(0.0, v);
        }

        pub fn getStdDev(self: *const RelationStatistics) f64 {
            return @sqrt(self.getVariance());
        }
    };

    pub const EntityStatistics = struct {
        count: usize,
        as_subject: usize,
        as_object: usize,
        total_confidence: f64,

        pub fn init() EntityStatistics {
            return EntityStatistics{
                .count = 0,
                .as_subject = 0,
                .as_object = 0,
                .total_confidence = 0.0,
            };
        }
    };

    pub fn init(allocator: Allocator, kernel: *ChaosCoreKernel) !CREVPipeline {
        var pipeline = CREVPipeline{
            .kernel = kernel,
            .triplet_buffer = try StreamBuffer.init(allocator, 10000),
            .knowledge_index = KnowledgeGraphIndex.init(allocator),
            .validation_threshold = 0.5,
            .extraction_count = 0,
            .validation_count = 0,
            .integration_count = 0,
            .conflict_count = 0,
            .allocator = allocator,
            .start_time = @as(i64, @truncate(std.time.nanoTimestamp())),
            .total_confidence_sum = 0.0,
            .relation_patterns = ArrayList(RelationPattern).init(allocator),
            .tokenizer_config = TokenizerConfig.default(),
            .relation_statistics = StringHashMap(RelationStatistics).init(allocator),
            .entity_statistics = StringHashMap(EntityStatistics).init(allocator),
            .is_running = true,
            .inference_hooks = ArrayList(InferenceHook).init(allocator),
        };
        errdefer pipeline.deinit();
        try pipeline.initializeDefaultPatterns();
        return pipeline;
    }

    fn initializeDefaultPatterns(self: *CREVPipeline) !void {
        try self.relation_patterns.append(.{ .pattern = " is a ", .relation_type = "is_a", .weight = 0.9 });
        try self.relation_patterns.append(.{ .pattern = " is ", .relation_type = "is", .weight = 0.7 });
        try self.relation_patterns.append(.{ .pattern = " has ", .relation_type = "has", .weight = 0.8 });
        try self.relation_patterns.append(.{ .pattern = " contains ", .relation_type = "contains", .weight = 0.85 });
        try self.relation_patterns.append(.{ .pattern = " belongs to ", .relation_type = "belongs_to", .weight = 0.85 });
        try self.relation_patterns.append(.{ .pattern = " part of ", .relation_type = "part_of", .weight = 0.85 });
        try self.relation_patterns.append(.{ .pattern = " located in ", .relation_type = "located_in", .weight = 0.8 });
        try self.relation_patterns.append(.{ .pattern = " works at ", .relation_type = "works_at", .weight = 0.8 });
        try self.relation_patterns.append(.{ .pattern = " created ", .relation_type = "created", .weight = 0.75 });
        try self.relation_patterns.append(.{ .pattern = " owns ", .relation_type = "owns", .weight = 0.8 });
        try self.relation_patterns.append(.{ .pattern = " uses ", .relation_type = "uses", .weight = 0.7 });
        try self.relation_patterns.append(.{ .pattern = " produces ", .relation_type = "produces", .weight = 0.75 });
        try self.relation_patterns.append(.{ .pattern = " causes ", .relation_type = "causes", .weight = 0.7 });
        try self.relation_patterns.append(.{ .pattern = " leads to ", .relation_type = "leads_to", .weight = 0.7 });
        try self.relation_patterns.append(.{ .pattern = " related to ", .relation_type = "related_to", .weight = 0.5 });
    }

    pub fn deinit(self: *CREVPipeline) void {
        self.is_running = false;
        self.triplet_buffer.deinit();
        self.knowledge_index.deinit();
        self.relation_patterns.deinit();

        var rel_it = self.relation_statistics.iterator();
        while (rel_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.relation_statistics.deinit();

        var ent_it = self.entity_statistics.iterator();
        while (ent_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.entity_statistics.deinit();

        self.inference_hooks.deinit();
    }

    pub fn processTextStream(self: *CREVPipeline, text: []const u8) !PipelineResult {
        const start_ns = @as(i64, @truncate(std.time.nanoTimestamp()));
        var result = PipelineResult.init();

        var triplets = try self.extractTriplets(text);
        defer triplets.deinit();

        result.triplets_extracted = triplets.items.len;
        self.extraction_count += triplets.items.len;

        for (triplets.items) |triplet| {
            var validation_result = try self.validateTriplet(triplet);
            defer validation_result.deinit();

            self.validation_count += 1;

            if (validation_result.is_valid) {
                result.triplets_validated += 1;
                triplet.confidence = clamp01(validation_result.confidence_adjusted);

                var integrated_triplet: *RelationalTriplet = triplet;

                if (validation_result.hasConflicts()) {
                    const resolved = try self.resolveConflicts(triplet, validation_result.conflicts.items);
                    result.conflicts_resolved += validation_result.conflictCount();
                    self.conflict_count += validation_result.conflictCount();

                    if (resolved != triplet) {
                        triplet.deinit();
                        self.allocator.destroy(triplet);
                        integrated_triplet = resolved;
                    }
                }

                try self.integrateTriplet(integrated_triplet);
                result.triplets_integrated += 1;
                self.integration_count += 1;
            } else {
                triplet.deinit();
                self.allocator.destroy(triplet);
            }
        }

        const end_ns = @as(i64, @truncate(std.time.nanoTimestamp()));
        result.processing_time_ns = end_ns - start_ns;
        result.stage = .indexing;
        return result;
    }

    pub fn processStructuredDataStream(self: *CREVPipeline, data: []const u8) !PipelineResult {
        const start_ns = @as(i64, @truncate(std.time.nanoTimestamp()));
        var result = PipelineResult.init();

        var triplets = try self.extractTripletsFromStructured(data);
        defer triplets.deinit();

        result.triplets_extracted = triplets.items.len;
        self.extraction_count += triplets.items.len;

        for (triplets.items) |triplet| {
            var validation_result = try self.validateTriplet(triplet);
            defer validation_result.deinit();

            self.validation_count += 1;

            if (validation_result.is_valid) {
                result.triplets_validated += 1;
                triplet.confidence = clamp01(validation_result.confidence_adjusted);

                try self.integrateTriplet(triplet);
                result.triplets_integrated += 1;
                self.integration_count += 1;
            } else {
                triplet.deinit();
                self.allocator.destroy(triplet);
            }
        }

        const end_ns = @as(i64, @truncate(std.time.nanoTimestamp()));
        result.processing_time_ns = end_ns - start_ns;
        result.stage = .indexing;
        return result;
    }

    pub fn processImageMetadataStream(self: *CREVPipeline, metadata: []const u8) !PipelineResult {
        const start_ns = @as(i64, @truncate(std.time.nanoTimestamp()));
        var result = PipelineResult.init();

        var triplets = try self.extractTripletsFromImageMetadata(metadata);
        defer triplets.deinit();

        result.triplets_extracted = triplets.items.len;
        self.extraction_count += triplets.items.len;

        for (triplets.items) |triplet| {
            var validation_result = try self.validateTriplet(triplet);
            defer validation_result.deinit();

            self.validation_count += 1;

            if (validation_result.is_valid) {
                result.triplets_validated += 1;
                triplet.confidence = clamp01(validation_result.confidence_adjusted);

                try self.integrateTriplet(triplet);
                result.triplets_integrated += 1;
                self.integration_count += 1;
            } else {
                triplet.deinit();
                self.allocator.destroy(triplet);
            }
        }

        const end_ns = @as(i64, @truncate(std.time.nanoTimestamp()));
        result.processing_time_ns = end_ns - start_ns;
        result.stage = .indexing;
        return result;
    }

    pub fn extractTriplets(self: *CREVPipeline, text: []const u8) !ArrayList(*RelationalTriplet) {
        var triplets = ArrayList(*RelationalTriplet).init(self.allocator);
        errdefer {
            for (triplets.items) |t| {
                t.deinit();
                self.allocator.destroy(t);
            }
            triplets.deinit();
        }

        var sentences = ArrayList([]const u8).init(self.allocator);
        defer sentences.deinit();

        var start: usize = 0;
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            const c = text[i];
            if (c == '.' or c == '!' or c == '?' or c == '\n') {
                if (i > start) {
                    const sentence = std.mem.trim(u8, text[start..i], " \t\r\n");
                    if (sentence.len >= self.tokenizer_config.min_entity_length) {
                        try sentences.append(sentence);
                    }
                }
                start = i + 1;
            }
        }
        if (start < text.len) {
            const sentence = std.mem.trim(u8, text[start..], " \t\r\n");
            if (sentence.len >= self.tokenizer_config.min_entity_length) {
                try sentences.append(sentence);
            }
        }

        for (sentences.items) |sentence| {
            var best_match: ?struct { match_start: usize, match_end: usize, pat: RelationPattern } = null;
            for (self.relation_patterns.items) |pattern| {
                if (matchPatternMorphemeAware(sentence, pattern.pattern, self.allocator)) |m| {
                    const match_len = m.match_end - m.match_start;
                    const best_len = if (best_match) |bm| bm.match_end - bm.match_start else @as(usize, 0);
                    if (best_match == null or match_len > best_len) {
                        best_match = .{ .match_start = m.match_start, .match_end = m.match_end, .pat = pattern };
                    }
                }
            }

            if (best_match) |m| {
                const subject = std.mem.trim(u8, sentence[0..m.match_start], " \t\r\n,;:");
                if (m.match_end < sentence.len) {
                    const object = std.mem.trim(u8, sentence[m.match_end..], " \t\r\n.,;:!?");
                    const pattern = m.pat;

                    if (subject.len >= self.tokenizer_config.min_entity_length and
                        subject.len <= self.tokenizer_config.max_entity_length and
                        object.len >= self.tokenizer_config.min_entity_length and
                        object.len <= self.tokenizer_config.max_entity_length and
                        pattern.relation_type.len > 0)
                    {
                        const confidence = clamp01(pattern.weight) * self.computeConfidence(subject, object);
                        if (confidence >= self.tokenizer_config.min_confidence_threshold) {
                            const triplet = try self.allocator.create(RelationalTriplet);
                            errdefer self.allocator.destroy(triplet);
                            triplet.* = try RelationalTriplet.init(
                                self.allocator,
                                subject,
                                pattern.relation_type,
                                object,
                                confidence,
                            );
                            try triplets.append(triplet);
                        }
                    }
                }
            }
        }

        return triplets;
    }

    fn extractTripletsFromStructured(self: *CREVPipeline, data: []const u8) !ArrayList(*RelationalTriplet) {
        var triplets = ArrayList(*RelationalTriplet).init(self.allocator);
        errdefer {
            for (triplets.items) |t| {
                t.deinit();
                self.allocator.destroy(t);
            }
            triplets.deinit();
        }

        var lines = std.mem.splitSequence(u8, data, "\n");
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;

            var parts = std.mem.splitSequence(u8, trimmed, ",");
            var fields = ArrayList([]const u8).init(self.allocator);
            defer fields.deinit();

            while (parts.next()) |part| {
                try fields.append(std.mem.trim(u8, part, " \t\""));
            }

            if (fields.items.len >= 3) {
                const conf = if (fields.items.len >= 4)
                    clamp01(std.fmt.parseFloat(f64, fields.items[3]) catch 0.8)
                else
                    0.8;

                const subj = fields.items[0];
                const rel = fields.items[1];
                const obj = fields.items[2];

                if (subj.len == 0 or rel.len == 0 or obj.len == 0) continue;

                const triplet = try self.allocator.create(RelationalTriplet);
                errdefer self.allocator.destroy(triplet);
                triplet.* = try RelationalTriplet.init(
                    self.allocator,
                    subj,
                    rel,
                    obj,
                    conf,
                );
                try triplets.append(triplet);
            }
        }

        return triplets;
    }

    fn extractTripletsFromImageMetadata(self: *CREVPipeline, metadata: []const u8) !ArrayList(*RelationalTriplet) {
        var triplets = ArrayList(*RelationalTriplet).init(self.allocator);
        errdefer {
            for (triplets.items) |t| {
                t.deinit();
                self.allocator.destroy(t);
            }
            triplets.deinit();
        }

        var lines = std.mem.splitSequence(u8, metadata, "\n");
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;

            if (std.mem.indexOfScalar(u8, trimmed, ':')) |colon_pos| {
                const key = std.mem.trim(u8, trimmed[0..colon_pos], " \t");
                const value = std.mem.trim(u8, trimmed[colon_pos + 1 ..], " \t");
                if (key.len == 0 or value.len == 0) continue;

                const triplet = try self.allocator.create(RelationalTriplet);
                errdefer self.allocator.destroy(triplet);
                triplet.* = try RelationalTriplet.init(
                    self.allocator,
                    "image",
                    key,
                    value,
                    0.9,
                );
                try triplet.setMetadata("source_type", "image_metadata");
                try triplets.append(triplet);
            }
        }

        return triplets;
    }

    fn computeConfidence(self: *CREVPipeline, subject: []const u8, object: []const u8) f64 {
        _ = self;
        var confidence: f64 = 1.0;

        const subject_len = @as(f64, @floatFromInt(subject.len));
        const object_len = @as(f64, @floatFromInt(object.len));

        if (subject_len < 3.0 or object_len < 3.0) {
            confidence *= 0.7;
        }

        if (subject_len > 50.0 or object_len > 50.0) {
            confidence *= 0.85;
        }

        var subject_upper: usize = 0;
        for (subject) |c| {
            if (c >= 'A' and c <= 'Z') subject_upper += 1;
        }
        if (subject.len > 0 and subject_upper == subject.len) {
            confidence *= 0.9;
        } else if (subject.len > 0 and subject[0] >= 'A' and subject[0] <= 'Z') {
            confidence *= 1.05;
        }

        return clamp01(confidence);
    }

    pub fn validateTriplet(self: *CREVPipeline, triplet: *RelationalTriplet) !ValidationResult {
        var result = ValidationResult.init(self.allocator, triplet);

        if (triplet.subject.len < self.tokenizer_config.min_entity_length or
            triplet.object.len < self.tokenizer_config.min_entity_length or
            triplet.relation.len == 0)
        {
            result.is_valid = false;
            result.setValidationMethod("basic_checks");
            return result;
        }

        triplet.confidence = clamp01(triplet.confidence);

        if (triplet.confidence < self.validation_threshold) {
            result.is_valid = false;
            result.confidence_adjusted = triplet.confidence;
            result.setValidationMethod("confidence_threshold");
            return result;
        }

        var existing_triplets = try self.knowledge_index.query(triplet.subject, null, triplet.object, self.allocator);
        defer existing_triplets.deinit();

        for (existing_triplets.items) |existing| {
            if (!self.checkConsistency(triplet, existing)) {
                try result.addConflict(existing);
            }
        }

        result.anomaly_score = try self.computeAnomalyScore(triplet);

        if (result.anomaly_score > 0.85) {
            result.is_valid = false;
            result.setValidationMethod("anomaly_detection");
            return result;
        }

        var adjusted = triplet.confidence * (1.0 - result.anomaly_score * 0.3);
        if (result.hasConflicts()) adjusted *= 0.9;
        result.confidence_adjusted = clamp01(adjusted);

        result.setValidationMethod("full_validation");
        return result;
    }

    fn computeAnomalyScore(self: *CREVPipeline, triplet: *RelationalTriplet) !f64 {
        var weighted_sum: f64 = 0.0;
        var total_weight: f64 = 0.0;

        if (self.relation_statistics.get(triplet.relation)) |stats| {
            if (stats.count > 10) {
                const std_dev = stats.getStdDev();
                if (std_dev > 0.0) {
                    const z = @abs(triplet.confidence - stats.avg_confidence) / std_dev;
                    const a = @min(1.0, z / 3.0);
                    const w = 0.3;
                    weighted_sum += a * w;
                    total_weight += w;
                }
            }
        }

        const subject_known = self.entity_statistics.contains(triplet.subject);
        const object_known = self.entity_statistics.contains(triplet.object);

        if (!subject_known and !object_known) {
            const w = 0.4;
            weighted_sum += 1.0 * w;
            total_weight += w;
        } else if (!subject_known or !object_known) {
            const w = 0.2;
            weighted_sum += 1.0 * w;
            total_weight += w;
        }

        if (!self.relation_statistics.contains(triplet.relation)) {
            const w = 0.15;
            weighted_sum += 1.0 * w;
            total_weight += w;
        }

        if (total_weight == 0.0) return 0.0;
        return clamp01(weighted_sum / total_weight);
    }

    pub fn checkConsistency(self: *CREVPipeline, triplet: *RelationalTriplet, existing: *RelationalTriplet) bool {
        _ = self;

        if (std.mem.eql(u8, triplet.relation, existing.relation)) {
            return true;
        }

        const contradicting_pairs = [_][2][]const u8{
            .{ "is_a", "is_not" },
            .{ "has", "lacks" },
            .{ "owns", "does_not_own" },
            .{ "contains", "excludes" },
            .{ "causes", "prevents" },
        };

        for (contradicting_pairs) |pair| {
            if ((std.mem.eql(u8, triplet.relation, pair[0]) and std.mem.eql(u8, existing.relation, pair[1])) or
                (std.mem.eql(u8, triplet.relation, pair[1]) and std.mem.eql(u8, existing.relation, pair[0])))
            {
                return false;
            }
        }

        return true;
    }

    pub fn resolveConflicts(
        self: *CREVPipeline,
        triplet: *RelationalTriplet,
        conflicts: []*RelationalTriplet,
    ) !*RelationalTriplet {
        if (conflicts.len == 0) return triplet;

        var best = triplet;
        for (conflicts) |c| {
            if (c.confidence > best.confidence) best = c;
        }

        if (best == triplet) return triplet;

        const new_triplet = try self.allocator.create(RelationalTriplet);
        errdefer self.allocator.destroy(new_triplet);
        new_triplet.* = try best.clone(self.allocator);

        const a = clamp01(triplet.confidence);
        const b = clamp01(best.confidence);
        const denom = a + b;
        new_triplet.confidence = if (denom > 0.0) (a * a + b * b) / denom else b;
        new_triplet.confidence = clamp01(new_triplet.confidence);

        return new_triplet;
    }

    pub fn integrateTriplet(self: *CREVPipeline, triplet: *RelationalTriplet) !void {
        try self.knowledge_index.index(triplet);
        self.total_confidence_sum += triplet.confidence;

        try self.updateStatistics(triplet);

        if (!self.triplet_buffer.push(triplet)) {
            _ = self.triplet_buffer.pop();
            _ = self.triplet_buffer.push(triplet);
        }

        const data = try std.fmt.allocPrint(self.allocator, "{s}|{s}|{s}|{d:.6}", .{
            triplet.subject,
            triplet.relation,
            triplet.object,
            triplet.confidence,
        });
        defer self.allocator.free(data);

        _ = try self.kernel.allocateMemory(data, null);
    }

    fn updateStatistics(self: *CREVPipeline, triplet: *RelationalTriplet) !void {
        const rel_gop = try self.relation_statistics.getOrPut(triplet.relation);
        if (!rel_gop.found_existing) {
            const key_copy = try self.allocator.dupe(u8, triplet.relation);
            errdefer self.allocator.free(key_copy);
            rel_gop.key_ptr.* = key_copy;
            rel_gop.value_ptr.* = RelationStatistics.init();
        }
        rel_gop.value_ptr.*.update(triplet.confidence);

        const subj_gop = try self.entity_statistics.getOrPut(triplet.subject);
        if (!subj_gop.found_existing) {
            const key_copy = try self.allocator.dupe(u8, triplet.subject);
            errdefer self.allocator.free(key_copy);
            subj_gop.key_ptr.* = key_copy;
            subj_gop.value_ptr.* = EntityStatistics.init();
        }
        subj_gop.value_ptr.*.count += 1;
        subj_gop.value_ptr.*.as_subject += 1;
        subj_gop.value_ptr.*.total_confidence += triplet.confidence;

        const obj_gop = try self.entity_statistics.getOrPut(triplet.object);
        if (!obj_gop.found_existing) {
            const key_copy = try self.allocator.dupe(u8, triplet.object);
            errdefer self.allocator.free(key_copy);
            obj_gop.key_ptr.* = key_copy;
            obj_gop.value_ptr.* = EntityStatistics.init();
        }
        obj_gop.value_ptr.*.count += 1;
        obj_gop.value_ptr.*.as_object += 1;
        obj_gop.value_ptr.*.total_confidence += triplet.confidence;
    }

    pub fn queryKnowledgeGraph(
        self: *CREVPipeline,
        subject: ?[]const u8,
        relation: ?[]const u8,
        object: ?[]const u8,
    ) !ArrayList(*RelationalTriplet) {
        return self.knowledge_index.query(subject, relation, object, self.allocator);
    }

    pub fn getPipelineStatistics(self: *CREVPipeline) PipelineStatistics {
        const uptime_ns = @as(i64, @truncate(std.time.nanoTimestamp())) - self.start_time;
        const uptime_ms_i64: i64 = @as(i64, @intCast(@max(@as(i128, 0), @divTrunc(uptime_ns, 1_000_000))));
        const uptime_sec = @as(f64, @floatFromInt(@max(@as(i128, 1), uptime_ns))) / 1_000_000_000.0;

        return PipelineStatistics{
            .total_extractions = self.extraction_count,
            .total_validations = self.validation_count,
            .total_integrations = self.integration_count,
            .average_confidence = if (self.integration_count > 0)
                self.total_confidence_sum / @as(f64, @floatFromInt(self.integration_count))
            else
                0.0,
            .conflict_rate = if (self.validation_count > 0)
                @as(f64, @floatFromInt(self.conflict_count)) / @as(f64, @floatFromInt(self.validation_count))
            else
                0.0,
            .throughput = @as(f64, @floatFromInt(self.integration_count)) / uptime_sec,
            .buffer_utilization = self.triplet_buffer.getUtilization(),
            .unique_subjects = self.knowledge_index.getUniqueSubjects(),
            .unique_relations = self.knowledge_index.getUniqueRelations(),
            .unique_objects = self.knowledge_index.getUniqueObjects(),
            .uptime_ms = uptime_ms_i64,
        };
    }

    pub fn shutdown(self: *CREVPipeline) void {
        self.is_running = false;
        self.triplet_buffer.clear();
    }

    pub fn addRelationPattern(self: *CREVPipeline, pattern: []const u8, relation_type: []const u8, weight_in: f64) !void {
        const p_copy = try self.allocator.dupe(u8, pattern);
        errdefer self.allocator.free(p_copy);
        const r_copy = try self.allocator.dupe(u8, relation_type);
        errdefer self.allocator.free(r_copy);

        try self.relation_patterns.append(.{
            .pattern = p_copy,
            .relation_type = r_copy,
            .weight = clamp01(weight_in),
        });
    }

    pub fn setValidationThreshold(self: *CREVPipeline, threshold: f64) void {
        self.validation_threshold = clamp01(threshold);
    }

    pub fn getKnowledgeGraphSize(self: *CREVPipeline) usize {
        return self.knowledge_index.count();
    }

    pub fn registerInferenceHook(self: *CREVPipeline, hook: InferenceHook) !void {
        try self.inference_hooks.append(hook);
    }

    pub fn clearInferenceHooks(self: *CREVPipeline) void {
        self.inference_hooks.clearRetainingCapacity();
    }

    pub fn processInferenceText(self: *CREVPipeline, text: []const u8) !PipelineResult {
        for (self.inference_hooks.items) |hook| {
            if (hook.pre_process) |cb| {
                cb(hook.context, text);
            }
        }

        var result = try self.processTextStream(text);

        for (self.inference_hooks.items) |hook| {
            if (hook.post_process) |cb| {
                cb(hook.context, &result);
            }
        }

        return result;
    }

    pub fn queryInferenceKnowledge(self: *CREVPipeline, subject: ?[]const u8, relation: ?[]const u8, object: ?[]const u8) !ArrayList(*RelationalTriplet) {
        for (self.inference_hooks.items) |hook| {
            if (hook.pre_query) |cb| {
                cb(hook.context, subject, relation, object);
            }
        }

        const results = try self.knowledge_index.queryMorphemeAware(subject, relation, object, self.allocator);

        for (self.inference_hooks.items) |hook| {
            if (hook.post_query) |cb| {
                cb(hook.context, results.items.len);
            }
        }

        return results;
    }

    pub fn getInferencePipelineStatistics(self: *CREVPipeline) PipelineStatistics {
        return self.getPipelineStatistics();
    }

    pub fn isRunning(self: *const CREVPipeline) bool {
        return self.is_running;
    }
};

test "ExtractionStage toString and fromString" {
    const testing = std.testing;

    try testing.expectEqualStrings("tokenization", ExtractionStage.tokenization.toString());
    try testing.expectEqualStrings("triplet_extraction", ExtractionStage.triplet_extraction.toString());
    try testing.expectEqualStrings("validation", ExtractionStage.validation.toString());
    try testing.expectEqualStrings("integration", ExtractionStage.integration.toString());
    try testing.expectEqualStrings("indexing", ExtractionStage.indexing.toString());

    try testing.expectEqual(ExtractionStage.tokenization, ExtractionStage.fromString("tokenization").?);
    try testing.expectEqual(ExtractionStage.validation, ExtractionStage.fromString("validation").?);
    try testing.expect(ExtractionStage.fromString("invalid") == null);
}

test "ExtractionStage next" {
    const testing = std.testing;

    try testing.expectEqual(ExtractionStage.triplet_extraction, ExtractionStage.tokenization.next().?);
    try testing.expectEqual(ExtractionStage.validation, ExtractionStage.triplet_extraction.next().?);
    try testing.expectEqual(ExtractionStage.integration, ExtractionStage.validation.next().?);
    try testing.expectEqual(ExtractionStage.indexing, ExtractionStage.integration.next().?);
    try testing.expect(ExtractionStage.indexing.next() == null);
}

test "RelationalTriplet initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var triplet = try RelationalTriplet.init(allocator, "Alice", "knows", "Bob", 0.9);
    defer triplet.deinit();

    try testing.expectEqualStrings("Alice", triplet.subject);
    try testing.expectEqualStrings("knows", triplet.relation);
    try testing.expectEqualStrings("Bob", triplet.object);
    try testing.expectApproxEqAbs(@as(f64, 0.9), triplet.confidence, 0.001);
}

test "RelationalTriplet clone" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var original = try RelationalTriplet.init(allocator, "Paris", "is_a", "City", 0.95);
    defer original.deinit();

    try original.setMetadata("source", "test");

    var cloned = try original.clone(allocator);
    defer cloned.deinit();

    try testing.expectEqualStrings(original.subject, cloned.subject);
    try testing.expectEqualStrings(original.relation, cloned.relation);
    try testing.expectEqualStrings(original.object, cloned.object);
    try testing.expectApproxEqAbs(original.confidence, cloned.confidence, 0.001);
    try testing.expectEqualStrings("test", cloned.getMetadata("source").?);
}

test "RelationalTriplet computeHash" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var triplet1 = try RelationalTriplet.init(allocator, "A", "B", "C", 0.5);
    defer triplet1.deinit();

    var triplet2 = try RelationalTriplet.init(allocator, "A", "B", "C", 0.5);
    defer triplet2.deinit();

    const hash1 = triplet1.computeHash();
    const hash2 = triplet2.computeHash();

    try testing.expect(hash1.len == 32);
    try testing.expect(hash2.len == 32);
}

test "RelationalTriplet equals" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var triplet1 = try RelationalTriplet.init(allocator, "A", "rel", "B", 0.9);
    defer triplet1.deinit();

    var triplet2 = try RelationalTriplet.init(allocator, "A", "rel", "B", 0.8);
    defer triplet2.deinit();

    var triplet3 = try RelationalTriplet.init(allocator, "A", "different", "B", 0.9);
    defer triplet3.deinit();

    try testing.expect(triplet1.equals(&triplet2));
    try testing.expect(!triplet1.equals(&triplet3));
}

test "KnowledgeGraphIndex initialization and indexing" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var index = KnowledgeGraphIndex.init(allocator);
    defer index.deinit();

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "Entity1", "related_to", "Entity2", 0.8);

    try index.index(triplet);

    try testing.expectEqual(@as(usize, 1), index.count());
}

test "KnowledgeGraphIndex query" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var index = KnowledgeGraphIndex.init(allocator);
    defer index.deinit();

    const triplet1 = try allocator.create(RelationalTriplet);
    triplet1.* = try RelationalTriplet.init(allocator, "Alice", "knows", "Bob", 0.9);
    try index.index(triplet1);

    const triplet2 = try allocator.create(RelationalTriplet);
    triplet2.* = try RelationalTriplet.init(allocator, "Alice", "works_at", "Company", 0.85);
    try index.index(triplet2);

    var results = try index.query("Alice", null, null, allocator);
    defer results.deinit();

    try testing.expectEqual(@as(usize, 2), results.items.len);
}

test "KnowledgeGraphIndex queryBySubject" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var index = KnowledgeGraphIndex.init(allocator);
    defer index.deinit();

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "TestSubject", "has", "TestObject", 0.7);
    try index.index(triplet);

    const results = index.queryBySubject("TestSubject");
    try testing.expectEqual(@as(usize, 1), results.len);

    const empty_results = index.queryBySubject("NonExistent");
    try testing.expectEqual(@as(usize, 0), empty_results.len);
}

test "KnowledgeGraphIndex remove" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var index = KnowledgeGraphIndex.init(allocator);
    defer index.deinit();

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "ToRemove", "relation", "Target", 0.6);
    try index.index(triplet);

    try testing.expectEqual(@as(usize, 1), index.count());

    const removed = index.remove(triplet);
    try testing.expect(removed);
    try testing.expectEqual(@as(usize, 0), index.count());

    triplet.deinit();
    allocator.destroy(triplet);
}

test "StreamBuffer push and pop" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buffer = try StreamBuffer.init(allocator, 5);
    defer buffer.deinit();

    try testing.expect(buffer.isEmpty());
    try testing.expect(!buffer.isFull());

    const triplet1 = try allocator.create(RelationalTriplet);
    triplet1.* = try RelationalTriplet.init(allocator, "A", "B", "C", 0.5);
    const ok = buffer.push(triplet1);
    try testing.expect(ok);

    try testing.expect(!buffer.isEmpty());
    try testing.expectEqual(@as(usize, 1), buffer.getSize());

    const popped = buffer.pop();
    try testing.expect(popped != null);
    try testing.expectEqualStrings("A", popped.?.subject);
    try testing.expect(buffer.isEmpty());

    popped.?.deinit();
    allocator.destroy(popped.?);
}

test "StreamBuffer capacity" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buffer = try StreamBuffer.init(allocator, 3);
    defer buffer.deinit();

    const triplet1 = try allocator.create(RelationalTriplet);
    triplet1.* = try RelationalTriplet.init(allocator, "1", "r", "a", 0.5);
    try testing.expect(buffer.push(triplet1));

    const triplet2 = try allocator.create(RelationalTriplet);
    triplet2.* = try RelationalTriplet.init(allocator, "2", "r", "b", 0.5);
    try testing.expect(buffer.push(triplet2));

    const triplet3 = try allocator.create(RelationalTriplet);
    triplet3.* = try RelationalTriplet.init(allocator, "3", "r", "c", 0.5);
    try testing.expect(buffer.push(triplet3));

    try testing.expect(buffer.isFull());

    const triplet4 = try allocator.create(RelationalTriplet);
    triplet4.* = try RelationalTriplet.init(allocator, "4", "r", "d", 0.5);
    const success = buffer.push(triplet4);
    try testing.expect(!success);
    try testing.expectEqual(@as(usize, 1), buffer.overflow_count);

    triplet4.deinit();
    allocator.destroy(triplet4);

    while (buffer.pop()) |t| {
        t.deinit();
        allocator.destroy(t);
    }
}

test "StreamBuffer peek" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buffer = try StreamBuffer.init(allocator, 5);
    defer buffer.deinit();

    try testing.expect(buffer.peek() == null);

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "Peek", "test", "value", 0.7);
    try testing.expect(buffer.push(triplet));

    const peeked = buffer.peek();
    try testing.expect(peeked != null);
    try testing.expectEqualStrings("Peek", peeked.?.subject);
    try testing.expectEqual(@as(usize, 1), buffer.getSize());

    const popped = buffer.pop().?;
    popped.deinit();
    allocator.destroy(popped);
}

test "PipelineResult merge" {
    const testing = std.testing;

    var result1 = PipelineResult.init();
    result1.triplets_extracted = 10;
    result1.triplets_validated = 8;
    result1.triplets_integrated = 7;

    var result2 = PipelineResult.init();
    result2.triplets_extracted = 5;
    result2.triplets_validated = 4;
    result2.triplets_integrated = 3;

    result1.merge(result2);

    try testing.expectEqual(@as(usize, 15), result1.triplets_extracted);
    try testing.expectEqual(@as(usize, 12), result1.triplets_validated);
    try testing.expectEqual(@as(usize, 10), result1.triplets_integrated);
}

test "CREVPipeline initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    try testing.expect(pipeline.is_running);
    try testing.expectEqual(@as(usize, 0), pipeline.extraction_count);
    try testing.expectApproxEqAbs(@as(f64, 0.5), pipeline.validation_threshold, 0.001);
}

test "CREVPipeline extractTriplets" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const text = "Paris is a city. The Eiffel Tower is located in Paris.";
    var triplets = try pipeline.extractTriplets(text);
    defer {
        for (triplets.items) |triplet| {
            triplet.deinit();
            allocator.destroy(triplet);
        }
        triplets.deinit();
    }

    try testing.expect(triplets.items.len > 0);
}

test "CREVPipeline processTextStream" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const text = "Python is a programming language. Python has modules.";
    const result = try pipeline.processTextStream(text);

    try testing.expect(result.triplets_extracted > 0);
    try testing.expect(result.processing_time_ns >= 0);
}

test "CREVPipeline processStructuredDataStream" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const data = "Alice,knows,Bob,0.9\nBob,works_at,Company,0.85";
    const result = try pipeline.processStructuredDataStream(data);

    try testing.expect(result.triplets_extracted == 2);
}

test "CREVPipeline validateTriplet" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    pipeline.setValidationThreshold(0.4);

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "TestSubjectEntity", "is_a", "TestObjectEntity", 0.95);
    defer {
        triplet.deinit();
        allocator.destroy(triplet);
    }

    var result = try pipeline.validateTriplet(triplet);
    defer result.deinit();

    try testing.expect(result.confidence_adjusted > 0);
}

test "CREVPipeline checkConsistency" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    var triplet1 = try RelationalTriplet.init(allocator, "A", "is_a", "B", 0.9);
    defer triplet1.deinit();

    var triplet2 = try RelationalTriplet.init(allocator, "A", "is_a", "B", 0.8);
    defer triplet2.deinit();

    try testing.expect(pipeline.checkConsistency(&triplet1, &triplet2));

    var triplet3 = try RelationalTriplet.init(allocator, "A", "is_not", "B", 0.7);
    defer triplet3.deinit();

    try testing.expect(!pipeline.checkConsistency(&triplet1, &triplet3));
}

test "CREVPipeline getPipelineStatistics" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const stats = pipeline.getPipelineStatistics();

    try testing.expectEqual(@as(usize, 0), stats.total_extractions);
    try testing.expectEqual(@as(usize, 0), stats.total_validations);
    try testing.expect(stats.uptime_ms >= 0);
}

test "CREVPipeline queryKnowledgeGraph" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "DirectEntity", "has_property", "TestProperty", 0.9);
    try pipeline.knowledge_index.index(triplet);

    var results = try pipeline.queryKnowledgeGraph("DirectEntity", null, null);
    defer results.deinit();

    try testing.expect(results.items.len > 0);
}

test "CREVPipeline shutdown" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = chaos_core.ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    try testing.expect(pipeline.isRunning());
    pipeline.shutdown();
    try testing.expect(!pipeline.isRunning());
}

test "ValidationResult initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var triplet = try RelationalTriplet.init(allocator, "S", "R", "O", 0.75);
    defer triplet.deinit();

    var result = ValidationResult.init(allocator, &triplet);
    defer result.deinit();

    try testing.expect(result.is_valid);
    try testing.expect(!result.hasConflicts());
    try testing.expectEqual(@as(usize, 0), result.conflictCount());
}

test "RelationStatistics update" {
    const testing = std.testing;

    var stats = CREVPipeline.RelationStatistics.init();

    stats.update(0.8);
    try testing.expectEqual(@as(usize, 1), stats.count);
    try testing.expectApproxEqAbs(@as(f64, 0.8), stats.avg_confidence, 0.001);

    stats.update(0.6);
    try testing.expectEqual(@as(usize, 2), stats.count);
    try testing.expectApproxEqAbs(@as(f64, 0.7), stats.avg_confidence, 0.001);

    try testing.expect(stats.getVariance() >= 0);
    try testing.expect(stats.getStdDev() >= 0);
}

test "StreamBuffer utilization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buffer = try StreamBuffer.init(allocator, 10);
    defer buffer.deinit();

    try testing.expectApproxEqAbs(@as(f64, 0.0), buffer.getUtilization(), 0.001);

    const triplet1 = try allocator.create(RelationalTriplet);
    triplet1.* = try RelationalTriplet.init(allocator, "1", "r", "a", 0.5);
    try testing.expect(buffer.push(triplet1));

    const triplet2 = try allocator.create(RelationalTriplet);
    triplet2.* = try RelationalTriplet.init(allocator, "2", "r", "b", 0.5);
    try testing.expect(buffer.push(triplet2));

    try testing.expectApproxEqAbs(@as(f64, 0.2), buffer.getUtilization(), 0.001);

    while (buffer.pop()) |t| {
        t.deinit();
        allocator.destroy(t);
    }
}
