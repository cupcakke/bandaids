const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const types = @import("../core/types.zig");
const BitSet = types.BitSet;
const Tensor = @import("../core/tensor.zig").Tensor;
const SSI = @import("../index/ssi.zig").SSI;
const stableHash = @import("../core/io.zig").stableHash;
const Error = types.Error;

pub const RankerConfig = struct {
    pub const STREAMING_BUFFER_SIZE: usize = 1024;
    pub const STREAMING_WINDOW_SIZE: usize = 512;
    pub const DEFAULT_TOP_N_RETRIEVAL: usize = 1000;
    pub const HASH_SEED_MULTIPLIER_A: u64 = 0x9e3779b97f4a7c15;
    pub const HASH_SEED_MULTIPLIER_B: u64 = 0x517cc1b727220a95;
    pub const LEARNING_RATE: f32 = 0.01;
    pub const DIVERSITY_WEIGHT: f32 = 0.3;
    pub const PROXIMITY_WEIGHT: f32 = 0.3;
    pub const MAX_RAW_SCORE: f32 = 100.0;
    pub const BASE_SCORE_WEIGHT: f32 = 0.4;
    pub const OVERLAP_WEIGHT: f32 = 0.3;
    pub const JACCARD_WEIGHT: f32 = 0.3;
};

fn tokenToLEBytes(token: u32) [4]u8 {
    return mem.toBytes(mem.nativeToLittle(u32, token));
}

fn tokensToLEBytes(allocator: Allocator, tokens: []const u32) ![]u8 {
    const buf = try allocator.alloc(u8, tokens.len * 4);
    for (tokens, 0..) |token, i| {
        const le = tokenToLEBytes(token);
        buf[i * 4 + 0] = le[0];
        buf[i * 4 + 1] = le[1];
        buf[i * 4 + 2] = le[2];
        buf[i * 4 + 3] = le[3];
    }
    return buf;
}

fn encodeNgramLE(ngram: []const u32, buf: []u8) void {
    for (ngram, 0..) |token, i| {
        const le = tokenToLEBytes(token);
        buf[i * 4 + 0] = le[0];
        buf[i * 4 + 1] = le[1];
        buf[i * 4 + 2] = le[2];
        buf[i * 4 + 3] = le[3];
    }
}

pub const Ranker = struct {
    ngram_weights: []f32,
    lsh_hash_params: []u64,
    num_hash_functions: usize,
    num_ngrams: usize,
    seed: u64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, num_ngrams: usize, num_hash_funcs: usize, seed: u64) !Ranker {
        if (num_ngrams == 0) return error.InvalidParameter;
        if (num_hash_funcs == 0) return error.InvalidParameter;

        const weights = try allocator.alloc(f32, num_ngrams);
        errdefer allocator.free(weights);

        var i: usize = 0;
        while (i < weights.len) : (i += 1) {
            const decay = 1.0 / @as(f32, @floatFromInt(i + 1));
            weights[i] = decay;
        }

        const hash_params = try allocator.alloc(u64, num_hash_funcs * 2);
        errdefer allocator.free(hash_params);

        i = 0;
        while (i < num_hash_funcs) : (i += 1) {
            const i_u64: u64 = @intCast(i);
            const i_plus_one: u64 = @intCast(i + 1);
            hash_params[i * 2] = seed +% (i_u64 *% RankerConfig.HASH_SEED_MULTIPLIER_A);
            hash_params[i * 2 + 1] = seed +% (i_plus_one *% RankerConfig.HASH_SEED_MULTIPLIER_B);
        }

        return .{
            .ngram_weights = weights,
            .lsh_hash_params = hash_params,
            .num_hash_functions = num_hash_funcs,
            .num_ngrams = num_ngrams,
            .seed = seed,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Ranker) void {
        self.allocator.free(self.ngram_weights);
        self.allocator.free(self.lsh_hash_params);
    }

    pub fn scoreSequence(self: *const Ranker, tokens: []const u32, ssi: *const SSI) !f32 {
        if (tokens.len == 0) return 0.0;

        var ngram_stack_buf: [1024]u8 = undefined;
        const max_ngram_bytes = @min(self.num_ngrams * 4, ngram_stack_buf.len);

        var ngram_score: f32 = 0.0;
        var gram: usize = 1;
        while (gram <= @min(self.num_ngrams, tokens.len)) : (gram += 1) {
            var start: usize = 0;
            while (start <= tokens.len - gram) : (start += 1) {
                const ngram = tokens[start .. start + gram];
                const needed = ngram.len * 4;
                if (needed > max_ngram_bytes) continue;
                encodeNgramLE(ngram, ngram_stack_buf[0..needed]);
                const h = stableHash(ngram_stack_buf[0..needed], self.seed);
                if (ssi.getSegment(h)) |s| {
                    if (!math.isNan(s.score) and !math.isInf(s.score)) {
                        const weight_idx = @min(gram - 1, self.ngram_weights.len - 1);
                        ngram_score += self.ngram_weights[weight_idx] * s.score;
                    }
                }
            }
        }

        const diversity_score = try self.computeTokenDiversity(tokens);
        const proximity = self.anchorProximity(tokens, ssi);

        var raw_score = ngram_score + RankerConfig.DIVERSITY_WEIGHT * diversity_score + RankerConfig.PROXIMITY_WEIGHT * proximity;
        raw_score = math.clamp(raw_score, 0.0, RankerConfig.MAX_RAW_SCORE);
        return raw_score / RankerConfig.MAX_RAW_SCORE;
    }

    pub fn scoreSequenceWithQuery(self: *const Ranker, tokens: []const u32, query: []const u32, ssi: *const SSI) !f32 {
        const base_score = try self.scoreSequence(tokens, ssi);

        const token_overlap = try self.computeTokenOverlap(tokens, query);
        const jaccard = try self.computeJaccardSimilarity(tokens, query);

        const combined_score = base_score * RankerConfig.BASE_SCORE_WEIGHT + token_overlap * RankerConfig.OVERLAP_WEIGHT + jaccard * RankerConfig.JACCARD_WEIGHT;
        return math.clamp(combined_score, 0.0, 1.0);
    }

    fn computeTokenDiversity(self: *const Ranker, tokens: []const u32) !f32 {
        if (tokens.len == 0) return 0.0;

        var unique_tokens = std.AutoHashMap(u32, void).init(self.allocator);
        defer unique_tokens.deinit();

        for (tokens) |token| {
            try unique_tokens.put(token, {});
        }

        const unique_count = unique_tokens.count();
        const diversity = @as(f32, @floatFromInt(unique_count)) / @as(f32, @floatFromInt(tokens.len));

        return diversity;
    }

    fn computeTokenOverlap(self: *const Ranker, tokens: []const u32, query: []const u32) !f32 {
        if (tokens.len == 0 or query.len == 0) return 0.0;

        var token_set = std.AutoHashMap(u32, void).init(self.allocator);
        defer token_set.deinit();

        for (tokens) |token| {
            try token_set.put(token, {});
        }

        var query_set = std.AutoHashMap(u32, void).init(self.allocator);
        defer query_set.deinit();

        for (query) |qtoken| {
            try query_set.put(qtoken, {});
        }

        var overlap: usize = 0;
        var it = token_set.keyIterator();
        while (it.next()) |key| {
            if (query_set.contains(key.*)) {
                overlap += 1;
            }
        }

        const max_unique = @max(token_set.count(), query_set.count());
        if (max_unique == 0) return 0.0;
        return @as(f32, @floatFromInt(overlap)) / @as(f32, @floatFromInt(max_unique));
    }

    fn computeJaccardSimilarity(self: *const Ranker, tokens: []const u32, query: []const u32) !f32 {
        if (tokens.len == 0 and query.len == 0) return 1.0;
        if (tokens.len == 0 or query.len == 0) return 0.0;

        var intersection: usize = 0;
        var union_size: usize = 0;

        var seen = std.AutoHashMap(u32, u8).init(self.allocator);
        defer seen.deinit();

        for (tokens) |token| {
            try seen.put(token, 1);
        }

        for (query) |qtoken| {
            if (seen.get(qtoken)) |val| {
                if (val == 1) {
                    try seen.put(qtoken, 2);
                }
            } else {
                try seen.put(qtoken, 1);
            }
        }

        var it = seen.iterator();
        while (it.next()) |entry| {
            union_size += 1;
            if (entry.value_ptr.* == 2) {
                intersection += 1;
            }
        }

        if (union_size == 0) return 0.0;
        return @as(f32, @floatFromInt(intersection)) / @as(f32, @floatFromInt(union_size));
    }

    fn anchorProximity(self: *const Ranker, tokens: []const u32, ssi: *const SSI) f32 {
        if (tokens.len == 0) return 0.0;

        var anchors: usize = 0;
        var total_dist: f32 = 0.0;
        var i: usize = 0;
        while (i < tokens.len) : (i += 1) {
            const le_bytes = tokenToLEBytes(tokens[i]);
            const h = stableHash(&le_bytes, self.seed);
            if (ssi.getSegment(h)) |s| {
                if (s.anchor_hash != 0) {
                    anchors += 1;
                    const pos_u64: u64 = @intCast(i);
                    const raw_dist: u64 = if (pos_u64 >= s.position) pos_u64 - s.position else s.position - pos_u64;
                    const clamped_dist: u64 = @min(raw_dist, std.math.maxInt(u32));
                    const dist: f32 = @floatFromInt(clamped_dist);
                    total_dist += dist;
                }
            }
        }
        if (anchors == 0) return 0.0;
        const denominator: u64 = math.mul(u64, @as(u64, anchors), @as(u64, tokens.len)) catch std.math.maxInt(u64);
        const denom_clamped: u64 = @min(denominator, std.math.maxInt(u32));
        const denom_f32: f32 = @floatFromInt(@max(denom_clamped, 1));
        return 1.0 - math.clamp(total_dist / denom_f32, 0.0, 1.0);
    }

    pub fn rankCandidates(self: *const Ranker, candidates: []types.RankedSegment, ssi: *const SSI, allocator: Allocator) !void {
        return self.rankCandidatesWithQuery(candidates, &[_]u32{}, ssi, allocator);
    }

    fn rearrangeCandidatesByIndices(candidates: []types.RankedSegment, indices: []const usize, scores: []const f32, allocator: Allocator) !void {
        var sorted_tokens = try allocator.alloc([]u32, candidates.len);
        defer allocator.free(sorted_tokens);
        @memset(sorted_tokens, &[_]u32{});

        var sorted_scores = try allocator.alloc(f32, candidates.len);
        defer allocator.free(sorted_scores);

        var sorted_positions = try allocator.alloc(u64, candidates.len);
        defer allocator.free(sorted_positions);

        var sorted_anchors = try allocator.alloc(bool, candidates.len);
        defer allocator.free(sorted_anchors);

        var dup_count: usize = 0;
        errdefer {
            var d: usize = 0;
            while (d < dup_count) : (d += 1) {
                allocator.free(sorted_tokens[d]);
            }
        }

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            const src_idx = indices[i];
            sorted_tokens[i] = try allocator.dupe(u32, candidates[src_idx].tokens);
            dup_count += 1;
            sorted_scores[i] = scores[src_idx];
            sorted_positions[i] = candidates[src_idx].position;
            sorted_anchors[i] = candidates[src_idx].anchor;
        }

        i = 0;
        while (i < candidates.len) : (i += 1) {
            allocator.free(candidates[i].tokens);
            candidates[i].tokens = sorted_tokens[i];
            candidates[i].score = sorted_scores[i];
            candidates[i].position = sorted_positions[i];
            candidates[i].anchor = sorted_anchors[i];
        }
    }

    fn sortCandidatesByScore(candidates: []types.RankedSegment, scores: []const f32, allocator: Allocator) !void {
        if (candidates.len == 0) return;

        var indices = try allocator.alloc(usize, candidates.len);
        defer allocator.free(indices);

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            indices[i] = i;
        }

        const SortContext = struct {
            scores: []const f32,
            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                const score_a = ctx.scores[a];
                const score_b = ctx.scores[b];
                if (math.isNan(score_a)) return false;
                if (math.isNan(score_b)) return true;
                return score_a > score_b;
            }
        };
        std.mem.sort(usize, indices, SortContext{ .scores = scores }, SortContext.lessThan);

        try rearrangeCandidatesByIndices(candidates, indices, scores, allocator);
    }

    pub fn rankCandidatesWithQuery(self: *const Ranker, candidates: []types.RankedSegment, query: []const u32, ssi: *const SSI, allocator: Allocator) !void {
        if (candidates.len == 0) return;

        var scores = try allocator.alloc(f32, candidates.len);
        defer allocator.free(scores);

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            if (query.len > 0) {
                scores[i] = try self.scoreSequenceWithQuery(candidates[i].tokens, query, ssi);
            } else {
                scores[i] = try self.scoreSequence(candidates[i].tokens, ssi);
            }
        }

        normalizeScoresStatic(scores);
        try sortCandidatesByScore(candidates, scores, allocator);
    }

    pub fn batchScore(self: *const Ranker, sequences: []const []const u32, ssi: *const SSI, allocator: Allocator) ![]f32 {
        if (sequences.len == 0) return allocator.alloc(f32, 0);

        const batch_size = sequences.len;
        var scores = try allocator.alloc(f32, batch_size);
        errdefer allocator.free(scores);

        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            scores[b] = try self.scoreSequence(sequences[b], ssi);
        }
        return scores;
    }

    pub fn topKHeap(self: *const Ranker, ssi: *const SSI, query: []const u32, k: usize, allocator: Allocator) ![]types.RankedSegment {
        if (k == 0) return allocator.alloc(types.RankedSegment, 0);

        const retrieval_count = k;

        var heap = std.PriorityQueue(types.RankedSegment, void, struct {
            pub fn lessThan(_: void, a: types.RankedSegment, b: types.RankedSegment) std.math.Order {
                if (math.isNan(a.score) and math.isNan(b.score)) return .eq;
                if (math.isNan(a.score)) return .lt;
                if (math.isNan(b.score)) return .gt;
                return std.math.order(a.score, b.score);
            }
        }.lessThan).init(allocator, {});
        defer heap.deinit();

        const candidates = try ssi.retrieveTopK(query, retrieval_count, allocator);
        defer {
            var i: usize = 0;
            while (i < candidates.len) : (i += 1) {
                candidates[i].deinit(allocator);
            }
            allocator.free(candidates);
        }

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            const cand = candidates[i];
            const score = try self.scoreSequenceWithQuery(cand.tokens, query, ssi);

            if (math.isNan(score) or math.isInf(score)) continue;

            if (heap.count() < k) {
                const ranked = try types.RankedSegment.init(allocator, cand.tokens, score, cand.position, cand.anchor);
                try heap.add(ranked);
            } else if (heap.peek()) |top| {
                if (score > top.score) {
                    var removed = heap.remove();
                    removed.deinit(allocator);
                    const ranked = try types.RankedSegment.init(allocator, cand.tokens, score, cand.position, cand.anchor);
                    try heap.add(ranked);
                }
            }
        }

        const result_count = heap.count();
        var top_n = try allocator.alloc(types.RankedSegment, result_count);
        errdefer {
            var j: usize = 0;
            while (j < top_n.len) : (j += 1) {
                if (j < result_count) {
                    top_n[j].deinit(allocator);
                }
            }
            allocator.free(top_n);
        }

        var idx: usize = result_count;
        while (heap.removeOrNull()) |item| {
            if (idx > 0) {
                idx -= 1;
                top_n[idx] = item;
            } else {
                var mutable_item = item;
                mutable_item.deinit(allocator);
            }
        }

        return top_n;
    }

    pub fn updateWeights(self: *Ranker, gradients: []const f32) void {
        var i: usize = 0;
        while (i < @min(self.ngram_weights.len, gradients.len)) : (i += 1) {
            const grad = gradients[i];
            if (math.isNan(grad) or math.isInf(grad)) continue;
            self.ngram_weights[i] -= RankerConfig.LEARNING_RATE * grad;
            self.ngram_weights[i] = math.clamp(self.ngram_weights[i], 0.0, 1.0);
        }
    }

    pub fn minHashSignature(self: *const Ranker, tokens: []const u32) ![]u64 {
        if (tokens.len == 0) {
            const sig = try self.allocator.alloc(u64, self.num_hash_functions);
            @memset(sig, std.math.maxInt(u64));
            return sig;
        }

        const sig = try self.allocator.alloc(u64, self.num_hash_functions);
        errdefer self.allocator.free(sig);

        var h: usize = 0;
        while (h < self.num_hash_functions) : (h += 1) {
            var min_hash: u64 = std.math.maxInt(u64);
            const seed_a = self.lsh_hash_params[h * 2];
            const seed_b = self.lsh_hash_params[h * 2 + 1];

            for (tokens) |token| {
                const le_bytes = tokenToLEBytes(token);
                const hash_val = stableHash(&le_bytes, seed_a) ^ seed_b;
                if (hash_val < min_hash) {
                    min_hash = hash_val;
                }
            }
            sig[h] = min_hash;
        }
        return sig;
    }

    pub fn jaccardSimilarityFromSignatures(sig1: []const u64, sig2: []const u64) f32 {
        if (sig1.len != sig2.len) return 0.0;
        if (sig1.len == 0) return 0.0;

        var matches: usize = 0;
        var i: usize = 0;
        while (i < sig1.len) : (i += 1) {
            if (sig1[i] == sig2[i]) {
                matches += 1;
            }
        }
        return @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(sig1.len));
    }

    pub fn estimateJaccard(set1: BitSet, set2: BitSet) f32 {
        const len1 = set1.bits.len;
        const len2 = set2.bits.len;
        const max_words = @max(len1, len2);

        if (max_words == 0) return 1.0;

        var intersect: usize = 0;
        var union_count: usize = 0;
        var i: usize = 0;
        while (i < max_words) : (i += 1) {
            const w1: u64 = if (i < len1) set1.bits[i] else 0;
            const w2: u64 = if (i < len2) set2.bits[i] else 0;
            intersect += @popCount(w1 & w2);
            union_count += @popCount(w1 | w2);
        }
        return if (union_count == 0) 1.0 else @as(f32, @floatFromInt(intersect)) / @as(f32, @floatFromInt(union_count));
    }

    pub fn vectorScore(embedding: *const Tensor, query_emb: *const Tensor) !f32 {
        if (!mem.eql(usize, embedding.shape.dims, query_emb.shape.dims)) return Error.ShapeMismatch;
        if (embedding.data.len == 0) return 0.0;

        var dot_prod: f32 = 0.0;
        var norm_emb: f32 = 0.0;
        var norm_query: f32 = 0.0;

        const len = embedding.data.len;
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const e = embedding.data[i];
            const q = query_emb.data[i];

            if (math.isNan(e) or math.isNan(q)) continue;
            if (math.isInf(e) or math.isInf(q)) continue;

            dot_prod += e * q;
            norm_emb += e * e;
            norm_query += q * q;
        }

        if (norm_emb <= 0.0 or norm_query <= 0.0) return 0.0;

        norm_emb = math.sqrt(norm_emb);
        norm_query = math.sqrt(norm_query);

        if (norm_emb == 0.0 or norm_query == 0.0) return 0.0;

        const result = dot_prod / (norm_emb * norm_query);
        return math.clamp(result, -1.0, 1.0);
    }

    pub fn dotProductScore(embedding: *const Tensor, query_emb: *const Tensor) !f32 {
        if (!mem.eql(usize, embedding.shape.dims, query_emb.shape.dims)) return Error.ShapeMismatch;
        if (embedding.data.len == 0) return 0.0;

        var dot_prod: f32 = 0.0;
        const len = embedding.data.len;
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const e = embedding.data[i];
            const q = query_emb.data[i];

            if (math.isNan(e) or math.isNan(q)) continue;
            if (math.isInf(e) or math.isInf(q)) continue;

            dot_prod += e * q;
        }
        return dot_prod;
    }

    pub fn weightedAverage(scores: []const f32, weights: []const f32) error{LengthMismatch}!f32 {
        if (scores.len != weights.len) return error.LengthMismatch;
        if (scores.len == 0) return 0.0;

        var num: f32 = 0.0;
        var den: f32 = 0.0;
        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            const s = scores[i];
            const w = weights[i];

            if (math.isNan(s) or math.isNan(w)) continue;
            if (math.isInf(s) or math.isInf(w)) continue;

            num += s * w;
            den += w;
        }

        if (den == 0.0) return 0.0;
        return num / den;
    }

    pub fn exponentialDecay(scores: []f32, decay_factor: f32) void {
        if (scores.len == 0) return;
        if (decay_factor <= 0.0 or decay_factor >= 1.0) return;

        var current_decay: f32 = 1.0;
        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            if (!math.isNan(scores[i]) and !math.isInf(scores[i])) {
                scores[i] *= current_decay;
            }
            current_decay *= decay_factor;
        }
    }

    pub fn normalizeScores(self: *const Ranker, scores: []f32) void {
        _ = self;
        normalizeScoresStatic(scores);
    }

    fn normalizeScoresStatic(scores: []f32) void {
        if (scores.len == 0) return;

        var min_score: f32 = math.inf(f32);
        var max_score: f32 = -math.inf(f32);
        var valid_count: usize = 0;

        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            const s = scores[i];
            if (math.isNan(s) or math.isInf(s)) continue;
            valid_count += 1;
            if (s < min_score) min_score = s;
            if (s > max_score) max_score = s;
        }

        if (valid_count == 0) {
            i = 0;
            while (i < scores.len) : (i += 1) {
                scores[i] = 0.0;
            }
            return;
        }
        if (max_score == min_score) {
            i = 0;
            while (i < scores.len) : (i += 1) {
                if (!math.isNan(scores[i]) and !math.isInf(scores[i])) {
                    scores[i] = 0.5;
                } else {
                    scores[i] = 0.0;
                }
            }
            return;
        }

        const range = max_score - min_score;
        i = 0;
        while (i < scores.len) : (i += 1) {
            if (math.isNan(scores[i]) or math.isInf(scores[i])) {
                scores[i] = 0.0;
            } else {
                scores[i] = (scores[i] - min_score) / range;
            }
        }
    }

    pub fn rankByMultipleCriteria(self: *const Ranker, candidates: []types.RankedSegment, criteria: [][]f32, weights: []const f32, allocator: Allocator) !void {
        _ = self;
        if (candidates.len == 0) return;
        if (criteria.len == 0) return;
        if (weights.len == 0) return;

        const num_cand = candidates.len;
        const num_crit = @min(criteria.len, weights.len);

        var combined = try allocator.alloc(f32, num_cand);
        defer allocator.free(combined);

        var c: usize = 0;
        while (c < num_cand) : (c += 1) {
            var crit_score: f32 = 0.0;
            var cr: usize = 0;
            while (cr < num_crit) : (cr += 1) {
                if (c < criteria[cr].len) {
                    const score_val = criteria[cr][c];
                    const weight_val = weights[cr];
                    if (!math.isNan(score_val) and !math.isNan(weight_val) and !math.isInf(score_val) and !math.isInf(weight_val)) {
                        crit_score += score_val * weight_val;
                    }
                }
            }
            combined[c] = crit_score;
        }

        try sortCandidatesByScore(candidates, combined, allocator);
    }

    pub fn streamingRank(self: *const Ranker, reader: anytype, ssi: *const SSI, k: usize, allocator: Allocator) ![]types.RankedSegment {
        if (k == 0) return allocator.alloc(types.RankedSegment, 0);

        var rolling_buffer = std.ArrayList(u32).init(allocator);
        defer rolling_buffer.deinit();

        var heap = std.PriorityQueue(types.RankedSegment, void, struct {
            pub fn lessThan(_: void, a: types.RankedSegment, b: types.RankedSegment) std.math.Order {
                if (math.isNan(a.score) and math.isNan(b.score)) return .eq;
                if (math.isNan(a.score)) return .lt;
                if (math.isNan(b.score)) return .gt;
                return std.math.order(a.score, b.score);
            }
        }.lessThan).init(allocator, {});
        defer {
            while (heap.removeOrNull()) |item| {
                var m = item;
                m.deinit(allocator);
            }
            heap.deinit();
        }

        var leftover_bytes: [3]u8 = undefined;
        var leftover_len: usize = 0;
        var position: u64 = 0;
        var read_buf: [RankerConfig.STREAMING_BUFFER_SIZE * @sizeOf(u32)]u8 = undefined;

        while (true) {
            const bytes_read = reader.read(&read_buf) catch |err| switch (err) {
                error.EndOfStream => break,
                else => return err,
            };
            if (bytes_read == 0) break;

            var combined_buf: []u8 = undefined;
            var combined_len: usize = 0;
            var combined_alloc: ?[]u8 = null;
            defer {
                if (combined_alloc) |ca| allocator.free(ca);
            }

            if (leftover_len > 0) {
                combined_len = leftover_len + bytes_read;
                combined_alloc = try allocator.alloc(u8, combined_len);
                combined_buf = combined_alloc.?;
                var ci: usize = 0;
                while (ci < leftover_len) : (ci += 1) {
                    combined_buf[ci] = leftover_bytes[ci];
                }
                var ri: usize = 0;
                while (ri < bytes_read) : (ri += 1) {
                    combined_buf[leftover_len + ri] = read_buf[ri];
                }
                leftover_len = 0;
            } else {
                combined_buf = read_buf[0..bytes_read];
                combined_len = bytes_read;
            }

            const full_tokens = combined_len / @sizeOf(u32);
            const remainder = combined_len % @sizeOf(u32);

            if (remainder > 0) {
                var ri: usize = 0;
                while (ri < remainder) : (ri += 1) {
                    leftover_bytes[ri] = combined_buf[full_tokens * @sizeOf(u32) + ri];
                }
                leftover_len = remainder;
            }

            var ti: usize = 0;
            while (ti < full_tokens) : (ti += 1) {
                const offset = ti * @sizeOf(u32);
                var token_bytes: [4]u8 = undefined;
                token_bytes[0] = combined_buf[offset + 0];
                token_bytes[1] = combined_buf[offset + 1];
                token_bytes[2] = combined_buf[offset + 2];
                token_bytes[3] = combined_buf[offset + 3];
                const token = mem.readInt(u32, &token_bytes, .little);
                try rolling_buffer.append(token);
            }

            while (rolling_buffer.items.len >= RankerConfig.STREAMING_WINDOW_SIZE) {
                const window = rolling_buffer.items[0..RankerConfig.STREAMING_WINDOW_SIZE];
                const score = try self.scoreSequence(window, ssi);

                if (!math.isNan(score) and !math.isInf(score)) {
                    if (heap.count() < k) {
                        const seg = try types.RankedSegment.init(allocator, window, score, position, false);
                        try heap.add(seg);
                    } else if (heap.peek()) |top| {
                        if (score > top.score) {
                            var removed = heap.remove();
                            removed.deinit(allocator);
                            const seg = try types.RankedSegment.init(allocator, window, score, position, false);
                            try heap.add(seg);
                        }
                    }
                }

                const shift = @min(rolling_buffer.items.len, RankerConfig.STREAMING_WINDOW_SIZE / 4);
                if (shift == 0) break;
                const remaining = rolling_buffer.items.len - shift;
                if (remaining > 0) {
                    std.mem.copyForwards(u32, rolling_buffer.items[0..remaining], rolling_buffer.items[shift..rolling_buffer.items.len]);
                }
                rolling_buffer.shrinkRetainingCapacity(remaining);
                position += shift;
            }
        }

        if (rolling_buffer.items.len > 0) {
            const tail = rolling_buffer.items;
            const score = try self.scoreSequence(tail, ssi);
            if (!math.isNan(score) and !math.isInf(score)) {
                if (heap.count() < k) {
                    const seg = try types.RankedSegment.init(allocator, tail, score, position, false);
                    try heap.add(seg);
                } else if (heap.peek()) |top| {
                    if (score > top.score) {
                        var removed = heap.remove();
                        removed.deinit(allocator);
                        const seg = try types.RankedSegment.init(allocator, tail, score, position, false);
                        try heap.add(seg);
                    }
                }
            }
        }

        const result_count = heap.count();
        var result = try allocator.alloc(types.RankedSegment, result_count);
        errdefer {
            var ei: usize = 0;
            while (ei < result.len) : (ei += 1) {
                result[ei].deinit(allocator);
            }
            allocator.free(result);
        }

        var idx: usize = result_count;
        while (heap.removeOrNull()) |item| {
            if (idx > 0) {
                idx -= 1;
                result[idx] = item;
            } else {
                var m = item;
                m.deinit(allocator);
            }
        }

        return result;
    }

    pub fn parallelScore(self: *const Ranker, sequences: [][]u32, ssi: *const SSI, num_threads: usize) ![]f32 {
        if (sequences.len == 0) return self.allocator.alloc(f32, 0);

        const scores = try self.allocator.alloc(f32, sequences.len);
        errdefer self.allocator.free(scores);

        if (num_threads <= 1 or sequences.len <= 1) {
            var i: usize = 0;
            while (i < sequences.len) : (i += 1) {
                scores[i] = try self.scoreSequence(sequences[i], ssi);
            }
            return scores;
        }

        const cpu_count = std.Thread.getCpuCount() catch @as(usize, 1);
        const effective_threads = @min(@min(num_threads, sequences.len), cpu_count);
        const chunk_size = sequences.len / effective_threads;
        const remainder_count = sequences.len % effective_threads;

        const ThreadContext = struct {
            ranker: *const Ranker,
            seqs: [][]u32,
            ssi_ptr: *const SSI,
            out: []f32,
            start: usize,
            end: usize,
            err_flag: bool,
        };

        var contexts = try self.allocator.alloc(ThreadContext, effective_threads);
        defer self.allocator.free(contexts);

        var threads = try self.allocator.alloc(std.Thread, effective_threads);
        defer self.allocator.free(threads);

        var thread_spawned = try self.allocator.alloc(bool, effective_threads);
        defer self.allocator.free(thread_spawned);
        @memset(thread_spawned, false);

        var offset: usize = 0;
        var t: usize = 0;
        while (t < effective_threads) : (t += 1) {
            const this_chunk = chunk_size + @as(usize, if (t < remainder_count) 1 else 0);
            contexts[t] = .{
                .ranker = self,
                .seqs = sequences,
                .ssi_ptr = ssi,
                .out = scores,
                .start = offset,
                .end = offset + this_chunk,
                .err_flag = false,
            };
            offset += this_chunk;
        }

        t = 0;
        while (t < effective_threads) : (t += 1) {
            threads[t] = std.Thread.spawn(.{}, struct {
                fn work(ctx: *ThreadContext) void {
                    var si: usize = ctx.start;
                    while (si < ctx.end) : (si += 1) {
                        ctx.out[si] = ctx.ranker.scoreSequence(ctx.seqs[si], ctx.ssi_ptr) catch {
                            ctx.err_flag = true;
                            return;
                        };
                    }
                }
            }.work, .{&contexts[t]}) catch {
                var si: usize = contexts[t].start;
                while (si < contexts[t].end) : (si += 1) {
                    scores[si] = self.scoreSequence(sequences[si], ssi) catch 0.0;
                }
                thread_spawned[t] = false;
                continue;
            };
            thread_spawned[t] = true;
        }

        t = 0;
        while (t < effective_threads) : (t += 1) {
            if (thread_spawned[t]) {
                threads[t].join();
            }
        }

        var had_error = false;
        t = 0;
        while (t < effective_threads) : (t += 1) {
            if (contexts[t].err_flag) had_error = true;
        }

        if (had_error) {
            self.allocator.free(scores);
            return error.ScoringFailed;
        }

        return scores;
    }

    pub fn calibrateWeights(self: *Ranker, training_data: [][]u32, labels: []const f32, ssi: *const SSI, epochs: usize) !void {
        if (training_data.len == 0 or labels.len == 0) return error.InvalidParameter;
        if (training_data.len != labels.len) return error.LengthMismatch;

        var gradients = try self.allocator.alloc(f32, self.ngram_weights.len);
        defer self.allocator.free(gradients);

        var gram_stack_buf: [1024]u8 = undefined;
        const max_gram_bytes = @min(self.num_ngrams * 4, gram_stack_buf.len);

        var epoch: usize = 0;
        while (epoch < epochs) : (epoch += 1) {
            @memset(gradients, 0.0);

            var i: usize = 0;
            while (i < training_data.len) : (i += 1) {
                const pred = try self.scoreSequence(training_data[i], ssi);
                const label = labels[i];

                if (math.isNan(pred) or math.isNan(label)) continue;
                if (math.isInf(pred) or math.isInf(label)) continue;

                const err_val = pred - label;

                var gram: usize = 1;
                while (gram <= @min(self.num_ngrams, training_data[i].len)) : (gram += 1) {
                    var start: usize = 0;
                    while (start <= training_data[i].len - gram) : (start += 1) {
                        const ngram = training_data[i][start .. start + gram];
                        const needed = ngram.len * 4;
                        if (needed > max_gram_bytes) continue;
                        encodeNgramLE(ngram, gram_stack_buf[0..needed]);
                        const h = stableHash(gram_stack_buf[0..needed], self.seed);
                        if (ssi.getSegment(h)) |s| {
                            if (!math.isNan(s.score) and !math.isInf(s.score)) {
                                const weight_idx = @min(gram - 1, self.ngram_weights.len - 1);
                                gradients[weight_idx] += err_val * s.score;
                            }
                        }
                    }
                }
            }

            const n_samples: f32 = @floatFromInt(training_data.len);
            var g: usize = 0;
            while (g < gradients.len) : (g += 1) {
                gradients[g] = gradients[g] / n_samples;
            }

            self.updateWeights(gradients);
        }
    }

    pub fn exportModel(self: *const Ranker, path: []const u8) !void {
        var file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        var writer = file.writer();
        try writer.writeInt(u8, 2, .little);
        try writer.writeInt(u64, @intCast(self.ngram_weights.len), .little);
        try writer.writeInt(u64, @intCast(self.num_ngrams), .little);
        var i: usize = 0;
        while (i < self.ngram_weights.len) : (i += 1) {
            const bits: u32 = @bitCast(self.ngram_weights[i]);
            try writer.writeInt(u32, bits, .little);
        }
        try writer.writeInt(u64, @intCast(self.num_hash_functions), .little);
        i = 0;
        while (i < self.lsh_hash_params.len) : (i += 1) {
            try writer.writeInt(u64, self.lsh_hash_params[i], .little);
        }
        try writer.writeInt(u64, self.seed, .little);
    }

    pub fn importModel(self: *Ranker, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        var reader = file.reader();
        const version = try reader.readInt(u8, .little);
        if (version != 2) return error.InvalidVersion;

        const num_w = try reader.readInt(u64, .little);
        const num_ng = try reader.readInt(u64, .little);

        const num_w_usize: usize = @intCast(num_w);
        const num_ng_usize: usize = @intCast(num_ng);

        if (self.ngram_weights.len != num_w_usize) {
            self.allocator.free(self.ngram_weights);
            self.ngram_weights = try self.allocator.alloc(f32, num_w_usize);
        }
        self.num_ngrams = num_ng_usize;

        var i: usize = 0;
        while (i < self.ngram_weights.len) : (i += 1) {
            const bits = try reader.readInt(u32, .little);
            var weight: f32 = @bitCast(bits);
            if (math.isNan(weight) or math.isInf(weight)) weight = 0.0;
            self.ngram_weights[i] = weight;
        }

        const num_h = try reader.readInt(u64, .little);
        const num_h_usize: usize = @intCast(num_h);
        self.num_hash_functions = num_h_usize;
        if (self.lsh_hash_params.len != num_h_usize * 2) {
            self.allocator.free(self.lsh_hash_params);
            self.lsh_hash_params = try self.allocator.alloc(u64, num_h_usize * 2);
        }

        i = 0;
        while (i < self.lsh_hash_params.len) : (i += 1) {
            self.lsh_hash_params[i] = try reader.readInt(u64, .little);
        }
        self.seed = try reader.readInt(u64, .little);
    }
};

test "Ranker score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2, 3 }, 0, false);
    const score = try ranker.scoreSequence(&.{ 1, 2 }, &ssi);
    try testing.expect(score >= 0.0);
}

test "MinHash signature deterministic" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 32, 42);
    defer ranker.deinit();
    const sig1 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig1);
    const sig2 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig2);
    try testing.expectEqualSlices(u64, sig1, sig2);
}

test "Jaccard similarity from signatures" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 32, 42);
    defer ranker.deinit();
    const sig1 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig1);
    const sig2 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig2);
    const sim = Ranker.jaccardSimilarityFromSignatures(sig1, sig2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim, @as(f32, 0.01));
}

test "Token diversity" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 1, 42);
    defer ranker.deinit();
    const div1 = try ranker.computeTokenDiversity(&.{ 1, 1, 1, 1 });
    const div2 = try ranker.computeTokenDiversity(&.{ 1, 2, 3, 4 });
    try testing.expect(div2 > div1);
}

test "Token overlap" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 1, 42);
    defer ranker.deinit();
    const overlap = try ranker.computeTokenOverlap(&.{ 1, 2, 3 }, &.{ 2, 3, 4 });
    try testing.expect(overlap > 0.0 and overlap <= 1.0);
}

test "Estimate Jaccard" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var set1 = try BitSet.init(gpa, 128);
    defer set1.deinit();
    set1.set(0);
    set1.set(64);
    var set2 = try BitSet.init(gpa, 128);
    defer set2.deinit();
    set2.set(0);
    const est = Ranker.estimateJaccard(set1, set2);
    try testing.expect(est >= 0.0 and est <= 1.0);
}

test "Estimate Jaccard empty sets" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var set1 = try BitSet.init(gpa, 64);
    defer set1.deinit();
    var set2 = try BitSet.init(gpa, 64);
    defer set2.deinit();
    const est = Ranker.estimateJaccard(set1, set2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), est, 0.01);
}

test "Vector cosine score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var emb = try Tensor.init(gpa, &.{3});
    defer emb.deinit();
    emb.data[0] = 1.0;
    emb.data[1] = 0.0;
    emb.data[2] = 0.0;
    var qemb = try Tensor.init(gpa, &.{3});
    defer qemb.deinit();
    qemb.data[0] = 1.0;
    qemb.data[1] = 0.0;
    qemb.data[2] = 0.0;
    const score = try Ranker.vectorScore(&emb, &qemb);
    try testing.expectApproxEqAbs(@as(f32, 1.0), score, @as(f32, 0.01));
}

test "Dot product score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var emb = try Tensor.init(gpa, &.{3});
    defer emb.deinit();
    emb.data[0] = 1.0;
    emb.data[1] = 2.0;
    emb.data[2] = 3.0;
    var qemb = try Tensor.init(gpa, &.{3});
    defer qemb.deinit();
    qemb.data[0] = 1.0;
    qemb.data[1] = 2.0;
    qemb.data[2] = 3.0;
    const score = try Ranker.dotProductScore(&emb, &qemb);
    try testing.expectApproxEqAbs(@as(f32, 14.0), score, @as(f32, 0.01));
}

test "Weighted average" {
    const testing = std.testing;
    const scores = [_]f32{ 0.5, 0.8, 0.3 };
    const weights = [_]f32{ 1.0, 2.0, 1.0 };
    const avg = try Ranker.weightedAverage(&scores, &weights);
    try testing.expect(avg > 0.0 and avg < 1.0);
}

test "Exponential decay" {
    const testing = std.testing;
    var scores = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    Ranker.exponentialDecay(&scores, 0.9);
    try testing.expect(scores[0] > scores[1]);
    try testing.expect(scores[1] > scores[2]);
    try testing.expect(scores[2] > scores[3]);
}

test "Normalize scores" {
    const testing = std.testing;
    var scores = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    Ranker.normalizeScoresStatic(&scores);
    try testing.expectApproxEqAbs(@as(f32, 0.0), scores[0], @as(f32, 0.01));
    try testing.expectApproxEqAbs(@as(f32, 1.0), scores[3], @as(f32, 0.01));
}
