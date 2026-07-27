const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const zk = @import("zk_verification.zig");
const obf = @import("dataset_obfuscation.zig");
const crypto = std.crypto;
const Blake3 = crypto.hash.Blake3;
const fs = std.fs;

pub const StoredCommitment = struct {
    input_commitment: [32]u8,
    output_commitment: [32]u8,
};

pub const VerifiedInferenceEngine = struct {
    allocator: Allocator,
    commitment_scheme: *zk.CommitmentScheme,
    homomorphic_enc: *obf.HomomorphicEncryption,
    differential_privacy: *zk.DifferentialPrivacy,
    dataset_fingerprint: *obf.DatasetFingerprint,
    proof_of_correctness: *obf.ProofOfCorrectness,
    inference_proofs: ArrayList(*zk.ZKInferenceProof),
    model_hash: [32]u8,
    verification_count: u64,
    successful_verifications: u64,
    zk_prover: ?*zk.ZKInferenceProver,
    use_zk_proofs: bool,
    layer_weights_s: ?[][][]f32,
    layer_weights_t: ?[][][]f32,
    num_layers: usize,
    embedding_dim: usize,
    weight_seed: u64,
    stored_commitments: ArrayList(StoredCommitment),

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);

        self.* = Self{
            .allocator = allocator,
            .commitment_scheme = try zk.CommitmentScheme.init(allocator),
            .homomorphic_enc = try obf.HomomorphicEncryption.init(allocator),
            .differential_privacy = try zk.DifferentialPrivacy.init(allocator, 1.0, 1e-5, 0.001),
            .dataset_fingerprint = try obf.DatasetFingerprint.init(allocator),
            .proof_of_correctness = try obf.ProofOfCorrectness.init(allocator),
            .inference_proofs = ArrayList(*zk.ZKInferenceProof).init(allocator),
            .model_hash = undefined,
            .verification_count = 0,
            .successful_verifications = 0,
            .zk_prover = null,
            .use_zk_proofs = false,
            .layer_weights_s = null,
            .layer_weights_t = null,
            .num_layers = 8,
            .embedding_dim = 32,
            .weight_seed = @as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))),
            .stored_commitments = ArrayList(StoredCommitment).init(allocator),
        };

        var model_hasher = Blake3.init(.{});
        const model_seed = "JAIDE_VERIFIED_MODEL_V40";
        model_hasher.update(model_seed);
        model_hasher.final(&self.model_hash);

        try self.initializeWeights();

        return self;
    }

    pub fn initWithZKProofs(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);

        self.* = Self{
            .allocator = allocator,
            .commitment_scheme = try zk.CommitmentScheme.init(allocator),
            .homomorphic_enc = try obf.HomomorphicEncryption.init(allocator),
            .differential_privacy = try zk.DifferentialPrivacy.init(allocator, 1.0, 1e-5, 0.001),
            .dataset_fingerprint = try obf.DatasetFingerprint.init(allocator),
            .proof_of_correctness = try obf.ProofOfCorrectness.init(allocator),
            .inference_proofs = ArrayList(*zk.ZKInferenceProof).init(allocator),
            .model_hash = undefined,
            .verification_count = 0,
            .successful_verifications = 0,
            .zk_prover = try zk.ZKInferenceProver.init(allocator),
            .use_zk_proofs = true,
            .layer_weights_s = null,
            .layer_weights_t = null,
            .num_layers = 8,
            .embedding_dim = 32,
            .weight_seed = @as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))),
            .stored_commitments = ArrayList(StoredCommitment).init(allocator),
        };

        var model_hasher = Blake3.init(.{});
        const model_seed = "JAIDE_VERIFIED_MODEL_V40";
        model_hasher.update(model_seed);
        model_hasher.final(&self.model_hash);

        try self.initializeWeights();

        return self;
    }

    pub fn initWithSeed(allocator: Allocator, seed: u64) !*Self {
        const self = try allocator.create(Self);

        self.* = Self{
            .allocator = allocator,
            .commitment_scheme = try zk.CommitmentScheme.init(allocator),
            .homomorphic_enc = try obf.HomomorphicEncryption.init(allocator),
            .differential_privacy = try zk.DifferentialPrivacy.init(allocator, 1.0, 1e-5, 0.001),
            .dataset_fingerprint = try obf.DatasetFingerprint.init(allocator),
            .proof_of_correctness = try obf.ProofOfCorrectness.init(allocator),
            .inference_proofs = ArrayList(*zk.ZKInferenceProof).init(allocator),
            .model_hash = undefined,
            .verification_count = 0,
            .successful_verifications = 0,
            .zk_prover = null,
            .use_zk_proofs = false,
            .layer_weights_s = null,
            .layer_weights_t = null,
            .num_layers = 8,
            .embedding_dim = 32,
            .weight_seed = seed,
            .stored_commitments = ArrayList(StoredCommitment).init(allocator),
        };

        var model_hasher = Blake3.init(.{});
        const model_seed = "JAIDE_VERIFIED_MODEL_V40";
        model_hasher.update(model_seed);
        model_hasher.final(&self.model_hash);

        try self.initializeWeights();

        return self;
    }

    fn initializeWeights(self: *Self) !void {
        self.layer_weights_s = try self.allocator.alloc([][]f32, self.num_layers);
        self.layer_weights_t = try self.allocator.alloc([][]f32, self.num_layers);

        var prng = std.Random.DefaultPrng.init(self.weight_seed);
        const rand = prng.random();

        var layer: usize = 0;
        while (layer < self.num_layers) : (layer += 1) {
            self.layer_weights_s.?[layer] = try self.allocator.alloc([]f32, self.embedding_dim);
            self.layer_weights_t.?[layer] = try self.allocator.alloc([]f32, self.embedding_dim);

            var i: usize = 0;
            while (i < self.embedding_dim) : (i += 1) {
                self.layer_weights_s.?[layer][i] = try self.allocator.alloc(f32, self.embedding_dim);
                self.layer_weights_t.?[layer][i] = try self.allocator.alloc(f32, self.embedding_dim);

                var j: usize = 0;
                while (j < self.embedding_dim) : (j += 1) {
                    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.embedding_dim)));
                    self.layer_weights_s.?[layer][i][j] = scale * rand.float(f32) * 2.0 - scale;
                    self.layer_weights_t.?[layer][i][j] = scale * rand.float(f32) * 2.0 - scale;
                }
            }
        }
    }

    pub fn loadWeightsFromFile(self: *Self, weights_path: []const u8) !void {
        const file = try fs.cwd().openFile(weights_path, .{});
        defer file.close();

        var reader = file.reader();

        if (self.layer_weights_s != null) {
            self.freeWeights();
        }

        self.layer_weights_s = try self.allocator.alloc([][]f32, self.num_layers);
        self.layer_weights_t = try self.allocator.alloc([][]f32, self.num_layers);

        var layer: usize = 0;
        while (layer < self.num_layers) : (layer += 1) {
            self.layer_weights_s.?[layer] = try self.allocator.alloc([]f32, self.embedding_dim);
            self.layer_weights_t.?[layer] = try self.allocator.alloc([]f32, self.embedding_dim);

            var i: usize = 0;
            while (i < self.embedding_dim) : (i += 1) {
                self.layer_weights_s.?[layer][i] = try self.allocator.alloc(f32, self.embedding_dim);
                self.layer_weights_t.?[layer][i] = try self.allocator.alloc(f32, self.embedding_dim);

                var j: usize = 0;
                while (j < self.embedding_dim) : (j += 1) {
                    var s_bytes: [4]u8 = undefined;
                    _ = try reader.readAll(&s_bytes);
                    self.layer_weights_s.?[layer][i][j] = @bitCast(s_bytes);

                    var t_bytes: [4]u8 = undefined;
                    _ = try reader.readAll(&t_bytes);
                    self.layer_weights_t.?[layer][i][j] = @bitCast(t_bytes);
                }
            }
        }

        self.use_zk_proofs = true;
    }

    fn freeWeights(self: *Self) void {
        if (self.layer_weights_s) |weights_s| {
            var layer: usize = 0;
            while (layer < self.num_layers) : (layer += 1) {
                var i: usize = 0;
                while (i < self.embedding_dim) : (i += 1) {
                    self.allocator.free(weights_s[layer][i]);
                }
                self.allocator.free(weights_s[layer]);
            }
            self.allocator.free(weights_s);
        }

        if (self.layer_weights_t) |weights_t| {
            var layer: usize = 0;
            while (layer < self.num_layers) : (layer += 1) {
                var i: usize = 0;
                while (i < self.embedding_dim) : (i += 1) {
                    self.allocator.free(weights_t[layer][i]);
                }
                self.allocator.free(weights_t[layer]);
            }
            self.allocator.free(weights_t);
        }

        self.layer_weights_s = null;
        self.layer_weights_t = null;
    }

    pub fn deinit(self: *Self) void {
        self.commitment_scheme.deinit();
        self.homomorphic_enc.deinit();
        self.differential_privacy.deinit();
        self.dataset_fingerprint.deinit();
        self.proof_of_correctness.deinit();

        for (self.inference_proofs.items) |proof| {
            proof.deinit();
        }
        self.inference_proofs.deinit();

        if (self.zk_prover) |prover| {
            prover.deinit();
        }

        self.freeWeights();

        self.stored_commitments.deinit();

        self.allocator.destroy(self);
    }

    pub fn performVerifiedInference(self: *Self, input: []const f32, output_buf: []f32) !void {
        if (input.len == 0 or output_buf.len == 0) {
            return error.InvalidInputOutput;
        }

        const input_commitment = try self.commitInput(input);

        var intermediate_1 = try self.allocator.alloc(f32, output_buf.len);
        defer self.allocator.free(intermediate_1);

        if (self.layer_weights_s != null and self.layer_weights_t != null) {
            self.matmul(input, intermediate_1, self.layer_weights_s.?, self.layer_weights_t.?);
        } else {
            self.scalarMultiply(input, intermediate_1, 1.732);
        }

        try self.proof_of_correctness.recordStep(
            1,
            input,
            intermediate_1,
            obf.ProofOfCorrectness.OperationType.ScalarMultiply,
        );

        const half = intermediate_1.len / 2;
        if (half > 0) {
            var j: usize = 0;
            while (j < half) : (j += 1) {
                const raw = intermediate_1[j + half];
                const clipped = if (raw < -5.0) @as(f32, -5.0) else if (raw > 5.0) @as(f32, 5.0) else raw;
                const scale = @exp(clipped);
                intermediate_1[j] *= scale;
            }
            j = 0;
            while (j < half) : (j += 1) {
                intermediate_1[j + half] += intermediate_1[j] * 1.732;
            }
        }

        try self.proof_of_correctness.recordStep(
            2,
            input,
            intermediate_1,
            obf.ProofOfCorrectness.OperationType.AffineCoupling,
        );

        var copy_i: usize = 0;
        while (copy_i < output_buf.len) : (copy_i += 1) {
            output_buf[copy_i] = intermediate_1[copy_i];
        }

        try self.proof_of_correctness.recordStep(
            3,
            intermediate_1,
            output_buf,
            obf.ProofOfCorrectness.OperationType.ScatterPermute,
        );

        for (output_buf) |*val| {
            const noisy = self.differential_privacy.addCalibratedNoise(@as(f64, val.*));
            val.* = @floatCast(noisy);
        }

        const output_commitment = try self.commitOutput(output_buf);

        try self.stored_commitments.append(StoredCommitment{
            .input_commitment = input_commitment,
            .output_commitment = output_commitment,
        });

        if (self.use_zk_proofs and self.zk_prover != null and self.layer_weights_s != null and self.layer_weights_t != null) {
            const inference_proof = try zk.ZKInferenceProof.initWithProver(self.allocator);

            const weights_s_slice = self.layer_weights_s.?[0..self.num_layers];
            const weights_t_slice = self.layer_weights_t.?[0..self.num_layers];

            inference_proof.proveInferenceWithZK(input, output_buf, weights_s_slice, weights_t_slice) catch |err| {
                std.debug.print("ZK proof generation failed: {}, falling back to hash-based proof\n", .{err});
                try inference_proof.proveInference(input, output_buf, self.model_hash);
            };

            try self.inference_proofs.append(inference_proof);
        } else {
            const inference_proof = try zk.ZKInferenceProof.init(self.allocator);
            try inference_proof.proveInference(input, output_buf, self.model_hash);
            try self.inference_proofs.append(inference_proof);
        }

        var commitment_verified = true;
        for (self.stored_commitments.items) |sc| {
            const in_commitment = self.commitment_scheme.commit(std.mem.sliceAsBytes(input)) catch {
                commitment_verified = false;
                break;
            };
            if (!std.mem.eql(u8, &sc.input_commitment, &in_commitment)) {
                commitment_verified = false;
                break;
            }
            const out_commitment = self.commitment_scheme.commit(std.mem.sliceAsBytes(output_buf)) catch {
                commitment_verified = false;
                break;
            };
            if (!std.mem.eql(u8, &sc.output_commitment, &out_commitment)) {
                commitment_verified = false;
                break;
            }
        }
        if (!commitment_verified) {
            self.verification_count += 1;
            try self.proof_of_correctness.finalize();
            return error.CommitmentVerificationFailed;
        }

        self.verification_count += 1;

        try self.proof_of_correctness.finalize();

        if (try self.verifyProofOfCorrectness()) {
            self.successful_verifications += 1;
        }
    }

    pub fn performVerifiedInferenceWithZK(
        self: *Self,
        input: []const f32,
        output_buf: []f32,
        weights_s: []const []const []const f32,
        weights_t: []const []const []const f32,
    ) !*zk.ZKProofBundle {
        if (input.len == 0 or output_buf.len == 0) {
            return error.InvalidInputOutput;
        }

        if (self.zk_prover == null) {
            self.zk_prover = try zk.ZKInferenceProver.init(self.allocator);
        }

        var intermediate = try self.allocator.alloc(f32, output_buf.len);
        defer self.allocator.free(intermediate);

        var i: usize = 0;
        while (i < output_buf.len and i < input.len) : (i += 1) {
            intermediate[i] = input[i];
        }

        var layer: usize = 0;
        while (layer < weights_s.len) : (layer += 1) {
            const half = output_buf.len / 2;
            if (half == 0) break;

            var out_idx: usize = 0;
            while (out_idx < half) : (out_idx += 1) {
                var s: f32 = 0.0;
                var in_idx: usize = 0;
                while (in_idx < half and in_idx < weights_s[layer].len) : (in_idx += 1) {
                    if (out_idx < weights_s[layer][in_idx].len) {
                        s += intermediate[in_idx + half] * weights_s[layer][in_idx][out_idx];
                    }
                }
                const clipped = if (s < -5.0) @as(f32, -5.0) else if (s > 5.0) @as(f32, 5.0) else s;
                const scale = @exp(clipped);
                intermediate[out_idx] *= scale;
            }

            out_idx = 0;
            while (out_idx < half) : (out_idx += 1) {
                var t: f32 = 0.0;
                var in_idx: usize = 0;
                while (in_idx < half and in_idx < weights_t[layer].len) : (in_idx += 1) {
                    if (out_idx < weights_t[layer][in_idx].len) {
                        t += intermediate[in_idx] * weights_t[layer][in_idx][out_idx];
                    }
                }
                intermediate[out_idx + half] += t;
            }
        }

        i = 0;
        while (i < output_buf.len) : (i += 1) {
            output_buf[i] = intermediate[i];
        }

        const proof_bundle = try self.zk_prover.?.proveInference(input, output_buf, weights_s, weights_t);

        self.verification_count += 1;
        if (proof_bundle.verification_status) {
            self.successful_verifications += 1;
        }

        return proof_bundle;
    }

    fn scalarMultiply(self: *Self, input: []const f32, output: []f32, factor: f32) void {
        _ = self;
        var i: usize = 0;
        while (i < output.len and i < input.len) : (i += 1) {
            output[i] = input[i] * factor;
        }
        while (i < output.len) : (i += 1) {
            output[i] = 0.0;
        }
    }

    fn matmul(self: *Self, input: []const f32, output: []f32, weights_s: [][][]f32, weights_t: [][][]f32) void {
        _ = self;
        var buf = output;
        var i: usize = 0;
        while (i < buf.len and i < input.len) : (i += 1) {
            buf[i] = input[i];
        }
        while (i < buf.len) : (i += 1) {
            buf[i] = 0.0;
        }
        var layer: usize = 0;
        while (layer < weights_s.len and layer < weights_t.len) : (layer += 1) {
            const half = buf.len / 2;
            if (half == 0) break;
            var out_idx: usize = 0;
            while (out_idx < half) : (out_idx += 1) {
                var s: f32 = 0.0;
                var in_idx: usize = 0;
                while (in_idx < half and in_idx < weights_s[layer].len) : (in_idx += 1) {
                    if (out_idx < weights_s[layer][in_idx].len) {
                        s += buf[in_idx + half] * weights_s[layer][in_idx][out_idx];
                    }
                }
                const clipped = if (s < -5.0) @as(f32, -5.0) else if (s > 5.0) @as(f32, 5.0) else s;
                const scale = @exp(clipped);
                buf[out_idx] *= scale;
            }
            out_idx = 0;
            while (out_idx < half) : (out_idx += 1) {
                var t: f32 = 0.0;
                var in_idx: usize = 0;
                while (in_idx < half and in_idx < weights_t[layer].len) : (in_idx += 1) {
                    if (out_idx < weights_t[layer][in_idx].len) {
                        t += buf[in_idx] * weights_t[layer][in_idx][out_idx];
                    }
                }
                buf[out_idx + half] += t;
            }
        }
    }

    fn commitInput(self: *Self, input: []const f32) ![32]u8 {
        const input_bytes = std.mem.sliceAsBytes(input);
        return self.commitment_scheme.commit(input_bytes);
    }

    fn commitOutput(self: *Self, output: []const f32) ![32]u8 {
        const output_bytes = std.mem.sliceAsBytes(output);
        return self.commitment_scheme.commit(output_bytes);
    }

    fn verifyProofOfCorrectness(self: *Self) !bool {
        if (self.inference_proofs.items.len == 0) {
            return false;
        }

        const latest_proof = self.inference_proofs.items[self.inference_proofs.items.len - 1];

        if (latest_proof.proof_bundle != null) {
            return latest_proof.verifyWithZK();
        }

        return latest_proof.verify(self.model_hash);
    }

    pub fn getVerificationRate(self: *Self) f64 {
        if (self.verification_count == 0) {
            return 0.0;
        }
        return @as(f64, @floatFromInt(self.successful_verifications)) / @as(f64, @floatFromInt(self.verification_count));
    }

    pub fn proveDatasetIsolation(self: *Self, sample_data: []const u8) !bool {
        var hasher = Blake3.init(.{});
        hasher.update(sample_data);
        var sample_hash: [32]u8 = undefined;
        hasher.final(&sample_hash);

        const exists = self.dataset_fingerprint.checkSimilarity(sample_hash);

        return !exists;
    }

    pub fn generateZKProofForQuery(self: *Self, query: []const f32, response: []const f32) ![32]u8 {
        if (self.use_zk_proofs and self.zk_prover != null and self.layer_weights_s != null and self.layer_weights_t != null) {
            const weights_s_slice = self.layer_weights_s.?[0..self.num_layers];
            const weights_t_slice = self.layer_weights_t.?[0..self.num_layers];

            const bundle = self.zk_prover.?.proveInference(query, response, weights_s_slice, weights_t_slice) catch {
                return self.generateHashProof(query, response);
            };
            defer bundle.deinit();

            var proof_hasher = Blake3.init(.{});
            proof_hasher.update(bundle.proof_json);
            var proof_hash: [32]u8 = undefined;
            proof_hasher.final(&proof_hash);

            return proof_hash;
        }

        return self.generateHashProof(query, response);
    }

    fn generateHashProof(self: *Self, query: []const f32, response: []const f32) ![32]u8 {
        const proof = try zk.ZKInferenceProof.init(self.allocator);
        defer proof.deinit();

        try proof.proveInference(query, response, self.model_hash);

        var proof_hasher = Blake3.init(.{});
        proof_hasher.update(&proof.input_commitment);
        proof_hasher.update(&proof.output_commitment);
        for (proof.computation_proof.items) |step| {
            proof_hasher.update(&step);
        }
        var proof_hash: [32]u8 = undefined;
        proof_hasher.final(&proof_hash);

        return proof_hash;
    }

    pub fn enableZKProofs(self: *Self) !void {
        if (self.zk_prover == null) {
            self.zk_prover = try zk.ZKInferenceProver.init(self.allocator);
        }
        if (self.layer_weights_s == null) {
            try self.initializeWeights();
        }
        self.use_zk_proofs = true;
    }

    pub fn disableZKProofs(self: *Self) void {
        self.use_zk_proofs = false;
    }

    pub fn isZKEnabled(self: *Self) bool {
        return self.use_zk_proofs and self.zk_prover != null;
    }
};

pub const BatchVerifier = struct {
    allocator: Allocator,
    batch_size: usize,
    accumulated_proofs: ArrayList([32]u8),
    batch_commitment: [32]u8,
    zk_proof_bundles: ArrayList(*zk.ZKProofBundle),
    use_zk_verification: bool,

    const Self = @This();

    pub fn init(allocator: Allocator, batch_size: usize) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .batch_size = batch_size,
            .accumulated_proofs = ArrayList([32]u8).init(allocator),
            .batch_commitment = undefined,
            .zk_proof_bundles = ArrayList(*zk.ZKProofBundle).init(allocator),
            .use_zk_verification = false,
        };
        return self;
    }

    pub fn initWithZK(allocator: Allocator, batch_size: usize) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .batch_size = batch_size,
            .accumulated_proofs = ArrayList([32]u8).init(allocator),
            .batch_commitment = undefined,
            .zk_proof_bundles = ArrayList(*zk.ZKProofBundle).init(allocator),
            .use_zk_verification = true,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.accumulated_proofs.deinit();
        for (self.zk_proof_bundles.items) |bundle| {
            bundle.deinit();
        }
        self.zk_proof_bundles.deinit();
        self.allocator.destroy(self);
    }

    pub fn addProof(self: *Self, proof_hash: [32]u8) !void {
        try self.accumulated_proofs.append(proof_hash);

        if (self.accumulated_proofs.items.len >= self.batch_size) {
            try self.finalizeBatch();
        }
    }

    pub fn addZKProofBundle(self: *Self, bundle: *zk.ZKProofBundle) !void {
        try self.zk_proof_bundles.append(bundle);

        var hasher = Blake3.init(.{});
        hasher.update(bundle.proof_json);
        var proof_hash: [32]u8 = undefined;
        hasher.final(&proof_hash);

        try self.accumulated_proofs.append(proof_hash);

        if (self.accumulated_proofs.items.len >= self.batch_size) {
            try self.finalizeBatch();
        }
    }

    fn finalizeBatch(self: *Self) !void {
        var hasher = Blake3.init(.{});

        for (self.accumulated_proofs.items) |proof| {
            hasher.update(&proof);
        }

        hasher.final(&self.batch_commitment);
        self.accumulated_proofs.clearRetainingCapacity();
    }

    pub fn verifyBatch(self: *Self, expected_commitment: [32]u8) bool {
        return std.mem.eql(u8, &self.batch_commitment, &expected_commitment);
    }

    pub fn verifyAllZKProofs(self: *Self) bool {
        for (self.zk_proof_bundles.items) |bundle| {
            if (!bundle.verification_status) {
                return false;
            }
        }
        return true;
    }

    pub fn getBatchVerificationStatus(self: *Self) struct { total: usize, verified: usize, success_rate: f64 } {
        var verified: usize = 0;
        for (self.zk_proof_bundles.items) |bundle| {
            if (bundle.verification_status) {
                verified += 1;
            }
        }

        const total = self.zk_proof_bundles.items.len;
        const rate: f64 = if (total > 0) @as(f64, @floatFromInt(verified)) / @as(f64, @floatFromInt(total)) else 0.0;

        return .{
            .total = total,
            .verified = verified,
            .success_rate = rate,
        };
    }
};

pub const ProofAggregator = struct {
    allocator: Allocator,
    aggregated_proofs: ArrayList(*zk.ZKProofBundle),
    merkle_root: [32]u8,
    proof_count: u64,

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .aggregated_proofs = ArrayList(*zk.ZKProofBundle).init(allocator),
            .merkle_root = undefined,
            .proof_count = 0,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        for (self.aggregated_proofs.items) |bundle| {
            bundle.deinit();
        }
        self.aggregated_proofs.deinit();
        @memset(&self.merkle_root, 0);
        self.proof_count = 0;
        self.allocator.destroy(self);
    }

    pub fn addProof(self: *Self, bundle: *zk.ZKProofBundle) !void {
        try self.aggregated_proofs.append(bundle);
        self.proof_count += 1;
        try self.updateMerkleRoot();
    }

    fn updateMerkleRoot(self: *Self) !void {
        if (self.aggregated_proofs.items.len == 0) {
            @memset(&self.merkle_root, 0);
            return;
        }

        var hashes = ArrayList([32]u8).init(self.allocator);
        defer hashes.deinit();

        for (self.aggregated_proofs.items) |bundle| {
            var hasher = Blake3.init(.{});
            hasher.update(bundle.proof_json);
            var hash: [32]u8 = undefined;
            hasher.final(&hash);
            try hashes.append(hash);
        }

        while (hashes.items.len > 1) {
            var next_level = ArrayList([32]u8).init(self.allocator);
            defer next_level.deinit();

            var i: usize = 0;
            while (i < hashes.items.len) : (i += 2) {
                var hasher = Blake3.init(.{});
                hasher.update(&hashes.items[i]);
                if (i + 1 < hashes.items.len) {
                    hasher.update(&hashes.items[i + 1]);
                } else {
                    hasher.update(&hashes.items[i]);
                }
                var hash: [32]u8 = undefined;
                hasher.final(&hash);
                try next_level.append(hash);
            }

            hashes.clearRetainingCapacity();
            try hashes.appendSlice(next_level.items);
        }

        self.merkle_root = hashes.items[0];
    }

    pub fn getMerkleRoot(self: *Self) [32]u8 {
        return self.merkle_root;
    }

    pub fn getProofCount(self: *Self) u64 {
        return self.proof_count;
    }

    pub fn verifyMerkleInclusion(self: *Self, proof_index: usize, proof_path: [][32]u8) bool {
        if (proof_index >= self.aggregated_proofs.items.len) {
            return false;
        }

        const bundle = self.aggregated_proofs.items[proof_index];
        var hasher = Blake3.init(.{});
        hasher.update(bundle.proof_json);
        var current_hash: [32]u8 = undefined;
        hasher.final(&current_hash);

        var idx = proof_index;
        for (proof_path) |sibling| {
            var next_hasher = Blake3.init(.{});
            if (idx % 2 == 0) {
                next_hasher.update(&current_hash);
                next_hasher.update(&sibling);
            } else {
                next_hasher.update(&sibling);
                next_hasher.update(&current_hash);
            }
            next_hasher.final(&current_hash);
            idx = idx / 2;
        }

        return std.mem.eql(u8, &current_hash, &self.merkle_root);
    }
};

