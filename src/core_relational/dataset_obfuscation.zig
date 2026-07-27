const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const crypto = std.crypto;
const Blake3 = crypto.hash.Blake3;
const Sha256 = crypto.hash.sha2.Sha256;

pub const CryptoConfig = struct {
    pub const PRIME_P: u256 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141;
    pub const PRIME_Q: u256 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140;
    pub const GENERATOR: u256 = 2;
    pub const NOISE_BITS: u32 = 128;
    pub const SECURITY_PARAMETER: u32 = 256;
};

pub const PaillierKeyPair = struct {
    n: u256,
    n_squared: u512,
    g: u256,
    lambda: u256,
    mu: u256,

    pub fn generate() PaillierKeyPair {
        const p = generatePrime128();
        const q = generatePrime128();

        const n: u256 = @as(u256, p) * @as(u256, q);
        const n_squared: u512 = @as(u512, n) * @as(u512, n);

        const p_minus_1 = p -% 1;
        const q_minus_1 = q -% 1;

        const lambda = lcm128(p_minus_1, q_minus_1);
        const g = n +% 1;

        const mu = modInverse256(@as(u256, lambda), n);

        return PaillierKeyPair{
            .n = n,
            .n_squared = n_squared,
            .g = g,
            .lambda = @as(u256, lambda),
            .mu = mu,
        };
    }
};

fn generatePrime128() u128 {
    while (true) {
        var buf: [16]u8 = undefined;
        crypto.random.bytes(&buf);
        var candidate = bytesToU128(&buf);
        candidate |= @as(u128, 1) << 126;
        candidate |= @as(u128, 1);
        if (isProbablePrime128(candidate)) {
            return candidate;
        }
        candidate += 2;
        var attempts: usize = 0;
        while (attempts < 1000) : (attempts += 1) {
            if (isProbablePrime128(candidate)) {
                return candidate;
            }
            candidate += 2;
        }
    }
}

fn isProbablePrime128(n: u128) bool {
    if (n < 2) return false;
    if (n == 2 or n == 3) return true;
    if (n % 2 == 0) return false;
    const small_primes = [_]u128{ 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 };
    for (small_primes) |p| {
        if (n == p) return true;
        if (n % p == 0) return false;
    }
    var d = n - 1;
    var r: u32 = 0;
    while (d % 2 == 0) {
        d /= 2;
        r += 1;
    }
    const witnesses = [_]u128{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
    for (witnesses) |a| {
        if (a >= n) continue;
        var x = modPow128(a, d, n);
        if (x == 1 or x == n - 1) continue;
        var j: u32 = 0;
        var found = false;
        while (j < r - 1) : (j += 1) {
            x = modMul128(x, x, n);
            if (x == n - 1) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

fn modPow128(base: u128, exp: u128, modulus: u128) u128 {
    if (modulus == 0) return 0;
    if (modulus == 1) return 0;
    var result: u256 = 1;
    var b: u256 = @as(u256, base) % @as(u256, modulus);
    var e = exp;
    while (e > 0) {
        if (e & 1 == 1) {
            result = (result * b) % @as(u256, modulus);
        }
        e = e >> 1;
        b = (b * b) % @as(u256, modulus);
    }
    return @truncate(result);
}

fn modMul128(a: u128, b: u128, modulus: u128) u128 {
    const product = @as(u256, a) * @as(u256, b);
    return @truncate(product % @as(u256, modulus));
}

fn bytesToU256(bytes: *const [32]u8) u256 {
    var result: u256 = 0;
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        result = (result << 8) | @as(u256, bytes[i]);
    }
    return result;
}

fn bytesToU128(bytes: *const [16]u8) u128 {
    var result: u128 = 0;
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        result = (result << 8) | @as(u128, bytes[i]);
    }
    return result;
}

fn gcd256(a: u256, b: u256) u256 {
    var x = a;
    var y = b;
    while (y != 0) {
        const temp = y;
        y = x % y;
        x = temp;
    }
    return x;
}

fn lcm256(a: u256, b: u256) u256 {
    if (a == 0 or b == 0) return 0;
    const g = gcd256(a, b);
    if (g == 0) return 0;
    return (a / g) *% b;
}

fn gcd128(a: u128, b: u128) u128 {
    var x = a;
    var y = b;
    while (y != 0) {
        const temp = y;
        y = x % y;
        x = temp;
    }
    return x;
}

fn lcm128(a: u128, b: u128) u128 {
    if (a == 0 or b == 0) return 0;
    const g = gcd128(a, b);
    if (g == 0) return 0;
    return (a / g) *% b;
}

fn modInverse256(a: u256, m: u256) u256 {
    if (m == 0) return 0;
    if (a == 0) return 0;

    var t: i512 = 0;
    var new_t: i512 = 1;
    var r: i512 = @intCast(m);
    var new_r: i512 = @intCast(a % m);

    while (new_r != 0) {
        const quotient = @divFloor(r, new_r);
        const temp_t = t - quotient * new_t;
        t = new_t;
        new_t = temp_t;

        const temp_r = r - quotient * new_r;
        r = new_r;
        new_r = temp_r;
    }

    if (r > 1) return 0;
    if (t < 0) t = t + @as(i512, @intCast(m));

    return @intCast(@as(u512, @intCast(t)) % @as(u512, m));
}

fn modPow256(base: u256, exp: u256, modulus: u256) u256 {
    if (modulus == 0) return 0;
    if (modulus == 1) return 0;

    var result: u512 = 1;
    var b: u512 = @as(u512, base) % @as(u512, modulus);
    var e = exp;

    while (e > 0) {
        if (e & 1 == 1) {
            result = (result * b) % @as(u512, modulus);
        }
        e = e >> 1;
        b = (b * b) % @as(u512, modulus);
    }

    return @truncate(result);
}

fn modInverse512(a: u512, m: u512) u512 {
    if (m == 0) return 0;
    if (a == 0) return 0;

    var t: i1024 = 0;
    var new_t: i1024 = 1;
    var r: i1024 = @intCast(m);
    var new_r: i1024 = @intCast(a % m);

    while (new_r != 0) {
        const quotient = @divFloor(r, new_r);
        const temp_t = t - quotient * new_t;
        t = new_t;
        new_t = temp_t;

        const temp_r = r - quotient * new_r;
        r = new_r;
        new_r = temp_r;
    }

    if (r > 1) return 0;
    if (t < 0) t = t + @as(i1024, @intCast(m));

    return @truncate(@as(u1024, @intCast(t)) % @as(u1024, m));
}

fn modPow512(base: u512, exp: u512, modulus: u512) u512 {
    if (modulus == 0) return 0;
    if (modulus == 1) return 0;

    var result: u512 = 1;
    var b: u512 = base % modulus;
    var e = exp;

    while (e > 0) {
        if (e & 1 == 1) {
            result = @rem(result *% b, modulus);
        }
        e = e >> 1;
        b = @rem(b *% b, modulus);
    }

    return result;
}

pub const HomomorphicEncryption = struct {
    allocator: Allocator,
    keys: PaillierKeyPair,
    noise_buffer: [64]u8,
    operation_count: u64,

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);

        var noise_buf: [64]u8 = undefined;
        crypto.random.bytes(&noise_buf);

        self.* = Self{
            .allocator = allocator,
            .keys = PaillierKeyPair.generate(),
            .noise_buffer = noise_buf,
            .operation_count = 0,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        secureZero(&self.keys.lambda);
        secureZero(&self.keys.mu);
        secureZeroBytes(&self.noise_buffer);
        self.allocator.destroy(self);
    }

    pub fn encrypt(self: *Self, plaintext: i64) !u512 {
        const sign: u256 = if (plaintext < 0) 1 else 0;
        const abs_val: u64 = if (plaintext < 0) @intCast(-plaintext) else @intCast(plaintext);
        const encoded: u256 = (@as(u256, sign) << 255) | @as(u256, abs_val);

        var r_bytes: [16]u8 = undefined;
        crypto.random.bytes(&r_bytes);
        var r = @as(u256, bytesToU128(&r_bytes)) % self.keys.n;

        while (r == 0 or gcd256(r, self.keys.n) != 1) {
            crypto.random.bytes(&r_bytes);
            r = @as(u256, bytesToU128(&r_bytes)) % self.keys.n;
        }

        const g_m = modPow512(@as(u512, self.keys.g), @as(u512, encoded), self.keys.n_squared);
        const r_n = modPow512(@as(u512, r), @as(u512, self.keys.n), self.keys.n_squared);

        const ciphertext: u512 = (g_m * r_n) % self.keys.n_squared;

        self.operation_count += 1;

        return ciphertext;
    }

    pub fn decrypt(self: *Self, ciphertext: u512) !i64 {
        const c_lambda = modPow512(ciphertext, @as(u512, self.keys.lambda), self.keys.n_squared);

        const l_value = if (c_lambda > 0) (c_lambda -% 1) / @as(u512, self.keys.n) else 0;

        const encoded: u256 = @truncate((l_value * @as(u512, self.keys.mu)) % @as(u512, self.keys.n));

        const sign_bit = (encoded >> 255) & 1;
        const magnitude: u64 = @truncate(encoded & (((@as(u256, 1) << 64) - 1)));

        const plaintext: i64 = if (sign_bit == 1) -@as(i64, @intCast(magnitude)) else @intCast(magnitude);

        return plaintext;
    }

    pub fn add(self: *Self, c1: u512, c2: u512) u512 {
        self.operation_count += 1;
        return (c1 * c2) % self.keys.n_squared;
    }

    pub fn multiplyScalar(self: *Self, c: u512, scalar: i64) u512 {
        self.operation_count += 1;
        if (scalar == 0) return 1;
        if (scalar == 1) return c;
        const abs_scalar: u64 = if (scalar < 0) @intCast(-scalar) else @intCast(scalar);
        const result = modPow512(c, @as(u512, abs_scalar), self.keys.n_squared);
        if (scalar < 0) {
            return modInverse512(result, self.keys.n_squared);
        }
        return result;
    }

    pub fn multiply(self: *Self, c1: u512, scalar: i64) u512 {
        return self.multiplyScalar(c1, scalar);
    }
};

fn secureZero(val: *u256) void {
    @as(*volatile u256, val).* = 0;
}

fn secureZeroBytes(buf: []u8) void {
    for (buf) |*b| {
        @as(*volatile u8, b).* = 0;
    }
}

pub const DatasetFingerprint = struct {
    allocator: Allocator,
    fingerprints: AutoHashMap([32]u8, FingerprintData),
    lsh_buckets: AutoHashMap(u64, ArrayList([32]u8)),
    num_hash_functions: usize,

    const FingerprintData = struct {
        sample_hash: [32]u8,
        encrypted_features: ArrayList(u512),
        timestamp: i64,
        access_count: u64,
        similarity_threshold: f64,
    };

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .fingerprints = AutoHashMap([32]u8, FingerprintData).init(allocator),
            .lsh_buckets = AutoHashMap(u64, ArrayList([32]u8)).init(allocator),
            .num_hash_functions = 8,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        var fp_iter = self.fingerprints.iterator();
        while (fp_iter.next()) |entry| {
            entry.value_ptr.encrypted_features.deinit();
        }
        self.fingerprints.deinit();

        var bucket_iter = self.lsh_buckets.iterator();
        while (bucket_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.lsh_buckets.deinit();

        self.allocator.destroy(self);
    }

    pub fn addSample(self: *Self, sample: []const u8, features: []const i64, he: *HomomorphicEncryption) !void {
        var hasher = Blake3.init(.{});
        hasher.update(sample);
        var len_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_bytes, @intCast(sample.len), .little);
        hasher.update(&len_bytes);
        var sample_hash: [32]u8 = undefined;
        hasher.final(&sample_hash);

        var encrypted_features = ArrayList(u512).init(self.allocator);
        for (features) |feat| {
            const encrypted = try he.encrypt(feat);
            try encrypted_features.append(encrypted);
        }

        const fingerprint = FingerprintData{
            .sample_hash = sample_hash,
            .encrypted_features = encrypted_features,
            .timestamp = @as(i64, @truncate(std.time.nanoTimestamp())),
            .access_count = 0,
            .similarity_threshold = 0.9,
        };

        if (self.fingerprints.fetchRemove(sample_hash)) |removed| {
            removed.value.encrypted_features.deinit();
        }
        try self.fingerprints.put(sample_hash, fingerprint);

        try self.indexToLSH(sample_hash, features);
    }

    fn indexToLSH(self: *Self, hash: [32]u8, features: []const i64) !void {
        var i: usize = 0;
        while (i < self.num_hash_functions) : (i += 1) {
            var lsh_hasher = Blake3.init(.{});
            lsh_hasher.update(&hash);
            const i_bytes = std.mem.asBytes(&i);
            lsh_hasher.update(i_bytes);

            if (features.len > 0) {
                const feat_idx = i % features.len;
                const feat_bytes = std.mem.asBytes(&features[feat_idx]);
                lsh_hasher.update(feat_bytes);
            }

            var lsh_hash: [32]u8 = undefined;
            lsh_hasher.final(&lsh_hash);

            const bucket_key = std.mem.readInt(u64, lsh_hash[0..8], .little);

            const gop = try self.lsh_buckets.getOrPut(bucket_key);
            if (!gop.found_existing) {
                gop.value_ptr.* = ArrayList([32]u8).init(self.allocator);
            }
            try gop.value_ptr.append(hash);
        }
    }

    pub fn checkSimilarity(self: *Self, query_hash: [32]u8) bool {
        if (self.fingerprints.contains(query_hash)) {
            return true;
        }

        var i: usize = 0;
        while (i < self.num_hash_functions) : (i += 1) {
            var lsh_hasher = Blake3.init(.{});
            lsh_hasher.update(&query_hash);
            const i_bytes = std.mem.asBytes(&i);
            lsh_hasher.update(i_bytes);

            var lsh_hash: [32]u8 = undefined;
            lsh_hasher.final(&lsh_hash);

            const bucket_key = std.mem.readInt(u64, lsh_hash[0..8], .little);

            if (self.lsh_buckets.get(bucket_key)) |bucket| {
                for (bucket.items) |stored_hash| {
                    if (hammingDistance(&query_hash, &stored_hash) < 32) {
                        return true;
                    }
                }
            }
        }

        return false;
    }
};

fn hammingDistance(a: *const [32]u8, b: *const [32]u8) u32 {
    var distance: u32 = 0;
    var i: usize = 0;
    while (i < 32) : (i += 1) {
        distance += @popCount(a[i] ^ b[i]);
    }
    return distance;
}

pub const SecureDataSampler = struct {
    allocator: Allocator,
    sample_pool: ArrayList(EncryptedSample),
    k_anonymity: usize,
    differential_privacy_budget: f64,
    consumed_budget: f64,

    const EncryptedSample = struct {
        id_hash: [32]u8,
        encrypted_data: ArrayList(u512),
        noise_signature: [32]u8,
        sensitivity: f64,
    };

    const Self = @This();

    pub fn init(allocator: Allocator, k_anonymity: usize, privacy_budget: f64) !*Self {
        if (k_anonymity == 0) return error.InvalidKAnonymity;
        if (privacy_budget <= 0.0) return error.InvalidPrivacyBudget;

        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .sample_pool = ArrayList(EncryptedSample).init(allocator),
            .k_anonymity = k_anonymity,
            .differential_privacy_budget = privacy_budget,
            .consumed_budget = 0.0,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        for (self.sample_pool.items) |*sample| {
            sample.encrypted_data.deinit();
        }
        self.sample_pool.deinit();
        self.allocator.destroy(self);
    }

    pub fn addEncryptedSample(self: *Self, data: []const i64, he: *HomomorphicEncryption, sensitivity: f64) !void {
        var id_bytes: [32]u8 = undefined;
        crypto.random.bytes(&id_bytes);

        var hasher = Sha256.init(.{});
        hasher.update(&id_bytes);
        var id_hash: [32]u8 = undefined;
        hasher.final(&id_hash);

        var encrypted_data = ArrayList(u512).init(self.allocator);
        for (data) |val| {
            const encrypted = try he.encrypt(val);
            try encrypted_data.append(encrypted);
        }

        var noise_sig: [32]u8 = undefined;
        crypto.random.bytes(&noise_sig);

        const sample = EncryptedSample{
            .id_hash = id_hash,
            .encrypted_data = encrypted_data,
            .noise_signature = noise_sig,
            .sensitivity = sensitivity,
        };

        try self.sample_pool.append(sample);
    }

    pub fn sampleWithKAnonymity(self: *Self, count: usize) !ArrayList(usize) {
        if (count < self.k_anonymity) {
            return error.InsufficientKAnonymity;
        }

        if (self.sample_pool.items.len < self.k_anonymity) {
            return error.InsufficientPoolSize;
        }

        const query_cost = @as(f64, @floatFromInt(count)) * 0.1;
        if (self.consumed_budget + query_cost > self.differential_privacy_budget) {
            return error.PrivacyBudgetExhausted;
        }

        self.consumed_budget += query_cost;

        var indices = ArrayList(usize).init(self.allocator);

        var shuffle_indices = try self.allocator.alloc(usize, self.sample_pool.items.len);
        defer self.allocator.free(shuffle_indices);

        var i: usize = 0;
        while (i < shuffle_indices.len) : (i += 1) {
            shuffle_indices[i] = i;
        }

        i = shuffle_indices.len;
        while (i > 1) {
            i -= 1;
            const j = crypto.random.uintLessThan(usize, i + 1);
            const temp = shuffle_indices[i];
            shuffle_indices[i] = shuffle_indices[j];
            shuffle_indices[j] = temp;
        }

        i = 0;
        while (i < count and i < shuffle_indices.len) : (i += 1) {
            try indices.append(shuffle_indices[i]);
        }

        return indices;
    }

    pub fn remainingBudget(self: *Self) f64 {
        return self.differential_privacy_budget - self.consumed_budget;
    }
};

pub const ProofOfCorrectness = struct {
    allocator: Allocator,
    computation_trace: ArrayList(TraceStep),
    final_commitment: [32]u8,
    merkle_root: [32]u8,
    chain_hash: [32]u8,

    pub const TraceStep = struct {
        step_number: u64,
        input_hash: [32]u8,
        output_hash: [32]u8,
        operation_type: OperationType,
        timestamp: i64,
    };

    pub const OperationType = enum {
        ScalarMultiply,
        MatrixMultiply,
        AffineCoupling,
        ScatterPermute,
        Aggregation,
        RSFForward,
        RSFInverse,
        OFTBMix,
        DifferentialPrivacy,
    };

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .computation_trace = ArrayList(TraceStep).init(allocator),
            .final_commitment = undefined,
            .merkle_root = undefined,
            .chain_hash = undefined,
        };
        @memset(&self.chain_hash, 0);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.computation_trace.deinit();
        self.allocator.destroy(self);
    }

    pub fn recordStep(self: *Self, step_num: u64, input: []const f32, output: []const f32, op_type: OperationType) !void {
        var input_hasher = Blake3.init(.{});
        const input_len_bytes = std.mem.asBytes(&input.len);
        input_hasher.update(input_len_bytes);
        for (input) |val| {
            const bytes = std.mem.asBytes(&val);
            input_hasher.update(bytes);
        }
        var input_hash: [32]u8 = undefined;
        input_hasher.final(&input_hash);

        var output_hasher = Blake3.init(.{});
        const output_len_bytes = std.mem.asBytes(&output.len);
        output_hasher.update(output_len_bytes);
        for (output) |val| {
            const bytes = std.mem.asBytes(&val);
            output_hasher.update(bytes);
        }
        var output_hash: [32]u8 = undefined;
        output_hasher.final(&output_hash);

        var chain_hasher = Blake3.init(.{});
        chain_hasher.update(&self.chain_hash);
        chain_hasher.update(&input_hash);
        chain_hasher.update(&output_hash);
        chain_hasher.final(&self.chain_hash);

        const step = TraceStep{
            .step_number = step_num,
            .input_hash = input_hash,
            .output_hash = output_hash,
            .operation_type = op_type,
            .timestamp = @as(i64, @truncate(std.time.nanoTimestamp())),
        };

        try self.computation_trace.append(step);
    }

    pub fn finalize(self: *Self) !void {
        try self.computeMerkleRoot();

        var hasher = Blake3.init(.{});
        hasher.update(&self.merkle_root);
        hasher.update(&self.chain_hash);

        const trace_len_bytes = std.mem.asBytes(&self.computation_trace.items.len);
        hasher.update(trace_len_bytes);

        hasher.final(&self.final_commitment);
    }

    fn computeMerkleRoot(self: *Self) !void {
        if (self.computation_trace.items.len == 0) {
            @memset(&self.merkle_root, 0);
            return;
        }

        var hashes = ArrayList([32]u8).init(self.allocator);
        defer hashes.deinit();

        for (self.computation_trace.items) |step| {
            var step_hasher = Blake3.init(.{});
            const step_num_bytes = std.mem.asBytes(&step.step_number);
            step_hasher.update(step_num_bytes);
            step_hasher.update(&step.input_hash);
            step_hasher.update(&step.output_hash);
            var step_hash: [32]u8 = undefined;
            step_hasher.final(&step_hash);
            try hashes.append(step_hash);
        }

        while (hashes.items.len > 1) {
            var next_level = ArrayList([32]u8).init(self.allocator);
            defer next_level.deinit();

            var i: usize = 0;
            while (i < hashes.items.len) : (i += 2) {
                var pair_hasher = Blake3.init(.{});
                pair_hasher.update(&hashes.items[i]);
                if (i + 1 < hashes.items.len) {
                    pair_hasher.update(&hashes.items[i + 1]);
                } else {
                    pair_hasher.update(&hashes.items[i]);
                }
                var pair_hash: [32]u8 = undefined;
                pair_hasher.final(&pair_hash);
                try next_level.append(pair_hash);
            }

            hashes.clearRetainingCapacity();
            try hashes.appendSlice(next_level.items);
        }

        self.merkle_root = hashes.items[0];
    }

    pub fn verify(self: *Self, expected_commitment: [32]u8) bool {
        return std.mem.eql(u8, &self.final_commitment, &expected_commitment);
    }

    pub fn getChainHash(self: *Self) [32]u8 {
        return self.chain_hash;
    }
};

pub const DatasetIsolation = struct {
    allocator: Allocator,
    isolation_barriers: ArrayList(IsolationBarrier),
    access_control: AutoHashMap(u64, AccessPolicy),

    const IsolationBarrier = struct {
        dataset_id: u64,
        access_key: [32]u8,
        encrypted_metadata: ArrayList(u8),
        access_log: ArrayList(AccessRecord),
        creation_time: i64,
        expiry_time: ?i128,
    };

    const AccessPolicy = struct {
        max_accesses: u64,
        current_accesses: u64,
        allowed_operations: u32,
        rate_limit_per_second: u32,
        last_access_time: i64,
    };

    const AccessRecord = struct {
        timestamp: i64,
        operation_hash: [32]u8,
        success: bool,
        client_hash: [32]u8,
    };

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = Self{
            .allocator = allocator,
            .isolation_barriers = ArrayList(IsolationBarrier).init(allocator),
            .access_control = AutoHashMap(u64, AccessPolicy).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        for (self.isolation_barriers.items) |*barrier| {
            secureZeroBytes(barrier.encrypted_metadata.items);
            barrier.encrypted_metadata.deinit();
            barrier.access_log.deinit();
        }
        self.isolation_barriers.deinit();
        self.access_control.deinit();
        self.allocator.destroy(self);
    }

    pub fn createBarrier(self: *Self, dataset_id: u64, expiry_seconds: ?i64) !u64 {
        var access_key: [32]u8 = undefined;
        crypto.random.bytes(&access_key);

        const current_time = @as(i64, @truncate(std.time.nanoTimestamp()));
        const expiry = if (expiry_seconds) |secs| current_time + @as(i128, secs) * std.time.ns_per_s else null;

        const barrier = IsolationBarrier{
            .dataset_id = dataset_id,
            .access_key = access_key,
            .encrypted_metadata = ArrayList(u8).init(self.allocator),
            .access_log = ArrayList(AccessRecord).init(self.allocator),
            .creation_time = current_time,
            .expiry_time = expiry,
        };

        try self.isolation_barriers.append(barrier);

        const policy = AccessPolicy{
            .max_accesses = 10000,
            .current_accesses = 0,
            .allowed_operations = 0xFFFFFFFF,
            .rate_limit_per_second = 100,
            .last_access_time = 0,
        };
        try self.access_control.put(dataset_id, policy);

        return self.isolation_barriers.items.len - 1;
    }

    pub fn logAccess(self: *Self, barrier_idx: usize, operation: []const u8, success: bool, client_id: []const u8) !void {
        if (barrier_idx >= self.isolation_barriers.items.len) {
            return error.InvalidBarrier;
        }

        const barrier = &self.isolation_barriers.items[barrier_idx];

        if (barrier.expiry_time) |expiry| {
            if (@as(i64, @truncate(std.time.nanoTimestamp())) > expiry) {
                return error.BarrierExpired;
            }
        }

        if (self.access_control.getPtr(barrier.dataset_id)) |policy| {
            if (policy.current_accesses >= policy.max_accesses) {
                return error.AccessLimitExceeded;
            }
            policy.current_accesses += 1;
            policy.last_access_time = @as(i64, @truncate(std.time.nanoTimestamp()));
        }

        var op_hasher = Blake3.init(.{});
        op_hasher.update(operation);
        var op_hash: [32]u8 = undefined;
        op_hasher.final(&op_hash);

        var client_hasher = Blake3.init(.{});
        client_hasher.update(client_id);
        var client_hash: [32]u8 = undefined;
        client_hasher.final(&client_hash);

        const record = AccessRecord{
            .timestamp = @as(i64, @truncate(std.time.nanoTimestamp())),
            .operation_hash = op_hash,
            .success = success,
            .client_hash = client_hash,
        };

        try barrier.access_log.append(record);
    }

    pub fn verifyAccess(self: *Self, barrier_idx: usize, access_key: [32]u8) bool {
        if (barrier_idx >= self.isolation_barriers.items.len) {
            return false;
        }

        const barrier = &self.isolation_barriers.items[barrier_idx];

        if (barrier.expiry_time) |expiry| {
            if (@as(i64, @truncate(std.time.nanoTimestamp())) > expiry) {
                return false;
            }
        }

        return std.mem.eql(u8, &barrier.access_key, &access_key);
    }
};

