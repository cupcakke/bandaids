const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const Complex = std.math.Complex;

pub const LogicGate = enum(u8) {
    HADAMARD = 0,
    PAULI_X = 1,
    PAULI_Y = 2,
    PAULI_Z = 3,
    PHASE = 4,
    CNOT = 5,
    TOFFOLI = 6,
    RELATIONAL_AND = 7,
    RELATIONAL_OR = 8,
    RELATIONAL_NOT = 9,
    RELATIONAL_XOR = 10,
    FRACTAL_TRANSFORM = 11,

    pub fn toString(self: LogicGate) []const u8 {
        return switch (self) {
            .HADAMARD => "hadamard",
            .PAULI_X => "pauli_x",
            .PAULI_Y => "pauli_y",
            .PAULI_Z => "pauli_z",
            .PHASE => "phase",
            .CNOT => "cnot",
            .TOFFOLI => "toffoli",
            .RELATIONAL_AND => "relational_and",
            .RELATIONAL_OR => "relational_or",
            .RELATIONAL_NOT => "relational_not",
            .RELATIONAL_XOR => "relational_xor",
            .FRACTAL_TRANSFORM => "fractal_transform",
        };
    }

    pub fn fromString(s: []const u8) ?LogicGate {
        if (std.mem.eql(u8, s, "hadamard")) return .HADAMARD;
        if (std.mem.eql(u8, s, "pauli_x")) return .PAULI_X;
        if (std.mem.eql(u8, s, "pauli_y")) return .PAULI_Y;
        if (std.mem.eql(u8, s, "pauli_z")) return .PAULI_Z;
        if (std.mem.eql(u8, s, "phase")) return .PHASE;
        if (std.mem.eql(u8, s, "cnot")) return .CNOT;
        if (std.mem.eql(u8, s, "toffoli")) return .TOFFOLI;
        if (std.mem.eql(u8, s, "relational_and")) return .RELATIONAL_AND;
        if (std.mem.eql(u8, s, "relational_or")) return .RELATIONAL_OR;
        if (std.mem.eql(u8, s, "relational_not")) return .RELATIONAL_NOT;
        if (std.mem.eql(u8, s, "relational_xor")) return .RELATIONAL_XOR;
        if (std.mem.eql(u8, s, "fractal_transform")) return .FRACTAL_TRANSFORM;
        return null;
    }

    pub fn isSingleQubit(self: LogicGate) bool {
        return switch (self) {
            .HADAMARD, .PAULI_X, .PAULI_Y, .PAULI_Z, .PHASE, .RELATIONAL_NOT, .FRACTAL_TRANSFORM => true,
            .CNOT, .RELATIONAL_AND, .RELATIONAL_OR, .RELATIONAL_XOR => false,
            .TOFFOLI => false,
        };
    }

    pub fn requiredQubits(self: LogicGate) usize {
        return switch (self) {
            .HADAMARD, .PAULI_X, .PAULI_Y, .PAULI_Z, .PHASE, .RELATIONAL_NOT, .FRACTAL_TRANSFORM => 1,
            .CNOT, .RELATIONAL_AND, .RELATIONAL_OR, .RELATIONAL_XOR => 2,
            .TOFFOLI => 3,
        };
    }
};

pub const QuantumState = struct {
    amplitudes: [2]Complex(f64),
    phase: f64,
    entanglement_degree: f64,

    const Self = @This();

    pub fn init(alpha_real: f64, alpha_imag: f64, beta_real: f64, beta_imag: f64, phase_val: f64, entangle: f64) Self {
        return Self{
            .amplitudes = .{
                Complex(f64).init(alpha_real, alpha_imag),
                Complex(f64).init(beta_real, beta_imag),
            },
            .phase = phase_val,
            .entanglement_degree = entangle,
        };
    }

    pub fn initBasis(is_one: bool, phase_val: f64, entangle: f64) Self {
        if (is_one) {
            return Self{
                .amplitudes = .{
                    Complex(f64).init(0.0, 0.0),
                    Complex(f64).init(1.0, 0.0),
                },
                .phase = phase_val,
                .entanglement_degree = entangle,
            };
        } else {
            return Self{
                .amplitudes = .{
                    Complex(f64).init(1.0, 0.0),
                    Complex(f64).init(0.0, 0.0),
                },
                .phase = phase_val,
                .entanglement_degree = entangle,
            };
        }
    }

    pub fn initFromComplex(alpha: Complex(f64), beta: Complex(f64), phase_val: f64, entangle: f64) Self {
        return Self{
            .amplitudes = .{ alpha, beta },
            .phase = phase_val,
            .entanglement_degree = entangle,
        };
    }

    pub fn normalize(self: *Self) void {
        const total_mag = self.totalMagnitude();
        if (total_mag > 0.0) {
            self.amplitudes[0] = Complex(f64).init(self.amplitudes[0].re / total_mag, self.amplitudes[0].im / total_mag);
            self.amplitudes[1] = Complex(f64).init(self.amplitudes[1].re / total_mag, self.amplitudes[1].im / total_mag);
        }
    }

    pub fn totalMagnitude(self: *const Self) f64 {
        return @sqrt(self.totalProbability());
    }

    pub fn totalProbability(self: *const Self) f64 {
        return self.prob0() + self.prob1();
    }

    pub fn prob0(self: *const Self) f64 {
        return self.amplitudes[0].re * self.amplitudes[0].re + self.amplitudes[0].im * self.amplitudes[0].im;
    }

    pub fn prob1(self: *const Self) f64 {
        return self.amplitudes[1].re * self.amplitudes[1].re + self.amplitudes[1].im * self.amplitudes[1].im;
    }

    pub fn conjugate(self: *const Self) Self {
        return Self{
            .amplitudes = .{
                Complex(f64).init(self.amplitudes[0].re, -self.amplitudes[0].im),
                Complex(f64).init(self.amplitudes[1].re, -self.amplitudes[1].im),
            },
            .phase = -self.phase,
            .entanglement_degree = self.entanglement_degree,
        };
    }

    pub fn add(self: *const Self, other: *const Self) Self {
        const sum_0_re = self.amplitudes[0].re + other.amplitudes[0].re;
        const sum_0_im = self.amplitudes[0].im + other.amplitudes[0].im;
        const sum_1_re = self.amplitudes[1].re + other.amplitudes[1].re;
        const sum_1_im = self.amplitudes[1].im + other.amplitudes[1].im;
        const norm_sq = sum_0_re * sum_0_re + sum_0_im * sum_0_im + sum_1_re * sum_1_re + sum_1_im * sum_1_im;
        if (norm_sq < 1e-30) {
            return Self{
                .amplitudes = .{
                    Complex(f64).init(1.0, 0.0),
                    Complex(f64).init(0.0, 0.0),
                },
                .phase = 0.0,
                .entanglement_degree = @max(self.entanglement_degree, other.entanglement_degree),
            };
        }
        const norm = std.math.sqrt(norm_sq);
        const result = Self{
            .amplitudes = .{
                Complex(f64).init(sum_0_re / norm, sum_0_im / norm),
                Complex(f64).init(sum_1_re / norm, sum_1_im / norm),
            },
            .phase = std.math.atan2(sum_0_im, sum_0_re),
            .entanglement_degree = @max(self.entanglement_degree, other.entanglement_degree),
        };
        return result;
    }

    pub fn scale(self: *const Self, factor: f64) Self {
        return Self{
            .amplitudes = .{
                Complex(f64).init(self.amplitudes[0].re * factor, self.amplitudes[0].im * factor),
                Complex(f64).init(self.amplitudes[1].re * factor, self.amplitudes[1].im * factor),
            },
            .phase = self.phase,
            .entanglement_degree = self.entanglement_degree,
        };
    }

    pub fn clone(self: *const Self) Self {
        return Self{
            .amplitudes = .{ self.amplitudes[0], self.amplitudes[1] },
            .phase = self.phase,
            .entanglement_degree = self.entanglement_degree,
        };
    }

    pub fn isNormalized(self: *const Self, epsilon: f64) bool {
        const prob = self.totalProbability();
        return @abs(prob - 1.0) < epsilon;
    }

    pub fn fidelity(self: *const Self, other: *const Self) f64 {
        const inner_real = self.amplitudes[0].re * other.amplitudes[0].re + self.amplitudes[0].im * other.amplitudes[0].im + self.amplitudes[1].re * other.amplitudes[1].re + self.amplitudes[1].im * other.amplitudes[1].im;
        const inner_imag = self.amplitudes[0].re * other.amplitudes[0].im - self.amplitudes[0].im * other.amplitudes[0].re + self.amplitudes[1].re * other.amplitudes[1].im - self.amplitudes[1].im * other.amplitudes[1].re;
        return inner_real * inner_real + inner_imag * inner_imag;
    }
};

pub const GateHistoryEntry = struct {
    gate: LogicGate,
    indices: ArrayList(usize),
    params: ?ArrayList(f64),
    timestamp: i64,

    const Self = @This();

    pub fn initEntry(allocator: Allocator, gate: LogicGate, indices: []const usize, params: ?[]const f64) !Self {
        var indices_list = ArrayList(usize).init(allocator);
        for (indices) |idx| {
            try indices_list.append(idx);
        }

        var params_list: ?ArrayList(f64) = null;
        if (params) |p| {
            params_list = ArrayList(f64).init(allocator);
            for (p) |param| {
                try params_list.?.append(param);
            }
        }

        const raw_ts = @as(i64, @truncate(std.time.nanoTimestamp()));
        const ts: i64 = if (raw_ts > std.math.maxInt(i64))
            std.math.maxInt(i64)
        else if (raw_ts < std.math.minInt(i64))
            std.math.minInt(i64)
        else
            @intCast(raw_ts);

        return Self{
            .gate = gate,
            .indices = indices_list,
            .params = params_list,
            .timestamp = ts,
        };
    }

    pub fn deinit(self: *Self) void {
        self.indices.deinit();
        if (self.params) |*p| {
            p.deinit();
        }
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        var indices_copy = ArrayList(usize).init(allocator);
        for (self.indices.items) |idx| {
            try indices_copy.append(idx);
        }

        var params_copy: ?ArrayList(f64) = null;
        if (self.params) |p| {
            params_copy = ArrayList(f64).init(allocator);
            for (p.items) |param| {
                try params_copy.?.append(param);
            }
        }

        return Self{
            .gate = self.gate,
            .indices = indices_copy,
            .params = params_copy,
            .timestamp = self.timestamp,
        };
    }
};

pub const GateSequenceEntry = struct {
    gate: LogicGate,
    indices: []const usize,
    params: ?[]const f64,
};

pub const MeasurementResult = struct {
    result: i32,
    probability_zero: f64,
    probability_one: f64,
    collapsed_state: QuantumState,
};

pub const RelationalQuantumLogic = struct {
    states: ArrayList(QuantumState),
    gate_history: ArrayList(GateHistoryEntry),
    allocator: Allocator,
    coherence_threshold: f64,
    max_entanglement_depth: usize,
    max_states: usize,

    const Self = @This();
    const DEFAULT_COHERENCE_THRESHOLD: f64 = 1e-10;
    const DEFAULT_MAX_ENTANGLEMENT_DEPTH: usize = 64;
    const DEFAULT_MAX_STATES: usize = 1024;
    const SQRT2_INV: f64 = 0.7071067811865476;
    const DEFAULT_PHASE_ANGLE: f64 = 0.7853981633974483;

    pub fn init(allocator: Allocator) Self {
        return Self{
            .states = ArrayList(QuantumState).init(allocator),
            .gate_history = ArrayList(GateHistoryEntry).init(allocator),
            .allocator = allocator,
            .coherence_threshold = DEFAULT_COHERENCE_THRESHOLD,
            .max_entanglement_depth = DEFAULT_MAX_ENTANGLEMENT_DEPTH,
            .max_states = DEFAULT_MAX_STATES,
        };
    }

    pub fn initWithOptions(allocator: Allocator, coherence_threshold: f64, max_entanglement_depth: usize) Self {
        return Self{
            .states = ArrayList(QuantumState).init(allocator),
            .gate_history = ArrayList(GateHistoryEntry).init(allocator),
            .allocator = allocator,
            .coherence_threshold = coherence_threshold,
            .max_entanglement_depth = max_entanglement_depth,
            .max_states = DEFAULT_MAX_STATES,
        };
    }

    pub fn deinit(self: *Self) void {
        self.states.deinit();
        for (self.gate_history.items) |*entry| {
            entry.deinit();
        }
        self.gate_history.deinit();
    }

    pub fn reset(self: *Self) void {
        self.states.clearRetainingCapacity();
        for (self.gate_history.items) |*entry| {
            entry.deinit();
        }
        self.gate_history.clearRetainingCapacity();
    }

    pub fn initializeState(self: *Self, alpha_real: f64, alpha_imag: f64, beta_real: f64, beta_imag: f64, phase: f64) !usize {
        if (self.states.items.len >= self.max_states) return error.TooManyStates;
        var state = QuantumState.init(alpha_real, alpha_imag, beta_real, beta_imag, phase, 0.0);
        state.normalize();
        try self.states.append(state);
        return self.states.items.len - 1;
    }

    pub fn initializeStateFromComplex(self: *Self, alpha: Complex(f64), beta: Complex(f64), phase: f64) !usize {
        if (self.states.items.len >= self.max_states) return error.TooManyStates;
        var state = QuantumState.initFromComplex(alpha, beta, phase, 0.0);
        state.normalize();
        try self.states.append(state);
        return self.states.items.len - 1;
    }

    pub fn initializeBasisState(self: *Self, is_one: bool) !usize {
        if (self.states.items.len >= self.max_states) return error.TooManyStates;
        var state = QuantumState.initBasis(is_one, 0.0, 0.0);
        state.normalize();
        try self.states.append(state);
        return self.states.items.len - 1;
    }

    pub fn getState(self: *const Self, qubit_index: usize) ?QuantumState {
        if (qubit_index >= self.states.items.len) {
            return null;
        }
        return self.states.items[qubit_index];
    }

    pub fn getStatePtr(self: *Self, qubit_index: usize) ?*QuantumState {
        if (qubit_index >= self.states.items.len) {
            return null;
        }
        return &self.states.items[qubit_index];
    }

    pub fn stateCount(self: *const Self) usize {
        return self.states.items.len;
    }

    pub fn historyCount(self: *const Self) usize {
        return self.gate_history.items.len;
    }

    fn validateDistinctIndices(indices: []const usize) !void {
        var i: usize = 0;
        while (i < indices.len) : (i += 1) {
            var j: usize = i + 1;
            while (j < indices.len) : (j += 1) {
                if (indices[i] == indices[j]) {
                    return error.DuplicateQubitIndex;
                }
            }
        }
    }

    pub fn applyGate(self: *Self, gate: LogicGate, qubit_indices: []const usize, params: ?[]const f64) !void {
        const required = gate.requiredQubits();
        if (qubit_indices.len < required) {
            return error.InsufficientQubits;
        }

        for (qubit_indices) |idx| {
            if (idx >= self.states.items.len) {
                return error.InvalidQubitIndex;
            }
        }

        if (required > 1) {
            try validateDistinctIndices(qubit_indices[0..required]);
        }

        switch (gate) {
            .HADAMARD => self.applyHadamard(qubit_indices[0]),
            .PAULI_X => self.applyPauliX(qubit_indices[0]),
            .PAULI_Y => self.applyPauliY(qubit_indices[0]),
            .PAULI_Z => self.applyPauliZ(qubit_indices[0]),
            .PHASE => {
                const theta = if (params != null and params.?.len > 0) params.?[0] else DEFAULT_PHASE_ANGLE;
                self.applyPhase(qubit_indices[0], theta);
            },
            .CNOT => self.applyCNOT(qubit_indices[0], qubit_indices[1]),
            .TOFFOLI => self.applyToffoli(qubit_indices[0], qubit_indices[1], qubit_indices[2]),
            .RELATIONAL_AND => try self.applyRelationalAnd(qubit_indices[0], qubit_indices[1]),
            .RELATIONAL_OR => try self.applyRelationalOr(qubit_indices[0], qubit_indices[1]),
            .RELATIONAL_NOT => self.applyRelationalNot(qubit_indices[0]),
            .RELATIONAL_XOR => try self.applyRelationalXor(qubit_indices[0], qubit_indices[1]),
            .FRACTAL_TRANSFORM => {
                const depth: u32 = if (params != null and params.?.len > 0) blk: {
                    const val = params.?[0];
                    if (std.math.isNan(val) or std.math.isInf(val) or val < 0.0 or val > 100.0) {
                        break :blk 3;
                    }
                    break :blk @intFromFloat(val);
                } else 3;
                self.applyFractalTransform(qubit_indices[0], depth);
            },
        }

        const history_entry = try GateHistoryEntry.initEntry(self.allocator, gate, qubit_indices, params);
        try self.gate_history.append(history_entry);
    }

    fn applyHadamard(self: *Self, qubit_idx: usize) void {
        if (qubit_idx >= self.states.items.len) return;
        var state = &self.states.items[qubit_idx];
        const a0 = state.amplitudes[0];
        const a1 = state.amplitudes[1];
        state.amplitudes[0] = Complex(f64).init(
            (a0.re + a1.re) * SQRT2_INV,
            (a0.im + a1.im) * SQRT2_INV,
        );
        state.amplitudes[1] = Complex(f64).init(
            (a0.re - a1.re) * SQRT2_INV,
            (a0.im - a1.im) * SQRT2_INV,
        );
    }

    fn applyPauliX(self: *Self, qubit_idx: usize) void {
        if (qubit_idx >= self.states.items.len) return;
        var state = &self.states.items[qubit_idx];
        const tmp = state.amplitudes[0];
        state.amplitudes[0] = state.amplitudes[1];
        state.amplitudes[1] = tmp;
    }

    fn applyPauliY(self: *Self, qubit_idx: usize) void {
        if (qubit_idx >= self.states.items.len) return;
        var state = &self.states.items[qubit_idx];
        const a0 = state.amplitudes[0];
        const a1 = state.amplitudes[1];
        state.amplitudes[0] = Complex(f64).init(a1.im, -a1.re);
        state.amplitudes[1] = Complex(f64).init(-a0.im, a0.re);
    }

    fn applyPauliZ(self: *Self, qubit_idx: usize) void {
        if (qubit_idx >= self.states.items.len) return;
        var state = &self.states.items[qubit_idx];
        state.amplitudes[1] = Complex(f64).init(-state.amplitudes[1].re, -state.amplitudes[1].im);
    }

    fn applyPhase(self: *Self, qubit_idx: usize, theta: f64) void {
        if (qubit_idx >= self.states.items.len) return;
        var state = &self.states.items[qubit_idx];
        state.phase += theta;
        const cos_theta = @cos(theta);
        const sin_theta = @sin(theta);
        const a1 = state.amplitudes[1];
        state.amplitudes[1] = Complex(f64).init(
            a1.re * cos_theta - a1.im * sin_theta,
            a1.re * sin_theta + a1.im * cos_theta,
        );
    }

    fn applyCNOT(self: *Self, control_idx: usize, target_idx: usize) void {
        if (control_idx >= self.states.items.len or target_idx >= self.states.items.len) return;

        var target = &self.states.items[target_idx];
        const control = self.states.items[control_idx];

        const c0_prob = control.prob0();
        const c1_prob = control.prob1();

        const old_t0 = target.amplitudes[0];
        const old_t1 = target.amplitudes[1];

        target.amplitudes[0] = Complex(f64).init(
            old_t0.re * c0_prob + old_t1.re * c1_prob,
            old_t0.im * c0_prob + old_t1.im * c1_prob,
        );
        target.amplitudes[1] = Complex(f64).init(
            old_t1.re * c0_prob + old_t0.re * c1_prob,
            old_t1.im * c0_prob + old_t0.im * c1_prob,
        );

        const superposition_term = 2.0 * @sqrt(c0_prob * c1_prob);
        const entanglement_inc = @min(1.0, @max(superposition_term, c1_prob));
        self.states.items[control_idx].entanglement_degree = @min(1.0, self.states.items[control_idx].entanglement_degree + entanglement_inc);
        target.entanglement_degree = @min(1.0, target.entanglement_degree + entanglement_inc);

        self.states.items[control_idx].normalize();
        target.normalize();
    }

    fn applyToffoli(self: *Self, control1_idx: usize, control2_idx: usize, target_idx: usize) void {
        if (control1_idx >= self.states.items.len or
            control2_idx >= self.states.items.len or
            target_idx >= self.states.items.len) return;

        const control1 = self.states.items[control1_idx];
        const control2 = self.states.items[control2_idx];

        if (control1.prob1() > 0.5 and control2.prob1() > 0.5) {
            var target = &self.states.items[target_idx];
            const tmp = target.amplitudes[0];
            target.amplitudes[0] = target.amplitudes[1];
            target.amplitudes[1] = tmp;
        }

        var c1 = &self.states.items[control1_idx];
        c1.entanglement_degree = @min(1.0, c1.entanglement_degree + 0.33);
        var c2 = &self.states.items[control2_idx];
        c2.entanglement_degree = @min(1.0, c2.entanglement_degree + 0.33);
        var tgt = &self.states.items[target_idx];
        tgt.entanglement_degree = @min(1.0, tgt.entanglement_degree + 0.33);
    }

    fn applyRelationalAnd(self: *Self, idx1: usize, idx2: usize) !void {
        if (idx1 >= self.states.items.len or idx2 >= self.states.items.len) return error.InvalidQubitIndex;
        if (self.states.items.len >= self.max_states) return error.TooManyStates;

        const state1 = self.states.items[idx1];
        const state2 = self.states.items[idx2];

        const a0_re = state1.amplitudes[0].re * state2.amplitudes[0].re - state1.amplitudes[0].im * state2.amplitudes[0].im;
        const a0_im = state1.amplitudes[0].re * state2.amplitudes[0].im + state1.amplitudes[0].im * state2.amplitudes[0].re;
        const a1_re = state1.amplitudes[1].re * state2.amplitudes[1].re - state1.amplitudes[1].im * state2.amplitudes[1].im;
        const a1_im = state1.amplitudes[1].re * state2.amplitudes[1].im + state1.amplitudes[1].im * state2.amplitudes[1].re;

        var result_state = QuantumState.init(
            a0_re,
            a0_im,
            a1_re,
            a1_im,
            (state1.phase + state2.phase) / 2.0,
            @min(1.0, state1.entanglement_degree + state2.entanglement_degree),
        );
        result_state.normalize();
        try self.states.append(result_state);
    }

    fn applyRelationalOr(self: *Self, idx1: usize, idx2: usize) !void {
        if (idx1 >= self.states.items.len or idx2 >= self.states.items.len) return error.InvalidQubitIndex;
        if (self.states.items.len >= self.max_states) return error.TooManyStates;

        const state1 = self.states.items[idx1];
        const state2 = self.states.items[idx2];

        var result_state = QuantumState.init(
            state1.amplitudes[0].re + state2.amplitudes[0].re,
            state1.amplitudes[0].im + state2.amplitudes[0].im,
            state1.amplitudes[1].re + state2.amplitudes[1].re,
            state1.amplitudes[1].im + state2.amplitudes[1].im,
            (state1.phase + state2.phase) / 2.0,
            @max(state1.entanglement_degree, state2.entanglement_degree),
        );
        result_state.normalize();
        try self.states.append(result_state);
    }

    fn applyRelationalNot(self: *Self, idx: usize) void {
        if (idx >= self.states.items.len) return;
        var state = &self.states.items[idx];
        const tmp = state.amplitudes[0];
        state.amplitudes[0] = state.amplitudes[1];
        state.amplitudes[1] = tmp;
        state.phase += std.math.pi;
        state.normalize();
    }

    fn applyRelationalXor(self: *Self, idx1: usize, idx2: usize) !void {
        if (idx1 >= self.states.items.len or idx2 >= self.states.items.len) return error.InvalidQubitIndex;
        if (self.states.items.len >= self.max_states) return error.TooManyStates;

        const state1 = self.states.items[idx1];
        const state2 = self.states.items[idx2];

        var result_state = QuantumState.init(
            state1.amplitudes[0].re - state2.amplitudes[0].re,
            state1.amplitudes[0].im - state2.amplitudes[0].im,
            state1.amplitudes[1].re - state2.amplitudes[1].re,
            state1.amplitudes[1].im - state2.amplitudes[1].im,
            @abs(state1.phase - state2.phase),
            (state1.entanglement_degree + state2.entanglement_degree) / 2.0,
        );
        result_state.normalize();
        try self.states.append(result_state);
    }

    fn applyFractalTransform(self: *Self, idx: usize, depth: u32) void {
        if (idx >= self.states.items.len) return;
        var state = &self.states.items[idx];

        const orig_phase = state.phase;
        var i: u32 = 0;
        while (i < depth) : (i += 1) {
            const scale_factor = 1.0 / std.math.pow(f64, 2.0, @as(f64, @floatFromInt(i)));
            const angle = orig_phase * scale_factor;
            const cos_angle = @cos(angle);
            const sin_angle = @sin(angle);

            const new_re_0 = state.amplitudes[0].re * cos_angle - state.amplitudes[0].im * sin_angle;
            const new_im_0 = state.amplitudes[0].re * sin_angle + state.amplitudes[0].im * cos_angle;
            const new_re_1 = state.amplitudes[1].re * cos_angle - state.amplitudes[1].im * sin_angle;
            const new_im_1 = state.amplitudes[1].re * sin_angle + state.amplitudes[1].im * cos_angle;

            state.amplitudes[0] = Complex(f64).init(new_re_0, new_im_0);
            state.amplitudes[1] = Complex(f64).init(new_re_1, new_im_1);
            state.normalize();
        }
        state.phase = std.math.atan2(state.amplitudes[0].im, state.amplitudes[0].re);
        state.normalize();
    }

    pub fn measure(self: *Self, qubit_idx: usize) MeasurementResult {
        if (qubit_idx >= self.states.items.len) {
            return MeasurementResult{
                .result = 0,
                .probability_zero = 0.0,
                .probability_one = 0.0,
                .collapsed_state = QuantumState.initBasis(false, 0.0, 0.0),
            };
        }

        var state = &self.states.items[qubit_idx];
        const p0 = state.prob0();
        const p1 = state.prob1();
        const total = p0 + p1;
        const norm_p0 = if (total > 0.0) p0 / total else 0.5;
        const random_val = std.crypto.random.float(f64);
        const result: i32 = if (random_val < norm_p0) 0 else 1;

        if (result == 1) {
            state.amplitudes[0] = Complex(f64).init(0.0, 0.0);
            state.amplitudes[1] = Complex(f64).init(1.0, 0.0);
        } else {
            state.amplitudes[0] = Complex(f64).init(1.0, 0.0);
            state.amplitudes[1] = Complex(f64).init(0.0, 0.0);
        }
        state.entanglement_degree = 0.0;

        return MeasurementResult{
            .result = result,
            .probability_zero = p0,
            .probability_one = p1,
            .collapsed_state = state.clone(),
        };
    }

    pub fn measureWithRandomness(self: *Self, qubit_idx: usize, random_value: f64) MeasurementResult {
        if (qubit_idx >= self.states.items.len) {
            return MeasurementResult{
                .result = 0,
                .probability_zero = 0.0,
                .probability_one = 0.0,
                .collapsed_state = QuantumState.initBasis(false, 0.0, 0.0),
            };
        }

        var state = &self.states.items[qubit_idx];
        const p0 = state.prob0();
        const p1 = state.prob1();
        const total = p0 + p1;
        const normalized_p0 = if (total > 0.0) p0 / total else 0.5;
        const result: i32 = if (random_value >= normalized_p0) 1 else 0;

        if (result == 1) {
            state.amplitudes[0] = Complex(f64).init(0.0, 0.0);
            state.amplitudes[1] = Complex(f64).init(1.0, 0.0);
        } else {
            state.amplitudes[0] = Complex(f64).init(1.0, 0.0);
            state.amplitudes[1] = Complex(f64).init(0.0, 0.0);
        }
        state.entanglement_degree = 0.0;

        return MeasurementResult{
            .result = result,
            .probability_zero = p0,
            .probability_one = p1,
            .collapsed_state = state.clone(),
        };
    }

    pub fn entangle(self: *Self, idx1: usize, idx2: usize) !void {
        if (idx1 >= self.states.items.len or idx2 >= self.states.items.len) return error.InvalidQubitIndex;
        if (idx1 == idx2) return error.DuplicateQubitIndex;

        var state1 = &self.states.items[idx1];
        var state2 = &self.states.items[idx2];

        const bell_a0 = Complex(f64).init(
            (state1.amplitudes[0].re * state2.amplitudes[0].re + state1.amplitudes[1].re * state2.amplitudes[1].re) * SQRT2_INV,
            (state1.amplitudes[0].im * state2.amplitudes[0].im + state1.amplitudes[1].im * state2.amplitudes[1].im) * SQRT2_INV,
        );
        const bell_a1 = Complex(f64).init(
            (state1.amplitudes[0].re * state2.amplitudes[1].re + state1.amplitudes[1].re * state2.amplitudes[0].re) * SQRT2_INV,
            (state1.amplitudes[0].im * state2.amplitudes[1].im + state1.amplitudes[1].im * state2.amplitudes[0].im) * SQRT2_INV,
        );

        state1.amplitudes[0] = bell_a0;
        state1.amplitudes[1] = bell_a1;
        state2.amplitudes[0] = bell_a0;
        state2.amplitudes[1] = bell_a1;

        state1.entanglement_degree = 1.0;
        state2.entanglement_degree = 1.0;

        state1.normalize();
        state2.normalize();
    }

    pub fn computeRelationalOutput(
        self: *Self,
        input_indices: []const usize,
        gate_sequence: []const GateSequenceEntry,
    ) !ArrayList(Complex(f64)) {
        for (gate_sequence) |entry| {
            try self.applyGate(entry.gate, entry.indices, entry.params);
        }

        var result = ArrayList(Complex(f64)).init(self.allocator);
        errdefer result.deinit();
        for (input_indices) |idx| {
            if (idx >= self.states.items.len) {
                return error.InvalidQubitIndex;
            }
            const state = self.states.items[idx];
            try result.append(Complex(f64).init(state.amplitudes[0].re, state.amplitudes[0].im));
        }
        return result;
    }

    pub fn computeRelationalOutputRaw(
        self: *Self,
        input_indices: []const usize,
    ) !ArrayList(Complex(f64)) {
        var result = ArrayList(Complex(f64)).init(self.allocator);
        errdefer result.deinit();
        for (input_indices) |idx| {
            if (idx >= self.states.items.len) {
                return error.InvalidQubitIndex;
            }
            const state = self.states.items[idx];
            try result.append(Complex(f64).init(state.amplitudes[0].re, state.amplitudes[0].im));
        }
        return result;
    }

    pub fn getTotalProbability(self: *const Self) f64 {
        var total: f64 = 0.0;
        for (self.states.items) |state| {
            total += state.totalProbability();
        }
        return total;
    }

    pub fn getAverageEntanglement(self: *const Self) f64 {
        if (self.states.items.len == 0) return 0.0;
        var total: f64 = 0.0;
        for (self.states.items) |state| {
            total += state.entanglement_degree;
        }
        return total / @as(f64, @floatFromInt(self.states.items.len));
    }

    pub fn isCoherent(self: *const Self) bool {
        for (self.states.items) |state| {
            if (state.totalProbability() < self.coherence_threshold) {
                return false;
            }
        }
        return true;
    }

    pub fn cloneStates(self: *const Self, allocator: Allocator) !ArrayList(QuantumState) {
        var cloned = ArrayList(QuantumState).init(allocator);
        for (self.states.items) |state| {
            try cloned.append(state.clone());
        }
        return cloned;
    }

    pub fn applyControlledGate(
        self: *Self,
        control_idx: usize,
        target_idx: usize,
        gate: LogicGate,
        params: ?[]const f64,
    ) !void {
        if (control_idx >= self.states.items.len or target_idx >= self.states.items.len) {
            return error.InvalidQubitIndex;
        }
        if (control_idx == target_idx) {
            return error.DuplicateQubitIndex;
        }

        const control = self.states.items[control_idx];
        if (control.prob1() > 0.5) {
            switch (gate) {
                .HADAMARD => self.applyHadamard(target_idx),
                .PAULI_X => self.applyPauliX(target_idx),
                .PAULI_Y => self.applyPauliY(target_idx),
                .PAULI_Z => self.applyPauliZ(target_idx),
                .PHASE => {
                    const theta = if (params != null and params.?.len > 0) params.?[0] else DEFAULT_PHASE_ANGLE;
                    self.applyPhase(target_idx, theta);
                },
                .RELATIONAL_NOT => self.applyRelationalNot(target_idx),
                .FRACTAL_TRANSFORM => {
                    const depth: u32 = if (params != null and params.?.len > 0) blk: {
                        const val = params.?[0];
                        if (std.math.isNan(val) or std.math.isInf(val) or val < 0.0 or val > 100.0) {
                            break :blk 3;
                        }
                        break :blk @intFromFloat(val);
                    } else 3;
                    self.applyFractalTransform(target_idx, depth);
                },
                .CNOT, .TOFFOLI, .RELATIONAL_AND, .RELATIONAL_OR, .RELATIONAL_XOR => {
                    return error.InvalidGateForControlled;
                },
            }
        }

        var control_ptr = &self.states.items[control_idx];
        control_ptr.entanglement_degree = @min(1.0, control_ptr.entanglement_degree + 0.25);
        var target = &self.states.items[target_idx];
        target.entanglement_degree = @min(1.0, target.entanglement_degree + 0.25);

        var indices_buf: [2]usize = .{ control_idx, target_idx };
        const history_entry = try GateHistoryEntry.initEntry(self.allocator, gate, &indices_buf, params);
        try self.gate_history.append(history_entry);
    }

    pub fn computeInferenceOutput(
        self: *Self,
        input_indices: []const usize,
        gate_sequence: []const GateSequenceEntry,
    ) !ArrayList(Complex(f64)) {
        for (gate_sequence) |entry| {
            try self.applyGate(entry.gate, entry.indices, entry.params);
        }

        var result = ArrayList(Complex(f64)).init(self.allocator);
        errdefer result.deinit();
        for (input_indices) |idx| {
            if (idx >= self.states.items.len) {
                return error.InvalidQubitIndex;
            }
            const state = self.states.items[idx];
            try result.append(Complex(f64).init(state.amplitudes[0].re, state.amplitudes[0].im));
        }
        return result;
    }

    pub fn serialize(self: *const Self, allocator: Allocator) !ArrayList(u8) {
        var buffer = ArrayList(u8).init(allocator);
        errdefer buffer.deinit();

        var state_count_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &state_count_buf, @as(u64, @intCast(self.states.items.len)), .little);
        try buffer.appendSlice(&state_count_buf);

        for (self.states.items) |state| {
            var a0_re_buf: [8]u8 = undefined;
            var a0_im_buf: [8]u8 = undefined;
            var a1_re_buf: [8]u8 = undefined;
            var a1_im_buf: [8]u8 = undefined;
            var phase_buf: [8]u8 = undefined;
            var entangle_buf: [8]u8 = undefined;

            @memcpy(&a0_re_buf, std.mem.asBytes(&state.amplitudes[0].re));
            @memcpy(&a0_im_buf, std.mem.asBytes(&state.amplitudes[0].im));
            @memcpy(&a1_re_buf, std.mem.asBytes(&state.amplitudes[1].re));
            @memcpy(&a1_im_buf, std.mem.asBytes(&state.amplitudes[1].im));
            @memcpy(&phase_buf, std.mem.asBytes(&state.phase));
            @memcpy(&entangle_buf, std.mem.asBytes(&state.entanglement_degree));

            try buffer.appendSlice(&a0_re_buf);
            try buffer.appendSlice(&a0_im_buf);
            try buffer.appendSlice(&a1_re_buf);
            try buffer.appendSlice(&a1_im_buf);
            try buffer.appendSlice(&phase_buf);
            try buffer.appendSlice(&entangle_buf);
        }

        var history_count_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &history_count_buf, @as(u64, @intCast(self.gate_history.items.len)), .little);
        try buffer.appendSlice(&history_count_buf);

        for (self.gate_history.items) |entry| {
            try buffer.append(@intFromEnum(entry.gate));

            var idx_count_buf: [8]u8 = undefined;
            std.mem.writeInt(u64, &idx_count_buf, @as(u64, @intCast(entry.indices.items.len)), .little);
            try buffer.appendSlice(&idx_count_buf);

            for (entry.indices.items) |idx| {
                var idx_buf: [8]u8 = undefined;
                std.mem.writeInt(u64, &idx_buf, @as(u64, @intCast(idx)), .little);
                try buffer.appendSlice(&idx_buf);
            }

            const has_params: u8 = if (entry.params != null) 1 else 0;
            try buffer.append(has_params);

            if (entry.params) |p| {
                var param_count_buf: [8]u8 = undefined;
                std.mem.writeInt(u64, &param_count_buf, @as(u64, @intCast(p.items.len)), .little);
                try buffer.appendSlice(&param_count_buf);

                for (p.items) |param| {
                    var param_buf: [8]u8 = undefined;
                    @memcpy(&param_buf, std.mem.asBytes(&param));
                    try buffer.appendSlice(&param_buf);
                }
            }

            var ts_buf: [8]u8 = undefined;
            std.mem.writeInt(i64, &ts_buf, entry.timestamp, .little);
            try buffer.appendSlice(&ts_buf);
        }

        return buffer;
    }

    pub fn deserialize(allocator: Allocator, data: []const u8) !Self {
        if (data.len < 8) return error.InvalidData;

        var self_obj = Self.init(allocator);
        errdefer self_obj.deinit();

        var offset: usize = 0;

        const state_count = std.mem.readInt(u64, data[0..8], .little);
        offset = 8;

        const bytes_per_state: usize = 48;
        var i: u64 = 0;
        while (i < state_count) : (i += 1) {
            if (offset + bytes_per_state > data.len) return error.InvalidData;

            var a0_re_bytes: [8]u8 = undefined;
            var a0_im_bytes: [8]u8 = undefined;
            var a1_re_bytes: [8]u8 = undefined;
            var a1_im_bytes: [8]u8 = undefined;
            var phase_bytes: [8]u8 = undefined;
            var entangle_bytes: [8]u8 = undefined;

            @memcpy(&a0_re_bytes, data[offset .. offset + 8]);
            @memcpy(&a0_im_bytes, data[offset + 8 .. offset + 16]);
            @memcpy(&a1_re_bytes, data[offset + 16 .. offset + 24]);
            @memcpy(&a1_im_bytes, data[offset + 24 .. offset + 32]);
            @memcpy(&phase_bytes, data[offset + 32 .. offset + 40]);
            @memcpy(&entangle_bytes, data[offset + 40 .. offset + 48]);

            const a0_re = @as(f64, @bitCast(a0_re_bytes));
            const a0_im = @as(f64, @bitCast(a0_im_bytes));
            const a1_re = @as(f64, @bitCast(a1_re_bytes));
            const a1_im = @as(f64, @bitCast(a1_im_bytes));
            const phase_val = @as(f64, @bitCast(phase_bytes));
            const entangle_val = @as(f64, @bitCast(entangle_bytes));

            try self_obj.states.append(QuantumState.init(a0_re, a0_im, a1_re, a1_im, phase_val, entangle_val));
            offset += bytes_per_state;
        }

        if (offset + 8 <= data.len) {
            const history_count = std.mem.readInt(u64, data[offset..][0..8], .little);
            offset += 8;

            var h: u64 = 0;
            while (h < history_count) : (h += 1) {
                if (offset + 1 > data.len) return error.InvalidData;
                const gate_byte = data[offset];
                offset += 1;

                const gate: LogicGate = @enumFromInt(gate_byte);

                if (offset + 8 > data.len) return error.InvalidData;
                const idx_count = std.mem.readInt(u64, data[offset..][0..8], .little);
                offset += 8;

                var indices_list = ArrayList(usize).init(allocator);
                var idx_i: u64 = 0;
                while (idx_i < idx_count) : (idx_i += 1) {
                    if (offset + 8 > data.len) {
                        indices_list.deinit();
                        return error.InvalidData;
                    }
                    const idx_val = std.mem.readInt(u64, data[offset..][0..8], .little);
                    offset += 8;
                    try indices_list.append(@as(usize, @intCast(idx_val)));
                }

                if (offset + 1 > data.len) {
                    indices_list.deinit();
                    return error.InvalidData;
                }
                const has_params = data[offset];
                offset += 1;

                var params_list: ?ArrayList(f64) = null;
                if (has_params == 1) {
                    if (offset + 8 > data.len) {
                        indices_list.deinit();
                        return error.InvalidData;
                    }
                    const param_count = std.mem.readInt(u64, data[offset..][0..8], .little);
                    offset += 8;

                    params_list = ArrayList(f64).init(allocator);
                    var p_i: u64 = 0;
                    while (p_i < param_count) : (p_i += 1) {
                        if (offset + 8 > data.len) {
                            indices_list.deinit();
                            params_list.?.deinit();
                            return error.InvalidData;
                        }
                        var param_bytes: [8]u8 = undefined;
                        @memcpy(&param_bytes, data[offset .. offset + 8]);
                        const param_val = @as(f64, @bitCast(param_bytes));
                        offset += 8;
                        try params_list.?.append(param_val);
                    }
                }

                if (offset + 8 > data.len) {
                    indices_list.deinit();
                    if (params_list) |*pl| pl.deinit();
                    return error.InvalidData;
                }
                const ts = std.mem.readInt(i64, data[offset..][0..8], .little);
                offset += 8;

                try self_obj.gate_history.append(.{
                    .gate = gate,
                    .indices = indices_list,
                    .params = params_list,
                    .timestamp = ts,
                });
            }
        }

        return self_obj;
    }
};

pub const QuantumCircuit = struct {
    gates: ArrayList(GateSequenceEntry),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .gates = ArrayList(GateSequenceEntry).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.gates.items) |entry| {
            self.allocator.free(entry.indices);
            if (entry.params) |p| {
                self.allocator.free(p);
            }
        }
        self.gates.deinit();
    }

    pub fn addGate(self: *Self, gate: LogicGate, indices: []const usize, params: ?[]const f64) !void {
        const indices_copy = try self.allocator.dupe(usize, indices);
        errdefer self.allocator.free(indices_copy);

        var params_copy: ?[]const f64 = null;
        if (params) |p| {
            params_copy = try self.allocator.dupe(f64, p);
        }
        errdefer {
            if (params_copy) |pc| {
                self.allocator.free(pc);
            }
        }

        try self.gates.append(.{
            .gate = gate,
            .indices = indices_copy,
            .params = params_copy,
        });
    }

    pub fn execute(self: *const Self, logic: *RelationalQuantumLogic) !void {
        for (self.gates.items) |entry| {
            try logic.applyGate(entry.gate, entry.indices, entry.params);
        }
    }

    pub fn gateCount(self: *const Self) usize {
        return self.gates.items.len;
    }
};

test "quantum_state_normalize" {
    var state = QuantumState.init(3.0, 4.0, 0.0, 0.0, 0.0, 0.0);
    state.normalize();
    const expected_prob: f64 = 1.0;
    const actual_prob = state.totalProbability();
    try std.testing.expectApproxEqAbs(expected_prob, actual_prob, 0.0001);
}

test "quantum_state_probability" {
    const state = QuantumState.init(0.6, 0.0, 0.8, 0.0, 0.0, 0.0);
    const prob = state.totalProbability();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), prob, 0.0001);
}

test "relational_quantum_logic_init_deinit" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    const idx = try logic.initializeState(1.0, 0.0, 0.0, 0.0, 0.0);
    try std.testing.expectEqual(@as(usize, 0), idx);
    try std.testing.expectEqual(@as(usize, 1), logic.stateCount());
}

test "apply_hadamard_gate" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeBasisState(false);
    try logic.applyGate(.HADAMARD, &[_]usize{0}, null);

    const state = logic.getState(0).?;
    try std.testing.expect(state.totalProbability() > 0.0);
    try std.testing.expectApproxEqAbs(state.prob0(), 0.5, 0.0001);
    try std.testing.expectApproxEqAbs(state.prob1(), 0.5, 0.0001);
}

test "apply_cnot_gate" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeBasisState(true);
    _ = try logic.initializeBasisState(false);

    try logic.applyGate(.CNOT, &[_]usize{ 0, 1 }, null);

    const control = logic.getState(0).?;
    const target = logic.getState(1).?;
    try std.testing.expect(control.entanglement_degree > 0.0);
    try std.testing.expect(target.entanglement_degree > 0.0);
    try std.testing.expectApproxEqAbs(target.prob1(), 1.0, 0.0001);
}

test "measure_qubit" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeState(0.8, 0.0, 0.6, 0.0, 0.0);

    const result = logic.measure(0);
    try std.testing.expect(result.result == 0 or result.result == 1);
    try std.testing.expect(result.probability_zero >= 0.0 and result.probability_zero <= 1.0);
    try std.testing.expect(result.probability_one >= 0.0 and result.probability_one <= 1.0);
}

test "entangle_qubits" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeBasisState(false);
    _ = try logic.initializeBasisState(true);

    try logic.entangle(0, 1);

    const state1 = logic.getState(0).?;
    const state2 = logic.getState(1).?;
    try std.testing.expectApproxEqAbs(state1.amplitudes[0].re, state2.amplitudes[0].re, 0.0001);
    try std.testing.expectApproxEqAbs(state1.amplitudes[1].re, state2.amplitudes[1].re, 0.0001);
    try std.testing.expectApproxEqAbs(state1.entanglement_degree, 1.0, 0.0001);
}

test "relational_and_gate" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeState(0.8, 0.0, 0.6, 0.0, 0.0);
    _ = try logic.initializeState(0.6, 0.0, 0.8, 0.0, 0.0);

    try logic.applyGate(.RELATIONAL_AND, &[_]usize{ 0, 1 }, null);

    try std.testing.expectEqual(@as(usize, 3), logic.stateCount());
    const result_state = logic.getState(2).?;
    try std.testing.expect(result_state.totalProbability() > 0.0);
}

test "fractal_transform" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeState(0.707, 0.0, 0.707, 0.0, std.math.pi / 4.0);

    const params = [_]f64{3.0};
    try logic.applyGate(.FRACTAL_TRANSFORM, &[_]usize{0}, &params);

    const state = logic.getState(0).?;
    try std.testing.expect(state.isNormalized(0.01));
}

test "compute_relational_output" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeBasisState(false);
    _ = try logic.initializeBasisState(true);

    const gate_sequence = [_]GateSequenceEntry{
        .{ .gate = .HADAMARD, .indices = &[_]usize{0}, .params = null },
        .{ .gate = .CNOT, .indices = &[_]usize{ 0, 1 }, .params = null },
    };

    var result = try logic.computeRelationalOutput(&[_]usize{ 0, 1 }, &gate_sequence);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.items.len);
}

test "logic_gate_enum" {
    try std.testing.expectEqual(@as(usize, 1), LogicGate.HADAMARD.requiredQubits());
    try std.testing.expectEqual(@as(usize, 2), LogicGate.CNOT.requiredQubits());
    try std.testing.expectEqual(@as(usize, 3), LogicGate.TOFFOLI.requiredQubits());
    try std.testing.expect(LogicGate.HADAMARD.isSingleQubit());
    try std.testing.expect(!LogicGate.CNOT.isSingleQubit());
}

test "gate_history_tracking" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeBasisState(false);
    try logic.applyGate(.HADAMARD, &[_]usize{0}, null);
    try logic.applyGate(.PAULI_X, &[_]usize{0}, null);

    try std.testing.expectEqual(@as(usize, 2), logic.historyCount());
}

test "basis_state_zero" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeBasisState(false);
    const state = logic.getState(0).?;
    try std.testing.expectApproxEqAbs(state.prob0(), 1.0, 0.0001);
    try std.testing.expectApproxEqAbs(state.prob1(), 0.0, 0.0001);
}

test "basis_state_one" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeBasisState(true);
    const state = logic.getState(0).?;
    try std.testing.expectApproxEqAbs(state.prob0(), 0.0, 0.0001);
    try std.testing.expectApproxEqAbs(state.prob1(), 1.0, 0.0001);
}

test "pauli_x_flips_basis" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeBasisState(false);
    try logic.applyGate(.PAULI_X, &[_]usize{0}, null);

    const state = logic.getState(0).?;
    try std.testing.expectApproxEqAbs(state.prob0(), 0.0, 0.0001);
    try std.testing.expectApproxEqAbs(state.prob1(), 1.0, 0.0001);
}

test "pauli_z_on_basis_one" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeBasisState(true);
    try logic.applyGate(.PAULI_Z, &[_]usize{0}, null);

    const state = logic.getState(0).?;
    try std.testing.expectApproxEqAbs(state.amplitudes[0].re, 0.0, 0.0001);
    try std.testing.expectApproxEqAbs(state.amplitudes[1].re, -1.0, 0.0001);
}

test "measure_with_randomness" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeBasisState(false);
    try logic.applyGate(.HADAMARD, &[_]usize{0}, null);

    const result_zero = logic.measureWithRandomness(0, 0.1);
    try std.testing.expectEqual(@as(i32, 0), result_zero.result);
}

test "serialize_deserialize" {
    var logic = RelationalQuantumLogic.init(std.testing.allocator);
    defer logic.deinit();

    _ = try logic.initializeBasisState(false);
    _ = try logic.initializeBasisState(true);

    var serialized = try logic.serialize(std.testing.allocator);
    defer serialized.deinit();

    var deserialized = try RelationalQuantumLogic.deserialize(std.testing.allocator, serialized.items);
    defer deserialized.deinit();

    try std.testing.expectEqual(logic.stateCount(), deserialized.stateCount());
}

test "quantum_state_amplitude_fields" {
    const state = QuantumState.init(0.6, 0.8, 0.0, 0.0, 0.5, 0.0);
    try std.testing.expectApproxEqAbs(state.amplitudes[0].re, 0.6, 0.0001);
    try std.testing.expectApproxEqAbs(state.amplitudes[0].im, 0.8, 0.0001);
}

test "quantum_state_normalize_syncs_amplitude_fields" {
    var state = QuantumState.init(3.0, 4.0, 0.0, 0.0, 0.0, 0.0);
    state.normalize();
    try std.testing.expectApproxEqAbs(state.amplitudes[0].re, state.amplitudes[0].re, 0.0001);
    try std.testing.expectApproxEqAbs(state.amplitudes[0].im, state.amplitudes[0].im, 0.0001);
}

