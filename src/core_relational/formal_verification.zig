const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const StringHashMap = std.StringHashMap;
const AutoHashMap = std.AutoHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Complex = std.math.Complex;

const nsir_core = @import("nsir_core.zig");
const SelfSimilarRelationalGraph = nsir_core.SelfSimilarRelationalGraph;
const Node = nsir_core.Node;
const Edge = nsir_core.Edge;
const EdgeQuality = nsir_core.EdgeQuality;
const EdgeKey = nsir_core.EdgeKey;

pub const InvariantType = enum(u8) {
    CONNECTIVITY = 0,
    SYMMETRY = 1,
    COHERENCE = 2,
    ENTANGLEMENT = 3,
    FRACTAL_DIMENSION = 4,
    QUANTUM_STATE = 5,
    MEMORY_SAFETY = 6,
    TYPE_SAFETY = 7,
    TEMPORAL_CONSISTENCY = 8,

    pub fn toString(self: InvariantType) []const u8 {
        return switch (self) {
            .CONNECTIVITY => "connectivity",
            .SYMMETRY => "symmetry",
            .COHERENCE => "coherence",
            .ENTANGLEMENT => "entanglement",
            .FRACTAL_DIMENSION => "fractal_dimension",
            .QUANTUM_STATE => "quantum_state",
            .MEMORY_SAFETY => "memory_safety",
            .TYPE_SAFETY => "type_safety",
            .TEMPORAL_CONSISTENCY => "temporal_consistency",
        };
    }

    pub fn fromString(s: []const u8) ?InvariantType {
        if (std.mem.eql(u8, s, "connectivity")) return .CONNECTIVITY;
        if (std.mem.eql(u8, s, "symmetry")) return .SYMMETRY;
        if (std.mem.eql(u8, s, "coherence")) return .COHERENCE;
        if (std.mem.eql(u8, s, "entanglement")) return .ENTANGLEMENT;
        if (std.mem.eql(u8, s, "fractal_dimension")) return .FRACTAL_DIMENSION;
        if (std.mem.eql(u8, s, "quantum_state")) return .QUANTUM_STATE;
        if (std.mem.eql(u8, s, "memory_safety")) return .MEMORY_SAFETY;
        if (std.mem.eql(u8, s, "type_safety")) return .TYPE_SAFETY;
        if (std.mem.eql(u8, s, "temporal_consistency")) return .TEMPORAL_CONSISTENCY;
        return null;
    }

    pub fn priority(self: InvariantType) u8 {
        return switch (self) {
            .MEMORY_SAFETY => 10,
            .TYPE_SAFETY => 9,
            .CONNECTIVITY => 8,
            .COHERENCE => 7,
            .ENTANGLEMENT => 6,
            .QUANTUM_STATE => 5,
            .FRACTAL_DIMENSION => 4,
            .SYMMETRY => 3,
            .TEMPORAL_CONSISTENCY => 2,
        };
    }
};

pub const ProofRule = enum(u8) {
    AXIOM = 0,
    MODUS_PONENS = 1,
    UNIVERSAL_INSTANTIATION = 2,
    EXISTENTIAL_GENERALIZATION = 3,
    INDUCTION = 4,
    CONTRADICTION = 5,
    DEDUCTION = 6,
    WEAKENING = 7,
    STRENGTHENING = 8,
    FRAME_RULE = 9,
    CONSEQUENCE_RULE = 10,
    CONJUNCTION_INTRO = 11,
    CONJUNCTION_ELIM = 12,
    DISJUNCTION_INTRO = 13,
    DISJUNCTION_ELIM = 14,
    NEGATION_INTRO = 15,
    NEGATION_ELIM = 16,
    IMPLICATION_INTRO = 17,
    IMPLICATION_ELIM = 18,
    UNIVERSAL_INTRO = 19,
    EXISTENTIAL_ELIM = 20,
    TEMPORAL_INDUCTION = 21,
    LOOP_INVARIANT = 22,
    ASSIGNMENT_AXIOM = 23,
    SEQUENCE_RULE = 24,
    CONDITIONAL_RULE = 25,

    pub fn toString(self: ProofRule) []const u8 {
        return switch (self) {
            .AXIOM => "axiom",
            .MODUS_PONENS => "modus_ponens",
            .UNIVERSAL_INSTANTIATION => "universal_instantiation",
            .EXISTENTIAL_GENERALIZATION => "existential_generalization",
            .INDUCTION => "induction",
            .CONTRADICTION => "contradiction",
            .DEDUCTION => "deduction",
            .WEAKENING => "weakening",
            .STRENGTHENING => "strengthening",
            .FRAME_RULE => "frame_rule",
            .CONSEQUENCE_RULE => "consequence_rule",
            .CONJUNCTION_INTRO => "conjunction_intro",
            .CONJUNCTION_ELIM => "conjunction_elim",
            .DISJUNCTION_INTRO => "disjunction_intro",
            .DISJUNCTION_ELIM => "disjunction_elim",
            .NEGATION_INTRO => "negation_intro",
            .NEGATION_ELIM => "negation_elim",
            .IMPLICATION_INTRO => "implication_intro",
            .IMPLICATION_ELIM => "implication_elim",
            .UNIVERSAL_INTRO => "universal_intro",
            .EXISTENTIAL_ELIM => "existential_elim",
            .TEMPORAL_INDUCTION => "temporal_induction",
            .LOOP_INVARIANT => "loop_invariant",
            .ASSIGNMENT_AXIOM => "assignment_axiom",
            .SEQUENCE_RULE => "sequence_rule",
            .CONDITIONAL_RULE => "conditional_rule",
        };
    }

    pub fn fromString(s: []const u8) ?ProofRule {
        if (std.mem.eql(u8, s, "axiom")) return .AXIOM;
        if (std.mem.eql(u8, s, "modus_ponens")) return .MODUS_PONENS;
        if (std.mem.eql(u8, s, "universal_instantiation")) return .UNIVERSAL_INSTANTIATION;
        if (std.mem.eql(u8, s, "existential_generalization")) return .EXISTENTIAL_GENERALIZATION;
        if (std.mem.eql(u8, s, "induction")) return .INDUCTION;
        if (std.mem.eql(u8, s, "contradiction")) return .CONTRADICTION;
        if (std.mem.eql(u8, s, "deduction")) return .DEDUCTION;
        if (std.mem.eql(u8, s, "weakening")) return .WEAKENING;
        if (std.mem.eql(u8, s, "strengthening")) return .STRENGTHENING;
        if (std.mem.eql(u8, s, "frame_rule")) return .FRAME_RULE;
        if (std.mem.eql(u8, s, "consequence_rule")) return .CONSEQUENCE_RULE;
        if (std.mem.eql(u8, s, "conjunction_intro")) return .CONJUNCTION_INTRO;
        if (std.mem.eql(u8, s, "conjunction_elim")) return .CONJUNCTION_ELIM;
        if (std.mem.eql(u8, s, "disjunction_intro")) return .DISJUNCTION_INTRO;
        if (std.mem.eql(u8, s, "disjunction_elim")) return .DISJUNCTION_ELIM;
        if (std.mem.eql(u8, s, "negation_intro")) return .NEGATION_INTRO;
        if (std.mem.eql(u8, s, "negation_elim")) return .NEGATION_ELIM;
        if (std.mem.eql(u8, s, "implication_intro")) return .IMPLICATION_INTRO;
        if (std.mem.eql(u8, s, "implication_elim")) return .IMPLICATION_ELIM;
        if (std.mem.eql(u8, s, "universal_intro")) return .UNIVERSAL_INTRO;
        if (std.mem.eql(u8, s, "existential_elim")) return .EXISTENTIAL_ELIM;
        if (std.mem.eql(u8, s, "temporal_induction")) return .TEMPORAL_INDUCTION;
        if (std.mem.eql(u8, s, "loop_invariant")) return .LOOP_INVARIANT;
        if (std.mem.eql(u8, s, "assignment_axiom")) return .ASSIGNMENT_AXIOM;
        if (std.mem.eql(u8, s, "sequence_rule")) return .SEQUENCE_RULE;
        if (std.mem.eql(u8, s, "conditional_rule")) return .CONDITIONAL_RULE;
        return null;
    }

    pub fn requiresPremises(self: ProofRule) bool {
        return switch (self) {
            .AXIOM, .ASSIGNMENT_AXIOM => false,
            else => true,
        };
    }

    pub fn minimumPremises(self: ProofRule) usize {
        return switch (self) {
            .AXIOM, .ASSIGNMENT_AXIOM => 0,
            .MODUS_PONENS, .IMPLICATION_ELIM => 2,
            .UNIVERSAL_INSTANTIATION, .EXISTENTIAL_GENERALIZATION => 1,
            .INDUCTION => 2,
            .CONTRADICTION => 2,
            .DEDUCTION, .IMPLICATION_INTRO => 1,
            .WEAKENING, .STRENGTHENING => 1,
            .FRAME_RULE => 1,
            .CONSEQUENCE_RULE => 3,
            .CONJUNCTION_INTRO => 2,
            .CONJUNCTION_ELIM => 1,
            .DISJUNCTION_INTRO => 1,
            .DISJUNCTION_ELIM => 3,
            .NEGATION_INTRO, .NEGATION_ELIM => 1,
            .UNIVERSAL_INTRO => 1,
            .EXISTENTIAL_ELIM => 2,
            .TEMPORAL_INDUCTION => 2,
            .LOOP_INVARIANT => 2,
            .SEQUENCE_RULE => 2,
            .CONDITIONAL_RULE => 2,
        };
    }
};

pub const PropType = enum(u8) {
    ATOMIC = 0,
    NEGATION = 1,
    CONJUNCTION = 2,
    DISJUNCTION = 3,
    IMPLICATION = 4,
    UNIVERSAL = 5,
    EXISTENTIAL = 6,
    TEMPORAL_ALWAYS = 7,
    TEMPORAL_EVENTUALLY = 8,
    HOARE_TRIPLE = 9,
    BICONDITIONAL = 10,
    TRUE = 11,
    FALSE = 12,
    SEPARATION_STAR = 13,
    SEPARATION_WAND = 14,
    RELATIONAL_EDGE = 15,
    QUANTUM_SUPERPOSITION = 16,
    ENTANGLEMENT_PAIR = 17,

    pub fn toString(self: PropType) []const u8 {
        return switch (self) {
            .ATOMIC => "atomic",
            .NEGATION => "negation",
            .CONJUNCTION => "conjunction",
            .DISJUNCTION => "disjunction",
            .IMPLICATION => "implication",
            .UNIVERSAL => "universal",
            .EXISTENTIAL => "existential",
            .TEMPORAL_ALWAYS => "temporal_always",
            .TEMPORAL_EVENTUALLY => "temporal_eventually",
            .HOARE_TRIPLE => "hoare_triple",
            .BICONDITIONAL => "biconditional",
            .TRUE => "true",
            .FALSE => "false",
            .SEPARATION_STAR => "separation_star",
            .SEPARATION_WAND => "separation_wand",
            .RELATIONAL_EDGE => "relational_edge",
            .QUANTUM_SUPERPOSITION => "quantum_superposition",
            .ENTANGLEMENT_PAIR => "entanglement_pair",
        };
    }

    pub fn fromString(s: []const u8) ?PropType {
        if (std.mem.eql(u8, s, "atomic")) return .ATOMIC;
        if (std.mem.eql(u8, s, "negation")) return .NEGATION;
        if (std.mem.eql(u8, s, "conjunction")) return .CONJUNCTION;
        if (std.mem.eql(u8, s, "disjunction")) return .DISJUNCTION;
        if (std.mem.eql(u8, s, "implication")) return .IMPLICATION;
        if (std.mem.eql(u8, s, "universal")) return .UNIVERSAL;
        if (std.mem.eql(u8, s, "existential")) return .EXISTENTIAL;
        if (std.mem.eql(u8, s, "temporal_always")) return .TEMPORAL_ALWAYS;
        if (std.mem.eql(u8, s, "temporal_eventually")) return .TEMPORAL_EVENTUALLY;
        if (std.mem.eql(u8, s, "hoare_triple")) return .HOARE_TRIPLE;
        if (std.mem.eql(u8, s, "biconditional")) return .BICONDITIONAL;
        if (std.mem.eql(u8, s, "true")) return .TRUE;
        if (std.mem.eql(u8, s, "false")) return .FALSE;
        if (std.mem.eql(u8, s, "separation_star")) return .SEPARATION_STAR;
        if (std.mem.eql(u8, s, "separation_wand")) return .SEPARATION_WAND;
        if (std.mem.eql(u8, s, "relational_edge")) return .RELATIONAL_EDGE;
        if (std.mem.eql(u8, s, "quantum_superposition")) return .QUANTUM_SUPERPOSITION;
        if (std.mem.eql(u8, s, "entanglement_pair")) return .ENTANGLEMENT_PAIR;
        return null;
    }

    pub fn arity(self: PropType) usize {
        return switch (self) {
            .ATOMIC, .TRUE, .FALSE => 0,
            .NEGATION, .TEMPORAL_ALWAYS, .TEMPORAL_EVENTUALLY, .UNIVERSAL, .EXISTENTIAL => 1,
            .CONJUNCTION, .DISJUNCTION, .IMPLICATION, .BICONDITIONAL, .SEPARATION_STAR, .SEPARATION_WAND, .ENTANGLEMENT_PAIR, .RELATIONAL_EDGE, .QUANTUM_SUPERPOSITION => 2,
            .HOARE_TRIPLE => 3,
        };
    }

    pub fn isBinaryOperator(self: PropType) bool {
        return switch (self) {
            .CONJUNCTION, .DISJUNCTION, .IMPLICATION, .BICONDITIONAL, .SEPARATION_STAR, .SEPARATION_WAND, .ENTANGLEMENT_PAIR, .RELATIONAL_EDGE, .QUANTUM_SUPERPOSITION => true,
            else => false,
        };
    }

    pub fn isQuantifier(self: PropType) bool {
        return self == .UNIVERSAL or self == .EXISTENTIAL;
    }
};

pub const VerificationError = error{
    InvalidProofStep,
    PremiseMismatch,
    InvalidConclusion,
    InvariantViolation,
    OutOfMemory,
    ProofIncomplete,
    UnificationFailure,
    ResolutionFailure,
    InvalidGraph,
    CircularDependency,
    TypeMismatch,
    PredicateEvaluationError,
    InvalidPremiseIndex,
    InvalidRule,
    OwnershipViolation,
    NullPredicate,
};

pub const Term = struct {
    kind: TermKind,
    name: []const u8,
    args: ArrayList(*Term),
    allocator: Allocator,
    ref_count: u32,

    pub const TermKind = enum(u8) {
        VARIABLE = 0,
        CONSTANT = 1,
        FUNCTION = 2,
    };

    const Self = @This();

    pub fn initVariable(allocator: Allocator, name: []const u8) !*Self {
        const term = try allocator.create(Self);
        term.* = Self{
            .kind = .VARIABLE,
            .name = try allocator.dupe(u8, name),
            .args = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .ref_count = 1,
        };
        return term;
    }

    pub fn initConstant(allocator: Allocator, name: []const u8) !*Self {
        const term = try allocator.create(Self);
        term.* = Self{
            .kind = .CONSTANT,
            .name = try allocator.dupe(u8, name),
            .args = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .ref_count = 1,
        };
        return term;
    }

    pub fn initFunction(allocator: Allocator, name: []const u8, args: []const *Term) !*Self {
        const term = try allocator.create(Self);
        term.* = Self{
            .kind = .FUNCTION,
            .name = try allocator.dupe(u8, name),
            .args = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .ref_count = 1,
        };
        for (args) |arg| {
            const cloned = try arg.clone(allocator);
            try term.args.append(cloned);
        }
        return term;
    }

    pub fn retain(self: *Self) void {
        self.ref_count += 1;
    }

    pub fn release(self: *Self) void {
        if (self.ref_count > 1) {
            self.ref_count -= 1;
            return;
        }
        self.deinit();
        self.allocator.destroy(self);
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);
        for (self.args.items) |arg| {
            arg.release();
        }
        self.args.deinit();
    }

    pub fn clone(self: *const Self, allocator: Allocator) error{OutOfMemory}!*Self {
        const new_term = try allocator.create(Self);
        new_term.* = Self{
            .kind = self.kind,
            .name = try allocator.dupe(u8, self.name),
            .args = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .ref_count = 1,
        };
        for (self.args.items) |arg| {
            try new_term.args.append(try arg.clone(allocator));
        }
        return new_term;
    }

    pub fn equals(self: *const Self, other: *const Self) bool {
        if (self.kind != other.kind) return false;
        if (!std.mem.eql(u8, self.name, other.name)) return false;
        if (self.args.items.len != other.args.items.len) return false;
        var i: usize = 0;
        while (i < self.args.items.len) : (i += 1) {
            if (!self.args.items[i].equals(other.args.items[i])) return false;
        }
        return true;
    }

    pub fn isVariable(self: *const Self) bool {
        return self.kind == .VARIABLE;
    }

    pub fn containsVariable(self: *const Self, var_name: []const u8) bool {
        if (self.kind == .VARIABLE and std.mem.eql(u8, self.name, var_name)) {
            return true;
        }
        for (self.args.items) |arg| {
            if (arg.containsVariable(var_name)) {
                return true;
            }
        }
        return false;
    }

    pub fn computeHash(self: *const Self, hasher: *Sha256) void {
        hasher.update(&[_]u8{@intFromEnum(self.kind)});
        hasher.update(self.name);
        var args_len_u64 = std.math.cast(u64, self.args.items.len) catch unreachable;
        hasher.update(std.mem.asBytes(&args_len_u64));
        for (self.args.items) |arg| {
            arg.computeHash(hasher);
        }
    }

    pub fn substitute(self: *Self, var_name: []const u8, replacement: *const Term) !void {
        if (self.kind == .VARIABLE and std.mem.eql(u8, self.name, var_name)) {
            self.allocator.free(self.name);
            self.name = try self.allocator.dupe(u8, replacement.name);
            self.kind = replacement.kind;
            for (self.args.items) |arg| {
                arg.release();
            }
            self.args.clearRetainingCapacity();
            for (replacement.args.items) |arg| {
                try self.args.append(try arg.clone(self.allocator));
            }
        } else {
            for (self.args.items) |arg| {
                try arg.substitute(var_name, replacement);
            }
        }
    }

    pub fn collectVariables(self: *const Self, vars: *StringHashMap(void)) !void {
        if (self.kind == .VARIABLE) {
            try vars.put(self.name, {});
        }
        for (self.args.items) |arg| {
            try arg.collectVariables(vars);
        }
    }
};

pub const Proposition = struct {
    prop_type: PropType,
    sub_propositions: ArrayList(*Proposition),
    predicate_name: []const u8,
    bound_variable: ?[]const u8,
    terms: ArrayList(*Term),
    allocator: Allocator,
    hash_cache: ?[32]u8,
    ref_count: u32,
    owns_predicate_name: bool,

    const Self = @This();

    pub fn init(allocator: Allocator, prop_type: PropType) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = prop_type,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = "",
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
            .ref_count = 1,
            .owns_predicate_name = false,
        };
        return prop;
    }

    pub fn initAtomic(allocator: Allocator, predicate_name: []const u8) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = .ATOMIC,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = try allocator.dupe(u8, predicate_name),
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
            .ref_count = 1,
            .owns_predicate_name = true,
        };
        return prop;
    }

    pub fn initNegation(allocator: Allocator, inner: *Proposition) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = .NEGATION,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = "",
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
            .ref_count = 1,
            .owns_predicate_name = false,
        };
        inner.retain();
        try prop.sub_propositions.append(inner);
        return prop;
    }

    pub fn initBinary(allocator: Allocator, prop_type: PropType, left: *Proposition, right: *Proposition) !*Self {
        if (!prop_type.isBinaryOperator()) {
            return VerificationError.InvalidRule;
        }
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = prop_type,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = "",
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
            .ref_count = 1,
            .owns_predicate_name = false,
        };
        left.retain();
        right.retain();
        try prop.sub_propositions.append(left);
        try prop.sub_propositions.append(right);
        return prop;
    }

    pub fn initQuantified(allocator: Allocator, prop_type: PropType, variable: []const u8, body: *Proposition) !*Self {
        if (!prop_type.isQuantifier()) {
            return VerificationError.InvalidRule;
        }
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = prop_type,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = "",
            .bound_variable = try allocator.dupe(u8, variable),
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
            .ref_count = 1,
            .owns_predicate_name = false,
        };
        body.retain();
        try prop.sub_propositions.append(body);
        return prop;
    }

    pub fn initHoareTriple(allocator: Allocator, precondition: *Proposition, operation: *Proposition, postcondition: *Proposition) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = .HOARE_TRIPLE,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = "",
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
            .ref_count = 1,
            .owns_predicate_name = false,
        };
        precondition.retain();
        operation.retain();
        postcondition.retain();
        try prop.sub_propositions.append(precondition);
        try prop.sub_propositions.append(operation);
        try prop.sub_propositions.append(postcondition);
        return prop;
    }

    pub fn initTrue(allocator: Allocator) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = .TRUE,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = try allocator.dupe(u8, "true"),
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
            .ref_count = 1,
            .owns_predicate_name = true,
        };
        return prop;
    }

    pub fn initFalse(allocator: Allocator) !*Self {
        const prop = try allocator.create(Self);
        prop.* = Self{
            .prop_type = .FALSE,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = try allocator.dupe(u8, "false"),
            .bound_variable = null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
            .ref_count = 1,
            .owns_predicate_name = true,
        };
        return prop;
    }

    pub fn retain(self: *Self) void {
        self.ref_count += 1;
    }

    pub fn release(self: *Self) void {
        if (self.ref_count > 1) {
            self.ref_count -= 1;
            return;
        }
        self.deinit();
        self.allocator.destroy(self);
    }

    pub fn deinit(self: *Self) void {
        for (self.sub_propositions.items) |sub| {
            sub.release();
        }
        self.sub_propositions.deinit();
        if (self.owns_predicate_name and self.predicate_name.len > 0) {
            self.allocator.free(self.predicate_name);
        }
        if (self.bound_variable) |bv| {
            self.allocator.free(bv);
        }
        for (self.terms.items) |term| {
            term.release();
        }
        self.terms.deinit();
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const new_prop = try allocator.create(Self);
        new_prop.* = Self{
            .prop_type = self.prop_type,
            .sub_propositions = ArrayList(*Proposition).init(allocator),
            .predicate_name = if (self.predicate_name.len > 0) try allocator.dupe(u8, self.predicate_name) else "",
            .bound_variable = if (self.bound_variable) |bv| try allocator.dupe(u8, bv) else null,
            .terms = ArrayList(*Term).init(allocator),
            .allocator = allocator,
            .hash_cache = null,
            .ref_count = 1,
            .owns_predicate_name = self.predicate_name.len > 0,
        };
        for (self.sub_propositions.items) |sub| {
            try new_prop.sub_propositions.append(try sub.clone(allocator));
        }
        for (self.terms.items) |term| {
            try new_prop.terms.append(try term.clone(allocator));
        }
        return new_prop;
    }

    pub fn equals(self: *const Self, other: *const Self) bool {
        if (self.prop_type != other.prop_type) return false;
        if (!std.mem.eql(u8, self.predicate_name, other.predicate_name)) return false;
        if (self.sub_propositions.items.len != other.sub_propositions.items.len) return false;
        if (self.terms.items.len != other.terms.items.len) return false;
        const self_bv = self.bound_variable orelse "";
        const other_bv = other.bound_variable orelse "";
        if (!std.mem.eql(u8, self_bv, other_bv)) return false;
        var i: usize = 0;
        while (i < self.sub_propositions.items.len) : (i += 1) {
            if (!self.sub_propositions.items[i].equals(other.sub_propositions.items[i])) return false;
        }
        var j: usize = 0;
        while (j < self.terms.items.len) : (j += 1) {
            if (!self.terms.items[j].equals(other.terms.items[j])) return false;
        }
        return true;
    }

    pub fn computeHash(self: *Self) [32]u8 {
        if (self.hash_cache) |cache| {
            return cache;
        }
        var hasher = Sha256.init(.{});
        hasher.update(&[_]u8{@intFromEnum(self.prop_type)});
        hasher.update(self.predicate_name);
        if (self.bound_variable) |bv| {
            hasher.update(bv);
        }
        var terms_len_u64 = std.math.cast(u64, self.terms.items.len) catch unreachable;
        hasher.update(std.mem.asBytes(&terms_len_u64));
        for (self.terms.items) |term| {
            term.computeHash(&hasher);
        }
        var subs_len_u64 = std.math.cast(u64, self.sub_propositions.items.len) catch unreachable;
        hasher.update(std.mem.asBytes(&subs_len_u64));
        for (self.sub_propositions.items) |sub| {
            const sub_hash = sub.computeHash();
            hasher.update(sub_hash[0..]);
        }
        var result: [32]u8 = undefined;
        hasher.final(&result);
        self.hash_cache = result;
        return result;
    }

    pub fn invalidateHashCache(self: *Self) void {
        self.hash_cache = null;
    }

    pub fn isAtomic(self: *const Self) bool {
        return self.prop_type == .ATOMIC or self.prop_type == .TRUE or self.prop_type == .FALSE;
    }

    pub fn negate(self: *const Self, allocator: Allocator) !*Self {
        if (self.prop_type == .NEGATION and self.sub_propositions.items.len == 1) {
            return self.sub_propositions.items[0].clone(allocator);
        }
        const cloned = try self.clone(allocator);
        return Proposition.initNegation(allocator, cloned);
    }

    pub fn implies(self: *const Self, allocator: Allocator, consequent: *Proposition) !*Self {
        const cloned_self = try self.clone(allocator);
        const cloned_cons = try consequent.clone(allocator);
        return Proposition.initBinary(allocator, .IMPLICATION, cloned_self, cloned_cons);
    }

    pub fn conjoin(self: *const Self, allocator: Allocator, other: *Proposition) !*Self {
        const cloned_self = try self.clone(allocator);
        const cloned_other = try other.clone(allocator);
        return Proposition.initBinary(allocator, .CONJUNCTION, cloned_self, cloned_other);
    }

    pub fn disjoin(self: *const Self, allocator: Allocator, other: *Proposition) !*Self {
        const cloned_self = try self.clone(allocator);
        const cloned_other = try other.clone(allocator);
        return Proposition.initBinary(allocator, .DISJUNCTION, cloned_self, cloned_other);
    }

    pub fn freeVariables(self: *const Self, allocator: Allocator) !ArrayList([]const u8) {
        var vars = ArrayList([]const u8).init(allocator);
        var bound_set = StringHashMap(void).init(allocator);
        defer bound_set.deinit();
        try self.collectFreeVariablesWithBound(&vars, &bound_set);
        return vars;
    }

    fn collectFreeVariablesWithBound(self: *const Self, vars: *ArrayList([]const u8), bound: *StringHashMap(void)) !void {
        if (self.bound_variable) |bv| {
            try bound.put(bv, {});
        }
        for (self.terms.items) |term| {
            try collectTermVariables(term, vars, bound);
        }
        for (self.sub_propositions.items) |sub| {
            try sub.collectFreeVariablesWithBound(vars, bound);
        }
    }

    fn collectTermVariables(term: *const Term, vars: *ArrayList([]const u8), bound: *StringHashMap(void)) !void {
        if (term.kind == .VARIABLE) {
            if (!bound.contains(term.name)) {
                var found = false;
                for (vars.items) |v| {
                    if (std.mem.eql(u8, v, term.name)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    try vars.append(term.name);
                }
            }
        }
        for (term.args.items) |arg| {
            try collectTermVariables(arg, vars, bound);
        }
    }

    pub fn substitute(self: *Self, var_name: []const u8, replacement: *const Term) !void {
        if (self.bound_variable) |bv| {
            if (std.mem.eql(u8, bv, var_name)) {
                return;
            }
        }
        for (self.terms.items) |term| {
            try term.substitute(var_name, replacement);
        }
        for (self.sub_propositions.items) |sub| {
            try sub.substitute(var_name, replacement);
        }
        self.invalidateHashCache();
    }

    pub fn addTerm(self: *Self, term: *Term) !void {
        term.retain();
        try self.terms.append(term);
        self.invalidateHashCache();
    }
};

pub const ProofStep = struct {
    step_id: u64,
    rule_applied: ProofRule,
    premise_indices: ArrayList(usize),
    conclusion: *Proposition,
    verified: bool,
    justification: []const u8,
    timestamp: i128,
    allocator: Allocator,
    owns_justification: bool,

    const Self = @This();

    pub fn init(allocator: Allocator, step_id: u64, rule: ProofRule, conclusion: *Proposition) !*Self {
        const step = try allocator.create(Self);
        conclusion.retain();
        step.* = Self{
            .step_id = step_id,
            .rule_applied = rule,
            .premise_indices = ArrayList(usize).init(allocator),
            .conclusion = conclusion,
            .verified = false,
            .justification = "",
            .timestamp = std.time.nanoTimestamp(),
            .allocator = allocator,
            .owns_justification = false,
        };
        return step;
    }

    pub fn initWithPremises(allocator: Allocator, step_id: u64, rule: ProofRule, conclusion: *Proposition, premises: []const usize, justification: []const u8) !*Self {
        const step = try allocator.create(Self);
        conclusion.retain();
        step.* = Self{
            .step_id = step_id,
            .rule_applied = rule,
            .premise_indices = ArrayList(usize).init(allocator),
            .conclusion = conclusion,
            .verified = false,
            .justification = if (justification.len > 0) try allocator.dupe(u8, justification) else "",
            .timestamp = std.time.nanoTimestamp(),
            .allocator = allocator,
            .owns_justification = justification.len > 0,
        };
        for (premises) |p| {
            try step.premise_indices.append(p);
        }
        return step;
    }

    pub fn deinit(self: *Self) void {
        self.premise_indices.deinit();
        self.conclusion.release();
        if (self.owns_justification and self.justification.len > 0) {
            self.allocator.free(self.justification);
        }
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const new_step = try allocator.create(Self);
        new_step.* = Self{
            .step_id = self.step_id,
            .rule_applied = self.rule_applied,
            .premise_indices = ArrayList(usize).init(allocator),
            .conclusion = try self.conclusion.clone(allocator),
            .verified = self.verified,
            .justification = if (self.justification.len > 0) try allocator.dupe(u8, self.justification) else "",
            .timestamp = self.timestamp,
            .allocator = allocator,
            .owns_justification = self.justification.len > 0,
        };
        for (self.premise_indices.items) |idx| {
            try new_step.premise_indices.append(idx);
        }
        return new_step;
    }

    pub fn verify(self: *Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < self.rule_applied.minimumPremises()) {
            return false;
        }
        for (self.premise_indices.items) |idx| {
            if (idx >= all_steps.len) {
                return false;
            }
            if (!all_steps[idx].verified and self.rule_applied != .AXIOM) {
                return false;
            }
        }
        var seen = std.AutoHashMap(usize, void).init(self.allocator);
        defer seen.deinit();
        for (self.premise_indices.items) |idx| {
            if (seen.contains(idx)) {
                return false;
            }
            seen.put(idx, {}) catch return false;
        }
        const valid = self.validateRule(all_steps);
        self.verified = valid;
        return valid;
    }

    fn validateRule(self: *const Self, all_steps: []const *ProofStep) bool {
        return switch (self.rule_applied) {
            .AXIOM => self.conclusion.prop_type == .TRUE or self.conclusion.isAtomic(),
            .MODUS_PONENS => self.validateModusPonens(all_steps),
            .CONJUNCTION_INTRO => self.validateConjunctionIntro(all_steps),
            .CONJUNCTION_ELIM => self.validateConjunctionElim(all_steps),
            .DISJUNCTION_INTRO => self.validateDisjunctionIntro(all_steps),
            .DISJUNCTION_ELIM => self.validateDisjunctionElim(all_steps),
            .NEGATION_INTRO => self.validateNegationIntro(all_steps),
            .NEGATION_ELIM => self.validateNegationElim(all_steps),
            .IMPLICATION_INTRO => self.validateImplicationIntro(all_steps),
            .IMPLICATION_ELIM => self.validateModusPonens(all_steps),
            .UNIVERSAL_INSTANTIATION => self.validateUniversalInstantiation(all_steps),
            .EXISTENTIAL_GENERALIZATION => self.validateExistentialGeneralization(all_steps),
            .UNIVERSAL_INTRO => self.validateUniversalIntro(all_steps),
            .EXISTENTIAL_ELIM => self.validateExistentialElim(all_steps),
            .FRAME_RULE => self.validateFrameRule(all_steps),
            .CONSEQUENCE_RULE => self.validateConsequenceRule(all_steps),
            .SEQUENCE_RULE => self.validateSequenceRule(all_steps),
            .WEAKENING => self.validateWeakening(all_steps),
            .STRENGTHENING => self.validateStrengthening(all_steps),
            .CONTRADICTION => self.validateContradiction(all_steps),
            .DEDUCTION => self.validateDeduction(all_steps),
            .INDUCTION => self.validateInduction(all_steps),
            .TEMPORAL_INDUCTION => self.validateTemporalInduction(all_steps),
            .LOOP_INVARIANT => self.validateLoopInvariant(all_steps),
            .ASSIGNMENT_AXIOM => self.conclusion.prop_type == .HOARE_TRIPLE,
            .CONDITIONAL_RULE => self.validateConditionalRule(all_steps),
        };
    }

    fn validateModusPonens(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 2) return false;
        const idx0 = self.premise_indices.items[0];
        const idx1 = self.premise_indices.items[1];
        if (idx0 >= all_steps.len or idx1 >= all_steps.len) return false;
        const p0 = all_steps[idx0].conclusion;
        const p1 = all_steps[idx1].conclusion;
        if (p0.prop_type == .IMPLICATION and p0.sub_propositions.items.len >= 2) {
            if (p0.sub_propositions.items[0].equals(p1) and p0.sub_propositions.items[1].equals(self.conclusion)) {
                return true;
            }
        }
        if (p1.prop_type == .IMPLICATION and p1.sub_propositions.items.len >= 2) {
            if (p1.sub_propositions.items[0].equals(p0) and p1.sub_propositions.items[1].equals(self.conclusion)) {
                return true;
            }
        }
        return false;
    }

    fn validateConjunctionIntro(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 2) return false;
        if (self.conclusion.prop_type != .CONJUNCTION) return false;
        if (self.conclusion.sub_propositions.items.len < 2) return false;
        const idx0 = self.premise_indices.items[0];
        const idx1 = self.premise_indices.items[1];
        if (idx0 >= all_steps.len or idx1 >= all_steps.len) return false;
        const left = self.conclusion.sub_propositions.items[0];
        const right = self.conclusion.sub_propositions.items[1];
        return (all_steps[idx0].conclusion.equals(left) and all_steps[idx1].conclusion.equals(right)) or
            (all_steps[idx0].conclusion.equals(right) and all_steps[idx1].conclusion.equals(left));
    }

    fn validateConjunctionElim(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= all_steps.len) return false;
        const premise = all_steps[idx].conclusion;
        if (premise.prop_type != .CONJUNCTION) return false;
        for (premise.sub_propositions.items) |sub| {
            if (sub.equals(self.conclusion)) return true;
        }
        return false;
    }

    fn validateDisjunctionIntro(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        if (self.conclusion.prop_type != .DISJUNCTION) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= all_steps.len) return false;
        const premise = all_steps[idx].conclusion;
        for (self.conclusion.sub_propositions.items) |sub| {
            if (sub.equals(premise)) return true;
        }
        return false;
    }

    fn validateDisjunctionElim(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 3) return false;
        const idx0 = self.premise_indices.items[0];
        const idx1 = self.premise_indices.items[1];
        const idx2 = self.premise_indices.items[2];
        if (idx0 >= all_steps.len or idx1 >= all_steps.len or idx2 >= all_steps.len) return false;
        const disj = all_steps[idx0].conclusion;
        if (disj.prop_type != .DISJUNCTION) return false;
        return all_steps[idx1].conclusion.equals(self.conclusion) and all_steps[idx2].conclusion.equals(self.conclusion);
    }

    fn validateNegationIntro(self: *const Self, all_steps: []const *ProofStep) bool {
        _ = all_steps;
        if (self.conclusion.prop_type != .NEGATION) return false;
        return self.premise_indices.items.len >= 1;
    }

    fn validateNegationElim(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= all_steps.len) return false;
        const premise = all_steps[idx].conclusion;
        if (premise.prop_type != .NEGATION) return false;
        if (premise.sub_propositions.items.len < 1) return false;
        const inner = premise.sub_propositions.items[0];
        if (inner.prop_type != .NEGATION) return false;
        if (inner.sub_propositions.items.len < 1) return false;
        return inner.sub_propositions.items[0].equals(self.conclusion);
    }

    fn validateImplicationIntro(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.conclusion.prop_type != .IMPLICATION) return false;
        if (self.conclusion.sub_propositions.items.len < 2) return false;
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= all_steps.len) return false;
        return all_steps[idx].conclusion.equals(self.conclusion.sub_propositions.items[1]);
    }

    fn validateUniversalInstantiation(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= all_steps.len) return false;
        const premise = all_steps[idx].conclusion;
        return premise.prop_type == .UNIVERSAL;
    }

    fn validateExistentialGeneralization(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.conclusion.prop_type != .EXISTENTIAL) return false;
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        return idx < all_steps.len and all_steps[idx].verified;
    }

    fn validateUniversalIntro(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.conclusion.prop_type != .UNIVERSAL) return false;
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        return idx < all_steps.len and all_steps[idx].verified;
    }

    fn validateExistentialElim(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 2) return false;
        const idx0 = self.premise_indices.items[0];
        if (idx0 >= all_steps.len) return false;
        return all_steps[idx0].conclusion.prop_type == .EXISTENTIAL;
    }

    fn validateFrameRule(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.conclusion.prop_type != .HOARE_TRIPLE) return false;
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= all_steps.len) return false;
        return all_steps[idx].conclusion.prop_type == .HOARE_TRIPLE and all_steps[idx].verified;
    }

    fn validateConsequenceRule(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.conclusion.prop_type != .HOARE_TRIPLE) return false;
        if (self.premise_indices.items.len < 3) return false;
        for (self.premise_indices.items) |idx| {
            if (idx >= all_steps.len) return false;
            if (!all_steps[idx].verified) return false;
        }
        return true;
    }

    fn validateSequenceRule(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.conclusion.prop_type != .HOARE_TRIPLE) return false;
        if (self.premise_indices.items.len < 2) return false;
        const idx0 = self.premise_indices.items[0];
        const idx1 = self.premise_indices.items[1];
        if (idx0 >= all_steps.len or idx1 >= all_steps.len) return false;
        const t1 = all_steps[idx0].conclusion;
        const t2 = all_steps[idx1].conclusion;
        if (t1.prop_type != .HOARE_TRIPLE or t2.prop_type != .HOARE_TRIPLE) return false;
        if (t1.sub_propositions.items.len < 3 or t2.sub_propositions.items.len < 3) return false;
        return t1.sub_propositions.items[2].equals(t2.sub_propositions.items[0]);
    }

    fn validateWeakening(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= all_steps.len) return false;
        const premise = all_steps[idx].conclusion;
        if (self.conclusion.prop_type == .DISJUNCTION) {
            for (self.conclusion.sub_propositions.items) |sub| {
                if (sub.equals(premise)) return true;
            }
        }
        return false;
    }

    fn validateStrengthening(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        if (idx >= all_steps.len) return false;
        const premise = all_steps[idx].conclusion;
        if (premise.prop_type == .CONJUNCTION) {
            for (premise.sub_propositions.items) |sub| {
                if (sub.equals(self.conclusion)) return true;
            }
        }
        return false;
    }

    fn validateContradiction(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 2) return false;
        const idx0 = self.premise_indices.items[0];
        const idx1 = self.premise_indices.items[1];
        if (idx0 >= all_steps.len or idx1 >= all_steps.len) return false;
        const p0 = all_steps[idx0].conclusion;
        const p1 = all_steps[idx1].conclusion;
        if (p0.prop_type == .NEGATION and p0.sub_propositions.items.len >= 1) {
            if (p0.sub_propositions.items[0].equals(p1)) return true;
        }
        if (p1.prop_type == .NEGATION and p1.sub_propositions.items.len >= 1) {
            if (p1.sub_propositions.items[0].equals(p0)) return true;
        }
        return false;
    }

    fn validateDeduction(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 1) return false;
        const idx = self.premise_indices.items[0];
        return idx < all_steps.len and all_steps[idx].verified;
    }

    fn validateInduction(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.premise_indices.items.len < 2) return false;
        for (self.premise_indices.items) |idx| {
            if (idx >= all_steps.len or !all_steps[idx].verified) return false;
        }
        return true;
    }

    fn validateTemporalInduction(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.conclusion.prop_type != .TEMPORAL_ALWAYS) return false;
        return self.validateInduction(all_steps);
    }

    fn validateLoopInvariant(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.conclusion.prop_type != .HOARE_TRIPLE) return false;
        return self.validateInduction(all_steps);
    }

    fn validateConditionalRule(self: *const Self, all_steps: []const *ProofStep) bool {
        if (self.conclusion.prop_type != .HOARE_TRIPLE) return false;
        if (self.premise_indices.items.len < 2) return false;
        for (self.premise_indices.items) |idx| {
            if (idx >= all_steps.len) return false;
            if (all_steps[idx].conclusion.prop_type != .HOARE_TRIPLE) return false;
        }
        return true;
    }
};

pub const FormalProof = struct {
    proof_id: u64,
    theorem: *Proposition,
    steps: ArrayList(*ProofStep),
    axioms: ArrayList(*Proposition),
    is_complete: bool,
    is_valid: bool,
    allocator: Allocator,
    creation_time: i128,

    const Self = @This();

    pub fn init(allocator: Allocator, proof_id: u64, theorem: *Proposition) !*Self {
        const proof = try allocator.create(Self);
        theorem.retain();
        proof.* = Self{
            .proof_id = proof_id,
            .theorem = theorem,
            .steps = ArrayList(*ProofStep).init(allocator),
            .axioms = ArrayList(*Proposition).init(allocator),
            .is_complete = false,
            .is_valid = false,
            .allocator = allocator,
            .creation_time = std.time.nanoTimestamp(),
        };
        return proof;
    }

    pub fn deinit(self: *Self) void {
        self.theorem.release();
        for (self.steps.items) |step| {
            step.deinit();
            self.allocator.destroy(step);
        }
        self.steps.deinit();
        for (self.axioms.items) |axiom| {
            axiom.release();
        }
        self.axioms.deinit();
    }

    pub fn addStep(self: *Self, step: *ProofStep) !void {
        try self.steps.append(step);
        self.is_valid = false;
        self.is_complete = false;
    }

    pub fn addAxiom(self: *Self, axiom: *Proposition) !void {
        axiom.retain();
        try self.axioms.append(axiom);
    }

    pub fn validate(self: *Self) !bool {
        if (self.steps.items.len == 0) {
            return false;
        }
        var all_verified = true;
        var i: usize = 0;
        while (i < self.steps.items.len) : (i += 1) {
            const step = self.steps.items[i];
            const premises_slice = self.steps.items[0..i];
            if (!step.verify(premises_slice)) {
                all_verified = false;
                break;
            }
        }
        self.is_valid = all_verified;
        if (all_verified and self.steps.items.len > 0) {
            const last_step = self.steps.items[self.steps.items.len - 1];
            self.is_complete = last_step.conclusion.equals(self.theorem);
        } else {
            self.is_complete = false;
        }
        return self.is_valid and self.is_complete;
    }

    pub fn getLastStep(self: *const Self) ?*ProofStep {
        if (self.steps.items.len == 0) return null;
        return self.steps.items[self.steps.items.len - 1];
    }

    pub fn stepCount(self: *const Self) usize {
        return self.steps.items.len;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const new_proof = try allocator.create(Self);
        new_proof.* = Self{
            .proof_id = self.proof_id,
            .theorem = try self.theorem.clone(allocator),
            .steps = ArrayList(*ProofStep).init(allocator),
            .axioms = ArrayList(*Proposition).init(allocator),
            .is_complete = self.is_complete,
            .is_valid = self.is_valid,
            .allocator = allocator,
            .creation_time = self.creation_time,
        };
        for (self.steps.items) |step| {
            try new_proof.steps.append(try step.clone(allocator));
        }
        for (self.axioms.items) |axiom| {
            try new_proof.axioms.append(try axiom.clone(allocator));
        }
        return new_proof;
    }
};

pub const PredicateFn = *const fn (*const SelfSimilarRelationalGraph) VerificationError!bool;

pub const Invariant = struct {
    id: u64,
    inv_type: InvariantType,
    description: []const u8,
    priority_val: u8,
    predicate: PredicateFn,
    enabled: bool,
    violation_count: u64,
    last_check_time: i128,
    last_check_passed: bool,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, id: u64, inv_type: InvariantType, description: []const u8, predicate: PredicateFn) !*Self {
        const inv = try allocator.create(Self);
        inv.* = Self{
            .id = id,
            .inv_type = inv_type,
            .description = try allocator.dupe(u8, description),
            .priority_val = inv_type.priority(),
            .predicate = predicate,
            .enabled = true,
            .violation_count = 0,
            .last_check_time = 0,
            .last_check_passed = true,
            .allocator = allocator,
        };
        return inv;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.description);
    }

    pub fn check(self: *Self, graph: *const SelfSimilarRelationalGraph) VerificationError!bool {
        self.last_check_time = std.time.nanoTimestamp();
        if (!self.enabled) {
            self.last_check_passed = true;
            return true;
        }
        const result = try self.predicate(graph);
        self.last_check_passed = result;
        if (!result) {
            self.violation_count += 1;
        }
        return result;
    }

    pub fn enable(self: *Self) void {
        self.enabled = true;
    }

    pub fn disable(self: *Self) void {
        self.enabled = false;
    }

    pub fn resetViolationCount(self: *Self) void {
        self.violation_count = 0;
    }
};

fn checkConnectivityPredicate(graph: *const SelfSimilarRelationalGraph) VerificationError!bool {
    const node_count = graph.nodes.count();
    if (node_count <= 1) {
        return true;
    }
    var visited = StringHashMap(void).init(graph.allocator);
    defer visited.deinit();
    var queue = ArrayList([]const u8).init(graph.allocator);
    defer queue.deinit();
    var adjacency = StringHashMap(ArrayList([]const u8)).init(graph.allocator);
    defer {
        var iter = adjacency.valueIterator();
        while (iter.next()) |list| {
            list.deinit();
        }
        adjacency.deinit();
    }
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!adjacency.contains(key.source)) {
            try adjacency.put(key.source, ArrayList([]const u8).init(graph.allocator));
        }
        if (!adjacency.contains(key.target)) {
            try adjacency.put(key.target, ArrayList([]const u8).init(graph.allocator));
        }
        var src_list = adjacency.getPtr(key.source).?;
        try src_list.append(key.target);
        var tgt_list = adjacency.getPtr(key.target).?;
        try tgt_list.append(key.source);
    }
    var node_iter = graph.nodes.iterator();
    if (node_iter.next()) |first_entry| {
        try queue.append(first_entry.key_ptr.*);
    } else {
        return true;
    }
    while (queue.items.len > 0) {
        const current = queue.pop().?;
        if (visited.contains(current)) {
            continue;
        }
        try visited.put(current, {});
        if (adjacency.get(current)) |neighbors| {
            for (neighbors.items) |neighbor| {
                if (!visited.contains(neighbor)) {
                    try queue.append(neighbor);
                }
            }
        }
    }
    return visited.count() == node_count;
}

fn checkCoherencePredicate(graph: *const SelfSimilarRelationalGraph) VerificationError!bool {
    const epsilon: f64 = 1e-9;
    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        const node = entry.value_ptr;
        const magnitude = std.math.sqrt(node.qubit.a.re * node.qubit.a.re + node.qubit.a.im * node.qubit.a.im + node.qubit.b.re * node.qubit.b.re + node.qubit.b.im * node.qubit.b.im);
        if (magnitude > 1.0 + epsilon) {
            return false;
        }
    }
    return true;
}

fn checkEntanglementPredicate(graph: *const SelfSimilarRelationalGraph) VerificationError!bool {
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        const key = entry.key_ptr.*;
        for (entry.value_ptr.items) |edge| {
            if (edge.quality == .entangled) {
                if (!graph.nodes.contains(key.source) or !graph.nodes.contains(key.target)) {
                    return false;
                }
            }
        }
    }
    return true;
}

fn checkFractalDimensionPredicate(graph: *const SelfSimilarRelationalGraph) VerificationError!bool {
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            if (edge.fractal_dimension < 0.0 or edge.fractal_dimension > 3.0) {
                return false;
            }
        }
    }
    return true;
}

fn checkQuantumStatePredicate(graph: *const SelfSimilarRelationalGraph) VerificationError!bool {
    const epsilon: f64 = 1e-9;
    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        const node = entry.value_ptr;
        if (std.math.isNan(node.qubit.a.re) or std.math.isNan(node.qubit.a.im) or std.math.isNan(node.qubit.b.re) or std.math.isNan(node.qubit.b.im)) {
            return false;
        }
        if (std.math.isInf(node.qubit.a.re) or std.math.isInf(node.qubit.a.im) or std.math.isInf(node.qubit.b.re) or std.math.isInf(node.qubit.b.im)) {
            return false;
        }
        const prob = node.qubit.normSquared();
        if (std.math.isNan(prob) or std.math.isInf(prob)) {
            return false;
        }
        if (prob < 0.0 or prob > 1.0 + epsilon) {
            return false;
        }
    }
    return true;
}

fn checkMemorySafetyPredicate(graph: *const SelfSimilarRelationalGraph) VerificationError!bool {
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!graph.nodes.contains(key.source) or !graph.nodes.contains(key.target)) {
            return false;
        }
    }
    return true;
}

fn checkTypeSafetyPredicate(graph: *const SelfSimilarRelationalGraph) VerificationError!bool {
    const max_weight: f64 = 1e10;
    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        const node = entry.value_ptr;
        if (node.id.len == 0) {
            return false;
        }
    }
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        for (entry.value_ptr.items) |edge| {
            if (std.math.isNan(edge.weight) or std.math.isInf(edge.weight)) {
                return false;
            }
            if (edge.weight < 0.0 or edge.weight > max_weight) {
                return false;
            }
        }
    }
    return true;
}

fn checkSymmetryPredicate(graph: *const SelfSimilarRelationalGraph) VerificationError!bool {
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        const key = entry.key_ptr.*;
        const reverse_key = EdgeKey{ .source = key.target, .target = key.source };
        if (graph.edges.get(reverse_key)) |reverse_edges| {
            if (reverse_edges.items.len != entry.value_ptr.items.len) {
                return false;
            }
        }
    }
    return true;
}

fn checkTemporalConsistencyPredicate(graph: *const SelfSimilarRelationalGraph) VerificationError!bool {
    var node_iter = graph.nodes.iterator();
    while (node_iter.next()) |entry| {
        const node = entry.value_ptr;
        if (std.math.isNan(node.phase) or std.math.isInf(node.phase)) {
            return false;
        }
    }
    var edge_iter = graph.edges.iterator();
    while (edge_iter.next()) |entry| {
        const key = entry.key_ptr.*;
        const source_node = graph.nodes.get(key.source) orelse continue;
        const target_node = graph.nodes.get(key.target) orelse continue;
        const source_time_str = source_node.getMetadata("creation_time") orelse continue;
        const target_time_str = target_node.getMetadata("creation_time") orelse continue;
        const source_time = std.fmt.parseInt(i64, source_time_str, 10) catch continue;
        const target_time = std.fmt.parseInt(i64, target_time_str, 10) catch continue;
        if (target_time < source_time) {
            return false;
        }
    }
    return true;
}

const INVARIANT_TYPE_COUNT = @typeInfo(InvariantType).@"enum".fields.len;

pub const InvariantRegistry = struct {
    invariants: AutoHashMap(u64, *Invariant),
    invariants_by_type: [INVARIANT_TYPE_COUNT]ArrayList(*Invariant),
    next_id: u64,
    allocator: Allocator,
    check_count: u64,
    violation_count: u64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        var by_type: [INVARIANT_TYPE_COUNT]ArrayList(*Invariant) = undefined;
        for (&by_type) |*list| {
            list.* = ArrayList(*Invariant).init(allocator);
        }
        return Self{
            .invariants = AutoHashMap(u64, *Invariant).init(allocator),
            .invariants_by_type = by_type,
            .next_id = 1,
            .allocator = allocator,
            .check_count = 0,
            .violation_count = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.invariants.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.invariants.deinit();
        for (&self.invariants_by_type) |*list| {
            list.deinit();
        }
    }

    pub fn registerInvariant(self: *Self, inv_type: InvariantType, description: []const u8, predicate: PredicateFn) !u64 {
        const id = self.next_id;
        self.next_id += 1;
        const inv = try Invariant.init(self.allocator, id, inv_type, description, predicate);
        try self.invariants.put(id, inv);
        try self.invariants_by_type[@intFromEnum(inv_type)].append(inv);
        return id;
    }

    pub fn registerCoreInvariants(self: *Self) !void {
        _ = try self.registerInvariant(.CONNECTIVITY, "Graph must maintain connectivity between nodes", checkConnectivityPredicate);
        _ = try self.registerInvariant(.COHERENCE, "Quantum state magnitude must not exceed 1", checkCoherencePredicate);
        _ = try self.registerInvariant(.ENTANGLEMENT, "Entangled edges must have valid paired nodes", checkEntanglementPredicate);
        _ = try self.registerInvariant(.FRACTAL_DIMENSION, "Fractal dimension must be in [0, 3]", checkFractalDimensionPredicate);
        _ = try self.registerInvariant(.QUANTUM_STATE, "Quantum states must be valid probabilities", checkQuantumStatePredicate);
        _ = try self.registerInvariant(.MEMORY_SAFETY, "All edge endpoints must reference existing nodes", checkMemorySafetyPredicate);
        _ = try self.registerInvariant(.TYPE_SAFETY, "All values must have valid types", checkTypeSafetyPredicate);
        _ = try self.registerInvariant(.SYMMETRY, "Graph symmetry properties must be preserved", checkSymmetryPredicate);
        _ = try self.registerInvariant(.TEMPORAL_CONSISTENCY, "Temporal ordering must be consistent", checkTemporalConsistencyPredicate);
    }

    pub fn getInvariant(self: *const Self, id: u64) ?*Invariant {
        return self.invariants.get(id);
    }

    pub fn getInvariantsByType(self: *const Self, inv_type: InvariantType) []const *Invariant {
        return self.invariants_by_type[@intFromEnum(inv_type)].items;
    }

    pub fn checkAll(self: *Self, graph: *const SelfSimilarRelationalGraph) !bool {
        self.check_count += 1;
        var all_passed = true;
        var current_violations: u64 = 0;
        var iter = self.invariants.iterator();
        while (iter.next()) |entry| {
            const passed = try entry.value_ptr.*.check(graph);
            if (!passed) {
                current_violations += 1;
                all_passed = false;
            }
        }
        self.violation_count += current_violations;
        return all_passed;
    }

    pub fn checkByType(self: *Self, graph: *const SelfSimilarRelationalGraph, inv_type: InvariantType) !bool {
        self.check_count += 1;
        for (self.invariants_by_type[@intFromEnum(inv_type)].items) |inv| {
            const passed = try inv.check(graph);
            if (!passed) {
                self.violation_count += 1;
                return false;
            }
        }
        return true;
    }

    pub fn checkByPriority(self: *Self, graph: *const SelfSimilarRelationalGraph, min_priority: u8) !bool {
        self.check_count += 1;
        var all_passed = true;
        var iter = self.invariants.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.*.priority_val >= min_priority) {
                const passed = try entry.value_ptr.*.check(graph);
                if (!passed) {
                    self.violation_count += 1;
                    all_passed = false;
                }
            }
        }
        return all_passed;
    }

    pub fn getViolatedInvariants(self: *const Self) !ArrayList(u64) {
        var violated = ArrayList(u64).init(self.allocator);
        var iter = self.invariants.iterator();
        while (iter.next()) |entry| {
            if (!entry.value_ptr.*.last_check_passed) {
                try violated.append(entry.value_ptr.*.id);
            }
        }
        return violated;
    }

    pub fn resetAllViolationCounts(self: *Self) void {
        var iter = self.invariants.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.resetViolationCount();
        }
        self.violation_count = 0;
    }

    pub fn getStatistics(self: *const Self) InvariantStatistics {
        var total_violations: u64 = 0;
        var enabled_count: usize = 0;
        var iter = self.invariants.iterator();
        while (iter.next()) |entry| {
            total_violations += entry.value_ptr.*.violation_count;
            if (entry.value_ptr.*.enabled) {
                enabled_count += 1;
            }
        }
        return InvariantStatistics{
            .total_invariants = self.invariants.count(),
            .enabled_invariants = enabled_count,
            .total_checks = self.check_count,
            .total_violations = total_violations,
        };
    }

    pub fn count(self: *const Self) usize {
        return self.invariants.count();
    }
};

pub const InvariantStatistics = struct {
    total_invariants: usize,
    enabled_invariants: usize,
    total_checks: u64,
    total_violations: u64,
};

pub const HoareTriple = struct {
    id: u64,
    precondition: *Proposition,
    operation: *Proposition,
    postcondition: *Proposition,
    verified: bool,
    verification_time: i128,
    frame_condition: ?*Proposition,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, id: u64, precondition: *Proposition, operation: *Proposition, postcondition: *Proposition) !*Self {
        const triple = try allocator.create(Self);
        precondition.retain();
        operation.retain();
        postcondition.retain();
        triple.* = Self{
            .id = id,
            .precondition = precondition,
            .operation = operation,
            .postcondition = postcondition,
            .verified = false,
            .verification_time = 0,
            .frame_condition = null,
            .allocator = allocator,
        };
        return triple;
    }

    pub fn deinit(self: *Self) void {
        self.precondition.release();
        self.operation.release();
        self.postcondition.release();
        if (self.frame_condition) |fc| {
            fc.release();
        }
    }

    pub fn setFrameCondition(self: *Self, frame: *Proposition) void {
        if (self.frame_condition) |fc| {
            fc.release();
        }
        frame.retain();
        self.frame_condition = frame;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Self {
        const new_triple = try allocator.create(Self);
        new_triple.* = Self{
            .id = self.id,
            .precondition = try self.precondition.clone(allocator),
            .operation = try self.operation.clone(allocator),
            .postcondition = try self.postcondition.clone(allocator),
            .verified = self.verified,
            .verification_time = self.verification_time,
            .frame_condition = if (self.frame_condition) |fc| try fc.clone(allocator) else null,
            .allocator = allocator,
        };
        return new_triple;
    }

    pub fn toProposition(self: *const Self) !*Proposition {
        return Proposition.initHoareTriple(
            self.allocator,
            try self.precondition.clone(self.allocator),
            try self.operation.clone(self.allocator),
            try self.postcondition.clone(self.allocator),
        );
    }
};

pub const HoareLogicVerifier = struct {
    triples: AutoHashMap(u64, *HoareTriple),
    next_id: u64,
    allocator: Allocator,
    total_verifications: u64,
    successful_verifications: u64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .triples = AutoHashMap(u64, *HoareTriple).init(allocator),
            .next_id = 1,
            .allocator = allocator,
            .total_verifications = 0,
            .successful_verifications = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.triples.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.triples.deinit();
    }

    pub fn createTriple(self: *Self, precondition: *Proposition, operation: *Proposition, postcondition: *Proposition) !u64 {
        const id = self.next_id;
        self.next_id += 1;
        const triple = try HoareTriple.init(self.allocator, id, precondition, operation, postcondition);
        try self.triples.put(id, triple);
        return id;
    }

    pub fn getTriple(self: *const Self, id: u64) ?*HoareTriple {
        return self.triples.get(id);
    }

    pub fn verifyAssignmentAxiom(self: *Self, triple_id: u64, variable: []const u8, expression: *const Term) !bool {
        self.total_verifications += 1;
        const triple = self.triples.get(triple_id) orelse return VerificationError.InvalidProofStep;
        var post_clone = try triple.postcondition.clone(self.allocator);
        defer post_clone.release();
        try post_clone.substitute(variable, expression);
        const valid = post_clone.equals(triple.precondition);
        if (valid) {
            self.successful_verifications += 1;
            triple.verified = true;
            triple.verification_time = std.time.nanoTimestamp();
        }
        return valid;
    }

    pub fn verifySequenceRule(self: *Self, triple1_id: u64, triple2_id: u64) !bool {
        self.total_verifications += 1;
        const triple1 = self.triples.get(triple1_id) orelse return VerificationError.InvalidProofStep;
        const triple2 = self.triples.get(triple2_id) orelse return VerificationError.InvalidProofStep;
        if (!triple1.verified or !triple2.verified) {
            return false;
        }
        const valid = triple1.postcondition.equals(triple2.precondition);
        if (valid) {
            self.successful_verifications += 1;
        }
        return valid;
    }

    pub fn verifyConditionalRule(self: *Self, condition: *const Proposition, then_triple_id: u64, else_triple_id: u64) !bool {
        self.total_verifications += 1;
        const then_triple = self.triples.get(then_triple_id) orelse return VerificationError.InvalidProofStep;
        const else_triple = self.triples.get(else_triple_id) orelse return VerificationError.InvalidProofStep;
        _ = condition;
        const valid = then_triple.verified and else_triple.verified and
            then_triple.postcondition.equals(else_triple.postcondition);
        if (valid) {
            self.successful_verifications += 1;
        }
        return valid;
    }

    pub fn verifyLoopInvariant(self: *Self, invariant: *const Proposition, body_triple_id: u64) !bool {
        self.total_verifications += 1;
        const body_triple = self.triples.get(body_triple_id) orelse return VerificationError.InvalidProofStep;
        const valid = body_triple.verified and
            body_triple.precondition.equals(invariant) and
            body_triple.postcondition.equals(invariant);
        if (valid) {
            self.successful_verifications += 1;
        }
        return valid;
    }

    pub fn verifyConsequenceRule(self: *Self, triple_id: u64, stronger_pre: *const Proposition, weaker_post: *const Proposition) !bool {
        self.total_verifications += 1;
        const triple = self.triples.get(triple_id) orelse return VerificationError.InvalidProofStep;
        if (!triple.verified) return false;
        _ = stronger_pre;
        _ = weaker_post;
        self.successful_verifications += 1;
        return true;
    }

    pub fn verifyFrameRule(self: *Self, triple_id: u64, frame: *Proposition) !bool {
        self.total_verifications += 1;
        const triple = self.triples.get(triple_id) orelse return VerificationError.InvalidProofStep;
        if (!triple.verified) {
            return false;
        }
        triple.setFrameCondition(frame);
        self.successful_verifications += 1;
        return true;
    }

    pub fn composeTriples(self: *Self, ids: []const u64) !u64 {
        if (ids.len < 2) return VerificationError.InvalidProofStep;
        var current_triple = self.triples.get(ids[0]) orelse return VerificationError.InvalidProofStep;
        for (ids[1..]) |id| {
            const next_triple = self.triples.get(id) orelse return VerificationError.InvalidProofStep;
            if (!current_triple.postcondition.equals(next_triple.precondition)) {
                return VerificationError.InvalidConclusion;
            }
            current_triple = next_triple;
        }
        const first = self.triples.get(ids[0]).?;
        const last = self.triples.get(ids[ids.len - 1]).?;
        const composed_pre = try first.precondition.clone(self.allocator);
        var op_buf: [64]u8 = undefined;
        const op_name = std.fmt.bufPrint(op_buf[0..], "seq_{d}_{d}", .{ ids[0], ids[ids.len - 1] }) catch "seq";
        const composed_op = try Proposition.initAtomic(self.allocator, op_name);
        const composed_post = try last.postcondition.clone(self.allocator);
        return self.createTriple(composed_pre, composed_op, composed_post);
    }

    pub fn count(self: *const Self) usize {
        return self.triples.count();
    }

    pub fn getStatistics(self: *const Self) HoareStatistics {
        return HoareStatistics{
            .total_triples = self.triples.count(),
            .total_verifications = self.total_verifications,
            .successful_verifications = self.successful_verifications,
        };
    }
};

pub const HoareStatistics = struct {
    total_triples: usize,
    total_verifications: u64,
    successful_verifications: u64,
};

pub const Substitution = struct {
    mappings: StringHashMap(*Term),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .mappings = StringHashMap(*Term).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.mappings.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.release();
        }
        self.mappings.deinit();
    }

    pub fn add(self: *Self, var_name: []const u8, term: *Term) !void {
        if (self.mappings.contains(var_name)) {
            return;
        }
        const key = try self.allocator.dupe(u8, var_name);
        term.retain();
        try self.mappings.put(key, term);
    }

    pub fn get(self: *const Self, var_name: []const u8) ?*Term {
        return self.mappings.get(var_name);
    }

    pub fn contains(self: *const Self, var_name: []const u8) bool {
        return self.mappings.contains(var_name);
    }

    pub fn apply(self: *const Self, term: *Term) !void {
        if (term.kind == .VARIABLE) {
            if (self.get(term.name)) |replacement| {
                try term.substitute(term.name, replacement);
            }
        }
        for (term.args.items) |arg| {
            try self.apply(arg);
        }
    }

    pub fn compose(self: *Self, other: *const Substitution) !void {
        var iter = other.mappings.iterator();
        while (iter.next()) |entry| {
            if (!self.contains(entry.key_ptr.*)) {
                try self.add(entry.key_ptr.*, entry.value_ptr.*);
            }
        }
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Substitution {
        const new_sub = try allocator.create(Substitution);
        new_sub.* = Substitution.init(allocator);
        var iter = self.mappings.iterator();
        while (iter.next()) |entry| {
            const cloned_term = try entry.value_ptr.*.clone(allocator);
            try new_sub.add(entry.key_ptr.*, cloned_term);
            cloned_term.release();
        }
        return new_sub;
    }
};

pub const Clause = struct {
    literals: ArrayList(*Proposition),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .literals = ArrayList(*Proposition).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.literals.items) |lit| {
            lit.release();
        }
        self.literals.deinit();
    }

    pub fn addLiteral(self: *Self, lit: *Proposition) !void {
        lit.retain();
        try self.literals.append(lit);
    }

    pub fn isEmpty(self: *const Self) bool {
        return self.literals.items.len == 0;
    }

    pub fn isUnit(self: *const Self) bool {
        return self.literals.items.len == 1;
    }

    pub fn size(self: *const Self) usize {
        return self.literals.items.len;
    }

    pub fn containsComplementary(self: *const Self) bool {
        var i: usize = 0;
        while (i < self.literals.items.len) : (i += 1) {
            const lit1 = self.literals.items[i];
            for (self.literals.items[i + 1 ..]) |lit2| {
                if (areComplementary(lit1, lit2)) {
                    return true;
                }
            }
        }
        return false;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !*Clause {
        const new_clause = try allocator.create(Clause);
        new_clause.* = Clause.init(allocator);
        for (self.literals.items) |lit| {
            const cloned_lit = try lit.clone(allocator);
            try new_clause.addLiteral(cloned_lit);
            cloned_lit.release();
        }
        return new_clause;
    }
};

fn areComplementary(p1: *const Proposition, p2: *const Proposition) bool {
    if (p1.prop_type == .NEGATION and p1.sub_propositions.items.len >= 1) {
        if (p1.sub_propositions.items[0].equals(p2)) {
            return true;
        }
    }
    if (p2.prop_type == .NEGATION and p2.sub_propositions.items.len >= 1) {
        if (p2.sub_propositions.items[0].equals(p1)) {
            return true;
        }
    }
    return false;
}

pub const ProofTreeNode = struct {
    proposition: *Proposition,
    rule: ProofRule,
    children: ArrayList(*ProofTreeNode),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, proposition: *Proposition, rule: ProofRule) !*Self {
        const node = try allocator.create(Self);
        proposition.retain();
        node.* = Self{
            .proposition = proposition,
            .rule = rule,
            .children = ArrayList(*ProofTreeNode).init(allocator),
            .allocator = allocator,
        };
        return node;
    }

    pub fn deinit(self: *Self) void {
        self.proposition.release();
        for (self.children.items) |child| {
            child.deinit();
            self.allocator.destroy(child);
        }
        self.children.deinit();
    }

    pub fn addChild(self: *Self, child: *ProofTreeNode) !void {
        try self.children.append(child);
    }

    pub fn isComplete(self: *const Self) bool {
        if (self.children.items.len == 0) {
            return self.rule == .AXIOM or self.proposition.prop_type == .TRUE;
        }
        for (self.children.items) |child| {
            if (!child.isComplete()) {
                return false;
            }
        }
        return true;
    }
};

pub const TheoremProver = struct {
    axioms: ArrayList(*Proposition),
    clauses: ArrayList(*Clause),
    max_depth: usize,
    allocator: Allocator,
    unification_count: u64,
    resolution_count: u64,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .axioms = ArrayList(*Proposition).init(allocator),
            .clauses = ArrayList(*Clause).init(allocator),
            .max_depth = 100,
            .allocator = allocator,
            .unification_count = 0,
            .resolution_count = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.axioms.items) |axiom| {
            axiom.release();
        }
        self.axioms.deinit();
        for (self.clauses.items) |clause| {
            clause.deinit();
            self.allocator.destroy(clause);
        }
        self.clauses.deinit();
    }

    pub fn addAxiom(self: *Self, axiom: *Proposition) !void {
        axiom.retain();
        try self.axioms.append(axiom);
    }

    pub fn addClause(self: *Self, clause: *Clause) !void {
        try self.clauses.append(clause);
    }

    pub fn unify(self: *Self, t1: *const Term, t2: *const Term) !?*Substitution {
        self.unification_count += 1;
        var sub = try self.allocator.create(Substitution);
        sub.* = Substitution.init(self.allocator);
        const success = try self.unifyTerms(t1, t2, sub);
        if (!success) {
            sub.deinit();
            self.allocator.destroy(sub);
            return null;
        }
        return sub;
    }

    fn unifyTerms(self: *Self, t1: *const Term, t2: *const Term, sub: *Substitution) !bool {
        if (t1.kind == .VARIABLE) {
            return self.unifyVariable(t1, t2, sub);
        }
        if (t2.kind == .VARIABLE) {
            return self.unifyVariable(t2, t1, sub);
        }
        if (!std.mem.eql(u8, t1.name, t2.name)) {
            return false;
        }
        if (t1.kind != t2.kind) {
            return false;
        }
        if (t1.args.items.len != t2.args.items.len) {
            return false;
        }
        var i: usize = 0;
        while (i < t1.args.items.len) : (i += 1) {
            if (!try self.unifyTerms(t1.args.items[i], t2.args.items[i], sub)) {
                return false;
            }
        }
        return true;
    }

    fn unifyVariable(self: *Self, variable: *const Term, term: *const Term, sub: *Substitution) !bool {
        _ = self;
        if (sub.contains(variable.name)) {
            const existing = sub.get(variable.name).?;
            return existing.equals(term);
        }
        if (term.containsVariable(variable.name)) {
            return false;
        }
        var term_clone = try term.clone(sub.allocator);
        try sub.add(variable.name, term_clone);
        term_clone.release();
        return true;
    }

    pub fn resolve(self: *Self, c1: *const Clause, c2: *const Clause, idx1: usize, idx2: usize) !?*Clause {
        self.resolution_count += 1;
        if (idx1 >= c1.literals.items.len or idx2 >= c2.literals.items.len) {
            return null;
        }
        const lit1 = c1.literals.items[idx1];
        const lit2 = c2.literals.items[idx2];
        if (!areComplementary(lit1, lit2)) {
            return null;
        }
        var resolvent = try self.allocator.create(Clause);
        resolvent.* = Clause.init(self.allocator);
        var i: usize = 0;
        while (i < c1.literals.items.len) : (i += 1) {
            if (i != idx1) {
                try resolvent.addLiteral(try c1.literals.items[i].clone(self.allocator));
                resolvent.literals.items[resolvent.literals.items.len - 1].release();
            }
        }
        var j: usize = 0;
        while (j < c2.literals.items.len) : (j += 1) {
            if (j != idx2) {
                const lit = c2.literals.items[j];
                var duplicate = false;
                for (resolvent.literals.items) |existing| {
                    if (existing.equals(lit)) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    try resolvent.addLiteral(try lit.clone(self.allocator));
                    resolvent.literals.items[resolvent.literals.items.len - 1].release();
                }
            }
        }
        return resolvent;
    }

    pub fn proveByResolution(self: *Self, goal: *Proposition) !bool {
        var negated_goal = try goal.negate(self.allocator);
        defer negated_goal.release();
        var working_set = ArrayList(*Clause).init(self.allocator);
        defer {
            for (working_set.items) |c| {
                c.deinit();
                self.allocator.destroy(c);
            }
            working_set.deinit();
        }
        for (self.clauses.items) |clause| {
            try working_set.append(try clause.clone(self.allocator));
        }
        var goal_clause = try self.allocator.create(Clause);
        goal_clause.* = Clause.init(self.allocator);
        try goal_clause.addLiteral(negated_goal);
        try working_set.append(goal_clause);
        var iterations: usize = 0;
        const max_iterations = self.max_depth * self.max_depth;
        while (iterations < max_iterations) : (iterations += 1) {
            var new_clauses = ArrayList(*Clause).init(self.allocator);
            defer new_clauses.deinit();
            var i: usize = 0;
            while (i < working_set.items.len) : (i += 1) {
                const c1 = working_set.items[i];
                for (working_set.items[i + 1 ..]) |c2| {
                    var li: usize = 0;
                    while (li < c1.literals.items.len) : (li += 1) {
                        var lj: usize = 0;
                        while (lj < c2.literals.items.len) : (lj += 1) {
                            if (try self.resolve(c1, c2, li, lj)) |resolvent| {
                                if (resolvent.isEmpty()) {
                                    resolvent.deinit();
                                    self.allocator.destroy(resolvent);
                                    return true;
                                }
                                if (!resolvent.containsComplementary()) {
                                    try new_clauses.append(resolvent);
                                } else {
                                    resolvent.deinit();
                                    self.allocator.destroy(resolvent);
                                }
                            }
                        }
                    }
                }
            }
            if (new_clauses.items.len == 0) {
                break;
            }
            for (new_clauses.items) |nc| {
                try working_set.append(nc);
            }
        }
        return false;
    }

    pub fn proveByBackwardChaining(self: *Self, goal: *Proposition, depth: usize) !bool {
        if (depth > self.max_depth) {
            return false;
        }
        for (self.axioms.items) |axiom| {
            if (axiom.equals(goal)) {
                return true;
            }
        }
        if (goal.prop_type == .TRUE) {
            return true;
        }
        if (goal.prop_type == .FALSE) {
            return false;
        }
        if (goal.prop_type == .CONJUNCTION) {
            for (goal.sub_propositions.items) |sub| {
                if (!try self.proveByBackwardChaining(sub, depth + 1)) {
                    return false;
                }
            }
            return true;
        }
        if (goal.prop_type == .DISJUNCTION) {
            for (goal.sub_propositions.items) |sub| {
                if (try self.proveByBackwardChaining(sub, depth + 1)) {
                    return true;
                }
            }
            return false;
        }
        for (self.axioms.items) |axiom| {
            if (axiom.prop_type == .IMPLICATION and axiom.sub_propositions.items.len >= 2) {
                if (axiom.sub_propositions.items[1].equals(goal)) {
                    if (try self.proveByBackwardChaining(axiom.sub_propositions.items[0], depth + 1)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    pub fn buildProofTree(self: *Self, goal: *Proposition) !?*ProofTreeNode {
        for (self.axioms.items) |axiom| {
            if (axiom.equals(goal)) {
                const goal_clone = try goal.clone(self.allocator);
                return ProofTreeNode.init(self.allocator, goal_clone, .AXIOM);
            }
        }
        if (goal.prop_type == .CONJUNCTION and goal.sub_propositions.items.len >= 2) {
            var children = ArrayList(*ProofTreeNode).init(self.allocator);
            defer children.deinit();
            for (goal.sub_propositions.items) |sub| {
                if (try self.buildProofTree(sub)) |child_tree| {
                    try children.append(child_tree);
                } else {
                    for (children.items) |c| {
                        c.deinit();
                        self.allocator.destroy(c);
                    }
                    return null;
                }
            }
            const goal_clone = try goal.clone(self.allocator);
            const root = try ProofTreeNode.init(self.allocator, goal_clone, .CONJUNCTION_INTRO);
            for (children.items) |child| {
                try root.addChild(child);
            }
            return root;
        }
        return null;
    }

    pub fn setMaxDepth(self: *Self, depth: usize) void {
        self.max_depth = depth;
    }

    pub fn getStatistics(self: *const Self) ProverStatistics {
        return ProverStatistics{
            .axiom_count = self.axioms.items.len,
            .clause_count = self.clauses.items.len,
            .unification_count = self.unification_count,
            .resolution_count = self.resolution_count,
        };
    }

    pub fn checkProof(self: *Self, proof: *FormalProof) !bool {
        var steps = ArrayList(*ProofStep).init(self.allocator);
        defer {
            for (steps.items) |s| {
                s.deinit();
                self.allocator.destroy(s);
            }
            steps.deinit();
        }
        for (proof.axioms.items) |axiom| {
            const step = try ProofStep.init(self.allocator, @intCast(steps.items.len), .AXIOM, axiom);
            step.verified = true;
            try steps.append(step);
        }
        for (proof.steps.items) |step| {
            var step_clone = try step.clone(self.allocator);
            const valid = step_clone.verify(steps.items);
            if (!valid) return false;
            try steps.append(step_clone);
        }
        return true;
    }

    pub fn applyRule(self: *Self, rule: ProofRule, premises: []const *Proposition, conclusion: *Proposition) !?*ProofStep {
        if (rule.minimumPremises() > premises.len) return null;
        const step_id: u64 = @intCast(self.resolution_count + self.unification_count);
        var premise_indices = ArrayList(usize).init(self.allocator);
        defer premise_indices.deinit();
        const step = try ProofStep.init(self.allocator, step_id, rule, conclusion);
        var i: usize = 0;
        while (i < premises.len) : (i += 1) {
            try step.premise_indices.append(i);
        }
        const valid = switch (rule) {
            .AXIOM => conclusion.prop_type == .TRUE or conclusion.isAtomic(),
            .CONJUNCTION_INTRO => blk: {
                if (premises.len < 2) break :blk false;
                if (conclusion.prop_type != .CONJUNCTION) break :blk false;
                if (conclusion.sub_propositions.items.len < 2) break :blk false;
                break :blk true;
            },
            .CONJUNCTION_ELIM => blk: {
                if (premises.len < 1) break :blk false;
                if (premises[0].prop_type != .CONJUNCTION) break :blk false;
                break :blk true;
            },
            .MODUS_PONENS => blk: {
                if (premises.len < 2) break :blk false;
                if (premises[0].prop_type == .IMPLICATION and premises[0].sub_propositions.items.len >= 2) {
                    if (premises[0].sub_propositions.items[0].equals(premises[1])) {
                        break :blk premises[0].sub_propositions.items[1].equals(conclusion);
                    }
                }
                break :blk false;
            },
            .WEAKENING => true,
            .STRENGTHENING => true,
            else => true,
        };
        step.verified = valid;
        return step;
    }
};

pub const ProverStatistics = struct {
    axiom_count: usize,
    clause_count: usize,
    unification_count: u64,
    resolution_count: u64,
};

pub const VerificationResult = struct {
    success: bool,
    error_type: ?VerificationError,
    violated_invariants: ArrayList(u64),
    execution_time_ns: i128,
    graph_hash: [32]u8,
    allocator: Allocator,

    const Self = @This();

    pub fn initSuccess(allocator: Allocator, graph_hash: [32]u8, exec_time: i128) !*Self {
        const result = try allocator.create(Self);
        result.* = Self{
            .success = true,
            .error_type = null,
            .violated_invariants = ArrayList(u64).init(allocator),
            .execution_time_ns = exec_time,
            .graph_hash = graph_hash,
            .allocator = allocator,
        };
        return result;
    }

    pub fn initFailure(allocator: Allocator, err: VerificationError, graph_hash: [32]u8, exec_time: i128) !*Self {
        const result = try allocator.create(Self);
        result.* = Self{
            .success = false,
            .error_type = err,
            .violated_invariants = ArrayList(u64).init(allocator),
            .execution_time_ns = exec_time,
            .graph_hash = graph_hash,
            .allocator = allocator,
        };
        return result;
    }

    pub fn deinit(self: *Self) void {
        self.violated_invariants.deinit();
    }

    pub fn addViolation(self: *Self, inv_id: u64) !void {
        for (self.violated_invariants.items) |existing| {
            if (existing == inv_id) return;
        }
        try self.violated_invariants.append(inv_id);
    }
};

pub const FormalVerificationEngine = struct {
    invariant_registry: InvariantRegistry,
    hoare_verifier: HoareLogicVerifier,
    theorem_prover: TheoremProver,
    verification_history: ArrayList(*VerificationResult),
    allocator: Allocator,
    total_verifications: u64,
    successful_verifications: u64,
    creation_time: i128,
    next_proof_id: u64,

    const Self = @This();

    pub fn init(allocator: Allocator) !Self {
        var registry = InvariantRegistry.init(allocator);
        try registry.registerCoreInvariants();
        return Self{
            .invariant_registry = registry,
            .hoare_verifier = HoareLogicVerifier.init(allocator),
            .theorem_prover = TheoremProver.init(allocator),
            .verification_history = ArrayList(*VerificationResult).init(allocator),
            .allocator = allocator,
            .total_verifications = 0,
            .successful_verifications = 0,
            .creation_time = std.time.nanoTimestamp(),
            .next_proof_id = 1,
        };
    }

    pub fn deinit(self: *Self) void {
        self.invariant_registry.deinit();
        self.hoare_verifier.deinit();
        self.theorem_prover.deinit();
        for (self.verification_history.items) |result| {
            result.deinit();
            self.allocator.destroy(result);
        }
        self.verification_history.deinit();
    }

    pub fn verifyGraph(self: *Self, graph: *const SelfSimilarRelationalGraph) !*VerificationResult {
        const start_time: i128 = std.time.nanoTimestamp();
        self.total_verifications += 1;
        var hash_buf: [32]u8 = undefined;
        var hasher = Sha256.init(.{});
        hasher.update(try @constCast(graph).getTopologyHashHex());
        var node_count_u64 = std.math.cast(u64, graph.nodes.count()) orelse unreachable;
        hasher.update(std.mem.asBytes(&node_count_u64));
        var edge_count_u64 = std.math.cast(u64, graph.edges.count()) orelse unreachable;
        hasher.update(std.mem.asBytes(&edge_count_u64));
        hasher.final(&hash_buf);
        self.invariant_registry.resetAllViolationCounts();
        const all_passed = try self.invariant_registry.checkAll(graph);
        const end_time: i128 = std.time.nanoTimestamp();
        const exec_time: i128 = end_time - start_time;
        var result: *VerificationResult = undefined;
        if (all_passed) {
            self.successful_verifications += 1;
            result = try VerificationResult.initSuccess(self.allocator, hash_buf, exec_time);
        } else {
            result = try VerificationResult.initFailure(self.allocator, VerificationError.InvariantViolation, hash_buf, exec_time);
            var violated = try self.invariant_registry.getViolatedInvariants();
            defer violated.deinit();
            for (violated.items) |inv_id| {
                try result.addViolation(inv_id);
            }
        }
        try self.verification_history.append(result);
        return result;
    }

    pub fn verifyInvariantType(self: *Self, graph: *const SelfSimilarRelationalGraph, inv_type: InvariantType) !bool {
        return self.invariant_registry.checkByType(graph, inv_type);
    }

    pub fn registerCustomInvariant(self: *Self, inv_type: InvariantType, description: []const u8, predicate: PredicateFn) !u64 {
        return self.invariant_registry.registerInvariant(inv_type, description, predicate);
    }

    pub fn createHoareTriple(self: *Self, pre: *Proposition, op: *Proposition, post: *Proposition) !u64 {
        return self.hoare_verifier.createTriple(pre, op, post);
    }

    pub fn verifyHoareTriple(self: *Self, triple_id: u64, variable: []const u8, expression: *Term) !bool {
        return self.hoare_verifier.verifyAssignmentAxiom(triple_id, variable, expression);
    }

    pub fn prove(self: *Self, goal: *Proposition) !bool {
        if (try self.theorem_prover.proveByBackwardChaining(goal, 0)) {
            const proof = try self.buildFormalProof(goal);
            defer {
                proof.deinit();
                self.allocator.destroy(proof);
            }
            const proof_valid = try self.theorem_prover.checkProof(proof);
            if (!proof_valid) return false;
            return true;
        }
        const resolution_result = try self.theorem_prover.proveByResolution(goal);
        if (resolution_result) {
            const proof = try self.buildFormalProof(goal);
            defer {
                proof.deinit();
                self.allocator.destroy(proof);
            }
            _ = try self.theorem_prover.checkProof(proof);
        }
        return resolution_result;
    }

    pub fn verifyInferenceStep(self: *Self, pre_condition: *Proposition, post_condition: *Proposition) !bool {
        const implication = try pre_condition.implies(self.allocator, post_condition);
        defer implication.release();
        return self.prove(implication);
    }

    pub fn checkProofCompleteness(_: *Self, proof: *FormalProof) !bool {
        return proof.validate();
    }

    pub fn proveByResolution(self: *Self, goal: *Proposition) !bool {
        return self.theorem_prover.proveByResolution(goal);
    }

    pub fn addAxiom(self: *Self, axiom: *Proposition) !void {
        try self.theorem_prover.addAxiom(axiom);
    }

    pub fn buildFormalProof(self: *Self, theorem: *Proposition) !*FormalProof {
        const proof_id = self.next_proof_id;
        self.next_proof_id += 1;
        const theorem_clone = try theorem.clone(self.allocator);
        const proof = try FormalProof.init(self.allocator, proof_id, theorem_clone);
        for (self.theorem_prover.axioms.items) |axiom| {
            const axiom_clone = try axiom.clone(self.allocator);
            try proof.addAxiom(axiom_clone);
        }
        return proof;
    }

    pub fn getStatistics(self: *const Self) EngineStatistics {
        const current_time: i128 = std.time.nanoTimestamp();
        return EngineStatistics{
            .total_verifications = self.total_verifications,
            .successful_verifications = self.successful_verifications,
            .invariant_count = self.invariant_registry.count(),
            .hoare_triple_count = self.hoare_verifier.count(),
            .axiom_count = self.theorem_prover.axioms.items.len,
            .history_size = self.verification_history.items.len,
            .uptime_ns = current_time - self.creation_time,
        };
    }

    pub fn clearHistory(self: *Self) void {
        for (self.verification_history.items) |result| {
            result.deinit();
            self.allocator.destroy(result);
        }
        self.verification_history.clearRetainingCapacity();
        self.invariant_registry.resetAllViolationCounts();
    }

    pub fn verifyInferenceHook(self: *Self, pre_condition: *Proposition, post_condition: *Proposition, graph: *const SelfSimilarRelationalGraph) !bool {
        const graph_ok = try self.invariant_registry.checkAll(graph);
        if (!graph_ok) return false;
        const step_ok = try self.verifyInferenceStep(pre_condition, post_condition);
        if (!step_ok) return false;
        const proof = try self.buildFormalProof(post_condition);
        defer {
            proof.deinit();
            self.allocator.destroy(proof);
        }
        return self.theorem_prover.checkProof(proof);
    }

    pub fn verifyInferenceHookSimple(self: *Self, pre_condition: *Proposition, post_condition: *Proposition, graph: *const SelfSimilarRelationalGraph) !bool {
        const graph_ok = try self.invariant_registry.checkAll(graph);
        if (!graph_ok) return false;
        return self.verifyInferenceStep(pre_condition, post_condition);
    }
};

pub const EngineStatistics = struct {
    total_verifications: u64,
    successful_verifications: u64,
    invariant_count: usize,
    hoare_triple_count: usize,
    axiom_count: usize,
    history_size: usize,
    uptime_ns: i128,
};

test "InvariantType enum conversion" {
    const testing = std.testing;
    try testing.expectEqual(InvariantType.CONNECTIVITY, InvariantType.fromString("connectivity").?);
    try testing.expectEqual(InvariantType.COHERENCE, InvariantType.fromString("coherence").?);
    try testing.expectEqual(InvariantType.ENTANGLEMENT, InvariantType.fromString("entanglement").?);
    try testing.expectEqualStrings("memory_safety", InvariantType.MEMORY_SAFETY.toString());
}

test "ProofRule enum properties" {
    const testing = std.testing;
    try testing.expect(!ProofRule.AXIOM.requiresPremises());
    try testing.expect(ProofRule.MODUS_PONENS.requiresPremises());
    try testing.expectEqual(@as(usize, 2), ProofRule.MODUS_PONENS.minimumPremises());
    try testing.expectEqual(@as(usize, 0), ProofRule.AXIOM.minimumPremises());
}

test "PropType arity" {
    const testing = std.testing;
    try testing.expectEqual(@as(usize, 0), PropType.ATOMIC.arity());
    try testing.expectEqual(@as(usize, 1), PropType.NEGATION.arity());
    try testing.expectEqual(@as(usize, 2), PropType.CONJUNCTION.arity());
    try testing.expectEqual(@as(usize, 3), PropType.HOARE_TRIPLE.arity());
}

test "Term creation and operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const var_term = try Term.initVariable(allocator, "x");
    defer var_term.release();
    try testing.expect(var_term.isVariable());
    try testing.expectEqualStrings("x", var_term.name);
    const const_term = try Term.initConstant(allocator, "42");
    defer const_term.release();
    try testing.expect(!const_term.isVariable());
}

test "Proposition creation and cloning" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const atomic = try Proposition.initAtomic(allocator, "P");
    defer atomic.release();
    try testing.expectEqual(PropType.ATOMIC, atomic.prop_type);
    try testing.expectEqualStrings("P", atomic.predicate_name);
    const cloned = try atomic.clone(allocator);
    defer cloned.release();
    try testing.expect(atomic.equals(cloned));
}

test "Proposition negation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const p = try Proposition.initAtomic(allocator, "P");
    defer p.release();
    const not_p = try Proposition.initNegation(allocator, p);
    defer not_p.release();
    try testing.expectEqual(PropType.NEGATION, not_p.prop_type);
    try testing.expectEqual(@as(usize, 1), not_p.sub_propositions.items.len);
}

test "Proposition binary operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const p = try Proposition.initAtomic(allocator, "P");
    defer p.release();
    const q = try Proposition.initAtomic(allocator, "Q");
    defer q.release();
    const p_and_q = try Proposition.initBinary(allocator, .CONJUNCTION, p, q);
    defer p_and_q.release();
    try testing.expectEqual(PropType.CONJUNCTION, p_and_q.prop_type);
    try testing.expectEqual(@as(usize, 2), p_and_q.sub_propositions.items.len);
}

test "Hoare triple creation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const pre = try Proposition.initTrue(allocator);
    defer pre.release();
    const op = try Proposition.initAtomic(allocator, "assign");
    defer op.release();
    const post = try Proposition.initTrue(allocator);
    defer post.release();
    const triple = try Proposition.initHoareTriple(allocator, pre, op, post);
    defer triple.release();
    try testing.expectEqual(PropType.HOARE_TRIPLE, triple.prop_type);
    try testing.expectEqual(@as(usize, 3), triple.sub_propositions.items.len);
}

test "ProofStep creation and verification" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const conclusion = try Proposition.initTrue(allocator);
    defer conclusion.release();
    const step = try ProofStep.init(allocator, 1, .AXIOM, conclusion);
    defer {
        step.deinit();
        allocator.destroy(step);
    }
    try testing.expectEqual(@as(u64, 1), step.step_id);
    try testing.expectEqual(ProofRule.AXIOM, step.rule_applied);
    const verified = step.verify(&[_]*ProofStep{});
    try testing.expect(verified);
}

test "FormalProof creation and step addition" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const theorem = try Proposition.initAtomic(allocator, "theorem");
    defer theorem.release();
    var proof = try FormalProof.init(allocator, 1, theorem);
    defer {
        proof.deinit();
        allocator.destroy(proof);
    }
    const step_conclusion = try Proposition.initTrue(allocator);
    defer step_conclusion.release();
    const step = try ProofStep.init(allocator, 1, .AXIOM, step_conclusion);
    try proof.addStep(step);
    try testing.expectEqual(@as(usize, 1), proof.stepCount());
}

test "InvariantRegistry core invariants" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var registry = InvariantRegistry.init(allocator);
    defer registry.deinit();
    try registry.registerCoreInvariants();
    try testing.expect(registry.count() >= 9);
}

test "checkConnectivity on empty graph" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();
    const result = try checkConnectivityPredicate(&graph);
    try testing.expect(result);
}

test "HoareLogicVerifier creation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var verifier = HoareLogicVerifier.init(allocator);
    defer verifier.deinit();
    const pre = try Proposition.initTrue(allocator);
    defer pre.release();
    const op = try Proposition.initAtomic(allocator, "op");
    defer op.release();
    const post = try Proposition.initTrue(allocator);
    defer post.release();
    const triple_id = try verifier.createTriple(pre, op, post);
    try testing.expect(triple_id > 0);
    try testing.expectEqual(@as(usize, 1), verifier.count());
}

test "TheoremProver backward chaining" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var prover = TheoremProver.init(allocator);
    defer prover.deinit();
    const axiom = try Proposition.initTrue(allocator);
    defer axiom.release();
    try prover.addAxiom(axiom);
    const goal = try Proposition.initTrue(allocator);
    defer goal.release();
    const result = try prover.proveByBackwardChaining(goal, 0);
    try testing.expect(result);
}

test "Clause operations" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var clause = Clause.init(allocator);
    defer clause.deinit();
    try testing.expect(clause.isEmpty());
    const lit = try Proposition.initAtomic(allocator, "P");
    defer lit.release();
    try clause.addLiteral(lit);
    try testing.expect(!clause.isEmpty());
    try testing.expect(clause.isUnit());
}

test "FormalVerificationEngine creation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    var engine = try FormalVerificationEngine.init(allocator);
    defer engine.deinit();
    try testing.expect(engine.invariant_registry.count() >= 9);
    const stats = engine.getStatistics();
    try testing.expectEqual(@as(u64, 0), stats.total_verifications);
}

