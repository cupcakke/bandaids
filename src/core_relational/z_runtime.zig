const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const StringHashMap = std.StringHashMap;
const Complex = std.math.Complex;

const nsir = @import("nsir_core.zig");
const SelfSimilarRelationalGraph = nsir.SelfSimilarRelationalGraph;
const Node = nsir.Node;
const Edge = nsir.Edge;
const EdgeQuality = nsir.EdgeQuality;
const Qubit = nsir.Qubit;

const quantum = @import("quantum_logic.zig");
const RelationalQuantumLogic = quantum.RelationalQuantumLogic;
const LogicGate = quantum.LogicGate;
const QuantumState = quantum.QuantumState;
const MeasurementResult = quantum.MeasurementResult;

pub const HistoryEntryType = enum(u8) {
    assign = 0,
    transform = 1,
    relate = 2,
    measure = 3,
    entangle = 4,

    pub fn toString(self: HistoryEntryType) []const u8 {
        return switch (self) {
            .assign => "assign",
            .transform => "transform",
            .relate => "relate",
            .measure => "measure",
            .entangle => "entangle",
        };
    }
};

pub const HistoryEntry = struct {
    entry_type: HistoryEntryType,
    value: []const u8,
    timestamp: i128,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, entry_type: HistoryEntryType, value: []const u8) !Self {
        return Self{
            .entry_type = entry_type,
            .value = try allocator.dupe(u8, value),
            .timestamp = @truncate(std.time.nanoTimestamp()),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.value);
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        return Self{
            .entry_type = self.entry_type,
            .value = try allocator.dupe(u8, self.value),
            .timestamp = self.timestamp,
            .allocator = allocator,
        };
    }
};

pub const ExecutionAction = enum(u8) {
    create_variable = 0,
    delete_variable = 1,
    relational_operation = 2,
    entangle_variables = 3,
    propagate_information = 4,
    fractal_transform = 5,
    measure = 6,
    quantum_circuit = 7,
    relational_expression = 8,

    pub fn toString(self: ExecutionAction) []const u8 {
        return switch (self) {
            .create_variable => "create_variable",
            .delete_variable => "delete_variable",
            .relational_operation => "relational_operation",
            .entangle_variables => "entangle_variables",
            .propagate_information => "propagate_information",
            .fractal_transform => "fractal_transform",
            .measure => "measure",
            .quantum_circuit => "quantum_circuit",
            .relational_expression => "relational_expression",
        };
    }
};

pub const ExecutionHistoryEntry = struct {
    action: ExecutionAction,
    primary_target: []const u8,
    secondary_targets: ArrayList([]const u8),
    operation_type: ?[]const u8,
    result_value: ?[]const u8,
    result_int: ?i32,
    result_float: ?f64,
    timestamp: i128,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, action: ExecutionAction, primary_target: []const u8) !Self {
        return Self{
            .action = action,
            .primary_target = try allocator.dupe(u8, primary_target),
            .secondary_targets = ArrayList([]const u8).init(allocator),
            .operation_type = null,
            .result_value = null,
            .result_int = null,
            .result_float = null,
            .timestamp = @truncate(std.time.nanoTimestamp()),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.primary_target);
        for (self.secondary_targets.items) |target| {
            self.allocator.free(target);
        }
        self.secondary_targets.deinit();
        if (self.operation_type) |op| {
            self.allocator.free(op);
        }
        if (self.result_value) |rv| {
            self.allocator.free(rv);
        }
    }

    pub fn addSecondaryTarget(self: *Self, target: []const u8) !void {
        const target_copy = try self.allocator.dupe(u8, target);
        errdefer self.allocator.free(target_copy);
        try self.secondary_targets.append(target_copy);
    }

    pub fn setOperationType(self: *Self, op_type: []const u8) !void {
        if (self.operation_type) |existing| {
            self.allocator.free(existing);
        }
        self.operation_type = try self.allocator.dupe(u8, op_type);
    }

    pub fn setResultValue(self: *Self, value: []const u8) !void {
        if (self.result_value) |existing| {
            self.allocator.free(existing);
        }
        self.result_value = try self.allocator.dupe(u8, value);
    }

    pub fn setResultInt(self: *Self, value: i32) void {
        self.result_int = value;
    }

    pub fn setResultFloat(self: *Self, value: f64) void {
        self.result_float = value;
    }

    pub fn clone(self: *const Self, allocator: Allocator) !Self {
        const primary_target_copy = try allocator.dupe(u8, self.primary_target);
        errdefer allocator.free(primary_target_copy);

        var new_entry = Self{
            .action = self.action,
            .primary_target = primary_target_copy,
            .secondary_targets = ArrayList([]const u8).init(allocator),
            .operation_type = null,
            .result_value = null,
            .result_int = self.result_int,
            .result_float = self.result_float,
            .timestamp = self.timestamp,
            .allocator = allocator,
        };
        errdefer {
            for (new_entry.secondary_targets.items) |t| {
                allocator.free(t);
            }
            new_entry.secondary_targets.deinit();
        }

        for (self.secondary_targets.items) |target| {
            try new_entry.addSecondaryTarget(target);
        }

        if (self.operation_type) |op| {
            new_entry.operation_type = try allocator.dupe(u8, op);
        }
        errdefer if (new_entry.operation_type) |op2| {
            allocator.free(op2);
        };

        if (self.result_value) |rv| {
            new_entry.result_value = try allocator.dupe(u8, rv);
        }

        return new_entry;
    }
};

pub const ZVariable = struct {
    name: []const u8,
    graph: *SelfSimilarRelationalGraph,
    logic: *RelationalQuantumLogic,
    history: ArrayList(HistoryEntry),
    creation_order: ArrayList([]const u8),
    allocator: Allocator,
    owns_graph: bool,
    owns_logic: bool,

    const Self = @This();

    pub fn init(allocator: Allocator, name: []const u8) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const graph = try allocator.create(SelfSimilarRelationalGraph);
        errdefer allocator.destroy(graph);
        graph.* = try SelfSimilarRelationalGraph.init(allocator);
        errdefer graph.deinit();

        const logic = try allocator.create(RelationalQuantumLogic);
        errdefer allocator.destroy(logic);
        logic.* = RelationalQuantumLogic.init(allocator);
        errdefer logic.deinit();

        const name_copy = try allocator.dupe(u8, name);
        errdefer allocator.free(name_copy);

        self.* = Self{
            .name = name_copy,
            .graph = graph,
            .logic = logic,
            .history = ArrayList(HistoryEntry).init(allocator),
            .creation_order = ArrayList([]const u8).init(allocator),
            .allocator = allocator,
            .owns_graph = true,
            .owns_logic = true,
        };

        return self;
    }

    pub fn initWithValue(allocator: Allocator, name: []const u8, initial_value: []const u8) !*Self {
        const self = try Self.init(allocator, name);
        errdefer self.deinit();
        try self.assign(initial_value);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.name);

        for (self.history.items) |*entry| {
            entry.deinit();
        }
        self.history.deinit();

        for (self.creation_order.items) |order_id| {
            self.allocator.free(order_id);
        }
        self.creation_order.deinit();

        if (self.owns_graph) {
            self.graph.deinit();
            self.allocator.destroy(self.graph);
        }

        if (self.owns_logic) {
            self.logic.deinit();
            self.allocator.destroy(self.logic);
        }

        self.allocator.destroy(self);
    }

    pub fn assign(self: *Self, value: []const u8) !void {
        const node_id = try self.graph.encodeInformation(value);

        const node_id_copy = try self.allocator.dupe(u8, node_id);
        errdefer self.allocator.free(node_id_copy);
        try self.creation_order.append(node_id_copy);

        self.allocator.free(node_id);

        var hasher = std.hash.Wyhash.init(0);
        hasher.update(value);
        const hash_val = hasher.final();
        const hash_f = @as(f64, @floatFromInt(hash_val % 10000)) * 0.001;
        const amp_real = @cos(hash_f);
        const amp_imag = @sin(hash_f);

        _ = try self.logic.initializeState(amp_real, amp_imag, 0.0, 0.0, 0.0);

        var history_entry = try HistoryEntry.init(self.allocator, .assign, value);
        errdefer history_entry.deinit();
        try self.history.append(history_entry);
    }

    pub fn getValue(self: *const Self) ?[]const u8 {
        if (self.creation_order.items.len == 0) {
            return null;
        }

        const latest_node_id = self.creation_order.items[self.creation_order.items.len - 1];
        return self.graph.decodeInformation(latest_node_id);
    }

    pub fn getLatestNodeId(self: *const Self) ?[]const u8 {
        if (self.creation_order.items.len == 0) {
            return null;
        }

        return self.creation_order.items[self.creation_order.items.len - 1];
    }

    pub fn relateTo(self: *Self, other: *const Self, relationship_type: EdgeQuality) !void {
        if (self.graph.nodeCount() == 0 or other.graph.nodeCount() == 0) {
            return;
        }

        const self_node_id_ref = self.getLatestNodeId() orelse return;
        const other_node_id_ref = other.getLatestNodeId() orelse return;

        const self_node_id = try self.allocator.dupe(u8, self_node_id_ref);
        defer self.allocator.free(self_node_id);

        const other_node_id = try self.allocator.dupe(u8, other_node_id_ref);
        defer self.allocator.free(other_node_id);

        const self_qubit = self.graph.getQuantumState(self_node_id) orelse Qubit.initBasis0();
        const other_qubit = other.graph.getQuantumState(other_node_id) orelse Qubit.initBasis0();
        const self_state = self_qubit.a;
        const other_state = other_qubit.a;
        const other_conj = other_state.conjugate();
        const correlation = self_state.mul(other_conj);

        const weight = correlation.magnitude();
        const fractal_dim = if (self.graph.edgeCount() > 0 and self.graph.nodeCount() > 0) blk: {
            var sum_dim: f64 = 0.0;
            var count: usize = 0;
            var eit = self.graph.edges.iterator();
            while (eit.next()) |entry| {
                for (entry.value_ptr.items) |e| {
                    sum_dim += e.fractal_dimension;
                    count += 1;
                }
            }
            break :blk if (count > 0) sum_dim / @as(f64, @floatFromInt(count)) else 1.0;
        } else 1.0;

        if (self.graph.getNodeConst(other_node_id) == null) {
            if (other.graph.getNodeConst(other_node_id)) |other_node| {
                const cloned = try other_node.clone(self.allocator);
                try self.graph.addNode(cloned);
            }
        }

        const edge = Edge.init(
            self.allocator,
            self_node_id,
            other_node_id,
            relationship_type,
            weight,
            correlation,
            fractal_dim,
        );
        try self.graph.addEdge(self_node_id, other_node_id, edge);

        if (self.logic.stateCount() > 0 and other.logic.stateCount() > 0) {
            const other_states_start = self.logic.stateCount();
            for (other.logic.states.items) |state| {
                try self.logic.states.append(state.clone());
            }
            if (other_states_start > 0 and other_states_start < self.logic.stateCount()) {
                try self.logic.entangle(other_states_start - 1, other_states_start);
            }
        }

        var buf: [128]u8 = undefined;
        const relate_str = std.fmt.bufPrint(&buf, "relate:{s}:{s}", .{ self_node_id, other_node_id }) catch "relate";
        var history_entry = try HistoryEntry.init(self.allocator, .relate, relate_str);
        errdefer history_entry.deinit();
        try self.history.append(history_entry);
    }

    pub fn transform(self: *Self, gate: LogicGate, params: ?[]const f64) !void {
        if (self.logic.stateCount() > 0) {
            const indices = [_]usize{self.logic.stateCount() - 1};
            try self.logic.applyGate(gate, &indices, params);
        }

        var history_entry = try HistoryEntry.init(self.allocator, .transform, gate.toString());
        errdefer history_entry.deinit();
        try self.history.append(history_entry);
    }

    pub fn measure(self: *Self) MeasurementResult {
        if (self.logic.stateCount() > 0) {
            const result = self.logic.measure(self.logic.stateCount() - 1);

            var buf: [64]u8 = undefined;
            const measure_str = std.fmt.bufPrint(&buf, "result:{d}:prob0:{d:.6}", .{ result.result, result.probability_zero }) catch "measure";
            var history_entry = HistoryEntry.init(self.allocator, .measure, measure_str) catch {
                return result;
            };
            self.history.append(history_entry) catch {
                history_entry.deinit();
            };

            return result;
        }

        return MeasurementResult{
            .result = 0,
            .probability_zero = 1.0,
            .probability_one = 0.0,
            .collapsed_state = QuantumState.init(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        };
    }

    pub fn getTopologySignature(self: *const Self) []const u8 {
        return self.graph.getTopologyHashHex();
    }

    pub fn getFractalDimension(self: *const Self) f64 {
        const node_count = self.graph.nodeCount();
        const edge_count = self.graph.edgeCount();
        if (node_count == 0) return 0.0;
        if (edge_count == 0) return @max(1.0, @as(f64, @floatFromInt(node_count)) / @as(f64, @floatFromInt(@max(node_count, 1))));
        var total_fractal_dim: f64 = 0.0;
        var edge_iter = self.graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |edge| {
                total_fractal_dim += edge.fractal_dimension;
            }
        }
        if (total_fractal_dim > 0.0 and edge_count > 0) {
            return total_fractal_dim / @as(f64, @floatFromInt(edge_count));
        }
        const ratio = @as(f64, @floatFromInt(edge_count)) / @as(f64, @floatFromInt(node_count));
        if (ratio <= 1.0) return 1.0;
        const dim = std.math.log2(ratio) + 1.0;
        return @max(1.0, dim);
    }

    pub fn historyCount(self: *const Self) usize {
        return self.history.items.len;
    }

    pub fn getHistoryEntry(self: *const Self, index: usize) ?HistoryEntry {
        if (index >= self.history.items.len) {
            return null;
        }
        return self.history.items[index];
    }

    pub fn clearHistory(self: *Self) void {
        for (self.history.items) |*entry| {
            entry.deinit();
        }
        self.history.clearRetainingCapacity();
    }

    pub fn copyStatesFrom(self: *Self, other: *const Self) !void {
        for (other.logic.states.items) |state| {
            try self.logic.states.append(state.clone());
        }
    }
};

pub const RelationalOperationType = enum(u8) {
    op_and = 0,
    op_or = 1,
    op_xor = 2,
    op_entangle = 3,

    pub fn toString(self: RelationalOperationType) []const u8 {
        return switch (self) {
            .op_and => "and",
            .op_or => "or",
            .op_xor => "xor",
            .op_entangle => "entangle",
        };
    }

    pub fn fromString(s: []const u8) ?RelationalOperationType {
        if (std.ascii.eqlIgnoreCase(s, "and")) return .op_and;
        if (std.ascii.eqlIgnoreCase(s, "or")) return .op_or;
        if (std.ascii.eqlIgnoreCase(s, "xor")) return .op_xor;
        if (std.ascii.eqlIgnoreCase(s, "entangle")) return .op_entangle;
        return null;
    }

    pub fn toGate(self: RelationalOperationType) ?LogicGate {
        return switch (self) {
            .op_and => .RELATIONAL_AND,
            .op_or => .RELATIONAL_OR,
            .op_xor => .RELATIONAL_XOR,
            .op_entangle => null,
        };
    }
};

pub const VariableState = struct {
    name: []const u8,
    value: ?[]const u8,
    node_count: usize,
    edge_count: usize,
    fractal_dimension: f64,
    topology_hash: []const u8,
    state_count: usize,
    history_count: usize,
};

pub const SystemState = struct {
    variable_count: usize,
    total_nodes: usize,
    total_edges: usize,
    average_fractal_dimension: f64,
    execution_history_length: usize,
    variables: ArrayList(VariableState),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .variable_count = 0,
            .total_nodes = 0,
            .total_edges = 0,
            .average_fractal_dimension = 0.0,
            .execution_history_length = 0,
            .variables = ArrayList(VariableState).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.variables.items) |vs| {
            self.allocator.free(vs.name);
            if (vs.value) |v| {
                self.allocator.free(v);
            }
            self.allocator.free(vs.topology_hash);
        }
        self.variables.deinit();
    }
};

pub const GateSpec = struct {
    gate_name: []const u8,
    indices: []const usize,
    params: ?[]const f64,
};

pub const ZRuntime = struct {
    variables: StringHashMap(*ZVariable),
    global_graph: *SelfSimilarRelationalGraph,
    global_logic: *RelationalQuantumLogic,
    execution_history: ArrayList(ExecutionHistoryEntry),
    allocator: Allocator,
    variable_name_storage: ArrayList([]const u8),

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const global_graph = try allocator.create(SelfSimilarRelationalGraph);
        errdefer allocator.destroy(global_graph);
        global_graph.* = try SelfSimilarRelationalGraph.init(allocator);
        errdefer global_graph.deinit();

        const global_logic = try allocator.create(RelationalQuantumLogic);
        errdefer allocator.destroy(global_logic);
        global_logic.* = RelationalQuantumLogic.init(allocator);
        errdefer global_logic.deinit();

        self.* = Self{
            .variables = StringHashMap(*ZVariable).init(allocator),
            .global_graph = global_graph,
            .global_logic = global_logic,
            .execution_history = ArrayList(ExecutionHistoryEntry).init(allocator),
            .allocator = allocator,
            .variable_name_storage = ArrayList([]const u8).init(allocator),
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        var var_iter = self.variables.iterator();
        while (var_iter.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        self.variables.deinit();

        for (self.variable_name_storage.items) |name| {
            self.allocator.free(name);
        }
        self.variable_name_storage.deinit();

        for (self.execution_history.items) |*entry| {
            entry.deinit();
        }
        self.execution_history.deinit();

        self.global_graph.deinit();
        self.allocator.destroy(self.global_graph);

        self.global_logic.deinit();
        self.allocator.destroy(self.global_logic);

        self.allocator.destroy(self);
    }

    pub fn createVariable(self: *Self, name: []const u8, initial_value: ?[]const u8) !*ZVariable {
        if (self.variables.contains(name)) {
            const existing = self.variables.get(name).?;
            existing.deinit();
            _ = self.variables.remove(name);
            var i: usize = 0;
            while (i < self.variable_name_storage.items.len) {
                if (std.mem.eql(u8, self.variable_name_storage.items[i], name)) {
                    self.allocator.free(self.variable_name_storage.items[i]);
                    _ = self.variable_name_storage.swapRemove(i);
                    break;
                }
                i += 1;
            }
        }

        const var_ptr = if (initial_value) |val|
            try ZVariable.initWithValue(self.allocator, name, val)
        else
            try ZVariable.init(self.allocator, name);
        errdefer var_ptr.deinit();

        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        try self.variable_name_storage.append(name_copy);

        try self.variables.put(name_copy, var_ptr);

        var history_entry = try ExecutionHistoryEntry.init(self.allocator, .create_variable, name);
        errdefer history_entry.deinit();
        if (initial_value) |val| {
            try history_entry.setResultValue(val);
        }
        try self.execution_history.append(history_entry);

        return var_ptr;
    }

    pub fn getVariable(self: *Self, name: []const u8) ?*ZVariable {
        return self.variables.get(name);
    }

    pub fn getVariableConst(self: *const Self, name: []const u8) ?*const ZVariable {
        if (self.variables.get(name)) |v| {
            return v;
        }
        return null;
    }

    pub fn deleteVariable(self: *Self, name: []const u8) bool {
        if (self.variables.fetchRemove(name)) |kv| {
            kv.value.deinit();

            var i: usize = 0;
            while (i < self.variable_name_storage.items.len) {
                if (std.mem.eql(u8, self.variable_name_storage.items[i], name)) {
                    self.allocator.free(self.variable_name_storage.items[i]);
                    _ = self.variable_name_storage.swapRemove(i);
                    break;
                }
                i += 1;
            }

            var history_entry = ExecutionHistoryEntry.init(self.allocator, .delete_variable, name) catch {
                return true;
            };
            self.execution_history.append(history_entry) catch {
                history_entry.deinit();
            };

            return true;
        }
        return false;
    }

    pub fn relationalOperation(
        self: *Self,
        var1_name: []const u8,
        var2_name: []const u8,
        operation: RelationalOperationType,
    ) !?*ZVariable {
        const var1 = self.getVariable(var1_name) orelse return null;
        const var2 = self.getVariable(var2_name) orelse return null;

        var result_name_buf: [256]u8 = undefined;
        const result_name = std.fmt.bufPrint(&result_name_buf, "{s}_{s}_{s}", .{
            var1_name,
            operation.toString(),
            var2_name,
        }) catch return null;

        const result = try self.createVariable(result_name, null);

        if (operation.toGate()) |gate| {
            if (var1.logic.stateCount() > 0 and var2.logic.stateCount() > 0) {
                try result.copyStatesFrom(var1);
                try result.copyStatesFrom(var2);

                if (result.logic.stateCount() >= 2) {
                    const indices = [_]usize{ 0, 1 };
                    try result.logic.applyGate(gate, &indices, null);
                }
            }
        }

        try result.relateTo(var1, .coherent);
        try result.relateTo(var2, .coherent);

        var history_entry = try ExecutionHistoryEntry.init(self.allocator, .relational_operation, result_name);
        errdefer history_entry.deinit();
        try history_entry.setOperationType(operation.toString());
        try history_entry.addSecondaryTarget(var1_name);
        try history_entry.addSecondaryTarget(var2_name);
        try self.execution_history.append(history_entry);

        return result;
    }

    pub fn entangleVariables(self: *Self, var1_name: []const u8, var2_name: []const u8) !bool {
        const var1 = self.getVariable(var1_name) orelse return false;
        const var2 = self.getVariable(var2_name) orelse return false;

        try var1.relateTo(var2, .entangled);

        var history_entry = try ExecutionHistoryEntry.init(self.allocator, .entangle_variables, var1_name);
        errdefer history_entry.deinit();
        try history_entry.addSecondaryTarget(var2_name);
        try self.execution_history.append(history_entry);

        return true;
    }

    pub fn propagateInformation(self: *Self, source_var_name: []const u8, depth: usize) !ArrayList([]const u8) {
        var affected_vars = ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (affected_vars.items) |av| {
                self.allocator.free(av);
            }
            affected_vars.deinit();
        }

        const source_var = self.getVariable(source_var_name) orelse return affected_vars;

        const source_node_id = source_var.getLatestNodeId() orelse return affected_vars;

        var affected_nodes = try source_var.graph.propagateInformation(source_node_id, depth);
        defer {
            for (affected_nodes.items) |node_id| {
                self.allocator.free(node_id);
            }
            affected_nodes.deinit();
        }

        var affected_set = StringHashMap(void).init(self.allocator);
        defer affected_set.deinit();
        for (affected_nodes.items) |affected_id| {
            try affected_set.put(affected_id, {});
        }

        var var_iter = self.variables.iterator();
        while (var_iter.next()) |var_entry| {
            const var_name = var_entry.key_ptr.*;
            const variable = var_entry.value_ptr.*;

            if (std.mem.eql(u8, var_name, source_var_name)) {
                continue;
            }

            var node_iter = variable.graph.nodes.iterator();
            var found = false;
            while (node_iter.next()) |node_entry| {
                const node_id = node_entry.key_ptr.*;
                if (affected_set.contains(node_id)) {
                    found = true;
                    break;
                }
            }

            if (found) {
                const var_name_copy = try self.allocator.dupe(u8, var_name);
                try affected_vars.append(var_name_copy);
            }
        }

        var history_entry = try ExecutionHistoryEntry.init(self.allocator, .propagate_information, source_var_name);
        errdefer history_entry.deinit();
        for (affected_vars.items) |affected_name| {
            try history_entry.addSecondaryTarget(affected_name);
        }
        try self.execution_history.append(history_entry);

        return affected_vars;
    }

    pub fn applyFractalTransform(self: *Self, var_name: []const u8, depth: i32) !bool {
        const variable = self.getVariable(var_name) orelse return false;

        const params = [_]f64{@as(f64, @floatFromInt(depth))};
        try variable.transform(.FRACTAL_TRANSFORM, &params);

        var history_entry = try ExecutionHistoryEntry.init(self.allocator, .fractal_transform, var_name);
        errdefer history_entry.deinit();
        history_entry.setResultInt(depth);
        try self.execution_history.append(history_entry);

        return true;
    }

    pub fn measureVariable(self: *Self, var_name: []const u8) !?MeasurementResult {
        const variable = self.getVariable(var_name) orelse return null;

        const result = variable.measure();

        var history_entry = try ExecutionHistoryEntry.init(self.allocator, .measure, var_name);
        errdefer history_entry.deinit();
        history_entry.setResultInt(result.result);
        history_entry.setResultFloat(result.probability_zero);
        try self.execution_history.append(history_entry);

        return result;
    }

    pub fn getSystemState(self: *const Self) !SystemState {
        var state = SystemState.init(self.allocator);
        errdefer state.deinit();

        state.variable_count = self.variables.count();
        state.execution_history_length = self.execution_history.items.len;

        var total_fractal: f64 = 0.0;

        var var_iter = self.variables.iterator();
        while (var_iter.next()) |entry| {
            const var_name = entry.key_ptr.*;
            const variable = entry.value_ptr.*;

            const node_count = variable.graph.nodeCount();
            const edge_count = variable.graph.edgeCount();
            const fractal_dim = variable.getFractalDimension();
            const topo_hash = variable.getTopologySignature();
            const current_value = variable.getValue();

            const name_duped = try self.allocator.dupe(u8, var_name);
            errdefer self.allocator.free(name_duped);

            const value_duped = if (current_value) |v| try self.allocator.dupe(u8, v) else null;
            errdefer if (value_duped) |vd| {
                self.allocator.free(vd);
            };

            const topo_duped = try self.allocator.dupe(u8, topo_hash);
            errdefer self.allocator.free(topo_duped);

            const var_state = VariableState{
                .name = name_duped,
                .value = value_duped,
                .node_count = node_count,
                .edge_count = edge_count,
                .fractal_dimension = fractal_dim,
                .topology_hash = topo_duped,
                .state_count = variable.logic.stateCount(),
                .history_count = variable.historyCount(),
            };

            try state.variables.append(var_state);

            state.total_nodes += node_count;
            state.total_edges += edge_count;
            total_fractal += fractal_dim;
        }

        if (state.variable_count > 0) {
            state.average_fractal_dimension = total_fractal / @as(f64, @floatFromInt(state.variable_count));
        }

        return state;
    }

    pub fn executeQuantumCircuit(self: *Self, var_name: []const u8, circuit: []const GateSpec) !bool {
        const variable = self.getVariable(var_name) orelse return false;

        for (circuit) |spec| {
            const gate = LogicGate.fromString(spec.gate_name) orelse continue;
            try variable.logic.applyGate(gate, spec.indices, spec.params);
        }

        var history_entry = try ExecutionHistoryEntry.init(self.allocator, .quantum_circuit, var_name);
        errdefer history_entry.deinit();
        if (circuit.len <= std.math.maxInt(i32)) {
            history_entry.setResultInt(@intCast(circuit.len));
        } else {
            history_entry.setResultInt(std.math.maxInt(i32));
        }
        try self.execution_history.append(history_entry);

        return true;
    }

    pub fn computeRelationalExpression(self: *Self, expression: []const u8) !?[]const u8 {
        return self.parseExpression(expression);
    }

    fn parseExpression(self: *Self, expression: []const u8) !?[]const u8 {
        var tokens = ArrayList([]const u8).init(self.allocator);
        defer tokens.deinit();

        var paren_depth: usize = 0;
        var token_start: usize = 0;
        var i: usize = 0;
        while (i < expression.len) : (i += 1) {
            const ch = expression[i];
            if (ch == '(') {
                paren_depth += 1;
            } else if (ch == ')') {
                if (paren_depth > 0) paren_depth -= 1;
            } else if (ch == ' ' and paren_depth == 0) {
                if (i > token_start) {
                    try tokens.append(expression[token_start..i]);
                }
                token_start = i + 1;
            }
        }
        if (token_start < expression.len) {
            try tokens.append(expression[token_start..]);
        }

        if (tokens.items.len == 0) return null;

        if (std.mem.eql(u8, tokens.items[0], "not") or std.mem.eql(u8, tokens.items[0], "NOT")) {
            if (tokens.items.len < 2) return null;
            const inner = try self.parseSubExpression(tokens.items[1..]);
            if (inner == null) return null;
            const result = try self.allocator.dupe(u8, inner.?);
            self.allocator.free(inner.?);
            return result;
        }

        return self.parseBinaryExpression(tokens.items);
    }

    fn parseSubExpression(self: *Self, tokens: []const []const u8) !?[]const u8 {
        if (tokens.len == 0) return null;

        if (tokens.len >= 3) {
            const op_type = RelationalOperationType.fromString(tokens[1]);
            if (op_type != null) {
                return self.evaluateBinaryOp(tokens[0], op_type.?, tokens[2]);
            }
        }

        if (tokens.len == 1) {
            const var_name = tokens[0];
            const variable = self.getVariable(var_name) orelse return null;
            if (variable.getValue()) |val| {
                return try self.allocator.dupe(u8, val);
            }
        }

        return self.parseBinaryExpression(tokens);
    }

    fn evaluateBinaryOp(self: *Self, var1_name: []const u8, op_type: RelationalOperationType, var2_name: []const u8) !?[]const u8 {
        if (op_type == .op_entangle) {
            _ = try self.entangleVariables(var1_name, var2_name);
            return try self.allocator.dupe(u8, "true");
        }

        const result_var = try self.relationalOperation(var1_name, var2_name, op_type);
        if (result_var) |rv| {
            if (rv.getValue()) |val| {
                return try self.allocator.dupe(u8, val);
            }
        }

        return null;
    }

    fn parseBinaryExpression(self: *Self, tokens: []const []const u8) !?[]const u8 {
        if (tokens.len < 3) {
            if (tokens.len == 1) {
                const var_name = tokens[0];
                const variable = self.getVariable(var_name) orelse return null;
                if (variable.getValue()) |val| {
                    return try self.allocator.dupe(u8, val);
                }
            }
            return null;
        }

        var best_op_idx: ?usize = null;
        var best_precedence: u8 = 0;
        for (tokens, 0..) |token, idx| {
            if (idx == 0) continue;
            const op = RelationalOperationType.fromString(token) orelse continue;
            const precedence: u8 = switch (op) {
                .op_and => 3,
                .op_xor => 2,
                .op_or => 1,
                .op_entangle => 0,
            };
            if (best_op_idx == null or precedence > best_precedence) {
                best_op_idx = idx;
                best_precedence = precedence;
            }
        }

        if (best_op_idx) |op_idx| {
            const op_type = RelationalOperationType.fromString(tokens[op_idx]) orelse return null;
            var left_result: ?[]const u8 = null;
            var right_result: ?[]const u8 = null;

            if (op_idx > 1) {
                left_result = try self.parseBinaryExpression(tokens[0..op_idx]);
            }
            if (tokens.len > op_idx + 2) {
                right_result = try self.parseBinaryExpression(tokens[op_idx + 1 ..]);
            }

            const left_name = if (left_result) |lr| lr else tokens[op_idx - 1];
            const right_name = if (right_result) |rr| rr else tokens[op_idx + 1];

            const result = try self.evaluateBinaryOp(left_name, op_type, right_name);

            if (left_result) |lr| self.allocator.free(lr);
            if (right_result) |rr| self.allocator.free(rr);

            return result;
        }

        const var1_name = tokens[0];
        const operator = tokens[1];
        const var2_name = tokens[2];
        const op_type = RelationalOperationType.fromString(operator) orelse return null;
        return self.evaluateBinaryOp(var1_name, op_type, var2_name);
    }

    pub fn variableCount(self: *const Self) usize {
        return self.variables.count();
    }

    pub fn executionHistoryLength(self: *const Self) usize {
        return self.execution_history.items.len;
    }

    pub fn getExecutionHistoryEntry(self: *const Self, index: usize) ?ExecutionHistoryEntry {
        if (index >= self.execution_history.items.len) {
            return null;
        }
        return self.execution_history.items[index];
    }

    pub fn clearExecutionHistory(self: *Self) void {
        for (self.execution_history.items) |*entry| {
            entry.deinit();
        }
        self.execution_history.clearRetainingCapacity();
    }

    pub fn reset(self: *Self) !void {
        var var_iter = self.variables.iterator();
        while (var_iter.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        self.variables.clearRetainingCapacity();

        for (self.variable_name_storage.items) |name| {
            self.allocator.free(name);
        }
        self.variable_name_storage.clearRetainingCapacity();

        self.clearExecutionHistory();

        try self.global_graph.clear();
        self.global_logic.reset();
    }

    pub fn getAllVariableNames(self: *const Self, allocator: Allocator) !ArrayList([]const u8) {
        var names = ArrayList([]const u8).init(allocator);
        errdefer {
            for (names.items) |n| {
                allocator.free(n);
            }
            names.deinit();
        }
        var iter = self.variables.iterator();
        while (iter.next()) |entry| {
            const name_copy = try allocator.dupe(u8, entry.key_ptr.*);
            errdefer allocator.free(name_copy);
            try names.append(name_copy);
        }
        return names;
    }

    pub fn hasVariable(self: *const Self, name: []const u8) bool {
        return self.variables.contains(name);
    }

    pub fn applyGlobalQuantumGate(self: *Self, gate: LogicGate, qubit_indices: []const usize, params: ?[]const f64) !void {
        try self.global_logic.applyGate(gate, qubit_indices, params);
    }

    pub fn initializeGlobalQubit(self: *Self, amp_real: f64, amp_imag: f64, phase: f64) !usize {
        return self.global_logic.initializeState(amp_real, amp_imag, phase);
    }

    pub fn measureGlobalQubit(self: *Self, qubit_index: usize) MeasurementResult {
        return self.global_logic.measure(qubit_index);
    }

    pub fn globalQubitCount(self: *const Self) usize {
        return self.global_logic.stateCount();
    }
};

test "HistoryEntry init and deinit" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var entry = try HistoryEntry.init(allocator, .assign, "test_value");
    defer entry.deinit();

    try testing.expectEqual(HistoryEntryType.assign, entry.entry_type);
    try testing.expectEqualStrings("test_value", entry.value);
}

test "ExecutionHistoryEntry operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var entry = try ExecutionHistoryEntry.init(allocator, .create_variable, "test_var");
    defer entry.deinit();

    try testing.expectEqual(ExecutionAction.create_variable, entry.action);
    try testing.expectEqualStrings("test_var", entry.primary_target);

    try entry.addSecondaryTarget("secondary1");
    try entry.addSecondaryTarget("secondary2");
    try testing.expectEqual(@as(usize, 2), entry.secondary_targets.items.len);

    try entry.setOperationType("and");
    try testing.expectEqualStrings("and", entry.operation_type.?);

    entry.setResultInt(42);
    try testing.expectEqual(@as(i32, 42), entry.result_int.?);

    entry.setResultFloat(3.14);
    try testing.expectApproxEqAbs(@as(f64, 3.14), entry.result_float.?, 0.001);
}

test "ZVariable init and deinit" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const variable = try ZVariable.init(allocator, "test_variable");
    defer variable.deinit();

    try testing.expectEqualStrings("test_variable", variable.name);
    try testing.expectEqual(@as(usize, 0), variable.graph.nodeCount());
    try testing.expectEqual(@as(usize, 0), variable.logic.stateCount());
}

test "ZVariable assign and getValue" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const variable = try ZVariable.init(allocator, "test_var");
    defer variable.deinit();

    try variable.assign("hello world");

    try testing.expectEqual(@as(usize, 1), variable.graph.nodeCount());
    try testing.expectEqual(@as(usize, 1), variable.logic.stateCount());

    const value = variable.getValue();
    try testing.expect(value != null);
    try testing.expectEqualStrings("hello world", value.?);

    try testing.expectEqual(@as(usize, 1), variable.historyCount());
}

test "ZVariable transform" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const variable = try ZVariable.initWithValue(allocator, "test_var", "value");
    defer variable.deinit();

    try variable.transform(.HADAMARD, null);

    try testing.expectEqual(@as(usize, 2), variable.historyCount());
}

test "ZVariable measure" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const variable = try ZVariable.initWithValue(allocator, "test_var", "value");
    defer variable.deinit();

    const result = variable.measure();

    try testing.expect(result.result == 0 or result.result == 1);
    try testing.expect(result.probability >= 0.0 and result.probability <= 1.0);
}

test "ZRuntime init and deinit" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const runtime = try ZRuntime.init(allocator);
    defer runtime.deinit();

    try testing.expectEqual(@as(usize, 0), runtime.variableCount());
    try testing.expectEqual(@as(usize, 0), runtime.executionHistoryLength());
}

test "ZRuntime createVariable and getVariable" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const runtime = try ZRuntime.init(allocator);
    defer runtime.deinit();

    const var1 = try runtime.createVariable("var1", "initial");
    try testing.expectEqualStrings("var1", var1.name);

    const retrieved = runtime.getVariable("var1");
    try testing.expect(retrieved != null);
    try testing.expectEqualStrings("var1", retrieved.?.name);

    try testing.expectEqual(@as(usize, 1), runtime.variableCount());
    try testing.expectEqual(@as(usize, 1), runtime.executionHistoryLength());
}

test "ZRuntime deleteVariable" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const runtime = try ZRuntime.init(allocator);
    defer runtime.deinit();

    _ = try runtime.createVariable("test_var", null);
    try testing.expectEqual(@as(usize, 1), runtime.variableCount());

    const deleted = runtime.deleteVariable("test_var");
    try testing.expect(deleted);
    try testing.expectEqual(@as(usize, 0), runtime.variableCount());

    const not_deleted = runtime.deleteVariable("nonexistent");
    try testing.expect(!not_deleted);
}

test "ZRuntime relationalOperation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const runtime = try ZRuntime.init(allocator);
    defer runtime.deinit();

    _ = try runtime.createVariable("a", "value_a");
    _ = try runtime.createVariable("b", "value_b");

    const result = try runtime.relationalOperation("a", "b", .op_and);
    try testing.expect(result != null);

    try testing.expectEqual(@as(usize, 3), runtime.variableCount());
}

test "ZRuntime entangleVariables" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const runtime = try ZRuntime.init(allocator);
    defer runtime.deinit();

    _ = try runtime.createVariable("x", "value_x");
    _ = try runtime.createVariable("y", "value_y");

    const success = try runtime.entangleVariables("x", "y");
    try testing.expect(success);

    const fail = try runtime.entangleVariables("x", "nonexistent");
    try testing.expect(!fail);
}

test "ZRuntime measureVariable" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const runtime = try ZRuntime.init(allocator);
    defer runtime.deinit();

    _ = try runtime.createVariable("quantum_var", "superposition");

    const result = try runtime.measureVariable("quantum_var");
    try testing.expect(result != null);
    try testing.expect(result.?.result == 0 or result.?.result == 1);

    const null_result = try runtime.measureVariable("nonexistent");
    try testing.expect(null_result == null);
}

test "ZRuntime applyFractalTransform" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const runtime = try ZRuntime.init(allocator);
    defer runtime.deinit();

    _ = try runtime.createVariable("fractal_var", "data");

    const success = try runtime.applyFractalTransform("fractal_var", 3);
    try testing.expect(success);

    const fail = try runtime.applyFractalTransform("nonexistent", 3);
    try testing.expect(!fail);
}

test "ZRuntime getSystemState" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const runtime = try ZRuntime.init(allocator);
    defer runtime.deinit();

    _ = try runtime.createVariable("var1", "value1");
    _ = try runtime.createVariable("var2", "value2");

    var state = try runtime.getSystemState();
    defer state.deinit();

    try testing.expectEqual(@as(usize, 2), state.variable_count);
    try testing.expectEqual(@as(usize, 2), state.variables.items.len);
    try testing.expectEqual(@as(usize, 2), runtime.executionHistoryLength());
}

test "ZRuntime computeRelationalExpression" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const runtime = try ZRuntime.init(allocator);
    defer runtime.deinit();

    _ = try runtime.createVariable("alpha", "val_alpha");
    _ = try runtime.createVariable("beta", "val_beta");

    const result = try runtime.computeRelationalExpression("alpha AND beta");
    if (result) |r| {
        defer allocator.free(r);
    }

    const entangle_result = try runtime.computeRelationalExpression("alpha ENTANGLE beta");
    if (entangle_result) |r| {
        defer allocator.free(r);
        try testing.expectEqualStrings("true", r);
    }
}

test "ZRuntime global quantum operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const runtime = try ZRuntime.init(allocator);
    defer runtime.deinit();

    const idx = try runtime.initializeGlobalQubit(1.0, 0.0, 0.0);
    try testing.expectEqual(@as(usize, 0), idx);
    try testing.expectEqual(@as(usize, 1), runtime.globalQubitCount());

    const indices = [_]usize{0};
    try runtime.applyGlobalQuantumGate(.HADAMARD, &indices, null);

    const result = runtime.measureGlobalQubit(0);
    try testing.expect(result.result == 0 or result.result == 1);
}

test "ZRuntime reset" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const runtime = try ZRuntime.init(allocator);
    defer runtime.deinit();

    _ = try runtime.createVariable("var1", "value1");
    _ = try runtime.createVariable("var2", "value2");

    try testing.expectEqual(@as(usize, 2), runtime.variableCount());

    try runtime.reset();

    try testing.expectEqual(@as(usize, 0), runtime.variableCount());
    try testing.expectEqual(@as(usize, 0), runtime.executionHistoryLength());
}

test "RelationalOperationType conversions" {
    const testing = std.testing;

    try testing.expectEqualStrings("and", RelationalOperationType.op_and.toString());
    try testing.expectEqualStrings("or", RelationalOperationType.op_or.toString());
    try testing.expectEqualStrings("xor", RelationalOperationType.op_xor.toString());

    try testing.expectEqual(RelationalOperationType.op_and, RelationalOperationType.fromString("AND").?);
    try testing.expectEqual(RelationalOperationType.op_or, RelationalOperationType.fromString("or").?);
    try testing.expectEqual(RelationalOperationType.op_xor, RelationalOperationType.fromString("XOR").?);

    try testing.expect(RelationalOperationType.fromString("invalid") == null);

    try testing.expectEqual(LogicGate.RELATIONAL_AND, RelationalOperationType.op_and.toGate().?);
    try testing.expectEqual(LogicGate.RELATIONAL_OR, RelationalOperationType.op_or.toGate().?);
    try testing.expectEqual(LogicGate.RELATIONAL_XOR, RelationalOperationType.op_xor.toGate().?);
    try testing.expect(RelationalOperationType.op_entangle.toGate() == null);
}

test "ZVariable relateTo" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const var1 = try ZVariable.initWithValue(allocator, "var1", "value1");
    defer var1.deinit();

    const var2 = try ZVariable.initWithValue(allocator, "var2", "value2");
    defer var2.deinit();

    try var1.relateTo(var2, .coherent);

    try testing.expect(var1.graph.edgeCount() > 0);
}

test "ZRuntime executeQuantumCircuit" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const runtime = try ZRuntime.init(allocator);
    defer runtime.deinit();

    _ = try runtime.createVariable("circuit_var", "initial");

    const hadamard_indices = [_]usize{0};
    const circuit = [_]GateSpec{
        .{ .gate_name = "hadamard", .indices = &hadamard_indices, .params = null },
    };

    const success = try runtime.executeQuantumCircuit("circuit_var", &circuit);
    try testing.expect(success);

    const fail = try runtime.executeQuantumCircuit("nonexistent", &circuit);
    try testing.expect(!fail);
}

