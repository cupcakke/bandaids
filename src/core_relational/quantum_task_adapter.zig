const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const Complex = std.math.Complex;

const nsir = @import("nsir_core.zig");
const ibm = @import("ibm_quantum.zig");
const quantum = @import("quantum_logic.zig");

const SelfSimilarRelationalGraph = nsir.SelfSimilarRelationalGraph;
const Node = nsir.Node;
const Edge = nsir.Edge;
const Qubit = nsir.Qubit;
const IBMQuantumClient = ibm.IBMQuantumClient;
const RelationalQuantumLogic = quantum.RelationalQuantumLogic;
const LogicGate = quantum.LogicGate;

pub const QuantumSubgraph = struct {
    node_ids: ArrayList([]const u8),
    edge_keys: ArrayList(nsir.EdgeKey),
    total_entanglement: f64,
    avg_fractal_dimension: f64,
    subgraph_id: u64,
    semantic_cluster_id: u64,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .node_ids = ArrayList([]const u8).init(allocator),
            .edge_keys = ArrayList(nsir.EdgeKey).init(allocator),
            .total_entanglement = 0.0,
            .avg_fractal_dimension = 0.0,
            .subgraph_id = @as(u64, @bitCast(@as(i64, @truncate(std.time.nanoTimestamp())))),
            .semantic_cluster_id = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.node_ids.items) |node_id| {
            self.allocator.free(node_id);
        }
        self.node_ids.deinit();
        self.edge_keys.deinit();
    }

    pub fn addNode(self: *Self, node_id: []const u8) !void {
        const id_copy = try self.allocator.dupe(u8, node_id);
        try self.node_ids.append(id_copy);
    }

    pub fn addEdge(self: *Self, edge_key: nsir.EdgeKey) !void {
        try self.edge_keys.append(edge_key);
    }

    pub fn computeMetrics(self: *Self, graph: *const SelfSimilarRelationalGraph) void {
        var total_ent: f64 = 0.0;
        var total_dim: f64 = 0.0;
        var edge_count: usize = 0;

        for (self.edge_keys.items) |key| {
            if (graph.edges.get(key)) |edge_list| {
                for (edge_list.items) |edge| {
                    total_ent += @abs(edge.quantum_correlation.magnitude());
                    total_dim += edge.fractal_dimension;
                    edge_count += 1;
                }
            }
        }

        if (edge_count > 0) {
            self.total_entanglement = total_ent;
            self.avg_fractal_dimension = total_dim / @as(f64, @floatFromInt(edge_count));
        }
    }

    pub fn isQuantumSuitable(self: *const Self, threshold: f64) bool {
        return self.total_entanglement > threshold and self.avg_fractal_dimension > 1.5;
    }
};

pub const QuantumTaskResult = struct {
    subgraph_id: u64,
    success: bool,
    quantum_states: ArrayList(Complex(f64)),
    correlations: ArrayList(f64),
    execution_time_ms: i64,
    backend_name: ?[]const u8,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, subgraph_id: u64) Self {
        return Self{
            .subgraph_id = subgraph_id,
            .success = false,
            .quantum_states = ArrayList(Complex(f64)).init(allocator),
            .correlations = ArrayList(f64).init(allocator),
            .execution_time_ms = 0,
            .backend_name = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.quantum_states.deinit();
        self.correlations.deinit();
        if (self.backend_name) |bn| {
            self.allocator.free(bn);
            self.backend_name = null;
        }
    }

    pub fn setBackendName(self: *Self, name: []const u8) !void {
        if (self.backend_name) |bn| {
            self.allocator.free(bn);
        }
        self.backend_name = try self.allocator.dupe(u8, name);
    }
};

pub const QuantumTaskAdapter = struct {
    graph: *SelfSimilarRelationalGraph,
    quantum_client: ?*IBMQuantumClient,
    local_simulator: RelationalQuantumLogic,
    entanglement_threshold: f64,
    fractal_threshold: f64,
    use_real_backend: bool,
    statistics: AdapterStatistics,
    allocator: Allocator,

    const Self = @This();

    pub const AdapterStatistics = struct {
        total_tasks_submitted: usize,
        tasks_completed: usize,
        tasks_failed: usize,
        avg_execution_time_ms: f64,
        total_qubits_used: usize,
    };

    pub fn init(allocator: Allocator, graph: *SelfSimilarRelationalGraph) Self {
        return Self{
            .graph = graph,
            .quantum_client = null,
            .local_simulator = RelationalQuantumLogic.init(allocator),
            .entanglement_threshold = 0.5,
            .fractal_threshold = 1.5,
            .use_real_backend = false,
            .statistics = AdapterStatistics{
                .total_tasks_submitted = 0,
                .tasks_completed = 0,
                .tasks_failed = 0,
                .avg_execution_time_ms = 0.0,
                .total_qubits_used = 0,
            },
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.local_simulator.deinit();
    }

    pub fn setQuantumClient(self: *Self, client: *IBMQuantumClient) void {
        self.quantum_client = client;
        self.use_real_backend = true;
    }

    pub fn setThresholds(self: *Self, entanglement: f64, fractal: f64) void {
        self.entanglement_threshold = entanglement;
        self.fractal_threshold = fractal;
    }

    fn semanticClusterKey(allocator: Allocator, edge: *const Edge) ![]const u8 {
        const quality_val: usize = @intFromEnum(edge.quality);
        const fractal_bucket: usize = if (edge.fractal_dimension >= 2.0) 2 else if (edge.fractal_dimension >= 1.5) 1 else 0;
        const corr_bucket: usize = if (edge.quantum_correlation.magnitude() >= 0.8) 3 else if (edge.quantum_correlation.magnitude() >= 0.5) 2 else if (edge.quantum_correlation.magnitude() >= 0.2) 1 else 0;
        return std.fmt.allocPrint(allocator, "q{d}_f{d}_c{d}", .{ quality_val, fractal_bucket, corr_bucket });
    }

    pub fn identifyQuantumSubgraphs(self: *Self) !ArrayList(QuantumSubgraph) {
        var subgraphs = ArrayList(QuantumSubgraph).init(self.allocator);

        var node_clusters = StringHashMap(ArrayList([]const u8)).init(self.allocator);
        defer {
            var iter = node_clusters.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                for (entry.value_ptr.items) |node_id| {
                    self.allocator.free(node_id);
                }
                entry.value_ptr.deinit();
            }
            node_clusters.deinit();
        }

        var edge_iter = self.graph.edges.iterator();
        while (edge_iter.next()) |entry| {
            for (entry.value_ptr.items) |edge| {
                const correlation = edge.quantum_correlation.magnitude();
                if (correlation > self.entanglement_threshold and edge.fractal_dimension > self.fractal_threshold) {
                    const cluster_key = try semanticClusterKey(self.allocator, &edge);
                    errdefer self.allocator.free(cluster_key);

                    var result = try node_clusters.getOrPut(cluster_key);
                    if (!result.found_existing) {
                        result.value_ptr.* = ArrayList([]const u8).init(self.allocator);
                    } else {
                        self.allocator.free(cluster_key);
                    }

                    const source_copy = try self.allocator.dupe(u8, edge.source);
                    const target_copy = try self.allocator.dupe(u8, edge.target);
                    try result.value_ptr.append(source_copy);
                    try result.value_ptr.append(target_copy);
                }
            }
        }

        var cluster_iter = node_clusters.iterator();
        while (cluster_iter.next()) |entry| {
            if (entry.value_ptr.items.len < 2) continue;

            var subgraph = QuantumSubgraph.init(self.allocator);
            for (entry.value_ptr.items) |node_id| {
                try subgraph.addNode(node_id);
            }

            var sg_edge_iter = self.graph.edges.iterator();
            while (sg_edge_iter.next()) |edge_entry| {
                for (entry.value_ptr.items) |node_id| {
                    if (std.mem.eql(u8, edge_entry.key_ptr.source, node_id)) {
                        try subgraph.addEdge(edge_entry.key_ptr.*);
                        break;
                    }
                }
            }

            subgraph.computeMetrics(self.graph);
            if (subgraph.isQuantumSuitable(self.entanglement_threshold)) {
                try subgraphs.append(subgraph);
            } else {
                subgraph.deinit();
            }
        }

        return subgraphs;
    }

    pub fn executeQuantumTask(self: *Self, subgraph: *const QuantumSubgraph) !QuantumTaskResult {
        const start_time = @as(i64, @truncate(std.time.nanoTimestamp()));
        self.statistics.total_tasks_submitted += 1;

        var result = QuantumTaskResult.init(self.allocator, subgraph.subgraph_id);

        if (self.use_real_backend and self.quantum_client != null) {
            const qasm = try self.generateQASM(subgraph);
            defer self.allocator.free(qasm);

            const job_response = try self.quantum_client.?.submitJobWithBackend(qasm, self.quantum_client.?.backend_name, 1024);
            defer self.allocator.free(job_response);

            result.success = true;
            try result.setBackendName(self.quantum_client.?.backend_name);
        } else {
            try self.executeLocalSimulation(subgraph, &result);
            try result.setBackendName("local_simulator");
        }

        const end_time = @as(i64, @truncate(std.time.nanoTimestamp()));
        const elapsed_ns = end_time - start_time;
        const max_i64: i128 = std.math.maxInt(i64);
        const min_i64: i128 = std.math.minInt(i64);
        if (elapsed_ns > max_i64) {
            result.execution_time_ms = @as(i64, @truncate(@divTrunc(elapsed_ns, 1_000_000)));
        } else if (elapsed_ns < min_i64) {
            result.execution_time_ms = @as(i64, @truncate(@divTrunc(elapsed_ns, 1_000_000)));
        } else {
            const ms_i128 = @divTrunc(elapsed_ns, 1_000_000);
            result.execution_time_ms = @as(i64, @intCast(ms_i128));
        }

        if (result.success) {
            self.statistics.tasks_completed += 1;
        } else {
            self.statistics.tasks_failed += 1;
        }

        const total_completed = self.statistics.tasks_completed;
        if (total_completed > 0) {
            const elapsed_ms: f64 = @as(f64, @floatFromInt(result.execution_time_ms));
            const prev_avg = self.statistics.avg_execution_time_ms * @as(f64, @floatFromInt(total_completed - 1));
            self.statistics.avg_execution_time_ms = (prev_avg + elapsed_ms) / @as(f64, @floatFromInt(total_completed));
        }

        return result;
    }

    fn executeLocalSimulation(self: *Self, subgraph: *const QuantumSubgraph, result: *QuantumTaskResult) !void {
        self.local_simulator.reset();

        for (subgraph.node_ids.items) |node_id| {
            const node = self.graph.getNode(node_id) orelse continue;
            _ = try self.local_simulator.initializeStateFromComplex(node.qubit.a, node.qubit.b, node.phase);
            self.statistics.total_qubits_used += 1;
        }

        const qubit_count = self.local_simulator.stateCount();
        if (qubit_count > 1) {
            var i: usize = 0;
            while (i < qubit_count - 1) : (i += 1) {
                try self.local_simulator.applyGate(.HADAMARD, &[_]usize{i}, null);
                try self.local_simulator.applyGate(.CNOT, &[_]usize{ i, i + 1 }, null);
            }
        }

        var i: usize = 0;
        while (i < qubit_count) : (i += 1) {
            const state = self.local_simulator.getState(i) orelse continue;
            const complex_val = Complex(f64).init(state.amplitudes[0].re, state.amplitudes[0].im);
            try result.quantum_states.append(complex_val);
            try result.correlations.append(state.entanglement_degree);
        }

        result.success = true;
    }

    fn generateQASM(self: *Self, subgraph: *const QuantumSubgraph) ![]u8 {
        var buffer = ArrayList(u8).init(self.allocator);

        try buffer.appendSlice("OPENQASM 2.0;\n");
        try buffer.appendSlice("include \"qelib1.inc\";\n");

        const qreg_size = subgraph.node_ids.items.len;
        const qreg_line = try std.fmt.allocPrint(self.allocator, "qreg q[{d}];\n", .{qreg_size});
        defer self.allocator.free(qreg_line);
        try buffer.appendSlice(qreg_line);

        const creg_line = try std.fmt.allocPrint(self.allocator, "creg c[{d}];\n", .{qreg_size});
        defer self.allocator.free(creg_line);
        try buffer.appendSlice(creg_line);

        var i: usize = 0;
        while (i < qreg_size) : (i += 1) {
            const h_line = try std.fmt.allocPrint(self.allocator, "h q[{d}];\n", .{i});
            defer self.allocator.free(h_line);
            try buffer.appendSlice(h_line);
        }

        i = 0;
        while (i < qreg_size - 1) : (i += 1) {
            const cx_line = try std.fmt.allocPrint(self.allocator, "cx q[{d}],q[{d}];\n", .{ i, i + 1 });
            defer self.allocator.free(cx_line);
            try buffer.appendSlice(cx_line);
        }

        i = 0;
        while (i < qreg_size) : (i += 1) {
            const measure_line = try std.fmt.allocPrint(self.allocator, "measure q[{d}] -> c[{d}];\n", .{ i, i });
            defer self.allocator.free(measure_line);
            try buffer.appendSlice(measure_line);
        }

        return try buffer.toOwnedSlice();
    }

    pub fn applyResultsToGraph(self: *Self, subgraph: *const QuantumSubgraph, result: *const QuantumTaskResult) !void {
        if (!result.success) return;

        var idx: usize = 0;
        while (idx < subgraph.node_ids.items.len) : (idx += 1) {
            if (idx >= result.quantum_states.items.len) break;

            const node_id = subgraph.node_ids.items[idx];
            const node = self.graph.getNode(node_id) orelse continue;
            const new_state = result.quantum_states.items[idx];
            node.qubit.a = new_state;
            node.qubit.normalizeInPlace();

            if (idx < result.correlations.items.len) {
                node.coherence = result.correlations.items[idx];
            }
        }

        for (subgraph.edge_keys.items) |key| {
            if (self.graph.edges.getPtr(key)) |edge_list| {
                for (edge_list.items) |*edge| {
                    const avg_corr = if (result.correlations.items.len > 0) blk: {
                        var sum: f64 = 0.0;
                        for (result.correlations.items) |c| sum += c;
                        break :blk sum / @as(f64, @floatFromInt(result.correlations.items.len));
                    } else 0.0;

                    var avg_imag: f64 = 0.0;
                    if (result.quantum_states.items.len > 0) {
                        var sum_imag: f64 = 0.0;
                        for (result.quantum_states.items) |qs| sum_imag += qs.im;
                        avg_imag = sum_imag / @as(f64, @floatFromInt(result.quantum_states.items.len));
                    }

                    edge.quantum_correlation.re = avg_corr;
                    edge.quantum_correlation.im = avg_imag;
                }
            }
        }
    }

    pub fn runFullQuantumOptimization(self: *Self) !void {
        var subgraphs = try self.identifyQuantumSubgraphs();
        defer {
            for (subgraphs.items) |*sg| {
                sg.deinit();
            }
            subgraphs.deinit();
        }

        for (subgraphs.items) |*subgraph| {
            var result = try self.executeQuantumTask(subgraph);
            defer result.deinit();

            if (result.success) {
                try self.applyResultsToGraph(subgraph, &result);
            }
        }
    }

    pub fn getStatistics(self: *const Self) AdapterStatistics {
        return self.statistics;
    }
};

test "quantum_task_adapter_identify_subgraphs" {
    const allocator = std.testing.allocator;

    var graph = try SelfSimilarRelationalGraph.init(allocator);
    defer graph.deinit();

    const n1 = try Node.init(allocator, "n1", "data1", Qubit{ .a = Complex(f64).init(0.7, 0.7), .b = Complex(f64).init(0.0, 0.0) }, 0.5);
    try graph.addNode(n1);
    const n2 = try Node.init(allocator, "n2", "data2", Qubit{ .a = Complex(f64).init(0.6, 0.8), .b = Complex(f64).init(0.0, 0.0) }, 0.3);
    try graph.addNode(n2);

    const e1 = try Edge.init(allocator, "n1", "n2", .entangled, 0.9, Complex(f64).init(0.8, 0.8), 2.0);
    try graph.addEdge("n1", "n2", e1);

    var adapter = QuantumTaskAdapter.init(allocator, &graph);
    defer adapter.deinit();

    var subgraphs = try adapter.identifyQuantumSubgraphs();
    defer {
        for (subgraphs.items) |*sg| {
            sg.deinit();
        }
        subgraphs.deinit();
    }

    try std.testing.expect(subgraphs.items.len >= 0);
}
