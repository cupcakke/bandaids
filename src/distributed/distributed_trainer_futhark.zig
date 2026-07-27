const std = @import("std");
const GPUCoordinator = @import("gpu_coordinator.zig").GPUCoordinator;
const MGT = @import("../tokenizer/mgt.zig").MGT;
const accel = @import("../hw/accel/accel_interface.zig");
const RSFAccelerator = accel.RSFAccelerator;
const FutharkArray2DF16 = accel.FutharkArray2DF16;
const FutharkArray1DF16 = accel.FutharkArray1DF16;
const FutharkArray3DF16 = accel.FutharkArray3DF16;
const PinnedMemory = accel.PinnedMemory;
const futhark = @import("../hw/accel/futhark_bindings.zig");
const core_relational = @import("../core_relational/mod.zig");
const CREVPipeline = core_relational.CREVPipeline;
const ChaosCoreKernel = core_relational.ChaosCoreKernel;
const nsir = core_relational.nsir_core;
const SelfSimilarRelationalGraph = core_relational.SelfSimilarRelationalGraph;
const EntangledStochasticSymmetryOptimizer = core_relational.EntangledStochasticSymmetryOptimizer;
const SurpriseMemoryManager = core_relational.SurpriseMemoryManager;
const TemporalGraph = core_relational.TemporalGraph;
const QuantumState = core_relational.QuantumState;
const ReasoningOrchestrator = core_relational.ReasoningOrchestrator;
const SignalPropagationEngine = core_relational.SignalPropagationEngine;
const ZRuntime = core_relational.ZRuntime;
const RelationalGraphProcessingUnit = core_relational.RelationalGraphProcessingUnit;
const FNDSManager = core_relational.FNDSManager;
const VPU = core_relational.VPU;
const PatternLocation = core_relational.PatternLocation;
const Tensor = @import("../core/tensor.zig").Tensor;
const sfd = @import("../optimizer/sfd.zig");

const _use_futhark_2d = FutharkArray2DF16;
const _use_futhark_1d = FutharkArray1DF16;
const _use_tensor = Tensor;

pub const CHECKPOINT_MAGIC: [8]u8 = .{ 'J', 'A', 'I', 'D', 'E', 'C', 'K', 'P' };
pub const CHECKPOINT_TRAILER: u32 = 0xDEADBEEF;

pub const MGTLanguage = @typeInfo(@TypeOf(MGT.init)).@"fn".params[4].type.?;

pub const TokenizerFactory = *const fn (
    allocator: std.mem.Allocator,
    vocabulary: []const []const u8,
    anchors: []const []const u8,
    max_merges: usize,
) anyerror!MGT;

const OwnedTokenList = struct {
    allocator: std.mem.Allocator,
    items: [][]const u8,

    fn deinit(self: *OwnedTokenList) void {
        for (self.items) |item| self.allocator.free(item);
        self.allocator.free(self.items);
    }
};

fn appendOwnedToken(
    allocator: std.mem.Allocator,
    items: *std.ArrayList([]const u8),
    token: []const u8,
) !void {
    if (token.len == 0) return TrainerError.InvalidTokenizerData;
    const owned = try allocator.dupe(u8, token);
    items.append(owned) catch |err| {
        allocator.free(owned);
        return err;
    };
}

fn loadTokenList(
    allocator: std.mem.Allocator,
    path: []const u8,
    maximum_size: usize,
) !OwnedTokenList {
    if (path.len == 0 or maximum_size == 0) return TrainerError.InvalidTokenizerData;
    const file = if (std.fs.path.isAbsolute(path))
        try std.fs.openFileAbsolute(path, .{ .mode = .read_only })
    else
        try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer file.close();
    const length_u64 = try file.getEndPos();
    const length = std.math.cast(usize, length_u64) orelse return TrainerError.FileTooLarge;
    if (length == 0 or length > maximum_size) return TrainerError.InvalidTokenizerData;
    const data = try allocator.alloc(u8, length);
    defer allocator.free(data);
    try file.reader().readNoEof(data);

    var items = std.ArrayList([]const u8).init(allocator);
    errdefer {
        for (items.items) |item| allocator.free(item);
        items.deinit();
    }

    var content = std.mem.trimLeft(u8, data, " \t\r\n");
    if (content.len >= 3 and std.mem.eql(u8, content[0..3], "\xEF\xBB\xBF")) {
        content = std.mem.trimLeft(u8, content[3..], " \t\r\n");
    }
    if (content.len == 0) return TrainerError.InvalidTokenizerData;

    if (content[0] == '[') {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            content,
            .{ .allocate = .alloc_always },
        );
        defer parsed.deinit();
        switch (parsed.value) {
            .array => |array| {
                for (array.items) |value| {
                    switch (value) {
                        .string => |token| try appendOwnedToken(allocator, &items, token),
                        else => return TrainerError.InvalidTokenizerData,
                    }
                }
            },
            else => return TrainerError.InvalidTokenizerData,
        }
    } else {
        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |raw_line| {
            const line = if (raw_line.len > 0 and raw_line[raw_line.len - 1] == '\r')
                raw_line[0 .. raw_line.len - 1]
            else
                raw_line;
            if (line.len == 0) continue;
            try appendOwnedToken(allocator, &items, line);
        }
    }

    if (items.items.len == 0) return TrainerError.InvalidTokenizerData;
    return OwnedTokenList{
        .allocator = allocator,
        .items = try items.toOwnedSlice(),
    };
}

pub const TrainerConfig = struct {
    learning_rate: f32 = 0.001,
    momentum: f32 = 0.0,
    max_line_size: usize = 10 * 1024 * 1024,
    max_tokenizer_file_size: usize = 1024 * 1024 * 1024,
    checkpoint_version: u32 = 8,
    reasoning_cycles: usize = 1,
    fnds_max_depth: usize = 6,
    fnds_branching: usize = 4,
    fnds_kg_max_depth: usize = 4,
    fnds_kg_branching: usize = 3,
    training_fnds_index_name: []const u8 = "distributed_training_tokens",
    knowledge_fnds_index_name: []const u8 = "knowledge_graph_patterns",
    embedding_seed: u64 = 42,
    spectral_iterations: usize = 5,
    gradient_clip_norm: f32 = 1.0,
    use_normalized_gradient_flow: bool = true,
    default_max_seq_len: usize = 256,
    temporal_sequence_tick: i64 = 1,
    training_variable_name: []const u8 = "distributed_training_session",
    max_id_length: usize = 1 << 20,
    max_edge_group_count: usize = 1 << 24,
    max_node_data_length: usize = 1 << 24,
    max_node_count: usize = 1 << 20,
    max_distributed_integer: u64 = 16_777_216,
    max_local_batch_size: usize = 1 << 20,
    tokenizer_vocabulary: []const []const u8 = &.{},
    tokenizer_anchors: []const []const u8 = &.{},
    tokenizer_vocabulary_path: ?[]const u8 = null,
    tokenizer_anchors_path: ?[]const u8 = null,
    tokenizer_max_merges: usize = 50_000,
    tokenizer_language: ?MGTLanguage = null,
    tokenizer_factory: ?TokenizerFactory = null,
    esso_initial_temperature: f64 = 1.0,
    esso_cooling_rate: f64 = 0.995,
    esso_max_iterations: usize = 1000,
    relational_gpu_rows: usize = 4,
    relational_gpu_columns: usize = 4,
    relational_pass_interval: usize = 50,
    max_knowledge_graph_input: usize = 64 * 1024 * 1024,
    reconstruction_alpha: f32 = 0.3,
    phase_a_steps: u64 = 500,
    phase_b_steps: u64 = 2000,
    shuffle_target_control: bool = false,
    target_source_frozen: bool = true,
    spectral_target_norm: f32 = 0.9,
    spectral_depth_compensation: bool = true,
};

pub const TrainerComponents = struct {
    tokenizer: MGT,
};

pub const TrainerError = error{
    InvalidModelDim,
    InvalidNumLayers,
    InvalidBatchSize,
    InvalidWorldSize,
    InvalidRank,
    InvalidMaxLineSize,
    InvalidCheckpointVersion,
    InvalidLearningRate,
    InvalidMomentum,
    InvalidHyperparameterAfterCast,
    InvalidWeightsShape,
    InvalidWeightValue,
    InvalidClipRange,
    InvalidLoss,
    InvalidPinnedMemorySize,
    IndexOutOfBounds,
    TokenIndexOutOfRange,
    CheckpointVersionMismatch,
    CheckpointMagicMismatch,
    CheckpointCorrupted,
    ModelDimMismatch,
    NumLayersMismatch,
    VocabSizeMismatch,
    EmptyDataset,
    InvalidDatasetPartition,
    InvalidEnvironmentValue,
    ValueOverflow,
    ConvertPrecisionLoss,
    AllocationFailed,
    InvalidQualityByte,
    NodeIdTooLong,
    NodeDataTooLong,
    EdgeCountTooLarge,
    DistributedIntegerPrecisionExceeded,
    InvalidDistributedInteger,
    InvalidReductionWeight,
    InvalidFloat16Value,
    InvalidGradient,
    InvalidQuantumState,
    InvalidEmbeddingWeight,
    InvalidEmbeddingShape,
    InvalidEdgeWeight,
    InvalidGraphIdentifier,
    InvalidTokenizerData,
    InvalidTokenizerConfiguration,
    InvalidOptimizerConfiguration,
    InvalidRelationalGPUConfiguration,
    InvalidTemporalConfiguration,
    InvalidSpectralState,
    InvalidCheckpointEmbeddingFlag,
    TrailingCheckpointData,
    TimestampOutOfRange,
    FileTooLarge,
    FutharkContextUnavailable,
    FutharkForwardFailed,
    FutharkTransformFailed,
    FutharkBackwardTransformFailed,
    FutharkGradientFailed,
    FutharkFullGradientFailed,
    FutharkProjectionFailed,
    FutharkGradientCopyFailed,
    EmptyKnowledgeGraphInput,
    KnowledgeGraphInputTooLarge,
    DistributedConfigMismatch,
    InvalidRelationalPassInterval,
    CheckpointSaveFailed,
};

fn createConfiguredTokenizer(
    allocator: std.mem.Allocator,
    config: TrainerConfig,
) !MGT {
    if (config.tokenizer_max_merges == 0 or config.max_tokenizer_file_size == 0) return TrainerError.InvalidTokenizerConfiguration;

    var vocabulary_path_owned: ?[]u8 = null;
    defer if (vocabulary_path_owned) |path| allocator.free(path);
    var anchors_path_owned: ?[]u8 = null;
    defer if (anchors_path_owned) |path| allocator.free(path);

    var owned_vocabulary: ?OwnedTokenList = null;
    defer if (owned_vocabulary) |*list| list.deinit();
    var owned_anchors: ?OwnedTokenList = null;
    defer if (owned_anchors) |*list| list.deinit();

    var vocabulary = config.tokenizer_vocabulary;
    if (vocabulary.len == 0) {
        const vocabulary_path = config.tokenizer_vocabulary_path orelse blk: {
            vocabulary_path_owned = std.process.getEnvVarOwned(allocator, "JAIDE_TOKENIZER_VOCAB") catch |err| switch (err) {
                error.EnvironmentVariableNotFound => return TrainerError.InvalidTokenizerConfiguration,
                else => return err,
            };
            break :blk vocabulary_path_owned.?;
        };
        owned_vocabulary = try loadTokenList(allocator, vocabulary_path, config.max_tokenizer_file_size);
        vocabulary = owned_vocabulary.?.items;
    }
    if (vocabulary.len == 0) return TrainerError.InvalidTokenizerData;

    var anchors = config.tokenizer_anchors;
    if (anchors.len == 0) {
        const configured_anchor_path = config.tokenizer_anchors_path;
        const anchor_path_opt: ?[]const u8 = if (configured_anchor_path) |path|
            path
        else blk: {
            anchors_path_owned = std.process.getEnvVarOwned(allocator, "JAIDE_TOKENIZER_ANCHORS") catch |err| switch (err) {
                error.EnvironmentVariableNotFound => break :blk null,
                else => return err,
            };
            break :blk anchors_path_owned.?;
        };
        if (anchor_path_opt) |anchor_path| {
            owned_anchors = try loadTokenList(allocator, anchor_path, config.max_tokenizer_file_size);
            anchors = owned_anchors.?.items;
        }
    }

    const max_vocab_size: ?u32 = if (config.tokenizer_max_merges > 0)
        std.math.cast(u32, config.tokenizer_max_merges) orelse return TrainerError.InvalidTokenizerConfiguration
    else
        null;

    if (config.tokenizer_factory) |tokenizer_factory| {
        return tokenizer_factory(allocator, vocabulary, anchors, config.tokenizer_max_merges);
    }

    var language_name_owned: ?[]u8 = null;
    defer if (language_name_owned) |name| allocator.free(name);
    const language = config.tokenizer_language orelse blk: {
        language_name_owned = std.process.getEnvVarOwned(allocator, "JAIDE_TOKENIZER_LANGUAGE") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => return TrainerError.InvalidTokenizerConfiguration,
            else => return err,
        };
        break :blk std.meta.stringToEnum(MGTLanguage, language_name_owned.?) orelse return TrainerError.InvalidTokenizerConfiguration;
    };
    return MGT.init(allocator, vocabulary, anchors, max_vocab_size, language);
}

const LayerSnapshot = struct {
    weights_s: []f16,
    weights_t: []f16,
    velocity_s: []f16,
    velocity_t: []f16,
};

fn CrcTrackingWriter(comptime WriterType: type) type {
    return struct {
        inner: WriterType,
        crc: std.hash.Crc32,

        const Self = @This();

        pub fn write(self: *Self, bytes: []const u8) WriterType.Error!usize {
            const written = try self.inner.write(bytes);
            self.crc.update(bytes[0..written]);
            return written;
        }

        pub const Writer = std.io.Writer(*Self, WriterType.Error, write);

        pub fn writer(self: *Self) Writer {
            return .{ .context = self };
        }
    };
}

fn CrcTrackingReader(comptime ReaderType: type) type {
    return struct {
        inner: ReaderType,
        crc: std.hash.Crc32,

        const Self = @This();

        pub fn read(self: *Self, buffer: []u8) ReaderType.Error!usize {
            const nread = try self.inner.read(buffer);
            self.crc.update(buffer[0..nread]);
            return nread;
        }

        pub const Reader = std.io.Reader(*Self, ReaderType.Error, read);

        pub fn reader(self: *Self) Reader {
            return .{ .context = self };
        }
    };
}

pub const DistributedTrainerFuthark = struct {
    allocator: std.mem.Allocator,
    coordinator: *GPUCoordinator,
    tokenizer: MGT,
    accelerator: *RSFAccelerator,
    model_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    local_batch_size: usize,
    global_step: u64,
    learning_rate: f32,
    momentum: f32,
    config: TrainerConfig,
    gpu_embedding: ?accel.EmbeddingAccelerator,
    crev_pipeline: CREVPipeline,
    crev_kernel: *ChaosCoreKernel,
    nsir_graph: *SelfSimilarRelationalGraph,
    knowledge_nsir_graph: *SelfSimilarRelationalGraph,
    esso: EntangledStochasticSymmetryOptimizer,
    surprise_memory: SurpriseMemoryManager,
    temporal_graph: TemporalGraph,
    signal_engine: *SignalPropagationEngine,
    z_runtime: *ZRuntime,
    r_gpu: RelationalGraphProcessingUnit,
    fnds_manager: FNDSManager,
    vpu: VPU,
    spectral_normalizer: sfd.SpectralNormalizer,
    gpu_spectral_u: ?accel.FutharkArray1DF32,
    gpu_spectral_v: ?accel.FutharkArray1DF32,
    training_fnds_tree_id: ?[32]u8,
    training_fnds_index_id: ?[]u8,
    knowledge_fnds_tree_id: ?[32]u8,
    knowledge_fnds_index_id: ?[]u8,
    knowledge_graph_nonce: [32]u8,
    temporal_logical_time: i64,
    training_variable_created: bool,
    vpu_lr_scale: f32,
    target_source: ?accel.EmbeddingAccelerator,
    shuffle_control_state: u64,

    pub fn init(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        local_batch_size: usize,
    ) !DistributedTrainerFuthark {
        return initWithConfig(allocator, coordinator, model_dim, 1, local_batch_size, .{});
    }

    pub fn initWithConfig(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        local_batch_size: usize,
        config: TrainerConfig,
    ) !DistributedTrainerFuthark {
        const tokenizer = try createConfiguredTokenizer(allocator, config);
        var tokenizer_transferred = false;
        errdefer if (!tokenizer_transferred) {
            var t = tokenizer;
            t.deinit();
        };
        const result = try initWithComponents(allocator, coordinator, model_dim, num_layers, local_batch_size, config, .{
            .tokenizer = tokenizer,
        });
        tokenizer_transferred = true;
        return result;
    }

    pub fn initWithComponents(
        allocator: std.mem.Allocator,
        coordinator: *GPUCoordinator,
        model_dim: usize,
        num_layers: usize,
        local_batch_size: usize,
        config: TrainerConfig,
        components_in: TrainerComponents,
    ) !DistributedTrainerFuthark {
        const components = components_in;
        var tokenizer_transferred = false;
        errdefer if (!tokenizer_transferred) {
            var t = components.tokenizer;
            t.deinit();
        };

        if (model_dim == 0 or model_dim % 2 != 0) return TrainerError.InvalidModelDim;
        if (num_layers == 0) return TrainerError.InvalidNumLayers;
        if (local_batch_size == 0) return TrainerError.InvalidBatchSize;
        if (coordinator.world_size == 0) return TrainerError.InvalidWorldSize;
        if (coordinator.rank >= coordinator.world_size) return TrainerError.InvalidRank;
        if (config.max_line_size == 0) return TrainerError.InvalidMaxLineSize;
        if (config.checkpoint_version == 0) return TrainerError.InvalidCheckpointVersion;
        if (!std.math.isFinite(config.esso_initial_temperature) or config.esso_initial_temperature <= 0.0) return TrainerError.InvalidOptimizerConfiguration;
        if (!std.math.isFinite(config.esso_cooling_rate) or config.esso_cooling_rate <= 0.0 or config.esso_cooling_rate > 1.0) return TrainerError.InvalidOptimizerConfiguration;
        if (config.esso_max_iterations == 0) return TrainerError.InvalidOptimizerConfiguration;
        if (config.relational_gpu_rows == 0 or config.relational_gpu_columns == 0) return TrainerError.InvalidRelationalGPUConfiguration;
        if (config.temporal_sequence_tick <= 0) return TrainerError.InvalidTemporalConfiguration;
        if (config.training_fnds_index_name.len == 0 or config.knowledge_fnds_index_name.len == 0) return TrainerError.InvalidGraphIdentifier;
        if (std.mem.eql(u8, config.training_fnds_index_name, config.knowledge_fnds_index_name)) return TrainerError.InvalidGraphIdentifier;
        if (!std.math.isFinite(config.gradient_clip_norm) or config.gradient_clip_norm <= 0.0) return TrainerError.InvalidGradient;
        if (config.relational_pass_interval == 0) return TrainerError.InvalidRelationalPassInterval;
        if (config.default_max_seq_len == 0 or config.default_max_seq_len > config.max_distributed_integer) return TrainerError.InvalidEnvironmentValue;
        try validateHyperparameters(config.learning_rate, config.momentum);

        const actual_model_dim = model_dim;

        const accelerator_ptr = try allocator.create(RSFAccelerator);
        var accelerator_ptr_committed = false;
        errdefer if (!accelerator_ptr_committed) allocator.destroy(accelerator_ptr);
        accelerator_ptr.* = try RSFAccelerator.initMultiLayerWithDepthScale(
            actual_model_dim,
            num_layers,
            allocator,
            config.spectral_depth_compensation,
        );
        var accelerator_committed = false;
        errdefer if (!accelerator_committed) accelerator_ptr.deinit();

        var gpu_embedding = try accel.EmbeddingAccelerator.init(
            &accelerator_ptr.ctx,
            components.tokenizer.next_token_id,
            actual_model_dim,
            config.embedding_seed,
        );
        var gpu_embedding_committed = false;
        errdefer if (!gpu_embedding_committed) gpu_embedding.deinit();

        var target_source: ?accel.EmbeddingAccelerator = null;
        var target_source_committed = false;
        errdefer if (!target_source_committed) {
            if (target_source) |*source| source.deinit();
        };
        if (config.target_source_frozen) {
            const frozen_source_flat = try gpu_embedding.readWeightFlat(allocator);
            defer allocator.free(frozen_source_flat);
            target_source = try accel.EmbeddingAccelerator.initWithWeights(
                &accelerator_ptr.ctx,
                components.tokenizer.next_token_id,
                actual_model_dim,
                frozen_source_flat,
            );
        }

        const crev_kernel_ptr = try allocator.create(ChaosCoreKernel);
        var crev_kernel_ptr_committed = false;
        errdefer if (!crev_kernel_ptr_committed) allocator.destroy(crev_kernel_ptr);
        crev_kernel_ptr.* = ChaosCoreKernel.init(allocator);
        var crev_kernel_committed = false;
        errdefer if (!crev_kernel_committed) crev_kernel_ptr.deinit();

        var crev_pipeline = try CREVPipeline.init(allocator, crev_kernel_ptr);
        var crev_pipeline_committed = false;
        errdefer if (!crev_pipeline_committed) crev_pipeline.deinit();

        const nsir_graph_ptr = try allocator.create(SelfSimilarRelationalGraph);
        var nsir_graph_ptr_committed = false;
        errdefer if (!nsir_graph_ptr_committed) allocator.destroy(nsir_graph_ptr);
        nsir_graph_ptr.* = try SelfSimilarRelationalGraph.init(allocator);
        var nsir_graph_committed = false;
        errdefer if (!nsir_graph_committed) nsir_graph_ptr.deinit();

        const knowledge_nsir_graph_ptr = try allocator.create(SelfSimilarRelationalGraph);
        var knowledge_nsir_graph_ptr_committed = false;
        errdefer if (!knowledge_nsir_graph_ptr_committed) allocator.destroy(knowledge_nsir_graph_ptr);
        knowledge_nsir_graph_ptr.* = try SelfSimilarRelationalGraph.init(allocator);
        var knowledge_nsir_graph_committed = false;
        errdefer if (!knowledge_nsir_graph_committed) knowledge_nsir_graph_ptr.deinit();

        var esso = EntangledStochasticSymmetryOptimizer.init(
            allocator,
            config.esso_initial_temperature,
            config.esso_cooling_rate,
            config.esso_max_iterations,
        );
        var esso_committed = false;
        errdefer if (!esso_committed) esso.deinit();

        var surprise_memory = SurpriseMemoryManager.init(
            allocator,
            &crev_kernel_ptr.storage,
            &crev_kernel_ptr.flow_analyzer,
        );
        var surprise_memory_committed = false;
        errdefer if (!surprise_memory_committed) surprise_memory.deinit();

        var temporal_graph_inst = TemporalGraph.init(allocator);
        var temporal_graph_committed = false;
        errdefer if (!temporal_graph_committed) temporal_graph_inst.deinit();

        const z_runtime_ptr = try ZRuntime.init(allocator);
        var z_runtime_committed = false;
        errdefer if (!z_runtime_committed) z_runtime_ptr.deinit();

        var r_gpu_inst = try RelationalGraphProcessingUnit.init(
            allocator,
            config.relational_gpu_rows,
            config.relational_gpu_columns,
        );
        var r_gpu_committed = false;
        errdefer if (!r_gpu_committed) r_gpu_inst.deinit();

        var fnds_manager_inst = try FNDSManager.init(allocator);
        var fnds_manager_committed = false;
        errdefer if (!fnds_manager_committed) fnds_manager_inst.deinit();

        var vpu_inst = try VPU.init(allocator);
        var vpu_committed = false;
        errdefer if (!vpu_committed) vpu_inst.deinit();

        const signal_engine_ptr = try allocator.create(SignalPropagationEngine);
        var signal_engine_ptr_committed = false;
        errdefer if (!signal_engine_ptr_committed) allocator.destroy(signal_engine_ptr);
        signal_engine_ptr.* = SignalPropagationEngine.init(
            allocator,
            nsir_graph_ptr,
            &crev_kernel_ptr.flow_analyzer,
        );
        var signal_engine_committed = false;
        errdefer if (!signal_engine_committed) signal_engine_ptr.deinit();

        const spectral_normalizer = sfd.SpectralNormalizer.init(config.spectral_iterations);
        var knowledge_graph_nonce: [32]u8 = undefined;
        std.crypto.random.bytes(knowledge_graph_nonce[0..]);

        tokenizer_transferred = true;
        accelerator_ptr_committed = true;
        accelerator_committed = true;
        gpu_embedding_committed = true;
        crev_kernel_ptr_committed = true;
        crev_kernel_committed = true;
        crev_pipeline_committed = true;
        nsir_graph_ptr_committed = true;
        nsir_graph_committed = true;
        knowledge_nsir_graph_ptr_committed = true;
        knowledge_nsir_graph_committed = true;
        esso_committed = true;
        surprise_memory_committed = true;
        temporal_graph_committed = true;
        z_runtime_committed = true;
        r_gpu_committed = true;
        fnds_manager_committed = true;
        vpu_committed = true;
        signal_engine_ptr_committed = true;
        signal_engine_committed = true;

        var trainer = DistributedTrainerFuthark{
            .allocator = allocator,
            .coordinator = coordinator,
            .tokenizer = components.tokenizer,
            .accelerator = accelerator_ptr,
            .model_dim = actual_model_dim,
            .num_layers = num_layers,
            .vocab_size = components.tokenizer.next_token_id,
            .local_batch_size = local_batch_size,
            .global_step = 0,
            .learning_rate = config.learning_rate,
            .momentum = config.momentum,
            .config = config,
            .gpu_embedding = gpu_embedding,
            .crev_pipeline = crev_pipeline,
            .crev_kernel = crev_kernel_ptr,
            .nsir_graph = nsir_graph_ptr,
            .knowledge_nsir_graph = knowledge_nsir_graph_ptr,
            .esso = esso,
            .surprise_memory = surprise_memory,
            .temporal_graph = temporal_graph_inst,
            .signal_engine = signal_engine_ptr,
            .z_runtime = z_runtime_ptr,
            .r_gpu = r_gpu_inst,
            .fnds_manager = fnds_manager_inst,
            .vpu = vpu_inst,
            .spectral_normalizer = spectral_normalizer,
            .gpu_spectral_u = null,
            .gpu_spectral_v = null,
            .training_fnds_tree_id = null,
            .training_fnds_index_id = null,
            .knowledge_fnds_tree_id = null,
            .knowledge_fnds_index_id = null,
            .knowledge_graph_nonce = knowledge_graph_nonce,
            .temporal_logical_time = 0,
            .training_variable_created = false,
            .vpu_lr_scale = 1.0,
            .target_source = target_source,
            .shuffle_control_state = config.embedding_seed ^ 0x5DEECE66D,
        };
        target_source_committed = true;

        trainer.verifyConfigConsistency(components.tokenizer.next_token_id) catch |err| {
            trainer.accelerator.deinit();
            allocator.destroy(trainer.accelerator);
            if (trainer.target_source) |*source| source.deinit();
            trainer.gpu_embedding.?.deinit();
            trainer.signal_engine.deinit();
            allocator.destroy(trainer.signal_engine);
            trainer.vpu.deinit();
            trainer.fnds_manager.deinit();
            trainer.r_gpu.deinit();
            trainer.z_runtime.deinit();
            trainer.temporal_graph.deinit();
            trainer.surprise_memory.deinit();
            trainer.esso.deinit();
            trainer.knowledge_nsir_graph.deinit();
            allocator.destroy(trainer.knowledge_nsir_graph);
            trainer.nsir_graph.deinit();
            allocator.destroy(trainer.nsir_graph);
            trainer.crev_pipeline.deinit();
            trainer.crev_kernel.deinit();
            allocator.destroy(trainer.crev_kernel);
            trainer.tokenizer.deinit();
            return err;
        };

        return trainer;
    }

    fn verifyConfigConsistency(self: *DistributedTrainerFuthark, local_vocab_size: usize) !void {
        if (self.coordinator.world_size <= 1) return;
        const vocab_u64 = std.math.cast(u64, local_vocab_size) orelse return TrainerError.ValueOverflow;
        const dim_u64 = std.math.cast(u64, self.model_dim) orelse return TrainerError.ValueOverflow;
        const layers_u64 = std.math.cast(u64, self.num_layers) orelse return TrainerError.ValueOverflow;
        if (vocab_u64 > self.config.max_distributed_integer) return TrainerError.DistributedIntegerPrecisionExceeded;
        if (dim_u64 > self.config.max_distributed_integer) return TrainerError.DistributedIntegerPrecisionExceeded;
        if (layers_u64 > self.config.max_distributed_integer) return TrainerError.DistributedIntegerPrecisionExceeded;

        const max_vocab = try self.allReduceMaximumU64(vocab_u64);
        if (max_vocab != vocab_u64) return TrainerError.DistributedConfigMismatch;
        const min_vocab_enc = try self.allReduceMaximumU64(self.config.max_distributed_integer - vocab_u64);
        if (min_vocab_enc != self.config.max_distributed_integer - vocab_u64) return TrainerError.DistributedConfigMismatch;

        const max_dim = try self.allReduceMaximumU64(dim_u64);
        if (max_dim != dim_u64) return TrainerError.DistributedConfigMismatch;
        const min_dim_enc = try self.allReduceMaximumU64(self.config.max_distributed_integer - dim_u64);
        if (min_dim_enc != self.config.max_distributed_integer - dim_u64) return TrainerError.DistributedConfigMismatch;

        const max_layers = try self.allReduceMaximumU64(layers_u64);
        if (max_layers != layers_u64) return TrainerError.DistributedConfigMismatch;
        const min_layers_enc = try self.allReduceMaximumU64(self.config.max_distributed_integer - layers_u64);
        if (min_layers_enc != self.config.max_distributed_integer - layers_u64) return TrainerError.DistributedConfigMismatch;
    }

    fn resetSpectralState(self: *DistributedTrainerFuthark) void {
        const ctx = &self.accelerator.ctx;
        if (self.gpu_spectral_u) |*u| u.free(ctx);
        if (self.gpu_spectral_v) |*v| v.free(ctx);
        self.gpu_spectral_u = null;
        self.gpu_spectral_v = null;
    }

    fn releaseTrainingFndsResources(self: *DistributedTrainerFuthark) void {
        if (self.training_fnds_index_id) |index_id| {
            _ = self.fnds_manager.removeIndex(index_id);
            self.allocator.free(index_id);
            self.training_fnds_index_id = null;
        }
        if (self.training_fnds_tree_id) |tree_id| {
            _ = self.fnds_manager.removeTree(tree_id);
            self.training_fnds_tree_id = null;
        }
    }

    fn releaseKnowledgeFndsResources(self: *DistributedTrainerFuthark) void {
        if (self.knowledge_fnds_index_id) |index_id| {
            _ = self.fnds_manager.removeIndex(index_id);
            self.allocator.free(index_id);
            self.knowledge_fnds_index_id = null;
        }
        if (self.knowledge_fnds_tree_id) |tree_id| {
            _ = self.fnds_manager.removeTree(tree_id);
            self.knowledge_fnds_tree_id = null;
        }
    }

    pub fn deinit(self: *DistributedTrainerFuthark) void {
        self.accelerator.sync() catch |err| {
            std.debug.print("[Rank {d}] WARN: accelerator.sync during deinit failed: {}\n", .{ self.coordinator.rank, err });
        };
        self.resetSpectralState();
        self.releaseTrainingFndsResources();
        self.releaseKnowledgeFndsResources();
        self.vpu.deinit();
        self.fnds_manager.deinit();
        self.r_gpu.deinit();
        self.z_runtime.deinit();
        self.signal_engine.deinit();
        self.allocator.destroy(self.signal_engine);
        self.temporal_graph.deinit();
        self.surprise_memory.deinit();
        self.esso.deinit();
        self.knowledge_nsir_graph.deinit();
        self.allocator.destroy(self.knowledge_nsir_graph);
        self.nsir_graph.deinit();
        self.allocator.destroy(self.nsir_graph);
        self.crev_pipeline.deinit();
        self.crev_kernel.deinit();
        self.allocator.destroy(self.crev_kernel);
        if (self.target_source) |*source| source.deinit();
        if (self.gpu_embedding) |*emb| emb.deinit();
        self.accelerator.deinit();
        self.allocator.destroy(self.accelerator);
        self.tokenizer.deinit();
    }

    pub fn rebindSignalEngine(self: *DistributedTrainerFuthark) void {
        self.signal_engine.deinit();
        self.signal_engine.* = SignalPropagationEngine.init(
            self.allocator,
            self.nsir_graph,
            &self.crev_kernel.flow_analyzer,
        );
    }

    fn pruneRelationalState(self: *DistributedTrainerFuthark) !void {
        if (self.training_variable_created) {
            _ = self.z_runtime.deleteVariable(self.config.training_variable_name);
            self.training_variable_created = false;
        }

        var new_nsir = try SelfSimilarRelationalGraph.init(self.allocator);
        self.nsir_graph.deinit();
        self.nsir_graph.* = new_nsir;
        new_nsir = undefined;

        self.temporal_graph.deinit();
        self.temporal_graph = TemporalGraph.init(self.allocator);
        self.temporal_logical_time = 0;
        self.releaseTrainingFndsResources();
        self.rebindSignalEngine();
        self.surprise_memory.deinit();
        self.surprise_memory = SurpriseMemoryManager.init(
            self.allocator,
            &self.crev_kernel.storage,
            &self.crev_kernel.flow_analyzer,
        );
    }

    pub fn reinitEmbedding(self: *DistributedTrainerFuthark) !void {
        var new_gpu_embedding = try accel.EmbeddingAccelerator.init(
            &self.accelerator.ctx,
            self.tokenizer.next_token_id,
            self.model_dim,
            self.config.embedding_seed,
        );
        errdefer new_gpu_embedding.deinit();

        self.resetSpectralState();
        if (self.gpu_embedding) |*old| old.deinit();
        self.gpu_embedding = new_gpu_embedding;
        self.vocab_size = self.tokenizer.next_token_id;

        if (self.config.target_source_frozen) {
            const frozen_source_flat = try self.gpu_embedding.?.readWeightFlat(self.allocator);
            defer self.allocator.free(frozen_source_flat);
            var new_target_source = try accel.EmbeddingAccelerator.initWithWeights(
                &self.accelerator.ctx,
                self.tokenizer.next_token_id,
                self.model_dim,
                frozen_source_flat,
            );
            errdefer new_target_source.deinit();
            if (self.target_source) |*old| old.deinit();
            self.target_source = new_target_source;
        } else {
            if (self.target_source) |*old| old.deinit();
            self.target_source = null;
        }
    }

    fn validateHyperparameters(learning_rate: f32, momentum: f32) TrainerError!void {
        if (!std.math.isFinite(learning_rate)) return TrainerError.InvalidLearningRate;
        if (!std.math.isFinite(momentum)) return TrainerError.InvalidMomentum;
        if (learning_rate < 0.0 or learning_rate > 65504.0) return TrainerError.InvalidLearningRate;
        if (momentum < 0.0 or momentum >= 1.0) return TrainerError.InvalidMomentum;
        const lr_f16: f16 = @floatCast(learning_rate);
        const momentum_f16: f16 = @floatCast(momentum);
        if (!std.math.isFinite(lr_f16)) return TrainerError.InvalidHyperparameterAfterCast;
        if (learning_rate > 0.0 and lr_f16 == @as(f16, 0.0)) return TrainerError.InvalidHyperparameterAfterCast;
        const momentum_back: f32 = @floatCast(momentum_f16);
        if (!std.math.isFinite(momentum_back) or !(momentum_back < 1.0)) return TrainerError.InvalidHyperparameterAfterCast;
    }

    fn checkedF32ToF16(value: f32) TrainerError!f16 {
        if (!std.math.isFinite(value)) return TrainerError.InvalidFloat16Value;
        if (value < -65504.0 or value > 65504.0) return TrainerError.InvalidFloat16Value;
        const converted: f16 = @floatCast(value);
        if (!std.math.isFinite(converted)) return TrainerError.InvalidFloat16Value;
        return converted;
    }

    fn safeUsizeToU32(value: usize) TrainerError!u32 {
        if (value > std.math.maxInt(u32)) return TrainerError.ValueOverflow;
        return @as(u32, @intCast(value));
    }

    fn openReadFile(path: []const u8) !std.fs.File {
        if (std.fs.path.isAbsolute(path)) return std.fs.openFileAbsolute(path, .{ .mode = .read_only });
        return std.fs.cwd().openFile(path, .{ .mode = .read_only });
    }

    fn createWriteFile(path: []const u8) !std.fs.File {
        if (std.fs.path.isAbsolute(path)) return std.fs.createFileAbsolute(path, .{ .mode = 0o600, .truncate = true });
        return std.fs.cwd().createFile(path, .{ .mode = 0o600, .truncate = true });
    }

    fn deletePath(path: []const u8) void {
        if (std.fs.path.isAbsolute(path)) {
            std.fs.deleteFileAbsolute(path) catch |err| switch (err) {
                error.FileNotFound => {},
                else => std.debug.print("WARN: deletePath({s}) failed: {}\n", .{ path, err }),
            };
            return;
        }
        std.fs.cwd().deleteFile(path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => std.debug.print("WARN: deletePath({s}) failed: {}\n", .{ path, err }),
        };
    }

    fn renamePath(from: []const u8, to: []const u8) !void {
        if (std.fs.path.isAbsolute(from)) return std.fs.renameAbsolute(from, to);
        return std.fs.cwd().rename(from, to);
    }

    fn syncContainingDirectory(path: []const u8) void {
        const dir_path = std.fs.path.dirname(path) orelse ".";
        var dir = if (std.fs.path.isAbsolute(dir_path))
            std.fs.openDirAbsolute(dir_path, .{}) catch return
        else
            std.fs.cwd().openDir(dir_path, .{}) catch return;
        defer dir.close();
        std.posix.fsync(dir.fd) catch {};
    }

    fn writeF32(writer: anytype, value: f32) !void {
        try writer.writeInt(u32, @as(u32, @bitCast(value)), .little);
    }

    fn readF32(reader: anytype) !f32 {
        const bits = try reader.readInt(u32, .little);
        return @as(f32, @bitCast(bits));
    }

    fn writeF64(writer: anytype, value: f64) !void {
        try writer.writeInt(u64, @as(u64, @bitCast(value)), .little);
    }

    fn readF64(reader: anytype) !f64 {
        const bits = try reader.readInt(u64, .little);
        return @as(f64, @bitCast(bits));
    }

    fn parseOptionalEnvironmentU64(
        self: *DistributedTrainerFuthark,
        name: []const u8,
    ) !?u64 {
        const owned = std.process.getEnvVarOwned(self.allocator, name) catch |err| switch (err) {
            error.EnvironmentVariableNotFound => return null,
            else => {
                std.debug.print("[Rank {d}] WARN: reading env '{s}' failed: {}\n", .{ self.coordinator.rank, name, err });
                return TrainerError.InvalidEnvironmentValue;
            },
        };
        defer self.allocator.free(owned);
        if (owned.len == 0) return TrainerError.InvalidEnvironmentValue;
        return std.fmt.parseInt(u64, owned, 10) catch return TrainerError.InvalidEnvironmentValue;
    }

    fn parseOptionalEnvironmentUsize(
        self: *DistributedTrainerFuthark,
        name: []const u8,
    ) !?usize {
        const value_opt = try self.parseOptionalEnvironmentU64(name);
        if (value_opt) |v| {
            return std.math.cast(usize, v) orelse TrainerError.InvalidEnvironmentValue;
        }
        return null;
    }

    fn getMaximumSequenceLength(self: *DistributedTrainerFuthark) !usize {
        const parsed = self.parseOptionalEnvironmentUsize("JAIDE_MAX_SEQ_LEN") catch |err| {
            std.debug.print("[Rank {d}] WARN: JAIDE_MAX_SEQ_LEN invalid: {} (using default {d})\n", .{ self.coordinator.rank, err, self.config.default_max_seq_len });
            return self.config.default_max_seq_len;
        };
        const result = parsed orelse self.config.default_max_seq_len;
        if (result == 0 or result > self.config.max_distributed_integer) return TrainerError.InvalidEnvironmentValue;
        return result;
    }

    fn isTokenizableText(self: *DistributedTrainerFuthark, text: []const u8) !bool {
        var token_list = std.ArrayList(u32).init(self.allocator);
        defer token_list.deinit();
        self.tokenizer.encode(text, &token_list) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return false,
        };
        return token_list.items.len > 1;
    }

    fn extractDatasetText(self: *DistributedTrainerFuthark, line: []const u8) !?[]u8 {
        const parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            line,
            .{ .allocate = .alloc_always },
        ) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => return null,
        };
        defer parsed.deinit();

        return switch (parsed.value) {
            .object => |obj| blk: {
                const text_value = obj.get("text") orelse break :blk null;
                break :blk switch (text_value) {
                    .string => |text| if (text.len > 0)
                        try self.allocator.dupe(u8, text)
                    else
                        null,
                    else => null,
                };
            },
            else => null,
        };
    }

    fn countUsableDatasetSamples(self: *DistributedTrainerFuthark, dataset_path: []const u8) !u64 {
        const file = try openReadFile(dataset_path);
        defer file.close();
        var buffered_reader = std.io.bufferedReader(file.reader());
        var reader = buffered_reader.reader();
        var count: u64 = 0;
        while (try reader.readUntilDelimiterOrEofAlloc(self.allocator, '\n', self.config.max_line_size)) |line| {
            defer self.allocator.free(line);
            const maybe_text = try self.extractDatasetText(line);
            if (maybe_text) |text| {
                defer self.allocator.free(text);
                if (try self.isTokenizableText(text)) {
                    count = try std.math.add(u64, count, 1);
                }
            }
        }
        return count;
    }

    fn appendDatasetRange(
        self: *DistributedTrainerFuthark,
        dataset_path: []const u8,
        start_valid_index: usize,
        count: usize,
        samples: *std.ArrayList([]const u8),
    ) !void {
        if (count == 0) return;
        const end_valid_index = try std.math.add(usize, start_valid_index, count);

        const file = try openReadFile(dataset_path);
        defer file.close();
        var buffered_reader = std.io.bufferedReader(file.reader());
        var reader = buffered_reader.reader();

        var valid_index: usize = 0;
        var appended: usize = 0;

        while (try reader.readUntilDelimiterOrEofAlloc(self.allocator, '\n', self.config.max_line_size)) |line| {
            defer self.allocator.free(line);
            if (appended == count or valid_index >= end_valid_index) break;

            const maybe_text = try self.extractDatasetText(line);
            const text = maybe_text orelse continue;
            var text_owned = true;
            defer if (text_owned) self.allocator.free(text);

            const usable = try self.isTokenizableText(text);
            if (!usable) continue;

            if (valid_index >= start_valid_index) {
                samples.append(text) catch |err| return err;
                text_owned = false;
                appended = try std.math.add(usize, appended, 1);
            }
            valid_index = try std.math.add(usize, valid_index, 1);
        }

        if (appended != count) return TrainerError.InvalidDatasetPartition;
    }

    fn readLayerMatrix(self: *DistributedTrainerFuthark, layer_idx: usize, kind: accel.WeightKind) ![]f16 {
        return self.accelerator.readLayerWeightsFlat(layer_idx, kind, self.allocator) catch |err| {
            std.debug.print("[Rank {d}] readLayerMatrix layer={d} kind={} err={}\n", .{ self.coordinator.rank, layer_idx, kind, err });
            return err;
        };
    }

    fn allReduceFloat32Values(self: *DistributedTrainerFuthark, values: []f32) !void {
        if (values.len == 0 or self.coordinator.world_size <= 1) return;
        const byte_count = try std.math.mul(usize, values.len, @sizeOf(f32));
        const device_values = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(device_values);
        try self.coordinator.copyHostToDevice(device_values, std.mem.sliceAsBytes(values), byte_count);
        try self.coordinator.allReduceFloat32(device_values, device_values, values.len);
        try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(values), device_values, byte_count);
        try self.coordinator.synchronize();
    }

    fn allReduceFloat32ValuesAvg(self: *DistributedTrainerFuthark, values: []f32) !void {
        if (values.len == 0 or self.coordinator.world_size <= 1) return;
        const byte_count = try std.math.mul(usize, values.len, @sizeOf(f32));
        const device_values = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(device_values);
        try self.coordinator.copyHostToDevice(device_values, std.mem.sliceAsBytes(values), byte_count);
        try self.coordinator.allReduceFloat32Avg(device_values, device_values, values.len);
        try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(values), device_values, byte_count);
        try self.coordinator.synchronize();
    }

    fn allReduceMaximumU64Raw(self: *DistributedTrainerFuthark, value: u64, limit: u64) !u64 {
        if (self.coordinator.world_size <= 1) return value;
        var arr = [1]f32{@as(f32, @floatFromInt(value))};
        const byte_count = @sizeOf(f32);
        const device_values = try self.coordinator.allocDeviceMemory(byte_count);
        defer self.coordinator.freeDeviceMemory(device_values);
        try self.coordinator.copyHostToDevice(device_values, std.mem.sliceAsBytes(arr[0..]), byte_count);
        try self.coordinator.allReduceFloat32Max(device_values, device_values, arr.len);
        try self.coordinator.copyDeviceToHost(std.mem.sliceAsBytes(arr[0..]), device_values, byte_count);
        try self.coordinator.synchronize();
        if (!std.math.isFinite(arr[0]) or arr[0] < 0.0 or arr[0] > @as(f32, @floatFromInt(limit))) return TrainerError.InvalidDistributedInteger;
        return @as(u64, @intFromFloat(arr[0]));
    }

    fn allReduceMaximumU64(self: *DistributedTrainerFuthark, value: u64) !u64 {
        if (self.coordinator.world_size <= 1) {
            if (value > self.config.max_distributed_integer) return TrainerError.DistributedIntegerPrecisionExceeded;
            return value;
        }
        const overflow_flag: u64 = if (value > self.config.max_distributed_integer) 1 else 0;
        const global_overflow = try self.allReduceMaximumU64Raw(overflow_flag, 1);
        if (global_overflow != 0) return TrainerError.DistributedIntegerPrecisionExceeded;
        return self.allReduceMaximumU64Raw(value, self.config.max_distributed_integer);
    }

    fn allReduceSumU64(self: *DistributedTrainerFuthark, value: u64) !u64 {
        if (self.coordinator.world_size <= 1) return value;
        const exact_integer_limit: u64 = 1 << 24;
        const world_size_u64 = std.math.cast(u64, self.coordinator.world_size) orelse return TrainerError.ValueOverflow;
        if (world_size_u64 == 0) return TrainerError.InvalidWorldSize;
        if (world_size_u64 > exact_integer_limit) return TrainerError.DistributedIntegerPrecisionExceeded;

        var radix_bits: u6 = 1;
        while (radix_bits < 24) {
            const candidate_bits: u6 = radix_bits + 1;
            const candidate_mask = (@as(u64, 1) << candidate_bits) - 1;
            if (candidate_mask > exact_integer_limit / world_size_u64) break;
            radix_bits = candidate_bits;
        }

        const bits_per_limb: usize = @intCast(radix_bits);
        const limb_count = (64 + bits_per_limb - 1) / bits_per_limb;
        const limb_mask = (@as(u64, 1) << radix_bits) - 1;
        var limb_values = [_]f32{0.0} ** 64;
        var limb_index: usize = 0;
        while (limb_index < limb_count) : (limb_index += 1) {
            const shift: u6 = @intCast(limb_index * bits_per_limb);
            const limb = (value >> shift) & limb_mask;
            limb_values[limb_index] = @as(f32, @floatFromInt(limb));
        }

        try self.allReduceFloat32Values(limb_values[0..limb_count]);

        var result: u128 = 0;
        var carry: u128 = 0;
        limb_index = 0;
        while (limb_index < limb_count) : (limb_index += 1) {
            const reduced = limb_values[limb_index];
            if (!std.math.isFinite(reduced) or reduced < 0.0 or reduced > @as(f32, @floatFromInt(exact_integer_limit))) return TrainerError.InvalidDistributedInteger;
            const rounded = @round(reduced);
            if (rounded != reduced) return TrainerError.InvalidDistributedInteger;
            const limb_sum: u64 = @intFromFloat(rounded);
            const total = @as(u128, limb_sum) + carry;
            const digit = total & @as(u128, limb_mask);
            const shift: u7 = @intCast(limb_index * bits_per_limb);
            result |= digit << shift;
            carry = total >> radix_bits;
        }

        if (carry != 0 or result > @as(u128, std.math.maxInt(u64))) return TrainerError.ValueOverflow;
        return @intCast(result);
    }

    fn reduceWeightedDeltaInPlace(
        self: *DistributedTrainerFuthark,
        delta: []f16,
        local_fraction: f32,
    ) !void {
        if (delta.len == 0) return;
        if (!std.math.isFinite(local_fraction) or local_fraction < 0.0 or local_fraction > 1.0) return TrainerError.InvalidReductionWeight;
        var values = try self.allocator.alloc(f32, delta.len);
        defer self.allocator.free(values);
        for (delta, 0..) |value, index| {
            const converted: f32 = @floatCast(value);
            values[index] = converted * local_fraction;
        }
        try self.allReduceFloat32Values(values);
        for (values, 0..) |value, index| {
            delta[index] = try checkedF32ToF16(value);
        }
    }

    fn applyLayerMatrix(
        self: *DistributedTrainerFuthark,
        layer_idx: usize,
        base: []const f16,
        delta: []const f16,
        kind: accel.WeightKind,
    ) !void {
        if (base.len != delta.len) return TrainerError.InvalidWeightsShape;
        const half = self.model_dim / 2;
        const columns = try std.math.add(usize, half, 1);
        const expected_length = try std.math.mul(usize, half, columns);
        if (base.len != expected_length) return TrainerError.InvalidWeightsShape;

        var merged = try self.allocator.alloc(f16, base.len);
        defer self.allocator.free(merged);
        for (base, delta, 0..) |base_value, delta_value, index| {
            const merged_value = @as(f32, @floatCast(base_value)) + @as(f32, @floatCast(delta_value));
            merged[index] = try checkedF32ToF16(merged_value);
        }
        switch (kind) {
            .weights_s => try self.accelerator.setLayerWeightsS(layer_idx, merged, half, columns),
            .weights_t => try self.accelerator.setLayerWeightsT(layer_idx, merged, half, columns),
            .velocity_s => try self.accelerator.setLayerVelocityS(layer_idx, merged, half, columns),
            .velocity_t => try self.accelerator.setLayerVelocityT(layer_idx, merged, half, columns),
        }
    }

    fn subtractLayerSnapshot(current: []f16, original: []const f16) !void {
        if (current.len != original.len) return TrainerError.InvalidWeightsShape;
        for (current, original) |*current_value, original_value| {
            const difference = @as(f32, @floatCast(current_value.*)) - @as(f32, @floatCast(original_value));
            current_value.* = try checkedF32ToF16(difference);
        }
    }

    fn freeLayerSnapshots(self: *DistributedTrainerFuthark, snapshots: []LayerSnapshot) void {
        for (snapshots) |snapshot| {
            if (snapshot.weights_s.len > 0) self.allocator.free(snapshot.weights_s);
            if (snapshot.weights_t.len > 0) self.allocator.free(snapshot.weights_t);
            if (snapshot.velocity_s.len > 0) self.allocator.free(snapshot.velocity_s);
            if (snapshot.velocity_t.len > 0) self.allocator.free(snapshot.velocity_t);
        }
        self.allocator.free(snapshots);
    }

    fn captureLayerSnapshots(self: *DistributedTrainerFuthark) ![]LayerSnapshot {
        const snapshots = try self.allocator.alloc(LayerSnapshot, self.num_layers);
        errdefer self.allocator.free(snapshots);
        for (snapshots) |*snapshot| {
            snapshot.* = .{
                .weights_s = &.{},
                .weights_t = &.{},
                .velocity_s = &.{},
                .velocity_t = &.{},
            };
        }
        for (snapshots, 0..) |*snapshot, layer_index| {
            snapshot.weights_s = self.readLayerMatrix(layer_index, .weights_s) catch |err| {
                var idx: usize = 0;
                while (idx <= layer_index) : (idx += 1) {
                    if (snapshots[idx].weights_s.len > 0) self.allocator.free(snapshots[idx].weights_s);
                    if (snapshots[idx].weights_t.len > 0) self.allocator.free(snapshots[idx].weights_t);
                    if (snapshots[idx].velocity_s.len > 0) self.allocator.free(snapshots[idx].velocity_s);
                    if (snapshots[idx].velocity_t.len > 0) self.allocator.free(snapshots[idx].velocity_t);
                }
                return err;
            };
            snapshot.weights_t = self.readLayerMatrix(layer_index, .weights_t) catch |err| {
                self.allocator.free(snapshot.weights_s);
                snapshot.weights_s = &.{};
                var idx: usize = 0;
                while (idx < layer_index) : (idx += 1) {
                    if (snapshots[idx].weights_s.len > 0) self.allocator.free(snapshots[idx].weights_s);
                    if (snapshots[idx].weights_t.len > 0) self.allocator.free(snapshots[idx].weights_t);
                    if (snapshots[idx].velocity_s.len > 0) self.allocator.free(snapshots[idx].velocity_s);
                    if (snapshots[idx].velocity_t.len > 0) self.allocator.free(snapshots[idx].velocity_t);
                }
                return err;
            };
            snapshot.velocity_s = self.readLayerMatrix(layer_index, .velocity_s) catch |err| {
                self.allocator.free(snapshot.weights_s);
                self.allocator.free(snapshot.weights_t);
                snapshot.weights_s = &.{};
                snapshot.weights_t = &.{};
                var idx: usize = 0;
                while (idx < layer_index) : (idx += 1) {
                    if (snapshots[idx].weights_s.len > 0) self.allocator.free(snapshots[idx].weights_s);
                    if (snapshots[idx].weights_t.len > 0) self.allocator.free(snapshots[idx].weights_t);
                    if (snapshots[idx].velocity_s.len > 0) self.allocator.free(snapshots[idx].velocity_s);
                    if (snapshots[idx].velocity_t.len > 0) self.allocator.free(snapshots[idx].velocity_t);
                }
                return err;
            };
            snapshot.velocity_t = self.readLayerMatrix(layer_index, .velocity_t) catch |err| {
                self.allocator.free(snapshot.weights_s);
                self.allocator.free(snapshot.weights_t);
                self.allocator.free(snapshot.velocity_s);
                snapshot.weights_s = &.{};
                snapshot.weights_t = &.{};
                snapshot.velocity_s = &.{};
                var idx: usize = 0;
                while (idx < layer_index) : (idx += 1) {
                    if (snapshots[idx].weights_s.len > 0) self.allocator.free(snapshots[idx].weights_s);
                    if (snapshots[idx].weights_t.len > 0) self.allocator.free(snapshots[idx].weights_t);
                    if (snapshots[idx].velocity_s.len > 0) self.allocator.free(snapshots[idx].velocity_s);
                    if (snapshots[idx].velocity_t.len > 0) self.allocator.free(snapshots[idx].velocity_t);
                }
                return err;
            };
        }
        return snapshots;
    }

    pub fn loadDataset(self: *DistributedTrainerFuthark, dataset_path: []const u8) ![][]const u8 {
        if (self.coordinator.world_size == 0) return TrainerError.InvalidWorldSize;
        if (self.coordinator.rank >= self.coordinator.world_size) return TrainerError.InvalidRank;

        const declared_total_opt = try self.parseOptionalEnvironmentU64("JAIDE_TOTAL_SAMPLES");
        const maximum_samples_opt = try self.parseOptionalEnvironmentU64("JAIDE_MAX_SAMPLES");

        var declared_total_synchronized: u64 = 0;
        if (declared_total_opt) |declared| {
            declared_total_synchronized = try self.allReduceMaximumU64(declared);
            const min_declared = try self.allReduceMaximumU64(self.config.max_distributed_integer - declared);
            if (min_declared != self.config.max_distributed_integer - declared) return TrainerError.DistributedConfigMismatch;
        }

        var max_samples_synchronized: ?u64 = null;
        if (maximum_samples_opt) |maximum| {
            if (maximum == 0) return TrainerError.InvalidEnvironmentValue;
            const max_max = try self.allReduceMaximumU64(maximum);
            const min_max_enc = try self.allReduceMaximumU64(self.config.max_distributed_integer - maximum);
            if (max_max != maximum or min_max_enc != self.config.max_distributed_integer - maximum) return TrainerError.DistributedConfigMismatch;
            max_samples_synchronized = maximum;
        }

        var valid_sample_count: u64 = 0;
        if (declared_total_opt != null) {
            valid_sample_count = declared_total_synchronized;
        } else {
            var root_count: u64 = 0;
            var root_error: u64 = 0;
            if (self.coordinator.isRoot()) {
                root_count = self.countUsableDatasetSamples(dataset_path) catch blk: {
                    root_error = 1;
                    break :blk 0;
                };
            }
            const global_error = try self.allReduceMaximumU64(root_error);
            if (global_error != 0) return TrainerError.EmptyDataset;
            valid_sample_count = try self.allReduceMaximumU64(root_count);
        }

        if (max_samples_synchronized) |maximum| {
            if (maximum < valid_sample_count) valid_sample_count = maximum;
        }

        if (valid_sample_count == 0) return TrainerError.EmptyDataset;

        const world_u64: u64 = @as(u64, self.coordinator.world_size);
        const rank_u64: u64 = @as(u64, self.coordinator.rank);
        const base_per_rank = valid_sample_count / world_u64;
        const remainder = valid_sample_count % world_u64;
        const samples_per_rank_u64: u64 = if (rank_u64 < remainder) base_per_rank + 1 else base_per_rank;
        const start_valid_index_u64: u64 = if (rank_u64 < remainder)
            rank_u64 * (base_per_rank + 1)
        else
            remainder * (base_per_rank + 1) + (rank_u64 - remainder) * base_per_rank;

        const samples_per_rank = std.math.cast(usize, samples_per_rank_u64) orelse return TrainerError.ValueOverflow;
        const start_valid_index = std.math.cast(usize, start_valid_index_u64) orelse return TrainerError.ValueOverflow;

        var samples = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (samples.items) |sample| self.allocator.free(sample);
            samples.deinit();
        }

        if (samples_per_rank > 0) {
            try self.appendDatasetRange(dataset_path, start_valid_index, samples_per_rank, &samples);
        }

        if (samples.items.len != samples_per_rank) return TrainerError.InvalidDatasetPartition;

        if (self.coordinator.isRoot()) {
            std.debug.print("[Rank {d}] Loaded {d} samples from total {d} (rank slice)\n", .{ self.coordinator.rank, samples.items.len, valid_sample_count });
        }

        return samples.toOwnedSlice();
    }

    pub fn trainEpoch(self: *DistributedTrainerFuthark, samples: [][]const u8) !f32 {
        if (self.local_batch_size == 0) return TrainerError.InvalidBatchSize;

        const local_batch_count: u64 = if (samples.len == 0) 0 else blk: {
            const inc = try std.math.add(usize, samples.len, self.local_batch_size - 1);
            break :blk @as(u64, inc / self.local_batch_size);
        };
        const target_batch_count = try self.allReduceMaximumU64(local_batch_count);

        var total_weighted_loss: f64 = 0.0;
        var total_sample_weight: f64 = 0.0;

        var batch_index: u64 = 0;
        var batch_start: usize = 0;
        while (batch_index < target_batch_count) : (batch_index += 1) {
            var batch: [][]const u8 = &.{};
            if (batch_start < samples.len) {
                const remaining = samples.len - batch_start;
                const batch_length = @min(self.local_batch_size, remaining);
                const batch_end = try std.math.add(usize, batch_start, batch_length);
                batch = samples[batch_start..batch_end];
                batch_start = batch_end;
            }

            const step_result = self.trainStepFuthark(batch) catch |err| {
                std.debug.print("[Rank {d}] trainStepFuthark ERROR at step {d}: {}\n", .{ self.coordinator.rank, self.global_step, err });
                return err;
            };

            if (!std.math.isFinite(step_result.loss)) return TrainerError.InvalidLoss;
            if (step_result.sample_weight > 0.0) {
                total_weighted_loss += @as(f64, step_result.loss) * step_result.sample_weight;
                total_sample_weight += step_result.sample_weight;
                self.global_step = try std.math.add(u64, self.global_step, 1);
            }

            if (self.coordinator.isRoot() and (self.global_step <= 50 or self.global_step % 10 == 0)) {
                std.debug.print(
                    "[Step {d}] Loss: {d:.6} | Recon: {d:.6} | SourceRMS: {d:.6}\n",
                    .{ self.global_step, step_result.loss, step_result.reconstruction_loss, step_result.source_rms },
                );
            }
        }

        const reduce_buf = [2]f64{total_weighted_loss, total_sample_weight};
        var reduce_f32 = [2]f32{
            @as(f32, @floatCast(reduce_buf[0])),
            @as(f32, @floatCast(reduce_buf[1])),
        };
        try self.allReduceFloat32Values(reduce_f32[0..]);

        const global_loss_sum: f64 = @as(f64, reduce_f32[0]);
        const global_weight: f64 = @as(f64, reduce_f32[1]);

        if (global_weight <= 0.0) {
            std.debug.print("[WARNING] No samples processed across all ranks\n", .{});
            try self.pruneRelationalState();
            return 0.0;
        }
        const result: f32 = @floatCast(global_loss_sum / global_weight);
        if (!std.math.isFinite(result)) return TrainerError.InvalidLoss;
        try self.pruneRelationalState();
        return result;
    }

    fn ensureTrainingFndsTree(self: *DistributedTrainerFuthark) ![32]u8 {
        if (self.training_fnds_tree_id) |tree_id| return tree_id;
        const tree_id = try self.fnds_manager.createTree(self.config.fnds_max_depth, self.config.fnds_branching);
        self.training_fnds_tree_id = tree_id;
        return tree_id;
    }

    fn ensureTrainingFndsIndex(self: *DistributedTrainerFuthark) ![]const u8 {
        if (self.training_fnds_index_id) |index_id| return index_id;
        const index_id = try self.allocator.dupe(u8, self.config.training_fnds_index_name);
        errdefer self.allocator.free(index_id);
        try self.fnds_manager.createIndex(index_id);
        if (self.fnds_manager.getIndex(index_id) == null) {
            _ = self.fnds_manager.removeIndex(index_id);
            return TrainerError.InvalidGraphIdentifier;
        }
        self.training_fnds_index_id = index_id;
        return index_id;
    }

    fn ensureKnowledgeFndsTree(self: *DistributedTrainerFuthark) ![32]u8 {
        if (self.knowledge_fnds_tree_id) |tree_id| return tree_id;
        const tree_id = try self.fnds_manager.createTree(self.config.fnds_kg_max_depth, self.config.fnds_kg_branching);
        self.knowledge_fnds_tree_id = tree_id;
        return tree_id;
    }

    fn ensureKnowledgeFndsIndex(self: *DistributedTrainerFuthark) ![]const u8 {
        if (self.knowledge_fnds_index_id) |index_id| return index_id;
        const index_id = try self.allocator.dupe(u8, self.config.knowledge_fnds_index_name);
        errdefer self.allocator.free(index_id);
        try self.fnds_manager.createIndex(index_id);
        if (self.fnds_manager.getIndex(index_id) == null) {
            _ = self.fnds_manager.removeIndex(index_id);
            return TrainerError.InvalidGraphIdentifier;
        }
        self.knowledge_fnds_index_id = index_id;
        return index_id;
    }

    fn runCoreRelationalPass(
        self: *DistributedTrainerFuthark,
        token_lists: []const std.ArrayList(u32),
    ) !void {
        var has_tokens = false;
        var maximum_sequence_span: usize = 0;
        for (token_lists) |token_list| {
            if (token_list.items.len == 0) continue;
            has_tokens = true;
            if (token_list.items.len > maximum_sequence_span) maximum_sequence_span = token_list.items.len;
            const le_bytes = try self.allocator.alloc(u8, token_list.items.len * @sizeOf(u32));
            defer self.allocator.free(le_bytes);
            for (token_list.items, 0..) |tok, i| {
                std.mem.writeInt(u32, le_bytes[i * 4 ..][0..4], tok, .little);
            }
            _ = self.nsir_graph.encodeInformation(le_bytes) catch |err| {
                std.debug.print("[Rank {d}] WARN: nsir_graph.encodeInformation failed: {}\n", .{ self.coordinator.rank, err });
            };
            _ = self.surprise_memory.storeWithSurprise(le_bytes, null) catch |err| {
                std.debug.print("[Rank {d}] WARN: surprise_memory.storeWithSurprise failed: {}\n", .{ self.coordinator.rank, err });
            };
        }

        if (!has_tokens) return;

        var graph_embeddings_opt: ?std.ArrayList(core_relational.F64x4) = self.vpu.computeGraphEmbeddings(self.nsir_graph) catch |err| blk: {
            std.debug.print("[Rank {d}] WARN: VPU.computeGraphEmbeddings failed: {}\n", .{ self.coordinator.rank, err });
            break :blk null;
        };
        if (graph_embeddings_opt) |*embeddings| {
            defer embeddings.deinit();
            if (embeddings.items.len > 0) {
                const hash = self.nsir_graph.getTopologyHash() catch return TrainerError.InvalidQuantumState;
                const theta: f64 = @as(f64, @floatFromInt(hash[0])) / 255.0 * std.math.pi;
                const phi: f64 = @as(f64, @floatFromInt(hash[1])) / 255.0 * std.math.pi;
                self.vpu.quantumVectorOps(embeddings.items, theta, phi);
                var magnitude_sum: f64 = 0.0;
                for (embeddings.items) |embed| {
                    var lane_sq: f64 = 0.0;
                    for (0..4) |lane| {
                        const v: f64 = embed.get(lane);
                        lane_sq += v * v;
                    }
                    magnitude_sum += @sqrt(lane_sq);
                }
                const mean_magnitude = magnitude_sum / @as(f64, @floatFromInt(embeddings.items.len));
                if (std.math.isFinite(mean_magnitude) and mean_magnitude > 0.0) {
                    const scale_candidate: f32 = @floatCast(@min(2.0, @max(0.5, 1.0 / (1.0 + mean_magnitude))));
                    if (std.math.isFinite(scale_candidate) and scale_candidate > 0.0) {
                        self.vpu_lr_scale = scale_candidate;
                    }
                }
            }
        }

        if (self.coordinator.world_size > 1) {
            const world_fraction: f32 = 1.0 / @as(f32, @floatFromInt(self.coordinator.world_size));
            var shared_scale = [1]f32{self.vpu_lr_scale * world_fraction};
            self.allReduceFloat32Values(shared_scale[0..]) catch |err| {
                std.debug.print(
                    "[Rank {d}] WARN: vpu scale reduction failed: {}\n",
                    .{ self.coordinator.rank, err },
                );
                shared_scale[0] = self.vpu_lr_scale;
            };
            if (std.math.isFinite(shared_scale[0]) and shared_scale[0] > 0.0) {
                self.vpu_lr_scale = @min(2.0, @max(0.5, shared_scale[0]));
            }
        }

        fnds_block: {
            const tree_id = self.ensureTrainingFndsTree() catch |err| {
                std.debug.print("[Rank {d}] WARN: ensureTrainingFndsTree failed: {}\n", .{ self.coordinator.rank, err });
                break :fnds_block;
            };
            const index_id = self.ensureTrainingFndsIndex() catch |err| {
                std.debug.print("[Rank {d}] WARN: ensureTrainingFndsIndex failed: {}\n", .{ self.coordinator.rank, err });
                break :fnds_block;
            };
            for (token_lists, 0..) |token_list, sample_index| {
                if (token_list.items.len == 0) continue;
                var node_id_buffer: [96]u8 = undefined;
                const node_id = std.fmt.bufPrint(
                    &node_id_buffer,
                    "rank_{d}_step_{d}_sample_{d}",
                    .{ self.coordinator.rank, self.global_step, sample_index },
                ) catch continue;
                const le_tok_bytes = self.allocator.alloc(u8, token_list.items.len * @sizeOf(u32)) catch continue;
                defer self.allocator.free(le_tok_bytes);
                for (token_list.items, 0..) |tok, i| {
                    std.mem.writeInt(u32, le_tok_bytes[i * 4 ..][0..4], tok, .little);
                }

                var pattern_location = PatternLocation.init(
                    self.allocator,
                    tree_id,
                    0,
                    node_id,
                    0,
                    @min(le_tok_bytes.len, 8 * @sizeOf(u32)),
                    1.0,
                ) catch |err| {
                    std.debug.print("[Rank {d}] WARN: PatternLocation.init failed: {}\n", .{ self.coordinator.rank, err });
                    continue;
                };
                var pattern_transferred = false;
                defer if (!pattern_transferred) pattern_location.deinit();

                _ = self.fnds_manager.insertIntoTree(tree_id, node_id, le_tok_bytes, 0) catch |err| {
                    std.debug.print("[Rank {d}] WARN: fnds insertIntoTree failed: {}\n", .{ self.coordinator.rank, err });
                    continue;
                };

                const pattern_bytes = le_tok_bytes[0..pattern_location.length];
                self.fnds_manager.addPatternToIndex(index_id, pattern_bytes, pattern_location) catch |err| {
                    std.debug.print("[Rank {d}] WARN: fnds addPatternToIndex failed: {}\n", .{ self.coordinator.rank, err });
                    continue;
                };
                pattern_transferred = true;
            }
        }

        self.r_gpu.distributeGraph(self.nsir_graph) catch |err| {
            std.debug.print("[Rank {d}] WARN: r_gpu.distributeGraph failed: {}\n", .{ self.coordinator.rank, err });
        };

        {
            var orchestrator = ReasoningOrchestrator.init(
                self.allocator,
                self.nsir_graph,
                &self.esso,
                self.crev_kernel,
            );
            defer orchestrator.deinit();
            _ = orchestrator.runHierarchicalReasoning(self.config.reasoning_cycles) catch |err| {
                std.debug.print("[Rank {d}] WARN: runHierarchicalReasoning failed: {}\n", .{ self.coordinator.rank, err });
            };
        }

        {
            const logical_time = self.temporal_logical_time;
            var node_iterator = self.nsir_graph.nodes.iterator();
            while (node_iterator.next()) |entry| {
                const node = entry.value_ptr;
                const quantum_state = QuantumState.init(
                    node.qubit.a.re,
                    node.qubit.a.im,
                    node.qubit.b.re,
                    node.qubit.b.im,
                    node.phase,
                    0.0,
                );
                self.temporal_graph.addNodeAtTime(node.id, quantum_state, logical_time) catch |err| switch (err) {
                    error.NodeAlreadyExists => {},
                    else => std.debug.print("[Rank {d}] WARN: temporal addNodeAtTime failed: {}\n", .{ self.coordinator.rank, err }),
                };
            }
            temporal_time_block: {
                const sequence_span = std.math.cast(i64, maximum_sequence_span) orelse {
                    std.debug.print("[Rank {d}] WARN: temporal sequence_span cast overflow\n", .{self.coordinator.rank});
                    break :temporal_time_block;
                };
                const logical_delta = std.math.mul(i64, sequence_span, self.config.temporal_sequence_tick) catch {
                    std.debug.print("[Rank {d}] WARN: temporal logical_delta mul overflow\n", .{self.coordinator.rank});
                    break :temporal_time_block;
                };
                const new_time = std.math.add(i64, self.temporal_logical_time, logical_delta) catch {
                    std.debug.print("[Rank {d}] WARN: temporal_logical_time add overflow, resetting\n", .{self.coordinator.rank});
                    self.temporal_graph.advanceTime(logical_delta);
                    self.temporal_logical_time = 0;
                    break :temporal_time_block;
                };
                self.temporal_graph.advanceTime(logical_delta);
                self.temporal_logical_time = new_time;
            }
        }

        self.signal_engine.propagateStep() catch |err| {
            std.debug.print("[Rank {d}] WARN: signal_engine.propagateStep failed: {}\n", .{ self.coordinator.rank, err });
        };

        if (!self.training_variable_created) {
            _ = self.z_runtime.createVariable(self.config.training_variable_name, null) catch |err| {
                std.debug.print("[Rank {d}] WARN: z_runtime.createVariable failed: {}\n", .{ self.coordinator.rank, err });
                return;
            };
            self.training_variable_created = true;
        }
    }

    pub const StepResult = struct {
        loss: f32,
        reconstruction_loss: f32,
        source_rms: f32,
        sample_weight: f64,
    };

    fn shouldRunRelationalPass(self: *DistributedTrainerFuthark) bool {
        const interval: u64 = std.math.cast(u64, self.config.relational_pass_interval) orelse return false;
        if (interval == 0) return false;
        const completed_step = std.math.add(u64, self.global_step, 1) catch return false;
        return completed_step % interval == 0;
    }

    fn reduceEmbeddingGradientInPlace(
        self: *DistributedTrainerFuthark,
        embedding: *accel.EmbeddingAccelerator,
        local_fraction: f32,
    ) !void {
        if (self.coordinator.world_size <= 1) return;
        if (!std.math.isFinite(local_fraction) or local_fraction < 0.0 or local_fraction > 1.0) return TrainerError.InvalidReductionWeight;
        const scale = try checkedF32ToF16(local_fraction);
        try embedding.scaleGradient(scale);
        try self.accelerator.sync();
        const gradient_buffer = try embedding.getGradientDevicePtr();
        try self.coordinator.allReduceFloat16(gradient_buffer.ptr, gradient_buffer.ptr, gradient_buffer.count);
        try self.coordinator.synchronize();
    }

    fn reduceLayerStateInPlace(
        self: *DistributedTrainerFuthark,
        local_fraction: f32,
    ) !void {
        if (self.coordinator.world_size <= 1) return;
        if (!std.math.isFinite(local_fraction) or local_fraction < 0.0 or local_fraction > 1.0) return TrainerError.InvalidReductionWeight;
        const scale = try checkedF32ToF16(local_fraction);
        const kinds = [_]accel.WeightKind{ .weights_s, .weights_t, .velocity_s, .velocity_t };

        var layer_index: usize = 0;
        while (layer_index < self.num_layers) : (layer_index += 1) {
            for (kinds) |kind| {
                try self.accelerator.scaleLayerMatrix(layer_index, kind, scale);
            }
        }
        try self.accelerator.sync();

        layer_index = 0;
        while (layer_index < self.num_layers) : (layer_index += 1) {
            for (kinds) |kind| {
                const buffer = try self.accelerator.getLayerDevicePtr(layer_index, kind);
                try self.coordinator.allReduceFloat16(buffer.ptr, buffer.ptr, buffer.count);
            }
        }
        try self.coordinator.synchronize();
    }

    fn updateSourceFromInputDelta(
        self: *DistributedTrainerFuthark,
        flat_input_tokens: []const u32,
        real_sequence_lengths: []const usize,
        input_delta: *FutharkArray3DF16,
        local_fraction: f32,
        learning_rate: f16,
    ) !void {
        if (self.gpu_embedding == null or flat_input_tokens.len == 0) return;
        const embedding = &self.gpu_embedding.?;
        const clip_norm = if (self.config.use_normalized_gradient_flow) self.config.gradient_clip_norm else 0.0;
        try embedding.backwardPaddedAccumulate(
            flat_input_tokens,
            real_sequence_lengths,
            input_delta,
            clip_norm,
        );
        try self.accelerator.sync();
        try self.reduceEmbeddingGradientInPlace(embedding, local_fraction);
        try embedding.applyUpdate(learning_rate);
        try self.accelerator.sync();

        if (self.config.spectral_iterations > 0) {
            const embedding_size = std.math.mul(usize, embedding.vocab_size, embedding.dim) catch return TrainerError.ValueOverflow;
            const spectral_stride: u64 = if (embedding_size > 10_000_000) 100 else 10;
            const completed_step = std.math.add(u64, self.global_step, 1) catch return TrainerError.ValueOverflow;
            if (completed_step % spectral_stride == 0) {
                try self.applyEmbeddingSpectralNormalization();
            }
        }
    }

    pub fn trainStepFuthark(self: *DistributedTrainerFuthark, batch: [][]const u8) !StepResult {
        var token_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator);
        defer {
            for (token_lists.items) |*list| list.deinit();
            token_lists.deinit();
        }

        for (batch) |text| {
            var token_list = std.ArrayList(u32).init(self.allocator);
            self.tokenizer.encode(text, &token_list) catch |err| {
                token_list.deinit();
                std.debug.print("[Rank {d}] WARN: tokenizer.encode failed: {} (skipping)\n", .{ self.coordinator.rank, err });
                continue;
            };
            token_lists.append(token_list) catch |err| {
                token_list.deinit();
                return err;
            };
        }

        var local_active_samples: u64 = 0;
        for (token_lists.items) |list| {
            if (list.items.len >= 2) local_active_samples = try std.math.add(u64, local_active_samples, 1);
        }
        const global_active_samples = try self.allReduceSumU64(local_active_samples);
        if (global_active_samples == 0) return StepResult{ .loss = 0.0, .reconstruction_loss = 0.0, .source_rms = 0.0, .sample_weight = 0.0 };

        const maximum_sequence_length = try self.getMaximumSequenceLength();
        var local_maximum_prediction_length: u64 = 0;
        for (token_lists.items) |*list| {
            const max_token_count = try std.math.add(usize, maximum_sequence_length, 1);
            if (list.items.len > max_token_count) list.shrinkRetainingCapacity(max_token_count);
            if (list.items.len >= 2) {
                const prediction_length: u64 = @intCast(list.items.len - 1);
                if (prediction_length > local_maximum_prediction_length) local_maximum_prediction_length = prediction_length;
            }
        }

        const maximum_prediction_length = try self.allReduceMaximumU64(local_maximum_prediction_length);
        if (maximum_prediction_length == 0) return StepResult{ .loss = 0.0, .reconstruction_loss = 0.0, .source_rms = 0.0, .sample_weight = 0.0 };
        const sequence_length = std.math.cast(usize, maximum_prediction_length) orelse return TrainerError.ValueOverflow;

        var local_token_count: u64 = 0;
        for (token_lists.items) |list| {
            if (list.items.len >= 2) local_token_count = try std.math.add(u64, local_token_count, @intCast(list.items.len - 1));
        }

        var active_lists = std.ArrayList(std.ArrayList(u32)).init(self.allocator);
        defer active_lists.deinit();
        for (token_lists.items) |list| {
            if (list.items.len >= 2) try active_lists.append(list);
        }
        if (active_lists.items.len == 0) return StepResult{ .loss = 0.0, .reconstruction_loss = 0.0, .source_rms = 0.0, .sample_weight = 0.0 };

        const effective_batch_size = active_lists.items.len;
        const real_sequence_lengths = try self.allocator.alloc(usize, effective_batch_size);
        defer self.allocator.free(real_sequence_lengths);
        for (active_lists.items, 0..) |token_list, index| {
            real_sequence_lengths[index] = @min(token_list.items.len - 1, sequence_length);
        }

        const flat_size = try std.math.mul(usize, effective_batch_size, sequence_length);
        const flat_input_tokens = try self.allocator.alloc(u32, flat_size);
        defer self.allocator.free(flat_input_tokens);
        const flat_target_tokens = try self.allocator.alloc(u32, flat_size);
        defer self.allocator.free(flat_target_tokens);
        @memset(flat_input_tokens, 0);
        @memset(flat_target_tokens, 0);

        for (active_lists.items, 0..) |token_list, batch_index| {
            const prediction_length = real_sequence_lengths[batch_index];
            var sequence_index: usize = 0;
            while (sequence_index < prediction_length) : (sequence_index += 1) {
                const flat_index = try std.math.add(
                    usize,
                    try std.math.mul(usize, batch_index, sequence_length),
                    sequence_index,
                );
                const input_token = token_list.items[sequence_index];
                const target_token = token_list.items[sequence_index + 1];
                if (self.gpu_embedding) |embedding| {
                    if (@as(usize, input_token) >= embedding.vocab_size or @as(usize, target_token) >= embedding.vocab_size) return TrainerError.TokenIndexOutOfRange;
                }
                flat_input_tokens[flat_index] = input_token;
                flat_target_tokens[flat_index] = target_token;
            }
        }

        const BatchTensors = struct {
            inputs: FutharkArray3DF16,
            targets: FutharkArray3DF16,
        };

        if (self.config.shuffle_target_control) {
            var permute_index: usize = flat_target_tokens.len;
            while (permute_index > 1) {
                permute_index -= 1;
                self.shuffle_control_state = self.shuffle_control_state *% 6364136223846793005 +% 1442695040888963407;
                const draw: usize = @intCast((self.shuffle_control_state >> 33) % @as(u64, @intCast(permute_index + 1)));
                const swap = flat_target_tokens[permute_index];
                flat_target_tokens[permute_index] = flat_target_tokens[draw];
                flat_target_tokens[draw] = swap;
            }
        }

        var tensors = if (self.gpu_embedding) |*embedding| embedding_block: {
            var inputs = try embedding.forwardPadded(flat_input_tokens, real_sequence_lengths, sequence_length);
            errdefer inputs.free(&self.accelerator.ctx);
            const targets = if (self.target_source) |*frozen_source|
                try frozen_source.forwardPadded(flat_target_tokens, real_sequence_lengths, sequence_length)
            else
                try embedding.forwardPadded(flat_target_tokens, real_sequence_lengths, sequence_length);
            break :embedding_block BatchTensors{ .inputs = inputs, .targets = targets };
        } else one_hot_block: {
            const batch_rows = try std.math.mul(usize, effective_batch_size, sequence_length);
            const data_elements = try std.math.mul(usize, batch_rows, self.model_dim);
            const data_size = try std.math.mul(usize, data_elements, @sizeOf(f16));
            var pinned_input = try PinnedMemory.alloc(data_size);
            defer pinned_input.free();
            var pinned_target = try PinnedMemory.alloc(data_size);
            defer pinned_target.free();
            const input_data = pinned_input.asSlice(f16) orelse return TrainerError.AllocationFailed;
            const target_data = pinned_target.asSlice(f16) orelse return TrainerError.AllocationFailed;
            if (input_data.len != data_elements or target_data.len != data_elements) return TrainerError.InvalidPinnedMemorySize;
            @memset(input_data, @as(f16, 0.0));
            @memset(target_data, @as(f16, 0.0));

            for (active_lists.items, 0..) |token_list, batch_index| {
                const prediction_length = real_sequence_lengths[batch_index];
                var sequence_index: usize = 0;
                while (sequence_index < prediction_length) : (sequence_index += 1) {
                    const row_index = try std.math.add(
                        usize,
                        try std.math.mul(usize, batch_index, sequence_length),
                        sequence_index,
                    );
                    const input_token: usize = @intCast(token_list.items[sequence_index]);
                    const target_token: usize = @intCast(flat_target_tokens[row_index]);
                    if (input_token >= self.model_dim or target_token >= self.model_dim) return TrainerError.TokenIndexOutOfRange;
                    const base_index = try std.math.mul(usize, row_index, self.model_dim);
                    const input_index = try std.math.add(usize, base_index, input_token);
                    const target_index = try std.math.add(usize, base_index, target_token);
                    if (input_index >= input_data.len or target_index >= target_data.len) return TrainerError.IndexOutOfBounds;
                    input_data[input_index] = 1.0;
                    target_data[target_index] = 1.0;
                }
            }

            var inputs = try FutharkArray3DF16.newFromFlat(
                &self.accelerator.ctx,
                input_data,
                effective_batch_size,
                sequence_length,
                self.model_dim,
            );
            errdefer inputs.free(&self.accelerator.ctx);
            const targets = try FutharkArray3DF16.newFromFlat(
                &self.accelerator.ctx,
                target_data,
                effective_batch_size,
                sequence_length,
                self.model_dim,
            );
            break :one_hot_block BatchTensors{ .inputs = inputs, .targets = targets };
        };
        defer tensors.inputs.free(&self.accelerator.ctx);
        defer tensors.targets.free(&self.accelerator.ctx);

        const global_token_count = if (self.coordinator.world_size > 1)
            try self.allReduceSumU64(local_token_count)
        else
            local_token_count;
        if (global_token_count == 0) return StepResult{ .loss = 0.0, .reconstruction_loss = 0.0, .source_rms = 0.0, .sample_weight = 0.0 };
        const local_fraction: f32 = if (self.coordinator.world_size > 1)
            @floatCast(
                @as(f64, @floatFromInt(local_token_count)) /
                    @as(f64, @floatFromInt(global_token_count)),
            )
        else
            1.0;
        if (!std.math.isFinite(local_fraction) or local_fraction < 0.0 or local_fraction > 1.0) return TrainerError.InvalidReductionWeight;

        const effective_learning_rate = self.learning_rate * self.vpu_lr_scale;
        const clamped_learning_rate: f32 = if (effective_learning_rate > 65504.0)
            65504.0
        else if (effective_learning_rate <= 0.0)
            self.learning_rate
        else
            effective_learning_rate;
        const learning_rate = try checkedF32ToF16(clamped_learning_rate);
        const momentum = try checkedF32ToF16(self.momentum);
        const completed_step = std.math.add(u64, self.global_step, 1) catch return TrainerError.ValueOverflow;
        const report_progress = self.coordinator.isRoot() and (completed_step <= 50 or completed_step % 10 == 0);
        const step_t0_ns = std.time.nanoTimestamp();
        if (report_progress) {
            std.debug.print(
                "[Rank 0] Step {d} start batch={d} seq={d} dim={d} layers={d} tokens={d}\n",
                .{ completed_step, effective_batch_size, sequence_length, self.model_dim, self.num_layers, local_token_count },
            );
        }

        const step_for_phase = self.global_step;
        const effective_reconstruction_alpha: f32 = blk: {
            if (self.config.phase_a_steps > 0 and step_for_phase < self.config.phase_a_steps) {
                break :blk 1.0;
            }
            const ramp_span = self.config.phase_b_steps;
            if (ramp_span == 0) break :blk self.config.reconstruction_alpha;
            const ramp_end = std.math.add(u64, self.config.phase_a_steps, ramp_span) catch {
                break :blk self.config.reconstruction_alpha;
            };
            if (step_for_phase >= ramp_end) break :blk self.config.reconstruction_alpha;
            const elapsed = step_for_phase - self.config.phase_a_steps;
            const progress = @as(f32, @floatFromInt(elapsed)) / @as(f32, @floatFromInt(ramp_span));
            const value = 1.0 - progress * (1.0 - self.config.reconstruction_alpha);
            if (!std.math.isFinite(value)) break :blk self.config.reconstruction_alpha;
            break :blk value;
        };
        const clamped_reconstruction_alpha = @max(@as(f32, 0.0), @min(@as(f32, 1.0), effective_reconstruction_alpha));
        const reconstruction_alpha_f16 = try checkedF32ToF16(clamped_reconstruction_alpha);

        const in_phase_a = self.config.phase_a_steps > 0 and step_for_phase < self.config.phase_a_steps;
        const forward_scale_value: f32 = if (in_phase_a) 0.0 else 1.0;
        const forward_scale_f16 = try checkedF32ToF16(forward_scale_value);

        var training_result = try self.accelerator.trainingStep(
            &tensors.inputs,
            &tensors.targets,
            real_sequence_lengths,
            learning_rate,
            momentum,
            reconstruction_alpha_f16,
            forward_scale_f16,
        );
        defer training_result.input_delta.free(&self.accelerator.ctx);
        try self.accelerator.sync();

        const step_backward_ns = std.time.nanoTimestamp() - step_t0_ns;
        if (report_progress) std.debug.print("[Rank 0] Step {d} RSF/OFTB backward complete dt={d}ms\n", .{ completed_step, @divTrunc(step_backward_ns, 1_000_000) });

        try self.updateSourceFromInputDelta(
            flat_input_tokens,
            real_sequence_lengths,
            &training_result.input_delta,
            local_fraction,
            learning_rate,
        );
        try self.reduceLayerStateInPlace(local_fraction);

        if (report_progress) std.debug.print("[Rank 0] Step {d} parameter reduction complete\n", .{completed_step});

        if (self.shouldRunRelationalPass()) {
            if (self.coordinator.isRoot()) std.debug.print("[Rank 0] Step {d} relational pass start\n", .{completed_step});
            self.runCoreRelationalPass(active_lists.items) catch |err| {
                std.debug.print("[Rank {d}] WARN: runCoreRelationalPass failed: {}\n", .{ self.coordinator.rank, err });
            };
            if (self.coordinator.isRoot()) std.debug.print("[Rank 0] Step {d} relational pass complete\n", .{completed_step});
        }

        const local_loss: f32 = @floatCast(training_result.loss);
        if (!std.math.isFinite(local_loss)) return TrainerError.InvalidLoss;
        const local_reconstruction_loss_raw: f32 = @floatCast(training_result.reconstruction_loss);
        const local_reconstruction_loss: f32 = if (std.math.isFinite(local_reconstruction_loss_raw))
            local_reconstruction_loss_raw
        else
            0.0;

        var reduced_pair = [2]f32{
            local_loss * local_fraction,
            local_reconstruction_loss * local_fraction,
        };
        var reduced_loss = local_loss;
        var reduced_reconstruction_loss = local_reconstruction_loss;
        if (self.coordinator.world_size > 1) {
            try self.allReduceFloat32Values(reduced_pair[0..]);
            reduced_loss = reduced_pair[0];
            reduced_reconstruction_loss = reduced_pair[1];
        }
        if (!std.math.isFinite(reduced_loss)) return TrainerError.InvalidLoss;
        if (!std.math.isFinite(reduced_reconstruction_loss)) reduced_reconstruction_loss = 0.0;

        const local_source_rms: f32 = if (self.gpu_embedding) |*embedding|
            embedding.sourceRootMeanSquare() catch 0.0
        else
            0.0;

        if (report_progress) std.debug.print(
            "[Rank 0] Step {d} complete loss={d:.6} recon={d:.6} alpha={d:.4} src_rms={d:.6}\n",
            .{ completed_step, reduced_loss, reduced_reconstruction_loss, clamped_reconstruction_alpha, local_source_rms },
        );
        return StepResult{
            .loss = reduced_loss,
            .reconstruction_loss = reduced_reconstruction_loss,
            .source_rms = local_source_rms,
            .sample_weight = @as(f64, @floatFromInt(local_token_count)),
        };
    }

    fn makeTemporaryPath(
        self: *DistributedTrainerFuthark,
        path: []const u8,
        suffix: []const u8,
    ) ![]u8 {
        const timestamp = std.time.nanoTimestamp();
        return std.fmt.allocPrint(self.allocator, "{s}.{s}.{d}.{d}.tmp", .{ path, suffix, self.coordinator.rank, timestamp });
    }

    fn makeTmpFilePath(
        self: *DistributedTrainerFuthark,
        suffix: []const u8,
    ) ![]u8 {
        const timestamp = std.time.nanoTimestamp();
        return std.fmt.allocPrint(self.allocator, "/tmp/jaide_{s}_{d}_{d}.tmp", .{ suffix, self.coordinator.rank, timestamp });
    }

    fn readWholeFile(self: *DistributedTrainerFuthark, path: []const u8, max_size: usize) ![]u8 {
        const file = try openReadFile(path);
        defer file.close();
        const length_u64 = try file.getEndPos();
        const length = std.math.cast(usize, length_u64) orelse return TrainerError.FileTooLarge;
        if (length == 0 or length > max_size) return TrainerError.FileTooLarge;
        const data = try self.allocator.alloc(u8, length);
        errdefer self.allocator.free(data);
        try file.reader().readNoEof(data);
        return data;
    }

    fn writeNsirGraph(writer: anytype, graph: *SelfSimilarRelationalGraph, config: TrainerConfig) !void {
        const node_count = try safeUsizeToU32(graph.nodes.count());
        try writer.writeInt(u32, node_count, .little);
        var node_iter = graph.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node = entry.value_ptr.*;
            if (node.id.len > config.max_id_length) return TrainerError.NodeIdTooLong;
            if (node.data.len > config.max_node_data_length) return TrainerError.NodeDataTooLong;
            if (!std.math.isFinite(node.qubit.a.re) or !std.math.isFinite(node.qubit.a.im) or
                !std.math.isFinite(node.qubit.b.re) or !std.math.isFinite(node.qubit.b.im) or
                !std.math.isFinite(node.phase)) return TrainerError.InvalidQuantumState;
            const id_len = try safeUsizeToU32(node.id.len);
            try writer.writeInt(u32, id_len, .little);
            try writer.writeAll(node.id);
            const data_len = try safeUsizeToU32(node.data.len);
            try writer.writeInt(u32, data_len, .little);
            try writer.writeAll(node.data);
            try writeF64(writer, node.qubit.a.re);
            try writeF64(writer, node.qubit.a.im);
            try writeF64(writer, node.qubit.b.re);
            try writeF64(writer, node.qubit.b.im);
            try writeF64(writer, node.phase);
        }

        const edge_key_count = try safeUsizeToU32(graph.edges.count());
        if (edge_key_count > config.max_edge_group_count) return TrainerError.EdgeCountTooLarge;
        try writer.writeInt(u32, edge_key_count, .little);
        var edge_iter = graph.edges.iterator();
        var total_edges: u64 = 0;
        while (edge_iter.next()) |entry| {
            const key = entry.key_ptr.*;
            const edge_list = entry.value_ptr.*;
            if (key.source.len > config.max_id_length) return TrainerError.NodeIdTooLong;
            if (key.target.len > config.max_id_length) return TrainerError.NodeIdTooLong;
            if (edge_list.items.len > config.max_edge_group_count) return TrainerError.EdgeCountTooLarge;
            total_edges = try std.math.add(u64, total_edges, @as(u64, @intCast(edge_list.items.len)));
            if (total_edges > @as(u64, config.max_edge_group_count)) return TrainerError.EdgeCountTooLarge;
            const src_len = try safeUsizeToU32(key.source.len);
            try writer.writeInt(u32, src_len, .little);
            try writer.writeAll(key.source);
            const tgt_len = try safeUsizeToU32(key.target.len);
            try writer.writeInt(u32, tgt_len, .little);
            try writer.writeAll(key.target);
            const count = try safeUsizeToU32(edge_list.items.len);
            try writer.writeInt(u32, count, .little);
            for (edge_list.items) |edge| {
                if (!std.math.isFinite(edge.weight)) return TrainerError.InvalidEdgeWeight;
                if (!std.math.isFinite(edge.quantum_correlation.re) or !std.math.isFinite(edge.quantum_correlation.im)) return TrainerError.InvalidEdgeWeight;
                if (!std.math.isFinite(edge.fractal_dimension)) return TrainerError.InvalidEdgeWeight;
                try writeF64(writer, edge.weight);
                try writer.writeByte(@intFromEnum(edge.quality));
                try writeF64(writer, edge.quantum_correlation.re);
                try writeF64(writer, edge.quantum_correlation.im);
                try writeF64(writer, edge.fractal_dimension);
            }
        }
    }

    fn readNsirGraph(
        allocator: std.mem.Allocator,
        reader: anytype,
        config: TrainerConfig,
    ) !*SelfSimilarRelationalGraph {
        const graph_ptr = try allocator.create(SelfSimilarRelationalGraph);
        var graph_ptr_committed = false;
        errdefer if (!graph_ptr_committed) allocator.destroy(graph_ptr);
        graph_ptr.* = try SelfSimilarRelationalGraph.init(allocator);
        var graph_committed = false;
        errdefer if (!graph_committed) graph_ptr.deinit();

        const node_count = try reader.readInt(u32, .little);
        if (@as(u64, node_count) > @as(u64, config.max_node_count)) return TrainerError.NodeDataTooLong;
        var ni: u32 = 0;
        while (ni < node_count) : (ni += 1) {
            const id_len = try reader.readInt(u32, .little);
            if (id_len > config.max_id_length) return TrainerError.NodeIdTooLong;
            const id = try allocator.alloc(u8, id_len);
            defer allocator.free(id);
            try reader.readNoEof(id);

            const data_len = try reader.readInt(u32, .little);
            if (data_len > config.max_node_data_length) return TrainerError.NodeDataTooLong;
            const data_bytes = try allocator.alloc(u8, data_len);
            defer allocator.free(data_bytes);
            try reader.readNoEof(data_bytes);

            const a_re = try readF64(reader);
            const a_im = try readF64(reader);
            const b_re = try readF64(reader);
            const b_im = try readF64(reader);
            const phase = try readF64(reader);
            if (!std.math.isFinite(a_re) or !std.math.isFinite(a_im) or !std.math.isFinite(b_re) or !std.math.isFinite(b_im) or !std.math.isFinite(phase)) return TrainerError.InvalidQuantumState;

            const qubit = nsir.Qubit.init(
                std.math.Complex(f64).init(a_re, a_im),
                std.math.Complex(f64).init(b_re, b_im),
            );
            const node = try nsir.Node.init(graph_ptr.allocator, id, data_bytes, qubit, phase);
            try graph_ptr.addNode(node);
        }

        const edge_key_count = try reader.readInt(u32, .little);
        if (edge_key_count > config.max_edge_group_count) return TrainerError.EdgeCountTooLarge;
        var ei: u32 = 0;
        var total_edges: u64 = 0;
        while (ei < edge_key_count) : (ei += 1) {
            const src_len = try reader.readInt(u32, .little);
            if (src_len > config.max_id_length) return TrainerError.NodeIdTooLong;
            const source = try allocator.alloc(u8, src_len);
            defer allocator.free(source);
            try reader.readNoEof(source);

            const tgt_len = try reader.readInt(u32, .little);
            if (tgt_len > config.max_id_length) return TrainerError.NodeIdTooLong;
            const target = try allocator.alloc(u8, tgt_len);
            defer allocator.free(target);
            try reader.readNoEof(target);

            const count = try reader.readInt(u32, .little);
            if (count > config.max_edge_group_count) return TrainerError.EdgeCountTooLarge;
            total_edges = try std.math.add(u64, total_edges, @as(u64, count));
            if (total_edges > @as(u64, config.max_edge_group_count)) return TrainerError.EdgeCountTooLarge;

            var k: u32 = 0;
            while (k < count) : (k += 1) {
                const weight = try readF64(reader);
                const quality_byte = try reader.readByte();
                const quality: nsir.EdgeQuality = std.meta.intToEnum(nsir.EdgeQuality, quality_byte) catch return TrainerError.InvalidQualityByte;
                const qc_re = try readF64(reader);
                const qc_im = try readF64(reader);
                const fd = try readF64(reader);
                if (!std.math.isFinite(weight) or !std.math.isFinite(qc_re) or !std.math.isFinite(qc_im) or !std.math.isFinite(fd)) return TrainerError.InvalidEdgeWeight;
                const edge = try nsir.Edge.init(
                    graph_ptr.allocator,
                    source,
                    target,
                    quality,
                    weight,
                    std.math.Complex(f64).init(qc_re, qc_im),
                    fd,
                );
                try graph_ptr.addEdge(source, target, edge);
            }
        }

        graph_ptr_committed = true;
        graph_committed = true;
        return graph_ptr;
    }

    pub fn saveCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        try self.coordinator.synchronize();

        if (!self.coordinator.isRoot()) {
            const global_save_status = try self.allReduceMaximumU64(0);
            if (global_save_status != 0) return TrainerError.CheckpointSaveFailed;
            return;
        }

        var broadcast_error: u64 = 0;
        errdefer {
            _ = self.allReduceMaximumU64(1) catch {};
        }

        try self.accelerator.sync();

        const tokenizer_tmp = try self.makeTemporaryPath(path, "tokenizer");
        defer self.allocator.free(tokenizer_tmp);
        var tokenizer_tmp_committed = false;
        defer if (!tokenizer_tmp_committed) deletePath(tokenizer_tmp);

        try self.tokenizer.saveVocab(tokenizer_tmp);

        const tokenizer_data = try self.readWholeFile(tokenizer_tmp, self.config.max_tokenizer_file_size);
        defer self.allocator.free(tokenizer_data);

        const checkpoint_tmp = try self.makeTemporaryPath(path, "checkpoint");
        defer self.allocator.free(checkpoint_tmp);
        var checkpoint_committed = false;
        defer if (!checkpoint_committed) deletePath(checkpoint_tmp);

        {
            const file = try createWriteFile(checkpoint_tmp);
            var file_closed = false;
            defer if (!file_closed) file.close();

            var bw = std.io.bufferedWriter(file.writer());
            const BW = @TypeOf(bw);
            var crc_writer = CrcTrackingWriter(BW.Writer){
                .inner = bw.writer(),
                .crc = std.hash.Crc32.init(),
            };
            const writer = crc_writer.writer();

            try writer.writeAll(CHECKPOINT_MAGIC[0..]);
            try writer.writeInt(u32, self.config.checkpoint_version, .little);
            try writer.writeInt(u64, self.global_step, .little);
            try writer.writeInt(u64, @as(u64, @intCast(self.model_dim)), .little);
            try writer.writeInt(u64, @as(u64, @intCast(self.num_layers)), .little);
            try writer.writeInt(u64, @as(u64, @intCast(self.vocab_size)), .little);
            try writer.writeInt(u64, @as(u64, @intCast(self.local_batch_size)), .little);
            try writeF32(writer, self.learning_rate);
            try writeF32(writer, self.momentum);

            const clip_min_f32: f32 = @floatCast(self.accelerator.clip_min);
            const clip_max_f32: f32 = @floatCast(self.accelerator.clip_max);
            if (!std.math.isFinite(clip_min_f32) or !std.math.isFinite(clip_max_f32) or !(clip_min_f32 < clip_max_f32)) return TrainerError.InvalidClipRange;

            var li_save: usize = 0;
            while (li_save < self.num_layers) : (li_save += 1) {
                const ws = try self.readLayerMatrix(li_save, .weights_s);
                defer self.allocator.free(ws);
                for (ws) |w| if (!std.math.isFinite(@as(f32, @floatCast(w)))) return TrainerError.InvalidWeightValue;
                try writer.writeInt(u64, @as(u64, ws.len), .little);
                for (ws) |w| try writeF32(writer, @floatCast(w));

                const wt = try self.readLayerMatrix(li_save, .weights_t);
                defer self.allocator.free(wt);
                for (wt) |w| if (!std.math.isFinite(@as(f32, @floatCast(w)))) return TrainerError.InvalidWeightValue;
                try writer.writeInt(u64, @as(u64, wt.len), .little);
                for (wt) |w| try writeF32(writer, @floatCast(w));

                const vs = try self.readLayerMatrix(li_save, .velocity_s);
                defer self.allocator.free(vs);
                for (vs) |v| if (!std.math.isFinite(@as(f32, @floatCast(v)))) return TrainerError.InvalidWeightValue;
                try writer.writeInt(u64, @as(u64, vs.len), .little);
                for (vs) |v| try writeF32(writer, @floatCast(v));

                const vt = try self.readLayerMatrix(li_save, .velocity_t);
                defer self.allocator.free(vt);
                for (vt) |v| if (!std.math.isFinite(@as(f32, @floatCast(v)))) return TrainerError.InvalidWeightValue;
                try writer.writeInt(u64, @as(u64, vt.len), .little);
                for (vt) |v| try writeF32(writer, @floatCast(v));
            }

            try writeF32(writer, clip_min_f32);
            try writeF32(writer, clip_max_f32);

            if (self.gpu_embedding) |*emb| {
                try writer.writeByte(1);
                try writer.writeInt(u64, @as(u64, emb.vocab_size), .little);
                try writer.writeInt(u64, @as(u64, emb.dim), .little);
                const total_elements = emb.vocab_size * emb.dim;
                const weight_f16_save = try self.allocator.alloc(f16, total_elements);
                defer self.allocator.free(weight_f16_save);
                if (futhark.futhark_values_f16_2d(self.accelerator.ctx.ctx, emb.weight.arr, @ptrCast(weight_f16_save.ptr)) != 0) return TrainerError.CheckpointSaveFailed;
                try self.accelerator.sync();
                try writer.writeInt(u64, @as(u64, total_elements), .little);
                for (weight_f16_save) |w| {
                    const wf32: f32 = @floatCast(w);
                    if (!std.math.isFinite(wf32)) return TrainerError.InvalidEmbeddingWeight;
                    _ = try checkedF32ToF16(wf32);
                    try writeF32(writer, wf32);
                }
                try writer.writeInt(u64, @as(u64, total_elements), .little);
                var zi: usize = 0;
                while (zi < total_elements) : (zi += 1) {
                    try writeF32(writer, 0.0);
                }
            } else {
                try writer.writeByte(0);
            }

            if (self.target_source) |*frozen_source| {
                try writer.writeByte(1);
                try writer.writeInt(u64, @as(u64, frozen_source.vocab_size), .little);
                try writer.writeInt(u64, @as(u64, frozen_source.dim), .little);
                const frozen_total = frozen_source.vocab_size * frozen_source.dim;
                const frozen_flat = try self.allocator.alloc(f16, frozen_total);
                defer self.allocator.free(frozen_flat);
                if (futhark.futhark_values_f16_2d(self.accelerator.ctx.ctx, frozen_source.weight.arr, @ptrCast(frozen_flat.ptr)) != 0) return TrainerError.CheckpointSaveFailed;
                try self.accelerator.sync();
                try writer.writeInt(u64, @as(u64, frozen_total), .little);
                for (frozen_flat) |value| {
                    const value_f32: f32 = @floatCast(value);
                    if (!std.math.isFinite(value_f32)) return TrainerError.InvalidEmbeddingWeight;
                    _ = try checkedF32ToF16(value_f32);
                    try writeF32(writer, value_f32);
                }
            } else {
                try writer.writeByte(0);
            }

            try writer.writeAll(self.knowledge_graph_nonce[0..]);

            try writeNsirGraph(writer, self.nsir_graph, self.config);
            try writeNsirGraph(writer, self.knowledge_nsir_graph, self.config);

            try writer.writeInt(u64, @as(u64, tokenizer_data.len), .little);
            try writer.writeAll(tokenizer_data);
            try writer.writeInt(u32, CHECKPOINT_TRAILER, .little);

            const final_crc = crc_writer.crc.final();
            try bw.flush();
            try bw.writer().writeInt(u32, final_crc, .little);
            try bw.flush();
            try file.sync();
            file.close();
            file_closed = true;
        }

        try renamePath(checkpoint_tmp, path);
        checkpoint_committed = true;
        tokenizer_tmp_committed = true;
        deletePath(tokenizer_tmp);
        syncContainingDirectory(path);

        broadcast_error = 0;
        const global_save_status = try self.allReduceMaximumU64(broadcast_error);
        if (global_save_status != 0) return TrainerError.CheckpointSaveFailed;

        std.debug.print("Checkpoint saved to {s} at step {d}\n", .{ path, self.global_step });
    }

    fn readCheckpointF16Array(
        self: *DistributedTrainerFuthark,
        reader: anytype,
        expected_length: usize,
    ) ![]f16 {
        const saved_length_u64 = try reader.readInt(u64, .little);
        const saved_length = std.math.cast(usize, saved_length_u64) orelse return TrainerError.InvalidWeightsShape;
        if (saved_length != expected_length) return TrainerError.InvalidWeightsShape;
        const values = try self.allocator.alloc(f16, saved_length);
        errdefer self.allocator.free(values);
        var i: usize = 0;
        while (i < saved_length) : (i += 1) {
            const v = try readF32(reader);
            if (!std.math.isFinite(v)) return TrainerError.InvalidWeightValue;
            values[i] = try checkedF32ToF16(v);
        }
        return values;
    }

    pub fn loadCheckpoint(self: *DistributedTrainerFuthark, path: []const u8) !void {
        const raw_data = blk: {
            const file = try openReadFile(path);
            defer file.close();
            const length_u64 = try file.getEndPos();
            const length = std.math.cast(usize, length_u64) orelse return TrainerError.FileTooLarge;
            if (length < 16) return TrainerError.CheckpointCorrupted;
            const data = try self.allocator.alloc(u8, length);
            errdefer self.allocator.free(data);
            try file.reader().readNoEof(data);
            break :blk data;
        };
        defer self.allocator.free(raw_data);

        if (raw_data.len < 4) return TrainerError.CheckpointCorrupted;
        const stored_crc = std.mem.readInt(u32, raw_data[raw_data.len - 4 ..][0..4], .little);
        var crc_check = std.hash.Crc32.init();
        crc_check.update(raw_data[0 .. raw_data.len - 4]);
        if (crc_check.final() != stored_crc) return TrainerError.CheckpointCorrupted;

        var fbs = std.io.fixedBufferStream(raw_data[0 .. raw_data.len - 4]);
        const reader = fbs.reader();

        var magic_buf: [8]u8 = undefined;
        try reader.readNoEof(magic_buf[0..]);
        if (!std.mem.eql(u8, magic_buf[0..], CHECKPOINT_MAGIC[0..])) return TrainerError.CheckpointMagicMismatch;

        const version = try reader.readInt(u32, .little);
        if (version != self.config.checkpoint_version) return TrainerError.CheckpointVersionMismatch;

        const saved_global_step = try reader.readInt(u64, .little);
        const saved_model_dim_u64 = try reader.readInt(u64, .little);
        const saved_num_layers_u64 = try reader.readInt(u64, .little);
        const saved_vocab_size_u64 = try reader.readInt(u64, .little);
        const saved_local_batch_size_u64 = try reader.readInt(u64, .little);
        const saved_learning_rate = try readF32(reader);
        const saved_momentum = try readF32(reader);

        const saved_model_dim = std.math.cast(usize, saved_model_dim_u64) orelse return TrainerError.ModelDimMismatch;
        const saved_num_layers = std.math.cast(usize, saved_num_layers_u64) orelse return TrainerError.NumLayersMismatch;
        const saved_vocab_size = std.math.cast(usize, saved_vocab_size_u64) orelse return TrainerError.VocabSizeMismatch;
        const saved_local_batch_size = std.math.cast(usize, saved_local_batch_size_u64) orelse return TrainerError.InvalidBatchSize;

        if (saved_model_dim != self.model_dim) return TrainerError.ModelDimMismatch;
        if (saved_num_layers != self.num_layers) return TrainerError.NumLayersMismatch;
        if (saved_vocab_size == 0 or saved_vocab_size > self.config.max_distributed_integer) return TrainerError.VocabSizeMismatch;
        if (saved_local_batch_size == 0 or saved_local_batch_size > self.config.max_local_batch_size) return TrainerError.InvalidBatchSize;
        try validateHyperparameters(saved_learning_rate, saved_momentum);

        const half = self.model_dim / 2;
        const columns = try std.math.add(usize, half, 1);
        const expected_length = try std.math.mul(usize, half, columns);

        const snapshots = try self.allocator.alloc(LayerSnapshot, self.num_layers);
        var snapshots_committed = false;
        errdefer if (!snapshots_committed) {
            for (snapshots) |snapshot| {
                if (snapshot.weights_s.len > 0) self.allocator.free(snapshot.weights_s);
                if (snapshot.weights_t.len > 0) self.allocator.free(snapshot.weights_t);
                if (snapshot.velocity_s.len > 0) self.allocator.free(snapshot.velocity_s);
                if (snapshot.velocity_t.len > 0) self.allocator.free(snapshot.velocity_t);
            }
            self.allocator.free(snapshots);
        };
        for (snapshots) |*snapshot| {
            snapshot.* = .{ .weights_s = &.{}, .weights_t = &.{}, .velocity_s = &.{}, .velocity_t = &.{} };
        }
        for (snapshots, 0..) |*snapshot, layer_index| {
            snapshot.weights_s = try self.readCheckpointF16Array(reader, expected_length);
            snapshot.weights_t = self.readCheckpointF16Array(reader, expected_length) catch |err| {
                self.allocator.free(snapshot.weights_s);
                snapshot.weights_s = &.{};
                var idx: usize = 0;
                while (idx < layer_index) : (idx += 1) {
                    if (snapshots[idx].weights_s.len > 0) self.allocator.free(snapshots[idx].weights_s);
                    if (snapshots[idx].weights_t.len > 0) self.allocator.free(snapshots[idx].weights_t);
                    if (snapshots[idx].velocity_s.len > 0) self.allocator.free(snapshots[idx].velocity_s);
                    if (snapshots[idx].velocity_t.len > 0) self.allocator.free(snapshots[idx].velocity_t);
                }
                return err;
            };
            snapshot.velocity_s = self.readCheckpointF16Array(reader, expected_length) catch |err| {
                self.allocator.free(snapshot.weights_s);
                self.allocator.free(snapshot.weights_t);
                snapshot.weights_s = &.{};
                snapshot.weights_t = &.{};
                var idx: usize = 0;
                while (idx < layer_index) : (idx += 1) {
                    if (snapshots[idx].weights_s.len > 0) self.allocator.free(snapshots[idx].weights_s);
                    if (snapshots[idx].weights_t.len > 0) self.allocator.free(snapshots[idx].weights_t);
                    if (snapshots[idx].velocity_s.len > 0) self.allocator.free(snapshots[idx].velocity_s);
                    if (snapshots[idx].velocity_t.len > 0) self.allocator.free(snapshots[idx].velocity_t);
                }
                return err;
            };
            snapshot.velocity_t = self.readCheckpointF16Array(reader, expected_length) catch |err| {
                self.allocator.free(snapshot.weights_s);
                self.allocator.free(snapshot.weights_t);
                self.allocator.free(snapshot.velocity_s);
                snapshot.weights_s = &.{};
                snapshot.weights_t = &.{};
                snapshot.velocity_s = &.{};
                var idx: usize = 0;
                while (idx < layer_index) : (idx += 1) {
                    if (snapshots[idx].weights_s.len > 0) self.allocator.free(snapshots[idx].weights_s);
                    if (snapshots[idx].weights_t.len > 0) self.allocator.free(snapshots[idx].weights_t);
                    if (snapshots[idx].velocity_s.len > 0) self.allocator.free(snapshots[idx].velocity_s);
                    if (snapshots[idx].velocity_t.len > 0) self.allocator.free(snapshots[idx].velocity_t);
                }
                return err;
            };
        }

        const clip_min_f32 = try readF32(reader);
        const clip_max_f32 = try readF32(reader);
        if (!std.math.isFinite(clip_min_f32) or !std.math.isFinite(clip_max_f32) or !(clip_min_f32 < clip_max_f32)) return TrainerError.InvalidClipRange;
        const clip_min = try checkedF32ToF16(clip_min_f32);
        const clip_max = try checkedF32ToF16(clip_max_f32);
        if (!(@as(f32, @floatCast(clip_min)) < @as(f32, @floatCast(clip_max)))) return TrainerError.ConvertPrecisionLoss;

        const has_embedding = try reader.readByte();
        if (has_embedding > 1) return TrainerError.InvalidCheckpointEmbeddingFlag;

        var pending_emb_weight: ?[]f16 = null;
        var pending_emb_vocab: usize = 0;
        var pending_emb_dim: usize = 0;
        defer if (pending_emb_weight) |w| self.allocator.free(w);

        if (has_embedding == 1) {
            const embedding_vocab_u64 = try reader.readInt(u64, .little);
            const embedding_dim_u64 = try reader.readInt(u64, .little);
            const embedding_vocab = std.math.cast(usize, embedding_vocab_u64) orelse return TrainerError.VocabSizeMismatch;
            const embedding_dim = std.math.cast(usize, embedding_dim_u64) orelse return TrainerError.ModelDimMismatch;
            if (embedding_vocab != saved_vocab_size) return TrainerError.VocabSizeMismatch;
            if (embedding_dim != self.model_dim) return TrainerError.ModelDimMismatch;
            if (embedding_vocab > self.config.max_distributed_integer) return TrainerError.VocabSizeMismatch;
            if (self.vocab_size > self.model_dim and has_embedding == 0) return TrainerError.TokenIndexOutOfRange;

            const total_w = embedding_vocab * embedding_dim;
            const w_len_u64 = try reader.readInt(u64, .little);
            const w_len = std.math.cast(usize, w_len_u64) orelse return TrainerError.InvalidEmbeddingShape;
            if (w_len != total_w) return TrainerError.InvalidEmbeddingShape;
            const weight_buf = try self.allocator.alloc(f16, w_len);
            var weight_buf_committed = false;
            errdefer if (!weight_buf_committed) self.allocator.free(weight_buf);
            for (weight_buf) |*value| {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return TrainerError.InvalidEmbeddingWeight;
                if (v < -65504.0 or v > 65504.0) return TrainerError.InvalidEmbeddingWeight;
                value.* = @floatCast(v);
            }
            const vel_len_u64 = try reader.readInt(u64, .little);
            const vel_len = std.math.cast(usize, vel_len_u64) orelse return TrainerError.InvalidEmbeddingShape;
            if (vel_len != total_w) return TrainerError.InvalidEmbeddingShape;
            var vel_idx: usize = 0;
            while (vel_idx < vel_len) : (vel_idx += 1) {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return TrainerError.InvalidEmbeddingWeight;
            }
            pending_emb_weight = weight_buf;
            pending_emb_vocab = embedding_vocab;
            pending_emb_dim = embedding_dim;
            weight_buf_committed = true;
        }

        const has_target_source = try reader.readByte();
        if (has_target_source > 1) return TrainerError.InvalidCheckpointEmbeddingFlag;

        var pending_target_weight: ?[]f16 = null;
        var pending_target_vocab: usize = 0;
        var pending_target_dim: usize = 0;
        defer if (pending_target_weight) |w| self.allocator.free(w);

        if (has_target_source == 1) {
            const target_vocab_u64 = try reader.readInt(u64, .little);
            const target_dim_u64 = try reader.readInt(u64, .little);
            const target_vocab = std.math.cast(usize, target_vocab_u64) orelse return TrainerError.VocabSizeMismatch;
            const target_dim = std.math.cast(usize, target_dim_u64) orelse return TrainerError.ModelDimMismatch;
            if (target_vocab != saved_vocab_size) return TrainerError.VocabSizeMismatch;
            if (target_dim != self.model_dim) return TrainerError.ModelDimMismatch;
            if (target_vocab > self.config.max_distributed_integer) return TrainerError.VocabSizeMismatch;

            const target_total = target_vocab * target_dim;
            const target_len_u64 = try reader.readInt(u64, .little);
            const target_len = std.math.cast(usize, target_len_u64) orelse return TrainerError.InvalidEmbeddingShape;
            if (target_len != target_total) return TrainerError.InvalidEmbeddingShape;
            const target_buf = try self.allocator.alloc(f16, target_len);
            var target_buf_committed = false;
            errdefer if (!target_buf_committed) self.allocator.free(target_buf);
            for (target_buf) |*value| {
                const v = try readF32(reader);
                if (!std.math.isFinite(v)) return TrainerError.InvalidEmbeddingWeight;
                if (v < -65504.0 or v > 65504.0) return TrainerError.InvalidEmbeddingWeight;
                value.* = @floatCast(v);
            }
            pending_target_weight = target_buf;
            pending_target_vocab = target_vocab;
            pending_target_dim = target_dim;
            target_buf_committed = true;
        }

        var loaded_nonce: [32]u8 = undefined;
        try reader.readNoEof(loaded_nonce[0..]);

        const new_training_graph = try readNsirGraph(self.allocator, reader, self.config);
        var new_training_graph_committed = false;
        errdefer if (!new_training_graph_committed) {
            new_training_graph.deinit();
            self.allocator.destroy(new_training_graph);
        };

        const new_knowledge_graph = try readNsirGraph(self.allocator, reader, self.config);
        var new_knowledge_graph_committed = false;
        errdefer if (!new_knowledge_graph_committed) {
            new_knowledge_graph.deinit();
            self.allocator.destroy(new_knowledge_graph);
        };

        const tokenizer_length_u64 = try reader.readInt(u64, .little);
        const tokenizer_length = std.math.cast(usize, tokenizer_length_u64) orelse return TrainerError.InvalidTokenizerData;
        if (tokenizer_length == 0 or tokenizer_length > self.config.max_tokenizer_file_size) return TrainerError.InvalidTokenizerData;
        const tokenizer_data = try self.allocator.alloc(u8, tokenizer_length);
        defer self.allocator.free(tokenizer_data);
        try reader.readNoEof(tokenizer_data);

        const trailer = try reader.readInt(u32, .little);
        if (trailer != CHECKPOINT_TRAILER) return TrainerError.CheckpointCorrupted;

        const tokenizer_tmp = try self.makeTmpFilePath("tokenizer");
        defer self.allocator.free(tokenizer_tmp);
        var tokenizer_tmp_committed = false;
        defer if (!tokenizer_tmp_committed) deletePath(tokenizer_tmp);

        {
            const tokenizer_file = try createWriteFile(tokenizer_tmp);
            var closed = false;
            defer if (!closed) tokenizer_file.close();
            try tokenizer_file.writer().writeAll(tokenizer_data);
            try tokenizer_file.sync();
            tokenizer_file.close();
            closed = true;
        }

        var new_tokenizer = try createConfiguredTokenizer(self.allocator, self.config);
        var new_tokenizer_committed = false;
        errdefer if (!new_tokenizer_committed) new_tokenizer.deinit();
        try new_tokenizer.loadVocab(tokenizer_tmp);
        if (new_tokenizer.next_token_id != saved_vocab_size) return TrainerError.VocabSizeMismatch;
        if (pending_emb_weight != null and pending_emb_vocab != new_tokenizer.next_token_id) return TrainerError.VocabSizeMismatch;
        if (pending_target_weight != null and pending_target_vocab != new_tokenizer.next_token_id) return TrainerError.VocabSizeMismatch;

        var new_accelerator_ptr = try self.allocator.create(RSFAccelerator);
        var new_accelerator_ptr_committed = false;
        errdefer if (!new_accelerator_ptr_committed) self.allocator.destroy(new_accelerator_ptr);
        new_accelerator_ptr.* = try RSFAccelerator.initMultiLayerWithDepthScale(
            self.model_dim,
            self.num_layers,
            self.allocator,
            self.config.spectral_depth_compensation,
        );
        var new_accelerator_committed = false;
        errdefer if (!new_accelerator_committed) new_accelerator_ptr.deinit();

        for (snapshots, 0..) |snapshot, layer_index| {
            try new_accelerator_ptr.setLayerWeightsS(layer_index, snapshot.weights_s, half, columns);
            try new_accelerator_ptr.setLayerWeightsT(layer_index, snapshot.weights_t, half, columns);
            try new_accelerator_ptr.setLayerVelocityS(layer_index, snapshot.velocity_s, half, columns);
            try new_accelerator_ptr.setLayerVelocityT(layer_index, snapshot.velocity_t, half, columns);
        }
        try new_accelerator_ptr.setClipRange(clip_min, clip_max);
        try new_accelerator_ptr.sync();

        var loaded_gpu_embedding: ?accel.EmbeddingAccelerator = null;
        var loaded_gpu_embedding_committed = false;
        errdefer if (!loaded_gpu_embedding_committed) {
            if (loaded_gpu_embedding) |*emb| emb.deinit();
        };
        if (pending_emb_weight) |wf16| {
            loaded_gpu_embedding = try accel.EmbeddingAccelerator.initWithWeights(
                &new_accelerator_ptr.ctx,
                pending_emb_vocab,
                pending_emb_dim,
                wf16,
            );
        }

        var loaded_target_source: ?accel.EmbeddingAccelerator = null;
        var loaded_target_source_committed = false;
        errdefer if (!loaded_target_source_committed) {
            if (loaded_target_source) |*source| source.deinit();
        };
        if (pending_target_weight) |target_f16| {
            loaded_target_source = try accel.EmbeddingAccelerator.initWithWeights(
                &new_accelerator_ptr.ctx,
                pending_target_vocab,
                pending_target_dim,
                target_f16,
            );
        }

        var new_signal_engine_ptr = try self.allocator.create(SignalPropagationEngine);
        var new_signal_engine_ptr_committed = false;
        errdefer if (!new_signal_engine_ptr_committed) self.allocator.destroy(new_signal_engine_ptr);
        new_signal_engine_ptr.* = SignalPropagationEngine.init(
            self.allocator,
            new_training_graph,
            &self.crev_kernel.flow_analyzer,
        );
        var new_signal_engine_committed = false;
        errdefer if (!new_signal_engine_committed) new_signal_engine_ptr.deinit();

        self.signal_engine.deinit();
        self.allocator.destroy(self.signal_engine);
        self.signal_engine = new_signal_engine_ptr;
        new_signal_engine_ptr_committed = true;
        new_signal_engine_committed = true;

        if (self.gpu_embedding) |*old_emb| old_emb.deinit();
        self.gpu_embedding = loaded_gpu_embedding;
        loaded_gpu_embedding_committed = true;

        if (self.target_source) |*old_source| old_source.deinit();
        self.target_source = loaded_target_source;
        loaded_target_source_committed = true;

        self.accelerator.deinit();
        self.allocator.destroy(self.accelerator);
        self.accelerator = new_accelerator_ptr;
        new_accelerator_ptr_committed = true;
        new_accelerator_committed = true;

        self.nsir_graph.deinit();
        self.allocator.destroy(self.nsir_graph);
        self.nsir_graph = new_training_graph;
        new_training_graph_committed = true;

        self.knowledge_nsir_graph.deinit();
        self.allocator.destroy(self.knowledge_nsir_graph);
        self.knowledge_nsir_graph = new_knowledge_graph;
        new_knowledge_graph_committed = true;

        self.tokenizer.deinit();
        self.tokenizer = new_tokenizer;
        new_tokenizer_committed = true;

        self.resetSpectralState();
        self.releaseTrainingFndsResources();
        self.releaseKnowledgeFndsResources();

        self.temporal_graph.deinit();
        self.temporal_graph = TemporalGraph.init(self.allocator);
        self.temporal_logical_time = 0;

        self.surprise_memory.deinit();
        self.surprise_memory = SurpriseMemoryManager.init(
            self.allocator,
            &self.crev_kernel.storage,
            &self.crev_kernel.flow_analyzer,
        );

        self.esso.deinit();
        self.esso = EntangledStochasticSymmetryOptimizer.init(
            self.allocator,
            self.config.esso_initial_temperature,
            self.config.esso_cooling_rate,
            self.config.esso_max_iterations,
        );

        self.vpu.deinit();
        self.vpu = VPU.init(self.allocator) catch |err| {
            std.debug.print("[Rank {d}] WARN: VPU reinit after load failed: {}\n", .{ self.coordinator.rank, err });
            return err;
        };

        if (self.training_variable_created) {
            _ = self.z_runtime.deleteVariable(self.config.training_variable_name);
        }

        self.vocab_size = saved_vocab_size;
        self.local_batch_size = saved_local_batch_size;
        self.learning_rate = saved_learning_rate;
        self.momentum = saved_momentum;
        self.global_step = saved_global_step;
        self.training_variable_created = false;
        self.vpu_lr_scale = 1.0;
        self.shuffle_control_state = self.config.embedding_seed ^ 0x5DEECE66D ^ saved_global_step;
        self.knowledge_graph_nonce = loaded_nonce;

        self.freeLayerSnapshots(snapshots);
        snapshots_committed = true;
        tokenizer_tmp_committed = true;
        deletePath(tokenizer_tmp);

        try self.coordinator.synchronize();

        std.debug.print("Checkpoint loaded from {s} at step {d}\n", .{ path, self.global_step });
    }


    fn ensureSpectralState(
        self: *DistributedTrainerFuthark,
        rows: usize,
        columns: usize,
    ) !void {
        if (self.gpu_spectral_u) |u| {
            if (self.gpu_spectral_v) |v| {
                if (u.len == rows and v.len == columns) return;
            }
        }

        self.resetSpectralState();

        const u_cpu = try self.allocator.alloc(f32, rows);
        defer self.allocator.free(u_cpu);
        const v_cpu = try self.allocator.alloc(f32, columns);
        defer self.allocator.free(v_cpu);

        var norm_squared: f64 = 0.0;
        for (u_cpu) |*value| {
            value.* = std.crypto.random.float(f32) - 0.5;
            norm_squared += @as(f64, value.*) * @as(f64, value.*);
        }
        const norm = @sqrt(norm_squared);
        if (!std.math.isFinite(norm) or norm <= 1e-12) return TrainerError.InvalidSpectralState;
        const inverse_norm: f32 = @floatCast(1.0 / norm);
        for (u_cpu) |*value| value.* *= inverse_norm;
        @memset(v_cpu, 0.0);

        const ctx = &self.accelerator.ctx;
        var new_u = try accel.FutharkArray1DF32.newFromSlice(ctx, u_cpu);
        errdefer new_u.free(ctx);
        const new_v = try accel.FutharkArray1DF32.newFromSlice(ctx, v_cpu);

        self.gpu_spectral_u = new_u;
        self.gpu_spectral_v = new_v;
    }

    fn applyEmbeddingSpectralNormalization(self: *DistributedTrainerFuthark) !void {
        if (self.gpu_embedding == null) return;
        const emb = &self.gpu_embedding.?;
        const rows = emb.vocab_size;
        const columns = emb.dim;
        if (rows == 0 or columns == 0 or self.spectral_normalizer.power_iterations == 0) return;

        try self.ensureSpectralState(rows, columns);
        const u_ptr = &self.gpu_spectral_u.?;
        const v_ptr = &self.gpu_spectral_v.?;

        if (self.coordinator.world_size > 1) {
            const u_vals = try u_ptr.valuesSlice(&self.accelerator.ctx, self.allocator);
            defer self.allocator.free(u_vals);
            try self.allReduceFloat32Values(u_vals);
            var u_sum_norm: f64 = 0.0;
            for (u_vals) |val| u_sum_norm += @as(f64, val) * @as(f64, val);
            const u_norm = @sqrt(u_sum_norm);
            if (!std.math.isFinite(u_norm) or u_norm <= 1e-12) return TrainerError.InvalidSpectralState;
            const inv_u_norm: f32 = @floatCast(1.0 / u_norm);
            for (u_vals) |*val| val.* *= inv_u_norm;
            const ctx = &self.accelerator.ctx;
            const replaced_u = try accel.FutharkArray1DF32.newFromSlice(ctx, u_vals);
            u_ptr.free(ctx);
            self.gpu_spectral_u = replaced_u;
        }

        try emb.spectralNormalize(u_ptr, v_ptr, self.spectral_normalizer.power_iterations);
        try self.accelerator.sync();

        if (self.coordinator.world_size > 1) {
            const u_post = try u_ptr.valuesSlice(&self.accelerator.ctx, self.allocator);
            defer self.allocator.free(u_post);
            try self.allReduceFloat32ValuesAvg(u_post);
            var u_post_norm: f64 = 0.0;
            for (u_post) |val| u_post_norm += @as(f64, val) * @as(f64, val);
            const u_pn = @sqrt(u_post_norm);
            if (std.math.isFinite(u_pn) and u_pn > 1e-12) {
                const inv_pn: f32 = @floatCast(1.0 / u_pn);
                for (u_post) |*val| val.* *= inv_pn;
            }
            const ctx = &self.accelerator.ctx;
            const replaced_u2 = try accel.FutharkArray1DF32.newFromSlice(ctx, u_post);
            u_ptr.free(ctx);
            self.gpu_spectral_u = replaced_u2;
        }
    }

    pub fn buildKnowledgeGraph(self: *DistributedTrainerFuthark, text: []const u8) !void {
        if (text.len == 0) return TrainerError.EmptyKnowledgeGraphInput;
        if (text.len > self.config.max_knowledge_graph_input) return TrainerError.KnowledgeGraphInputTooLarge;

        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        hasher.update(self.knowledge_graph_nonce[0..]);
        hasher.update(text);
        var digest: [32]u8 = undefined;
        hasher.final(&digest);

        var node_id_buffer: [67]u8 = undefined;
        node_id_buffer[0] = 'k';
        node_id_buffer[1] = 'g';
        node_id_buffer[2] = '_';
        const hexadecimal = "0123456789abcdef";
        for (digest, 0..) |byte, index| {
            node_id_buffer[3 + index * 2] = hexadecimal[byte >> 4];
            node_id_buffer[4 + index * 2] = hexadecimal[byte & 0x0f];
        }
        const node_id = node_id_buffer[0..67];

        const tree_id = try self.ensureKnowledgeFndsTree();
        const index_id = try self.ensureKnowledgeFndsIndex();

        _ = try self.crev_pipeline.processTextStream(text);

        const text_bytes = std.mem.sliceAsBytes(text);
        _ = try self.knowledge_nsir_graph.encodeInformation(text_bytes);

        var pattern_location = try PatternLocation.init(
            self.allocator,
            tree_id,
            0,
            node_id,
            0,
            text_bytes.len,
            1.0,
        );
        var pattern_transferred = false;
        defer if (!pattern_transferred) pattern_location.deinit();

        _ = try self.fnds_manager.insertIntoTree(tree_id, node_id, text_bytes, 0);

        try self.fnds_manager.addPatternToIndex(index_id, text_bytes, pattern_location);
        pattern_transferred = true;

        try self.r_gpu.distributeGraph(self.knowledge_nsir_graph);
        try self.signal_engine.propagateStep();
    }
};
