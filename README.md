JAIDE v40

A fifth root architecture.

Machine learning has produced four architectural roots so far. The perceptron established the trainable linear threshold unit. The convolutional network added weight sharing over local receptive fields. The recurrent network added state carried across time. The transformer replaced recurrence with all-pairs attention. Every one of these is built on the same primitive: a nonlinearity applied to a weighted sum, sigma of W times x plus b, composed in depth, trained by storing activations on the forward pass and consuming them on the backward pass.

RSF is not built on that primitive. Its atomic operation is a bijective, invertible cross-affine coupling map. There is no sigma of Wx plus b anywhere in the model. There is no MLP, no ReLU, no LayerNorm, no BatchNorm, no softmax, no attention, no convolution, no recurrence, and no hidden state. What replaces them is a single algebraically invertible transform, applied in depth, with a deterministic parameter-free mixing block between layers.

The analogy to 2017 is deliberate. Attention Is All You Need took attention out of the encoder-decoder RNN context where it had lived as an auxiliary mechanism and made it the sole primitive. RSF does the same thing to affine coupling: it takes the coupling layer out of the normalizing-flow context, where it existed to make density estimation tractable, and makes it the entire network. This is not a wrapper. RevNet and Reformer are reversibility retrofitted onto existing architectures and still contain convolutions or attention underneath. RSF contains neither.

This repository is the full system built on that primitive, written from scratch in Zig, with GPU kernels in Futhark, hardware models in Clash Haskell, a machine-checked proof of the core invertibility theorem in Lean 4, and a zero-knowledge inference circuit in Circom. No PyTorch, no TensorFlow, no Zig package dependencies.


Why the primitive is different

A conventional layer computes y equals sigma of Wx plus b. That map is not injective. ReLU destroys the sign of every negative pre-activation. Pooling discards spatial detail. Softmax collapses a vector onto a simplex. Each of these is a deliberate information loss, justified by the Information Bottleneck view that discarding nuisance variation is what makes a representation useful. Because information is destroyed, the input to each layer cannot be recovered from its output, so training must cache every intermediate activation. That is what makes backpropagation cost O of depth times sequence length times width in memory.

An RSF layer splits its input into halves x1 and x2 and computes:

    scale = exp(clip(Ws * x2 + bs))
    y1    = x1 * scale
    trans = Wt * y1 + bt
    y2    = x2 + trans

Scale is derived from x2 before x2 is touched, and trans is derived from the already-updated y1. That ordering is what makes the map exactly invertible:

    trans = Wt * y1 + bt
    x2    = y2 - trans
    scale = exp(clip(Ws * x2 + bs))
    x1    = y1 / scale

Because scale is an exponential, it is strictly positive by construction, so the division can never divide by zero and the Jacobian determinant is never zero. Bijectivity is not an empirical property that happens to hold on the trained weights. It is a structural guarantee of the algebra, valid for every weight configuration the optimizer can reach. Both directions are implemented at rsf.zig lines 340 to 382, and the clipped exponential is at line 335.

The consequences follow mechanically rather than by design choice.

Information is preserved at one hundred percent. Mutual information between any two layers is maximal, because a bijection cannot lose bits. This is a direct break with the Information Bottleneck principle rather than an approximation of it.

The Jacobian is triangular. Splitting into x1 and x2 and updating them in sequence means each output block depends on only one input block plus already-computed quantities. The determinant is therefore the product of the diagonal, which is the product of exp of the scale pre-activations. Computing it is O of dim, not O of dim cubed.

Backward memory is depth-independent. Because x1 and x2 can be reconstructed from y1 and y2 algebraically, activations are never stored. The signature of the backward routine makes this visible: backwardFromOutputsRow at rsf.zig line 384 takes y1_row and y2_row, the outputs, and has no activation buffer parameter at all. Working memory is O of dim per row regardless of how many layers deep the model is. A transformer needs O of layers times sequence times width.

The coupling is subnetwork-free. NICE and RealNVP use internal MLPs to compute their scale and translation functions. RSF uses a single matrix-vector product for each, computeScaleRow and computeTranslationRow, so the primitive stays atomic instead of being a container for a conventional network.


What replaces the missing pieces

Removing attention, convolution and recurrence leaves four jobs that something has to do.

Global mixing without attention is handled by OFTB, an orthogonal fractal transform block. It is a Haar-wavelet butterfly over the halves of the vector, scaled by FRACTAL_SCALE which is one over the square root of two, approximately 0.7071. It has no trainable parameters at all. Forward produces the difference and sum, each scaled; backward reverses it. Between coupling layers a fixed deterministic permutation shuffles elements so that every coordinate eventually interacts with every other, giving global context without an O of N squared attention matrix. The scatter kernel and its composition with the flow are in futhark_kernels.fut at lines 82 to 120. OFTB itself is in processor/oftb.zig.

Spatial and temporal dependency is handled by OFTB for the spatial axis and by TemporalGraph for the temporal one, rather than by convolution or recurrence.

Context is not held in a KV cache. It lives in a versioned temporal graph with bitemporal snapshots and time-range queries, paired with a surprise-based memory manager that retains events by novelty using hash-distance scoring. Nothing grows linearly with sequence length the way a KV cache does.

Symbolic relation is handled by NSIR, a self-similar relational graph running alongside the neural stack. Queries against it are sparse relational lookups at O of d, not dense softmax over all positions. Its edges carry a quality drawn from a five-state quantum-inspired enum defined at nsir_core.zig line 12: superposition, entangled, coherent, collapsed, and fractal. This is the dual-system design. The RSF stack is the neural half; the NSIR graph is the symbolic relational half; they share state rather than being pipelined.


The rest of the departures

There is no separate bias tensor anywhere in the model. Bias is fused into the weight matrix using homogeneous coordinates, giving a matrix of shape dim by dim plus one, for dim squared plus dim parameters per matrix. The extra column is read directly as the bias term in computeScaleRow.

Objectives are not next-token prediction. The model builds a hierarchical latent representation, and reasoning is convergence-driven, run in local, global, hierarchical and meta phases by the reasoning orchestrator, rather than being a single autoregressive sampling loop.

Optimization is second order. SFD, Spectral Fisher Descent, approximates the diagonal of the Fisher Information Matrix with spectral cutting, rather than following raw gradients.

Stability comes from topology rather than normalization layers. There is no LayerNorm and no BatchNorm. Instead each layer is spectrally constrained by power iteration, 30 iterations against a target norm of 0.9, which guarantees Lipschitz continuity. The exponential argument is clipped to a configured range, default minus five to five, validated to never exceed plus or minus twenty. Above that sit a GradientFlowController and dynamic loss scaling. Multi-layer gradient stability is therefore a property of the construction, not a patch applied on top of it.

Correctness is machine-checked, not merely tested. The Lean 4 proof in verification/oftb.lean states the invertibility theorem over an abstract ScaleAlgebra structure, so it is a claim about the algebra the code implements rather than about floating-point rounding. The theorems mix_round_trip and mix_round_trip_rev establish that composing the two directions in either order is the identity. This is the Curry-Howard direction: the network is a proof of a theorem, not heuristic code that seems to work. On top of that, empirical roundtrip checks run at a tolerance of 1e-4, and the repository carries 359 inline test blocks across its modules.

In approximation-theoretic terms the model is a universal approximator within the class of diffeomorphisms, following CF-INN theory. The whole thing is analytically derived, from geometric flows and contextual type theory downward, rather than assembled by heuristic experiment.


Invertibility as a learning signal

The same property that removes the activation cache also changes what a training step is worth.

In a conventional architecture every step produces exactly one learning signal: the gradient of the output loss. Weights learn only from how accurately the next token was predicted. Intermediate states are computed during the backward pass and then discarded, because they have no other use.

Because RSF is bijective, the backward pass produces a second quantity at no extra cost. Running gradients back through the stack requires reconstructing each layer's input from its output, and backwardFromOutputsRow does exactly that: alongside the gradients it writes x1_row_out and x2_row_out, the recovered inputs. backwardOnCore then copies those into y1_row and y2_row before descending to the next layer down. After the final layer is processed, those buffers hold a reconstruction of the original network input. This is not a side effect that happens to be useful. It is an exact algebraic consequence of the inverse being closed-form.

So a single forward and backward pair carries three independent signals rather than one.

The forward signal is the ordinary next-token prediction loss. This is the only one a conventional architecture can extract.

The reconstruction signal is the discrepancy between the recovered input and the true input:

    L_recon = (1/N) * sum_i || x_hat_i - x_i ||^2

This measures how well the weight matrices actually preserve information, which is to say how genuinely invertible the network is at its current weights. Its computational cost is zero, because x_hat is already sitting in the backward buffers.

The combined embedding signal weights the two together when updating the embedding table:

    grad_emb = grad_fwd + alpha * grad_recon

The embedding therefore learns two things at once: how to predict, and how to carry information through the stack without losing it.

This is only available because the inverse is exact. In a transformer, intermediate activations recovered during the backward pass are either values cached from the forward pass or approximations reassembled from them, and they do not correspond to the original input under any interpretation. In RSF the backward pass applies the true inverse, so the reconstruction is the input, and the reconstruction loss is a real geometric measure of whether the weights preserve the structure of what was fed in.

The practical effect is gradient density. The embedding and the early layers receive feedback from two directions on every step, for the same arithmetic, giving roughly two to three times the signal per unit of computation. A phased curriculum exploits this by starting with the reconstruction term alone, at alpha equal to one, so the weight matrices become well conditioned first, then folding in the forward term progressively. That ordering avoids the failure mode where a network learns from incoherent prediction gradients while its weights are still badly conditioned.

Architecture summary

    primitive      bijective cross-affine coupling, exp-positive scale
    mixing         OFTB Haar butterfly, parameter-free, 1/sqrt(2)
    permutation    fixed deterministic scatter between layers
    bias           fused, homogeneous coordinates, dim by (dim+1)
    backward       reconstruction from outputs, no activation cache
    jacobian       triangular, det = product of exp(s_j)
    stability      spectral norm 0.9 via 30 power iterations, exp clipping
    optimizer      SFD, diagonal Fisher with spectral cutting
    context        versioned temporal graph plus surprise memory
    symbolic       NSIR relational graph, five-state edge quality
    training       triple signal: forward loss, reconstruction, combined
    verification   Lean 4 proof, Circom ZK circuit, 359 test blocks

Absent by construction: MLP, ReLU, LayerNorm, BatchNorm, softmax, self-attention, convolution, pooling, recurrence, hidden state, KV cache, separate bias tensor, stored activations.


Repository layout

    build.zig.txt          build script, rename to build.zig
    build.zig.zon.txt      package manifest, rename to build.zig.zon
    scripts/
      modal_status_bench.py   end-to-end Modal harness: build, train, serve
      setup_modal_token.sh    credential sanitizer for the Modal CLI
      upload_vocab.py         pushes tokenizer.vocab to the Modal volume
    src/
      core/                tensor, types, memory, io, model_io, embedding
      processor/           RSF coupling flow, OFTB butterfly block
      optimizer/           Spectral Fisher Descent and optimizer toolkit
      tokenizer/           MGT morphological and BPE tokenizer
      index/               SSI hash-partitioned segment index
      ranker/              relevance scoring over SSI results
      core_relational/     26-module relational graph substrate
      hw/accel/            CUDA and Futhark bindings, kernels, fractal LPU
      hw/rtl/              Clash hardware models and software simulator
      distributed/         NCCL coordinator, GPU trainer, Modal client
      api/                 HTTP inference server
      verification/        Lean 4 proof of OFTB invertibility
      zk/                  Circom inference-trace circuit
      tests/               benchmarks, refcount stress test, C ABI test

The system spans 57 Zig sources, two Futhark kernel libraries, one Circom circuit, one Lean proof, three Clash hardware modules, two Python deployment scripts and one C ABI test.


Building

Zig 0.14.1 and Futhark 0.26.4 are required for any build. CUDA 12.8 with NCCL is needed for the gpu flag, circom 2.1.8 with snarkjs for zk, Lean 4 with lake for verify, and GHC with Clash for rtl.

The build files carry a text suffix so the toolchain does not pick them up until renamed. After renaming and syncing the single Futhark package dependency, a default build produces the inference server:

    mv build.zig.txt build.zig
    mv build.zig.zon.txt build.zig.zon
    cd src/hw/accel && futhark pkg sync && cd -
    zig build -Doptimize=ReleaseSafe

Every build runs the Futhark CPU library step before compiling any Zig, because the Zig sources include the generated C header. The gpu flag additionally produces the distributed trainer:

    zig build -Dgpu=true -Doptimize=ReleaseSafe

The inference server takes port, host, model and require-api-key options, and reads a model path from the environment when the flag is absent.


Build system

Four boolean options gate the optional toolchains, all defaulting to false, so a default build needs only Zig and Futhark. The gpu flag adds the distributed trainer, compiles the CUDA kernel library and links cuda, cudart, nvrtc, nccl, m, pthread and dl. The zk flag runs circom and then a snarkjs Groth16 setup against a powers-of-tau file. The verify flag runs a lake build over the Lean proof. The rtl flag compiles the Clash modules with GHC into a shared library and builds the simulator.

Named steps are distributed-futhark for the trainer alone, bench to build and run the four benchmark binaries, test-all for every test plus the C ABI check, test-c-api on its own, and zk, verify and rtl for the toolchain-gated artifacts.

Two separate option sets exist rather than one. The CPU set hardcodes gpu acceleration to false and the GPU set passes the flag through, and every CPU artifact receives the CPU set. This is why a machine with CUDA installed still produces a server binary that runs correctly without a device present.

One linking decision looks like an omission and is not. The distributed binary links the CUDA-generated C file but not the CPU one, because both Futhark libraries export identical context and entry point symbol names. Linking both would be a duplicate symbol error, not a missing symbol error, so only the CUDA library is included.


Core numeric layer

core/tensor.zig, is the foundation.

Data is 32-byte aligned and operated on in 8-wide f32 SIMD vectors. Each tensor holds a heap-allocated atomic refcount and copy-on-write flag alongside its data pointer, base pointer and its own Shape, at lines 218 to 238. Views share the buffer but own their Shape independently, which is why release at lines 262 to 281 always frees the shape and only frees the buffer at zero. A non-final release deliberately leaves the struct usable because other holders still depend on it.

ensureWritable at lines 296 to 334 implements copy-on-write and is called by every mutating method. At refcount one it clears the flag and returns; otherwise it duplicates the buffer, contiguously or through the strided iterator, and installs fresh refcount and flag cells.

effectiveCpuCount at line 59 reads cgroup v2 cpu.max, falls back to cgroup v1 quota and period, then to the host count, clamping to between 1 and 8. Every benchmark prints it alongside the cgroup source so results are attributable.

Matmul at lines 1046 to 1109 validates rank-2 shapes with matching inner dimensions, copies both operands contiguous, transposes the right side so both walks are row-major, then runs a 32 by 32 blocked kernel with vector accumulators across up to eight threads.

Shape.init rejects zero dimensions, more than eight dimensions and zero-length axes, with checked multiplication on both total size and strides. Division pre-scans the divisor for zeros and returns an error before writing anything, so a failed divide leaves the tensor untouched.

Four modules support it. types.zig carries the shared error set and four checked fixed-point formats, Q8.8, Q16.16, a 64-bit variant and Q32.32, each detecting overflow and division by zero rather than wrapping, which is what the RTL and quantum paths use where float behaviour would be unreproducible. It also holds a seedable PRNG with uniform, normal and entropy-reseed paths, a bit set with union and intersection, a fixed-capacity token ring buffer, a bloom filter, and the RankedSegment type the index and ranker exchange.

memory.zig is the allocator layer. Arena, slab, pool, buddy and page allocators each expose both a direct interface and a standard vtable adapter, so any of them can back a tensor, and each has a secure variant that zeroes memory before releasing it. Alongside them sit lock-free and mutex-guarded queues and stacks, a read-write lock, a tracking allocator that records statistics, constant-time comparison for secret material, and a wide surface of raw memory operations including compression, AEAD encryption, and virtual memory mapping, protection and locking.

io.zig provides memory-mapped files guarded against double close, a durable writer with optional fsync on flush, buffered readers and writers, atomic write through a temp file and rename, and stableHash, the seeded hash that makes ranker scores reproducible across runs and machines.

learned_embedding.zig is the trainable token table. Weights initialize to small uniform noise, forward gathers rows with token ids clamped into range, backward scatters gradients back with the same clamping, and applyGradients runs SGD with momentum. It flattens parameters and gradients into and out of contiguous buffers, which is exactly the interface the distributed trainer needs for all-reduce.

Model files open with the eight bytes JAIDE40 and a null, then a u32 version of 1, then JSON metadata, then payloads, ending with a 32-byte SHA-256 digest. Every byte is fed through the hasher as it is written and re-derived on import.


The RSF implementation

Initialization sets up a stack that is stable before the first step rather than one that has to be stabilized during training. Weights are Xavier-initialized with a bound of the square root of six over fan-in plus fan-out, the bias column is explicitly zeroed so a layer starts as a pure scaling, and both matrices are then constrained to a spectral norm of 0.9 through thirty power iterations. That constraint is what bounds the Lipschitz constant of each layer, and with it the composition of the whole stack, which is why the architecture needs no normalization layer. The power iteration accumulates its inner products in f64 before casting back to f32, with guards at 1e-12 on both normalizations, so the estimate does not collapse on a near-singular matrix. Layer l seeds from l times 10007 and the spectral constraint uses a further offset of nine million, so no two layers and no two phases share a random stream.

The lifetime model is unusual enough to explain. Both RSF and RSFLayer are handles: the struct holds a single u64 identifier, and the real state lives in a process-global registry behind a mutex, with each core individually protected by a read-write lock. This exists because a coupling layer holds weight matrices that are expensive to duplicate and dangerous to share accidentally. The registry records the memory address of the handle that first used a given identifier, and if a different address later presents the same one, the call returns an error rather than proceeding. In other words, copying the struct and using both copies is caught immediately as an error instead of producing a double free later. Destruction only happens when the owning address releases.

Reads take the shared lock and writes take the exclusive one, so many forward passes can run concurrently against one model while a weight update serializes against all of them. On a GPU build the forward path is more involved: it takes the exclusive lock, tries the device path, and if that fails resyncs the weights and tries exactly once more before falling back to the CPU path under a shared lock. A pair of version counters, one for CPU weights and one for GPU, decides whether the device copy is current, and the mutation path bumps the CPU counter with wrapping addition so the comparison stays correct across overflow.


Optimizer

Spectral Fisher Descent is second-order: instead of following the raw gradient it approximates the diagonal of the Fisher Information Matrix and preconditions the step with it, using spectral cutting to keep the estimate bounded. It maintains the Fisher diagonal alongside momentum and velocity buffers, starting the Fisher at one so the first steps behave like plain momentum until curvature information accumulates.

Validation happens at construction rather than at the first bad step: both betas must lie strictly inside zero to one, epsilon and the clip threshold must be positive, and fisher_max must be finite and positive. The update path then re-checks that the learning rate is finite and non-negative and that gradient, parameter and configured sizes all agree before touching anything. Defaults are beta1 at 0.9, beta2 at 0.999, epsilon at 1e-8, a clip threshold of 1.0, a Fisher ceiling of 1e6 and ten warmup steps.

The module carries its own lighter tensor type rather than reusing the core one, with conversions in both directions, because the optimizer needs precision tagging and quantization state that the core tensor does not model. Its matmul mirrors the core implementation, blocked at thirty-two, SIMD-accumulated and threaded. An FP4 quantizer snaps absolute values onto the ladder zero, one half, one, one and a half, two, three, four and six, clamping at plus or minus six and passing non-finite values through untouched, and an erf approximation with the standard Abramowitz-Stegun coefficients supports the probabilistic paths.

Surrounding SFD is the machinery that keeps a deep reversible stack stable in practice. KFACBlock maintains Kronecker-factored curvature with damping and preconditions gradients through it. SpectralNormalizer bounds singular values by power iteration. GradientFlowController implements the normalized gradient flow the trainer enables by default. MARSVarianceReducer cuts gradient variance. LRScheduler covers the usual schedules including full cosine annealing. DynamicLossScaler and MixedPrecisionTrainer handle the f16 path, scaling losses up to keep small gradients representable and backing off on overflow. GaussianProcess and BayesianOptimizer search hyperparameters from observed runs. B200MemoryManager and B200Profiler tune for Blackwell specifically. SophiaSOAPOptimizer is an alternative second-order method kept alongside SFD.


Tokenizer, index, ranker

MGT is a morphological and byte-pair hybrid rather than pure BPE, because purely statistical merges handle agglutinative morphology poorly. It keeps separate prefix, suffix and root tables per language, currently English and Hungarian, alongside the learned merge table, and consults the longest morphological match before falling back to statistical merges. Token ids zero through three are permanently PAD, UNK, BOS and EOS, and a vocabulary cap below four is rejected outright. A 256-entry byte table sits underneath everything as a total fallback, which guarantees that any byte sequence is representable and that decode is never lossy. Anchor tokens are tracked in a separate map, and tokenizeWithAnchors returns their positions alongside the token stream so downstream stages can find semantic pivots without re-scanning. BPE training is parallel, with one worker context counting pair frequencies across sequences and another rewriting sequences once a merge is chosen. Beyond encode and decode there are batch paths, vocabulary persistence, a coverage measure reporting what fraction of a corpus the vocabulary spans, and direct tensor bridges in both directions.

SSI is the segment store. It is a tree with sixty-four children per internal node, from a six-bit bucket width, and a default height of six, so lookups reach a leaf in six hash-directed steps. Leaves hold segments carrying their tokens, a position, a score and an anchor hash, and collisions extend into linked chains rather than probing. Segments hash two ways: over their tokens alone, and a full hash mixing position, the raw score bits, the anchor hash, the token count and every token, so identical token runs at different positions remain distinguishable. Beyond insertion and top-k retrieval it supports compaction, in-place score updates, merging another index, splitting on a score threshold into a new index, rebalancing, structural validation, serialization over any reader and writer, and export to a fixed 134-column tensor layout for bulk numeric work.

The ranker scores candidates from that index. The final score blends three signals at fixed weights, base score at 0.4, token overlap at 0.3 and Jaccard similarity at 0.3, with diversity and proximity weights of 0.3 available for the multi-criteria path. Every token is hashed through an explicit little-endian byte encoding rather than raw memory, so scores are bit-identical on any architecture and a model ranked on one machine ranks the same on another. It computes MinHash signatures for cheap set similarity, cosine and dot-product scores over embeddings, exponential decay and normalization over score arrays, streaming top-k over any reader with a bounded heap, parallel scoring across threads, and supervised weight calibration that runs epochs against labelled data.


The relational layer

This is the symbolic half of the dual system. Twenty-six modules hold, transform, verify and remember relational structure while the RSF stack does the numerics. Everything is exposed through mod.zig, a facade that imports all twenty-four submodules with public declarations and then lifts several hundred individual types into the top namespace, so calling code writes core_relational.SelfSimilarRelationalGraph without knowing which file it lives in. Where two modules define the same name the facade disambiguates: ProcessingCore becomes ChaosProcessingCore and RGPUProcessingCore, CoreState splits the same way, SystemState becomes SystemState and SecuritySystemState, QuantumCircuit becomes QuantumCircuit and HardwareQuantumCircuit, VerificationResult becomes FormalVerificationResult, Term becomes TypeTheoryTerm. The C entry points are re-exported here too, which means the entire relational engine is reachable over FFI from Python, Rust or C++. The single test in the facade is the only place in the codebase where ZRuntime and ChaosCoreKernel appear together, creating two variables, entangling them, then allocating and reading back a memory block, so it validates the integration seam rather than any one module.

nsir_core: the substrate

Every other module in this directory imports its types from here, and this file imports nothing from the relational layer itself. It defines NSIR, Non-Sequential Information Representation.

A Qubit holds two complex amplitudes with the invariant that their squared magnitudes sum to one, enforced at construction, falling back to the zero basis state if the norm is NaN or infinite. A Node carries an identifier, a raw data payload, a Qubit, a phase in zero to two pi, a coherence value and a metadata map. An Edge is the richer structure: source and target, an EdgeQuality, a weight, a complex quantum_correlation and a fractal_dimension. Two ownership flags let an edge either own its endpoint strings or borrow canonical pointers from the node map, which is how the graph guarantees edges always reference the stored strings rather than copies.

EdgeQuality has five states and they carry real semantics. Superposition means both endpoints are unresolved. Entangled means the two nodes share a state and measuring one collapses both. Coherent is a stable settled link. Collapsed means measurement already happened. Fractal marks a scale-invariant self-similar connection.

Edges are stored as a map from edge key to a list, so parallel edges between the same pair are first-class rather than an error. Entanglement is stored separately as a map from an unordered node pair to a TwoQubit holding four amplitudes over the basis states; entangleNodes writes the maximally entangled Bell state and adds bidirectional entangled-quality edges. Measurement implements the physical postulate properly: for an entangled node it samples the joint state by cumulative probability, forces both nodes into the measured basis state, deletes the entanglement record and downgrades the connecting edges to collapsed.

Gates are function pointers over a Qubit. Hadamard produces the sum and difference over root two, Pauli X swaps the amplitudes, Pauli Y applies the imaginary swap, Pauli Z flips the sign of the second amplitude, and phase gates exist in both a comptime-specialized and a runtime form.

The topology hash is order-independent. Each node, edge group and entanglement is hashed separately including metadata, the digests are sorted, then folded into one final SHA-256, so two structurally identical graphs hash identically regardless of map iteration order.

Two methods form the bridge to the numeric side. exportNodeEmbeddings emits a float tensor of four values per node, the real and imaginary parts of both amplitudes, and importNodeEmbeddings writes a tensor back into the node states with renormalization. bulkImportFromGPU ingests the arrays a Futhark kernel produces, hash array plus four amplitude channels plus edge endpoint indices, and it is transactional: on failure every node added during the call is rolled back. Negative indices, self-loops and out-of-range endpoints are filtered, and a missing endpoint is logged and skipped rather than aborting the batch.

vpu: the vector processing unit

The most complex component of the hardware abstraction layer, and the numerical foundation the whole relational engine computes on. It provides SIMD linear algebra, a dedicated allocator, graph vectorization and its own instruction set.

VectorType enumerates the six supported shapes, four and eight lane single precision, two and four lane double, four and eight lane integer, and for each one records lane count, element size, total byte size and required alignment, sixteen bytes for the narrow forms and thirty-two for the wide. That alignment is what makes the vectors directly consumable by SSE and AVX rather than forcing a scalar fallback. SimdVector is a comptime generic over element type and lane count built on Zig's native vector builtin, implementing add, subtract, multiply, divide, scale, dot product, magnitude, normalize, fused multiply-add, square root, minimum, maximum, absolute value, reductions, blend, shuffle, cross product, distance, linear interpolation, clamp, negate and reflect. Each lowers to hardware vector registers, so a single instruction operates on four or eight floats at once.

VectorBatch and VectorBatchEntry form a type-erased container aligned to thirty-two bytes, letting heterogeneous vector types sit together and be processed under one batch operation of normalize, scale, absolute value or square root, while tracking how many were processed, skipped and handled individually.

Matrix4x4 and MatrixOps handle four by four float matrices. The SIMD multiply transposes and uses dot products so the inner loop stays vectorized, QR decomposition runs Gram-Schmidt orthogonalization, inversion goes through the adjugate, and the determinant uses Laplace expansion. These are the numerical basis for the bijective affine transforms in the RSF layers.

RelationalVectorOps is the bridge between the compute layer and the semantic representation, and the most important connection in the file. computeNodeSimilarity scores two nodes as a weighted combination of three normalized terms: phase difference at 0.3, magnitude difference at 0.3, and the quantum state inner product at 0.4. vectorizeGraph walks nodes in sorted key order, so output is deterministic, and turns each into a four-lane double vector of phase, magnitude and the two parts of the quantum state. computeEdgeVectorBatch does the same for edges, packing weight, both parts of the quantum coupling and the fractal dimension. computeGraphLaplacian builds the Laplacian with explicit NaN, infinity and negative guards on every adjacency entry, spectralEmbedding derives a normalized embedding from it, and applyQuantumRotation rotates in two independent planes by theta and phi, simulating quantum logic gates in vector space.

MemoryPool is a thirty-two byte aligned free-list allocator with a thirty-two byte minimum unit and coalescing of adjacent free blocks to limit fragmentation; the minimum and the alignment together guarantee every region stays SIMD-usable. VectorCache is an LRU keyed by unsigned 64-bit values that evicts the least recently used entry at capacity, which stops the unit recomputing graph embeddings and similarity matrices it already produced. VPUStatistics counts operations, SIMD instructions issued, cache hits and misses, memory allocated and freed, vectors processed, and matrix and graph operations, deriving a cache hit rate and a SIMD efficiency ratio for self-monitoring.

The VPU struct integrates all of it, defaulting to a one megabyte pool, a batch size of 256 and a 1024 entry cache. computeGraphEmbeddings vectorizes then normalizes, computeSimilarityMatrix produces cosine similarity across all embeddings, powerIteration finds the dominant eigenvector, batchMatmul runs SIMD multiplication over batches, and quantumVectorOps applies rotation across a batch. Every operation advances a cycle counter that functions as a virtual processor clock.

LNSValue implements a logarithmic number system and is the reason the unit can be trusted with the numbers RSF produces. A value is stored as the natural log of its absolute value in the mantissa with a separate sign flag, and zero is encoded as negative infinity. Multiplication becomes addition of mantissas and division becomes subtraction, both exact in log space with no rounding, and the code short-circuits correctly when either operand is the zero encoding. This matters for probability weights and spectral coefficients where ordinary floating point loses precision at extreme magnitudes.

LNSInstruction defines the instruction set, and it is what makes this a processor rather than a math library. It carries two RSF-specific instructions, rsf_scatter and rsf_affine_couple, which encode the scatter permutation and the affine coupling step directly as machine operations, plus tensor load and store, logarithmic add and multiply, a graph transform, conditional and unconditional jumps, and halt.

Its place in the system is specific. After the Futhark GPU kernels finish the large parallel work including canonical signature computation, the VPU takes the results on the CPU side and performs node vectorization, similarity matrices, spectral embeddings, quantum rotations and the LNS-based affine coupling steps. Its fused multiply-add, dot product and logarithmic instructions are the numerical foundation for reconstructing activations during the backward pass, which is what lets the RSF stack run at memory proportional to width instead of the transformer's sequence length times width.

signal_propagation: what replaces attention

This module simulates a wave-like, quantum-phase-aware propagation across the graph, and it is the mechanism that stands in for attention.

SignalState describes a signal with amplitude, phase, frequency and a nanosecond timestamp. Its advance method is a discrete approximation of the wave equation, adding two pi times frequency times delta time to the phase each step and normalizing modulo two pi, and its complex representation multiplies amplitude by cosine and sine of the phase so the signal can interact directly with node quantum states. The combine method averages two signals by linear superposition across amplitude, phase and frequency, which means that when several sources reach one node they interfere rather than overwrite. That is precisely what replaces the summation step of multi-head attention, except driven by graph topology.

ActivationTrace keeps the full activation history per node: every past signal state, an activation count, first and last activation timestamps, and average amplitude and frequency, which the inference pipeline reads as a node's importance.

The engine holds a graph pointer, a data flow analyzer pointer, a map of activation traces, and simulation parameters defaulting to a time step of 0.01 and a propagation speed of 1.0. initiateSignal injects into a source node, setting the node phase from the signal, scaling the qubit amplitude by the signal amplitude relative to current magnitude, then renormalizing so the squared magnitudes still sum to one. That is where external input enters the quantum state representation.

propagateStep is the core. It iterates every edge, derives the source signal from the node qubit magnitude and phase, multiplies amplitude by edge weight as attenuation, adds the angle of the edge quantum correlation to the phase, advances by the time step, combines when several signals land on one target, and finally blends the target qubit seventy to thirty between old and new magnitude, which is an exponential moving average. Every activated node is also reported to the data flow analyzer with its identifier hashed into a sixteen byte key, so propagation patterns feed the chaos dynamics analysis and become part of the system's self-organizing behavior.

propagateInferenceSignal runs a fixed five steps and propagateForInference takes a step count; both return the summed average amplitude across nodes as a scalar activation score. getInferenceActivationMap returns per-node average amplitude, and that map is what the upper RSF layers read to decide which nodes are relevant to the current step. Functionally it expresses what attention scores express, but relevance emerges from graph topology and phase dynamics rather than from a learned query-key product. An inference hook interface with three optional callbacks, on step complete, on signal initiated and on propagation complete, lets higher layers observe and steer without reaching into engine state.

The tests check exactly what matters: that ten steps genuinely increase activation counts, that the hook interface returns a correct score, and that qubit normalization survives every propagation step to within a millionth, which guarantees numerical stability over long sequences.

surprise_memory: what replaces the KV cache

Rather than storing every key and value for every token at every layer, this module keeps only content that differs meaningfully from what is already held.

Its constants set the behavior: a default surprise threshold of 0.3, a temporal novelty window of exactly twenty-four hours in nanoseconds, a Jaccard sample size of 1000, a bigram space of 65536 for all possible byte pairs, a frequency saturation of 8.0 and a cap of 100 entanglement pairs.

SurpriseMetrics scores novelty on three orthogonal axes. Jaccard dissimilarity measures bigram-level structural difference against existing blocks. Content hash distance is the Hamming distance between SHA-256 digests normalized over 128 bits. Temporal novelty measures how old existing blocks are within the twenty-four hour window. The combined score is their mean, tested against the threshold.

SurpriseRecord tracks each block's lifecycle and converts the score into a retention priority, recomputing a weight as 0.5 plus 0.3 times a recency term plus 0.2 times a frequency term, where recency is one over one plus age in milliseconds and frequency saturates at eight accesses, then multiplying the surprise score by that weight. Fresh, frequently touched, surprising content ranks high; stale repetitive content decays.

The manager holds a pointer to the content-addressable storage where blocks physically live, a pointer to the data flow analyzer, a map of records, a mutex and a monotonic epoch guaranteeing timestamps never decrease. Content hashing takes SHA-256 and XORs the two halves into a sixteen byte identifier. Bigram presence builds a bit vector over 1024 words where each bit marks one possible byte pair, sampling with a stride on large inputs so the cost stays constant regardless of input size.

Surprise computation returns maximum novelty on an empty store, otherwise samples up to a thousand existing blocks with a stride, finds the minimum Jaccard and hash distances across them, computes temporal novelty and assembles the metrics. Storage first checks whether the content already exists by content address, and if so only updates the access record, so identical content is never stored twice.

Eviction is adaptive: past the target capacity it refreshes all retention priorities, uses a heap-based partial sort to find the lowest, and drops those. Organization by entanglement collects the high-surprise blocks up to the cap of a hundred and entangles every pair through the storage layer, so semantic links form between surprising content.

The inference entry point computes the metrics and returns two decisions, cache when combined surprise exceeds 0.3 and propagate when it exceeds 0.15, and that second flag is what tells the orchestrator whether to run signal propagation over the graph. Caching a result concatenates input and output and stores the pair, so a later similar input scores low surprise and the earlier result can be reused.

quantum_logic: gates over graph state

This defines the quantum state representation the whole relational layer computes on, and every other module here imports it.

A QuantumState holds two complex amplitudes, a global phase and an entanglement degree in the unit interval, and that last field is what the runtime uses as the correlation weight when it creates graph edges. Normalization enforces that the squared magnitudes sum to one, the two probability methods implement the Born rule, fidelity returns the squared inner product as a similarity measure between states, and addition superposes two states with renormalization while preserving the larger entanglement degree.

Twelve gate types split into two families. The standard set is Hadamard, the three Pauli operators, phase, controlled not and Toffoli. The JAIDE-specific set is relational and, or, not and exclusive or plus a fractal transform, and these are not part of standard quantum mechanics; they are the system's own operators encoding graph relations as state transformations.

The engine manages up to 1024 states, logs every gate application with a nanosecond timestamp so the full computation history can be replayed and audited, and a coherence threshold catches any state whose total probability has collapsed to nothing.

The controlled not implementation is a deliberate approximation and worth understanding. Rather than applying the true unitary over a joint two-qubit space, it mixes the target amplitudes weighted by the control's two probabilities and raises the entanglement degree by twice the square root of their product, so maximum entanglement growth happens exactly when the control sits in equal superposition. That approximation is what allows each qubit to be stored independently, giving memory linear in qubit count rather than exponential in the dimension of the joint Hilbert space.

The relational and gate does not modify an existing state but produces a new one whose amplitudes are the componentwise complex products of the inputs, with phase averaged and entanglement summed to a cap of one, then renormalized. Semantically the result is strong only when both inputs were strong, which is logical conjunction at the level of amplitudes. Or superposes by addition, exclusive or by difference, and not swaps the amplitudes with a pi phase shift.

The fractal transform is the most distinctive. It applies a phase rotation repeatedly, where each iteration uses half the angle of the previous one, so the sequence of angles forms a geometric series, and the whole amplitude vector rotates by each in turn with renormalization between. The effect is a self-similar structure in phase space where every level is a halved copy of the one above, which is the quantum-logical counterpart of the spectral transform in the RSF layers.

Measurement uses cryptographic randomness, collapses to whichever basis state the sample selects, and zeroes the entanglement degree, with a deterministic variant taking an injected value so the behavior is testable. Serialization writes forty-eight bytes per state in little-endian order plus the full gate history, which is what lets the temporal graph store and restore quantum state at any past version.

fnds: fractal data structures

This organizes the graph's data into a self-similar hierarchy rather than a flat sequence.

A fractal node carries an identifier, data, a weight, a scale factor, a SHA-256 signature and metadata, and the signature is computed over the identifier, data, weight and scale in canonical little-endian order so any change to any field changes the fingerprint; setting metadata also refreshes it. Fractal edges come in four types, hierarchical for parent and child, sibling, cross-level, and self-similar for links between structurally identical regions at different scales, and each carries the ratio between the two levels' scale factors.

A level holds its own scale factor, starting at one at the root and dividing by the branching factor at each descent, its own node and edge maps and parent and child pointers. Its local fractal dimension comes from box counting: it estimates counts at four box sizes by hashing each node and edge into a pseudo-coordinate, then fits a line through the log-log points, which approximates the Hausdorff dimension where one is a line and two is a plane. Adding a child level walks the ancestor chain first and refuses the link if it would create a cycle, so the hierarchy stays a tree.

The tree takes a maximum depth and a branching factor of at least two, holds a random identifier, and places each insertion by hashing the tree identifier against the depth, which means the same content lands in the same place deterministically. Traversal supports four orders, of which the fractal order is the interesting one: it halves the children and walks the first half forward and the second half backward, producing a symmetric pattern useful for surfacing symmetric structure. Balancing collects every node, sorts identifiers, computes the optimal depth from the logarithm of the node count in the branching factor, and rebuilds evenly, with a check every hundredth insertion that actual depth has not drifted more than two levels past optimal. Global dimension averages each level's local dimension with the mean of its children recursively.

A pattern location records which tree, level, node, offset and length an occurrence sits at along with a confidence. The self-similar index maps pattern keys to those locations, and similarity scoring averages a length ratio with a positional match ratio, sorting results descending and returning only those above the threshold, which gives fuzzy pattern matching over graph data. The index also computes the fractal dimension of the pattern length distribution, where a power-law distribution of the kind natural language exhibits yields a positive dimension.

Two supporting structures round it out. A coalesced hash map keeps primary buckets in the first eighty-six percent of capacity and pushes collisions into a cellar in the remaining fourteen, chaining through a next-index field so every element lives in one array, which gives better cache locality than external chaining, with a free list for reuse. An LRU cache pairs a doubly linked list with a map and evicts from the tail on either a count or a memory ceiling, defaulting to ten megabytes.

The manager holds the trees, the indexes and the cache, averages every tree and index dimension into one global figure, and reports statistics covering mean depth, total nodes, indexed patterns and locations, cache hit rate and the timestamp of the last operation.

c_api: the stable C boundary

This is the single C ABI surface for the entire relational core, and it is how the inference server, the distributed trainer and any external tooling reach the graph without knowing Zig types.

Two opaque handle types wrap the internals. The graph handle points at a context bundling the graph itself, a mutex and an allocator, and every exported function takes that mutex first, so the whole interface is thread-safe, which the server's concurrent request handling requires. Every function is declared with the C calling convention.

The surface divides three ways. Lifecycle and topology covers creation, destruction, clearing, node and edge addition and removal, counts, and a topology hash returning the SHA-256 fingerprint of current structure, which the verification and security subsystems depend on. Quantum state covers setting node state with normalization applied on write, reading the probability of the first basis state, and measurement that collapses and returns a single bit. Gate application exposes identity, Hadamard and the three Pauli operators against any node.

Edges are not simple weighted links across this boundary either: weight is clamped to the unit interval, and quality, quantum correlation and fractal dimension are all readable, with those values feeding directly into the optimizer's energy function.

The optimizer implemented here is an adaptive simulated annealer whose energy sums weight times fractal dimension across edges, correlation magnitudes and node amplitudes using Kahan accumulation for numerical stability. Perturbation randomly adjusts edge weights and node states, acceptance follows the Metropolis rule, stagnation triggers automatic reheating, and the cooling rate adapts to the observed acceptance rate.

Two functions form the data gateway: encoding takes raw bytes and returns a node identifier, decoding returns the payload, and together they replace what an embedding lookup table does in a conventional model. Nineteen negative error codes cover null pointers, allocation failure, missing nodes and edges, invalid quality, optimization failure, invalid strings, duplicates, invalid parameters, math errors, uninitialized state, self-reference, invalid state, threading and unknown gates.

temporal_graph: what replaces the context window

This layer stores the complete temporal history of every node and edge at nanosecond resolution and allows the graph state at any past moment to be queried, restored or snapshotted.

A NodeVersion holds a version number, a nanosecond timestamp, a quantum state and a property map, and delegates probability and magnitude to the state so any historical version remains queryable in quantum terms. An EdgeVersion holds a version, timestamp, weight and quality. TemporalNode keeps a version list, returns the latest version at or before a given timestamp, rolls back by version or by time, and returns all versions in a range. TemporalEdge adds a validity interval; invalidating an edge sets its end bound rather than deleting it, so the edge disappears from later queries while remaining visible in earlier ones.

A GraphSnapshot records version numbers rather than data, so restoring is a matter of moving every node and edge back to the recorded version. Every mutation writes a history entry recording the operation, entity type, identifier and the version before and after, giving a complete audit trail.

computeGraphStateAt returns every node quantum state and edge weight at a given timestamp, which is how the RSF stack looks back into earlier graph states. recordInferenceObservation adds a version for the current state, and a combined method updates and snapshots in one call so any inference step can be rewound.

TemporalQuery is declarative: a time range, optional node and edge filter functions and a flag for including invalidated edges. Three prebuilt filters select by entanglement above a half, by superposition quality and by coherent or superposition quality, and these give the selective-attention behavior a topological rather than learned basis. An InferenceContext wraps the whole thing in a narrow interface for the pipeline, exposing active edges for a node, version lookups by time, observation recording, time advance and full state queries.

The consequence is that the context window is not a fixed token budget. Nothing is lost when it fills, because nothing fills; the graph topology, the quantum state filters and the temporal queries jointly determine which past states are relevant now.

chaos_core: self-organizing execution

This is where the abstract graph becomes physical resource management, deciding where data lives in memory and which core runs which task.

ContentAddressableStorage replaces position-based addressing with content-based identity. A block identifier is a SHA-256 of the content hash combined with a nanosecond timestamp, and storing content that already exists returns the existing identifier and refreshes the access time rather than allocating again. Each block records an affinity core and a set of entangled blocks, mirroring the graph edge structure at the memory level. Eviction runs in two passes, taking unentangled blocks first and only touching entangled ones if that was not enough.

DynamicTaskScheduler assigns tasks by data locality, not just priority: a core scores ten if the task's data dependency already lives there, one if it lives elsewhere, plus five times the inverse of its load. Work moves to where the data already is. Task descriptors carry an inference priority and context identifier, so the scheduler is tuned for the inference pipeline rather than being general purpose.

DataFlowAnalyzer watches which core touches which block and builds a per-block affinity profile, so a block accessed two thirds of the time from core zero carries a 0.67 affinity to it. It also builds a co-occurrence graph from task dependencies, and that correlation drives entanglement propagation.

The kernel runs all of it. Each cycle schedules tasks, updates core states and optimizes data placement, migrating any block whose measured best core differs from its current affinity when the affinity exceeds 0.6. Every hundredth cycle it rebalances, taking a quarter of the blocks from cores above 1.3 times average load and distributing them to cores below 0.7 times average. The critical integration is executing a graph on the kernel: every node becomes a memory block, every edge becomes an entanglement between the corresponding blocks, then self-organization runs, which means graph topology literally determines physical memory layout. Entangling two blocks also pulls in the first block's correlated neighbors above 0.5, so transitive graph relationships propagate into memory.

crev_pipeline: text into knowledge

CREV stands for continuous relational extraction and validation, and it is the path from raw text into the graph.

Beneath it sit three helpers: a word tokenizer splitting on whitespace and punctuation, a stemmer handling the common English suffixes, and a morpheme-aware matcher combining them so running and run match the same pattern.

The unit is a RelationalTriplet holding subject, relation and object as owned strings, a confidence clamped to the unit interval, a SHA-256 source hash computed from the three fields alone so it is time-independent, an extraction timestamp and a metadata map. Its most important method converts the triplet into graph elements: subject and object hash into sixteen character identifiers and become nodes whose quantum state is confidence as the real amplitude and the square root of one minus confidence squared as the imaginary part, so the pair is unit-normalized and uncertainty is encoded geometrically, with phase derived from the extraction time over a 360 second period. The relation becomes a coherent edge weighted by confidence.

KnowledgeGraphIndex maintains three inverted indexes over subject, relation and object, and queries filter on whichever gives the smallest candidate set. A morpheme-aware query trades that for stem-based matching across all triplets. A StreamBuffer of ten thousand entries holds triplets in flight, dropping the oldest when full so the pipeline never blocks.

The pipeline itself holds a kernel pointer, the buffer, the index, per-relation and per-entity statistics and inference hooks, and initializes with fifteen relation patterns weighted by reliability, from is a at 0.9 and contains at 0.85 down to related to at 0.5. Processing text splits into sentences, finds the longest matching pattern per sentence with morpheme awareness, computes confidence from the pattern weight adjusted by entity length and capitalization, and keeps the triplet if confidence clears 0.3.

Validation runs four stages: basic checks, a confidence threshold at 0.5, consistency against indexed triplets using contradiction pairs such as is a against is not and causes against prevents, and anomaly detection. The anomaly score uses Welford's online algorithm: once a relation has appeared more than ten times the system knows its mean and standard deviation, and a triplet more than three deviations out drives the score toward one, with unknown entities and unknown relation types adding weighted contributions. Above 0.85 the triplet is rejected. Conflict resolution prefers the higher confidence and combines the two values as the sum of squares over the sum, which rewards the stronger claim more than an average would.

Integration is the key seam: a validated triplet is indexed, buffered, counted, then serialized and allocated into the kernel's content-addressable store, so it lives simultaneously in the knowledge index and in the self-organizing memory where locality optimization and entanglement apply to it. Three input formats are handled, free text, comma-separated structured records, and image metadata that becomes image to key to value triplets at 0.9 confidence, which makes the knowledge graph multimodal.

esso_optimizer: symmetry-aware annealing

This is the dedicated energy minimizer for the graph, richer than the compact version in the C API.

SymmetryGroup defines seven symmetries with their rotation angles and group orders, from identity through reflection and the three rotations to translation and a custom rotation whose order is recovered by rounding two pi over the angle. SymmetryTransform describes a full affine transform with origin, parameters and scale, applies the appropriate geometry per group, applies directly to node quantum states by shifting phase for rotations and mirroring amplitude parts for reflections, and composes two transforms by matrix product, identifying the result group from the sign of the determinant and locating the fixed point by solving the linear system.

EntanglementInfo tracks correlation strength, phase difference, creation and update times and an interaction count. Updates use a running weighted average for strength and a circular mean for phase, averaging the cosine and sine components before converting back, which avoids the discontinuity at the wrap point. Decay is exponential with a sixty second half life, so an entanglement left untouched for a minute weakens by half.

The optimization state holds the graph, current energy, entanglement percentage, iteration count and the entanglement map keyed by lexicographically ordered node pairs so direction does not matter. An undo log records the original edge weights and fractal dimensions, node phases and qubit amplitudes before each step, so a rejected proposal is reverted exactly rather than approximately; only the topology-changing step needs a full graph clone.

The loop clones the input, computes initial energy, detects symmetries, then iterates up to ten thousand times. Each iteration decays and drifts the entanglement map, re-detects symmetries every fiftieth iteration, applies one of seven move types, computes the new energy and accepts by the Metropolis criterion, reverting through the undo log otherwise. The seven moves perturb edge weights, node phases, new entanglements, a random non-identity symmetry transform across all node states, qubit amplitudes with renormalization, fractal dimensions, and finally graph topology by adding or removing an edge.

Cooling adapts to the acceptance rate: above 0.6 the rate tightens to cool faster, below 0.2 it loosens, bounded either side. Stagnation past a tenth of the iteration budget triggers reheating by the configured factor, which escapes local minima.

Symmetry detection projects each node to a two-dimensional position from its amplitude parts and phase, computes the centroid and inertia tensor, derives a principal axis angle and eccentricity, then tests reflection by mirroring across that axis and rotation at two, three, four and six fold, accepting when over thirty percent of points map close to an existing point. Phase coherence is measured by the norm of the summed phase vectors, and above a half it registers as a custom rotation at the mean phase.

The tensor modulation method is the direct bridge to inference: it computes average entanglement strength from the best state and scales tensor values by one plus a tenth of that plus a hundredth per detected symmetry, capped at double, so topological and quantum properties of the graph directly shape the generated output. Four selectable objectives ship: a default combining edge and node terms with average entanglement, a connectivity objective, a quantum coherence objective, and one that drives average fractal dimension toward 1.5, the dimension of Brownian motion and therefore of scale-invariant self-similar structure.

r_gpu: parallel graph processing

This models a relational graph processing unit in software, five subsystems in one.

A processing core sits at a grid coordinate with a state, a neighbor list, its own local graph, an incoming message queue and activity counters from which workload is the active fraction. The asynchronous network on chip builds a two-dimensional mesh, gives every core its four neighbors, and precomputes XY routes for every source and destination pair, moving in X first then Y, which is the deterministic deadlock-free mesh routing. Messages travel through a priority queue with arrival order breaking ties, and five message types cover weight updates, graph synchronization, isomorphism results, power control and data transfer.

The isomorphism processor computes a canonical form per graph: each node gets a signature of out-degree, in-degree, weight sum and edge quality sum, signatures are sorted, edges are sorted as index triples, and the whole thing serializes to a string that is order-independent, so structurally identical graphs produce identical strings. Two graphs are isomorphic when their canonical forms match, and subgraph search slides a window across the main graph.

Dynamic edge weighting keeps per-edge history and multiplies four factors: a history adjustment from the most recent weight, a temporal adjustment growing logarithmically with history length, a spatial adjustment from the trend between the last two weights, and a semantic adjustment from the full history average. Weight propagation spreads outward with a decay of nine tenths per iteration.

Sparse activation only wakes cores above a ten percent load, banking the rest as saved energy, and the power gating controller switches off cores below ten percent utilization when current draw exceeds half the budget and switches them back on above eighty percent.

The unit ties these together with three operations: distributing the global graph across cores round-robin with the relevant edges cloned alongside, running isomorphism search in parallel across active non-gated cores, and synchronizing the local graphs back into one global graph with duplicate filtering.

formal_verification: proving the graph is sound

Nine invariant types are ordered by priority so the most critical failure is handled first: memory safety at ten, type safety at nine, connectivity at eight, then coherence, entanglement, quantum state, fractal dimension, symmetry and temporal consistency.

Twenty-six proof rules cover propositional logic, first-order logic, Hoare logic and temporal induction, each declaring its minimum premise count. Eighteen proposition types include the JAIDE-specific ones: Hoare triples, separation logic star and wand for disjointness, temporal always and eventually, and relational edge, quantum superposition and entanglement pair, which encode the graph's physical concepts as logical formulas.

Terms and propositions are reference counted with SHA-256 structural hashes cached and invalidated on substitution, so equality is a hash comparison. A proof step verifies premise count, index validity, prior verification and absence of duplicates before running rule-specific structural checks: modus ponens confirms one premise is an implication whose antecedent is the other premise, the sequence rule confirms the first triple's postcondition matches the second's precondition, the loop invariant rule confirms precondition and postcondition both equal the invariant. A full proof validates that every step references only earlier steps and that the final conclusion matches the theorem.

Nine built-in predicates check the graph directly. Connectivity runs breadth-first treating edges as undirected. Coherence confirms every qubit magnitude stays within one plus a tolerance. Quantum state confirms no NaN or infinity and a squared norm in range. Memory safety confirms every edge endpoint exists, catching dangling edges. Type safety confirms non-empty identifiers and finite bounded weights. Fractal dimension confirms the zero to three range. Symmetry confirms every edge has a matching reverse. Temporal consistency confirms finite phases and non-decreasing creation times along edges.

The Hoare verifier formalizes the RSF layer transforms as triples, checking the assignment axiom by substitution, the sequence rule by matching conditions, and the loop invariant rule, and composing several triples into one, which means a stack of layers reduces to a single triple. The theorem prover works both by backward chaining to a depth of a hundred and by resolution refutation, unifying with Robinson's algorithm including the occurs check.

The engine registers all nine invariants at construction, hashes the graph topology, checks every invariant and reports which failed with elapsed time. Its inference hook runs three stages, graph invariants, then the implication between the step's pre and postcondition, then building and checking a formal proof, with a simplified variant that skips proof construction for speed.

type_theory: dependent types and category theory

Twenty-one type kinds span base types, composites, the dependent triple of Pi, Sigma and identity types, and the special kinds including a quantum type that carries a base type and a dimension.

The Type structure holds every kind in one representation with parameters, fields, bound variable and body, left and right types, universe level and a cached SHA-256 for structural identity. Substitution replaces free variables and is the basis of dependent application: applying a Pi type to an argument substitutes the bound variable in the body with the argument's type, which is exactly the mechanism absent from conventional architectures. Type contexts chain with a parent pointer and look up backward so shadowing works.

Terms cover eighteen kinds including reflexivity and the J eliminator, which encode the introduction and elimination rules of the identity type. The type checker infers recursively, resolving variables from context, literals from value, lambdas from body, and applications from the codomain with substitution for dependent functions. Subtyping implements the numeric hierarchy from natural through integer and real to complex, bottom below everything, top above everything, contravariant domains and covariant codomains for functions, and level comparison for universes, with unification finding the least common supertype.

Identity types provide reflexivity, symmetry and transitivity with a check that the chain endpoints match. Universes are cumulative with a least upper bound. Inductive types are defined Coq-style with constructors, parameters and indices, with built-ins for naturals, booleans and lists, and a recursor for structural recursion.

Propositions as types implements Curry-Howard directly: conjunction is a tuple, disjunction a sum, implication a function, negation a function into bottom, universal quantification a Pi type, existential quantification a Sigma type, truth the unit type and falsity the bottom type. Linear types add four modes, linear demanding exactly one use, affine at most one, relevant at least one and unrestricted any number, with a validator logging unused, overused, dropped and duplicated violations. The linear mode matters directly: using every resource exactly once is a necessary condition for bijectivity.

The category layer provides categories with cached composition, functors that verify preservation of identity and composition, natural transformations that verify the naturality square commutes, monads that verify the unit and associativity laws, and cartesian closed categories that check for a terminal object, products and exponentials, which are necessary and sufficient for modelling lambda calculus.

The engine exposes proof methods for type judgments, subtyping, equality, linear usage, functor laws and monad laws, each returning a result with validity, a textual derivation and any error, so every RSF transformation step can carry a formal proof.

security_proofs: information flow and access control

Security and integrity levels form complementary lattices. Confidentiality runs public through top secret with join as maximum, integrity runs untrusted through kernel with join as minimum, which is the inverted lattice the Biba model requires. Access rights are a bitmask supporting the usual set operations.

A security label combines a confidentiality level, an integrity level and a category set, and domination requires both levels to be at least as high and the categories to be a superset, which is the mathematical basis of mandatory access control. Principals carry a clearance and rights; objects carry a label, owner and data hash verified in constant time.

Information flow analysis builds a flow graph with explicit, implicit and covert edge types, computes the transitive closure by breadth-first search, and flags every flow from a higher to a lower level as illegal, checking both direct edges and the closure. Severity scales with the level gap using overflow-safe arithmetic. Non-interference is verified by confirming no high node reaches a low node through the closure, and proved separately by bisimulation: the prover partitions variables into high and low by observer level, compares the low projections of two states, applies the unwinding lemma and concludes, where matching low projections mean high inputs cannot affect low observations.

Access control runs four layers, a matrix check, a policy check over subject and object patterns, a clearance check and a combined decision, alongside separation of duties constraints that fail when a subject holds too many conflicting roles, and a least privilege check for unnecessary rights.

Integrity uses two cryptographic structures. A hash chain links each block to its predecessor by hashing index, data, previous hash and timestamp, with constant-time verification of both the data and block digests, giving a tamper-evident log. A Merkle tree builds over data hashes, duplicating the last leaf when odd, and generates sibling-path proofs so membership can be verified without rehashing everything. Commitments generate a random salt, publish the hash of value and salt, and verify in constant time. Cryptographic proofs cover knowledge, membership, range, equality and integrity claims.

Bell-LaPadula enforces confidentiality with no read up and no write down. Biba enforces integrity with the dual, no read down and no write up. Every proof is finalized with a binding hash over its identifier, type, property and step descriptions, so a proof cannot be edited after the fact without invalidating it.

The critical integration is proving information flow security over the graph itself: each node is classified by the magnitude of its first amplitude, below 0.2 public, below 0.4 internal, below 0.6 confidential, below 0.8 secret and above that top secret, edges become explicit flows, and the analysis then detects illegal flows. Node quantum state determines security level and graph topology determines the flow policy, and the engine verifies the result formally.

verified_inference_engine: proof-carrying inference

Six cryptographic components combine into one engine: a Blake3 commitment scheme for inputs and outputs, homomorphic encryption, differential privacy calibrated at epsilon one, delta a hundred-thousandth and sensitivity a thousandth, a dataset fingerprint, a step-by-step proof of correctness, and a zero-knowledge prover. It also holds the model hash and the per-layer weight matrices for eight layers at thirty-two by thirty-two.

Weights initialize with Xavier scaling at one over the square root of the embedding dimension, spread uniformly in that range, and those two matrices are exactly the affine coupling parameters. The matmul is the coupling step itself: split the buffer in half, accumulate the scale weights against the second half, clip to plus or minus five, multiply the first half by the exponential, then accumulate the translation weights against the first half and add to the second. The inverse is division by the exponential and subtraction of the translation, which is why activations need never be stored.

A verified inference runs fourteen sequential phases: commit the input, run the matmul, record the scalar multiply step, run the affine coupling, record it, copy to output, record the scatter permute, add differential privacy noise per value, commit the output, store the input-output commitment pair, generate a zero-knowledge proof or fall back to hashing, re-verify every previously stored commitment, finalize the proof of correctness, and verify the latest proof. That re-verification step means tampering anywhere in the history is detected immediately.

Dataset isolation is provable: hashing a sample and checking it against the fingerprint establishes cryptographically that the model did not memorize it. A batch verifier accumulates proof hashes and commits them together with a success rate, and a proof aggregator builds a Blake3 Merkle tree over all proof bundles so any individual inference can be shown to belong to the complete set.

zk_verification: the proving stack

Six mechanisms layer here, from the toolchain driver up to federated aggregation.

The circuit configuration points at the Circom source, the compiled WASM, the proving key, the verification key and the witness and proof directories, defaulting to eight layers, thirty-two dimensions, sixty-four bits of precision and a five minute timeout, matching the engine above.

A Groth16 proof encodes the three curve elements over BN128, the standard Ethereum curve, giving constant proof size and constant verification cost. The Circom prover drives the external toolchain as child processes: compiling the circuit to R1CS, WASM and symbols, running the Groth16 setup and exporting the verification key, generating the witness, producing the proof, and verifying it by looking for the success marker in output.

The inference witness carries the tokens, both weight matrix sets, the expected output and the commitments, converting floats to fixed point at a scale of a million with clamping, because arithmetic circuits work only over integers. Commitments are Blake3 digests of the tokens, outputs and each layer's matrices, converted into 256-bit integers, and the whole witness serializes to the JSON the witness generator consumes, so the circuit checks exactly the computation the stack performs.

Proving runs twelve steps end to end, building the witness, converting to fixed point, computing commitments, generating unique filenames from a random nonce, writing the JSON, generating the witness and proof, reading both artifacts into a bundle, verifying immediately and returning.

The supporting primitives are a Pedersen-style commitment with nonce and blinding factor, a range proof committing bit by bit so a value can be shown in bounds without disclosure, a Merkle membership proof, a simplified Schnorr signature for model identity, and differential privacy offering Gaussian, Laplace and calibrated noise, where the configured parameters give a noise scale around three thousandths. The inference proof itself works in two modes, a hash-chain fallback that folds a step hash across the eight layers, and full mode running the real Groth16 pipeline. Secure aggregation commits each participant and averages only once the threshold is met, enabling federated learning without exposing any participant's data.

dataset_obfuscation: privacy at the data layer

The configuration constants use the secp256k1 group order and a 256-bit security parameter, so the cryptographic strength matches industry practice.

The Paillier implementation generates 128-bit primes with the top and bottom bits forced and confirms them by Miller-Rabin with twelve witnesses, then derives the modulus, its square, the Carmichael totient and its inverse using extended Euclid over wide intermediates. Encryption encodes the sign in a high bit and the magnitude low, draws a coprime nonce and computes the standard ciphertext. Addition of ciphertexts multiplies them modulo the square, which decrypts to the sum, and scalar multiplication exponentiates, taking the modular inverse for negative scalars. Deinitialization securely zeroes the private values so a memory dump cannot recover them. That homomorphism is what lets RSF layer features be stored and combined while still encrypted.

The dataset fingerprint indexes samples by locality-sensitive hashing across eight functions into buckets, encrypting the features, and similarity checking looks for an exact match first then scans the buckets for any digest within a Hamming distance of thirty-two. That is the mechanism behind proving a sample was absent from training.

The secure sampler enforces k-anonymity and a privacy budget: a query must request at least k samples, costs a tenth per sample, fails when the budget would be exceeded, and returns a cryptographically shuffled index order.

Proof of correctness enumerates exactly the RSF operations, scalar multiply, matrix multiply, affine coupling, scatter permute, aggregation, forward, inverse, OFTB mix and differential privacy. Each recorded step hashes its input and output and folds them into a running chain hash, and finalization builds a Merkle tree over the steps and commits to the root, the chain hash and the length together, so the whole computation is auditable and any tampered step is detectable.

Dataset isolation protects data with access keys, expiry, a default cap of ten thousand accesses and a hundred per second rate limit, hashing the operation and client identifier into the audit log so no raw identifiers are stored.

quantum_hardware and ibm_quantum: real devices

The hardware module carries documented physical parameters for five IBM processor generations, the relaxation and dephasing times, readout error and two-qubit gate error, taken from published calibration data rather than invented. Fidelity estimation multiplies gate fidelity, readout fidelity and a depth factor, so a fifty-deep circuit with a hundred two-qubit gates on Heron comes out near forty-five percent, and that estimate feeds straight back into the inference pipeline.

Calibration is fetched over HTTP from the cloud endpoint and parsed per qubit and per gate, with unit normalization when values arrive in seconds, and when the network is unavailable the module synthesizes calibration from the documented specifications with bounded random variation so offline runs still use realistic parameters. A mock mode makes every path exercisable without credentials.

Circuits support thirty-four gate types and serialize to OpenQASM 3, with depth tracked per qubit. Simulation allocates a state vector up to twenty qubits, applies every instruction, then samples; hardware execution mixes the ideal distribution with noise weighted by the estimated fidelity. Five algorithms ship: Bell and GHZ state preparation, the quantum Fourier transform, Grover search with the optimal iteration count, and a variational ansatz with rotation and entanglement layers. The hybrid optimizer computes expectation values from measurement parity and gradients by the parameter-shift rule, which gives analytic gradients over circuit parameters usable by classical optimizers.

The separate client module is deliberately dependency-free, importing only the standard library. It handles authentication, submits QASM jobs at a default of 1024 shots, polls results, and zeroes the token and resource name before freeing them.

quantum_task_adapter: choosing what to run on hardware

This decides which parts of the graph are worth sending to a quantum device. A subgraph collects node identifiers and edge keys and computes two metrics, total entanglement as the summed correlation magnitude and average fractal dimension, and it is considered quantum-suitable only when entanglement exceeds the threshold and average fractal dimension exceeds 1.5. Identification walks the edges, keeps those above both the entanglement and fractal thresholds, groups them into semantic clusters, and builds a subgraph per cluster with at least two members. Execution either generates QASM and submits to the real backend at 1024 shots, or runs the local simulator, and the results are written back into the graph node states.

z_runtime: variables as graph state

This redefines what a variable is. A ZVariable is not an address and a value but a graph plus a quantum logic engine, where the value is encoded in graph nodes and the variable's quantum state derives deterministically from the content hash: assignment encodes the value into the graph, hashes it, and initializes the amplitudes from the cosine and sine of a scaled hash, so identical content always produces the same quantum state.

Relating two variables computes their quantum correlation as the product of one amplitude with the conjugate of the other, uses its magnitude as the edge weight, sets the edge correlation to the product itself, averages existing fractal dimensions, clones the target node if it is foreign, creates the edge and entangles the two in the logic engine, so a relation is simultaneously a topological link and a physical entanglement.

The runtime holds a variable map, a global graph, a global logic engine and an execution history logging every operation with a nanosecond timestamp, primary and secondary targets and the result. A relational operation between two variables produces a new variable named after both, copies their states, applies the corresponding gate and links the result to both sources coherently, so the output of a logical operation is itself a graph-connected variable rather than a scalar. Information propagation walks the graph to a given depth from a source and reports which other variables' graphs contain the touched nodes, lifting signal propagation up to the variable level. An expression evaluator parses text with precedence over and, exclusive or, or and entangle. A system state snapshot reports every variable's node and edge counts, fractal dimension, topology hash and state counts along with aggregates.

reasoning_orchestrator: convergence-driven thinking

This optimizes graph state by minimizing a physical energy function and returns a modulation factor that scales the RSF weights.

Three thought levels operate: local perturbation at node granularity, global symmetry-based transformation, and meta orchestration alternating between them. A phase tracks a target energy of a hundredth, current and previous energy, a convergence threshold of a millionth, timestamps and captured symmetry pattern identifiers, and converges when the relative energy change falls below the threshold.

Numerical guards run throughout: weights clamp to the unit interval, fractal dimensions to one through three, complex magnitudes to one, and any NaN or infinity in a qubit resets it to the zero basis state, so optimization can never produce an invalid state. Snapshots of node and edge state are taken before each perturbation and restored when energy rises, making the search a greedy descent.

Local perturbation applies changes on the order of a few thousandths to qubit amplitudes and renormalizes, and edge updates shift weights slightly and correlations by a tenth of that. The global phase detects symmetries, records a pattern identifier per transform, applies them to node states and renormalizes. Fractal rebalancing moves every edge a tenth of the way toward the mean dimension, homogenizing the structure. Chaos relaxation damps phases and correlations by a thousandth each cycle while injecting a golden-ratio-based phase bias on the order of a millionth, which prevents settling into a local minimum.

The energy function sums a structural term of weight times fractal dimension and a correlation magnitude term per edge, plus a phase term per node that is minimized when phases align near zero or pi. The full run cycles local, global and meta phases, averages their energies and stops on convergence, returning the best energy, a modulation factor of one over one plus that energy, the phase count and the pattern count. Applying that factor to a tensor is the direct control path: low graph energy means a factor near one and weights pass through, high energy means a factor near zero and weights are damped, so the orchestrator regulates how much the stack trusts the current graph state.

safety: the primitives underneath

This file imports nothing but the standard library and everything else leans on it. Checked integer casts use compile-time type information to reject overflow and underflow, so values crossing the C boundary or feeding size computations cannot silently truncate. Pointer casts check for null, non-pointer types, zero addresses and misalignment before converting.

The secure random generator combines two sources, XORing the operating system generator with a linear congruential state seeded from it, and uses rejection sampling for ranges to avoid modulo bias, so even a compromised system generator leaves entropy. A monotonic clock provides the nanosecond timestamps used for block identifiers, extraction times, entanglement records and message stamps throughout the system.

Secure zeroing uses volatile writes so the compiler cannot elide it, which matters because a plain memory set on a value never read again is legally removable. Constant-time comparison accumulates length differences and byte differences with bitwise or across the full length regardless of where the first mismatch falls, so timing reveals nothing, and it backs the token comparisons and block identifier checks. Bounds-checked slicing and copying return errors instead of overrunning, and a 512-bit integer type provides the wide arithmetic the hashing and modular steps need.


Hardware acceleration

The stack is layered so the same Zig source compiles with or without a GPU.

cuda_bindings.zig declares two parallel structs and selects at compile time, using the real extern symbols when the flag is on and a stub otherwise. The stub returns an initialization error for allocations and success for cleanup, so CPU builds link with no CUDA installed. futhark_bindings.zig declares the Futhark C API by hand rather than importing generated headers, with no-op equivalents of the GPU-only config setters when the flag is off.

accel_interface.zig wraps this in typed handles and an RSFAccelerator mirroring the model on device. It configures Futhark with group size 256, 128 groups and tile size 32, and reads JAIDE_FUTHARK_CACHE to persist NVRTC kernels across container starts, warning when unset, at lines 53 to 56. Raw f16 device pointers are exposed so NCCL can all-reduce gradients in place. setClipRange rejects any range other than minus five to five when GPU is on, because the compiled kernel bakes that constant in.

futhark_kernels.fut, compiled for CPU in f32, exposes roughly a hundred entries covering matmul, Fisher and natural gradient updates, RSF forward and backward, the scatter permutation, SSI hashing and search, LSH, and a large family of relational graph kernels. main.fut, compiled for CUDA in f16, provides a fused training step, padded and masked batch operations, OFTB, embedding forward, backward, update and spectral normalization, and graph_batch_encode which feeds bulkImportFromGPU.

fractal_lpu.zig partitions memory as a self-similar quadtree rather than a flat pool, matching the fractal structure of the graph it serves. Each tile owns a base address, a size, up to four children and a compute unit array sized two to the power of its level capped at six, with coherence decaying by a factor of 0.9 at each level down. Subdivision stops on any of four conditions: the tile is at or below the minimum size, the level has reached the box-counting limit, there are fewer than four child slots, or a quarter-sized child would fall below the minimum. Mapping a graph node into a tile clamps its weight, records it in the tile's entanglement map, and increments the pending operation count on the compute unit chosen by hash. Load balancing caps any unit exceeding the average by more than the balance factor. Fixed-point execution converts coherence into a Q16.16 scale, distributes the input across compute units, and saturates explicitly at the signed 32-bit bounds rather than wrapping. Defaults are a Hausdorff dimension of 1.5, four box-counting levels, a 4096-byte minimum tile and a coherence threshold of 0.7.


Distributed training

gpu_coordinator.zig owns everything that touches NCCL and the CUDA runtime. Errors are never swallowed: each call goes through a checker that prints the driver's own error string with a tag identifying the call site before returning a Zig error, and the cleanup paths use logging variants that report without failing, so a teardown after an error does not mask the original cause.

Initialization validates in order: world size nonzero, rank below world size, both representable as a C int, at least one visible device, and local rank below the device count. Then it makes a deliberate choice. When world size is one, ncclCommInitRank is skipped entirely, because NCCL probes the network stack including InfiniBand and shared memory even for a single rank, producing warnings and costing about half a second for a communicator that will never communicate. After that it creates a stream, allocates and zeroes a single-float device buffer used as a barrier target, and synchronizes, with rollback registered at each step.

The collective surface covers all-reduce in f32 and f16 with sum, average, max and min variants, broadcast in both precisions, all-gather, reduce-scatter, and a barrier implemented as an all-reduce over that one float. Two comptime helpers convert pointers and slices to opaque pointers, rejecting unbounded pointer types with a compile error rather than a runtime check, refusing const where mutability is required, and returning an error for empty slices.

distributed_trainer_futhark.zig is the loop itself. Checkpoints open with the eight bytes JAIDECKP and close with a trailer of 0xDEADBEEF at format version seven, so a truncated write is detectable from both ends. TrainerConfig carries more than forty defaulted fields, and its character is defensive: alongside the expected learning rate of 0.001, gradient clip norm of 1.0, five spectral iterations and a default sequence length of 256, it sets hard ceilings on identifier length, node count, edge group count and node data length, and a maximum distributed integer of 16,777,216, which is the largest integer f32 represents exactly and therefore the point past which gradient accumulation would start silently losing counts. The relational pass interval of fifty controls how often the graph subsystems run relative to plain training steps, and the ESSO temperature and cooling rate govern the topology optimizer that runs alongside.

One detail is worth singling out. The tokenizer language type is not imported but derived reflectively from the fifth parameter of MGT.init, so if the tokenizer signature changes the trainer fails to compile rather than silently passing the wrong enum.

main_distributed_futhark.zig is the process entry point and handles the parts that involve the outside world. The NCCL unique identifier is exchanged through the filesystem: rank zero deletes any stale ready marker, writes the identifier, then creates the marker, while every other rank polls for the marker before reading. Dataset loading is rank-aware in two passes, first parsing each JSONL line for a text field and skipping malformed or empty records, then re-reading and keeping only the half-open index range belonging to this rank, erroring out if the count does not match the expected shard size rather than training on a short shard. Stage failures are aggregated across all ranks, so one rank cannot report success while another has failed. Passing a deploy flag switches the binary into a Modal submission mode that polls job status every thirty seconds for up to three hours.


Inference server

The server is where the two halves of the system meet in one process. Its state struct holds thirty fields, and the list is the clearest statement of what a JAIDE inference actually involves: the model, the SSI index and the ranker, a learned embedding, an NSIR relational graph, a chaos kernel, an ESSO optimizer, surprise memory, a temporal graph, a verified inference engine, a signal propagation engine, a Z runtime, a fractal LPU, a relational graph processing unit, a VPU, an FNDS manager, a CREV pipeline, a formal verification engine, a security proof engine and a quantum task adapter, alongside a rate limiter, an atomic request counter, an atomic active connection count and a thread pool. Most are optional, so the server degrades to a plain forward pass when a subsystem fails to initialize instead of refusing to start.

Three endpoints are served. A GET to /v1/health returns a status string, uptime in seconds, whether a model is loaded and a version. A POST to /v1/inference takes a text field and a token budget and returns the generated token ids, optionally the decoded text, the input tokens, embeddings, and the processing time in milliseconds. A POST to /v1/batch_inference takes a list of texts and runs them through the same path.

Connections are handled by a thread pool sized from num_worker_threads, defaulting to four. Each accepted socket gets a keep-alive timeout, defaulting to five seconds, applied directly to the socket, and the handler loop reads the connection header to decide whether to serve another request on the same socket or close. Rate limiting keys a map of timestamp logs by client address over a sixty second window, with a map-level mutex and a per-entry mutex so two clients never serialize against each other. When require_api_key is set the server reads JAIDE_API_KEY from the environment and compares it against the authorization header on every request.

One configuration detail deserves attention before deployment. The ServerConfig struct defaults to binding 127.0.0.1, ten requests per minute, a sixteen megabyte body cap and API keys required. The main function overrides all four, to 0.0.0.0, sixty per minute, one megabyte and keys off. The shipped binary therefore listens on every interface with authentication disabled unless the operator passes the flag explicitly. The command line accepts port, host, model and require-api-key options; the help text also lists a dataset option that is never parsed, and JAIDE_MODEL_PATH fills the model path only when the flag was absent.


Formal verification

The Lean proof establishes that the OFTB block is exactly invertible, and it is written to be believable rather than merely present. It imports no Mathlib, so nothing is assumed beyond Lean 4 core, and it rebuilds the arithmetic it needs from first principles: successor injectivity, the two addition lemmas on naturals, congruence for cons, append and addition, and a split_at function with proofs that the halves have the expected lengths, that concatenating them recovers the original list, and that splitting a concatenation at the length of its first part returns exactly the two parts.

The central move is proving over an algebraic structure instead of over floating point. ScaleAlgebra axiomatizes addition, subtraction and multiplication together with a scale constant and the values one and two, and demands distributivity over both addition and subtraction, commutativity and associativity of multiplication, multiplication by one, four butterfly identities, and one final law: two times scale, times scale, equals one. That last axiom is the abstract content of two times one over root two, times one over root two, equalling one. Any type satisfying these laws is covered, so the theorem is about the algebra the implementation performs, not about the rounding behaviour of f32.

From there the argument is mechanical. Proofs for the first and second components show the forward pair recovers the originals, with reversed variants for the other direction, and prove_inverse_pair packages each as a conjunction. zipWith_inverse_list lifts both directions elementwise over lists of equal length by structural induction, with the impossible length cases discharged through the successor lemmas. Two length theorems then show that forwardCore and backwardCore each preserve the total length of dim plus dim, which is what allows the round-trip statement to construct the precondition it needs for the second application. The final theorems, mix_round_trip and mix_round_trip_rev, state that composing the two directions in either order returns the input unchanged.

The file also mirrors the runtime validation the Zig code performs, so the guards are proved rather than assumed. Two theorems establish that an empty shape and any shape containing a zero dimension are both rejected, and another that a transform on a list whose length is not exactly twice the dimension is rejected. The maximum representable size is written out in full as 18446744073709551615 rather than referenced symbolically.


Zero-knowledge circuit

The Circom circuit proves that a specific inference ran correctly without revealing the weights that produced it. It targets Circom 2.1.8, pulls poseidon, comparators, bitify and mux1 from circomlib, and works in fixed point at a scale of one million. Constants are written as functions rather than literals so the same value can parameterize a template and appear inside its body.

Soundness rests on a small primitive worth reading closely. SafeIsZero computes an inverse as an unconstrained witness, multiplies the input by it, and takes the output as one minus that product. Written that way alone, a malicious prover could supply any inverse. The template therefore adds a final constraint that the input times the output must equal zero, which forces the witness to be genuine, and SafeIsEqual builds on top of it.

The hard part of the circuit is the exponential. The RSF scale is an exponential, but exponentials are not natively expressible in an arithmetic circuit, so RSFLayerComputation approximates it with a third-order Taylor expansion in fixed point, assembling a numerator from the constant, linear, quadratic and cubic terms with the appropriate scale factors and then dividing by six times the cube of the scale through SignedDivByConstant with an explicit remainder bit bound. The intermediate is range-checked with a sixty-five bit decomposition so the field element cannot silently wrap. The layer template asserts that the dimension is positive and even, splits into halves exactly as the Zig code does, accumulates the weighted sums through a chain of partial signals so each constraint stays quadratic, and reads the bias from the final column of the weight matrix, mirroring the homogeneous-coordinate layout used in the model.

Around it sit the supporting templates: signed absolute value, division by a constant, Poseidon commitment and chaining, Merkle proof verification, range proofs, batch inference verification, a noise bound check that verifies every coordinate of a perturbation stays under a limit, aggregation verification, a differential privacy proof and a secure aggregation proof across participants.

FullInferenceProof composes them into the top level. It takes the input tokens, the per-layer weight matrices, an expected output, input and output commitments, a commitment per layer and a maximum squared error. It hashes the input through a Poseidon chain and checks it against the supplied commitment, threads the tokens through every layer in sequence carrying the intermediate outputs, checks each layer against its own commitment, and folds all the per-layer validity flags into a single output by multiplication, so one failure anywhere zeroes the result. The instantiated main component is eight layers at dimension thirty-two with sixty-four precision bits, declaring the tokens, expected output and both commitments public.


by Kollár Jade
