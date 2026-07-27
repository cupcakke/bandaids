const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const gpu_enabled = b.option(bool, "gpu", "Enable GPU/CUDA via Futhark CUDA backend") orelse false;
    const zk_enabled = b.option(bool, "zk", "Compile Circom ZK circuits") orelse false;
    const verify_enabled = b.option(bool, "verify", "Run Lean verification") orelse false;
    const rtl_enabled = b.option(bool, "rtl", "Compile Haskell RTL simulation") orelse false;

    const cpu_build_options = b.addOptions();
    cpu_build_options.addOption(bool, "gpu_acceleration", false);
    cpu_build_options.addOption(bool, "zk_enabled", zk_enabled);
    cpu_build_options.addOption(bool, "verify_enabled", verify_enabled);
    cpu_build_options.addOption(bool, "rtl_enabled", rtl_enabled);

    const gpu_build_options = b.addOptions();
    gpu_build_options.addOption(bool, "gpu_acceleration", gpu_enabled);
    gpu_build_options.addOption(bool, "zk_enabled", zk_enabled);
    gpu_build_options.addOption(bool, "verify_enabled", verify_enabled);
    gpu_build_options.addOption(bool, "rtl_enabled", rtl_enabled);

    const futhark_c = b.path("src/hw/accel/futhark_kernels.c");
    const futhark_gpu_c = b.path("src/hw/accel/main_gpu.c");
    const futhark_include = b.path("src/hw/accel");

    const futhark_cpu_step = b.addSystemCommand(&.{
        "futhark", "c", "--library",
        "src/hw/accel/futhark_kernels.fut",
        "-o",      "src/hw/accel/futhark_kernels",
    });

    const futhark_gpu_step = b.addSystemCommand(&.{
        "futhark", "cuda", "--library",
        "src/hw/accel/main.fut",
        "-o",      "src/hw/accel/main_gpu",
    });

    const core_relational_mod = b.createModule(.{
        .root_source_file = b.path("src/core_relational/mod.zig"),
        .target = target,
        .optimize = optimize,
    });

    const inference_server_exe = b.addExecutable(.{
        .name = "jaide-inference-server",
        .root_source_file = b.path("src/inference_server_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    inference_server_exe.linkLibC();
    inference_server_exe.addIncludePath(futhark_include);
    inference_server_exe.addCSourceFile(.{ .file = futhark_c, .flags = &.{"-O2"} });
    inference_server_exe.step.dependOn(&futhark_cpu_step.step);
    inference_server_exe.root_module.addOptions("build_options", cpu_build_options);
    inference_server_exe.root_module.addImport("core_relational", core_relational_mod);
    b.installArtifact(inference_server_exe);

    if (gpu_enabled) {
        const distributed_futhark_exe = b.addExecutable(.{
            .name = "jaide-distributed-futhark",
            .root_source_file = b.path("src/main_distributed_futhark.zig"),
            .target = target,
            .optimize = optimize,
        });
        distributed_futhark_exe.linkLibC();
        // The CUDA-generated library (main_gpu.c, compiled via `futhark cuda
        // --library` from main.fut) already contains every GPU entry point the
        // distributed trainer needs and re-uses the same futhark_context_* and
        // futhark_entry_* symbol names as the CPU library generated from
        // futhark_kernels.fut. Linking both object files into the same binary
        // is therefore a duplicate-symbol linker error, not a missing-symbol
        // one, so only the CUDA library is linked here.
        distributed_futhark_exe.addCSourceFile(.{ .file = futhark_gpu_c, .flags = &.{"-O2"} });
        distributed_futhark_exe.addIncludePath(futhark_include);
        distributed_futhark_exe.addIncludePath(.{ .cwd_relative = "/usr/local/cuda/include" });
        distributed_futhark_exe.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/lib64" });
        distributed_futhark_exe.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/lib64/stubs" });
        distributed_futhark_exe.linkSystemLibrary("cuda");
        distributed_futhark_exe.linkSystemLibrary("cudart");
        distributed_futhark_exe.linkSystemLibrary("nvrtc");
        distributed_futhark_exe.linkSystemLibrary("nccl");
        distributed_futhark_exe.linkSystemLibrary("m");
        distributed_futhark_exe.linkSystemLibrary("pthread");
        distributed_futhark_exe.linkSystemLibrary("dl");
        distributed_futhark_exe.root_module.addOptions("build_options", gpu_build_options);
        distributed_futhark_exe.root_module.addImport("core_relational", core_relational_mod);
        distributed_futhark_exe.step.dependOn(&futhark_gpu_step.step);
        b.installArtifact(distributed_futhark_exe);

        const distributed_futhark_install = b.addInstallArtifact(distributed_futhark_exe, .{});
        const distributed_futhark_step = b.step("distributed-futhark", "Build only the Futhark-accelerated distributed trainer");
        distributed_futhark_step.dependOn(&distributed_futhark_install.step);
    }

    const TestSpec = struct {
        step: []const u8,
        wrapper: []const u8,
        desc: []const u8,
    };

    const test_specs = [_]TestSpec{
        .{ .step = "test-tensor", .wrapper = "src/test_root_tensor.zig", .desc = "Run tensor tests" },
        .{ .step = "test-memory", .wrapper = "src/test_root_memory.zig", .desc = "Run memory tests" },
        .{ .step = "test-embedding", .wrapper = "src/test_root_embedding.zig", .desc = "Run embedding tests" },
        .{ .step = "test-rsf", .wrapper = "src/test_root_rsf.zig", .desc = "Run RSF tests" },
        .{ .step = "test-oftb", .wrapper = "src/test_root_oftb.zig", .desc = "Run OFTB tests" },
        .{ .step = "test-nsir", .wrapper = "src/test_root_nsir.zig", .desc = "Run NSIR graph tests" },
        .{ .step = "test-reasoning", .wrapper = "src/test_root_reasoning.zig", .desc = "Run reasoning orchestrator tests" },
        .{ .step = "test-crev", .wrapper = "src/test_root_crev.zig", .desc = "Run CREV pipeline tests" },
        .{ .step = "test-surprise", .wrapper = "src/test_root_surprise.zig", .desc = "Run surprise memory tests" },
        .{ .step = "test-temporal", .wrapper = "src/test_root_temporal.zig", .desc = "Run temporal graph tests" },
        .{ .step = "test-vpu", .wrapper = "src/test_root_vpu.zig", .desc = "Run VPU tests" },
        .{ .step = "test-fnds", .wrapper = "src/test_root_fnds.zig", .desc = "Run FNDS tests" },
        .{ .step = "test-formal", .wrapper = "src/test_root_formal.zig", .desc = "Run formal verification tests" },
        .{ .step = "test-security", .wrapper = "src/test_root_security.zig", .desc = "Run security proofs tests" },
        .{ .step = "test-quantum-adapter", .wrapper = "src/test_root_quantum_adapter.zig", .desc = "Run quantum task adapter tests" },
        .{ .step = "test-signal", .wrapper = "src/test_root_signal.zig", .desc = "Run signal propagation tests" },
        .{ .step = "stress-refcount", .wrapper = "src/test_root_stress_refcount.zig", .desc = "Run tensor refcount stress test" },
    };

    const test_all_step = b.step("test-all", "Run all tests");

    inline for (test_specs) |spec| {
        const t = b.addTest(.{
            .root_source_file = b.path(spec.wrapper),
            .target = target,
            .optimize = optimize,
        });
        t.linkLibC();
        t.addCSourceFile(.{ .file = futhark_c, .flags = &.{"-O2"} });
        t.addIncludePath(futhark_include);
        t.step.dependOn(&futhark_cpu_step.step);
        t.root_module.addOptions("build_options", cpu_build_options);
        const run = b.addRunArtifact(t);
        const step = b.step(spec.step, spec.desc);
        step.dependOn(&run.step);
        test_all_step.dependOn(&run.step);
    }

    const c_api_test = b.addExecutable(.{
        .name = "jaide-c-api-test",
        .target = target,
        .optimize = optimize,
    });
    c_api_test.addCSourceFile(.{ .file = b.path("src/tests/c_api_test.c"), .flags = &.{"-O2"} });
    c_api_test.addIncludePath(b.path("src/core_relational"));
    c_api_test.linkLibC();
    b.installArtifact(c_api_test);
    const c_api_test_run = b.addRunArtifact(c_api_test);
    const c_api_test_step = b.step("test-c-api", "Run C API test");
    c_api_test_step.dependOn(&c_api_test_run.step);
    test_all_step.dependOn(&c_api_test_run.step);

    if (zk_enabled) {
        const circom_step = b.addSystemCommand(&.{
            "circom",   "src/zk/inference_trace.circom",
            "--r1cs",   "--wasm",
            "--sym",    "-o",
            "src/zk/",
        });
        const snarkjs_setup = b.addSystemCommand(&.{
            "snarkjs",                        "groth16",
            "setup",                          "src/zk/inference_trace.r1cs",
            "pot12_final.ptau",               "src/zk/inference_trace.zkey",
        });
        snarkjs_setup.step.dependOn(&circom_step.step);
        const snarkjs_vkey = b.addSystemCommand(&.{
            "snarkjs",              "zkey",
            "export",               "verificationkey",
            "src/zk/inference_trace.zkey",
            "src/zk/verification_key.json",
        });
        snarkjs_vkey.step.dependOn(&snarkjs_setup.step);
        const zk_step = b.step("zk", "Compile ZK circuits");
        zk_step.dependOn(&snarkjs_vkey.step);
    }

    if (verify_enabled) {
        const lean_step = b.addSystemCommand(&.{ "lake", "build" });
        lean_step.setCwd(b.path("src/verification"));
        const verify_step = b.step("verify", "Run Lean formal verification");
        verify_step.dependOn(&lean_step.step);
        test_all_step.dependOn(&lean_step.step);
    }

    if (rtl_enabled) {
        const ghc_step = b.addSystemCommand(&.{
            "ghc", "-O2", "-dynamic", "-shared", "-fPIC",
            "src/hw/rtl/MemoryArbiter.hs",
            "src/hw/rtl/RankerCore.hs",
            "src/hw/rtl/SSISearch.hs",
            "-o", "src/hw/rtl/librtl_sim.so",
        });
        const rtl_exe = b.addExecutable(.{
            .name = "jaide-rtl-sim",
            .root_source_file = b.path("src/hw/rtl/rtl_sim_main.zig"),
            .target = target,
            .optimize = optimize,
        });
        rtl_exe.linkLibC();
        rtl_exe.step.dependOn(&ghc_step.step);
        b.installArtifact(rtl_exe);
        const rtl_install = b.addInstallArtifact(rtl_exe, .{});
        const rtl_step = b.step("rtl", "Build Haskell RTL simulation");
        rtl_step.dependOn(&rtl_install.step);
    }

    const bench_deps = b.createModule(.{
        .root_source_file = b.path("src/_bench_deps.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_deps.addOptions("build_options", cpu_build_options);

    const bench_step = b.step("bench", "Run all benchmarks");

    const bench_sources = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "bench-rsf", .path = "src/tests/bench_rsf.zig" },
        .{ .name = "bench-matmul", .path = "src/tests/bench_matmul.zig" },
        .{ .name = "bench-tensor-ops", .path = "src/tests/bench_tensor_ops.zig" },
        .{ .name = "bench-sfd", .path = "src/tests/bench_sfd.zig" },
    };

    inline for (bench_sources) |src| {
        const exe = b.addExecutable(.{
            .name = src.name,
            .root_source_file = b.path(src.path),
            .target = target,
            .optimize = optimize,
        });
        exe.linkLibC();
        exe.addIncludePath(futhark_include);
        exe.addCSourceFile(.{ .file = futhark_c, .flags = &.{"-O2"} });
        exe.step.dependOn(&futhark_cpu_step.step);
        exe.root_module.addOptions("build_options", cpu_build_options);
        exe.root_module.addImport("deps", bench_deps);
        b.installArtifact(exe);
        const run = b.addRunArtifact(exe);
        bench_step.dependOn(&run.step);
    }
}
