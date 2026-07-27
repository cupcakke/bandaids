const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const futhark_c = b.path("hw/accel/futhark_kernels.c");
    const futhark_include = b.path("hw/accel");

    const main_exe = b.addExecutable(.{
        .name = "jaide",
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_exe.linkLibC();
    main_exe.addCSourceFile(.{ .file = futhark_c, .flags = &.{"-O2"} });
    main_exe.addIncludePath(futhark_include);
    b.installArtifact(main_exe);

    const run_cmd = b.addRunArtifact(main_exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the JAIDE main executable");
    run_step.dependOn(&run_cmd.step);
}

