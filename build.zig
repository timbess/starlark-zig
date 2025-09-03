const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe_mod = b.addModule("starlark", .{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const libgc = compileLibgc(b, optimize, target) catch @panic("oom");
    exe_mod.addImport("zig_libgc", libgc);

    const modules = [_]*std.Build.Module{
        exe_mod,
        libgc,
    };

    const exe = b.addExecutable(.{
        .name = "starlark",
        .root_module = exe_mod,
    });

    const doc = b.step("doc", "Generate docs");
    const exe_docs = b.addInstallDirectory(.{
        .source_dir = exe.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs/exe", // Install to zig-out/docs/
    });
    doc.dependOn(&exe_docs.step);

    const install_exe = b.addInstallArtifact(exe, .{});
    b.getInstallStep().dependOn(&install_exe.step);

    const run = b.step("run", "Executes starlark binary");
    run.dependOn(&b.addRunArtifact(exe).step);

    const check = b.step("check", "Check without building a full binary");
    check.dependOn(&exe.step);

    const tests = b.step("test", "Run tests");
    const build_tests = b.step("build-test", "Build tests as binary");
    for (modules) |mod| {
        const exe_tests = b.addTest(.{
            .root_module = mod,
        });

        // Build tests without running for debugging.
        const install_exe_tests = b.addInstallArtifact(exe_tests, .{});
        build_tests.dependOn(&install_exe_tests.step);

        // Build & Run tests.
        const run_exe_tests = b.addRunArtifact(exe_tests);
        tests.dependOn(&run_exe_tests.step);
    }
}

fn compileLibgc(b: *std.Build, optimize: std.builtin.OptimizeMode, target: std.Build.ResolvedTarget) !*std.Build.Module {
    const libgc_dep = b.dependency("libgc", .{
        .optimize = optimize,
        .target = target,
        .BUILD_SHARED_LIBS = false,
        // TODO: I think this is what we want since the GC won't be used in a multi threaded context and we don't want it to stop the world or lock anything.
        .enable_threads = false,
        .enable_gc_assertions = optimize == .Debug,
        .enable_gc_debug = optimize == .Debug,
    });

    const cmake_configure = b.addSystemCommand(&.{
        "cmake",
        "-G",
        "Ninja",
        "-B",
    });
    const build_output = cmake_configure.addOutputDirectoryArg("libgc_build");
    cmake_configure.addArg("-S");
    cmake_configure.addDirectoryArg(libgc_dep.path("."));
    cmake_configure.addArgs(&.{
        "-DBUILD_SHARED_LIBS=OFF",
        "-Denable_threads=OFF",
        "-DCMAKE_SUPPRESS_DEVELOPER_WARNINGS=1", // I literally don't care, just build.
        "-Wno-deprecated",
        // cmakeOptFlag("-Denable_gc_assertions={s}", optimize == .Debug),
        "-DCMAKE_BUILD_TYPE=Release",
    });
    // const install_dir = cmake_configure.addPrefixedOutputDirectoryArg("-DCMAKE_INSTALL_PREFIX=", "libgc_install_dir");

    const cmake_build = b.addSystemCommand(&.{
        "cmake",
        "--build",
    });
    cmake_build.addDirectoryArg(build_output);
    // Ninja logs and breaks ZLS :(
    _ = cmake_build.captureStdErr();
    _ = cmake_build.captureStdOut();

    const cmake_install = b.addSystemCommand(&.{
        "cmake",
        "--install",
    });
    cmake_install.addDirectoryArg(build_output);
    cmake_install.addArgs(&.{
        "--prefix",
    });
    const install_dir = cmake_install.addOutputDirectoryArg("libgc_install_dir");

    cmake_install.step.dependOn(&cmake_build.step);

    const libgc_mod = b.createModule(.{
        .root_source_file = b.path("src/zig_libgc.zig"),
        .target = target,
        .optimize = optimize,
    });
    const includeDir = try install_dir.join(b.allocator, "include");
    const libDir = try install_dir.join(b.allocator, "lib");
    libgc_mod.addIncludePath(includeDir);
    libgc_mod.addLibraryPath(libDir);
    libgc_mod.linkSystemLibrary("gc", .{});
    libgc_mod.linkSystemLibrary("cord", .{});
    return libgc_mod;
}
