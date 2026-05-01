const std = @import("std");
const Ast = @import("Ast.zig");
const Compiler = @import("Compiler.zig");
const Runtime = @import("Runtime.zig");
const stdlib = @import("stdlib.zig");
const test_predeclared = @import("test_predeclared.zig");

const spec_files = .{
    .{ "assign", @embedFile("testdata/assign.star") },
    .{ "benchmark", @embedFile("testdata/benchmark.star") },
    .{ "bool", @embedFile("testdata/bool.star") },
    .{ "builtins", @embedFile("testdata/builtins.star") },
    .{ "bytes", @embedFile("testdata/bytes.star") },
    .{ "control", @embedFile("testdata/control.star") },
    .{ "dict", @embedFile("testdata/dict.star") },
    .{ "float", @embedFile("testdata/float.star") },
    .{ "function", @embedFile("testdata/function.star") },
    .{ "function_param", @embedFile("testdata/function_param.star") },
    .{ "int", @embedFile("testdata/int.star") },
    .{ "json", @embedFile("testdata/json.star") },
    .{ "list", @embedFile("testdata/list.star") },
    .{ "math", @embedFile("testdata/math.star") },
    .{ "misc", @embedFile("testdata/misc.star") },
    .{ "module", @embedFile("testdata/module.star") },
    .{ "paths", @embedFile("testdata/paths.star") },
    .{ "proto", @embedFile("testdata/proto.star") },
    .{ "recursion", @embedFile("testdata/recursion.star") },
    .{ "set", @embedFile("testdata/set.star") },
    .{ "string", @embedFile("testdata/string.star") },
    .{ "time", @embedFile("testdata/time.star") },
    .{ "tuple", @embedFile("testdata/tuple.star") },
    .{ "while", @embedFile("testdata/while.star") },
};

fn splitChunks(source: []const u8) ChunkIterator {
    return .{ .source = source, .pos = 0 };
}

const ChunkIterator = struct {
    source: []const u8,
    pos: usize,

    fn next(self: *ChunkIterator) ?[]const u8 {
        if (self.pos >= self.source.len) return null;
        const start = self.pos;
        const separator = "\n---\n";
        if (std.mem.indexOfPos(u8, self.source, start, separator)) |sep_pos| {
            self.pos = sep_pos + separator.len;
            return self.source[start..sep_pos];
        } else {
            self.pos = self.source.len;
            const chunk = self.source[start..];
            if (std.mem.trim(u8, chunk, " \t\n\r").len == 0) return null;
            return chunk;
        }
    }
};

fn stripAndNullTerminate(allocator: std.mem.Allocator, chunk: []const u8) ![:0]const u8 {
    // Copy chunk to a new buffer, stripping load() and # option: lines, null-terminated.
    var buf = std.ArrayList(u8).empty;
    var pos: usize = 0;

    while (pos < chunk.len) {
        const line_end = std.mem.indexOfPos(u8, chunk, pos, "\n") orelse chunk.len;
        const line = chunk[pos..line_end];
        const trimmed = std.mem.trimStart(u8, line, " \t");

        const skip = std.mem.startsWith(u8, trimmed, "# option:");

        if (!skip) {
            try buf.appendSlice(allocator, line);
            if (line_end < chunk.len) {
                try buf.append(allocator, '\n');
            }
        }

        pos = if (line_end < chunk.len) line_end + 1 else chunk.len;
    }

    // Null-terminate
    try buf.append(allocator, 0);
    const slice = buf.items[0 .. buf.items.len - 1 :0];
    return slice;
}

fn chunkExpectsError(chunk: []const u8) bool {
    return std.mem.indexOf(u8, chunk, "###") != null;
}

const TestReport = struct {
    file_name: []const u8,
    total_chunks: usize,
    passed_chunks: usize,
    skipped_chunks: usize,

    fn print(self: TestReport) void {
        std.debug.print("  {s: <24} {d}/{d} passed", .{ self.file_name, self.passed_chunks, self.total_chunks });
        if (self.skipped_chunks > 0) {
            std.debug.print(" ({d} error-chunks skipped)", .{self.skipped_chunks});
        }
        std.debug.print("\n", .{});
    }
};

fn runSpecFile(comptime name: []const u8, comptime source: []const u8) TestReport {
    var chunks = splitChunks(source);
    var total: usize = 0;
    var passed: usize = 0;
    var skipped: usize = 0;

    while (chunks.next()) |chunk| {
        total += 1;

        if (chunkExpectsError(chunk)) {
            skipped += 1;
            continue;
        }

        // Try to execute the chunk
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const gc = arena.allocator();

        const result = blk: {
            const null_term = stripAndNullTerminate(gc, chunk) catch break :blk false;
            if (std.mem.trim(u8, null_term, " \t\n\r").len == 0) {
                // Genuinely empty chunk (just whitespace) — count as pass.
                break :blk true;
            }

            var ast = Ast.parse(std.testing.allocator, null_term) catch break :blk false;
            defer ast.deinit();

            const module = Compiler.compile(std.testing.allocator, name, null_term, &ast, .{}, gc) catch break :blk false;
            var rt = Runtime.init(std.testing.allocator, .{ .gc = gc }) catch break :blk false;
            defer rt.deinit();

            rt.registerStdlib(stdlib.Stdlib) catch break :blk false;
            test_predeclared.register(&rt, gc) catch break :blk false;
            rt.execModule(&module) catch {
                test_predeclared.reset();
                break :blk false;
            };

            const err_count = test_predeclared.getErrorCount();
            test_predeclared.reset();
            break :blk err_count == 0;
        };

        if (result) {
            passed += 1;
        }
    }

    return .{
        .file_name = name ++ ".star",
        .total_chunks = total,
        .passed_chunks = passed,
        .skipped_chunks = skipped,
    };
}

test "spec conformance report" {
    std.debug.print("\n\nStarlark Spec Conformance Report\n", .{});
    std.debug.print("================================\n", .{});

    var total_chunks: usize = 0;
    var total_passed: usize = 0;

    inline for (spec_files) |entry| {
        const report = runSpecFile(entry[0], entry[1]);
        report.print();
        total_chunks += report.total_chunks;
        total_passed += report.passed_chunks;
    }

    std.debug.print("--------------------------------\n", .{});
    std.debug.print("  Total: {d}/{d} chunks passing\n\n", .{ total_passed, total_chunks });

    test_predeclared.deinit();

    // Fail the test to ensure the report is always visible in test output.
    // Remove this once all spec tests pass.
    if (total_passed < total_chunks) {
        return error.SpecConformanceIncomplete;
    }
}
