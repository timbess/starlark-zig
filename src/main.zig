const std = @import("std");
const builtin = @import("builtin");

const Tokenizer = @import("Tokenizer.zig");
const Parser = @import("Parser.zig");
const Ast = @import("Ast.zig");
const Compiler = @import("Compiler.zig");
const Runtime = @import("Runtime.zig");

pub fn main() !void {
    var gpa: std.mem.Allocator = std.heap.smp_allocator;

    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    if (builtin.mode == .Debug) {
        gpa = debug_allocator.allocator();
    }

    defer if (builtin.mode == .Debug) {
        _ = debug_allocator.deinit();
    };

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    var stdout = &stdout_writer.interface;
    defer stdout.flush() catch @panic("Failed to flush");

    const code: [:0]const u8 =
        \\asdf = 1
        // \\def foo(a):
        // \\    b = 2
    ;
    var ast: Ast = try .parse(gpa, code);
    defer ast.deinit();

    try stdout.print("AST:\n{f}\n", .{Ast.DebugNodeFormatter{ .data = &ast }});

    const module = try Compiler.compile(gpa, "<main>", code, &ast, .{});
    var runtime = try Runtime.init(gpa, .{});
    defer runtime.deinit();

    try runtime.execModule(&module);
    try runtime.stepUntilDone();

    const asdf_result = runtime.globals.get("asdf");
    const asdf_result_num = try Runtime.downCast(Runtime.StarInt, asdf_result.?);

    try stdout.print("asdf = {d}", .{asdf_result_num.num});
}

test {
    // TODO: Delete me
    // _ = @import("Runtime.zig");
    _ = @import("Compiler.zig");
    std.testing.refAllDecls(@This());
}
