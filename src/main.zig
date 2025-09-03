const std = @import("std");
const builtin = @import("builtin");

const Tokenizer = @import("Tokenizer.zig");
const Parser = @import("Parser.zig");
const Ast = @import("Ast.zig");

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
    ;
    const ast: Ast = try .parse(gpa, code);
    for (0..ast.tokens.len) |i| {
        const token = ast.tokens.get(i);
        if (token.tag == .invalid) return error.WTF;
        try stdout.print("Result: {any}\n", .{token});
        try stdout.print("Raw: ###{s}###\n", .{code[token.loc.start..token.loc.end]});
    }
}

test {
    // TODO: Delete me
    _ = @import("Runtime.zig");
    _ = @import("Compiler.zig");
    std.testing.refAllDecls(@This());
}
