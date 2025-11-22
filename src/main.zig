const std = @import("std");
const builtin = @import("builtin");

const Tokenizer = @import("Tokenizer.zig");
const Parser = @import("Parser.zig");
const Ast = @import("Ast.zig");
const Compiler = @import("Compiler.zig");
const Runtime = @import("Runtime.zig");
const Gc = @import("zig_libgc");

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        // .{ .scope = .parser, .level = .debug },
    },
};

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

    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    var code_buf: [4096]u8 = undefined;
    var file: ?std.fs.File = null;
    defer if (file) |f| f.close();

    var code_reader: *std.Io.Reader = blk: {
        if (args.len > 1 and !std.mem.eql(u8, args[1], "-")) {
            const f = try std.fs.cwd().openFile(args[1], .{ .mode = .read_only });
            file = f;
            var file_reader = f.reader(&code_buf);
            break :blk &file_reader.interface;
        } else {
            var stdin_reader = std.fs.File.stdin().reader(&code_buf);
            break :blk &stdin_reader.interface;
        }
    };

    var code_buffer: [1024 * 1024 * 5]u8 = undefined;
    const n = try code_reader.readSliceShort(&code_buffer);
    code_buffer[n] = 0;
    const code: [:0]const u8 = code_buffer[0..n :0];

    var ast: Ast = try .parse(gpa, code);
    defer ast.deinit();

    // try stdout.print("AST:\n{f}\n", .{Ast.DebugNodeFormatter{ .data = &ast }});

    const gc = Gc.allocator();
    const module = try Compiler.compile(gpa, "<main>", code, &ast, .{}, gc);
    var runtime = try Runtime.init(gpa, .{ .gc = gc });
    defer runtime.deinit();

    try runtime.registerStdlib(Stdlib);
    try runtime.execModule(&module);
}

const Stdlib = Runtime.StarNativeModule(struct {
    pub fn concat(rt: *Runtime, args: struct { a: []const u8, b: []const u8 }) Runtime.Error!?*Runtime.StarObj {
        const new = try rt.gc.create(Runtime.StarStr);
        new.* = .{ .str = try std.mem.concat(rt.gc, u8, &.{ args.a, args.b }) };
        return &new.obj;
    }

    pub fn print(_: *Runtime, args: []*Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        var stdout_buf: [4096]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
        var stdout = &stdout_writer.interface;
        defer stdout.flush() catch @panic("Failed to flush");

        for (args, 0..) |arg, i| {
            if (i > 0) try stdout.print(" ", .{});

            if (arg.vtable.str) |str_fn| {
                const str_obj = try str_fn(&arg.vtable, Gc.allocator());
                const str_val = try Runtime.downCast(Runtime.StarStr, str_obj);
                try stdout.print("{s}", .{str_val.str});
            } else {
                try stdout.print("<object>", .{});
            }
        }
        try stdout.print("\n", .{});
        return null;
    }
});

test {
    std.testing.refAllDecls(@This());
}
