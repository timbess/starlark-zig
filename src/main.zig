const std = @import("std");
const builtin = @import("builtin");

const Tokenizer = @import("Tokenizer.zig");
const Parser = @import("Parser.zig");
const Ast = @import("Ast.zig");
const Compiler = @import("Compiler.zig");
const Runtime = @import("Runtime.zig");
const Gc = @import("zig_libgc");
const stdlib = @import("stdlib.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        // .{ .scope = .parser, .level = .debug },
    },
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
    var stdout = &stdout_writer.interface;
    defer stdout.flush() catch @panic("Failed to flush");

    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var code_buf: [4096]u8 = undefined;
    var file: ?std.Io.File = null;
    defer if (file) |f| f.close(io);

    var file_reader: ?std.Io.File.Reader = null;
    var code_reader: *std.Io.Reader = blk: {
        if (args.len > 1 and !std.mem.eql(u8, args[1], "-")) {
            const f = try std.Io.Dir.cwd().openFile(io, args[1], .{ .mode = .read_only });
            file = f;
            file_reader = f.reader(io, &code_buf);
            break :blk &file_reader.?.interface;
        } else {
            file_reader = std.Io.File.stdin().reader(io, &code_buf);
            break :blk &file_reader.?.interface;
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
    const module = try Compiler.compile(gpa, if (file) |_| args[1] else "<stdin>", code, &ast, .{}, gc);
    var frame_alloc = std.heap.stackFallback(@sizeOf(Runtime.Frame) * 4096, gc);
    var runtime = try Runtime.init(gpa, .{
        .gc = gc,
        .frame_alloc = frame_alloc.get(),
        .output = stdout,
    });
    defer runtime.deinit();

    try runtime.registerStdlib(stdlib.Stdlib);

    var diag: Runtime.Diagnostic = .{};
    defer diag.deinit();
    runtime.diagnostic = &diag;

    runtime.execModule(&module) catch {
        var stderr_buf: [4096]u8 = undefined;
        var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buf);
        var stderr = &stderr_writer.interface;

        diag.format(stderr) catch {};
        stderr.flush() catch {};
        std.process.exit(1);
    };
}

test {
    std.testing.refAllDecls(@This());
    _ = @import("spec_tests.zig");
    _ = @import("test_predeclared.zig");
}
