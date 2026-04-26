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
    });
    defer runtime.deinit();

    try runtime.registerStdlib(Stdlib);

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

const Stdlib = Runtime.StarNativeModule(struct {
    pub fn concat(rt: *Runtime, args: struct { a: []const u8, b: []const u8 }) Runtime.Error!?*Runtime.StarObj {
        const new = try rt.gc.create(Runtime.StarStr);
        new.* = .{ .str = try std.mem.concat(rt.gc, u8, &.{ args.a, args.b }) };
        return &new.obj;
    }

    pub fn print(_: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        const io = std.Io.Threaded.global_single_threaded.io();
        var stdout_buf: [4096]u8 = undefined;
        var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buf);
        var stdout = &stdout_writer.interface;
        defer stdout.flush() catch @panic("Failed to flush");

        for (args, 0..) |arg, i| {
            if (i > 0) try stdout.print(" ", .{});

            if (try arg.getMethodDunder(.str)) |str_method| {
                const str_obj = try str_method.call(Gc.allocator(), &.{});
                const str_val = try Runtime.downCast(Runtime.StarStr, str_obj);
                try stdout.print("{s}", .{str_val.str});
            } else {
                try stdout.print("<object>", .{});
            }
        }
        try stdout.print("\n", .{});
        return null;
    }

    pub fn range(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        var start: i64 = 0;
        var stop: i64 = 0;
        var step_val: i64 = 1;

        switch (args.len) {
            1 => {
                const stop_int = try Runtime.downCast(Runtime.StarInt, args[0]);
                stop = stop_int.num;
            },
            2 => {
                const start_int = try Runtime.downCast(Runtime.StarInt, args[0]);
                const stop_int = try Runtime.downCast(Runtime.StarInt, args[1]);
                start = start_int.num;
                stop = stop_int.num;
            },
            3 => {
                const start_int = try Runtime.downCast(Runtime.StarInt, args[0]);
                const stop_int = try Runtime.downCast(Runtime.StarInt, args[1]);
                const step_int = try Runtime.downCast(Runtime.StarInt, args[2]);
                start = start_int.num;
                stop = stop_int.num;
                step_val = step_int.num;
            },
            else => return Runtime.RuntimeError.ArityMismatch,
        }

        const iter = try Runtime.StarRangeIter.init(rt.gc, start, stop, step_val);
        return &iter.obj;
    }

    pub fn len(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;

        if (Runtime.downCast(Runtime.StarList, args[0])) |list| {
            const result = try Runtime.StarInt.init(rt.gc, @intCast(list.items.items.len));
            return &result.obj;
        } else |_| if (Runtime.downCast(Runtime.StarStr, args[0])) |str| {
            const result = try Runtime.StarInt.init(rt.gc, @intCast(str.str.len));
            return &result.obj;
        } else |_| {
            return Runtime.TypeError.TypeMismatch;
        }
    }
});

test {
    std.testing.refAllDecls(@This());
}
