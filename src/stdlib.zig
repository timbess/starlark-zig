const std = @import("std");
const Runtime = @import("Runtime.zig");
const Gc = @import("zig_libgc");

pub const Stdlib = Runtime.StarNativeModule(struct {
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
