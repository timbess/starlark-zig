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

    pub fn @"type"(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        const result = try Runtime.StarStr.init(rt.gc, args[0].vtable.type_name);
        return &result.obj;
    }

    pub fn float(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        if (Runtime.downCast(Runtime.StarFloat, args[0])) |_| {
            return args[0];
        } else |_| {}
        if (Runtime.downCast(Runtime.StarInt, args[0])) |i| {
            const result = try Runtime.StarFloat.init(rt.gc, @floatFromInt(i.num));
            return &result.obj;
        } else |_| {}
        if (Runtime.downCast(Runtime.StarStr, args[0])) |s| {
            const n = std.fmt.parseFloat(f64, s.str) catch return Runtime.RuntimeError.TypeMismatch;
            const result = try Runtime.StarFloat.init(rt.gc, n);
            return &result.obj;
        } else |_| {}
        return Runtime.RuntimeError.TypeMismatch;
    }

    pub fn str(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        if (try args[0].getMethodDunder(.str)) |str_method| {
            return try str_method.call(rt.gc, &.{});
        }
        return Runtime.RuntimeError.TypeMismatch;
    }

    pub fn len(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;

        if (Runtime.downCast(Runtime.StarList, args[0])) |list| {
            const result = try Runtime.StarInt.init(rt.gc, @intCast(list.items.items.len));
            return &result.obj;
        } else |_| if (Runtime.downCast(Runtime.StarStr, args[0])) |s| {
            const result = try Runtime.StarInt.init(rt.gc, @intCast(s.str.len));
            return &result.obj;
        } else |_| {
            return Runtime.TypeError.TypeMismatch;
        }
    }
});
