const std = @import("std");
const Runtime = @import("Runtime.zig");
const Gc = @import("zig_libgc");

pub const Stdlib = Runtime.StarNativeModule(struct {
    pub fn concat(rt: *Runtime, args: struct { a: []const u8, b: []const u8 }) Runtime.Error!?*Runtime.StarObj {
        const new = try rt.gc.create(Runtime.StarStr);
        new.* = .{ .str = try std.mem.concat(rt.gc, u8, &.{ args.a, args.b }) };
        return &new.obj;
    }

    pub fn print(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        const out = rt.output;
        defer out.flush() catch @panic("Failed to flush");

        for (args, 0..) |arg, i| {
            if (i > 0) try out.print(" ", .{});

            if (try arg.getMethodDunder(.str)) |str_method| {
                const str_obj = try str_method.call(rt, &.{});
                const str_val = try Runtime.downCast(Runtime.StarStr, str_obj);
                try out.print("{s}", .{str_val.str});
            } else {
                try out.print("<object>", .{});
            }
        }
        try out.print("\n", .{});
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
        if (args.len == 0) {
            const result = try Runtime.StarFloat.init(rt.gc, 0.0);
            return &result.obj;
        }
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        if (Runtime.downCast(Runtime.StarFloat, args[0])) |_| {
            return args[0];
        } else |_| {}
        if (Runtime.downCast(Runtime.StarInt, args[0])) |i| {
            const result = try Runtime.StarFloat.init(rt.gc, @floatFromInt(i.num));
            return &result.obj;
        } else |_| {}
        if (Runtime.downCast(Runtime.StarBool, args[0])) |b| {
            const result = try Runtime.StarFloat.init(rt.gc, if (b.value) 1.0 else 0.0);
            return &result.obj;
        } else |_| {}
        if (Runtime.downCast(Runtime.StarStr, args[0])) |s| {
            const n = std.fmt.parseFloat(f64, s.str) catch return Runtime.RuntimeError.TypeMismatch;
            const result = try Runtime.StarFloat.init(rt.gc, n);
            return &result.obj;
        } else |_| {}
        return Runtime.RuntimeError.TypeMismatch;
    }

    pub fn int(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len == 0) {
            const result = try Runtime.StarInt.init(rt.gc, 0);
            return &result.obj;
        }
        if (args.len > 2) return Runtime.RuntimeError.ArityMismatch;
        if (Runtime.downCast(Runtime.StarInt, args[0])) |_| {
            return args[0];
        } else |_| {}
        if (Runtime.downCast(Runtime.StarBool, args[0])) |b| {
            const result = try Runtime.StarInt.init(rt.gc, if (b.value) 1 else 0);
            return &result.obj;
        } else |_| {}
        if (Runtime.downCast(Runtime.StarFloat, args[0])) |f| {
            const result = try Runtime.StarInt.init(rt.gc, @intFromFloat(@trunc(f.num)));
            return &result.obj;
        } else |_| {}
        if (Runtime.downCast(Runtime.StarStr, args[0])) |s| {
            const base: u8 = if (args.len == 2) blk: {
                const b = try Runtime.downCast(Runtime.StarInt, args[1]);
                break :blk @intCast(b.num);
            } else 10;
            const n = std.fmt.parseInt(i64, s.str, base) catch return Runtime.RuntimeError.TypeMismatch;
            const result = try Runtime.StarInt.init(rt.gc, n);
            return &result.obj;
        } else |_| {}
        return Runtime.RuntimeError.TypeMismatch;
    }

    pub fn @"bool"(_: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len == 0) return Runtime.StarBool.get(false);
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        return Runtime.StarBool.get(args[0].isTruthy());
    }

    pub fn tuple(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len == 0) {
            const t = try Runtime.StarTuple.init(rt.gc, &.{});
            return &t.obj;
        }
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        const items = Runtime.iterateToOwned(rt, args[0]) catch {
            rt.setErrorMessage( "tuple: got {s}, want iterable", .{args[0].vtable.type_name});
            return Runtime.RuntimeError.TypeMismatch;
        };
        const t = try Runtime.StarTuple.init(rt.gc, items);
        return &t.obj;
    }

    pub fn list(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len == 0) {
            const l = try Runtime.StarList.init(rt.gc, &.{});
            return &l.obj;
        }
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        const items = Runtime.iterateToOwned(rt, args[0]) catch {
            rt.setErrorMessage( "list: got {s}, want iterable", .{args[0].vtable.type_name});
            return Runtime.RuntimeError.TypeMismatch;
        };
        const l = try Runtime.StarList.init(rt.gc, items);
        return &l.obj;
    }

    pub fn dict(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len == 0) {
            const d = try Runtime.StarDict.init(rt, &.{});
            return &d.obj;
        }
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        const d = try Runtime.StarDict.init(rt, &.{});
        if (Runtime.downCast(Runtime.StarDict, args[0])) |src| {
            for (src.keys.items, src.values.items) |k, v| {
                try d.put(rt, k, v);
            }
            return &d.obj;
        } else |_| {}
        const items = try Runtime.iterateToOwned(rt, args[0]);
        for (items) |it| {
            if (Runtime.downCast(Runtime.StarTuple, it)) |tup| {
                if (tup.items.len != 2) return Runtime.RuntimeError.TypeMismatch;
                try d.put(rt, tup.items[0], tup.items[1]);
            } else |_| if (Runtime.downCast(Runtime.StarList, it)) |li| {
                if (li.items.items.len != 2) return Runtime.RuntimeError.TypeMismatch;
                try d.put(rt, li.items.items[0], li.items.items[1]);
            } else |_| {
                return Runtime.RuntimeError.TypeMismatch;
            }
        }
        return &d.obj;
    }

    pub fn dir(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        var names = std.ArrayList(*Runtime.StarObj).empty;
        var it = args[0].attributes.iterator();
        while (it.next()) |e| {
            const name = e.key_ptr.*;
            if (std.mem.startsWith(u8, name, "__") and std.mem.endsWith(u8, name, "__")) continue;
            const dup = try rt.gc.dupe(u8, name);
            try names.append(rt.gc, &(try Runtime.StarStr.init(rt.gc, dup)).obj);
        }
        std.mem.sort(*Runtime.StarObj, names.items, {}, struct {
            fn lt(_: void, a: *Runtime.StarObj, b: *Runtime.StarObj) bool {
                const sa = Runtime.downCast(Runtime.StarStr, a) catch return false;
                const sb = Runtime.downCast(Runtime.StarStr, b) catch return false;
                return std.mem.order(u8, sa.str, sb.str) == .lt;
            }
        }.lt);
        const l = try Runtime.StarList.init(rt.gc, names.items);
        return &l.obj;
    }

    pub fn abs(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        if (Runtime.downCast(Runtime.StarInt, args[0])) |i| {
            const v: i64 = if (i.num < 0) -i.num else i.num;
            const r = try Runtime.StarInt.init(rt.gc, v);
            return &r.obj;
        } else |_| {}
        if (Runtime.downCast(Runtime.StarFloat, args[0])) |f| {
            const r = try Runtime.StarFloat.init(rt.gc, @abs(f.num));
            return &r.obj;
        } else |_| {}
        return Runtime.RuntimeError.TypeMismatch;
    }

    pub fn min(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len == 0) return Runtime.RuntimeError.ArityMismatch;
        const items: []const *Runtime.StarObj = if (args.len == 1)
            try Runtime.iterateToOwned(rt, args[0])
        else
            args;
        if (items.len == 0) return Runtime.RuntimeError.TypeMismatch;
        var best = items[0];
        for (items[1..]) |it| {
            const m = try best.getMethodDunder(.gt) orelse return Runtime.RuntimeError.TypeMismatch;
            const r = try m.call(rt, &.{it});
            const b = try Runtime.downCast(Runtime.StarBool, r);
            if (b.value) best = it;
        }
        return best;
    }

    pub fn max(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len == 0) return Runtime.RuntimeError.ArityMismatch;
        const items: []const *Runtime.StarObj = if (args.len == 1)
            try Runtime.iterateToOwned(rt, args[0])
        else
            args;
        if (items.len == 0) return Runtime.RuntimeError.TypeMismatch;
        var best = items[0];
        for (items[1..]) |it| {
            const m = try best.getMethodDunder(.lt) orelse return Runtime.RuntimeError.TypeMismatch;
            const r = try m.call(rt, &.{it});
            const b = try Runtime.downCast(Runtime.StarBool, r);
            if (b.value) best = it;
        }
        return best;
    }

    pub fn sum(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len < 1 or args.len > 2) return Runtime.RuntimeError.ArityMismatch;
        const items = try Runtime.iterateToOwned(rt, args[0]);
        const start_obj: *Runtime.StarObj = if (args.len == 2) args[1] else &(try Runtime.StarInt.init(rt.gc, 0)).obj;
        var acc = start_obj;
        for (items) |it| {
            const m = try acc.getMethodDunder(.add) orelse return Runtime.RuntimeError.TypeMismatch;
            acc = try m.call(rt, &.{it});
        }
        return acc;
    }

    pub fn sorted(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        const items = try Runtime.iterateToOwned(rt, args[0]);
        const Cmp = struct {
            rt: *Runtime,
            fail: *bool,
            fn lt(self: @This(), a: *Runtime.StarObj, b: *Runtime.StarObj) bool {
                const m = a.getMethodDunder(.lt) catch {
                    self.fail.* = true;
                    return false;
                } orelse {
                    self.fail.* = true;
                    return false;
                };
                const r = m.call(self.rt, &.{b}) catch {
                    self.fail.* = true;
                    return false;
                };
                const bv = Runtime.downCast(Runtime.StarBool, r) catch {
                    self.fail.* = true;
                    return false;
                };
                return bv.value;
            }
        };
        var fail = false;
        std.mem.sort(*Runtime.StarObj, items, Cmp{ .rt = rt, .fail = &fail }, Cmp.lt);
        if (fail) return Runtime.RuntimeError.TypeMismatch;
        const l = try Runtime.StarList.init(rt.gc, items);
        return &l.obj;
    }

    pub fn reversed(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        const items = try Runtime.iterateToOwned(rt, args[0]);
        std.mem.reverse(*Runtime.StarObj, items);
        const l = try Runtime.StarList.init(rt.gc, items);
        return &l.obj;
    }

    pub fn enumerate(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len < 1 or args.len > 2) return Runtime.RuntimeError.ArityMismatch;
        const items = try Runtime.iterateToOwned(rt, args[0]);
        const start: i64 = if (args.len == 2) (try Runtime.downCast(Runtime.StarInt, args[1])).num else 0;
        const out = try rt.gc.alloc(*Runtime.StarObj, items.len);
        for (items, 0..) |it, i| {
            const pair = try rt.gc.alloc(*Runtime.StarObj, 2);
            pair[0] = &(try Runtime.StarInt.init(rt.gc, start + @as(i64, @intCast(i)))).obj;
            pair[1] = it;
            out[i] = &(try Runtime.StarTuple.init(rt.gc, pair)).obj;
        }
        const l = try Runtime.StarList.init(rt.gc, out);
        return &l.obj;
    }

    pub fn zip(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        var min_len: usize = std.math.maxInt(usize);
        const cols = try rt.gc.alloc([]*Runtime.StarObj, args.len);
        for (args, 0..) |a, i| {
            cols[i] = try Runtime.iterateToOwned(rt, a);
            if (cols[i].len < min_len) min_len = cols[i].len;
        }
        if (args.len == 0) min_len = 0;
        var out = std.ArrayList(*Runtime.StarObj).empty;
        var i: usize = 0;
        while (i < min_len) : (i += 1) {
            const row = try rt.gc.alloc(*Runtime.StarObj, args.len);
            for (cols, 0..) |c, j| row[j] = c[i];
            try out.append(rt.gc, &(try Runtime.StarTuple.init(rt.gc, row)).obj);
        }
        const l = try Runtime.StarList.init(rt.gc, out.items);
        return &l.obj;
    }

    pub fn any(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        const items = try Runtime.iterateToOwned(rt, args[0]);
        for (items) |it| if (it.isTruthy()) return Runtime.StarBool.get(true);
        return Runtime.StarBool.get(false);
    }

    pub fn all(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        const items = try Runtime.iterateToOwned(rt, args[0]);
        for (items) |it| if (!it.isTruthy()) return Runtime.StarBool.get(false);
        return Runtime.StarBool.get(true);
    }

    pub fn repr(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        if (Runtime.downCast(Runtime.StarStr, args[0])) |s| {
            const buf = try rt.gc.alloc(u8, s.str.len + 2);
            buf[0] = '"';
            @memcpy(buf[1 .. 1 + s.str.len], s.str);
            buf[buf.len - 1] = '"';
            const r = try Runtime.StarStr.init(rt.gc, buf);
            return &r.obj;
        } else |_| {}
        if (try args[0].getMethodDunder(.str)) |m| {
            return try m.call(rt, &.{});
        }
        return Runtime.RuntimeError.TypeMismatch;
    }

    pub fn hasattr(_: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 2) return Runtime.RuntimeError.ArityMismatch;
        const name = try Runtime.downCast(Runtime.StarStr, args[1]);
        return Runtime.StarBool.get(args[0].attributes.contains(name.str));
    }

    pub fn getattr(_: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len < 2 or args.len > 3) return Runtime.RuntimeError.ArityMismatch;
        const name = try Runtime.downCast(Runtime.StarStr, args[1]);
        if (args[0].attributes.get(name.str)) |v| return v;
        if (args.len == 3) return args[2];
        return Runtime.RuntimeError.AttributeMissing;
    }

    pub fn str(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
        if (try args[0].getMethodDunder(.str)) |str_method| {
            return try str_method.call(rt, &.{});
        }
        return Runtime.RuntimeError.TypeMismatch;
    }

    pub fn len(rt: *Runtime, args: []const *Runtime.StarObj) Runtime.Error!?*Runtime.StarObj {
        if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;

        if (Runtime.downCast(Runtime.StarList, args[0])) |list_| {
            const result = try Runtime.StarInt.init(rt.gc, @intCast(list_.items.items.len));
            return &result.obj;
        } else |_| if (Runtime.downCast(Runtime.StarStr, args[0])) |s| {
            const result = try Runtime.StarInt.init(rt.gc, @intCast(s.str.len));
            return &result.obj;
        } else |_| if (Runtime.downCast(Runtime.StarTuple, args[0])) |t| {
            const result = try Runtime.StarInt.init(rt.gc, @intCast(t.items.len));
            return &result.obj;
        } else |_| if (Runtime.downCast(Runtime.StarDict, args[0])) |d| {
            const result = try Runtime.StarInt.init(rt.gc, @intCast(d.keys.items.len));
            return &result.obj;
        } else |_| {
            return Runtime.TypeError.TypeMismatch;
        }
    }
});
