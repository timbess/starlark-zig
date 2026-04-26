const std = @import("std");
const Ast = @import("Ast.zig");
const Compiler = @import("Compiler.zig");
const Runtime = @import("Runtime.zig");

const StarObj = Runtime.StarObj;
const StarFunc = Runtime.StarFunc;
const StarStr = Runtime.StarStr;
const StarFloat = Runtime.StarFloat;
const StarBool = Runtime.StarBool;
const StarNone = Runtime.StarNone;
const Error = Runtime.Error;

var error_messages: std.ArrayList([]const u8) = .empty;
var error_allocator: std.mem.Allocator = undefined;
var initialized = false;

pub fn getErrorCount() usize {
    return error_messages.items.len;
}

pub fn getErrors() []const []const u8 {
    return error_messages.items;
}

pub fn reset() void {
    error_messages.clearRetainingCapacity();
}

pub fn deinit() void {
    if (initialized) {
        error_messages.deinit(error_allocator);
        initialized = false;
    }
}

// The 6 predeclared builtins that assert.star depends on.
// assert.star builds the assert module in pure Starlark using these.

fn errorBuiltin(_: *Runtime, args: []const *Runtime.StarObj) Error!*StarObj {
    if (args.len < 1) return Runtime.RuntimeError.ArityMismatch;
    if (Runtime.downCast(StarStr, args[0])) |msg| {
        error_messages.append(error_allocator, msg.str) catch {};
    } else |_| {
        error_messages.append(error_allocator, "<non-string error>") catch {};
    }
    return StarNone.instance;
}

fn catchBuiltin(rt: *Runtime, args: []const *Runtime.StarObj) Error!*StarObj {
    if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
    const func_obj = args[0];
    const maybe_func = Runtime.downCast(StarFunc, func_obj) catch return Runtime.RuntimeError.CallUndefined;

    if (maybe_func.native_fn) |f| {
        if (f(rt, &.{})) |_| {
            return StarNone.instance;
        } else |err| {
            const err_str = try StarStr.init(rt.gc, @errorName(err));
            return &err_str.obj;
        }
    }

    // Interpreted functions: stub until stackPush/callFn are pub
    return StarNone.instance;
}

fn matchesBuiltin(_: *Runtime, args: []const *Runtime.StarObj) Error!*StarObj {
    if (args.len != 2) return Runtime.RuntimeError.ArityMismatch;
    const pattern = Runtime.downCast(StarStr, args[0]) catch return Runtime.TypeError.TypeMismatch;
    const str = Runtime.downCast(StarStr, args[1]) catch return Runtime.TypeError.TypeMismatch;
    const found = std.mem.indexOf(u8, str.str, pattern.str) != null;
    return StarBool.get(found);
}

fn moduleBuiltin(rt: *Runtime, args: []const *Runtime.StarObj) Error!*StarObj {
    // module(name, **kwargs) — creates a StarObj with kwargs as attributes.
    // Full implementation requires kwargs support in the call convention.
    // For now: creates an empty object. kwargs will populate it once implemented.
    _ = args;
    const obj = try rt.gc.create(StarObj);
    obj.* = .{ .vtable = .{ .name = "module" } };
    return obj;
}

fn freezeBuiltin(_: *Runtime, args: []const *Runtime.StarObj) Error!*StarObj {
    if (args.len != 1) return Runtime.RuntimeError.ArityMismatch;
    return StarNone.instance;
}

fn floateqBuiltin(_: *Runtime, args: []const *Runtime.StarObj) Error!*StarObj {
    if (args.len != 2) return Runtime.RuntimeError.ArityMismatch;
    const a = Runtime.downCast(StarFloat, args[0]) catch return Runtime.TypeError.TypeMismatch;
    const b = Runtime.downCast(StarFloat, args[1]) catch return Runtime.TypeError.TypeMismatch;
    const diff = @abs(a.num - b.num);
    const epsilon = @max(@abs(a.num), @abs(b.num)) * std.math.floatEps(f64);
    return StarBool.get(diff <= epsilon);
}

const assert_star_source = @embedFile("testdata/assert.star");

pub fn register(rt: *Runtime, gc: std.mem.Allocator) !void {
    if (!initialized) {
        error_allocator = std.testing.allocator;
        error_messages = std.ArrayList([]const u8).empty;
        initialized = true;
    }

    const Predeclared = Runtime.StarNativeModule(struct {
        pub fn @"error"(_: *Runtime, args: []const *Runtime.StarObj) Error!?*StarObj {
            return errorBuiltin(undefined, args);
        }

        pub fn catch_fn(r: *Runtime, args: []const *Runtime.StarObj) Error!?*StarObj {
            return catchBuiltin(r, args);
        }

        pub fn matches(r: *Runtime, args: []const *Runtime.StarObj) Error!?*StarObj {
            return matchesBuiltin(r, args);
        }

        pub fn module(r: *Runtime, args: []const *Runtime.StarObj) Error!?*StarObj {
            return moduleBuiltin(r, args);
        }

        pub fn _freeze(r: *Runtime, args: []const *Runtime.StarObj) Error!?*StarObj {
            return freezeBuiltin(r, args);
        }

        pub fn _floateq(r: *Runtime, args: []const *Runtime.StarObj) Error!?*StarObj {
            return floateqBuiltin(r, args);
        }
    });
    try rt.registerStdlib(Predeclared);

    // Execute assert.star to define the assert module.
    // This will fail until tuples, kwargs, default params, and/or, in/not in,
    // % formatting, type(), and float() are implemented.
    execAssertStar(rt, gc) catch {};
}

fn execAssertStar(rt: *Runtime, gc: std.mem.Allocator) !void {
    const source: [:0]const u8 = assert_star_source[0..assert_star_source.len :0];
    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
    const module = try Compiler.compile(std.testing.allocator, "assert.star", source, &ast, .{}, gc);
    try rt.execModule(&module);
}
