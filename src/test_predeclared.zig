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
const NativeCall = Runtime.NativeCall;
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
    const func = Runtime.downCast(StarFunc, func_obj) catch return Runtime.RuntimeError.CallUndefined;

    if (func.native_fn) |f| {
        if (f(rt, .{ .args = &.{} })) |_| {
            return StarNone.instance;
        } else |err| {
            const err_str = try StarStr.init(rt.gc, @errorName(err));
            return &err_str.obj;
        }
    }

    // Non-native (interpreted) Star function: invoke it via the runtime's
    // call machinery and capture any error. Mirrors Runtime.execModule's
    // call/stepUntilDepth pattern but unwinds frames + stack on error so
    // the outer interpreter loop sees a clean state.
    const initial_depth = rt.frames.items.len;
    const sp_base = rt.stack.items.len;
    rt.callFn(func, &.{}, sp_base) catch |err| {
        return errString(rt, err);
    };
    rt.stepUntilDepth(initial_depth) catch |err| {
        // Unwind anything pushed during the failed call.
        rt.frames.shrinkRetainingCapacity(initial_depth);
        rt.stack.shrinkRetainingCapacity(sp_base);
        return errString(rt, err);
    };
    // Discard the return value — assert.fails ignores it.
    _ = rt.stackPop() catch {};
    return StarNone.instance;
}

fn errString(rt: *Runtime, err: anyerror) Error!*StarObj {
    // Map specific runtime errors to the user-facing strings the Starlark
    // spec tests expect to match. matchesBuiltin does substring search, so
    // the returned string just has to contain the pattern.
    const text: []const u8 = switch (err) {
        error.StarlarkStackOverflow => "Starlark stack overflow",
        else => @errorName(err),
    };
    const err_str = try StarStr.init(rt.gc, text);
    return &err_str.obj;
}

fn matchesBuiltin(_: *Runtime, args: []const *Runtime.StarObj) Error!*StarObj {
    if (args.len != 2) return Runtime.RuntimeError.ArityMismatch;
    const pattern = Runtime.downCast(StarStr, args[0]) catch return Runtime.TypeError.TypeMismatch;
    const str = Runtime.downCast(StarStr, args[1]) catch return Runtime.TypeError.TypeMismatch;
    const found = std.mem.indexOf(u8, str.str, pattern.str) != null;
    return StarBool.get(found);
}

fn moduleBuiltin(rt: *Runtime, call: NativeCall) Error!*StarObj {
    // module(name, **kwargs): each kwarg becomes an attribute on the returned object.
    const obj = try rt.gc.create(StarObj);
    obj.* = .{ .vtable = .{ .name = "module", .type_name = "module" } };
    for (call.kwargs) |kw| {
        try obj.attributes.put(rt.gc, kw.name, kw.value);
    }
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

fn loadBuiltin(rt: *Runtime, args: []const *Runtime.StarObj) Error!*StarObj {
    if (args.len < 2) return Runtime.RuntimeError.ArityMismatch;
    const path_obj = Runtime.downCast(StarStr, args[0]) catch return Runtime.RuntimeError.TypeMismatch;

    const raw_source: ?[]const u8 = if (std.mem.eql(u8, path_obj.str, "assert.star"))
        assert_star_source
    else
        null;

    const source_bytes = raw_source orelse return StarNone.instance;

    const buf = rt.gc.allocSentinel(u8, source_bytes.len, 0) catch return error.OutOfMemory;
    @memcpy(buf, source_bytes);
    const source: [:0]const u8 = buf;

    var ast = Ast.parse(std.testing.allocator, source) catch return error.TypeMismatch;
    defer ast.deinit();

    const module = Compiler.compile(std.testing.allocator, path_obj.str, source, &ast, .{}, rt.gc) catch return error.TypeMismatch;

    const saved_source = rt.current_source;
    const saved_filename = rt.current_filename;
    defer {
        rt.current_source = saved_source;
        rt.current_filename = saved_filename;
    }
    try rt.execModule(&module);

    return StarNone.instance;
}

pub fn register(rt: *Runtime, gc: std.mem.Allocator) !void {
    _ = gc;
    if (!initialized) {
        error_allocator = std.testing.allocator;
        error_messages = std.ArrayList([]const u8).empty;
        initialized = true;
    }

    const Predeclared = Runtime.StarNativeModule(struct {
        pub fn @"error"(_: *Runtime, args: []const *Runtime.StarObj) Error!?*StarObj {
            return errorBuiltin(undefined, args);
        }
        pub fn @"catch"(r: *Runtime, args: []const *Runtime.StarObj) Error!?*StarObj {
            return catchBuiltin(r, args);
        }
        pub fn matches(r: *Runtime, args: []const *Runtime.StarObj) Error!?*StarObj {
            return matchesBuiltin(r, args);
        }
        pub fn module(r: *Runtime, call: NativeCall) Error!?*StarObj {
            return moduleBuiltin(r, call);
        }
        pub fn _freeze(r: *Runtime, args: []const *Runtime.StarObj) Error!?*StarObj {
            return freezeBuiltin(r, args);
        }
        pub fn _floateq(r: *Runtime, args: []const *Runtime.StarObj) Error!?*StarObj {
            return floateqBuiltin(r, args);
        }
        pub fn load(r: *Runtime, args: []const *Runtime.StarObj) Error!?*StarObj {
            return loadBuiltin(r, args);
        }
    });
    try rt.registerStdlib(Predeclared);
}

test "multi-line expression inside parens (paren_depth tracking)" {
    const source: [:0]const u8 =
        \\def f(x):
        \\    if (x == 1 or
        \\        x == 2):
        \\        pass
        \\
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
}

test "multi-level dedent emits multiple block_end tokens" {
    const source: [:0]const u8 =
        \\def f(x):
        \\    if x:
        \\        if x:
        \\            pass
        \\    return x
        \\
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
}

test "default is evaluated at definition time, not call time" {
    // The default expression `seed` is a global lookup that must resolve when
    // `def f` runs (yielding 7). Mutating `seed` afterwards must not change
    // the bound default.
    const source: [:0]const u8 =
        \\seed = 7
        \\def f(x = seed):
        \\    return x
        \\seed = 99
        \\a = f()
        \\b = f()
    ;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();
    try register(&rt, gc);

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
    const module = try Compiler.compile(std.testing.allocator, "<test>", source, &ast, .{}, gc);
    try rt.execModule(&module);

    const a = try Runtime.downCast(Runtime.StarInt, rt.globals.get("a").?);
    try std.testing.expectEqual(@as(i64, 7), a.num);
    const b = try Runtime.downCast(Runtime.StarInt, rt.globals.get("b").?);
    try std.testing.expectEqual(@as(i64, 7), b.num);
}

test "function with default parameter" {
    const source: [:0]const u8 =
        \\def f(x, msg = "hello"):
        \\    pass
        \\
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
}

test "module call with kwargs spanning multiple lines" {
    const source: [:0]const u8 =
        \\x = module(
        \\    "test",
        \\    a = 1,
        \\    b = 2,
        \\)
        \\
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
}

test "function with docstring as first statement" {
    const source: [:0]const u8 =
        \\def f(x):
        \\    "this is a docstring"
        \\    return x
        \\
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
}

test "load() inside chunk works" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();
    try register(&rt, gc);

    const chunk: [:0]const u8 =
        \\load("assert.star", "assert")
        \\assert.eq(1 + 1, 2)
        \\
    ;

    var ast = try Ast.parse(std.testing.allocator, chunk);
    defer ast.deinit();
    const module = try Compiler.compile(std.testing.allocator, "<chunk>", chunk, &ast, .{}, gc);
    try rt.execModule(&module);

    if (getErrorCount() > 0) {
        for (getErrors()) |msg| {
            std.debug.print("assertion: {s}\n", .{msg});
        }
        reset();
        return error.AssertionFailed;
    }
}

test "simple test chunk: load, def, call, assert.eq" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();
    try register(&rt, gc);

    const chunk: [:0]const u8 =
        \\load("assert.star", "assert")
        \\
        \\def add(a, b):
        \\    return a + b
        \\
        \\assert.eq(add(2, 3), 5)
        \\assert.eq(add(0, 0), 0)
        \\assert.true(True)
        \\
    ;

    var ast = try Ast.parse(std.testing.allocator, chunk);
    defer ast.deinit();
    const module = try Compiler.compile(std.testing.allocator, "<chunk>", chunk, &ast, .{}, gc);
    try rt.execModule(&module);

    if (getErrorCount() > 0) {
        for (getErrors()) |msg| {
            std.debug.print("  assertion: {s}\n", .{msg});
        }
        reset();
        return error.AssertionFailed;
    }
}

test "type() returns spec-conformant type names" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    const stdlib = @import("stdlib.zig");
    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();
    try rt.registerStdlib(stdlib.Stdlib);

    const source: [:0]const u8 =
        \\def f():
        \\    return 0
        \\t_int = type(1)
        \\t_str = type("x")
        \\t_list = type([1, 2])
        \\t_tuple = type((1, 2))
        \\t_dict = type({})
        \\t_none = type(None)
        \\t_bool = type(True)
        \\t_native = type(print)
        \\t_func = type(f)
        \\
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
    const module = try Compiler.compile(std.testing.allocator, "<test>", source, &ast, .{}, gc);
    try rt.execModule(&module);

    const cases = [_]struct { name: []const u8, expected: []const u8 }{
        .{ .name = "t_int", .expected = "int" },
        .{ .name = "t_str", .expected = "string" },
        .{ .name = "t_list", .expected = "list" },
        .{ .name = "t_tuple", .expected = "tuple" },
        .{ .name = "t_dict", .expected = "dict" },
        .{ .name = "t_none", .expected = "NoneType" },
        .{ .name = "t_bool", .expected = "bool" },
        .{ .name = "t_native", .expected = "builtin_function_or_method" },
        .{ .name = "t_func", .expected = "function" },
    };
    for (cases) |c| {
        const obj = rt.globals.get(c.name) orelse return error.MissingGlobal;
        const s = try Runtime.downCast(StarStr, obj);
        try std.testing.expectEqualStrings(c.expected, s.str);
    }
}

test "module(...) builds object whose attributes match its kwargs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();
    try register(&rt, gc);

    const source: [:0]const u8 =
        \\m = module("name", a = 1, b = 10)
        \\
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
    const module = try Compiler.compile(std.testing.allocator, "<test>", source, &ast, .{}, gc);
    try rt.execModule(&module);

    const m_obj = rt.globals.get("m") orelse return error.MissingGlobal;
    const a = m_obj.attributes.get("a") orelse return error.MissingAttr;
    const b = m_obj.attributes.get("b") orelse return error.MissingAttr;
    const a_int = try Runtime.downCast(Runtime.StarInt, a);
    const b_int = try Runtime.downCast(Runtime.StarInt, b);
    try std.testing.expectEqual(@as(i64, 1), a_int.num);
    try std.testing.expectEqual(@as(i64, 10), b_int.num);
}

test "assert.star executes (full integration)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();

    try register(&rt, gc);

    var diag: Runtime.Diagnostic = .{};
    defer diag.deinit();
    rt.diagnostic = &diag;

    const source = try std.testing.allocator.allocSentinel(u8, assert_star_source.len, 0);
    defer std.testing.allocator.free(source);
    @memcpy(source, assert_star_source);

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();

    const module = try Compiler.compile(std.testing.allocator, "assert.star", source, &ast, .{}, gc);
    rt.execModule(&module) catch |err| {
        std.debug.print("\nassert.star exec error: {s}\n", .{@errorName(err)});
        if (diag.trace.len > 0) {
            for (diag.trace) |entry| {
                std.debug.print("  at {s}, offset {d}\n", .{ entry.func_name, entry.source_loc.byte_offset });
                // Find the line
                var line: u32 = 1;
                var i: u32 = 0;
                while (i < entry.source_loc.byte_offset and i < source.len) : (i += 1) {
                    if (source[i] == '\n') line += 1;
                }
                std.debug.print("  line {d}\n", .{line});
            }
        }
        return err;
    };

    // Verify the assert global was created
    const assert_global = rt.globals.get("assert") orelse return error.AssertGlobalMissing;
    _ = assert_global;
}

test "assert.star parses and compiles" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    const source = try std.testing.allocator.allocSentinel(u8, assert_star_source.len, 0);
    defer std.testing.allocator.free(source);
    @memcpy(source, assert_star_source);

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();

    const module = try Compiler.compile(std.testing.allocator, "assert.star", source, &ast, .{}, gc);
    _ = module;
}

test "lambda + new operators end-to-end" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    const stdlib = @import("stdlib.zig");
    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();
    try rt.registerStdlib(stdlib.Stdlib);

    const source: [:0]const u8 =
        \\add = lambda x, y: x + y
        \\half = (lambda n: n // 2)
        \\quot = lambda n: n / 2
        \\merge = lambda a, b: a | b
        \\a = add(3, 4)
        \\b = half(10)
        \\c = quot(7)
        \\d = merge(5, 3)
        \\e = (lambda: 42)()
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
    const module = try Compiler.compile(std.testing.allocator, "<test>", source, &ast, .{}, gc);
    try rt.execModule(&module);

    const a = try Runtime.downCast(Runtime.StarInt, rt.globals.get("a").?);
    try std.testing.expectEqual(@as(i64, 7), a.num);
    const b = try Runtime.downCast(Runtime.StarInt, rt.globals.get("b").?);
    try std.testing.expectEqual(@as(i64, 5), b.num);
    const c = try Runtime.downCast(Runtime.StarFloat, rt.globals.get("c").?);
    try std.testing.expectEqual(@as(f64, 3.5), c.num);
    const d = try Runtime.downCast(Runtime.StarInt, rt.globals.get("d").?);
    try std.testing.expectEqual(@as(i64, 7), d.num);
    const e = try Runtime.downCast(Runtime.StarInt, rt.globals.get("e").?);
    try std.testing.expectEqual(@as(i64, 42), e.num);
}

test "catch(f) on a Star function that errors returns an error string" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();
    try register(&rt, gc);

    const source: [:0]const u8 =
        \\def boom():
        \\    return undef_global
        \\msg = catch(boom)
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
    const module = try Compiler.compile(std.testing.allocator, "<test>", source, &ast, .{}, gc);
    try rt.execModule(&module);

    const msg = rt.globals.get("msg") orelse return error.MissingGlobal;
    const msg_str = try Runtime.downCast(StarStr, msg);
    try std.testing.expect(std.mem.indexOf(u8, msg_str.str, "Undefined") != null);
}

test "catch(f) on a Star function that succeeds returns None" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();
    try register(&rt, gc);

    const source: [:0]const u8 =
        \\def ok():
        \\    return 42
        \\msg = catch(ok)
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
    const module = try Compiler.compile(std.testing.allocator, "<test>", source, &ast, .{}, gc);
    try rt.execModule(&module);

    const msg = rt.globals.get("msg") orelse return error.MissingGlobal;
    try std.testing.expectEqual(StarNone.instance, msg);
}

test "conditional (ternary) expression end-to-end" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    const stdlib = @import("stdlib.zig");
    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();
    try rt.registerStdlib(stdlib.Stdlib);

    // Covers: simple ternary, ternary as argument, ternary in lambda,
    // right-associative chaining, and the disambiguation case where `if`
    // following a complete expression starts a NEW statement (not a ternary).
    const source: [:0]const u8 =
        \\a = 1 if True else 0
        \\b = 1 if False else 99
        \\c = (lambda x: 1 if x else 0)(True)
        \\d = (lambda x: 1 if x else 0)(False)
        \\e = "small" if 3 < 4 else "big"
        \\f = "a" if True else "b" if False else "c"
        \\g = 0
        \\if True:
        \\    g = 7
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
    const module = try Compiler.compile(std.testing.allocator, "<test>", source, &ast, .{}, gc);
    try rt.execModule(&module);

    const a = try Runtime.downCast(Runtime.StarInt, rt.globals.get("a").?);
    try std.testing.expectEqual(@as(i64, 1), a.num);
    const b = try Runtime.downCast(Runtime.StarInt, rt.globals.get("b").?);
    try std.testing.expectEqual(@as(i64, 99), b.num);
    const c = try Runtime.downCast(Runtime.StarInt, rt.globals.get("c").?);
    try std.testing.expectEqual(@as(i64, 1), c.num);
    const d = try Runtime.downCast(Runtime.StarInt, rt.globals.get("d").?);
    try std.testing.expectEqual(@as(i64, 0), d.num);
    const e = try Runtime.downCast(Runtime.StarStr, rt.globals.get("e").?);
    try std.testing.expectEqualStrings("small", e.str);
    const f = try Runtime.downCast(Runtime.StarStr, rt.globals.get("f").?);
    try std.testing.expectEqualStrings("a", f.str);
    // g checks the disambiguation: after `g = 0`, a fresh `if True:` must
    // be parsed as a new if-statement, not a spurious ternary continuation.
    const g = try Runtime.downCast(Runtime.StarInt, rt.globals.get("g").?);
    try std.testing.expectEqual(@as(i64, 7), g.num);
}

test "while + augmented assignment end-to-end" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    const stdlib = @import("stdlib.zig");
    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();
    try rt.registerStdlib(stdlib.Stdlib);

    const source: [:0]const u8 =
        \\def sum(n):
        \\    r = 0
        \\    while n > 0:
        \\        r += n
        \\        n -= 1
        \\    return r
        \\
        \\def while_break(n):
        \\    r = 0
        \\    while n > 0:
        \\        if n == 5:
        \\            break
        \\        r += n
        \\        n -= 1
        \\    return r
        \\
        \\def while_continue(n):
        \\    r = 0
        \\    while n > 0:
        \\        if n % 2 == 0:
        \\            n -= 1
        \\            continue
        \\        r += n
        \\        n -= 1
        \\    return r
        \\
        \\a = sum(5)
        \\b = while_break(10)
        \\c = while_continue(10)
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();
    const module = try Compiler.compile(std.testing.allocator, "<test>", source, &ast, .{}, gc);
    try rt.execModule(&module);

    const a = try Runtime.downCast(Runtime.StarInt, rt.globals.get("a").?);
    try std.testing.expectEqual(@as(i64, 15), a.num); // 5+4+3+2+1
    const b = try Runtime.downCast(Runtime.StarInt, rt.globals.get("b").?);
    try std.testing.expectEqual(@as(i64, 40), b.num); // 10+9+8+7+6
    const c = try Runtime.downCast(Runtime.StarInt, rt.globals.get("c").?);
    try std.testing.expectEqual(@as(i64, 25), c.num); // 9+7+5+3+1
}
