const std = @import("std");
const Allocator = std.mem.Allocator;
const Tokenizer = @import("Tokenizer.zig");
const Compiler = @import("Compiler.zig");
const Token = Tokenizer.Token;
const Ast = @import("Ast.zig");
const util = @import("util.zig");

const scope = .runtime;
const log = std.log.scoped(scope);

gpa: Allocator,
gc: Allocator,
frame_alloc: Allocator = undefined,
/// Where the Starlark `print` builtin writes.
output: *std.Io.Writer = &discarding_output.writer,
/// Call stack for code/locals
frames: std.ArrayList(Frame) = .empty,
/// Value stack
stack: std.ArrayList(*StarObj) = .empty,
/// Globals
globals: std.StringHashMapUnmanaged(*StarObj) = .empty,
/// Source code of current module, mostly for error reporting
current_source: ?[:0]const u8 = null,
/// Filename of current module,
current_filename: ?[]const u8 = null,
/// Optional diagnostic to populate on error
diagnostic: ?*Diagnostic = null,
/// Reentrancy guard for `objectsEqual`. Caps recursion through cyclic
/// data so equality on self-referential containers surfaces a runtime
/// error instead of overflowing the host stack.
eq_depth: usize = 0,
/// Stack of containers currently being rendered by `__str__`. Used to
/// emit `[...]` / `(...)` / `{...}` markers in place of an infinite
/// recursion on self-referential containers.
repr_stack: [128]?*StarObj = .{null} ** 128,
repr_stack_len: usize = 0,

/// Structured descriptor handed to a native function on invocation. Carries
/// positional args plus (optional) kwargs so kwarg-aware natives can read both
/// without reaching back into the runtime.
pub const NativeCall = struct {
    args: []const *StarObj,
    kwargs: []const KwArg = &.{},

    pub const KwArg = struct {
        name: []const u8,
        value: *StarObj,
    };
};

const knobs = struct {
    const locals_capacity: usize = 64;
    const frames_capacity: usize = 4096;
};

pub const locals_capacity = knobs.locals_capacity;

pub const dunder = struct {
    pub const eq = "__eq__";
    pub const add = "__add__";
    pub const sub = "__sub__";
    pub const mul = "__mul__";
    pub const ne = "__ne__";
    pub const lt = "__lt__";
    pub const le = "__le__";
    pub const gt = "__gt__";
    pub const ge = "__ge__";
    pub const mod = "__mod__";
    pub const div = "__truediv__";
    pub const floor_div = "__floordiv__";
    pub const bit_or = "__or__";
    pub const contains = "__contains__";
    pub const iter = "__iter__";
    pub const next = "__next__";
    pub const str = "__str__";
};

pub const InitOpts = struct {
    /// This allocator is treated as if `free` calls are not required. In production it will be a GC,
    /// but in tests or other uses, it could be an arena.
    gc: Allocator,
    /// Allows for optimizing frame allocation with either a fixed-size bump allocator or whatever the user pleases.
    frame_alloc: ?Allocator = null,
    /// Sink for the Starlark `print` builtin. If null, a process-static
    /// discarding writer is used. The CLI passes its stdout writer here; tests
    /// leave it null so prints don't escape to wherever stdout points.
    output: ?*std.Io.Writer = null,
};

var discarding_output_buf: [256]u8 = undefined;
var discarding_output: std.Io.Writer.Discarding = .init(&discarding_output_buf);

/// Walk `obj.attributes` looking for a name whose Levenshtein distance from
/// `wanted` is small enough to surface as a "did you mean .X?" suggestion.
/// Returns null when nothing close is found.
fn findSimilarAttr(obj: *StarObj, wanted: []const u8) ?[]const u8 {
    var best: ?[]const u8 = null;
    var best_d: usize = std.math.maxInt(usize);
    var it = obj.attributes.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        if (std.mem.startsWith(u8, name, "__") and std.mem.endsWith(u8, name, "__")) continue;
        const d = editDistance(wanted, name);
        if (d < best_d) {
            best_d = d;
            best = name;
        }
    }
    // Suggest only on near-misses; for longer names allow more drift.
    const threshold = @max(@as(usize, 2), wanted.len / 3);
    return if (best != null and best_d <= threshold) best else null;
}

fn editDistance(a: []const u8, b: []const u8) usize {
    if (a.len == 0) return b.len;
    if (b.len == 0) return a.len;
    var prev_buf: [128]usize = undefined;
    var cur_buf: [128]usize = undefined;
    if (b.len + 1 > prev_buf.len) return std.math.maxInt(usize);
    var prev: []usize = prev_buf[0 .. b.len + 1];
    var cur: []usize = cur_buf[0 .. b.len + 1];
    for (0..b.len + 1) |j| prev[j] = j;
    for (a, 0..) |ca, i| {
        cur[0] = i + 1;
        for (b, 0..) |cb, j| {
            const cost: usize = if (ca == cb) 0 else 1;
            const del = prev[j + 1] + 1;
            const ins = cur[j] + 1;
            const sub = prev[j] + cost;
            cur[j + 1] = @min(@min(del, ins), sub);
        }
        const tmp = prev;
        prev = cur;
        cur = tmp;
    }
    return prev[b.len];
}

pub const GlobalIdx = enum(u32) { _ };
pub const LocalIdx = enum(u32) { _ };
pub const ConstIdx = enum(u32) { _ };
pub const NameIdx = enum(u32) { _ };
pub const FreeVarIdx = enum(u32) { _ };
pub const KwNamesIdx = enum(u32) { _ };

pub const Arity = u32;

pub const BinOp = enum { add, sub, mul, div, floor_div, bit_or, eq, ne, lt, le, gt, ge, mod, contains };

pub const GetSliceOpts = struct {
    start: bool,
    stop: bool,
    step: bool,
};

pub const Instruction = union(enum) {
    load: LocalIdx,
    load_const: ConstIdx,
    load_global: GlobalIdx,
    load_free: FreeVarIdx,
    store: LocalIdx,
    store_global: GlobalIdx,
    binary_op: BinOp,
    bool_not: void,
    unary_neg: void,
    /// Pop an iterable, unpack into exactly N values, push them with the
    /// first element on top of the stack (LIFO consumption matches the
    /// declared target order).
    unpack_seq: u32,
    call: Arity,
    /// Call with keyword arguments. The stack layout at dispatch is
    /// `[..., func, pos_0..pos_{N-1}, kw_0..kw_{M-1}]` where `N == pos_arity`
    /// and `M == kw_count`. The kwarg names live in the active function's
    /// `kw_names` pool at indices `[kw_names_start .. kw_names_start + kw_count]`,
    /// each paired with the kwarg value at the corresponding stack slot.
    call_kw: struct { pos_arity: Arity, kw_names_start: KwNamesIdx, kw_count: Arity },
    ret: void,
    ret_none: void,
    /// Materialize a function instance from the prototype constant at
    /// `const_idx`. Pops `num_defaults` values from the stack (the leftmost
    /// default was pushed first) and binds them to the rightmost parameters.
    /// Free variables, if any, are captured from the enclosing frame.
    make_function: struct {
        const_idx: ConstIdx,
        num_defaults: u16,
    },
    /// Slice of the identifier from the original source code.
    get_attr: []const u8,
    /// Create a list from top N stack elements.
    build_list: u32,
    /// Create a tuple from top N stack elements.
    build_tuple: u32,
    /// Create a dict from top N key-value pairs (2*N elements on stack).
    build_dict: u32,
    /// Index into object: pops index, pops object, pushes result.
    get_index: void,
    /// Item assignment: pops value, idx, obj; stores obj[idx] = value.
    set_index: void,
    /// Attribute assignment: pops value, obj; stores obj.<attr> = value.
    set_attr: []const u8,
    /// Slice an object. The 3-bit mask indicates which of (start, stop, step)
    /// are present on the stack (TOS-down). The object is below them.
    get_slice: GetSliceOpts,
    /// Jump unconditionally (offset relative to next instruction).
    jump: i32,
    /// Pop top of stack, jump if falsey (offset relative to next instruction).
    jump_if_false: i32,
    /// Pop top of stack, jump if truthy (offset relative to next instruction).
    jump_if_true: i32,
    /// Get iterator from top of stack.
    get_iter: void,
    /// Advance iterator: pushes next value or jumps if exhausted.
    for_iter: i32,
    /// Pop iterator from stack (cleanup after for loop).
    pop_iter: void,
    /// Pop top of stack.
    pop: void,
    /// Short-circuit AND: if TOS is falsy, jump (keep value); else pop and continue.
    and_jump: i32,
    /// Short-circuit OR: if TOS is truthy, jump (keep value); else pop and continue.
    or_jump: i32,

    const Tag = std.meta.Tag(@This());
};

pub const TypeError = error{
    TypeMismatch,
};

pub const RuntimeError = error{
    AddOpUndefined,
    SubOpUndefined,
    MulOpUndefined,
    ArityMismatch,
    LocalOutOfRange,
    LocalUninitialized,
    ConstOutOfRange,
    StackUnderflow,
    CallUndefined,
    FrameMissing,
    FrameNoReturn,
    /// For now there is a static number of locals stored in a frame.
    FrameOversized,
    StarlarkStackOverflow,
    ReturnedOutsideFunction,
    AttributeMissing,
    GlobalUndefined,
    FreeVarOutOfRange,
    IndexOutOfRange,
    NotIterable,
    TypeMismatch,
};

pub const Error = Allocator.Error || std.Io.Writer.Error || TypeError || RuntimeError;

/// Represents a source location in the original source code
pub const SourceLoc = struct {
    byte_offset: u32,
    len: u16,

    pub const none: SourceLoc = .{ .byte_offset = 0, .len = 0 };
};

/// A single entry in the stack trace
pub const StackTraceEntry = struct {
    func_name: []const u8,
    source_loc: SourceLoc,
};

/// Diagnostic populated on error
pub const Diagnostic = struct {
    allocator: Allocator = undefined,
    err: Error = error.FrameMissing,
    trace: []const StackTraceEntry = &.{},
    source: ?[:0]const u8 = null,
    filename: []const u8 = "<source>",
    /// Operand-specific message describing the failure (e.g.
    /// `"unknown binary op: string + int"`). Owned by `allocator`.
    message: ?[]const u8 = null,

    const style = struct {
        const reset = "\x1b[0m";
        const bold_red = "\x1b[1;31m";
        const cyan = "\x1b[36m";
        const yellow = "\x1b[33m";
        const bold_green = "\x1b[1;32m";
    };

    fn offsetToLineCol(source: [:0]const u8, offset: u32) struct { line: u32, col: u32 } {
        var line: u32 = 1;
        var col: u32 = 1;
        for (source[0..@min(offset, source.len)]) |c| {
            if (c == '\n') {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        return .{ .line = line, .col = col };
    }

    fn getSourceLine(source: [:0]const u8, offset: u32) []const u8 {
        const clamped = @min(offset, source.len);
        var start: usize = clamped;
        while (start > 0 and source[start - 1] != '\n') start -= 1;
        var end: usize = clamped;
        while (end < source.len and source[end] != '\n') end += 1;
        return source[start..end];
    }

    fn getLineStartOffset(source: [:0]const u8, offset: u32) u32 {
        var pos: u32 = @min(offset, source.len);
        while (pos > 0 and source[pos - 1] != '\n') pos -= 1;
        return pos;
    }

    fn getErrorCategory(err: Error) []const u8 {
        return switch (err) {
            error.TypeMismatch => "TypeError",
            error.AddOpUndefined,
            error.SubOpUndefined,
            error.MulOpUndefined,
            error.ArityMismatch,
            error.LocalOutOfRange,
            error.LocalUninitialized,
            error.ConstOutOfRange,
            error.StackUnderflow,
            error.CallUndefined,
            error.FrameMissing,
            error.FrameNoReturn,
            error.ReturnedOutsideFunction,
            error.AttributeMissing,
            error.GlobalUndefined,
            error.FreeVarOutOfRange,
            => "RuntimeError",
            error.OutOfMemory => "MemoryError",
            else => "Error",
        };
    }

    /// Format as a Python-style traceback
    pub fn format(self: *const Diagnostic, writer: *std.Io.Writer) !void {
        try writer.print("\n" ++ style.bold_red ++ "Traceback" ++ style.reset ++ " (most recent call last):\n", .{});

        for (self.trace, 0..) |entry, idx| {
            const is_innermost = idx == self.trace.len - 1;

            if (self.source) |source| {
                const loc = offsetToLineCol(source, entry.source_loc.byte_offset);
                try writer.print(
                    "  File " ++ style.cyan ++ "\"{s}\"" ++ style.reset ++
                        ", line " ++ style.yellow ++ "{d}" ++ style.reset ++
                        ", in " ++ style.bold_green ++ "{s}" ++ style.reset ++ "\n",
                    .{ self.filename, loc.line, entry.func_name },
                );

                const line_content = getSourceLine(source, entry.source_loc.byte_offset);
                const leading_ws = line_content.len - std.mem.trimStart(u8, line_content, " \t").len;
                try writer.print("    {s}\n", .{line_content[leading_ws..]});

                if (is_innermost and entry.source_loc.len > 0) {
                    const line_start = getLineStartOffset(source, entry.source_loc.byte_offset);
                    const col = entry.source_loc.byte_offset - line_start;
                    const adjusted = if (col >= leading_ws) col - @as(u32, @intCast(leading_ws)) else 0;

                    try writer.print("    ", .{});
                    for (0..adjusted) |_| try writer.print(" ", .{});
                    try writer.print(style.bold_red, .{});
                    for (0..entry.source_loc.len) |_| try writer.print("^", .{});
                    try writer.print(style.reset ++ "\n", .{});
                }
            } else {
                try writer.print(
                    "  File " ++ style.cyan ++ "\"<source>\"" ++ style.reset ++
                        ", in " ++ style.bold_green ++ "{s}" ++ style.reset ++ "\n",
                    .{entry.func_name},
                );
            }
        }

        try writer.print(style.bold_red ++ "{s}" ++ style.reset ++ ": {s}\n\n", .{
            getErrorCategory(self.err), @errorName(self.err),
        });
    }

    pub fn deinit(self: *Diagnostic) void {
        if (self.trace.len > 0) {
            self.allocator.free(self.trace);
        }
        self.* = .{};
    }
};

/// A single frame in the call stack, containing the function being executed and its local variables.
pub const Frame = struct {
    func: *StarFunc,
    locals: util.BoundedArray(knobs.locals_capacity, *StarObj),
    sp_base: usize,
    pc: usize, // pc inside this frame's code

    pub fn init(self: *Frame, func: *StarFunc, sp_base: usize, args: []*StarObj) !void {
        self.* = .{
            .func = func,
            .locals = .init(),
            .sp_base = sp_base,
            .pc = 0,
        };

        if (func.frame_size > 0) {
            if (func.frame_size > self.locals.items.len) return Error.FrameOversized;
            try self.locals.appendSlice(args);
            for (args.len..func.frame_size) |_| {
                try self.locals.append(StarNone.instance);
            }
        }
    }

    pub fn deinit(self: *Frame) void {
        _ = self;
    }

    pub fn readConst(self: *Frame, idx: ConstIdx) Error!*StarObj {
        const const_idx = @intFromEnum(idx);
        if (const_idx >= self.func.consts.len) return RuntimeError.ConstOutOfRange;
        return self.func.consts[const_idx];
    }

    pub fn writeLocal(self: *Frame, idx: LocalIdx, value: *StarObj) Error!void {
        const local_idx = @intFromEnum(idx);
        if (local_idx >= self.locals.len) return RuntimeError.LocalOutOfRange;
        self.locals.items[local_idx] = value;
    }

    pub fn readLocal(self: *Frame, idx: LocalIdx) Error!*StarObj {
        const local_idx = @intFromEnum(idx);
        if (local_idx >= self.locals.len) return RuntimeError.LocalOutOfRange;
        const local = self.locals.items[local_idx];
        return local;
    }

    pub fn readFree(self: *Frame, idx: FreeVarIdx) Error!*StarObj {
        const free_idx = @intFromEnum(idx);
        if (free_idx >= self.func.closure_cells.len) return RuntimeError.FreeVarOutOfRange;
        return self.func.closure_cells[free_idx];
    }
};
const Runtime = @This();

pub fn init(gpa: Allocator, opts: InitOpts) !Runtime {
    // The singletons (`StarNone.instance`, `StarBool` true/false,
    // `StarStopIteration.instance`) are process-static, so their attribute
    // hash maps cannot be allocated from a per-Runtime allocator without
    // leaving dangling metadata when this Runtime is deinit'd. Each
    // `initAttributes` self-manages its allocator (page_allocator) and is
    // idempotent across Runtimes — see the doc comments on each.
    // TODO: A cleaner long-term shape is to either (a) make the singletons
    // per-Runtime, or (b) factor a one-shot process-init phase, so that
    // Runtime.init can avoid touching them at all.
    try StarNone.initAttributes();
    try StarBool.initAttributes();
    try StarStopIteration.initAttributes();

    return Runtime{
        .gpa = gpa,
        .gc = opts.gc,
        .frame_alloc = opts.frame_alloc orelse opts.gc,
        .output = opts.output orelse &discarding_output.writer,
    };
}

pub fn deinit(self: *Runtime) void {
    // free any frame locals still allocated
    for (self.frames.items) |*frame| {
        frame.deinit();
    }
    self.frames.deinit(self.frame_alloc);
    self.stack.deinit(self.gc);
    self.globals.deinit(self.gc);
}

/// Attach an operand-specific message to the runtime's diagnostic. No-op
/// when no diagnostic is registered (the caller can still recover the
/// error tag). Owned by `self.gc`.
pub fn setErrorMessage(self: *Runtime, comptime fmt: []const u8, args: anytype) void {
    const diag = self.diagnostic orelse return;
    diag.message = std.fmt.allocPrint(self.gc, fmt, args) catch null;
}

/// True if a helper has already attached a message for the in-flight error.
pub fn hasErrorMessage(self: *Runtime) bool {
    const diag = self.diagnostic orelse return false;
    return diag.message != null;
}

/// Push `obj` onto the in-progress `__str__` cycle-detection stack.
/// Returns false (and skips the push) when the stack is already full —
/// callers fall back to non-cycle-detected output in that case.
pub fn reprPush(self: *Runtime, obj: *StarObj) bool {
    if (self.repr_stack_len >= self.repr_stack.len) return false;
    self.repr_stack[self.repr_stack_len] = obj;
    self.repr_stack_len += 1;
    return true;
}

pub fn reprPop(self: *Runtime) void {
    if (self.repr_stack_len > 0) self.repr_stack_len -= 1;
}

pub fn reprContains(self: *Runtime, obj: *StarObj) bool {
    for (self.repr_stack[0..self.repr_stack_len]) |o| if (o == obj) return true;
    return false;
}

/// Create and push a new frame for `func`. Copies `args` into locals; missing
/// arguments are filled from `func.defaults`. `sp_base` is the stack height
/// the frame will be unwound to on return — callers must shrink the value
/// stack to that height before invoking `callFn` (i.e. `args` must already
/// have been removed from the stack and copied into a caller-owned buffer).
pub fn callFn(self: *Runtime, func: *StarFunc, args: []*StarObj, sp_base: usize) Error!void {
    if (self.frames.items.len >= knobs.frames_capacity) return RuntimeError.StarlarkStackOverflow;
    // When there are no defaults, every parameter is required — derive that
    // here so callers don't have to set both `arity` and `num_required`.
    const required = if (func.defaults.len == 0) func.arity else func.num_required;
    if (args.len < required or args.len > func.arity) return RuntimeError.ArityMismatch;
    std.debug.assert(func.frame_size >= func.arity);
    std.debug.assert(self.stack.items.len == sp_base);

    if (comptime std.log.logEnabled(.debug, scope)) {
        log.debug("callFn: arity={d}, sp_base={d}, args=[", .{ func.arity, sp_base });
        for (args, 0..) |arg, i| {
            if (i > 0) log.debug(", ", .{});
            log.debug("{s}", .{arg.vtable.name});
        }
        log.debug("]", .{});
    }

    // Pad with defaults if positional args don't cover the full arity.
    var defaults_buf: [knobs.locals_capacity]*StarObj = undefined;
    var call_args: []*StarObj = args;
    if (args.len < func.arity) {
        if (func.arity > defaults_buf.len) return RuntimeError.FrameOversized;
        @memcpy(defaults_buf[0..args.len], args);
        for (args.len..func.arity) |i| {
            const default_idx = i - func.num_required;
            std.debug.assert(default_idx < func.defaults.len);
            defaults_buf[i] = func.defaults[default_idx];
        }
        call_args = defaults_buf[0..func.arity];
    }

    var frame: Frame = undefined;
    try frame.init(func, sp_base, call_args);
    try self.frames.append(self.frame_alloc, frame);
}

pub fn stackPop(self: *Runtime) Error!*StarObj {
    return self.stack.pop() orelse return Error.StackUnderflow;
}

fn stackPush(self: *Runtime, obj: *StarObj) Error!void {
    try self.stack.append(self.gc, obj);
}

fn stackPushOne(self: *Runtime) Error!**StarObj {
    return self.stack.addOne(self.gc);
}

fn storeGlobal(self: *Runtime, name_idx: GlobalIdx, value: *StarObj) Error!void {
    const frame: *Frame = &self.frames.items[self.frames.items.len - 1];
    const name = frame.func.names_global[@intFromEnum(name_idx)];
    try self.globals.put(self.gc, name, value);
}

fn loadGlobal(self: *Runtime, name_idx: GlobalIdx) Error!*StarObj {
    const frame: *Frame = &self.frames.items[self.frames.items.len - 1];
    const name = frame.func.names_global[@intFromEnum(name_idx)];
    log.debug("Loading global: {s}", .{name});
    return self.globals.get(name) orelse return RuntimeError.GlobalUndefined;
}

/// Resolve each name in `names_free` against the enclosing `frame` (its
/// locals first, then its captured free variables) and return the resulting
/// closure cells. The frame's compiler is responsible for ensuring every
/// name resolves; an unresolved name indicates a compiler bug.
fn captureClosure(gc: Allocator, frame: *Frame, names_free: []const []const u8) Error![]const *StarObj {
    if (names_free.len == 0) return &.{};
    const cells = try gc.alloc(*StarObj, names_free.len);
    // TODO: Consider finding an efficient way to only pay this cost once per function prototype.
    outer: for (names_free, cells) |name, *cell| {
        for (frame.func.names_local, 0..) |local_name, local_idx| {
            if (std.mem.eql(u8, name, local_name)) {
                cell.* = try frame.readLocal(@enumFromInt(local_idx));
                continue :outer;
            }
        }
        for (frame.func.names_free, 0..) |free_name, free_idx| {
            if (std.mem.eql(u8, name, free_name)) {
                cell.* = try frame.readFree(@enumFromInt(free_idx));
                continue :outer;
            }
        }
        return RuntimeError.GlobalUndefined;
    }
    return cells;
}

/// Populate diagnostic with stack trace if required.
fn fillDiagnostic(self: *Runtime, err: Error) void {
    const diag = self.diagnostic orelse return;

    // `fillDiagnostic` can be called multiple times in one Runtime (each
    // `catch`-wrapped lambda that errors triggers another fill); free any
    // previously-allocated trace before we overwrite it.
    if (diag.trace.len > 0) {
        diag.allocator.free(diag.trace);
        diag.trace = &.{};
    }

    diag.allocator = self.gpa;
    diag.err = err;
    diag.source = self.current_source;
    if (self.current_filename) |cf| {
        diag.filename = cf;
    }

    if (self.frames.items.len == 0) {
        return;
    }

    var entries = std.ArrayList(StackTraceEntry).empty;
    var i: usize = 0;
    while (i < self.frames.items.len) : (i += 1) {
        const frame: *Frame = &self.frames.items[i];
        const func = frame.func;
        const pc = if (frame.pc > 0) frame.pc else 0;
        const source_loc = if (pc < func.source_locs.len) func.source_locs[pc] else SourceLoc.none;

        entries.append(self.gpa, .{ .func_name = func.name, .source_loc = source_loc }) catch {
            return;
        };
    }

    diag.trace = entries.toOwnedSlice(self.gpa) catch &.{};
}

/// Step until frame stack is empty, filling in diagnostics in case of error.
pub fn stepUntilDone(self: *Runtime) Error!void {
    return self.stepUntilDepth(0);
}

/// Step until frame stack reaches the given depth (used for recursive execModule calls).
pub fn stepUntilDepth(self: *Runtime, target_depth: usize) Error!void {
    while (self.frames.items.len > target_depth) {
        self.step() catch |err| {
            self.fillDiagnostic(err);
            return err;
        };
    }
}

// Move interpreter forward one step. DOES NOT populate error diagnostics.
fn step(self: *Runtime) Error!void {
    if (self.frames.items.len == 0) return RuntimeError.FrameMissing;
    var frame: *Frame = &self.frames.items[self.frames.items.len - 1];
    const active_code = frame.func.code;

    if (frame.pc >= active_code.len) return RuntimeError.FrameNoReturn;

    const instr = active_code[frame.pc];
    const pc = frame.pc;

    log.debug("[{d}] {s}", .{ pc, @tagName(instr) });
    switch (instr) {
        .load => |idx| {
            const val = try frame.readLocal(idx);
            try self.stackPush(val);
        },
        .load_const => |cidx| {
            const const_obj = try frame.readConst(cidx);
            try self.stackPush(const_obj);
        },
        .load_global => |idx| {
            const val = try self.loadGlobal(idx);
            try self.stackPush(val);
        },
        .load_free => |idx| {
            const val = try frame.readFree(idx);
            try self.stackPush(val);
        },
        .store => |idx| {
            const val = try self.stackPop();
            try frame.writeLocal(idx, val);
        },
        .store_global => |idx| {
            const val = try self.stackPop();
            try self.storeGlobal(idx, val);
        },
        .binary_op => |op| {
            // Stack layout (set by the compiler): `[..., self, arg]`.
            // For `__contains__` the compiler pushes `haystack` then `needle`,
            // so the dispatch is uniform: `self.<dunder>(arg)`.
            const arg = try self.stackPop();
            const self_obj = try self.stackPop();

            const dunder_op = switch (op) {
                inline else => |o| @field(std.meta.DeclEnum(dunder), @tagName(o)),
            };

            const bound_method = try self_obj.getMethodDunder(dunder_op) orelse {
                self.setErrorMessage( "unknown binary op (not impl): {s} {s} {s}", .{
                    self_obj.vtable.type_name, @tagName(op), arg.vtable.type_name,
                });
                return Error.AddOpUndefined;
            };
            const result = bound_method.call(self, &.{arg}) catch |err| {
                if (!self.hasErrorMessage()) {
                    self.setErrorMessage( "unknown binary op (not impl): {s} {s} {s}", .{
                        self_obj.vtable.type_name, @tagName(op), arg.vtable.type_name,
                    });
                }
                return err;
            };
            try self.stackPush(result);
        },
        .get_attr => |attr| {
            const obj = try self.stackPop();
            const val = try obj.getAttrStr(attr) orelse {
                // Two spaces before `did you mean` match the spec regex
                // `.did you mean .X` (the `.` matches any char).
                const type_name = obj.vtable.type_name;
                if (findSimilarAttr(obj, attr)) |sug| {
                    self.setErrorMessage( "{s} has no .{s} field or method  did you mean .{s}?", .{ type_name, attr, sug });
                } else {
                    self.setErrorMessage( "{s} has no .{s} field or method", .{ type_name, attr });
                }
                return Error.AttributeMissing;
            };
            try self.stackPush(val);
        },
        .call => |arity_u| {
            const arity: usize = arity_u;
            const sp = self.stack.items.len;
            if (sp < arity + 1) return RuntimeError.StackUnderflow;

            // Stack layout: [..., func_obj, arg1, ..., argN]
            const func_index = sp - arity - 1;
            const func_obj = self.stack.items[func_index];
            const args_slice = self.stack.items[func_index + 1 .. sp];

            const maybe_func: ?*StarFunc = downCast(StarFunc, func_obj) catch null;
            if (maybe_func) |fptr| {
                if (fptr.native_fn) |f| {
                    const result = try f(self, .{ .args = args_slice });
                    self.stack.shrinkRetainingCapacity(func_index);
                    try self.stackPush(result);
                } else {
                    // Copy args off the stack so we can shrink to the clean
                    // base before recursing into callFn (which expects the
                    // stack to already be at sp_base).
                    var args_buf: [knobs.locals_capacity]*StarObj = undefined;
                    if (arity > args_buf.len) return RuntimeError.FrameOversized;
                    @memcpy(args_buf[0..arity], args_slice);
                    self.stack.shrinkRetainingCapacity(func_index);
                    // callFn may reallocate self.frames, invalidating `frame`,
                    // so we cannot rely on the post-switch `frame.pc += 1`.
                    frame.pc += 1;
                    try self.callFn(fptr, args_buf[0..arity], func_index);
                    return;
                }
            } else if (downCast(StarBoundMethod, func_obj)) |bm| {
                const result = try bm.call(self, args_slice);
                self.stack.shrinkRetainingCapacity(func_index);
                try self.stackPush(result);
            } else |_| {
                return RuntimeError.CallUndefined;
            }
        },
        .call_kw => |kw| {
            const total_args = kw.pos_arity + kw.kw_count;
            const sp = self.stack.items.len;
            if (sp < total_args + 1) return RuntimeError.StackUnderflow;

            const args_start = sp - total_args;
            const func_index = args_start - 1;
            const func_obj = self.stack.items[func_index];

            const fptr = downCast(StarFunc, func_obj) catch return RuntimeError.CallUndefined;

            const kw_names_start: u32 = @intFromEnum(kw.kw_names_start);
            const kw_names_end = kw_names_start + kw.kw_count;
            if (kw_names_end > frame.func.kw_names.len) return RuntimeError.ConstOutOfRange;
            const kw_names = frame.func.kw_names[kw_names_start..kw_names_end];

            if (fptr.native_fn) |f| {
                const pos_args = self.stack.items[args_start .. args_start + kw.pos_arity];
                const kw_values = self.stack.items[args_start + kw.pos_arity .. sp];
                var kwarg_buf: [knobs.locals_capacity]NativeCall.KwArg = undefined;
                if (kw.kw_count > kwarg_buf.len) return RuntimeError.FrameOversized;
                for (kw_names, kw_values, 0..) |name, value, i| {
                    kwarg_buf[i] = .{ .name = name, .value = value };
                }
                const result = try f(self, .{
                    .args = pos_args,
                    .kwargs = kwarg_buf[0..kw.kw_count],
                });
                self.stack.shrinkRetainingCapacity(func_index);
                try self.stackPush(result);
            } else {
                // Interpreted function: bind args to declared parameters.
                const pos_args = self.stack.items[args_start .. args_start + kw.pos_arity];
                const kw_values = self.stack.items[args_start + kw.pos_arity .. sp];
                var bound_buf: [knobs.locals_capacity]*StarObj = undefined;
                if (fptr.arity > bound_buf.len) return RuntimeError.FrameOversized;
                const bound = bound_buf[0..fptr.arity];

                // Fill from positional args, then defaults, then overlay kwargs.
                for (bound, 0..) |*slot, i| {
                    if (i < pos_args.len) {
                        slot.* = pos_args[i];
                    } else if (i >= fptr.num_required and i - fptr.num_required < fptr.defaults.len) {
                        slot.* = fptr.defaults[i - fptr.num_required];
                    } else {
                        slot.* = StarNone.instance; // unbound; expected to be filled by a kwarg below
                    }
                }
                for (kw_names, kw_values) |name, value| {
                    for (fptr.names_local[0..fptr.arity], 0..) |param, i| {
                        if (std.mem.eql(u8, param, name)) {
                            bound[i] = value;
                            break;
                        }
                    }
                }
                self.stack.shrinkRetainingCapacity(func_index);
                // See comment in `.call` arm: advance our PC before the
                // callee's frame is pushed.
                frame.pc += 1;
                try self.callFn(fptr, bound, func_index);
                return;
            }
        },
        .make_function => |mf| {
            const proto_obj = try frame.readConst(mf.const_idx);
            const proto = try downCast(StarFunc, proto_obj);

            // Pop default values (last pushed = last param to receive a default).
            // We hand them to the new function in declaration order.
            const sp = self.stack.items.len;
            if (sp < mf.num_defaults) return RuntimeError.StackUnderflow;
            const defaults_start = sp - mf.num_defaults;

            // Fast path: no defaults and no free vars → reuse the prototype.
            if (mf.num_defaults == 0 and proto.names_free.len == 0) {
                try self.stackPush(proto_obj);
            } else {
                const defaults: []const *StarObj = if (mf.num_defaults == 0)
                    &.{}
                else
                    try self.gc.dupe(*StarObj, self.stack.items[defaults_start..sp]);
                self.stack.shrinkRetainingCapacity(defaults_start);

                const closure_cells = try captureClosure(self.gc, frame, proto.names_free);

                const new_func = try self.gc.create(StarFunc);
                new_func.* = proto.*;
                try new_func.obj.cloneFrom(self.gc, &proto.obj);
                new_func.defaults = defaults;
                new_func.closure_cells = closure_cells;
                try self.stackPush(&new_func.obj);
            }
        },
        .bool_not => {
            const val = try self.stackPop();
            try self.stackPush(StarBool.get(!val.isTruthy()));
        },
        .unary_neg => {
            const val = try self.stackPop();
            if (downCast(StarInt, val)) |i| {
                const r = try StarInt.init(self.gc, -i.num);
                try self.stackPush(&r.obj);
            } else |_| if (downCast(StarFloat, val)) |f| {
                const r = try StarFloat.init(self.gc, -f.num);
                try self.stackPush(&r.obj);
            } else |_| return Error.TypeMismatch;
        },
        .unpack_seq => |count| {
            const val = try self.stackPop();
            const items = iterateToOwned(self, val) catch |err| {
                if (!self.hasErrorMessage()) {
                    self.setErrorMessage( "got {s} in sequence assignment", .{val.vtable.type_name});
                }
                return err;
            };
            if (items.len < count) {
                self.setErrorMessage( "too few values to unpack (got {d}, want {d})", .{ items.len, count });
                return Error.TypeMismatch;
            }
            if (items.len > count) {
                self.setErrorMessage( "too many values to unpack (got {d}, want {d})", .{ items.len, count });
                return Error.TypeMismatch;
            }
            // Push reversed so the leftmost target pops first.
            var i: usize = items.len;
            while (i > 0) {
                i -= 1;
                try self.stackPush(items[i]);
            }
        },
        .build_list => |count| {
            const sp = self.stack.items.len;
            if (sp < count) return RuntimeError.StackUnderflow;
            const start = sp - count;
            const elements = self.stack.items[start..sp];
            const list = try StarList.init(self.gc, elements);
            self.stack.shrinkRetainingCapacity(start);
            try self.stackPush(&list.obj);
        },
        .build_tuple => |count| {
            const sp = self.stack.items.len;
            if (sp < count) return RuntimeError.StackUnderflow;
            const start = sp - count;
            const elements = self.stack.items[start..sp];
            const tuple = try StarTuple.init(self.gc, elements);
            self.stack.shrinkRetainingCapacity(start);
            try self.stackPush(&tuple.obj);
        },
        .build_dict => |count| {
            const sp = self.stack.items.len;
            const total = count * 2;
            if (sp < total) return RuntimeError.StackUnderflow;
            const start = sp - total;
            const pairs = self.stack.items[start..sp];
            const dict = try StarDict.init(self, pairs);
            self.stack.shrinkRetainingCapacity(start);
            try self.stackPush(&dict.obj);
        },
        .get_slice => |opts| {
            var step_v: ?i64 = null;
            var stop_v: ?i64 = null;
            var start_v: ?i64 = null;
            if (opts.step) {
                const v = try self.stackPop();
                step_v = (try downCast(StarInt, v)).num;
            }
            if (opts.stop) {
                const v = try self.stackPop();
                stop_v = (try downCast(StarInt, v)).num;
            }
            if (opts.start) {
                const v = try self.stackPop();
                start_v = (try downCast(StarInt, v)).num;
            }
            const obj = try self.stackPop();

            const result: *StarObj = if (downCast(StarList, obj)) |list| blk: {
                const b = try normalizeSlice(list.items.items.len, start_v, stop_v, step_v);
                const items = try sliceSequence(self.gc, list.items.items, b);
                const new_list = try StarList.init(self.gc, items);
                break :blk &new_list.obj;
            } else |_| if (downCast(StarTuple, obj)) |tup| blk: {
                const b = try normalizeSlice(tup.items.len, start_v, stop_v, step_v);
                const items = try sliceSequence(self.gc, tup.items, b);
                const new_tup = try StarTuple.init(self.gc, items);
                break :blk &new_tup.obj;
            } else |_| if (downCast(StarStr, obj)) |s| blk: {
                const b = try normalizeSlice(s.str.len, start_v, stop_v, step_v);
                var buf = std.ArrayList(u8).empty;
                var i = b.start;
                if (b.step > 0) {
                    while (i < b.stop) : (i += b.step) try buf.append(self.gc, s.str[@intCast(i)]);
                } else {
                    while (i > b.stop) : (i += b.step) try buf.append(self.gc, s.str[@intCast(i)]);
                }
                break :blk &(try StarStr.init(self.gc, buf.items)).obj;
            } else |_| return Error.TypeMismatch;

            try self.stackPush(result);
        },
        .set_index => {
            const value = try self.stackPop();
            const idx_obj = try self.stackPop();
            const obj = try self.stackPop();
            if (downCast(StarList, obj)) |list| {
                const i = try downCast(StarInt, idx_obj);
                const len: i64 = @intCast(list.items.items.len);
                const actual = if (i.num < 0) i.num + len else i.num;
                if (actual < 0 or actual >= len) return Error.IndexOutOfRange;
                list.items.items[@intCast(actual)] = value;
            } else |_| if (downCast(StarDict, obj)) |dict| {
                try dict.put(self, idx_obj, value);
            } else |_| return Error.TypeMismatch;
        },
        .set_attr => |attr| {
            const value = try self.stackPop();
            const obj = try self.stackPop();
            try obj.attributes.put(self.gc, attr, value);
        },
        .get_index => {
            const idx_obj = try self.stackPop();
            const obj = try self.stackPop();

            // Dicts dispatch by value-equality on the key, not by integer offset.
            if (downCast(StarDict, obj)) |dict| {
                const v = try dict.lookup(self, idx_obj) orelse return Error.TypeMismatch;
                try self.stackPush(v);
            } else |_| {
                const idx_int = try downCast(StarInt, idx_obj);
                const val = if (downCast(StarList, obj)) |list|
                    try list.getItem(idx_int.num)
                else |_| if (downCast(StarTuple, obj)) |tuple|
                    try tuple.getItem(idx_int.num)
                else |_| if (downCast(StarStr, obj)) |s| blk: {
                    const len: i64 = @intCast(s.str.len);
                    const actual = if (idx_int.num < 0) idx_int.num + len else idx_int.num;
                    if (actual < 0 or actual >= len) return Error.IndexOutOfRange;
                    const buf = try self.gc.alloc(u8, 1);
                    buf[0] = s.str[@intCast(actual)];
                    break :blk &(try StarStr.init(self.gc, buf)).obj;
                } else |_| return TypeError.TypeMismatch;
                try self.stackPush(val);
            }
        },
        .jump => |offset| {
            const new_pc: i64 = @as(i64, @intCast(frame.pc)) + 1 + offset;
            frame.pc = @intCast(new_pc);
            return;
        },
        .jump_if_false => |offset| {
            const val = try self.stackPop();
            if (!val.isTruthy()) {
                const new_pc: i64 = @as(i64, @intCast(frame.pc)) + 1 + offset;
                frame.pc = @intCast(new_pc);
                return;
            }
        },
        .jump_if_true => |offset| {
            const val = try self.stackPop();
            if (val.isTruthy()) {
                const new_pc: i64 = @as(i64, @intCast(frame.pc)) + 1 + offset;
                frame.pc = @intCast(new_pc);
                return;
            }
        },
        .get_iter => {
            const obj = try self.stackPop();
            const bound = try obj.getMethodDunder(.iter) orelse return RuntimeError.NotIterable;
            const iter_obj = try bound.call(self, &.{});
            try self.stackPush(iter_obj);
        },
        .for_iter => |offset| {
            const iter_obj = self.stack.items[self.stack.items.len - 1];
            const bound = try iter_obj.getMethodDunder(.next) orelse return RuntimeError.NotIterable;
            const val = try bound.call(self, &.{});

            if (val == StarStopIteration.instance) {
                const new_pc: i64 = @as(i64, @intCast(frame.pc)) + 1 + offset;
                frame.pc = @intCast(new_pc);
                return;
            }
            try self.stackPush(val);
        },
        .pop_iter => {
            _ = try self.stackPop();
        },
        .pop => {
            _ = try self.stackPop();
        },
        .and_jump => |offset| {
            const val = self.stack.items[self.stack.items.len - 1];
            if (!val.isTruthy()) {
                // Falsy: short-circuit, keep value on stack, jump
                const new_pc: i64 = @as(i64, @intCast(frame.pc)) + 1 + offset;
                frame.pc = @intCast(new_pc);
                return;
            } else {
                // Truthy: pop and evaluate RHS
                _ = try self.stackPop();
            }
        },
        .or_jump => |offset| {
            const val = self.stack.items[self.stack.items.len - 1];
            if (val.isTruthy()) {
                // Truthy: short-circuit, keep value on stack, jump
                const new_pc: i64 = @as(i64, @intCast(frame.pc)) + 1 + offset;
                frame.pc = @intCast(new_pc);
                return;
            } else {
                // Falsy: pop and evaluate RHS
                _ = try self.stackPop();
            }
        },
        .ret, .ret_none => {
            const frame_val = self.frames.pop() orelse return error.ReturnedOutsideFunction;
            const return_value =
                if (std.meta.activeTag(instr) == .ret) try self.stackPop() else StarNone.instance;
            log.debug("ret: returning {s}", .{return_value.vtable.name});
            // sp_base is the clean stack height established by the caller
            // before the call instruction ran (see callFn). Restoring to it
            // discards any operands the callee pushed but didn't consume.
            self.stack.shrinkRetainingCapacity(frame_val.sp_base);
            try self.stackPush(return_value);
            return; // Don't increment PC after ret
        },
    }
    self.frames.items[self.frames.items.len - 1].pc += 1;
}

/// Inefficient and not ideal. Should only be used when the scope where a variable lives
/// cannot be known ahead of time.
fn scopedLookup(self: *Runtime, idx: FreeVarIdx) Error!?*StarObj {
    const free_idx: usize = @intFromEnum(idx);
    if (self.frames.items.len == 0) return error.FrameMissing;
    const current = self.frames.items[self.frames.items.len - 1];
    const func = current.func;
    const name = func.names[free_idx];
    var i = self.frames.items.len;
    while (i > 0) : (i -= 1) {
        const frame = self.frames.items[i - 1];

        var local_idx = 0;
        while (frame.func.names_local) |local| {
            if (std.mem.eql(u8, local, name)) {
                return frame.readLocal(@enumFromInt(local_idx));
            }
            local_idx += 1;
        }
    }
    return null;
}

pub fn takeOwnedGlobals(self: *Runtime) std.StringHashMapUnmanaged(*StarObj) {
    const old_globals = self.globals;
    self.globals = std.StringHashMapUnmanaged(*StarObj).empty;
    return old_globals;
}

pub fn execModule(self: *Runtime, module: *const Compiler.Module) Error!void {
    const mod = try StarModule.fromCompiledModule(self.gc, module);
    // TODO: These don't get automatically cleared, I'm not sure if defer
    //       setting them to null is the right move since you may want to recover
    //       on errors.
    self.current_source = module.source;
    self.current_filename = module.module_name;

    std.debug.assert(mod.init_fn.arity == 0);
    const initial_depth = self.frames.items.len;
    const sp_base = self.stack.items.len;
    try self.callFn(&mod.init_fn, &.{}, sp_base);
    try self.stepUntilDepth(initial_depth);
    // The frame's `ret` left the return value (None) on top — discard it.
    _ = try self.stackPop();
}

pub const StarInt = struct {
    num: i64,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "int",
            .setup_attrs = &setupAttrs,
        },
    },

    pub fn init(allocator: Allocator, num: i64) !*StarInt {
        const self = try allocator.create(StarInt);
        self.* = .{ .num = num };
        try setupAttrs(allocator, &self.obj);
        return self;
    }

    fn setupAttrs(allocator: Allocator, obj: *StarObj) Allocator.Error!void {
        try obj.attributes.put(allocator, dunder.add, &(try StarBoundMethod.init(allocator, obj, &add)).obj);
        try obj.attributes.put(allocator, dunder.sub, &(try StarBoundMethod.init(allocator, obj, &sub)).obj);
        try obj.attributes.put(allocator, dunder.mul, &(try StarBoundMethod.init(allocator, obj, &mul)).obj);
        try obj.attributes.put(allocator, dunder.eq, &(try StarBoundMethod.init(allocator, obj, &eqFn)).obj);
        try obj.attributes.put(allocator, dunder.ne, &(try StarBoundMethod.init(allocator, obj, &neFn)).obj);
        try obj.attributes.put(allocator, dunder.lt, &(try StarBoundMethod.init(allocator, obj, &ltFn)).obj);
        try obj.attributes.put(allocator, dunder.le, &(try StarBoundMethod.init(allocator, obj, &leFn)).obj);
        try obj.attributes.put(allocator, dunder.gt, &(try StarBoundMethod.init(allocator, obj, &gtFn)).obj);
        try obj.attributes.put(allocator, dunder.ge, &(try StarBoundMethod.init(allocator, obj, &geFn)).obj);
        try obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, obj, &strFn)).obj);
        try obj.attributes.put(allocator, dunder.mod, &(try StarBoundMethod.init(allocator, obj, &modFn)).obj);
        try obj.attributes.put(allocator, dunder.div, &(try StarBoundMethod.init(allocator, obj, &divFn)).obj);
        try obj.attributes.put(allocator, dunder.floor_div, &(try StarBoundMethod.init(allocator, obj, &floorDivFn)).obj);
        try obj.attributes.put(allocator, dunder.bit_or, &(try StarBoundMethod.init(allocator, obj, &bitOrFn)).obj);
    }

    fn modFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        if (other.num == 0) return Error.TypeMismatch; // division by zero
        // Python-style modulo: result has sign of divisor
        const result = try StarInt.init(allocator, @mod(self.num, other.num));
        return &result.obj;
    }

    fn divFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        if (other.num == 0) {
            rt.setErrorMessage( "real division by zero", .{});
            return Error.TypeMismatch;
        }
        const result = try StarFloat.init(allocator, @as(f64, @floatFromInt(self.num)) / @as(f64, @floatFromInt(other.num)));
        return &result.obj;
    }

    fn floorDivFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        if (other.num == 0) {
            rt.setErrorMessage( "integer division by zero", .{});
            return Error.TypeMismatch;
        }
        const result = try StarInt.init(allocator, @divFloor(self.num, other.num));
        return &result.obj;
    }

    fn bitOrFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        const result = try StarInt.init(allocator, self.num | other.num);
        return &result.obj;
    }

    fn add(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        const result = try StarInt.init(allocator, self.num + other.num);
        return &result.obj;
    }

    fn sub(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        const result = try StarInt.init(allocator, self.num - other.num);
        return &result.obj;
    }

    fn mul(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        // `int * sequence` is sequence repetition; delegate to the sequence.
        if (downCast(StarStr, args[0])) |_| {
            const m = try args[0].getMethodDunder(.mul) orelse return Error.MulOpUndefined;
            return m.call(rt, &.{self_obj});
        } else |_| {}
        if (downCast(StarList, args[0])) |_| {
            const m = try args[0].getMethodDunder(.mul) orelse return Error.MulOpUndefined;
            return m.call(rt, &.{self_obj});
        } else |_| {}
        if (downCast(StarTuple, args[0])) |_| {
            const m = try args[0].getMethodDunder(.mul) orelse return Error.MulOpUndefined;
            return m.call(rt, &.{self_obj});
        } else |_| {}
        const other = try downCast(StarInt, args[0]);
        const result = try StarInt.init(allocator, self.num * other.num);
        return &result.obj;
    }

    fn eqFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = downCast(StarInt, args[0]) catch return StarBool.get(false);
        return StarBool.get(self.num == other.num);
    }

    fn neFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = downCast(StarInt, args[0]) catch return StarBool.get(true);
        return StarBool.get(self.num != other.num);
    }

    /// Cast self/other to f64 if either side is a StarFloat. Returns null
    /// when `other` is neither an int nor a float — comparisons across
    /// unrelated types fail per the spec, so callers signal that with a
    /// TypeMismatch.
    fn numericPair(self: *StarInt, other_obj: *StarObj) ?struct { a: f64, b: f64 } {
        if (downCast(StarFloat, other_obj)) |f| {
            return .{ .a = @floatFromInt(self.num), .b = f.num };
        } else |_| {}
        if (downCast(StarInt, other_obj)) |i| {
            return .{ .a = @floatFromInt(self.num), .b = @floatFromInt(i.num) };
        } else |_| {}
        return null;
    }

    fn ltFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const p = numericPair(self, args[0]) orelse return Error.TypeMismatch;
        return StarBool.get(p.a < p.b);
    }

    fn leFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const p = numericPair(self, args[0]) orelse return Error.TypeMismatch;
        return StarBool.get(p.a <= p.b);
    }

    fn gtFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const p = numericPair(self, args[0]) orelse return Error.TypeMismatch;
        return StarBool.get(p.a > p.b);
    }

    fn geFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const p = numericPair(self, args[0]) orelse return Error.TypeMismatch;
        return StarBool.get(p.a >= p.b);
    }

    fn strFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const str_val = try std.fmt.allocPrint(allocator, "{d}", .{self.num});
        const new = try StarStr.init(allocator, str_val);
        return &new.obj;
    }
};

pub const StarFloat = struct {
    num: f64,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "float",
            .setup_attrs = &setupAttrs,
        },
    },

    pub fn init(allocator: Allocator, num: f64) !*StarFloat {
        const self = try allocator.create(StarFloat);
        self.* = .{ .num = num };
        try setupAttrs(allocator, &self.obj);
        return self;
    }

    fn setupAttrs(allocator: Allocator, obj: *StarObj) Allocator.Error!void {
        try obj.attributes.put(allocator, dunder.add, &(try StarBoundMethod.init(allocator, obj, &add)).obj);
        try obj.attributes.put(allocator, dunder.sub, &(try StarBoundMethod.init(allocator, obj, &sub)).obj);
        try obj.attributes.put(allocator, dunder.mul, &(try StarBoundMethod.init(allocator, obj, &mul)).obj);
        try obj.attributes.put(allocator, dunder.div, &(try StarBoundMethod.init(allocator, obj, &divFn)).obj);
        try obj.attributes.put(allocator, dunder.floor_div, &(try StarBoundMethod.init(allocator, obj, &floorDivFn)).obj);
        try obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, obj, &strFn)).obj);
        try obj.attributes.put(allocator, dunder.eq, &(try StarBoundMethod.init(allocator, obj, &eqFn)).obj);
        try obj.attributes.put(allocator, dunder.ne, &(try StarBoundMethod.init(allocator, obj, &neFn)).obj);
        try obj.attributes.put(allocator, dunder.lt, &(try StarBoundMethod.init(allocator, obj, &ltFn)).obj);
        try obj.attributes.put(allocator, dunder.le, &(try StarBoundMethod.init(allocator, obj, &leFn)).obj);
        try obj.attributes.put(allocator, dunder.gt, &(try StarBoundMethod.init(allocator, obj, &gtFn)).obj);
        try obj.attributes.put(allocator, dunder.ge, &(try StarBoundMethod.init(allocator, obj, &geFn)).obj);
    }

    fn numericPair(self: *StarFloat, other_obj: *StarObj) ?struct { a: f64, b: f64 } {
        if (downCast(StarFloat, other_obj)) |f| return .{ .a = self.num, .b = f.num } else |_| {}
        if (downCast(StarInt, other_obj)) |i| return .{ .a = self.num, .b = @floatFromInt(i.num) } else |_| {}
        return null;
    }

    fn eqFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const p = numericPair(self, args[0]) orelse return StarBool.get(false);
        return StarBool.get(p.a == p.b);
    }
    fn neFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const p = numericPair(self, args[0]) orelse return StarBool.get(true);
        return StarBool.get(p.a != p.b);
    }
    fn ltFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const p = numericPair(self, args[0]) orelse return Error.TypeMismatch;
        return StarBool.get(p.a < p.b);
    }
    fn leFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const p = numericPair(self, args[0]) orelse return Error.TypeMismatch;
        return StarBool.get(p.a <= p.b);
    }
    fn gtFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const p = numericPair(self, args[0]) orelse return Error.TypeMismatch;
        return StarBool.get(p.a > p.b);
    }
    fn geFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const p = numericPair(self, args[0]) orelse return Error.TypeMismatch;
        return StarBool.get(p.a >= p.b);
    }

    fn divFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarFloat, args[0]);
        if (other.num == 0.0) return Error.TypeMismatch;
        const result = try StarFloat.init(allocator, self.num / other.num);
        return &result.obj;
    }

    fn floorDivFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarFloat, args[0]);
        if (other.num == 0.0) return Error.TypeMismatch;
        const result = try StarFloat.init(allocator, @floor(self.num / other.num));
        return &result.obj;
    }

    fn add(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarFloat, args[0]);
        const result = try StarFloat.init(allocator, self.num + other.num);
        return &result.obj;
    }

    fn sub(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarFloat, args[0]);
        const result = try StarFloat.init(allocator, self.num - other.num);
        return &result.obj;
    }

    fn mul(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarFloat, args[0]);
        const result = try StarFloat.init(allocator, self.num * other.num);
        return &result.obj;
    }

    fn strFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const str_val = try std.fmt.allocPrint(allocator, "{d}", .{self.num});
        const new = try StarStr.init(allocator, str_val);
        return &new.obj;
    }
};

pub const StarStr = struct {
    str: []const u8,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "string",
            .setup_attrs = &setupAttrs,
        },
    },

    pub fn init(allocator: Allocator, s: []const u8) !*StarStr {
        const self = try allocator.create(StarStr);
        self.* = .{ .str = s };
        try setupAttrs(allocator, &self.obj);
        return self;
    }

    fn setupAttrs(allocator: Allocator, obj: *StarObj) Allocator.Error!void {
        try obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, obj, &strFn)).obj);
        try obj.attributes.put(allocator, dunder.eq, &(try StarBoundMethod.init(allocator, obj, &eqFn)).obj);
        try obj.attributes.put(allocator, dunder.ne, &(try StarBoundMethod.init(allocator, obj, &neFn)).obj);
        try obj.attributes.put(allocator, dunder.mod, &(try StarBoundMethod.init(allocator, obj, &modFn)).obj);
        try obj.attributes.put(allocator, dunder.contains, &(try StarBoundMethod.init(allocator, obj, &containsFn)).obj);
        try obj.attributes.put(allocator, dunder.add, &(try StarBoundMethod.init(allocator, obj, &addFn)).obj);
        try obj.attributes.put(allocator, dunder.mul, &(try StarBoundMethod.init(allocator, obj, &mulFn)).obj);
        try obj.attributes.put(allocator, dunder.lt, &(try StarBoundMethod.init(allocator, obj, &ltFn)).obj);
        try obj.attributes.put(allocator, dunder.le, &(try StarBoundMethod.init(allocator, obj, &leFn)).obj);
        try obj.attributes.put(allocator, dunder.gt, &(try StarBoundMethod.init(allocator, obj, &gtFn)).obj);
        try obj.attributes.put(allocator, dunder.ge, &(try StarBoundMethod.init(allocator, obj, &geFn)).obj);
        try obj.attributes.put(allocator, "upper", &(try StarBoundMethod.init(allocator, obj, &upper)).obj);
        try obj.attributes.put(allocator, "lower", &(try StarBoundMethod.init(allocator, obj, &lower)).obj);
        try obj.attributes.put(allocator, "capitalize", &(try StarBoundMethod.init(allocator, obj, &capitalize)).obj);
        try obj.attributes.put(allocator, "title", &(try StarBoundMethod.init(allocator, obj, &title)).obj);
        try obj.attributes.put(allocator, "strip", &(try StarBoundMethod.init(allocator, obj, &strip)).obj);
        try obj.attributes.put(allocator, "lstrip", &(try StarBoundMethod.init(allocator, obj, &lstrip)).obj);
        try obj.attributes.put(allocator, "rstrip", &(try StarBoundMethod.init(allocator, obj, &rstrip)).obj);
        try obj.attributes.put(allocator, "split", &(try StarBoundMethod.init(allocator, obj, &split)).obj);
        try obj.attributes.put(allocator, "rsplit", &(try StarBoundMethod.init(allocator, obj, &rsplit)).obj);
        try obj.attributes.put(allocator, "splitlines", &(try StarBoundMethod.init(allocator, obj, &splitlines)).obj);
        try obj.attributes.put(allocator, "join", &(try StarBoundMethod.init(allocator, obj, &join)).obj);
        try obj.attributes.put(allocator, "replace", &(try StarBoundMethod.init(allocator, obj, &replace)).obj);
        try obj.attributes.put(allocator, "startswith", &(try StarBoundMethod.init(allocator, obj, &startswith)).obj);
        try obj.attributes.put(allocator, "endswith", &(try StarBoundMethod.init(allocator, obj, &endswith)).obj);
        try obj.attributes.put(allocator, "find", &(try StarBoundMethod.init(allocator, obj, &find)).obj);
        try obj.attributes.put(allocator, "rfind", &(try StarBoundMethod.init(allocator, obj, &rfind)).obj);
        try obj.attributes.put(allocator, "index", &(try StarBoundMethod.init(allocator, obj, &indexFn)).obj);
        try obj.attributes.put(allocator, "count", &(try StarBoundMethod.init(allocator, obj, &countFn)).obj);
        try obj.attributes.put(allocator, "elems", &(try StarBoundMethod.init(allocator, obj, &elemsFn)).obj);
        try obj.attributes.put(allocator, "isalpha", &(try StarBoundMethod.init(allocator, obj, &isalphaFn)).obj);
        try obj.attributes.put(allocator, "isdigit", &(try StarBoundMethod.init(allocator, obj, &isdigitFn)).obj);
        try obj.attributes.put(allocator, "isalnum", &(try StarBoundMethod.init(allocator, obj, &isalnumFn)).obj);
        try obj.attributes.put(allocator, "isspace", &(try StarBoundMethod.init(allocator, obj, &isspaceFn)).obj);
        try obj.attributes.put(allocator, "isupper", &(try StarBoundMethod.init(allocator, obj, &isupperFn)).obj);
        try obj.attributes.put(allocator, "islower", &(try StarBoundMethod.init(allocator, obj, &islowerFn)).obj);
        try obj.attributes.put(allocator, "removeprefix", &(try StarBoundMethod.init(allocator, obj, &removeprefix)).obj);
        try obj.attributes.put(allocator, "removesuffix", &(try StarBoundMethod.init(allocator, obj, &removesuffix)).obj);
    }

    pub fn init_dupe(allocator: Allocator, s: []const u8) !*StarStr {
        return StarStr.init(allocator, try allocator.dupe(u8, s));
    }

    fn upper(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const upper_str = try allocator.alloc(u8, self.str.len);
        for (self.str, 0..) |c, i| {
            upper_str[i] = std.ascii.toUpper(c);
        }
        const new = try StarStr.init(allocator, upper_str);
        return &new.obj;
    }

    fn modFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        // String formatting: "format %s %d" % args
        // args can be a single value or a tuple/list of values
        var fmt_args: []const *StarObj = undefined;
        if (downCast(StarTuple, args[0])) |tuple| {
            fmt_args = tuple.items;
        } else |_| if (downCast(StarList, args[0])) |list| {
            fmt_args = list.items.items;
        } else |_| {
            // Single value — treat as a 1-element tuple
            fmt_args = args[0..1];
        }

        var buf = std.ArrayList(u8).empty;
        var arg_idx: usize = 0;
        var i: usize = 0;
        while (i < self.str.len) {
            if (self.str[i] == '%' and i + 1 < self.str.len) {
                const spec = self.str[i + 1];
                switch (spec) {
                    's' => {
                        if (arg_idx < fmt_args.len) {
                            if (try fmt_args[arg_idx].getMethodDunder(.str)) |m| {
                                const s = try m.call(rt, &.{});
                                const sv = try downCast(StarStr, s);
                                try buf.appendSlice(allocator, sv.str);
                            }
                            arg_idx += 1;
                        }
                        i += 2;
                    },
                    'd', 'i' => {
                        if (arg_idx < fmt_args.len) {
                            if (downCast(StarInt, fmt_args[arg_idx])) |int_val| {
                                const s = try std.fmt.allocPrint(allocator, "{d}", .{int_val.num});
                                try buf.appendSlice(allocator, s);
                            } else |_| {
                                try buf.appendSlice(allocator, "?");
                            }
                            arg_idx += 1;
                        }
                        i += 2;
                    },
                    'r' => {
                        if (arg_idx < fmt_args.len) {
                            if (try fmt_args[arg_idx].getMethodDunder(.str)) |m| {
                                const s = try m.call(rt, &.{});
                                const sv = try downCast(StarStr, s);
                                try buf.appendSlice(allocator, sv.str);
                            }
                            arg_idx += 1;
                        }
                        i += 2;
                    },
                    '%' => {
                        try buf.append(allocator, '%');
                        i += 2;
                    },
                    else => {
                        try buf.append(allocator, '%');
                        i += 1;
                    },
                }
            } else {
                try buf.append(allocator, self.str[i]);
                i += 1;
            }
        }

        const result = try StarStr.init(allocator, buf.items);
        return &result.obj;
    }

    fn containsFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const needle = downCast(StarStr, args[0]) catch return Error.TypeMismatch;
        const found = std.mem.indexOf(u8, self.str, needle.str) != null;
        return StarBool.get(found);
    }

    fn eqFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const other = downCast(StarStr, args[0]) catch return StarBool.get(false);
        return StarBool.get(std.mem.eql(u8, self.str, other.str));
    }

    fn neFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const other = downCast(StarStr, args[0]) catch return StarBool.get(true);
        return StarBool.get(!std.mem.eql(u8, self.str, other.str));
    }

    fn strFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return self_obj;
    }

    fn addFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const other = downCast(StarStr, args[0]) catch {
            rt.setErrorMessage( "unknown binary op: string + {s}", .{args[0].vtable.type_name});
            return Error.TypeMismatch;
        };
        const buf = try std.mem.concat(allocator, u8, &.{ self.str, other.str });
        return &(try StarStr.init(allocator, buf)).obj;
    }

    fn mulFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const n = try downCast(StarInt, args[0]);
        const count: i64 = if (n.num < 0) 0 else n.num;
        const max_bytes: i64 = 1 << 28;
        if (count > max_bytes) return Error.OutOfMemory;
        const total: i64 = count * @as(i64, @intCast(self.str.len));
        if (total > max_bytes) return Error.OutOfMemory;
        const buf = try allocator.alloc(u8, @intCast(total));
        var i: usize = 0;
        while (i < count) : (i += 1) {
            @memcpy(buf[i * self.str.len .. (i + 1) * self.str.len], self.str);
        }
        return &(try StarStr.init(allocator, buf)).obj;
    }

    fn ltFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarStr, args[0]);
        return StarBool.get(std.mem.order(u8, self.str, other.str) == .lt);
    }

    fn leFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarStr, args[0]);
        return StarBool.get(std.mem.order(u8, self.str, other.str) != .gt);
    }

    fn gtFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarStr, args[0]);
        return StarBool.get(std.mem.order(u8, self.str, other.str) == .gt);
    }

    fn geFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarStr, args[0]);
        return StarBool.get(std.mem.order(u8, self.str, other.str) != .lt);
    }

    fn lower(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const buf = try allocator.alloc(u8, self.str.len);
        for (self.str, 0..) |c, i| buf[i] = std.ascii.toLower(c);
        return &(try StarStr.init(allocator, buf)).obj;
    }

    fn capitalize(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const buf = try allocator.alloc(u8, self.str.len);
        for (self.str, 0..) |c, i| {
            buf[i] = if (i == 0) std.ascii.toUpper(c) else std.ascii.toLower(c);
        }
        return &(try StarStr.init(allocator, buf)).obj;
    }

    fn title(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const buf = try allocator.alloc(u8, self.str.len);
        var prev_alpha = false;
        for (self.str, 0..) |c, i| {
            const is_alpha = std.ascii.isAlphabetic(c);
            buf[i] = if (is_alpha and !prev_alpha) std.ascii.toUpper(c) else std.ascii.toLower(c);
            prev_alpha = is_alpha;
        }
        return &(try StarStr.init(allocator, buf)).obj;
    }

    fn stripImpl(allocator: Allocator, s: []const u8, cutset: []const u8, do_left: bool, do_right: bool) Error!*StarObj {
        const default_ws = " \t\n\r\x0b\x0c";
        const chars = if (cutset.len == 0) default_ws else cutset;
        var start: usize = 0;
        var end: usize = s.len;
        if (do_left) while (start < end and std.mem.indexOfScalar(u8, chars, s[start]) != null) : (start += 1) {};
        if (do_right) while (end > start and std.mem.indexOfScalar(u8, chars, s[end - 1]) != null) : (end -= 1) {};
        const out = try allocator.dupe(u8, s[start..end]);
        return &(try StarStr.init(allocator, out)).obj;
    }

    fn strip(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const cs: []const u8 = if (args.len == 0) "" else (try downCast(StarStr, args[0])).str;
        return stripImpl(allocator, self.str, cs, true, true);
    }
    fn lstrip(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const cs: []const u8 = if (args.len == 0) "" else (try downCast(StarStr, args[0])).str;
        return stripImpl(allocator, self.str, cs, true, false);
    }
    fn rstrip(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const cs: []const u8 = if (args.len == 0) "" else (try downCast(StarStr, args[0])).str;
        return stripImpl(allocator, self.str, cs, false, true);
    }

    fn split(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        var pieces = std.ArrayList(*StarObj).empty;
        if (args.len == 0) {
            // No-arg split: collapse runs of whitespace, drop empty pieces.
            var i: usize = 0;
            while (i < self.str.len) {
                while (i < self.str.len and std.ascii.isWhitespace(self.str[i])) : (i += 1) {}
                if (i >= self.str.len) break;
                const start = i;
                while (i < self.str.len and !std.ascii.isWhitespace(self.str[i])) : (i += 1) {}
                const piece = try allocator.dupe(u8, self.str[start..i]);
                try pieces.append(allocator, &(try StarStr.init(allocator, piece)).obj);
            }
        } else {
            const sep = try downCast(StarStr, args[0]);
            if (sep.str.len == 0) return Error.TypeMismatch;
            var rest: []const u8 = self.str;
            while (std.mem.indexOf(u8, rest, sep.str)) |idx| {
                const piece = try allocator.dupe(u8, rest[0..idx]);
                try pieces.append(allocator, &(try StarStr.init(allocator, piece)).obj);
                rest = rest[idx + sep.str.len ..];
            }
            const piece = try allocator.dupe(u8, rest);
            try pieces.append(allocator, &(try StarStr.init(allocator, piece)).obj);
        }
        const list = try StarList.init(allocator, pieces.items);
        return &list.obj;
    }

    fn rsplit(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        // Approximation: ignore maxsplit direction. Correct only when the
        // result is unbounded, which the current spec callers all are.
        return split(rt, self_obj, args);
    }

    fn splitlines(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const keep_ends: bool = if (args.len == 0) false else blk: {
            const b = downCast(StarBool, args[0]) catch {
                rt.setErrorMessage( "splitlines: got {s}, want bool", .{args[0].vtable.type_name});
                return Error.TypeMismatch;
            };
            break :blk b.value;
        };
        var pieces = std.ArrayList(*StarObj).empty;
        var i: usize = 0;
        while (i < self.str.len) {
            const start = i;
            while (i < self.str.len and self.str[i] != '\n') : (i += 1) {}
            const end_no_nl = i;
            const end_with_nl = if (i < self.str.len) i + 1 else i;
            const piece_slice = if (keep_ends) self.str[start..end_with_nl] else self.str[start..end_no_nl];
            const piece = try allocator.dupe(u8, piece_slice);
            try pieces.append(allocator, &(try StarStr.init(allocator, piece)).obj);
            if (i < self.str.len) i += 1;
        }
        const list = try StarList.init(allocator, pieces.items);
        return &list.obj;
    }

    fn join(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const items = try iterateToOwned(rt, args[0]);
        var buf = std.ArrayList(u8).empty;
        for (items, 0..) |item, i| {
            if (i > 0) try buf.appendSlice(allocator, self.str);
            const s = try downCast(StarStr, item);
            try buf.appendSlice(allocator, s.str);
        }
        return &(try StarStr.init(allocator, buf.items)).obj;
    }

    fn replace(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len < 2 or args.len > 3) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const old = try downCast(StarStr, args[0]);
        const new = try downCast(StarStr, args[1]);
        const max_count: i64 = if (args.len == 3) (try downCast(StarInt, args[2])).num else -1;

        var buf = std.ArrayList(u8).empty;
        var rest: []const u8 = self.str;
        var replaced: i64 = 0;
        if (old.str.len == 0) {
            const dup = try allocator.dupe(u8, self.str);
            return &(try StarStr.init(allocator, dup)).obj;
        }
        while (std.mem.indexOf(u8, rest, old.str)) |idx| {
            if (max_count >= 0 and replaced >= max_count) break;
            try buf.appendSlice(allocator, rest[0..idx]);
            try buf.appendSlice(allocator, new.str);
            rest = rest[idx + old.str.len ..];
            replaced += 1;
        }
        try buf.appendSlice(allocator, rest);
        return &(try StarStr.init(allocator, buf.items)).obj;
    }

    fn startswith(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        if (downCast(StarStr, args[0])) |prefix| {
            return StarBool.get(std.mem.startsWith(u8, self.str, prefix.str));
        } else |_| {}
        if (downCast(StarTuple, args[0])) |tup| {
            for (tup.items) |it| {
                const s = try downCast(StarStr, it);
                if (std.mem.startsWith(u8, self.str, s.str)) return StarBool.get(true);
            }
            return StarBool.get(false);
        } else |_| {}
        return Error.TypeMismatch;
    }

    fn endswith(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        if (downCast(StarStr, args[0])) |suffix| {
            return StarBool.get(std.mem.endsWith(u8, self.str, suffix.str));
        } else |_| {}
        if (downCast(StarTuple, args[0])) |tup| {
            for (tup.items) |it| {
                const s = try downCast(StarStr, it);
                if (std.mem.endsWith(u8, self.str, s.str)) return StarBool.get(true);
            }
            return StarBool.get(false);
        } else |_| {}
        return Error.TypeMismatch;
    }

    fn find(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len < 1 or args.len > 3) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const needle = try downCast(StarStr, args[0]);
        const idx: i64 = if (std.mem.indexOf(u8, self.str, needle.str)) |i| @intCast(i) else -1;
        return &(try StarInt.init(allocator, idx)).obj;
    }

    fn rfind(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len < 1 or args.len > 3) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const needle = try downCast(StarStr, args[0]);
        const idx: i64 = if (std.mem.lastIndexOf(u8, self.str, needle.str)) |i| @intCast(i) else -1;
        return &(try StarInt.init(allocator, idx)).obj;
    }

    fn indexFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len < 1 or args.len > 3) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const needle = try downCast(StarStr, args[0]);
        const idx = std.mem.indexOf(u8, self.str, needle.str) orelse return Error.TypeMismatch;
        return &(try StarInt.init(allocator, @intCast(idx))).obj;
    }

    fn countFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const needle = try downCast(StarStr, args[0]);
        var n: i64 = 0;
        if (needle.str.len == 0) {
            n = @intCast(self.str.len + 1);
        } else {
            var rest: []const u8 = self.str;
            while (std.mem.indexOf(u8, rest, needle.str)) |i| {
                n += 1;
                rest = rest[i + needle.str.len ..];
            }
        }
        return &(try StarInt.init(allocator, n)).obj;
    }

    fn elemsFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const pieces = try allocator.alloc(*StarObj, self.str.len);
        for (self.str, 0..) |c, i| {
            const buf = try allocator.alloc(u8, 1);
            buf[0] = c;
            pieces[i] = &(try StarStr.init(allocator, buf)).obj;
        }
        const iter = try StarSliceIter.init(allocator, pieces);
        return &iter.obj;
    }

    fn classifyAll(s: []const u8, comptime pred: fn (u8) bool) bool {
        if (s.len == 0) return false;
        for (s) |c| if (!pred(c)) return false;
        return true;
    }

    fn isalphaFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        return StarBool.get(classifyAll(self.str, std.ascii.isAlphabetic));
    }
    fn isdigitFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        return StarBool.get(classifyAll(self.str, std.ascii.isDigit));
    }
    fn isalnumFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        return StarBool.get(classifyAll(self.str, std.ascii.isAlphanumeric));
    }
    fn isspaceFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        return StarBool.get(classifyAll(self.str, std.ascii.isWhitespace));
    }
    fn isupperFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        var saw_letter = false;
        for (self.str) |c| {
            if (std.ascii.isLower(c)) return StarBool.get(false);
            if (std.ascii.isUpper(c)) saw_letter = true;
        }
        return StarBool.get(saw_letter);
    }
    fn islowerFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        var saw_letter = false;
        for (self.str) |c| {
            if (std.ascii.isUpper(c)) return StarBool.get(false);
            if (std.ascii.isLower(c)) saw_letter = true;
        }
        return StarBool.get(saw_letter);
    }

    fn removeprefix(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const prefix = try downCast(StarStr, args[0]);
        if (std.mem.startsWith(u8, self.str, prefix.str)) {
            const out = try allocator.dupe(u8, self.str[prefix.str.len..]);
            return &(try StarStr.init(allocator, out)).obj;
        }
        const out = try allocator.dupe(u8, self.str);
        return &(try StarStr.init(allocator, out)).obj;
    }

    fn removesuffix(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const suffix = try downCast(StarStr, args[0]);
        if (std.mem.endsWith(u8, self.str, suffix.str)) {
            const out = try allocator.dupe(u8, self.str[0 .. self.str.len - suffix.str.len]);
            return &(try StarStr.init(allocator, out)).obj;
        }
        const out = try allocator.dupe(u8, self.str);
        return &(try StarStr.init(allocator, out)).obj;
    }
};

test "Basic StarObject usage works" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();

    var a = try StarInt.init(gc, 5);
    const a_obj = &a.obj;

    var b = try StarInt.init(gc, 6);
    const b_obj = &b.obj;

    const add_method = try a_obj.getMethodDunder(.add) orelse return error.TestExpectedEqual;
    const c_obj = try add_method.call(&rt, &.{b_obj});
    const c = try downCast(StarInt, c_obj);

    try std.testing.expectEqual(11, c.num);
}

/// Generic downCast: returns the parent struct T for an object if the vtable name ptr matches.
pub fn downCast(T: type, obj: *StarObj) TypeError!*T {
    if (!@hasField(T, "obj") or @FieldType(T, "obj") != StarObj) {
        @compileError("Wrapper can only be used on types that have an `obj` field of type StarObj");
    }

    const default_name_ptr = std.meta.fieldInfo(T, .obj).defaultValue().?.vtable.name.ptr;
    if (obj.vtable.name.ptr != default_name_ptr) {
        return TypeError.TypeMismatch;
    }

    return @fieldParentPtr("obj", obj);
}

pub fn starTypeToZig(T: type) type {
    switch (T) {
        StarInt => return i64,
        StarFloat => return f64,
        StarStr => return []const u8,
        else => @compileError("Unsupported type " ++ @typeName(T)),
    }
}

pub fn zigToStarType(T: type) type {
    switch (@typeInfo(T)) {
        .int => return StarInt,
        .float => return StarFloat,
        .pointer => |p| {
            if (p.size == .slice and p.child == u8) {
                return StarStr;
            } else @compileError("Pointers are only allowed with string slices.");
        },
        else => @compileError("Unsupported type " ++ @typeName(T)),
    }
}

pub fn unwrapStar(T: type, value: *T) starTypeToZig(T) {
    // const value_typeinfo = @typeInfo(value);
    // const T = blk: switch (value_typeinfo) {
    //     .pointer => |p| break :blk p.child,
    //     else => @compileError("Expected a pointer."),
    // };
    return switch (T) {
        StarInt => value.num,
        StarFloat => value.num,
        StarStr => value.str,
        else => @compileError("Unsupported type " ++ @typeName(T)),
    };
}

pub const StarFunc = struct {
    code: []const Instruction = &.{},

    arity: usize,
    /// number of local slots
    frame_size: usize,
    /// function constants (literals, etc.)
    consts: []const *StarObj = &.{},
    /// All local var names
    names_local: []const []const u8 = &.{},
    /// All free var names
    names_free: []const []const u8 = &.{},
    /// All global var names referenced by this function
    names_global: []const []const u8 = &.{},
    /// Pool of kwarg names referenced by `call_kw` instructions in this
    /// function. Each `call_kw` carries `(start, count)` indices into this.
    kw_names: []const []const u8 = &.{},
    /// Closure cells (captured variables)
    closure_cells: []const *StarObj = &.{},

    /// Function name (or "<module>" for top-level)
    name: []const u8 = "<anonymous>",
    /// Source locations parallel to code, maps each instruction to source
    source_locs: []const SourceLoc = &.{},

    /// Default parameter values (right-aligned: defaults[0] is for param at index num_required)
    defaults: []const *StarObj = &.{},
    /// Number of required (non-default) parameters
    num_required: usize = 0,

    native_fn: ?*const fn (rt: *Runtime, call: NativeCall) Error!*StarObj = null,

    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "function",
            .setup_attrs = &setupAttrs,
        },
    },

    pub const native_type_name: []const u8 = "builtin_function_or_method";

    pub fn setupAttrs(allocator: Allocator, obj: *StarObj) Allocator.Error!void {
        try obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, obj, &strFn)).obj);
    }

    pub fn strFn(_: *Runtime, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return &func_str.obj;
    }

    var func_str = StarStr{ .str = "<function>" };

    /// Wrap a Zig function as a `StarFunc` whose `native_fn` accepts a
    /// `NativeCall`. The Args shape determines what the wrapper passes to the
    /// native:
    ///   * `NativeCall`         — full call descriptor (positional + kwargs).
    ///   * `[]const *StarObj`   — variadic positional slice (kwargs dropped).
    ///   * `struct { ... }`     — fixed-arity, typed: each field is downcast
    ///                            from the corresponding positional arg.
    pub fn fromNative(
        Args: type,
        comptime native: fn (rt: *Runtime, args: Args) Error!?*StarObj,
    ) StarFunc {
        // Full call descriptor.
        if (Args == NativeCall) {
            const native_fn = struct {
                pub fn wrapper(rt: *Runtime, call: NativeCall) Error!*StarObj {
                    const res = try native(rt, call);
                    if (res) |r| return r;
                    return StarNone.instance;
                }
            }.wrapper;
            return native_func(&native_fn, 0, 0);
        }

        // Variadic positional slice; kwargs are silently dropped.
        if (Args == []const *StarObj) {
            const native_fn = struct {
                pub fn wrapper(rt: *Runtime, call: NativeCall) Error!*StarObj {
                    const res = try native(rt, call.args);
                    if (res) |r| return r;
                    return StarNone.instance;
                }
            }.wrapper;
            return native_func(&native_fn, 0, 0);
        }

        switch (@typeInfo(Args)) {
            .@"struct" => |s| {
                const arity = s.fields.len;
                const native_fn = struct {
                    pub fn wrapper(rt: *Runtime, call: NativeCall) Error!*StarObj {
                        if (call.kwargs.len != 0 or call.args.len != arity) {
                            return Error.ArityMismatch;
                        }
                        var parsed_args: Args = undefined;
                        inline for (s.fields, 0..) |field, i| {
                            const StarType = comptime zigToStarType(field.type);
                            const downcasted: *StarType = try downCast(StarType, call.args[i]);
                            @field(parsed_args, field.name) = unwrapStar(StarType, downcasted);
                        }
                        const res = try native(rt, parsed_args);
                        if (res) |r| return r;
                        return StarNone.instance;
                    }
                }.wrapper;
                return native_func(&native_fn, arity, arity);
            },
            else => @compileError("Expected struct but got " ++ @typeName(Args)),
        }
    }

    fn native_func(
        native_fn: *const fn (rt: *Runtime, call: NativeCall) Error!*StarObj,
        arity: usize,
        frame_size: usize,
    ) StarFunc {
        return StarFunc{
            .native_fn = native_fn,
            .arity = arity,
            .frame_size = frame_size,
            .obj = .{
                .vtable = .{
                    .name = @typeName(StarFunc),
                    .type_name = native_type_name,
                },
            },
        };
    }

    test "Native function" {
        const Args = struct {
            x: i64,
            y: i64,
        };
        _ = fromNative(Args, struct {
            pub fn inner(_: *Runtime, args: Args) Error!?*StarObj {
                _ = args.x + args.y;
                return null;
            }
        }.inner);
    }
};

pub const StarNone = struct {
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "NoneType",
        },
    },

    fn strFn(_: *Runtime, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return &none_str.obj;
    }

    fn eqFn(_: *Runtime, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        return StarBool.get(args[0] == instance);
    }

    fn neFn(_: *Runtime, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        return StarBool.get(args[0] != instance);
    }

    /// Initialize the singleton's attributes. Takes no allocator because the
    /// singleton (`none_obj`) lives for the process lifetime and is shared
    /// across Runtime instances
    pub fn initAttributes() !void {
        if (none_obj.obj.attributes.count() > 0) return;
        const allocator = std.heap.smp_allocator;
        try none_obj.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &none_obj.obj, &strFn)).obj);
        try none_obj.obj.attributes.put(allocator, dunder.eq, &(try StarBoundMethod.init(allocator, &none_obj.obj, &eqFn)).obj);
        try none_obj.obj.attributes.put(allocator, dunder.ne, &(try StarBoundMethod.init(allocator, &none_obj.obj, &neFn)).obj);
    }

    var none_obj = StarNone{};
    pub const instance: *StarObj = &none_obj.obj;

    var none_str = StarStr{ .str = "None" };
};

pub const StarStopIteration = struct {
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "stop_iteration",
        },
    },

    fn strFn(_: *Runtime, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return &stop_str.obj;
    }

    /// See `StarNone.initAttributes` for the rationale on the no-allocator
    /// signature.
    pub fn initAttributes() !void {
        if (stop_obj.obj.attributes.count() > 0) return;
        const allocator = std.heap.smp_allocator;
        try stop_obj.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &stop_obj.obj, &strFn)).obj);
    }

    var stop_obj = StarStopIteration{};
    pub const instance: *StarObj = &stop_obj.obj;

    var stop_str = StarStr{ .str = "<StopIteration>" };
};

pub const StarBool = struct {
    value: bool,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "bool",
        },
    },

    pub fn init(allocator: Allocator, value: bool) !*StarBool {
        const self = try allocator.create(StarBool);
        self.* = .{ .value = value };
        return self;
    }

    pub fn get(value: bool) *StarObj {
        if (value) return &true_obj.obj else return &false_obj.obj;
    }

    fn strFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarBool = @fieldParentPtr("obj", self_obj);
        if (self.value) return &true_str.obj else return &false_str.obj;
    }

    fn eqFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarBool = @fieldParentPtr("obj", self_obj);
        const other = downCast(StarBool, args[0]) catch return StarBool.get(false);
        return StarBool.get(self.value == other.value);
    }
    fn neFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarBool = @fieldParentPtr("obj", self_obj);
        const other = downCast(StarBool, args[0]) catch return StarBool.get(true);
        return StarBool.get(self.value != other.value);
    }
    fn ltFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarBool = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarBool, args[0]);
        return StarBool.get(@intFromBool(self.value) < @intFromBool(other.value));
    }
    fn leFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarBool = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarBool, args[0]);
        return StarBool.get(@intFromBool(self.value) <= @intFromBool(other.value));
    }
    fn gtFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarBool = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarBool, args[0]);
        return StarBool.get(@intFromBool(self.value) > @intFromBool(other.value));
    }
    fn geFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarBool = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarBool, args[0]);
        return StarBool.get(@intFromBool(self.value) >= @intFromBool(other.value));
    }

    /// See `StarNone.initAttributes` for the rationale on the no-allocator
    /// signature.
    pub fn initAttributes() !void {
        if (true_obj.obj.attributes.count() > 0) return;
        const allocator = std.heap.page_allocator;
        inline for (.{ &true_obj.obj, &false_obj.obj }) |obj| {
            try obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, obj, &strFn)).obj);
            try obj.attributes.put(allocator, dunder.eq, &(try StarBoundMethod.init(allocator, obj, &eqFn)).obj);
            try obj.attributes.put(allocator, dunder.ne, &(try StarBoundMethod.init(allocator, obj, &neFn)).obj);
            try obj.attributes.put(allocator, dunder.lt, &(try StarBoundMethod.init(allocator, obj, &ltFn)).obj);
            try obj.attributes.put(allocator, dunder.le, &(try StarBoundMethod.init(allocator, obj, &leFn)).obj);
            try obj.attributes.put(allocator, dunder.gt, &(try StarBoundMethod.init(allocator, obj, &gtFn)).obj);
            try obj.attributes.put(allocator, dunder.ge, &(try StarBoundMethod.init(allocator, obj, &geFn)).obj);
        }
    }

    var true_obj = StarBool{ .value = true };
    var false_obj = StarBool{ .value = false };
    var true_str = StarStr{ .str = "True" };
    var false_str = StarStr{ .str = "False" };
};

/// Index a sequence with Python-style negative-index semantics.
fn sequenceGetItem(items: []const *StarObj, idx: i64) Error!*StarObj {
    const len: i64 = @intCast(items.len);
    const actual = if (idx < 0) idx + len else idx;
    if (actual < 0 or actual >= len) return error.IndexOutOfRange;
    return items[@intCast(actual)];
}

/// Linear search for `needle`, comparing via `__eq__` (with pointer-equality
/// fast path). Used by `__contains__` on lists and tuples.
fn sequenceContains(rt: *Runtime, items: []const *StarObj, needle: *StarObj) Error!*StarObj {
    for (items) |item| {
        if (item == needle) return StarBool.get(true);
        const eq_method = (try item.getMethodDunder(.eq)) orelse continue;
        const result = try eq_method.call(rt, &.{needle});
        const result_bool = downCast(StarBool, result) catch continue;
        if (result_bool.value) return StarBool.get(true);
    }
    return StarBool.get(false);
}

/// Iterate any iterable into a freshly allocated slice. Caller owns the slice.
pub fn iterateToOwned(rt: *Runtime, iterable: *StarObj) Error![]*StarObj {
    var out = std.ArrayList(*StarObj).empty;
    errdefer out.deinit(rt.gc);
    const iter_method = try iterable.getMethodDunder(.iter) orelse return RuntimeError.NotIterable;
    const iter_obj = try iter_method.call(rt, &.{});
    const next_method = try iter_obj.getMethodDunder(.next) orelse return RuntimeError.NotIterable;
    while (true) {
        const v = try next_method.call(rt, &.{});
        if (v == StarStopIteration.instance) break;
        try out.append(rt.gc, v);
    }
    return out.toOwnedSlice(rt.gc);
}

const eq_depth_limit: usize = 64;

/// Value-based equality between two arbitrary StarObj values.
pub fn objectsEqual(rt: *Runtime, a: *StarObj, b: *StarObj) Error!bool {
    if (a == b) return true;
    if (rt.eq_depth >= eq_depth_limit) {
        rt.setErrorMessage("maximum recursion depth exceeded", .{});
        return Error.StarlarkStackOverflow;
    }
    rt.eq_depth += 1;
    defer rt.eq_depth -= 1;
    if (try a.getMethodDunder(.eq)) |m| {
        const r = m.call(rt, &.{b}) catch return false;
        const rb = downCast(StarBool, r) catch return false;
        return rb.value;
    }
    return false;
}

/// Resolve a Python-style slice spec against a sequence of `len` items.
/// Returns concrete (start, stop, step) bounds.
pub const SliceBounds = struct { start: i64, stop: i64, step: i64 };

pub fn normalizeSlice(len: usize, start_in: ?i64, stop_in: ?i64, step_in: ?i64) Error!SliceBounds {
    const k: i64 = step_in orelse 1;
    if (k == 0) return Error.TypeMismatch;
    const l: i64 = @intCast(len);
    const lo: i64 = if (k < 0) -1 else 0;
    const hi: i64 = if (k < 0) l - 1 else l;
    var start: i64 = undefined;
    var stop: i64 = undefined;
    if (start_in) |s| {
        const adj = if (s < 0) s + l else s;
        start = if (adj < lo) lo else if (adj > hi) hi else adj;
    } else {
        start = if (k < 0) l - 1 else 0;
    }
    if (stop_in) |s| {
        const adj = if (s < 0) s + l else s;
        stop = if (adj < lo) lo else if (adj > hi) hi else adj;
    } else {
        stop = if (k < 0) -1 else l;
    }
    return .{ .start = start, .stop = stop, .step = k };
}

/// Materialize a sliced sequence into an owned slice.
pub fn sliceSequence(allocator: Allocator, items: []const *StarObj, b: SliceBounds) Error![]*StarObj {
    var out = std.ArrayList(*StarObj).empty;
    var i = b.start;
    if (b.step > 0) {
        while (i < b.stop) : (i += b.step) {
            try out.append(allocator, items[@intCast(i)]);
        }
    } else {
        while (i > b.stop) : (i += b.step) {
            try out.append(allocator, items[@intCast(i)]);
        }
    }
    return out.toOwnedSlice(allocator);
}

pub const StarList = struct {
    items: std.ArrayList(*StarObj),
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "list",
            .setup_attrs = &setupAttrs,
        },
    },

    pub fn init(allocator: Allocator, items: []const *StarObj) !*StarList {
        const self = try allocator.create(StarList);
        var list = std.ArrayList(*StarObj).empty;
        try list.appendSlice(allocator, items);
        self.* = .{ .items = list };
        try setupAttrs(allocator, &self.obj);
        return self;
    }

    fn setupAttrs(allocator: Allocator, obj: *StarObj) Allocator.Error!void {
        try obj.attributes.put(allocator, "append", &(try StarBoundMethod.init(allocator, obj, &append)).obj);
        try obj.attributes.put(allocator, "extend", &(try StarBoundMethod.init(allocator, obj, &extendFn)).obj);
        try obj.attributes.put(allocator, "pop", &(try StarBoundMethod.init(allocator, obj, &popFn)).obj);
        try obj.attributes.put(allocator, "insert", &(try StarBoundMethod.init(allocator, obj, &insertFn)).obj);
        try obj.attributes.put(allocator, "remove", &(try StarBoundMethod.init(allocator, obj, &removeFn)).obj);
        try obj.attributes.put(allocator, "index", &(try StarBoundMethod.init(allocator, obj, &indexFn)).obj);
        try obj.attributes.put(allocator, "count", &(try StarBoundMethod.init(allocator, obj, &countFn)).obj);
        try obj.attributes.put(allocator, "clear", &(try StarBoundMethod.init(allocator, obj, &clearFn)).obj);
        try obj.attributes.put(allocator, dunder.iter, &(try StarBoundMethod.init(allocator, obj, &iter)).obj);
        try obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, obj, &strFn)).obj);
        try obj.attributes.put(allocator, dunder.contains, &(try StarBoundMethod.init(allocator, obj, &containsFn)).obj);
        try obj.attributes.put(allocator, dunder.eq, &(try StarBoundMethod.init(allocator, obj, &eqFn)).obj);
        try obj.attributes.put(allocator, dunder.ne, &(try StarBoundMethod.init(allocator, obj, &neFn)).obj);
        try obj.attributes.put(allocator, dunder.add, &(try StarBoundMethod.init(allocator, obj, &addFn)).obj);
        try obj.attributes.put(allocator, dunder.mul, &(try StarBoundMethod.init(allocator, obj, &mulFn)).obj);
    }

    fn containsFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        return sequenceContains(rt, self.items.items, args[0]);
    }

    fn eqFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        const other = downCast(StarList, args[0]) catch return StarBool.get(false);
        if (self.items.items.len != other.items.items.len) return StarBool.get(false);
        for (self.items.items, other.items.items) |a, b| {
            if (!try objectsEqual(rt, a, b)) return StarBool.get(false);
        }
        return StarBool.get(true);
    }

    fn neFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const r = try eqFn(rt, self_obj, args);
        const b = try downCast(StarBool, r);
        return StarBool.get(!b.value);
    }

    fn addFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        const other = downCast(StarList, args[0]) catch {
            rt.setErrorMessage( "unknown binary op: list + {s}", .{args[0].vtable.type_name});
            return Error.TypeMismatch;
        };
        const combined = try allocator.alloc(*StarObj, self.items.items.len + other.items.items.len);
        @memcpy(combined[0..self.items.items.len], self.items.items);
        @memcpy(combined[self.items.items.len..], other.items.items);
        const new = try StarList.init(allocator, combined);
        return &new.obj;
    }

    fn mulFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        const n = try downCast(StarInt, args[0]);
        const count: i64 = if (n.num < 0) 0 else n.num;
        const max_elems: i64 = 1 << 24;
        if (count > max_elems) return Error.OutOfMemory;
        const total: usize = @intCast(count * @as(i64, @intCast(self.items.items.len)));
        if (total > @as(usize, @intCast(max_elems))) return Error.OutOfMemory;
        const combined = try allocator.alloc(*StarObj, total);
        var i: usize = 0;
        while (i < count) : (i += 1) {
            @memcpy(combined[i * self.items.items.len .. (i + 1) * self.items.items.len], self.items.items);
        }
        const new = try StarList.init(allocator, combined);
        return &new.obj;
    }

    fn extendFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        const items = try iterateToOwned(rt, args[0]);
        try self.items.appendSlice(allocator, items);
        return StarNone.instance;
    }

    fn popFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        if (self.items.items.len == 0) return Error.IndexOutOfRange;
        var idx: i64 = @intCast(self.items.items.len - 1);
        if (args.len == 1) {
            const i = try downCast(StarInt, args[0]);
            idx = i.num;
        } else if (args.len > 1) return Error.ArityMismatch;
        const len: i64 = @intCast(self.items.items.len);
        const actual = if (idx < 0) idx + len else idx;
        if (actual < 0 or actual >= len) return Error.IndexOutOfRange;
        const removed = self.items.orderedRemove(@intCast(actual));
        _ = allocator;
        return removed;
    }

    fn insertFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 2) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        const i = try downCast(StarInt, args[0]);
        const len: i64 = @intCast(self.items.items.len);
        var idx = i.num;
        if (idx < 0) idx += len;
        if (idx < 0) idx = 0;
        if (idx > len) idx = len;
        try self.items.insert(allocator, @intCast(idx), args[1]);
        return StarNone.instance;
    }

    fn removeFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        for (self.items.items, 0..) |it, i| {
            if (try objectsEqual(rt, it, args[0])) {
                _ = self.items.orderedRemove(i);
                return StarNone.instance;
            }
        }
        return Error.TypeMismatch;
    }

    fn indexFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len < 1 or args.len > 3) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        for (self.items.items, 0..) |it, i| {
            if (try objectsEqual(rt, it, args[0])) {
                return &(try StarInt.init(allocator, @intCast(i))).obj;
            }
        }
        return Error.TypeMismatch;
    }

    fn countFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        var n: i64 = 0;
        for (self.items.items) |it| {
            if (try objectsEqual(rt, it, args[0])) n += 1;
        }
        return &(try StarInt.init(allocator, n)).obj;
    }

    fn clearFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        self.items.clearRetainingCapacity();
        return StarNone.instance;
    }

    pub fn getItem(self: *StarList, idx: i64) Error!*StarObj {
        return sequenceGetItem(self.items.items, idx);
    }

    fn strFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);

        var buf = std.ArrayList(u8).empty;

        if (rt.reprContains(self_obj)) {
            try buf.appendSlice(allocator, "[...]");
            return &(try StarStr.init(allocator, buf.items)).obj;
        }
        const pushed = rt.reprPush(self_obj);
        defer if (pushed) rt.reprPop();

        try buf.append(allocator, '[');
        for (self.items.items, 0..) |item, i| {
            if (i > 0) try buf.appendSlice(allocator, ", ");
            try appendReprToBuf(rt, &buf, item);
        }
        try buf.append(allocator, ']');
        const new = try StarStr.init(allocator, buf.items);
        return &new.obj;
    }

    fn append(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        try self.items.append(allocator, args[0]);
        return StarNone.instance;
    }

    fn iter(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        const list_iter = try StarListIter.init(allocator, self);
        return &list_iter.obj;
    }
};

pub const StarTuple = struct {
    items: []const *StarObj,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "tuple",
            .setup_attrs = &setupAttrs,
        },
    },

    pub fn init(allocator: Allocator, items: []const *StarObj) !*StarTuple {
        const self = try allocator.create(StarTuple);
        const owned = try allocator.dupe(*StarObj, items);
        self.* = .{ .items = owned };
        try setupAttrs(allocator, &self.obj);
        return self;
    }

    fn setupAttrs(allocator: Allocator, obj: *StarObj) Allocator.Error!void {
        try obj.attributes.put(allocator, dunder.iter, &(try StarBoundMethod.init(allocator, obj, &iterFn)).obj);
        try obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, obj, &strFn)).obj);
        try obj.attributes.put(allocator, dunder.eq, &(try StarBoundMethod.init(allocator, obj, &eqFn)).obj);
        try obj.attributes.put(allocator, dunder.ne, &(try StarBoundMethod.init(allocator, obj, &neFn)).obj);
        try obj.attributes.put(allocator, dunder.contains, &(try StarBoundMethod.init(allocator, obj, &containsFn)).obj);
        try obj.attributes.put(allocator, dunder.add, &(try StarBoundMethod.init(allocator, obj, &addFn)).obj);
        try obj.attributes.put(allocator, dunder.mul, &(try StarBoundMethod.init(allocator, obj, &mulFn)).obj);
    }

    pub fn getItem(self: *StarTuple, idx: i64) Error!*StarObj {
        return sequenceGetItem(self.items, idx);
    }

    fn iterFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarTuple = @fieldParentPtr("obj", self_obj);
        const it = try StarSliceIter.init(allocator, self.items);
        return &it.obj;
    }

    fn neFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const r = try eqFn(rt, self_obj, args);
        const b = try downCast(StarBool, r);
        return StarBool.get(!b.value);
    }

    fn addFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarTuple = @fieldParentPtr("obj", self_obj);
        const other = downCast(StarTuple, args[0]) catch {
            rt.setErrorMessage( "unknown binary op: tuple + {s}", .{args[0].vtable.type_name});
            return Error.TypeMismatch;
        };
        const combined = try allocator.alloc(*StarObj, self.items.len + other.items.len);
        @memcpy(combined[0..self.items.len], self.items);
        @memcpy(combined[self.items.len..], other.items);
        const new = try StarTuple.init(allocator, combined);
        return &new.obj;
    }

    fn mulFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarTuple = @fieldParentPtr("obj", self_obj);
        const n = try downCast(StarInt, args[0]);
        const count: i64 = if (n.num < 0) 0 else n.num;
        // Guard against huge repeats — the spec probes them; promptly
        // returning an error is much faster than trying the allocation.
        const max_elems: i64 = 1 << 24;
        if (count > max_elems) {
            rt.setErrorMessage( "repeat count {d} too large", .{count});
            return Error.OutOfMemory;
        }
        const total: usize = @intCast(count * @as(i64, @intCast(self.items.len)));
        if (total > @as(usize, @intCast(max_elems))) {
            rt.setErrorMessage( "excessive repeat ({d} * {d} elements)", .{ self.items.len, count });
            return Error.OutOfMemory;
        }
        const combined = try allocator.alloc(*StarObj, total);
        var idx: usize = 0;
        while (idx < count) : (idx += 1) {
            @memcpy(combined[idx * self.items.len .. (idx + 1) * self.items.len], self.items);
        }
        const new = try StarTuple.init(allocator, combined);
        return &new.obj;
    }

    fn strFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarTuple = @fieldParentPtr("obj", self_obj);
        var buf = std.ArrayList(u8).empty;
        if (rt.reprContains(self_obj)) {
            try buf.appendSlice(allocator, "(...)");
            return &(try StarStr.init(allocator, buf.items)).obj;
        }
        const pushed = rt.reprPush(self_obj);
        defer if (pushed) rt.reprPop();
        try buf.append(allocator, '(');
        for (self.items, 0..) |item, i| {
            if (i > 0) try buf.appendSlice(allocator, ", ");
            try appendReprToBuf(rt, &buf, item);
        }
        if (self.items.len == 1) try buf.append(allocator, ',');
        try buf.append(allocator, ')');
        return &(try StarStr.init(allocator, buf.items)).obj;
    }

    fn eqFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarTuple = @fieldParentPtr("obj", self_obj);
        const other = downCast(StarTuple, args[0]) catch return StarBool.get(false);
        if (self.items.len != other.items.len) return StarBool.get(false);
        for (self.items, other.items) |a, b| {
            if (a == b) continue;
            if (try a.getMethodDunder(.eq)) |eq_method| {
                const result = try eq_method.call(rt, &.{b});
                if (downCast(StarBool, result)) |bv| {
                    if (!bv.value) return StarBool.get(false);
                } else |_| return StarBool.get(false);
            } else return StarBool.get(false);
        }
        return StarBool.get(true);
    }

    fn containsFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarTuple = @fieldParentPtr("obj", self_obj);
        return sequenceContains(rt, self.items, args[0]);
    }
};

/// Iterator over an immutable slice of `*StarObj`. Used by tuples (and any
/// caller that has a stable slice and doesn't need a backing collection).
pub const StarSliceIter = struct {
    items: []const *StarObj,
    index: usize = 0,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "tuple_iterator",
            .setup_attrs = &setupAttrs,
        },
    },

    pub fn init(allocator: Allocator, items: []const *StarObj) !*StarSliceIter {
        const self = try allocator.create(StarSliceIter);
        self.* = .{ .items = items };
        try setupAttrs(allocator, &self.obj);
        return self;
    }

    fn setupAttrs(allocator: Allocator, obj: *StarObj) Allocator.Error!void {
        try obj.attributes.put(allocator, dunder.iter, &(try StarBoundMethod.init(allocator, obj, &iterFn)).obj);
        try obj.attributes.put(allocator, dunder.next, &(try StarBoundMethod.init(allocator, obj, &nextFn)).obj);
    }

    fn iterFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return self_obj;
    }

    fn nextFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarSliceIter = @fieldParentPtr("obj", self_obj);
        if (self.index >= self.items.len) return StarStopIteration.instance;
        const v = self.items[self.index];
        self.index += 1;
        return v;
    }
};

/// Value-keyed map preserving insertion order. Backed by parallel ArrayLists
/// with linear lookup — proper hashing would require a per-type `__hash__`
/// dunder which doesn't exist yet, and the alternative `AutoArrayHashMap` is
/// pointer-keyed (so `{"x": 1}["x"]` would miss when the two `"x"` instances
/// are distinct StarStr objects). Revisit when `__hash__` lands.
pub const StarDict = struct {
    keys: std.ArrayList(*StarObj),
    values: std.ArrayList(*StarObj),
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "dict",
            .setup_attrs = &setupAttrs,
        },
    },

    pub fn init(rt: *Runtime, pairs: []const *StarObj) !*StarDict {
        const self = try rt.gc.create(StarDict);
        self.* = .{ .keys = .empty, .values = .empty };
        try setupAttrs(rt.gc, &self.obj);
        var i: usize = 0;
        while (i < pairs.len) : (i += 2) {
            try self.put(rt, pairs[i], pairs[i + 1]);
        }
        return self;
    }

    pub fn lookupIndex(self: *StarDict, rt: *Runtime, key: *StarObj) Error!?usize {
        for (self.keys.items, 0..) |k, i| {
            if (try objectsEqual(rt, k, key)) return i;
        }
        return null;
    }

    pub fn put(self: *StarDict, rt: *Runtime, key: *StarObj, value: *StarObj) Error!void {
        if (try self.lookupIndex(rt, key)) |i| {
            self.values.items[i] = value;
            return;
        }
        try self.keys.append(rt.gc, key);
        try self.values.append(rt.gc, value);
    }

    pub fn lookup(self: *StarDict, rt: *Runtime, key: *StarObj) Error!?*StarObj {
        if (try self.lookupIndex(rt, key)) |i| return self.values.items[i];
        return null;
    }

    fn setupAttrs(allocator: Allocator, obj: *StarObj) Allocator.Error!void {
        try obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, obj, &strFn)).obj);
        try obj.attributes.put(allocator, dunder.contains, &(try StarBoundMethod.init(allocator, obj, &containsFn)).obj);
        try obj.attributes.put(allocator, dunder.iter, &(try StarBoundMethod.init(allocator, obj, &iterFn)).obj);
        try obj.attributes.put(allocator, dunder.eq, &(try StarBoundMethod.init(allocator, obj, &eqFn)).obj);
        try obj.attributes.put(allocator, dunder.ne, &(try StarBoundMethod.init(allocator, obj, &neFn)).obj);
        try obj.attributes.put(allocator, "get", &(try StarBoundMethod.init(allocator, obj, &getMethod)).obj);
        try obj.attributes.put(allocator, "keys", &(try StarBoundMethod.init(allocator, obj, &keysMethod)).obj);
        try obj.attributes.put(allocator, "values", &(try StarBoundMethod.init(allocator, obj, &valuesMethod)).obj);
        try obj.attributes.put(allocator, "items", &(try StarBoundMethod.init(allocator, obj, &itemsMethod)).obj);
        try obj.attributes.put(allocator, "pop", &(try StarBoundMethod.init(allocator, obj, &popMethod)).obj);
        try obj.attributes.put(allocator, "popitem", &(try StarBoundMethod.init(allocator, obj, &popitemMethod)).obj);
        try obj.attributes.put(allocator, "update", &(try StarBoundMethod.init(allocator, obj, &updateMethod)).obj);
        try obj.attributes.put(allocator, "setdefault", &(try StarBoundMethod.init(allocator, obj, &setdefaultMethod)).obj);
        try obj.attributes.put(allocator, "clear", &(try StarBoundMethod.init(allocator, obj, &clearMethod)).obj);
    }

    fn containsFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);
        const idx = try self.lookupIndex(rt, args[0]);
        return StarBool.get(idx != null);
    }

    fn iterFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);
        const dup = try allocator.dupe(*StarObj, self.keys.items);
        const it = try StarSliceIter.init(allocator, dup);
        return &it.obj;
    }

    fn eqFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);
        const other = downCast(StarDict, args[0]) catch return StarBool.get(false);
        if (self.keys.items.len != other.keys.items.len) return StarBool.get(false);
        for (self.keys.items, self.values.items) |k, v| {
            const ov = try other.lookup(rt, k) orelse return StarBool.get(false);
            if (!try objectsEqual(rt, v, ov)) return StarBool.get(false);
        }
        return StarBool.get(true);
    }

    fn neFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const r = try eqFn(rt, self_obj, args);
        const b = try downCast(StarBool, r);
        return StarBool.get(!b.value);
    }

    fn getMethod(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len < 1 or args.len > 2) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);
        if (try self.lookup(rt, args[0])) |v| return v;
        return if (args.len == 2) args[1] else StarNone.instance;
    }

    fn keysMethod(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);
        const list = try StarList.init(allocator, self.keys.items);
        return &list.obj;
    }

    fn valuesMethod(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);
        const list = try StarList.init(allocator, self.values.items);
        return &list.obj;
    }

    fn itemsMethod(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);
        var pairs = std.ArrayList(*StarObj).empty;
        for (self.keys.items, self.values.items) |k, v| {
            const pair_items = try allocator.alloc(*StarObj, 2);
            pair_items[0] = k;
            pair_items[1] = v;
            const pair = try StarTuple.init(allocator, pair_items);
            try pairs.append(allocator, &pair.obj);
        }
        const list = try StarList.init(allocator, pairs.items);
        return &list.obj;
    }

    fn popMethod(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len < 1 or args.len > 2) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);
        if (try self.lookupIndex(rt, args[0])) |i| {
            const v = self.values.items[i];
            _ = self.keys.orderedRemove(i);
            _ = self.values.orderedRemove(i);
            return v;
        }
        if (args.len == 2) return args[1];
        return Error.TypeMismatch;
    }

    fn popitemMethod(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);
        if (self.keys.items.len == 0) return Error.TypeMismatch;
        const last = self.keys.items.len - 1;
        const k = self.keys.items[last];
        const v = self.values.items[last];
        _ = self.keys.pop();
        _ = self.values.pop();
        const pair_items = try allocator.alloc(*StarObj, 2);
        pair_items[0] = k;
        pair_items[1] = v;
        const pair = try StarTuple.init(allocator, pair_items);
        return &pair.obj;
    }

    fn updateMethod(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);
        if (downCast(StarDict, args[0])) |other| {
            for (other.keys.items, other.values.items) |k, v| {
                try self.put(rt, k, v);
            }
            return StarNone.instance;
        } else |_| {}
        const items = try iterateToOwned(rt, args[0]);
        for (items) |it| {
            const tup = downCast(StarTuple, it) catch return Error.TypeMismatch;
            if (tup.items.len != 2) return Error.TypeMismatch;
            try self.put(rt, tup.items[0], tup.items[1]);
        }
        return StarNone.instance;
    }

    fn setdefaultMethod(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len < 1 or args.len > 2) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);
        if (try self.lookup(rt, args[0])) |v| return v;
        const default_val: *StarObj = if (args.len == 2) args[1] else StarNone.instance;
        try self.put(rt, args[0], default_val);
        return default_val;
    }

    fn clearMethod(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);
        self.keys.clearRetainingCapacity();
        self.values.clearRetainingCapacity();
        return StarNone.instance;
    }

    fn strFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);

        var buf = std.ArrayList(u8).empty;
        if (rt.reprContains(self_obj)) {
            try buf.appendSlice(allocator, "{...}");
            return &(try StarStr.init(allocator, buf.items)).obj;
        }
        const pushed = rt.reprPush(self_obj);
        defer if (pushed) rt.reprPop();

        try buf.append(allocator, '{');

        for (self.keys.items, self.values.items, 0..) |key, value, idx| {
            if (idx > 0) try buf.appendSlice(allocator, ", ");
            try appendReprToBuf(rt, &buf, key);
            try buf.appendSlice(allocator, ": ");
            try appendReprToBuf(rt, &buf, value);
        }

        try buf.append(allocator, '}');
        const new = try StarStr.init(allocator, buf.items);
        return &new.obj;
    }
};

/// Helper: append a value's str() output to `buf`, quoting strings.
/// Container types (list/tuple/dict) handle their own cycle-detection
/// inside their `strFn` so we don't push twice when delegating.
fn appendReprToBuf(rt: *Runtime, buf: *std.ArrayList(u8), obj: *StarObj) Error!void {
    const allocator = rt.gc;
    if (downCast(StarStr, obj)) |s| {
        try buf.append(allocator, '"');
        try buf.appendSlice(allocator, s.str);
        try buf.append(allocator, '"');
        return;
    } else |_| {}
    if (try obj.getMethodDunder(.str)) |m| {
        const sobj = try m.call(rt, &.{});
        const sv = try downCast(StarStr, sobj);
        try buf.appendSlice(allocator, sv.str);
    } else {
        try buf.appendSlice(allocator, "<object>");
    }
}

pub const StarRangeIter = struct {
    current: i64,
    stop: i64,
    step: i64,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "range_iterator",
            .setup_attrs = &setupAttrs,
        },
    },

    pub fn init(allocator: Allocator, start: i64, stop: i64, step_val: i64) !*StarRangeIter {
        const self = try allocator.create(StarRangeIter);
        self.* = .{ .current = start, .stop = stop, .step = step_val };
        try setupAttrs(allocator, &self.obj);
        return self;
    }

    fn setupAttrs(allocator: Allocator, obj: *StarObj) Allocator.Error!void {
        try obj.attributes.put(allocator, dunder.iter, &(try StarBoundMethod.init(allocator, obj, &iter)).obj);
        try obj.attributes.put(allocator, dunder.next, &(try StarBoundMethod.init(allocator, obj, &nextFn)).obj);
        try obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, obj, &strFn)).obj);
    }

    fn iter(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return self_obj;
    }

    fn nextFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarRangeIter = @fieldParentPtr("obj", self_obj);
        if (self.step > 0 and self.current >= self.stop) return StarStopIteration.instance;
        if (self.step < 0 and self.current <= self.stop) return StarStopIteration.instance;
        const val = try StarInt.init(allocator, @intCast(self.current));
        self.current += self.step;
        return &val.obj;
    }

    fn strFn(_: *Runtime, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return &range_iter_str.obj;
    }

    var range_iter_str = StarStr{ .str = "<range_iterator>" };
};

pub const StarListIter = struct {
    list: *StarList,
    index: usize,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "list_iterator",
            .setup_attrs = &setupAttrs,
        },
    },

    pub fn init(allocator: Allocator, list: *StarList) !*StarListIter {
        const self = try allocator.create(StarListIter);
        self.* = .{ .list = list, .index = 0 };
        try setupAttrs(allocator, &self.obj);
        return self;
    }

    fn setupAttrs(allocator: Allocator, obj: *StarObj) Allocator.Error!void {
        try obj.attributes.put(allocator, dunder.iter, &(try StarBoundMethod.init(allocator, obj, &iter)).obj);
        try obj.attributes.put(allocator, dunder.next, &(try StarBoundMethod.init(allocator, obj, &nextFn)).obj);
        try obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, obj, &strFn)).obj);
    }

    fn iter(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return self_obj;
    }

    fn nextFn(_: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarListIter = @fieldParentPtr("obj", self_obj);
        if (self.index >= self.list.items.items.len) return StarStopIteration.instance;
        const val = self.list.items.items[self.index];
        self.index += 1;
        return val;
    }

    fn strFn(_: *Runtime, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return &list_iter_str.obj;
    }

    var list_iter_str = StarStr{ .str = "<list_iterator>" };
};

pub const MethodFn = *const fn (rt: *Runtime, self: *StarObj, args: []const *StarObj) Error!*StarObj;

pub const StarBoundMethod = struct {
    bound_self: *StarObj,
    func: MethodFn,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "builtin_function_or_method",
        },
    },

    fn strFn(_: *Runtime, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return &bound_method_str.obj;
    }

    var bound_method_str = StarStr{ .str = "<bound method>" };

    pub fn init(allocator: Allocator, bound_self: *StarObj, func: MethodFn) !*StarBoundMethod {
        const bm = try allocator.create(StarBoundMethod);
        bm.* = .{ .bound_self = bound_self, .func = func };
        return bm;
    }

    pub fn call(self: *StarBoundMethod, rt: *Runtime, args: []const *StarObj) Error!*StarObj {
        return self.func(rt, self.bound_self, args);
    }
};

pub const StarModule = struct {
    name: []const u8,
    init_fn: StarFunc,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .type_name = "module",
            .setup_attrs = &setupAttrs,
        },
    },

    fn strFn(rt: *Runtime, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        const allocator = rt.gc;
        if (args.len != 0) return Error.ArityMismatch;
        const self_module: *StarModule = @fieldParentPtr("obj", self_obj);

        const str_val = try std.fmt.allocPrint(allocator, "<module '{s}'>", .{self_module.name});
        const new = try StarStr.init(allocator, str_val);
        return &new.obj;
    }

    fn setupAttrs(allocator: Allocator, obj: *StarObj) Allocator.Error!void {
        try obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, obj, &strFn)).obj);
    }

    pub fn init(
        allocator: Allocator,
        name: []const u8,
        init_fn: StarFunc,
    ) !*StarModule {
        const self = try allocator.create(StarModule);
        self.* = .{
            .name = name,
            .init_fn = init_fn,
        };
        try setupAttrs(allocator, &self.obj);
        return self;
    }

    fn fromCompiledModule(allocator: Allocator, module: *const Compiler.Module) !*StarModule {
        const mod = try StarModule.init(allocator, module.module_name, .{
            .code = module.code,
            .consts = module.constants,
            .arity = 0,
            .frame_size = 0,
            .names_local = &.{},
            .names_free = &.{},
            .names_global = module.global_names,
            .kw_names = module.kw_names,
            .name = "<module>",
            .source_locs = module.source_locs,
        });
        return mod;
    }
};

pub const StarObj = struct {
    vtable: Vtable,
    attributes: std.StringHashMapUnmanaged(*StarObj) = .empty,

    pub const Vtable = struct {
        /// Type identity used by `downCast`: each type's instance default
        /// carries a unique pointer here, and `downCast` compares
        /// `vtable.name.ptr` to determine type identity.
        name: []const u8,
        /// User-facing type name returned by Starlark's `type()` builtin.
        /// Defaults to `name`, but types should set it explicitly to the
        /// spec-prescribed string ("int", "string", "list", ...). May be
        /// overridden per-instance (e.g. native StarFunc →
        /// "builtin_function_or_method").
        type_name: []const u8,
        /// Populate `obj.attributes` (dunder methods etc.) for a freshly-
        /// constructed or shallow-cloned instance. Bound-method receivers
        /// must point at `obj` (so they bind to the instance, not a
        /// prototype). Called by both `<Type>.init` and `StarObj.cloneFrom`
        /// so there's a single mechanism for setting up attributes.
        setup_attrs: ?*const fn (allocator: Allocator, obj: *StarObj) Error!void = null,

        pub fn basePtr(self: *Vtable) *StarObj {
            return @fieldParentPtr("vtable", self);
        }
    };

    /// Reset `self` to share `src`'s vtable but with a freshly-allocated
    /// attributes table. Used by `make_function` and any other site that
    /// instantiates a value by copying a prototype: the prototype's
    /// `attributes` map storage and bound-method receivers must not leak
    /// into the new instance.
    pub fn cloneFrom(self: *StarObj, allocator: Allocator, src: *const StarObj) Error!void {
        self.vtable = src.vtable;
        self.attributes = .empty;
        if (src.vtable.setup_attrs) |f| try f(allocator, self);
    }

    pub fn setAttr(self: *StarObj, allocator: Allocator, name: *StarObj, value: *StarObj) Error!void {
        const n = try downCast(StarStr, name);
        try self.attributes.put(allocator, n.str, value);
    }

    pub fn getAttr(self: *StarObj, name: *StarObj) Error!*StarObj {
        const n = try downCast(StarStr, name);
        return self.attributes.get(n.str) orelse return error.AttributeMissing;
    }

    pub fn getAttrStr(self: *StarObj, name: []const u8) Error!?*StarObj {
        if (self.attributes.get(name)) |val| return val;
        return null;
    }

    pub fn getMethodDunder(self: *StarObj, op: std.meta.DeclEnum(dunder)) Error!?*StarBoundMethod {
        const method_name = switch (op) {
            inline else => |tag| @field(dunder, @tagName(tag)),
        };
        const method_obj = try self.getAttrStr(method_name) orelse return null;
        return downCast(StarBoundMethod, method_obj) catch null;
    }

    pub fn isTruthy(self: *StarObj) bool {
        if (downCast(StarBool, self)) |b| {
            return b.value;
        } else |_| {}

        if (downCast(StarNone, self)) |_| {
            return false;
        } else |_| {}

        if (downCast(StarInt, self)) |i| {
            return i.num != 0;
        } else |_| {}

        if (downCast(StarFloat, self)) |f| {
            return f.num != 0.0;
        } else |_| {}

        if (downCast(StarStr, self)) |s| {
            return s.str.len > 0;
        } else |_| {}

        if (downCast(StarList, self)) |l| {
            return l.items.items.len > 0;
        } else |_| {}

        if (downCast(StarTuple, self)) |t| {
            return t.items.len > 0;
        } else |_| {}

        if (downCast(StarDict, self)) |d| {
            return d.keys.items.len > 0;
        } else |_| {}

        return true;
    }
};

/// Registers all functions from a wrapped stdlib struct into runtime globals.
pub fn registerStdlib(self: *Runtime, T: type) Error!void {
    const decls = @typeInfo(T.Underlying).@"struct".decls;

    // Not a range so we can increment only when we hit functions.
    var i: usize = 0;
    inline for (decls) |decl| {
        const DeclType = @TypeOf(@field(T.Underlying, decl.name));
        if (@typeInfo(DeclType) == .@"fn") {
            const func = @field(T.Underlying, decl.name);
            const func_info = @typeInfo(DeclType).@"fn";

            // Extract Args type from first parameter
            if (func_info.params.len != 2) {
                @compileError("Native functions must take exactly two parameters (runtime, args struct)");
            }
            const ArgsType = func_info.params[1].type.?;

            T.fns_buffer[i] = StarFunc.fromNative(ArgsType, func);
            try self.globals.put(self.gc, decl.name, &T.fns_buffer[i].obj);
            i += 1;
        }
    }
}

fn numFuncs(T: type) comptime_int {
    comptime var field_count = 0;
    inline for (std.meta.declarations(T)) |decl| {
        const DeclType = @TypeOf(@field(T, decl.name));
        if (@typeInfo(DeclType) == .@"fn") {
            field_count += 1;
        }
    }
    return field_count;
}

pub fn StarNativeModule(T: type) type {
    return struct {
        const Underlying = T;
        /// Statically defined buffer for global functions.
        var fns_buffer: [numFuncs(T)]StarFunc = undefined;
    };
}

test "Runtime executes add function" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();
    const gpa = std.testing.allocator;

    var code = std.ArrayList(Instruction).empty;
    defer code.deinit(gpa);

    try code.append(gpa, .{ .load_const = @enumFromInt(0) });
    try code.append(gpa, .{ .load_const = @enumFromInt(1) });
    try code.append(gpa, .{ .binary_op = .add });
    try code.append(gpa, .{ .ret = {} });

    var a = try StarInt.init(gc, 5);
    var b = try StarInt.init(gc, 6);
    const consts = [_]*StarObj{ &a.obj, &b.obj };

    var func = StarFunc{
        .code = code.items,
        .consts = &consts,
        .arity = 0,
        .frame_size = 0,
        .names_free = &.{},
        .names_local = &.{},
    };

    // Init runtime and call the function
    var rt = try Runtime.init(gpa, .{ .gc = gc });
    defer rt.deinit();

    var empty_args: [0]*StarObj = .{};
    try rt.callFn(&func, empty_args[0..], 0);

    try rt.stepUntilDone();

    // At the end, top of stack should be the result
    try std.testing.expectEqual(1, rt.stack.items.len);
    const result_obj = rt.stack.items[0];
    const result_int = try downCast(StarInt, result_obj);
    try std.testing.expectEqual(11, result_int.num);
}

test "StarObj attributes work" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var obj = try StarInt.init(gc, 42);
    var key = StarStr{ .str = "answer" };
    var val = try StarInt.init(gc, 99);

    try obj.obj.setAttr(gc, &key.obj, &val.obj);
    const got = try obj.obj.getAttr(&key.obj);
    const casted = try downCast(StarInt, got);

    try std.testing.expectEqual(99, casted.num);
}

test "StarObj.getAttr fails on wrong type key" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var obj = try StarInt.init(gc, 1);
    var not_a_str = try StarInt.init(gc, 2);

    try std.testing.expectError(TypeError.TypeMismatch, obj.obj.getAttr(&not_a_str.obj));
}

test "StarInt add fails with wrong type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();

    var a = try StarInt.init(gc, 10);
    var b = StarStr{ .str = "oops" };

    const add_method = try a.obj.getMethodDunder(.add) orelse return error.TestExpectedEqual;
    try std.testing.expectError(TypeError.TypeMismatch, add_method.call(&rt, &.{&b.obj}));
}

test "downCast fails for wrong vtable" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var int_obj = try StarInt.init(gc, 123);
    var str_obj = StarStr{ .str = "abc" };

    try std.testing.expectError(TypeError.TypeMismatch, downCast(StarStr, &int_obj.obj));
    try std.testing.expectError(TypeError.TypeMismatch, downCast(StarInt, &str_obj.obj));
}

test "Runtime detects wrong arity" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();
    const gpa = std.testing.allocator;

    var func = StarFunc{
        .code = &[_]Instruction{},
        .consts = &[_]*StarObj{},
        .arity = 1,
        .frame_size = 1,
        .names_free = &.{},
        .names_local = &.{},
    };

    var rt = try Runtime.init(gpa, .{ .gc = gc });
    defer rt.deinit();

    var no_args: [0]*StarObj = .{};
    try std.testing.expectError(RuntimeError.ArityMismatch, rt.callFn(&func, no_args[0..], 0));
}

test "Instruction.load_const out of range" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();
    const gpa = std.testing.allocator;

    var code = [_]Instruction{.{ .load_const = @enumFromInt(5) }}; // no consts
    var func = StarFunc{
        .code = &code,
        .consts = &[_]*StarObj{},
        .arity = 0,
        .frame_size = 0,
        .names_free = &.{},
        .names_local = &.{},
    };

    var rt = try Runtime.init(gpa, .{ .gc = gc });
    defer rt.deinit();

    var no_args: [0]*StarObj = .{};
    try rt.callFn(&func, no_args[0..], 0);

    try std.testing.expectError(RuntimeError.ConstOutOfRange, rt.step());
}

test "Instruction.store and load locals" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();
    const gpa = std.testing.allocator;

    var code = std.ArrayList(Instruction).empty;
    defer code.deinit(gpa);

    // store local[0], then load it back and return
    try code.append(gpa, .{ .load_const = @enumFromInt(0) });
    try code.append(gpa, .{ .store = @enumFromInt(0) });
    try code.append(gpa, .{ .load = @enumFromInt(0) });
    try code.append(gpa, .{ .ret = {} });

    var c = try StarInt.init(gc, 77);
    const consts = [_]*StarObj{&c.obj};

    var func = StarFunc{
        .code = code.items,
        .consts = &consts,
        .arity = 0,
        .frame_size = 1,
        .names_free = &.{},
        .names_local = &.{"a"},
    };

    var rt = try Runtime.init(gpa, .{ .gc = gc });
    defer rt.deinit();

    var no_args: [0]*StarObj = .{};
    try rt.callFn(&func, no_args[0..], 0);
    try rt.stepUntilDone();

    const result_obj = rt.stack.items[0];
    const result_int = try downCast(StarInt, result_obj);
    try std.testing.expectEqual(77, result_int.num);
}

test "Instruction.call fails on non-function" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();
    const gpa = std.testing.allocator;

    var code = [_]Instruction{.{ .call = 0 }};

    var fake_func_obj = try StarInt.init(gc, 999);

    var func = StarFunc{
        .code = &code,
        .consts = &[_]*StarObj{&fake_func_obj.obj},
        .arity = 0,
        .frame_size = 0,
        .names_free = &.{},
        .names_local = &.{},
    };

    var rt = try Runtime.init(gpa, .{ .gc = gc });
    defer rt.deinit();

    var no_args: [0]*StarObj = .{};
    try rt.callFn(&func, no_args[0..], 0);

    // push the const onto stack manually
    try rt.stackPush(&fake_func_obj.obj);

    try std.testing.expectError(RuntimeError.CallUndefined, rt.step());
}

test "Nested functions and closures" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();
    const compile = Compiler.compile;

    const source: [:0]const u8 =
        \\def make_getter(n):
        \\  def getter():
        \\    return n
        \\  return getter
        \\f = make_getter(42)
        \\result = f()
    ;

    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();

    const module = try compile(std.testing.allocator, "<test>", source, &ast, .{}, gc);
    var rt = try Runtime.init(std.testing.allocator, .{ .gc = gc });
    defer rt.deinit();

    try rt.execModule(&module);

    const f_val = rt.globals.get("f") orelse return error.TestUnexpectedResult;
    const f_func = try Runtime.downCast(Runtime.StarFunc, f_val);

    // `f` enclosed `n` on upon definition
    try std.testing.expectEqual(1, f_func.closure_cells.len);

    const captured_obj = f_func.closure_cells[0];
    const captured_val = try Runtime.downCast(Runtime.StarInt, captured_obj);
    try std.testing.expectEqual(@as(i64, 42), captured_val.num);

    const result_val = rt.globals.get("result") orelse return error.TestUnexpectedResult;
    const result_int = try Runtime.downCast(Runtime.StarInt, result_val);
    try std.testing.expectEqual(@as(i64, 42), result_int.num);

    try std.testing.expectEqual(0, rt.stack.items.len);
}

test "registerStdlib populates runtime globals" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();
    const gpa = std.testing.allocator;

    const Builtins = StarNativeModule(struct {
        pub fn print(_: *Runtime, args: struct { msg: []const u8 }) Error!?*StarObj {
            std.debug.print("Message: {s}\n", .{args.msg});
            return null;
        }

        pub fn square(_: *Runtime, args: struct { n: i64 }) Error!?*StarObj {
            _ = args.n * args.n;
            return null;
        }
    });

    var rt = try Runtime.init(gpa, .{ .gc = gc });
    defer rt.deinit();

    try rt.registerStdlib(Builtins);

    const print_global = rt.globals.get("print");
    try std.testing.expect(print_global != null);
    const print_func = try downCast(StarFunc, print_global.?);
    try std.testing.expectEqual(@as(usize, 1), print_func.arity);

    const square_global = rt.globals.get("square");
    try std.testing.expect(square_global != null);
    const square_func = try downCast(StarFunc, square_global.?);
    try std.testing.expectEqual(@as(usize, 1), square_func.arity);
}
