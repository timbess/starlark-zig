const std = @import("std");
const Allocator = std.mem.Allocator;
const Tokenizer = @import("Tokenizer.zig");
const Compiler = @import("Compiler.zig");
const Token = Tokenizer.Token;
const Ast = @import("Ast.zig");

const scope = .runtime;
const log = std.log.scoped(scope);

gpa: Allocator,
gc: Allocator,
/// Call stack for code/locals
frames: std.SegmentedList(Frame, 4096) = .{},
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

const knobs = struct {
    const locals_initial_capacity: usize = 16;
};

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
    pub const iter = "__iter__";
    pub const next = "__next__";
    pub const str = "__str__";
};

pub const InitOpts = struct {
    gc: Allocator,
};

pub const GlobalIdx = enum(u32) { _ };
pub const LocalIdx = enum(u32) { _ };
pub const ConstIdx = enum(u32) { _ };
pub const NameIdx = enum(u32) { _ };
pub const FreeVarIdx = enum(u32) { _ };

pub const Arity = u32;

pub const BinOp = enum { add, sub, mul, eq, ne, lt, le, gt, ge };

pub const Instruction = union(enum) {
    load: LocalIdx,
    load_const: ConstIdx,
    load_global: GlobalIdx,
    load_free: FreeVarIdx,
    store: LocalIdx,
    store_global: GlobalIdx,
    binary_op: BinOp,
    bool_not: void,
    call: Arity,
    ret: void,
    ret_none: void,
    make_closure: ConstIdx,
    /// Slice of the identifier from the original source code.
    get_attr: []const u8,
    /// Create a list from top N stack elements.
    build_list: u32,
    /// Create a dict from top N key-value pairs (2*N elements on stack).
    build_dict: u32,
    /// Index into object: pops index, pops object, pushes result.
    get_index: void,
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
        for (source[0..@min(offset, @as(u32, @intCast(source.len)))]) |c| {
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
        const clamped = @min(offset, @as(u32, @intCast(source.len)));
        var start: usize = clamped;
        while (start > 0 and source[start - 1] != '\n') start -= 1;
        var end: usize = clamped;
        while (end < source.len and source[end] != '\n') end += 1;
        return source[start..end];
    }

    fn getLineStartOffset(source: [:0]const u8, offset: u32) u32 {
        var pos: u32 = @min(offset, @as(u32, @intCast(source.len)));
        while (pos > 0 and source[pos - 1] != '\n') pos -= 1;
        return pos;
    }

    fn getErrorCategory(err: Error) []const u8 {
        return switch (err) {
            error.TypeMismatch => "TypeError",
            error.AddOpUndefined, error.SubOpUndefined, error.MulOpUndefined,
            error.ArityMismatch, error.LocalOutOfRange, error.LocalUninitialized,
            error.ConstOutOfRange, error.StackUnderflow, error.CallUndefined,
            error.FrameMissing, error.FrameNoReturn, error.ReturnedOutsideFunction,
            error.AttributeMissing, error.GlobalUndefined, error.FreeVarOutOfRange,
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
                const leading_ws = line_content.len - std.mem.trimLeft(u8, line_content, " \t").len;
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

const Frame = struct {
    func: *StarFunc,
    locals: std.SegmentedList(*StarObj, knobs.locals_initial_capacity),
    sp_base: usize,
    pc: usize, // pc inside this frame's code

    pub fn readConst(self: *Frame, idx: ConstIdx) Error!*StarObj {
        const const_idx = @intFromEnum(idx);
        if (const_idx >= self.func.consts.len) return RuntimeError.ConstOutOfRange;
        return self.func.consts[const_idx];
    }

    pub fn writeLocal(self: *Frame, idx: LocalIdx, value: *StarObj) Error!void {
        const local_idx = @intFromEnum(idx);
        if (local_idx >= self.locals.len) return RuntimeError.LocalOutOfRange;
        self.locals.at(local_idx).* = value;
    }

    pub fn readLocal(self: *Frame, idx: LocalIdx) Error!*StarObj {
        const local_idx = @intFromEnum(idx);
        if (local_idx >= self.locals.len) return RuntimeError.LocalOutOfRange;
        const local = self.locals.at(local_idx).*;
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
    // TODO: This isn't super ideal given that multiple Runtimes will share these globals.
    //       The options are:
    //          1. Attach them to the runtime and pass it around _everywhere_.
    //          2. Make them thread local and only allow a Runtime per thread.
    try StarNone.initAttributes(opts.gc);
    try StarBool.initAttributes(opts.gc);
    try StarStopIteration.initAttributes(opts.gc);
    return Runtime{
        .gpa = gpa,
        .gc = opts.gc,
    };
}

pub fn deinit(self: *Runtime) void {
    // free any frame locals still allocated
    var frame_iter = self.frames.iterator(0);
    while (frame_iter.next()) |frame| {
        frame.locals.deinit(self.gc);
    }
    self.frames.deinit(self.gc);
    self.stack.deinit(self.gc);
    self.globals.deinit(self.gc);
}

/// Create and push a new frame for `func`. Copies args into locals[0..arity).
/// ret_addr is the caller's next-pc (an index into the caller's code).
fn callFn(self: *Runtime, func: *StarFunc, args: []*StarObj) Error!void {
    if (args.len != func.arity) return RuntimeError.ArityMismatch;
    std.debug.assert(func.frame_size >= func.arity);

    if (comptime std.log.logEnabled(.debug, scope)) {
        log.debug("callFn: arity={d}, args=[", .{func.arity});
        for (args, 0..) |arg, i| {
            if (i > 0) log.debug(", ", .{});
            log.debug("{s}", .{arg.vtable.name});
        }
        log.debug("]", .{});
    }

    const sp_base = self.stack.items.len;
    log.debug("sp_base: {d}", .{sp_base});

    var frame = Frame{
        .func = func,
        .locals = .{},
        .sp_base = sp_base,
        .pc = 0,
    };
    if (func.frame_size > 0) {
        try frame.locals.growCapacity(self.gc, func.frame_size);
        try frame.locals.appendSlice(self.gc, args);
        for (args.len..func.frame_size) |_| {
            try frame.locals.append(self.gc, StarNone.instance);
        }
    }
    try self.frames.append(self.gc, frame);
}

fn stackPop(self: *Runtime) Error!*StarObj {
    return self.stack.pop() orelse return Error.StackUnderflow;
}

fn stackPush(self: *Runtime, obj: *StarObj) Error!void {
    try self.stack.append(self.gc, obj);
}

fn stackPushOne(self: *Runtime) Error!**StarObj {
    return self.stack.addOne(self.gc);
}

fn storeGlobal(self: *Runtime, name_idx: GlobalIdx, value: *StarObj) Error!void {
    const frame: *Frame = self.frames.at(self.frames.len - 1);
    const name = frame.func.names_global[@intFromEnum(name_idx)];
    try self.globals.put(self.gc, name, value);
}

fn loadGlobal(self: *Runtime, name_idx: GlobalIdx) Error!*StarObj {
    const frame: *Frame = self.frames.at(self.frames.len - 1);
    const name = frame.func.names_global[@intFromEnum(name_idx)];
    log.debug("Loading global: {s}", .{name});
    return self.globals.get(name) orelse return RuntimeError.GlobalUndefined;
}

/// Populate diagnostic with stack trace if required.
fn fillDiagnostic(self: *Runtime, err: Error) void {
    const diag = self.diagnostic orelse return;

    diag.allocator = self.gpa;
    diag.err = err;
    diag.source = self.current_source;
    if (self.current_filename) |cf| {
        diag.filename = cf;
    }

    if (self.frames.len == 0) {
        diag.trace = &.{};
        return;
    }

    var entries = std.ArrayList(StackTraceEntry).empty;
    var i: usize = 0;
    while (i < self.frames.len) : (i += 1) {
        const frame: *Frame = self.frames.at(i);
        const func = frame.func;
        const pc = if (frame.pc > 0) frame.pc else 0;
        const source_loc = if (pc < func.source_locs.len) func.source_locs[pc] else SourceLoc.none;

        entries.append(self.gpa, .{ .func_name = func.name, .source_loc = source_loc }) catch {
            diag.trace = &.{};
            return;
        };
    }

    diag.trace = entries.toOwnedSlice(self.gpa) catch &.{};
}

/// Step until frame stack is empty, filling in diagnostics in case of error.
pub fn stepUntilDone(self: *Runtime) Error!void {
    while (self.frames.len > 0) {
        self.step() catch |err| {
            self.fillDiagnostic(err);
            return err;
        };
    }
}

// Move interpreter forward one step. DOES NOT populate error diagnostics.
fn step(self: *Runtime) Error!void {
    if (self.frames.len == 0) return RuntimeError.FrameMissing;
    var frame: *Frame = self.frames.at(self.frames.len - 1);
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
            const rhs = try self.stackPop();
            const lhs = try self.stackPop();

            // Sorta nice that it enforces that the dunder struct must match the op name.
            const dunder_op = switch (op) {inline else => |o| @field(std.meta.DeclEnum(dunder), @tagName(o))};

            const bound_method = try lhs.getMethodDunder(dunder_op) orelse return Error.AddOpUndefined;
            const result = try bound_method.call(self.gc, &.{rhs});

            try self.stackPush(result);
        },
        .get_attr => |attr| {
            const obj = try self.stackPop();
            const val = try obj.getAttrStr(attr) orelse return Error.AttributeMissing;
            try self.stackPush(val);
        },
        .call => |arity_u| {
            const arity = @as(usize, arity_u);

            const sp = self.stack.items.len;
            if (sp < arity + 1) return RuntimeError.StackUnderflow;

            // layout: [..., func_obj, arg1, arg2, ... argN]
            const args_start = sp - arity;
            const func_index = args_start - 1;
            const func_obj = self.stack.items[func_index];
            log.debug("=== Calling function, arity={d}, type={s} ===", .{ arity, func_obj.vtable.name });

            const args_slice: []*StarObj = self.stack.items[args_start .. args_start + arity];

            // Try interpreted function first: downCast StarFunc
            const maybe_func: ?*StarFunc = downCast(StarFunc, func_obj) catch |e| switch (e) {
                TypeError.TypeMismatch => null,
                else => return e,
            };

            if (maybe_func) |fptr| {
                if (fptr.native_fn) |f| {
                    const result = try f(self, args_slice);
                    self.stack.shrinkRetainingCapacity(func_index);
                    try self.stackPush(result);
                } else try self.callFn(fptr, args_slice);
            } else if (downCast(StarBoundMethod, func_obj)) |bm| {
                const result = try bm.call(self.gc, args_slice);
                self.stack.shrinkRetainingCapacity(func_index);
                try self.stackPush(result);
            } else |_| {
                return RuntimeError.CallUndefined;
            }
        },
        .make_closure => |cidx| {
            log.debug("=== make_closure called, cidx={d}, frame.func.arity={d}, frame.func.names_local.len={d} ===", .{ @intFromEnum(cidx), frame.func.arity, frame.func.names_local.len });
            for (frame.func.names_local, 0..) |name, idx| {
                const val = try frame.readLocal(@enumFromInt(idx));
                log.debug("  local[{d}] '{s}' = {s}", .{ idx, name, val.vtable.name });
            }
            // Get the base function from constants
            const const_obj = try frame.readConst(cidx);
            const base_func = try downCast(StarFunc, const_obj);

            // Doesn't enclose over free variables.
            if (base_func.names_free.len == 0) {
                try self.stackPush(const_obj);
                return;
            }

            // Allocate closure_cells array
            const closure_cells = try self.gc.alloc(*StarObj, base_func.names_free.len);

            // Populate closure_cells by looking up each free variable name in current scope
            for (base_func.names_free, 0..) |name, i| {
                var found = false;

                for (frame.func.names_local, 0..) |local_name, local_idx| {
                    if (std.mem.eql(u8, name, local_name)) {
                        const captured = try frame.readLocal(@enumFromInt(local_idx));
                        log.debug("Capturing '{s}' from local[{d}], type: {s}", .{ name, local_idx, captured.vtable.name });
                        closure_cells[i] = captured;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    for (frame.func.names_free, 0..) |free_name, free_idx| {
                        if (std.mem.eql(u8, name, free_name)) {
                            closure_cells[i] = try frame.readFree(@enumFromInt(free_idx));
                            found = true;
                            break;
                        }
                    }
                }

                if (!found) {
                    return RuntimeError.GlobalUndefined; // Should never happen if compiler is correct
                }
            }

            // Create new function with captured cells
            const closure_func = try self.gc.create(StarFunc);
            closure_func.* = .{
                .code = base_func.code,
                .consts = base_func.consts,
                .arity = base_func.arity,
                .frame_size = base_func.frame_size,
                .names_local = base_func.names_local,
                .names_free = base_func.names_free,
                .closure_cells = closure_cells,
                .name = base_func.name,
                .source_locs = base_func.source_locs,
            };

            try self.stackPush(&closure_func.obj);
        },
        .bool_not => {
            const val = try self.stackPop();
            try self.stackPush(StarBool.get(!val.isTruthy()));
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
        .build_dict => |count| {
            const sp = self.stack.items.len;
            const total = count * 2;
            if (sp < total) return RuntimeError.StackUnderflow;
            const start = sp - total;
            const pairs = self.stack.items[start..sp];
            const dict = try StarDict.init(self.gc, pairs);
            self.stack.shrinkRetainingCapacity(start);
            try self.stackPush(&dict.obj);
        },
        .get_index => {
            const idx_obj = try self.stackPop();
            const obj = try self.stackPop();

            const list = downCast(StarList, obj) catch return TypeError.TypeMismatch;
            const idx_int = try downCast(StarInt, idx_obj);
            const val = try list.getItem(@intCast(idx_int.num));
            try self.stackPush(val);
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
            const iter_obj = try bound.call(self.gc, &.{});
            try self.stackPush(iter_obj);
        },
        .for_iter => |offset| {
            const iter_obj = self.stack.items[self.stack.items.len - 1];
            const bound = try iter_obj.getMethodDunder(.next) orelse return RuntimeError.NotIterable;
            const val = try bound.call(self.gc, &.{});

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
        .ret, .ret_none => {
            var frame_val = self.frames.pop() orelse return error.ReturnedOutsideFunction;
            frame_val.locals.deinit(self.gpa);
            const return_value =
                if (std.meta.activeTag(instr) == .ret) try self.stackPop() else StarNone.instance;
            log.debug("ret: returning {s}", .{return_value.vtable.name});
            // Pop args, and function reference.
            const new_sp = frame_val.sp_base - frame_val.func.arity - 1;
            std.debug.assert(new_sp >= 0);
            self.stack.shrinkRetainingCapacity(new_sp);
            try self.stackPush(return_value);
            return; // Don't increment PC after ret
        },
    }
    frame.pc += 1;
}

/// Inefficient and not ideal. Should only be used when the scope where a variable lives
/// cannot be known ahead of time.
fn scopedLookup(self: *Runtime, idx: FreeVarIdx) Error!?*StarObj {
    const free_idx: usize = @intFromEnum(idx);
    if (self.frames.items.len == 0) return error.FrameMissing;
    const current = self.frames.items[self.frames.items.len - 1];
    const func = current.func;
    const name = func.names[free_idx];
    var i = self.frames.len;
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
    try self.stackPush(&mod.init_fn.obj);
    try self.callFn(&mod.init_fn, &.{});
    try self.stepUntilDone();
    _ = try self.stackPop();
}

pub const StarInt = struct {
    num: u64,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
        },
    },

    pub fn init(allocator: Allocator, num: u64) !*StarInt {
        const self = try allocator.create(StarInt);
        self.* = .{ .num = num };
        try self.obj.attributes.put(allocator, dunder.add, &(try StarBoundMethod.init(allocator, &self.obj, &add)).obj);
        try self.obj.attributes.put(allocator, dunder.sub, &(try StarBoundMethod.init(allocator, &self.obj, &sub)).obj);
        try self.obj.attributes.put(allocator, dunder.mul, &(try StarBoundMethod.init(allocator, &self.obj, &mul)).obj);
        try self.obj.attributes.put(allocator, dunder.eq, &(try StarBoundMethod.init(allocator, &self.obj, &eqFn)).obj);
        try self.obj.attributes.put(allocator, dunder.ne, &(try StarBoundMethod.init(allocator, &self.obj, &neFn)).obj);
        try self.obj.attributes.put(allocator, dunder.lt, &(try StarBoundMethod.init(allocator, &self.obj, &ltFn)).obj);
        try self.obj.attributes.put(allocator, dunder.le, &(try StarBoundMethod.init(allocator, &self.obj, &leFn)).obj);
        try self.obj.attributes.put(allocator, dunder.gt, &(try StarBoundMethod.init(allocator, &self.obj, &gtFn)).obj);
        try self.obj.attributes.put(allocator, dunder.ge, &(try StarBoundMethod.init(allocator, &self.obj, &geFn)).obj);
        try self.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &self.obj, &strFn)).obj);
        return self;
    }

    fn add(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        const result = try StarInt.init(allocator, self.num + other.num);
        return &result.obj;
    }

    fn sub(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        const result = try StarInt.init(allocator, self.num - other.num);
        return &result.obj;
    }

    fn mul(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        const result = try StarInt.init(allocator, self.num * other.num);
        return &result.obj;
    }

    fn eqFn(_: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        return StarBool.get(self.num == other.num);
    }

    fn neFn(_: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        return StarBool.get(self.num != other.num);
    }

    fn ltFn(_: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        return StarBool.get(self.num < other.num);
    }

    fn leFn(_: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        return StarBool.get(self.num <= other.num);
    }

    fn gtFn(_: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        return StarBool.get(self.num > other.num);
    }

    fn geFn(_: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarInt = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarInt, args[0]);
        return StarBool.get(self.num >= other.num);
    }

    fn strFn(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
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
        },
    },

    pub fn init(allocator: Allocator, num: f64) !*StarFloat {
        const self = try allocator.create(StarFloat);
        self.* = .{ .num = num };
        try self.obj.attributes.put(allocator, dunder.add, &(try StarBoundMethod.init(allocator, &self.obj, &add)).obj);
        try self.obj.attributes.put(allocator, dunder.sub, &(try StarBoundMethod.init(allocator, &self.obj, &sub)).obj);
        try self.obj.attributes.put(allocator, dunder.mul, &(try StarBoundMethod.init(allocator, &self.obj, &mul)).obj);
        try self.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &self.obj, &strFn)).obj);
        return self;
    }

    fn add(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarFloat, args[0]);
        const result = try StarFloat.init(allocator, self.num + other.num);
        return &result.obj;
    }

    fn sub(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarFloat, args[0]);
        const result = try StarFloat.init(allocator, self.num - other.num);
        return &result.obj;
    }

    fn mul(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarFloat = @fieldParentPtr("obj", self_obj);
        const other = try downCast(StarFloat, args[0]);
        const result = try StarFloat.init(allocator, self.num * other.num);
        return &result.obj;
    }

    fn strFn(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
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
        },
    },

    pub fn init(allocator: Allocator, s: []const u8) !*StarStr {
        const self = try allocator.create(StarStr);
        self.* = .{ .str = s };
        try self.obj.attributes.put(allocator, "upper", &(try StarBoundMethod.init(allocator, &self.obj, &upper)).obj);
        try self.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &self.obj, &strFn)).obj);
        return self;
    }

    pub fn init_dupe(allocator: Allocator, s: []const u8) !*StarStr {
        return StarStr.init(allocator, try allocator.dupe(u8, s));
    }

    fn upper(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarStr = @fieldParentPtr("obj", self_obj);
        const upper_str = try allocator.alloc(u8, self.str.len);
        for (self.str, 0..) |c, i| {
            upper_str[i] = std.ascii.toUpper(c);
        }
        const new = try StarStr.init(allocator, upper_str);
        return &new.obj;
    }

    fn strFn(_: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return self_obj;
    }
};

test "Basic StarObject usage works" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var a = try StarInt.init(gc, 5);
    const a_obj = &a.obj;

    var b = try StarInt.init(gc, 6);
    const b_obj = &b.obj;

    const add_method = try a_obj.getMethodDunder(.add) orelse return error.TestExpectedEqual;
    const c_obj = try add_method.call(gc, &.{b_obj});
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
        StarInt => return u64,
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
    /// Closure cells (captured variables)
    closure_cells: []const *StarObj = &.{},

    /// Function name (or "<module>" for top-level)
    name: []const u8 = "<anonymous>",
    /// Source locations parallel to code, maps each instruction to source
    source_locs: []const SourceLoc = &.{},

    native_fn: ?*const fn (rt: *Runtime, args: []const *StarObj) Error!*StarObj = null,

    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
        },
    },

    pub fn strFn(_: Allocator, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return &func_str.obj;
    }

    var func_str = StarStr{ .str = "<function>" };

    pub fn fromNative(
        // TODO: kwargs
        Args: type,
        comptime native: fn (rt: *Runtime, args: Args) Error!?*StarObj,
    ) StarFunc {
        // Handle variadic args: []const *StarObj
        if (Args == []const *StarObj) {
            const native_fn = struct {
                pub fn wrapper(rt: *Runtime, args: []const *StarObj) Error!*StarObj {
                    const res = try native(rt, args);
                    if (res) |r| return r;
                    return StarNone.instance;
                }
            }.wrapper;
            return StarFunc{
                .native_fn = &native_fn,
                .arity = 0, // variadic
                .frame_size = 0,
            };
        }

        const args_typeinfo = @typeInfo(Args);
        switch (args_typeinfo) {
            .@"struct" => |s| {
                const arity = s.fields.len;
                const native_fn = struct {
                    pub fn wrapper(rt: *Runtime, args: []const *StarObj) Error!*StarObj {
                        var parsed_args: Args = undefined;
                        if (args.len != arity) return Error.ArityMismatch;
                        inline for (s.fields, 0..) |field, i| {
                            const StarType = comptime zigToStarType(field.type);
                            const downcasted: *StarType = try downCast(StarType, args[i]);
                            @field(parsed_args, field.name) = unwrapStar(StarType, downcasted);
                        }
                        const res = try native(rt, parsed_args);

                        if (res) |r| return r;
                        return StarNone.instance;
                    }
                }.wrapper;
                return StarFunc{
                    .native_fn = &native_fn,
                    .arity = arity,
                    .frame_size = arity,
                };
            },
            else => @compileError("Expected struct but got " ++ @typeName(Args)),
        }
    }

    test "Native function" {
        const Args = struct {
            x: u64,
            y: u64,
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
        },
    },

    fn strFn(_: Allocator, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return &none_str.obj;
    }

    pub fn initAttributes(allocator: Allocator) !void {
        if (none_obj.obj.attributes.count() > 0) return;
        try none_obj.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &none_obj.obj, &strFn)).obj);
    }

    var none_obj = StarNone{};
    pub const instance: *StarObj = &none_obj.obj;

    var none_str = StarStr{ .str = "None" };
};

pub const StarStopIteration = struct {
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
        },
    },

    fn strFn(_: Allocator, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return &stop_str.obj;
    }

    pub fn initAttributes(allocator: Allocator) !void {
        if (stop_obj.obj.attributes.count() > 0) return;
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

    fn strFn(_: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarBool = @fieldParentPtr("obj", self_obj);
        if (self.value) return &true_str.obj else return &false_str.obj;
    }

    pub fn initAttributes(allocator: Allocator) !void {
        if (true_obj.obj.attributes.count() > 0) return;
        try true_obj.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &true_obj.obj, &strFn)).obj);
        try false_obj.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &false_obj.obj, &strFn)).obj);
    }

    var true_obj = StarBool{ .value = true };
    var false_obj = StarBool{ .value = false };
    var true_str = StarStr{ .str = "True" };
    var false_str = StarStr{ .str = "False" };
};

pub const StarList = struct {
    items: std.ArrayList(*StarObj),
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
        },
    },

    pub fn init(allocator: Allocator, items: []const *StarObj) !*StarList {
        const self = try allocator.create(StarList);
        var list = std.ArrayList(*StarObj).empty;
        try list.appendSlice(allocator, items);
        self.* = .{ .items = list };
        try self.obj.attributes.put(allocator, "append", &(try StarBoundMethod.init(allocator, &self.obj, &append)).obj);
        try self.obj.attributes.put(allocator, dunder.iter, &(try StarBoundMethod.init(allocator, &self.obj, &iter)).obj);
        try self.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &self.obj, &strFn)).obj);
        return self;
    }

    pub fn getItem(self: *StarList, idx: i64) Error!*StarObj {
        const len: i64 = @intCast(self.items.items.len);
        var actual_idx = idx;
        if (actual_idx < 0) actual_idx += len;
        if (actual_idx < 0 or actual_idx >= len) return error.IndexOutOfRange;
        return self.items.items[@intCast(actual_idx)];
    }

    fn strFn(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);

        var buf = std.ArrayList(u8).empty;
        try buf.append(allocator, '[');

        for (self.items.items, 0..) |item, i| {
            if (i > 0) try buf.appendSlice(allocator, ", ");
            if (try item.getMethodDunder(.str)) |str_method| {
                const str_obj = try str_method.call(allocator, &.{});
                const str_val = try downCast(StarStr, str_obj);
                try buf.appendSlice(allocator, str_val.str);
            } else {
                try buf.appendSlice(allocator, "<object>");
            }
        }

        try buf.append(allocator, ']');
        const new = try StarStr.init(allocator, buf.items);
        return &new.obj;
    }

    fn append(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 1) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        try self.items.append(allocator, args[0]);
        return StarNone.instance;
    }

    fn iter(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarList = @fieldParentPtr("obj", self_obj);
        const list_iter = try StarListIter.init(allocator, self);
        return &list_iter.obj;
    }
};

pub const StarDict = struct {
    entries: std.AutoArrayHashMapUnmanaged(*StarObj, *StarObj),
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
        },
    },

    pub fn init(allocator: Allocator, pairs: []const *StarObj) !*StarDict {
        const self = try allocator.create(StarDict);
        var entries = std.AutoArrayHashMapUnmanaged(*StarObj, *StarObj).empty;
        var i: usize = 0;
        while (i < pairs.len) : (i += 2) {
            try entries.put(allocator, pairs[i], pairs[i + 1]);
        }
        self.* = .{ .entries = entries };
        try self.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &self.obj, &strFn)).obj);
        return self;
    }

    fn strFn(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarDict = @fieldParentPtr("obj", self_obj);

        var buf = std.ArrayList(u8).empty;
        try buf.append(allocator, '{');

        var idx: usize = 0;
        for (self.entries.keys(), self.entries.values()) |key, value| {
            if (idx > 0) try buf.appendSlice(allocator, ", ");
            if (try key.getMethodDunder(.str)) |str_method| {
                const str_obj = try str_method.call(allocator, &.{});
                const str_val = try downCast(StarStr, str_obj);
                try buf.appendSlice(allocator, str_val.str);
            } else {
                try buf.appendSlice(allocator, "<object>");
            }
            try buf.appendSlice(allocator, ": ");
            if (try value.getMethodDunder(.str)) |str_method| {
                const str_obj = try str_method.call(allocator, &.{});
                const str_val = try downCast(StarStr, str_obj);
                try buf.appendSlice(allocator, str_val.str);
            } else {
                try buf.appendSlice(allocator, "<object>");
            }
            idx += 1;
        }

        try buf.append(allocator, '}');
        const new = try StarStr.init(allocator, buf.items);
        return &new.obj;
    }
};

pub const StarRangeIter = struct {
    current: i64,
    stop: i64,
    step: i64,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
        },
    },

    pub fn init(allocator: Allocator, start: i64, stop: i64, step_val: i64) !*StarRangeIter {
        const self = try allocator.create(StarRangeIter);
        self.* = .{ .current = start, .stop = stop, .step = step_val };
        try self.obj.attributes.put(allocator, dunder.iter, &(try StarBoundMethod.init(allocator, &self.obj, &iter)).obj);
        try self.obj.attributes.put(allocator, dunder.next, &(try StarBoundMethod.init(allocator, &self.obj, &nextFn)).obj);
        try self.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &self.obj, &strFn)).obj);
        return self;
    }

    fn iter(_: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return self_obj;
    }

    fn nextFn(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarRangeIter = @fieldParentPtr("obj", self_obj);
        if (self.step > 0 and self.current >= self.stop) return StarStopIteration.instance;
        if (self.step < 0 and self.current <= self.stop) return StarStopIteration.instance;
        const val = try StarInt.init(allocator, @intCast(self.current));
        self.current += self.step;
        return &val.obj;
    }

    fn strFn(_: Allocator, _: *StarObj, args: []const *StarObj) Error!*StarObj {
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
        },
    },

    pub fn init(allocator: Allocator, list: *StarList) !*StarListIter {
        const self = try allocator.create(StarListIter);
        self.* = .{ .list = list, .index = 0 };

        try self.obj.attributes.put(allocator, dunder.iter, &(try StarBoundMethod.init(allocator, &self.obj, &iter)).obj);
        try self.obj.attributes.put(allocator, dunder.next, &(try StarBoundMethod.init(allocator, &self.obj, &nextFn)).obj);
        try self.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &self.obj, &strFn)).obj);
        return self;
    }

    fn iter(_: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return self_obj;
    }

    fn nextFn(_: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self: *StarListIter = @fieldParentPtr("obj", self_obj);
        if (self.index >= self.list.items.items.len) return StarStopIteration.instance;
        const val = self.list.items.items[self.index];
        self.index += 1;
        return val;
    }

    fn strFn(_: Allocator, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return &list_iter_str.obj;
    }

    var list_iter_str = StarStr{ .str = "<list_iterator>" };
};

pub const MethodFn = *const fn (allocator: Allocator, self: *StarObj, args: []const *StarObj) Error!*StarObj;

pub const StarBoundMethod = struct {
    bound_self: *StarObj,
    func: MethodFn,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
        },
    },

    fn strFn(_: Allocator, _: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        return &bound_method_str.obj;
    }

    var bound_method_str = StarStr{ .str = "<bound method>" };

    pub fn init(allocator: Allocator, bound_self: *StarObj, func: MethodFn) !*StarBoundMethod {
        const bm = try allocator.create(StarBoundMethod);
        bm.* = .{ .bound_self = bound_self, .func = func };
        return bm;
    }

    pub fn call(self: *StarBoundMethod, allocator: Allocator, args: []const *StarObj) Error!*StarObj {
        return self.func(allocator, self.bound_self, args);
    }
};

pub const StarModule = struct {
    name: []const u8,
    init_fn: StarFunc,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
        },
    },

    fn strFn(allocator: Allocator, self_obj: *StarObj, args: []const *StarObj) Error!*StarObj {
        if (args.len != 0) return Error.ArityMismatch;
        const self_module: *StarModule = @fieldParentPtr("obj", self_obj);

        const str_val = try std.fmt.allocPrint(allocator, "<module '{s}'>", .{self_module.name});
        const new = try StarStr.init(allocator, str_val);
        return &new.obj;
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
        try self.obj.attributes.put(allocator, dunder.str, &(try StarBoundMethod.init(allocator, &self.obj, &strFn)).obj);
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
        name: []const u8,

        pub fn basePtr(self: *Vtable) *StarObj {
            return @fieldParentPtr("vtable", self);
        }
    };

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

        if (downCast(StarStr, self)) |s| {
            return s.str.len > 0;
        } else |_| {}

        if (downCast(StarList, self)) |l| {
            return l.items.items.len > 0;
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
    try rt.stackPush(&func.obj);
    try rt.callFn(&func, empty_args[0..]);

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

    var a = try StarInt.init(gc, 10);
    var b = StarStr{ .str = "oops" };

    const add_method = try a.obj.getMethodDunder(.add) orelse return error.TestExpectedEqual;
    try std.testing.expectError(TypeError.TypeMismatch, add_method.call(gc, &.{&b.obj}));
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
    try rt.stackPush(&func.obj);
    try std.testing.expectError(RuntimeError.ArityMismatch, rt.callFn(&func, no_args[0..]));
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
    try rt.stackPush(&func.obj);
    try rt.callFn(&func, no_args[0..]);

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
    try rt.stackPush(&func.obj);
    try rt.callFn(&func, no_args[0..]);
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
    try rt.stackPush(&func.obj);
    try rt.callFn(&func, no_args[0..]);

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
    try std.testing.expectEqual(@as(u64, 42), captured_val.num);

    const result_val = rt.globals.get("result") orelse return error.TestUnexpectedResult;
    const result_int = try Runtime.downCast(Runtime.StarInt, result_val);
    try std.testing.expectEqual(@as(u64, 42), result_int.num);

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

        pub fn square(_: *Runtime, args: struct { n: u64 }) Error!?*StarObj {
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
