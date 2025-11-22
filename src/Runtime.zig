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

const knobs = struct {
    const locals_initial_capacity: usize = 16;
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

pub const BinOp = enum { add, sub, mul };

pub const Instruction = union(enum) {
    load: LocalIdx,
    load_const: ConstIdx,
    load_global: GlobalIdx,
    load_free: FreeVarIdx,
    store: LocalIdx,
    store_global: GlobalIdx,
    binary_op: BinOp,
    call: Arity,
    ret: void,
    ret_none: void,
    make_closure: ConstIdx,
    /// Slice of the identifier from the original source code.
    get_attr: []const u8,

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
};

pub const Error = Allocator.Error || std.Io.Writer.Error || TypeError || RuntimeError;

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

// Step until frame stack is empty
pub fn stepUntilDone(self: *Runtime) Error!void {
    while (self.frames.len > 0) {
        try self.step();
    }
}

// Move interpreter forward one step.
pub fn step(self: *Runtime) Error!void {
    if (self.frames.len == 0) return RuntimeError.FrameMissing;
    var frame: *Frame = self.frames.at(self.frames.len - 1);
    const active_code = frame.func.code;

    if (frame.pc >= active_code.len) return RuntimeError.FrameNoReturn;

    const instr = active_code[frame.pc];
    const pc = frame.pc;

    defer {
        frame.pc += 1;
    }

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

            const result: *StarObj = switch (op) {
                .add => try lhs.addOp(self.gc, rhs),
                .sub => try lhs.subOp(self.gc, rhs),
                .mul => try lhs.mulOp(self.gc, rhs),
            };

            try self.stackPush(result);
        },
        .get_attr => |attr| {
            const obj = try self.stackPop();
            const val = obj.attributes.get(attr) orelse return Error.AttributeMissing;
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
                    try self.stackPush(try f(self, args_slice));
                } else try self.callFn(fptr, args_slice);
            } else {
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
            };

            try self.stackPush(&closure_func.obj);
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
        },
    }
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

pub fn execModule(self: *Runtime, module: *const Compiler.Module) !void {
    const mod = try StarModule.fromCompiledModule(self.gc, module);

    std.debug.assert(mod.init_fn.arity == 0);
    try self.stackPush(&mod.init_fn.obj);
    try self.callFn(&mod.init_fn, &.{});
    try self.stepUntilDone();
    // Module inits return None
    _ = try self.stackPop();
}

pub const StarInt = struct {
    num: u64,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .add_op = &addOp,
            .sub_op = &subOp,
            .mul_op = &mulOp,
            .str = &str,
        },
    },

    pub fn init(allocator: Allocator, num: u64) !*StarInt {
        const self = try allocator.create(StarInt);
        self.* = .{ .num = num };
        return self;
    }

    fn addOp(vt: *StarObj.Vtable, allocator: Allocator, other: *StarObj) Error!*StarObj {
        // Find the StarInt that owns this vtable:
        const owner_starobj = vt.basePtr(); // *StarObj
        const self_int: *StarInt = @fieldParentPtr("obj", owner_starobj);

        const o = try downCast(StarInt, other);

        const new = try StarInt.init(allocator, self_int.num + o.num);
        return &new.obj;
    }

    fn subOp(vt: *StarObj.Vtable, allocator: Allocator, other: *StarObj) Error!*StarObj {
        const owner_starobj = vt.basePtr();
        const self_int: *StarInt = @fieldParentPtr("obj", owner_starobj);

        const o = try downCast(StarInt, other);

        const new = try StarInt.init(allocator, self_int.num - o.num);
        return &new.obj;
    }

    fn mulOp(vt: *StarObj.Vtable, allocator: Allocator, other: *StarObj) Error!*StarObj {
        const owner_starobj = vt.basePtr();
        const self_int: *StarInt = @fieldParentPtr("obj", owner_starobj);

        const o = try downCast(StarInt, other);

        const new = try StarInt.init(allocator, self_int.num * o.num);
        return &new.obj;
    }

    fn str(vt: *StarObj.Vtable, allocator: Allocator) Error!*StarObj {
        const owner_starobj = vt.basePtr();
        const self_int: *StarInt = @fieldParentPtr("obj", owner_starobj);

        const str_val = try std.fmt.allocPrint(allocator, "{d}", .{self_int.num});
        const new = try allocator.create(StarStr);
        new.* = .{ .str = str_val };
        return &new.obj;
    }
};

pub const StarFloat = struct {
    num: f64,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .add_op = &addOp,
            .sub_op = &subOp,
            .mul_op = &mulOp,
            .str = &str,
        },
    },

    pub fn init(allocator: Allocator, num: f64) !*StarFloat {
        const self = try allocator.create(StarFloat);
        self.* = .{ .num = num };
        return self;
    }

    fn addOp(vt: *StarObj.Vtable, allocator: Allocator, other: *StarObj) Error!*StarObj {
        // Find the StarFloat that owns this vtable:
        const owner_starobj = vt.basePtr(); // *StarObj
        const self_int: *StarFloat = @fieldParentPtr("obj", owner_starobj);

        const o = try downCast(StarFloat, other);

        const new = try StarFloat.init(allocator, self_int.num + o.num);
        return &new.obj;
    }

    fn subOp(vt: *StarObj.Vtable, allocator: Allocator, other: *StarObj) Error!*StarObj {
        const owner_starobj = vt.basePtr();
        const self_float: *StarFloat = @fieldParentPtr("obj", owner_starobj);

        const o = try downCast(StarFloat, other);

        const new = try StarFloat.init(allocator, self_float.num - o.num);
        return &new.obj;
    }

    fn mulOp(vt: *StarObj.Vtable, allocator: Allocator, other: *StarObj) Error!*StarObj {
        const owner_starobj = vt.basePtr();
        const self_float: *StarFloat = @fieldParentPtr("obj", owner_starobj);

        const o = try downCast(StarFloat, other);

        const new = try StarFloat.init(allocator, self_float.num * o.num);
        return &new.obj;
    }

    fn str(vt: *StarObj.Vtable, allocator: Allocator) Error!*StarObj {
        const owner_starobj = vt.basePtr();
        const self_float: *StarFloat = @fieldParentPtr("obj", owner_starobj);

        const str_val = try std.fmt.allocPrint(allocator, "{d}", .{self_float.num});
        const new = try allocator.create(StarStr);
        new.* = .{ .str = str_val };
        return &new.obj;
    }
};

pub const StarStr = struct {
    str: []const u8,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .str = &strFn,
        },
    },

    pub fn init_dupe(allocator: Allocator, str: []const u8) !*StarStr {
        const self = try allocator.create(StarStr);
        self.* = .{ .str = try allocator.dupe(u8, str) };
        return self;
    }

    fn strFn(vt: *StarObj.Vtable, allocator: Allocator) Error!*StarObj {
        _ = allocator;
        const owner_starobj = vt.basePtr();
        return owner_starobj;
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

    const c_obj = try a_obj.addOp(gc, b_obj);
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

    native_fn: ?*const fn (rt: *Runtime, args: []*StarObj) Error!*StarObj = null,

    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .str = &str,
        },
    },

    fn str(vt: *StarObj.Vtable, allocator: Allocator) Error!*StarObj {
        _ = vt;
        _ = allocator;
        return &func_str.obj;
    }

    var func_str = StarStr{ .str = "<function>" };

    pub fn fromNative(
        // TODO: kwargs
        Args: type,
        comptime native: fn (rt: *Runtime, args: Args) Error!?*StarObj,
    ) StarFunc {
        // Handle variadic args: []*StarObj
        if (Args == []*StarObj) {
            const native_fn = struct {
                pub fn wrapper(rt: *Runtime, args: []*StarObj) Error!*StarObj {
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
                    pub fn wrapper(rt: *Runtime, args: []*StarObj) Error!*StarObj {
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
            .str = &str,
        },
    },

    fn str(vt: *StarObj.Vtable, allocator: Allocator) Error!*StarObj {
        _ = vt;
        _ = allocator;
        return &none_str.obj;
    }

    var none_obj = StarNone{};
    pub const instance: *StarObj = &none_obj.obj;

    var none_str = StarStr{ .str = "None" };
};

pub const StarModule = struct {
    name: []const u8,
    init_fn: StarFunc,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .str = &str,
        },
    },

    fn str(vt: *StarObj.Vtable, allocator: Allocator) Error!*StarObj {
        const owner_starobj = vt.basePtr();
        const self_module: *StarModule = @fieldParentPtr("obj", owner_starobj);

        const str_val = try std.fmt.allocPrint(allocator, "<module '{s}'>", .{self_module.name});
        const new = try allocator.create(StarStr);
        new.* = .{ .str = str_val };
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
        });
        return mod;
    }
};

pub const StarObj = struct {
    vtable: Vtable,
    attributes: std.StringHashMapUnmanaged(*StarObj) = .empty,

    pub const Vtable = struct {
        name: []const u8,
        // optional add operator
        add_op: ?*const fn (*Vtable, allocator: Allocator, other: *StarObj) Error!*StarObj = null,
        // optional sub operator
        sub_op: ?*const fn (*Vtable, allocator: Allocator, other: *StarObj) Error!*StarObj = null,
        // optional mul operator
        mul_op: ?*const fn (*Vtable, allocator: Allocator, other: *StarObj) Error!*StarObj = null,
        // optional str method for string representation
        str: ?*const fn (*Vtable, allocator: Allocator) Error!*StarObj = null,

        pub fn basePtr(self: *Vtable) *StarObj {
            return @fieldParentPtr("vtable", self);
        }
    };

    pub fn setAttrConcrete(self: *StarObj, allocator: Allocator, name: *StarStr, value: *StarObj) Allocator.Error!void {
        try self.attributes.put(allocator, name.str, value);
    }

    pub fn setAttr(self: *StarObj, allocator: Allocator, name: *StarObj, value: *StarObj) Error!void {
        const n = try downCast(StarStr, name);
        try self.attributes.put(allocator, n.str, value);
    }

    pub fn getAttrConcrete(self: *StarObj, name: *StarStr) Error!*StarObj {
        return self.attributes.get(name.str) orelse return error.AttributeMissing;
    }

    pub fn getAttr(self: *StarObj, name: *StarObj) Error!*StarObj {
        const n = try downCast(StarStr, name);
        return self.attributes.get(n.str) orelse return error.AttributeMissing;
    }

    pub fn addOp(self: *StarObj, allocator: Allocator, other: *StarObj) Error!*StarObj {
        if (self.vtable.add_op) |f| {
            return try f(&self.vtable, allocator, other);
        }
        return RuntimeError.AddOpUndefined;
    }

    pub fn subOp(self: *StarObj, allocator: Allocator, other: *StarObj) Error!*StarObj {
        if (self.vtable.sub_op) |f| {
            return try f(&self.vtable, allocator, other);
        }
        return RuntimeError.SubOpUndefined;
    }

    pub fn mulOp(self: *StarObj, allocator: Allocator, other: *StarObj) Error!*StarObj {
        if (self.vtable.mul_op) |f| {
            return try f(&self.vtable, allocator, other);
        }
        return RuntimeError.MulOpUndefined;
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
    const result = try downCast(StarInt, result_obj);
    try std.testing.expectEqual(11, result.num);
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

test "StarInt addOp fails with wrong type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    var a = try StarInt.init(gc, 10);
    var b = StarStr{ .str = "oops" };

    try std.testing.expectError(TypeError.TypeMismatch, a.obj.addOp(gc, &b.obj));
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
    const result = try downCast(StarInt, result_obj);
    try std.testing.expectEqual(77, result.num);
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
