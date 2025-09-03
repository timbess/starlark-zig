const std = @import("std");
const Allocator = std.mem.Allocator;
const Tokenizer = @import("Tokenizer.zig");
const Token = Tokenizer.Token;
const Ast = @import("Ast.zig");
const Gc = @import("zig_libgc");

gpa: Allocator,
gc: Allocator,
/// Call stack for code/locals
frames: std.ArrayList(Frame) = .empty,
/// Value stack
stack: std.ArrayList(*StarObj) = .empty,

pub const InitOpts = struct {};

pub const LocalIdx = enum(u32) { _ };
pub const ConstIdx = enum(u32) { _ };
pub const FreeVarIdx = enum(u32) { _ };

pub const Arity = u32;

pub const BinOp = enum { add };

pub const Instruction = union(enum) {
    load: LocalIdx,
    load_const: ConstIdx,
    store: LocalIdx,
    binary_op: BinOp,
    call: Arity,
    ret: void,

    const Tag = std.meta.Tag(@This());
};

pub const TypeError = error{
    TypeMismatch,
};

pub const RuntimeError = error{
    AddOpUndefined,
    WrongArity,
    LocalOutOfRange,
    UninitializedLocal,
    ConstOutOfRange,
    StackUnderflow,
    CallUndefined,
    NoFrame,
    ReturnedOutsideFunction,
    AttributeMissing,
};

pub const Error = Allocator.Error || TypeError || RuntimeError;

const Frame = struct {
    func: *StarFunc,
    /// Segmented list, first 16 elements are stack allocated.
    locals: std.SegmentedList(*StarObj, 16),
    return_addr: **StarObj,
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
};
const Runtime = @This();

pub fn init(gpa: Allocator, opts: InitOpts) !Runtime {
    _ = opts;
    return Runtime{
        .gpa = gpa,
        .gc = Gc.allocator(),
    };
}

pub fn deinit(self: *Runtime) void {
    // free any frame locals still allocated
    for (self.frames.items) |*frame| {
        frame.locals.deinit(self.gc);
    }
    self.frames.deinit(self.gc);
    self.stack.deinit(self.gc);
}

/// Create and push a new frame for `func`. Copies args into locals[0..arity).
/// ret_addr is the caller's next-pc (an index into the caller's code).
fn call_fn(self: *Runtime, func: *StarFunc, args: []*StarObj) Error!void {
    if (args.len != func.arity) return RuntimeError.WrongArity;
    std.debug.assert(func.frame_size >= func.arity);

    const return_addr = try self.stackPushOne();

    // create frame value and push
    var frame = Frame{
        .func = func,
        .locals = .{},
        .return_addr = return_addr,
        .sp_base = self.stack.items.len,
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

inline fn stackPop(self: *Runtime) Error!*StarObj {
    return self.stack.pop() orelse return Error.StackUnderflow;
}

inline fn stackPush(self: *Runtime, obj: *StarObj) Error!void {
    try self.stack.append(self.gc, obj);
}

inline fn stackPushOne(self: *Runtime) Error!**StarObj {
    return self.stack.addOne(self.gc);
}

// Move interpreter forward one step.
pub fn step(self: *Runtime) Error!void {
    if (self.frames.items.len == 0) return RuntimeError.NoFrame;
    var frame = &self.frames.items[self.frames.items.len - 1];
    const active_code = frame.func.code;

    if (frame.pc >= active_code.len) return RuntimeError.NoFrame;

    const instr = active_code[frame.pc];

    defer frame.pc += 1;
    switch (instr) {
        .load => |idx| {
            const val = try frame.readLocal(idx);
            try self.stackPush(val);
        },
        .load_const => |cidx| {
            const const_obj = try frame.readConst(cidx);
            try self.stackPush(const_obj);
        },
        .store => |idx| {
            const val = try self.stackPop();
            try frame.writeLocal(idx, val);
        },
        .binary_op => |op| {
            const rhs = try self.stackPop();
            const lhs = try self.stackPop();

            const result: *StarObj = switch (op) {
                .add => try lhs.addOp(rhs),
            };

            try self.stackPush(result);
        },
        .call => |arity_u| {
            const arity = @as(usize, arity_u);

            const sp = self.stack.items.len;
            if (sp < arity + 1) return RuntimeError.StackUnderflow;

            // layout: [..., func_obj, arg1, arg2, ... argN]
            const args_start = sp - arity;
            const func_index = args_start - 1;
            const func_obj = self.stack.items[func_index];

            const args_slice: []*StarObj = self.stack.items[args_start .. args_start + arity];

            // Try interpreted function first: downCast StarFunc
            const maybe_func = downCast(StarFunc, func_obj) catch |e| switch (e) {
                TypeError.TypeMismatch => null,
                else => return e,
            };

            if (maybe_func) |fptr| {
                try self.call_fn(fptr, args_slice);
            } else {
                return RuntimeError.CallUndefined;
            }
        },
        .ret => {
            var frame_val = self.frames.pop() orelse return error.ReturnedOutsideFunction;
            frame_val.locals.deinit(self.gpa);
            const return_value = try self.stackPop();
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
    if (self.frames.items.len == 0) return error.NoFrame;
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

pub const StarInt = struct {
    num: u64,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .add_op = &add_op,
        },
    },

    pub fn init(num: u64) !*StarInt {
        const self = try Gc.allocator().create(StarInt);
        self.* = .{ .num = num };
        return self;
    }

    fn add_op(vt: *StarObj.Vtable, other: *StarObj) Error!*StarObj {
        // Find the StarInt that owns this vtable:
        const owner_starobj = vt.basePtr(); // *StarObj
        const self_int: *StarInt = @fieldParentPtr("obj", owner_starobj);

        const o = try downCast(StarInt, other);

        const new = try StarInt.init(self_int.num + o.num);
        return &new.obj;
    }
};

pub const StarFloat = struct {
    num: f64,
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
            .add_op = &add_op,
        },
    },

    pub fn init(num: f64) !*StarFloat {
        const self = try Gc.allocator().create(StarFloat);
        self.* = .{ .num = num };
        return self;
    }

    fn add_op(vt: *StarObj.Vtable, other: *StarObj) Error!*StarObj {
        // Find the StarFloat that owns this vtable:
        const owner_starobj = vt.basePtr(); // *StarObj
        const self_int: *StarFloat = @fieldParentPtr("obj", owner_starobj);

        const o = try downCast(StarFloat, other);

        const new = try StarFloat.init(self_int.num + o.num);
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

    pub fn init_dupe(str: []const u8) !*StarStr {
        const allocator = Gc.allocator();
        const self = try allocator.create(StarStr);
        self.* = .{ .str = try allocator.dupe(u8, str) };
        return self;
    }
};

test "Basic StarObject usage works" {
    var a = try StarInt.init(5);
    const a_obj = &a.obj;

    var b = try StarInt.init(6);
    const b_obj = &b.obj;

    const c_obj = try a_obj.addOp(b_obj);
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

pub const StarFunc = struct {
    code: []const Instruction,
    consts: []const *StarObj, // function constants (literals, etc.)
    arity: usize,
    /// number of local slots
    frame_size: usize,
    /// All local var names
    names_local: []const []const u8,
    /// All free var names
    names_free: []const []const u8,

    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
        },
    },
};

pub const StarNone = struct {
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
        },
    },

    var none_obj = StarNone{};
    pub const instance: *StarObj = &none_obj.obj;
};

pub const StarModule = struct {
    obj: StarObj = .{
        .vtable = StarObj.Vtable{
            .name = @typeName(@This()),
        },
    },

    pub fn init() !*StarModule {
        const alloc = Gc.allocator();
        const self = try alloc.create(StarModule);
        self.* = .{};
        return self;
    }
};

pub const StarObj = struct {
    vtable: Vtable,
    attributes: std.StringHashMapUnmanaged(*StarObj) = .empty,

    pub const Vtable = struct {
        name: []const u8,
        // optional add operator
        add_op: ?*const fn (*Vtable, other: *StarObj) Error!*StarObj = null,

        pub fn basePtr(self: *Vtable) *StarObj {
            return @fieldParentPtr("vtable", self);
        }
    };

    pub fn setAttrConcrete(self: *StarObj, name: *StarStr, value: *StarObj) Allocator.Error!void {
        const gca = Gc.allocator();
        try self.attributes.put(gca, name.str, value);
    }

    pub fn setAttr(self: *StarObj, name: *StarObj, value: *StarObj) Error!void {
        const gca = Gc.allocator();
        const n = try downCast(StarStr, name);
        try self.attributes.put(gca, n.str, value);
    }

    pub fn getAttrConcrete(self: *StarObj, name: *StarStr) Error!*StarObj {
        return self.attributes.get(name.str) orelse return error.AttributeMissing;
    }

    pub fn getAttr(self: *StarObj, name: *StarObj) Error!*StarObj {
        const n = try downCast(StarStr, name);
        return self.attributes.get(n.str) orelse return error.AttributeMissing;
    }

    pub fn addOp(self: *StarObj, other: *StarObj) Error!*StarObj {
        if (self.vtable.add_op) |f| {
            return try f(&self.vtable, other);
        }
        return RuntimeError.AddOpUndefined;
    }
};

test "Runtime executes add function" {
    const gpa = std.testing.allocator;

    var code = std.ArrayList(Instruction).empty;
    defer code.deinit(gpa);

    try code.append(gpa, .{ .load_const = @enumFromInt(0) });
    try code.append(gpa, .{ .load_const = @enumFromInt(1) });
    try code.append(gpa, .{ .binary_op = .add });
    try code.append(gpa, .{ .ret = {} });

    var a = try StarInt.init(5);
    var b = try StarInt.init(6);
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
    var rt = try Runtime.init(gpa, .{});
    defer rt.deinit();

    var empty_args: [0]*StarObj = .{};
    try rt.call_fn(&func, empty_args[0..]);

    // Step until frame stack is empty
    while (rt.frames.items.len > 0) {
        try rt.step();
    }

    // At the end, top of stack should be the result
    try std.testing.expectEqual(1, rt.stack.items.len);
    const result_obj = rt.stack.items[0];
    const result = try downCast(StarInt, result_obj);
    try std.testing.expectEqual(11, result.num);
}

test "StarObj attributes work" {
    var obj = try StarInt.init(42);
    var key = StarStr{ .str = "answer" };
    var val = try StarInt.init(99);

    try obj.obj.setAttr(&key.obj, &val.obj);
    const got = try obj.obj.getAttr(&key.obj);
    const casted = try downCast(StarInt, got);

    try std.testing.expectEqual(99, casted.num);
}

test "StarObj.getAttr fails on wrong type key" {
    var obj = try StarInt.init(1);
    var not_a_str = try StarInt.init(2);

    try std.testing.expectError(TypeError.TypeMismatch, obj.obj.getAttr(&not_a_str.obj));
}

test "StarInt addOp fails with wrong type" {
    var a = try StarInt.init(10);
    var b = StarStr{ .str = "oops" };

    try std.testing.expectError(TypeError.TypeMismatch, a.obj.addOp(&b.obj));
}

test "downCast fails for wrong vtable" {
    var int_obj = try StarInt.init(123);
    var str_obj = StarStr{ .str = "abc" };

    try std.testing.expectError(TypeError.TypeMismatch, downCast(StarStr, &int_obj.obj));
    try std.testing.expectError(TypeError.TypeMismatch, downCast(StarInt, &str_obj.obj));
}

test "Runtime detects wrong arity" {
    const gpa = std.testing.allocator;

    var func = StarFunc{
        .code = &[_]Instruction{},
        .consts = &[_]*StarObj{},
        .arity = 1,
        .frame_size = 1,
        .names_free = &.{},
        .names_local = &.{},
    };

    var rt = try Runtime.init(gpa, .{});
    defer rt.deinit();

    var no_args: [0]*StarObj = .{};
    try std.testing.expectError(RuntimeError.WrongArity, rt.call_fn(&func, no_args[0..]));
}

test "Instruction.load_const out of range" {
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

    var rt = try Runtime.init(gpa, .{});
    defer rt.deinit();

    var no_args: [0]*StarObj = .{};
    try rt.call_fn(&func, no_args[0..]);

    try std.testing.expectError(RuntimeError.ConstOutOfRange, rt.step());
}

test "Instruction.store and load locals" {
    const gpa = std.testing.allocator;

    var code = std.ArrayList(Instruction).empty;
    defer code.deinit(gpa);

    // store local[0], then load it back and return
    try code.append(gpa, .{ .load_const = @enumFromInt(0) });
    try code.append(gpa, .{ .store = @enumFromInt(0) });
    try code.append(gpa, .{ .load = @enumFromInt(0) });
    try code.append(gpa, .{ .ret = {} });

    var c = try StarInt.init(77);
    const consts = [_]*StarObj{&c.obj};

    var func = StarFunc{
        .code = code.items,
        .consts = &consts,
        .arity = 0,
        .frame_size = 1,
        .names_free = &.{},
        .names_local = &.{"a"},
    };

    var rt = try Runtime.init(gpa, .{});
    defer rt.deinit();

    var no_args: [0]*StarObj = .{};
    try rt.call_fn(&func, no_args[0..]);

    while (rt.frames.items.len > 0) {
        try rt.step();
    }

    const result_obj = rt.stack.items[0];
    const result = try downCast(StarInt, result_obj);
    try std.testing.expectEqual(77, result.num);
}

test "Instruction.call fails on non-function" {
    const gpa = std.testing.allocator;

    var code = [_]Instruction{.{ .call = 0 }};

    var fake_func_obj = try StarInt.init(999);

    var func = StarFunc{
        .code = &code,
        .consts = &[_]*StarObj{&fake_func_obj.obj},
        .arity = 0,
        .frame_size = 0,
        .names_free = &.{},
        .names_local = &.{},
    };

    var rt = try Runtime.init(gpa, .{});
    defer rt.deinit();

    var no_args: [0]*StarObj = .{};
    try rt.call_fn(&func, no_args[0..]);

    // push the const onto stack manually
    try rt.stackPush(&fake_func_obj.obj);

    try std.testing.expectError(RuntimeError.CallUndefined, rt.step());
}
