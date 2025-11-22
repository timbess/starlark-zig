/// This isn't a great compiler, I will circle back to rewriting it once the more important
/// bits have come together.
const std = @import("std");
const Ast = @import("Ast.zig");
const Token = @import("Tokenizer.zig").Token;
const Runtime = @import("Runtime.zig");
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.compiler);

pub const Error = error{
    AstEmpty,
    AstInvalid,
    LiteralInvalid,
    BigIntUnsupported,
    VariableUndefined,
    VariableRedefinition,
} || std.mem.Allocator.Error;

pub const Module = struct {
    module_name: []const u8,
    code: []Runtime.Instruction,
    constants: []*Runtime.StarObj,
    global_names: [][]const u8,
};

pub const Function = struct {
    code: []Runtime.Instruction,
    constants: []*Runtime.StarObj,
    local_names: [][]const u8,
    free_names: [][]const u8,
    closure_cells: []*Runtime.StarObj,
};

/// Tweak these to optimize the compiler later.
const knobs = struct {
    const string_avg_count: usize = 128;
    const string_avg_len: usize = 4096;
    const ast_depth: usize = 4096;
    const fn_binding_count: usize = 16;
};

const StringInterner = struct {
    strings: std.StringHashMap([]const u8),
    allocator: Allocator,

    fn init(allocator: Allocator) !StringInterner {
        var strings = std.StringHashMap([]const u8).init(allocator);
        try strings.ensureTotalCapacity(knobs.string_avg_count);
        return .{
            .strings = strings,
            .allocator = allocator,
        };
    }

    fn deinit(self: *StringInterner) void {
        self.strings.deinit();
    }

    inline fn intern(self: *StringInterner, str: []const u8) ![]const u8 {
        const gop = try self.strings.getOrPut(str);
        if (!gop.found_existing) {
            const owned = try self.allocator.dupe(u8, str);
            gop.key_ptr.* = owned;
            gop.value_ptr.* = owned;
        }
        return gop.value_ptr.*;
    }
};

const VarInfo = struct {
    kind: enum(u8) { local, global, free },
    index: u32,
};

const Scope = struct {
    parent: ?*Scope,
    bindings: std.StringHashMap(VarInfo),

    fn init(allocator: Allocator, parent: ?*Scope) !Scope {
        var bindings = std.StringHashMap(VarInfo).init(allocator);
        try bindings.ensureTotalCapacity(knobs.fn_binding_count);
        return .{
            .parent = parent,
            .bindings = bindings,
        };
    }

    fn deinit(self: *Scope) void {
        self.bindings.deinit();
    }

    inline fn define(self: *Scope, name: []const u8, info: VarInfo) !void {
        try self.bindings.put(name, info);
    }

    inline fn resolveLocal(self: *Scope, name: []const u8) ?VarInfo {
        return self.bindings.get(name);
    }

    // Iterative resolve instead of recursive
    fn resolve(self: *Scope, name: []const u8) ?VarInfo {
        var current: ?*Scope = self;
        while (current) |scope| {
            if (scope.bindings.get(name)) |info| return info;
            current = scope.parent;
        }
        return null;
    }
};

const CodeBuilder = struct {
    gc: Allocator,
    code: std.ArrayList(Runtime.Instruction) = .empty,
    constants: std.ArrayList(*Runtime.StarObj) = .empty,
    const_map: std.AutoHashMapUnmanaged(*Runtime.StarObj, Runtime.ConstIdx) = .empty,
    interner: *StringInterner,

    fn init(gc: Allocator, interner: *StringInterner) CodeBuilder {
        return .{
            .gc = gc,
            .interner = interner,
        };
    }

    inline fn emit(self: *@This(), instruction: Runtime.Instruction) !void {
        try self.code.append(self.gc, instruction);
    }

    inline fn addConst(self: *@This(), obj: *Runtime.StarObj) !Runtime.ConstIdx {
        const gop = try self.const_map.getOrPut(self.gc, obj);
        if (!gop.found_existing) {
            const idx: Runtime.ConstIdx = @enumFromInt(self.constants.items.len);
            try self.constants.append(self.gc, obj);
            gop.value_ptr.* = idx;
        }
        return gop.value_ptr.*;
    }

    inline fn internString(self: *@This(), str: []const u8) ![]const u8 {
        return self.interner.intern(str);
    }
};

const FunctionCompiler = struct {
    base: CodeBuilder,
    local_names: std.ArrayList([]const u8) = .empty,
    free_names: std.ArrayList([]const u8) = .empty,
    free_map: std.StringHashMapUnmanaged(Runtime.FreeVarIdx) = .empty,
    global_names: std.ArrayList([]const u8) = .empty,
    global_map: std.StringHashMapUnmanaged(Runtime.GlobalIdx) = .empty,
    scope: Scope,
    arity: usize,
    arena: Allocator,

    fn init(gc: Allocator, arena: Allocator, interner: *StringInterner, parent_scope: ?*Scope) !FunctionCompiler {
        return .{
            .base = CodeBuilder.init(gc, interner),
            .scope = try Scope.init(arena, parent_scope),
            .arity = 0,
            .arena = arena,
        };
    }

    fn deinit(self: *FunctionCompiler) void {
        self.scope.deinit();
    }

    inline fn defineLocal(self: *FunctionCompiler, name: []const u8) !Runtime.LocalIdx {
        const interned = try self.base.internString(name);
        const idx: u32 = @intCast(self.local_names.items.len);
        try self.local_names.append(self.base.gc, interned);
        try self.scope.define(interned, .{ .kind = .local, .index = idx });
        return @enumFromInt(idx);
    }

    inline fn defineFreeVar(self: *FunctionCompiler, name: []const u8) !Runtime.FreeVarIdx {
        const interned = try self.base.internString(name);
        const gop = try self.free_map.getOrPut(self.arena, interned);
        if (!gop.found_existing) {
            const idx: Runtime.FreeVarIdx = @enumFromInt(self.free_names.items.len);
            try self.free_names.append(self.base.gc, interned);
            gop.value_ptr.* = idx;
        }
        return gop.value_ptr.*;
    }

    inline fn defineGlobal(self: *FunctionCompiler, name: []const u8) !Runtime.GlobalIdx {
        const interned = try self.base.internString(name);
        const gop = try self.global_map.getOrPut(self.arena, interned);
        if (!gop.found_existing) {
            const idx: Runtime.GlobalIdx = @enumFromInt(self.global_names.items.len);
            try self.global_names.append(self.base.gc, interned);
            gop.value_ptr.* = idx;
        }
        return gop.value_ptr.*;
    }

    fn resolveVar(self: *FunctionCompiler, name: []const u8) !VarInfo {
        // 1. Check local scope (this function's locals)
        if (self.scope.resolveLocal(name)) |info| {
            return info;
        }

        // 2. Walk up parent scopes to determine if it's a free var or global
        var current_parent: ?*Scope = self.scope.parent;
        while (current_parent) |parent| {
            if (parent.resolveLocal(name)) |_| {
                // Found in a parent scope
                if (parent.parent == null) {
                    // Parent is the module scope (root) - this is a global
                    const global_idx = try self.defineGlobal(name);
                    return .{ .kind = .global, .index = @intFromEnum(global_idx) };
                } else {
                    // Parent is an enclosing function scope - it's a free variable
                    const free_idx = try self.defineFreeVar(name);
                    return .{ .kind = .free, .index = @intFromEnum(free_idx) };
                }
            }
            current_parent = parent.parent;
        }

        // 3. Not found in any scope - treat as builtin (will be looked up at runtime)
        const global_idx = try self.defineGlobal(name);
        return .{ .kind = .global, .index = @intFromEnum(global_idx) };
    }
};

const ModuleCompiler = struct {
    base: CodeBuilder,
    global_names: std.ArrayList([]const u8) = .empty,
    global_map: std.StringHashMapUnmanaged(u32) = .empty,
    scope: Scope,
    arena: Allocator,

    fn init(gc: Allocator, arena: Allocator, interner: *StringInterner) !ModuleCompiler {
        return .{
            .base = CodeBuilder.init(gc, interner),
            .scope = try Scope.init(arena, null),
            .arena = arena,
        };
    }

    fn deinit(self: *ModuleCompiler) void {
        self.scope.deinit();
    }

    inline fn defineGlobal(self: *ModuleCompiler, name: []const u8) !Runtime.GlobalIdx {
        const interned = try self.base.internString(name);
        const gop = try self.global_map.getOrPut(self.arena, interned);
        if (!gop.found_existing) {
            const idx: u32 = @intCast(self.global_names.items.len);
            try self.global_names.append(self.base.gc, interned);
            gop.value_ptr.* = idx;
            try self.scope.define(interned, .{ .kind = .global, .index = idx });
        }
        return @enumFromInt(gop.value_ptr.*);
    }

    inline fn resolveGlobal(self: *ModuleCompiler, name: []const u8) !Runtime.GlobalIdx {
        if (self.global_map.get(name)) |idx| {
            return @enumFromInt(idx);
        }
        return error.VariableUndefined;
    }
};

inline fn compileLiteral(
    builder: *CodeBuilder,
    ast: *const Ast,
    source: [:0]const u8,
    node: Ast.Node,
) Error!*Runtime.StarObj {
    const token = ast.tokens.get(@intFromEnum(node.main_token));
    switch (token.tag) {
        .number_literal => {
            const number_src = source[token.loc.start..token.loc.end];
            const number_res = std.zig.parseNumberLiteral(number_src);
            return switch (number_res) {
                .int => |n| &(try Runtime.StarInt.init(builder.gc, n)).obj,
                .big_int => error.BigIntUnsupported,
                .failure => error.LiteralInvalid,
                .float => {
                    const n = std.fmt.parseFloat(f64, number_src) catch return error.LiteralInvalid;
                    return &(try Runtime.StarFloat.init(builder.gc, n)).obj;
                },
            };
        },
        .string => {
            const string_literal = source[token.loc.start..token.loc.end];
            // Parse string literal directly with gc allocator
            const str_parsed = std.zig.string_literal.parseAlloc(builder.gc, string_literal) catch return error.LiteralInvalid;
            defer builder.gc.free(str_parsed);
            const interned = try builder.internString(str_parsed);
            const str_star = try builder.gc.create(Runtime.StarStr);
            str_star.* = .{ .str = interned };
            return &str_star.obj;
        },
        else => return error.LiteralInvalid,
    }
}

fn compileExpr(
    ast: *const Ast,
    source: [:0]const u8,
    builder: *CodeBuilder,
    scope: *Scope,
    expr_idx: Ast.Node.Index,
    func_compiler: ?*FunctionCompiler,
    module_compiler: ?*ModuleCompiler,
) Error!void {
    const ExprWork = struct {
        kind: enum {
            compile_expr,
            emit_store,
            emit_call,
            emit_binop,
            emit_getattr,
        },
        node_idx: Ast.Node.Index,
        extra: u32 = 0, // Used for arity in calls, binop type
    };

    var work_buffer: [256]ExprWork = undefined;
    var work_stack = std.ArrayList(ExprWork).initBuffer(&work_buffer);

    try work_stack.appendBounded(.{ .kind = .compile_expr, .node_idx = expr_idx });

    while (work_stack.pop()) |work| {
        switch (work.kind) {
            .compile_expr => {
                const node = ast.nodes.get(@intFromEnum(work.node_idx));

                switch (node.data) {
                    .literal => {
                        const obj = try compileLiteral(builder, ast, source, node);
                        const idx = try builder.addConst(obj);
                        try builder.emit(.{ .load_const = idx });
                    },
                    .identifier => {
                        const ident_tok = ast.tokens.get(@intFromEnum(node.main_token));
                        const ident_src = source[ident_tok.loc.start..ident_tok.loc.end];

                        const var_info = if (func_compiler) |fc|
                            try fc.resolveVar(ident_src)
                        else if (module_compiler) |mc| blk: {
                            // At module level, auto-define undefined identifiers as globals
                            if (scope.resolve(ident_src)) |info| {
                                break :blk info;
                            } else {
                                const global_idx = try mc.defineGlobal(ident_src);
                                break :blk VarInfo{ .kind = .global, .index = @intFromEnum(global_idx) };
                            }
                        } else {
                            return error.VariableUndefined;
                        };

                        switch (var_info.kind) {
                            .local => try builder.emit(.{ .load = @enumFromInt(var_info.index) }),
                            .global => try builder.emit(.{ .load_global = @enumFromInt(var_info.index) }),
                            .free => try builder.emit(.{ .load_free = @enumFromInt(var_info.index) }),
                        }
                    },
                    .call => |c| {
                        const args_node = ast.nodes.get(@intFromEnum(c.args));
                        if (args_node.data != .call_args) return error.AstInvalid;

                        const arity: u32 = @intCast(args_node.data.call_args.args.len);

                        try work_stack.appendBounded(.{ .kind = .emit_call, .node_idx = work.node_idx, .extra = arity });

                        var i: usize = args_node.data.call_args.args.len;
                        while (i > 0) {
                            i -= 1;
                            try work_stack.appendBounded(.{ .kind = .compile_expr, .node_idx = args_node.data.call_args.args[i] });
                        }

                        try work_stack.appendBounded(.{ .kind = .compile_expr, .node_idx = c.func });
                    },
                    .add, .sub, .mul => |binop| {
                        switch (std.meta.activeTag(node.data)) {
                            inline else => |t| {
                                // We already know it's add/sub/mul
                                if (!@hasField(Runtime.BinOp, @tagName(t))) unreachable;
                                try work_stack.appendBounded(.{ .kind = .emit_binop, .node_idx = work.node_idx, .extra = @intFromEnum(@field(Runtime.BinOp, @tagName(t))) });
                            },
                        }
                        try work_stack.appendBounded(.{ .kind = .compile_expr, .node_idx = binop.rhs });
                        try work_stack.appendBounded(.{ .kind = .compile_expr, .node_idx = binop.lhs });
                    },
                    .get_attribute => |getattr| {
                        try work_stack.appendBounded(.{ .kind = .emit_getattr, .node_idx = work.node_idx, .extra = @intFromEnum(getattr.attr) });
                        try work_stack.appendBounded(.{ .kind = .compile_expr, .node_idx = getattr.obj });
                    },
                    else => {
                        log.err("Unexpected expression node: {any}", .{std.meta.activeTag(node.data)});
                        return Error.AstInvalid;
                    },
                }
            },
            .emit_getattr => {
                const loc: Token.Loc = ast.tokens.items(.loc)[work.extra];
                try builder.emit(.{ .get_attr = source[loc.start..loc.end] });
            },
            .emit_call => {
                try builder.emit(.{ .call = work.extra });
            },
            .emit_binop => {
                const binop: Runtime.BinOp = @enumFromInt(work.extra);
                try builder.emit(.{ .binary_op = binop });
            },
            .emit_store => {
                // Used by other compilation contexts
            },
        }
    }
}

fn compileFunction(
    gc: Allocator,
    arena: Allocator,
    ast: *const Ast,
    source: [:0]const u8,
    interner: *StringInterner,
    parent_scope: *Scope,
    body_idx: Ast.Node.Index,
    args_idx: Ast.Node.Index,
) Error!*Runtime.StarFunc {
    var func_compiler = try FunctionCompiler.init(gc, arena, interner, parent_scope);
    defer func_compiler.deinit();

    const args_node = ast.nodes.get(@intFromEnum(args_idx));
    const fn_args = args_node.data.fn_args.positional;

    func_compiler.arity = fn_args.len;

    for (fn_args) |arg_idx| {
        const arg_node = ast.nodes.get(@intFromEnum(arg_idx));
        const arg_tok = ast.tokens.get(@intFromEnum(arg_node.data.fn_arg.binding));
        _ = try func_compiler.defineLocal(source[arg_tok.loc.start..arg_tok.loc.end]);
    }

    const body_node = ast.nodes.get(@intFromEnum(body_idx));
    const block_data = body_node.data.block;

    for (block_data.statements) |stmt_idx| {
        const stmt_node = ast.nodes.get(@intFromEnum(stmt_idx));

        switch (stmt_node.data) {
            .var_definition => |v| {
                try compileExpr(ast, source, &func_compiler.base, &func_compiler.scope, v.value, &func_compiler, null);

                const binding_tok = ast.tokens.get(@intFromEnum(v.binding));
                const binding_src = source[binding_tok.loc.start..binding_tok.loc.end];
                const local_idx = try func_compiler.defineLocal(binding_src);
                try func_compiler.base.emit(.{ .store = local_idx });
            },
            .def_proto => |d| {
                const name_tok = ast.tokens.get(@intFromEnum(d.name));
                const fn_name = source[name_tok.loc.start..name_tok.loc.end];

                const nested_func_obj = try compileFunction(gc, arena, ast, source, interner, &func_compiler.scope, d.body, d.args);

                const func_idx = try func_compiler.base.addConst(&nested_func_obj.obj);

                if (nested_func_obj.names_free.len > 0) {
                    try func_compiler.base.emit(.{ .make_closure = func_idx });
                } else {
                    try func_compiler.base.emit(.{ .load_const = func_idx });
                }

                const local_idx = try func_compiler.defineLocal(fn_name);
                try func_compiler.base.emit(.{ .store = local_idx });
            },
            .@"return" => |ret_expr_idx| {
                try compileExpr(ast, source, &func_compiler.base, &func_compiler.scope, ret_expr_idx, &func_compiler, null);
                try func_compiler.base.emit(.ret);
            },
            .call, .add, .sub, .mul, .literal, .identifier => {
                try compileExpr(ast, source, &func_compiler.base, &func_compiler.scope, stmt_idx, &func_compiler, null);
            },
            else => {
                log.err("Unexpected statement node: {any}", .{std.meta.activeTag(stmt_node.data)});
                return Error.AstInvalid;
            },
        }
    }

    // TODO: This needs to validate all branches for returns first.
    const last_tag = std.meta.activeTag(func_compiler.base.code.items[func_compiler.base.code.items.len - 1]);
    if (func_compiler.base.code.items.len == 0 or last_tag != .ret or last_tag != .ret_none) {
        try func_compiler.base.emit(.ret_none);
    }

    const func = try gc.create(Runtime.StarFunc);
    func.* = .{
        .code = try func_compiler.base.code.toOwnedSlice(gc),
        .consts = try func_compiler.base.constants.toOwnedSlice(gc),
        .arity = fn_args.len,
        .frame_size = func_compiler.local_names.items.len,
        .names_local = try func_compiler.local_names.toOwnedSlice(gc),
        .names_free = try func_compiler.free_names.toOwnedSlice(gc),
        .names_global = try func_compiler.global_names.toOwnedSlice(gc),
        .closure_cells = &.{},
    };

    return func;
}

fn compileModule(
    gpa: Allocator,
    module_name: []const u8,
    source: [:0]const u8,
    ast: *const Ast,
    interner: *StringInterner,
    gc: Allocator,
) Error!Module {
    var arena_allocator = std.heap.ArenaAllocator.init(gpa);
    const arena = arena_allocator.allocator();
    defer arena_allocator.deinit();

    var module_compiler = try ModuleCompiler.init(gc, arena, interner);
    defer module_compiler.deinit();

    if (ast.nodes.len == 0) return Error.AstEmpty;
    const root_node = ast.nodes.get(0);
    if (root_node.data != .block) return Error.AstInvalid;

    for (root_node.data.block.statements) |stmt_idx| {
        const stmt_node = ast.nodes.get(@intFromEnum(stmt_idx));

        switch (stmt_node.data) {
            .var_definition => |v| {
                try compileExpr(ast, source, &module_compiler.base, &module_compiler.scope, v.value, null, &module_compiler);

                const binding_tok = ast.tokens.get(@intFromEnum(v.binding));
                const binding_src = source[binding_tok.loc.start..binding_tok.loc.end];
                const global_idx = try module_compiler.defineGlobal(binding_src);
                try module_compiler.base.emit(.{ .store_global = global_idx });
            },
            .def_proto => |d| {
                const name_tok = ast.tokens.get(@intFromEnum(d.name));
                const fn_name = source[name_tok.loc.start..name_tok.loc.end];

                const func_obj = try compileFunction(gc, arena, ast, source, interner, &module_compiler.scope, d.body, d.args);
                const func_idx = try module_compiler.base.addConst(&func_obj.obj);

                if (func_obj.names_free.len > 0) {
                    try module_compiler.base.emit(.{ .make_closure = func_idx });
                } else {
                    try module_compiler.base.emit(.{ .load_const = func_idx });
                }

                const global_idx = try module_compiler.defineGlobal(fn_name);
                try module_compiler.base.emit(.{ .store_global = global_idx });
            },
            .@"return" => return Error.AstInvalid,
            .call, .add, .sub, .mul, .literal, .identifier => {
                try compileExpr(ast, source, &module_compiler.base, &module_compiler.scope, stmt_idx, null, &module_compiler);
            },
            else => {
                log.err("Unexpected top-level statement: {any}", .{std.meta.activeTag(stmt_node.data)});
                return Error.AstInvalid;
            },
        }
    }

    try module_compiler.base.emit(.ret_none);
    return Module{
        .module_name = module_name,
        .code = try module_compiler.base.code.toOwnedSlice(gc),
        .constants = try module_compiler.base.constants.toOwnedSlice(gc),
        .global_names = try module_compiler.global_names.toOwnedSlice(gc),
    };
}

pub const Opts = struct {
    stack_depth: usize = knobs.ast_depth,
};

pub fn compile(fallback: Allocator, module_name: []const u8, source: [:0]const u8, ast: *const Ast, comptime opts: Opts, gc: Allocator) Error!Module {
    if (ast.nodes.len == 0) return Error.AstEmpty;

    var stack_fallback = std.heap.stackFallback(opts.stack_depth, fallback);
    const gpa = stack_fallback.get();

    var interner = try StringInterner.init(gc);
    defer interner.deinit();

    return try compileModule(gpa, module_name, source, ast, &interner, gc);
}

test compile {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();
    const Instruction = Runtime.Instruction;

    const source: [:0]const u8 =
        \\asdf = 1
        \\def foo():
        \\  b = 2
        \\  return b
    ;
    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();

    const module = try compile(std.testing.allocator, "<test>", source, &ast, .{
        .stack_depth = 0,
    }, gc);

    // Check global names
    try std.testing.expectEqual(2, module.global_names.len);
    try std.testing.expectEqualStrings("asdf", module.global_names[0]);
    try std.testing.expectEqualStrings("foo", module.global_names[1]);

    // Check constants
    try std.testing.expectEqual(2, module.constants.len);

    const int_const = try Runtime.downCast(Runtime.StarInt, module.constants[0]);
    try std.testing.expectEqual(1, int_const.num);

    const func_const = try Runtime.downCast(Runtime.StarFunc, module.constants[1]);
    try std.testing.expectEqual(0, func_const.arity);
    try std.testing.expectEqual(1, func_const.frame_size);

    // Check module code structure
    try std.testing.expectEqual(5, module.code.len);
    try std.testing.expectEqual(Instruction{ .load_const = @enumFromInt(0) }, module.code[0]);
    try std.testing.expectEqual(Instruction{ .store_global = @enumFromInt(0) }, module.code[1]);
    try std.testing.expectEqual(Instruction{ .load_const = @enumFromInt(1) }, module.code[2]);
    try std.testing.expectEqual(Instruction{ .store_global = @enumFromInt(1) }, module.code[3]);

    // Check function code
    try std.testing.expectEqual(5, func_const.code.len);
    try std.testing.expectEqual(Instruction{ .load_const = @enumFromInt(0) }, func_const.code[0]);
    try std.testing.expectEqual(Instruction{ .store = @enumFromInt(0) }, func_const.code[1]);
    try std.testing.expectEqual(Instruction{ .load = @enumFromInt(0) }, func_const.code[2]);
    try std.testing.expectEqual(Instruction{ .ret = {} }, func_const.code[3]);
}

test "compile function call" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    const source: [:0]const u8 =
        \\def add():
        \\  return 1
        \\x = add()
    ;
    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();

    const module = try compile(std.testing.allocator, "<test>", source, &ast, .{
        .stack_depth = 0,
    }, gc);

    try std.testing.expectEqual(2, module.global_names.len);
    try std.testing.expectEqualStrings("add", module.global_names[0]);
    try std.testing.expectEqualStrings("x", module.global_names[1]);

    try std.testing.expectEqual(1, module.constants.len);
    const func_const = try Runtime.downCast(Runtime.StarFunc, module.constants[0]);

    try std.testing.expectEqual(0, func_const.arity);
    try std.testing.expectEqual(0, func_const.frame_size);

    try std.testing.expectEqual(1, func_const.consts.len);
    const func_int = try Runtime.downCast(Runtime.StarInt, func_const.consts[0]);
    try std.testing.expectEqual(1, func_int.num);
}

test "nested function can access module globals and builtins" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gc = arena.allocator();

    const source: [:0]const u8 =
        \\x = 42
        \\def outer():
        \\  def inner():
        \\    return x
        \\  return inner
    ;
    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();

    const module = try compile(std.testing.allocator, "<test>", source, &ast, .{
        .stack_depth = 0,
    }, gc);

    // Module should have x and outer as globals
    try std.testing.expectEqual(2, module.global_names.len);
    try std.testing.expectEqualStrings("x", module.global_names[0]);
    try std.testing.expectEqualStrings("outer", module.global_names[1]);

    // Get the outer function
    const outer_func = try Runtime.downCast(Runtime.StarFunc, module.constants[1]);

    // Get the inner function (constant in outer)
    const inner_func = try Runtime.downCast(Runtime.StarFunc, outer_func.consts[0]);

    // Inner function should have 'x' in its names_global (not names_free!)
    try std.testing.expectEqual(1, inner_func.names_global.len);
    try std.testing.expectEqualStrings("x", inner_func.names_global[0]);

    // Inner should emit load_global for 'x'
    const Instruction = Runtime.Instruction;
    try std.testing.expectEqual(Instruction{ .load_global = @enumFromInt(0) }, inner_func.code[0]);
    try std.testing.expectEqual(Instruction{ .ret = {} }, inner_func.code[1]);
}
