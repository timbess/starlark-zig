const std = @import("std");
const Ast = @import("Ast.zig");
const Runtime = @import("Runtime.zig");
const Gc = @import("zig_libgc");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

// TODO:
// * Compile Instructions into the module.
// * Add MAKE_FUNCTION and have it index into the module instructions.
// * Rewrite this to output custom non-runtime types. (maybe?)

pub const Error = error{
    AstEmpty,
    LiteralInvalid,
    BigIntUnsupported,
} || std.mem.Allocator.Error;

pub const Module = struct {
    code: []Runtime.Instruction,
    constants: []*Runtime.StarObj,
    names: [][]const u8,
};

pub const Function = struct {
    code: []Runtime.Instruction,
    constants: []*Runtime.StarObj,
    names: [][]const u8,
    locals: [][]const u8,
    globals: [][]const u8,
};

/// Builds a list of dependencies for a given node
fn getDeps(data: Ast.Node.Data) []const Ast.Node.Index {
    return switch (data) {
        .block => |r| r.statements, // depends on each statement
        .var_definition => |v| &[_]Ast.Node.Index{v.value},
        else => &.{}, // literals have no deps
    };
}

/// Topological sort using Kahn’s algorithm
fn topoSort(gpa: std.mem.Allocator, ast: *const Ast) ![]u32 {
    var indegree = try gpa.alloc(usize, ast.nodes.len);
    defer gpa.free(indegree);
    @memset(indegree, 0);

    // build indegree counts
    for (0..ast.nodes.len) |i| {
        const node = ast.nodes.items(.data)[i];
        for (getDeps(node)) |dep_id| {
            const j = @intFromEnum(dep_id);
            indegree[j] += 1;
        }
    }

    var queue = std.ArrayList(u32).empty;
    defer queue.deinit(gpa);

    for (indegree, 0..) |deg, id| {
        if (deg == 0) try queue.append(gpa, @intCast(id));
    }

    var order = std.ArrayList(u32).empty;

    while (queue.items.len > 0) {
        const n = queue.orderedRemove(0);
        try order.append(gpa, n);

        for (getDeps(ast.nodes.items(.data)[n])) |dep_id| {
            const i = @intFromEnum(dep_id);
            indegree[i] -= 1;
            if (indegree[i] == 0) {
                try queue.append(gpa, i);
            }
        }
    }

    reverseArray(u32, order.items);
    return order.toOwnedSlice(gpa);
}

fn reverseArray(T: type, arr: []T) void {
    var left: usize = 0;
    var right: usize = arr.len - 1;

    while (left < right) {
        // Swap elements at i and j
        const temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;

        left += 1;
        right -= 1;
    }
}

/// Iteratively evaluate nodes in topo order
fn compileInOrder(
    gpa: std.mem.Allocator,
    source: [:0]const u8,
    ast: *const Ast,
    node_order: []const u32,
) Error!Module {
    var arena_allocator = std.heap.ArenaAllocator.init(gpa);
    const arena = arena_allocator.allocator();
    defer arena_allocator.deinit();

    // var results = try gpa.alloc(?Value, node_order.len + 1);
    // defer gpa.free(results);
    // if (builtin.mode == .Debug) @memset(results, null);

    const gc = Gc.allocator();

    const ModuleBuilder = struct {
        gc: Allocator,
        code: std.ArrayList(Runtime.Instruction),
        constants: std.ArrayList(*Runtime.StarObj) = .empty,
        names: std.ArrayList([]const u8) = .empty,

        // Ephemeral data
        name_to_idx: std.StringHashMap(Runtime.NameIdx),

        pub fn emit(self: *@This(), instruction: Runtime.Instruction) Allocator.Error!void {
            try self.code.append(self.gc, instruction);
        }

        pub fn addConst(self: *@This(), obj: *Runtime.StarObj) Allocator.Error!Runtime.ConstIdx {
            const idx = self.constants.items.len;
            try self.constants.append(self.gc, obj);
            return @enumFromInt(idx);
        }

        pub fn defineName(self: *@This(), name: []const u8) Allocator.Error!Runtime.NameIdx {
            if (self.name_to_idx.get(name)) |v| return v;
            const idx: Runtime.NameIdx = @enumFromInt(self.names.items.len);
            try self.names.append(self.gc, name);
            try self.name_to_idx.put(name, idx);
            return idx;
        }
    };
    var module = ModuleBuilder{
        .gc = gc,
        .code = try .initCapacity(gc, ast.nodes.len * 2),
        .name_to_idx = .init(arena),
    };

    const func: ?Function = null;
    // var local_names: std.ArrayList([]const u8) = .empty;

    for (node_order) |node_idx| {
        const node = ast.nodes.get(node_idx);

        switch (node.data) {
            .literal => {
                const token = ast.tokens.get(@intFromEnum(node.main_token));
                switch (token.tag) {
                    .number_literal => {
                        const number_src = source[token.loc.start..token.loc.end];
                        const number_res = std.zig.parseNumberLiteral(number_src);
                        const val = brk: switch (number_res) {
                            .int => |n| &(try Runtime.StarInt.init(n)).obj,
                            .big_int => return error.BigIntUnsupported,
                            .failure => return error.LiteralInvalid,
                            .float => {
                                const n = std.fmt.parseFloat(f64, number_src) catch return error.LiteralInvalid;
                                break :brk &(try Runtime.StarFloat.init(n)).obj;
                            },
                        };

                        try module.emit(.{ .load_const = try module.addConst(val) });
                    },
                    .string => {
                        const string_literal = source[token.loc.start..token.loc.end];
                        const str_parsed = std.zig.string_literal.parseAlloc(Gc.allocator(), string_literal) catch return error.LiteralInvalid;
                        const str_star = try Gc.allocator().create(Runtime.StarStr);
                        str_star.str = str_parsed;
                        try module.emit(.{ .load_const = try module.addConst(&str_star.obj) });
                    },
                    else => return error.LiteralInvalid,
                }
            },
            .var_definition => |v| {
                const binding_tok = ast.tokens.get(@intFromEnum(v.binding));
                const binding_src = source[binding_tok.loc.start..binding_tok.loc.end];

                if (func == null) {
                    try module.emit(.{ .store_global = try module.defineName(binding_src) });
                } else {
                    @panic("Not Implemented");
                    // try module.emit(.{ .store = binding_idx });
                }
            },
            .block => {
                try module.emit(.ret);
            },
            else => @panic("unsupported AST node"),
        }

        // results[node_idx] = value;
    }

    return Module{
        .code = try module.code.toOwnedSlice(gc),
        .constants = try module.constants.toOwnedSlice(gc),
        .names = try module.names.toOwnedSlice(gc),
    };
}

pub const Opts = struct {
    stack_depth: usize = 4096,
};

pub fn compile(fallback: Allocator, source: [:0]const u8, ast: *const Ast, comptime opts: Opts) Error!Module {
    if (ast.nodes.len == 0) return error.AstEmpty;

    var stack_fallback = std.heap.stackFallback(opts.stack_depth, fallback);
    const gpa = stack_fallback.get();
    const order = try topoSort(gpa, ast);
    defer gpa.free(order);

    return try compileInOrder(gpa, source, ast, order);
}

test compile {
    const Instruction = Runtime.Instruction;

    const source: [:0]const u8 = "asdf = 1";
    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();

    const module = try compile(std.testing.allocator, source, &ast, .{
        .stack_depth = 0, // Set to 0 to force usage of the fallback allocator and detect leaks.
    });

    var Test = struct {
        module: Module,
        c_idx: usize = 0,

        pub fn next_op(self: *@This()) ?Runtime.Instruction {
            if (self.c_idx >= self.module.code.len) return null;
            defer self.c_idx += 1;
            return self.module.code[self.c_idx];
        }
    }{ .module = module };

    try std.testing.expectEqualStrings(module.names[0], "asdf");

    const first_const = try Runtime.downCast(Runtime.StarInt, module.constants[0]);
    try std.testing.expectEqual(1, first_const.num);

    try std.testing.expectEqual(Instruction{ .load_const = @enumFromInt(0) }, Test.next_op());
    try std.testing.expectEqual(Instruction{ .store_global = @enumFromInt(0) }, Test.next_op());
    try std.testing.expectEqual(Instruction{ .ret = {} }, Test.next_op());
    try std.testing.expectEqual(null, Test.next_op());
}
