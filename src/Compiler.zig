const std = @import("std");
const Ast = @import("Ast.zig");
const Runtime = @import("Runtime.zig");
const Gc = @import("zig_libgc");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

pub const Error = error{
    AstEmpty,
    LiteralInvalid,
    BigIntUnsupported,
} || std.mem.Allocator.Error;

/// Builds a list of dependencies for a given node
fn getDeps(node: Ast.Node) []const Ast.Node.Index {
    return switch (node.data) {
        .root => |r| r.statements, // depends on each statement
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
        const node = ast.nodes.get(i);
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

        for (getDeps(ast.nodes.get(n))) |dep_id| {
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
) Error!*Runtime.StarModule {
    var results = try gpa.alloc(?*Runtime.StarObj, node_order.len + 1);
    defer gpa.free(results);
    if (builtin.mode == .Debug) @memset(results, null);

    const module = try Runtime.StarModule.init();

    for (node_order) |node_idx| {
        const node = ast.nodes.get(node_idx);
        var value: *Runtime.StarObj = undefined;

        switch (node.data) {
            .literal => {
                const token = ast.tokens.get(@intFromEnum(node.main_token));
                switch (token.tag) {
                    .number_literal => {
                        const number_src = source[token.loc.start..token.loc.end];
                        const number_res = std.zig.parseNumberLiteral(number_src);
                        switch (number_res) {
                            .int => |n| value = &(try Runtime.StarInt.init(n)).obj,
                            .big_int => return error.BigIntUnsupported,
                            .failure => return error.LiteralInvalid,
                            .float => {
                                const n = std.fmt.parseFloat(f64, number_src) catch return error.LiteralInvalid;
                                value = &(try Runtime.StarFloat.init(n)).obj;
                            },
                        }
                    },
                    .string => {
                        const string_literal = source[token.loc.start..token.loc.end];
                        const str_parsed = std.zig.string_literal.parseAlloc(Gc.allocator(), string_literal) catch return error.LiteralInvalid;
                        const str_star = try Gc.allocator().create(Runtime.StarStr);
                        str_star.str = str_parsed;
                        value = &str_star.obj;
                    },
                    else => return error.LiteralInvalid,
                }
            },
            .var_definition => |v| {
                const rhs_value = results[@intFromEnum(v.value)].?;
                const binding_tok = ast.tokens.get(@intFromEnum(v.binding));
                const binding_src = source[binding_tok.loc.start..binding_tok.loc.end];
                const binding = try Runtime.StarStr.init_dupe(binding_src);
                try module.obj.setAttrConcrete(binding, rhs_value);
                value = rhs_value;
            },
            .root => {
                // root doesn't evaluate to a runtime value, but give it a sentinel
                value = Runtime.StarNone.instance;
            },
            else => @panic("unsupported AST node"),
        }

        results[node_idx] = value;
    }

    return module;
}

pub const Opts = struct {
    stack_depth: usize = 4096,
};

pub fn compile(fallback: Allocator, source: [:0]const u8, ast: *const Ast, comptime opts: Opts) Error!*Runtime.StarModule {
    if (ast.nodes.len == 0) return error.AstEmpty;

    var stack_fallback = std.heap.stackFallback(opts.stack_depth, fallback);
    const gpa = stack_fallback.get();
    const order = try topoSort(gpa, ast);
    defer gpa.free(order);

    return try compileInOrder(gpa, source, ast, order);
}

test compile {
    const source: [:0]const u8 = "asdf = 1";
    var ast = try Ast.parse(std.testing.allocator, source);
    defer ast.deinit();

    const module = try compile(std.testing.allocator, source, &ast, .{
        .stack_depth = 0, // Set to 0 to force usage of the fallback allocator and detect leaks.
    });

    const name = try Runtime.StarStr.init_dupe("asdf");
    const value_obj = try module.obj.getAttrConcrete(name);

    const value_num = try Runtime.downCast(Runtime.StarInt, value_obj);
    try std.testing.expectEqual(1, value_num.num);
}
