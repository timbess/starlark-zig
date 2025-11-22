const std = @import("std");
const Tokenizer = @import("Tokenizer.zig");
const Token = Tokenizer.Token;
const Parser = @import("Parser.zig");
const Allocator = std.mem.Allocator;
const Arena = std.heap.ArenaAllocator;
const Ast = @This();

gpa: Allocator,
arena: Arena,
tokens: TokenList.Slice,
nodes: NodeList.Slice,
errors: std.ArrayList(Error),
code: [:0]const u8,

pub fn parse(gpa: Allocator, code: [:0]const u8) !Ast {
    var tokenizer = Tokenizer.init(code);
    var parser = try Parser.init(gpa, &tokenizer);
    errdefer parser.deinit();

    try parser.parse();

    var res = parser.toOwned();
    return Ast{
        .gpa = gpa,
        .arena = res.arena,
        .nodes = res.nodes.toOwnedSlice(),
        .tokens = res.tokens.toOwnedSlice(),
        .errors = res.errors,
        .code = code,
    };
}

pub fn deinit(self: *Ast) void {
    self.arena.deinit();
    self.tokens.deinit(self.gpa);
    self.nodes.deinit(self.gpa);
    self.errors.deinit(self.gpa);
}

const NodeList = std.MultiArrayList(Node);
const TokenList = std.MultiArrayList(Token);

pub const Error = struct {
    tag: Tag,
    token: Token.Index,

    pub const Tag = enum {
        expected_expr,
        expected_operator,
        expected_fn_arg,
        unexpected_eof,
    };
};

pub const Node = struct {
    main_token: Token.Index,
    data: Data,

    pub const Index = enum(u32) { none = std.math.maxInt(u32), _ };

    pub const Tag = enum {
        block,
        var_definition,
        def_proto,
        fn_arg,
        fn_args,
        literal,
        @"return",
        identifier,
        call,
        call_args,
        bool_and,
        bool_or,
        add,
        sub,
        mul,
        get_attribute,
    };

    pub const Data = union(Tag) {
        block: struct {
            statements: []const Node.Index,
        },
        var_definition: struct {
            binding: Token.Index,
            value: Node.Index,
        },
        def_proto: struct {
            name: Token.Index,
            args: Node.Index,
            body: Node.Index,
        },
        fn_arg: struct {
            binding: Token.Index,
        },
        fn_args: struct {
            positional: []const Node.Index,
        },
        literal: void, // Use token
        @"return": Node.Index,
        identifier: void,
        call: struct {
            func: Node.Index,
            args: Node.Index,
        },
        call_args: struct {
            args: []const Node.Index,
        },
        bool_and: BinOp,
        bool_or: BinOp,
        add: BinOp,
        sub: BinOp,
        mul: BinOp,
        get_attribute: struct {
            obj: Node.Index,
            attr: Token.Index,
        },
    };
};

pub const BinOp = struct { lhs: Node.Index, rhs: Node.Index };

pub const DebugNodeFormatter = std.fmt.Formatter(*const Ast, struct {
    pub fn format(
        ast: *const Ast,
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        if (ast.nodes.len > 0)
            try (formatNode(ast, writer, @enumFromInt(0), 0) catch std.Io.Writer.Error.WriteFailed);
    }

    pub fn formatNode(
        ast: *const Ast,
        writer: anytype,
        root_idx: Node.Index,
        indent: usize,
    ) !void {
        const Frame = struct {
            idx: Node.Index,
            state: enum {
                start,
                after_args,
                close_array,
                close_node,
            },
            indent: usize,
        };

        var stack_buf: [1024]Frame = undefined;
        var stack = std.ArrayList(Frame).initBuffer(&stack_buf);

        try stack.appendBounded(.{ .idx = root_idx, .state = .start, .indent = indent });

        while (stack.pop()) |*frame| {
            if (frame.idx == .none) {
                try writer.print("none", .{});
                continue;
            }

            const node = ast.nodes.get(@intFromEnum(frame.idx));
            const tag = node.data;

            switch (frame.state) {
                .start => {
                    // Print opening with indent
                    for (0..frame.indent) |_| _ = try writer.write("  ");
                    try writer.print("{s}(", .{@tagName(tag)});
                    try formatToken(ast, writer, node.main_token);

                    switch (tag) {
                        .block => |b| {
                            try writer.print(", statements=[\n", .{});

                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_array, .indent = frame.indent + 1 });

                            if (b.statements.len > 0) {
                                for (b.statements, 0..) |_, i| {
                                    const rev = b.statements.len - 1 - i;
                                    try stack.appendBounded(.{ .idx = b.statements[rev], .state = .start, .indent = frame.indent + 1 });
                                }
                            }
                        },
                        .var_definition => |v| {
                            try writer.print(", binding=", .{});
                            try formatToken(ast, writer, v.binding);
                            try writer.print(", value=\n", .{});

                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = v.value, .state = .start, .indent = frame.indent + 1 });
                        },
                        .def_proto => |d| {
                            try writer.print(", name=", .{});
                            try formatToken(ast, writer, d.name);
                            try writer.print(", args=\n", .{});

                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = d.body, .state = .start, .indent = frame.indent + 1 });
                            try stack.appendBounded(.{ .idx = frame.idx, .state = .after_args, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = d.args, .state = .start, .indent = frame.indent + 1 });
                        },
                        .fn_arg => |a| {
                            try writer.print(", binding=", .{});
                            try formatToken(ast, writer, a.binding);
                            try writer.print(")\n", .{});
                        },
                        .fn_args => |fa| {
                            try writer.print(", positional=[\n", .{});

                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_array, .indent = frame.indent + 1 });

                            if (fa.positional.len > 0) {
                                // Push children in reverse
                                for (fa.positional, 0..) |_, i| {
                                    const rev = fa.positional.len - 1 - i;
                                    try stack.appendBounded(.{ .idx = fa.positional[rev], .state = .start, .indent = frame.indent + 1 });
                                }
                            }
                        },
                        .literal => {
                            try writer.print(")\n", .{});
                        },
                        .@"return" => |rv| {
                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = rv, .state = .start, .indent = frame.indent });
                        },
                        .identifier => {
                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                        },
                        .call => |c| {
                            try writer.print(", func=\n", .{});

                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });

                            // Push args node
                            try stack.appendBounded(.{ .idx = c.args, .state = .start, .indent = frame.indent + 1 });

                            // Print args label
                            for (0..frame.indent) |_| _ = try writer.write("  ");
                            try writer.print(", args=\n", .{});

                            // Push func
                            try stack.appendBounded(.{ .idx = c.func, .state = .start, .indent = frame.indent + 1 });
                        },
                        .call_args => |ca| {
                            try writer.print(", args=[\n", .{});

                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_array, .indent = frame.indent + 1 });

                            if (ca.args.len > 0) {
                                for (ca.args, 0..) |_, i| {
                                    const rev = ca.args.len - 1 - i;
                                    try stack.appendBounded(.{ .idx = ca.args[rev], .state = .start, .indent = frame.indent + 1 });
                                }
                            }
                        },
                        .bool_and, .bool_or, .add, .sub => |binop| {
                            try writer.print(", lhs=\n", .{});

                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = binop.rhs, .state = .start, .indent = frame.indent + 1 });

                            for (0..frame.indent) |_| _ = try writer.write("  ");
                            try writer.print(", rhs=\n", .{});

                            try stack.appendBounded(.{ .idx = binop.lhs, .state = .start, .indent = frame.indent + 1 });
                        },
                    }
                },
                .after_args => {
                    // For def_proto: print ", body=" between args and body
                    for (0..frame.indent) |_| _ = try writer.write("  ");
                    try writer.print(", body=\n", .{});
                },
                .close_array => {
                    for (0..frame.indent) |_| _ = try writer.write("  ");
                    try writer.print("]\n", .{});
                },
                .close_node => {
                    for (0..frame.indent) |_| _ = try writer.write("  ");
                    try writer.print(")\n", .{});
                },
            }
        }
    }

    pub fn formatToken(
        ast: *const Ast,
        writer: anytype,
        tok_idx: Token.Index,
    ) !void {
        if (tok_idx == .none) {
            try writer.print("token(none)", .{});
            return;
        }
        const tok = ast.tokens.get(@intFromEnum(tok_idx));
        const raw = ast.code[tok.loc.start..tok.loc.end];
        try writer.print("token({s})", .{raw});
    }
}.format);

test Ast {
    var ast = try Ast.parse(std.testing.allocator,
        \\asdf = 1
    );
    defer ast.deinit();
}
