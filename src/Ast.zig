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
        bool_not,
        add,
        sub,
        mul,
        eq,
        ne,
        lt,
        le,
        gt,
        ge,
        get_attribute,
        list_literal,
        dict_literal,
        index,
        @"if",
        @"for",
        @"while",
        if_expression,
        tuple_literal,
        mod,
        div,
        floor_div,
        bit_or,
        contains,
        lambda,
        @"break",
        @"continue",
        pass,
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
            default: Node.Index, // .none if no default
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
            kwargs: []const KwArg = &.{},
        },
        bool_and: BinOp,
        bool_or: BinOp,
        bool_not: Node.Index,
        add: BinOp,
        sub: BinOp,
        mul: BinOp,
        eq: BinOp,
        ne: BinOp,
        lt: BinOp,
        le: BinOp,
        gt: BinOp,
        ge: BinOp,
        get_attribute: struct {
            obj: Node.Index,
            attr: Token.Index,
        },
        list_literal: struct {
            elements: []const Node.Index,
        },
        dict_literal: struct {
            keys: []const Node.Index,
            values: []const Node.Index,
        },
        index: struct {
            obj: Node.Index,
            idx: Node.Index,
        },
        @"if": struct {
            condition: Node.Index,
            then_body: Node.Index,
            else_body: Node.Index,
        },
        @"for": struct {
            binding: Token.Index,
            iterable: Node.Index,
            body: Node.Index,
        },
        @"while": struct {
            condition: Node.Index,
            body: Node.Index,
        },
        if_expression: struct {
            condition: Node.Index,
            then_branch: Node.Index,
            else_branch: Node.Index,
        },
        tuple_literal: struct {
            elements: []const Node.Index,
        },
        mod: BinOp,
        div: BinOp,
        floor_div: BinOp,
        bit_or: BinOp,
        contains: BinOp,
        lambda: Lambda,
        @"break": void,
        @"continue": void,
        pass: void,
    };
};

pub const BinOp = struct { lhs: Node.Index, rhs: Node.Index };

pub const Lambda = struct { args: Node.Index, body: Node.Index };

pub const KwArg = struct { name: Token.Index, value: Node.Index };

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
                        .bool_and, .bool_or, .add, .sub, .mul, .eq, .ne, .lt, .le, .gt, .ge, .index => |binop| {
                            try writer.print(", lhs=\n", .{});

                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = binop.rhs, .state = .start, .indent = frame.indent + 1 });

                            for (0..frame.indent) |_| _ = try writer.write("  ");
                            try writer.print(", rhs=\n", .{});

                            try stack.appendBounded(.{ .idx = binop.lhs, .state = .start, .indent = frame.indent + 1 });
                        },
                        .bool_not => |operand| {
                            try writer.print(", operand=\n", .{});
                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = operand, .state = .start, .indent = frame.indent + 1 });
                        },
                        .get_attribute => |ga| {
                            try writer.print(", attr=", .{});
                            try formatToken(ast, writer, ga.attr);
                            try writer.print(", obj=\n", .{});
                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = ga.obj, .state = .start, .indent = frame.indent + 1 });
                        },
                        .list_literal => |ll| {
                            try writer.print(", elements=[\n", .{});
                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_array, .indent = frame.indent + 1 });
                            if (ll.elements.len > 0) {
                                for (ll.elements, 0..) |_, i| {
                                    const rev = ll.elements.len - 1 - i;
                                    try stack.appendBounded(.{ .idx = ll.elements[rev], .state = .start, .indent = frame.indent + 1 });
                                }
                            }
                        },
                        .@"if" => |if_node| {
                            try writer.print(", cond=\n", .{});
                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = if_node.else_body, .state = .start, .indent = frame.indent + 1 });
                            try stack.appendBounded(.{ .idx = if_node.then_body, .state = .start, .indent = frame.indent + 1 });
                            try stack.appendBounded(.{ .idx = if_node.condition, .state = .start, .indent = frame.indent + 1 });
                        },
                        .@"for" => |for_node| {
                            try writer.print(", binding=", .{});
                            try formatToken(ast, writer, for_node.binding);
                            try writer.print(", iterable=\n", .{});
                            try stack.appendBounded(.{ .idx = frame.idx, .state = .close_node, .indent = frame.indent });
                            try stack.appendBounded(.{ .idx = for_node.body, .state = .start, .indent = frame.indent + 1 });
                            try stack.appendBounded(.{ .idx = for_node.iterable, .state = .start, .indent = frame.indent + 1 });
                        },
                        .@"break", .@"continue" => {
                            try writer.print(")\n", .{});
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
