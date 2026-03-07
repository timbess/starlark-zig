const std = @import("std");
const Parser = @This();

const Tokenizer = @import("Tokenizer.zig");
const Token = Tokenizer.Token;
const Allocator = std.mem.Allocator;
const Arena = std.heap.ArenaAllocator;
const Ast = @import("Ast.zig");
const Node = Ast.Node;

/// Used only for backing data structures, not the nested data IN the nodes.
gpa: Allocator,
/// Used for nested data inside the nodes (literals, strings, dynamically sized things).
arena: Arena,
source: [:0]const u8,
nodes: std.MultiArrayList(Node) = .empty,
tokens: std.MultiArrayList(Token) = .empty,
scratch: std.ArrayList(Node.Index) = .empty,
errors: std.ArrayList(Ast.Error) = .empty,
token_idx: u32 = 0,

const scope = .parser;
const log = std.log.scoped(scope);

const SIZING_FACTOR = 1 / 8;

const Error = error{ UnexpectedToken, ParseError } || Allocator.Error;

pub fn init(gpa: Allocator, tokenizer: *Tokenizer) !Parser {
    var tokens = std.MultiArrayList(Token).empty;
    try tokens.ensureTotalCapacity(gpa, tokenizer.source.len * SIZING_FACTOR);
    errdefer tokens.deinit(gpa);

    while (tokenizer.next()) |t| {
        if (std.log.logEnabled(.debug, scope)) {
            log.debug("Token: {any} '{s}'", .{ t.tag, try tokenizer.read_raw_token(t) });
        }
        try tokens.append(gpa, t);
        if (t.tag == .invalid) return error.UnexpectedToken;
        if (t.tag == .eof) break;
    } else |e| {
        return e;
    }

    return Parser{
        .gpa = gpa,
        .source = tokenizer.source,
        .arena = .init(gpa),
        .tokens = tokens,
    };
}

fn consumeNext(self: *Parser) Token.Index {
    const result = self.token_idx;
    self.advance();
    return @enumFromInt(result);
}

fn advance(self: *Parser) void {
    self.token_idx += 1;
}

fn consume(self: *Parser, tag: Token.Tag) ?Token.Index {
    if (self.currentTag() == tag) return self.consumeNext();
    return null;
}

fn expectConsume(self: *Parser, tag: Token.Tag) Error!Token.Index {
    if (self.currentTag() == tag) return self.consumeNext();
    return error.UnexpectedToken;
}

fn consumeOneOf(self: *Parser, comptime tags: []const Token.Tag) ?Token.Index {
    const current = self.currentTag();
    inline for (tags) |tag|
        if (current == tag) return self.consumeNext();
    return null;
}

fn consumeMany(self: *Parser, tags: []const Token.Tag) ?Token.Index {
    const ahead = self.tokens.items(.tag)[self.token_idx..];
    if (!std.mem.startsWith(Token.Tag, ahead, tags)) return null;
    const result = self.token_idx;
    self.token_idx += tags.len;
    return @enumFromInt(result);
}

fn tokenTag(self: *Parser, idx: u32) ?Token.Tag {
    return self.tokens.items(.tag)[@intCast(idx)];
}

fn currentTag(self: *Parser) ?Token.Tag {
    return self.tokenTag(self.token_idx);
}

fn tokenIdx(self: *Parser) Token.Index {
    return @enumFromInt(self.token_idx);
}

pub fn addNode(self: *Parser, node: Node) Allocator.Error!Node.Index {
    const result: Node.Index = @enumFromInt(self.nodes.len);
    try self.nodes.append(self.gpa, node);
    return result;
}

pub fn nodeData(self: *Parser, idx: Node.Index) *Node.Data {
    return &self.nodes.items(.data)[@intFromEnum(idx)];
}

pub fn parse(self: *Parser) !void {
    std.debug.assert(self.nodes.len == 0);

    // TODO: Should I just return this?
    _ = try self.blockExpr(.{ .root_level = true });

    std.debug.assert(@intFromEnum(self.tokenIdx()) == self.tokens.len);
}

// TODO: Rethink this design.
const BlockOpts = struct { root_level: bool = false };
pub fn blockExpr(self: *Parser, comptime opts: BlockOpts) !Node.Index {
    const old_len = self.scratch.items.len;
    defer self.scratch.shrinkRetainingCapacity(old_len);

    const current_token =
        if (opts.root_level) self.tokenIdx() else try self.expectConsume(.block_start);

    const node_idx = try self.addNode(.{
        .main_token = current_token,
        .data = .{ .block = .{ .statements = &.{} } },
    });
    errdefer self.nodes.clearRetainingCapacity();

    while (self.token_idx < self.tokens.len) {
        if (opts.root_level) {
            if (self.consume(.eof)) |_| {
                break;
            }
        } else {
            if (self.consumeOneOf(&.{ .block_end, .eof })) |_| {
                break;
            }
        }
        try self.scratch.append(self.gpa, try self.expectExpr());
    }

    const statements = try self.arena.allocator().dupe(Node.Index, self.scratch.items[old_len..]);
    self.nodeData(node_idx).* = .{ .block = .{ .statements = statements } };

    return node_idx;
}

pub fn parseFnArgs(self: *Parser) Error!Node.Index {
    const old_len = self.scratch.items.len;
    defer self.scratch.shrinkRetainingCapacity(old_len);

    const main_token = try self.expectConsume(.l_paren);

    const fn_args_idx = try self.addNode(.{
        .main_token = main_token,
        .data = .{ .fn_args = .{ .positional = &.{} } },
    });

    while (true) {
        if (self.consume(.identifier)) |arg_name| {
            try self.scratch.append(self.gpa, try self.addNode(.{
                .main_token = arg_name,
                .data = .{ .fn_arg = .{ .binding = arg_name } },
            }));
        }
        switch (self.currentTag() orelse return self.fail(.unexpected_eof)) {
            .comma => self.advance(),
            .r_paren => {
                self.advance(); // already checked r_paren
                const scratch_statements = self.scratch.items[old_len..];
                if (scratch_statements.len == 0) return fn_args_idx;
                const statements = try self.arena.allocator().dupe(Node.Index, scratch_statements);
                self.nodeData(fn_args_idx).fn_args.positional = statements;
                return fn_args_idx;
            },
            else => return self.fail(.expected_fn_arg),
        }
    }
}

pub fn expectExpr(self: *Parser) Error!Node.Index {
    return try self.parseExpr(0) orelse self.fail(.expected_expr);
}

// const Assoc = enum {
//     left,
//     none,
// };

const OperInfo = struct {
    prec: i8,
    tag: Node.Tag,
    // assoc: Assoc = Assoc.left,
};

const operTable = std.enums.directEnumArrayDefault(Token.Tag, OperInfo, .{ .prec = -1, .tag = Node.Tag.block }, 0, .{
    .keyword_or = .{ .prec = 10, .tag = .bool_or },
    .keyword_and = .{ .prec = 20, .tag = .bool_and },

    .eq_eq = .{ .prec = 25, .tag = .eq },
    .bang_eq = .{ .prec = 25, .tag = .ne },
    .lt = .{ .prec = 25, .tag = .lt },
    .lt_eq = .{ .prec = 25, .tag = .le },
    .gt = .{ .prec = 25, .tag = .gt },
    .gt_eq = .{ .prec = 25, .tag = .ge },

    .plus = .{ .prec = 30, .tag = .add },
    .minus = .{ .prec = 30, .tag = .sub },

    .star = .{ .prec = 40, .tag = .mul },
});

pub fn parseExpr(self: *Parser, min_prec: i32) Error!?Node.Index {
    std.debug.assert(min_prec >= 0);
    var node = try self.parseExprWithoutOperators() orelse return null;

    // const banned_prec: i8 = -1;

    while (true) {
        const tok_tag = self.currentTag() orelse return error.ParseError;
        const info = operTable[@as(usize, @intCast(@intFromEnum(tok_tag)))];
        if (info.prec < min_prec) {
            break;
        }
        // if (info.prec == banned_prec) {
        //     @panic("handle chained comparisons");
        // }

        const oper_token = self.consumeNext();

        const rhs = try self.parseExpr(info.prec + 1) orelse {
            try self.warn(.expected_expr);
            return node;
        };

        // TODO: Rethink the performance characteristics of this.
        switch (info.tag) {
            inline else => |t| {
                if (@FieldType(Node.Data, @tagName(t)) == Ast.BinOp) {
                    node = try self.addNode(.{
                        .main_token = oper_token,
                        .data = @unionInit(Node.Data, @tagName(t), .{ .lhs = node, .rhs = rhs }),
                    });
                } else return self.fail(.expected_operator);
            },
        }

        // if (info.assoc == Assoc.none) {
        //     banned_prec = info.prec;
        // }
    }

    return node;
}

pub fn parseExprWithoutOperators(self: *Parser) Error!?Node.Index {
    if (self.currentTag()) |t| {
        log.debug("parseExpr hit tag: {any}", .{t});
        var node_idx = blk: switch (t) {
            .identifier => {
                const ident = self.consumeNext();
                if (self.consume(.eq)) |_| {
                    const node_idx = try self.addNode(.{
                        .data = .{
                            .var_definition = .{
                                .binding = ident,
                                .value = undefined,
                            },
                        },
                        .main_token = ident,
                    });
                    errdefer self.nodes.orderedRemove(@intFromEnum(node_idx));
                    const value = try self.expectExpr();
                    self.nodeData(node_idx).var_definition.value = value;
                    break :blk node_idx;
                } else {
                    break :blk try self.addNode(.{
                        .data = .{ .identifier = {} },
                        .main_token = ident,
                    });
                }
            },
            .number_literal, .string, .keyword_true, .keyword_false => {
                const num = self.consumeNext();
                break :blk try self.addNode(.{
                    .data = .{ .literal = {} },
                    .main_token = num,
                });
            },
            .keyword_not => {
                const not_tok = self.consumeNext();
                const operand = try self.expectExpr();
                break :blk try self.addNode(.{
                    .data = .{ .bool_not = operand },
                    .main_token = not_tok,
                });
            },
            .l_bracket => {
                break :blk try self.parseListLiteral();
            },
            .l_brace => {
                break :blk try self.parseDictLiteral();
            },
            .keyword_if => {
                break :blk try self.parseIfStatement();
            },
            .keyword_for => {
                break :blk try self.parseForLoop();
            },
            .keyword_break => {
                const tok = self.consumeNext();
                break :blk try self.addNode(.{
                    .data = .{ .@"break" = {} },
                    .main_token = tok,
                });
            },
            .keyword_continue => {
                const tok = self.consumeNext();
                break :blk try self.addNode(.{
                    .data = .{ .@"continue" = {} },
                    .main_token = tok,
                });
            },
            .keyword_def => {
                self.advance();

                const ident = try self.expectConsume(.identifier);
                const node_idx = try self.addNode(.{
                    .main_token = ident,
                    .data = .{ .def_proto = undefined },
                });

                const args = try self.parseFnArgs();
                _ = try self.expectConsume(.colon);

                const body = try self.blockExpr(.{});
                self.nodeData(node_idx).* = .{
                    .def_proto = .{
                        .name = ident,
                        .args = args, // TODO: Parse args.
                        .body = body,
                    },
                };

                break :blk node_idx;
            },
            .keyword_return => {
                const ret = self.consumeNext();
                const node_idx = try self.addNode(.{
                    .main_token = ret,
                    .data = .{ .@"return" = undefined },
                });
                const value = try self.expectExpr();
                self.nodeData(node_idx).@"return" = value;
                break :blk node_idx;
            },
            else => return self.fail(.expected_expr),
        };

        // Parse function call(s)
        while (try self.parseSuffix(node_idx)) |new_idx| {
            node_idx = new_idx;
        }

        return node_idx;
    }
    return null;
}

fn parseSuffix(self: *Parser, expr_idx: Node.Index) !?Node.Index {
    // Function calls
    if (self.consume(.l_paren)) |lparen| {
        const old_len = self.scratch.items.len;
        defer self.scratch.shrinkRetainingCapacity(old_len);

        // Parse arguments
        while (true) {
            if (self.currentTag() == .r_paren) {
                break;
            }

            const arg = try self.expectExpr();
            try self.scratch.append(self.gpa, arg);

            if (self.consume(.comma)) |_| {
                continue;
            } else if (self.currentTag() == .r_paren) {
                break;
            } else {
                return self.fail(.expected_expr);
            }
        }

        _ = try self.expectConsume(.r_paren);

        const args_slice = try self.arena.allocator().dupe(Node.Index, self.scratch.items[old_len..]);
        const args_node = try self.addNode(.{
            .data = .{ .call_args = .{ .args = args_slice } },
            .main_token = lparen,
        });

        return try self.addNode(.{
            .data = .{
                .call = .{
                    .func = expr_idx,
                    .args = args_node,
                },
            },
            .main_token = lparen,
        });
    }

    if (self.consume(.dot)) |_| {
        const attr_name = try self.expectConsume(.identifier);

        return try self.addNode(.{ .main_token = attr_name, .data = .{
            .get_attribute = .{
                .attr = attr_name,
                .obj = expr_idx,
            },
        } });
    }

    if (self.consume(.l_bracket)) |lbracket| {
        const idx_expr = try self.expectExpr();
        _ = try self.expectConsume(.r_bracket);

        return try self.addNode(.{
            .main_token = lbracket,
            .data = .{
                .index = .{
                    .obj = expr_idx,
                    .idx = idx_expr,
                },
            },
        });
    }

    return null;
}

fn parseListLiteral(self: *Parser) Error!Node.Index {
    const old_len = self.scratch.items.len;
    defer self.scratch.shrinkRetainingCapacity(old_len);

    const lbracket = try self.expectConsume(.l_bracket);

    while (true) {
        if (self.currentTag() == .r_bracket) {
            break;
        }

        const elem = try self.expectExpr();
        try self.scratch.append(self.gpa, elem);

        if (self.consume(.comma)) |_| {
            continue;
        } else if (self.currentTag() == .r_bracket) {
            break;
        } else {
            return self.fail(.expected_expr);
        }
    }

    _ = try self.expectConsume(.r_bracket);

    const elements = try self.arena.allocator().dupe(Node.Index, self.scratch.items[old_len..]);
    return try self.addNode(.{
        .main_token = lbracket,
        .data = .{ .list_literal = .{ .elements = elements } },
    });
}

fn parseDictLiteral(self: *Parser) Error!Node.Index {
    const old_len = self.scratch.items.len;
    defer self.scratch.shrinkRetainingCapacity(old_len);

    const lbrace = try self.expectConsume(.l_brace);

    while (true) {
        if (self.currentTag() == .r_brace) {
            break;
        }

        const key = try self.expectExpr();
        try self.scratch.append(self.gpa, key);
        _ = try self.expectConsume(.colon);
        const value = try self.expectExpr();
        try self.scratch.append(self.gpa, value);

        if (self.consume(.comma)) |_| {
            continue;
        } else if (self.currentTag() == .r_brace) {
            break;
        } else {
            return self.fail(.expected_expr);
        }
    }

    _ = try self.expectConsume(.r_brace);

    const interleaved = self.scratch.items[old_len..];
    const count = interleaved.len / 2;
    const arena = self.arena.allocator();
    const keys_then_values = try arena.alloc(Node.Index, interleaved.len);
    for (0..count) |i| {
        keys_then_values[i] = interleaved[i * 2];
        keys_then_values[i + count] = interleaved[i * 2 + 1];
    }
    return try self.addNode(.{
        .main_token = lbrace,
        .data = .{ .dict_literal = .{
            .keys = keys_then_values[0..count],
            .values = keys_then_values[count..],
        } },
    });
}

fn parseIfStatement(self: *Parser) Error!Node.Index {
    const if_tok = self.consume(.keyword_if) orelse self.consume(.keyword_elif) orelse return self.fail(.expected_expr);
    const condition = try self.expectExpr();
    _ = try self.expectConsume(.colon);
    const then_body = try self.blockExpr(.{});

    var else_body: Node.Index = .none;

    if (self.currentTag() == .keyword_elif) {
        else_body = try self.parseIfStatement();
    } else if (self.consume(.keyword_else)) |_| {
        _ = try self.expectConsume(.colon);
        else_body = try self.blockExpr(.{});
    }

    return try self.addNode(.{
        .main_token = if_tok,
        .data = .{
            .@"if" = .{
                .condition = condition,
                .then_body = then_body,
                .else_body = else_body,
            },
        },
    });
}

fn parseForLoop(self: *Parser) Error!Node.Index {
    const for_tok = try self.expectConsume(.keyword_for);
    const binding = try self.expectConsume(.identifier);
    _ = try self.expectConsume(.keyword_in);
    const iterable = try self.expectExpr();
    _ = try self.expectConsume(.colon);
    const body = try self.blockExpr(.{});

    return try self.addNode(.{
        .main_token = for_tok,
        .data = .{
            .@"for" = .{
                .binding = binding,
                .iterable = iterable,
                .body = body,
            },
        },
    });
}

pub fn warn(self: *Parser, tag: Ast.Error.Tag) Allocator.Error!void {
    try self.errors.append(self.gpa, .{ .tag = tag, .token = self.tokenIdx() });
}

pub fn warnMsg(self: *Parser, msg: Ast.Error) Allocator.Error!void {
    try self.errors.append(self.gpa, msg);
}

pub fn fail(self: *Parser, tag: Ast.Error.Tag) Error {
    try self.warn(tag);
    return error.ParseError;
}

pub fn failMsg(self: *Parser, msg: Ast.Error) Error {
    try self.warnMsg(msg);
    return error.ParseError;
}

pub fn deinit(self: *Parser) void {
    self.tokens.deinit(self.gpa);
    self.nodes.deinit(self.gpa);
    self.scratch.deinit(self.gpa);
    self.errors.deinit(self.gpa);
    self.arena.deinit();
}

pub fn toOwned(self: *Parser) struct {
    tokens: std.MultiArrayList(Token),
    nodes: std.MultiArrayList(Node),
    errors: std.ArrayList(Ast.Error),
    arena: Arena,
} {
    defer self.* = undefined; // Invalidate self unless init is called.
    self.scratch.deinit(self.gpa);
    return .{
        .tokens = self.tokens,
        .nodes = self.nodes,
        .errors = self.errors,
        .arena = self.arena,
    };
}

test Parser {
    {
        const code: [:0]const u8 =
            \\asdf = 1
            \\foo = 2.0
            \\def my_fn(a):
            \\    b = 1
            \\    return {"b": b}
        ;

        var tokenizer = Tokenizer.init(code);
        var parser: Parser = try .init(std.testing.allocator, &tokenizer);
        defer parser.deinit();

        try parser.parse();

        const block: Node = parser.nodes.get(0);
        try std.testing.expectEqual(.block, std.meta.activeTag(block.data));
        try std.testing.expectEqual(3, block.data.block.statements.len);

        const asdf_var_definition: Node = parser.nodes.get(1);
        try std.testing.expectEqual(.var_definition, std.meta.activeTag(asdf_var_definition.data));
        const asdf_lit: Node = parser.nodes.get(2);
        try std.testing.expectEqual(.literal, std.meta.activeTag(asdf_lit.data));
        const foo_var_definition: Node = parser.nodes.get(3);
        try std.testing.expectEqual(.var_definition, std.meta.activeTag(foo_var_definition.data));
        const foo_lit: Node = parser.nodes.get(4);
        try std.testing.expectEqual(.literal, std.meta.activeTag(foo_lit.data));
        const fn_definition: Node = parser.nodes.get(5);
        try std.testing.expectEqual(.def_proto, std.meta.activeTag(fn_definition.data));
        const fn_args_definition: Node = parser.nodes.get(6);
        try std.testing.expectEqual(.fn_args, std.meta.activeTag(fn_args_definition.data));
        const fn_body_definition: Node = parser.nodes.get(8);
        try std.testing.expectEqual(.block, std.meta.activeTag(fn_body_definition.data));
        const return_node: Node = parser.nodes.get(@intFromEnum(fn_body_definition.data.block.statements[1]));
        try std.testing.expectEqual(.@"return", std.meta.activeTag(return_node.data));
        const dict: Node = parser.nodes.get(@intFromEnum(return_node.data.@"return"));
        try std.testing.expectEqual(.dict_literal, std.meta.activeTag(dict.data));

        // asdf lit token
        const asdf_lit_token = parser.tokens.get(@intFromEnum(asdf_lit.main_token));
        const asdf_raw_lit = try tokenizer.read_raw_token(asdf_lit_token);
        try std.testing.expectEqualStrings("1", asdf_raw_lit);

        // asdf ident token
        const asdf_binding_token = parser.tokens.get(@intFromEnum(asdf_var_definition.main_token));
        const asdf_raw_binding = try tokenizer.read_raw_token(asdf_binding_token);
        try std.testing.expectEqualStrings("asdf", asdf_raw_binding);

        // foo lit token
        const foo_lit_token = parser.tokens.get(@intFromEnum(foo_lit.main_token));
        const foo_raw_lit = try tokenizer.read_raw_token(foo_lit_token);
        try std.testing.expectEqualStrings("2.0", foo_raw_lit);

        // foo ident token
        const foo_binding_token = parser.tokens.get(@intFromEnum(foo_var_definition.main_token));
        const foo_raw_binding = try tokenizer.read_raw_token(foo_binding_token);
        try std.testing.expectEqualStrings("foo", foo_raw_binding);

        // Fn token
        const fn_token = parser.tokens.get(@intFromEnum(fn_definition.main_token));
        const raw_fn = try tokenizer.read_raw_token(fn_token);
        try std.testing.expectEqualStrings("my_fn", raw_fn);

        // Fn Arg token
        const fn_args_token = parser.tokens.get(@intFromEnum(fn_args_definition.main_token));
        const raw_fn_args = try tokenizer.read_raw_token(fn_args_token);
        try std.testing.expectEqualStrings("(", raw_fn_args);

        // All args
        const args = fn_args_definition.data.fn_args.positional;
        try std.testing.expectEqual(1, args.len);

        // arg[0]
        const arg_node = parser.nodes.get(@intFromEnum(args[0]));
        const arg_token = parser.tokens.get(@intFromEnum(arg_node.main_token));
        const raw_arg_token = try tokenizer.read_raw_token(arg_token);

        try std.testing.expectEqualStrings("a", raw_arg_token);

        // Fn body
        const fn_body_statements = fn_body_definition.data.block.statements;
        try std.testing.expectEqual(2, fn_body_statements.len);

        // literal dict in return values
        const dict_keys = dict.data.dict_literal.keys;
        const dict_values = dict.data.dict_literal.values;
        try std.testing.expectEqual(dict_keys.len, dict_values.len);

        const key = parser.nodes.get(@intFromEnum(dict_keys[0]));
        const key_token = parser.tokens.get(@intFromEnum(key.main_token));
        try std.testing.expectEqualStrings("\"b\"", try tokenizer.read_raw_token(key_token));
    }
    {}
}
