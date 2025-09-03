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

const SIZING_FACTOR = 1 / 8;

pub fn init(gpa: Allocator, tokenizer: *Tokenizer) !Parser {
    var tokens = std.MultiArrayList(Token).empty;
    try tokens.ensureTotalCapacity(gpa, tokenizer.source.len * SIZING_FACTOR);
    errdefer tokens.deinit(gpa);

    while (tokenizer.next()) |t| {
        if (t.tag == .eof) break;
        try tokens.append(gpa, t);
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

inline fn nextToken(self: *Parser) Token.Index {
    const result = self.token_idx;
    self.token_idx += 1;
    return @enumFromInt(result);
}

inline fn consume(self: *Parser, tag: Token.Tag) ?Token.Index {
    if (self.currentTag() == tag) return self.nextToken();
    return null;
}

inline fn consumeMany(self: *Parser, tags: []const Token.Tag) ?Token.Index {
    const ahead = self.tokens.items(.tag)[self.token_idx..];
    if (!std.mem.startsWith(Token.Tag, ahead, tags)) return null;
    const result = self.token_idx;
    self.token_idx += tags.len;
    return @enumFromInt(result);
}

inline fn tokenTag(self: *Parser, idx: u32) ?Token.Tag {
    return self.tokens.items(.tag)[@intCast(idx)];
}

inline fn currentTag(self: *Parser) ?Token.Tag {
    return self.tokenTag(self.token_idx);
}

inline fn tokenIdx(self: *Parser) Token.Index {
    return @enumFromInt(self.token_idx);
}

pub fn addNode(self: *Parser, node: Node) Allocator.Error!Node.Index {
    const result: Node.Index = @enumFromInt(self.nodes.len);
    try self.nodes.append(self.gpa, node);
    return result;
}

pub fn parse(self: *Parser) !void {
    std.debug.assert(self.nodes.len == 0);

    const old_len = self.scratch.items.len;
    defer self.scratch.shrinkRetainingCapacity(old_len);

    _ = try self.addNode(.{
        .main_token = @enumFromInt(0),
        .data = .{ .root = .{ .statements = &.{} } },
    });
    errdefer self.nodes.clearRetainingCapacity();

    while (self.token_idx < self.tokens.len) {
        try self.scratch.append(self.gpa, try self.expectExpr());
    }

    const statements = try self.arena.allocator().dupe(Node.Index, self.scratch.items[old_len..]);
    self.nodes.items(.data)[0].root.statements = statements;
}

pub fn expectExpr(self: *Parser) !Node.Index {
    if (self.currentTag()) |t| {
        switch (t) {
            .identifier => {
                const ident = self.nextToken();
                if (self.consume(.eq)) |_| {
                    return try self.addNode(.{
                        .data = .{ .var_definition = .{
                            .binding = ident,
                            .value = try self.expectExpr(),
                        } },
                        .main_token = ident,
                    });
                }
            },
            .number_literal => {
                const num = self.nextToken();
                return try self.addNode(.{
                    .data = .{ .literal = {} },
                    .main_token = num,
                });
            },
            else => return self.fail(.expected_expr),
        }
    }
    return error.Unimplemented;
}

pub fn fail(self: *Parser, tag: Ast.Error.Tag) error{ ParseError, OutOfMemory } {
    try self.errors.append(self.gpa, .{ .tag = tag, .token = self.tokenIdx() });
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
        ;

        var tokenizer = Tokenizer.init(code);
        var parser: Parser = try .init(std.testing.allocator, &tokenizer);
        defer parser.deinit();

        try parser.parse();

        const root: Node = parser.nodes.get(0);
        try std.testing.expectEqual(.root, std.meta.activeTag(root.data));
        const lit: Node = parser.nodes.get(1);
        try std.testing.expectEqual(.literal, std.meta.activeTag(lit.data));
        const var_definition: Node = parser.nodes.get(2);
        try std.testing.expectEqual(.var_definition, std.meta.activeTag(var_definition.data));

        try std.testing.expectEqual(2, root.data.root.statements.len);

        const lit_token = parser.tokens.get(@intFromEnum(lit.main_token));
        const raw_lit = try tokenizer.read_raw_token(lit_token);
        try std.testing.expectEqualStrings("1", raw_lit);

        const binding_token = parser.tokens.get(@intFromEnum(var_definition.data.var_definition.binding));
        const raw_binding = try tokenizer.read_raw_token(binding_token);
        try std.testing.expectEqualStrings("asdf", raw_binding);
    }
    {}
}
