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
    };
};

pub const Node = struct {
    main_token: Token.Index,
    data: Data,

    pub const Index = enum(u32) { none = std.math.maxInt(u32), _ };

    const Tag = enum {
        root,
        var_definition,
        def_proto,
        literal,
    };

    pub const Data = union(Tag) {
        root: struct {
            statements: []const Node.Index,
        },
        var_definition: struct {
            binding: Token.Index,
            value: Node.Index,
        },
        def_proto: struct {
            name: Token.Index,
            args: []const Node.Index,
            body: Node.Index,
        },
        literal: void, // Use token
    };
};

test Ast {
    var ast = try Ast.parse(std.testing.allocator,
        \\asdf = 1
    );
    defer ast.deinit();
}
