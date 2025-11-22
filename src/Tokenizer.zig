const std = @import("std");
const Tokenizer = @This();

source: [:0]const u8,
index: usize = 0,
indent_level: usize = 0,
indent_size: usize = 0,
state: State = .start,

const State = enum {
    start,
    identifier,
    indentation,
    invalid,
    number,
    number_w_decimal,
    string,
    string_escape_char,
    operator_two_char,

    const initial_state: State = .start;
};

pub const Token = struct {
    tag: Tag,
    loc: Loc,

    pub const Index = enum(u32) { none = std.math.maxInt(u32), _ };

    pub const Tag = enum {
        identifier,
        l_paren,
        r_paren,
        comma,
        string,
        number_literal,
        eof,
        invalid,
        eq,
        block_start,
        block_end,
        keyword_def,
        keyword_return,
        keyword_and,
        keyword_or,
        colon,
        star,
        starstar,
        minus,
        minus_eq,
        plus,
        plus_eq,
        dot,
    };

    pub const Loc = struct {
        start: u64,
        end: u64,
    };

    const keywords = std.StaticStringMap(Tag).initComptime(.{
        .{ "def", .keyword_def },
        .{ "return", .keyword_return },
        .{ "and", .keyword_and },
        .{ "or", .keyword_or },
    });

    pub fn getKeyword(name: []const u8) ?Tag {
        return keywords.get(name);
    }
};

const Error = error{};

pub fn init(source: [:0]const u8) Tokenizer {
    return Tokenizer{
        .source = source,
    };
}

// TODO: Does this need to be fallible if tokens can be invalid? Maybe invalid shouldn't be a tag.
pub fn next(self: *Tokenizer) Error!Token {
    var result: Token = .{ .tag = undefined, .loc = .{
        .start = self.index,
        .end = undefined,
    } };
    var indent: usize = 0;
    state: switch (State.initial_state) {
        .start => {
            switch (self.source[self.index]) {
                // String is null terminated to make EOF handling nicer.
                0 => {
                    if (self.indent_size > 0 and self.indent_level > 0) {
                        self.indent_level -= self.indent_size;
                        return .{
                            .tag = .block_end,
                            .loc = .{
                                .start = self.index,
                                .end = self.index,
                            },
                        };
                    }

                    if (self.index == self.source.len) {
                        return .{
                            .tag = .eof,
                            .loc = .{
                                .start = self.index,
                                .end = self.index,
                            },
                        };
                    } else {
                        continue :state .invalid;
                    }
                },
                '\n', '\r' => {
                    indent = 0;
                    self.index += 1;
                    continue :state .indentation;
                },
                // Should only handle mid-line WS.
                '\t', ' ' => {
                    self.index += 1;
                    result.loc.start = self.index;
                    continue :state .start;
                },
                '=' => {
                    // TODO: Handle ==
                    self.index += 1;
                    result.tag = .eq;
                },
                '*' => {
                    self.index += 1;
                    result.tag = .star;
                    continue :state .operator_two_char;
                },
                '+' => {
                    self.index += 1;
                    result.tag = .plus;
                    continue :state .operator_two_char;
                },
                '-' => {
                    self.index += 1;
                    result.tag = .minus;
                    continue :state .operator_two_char;
                },
                '0'...'9' => {
                    continue :state .number;
                },
                'a'...'z', 'A'...'Z', '_' => {
                    self.index += 1;
                    continue :state .identifier;
                },
                '.' => {
                    self.index += 1;
                    result.tag = .dot;
                },
                ',' => {
                    self.index += 1;
                    result.tag = .comma;
                },
                '"' => {
                    continue :state .string;
                },
                '(' => {
                    self.index += 1;
                    result.tag = .l_paren;
                },
                ')' => {
                    self.index += 1;
                    result.tag = .r_paren;
                },
                ':' => {
                    self.index += 1;
                    result.tag = .colon;
                },
                else => continue :state .invalid,
            }
        },
        // Consume prefixed indentation to start/end blocks.
        .indentation => {
            switch (self.source[self.index]) {
                ' ', '\t' => {
                    indent += 1;
                    self.index += 1;
                    continue :state .indentation;
                },
                else => {
                    if (indent == self.indent_level) {
                        // skip WS when next line is equal indent
                        result.loc.start = self.index;
                        continue :state .start;
                    } else if (indent < self.indent_level) {
                        // We don't care about capturing ws when blocks end
                        result.loc.start = self.index;
                        self.indent_level = indent;
                        result.tag = .block_end;
                    } else {
                        self.indent_level = indent;
                        // TODO: This is likely wrong, detecting indentation needs to be smarter.
                        if (self.indent_size == 0) {
                            self.indent_size = indent;
                            // TODO: Is there a way to do this without a runtime known rhs?
                        } else if (@mod(indent, self.indent_size) != 0) {
                            continue :state .invalid;
                        }
                        result.tag = .block_start;
                    }
                },
            }
        },
        .identifier => {
            switch (self.source[self.index]) {
                'a'...'z', 'A'...'Z', '_' => {
                    self.index += 1;
                    continue :state .identifier;
                },
                else => {
                    const name = self.source[result.loc.start..self.index];
                    if (Token.getKeyword(name)) |t| {
                        result.tag = t;
                    } else {
                        result.tag = .identifier;
                    }
                },
            }
        },
        .invalid => {
            self.index += 1;
            switch (self.source[self.index]) {
                0 => if (self.index == self.source.len) {
                    result.tag = .invalid;
                } else {
                    continue :state .invalid;
                },
                '\n' => result.tag = .invalid,
                else => continue :state .invalid,
            }
        },
        .number, .number_w_decimal => |v| {
            switch (self.source[self.index]) {
                '0'...'9', '.' => |d| {
                    // TODO: Is there a way to do this without a dynamic branch?
                    if (v == .number and d == '.') continue :state .number_w_decimal;
                    self.index += 1;
                    continue :state v;
                },
                else => {
                    result.tag = .number_literal;
                },
            }
        },
        .string => {
            self.index += 1;
            switch (self.source[self.index]) {
                0 => {
                    if (self.index != self.source.len) {
                        continue :state .invalid;
                    } else {
                        result.tag = .invalid;
                    }
                },
                '\n' => result.tag = .invalid,
                '\\' => continue :state .string_escape_char,
                '"' => {
                    self.index += 1;
                    result.tag = .string;
                },
                // Control characters are not allowed in strings.
                0x01...0x09, 0x0b...0x1f, 0x7f => {
                    continue :state .invalid;
                },
                else => continue :state .string,
            }
        },
        .string_escape_char => {
            // Ignore whatever comes next and parse it later.
            self.index += 1;
            switch (self.source[self.index]) {
                0, '\n' => result.tag = .invalid,
                else => continue :state .string,
            }
        },
        .operator_two_char => {
            switch (self.source[self.index]) {
                0 => {
                    if (self.index != self.source.len) {
                        continue :state .invalid;
                    } else {
                        result.tag = .invalid;
                    }
                },
                '*' => {
                    if (result.tag != .star) continue :state .invalid;
                    self.index += 1;
                    result.tag = .starstar;
                },
                '=' => {
                    switch (result.tag) {
                        .minus => result.tag = .minus_eq,
                        .plus => result.tag = .plus_eq,
                        else => continue :state .invalid,
                    }
                    self.index += 1;
                },
                else => {
                    // Not a two-char operator, return the single-char operator token
                },
            }
        },
    }
    result.loc.end = self.index;
    return result;
}

pub fn read_raw_token(self: *Tokenizer, token: Token) ![]const u8 {
    return self.source[token.loc.start..token.loc.end];
}

test Tokenizer {
    const Case = struct {
        tag: Token.Tag,
        text: []const u8,
    };

    const Test = struct {
        fn expectToken(t: *Tokenizer, expected: Case) !void {
            const token = try t.next();

            try std.testing.expectEqual(expected.tag, token.tag);
            try std.testing.expectEqualSlices(
                u8,
                expected.text,
                try t.read_raw_token(token),
            );
        }
    };

    // Many of these have invalid syntax, but can still be tokenized.
    {
        const code: [:0]const u8 =
            \\asdf = 1
            \\foo = 2.0
        ;

        var tokenizer = Tokenizer.init(code);

        // zig fmt: off
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "asdf" });
        try Test.expectToken(&tokenizer, .{ .tag = .eq,             .text = "=" });
        try Test.expectToken(&tokenizer, .{ .tag = .number_literal, .text = "1" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "foo" });
        try Test.expectToken(&tokenizer, .{ .tag = .eq,             .text = "=" });
        try Test.expectToken(&tokenizer, .{ .tag = .number_literal, .text = "2.0" });
        try Test.expectToken(&tokenizer, .{ .tag = .eof,            .text = "" });
        // zig fmt: on
    }
    {
        const code: [:0]const u8 =
            \\asdf = 1
            \\  foo = 2.0
            \\    bar = 3.0
        ;

        var tokenizer = Tokenizer.init(code);

        // zig fmt: off
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "asdf" });
        try Test.expectToken(&tokenizer, .{ .tag = .eq,             .text = "=" });
        try Test.expectToken(&tokenizer, .{ .tag = .number_literal, .text = "1" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_start,    .text = "\n  " });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "foo" });
        try Test.expectToken(&tokenizer, .{ .tag = .eq,             .text = "=" });
        try Test.expectToken(&tokenizer, .{ .tag = .number_literal, .text = "2.0" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_start,    .text = "\n    " });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "bar" });
        try Test.expectToken(&tokenizer, .{ .tag = .eq,             .text = "=" });
        try Test.expectToken(&tokenizer, .{ .tag = .number_literal, .text = "3.0" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_end,      .text = "" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_end,      .text = "" });
        try Test.expectToken(&tokenizer, .{ .tag = .eof,            .text = "" });
        // zig fmt: on
    }
    {
        const code: [:0]const u8 =
            \\asdf = 1
            \\  foo = 2.0
            \\    bar = 3.0
            \\  1
            \\
            \\def my_fn(name):
            \\  println(name)
            \\  return 1
        ;

        var tokenizer = Tokenizer.init(code);

        // zig fmt: off
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "asdf" });
        try Test.expectToken(&tokenizer, .{ .tag = .eq,             .text = "=" });
        try Test.expectToken(&tokenizer, .{ .tag = .number_literal, .text = "1" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_start,    .text = "\n  " });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "foo" });
        try Test.expectToken(&tokenizer, .{ .tag = .eq,             .text = "=" });
        try Test.expectToken(&tokenizer, .{ .tag = .number_literal, .text = "2.0" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_start,    .text = "\n    " });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "bar" });
        try Test.expectToken(&tokenizer, .{ .tag = .eq,             .text = "=" });
        try Test.expectToken(&tokenizer, .{ .tag = .number_literal, .text = "3.0" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_end,      .text = "" });
        try Test.expectToken(&tokenizer, .{ .tag = .number_literal, .text = "1" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_end,      .text = "" });
        try Test.expectToken(&tokenizer, .{ .tag = .keyword_def,    .text = "def" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "my_fn" });
        try Test.expectToken(&tokenizer, .{ .tag = .l_paren,        .text = "(" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "name" });
        try Test.expectToken(&tokenizer, .{ .tag = .r_paren,        .text = ")" });
        try Test.expectToken(&tokenizer, .{ .tag = .colon,          .text = ":" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_start,    .text = "\n  " });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "println" });
        try Test.expectToken(&tokenizer, .{ .tag = .l_paren,        .text = "(" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "name" });
        try Test.expectToken(&tokenizer, .{ .tag = .r_paren,        .text = ")" });
        try Test.expectToken(&tokenizer, .{ .tag = .keyword_return, .text = "return" });
        try Test.expectToken(&tokenizer, .{ .tag = .number_literal, .text = "1" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_end,      .text = "" });
        try Test.expectToken(&tokenizer, .{ .tag = .eof,            .text = "" });
        // zig fmt: on
    }
}
