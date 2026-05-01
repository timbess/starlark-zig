const std = @import("std");
const Tokenizer = @This();

source: [:0]const u8,
index: usize = 0,
indent_stack: [32]usize = undefined,
indent_depth: usize = 0,
paren_depth: usize = 0,
/// When > 0, we're in the middle of emitting block_end tokens for a multi-
/// level dedent. Each emission pops one level off `indent_stack` and
/// decrements this counter.
pending_dedents: usize = 0,
/// The opening quote character of the string currently being tokenized.
string_quote: u8 = 0,
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
        l_bracket,
        r_bracket,
        l_brace,
        r_brace,
        comma,
        string,
        number_literal,
        eof,
        invalid,
        eq,
        eq_eq,
        bang_eq,
        lt,
        lt_eq,
        gt,
        gt_eq,
        block_start,
        block_end,
        keyword_def,
        keyword_return,
        keyword_and,
        keyword_or,
        keyword_not,
        keyword_true,
        keyword_false,
        keyword_if,
        keyword_elif,
        keyword_else,
        keyword_for,
        keyword_in,
        keyword_break,
        keyword_continue,
        keyword_none,
        keyword_pass,
        percent,
        percent_eq,
        colon,
        star,
        star_eq,
        starstar,
        minus,
        minus_eq,
        plus,
        plus_eq,
        slash,
        slash_eq,
        slash_slash,
        slash_slash_eq,
        pipe,
        pipe_eq,
        dot,
        keyword_lambda,
        keyword_while,
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
        .{ "not", .keyword_not },
        .{ "True", .keyword_true },
        .{ "False", .keyword_false },
        .{ "if", .keyword_if },
        .{ "elif", .keyword_elif },
        .{ "else", .keyword_else },
        .{ "for", .keyword_for },
        .{ "in", .keyword_in },
        .{ "break", .keyword_break },
        .{ "continue", .keyword_continue },
        .{ "None", .keyword_none },
        .{ "pass", .keyword_pass },
        .{ "lambda", .keyword_lambda },
        .{ "while", .keyword_while },
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

/// Current open block's indent width, or 0 at module level.
inline fn currentIndent(self: *const Tokenizer) usize {
    return if (self.indent_depth == 0) 0 else self.indent_stack[self.indent_depth - 1];
}

// TODO: Does this need to be fallible if tokens can be invalid? Maybe invalid shouldn't be a tag.
pub fn next(self: *Tokenizer) Error!Token {
    // Emit any pending dedent block_end tokens before processing input. Each
    // pending emit pops one level off the indent stack.
    if (self.pending_dedents > 0) {
        self.pending_dedents -= 1;
        if (self.indent_depth > 0) self.indent_depth -= 1;
        return .{
            .tag = .block_end,
            .loc = .{ .start = self.index, .end = self.index },
        };
    }

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
                    if (self.indent_depth > 0) {
                        self.indent_depth -= 1;
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
                    self.index += 1;
                    if (self.paren_depth > 0) {
                        // Inside parens/brackets/braces: skip indentation processing
                        result.loc.start = self.index;
                        continue :state .start;
                    }
                    indent = 0;
                    continue :state .indentation;
                },
                // Should only handle mid-line WS.
                '\t', ' ' => {
                    self.index += 1;
                    result.loc.start = self.index;
                    continue :state .start;
                },
                '#' => {
                    // Skip comment until end of line
                    while (self.source[self.index] != 0 and self.source[self.index] != '\n') {
                        self.index += 1;
                    }
                    result.loc.start = self.index;
                    continue :state .start;
                },
                '=' => {
                    self.index += 1;
                    result.tag = .eq;
                    continue :state .operator_two_char;
                },
                '!' => {
                    self.index += 1;
                    if (self.source[self.index] == '=') {
                        self.index += 1;
                        result.tag = .bang_eq;
                    } else {
                        continue :state .invalid;
                    }
                },
                '<' => {
                    self.index += 1;
                    result.tag = .lt;
                    continue :state .operator_two_char;
                },
                '>' => {
                    self.index += 1;
                    result.tag = .gt;
                    continue :state .operator_two_char;
                },
                '[' => {
                    self.index += 1;
                    self.paren_depth += 1;
                    result.tag = .l_bracket;
                },
                ']' => {
                    self.index += 1;
                    if (self.paren_depth > 0) self.paren_depth -= 1;
                    result.tag = .r_bracket;
                },
                '{' => {
                    self.index += 1;
                    self.paren_depth += 1;
                    result.tag = .l_brace;
                },
                '}' => {
                    self.index += 1;
                    if (self.paren_depth > 0) self.paren_depth -= 1;
                    result.tag = .r_brace;
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
                '/' => {
                    self.index += 1;
                    result.tag = .slash;
                    continue :state .operator_two_char;
                },
                '|' => {
                    self.index += 1;
                    result.tag = .pipe;
                    continue :state .operator_two_char;
                },
                '0'...'9' => {
                    continue :state .number;
                },
                'a'...'z', 'A'...'Z', '_' => {
                    self.index += 1;
                    continue :state .identifier;
                },
                '%' => {
                    self.index += 1;
                    result.tag = .percent;
                    continue :state .operator_two_char;
                },
                '.' => {
                    self.index += 1;
                    result.tag = .dot;
                },
                ',' => {
                    self.index += 1;
                    result.tag = .comma;
                },
                '"', '\'' => {
                    self.string_quote = self.source[self.index];
                    continue :state .string;
                },
                '(' => {
                    self.index += 1;
                    self.paren_depth += 1;
                    result.tag = .l_paren;
                },
                ')' => {
                    self.index += 1;
                    if (self.paren_depth > 0) self.paren_depth -= 1;
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
                ' ' => {
                    indent += 1;
                    self.index += 1;
                    continue :state .indentation;
                },
                // A tab advances to the next column that is a multiple of 8.
                '\t' => {
                    indent = (indent / 8 + 1) * 8;
                    self.index += 1;
                    continue :state .indentation;
                },
                else => {
                    const cur = self.currentIndent();
                    if (indent == cur) {
                        // Same indent as current block — no token, just continue
                        // tokenizing the line.
                        result.loc.start = self.index;
                        continue :state .start;
                    } else if (indent > cur) {
                        // Open a new block at this indent level.
                        if (self.indent_depth >= self.indent_stack.len) continue :state .invalid;
                        self.indent_stack[self.indent_depth] = indent;
                        self.indent_depth += 1;
                        result.tag = .block_start;
                    } else {
                        // Close one or more blocks. Pop until the top matches
                        // `indent`. The new indent must match an existing
                        // open level exactly (Python/Starlark rule). Emit one
                        // block_end now and queue the rest as pending dedents.
                        result.loc.start = self.index;
                        var pops: usize = 0;
                        while (self.indent_depth > pops and self.indent_stack[self.indent_depth - 1 - pops] > indent) {
                            pops += 1;
                        }
                        const new_top = if (self.indent_depth == pops) 0 else self.indent_stack[self.indent_depth - 1 - pops];
                        if (new_top != indent) continue :state .invalid;
                        // Pop one level now, queue the rest.
                        self.indent_depth -= 1;
                        if (pops > 1) self.pending_dedents = pops - 1;
                        result.tag = .block_end;
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
            const c = self.source[self.index];
            if (c == self.string_quote) {
                self.index += 1;
                result.tag = .string;
            } else switch (c) {
                0 => {
                    if (self.index != self.source.len) {
                        continue :state .invalid;
                    } else {
                        result.tag = .invalid;
                    }
                },
                '\n' => result.tag = .invalid,
                '\\' => continue :state .string_escape_char,
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
                '/' => {
                    if (result.tag != .slash) continue :state .invalid;
                    self.index += 1;
                    result.tag = .slash_slash;
                    continue :state .operator_two_char;
                },
                '=' => {
                    switch (result.tag) {
                        .minus => result.tag = .minus_eq,
                        .plus => result.tag = .plus_eq,
                        .star => result.tag = .star_eq,
                        .slash => result.tag = .slash_eq,
                        .slash_slash => result.tag = .slash_slash_eq,
                        .percent => result.tag = .percent_eq,
                        .pipe => result.tag = .pipe_eq,
                        .eq => result.tag = .eq_eq,
                        .lt => result.tag = .lt_eq,
                        .gt => result.tag = .gt_eq,
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
    {
        const code: [:0]const u8 =
            \\{"foo": 123}
        ;

        var tokenizer = Tokenizer.init(code);

        // zig fmt: off
        try Test.expectToken(&tokenizer, .{ .tag = .l_brace,        .text = "{" });
        try Test.expectToken(&tokenizer, .{ .tag = .string,         .text = "\"foo\"" });
        try Test.expectToken(&tokenizer, .{ .tag = .colon,          .text = ":" });
        try Test.expectToken(&tokenizer, .{ .tag = .number_literal, .text = "123" });
        try Test.expectToken(&tokenizer, .{ .tag = .r_brace,        .text = "}" });
        // zig fmt: on
    }
    {
        const code: [:0]const u8 =
            \\'foo'
            \\"can't"
            \\'say "hi"'
            \\'don\'t'
            \\"a\"b"
            \\''
            \\""
        ;

        var tokenizer = Tokenizer.init(code);

        // zig fmt: off
        try Test.expectToken(&tokenizer, .{ .tag = .string, .text = "'foo'" });
        try Test.expectToken(&tokenizer, .{ .tag = .string, .text = "\"can't\"" });
        try Test.expectToken(&tokenizer, .{ .tag = .string, .text = "'say \"hi\"'" });
        try Test.expectToken(&tokenizer, .{ .tag = .string, .text = "'don\\'t'" });
        try Test.expectToken(&tokenizer, .{ .tag = .string, .text = "\"a\\\"b\"" });
        try Test.expectToken(&tokenizer, .{ .tag = .string, .text = "''" });
        try Test.expectToken(&tokenizer, .{ .tag = .string, .text = "\"\"" });
        try Test.expectToken(&tokenizer, .{ .tag = .eof,    .text = "" });
        // zig fmt: on
    }
    {
        const code: [:0]const u8 =
            \\{'k': "v"}
        ;

        var tokenizer = Tokenizer.init(code);

        // zig fmt: off
        try Test.expectToken(&tokenizer, .{ .tag = .l_brace, .text = "{" });
        try Test.expectToken(&tokenizer, .{ .tag = .string,  .text = "'k'" });
        try Test.expectToken(&tokenizer, .{ .tag = .colon,   .text = ":" });
        try Test.expectToken(&tokenizer, .{ .tag = .string,  .text = "\"v\"" });
        try Test.expectToken(&tokenizer, .{ .tag = .r_brace, .text = "}" });
        // zig fmt: on
    }
    {
        // Tab indentation: each tab advances to the next multiple of 8 columns.
        // First body uses 4 spaces (indent_size=4); inner body uses one tab
        // (column 8) which is a valid 2x indent_size.
        const code: [:0]const u8 = "def f():\n    if x:\n\tpass\n";

        var tokenizer = Tokenizer.init(code);

        // zig fmt: off
        try Test.expectToken(&tokenizer, .{ .tag = .keyword_def,    .text = "def" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "f" });
        try Test.expectToken(&tokenizer, .{ .tag = .l_paren,        .text = "(" });
        try Test.expectToken(&tokenizer, .{ .tag = .r_paren,        .text = ")" });
        try Test.expectToken(&tokenizer, .{ .tag = .colon,          .text = ":" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_start,    .text = "\n    " });
        try Test.expectToken(&tokenizer, .{ .tag = .keyword_if,     .text = "if" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "x" });
        try Test.expectToken(&tokenizer, .{ .tag = .colon,          .text = ":" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_start,    .text = "\n\t" });
        try Test.expectToken(&tokenizer, .{ .tag = .keyword_pass,   .text = "pass" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_end,      .text = "" });
        try Test.expectToken(&tokenizer, .{ .tag = .block_end,      .text = "" });
        try Test.expectToken(&tokenizer, .{ .tag = .eof,            .text = "" });
        // zig fmt: on
    }
    {
        const code: [:0]const u8 =
            \\a / b
            \\a // b
            \\a | b
            \\lambda x: x
        ;

        var tokenizer = Tokenizer.init(code);

        // zig fmt: off
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "a" });
        try Test.expectToken(&tokenizer, .{ .tag = .slash,          .text = "/" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "b" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "a" });
        try Test.expectToken(&tokenizer, .{ .tag = .slash_slash,    .text = "//" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "b" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "a" });
        try Test.expectToken(&tokenizer, .{ .tag = .pipe,           .text = "|" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "b" });
        try Test.expectToken(&tokenizer, .{ .tag = .keyword_lambda, .text = "lambda" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "x" });
        try Test.expectToken(&tokenizer, .{ .tag = .colon,          .text = ":" });
        try Test.expectToken(&tokenizer, .{ .tag = .identifier,     .text = "x" });
        try Test.expectToken(&tokenizer, .{ .tag = .eof,            .text = "" });
        // zig fmt: on
    }
}
