const Error = error{OutOfMemory};

pub fn BoundedArray(n: comptime_int, T: type) type {
    return struct {
        items: [n]T,
        len: usize,

        const Self = @This();

        pub fn init() Self {
            return Self{
                .items = undefined,
                .len = 0,
            };
        }

        pub fn append(self: *Self, value: T) Error!void {
            if (self.len >= n) {
                return error.OutOfMemory;
            }
            self.items[self.len] = value;
            self.len += 1;
        }

        pub fn appendSlice(self: *Self, slice: []const T) Error!void {
            if (self.len + slice.len > n) {
                return error.OutOfMemory;
            }
            @memcpy(self.items[self.len .. self.len + slice.len], slice);
            self.len += slice.len;
        }

        pub fn get(self: *const Self, index: usize) Error!T {
            if (index >= self.len) {
                return error.OutOfMemory;
            }
            return self.items[index];
        }

        pub fn clear(self: *Self) void {
            self.len = 0;
        }
    };
}
