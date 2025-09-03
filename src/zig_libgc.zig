const std = @import("std");
const lib = @cImport({
    @cInclude("gc.h");
});

const Self = @This();

const vtable = std.mem.Allocator.VTable{
    .alloc = alloc,
    .resize = resize,
    .free = free,
    .remap = remap,
};

pub fn setFindLeak(v: bool) void {
    lib.GC_set_find_leak(@intFromBool(v));
}

pub fn collectALittle() void {
    _ = lib.GC_collect_a_little();
}

pub fn collect() void {
    lib.GC_gcollect();
}

pub fn dump() void {
    lib.GC_dump();
}

pub fn should_invoke_finalizers() bool {
    return lib.GC_should_invoke_finalizers() != 0;
}

pub fn init() void {
    if (lib.GC_is_init_called() == 0) {
        // TODO: Not clear on why this is necessary, but it fails to init otherwise
        lib.GC_clear_roots();
        // lib.GC_clear_exclusion_table();
        lib.GC_init();
        lib.GC_set_finalize_on_demand(@intFromBool(true));
    }
}

pub fn getHeapUse() usize {
    return lib.GC_get_memory_use();
}

pub fn getHeapSize() usize {
    return lib.GC_get_heap_size();
}

pub fn allocator() std.mem.Allocator {
    init();
    return .{ .ptr = undefined, .vtable = &vtable };
}

pub inline fn shouldInvokeFinalizers() bool {
    return lib.GC_should_invoke_finalizers() != 0;
}

pub fn invokeFinalizers() i32 {
    if (shouldInvokeFinalizers()) {
        return lib.GC_invoke_finalizers();
    }
    return 0;
}

inline fn twoPtrFn(a: type, b: type) type {
    const aInfo = @typeInfo(a);
    const bInfo = @typeInfo(b);
    if (aInfo != .pointer) {
        @compileError("Expected a pointer type and got: " ++ @typeName(a));
    }
    if (bInfo != .pointer) {
        @compileError("Expected a pointer type and got: " ++ @typeName(a));
    }
    return *const fn (
        *aInfo.pointer.child,
        *bInfo.pointer.child,
    ) void;
}

pub fn registerFinalizer(ptr: anytype, finalizer_data: anytype, comptime finalizer: twoPtrFn(@TypeOf(ptr), @TypeOf(finalizer_data))) void {
    const wrapper = struct {
        pub fn wrap(ptr_raw: ?*anyopaque, finalizer_data_raw: ?*anyopaque) callconv(.c) void {
            finalizer(@ptrCast(@alignCast(ptr_raw)), @ptrCast(@alignCast(finalizer_data_raw)));
        }
    };
    lib.GC_register_finalizer_ignore_self(ptr, &wrapper.wrap, finalizer_data, null, null);
}

fn alloc(_: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
    _ = ret_addr;
    if (len == 0) return &[0]u8{};

    const ptr = lib.GC_memalign(alignment.toByteUnits(), len);
    if (ptr) |p| {
        return @ptrCast(p);
    }
    return null;
}

fn alignedFree(ptr: [*]u8) void {
    lib.GC_free(ptr);
}

fn resize(_: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
    _ = ret_addr;
    _ = alignment;
    if (new_len <= memory.len) {
        return true;
    }

    return new_len <= lib.GC_size(memory.ptr);
}

fn remap(_: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
    _ = memory;
    _ = alignment;
    _ = new_len;
    _ = ret_addr;
    return null;
    // TODO: This breaks alignment somehow
    // return @ptrCast(lib.GC_realloc(@ptrCast(memory.ptr), alignment.forward(new_len)));
}

fn free(_: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
    _ = ret_addr;
    _ = alignment;
    lib.GC_free(memory.ptr);
}

const Node = struct {
    value: i32,
    next: ?*Node = null,
};

test "GcAllocator" {
    const a = allocator();

    try std.heap.testAllocator(a);
    try std.heap.testAllocatorAligned(a);
    try std.heap.testAllocatorAlignedShrink(a);

    // TODO: look into this
    // try std.heap.testAllocatorLargeAlignment(a);
}

const Finalizer = struct {
    called: bool = false,
    fn finalize(obj: *Node, self: *Finalizer) void {
        _ = obj;
        self.called = true;
    }
};

test "GcAllocator Finalizer" {
    var finalizer = Finalizer{};

    // If this gets inlined, the pointers seem to not get erased from the stack in debug builds
    try (struct {
        finalizer: *Finalizer,
        noinline fn new_stack_frame(self: @This()) !void {
            const a = allocator();
            const node: *Node = try a.create(Node);
            registerFinalizer(node, self.finalizer, &Finalizer.finalize);
        }
    }{ .finalizer = &finalizer }).new_stack_frame();

    collect();

    {
        const c = invokeFinalizers();
        try std.testing.expectEqual(0, c);
    }

    try std.testing.expect(!finalizer.called);
    collect();

    {
        const c = invokeFinalizers();
        try std.testing.expectEqual(1, c);
    }

    try std.testing.expect(finalizer.called);
}
