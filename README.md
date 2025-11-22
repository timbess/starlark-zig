# starlark-zig

Parser/Compiler/Runtime for Starlark written in Zig with bdwgc as the garbage collector. Still very much WIP.

To run it:

```sh
# This is a checked in test file to play with.
zig build run <./test.star
```

To do a watch build of that, run:

```sh
zig build run-test-star --watch
```

That will rerun the build/interpreter on every change to a source file or `test.star`.

# TODO

- [ ] Make pool for small numbers/strings.
- [ ] Figure out a good multi-threading model.
- [ ] Test long running processes and GC.
- [ ] Parse/Compile attribute access.
