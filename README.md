# cl-mlx

Common Lisp bindings to MLX's C API for lazy array computation and inference on macOS Apple Silicon.

[Installation](docs/install.md) · [Arrays and ownership](docs/arrays.md) · [Coverage](docs/PLAN.md)

```sh
./build-native.sh
sbcl --non-interactive --eval '(require :asdf)' --eval '(asdf:test-system "cl-mlx")'
```

See [the lazy graph example](examples/lazy.lisp). [MIT](LICENSE).
