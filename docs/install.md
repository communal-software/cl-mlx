# Installation

Build the native libraries explicitly, then load `cl-mlx`. ASDF does not download or compile native code.

## Requirements

macOS Apple Silicon, Xcode with Metal tools, CMake, Git and SBCL. Lisp dependencies are CFFI, trivial-garbage and Bordeaux Threads.

```sh
git clone https://github.com/communal-software/cl-mlx ~/quicklisp/local-projects/cl-mlx
cd ~/quicklisp/local-projects/cl-mlx
./build-native.sh
```

```lisp
(asdf:load-system "cl-mlx")
(cl-mlx:initialize)
(cl-mlx:available-p)
```

The build uses mlx-c v0.7.0 at commit `a341b4925024b88b2c593468f16e12f5e17315da`, with MLX v0.32.2. Native artifacts live in gitignored `.native/`. Set `CL_MLX_BUILD_JOBS` to control build parallelism; the default is four jobs.

## Library locations

The default shim is `.native/shim/libclmlx.dylib`; the Metal library is `.native/build/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib`.

Override these with `CL_MLX_LIBRARY` and `CL_MLX_METALLIB`, or `initialize`'s `:library` and `:metallib` arguments. Keep the shim, mlx-c and MLX from the same pinned build.

## Tests

```sh
sbcl --non-interactive --eval '(require :asdf)' --eval '(asdf:test-system "cl-mlx")'
```

Tests execute real CPU/GPU graphs, fused operations, 4-bit matmul, symmetric Q8 groups, cleanup and native error paths. Loading the Lisp system alone does not load the native library. Rebuild the shim after updating bindings that add native entry points.
