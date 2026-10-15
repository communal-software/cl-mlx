# Coverage

`cl-mlx` provides reusable lazy array bindings used by `cl-inference/mlx`.

| Deliverable | Status |
|---|---|
| Repository, MIT license and native build | Complete |
| Inference array operations and fused blocks | Complete |
| CPU/GPU scopes, errors and deterministic cleanup | Complete |
| Native numerical tests and lazy graph example | Complete |
| Device-side symmetric S8/F32 group-32 encoding and decoding | Complete |

## Limitations

| Deferred capability | Tracker |
|---|---|
| Remaining array operations, reductions, linalg and FFT | [#1](https://github.com/communal-software/cl-mlx/issues/1) |
| Graph transforms, differentiation and compilation | [#2](https://github.com/communal-software/cl-mlx/issues/2) |
| Native file IO, memory controls, custom streams and async evaluation | [#3](https://github.com/communal-software/cl-mlx/issues/3) |

The verified runtime is SBCL on an M1 Mac. Other Lisp implementations and platforms are not verified.
