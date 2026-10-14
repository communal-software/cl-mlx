# Arrays and ownership

Operations construct lazy graphs. `eval` evaluates a group of outputs together; `to-vector` copies F32 data to Lisp.

```lisp
(cl-mlx:with-array-scope ()
  (let* ((x (cl-mlx:from-data '(1 2 3 4) '(2 2)))
         (y (cl-mlx:matmul x (cl-mlx:transpose x))))
    (cl-mlx:eval y)
    (cl-mlx:to-vector y :evaluate nil)))
; => #(5.0 11.0 11.0 25.0)
```

## Lifetime

`with-array-scope` releases every array created inside it, including on errors. `retain` transfers an array out of the current scope; its caller owns it and calls `release`. Release is idempotent. Access after release signals `mlx-error`. Finalizers provide a fallback; use explicit cleanup for predictable memory use.

Graphs retain their input dependencies, so releasing an input handle does not invalidate an output graph. `from-pointer` copies data into MLX-owned memory; the source buffer can be released after the call.

`live-array-count` reports live Lisp-owned handles, not GPU bytes.

## Operations

| Capability | Functions |
|---|---|
| Construction and inspection | `from-data`, `from-pointer`, `zeros`, `shape`, `dtype`, `size` |
| Shape and indexing | `reshape`, `transpose`, `slice`, `slice-update`, `take` |
| Arithmetic | `add`, `multiply`, `matmul`, `sigmoid`, `astype` |
| Fused blocks | `rmsnorm`, `rope`, `attention` |
| Affine 4-bit weights | `quantized-matmul`, `dequantize` |
| Execution | `eval`, `to-vector`, `with-device`, `device`, `synchronize` |

Shapes and slice vectors use axis order, with exclusive slice stops. `from-data` accepts flat Lisp sequences for F32/F64/S32/U32/U8; `from-pointer` also accepts raw F16/BF16 encodings. Device-specific dtype restrictions are enforced by MLX.

`to-vector` requires row-contiguous F32 data. It evaluates by default; `:evaluate nil` only reads an already evaluated array. Use `astype` to convert another dtype and `eval` to evaluate it.

## Devices and threads

GPU execution is the default. `with-device (:cpu)` selects CPU graph construction within its body. The shim creates a distinct stream per native thread and device. `synchronize` waits for the current thread's stream. Arrays may cross threads; do not concurrently release or mutate the same array handle.

Native calls preserve the calling thread's floating-point environment. Errors are captured in native thread-local storage and signalled as `mlx-error` after the call returns.[^handler]

[^handler]: mlx-c's error handler is process-global. The shim installs its handler once; another consumer sharing the same native mlx-c library must not replace it while cl-mlx is active.
