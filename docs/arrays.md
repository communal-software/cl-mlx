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
| Symmetric Q8 groups | `symmetric-quantize`, `symmetric-dequantize` |
| Execution | `eval`, `to-vector`, `with-device`, `device`, `synchronize` |

Shapes and slice vectors use axis order, with exclusive slice stops. `from-data` accepts flat Lisp sequences for F32/F64/S32/U32/U8; `from-pointer` also accepts raw F16/BF16 encodings. Device-specific dtype restrictions are enforced by MLX.

`to-vector` requires row-contiguous F32 data. It evaluates by default; `:evaluate nil` only reads an already evaluated array. Use `astype` to convert another dtype and `eval` to evaluate it.

## Symmetric Q8

`symmetric-quantize` returns two owned arrays: S8 codes with the input shape, and F32 scales with the last dimension replaced by `ceiling(dimension / 32)`. Groups stay within each row of the final axis; partial groups are supported.

```lisp
(cl-mlx:with-array-scope ()
  (multiple-value-bind (codes scales)
      (cl-mlx:symmetric-quantize (cl-mlx:from-data '(127 -127 0.5 1.5) '(1 4)))
    (cl-mlx:to-vector (cl-mlx:symmetric-dequantize codes scales))))
; => #(127.0 -127.0 0.0 2.0)
```

Input converts to F32. Each group uses scale `max(abs(group))/127`, signed codes in `[-127,127]` and nearest rounding with ties to even. Zero groups use scale one; other scales have a floor of the smallest normal F32 value. Dequantization returns row-contiguous F32 data. This encoding is separate from affine quantized weights and GGUF Q8_0.

Encoding checks finiteness through a device reduction and reads its scalar result, synchronizing that check. The payload stays on the selected device. Graph construction for decoding remains lazy. Both outputs follow the ordinary scope/retain/release rules; an invalid final dimension, non-finite input or mismatched decoder shapes signals `mlx-error`.

## Devices and threads

GPU execution is the default. `with-device (:cpu)` selects CPU graph construction within its body. The shim creates a distinct stream per native thread and device. `synchronize` waits for the current thread's stream. Arrays may cross threads; do not concurrently release or mutate the same array handle.

Native calls preserve the calling thread's floating-point environment. Errors are captured in native thread-local storage and signalled as `mlx-error` after the call returns.[^handler]

[^handler]: mlx-c's error handler is process-global. The shim installs its handler once; another consumer sharing the same native mlx-c library must not replace it while cl-mlx is active.
