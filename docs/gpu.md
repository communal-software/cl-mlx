# Custom Metal kernels

`cl-mlx/gpu` runs [cl-gpu](https://github.com/communal-software/cl-gpu) DSL kernels as lazy MLX operations. Inputs and outputs are ordinary MLX arrays. Load this optional system after rebuilding the native shim:

```lisp
(asdf:load-system "cl-mlx/gpu")
```

The [runnable vector example](../examples/custom-kernels.lisp) composes a custom kernel with ordinary MLX addition and evaluates the final result once.

## Define a wrapper

```lisp
(cl-gpu:define-kernel scale-values
    ((out (cl-gpu:ptr cl-gpu:f32 :global))
     (x (cl-gpu:ptr cl-gpu:f32 :global)) (gain cl-gpu:f32))
  (let ((i (cl-gpu:global-id 0)))
    (setf (aref out i) (* (aref x i) gain))))

(cl-mlx/gpu:define-kernel scale-array
    (cl-gpu:find-kernel 'scale-values) (x gain)
  :outputs ((out (cl-mlx:shape x)))
  :grid (cl-mlx:size x) :group 32)

(cl-mlx:with-array-scope ()
  (let* ((x (cl-mlx:from-data '(1 2 3) '(3)))
         (out (first (scale-array x 2.0))))
    (cl-mlx:to-vector out)))
; => #(2.0 4.0 6.0)
```

The wrapper uses required arguments named after the DSL inputs and value parameters; names match across packages. Output declarations name DSL output parameters and contain shape expressions. All expressions run in the wrapper's argument bindings. The kernel expression is resolved on each call, so redefining the DSL kernel selects a fresh specialization.

Calls return a list of lazy arrays in output declaration order. Shapes describe scalar storage, not DSL vector counts. Grid and group each contain one to three positive int32 dimensions, or one integer; missing dimensions are one. Grid counts threads. Kernels must guard accesses when their launch extends beyond the data.

## Argument layout

| DSL parameter | Wrapper value |
|---|---|
| Scalar buffer pointer | MLX array of matching dtype and positive rank |
| Scalar value | Lisp value or matching zero-dimensional MLX array |
| Value `vec T N` | Sequence of N values or MLX array with shape `(N)` |
| Buffer pointer to `vec T N` | MLX array with rank at least two and final dimension N |
| Atomic output pointer | Fresh MLX array with the atomic element dtype |

Supported scalar elements are F16, BF16, F32, signed/unsigned 8/16/32/64-bit integers and booleans, subject to Metal's DSL type support. Integer DSL names `i8`–`i64` map to MLX `:s8`–`:s64`. Arrays require exact dtypes; Lisp numeric values convert to their declared type. Boolean values are `t` or `nil`.

Vector buffers store adjacent scalar lanes: `(vec f32 3)` uses three floats, without Metal's native fourth padding lane. Component loads/stores and pointer offsets follow that layout. Noncontiguous inputs normalize lazily on the GPU when needed; contiguous inputs avoid that copy.[^layout]

## Initialization and templates

Read/write outputs require `:init-value`, for example `:init-value 0`. MLX allocates fresh output arrays and initializes every output to that value. Input arrays are borrowed; this API does not mutate them. Access classification is conservative: an output read after a write still requires initialization. Write-only outputs without initialization must fill every location the caller reads.

Use `:templates ((n n) (enabled enabled))` to specialize integer or boolean scalar parameters using expressions in wrapper argument bindings. Arguments with those parameter names become templates instead of runtime inputs. Integers must fit both their declared type and signed int32. Floating-point scalars and vectors remain runtime arguments.

## Ownership and execution

Outputs follow ordinary [array scope, retain and release rules](arrays.md). Graph construction does not evaluate arrays or transfer results to Lisp. Outputs remain usable after input handles are released or the kernel cache is cleared.

Specializations are cached per wrapper, source kernel identity, template values and initialization mode. Each wrapper serializes cache access and graph construction; evaluation uses the calling thread's MLX GPU stream. Different wrappers have separate locks. `clear-kernel-cache` releases cached native handles; `live-kernel-count` reports their count. Array handles and abandoned wrapper caches also have finalizers.[^cache]

## Tests

```sh
sbcl --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-mlx/gpu")'
```

Tests require Metal execution on Apple Silicon. They cover lazy graphs, typed arguments, packed vectors, normalization, shared memory, atomics, templates, source redefinition, errors and ownership. The base `cl-mlx` system does not load cl-gpu; the bridge loads no cl-gpu Metal runtime.

## Limitations

- Metal GPU execution on SBCL is verified. CPU execution and other targets are unsupported.
- The pinned MLX C API does not expose compiler options; kernels requesting `precise-math` are rejected.
- Shared/private pointer parameters, read-only atomic inputs and mixed atomic/non-atomic outputs are rejected. Static shared/local arrays remain supported.
- Pointer aliases require initializers. Wrapper lambda lists contain required arguments only.
- MLX supplies one floating-point initialization value for all outputs; per-output values and initial array contents are unsupported.
- GGUF inference kernels, GPU sampling and inference fusion are separate integrations.

[^layout]: MLX chooses constant or device address space for input buffers based on their size. The host adapter preserves that choice rather than casting away address-space or read-only qualifiers.

[^cache]: Specializations remain cached until explicitly cleared or their wrapper becomes unreachable. Clearing a cache does not force evaluation or synchronization. It serializes with graph construction through each wrapper's lock.
