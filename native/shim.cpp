#include <mlx/c/mlx.h>
#include <cstdio>
#include <cfenv>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>
#include <limits>

static thread_local char last_error[4096];
static std::atomic<size_t> live_arrays{0};
static void capture_error(const char* message, void*) {
  std::snprintf(last_error, sizeof(last_error), "%s", message);
}
static std::once_flag initialized;
static void begin() {
  std::call_once(initialized, [] { mlx_set_error_handler(capture_error, nullptr, nullptr); });
  last_error[0] = 0;
}
struct native_guard {
  fenv_t environment;
  native_guard() { feholdexcept(&environment); begin(); }
  ~native_guard() { fesetenv(&environment); }
};
static mlx_array a(void* p) { return {p}; }
static mlx_optional_int oi(int x) { return {x, true}; }
static mlx_optional_float of(float x) { return {x, true}; }
static void* finish(int status, mlx_array result) {
  if (status || last_error[0]) {
    char saved[4096];
    std::memcpy(saved, last_error, sizeof(saved));
    mlx_array_free(result);
    std::memcpy(last_error, saved, sizeof(saved));
    return nullptr;
  }
  if (result.ctx) ++live_arrays;
  return result.ctx;
}
struct stream_owner {
  mlx_stream stream;
  explicit stream_owner(mlx_device_type type) {
    auto dev = mlx_device_new_type(type, 0);
    stream = mlx_stream_new_device(dev);
    mlx_device_free(dev);
  }
  ~stream_owner() { mlx_stream_free(stream); }
};
static thread_local int active_device = 1;
static mlx_stream s() {
  if (active_device == 0) {
    static thread_local stream_owner owner(MLX_CPU);
    return owner.stream;
  }
  static thread_local stream_owner owner(MLX_GPU);
  return owner.stream;
}

struct array_owner {
  mlx_array value;
  array_owner() : value(mlx_array_new()) {}
  explicit array_owner(float x) : value(mlx_array_new_float32(x)) {}
  ~array_owner() { mlx_array_free(value); }
  void* release() { auto p = value.ctx; value.ctx = nullptr; ++live_arrays; return p; }
};

static int q8_group(mlx_array* result, mlx_array input, std::vector<int>& shape) {
  int rank = mlx_array_ndim(input);
  if (rank < 1 || mlx_array_shape(input)[rank - 1] <= 0) {
    capture_error("Q8 needs a positive final dimension", nullptr); return -1;
  }
  shape.assign(mlx_array_shape(input), mlx_array_shape(input) + rank);
  int axis = rank - 1, low = 0, high = (32 - shape.back() % 32) % 32;
  array_owner padded, zero(0.0f);
  if (mlx_pad(&padded.value, input, &axis, 1, &low, 1, &high, 1,
              zero.value, "constant", s())) return -1;
  auto grouped = shape;
  grouped.back() = (shape.back() + high) / 32;
  grouped.push_back(32);
  return mlx_reshape(result, padded.value, grouped.data(), grouped.size(), s());
}

static int q8_ungroup(mlx_array* result, mlx_array input, const std::vector<int>& shape) {
  auto padded = shape;
  padded.back() = ((shape.back() + 31) / 32) * 32;
  array_owner flat;
  if (mlx_reshape(&flat.value, input, padded.data(), padded.size(), s())) return -1;
  std::vector<int> start(shape.size(), 0), strides(shape.size(), 1);
  array_owner sliced;
  if (mlx_slice(&sliced.value, flat.value, start.data(), start.size(), shape.data(), shape.size(),
                 strides.data(), strides.size(), s())) return -1;
  return mlx_contiguous(result, sliced.value, false, s());
}

extern "C" {
int clmlx_symmetric_quantize(void* x, void** codes_out, void** scales_out) {
  native_guard guard;
  *codes_out = *scales_out = nullptr;
  array_owner input, finite, all;
  bool valid = false;
  if (mlx_astype(&input.value, a(x), MLX_FLOAT32, s()) ||
      mlx_isfinite(&finite.value, input.value, s()) ||
      mlx_all(&all.value, finite.value, false, s()) ||
      mlx_array_item_bool(&valid, all.value)) return -1;
  if (!valid) { capture_error("Non-finite Q8 input", nullptr); return -1; }
  array_owner grouped, magnitude, peak, raw_scale, safe_scale, is_zero, scale;
  array_owner divisor(127.0f), minimum(std::numeric_limits<float>::min()), one(1.0f), zero(0.0f);
  array_owner divided, rounded, clipped, codes, flat_codes, scales;
  array_owner low(-127.0f), high(127.0f);
  std::vector<int> shape;
  if (q8_group(&grouped.value, input.value, shape) ||
      mlx_abs(&magnitude.value, grouped.value, s()) ||
      mlx_max_axis(&peak.value, magnitude.value, -1, true, s()) ||
      mlx_divide(&raw_scale.value, peak.value, divisor.value, s()) ||
      mlx_maximum(&safe_scale.value, raw_scale.value, minimum.value, s()) ||
      mlx_equal(&is_zero.value, peak.value, zero.value, s()) ||
      mlx_where(&scale.value, is_zero.value, one.value, safe_scale.value, s()) ||
      mlx_divide(&divided.value, grouped.value, scale.value, s()) ||
      mlx_round(&rounded.value, divided.value, 0, s()) ||
      mlx_clip(&clipped.value, rounded.value, low.value, high.value, s()) ||
      mlx_astype(&codes.value, clipped.value, MLX_INT8, s()) ||
      q8_ungroup(&flat_codes.value, codes.value, shape)) return -1;
  shape.back() = (shape.back() + 31) / 32;
  if (mlx_reshape(&scales.value, scale.value, shape.data(), shape.size(), s())) return -1;
  *codes_out = flat_codes.release(); *scales_out = scales.release();
  return 0;
}

void* clmlx_symmetric_dequantize(void* codes, void* scales) {
  native_guard guard;
  array_owner grouped, scale, floats, decoded, bounded, result;
  array_owner low(-std::numeric_limits<float>::max()), high(std::numeric_limits<float>::max());
  std::vector<int> shape;
  if (q8_group(&grouped.value, a(codes), shape)) return nullptr;
  auto scale_shape = shape;
  scale_shape.back() = (shape.back() + 31) / 32;
  scale_shape.push_back(1);
  if (mlx_reshape(&scale.value, a(scales), scale_shape.data(), scale_shape.size(), s()) ||
      mlx_astype(&floats.value, grouped.value, MLX_FLOAT32, s()) ||
      mlx_multiply(&decoded.value, floats.value, scale.value, s()) ||
      mlx_clip(&bounded.value, decoded.value, low.value, high.value, s()) ||
      q8_ungroup(&result.value, bounded.value, shape)) return nullptr;
  return result.release();
}
const char* clmlx_error() { return last_error; }
size_t clmlx_live_arrays() { return live_arrays.load(); }
int clmlx_device() { return active_device; }
void clmlx_set_device(int device) { active_device = device; }
int clmlx_synchronize() { native_guard guard; return mlx_synchronize(s()); }
int clmlx_initialize(const char* metallib) {
  native_guard guard;
  return mlx_metal_set_metallib_path(metallib);
}
int clmlx_available() {
  native_guard guard; bool available = false;
  return mlx_metal_is_available(&available) == 0 && available;
}
void* clmlx_from_data(const void* data, const int* shape, int rank, int dtype) {
  native_guard guard; auto result = mlx_array_new_data(data, shape, rank, static_cast<mlx_dtype>(dtype));
  return finish(0, result);
}
int clmlx_free(void* p) {
  native_guard guard;
  int status = mlx_array_free(a(p));
  if (status == 0 && p) --live_arrays;
  return status;
}
int clmlx_rank(void* p) { native_guard guard; return mlx_array_ndim(a(p)); }
int clmlx_dim(void* p, int axis) { native_guard guard; return mlx_array_dim(a(p), axis); }
int clmlx_dtype(void* p) { native_guard guard; return mlx_array_dtype(a(p)); }
size_t clmlx_size(void* p) { native_guard guard; return mlx_array_size(a(p)); }
const float* clmlx_data_f32(void* p) {
  native_guard guard;
  auto array = a(p);
  auto rank = mlx_array_ndim(array);
  auto shape = mlx_array_shape(array);
  auto strides = mlx_array_strides(array);
  size_t expected = 1;
  for (int axis = static_cast<int>(rank) - 1; axis >= 0; --axis) {
    if (shape[axis] > 1 && strides[axis] != expected) {
      capture_error("Host transfer requires a row-contiguous array", nullptr);
      return nullptr;
    }
    expected *= shape[axis];
  }
  return mlx_array_data_float32(array);
}
int clmlx_eval(void* const* pointers, int count) {
  native_guard guard; std::vector<mlx_array> arrays;
  for (int i = 0; i < count; ++i) arrays.push_back(a(pointers[i]));
  auto vec = mlx_vector_array_new_data(arrays.data(), arrays.size());
  int status = mlx_eval(vec);
  mlx_vector_array_free(vec);
  return status;
}
void* clmlx_zeros(const int* shape, int rank, int dtype) {
  native_guard guard; auto r = mlx_array_new();
  int status = mlx_zeros(&r, shape, rank, static_cast<mlx_dtype>(dtype), s());
  return finish(status, r);
}
#define BINARY(name) \
void* clmlx_##name(void* x, void* y) { \
  native_guard guard; auto r = mlx_array_new(); \
  int status = mlx_##name(&r, a(x), a(y), s()); return finish(status, r); }
BINARY(add)
BINARY(multiply)
BINARY(matmul)
#define UNARY(name) \
void* clmlx_##name(void* x) { \
  native_guard guard; auto r = mlx_array_new(); \
  int status = mlx_##name(&r, a(x), s()); return finish(status, r); }
UNARY(sigmoid)
void* clmlx_astype(void* x, int dtype) {
  native_guard guard; auto r = mlx_array_new();
  int status = mlx_astype(&r, a(x), static_cast<mlx_dtype>(dtype), s()); return finish(status, r);
}
#define SHAPE_OP(name) \
void* clmlx_##name(void* x, const int* shape, int rank) { \
  native_guard guard; auto r = mlx_array_new(); \
  int status = mlx_##name(&r, a(x), shape, rank, s()); return finish(status, r); }
SHAPE_OP(reshape)
SHAPE_OP(transpose_axes)
void* clmlx_slice(void* x, const int* start, const int* stop, const int* strides, int rank) {
  native_guard guard; auto r = mlx_array_new();
  int status = mlx_slice(&r, a(x), start, rank, stop, rank, strides, rank, s()); return finish(status, r);
}
void* clmlx_slice_update(void* x, void* update, const int* start, const int* stop,
                        const int* strides, int rank) {
  native_guard guard; auto r = mlx_array_new();
  int status = mlx_slice_update(&r, a(x), a(update), start, rank, stop, rank, strides, rank, s());
  return finish(status, r);
}
void* clmlx_take(void* x, void* indices, int axis) {
  native_guard guard; auto r = mlx_array_new();
  int status = mlx_take_axis(&r, a(x), a(indices), axis, s()); return finish(status, r);
}
void* clmlx_rmsnorm(void* x, void* weight, float epsilon) {
  native_guard guard; auto r = mlx_array_new();
  int status = mlx_fast_rms_norm(&r, a(x), a(weight), epsilon, s()); return finish(status, r);
}
void* clmlx_rope(void* x, int dims, int traditional, float base, int offset, void* freqs) {
  native_guard guard; auto r = mlx_array_new();
  int status = mlx_fast_rope(&r, a(x), dims, traditional, mlx_optional_float{base, freqs == nullptr}, 1.0f, offset, a(freqs), s());
  return finish(status, r);
}
void* clmlx_attention(void* q, void* k, void* v, float scale, const char* mode, void* mask) {
  native_guard guard; auto r = mlx_array_new();
  int status = mlx_fast_scaled_dot_product_attention(&r, a(q), a(k), a(v), scale,
                                                     mode, a(mask), a(nullptr), false, s());
  return finish(status, r);
}
void* clmlx_quantized_matmul(void* x, void* w, void* scales, void* biases, int group_size) {
  native_guard guard; auto r = mlx_array_new();
  int status = mlx_quantized_matmul(&r, a(x), a(w), a(scales), a(biases), true,
                                    oi(group_size), oi(4), "affine", s());
  return finish(status, r);
}
void* clmlx_dequantize(void* w, void* scales, void* biases, int group_size) {
  native_guard guard; auto r = mlx_array_new();
  mlx_optional_dtype dtype = {MLX_FLOAT32, true};
  int status = mlx_dequantize(&r, a(w), a(scales), a(biases), oi(group_size), oi(4),
                            "affine", a(nullptr), dtype, s());
  return finish(status, r);
}

static std::atomic<size_t> live_kernels{0};
size_t clmlx_live_kernels() { return live_kernels; }
void* clmlx_kernel_new(const char* name, const char** inputs, int ni,
                       const char** outputs, int no, const char* source,
                       const char* header, int atomic) {
  native_guard guard;
  auto in = mlx_vector_string_new_data(inputs, ni);
  auto out = mlx_vector_string_new_data(outputs, no);
  auto kernel = mlx_fast_metal_kernel_new(name, in, out, source, header, true, atomic != 0);
  mlx_vector_string_free(in); mlx_vector_string_free(out);
  if (last_error[0]) { mlx_fast_metal_kernel_free(kernel); return nullptr; }
  if (kernel.ctx) ++live_kernels;
  return kernel.ctx;
}
void clmlx_kernel_free(void* kernel) {
  if (kernel) { mlx_fast_metal_kernel_free({kernel}); --live_kernels; }
}
int clmlx_kernel_apply(void* kernel, void** inputs, int ni, void** outputs, int no,
                       const int* dimensions, const int* ranks, const int* dtypes,
                       const int* grid, const int* group, int initialize, float init,
                       const char** names, const int* kinds, const int* values, int nt) {
  native_guard guard;
  if (active_device != 1) { capture_error("Custom Metal kernels require the GPU device", nullptr); return -1; }
  auto config = mlx_fast_metal_kernel_config_new();
  auto in = mlx_vector_array_new();
  auto out = mlx_vector_array_new();
  int status = 0, offset = 0;
  for (int i = 0; i < ni && !status; ++i) status = mlx_vector_array_append_value(in, a(inputs[i]));
  for (int i = 0; i < no && !status; ++i) {
    status = mlx_fast_metal_kernel_config_add_output_arg(config, dimensions + offset, ranks[i], mlx_dtype(dtypes[i]));
    offset += ranks[i];
  }
  if (!status) status = mlx_fast_metal_kernel_config_set_grid(config, grid[0], grid[1], grid[2]);
  if (!status) status = mlx_fast_metal_kernel_config_set_thread_group(config, group[0], group[1], group[2]);
  if (!status && initialize) status = mlx_fast_metal_kernel_config_set_init_value(config, init);
  for (int i = 0; i < nt && !status; ++i) {
    status = kinds[i] == 0
      ? mlx_fast_metal_kernel_config_add_template_arg_bool(config, names[i], values[i] != 0)
      : mlx_fast_metal_kernel_config_add_template_arg_int(config, names[i], values[i]);
  }
  if (!status) status = mlx_fast_metal_kernel_apply(&out, {kernel}, in, config, s());
  for (int i = 0; i < no && !status; ++i) {
    auto result = mlx_array_new();
    status = mlx_vector_array_get(&result, out, i);
    outputs[i] = finish(status, result);
    if (!outputs[i]) status = -1;
  }
  mlx_fast_metal_kernel_config_free(config);
  mlx_vector_array_free(in); mlx_vector_array_free(out);
  if (status || last_error[0]) {
    for (int i = 0; i < no; ++i) {
      if (outputs[i]) { mlx_array_free(a(outputs[i])); --live_arrays; outputs[i] = nullptr; }
    }
    return -1;
  }
  return 0;
}

}
