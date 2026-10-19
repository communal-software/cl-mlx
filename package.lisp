(defpackage #:cl-mlx
  (:use #:cl)
  (:shadow #:array #:eval)
  (:export #:with-device #:device #:synchronize #:array #:mlx-error #:initialize #:available-p #:from-data #:from-pointer
           #:zeros #:shape #:dtype #:size #:release #:released-p #:with-array-scope
           #:retain #:live-array-count #:eval #:to-vector #:astype #:reshape #:transpose
           #:slice #:slice-update #:take #:add #:multiply #:matmul #:sigmoid #:gelu
           #:rmsnorm #:rope #:attention #:quantized-matmul #:dequantize
           #:symmetric-quantize #:symmetric-dequantize))
