(defpackage #:cl-mlx/gpu
  (:use #:cl)
  (:shadow #:define-kernel)
  (:local-nicknames (#:mx #:cl-mlx) (#:gpu #:cl-gpu))
  (:export #:define-kernel #:clear-kernel-cache #:live-kernel-count))
