(defpackage #:cl-mlx/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:mx #:cl-mlx))
  (:export #:run-tests))
(in-package #:cl-mlx/tests)
(def-suite :cl-mlx)
(defun run-tests () (run! :cl-mlx))
