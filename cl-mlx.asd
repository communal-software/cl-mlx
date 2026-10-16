(asdf:defsystem "cl-mlx"
  :description "Common Lisp bindings to MLX's C API for lazy array computation and inference"
  :author "George Watson"
  :license "MIT"
  :version "0.0.0"
  :depends-on ("cffi" "trivial-garbage" "bordeaux-threads")
  :serial t
  :components ((:file "package") (:file "ffi") (:file "arrays"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-mlx/tests"))))

(asdf:defsystem "cl-mlx/tests"
  :depends-on ("cl-mlx" "fiveam")
  :serial t
  :components ((:file "tests/package") (:file "tests/arrays"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-mlx/tests :run-tests)
               (error "cl-mlx tests failed"))))

(asdf:defsystem "cl-mlx/gpu"
  :description "cl-gpu DSL kernels in MLX lazy graphs"
  :license "MIT"
  :depends-on ("cl-mlx" "cl-gpu")
  :serial t
  :components ((:file "gpu/package") (:file "gpu/printer") (:file "gpu/kernels"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-mlx/gpu/tests"))))

(asdf:defsystem "cl-mlx/gpu/tests"
  :depends-on ("cl-mlx/gpu" "fiveam")
  :serial t
  :components ((:file "tests/gpu"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-mlx/gpu/tests :run-tests)
               (error "cl-mlx GPU tests failed"))))
