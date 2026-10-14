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
