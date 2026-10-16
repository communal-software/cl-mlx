(in-package #:cl-mlx/gpu)

(defclass mlx-printer (gpu:embedded-msl-printer) ())

(defun wire-name (variable)
  (format nil "cl_gpu_arg_~d" (gpu:ir-variable-id variable)))

(defmethod gpu::begin-kernel :after ((printer mlx-printer) kernel)
  (labels ((name (variable)
             (setf (gethash variable (gpu::printer-names printer))
                   (format nil "cl_gpu_value_~d" (gpu:ir-variable-id variable)))))
    (mapc #'name (gpu:ir-kernel-parameters kernel))
    (gpu:map-ir (lambda (node)
                  (typecase node
                    (gpu:ir-var (name (gpu:ir-var-variable node)))
                    (gpu:ir-declare (name (gpu:ir-declare-variable node)))
                    (gpu::ir-array (name (gpu::ir-array-variable node))))) kernel)))

(defun packed-vector (type)
  (and (typep type 'gpu:pointer-type)
       (member (gpu:pointer-type-space type) '(:global :constant))
       (typep (gpu:pointer-type-pointee type) 'gpu:vector-type)
       (gpu:pointer-type-pointee type)))

(defmethod gpu::type-text ((printer mlx-printer) (type gpu:pointer-type))
  (let ((vector (packed-vector type)))
    (if vector
        (format nil "~a ~a*" (gpu::space-text printer (gpu:pointer-type-space type))
                (gpu::scalar-text printer (gpu:vector-type-element vector)))
        (call-next-method))))

(defmethod gpu::print-prologue ((printer mlx-printer) kernel)
  (dolist (variable (gpu:ir-kernel-parameters kernel))
    (let* ((type (gpu:ir-variable-type variable))
           (name (gpu::variable-name printer variable))
           (wire (wire-name variable)))
      (cond ((typep type 'gpu:pointer-type)
             (gpu::emit-line printer "auto ~a = ~a;" name wire))
            ((typep type 'gpu:vector-type)
             (gpu::emit-line printer "~a ~a = ~a(~{~a~^, ~});"
                             (gpu::type-text printer type) name (gpu::type-text printer type)
                             (loop for i below (gpu:vector-type-length type)
                                   collect (format nil "~a[~d]" wire i))))
            (t (gpu::emit-line printer "~a ~a = ~a;" (gpu::type-text printer type) name wire)))))
  (gpu::print-shared-arrays printer kernel))

(defmethod gpu::print-statement ((printer mlx-printer) (statement gpu:ir-declare))
  (if (typep (gpu:ir-variable-type (gpu:ir-declare-variable statement)) 'gpu:pointer-type)
      (if (gpu:ir-declare-init statement)
          (gpu::emit-line printer "auto ~a = ~a;"
                          (gpu::variable-name printer (gpu:ir-declare-variable statement))
                          (gpu::expr-text printer (gpu:ir-declare-init statement)))
          (gpu::target-fail "MLX pointer aliases require an initializer"))
      (call-next-method)))

(defun vector-helper (printer vector store)
  (let* ((type (gpu::scalar-text printer (gpu:vector-type-element vector)))
         (width (gpu:vector-type-length vector))
         (name (format nil "cl_gpu_~a_~a~d" (if store "store" "load") type width)))
    (pushnew name (gpu::printer-helpers printer) :test #'string=)
    name))

(defmethod gpu::helper-definition ((printer mlx-printer) name)
  (if (or (search "cl_gpu_load_" name) (search "cl_gpu_store_" name))
      (let* ((store (search "cl_gpu_store_" name))
             (type (subseq name (if store 13 12) (1- (length name))))
             (width (digit-char-p (char name (1- (length name))))))
        (with-output-to-string (out)
          (if store
              (format out "template<typename P> inline void ~a(P p, long i, ~a~d v) {~%" name type width)
              (format out "template<typename P> inline ~a~d ~a(P p, long i) {~%    return ~a~d("
                      type width name type width))
          (dotimes (i width)
            (if store
                (format out "    p[i * ~d + ~d] = v.~a;~%" width i (char "xyzw" i))
                (format out "~:[~;, ~]p[i * ~d + ~d]" (plusp i) width i)))
          (unless store (format out ");~%"))
          (format out "}~%")))
      (call-next-method)))

(defmethod gpu::render ((printer mlx-printer) (expr gpu:ir-index))
  (let ((vector (packed-vector (gpu:ir-expr-type (gpu:ir-index-base expr)))))
    (if vector
        (values (format nil "~a(~a, ~a)" (vector-helper printer vector nil)
                        (gpu::expr-text printer (gpu:ir-index-base expr))
                        (gpu::expr-text printer (gpu:ir-index-index expr))) gpu::+postfix-precedence+)
        (call-next-method))))

(defmethod gpu::render ((printer mlx-printer) (expr gpu:ir-component))
  (let* ((base (gpu:ir-component-base expr))
         (vector (and (typep base 'gpu:ir-index)
                      (packed-vector (gpu:ir-expr-type (gpu:ir-index-base base))))))
    (if vector
        (values (format nil "~a[(~a) * ~d + ~d]"
                        (gpu::expr-text printer (gpu:ir-index-base base) gpu::+postfix-precedence+)
                        (gpu::expr-text printer (gpu:ir-index-index base))
                        (gpu:vector-type-length vector)
                        (gpu:ir-literal-value (gpu:ir-component-index expr))) gpu::+postfix-precedence+)
        (call-next-method))))

(defmethod gpu::assign-text ((printer mlx-printer) assign)
  (let* ((target (gpu:ir-assign-target assign))
         (vector (and (typep target 'gpu:ir-index)
                      (packed-vector (gpu:ir-expr-type (gpu:ir-index-base target))))))
    (if vector
        (format nil "~a(~a, ~a, ~a)" (vector-helper printer vector t)
                (gpu::expr-text printer (gpu:ir-index-base target))
                (gpu::expr-text printer (gpu:ir-index-index target))
                (gpu::expr-text printer (gpu:ir-assign-value assign)))
        (call-next-method))))

(defmethod gpu::render ((printer mlx-printer) (expr gpu:ir-op))
  (let* ((operands (gpu:ir-op-operands expr))
         (vector (and operands (packed-vector (gpu:ir-expr-type (first operands)))))
         (operator (gpu:ir-op-operator expr)))
    (if (and vector (member operator '(:add :sub)))
        (values
         (if (typep (gpu:ir-expr-type expr) 'gpu:pointer-type)
             (format nil "~a + (~a) * ~d" (gpu::expr-text printer (first operands) 11)
                     (gpu::expr-text printer (second operands)) (gpu:vector-type-length vector))
             (format nil "(~a - ~a) / ~d" (gpu::expr-text printer (first operands) 11)
                     (gpu::expr-text printer (second operands) 11) (gpu:vector-type-length vector)))
         (if (typep (gpu:ir-expr-type expr) 'gpu:pointer-type) 10 11))
        (call-next-method))))
