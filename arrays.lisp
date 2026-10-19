(in-package #:cl-mlx)

(defparameter *dtypes*
  '(:bool :u8 :u16 :u32 :u64 :s8 :s16 :s32 :s64 :f16 :f32 :f64 :bf16 :complex64))
(defvar *array-scope* nil)
(defvar *scope-active* nil)
(defvar *evaluation-count* 0)

(defstruct (array (:constructor %make-array (pointer)))
  pointer (released-p nil))

(defun released-p (array) (array-released-p array))

(defun dtype-id (type)
  (or (position type *dtypes*) (error 'mlx-error :message (format nil "Unsupported dtype ~S" type))))

(defun pointer (array)
  (if array
      (progn
        (when (released-p array) (error 'mlx-error :message "Array has been released"))
        (array-pointer array))
      (cffi:null-pointer)))

(defun wrap (pointer)
  (when (cffi:null-pointer-p pointer)
    (error 'mlx-error :message (let ((message (%error)))
                                (if (plusp (length message)) message "MLX returned no array"))))
  (let ((array (%make-array pointer)))
    (trivial-garbage:finalize array (lambda () (%free pointer)))
    (when *scope-active* (push array *array-scope*))
    array))

(defun release (array)
  (when (and array (not (released-p array)))
    (trivial-garbage:cancel-finalization array)
    (setf (array-released-p array) t)
    (check-status (%free (array-pointer array))))
  nil)

(defun retain (array)
  "Keep ARRAY after the current array scope exits."
  (setf *array-scope* (delete array *array-scope* :test #'eq))
  array)

(defun live-array-count () (initialize) (%live-arrays))

(defmacro with-array-scope (() &body body)
  `(let ((*array-scope* nil) (*scope-active* t))
     (unwind-protect (progn ,@body)
       (dolist (array *array-scope*) (release array)))))

(defmacro with-ints ((pointer values) &body body)
  (let ((vector (gensym)))
    `(let ((,vector (coerce ,values 'vector)))
       (cffi:with-foreign-object (,pointer :int (max 1 (length ,vector)))
         (dotimes (i (length ,vector))
           (setf (cffi:mem-aref ,pointer :int i) (aref ,vector i)))
         ,@body))))

(defun device () (initialize) (if (zerop (%device)) :cpu :gpu))

(defmacro with-device ((device) &body body)
  (let ((old (gensym)) (value (gensym)))
    `(progn
       (initialize)
       (let ((,old (%device)) (,value ,device))
         (%set-device (ecase ,value (:cpu 0) (:gpu 1)))
         (unwind-protect (progn ,@body) (%set-device ,old))))))

(defun synchronize () (initialize) (check-status (%synchronize)) nil)

(defun available-p () (initialize) (not (zerop (%available))))
(defun shape (array)
  (let* ((p (pointer array)) (rank (%rank p)))
    (check-status 0)
    (loop for axis below rank collect (%dim p axis))))
(defun dtype (array) (let ((id (%dtype (pointer array)))) (check-status 0) (nth id *dtypes*)))
(defun size (array) (let ((size (%size (pointer array)))) (check-status 0) size))

(defun checked-shape (shape)
  (let ((shape (coerce shape 'list)))
    (unless (every (lambda (n) (typep n '(integer 0 2147483647))) shape)
      (error 'mlx-error :message "Array dimensions must be non-negative int32 values"))
    shape))

(defun from-pointer (data shape &key (dtype :f32))
  "Copy DATA into an MLX-owned array with SHAPE and DTYPE."
  (initialize)
  (let ((shape (checked-shape shape)))
    (with-ints (dims shape) (wrap (%from-data data dims (length shape) (dtype-id dtype))))))

(defun from-data (data shape &key (dtype :f32))
  (let* ((shape (checked-shape shape)) (data (coerce data 'vector))
         (type (case dtype (:f32 :float) (:f64 :double) (:s32 :int32) (:u32 :uint32)
                        (:u8 :uint8) (t (error 'mlx-error :message "FROM-DATA supports F32/F64/S32/U32/U8; use FROM-POINTER for raw encodings")))))
    (unless (= (length data) (reduce #'* shape :initial-value 1))
      (error 'mlx-error :message "Data length does not match shape"))
    (cffi:with-foreign-object (buffer type (max 1 (length data)))
      (dotimes (i (length data))
        (setf (cffi:mem-aref buffer type i)
              (case dtype (:f32 (coerce (aref data i) 'single-float))
                          (:f64 (coerce (aref data i) 'double-float)) (t (aref data i)))))
      (from-pointer buffer shape :dtype dtype))))

(defun zeros (shape &key (dtype :f32))
  (initialize)
  (let ((shape (checked-shape shape)))
    (with-ints (dims shape) (wrap (%zeros dims (length shape) (dtype-id dtype))))))

(defun eval (&rest arrays)
  (unless arrays (return-from eval nil))
  (cffi:with-foreign-object (pointers :pointer (length arrays))
    (loop for array in arrays for i from 0
          do (setf (cffi:mem-aref pointers :pointer i) (pointer array)))
    (incf *evaluation-count*)
    (check-status (%eval pointers (length arrays))))
  (first arrays))

(defun to-vector (array &key (evaluate t))
  "Copy a contiguous F32 array to Lisp; EVALUATE NIL requires an evaluated array."
  (unless (eq (dtype array) :f32)
    (error 'mlx-error :message "TO-VECTOR requires F32; use ASTYPE first"))
  (when evaluate (eval array))
  (let ((p (%data-f32 (pointer array))) (out (make-array (size array) :element-type 'single-float)))
    (check-status 0)
    (when (and (plusp (length out)) (cffi:null-pointer-p p))
      (error 'mlx-error :message "Array data is not evaluated or contiguous"))
    (dotimes (i (length out) out) (setf (aref out i) (cffi:mem-aref p :float i)))))

(defmacro binary (name native)
  `(defun ,name (a b) (wrap (,native (pointer a) (pointer b)))))
(binary add %add)
(binary multiply %multiply)
(binary matmul %matmul)
(defun sigmoid (a) (wrap (%sigmoid (pointer a))))
(defun astype (a dtype) (wrap (%astype (pointer a) (dtype-id dtype))))
(defun reshape (a shape)
  (with-ints (dims shape) (wrap (%reshape (pointer a) dims (length shape)))))
(defun transpose (a &optional (axes (reverse (loop for i below (length (shape a)) collect i))))
  (with-ints (dims axes) (wrap (%transpose (pointer a) dims (length axes)))))

(defun slice (a start stop &optional (strides (make-list (length start) :initial-element 1)))
  (unless (= (length start) (length stop) (length strides) (length (shape a)))
    (error 'mlx-error :message "Slice vectors must match array rank"))
  (with-ints (starts start)
    (with-ints (stops stop)
      (with-ints (steps strides)
        (wrap (%slice (pointer a) starts stops steps (length start)))))))

(defun slice-update (a update start stop &optional (strides (make-list (length start) :initial-element 1)))
  (unless (= (length start) (length stop) (length strides) (length (shape a)))
    (error 'mlx-error :message "Slice vectors must match array rank"))
  (with-ints (starts start)
    (with-ints (stops stop)
      (with-ints (steps strides)
        (wrap (%slice-update (pointer a) (pointer update) starts stops steps (length start)))))))

(defun take (a indices &key (axis 0)) (wrap (%take (pointer a) (pointer indices) axis)))
(defun rmsnorm (a weight epsilon)
  (wrap (%rmsnorm (pointer a) (pointer weight) (coerce epsilon 'single-float))))
(defun rope (a dims &key (traditional nil) (base 10000) (offset 0) freqs)
  (wrap (%rope (pointer a) dims (if traditional 1 0) (coerce base 'single-float) offset (pointer freqs))))
(defun attention (q k v scale &key (causal nil) mask)
  (wrap (%attention (pointer q) (pointer k) (pointer v) (coerce scale 'single-float)
                    (if causal "causal" "") (pointer mask))))
(defun quantized-matmul (x weight scales biases group-size)
  (wrap (%quantized-matmul (pointer x) (pointer weight) (pointer scales) (pointer biases) group-size)))
(defun dequantize (weight scales biases group-size)
  (wrap (%dequantize (pointer weight) (pointer scales) (pointer biases) group-size)))

(defun symmetric-quantize (array)
  "Return signed Q8 codes and F32 scales for groups of 32 on ARRAY's last axis."
  (cffi:with-foreign-objects ((codes :pointer) (scales :pointer))
    (check-status (%symmetric-quantize (pointer array) codes scales))
    (values (wrap (cffi:mem-ref codes :pointer)) (wrap (cffi:mem-ref scales :pointer)))))

(defun symmetric-dequantize (codes scales)
  "Decode signed Q8 CODES with one F32 scale per last-axis group of 32."
  (let* ((shape (shape codes)) (expected (copy-list shape)))
    (unless (and shape (plusp (car (last shape))) (eq (dtype codes) :s8) (eq (dtype scales) :f32))
      (error 'mlx-error :message "Q8 requires S8 codes, F32 scales and a positive final dimension"))
    (setf (car (last expected)) (ceiling (car (last shape)) 32))
    (unless (equal expected (shape scales))
      (error 'mlx-error :message "Q8 scales do not match the code groups"))
    (wrap (%symmetric-dequantize (pointer codes) (pointer scales)))))

(defun gelu (a)
  "Lazy GELU with the standard tanh approximation."
  (wrap (%gelu (pointer a))))
