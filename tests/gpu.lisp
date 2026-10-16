(defpackage #:cl-mlx/gpu/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:mx #:cl-mlx) (#:gpu #:cl-gpu) (#:bridge #:cl-mlx/gpu))
  (:export #:run-tests))
(in-package #:cl-mlx/gpu/tests)
(def-suite :cl-mlx/gpu)
(in-suite :cl-mlx/gpu)
(defun run-tests ()
  (unwind-protect (run! :cl-mlx/gpu) (bridge:clear-kernel-cache)))

(gpu:define-kernel scale-raw ((out (gpu:ptr gpu:f32 :global)) (x (gpu:ptr gpu:f32 :global)) (gain gpu:f32))
  (let ((i (gpu:global-id 0))) (setf (aref out i) (* (aref x i) gain))))
(bridge:define-kernel scale-array (gpu:find-kernel 'scale-raw) (x gain)
  :outputs ((out (mx:shape x))) :grid (mx:size x) :group 32)

(defmacro vector-fixture (width)
  (let ((raw (intern (format nil "VECTOR~d-RAW" width)))
        (wrapper (intern (format nil "VECTOR~d" width))))
    `(progn
       (gpu:define-kernel ,raw ((out (gpu:ptr (gpu:vec gpu:f32 ,width) :global))
                               (x (gpu:ptr (gpu:vec gpu:f32 ,width) :global))
                               (gain (gpu:vec gpu:f32 ,width)))
         (let ((i (gpu:global-id 0))) (setf (aref out i) (+ (aref x i) gain))))
       (bridge:define-kernel ,wrapper (gpu:find-kernel ',raw) (x gain)
         :outputs ((out (mx:shape x))) :grid (first (mx:shape x)) :group 32))))
(vector-fixture 2)
(vector-fixture 3)
(vector-fixture 4)

(gpu:define-kernel vector-tail-raw ((out (gpu:ptr (gpu:vec gpu:f32 3) :global))
                                  (x (gpu:ptr (gpu:vec gpu:f32 3) :global)))
  (let ((i (gpu:global-id 0)) (tail (gpu:ptr+ x 1)))
    (setf (aref out i) (aref tail i))
    (setf (aref (aref out i) 1) 9.0)))
(bridge:define-kernel vector-tail (gpu:find-kernel 'vector-tail-raw) (x)
  :outputs ((out (list (1- (first (mx:shape x))) 3))) :grid (1- (first (mx:shape x))) :group 32)

(gpu:define-kernel count-raw ((out (gpu:ptr gpu:u32 :global)) (n gpu:u32))
  (let ((i (gpu:global-id 0))) (setf (aref out i) n)))
(bridge:define-kernel count-array (gpu:find-kernel 'count-raw) (n)
  :outputs ((out '(3))) :grid 3 :group 32 :templates ((n n)))

(gpu:define-kernel accumulate-raw ((out (gpu:ptr gpu:f32 :global)))
  (let ((alias out)) (setf (aref alias (gpu:global-id 0)) (+ (aref alias (gpu:global-id 0)) 2.0))))
(bridge:define-kernel accumulate-array (gpu:find-kernel 'accumulate-raw) ()
  :outputs ((out '(4))) :grid 4 :group 32 :init-value 0)
(bridge:define-kernel uninitialized-array (gpu:find-kernel 'accumulate-raw) ()
  :outputs ((out '(4))) :grid 4 :group 32)

(gpu:define-kernel atomic-raw ((out (gpu:ptr (gpu:atomic gpu:u32) :global)))
  (gpu:atomic-add out 1))
(bridge:define-kernel atomic-array (gpu:find-kernel 'atomic-raw) ()
  :outputs ((out '(1))) :grid 32 :group 32 :init-value 0)

(gpu:define-kernel shared-raw ((out (gpu:ptr gpu:f32 :global)) (x (gpu:ptr gpu:f32 :global)))
  (let ((tile (gpu:shared-array gpu:f32 4 :zero)) (i (gpu:local-id 0)))
    (setf (aref tile i) (aref x i))
    (gpu:barrier)
    (when (= i 0) (setf (aref out 0) (+ (aref tile 0) (aref tile 1) (aref tile 2) (aref tile 3))))))
(bridge:define-kernel shared-array (gpu:find-kernel 'shared-raw) (x)
  :outputs ((out '(1))) :grid 4 :group 4)

(gpu:define-kernel multiple-raw ((a (gpu:ptr gpu:f32 :global)) (b (gpu:ptr gpu:f32 :global)))
  (setf (aref a 0) 1.0 (aref b 0) 2.0))
(bridge:define-kernel multiple-array (gpu:find-kernel 'multiple-raw) ()
  :outputs ((b '(1)) (a '(1))) :grid 1 :group 1)

(test lazy-composition-and-borrowed-lifetime
  (bridge:clear-kernel-cache)
  (let ((before (mx:live-array-count)))
    (mx:with-array-scope ()
      (let* ((x (mx:from-data '(1 2 3 4) '(4)))
             (evaluations mx::*evaluation-count*)
             (out (first (scale-array x 2.0)))
             (result (mx:add out out)))
        (is (= evaluations mx::*evaluation-count*))
        (mx:release x)
        (bridge:clear-kernel-cache)
        (is (equalp #(4 8 12 16) (mx:to-vector result)))))
    (is (= before (mx:live-array-count)))))

(test vectors-packed-lanes-and-offsets
  (mx:with-array-scope ()
    (loop for width in '(2 3 4)
          for function in (list #'vector2 #'vector3 #'vector4)
          do (let* ((values (loop for i from 1 to (* 3 width) collect i))
                    (x (mx:from-data values (list 3 width)))
                    (result (first (funcall function x (make-list width :initial-element 2.0)))))
               (is (equalp (coerce (mapcar (lambda (v) (+ v 2)) values) 'vector) (mx:to-vector result)))))
    (let* ((x (mx:from-data '(1 2 3 4 5 6 7 8 9) '(3 3)))
           (result (first (vector-tail x))))
      (is (equalp #(4 9 6 7 9 9) (mx:to-vector result))))))

(test strided-input-normalization
  (mx:with-array-scope ()
    (let* ((x (mx:transpose (mx:from-data '(1 2 3 4 5 6) '(2 3))))
           (out (first (scale-array x 2))))
      (is (equalp #(2 8 4 10 6 12) (mx:to-vector out))))))

(test templates-and-cache-lifetime
  (bridge:clear-kernel-cache)
  (mx:with-array-scope ()
    (is (equalp #(3 3 3) (mx:to-vector (mx:astype (first (count-array 3)) :f32))))
    (let ((count (bridge:live-kernel-count)))
      (count-array 3)
      (is (= count (bridge:live-kernel-count)))
      (count-array 4)
      (is (= (1+ count) (bridge:live-kernel-count)))))
  (bridge:clear-kernel-cache)
  (is (zerop (bridge:live-kernel-count))))

(test initialized-atomic-and-shared-outputs
  (mx:with-array-scope ()
    (is (equalp #(2 2 2 2) (mx:to-vector (first (accumulate-array)))))
    (is (equalp #(32) (mx:to-vector (mx:astype (first (atomic-array)) :f32))))
    (is (equalp #(10) (mx:to-vector (first (shared-array (mx:from-data '(1 2 3 4) '(4))))))))
  (signals gpu:kernel-target-error (uninitialized-array)))

(test declared-output-order
  (mx:with-array-scope ()
    (let ((outputs (multiple-array)))
      (is (equalp #(2) (mx:to-vector (first outputs))))
      (is (equalp #(1) (mx:to-vector (second outputs)))))))

(test validation-and-cleanup
  (let ((before (mx:live-array-count)))
    (mx:with-array-scope ()
      (let ((x (mx:from-data '(1 2 3) '(3))))
        (signals mx:mlx-error (scale-array x (mx:from-data '(1) '(1))))
        (signals mx:mlx-error (scale-array (mx:astype x :u32) 2))
        (mx:with-device (:cpu) (signals mx:mlx-error (scale-array x 2)))
        (signals mx:mlx-error (count-array -1))))
    (is (= before (mx:live-array-count)))))

(gpu:define-kernel boolean-raw ((out (gpu:ptr gpu:f32 :global)) (enabled gpu:bool))
  (setf (aref out 0) (if enabled 1.0 0.0)))
(bridge:define-kernel boolean-array (gpu:find-kernel 'boolean-raw) (enabled)
  :outputs ((out '(1))) :grid 1 :group 1 :templates ((enabled enabled)))

(gpu:define-kernel writable-value-raw ((out (gpu:ptr gpu:u32 :global)) (n gpu:u32))
  (setf n (+ n 1)) (setf (aref out 0) n))
(bridge:define-kernel writable-value (gpu:find-kernel 'writable-value-raw) (n)
  :outputs ((out '(1))) :grid 1 :group 1)

(gpu:define-kernel precise-raw ((out (gpu:ptr gpu:f32 :global)))
  (declare (gpu::precise-math)) (setf (aref out 0) 1.0))
(bridge:define-kernel precise-array (gpu:find-kernel 'precise-raw) ()
  :outputs ((out '(1))) :grid 1 :group 1)

(gpu:define-kernel mixed-atomic-raw ((out (gpu:ptr gpu:f32 :global))
                                  (count (gpu:ptr (gpu:atomic gpu:u32) :global)))
  (setf (aref out 0) 1.0) (gpu:atomic-add count 1))
(bridge:define-kernel mixed-atomic-array (gpu:find-kernel 'mixed-atomic-raw) ()
  :outputs ((out '(1)) (count '(1))) :grid 1 :group 1 :init-value 0)

(test typed-values-and-template-booleans
  (mx:with-array-scope ()
    (is (equalp #(1) (mx:to-vector (first (boolean-array t)))))
    (is (equalp #(0) (mx:to-vector (first (boolean-array nil)))))
    (is (equalp #(8) (mx:to-vector (mx:astype (first (writable-value 7)) :f32))))
    (is (equalp #(8) (mx:to-vector (mx:astype (first (writable-value (mx:from-data '(7) nil :dtype :u32))) :f32))))
    (signals mx:mlx-error (boolean-array 1))
    (signals mx:mlx-error (precise-array))
    (signals mx:mlx-error (mixed-atomic-array))))

(test source-redefinition-invalidates-cache
  (bridge:clear-kernel-cache)
  (let ((original (gpu:find-kernel 'scale-raw))
        (original-function (symbol-function 'scale-raw)))
    (unwind-protect
         (mx:with-array-scope ()
           (let* ((x (mx:from-data '(2 3) '(2)))
                  (old (first (scale-array x 2))))
             (eval '(gpu:define-kernel scale-raw ((out (gpu:ptr gpu:f32 :global))
                                                 (x (gpu:ptr gpu:f32 :global)) (gain gpu:f32))
                      (let ((i (gpu:global-id 0)))
                        (setf (aref out i) (+ (aref x i) gain)))))
             (is (equalp #(4 5) (mx:to-vector (first (scale-array x 2)))))
             (is (equalp #(4 6) (mx:to-vector old)))
             (is (= 2 (bridge:live-kernel-count)))))
      (setf (get 'scale-raw 'gpu::gpu-kernel) original
            (symbol-function 'scale-raw) original-function)
      (bridge:clear-kernel-cache))))

(test unsupported-bindings-and-invalid-launches
  (signals gpu:kernel-target-error
    (gpu:kernel-parts
     (gpu::make-gpu-kernel 'shared-pointer '((out (gpu:ptr gpu:f32 :shared))) nil) :metal))
  (dolist (value '(0 -1 (1 2 3 4) (1 0) 1.5))
    (signals mx:mlx-error (bridge::launch-dimensions value)))
  (mx:with-array-scope ()
    (let* ((state (bridge::new-wrapper))
           (kernel (gpu:find-kernel 'scale-raw))
           (x (mx:from-data '(1 2) '(2))))
      (signals mx:mlx-error
        (bridge::call-wrapper state kernel (list (list 'x x) (list 'gain 2)) '((bad (2))) 2 2 nil nil nil))
      (signals mx:mlx-error
        (bridge::call-wrapper state kernel (list (list 'x x)) '((out (2))) 2 2 nil nil nil))
      (signals mx:mlx-error
        (bridge::call-wrapper state kernel (list (list 'x x) (list 'gain 2)) '((out (-1))) 2 2 nil nil nil)))))

(test native-launch-failure-releases-handles
  (let ((before (mx:live-array-count)))
    (handler-case
        (mx:with-array-scope ()
          (let* ((x (mx:from-data '(1 2) '(2)))
                 (out (first (bridge::call-wrapper (bridge::new-wrapper) (gpu:find-kernel 'scale-raw)
                                                  (list (list 'x x) (list 'gain 2)) '((out (2)))
                                                  2048 2048 nil nil nil))))
            (mx:eval out)
            (fail "Oversized threadgroup unexpectedly succeeded")))
      (mx:mlx-error () (pass)))
    (is (= before (mx:live-array-count)))))

(defmacro typed-fixture (name type)
  (let ((raw (intern (format nil "~a-RAW" name))))
    `(progn
       (gpu:define-kernel ,raw ((out (gpu:ptr ,type :global)) (x (gpu:ptr ,type :global)) (gain ,type))
         (let ((i (gpu:global-id 0))) (setf (aref out i) (+ (aref x i) gain))))
       (bridge:define-kernel ,name (gpu:find-kernel ',raw) (x gain)
         :outputs ((out (mx:shape x))) :grid (mx:size x) :group 32))))
(typed-fixture half-array gpu:f16)
(typed-fixture bfloat-array gpu:bf16)
(typed-fixture signed8-array gpu:i8)
(typed-fixture signed16-array gpu:i16)
(typed-fixture signed32-array gpu:i32)
(typed-fixture signed64-array gpu:i64)
(typed-fixture unsigned8-array gpu:u8)
(typed-fixture unsigned16-array gpu:u16)
(typed-fixture unsigned32-array gpu:u32)
(typed-fixture unsigned64-array gpu:u64)

(test scalar-element-dtype-mapping
  (mx:with-array-scope ()
    (let ((base (mx:from-data '(1 2 3) '(3))))
      (loop for dtype in '(:f16 :bf16 :s8 :s16 :s32 :s64 :u8 :u16 :u32 :u64)
            for function in (list #'half-array #'bfloat-array #'signed8-array #'signed16-array
                                  #'signed32-array #'signed64-array #'unsigned8-array #'unsigned16-array
                                  #'unsigned32-array #'unsigned64-array)
            do (is (equalp #(3 4 5) (mx:to-vector (mx:astype (first (funcall function (mx:astype base dtype) 2)) :f32))))))))

(gpu:define-kernel host-names-raw ((out (gpu:ptr gpu:f32 :global))
                                 (thread_position_in_grid (gpu:ptr gpu:f32 :global))
                                 (thread_position_in_grid_shape gpu:f32))
  (setf (aref out 0) (+ (aref thread_position_in_grid 0) thread_position_in_grid_shape)))
(bridge:define-kernel host-names (gpu:find-kernel 'host-names-raw)
    (thread_position_in_grid thread_position_in_grid_shape)
  :outputs ((out '(1))) :grid 1 :group 1)

(test parameter-names-do-not-trigger-host-signature-collisions
  (mx:with-array-scope ()
    (is (equalp #(5) (mx:to-vector (first (host-names (mx:from-data '(2) '(1)) 3)))))))

(test wrapper-cache-across-native-threads
  (bridge:clear-kernel-cache)
  (let ((threads
          (loop repeat 4 collect
            (bordeaux-threads:make-thread
             (lambda ()
               (mx:with-array-scope ()
                 (mx:to-vector (first (scale-array (mx:from-data '(1 2 3) '(3)) 2)))))))))
    (dolist (thread threads) (is (equalp #(2 4 6) (bordeaux-threads:join-thread thread))))
    (is (= 1 (bridge:live-kernel-count))))
  (bridge:clear-kernel-cache))
