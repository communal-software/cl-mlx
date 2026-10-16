(asdf:load-system "cl-mlx/gpu")

(defpackage #:cl-mlx/custom-example
  (:use #:cl)
  (:local-nicknames (#:mx #:cl-mlx) (#:gpu #:cl-gpu) (#:bridge #:cl-mlx/gpu)))
(in-package #:cl-mlx/custom-example)

(gpu:define-kernel add-vectors ((out (gpu:ptr (gpu:vec gpu:f32 3) :global))
                              (x (gpu:ptr (gpu:vec gpu:f32 3) :global))
                              (gain (gpu:vec gpu:f32 3)))
  (let ((i (gpu:global-id 0))) (setf (aref out i) (+ (aref x i) gain))))

(bridge:define-kernel add-array-vectors (gpu:find-kernel 'add-vectors) (x gain)
  :outputs ((out (mx:shape x))) :grid (first (mx:shape x)) :group 32)

(unwind-protect
     (mx:with-array-scope ()
       (let* ((x (mx:from-data '(1 2 3 4 5 6) '(2 3)))
              (custom (first (add-array-vectors x '(1 2 3))))
              (result (mx:add custom custom)))
         (mx:eval result)
         (print (mx:to-vector result :evaluate nil))))
  (bridge:clear-kernel-cache))
