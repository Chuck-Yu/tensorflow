### Pre-build
Generate `libtensorflow_cc.so` and several header files(xxx_ops.h).
```shell
bazel build //tensorflow:libtensorflow_cc.so
```
General `libtensorflow_framework.so`
```shell
bazel build //tensorflow:libtensorflow_framework.so
```

### Run

Under TensorFlow root directory:
```shell
$ tensorflow/contrib/pi_examples/object_detection/gen/bin/obj_detection_demo
```