# 深度学习识别电线实现说明

## 安装TensorFlow
TensorFlow的下载以及安装的步骤可以参考[`GitHub`](https://github.com/tensorflow/tensorflow)。
或者可以参考极客学院提供的中文版文档，[传送门](http://wiki.jikexueyuan.com/project/tensorflow-zh/get_started/os_setup.html)。
这里只给出Ubuntu Pip安装的教程。其他操作系统和安装方法可通过上面的链接获得。

### Pip 安装
Pip 是一个 Python 的软件包安装与管理工具。首先安装Pip（或者python3的pip3）：
```shell
$ sudo apt-get install python-pip python-dev
```
安装TensorFlow：
```shell
# Ubuntu/Linux 64-bit, CPU only, Python 2.7:
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7. Requires CUDA toolkit 7.5 and CuDNN v4.
# For other versions, see "Install from sources" below.
$ sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-1.2.0-cp27-none-linux_x86_64.whl
```

安装完后，可以运行一个简单的Python代码测试。
```shell
$ python
```
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
'Hello, TensorFlow!'
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a + b)
42
>>> sess.close()
```

## 在Ubuntu上运行TensorFlow
首先，需要把TensorFlow的Repo克隆到本地。
```shell
$ git checkout https://github.com/tensorflow/tensorflow.git
```
[`makefile`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/makefile)目录下提供了
如何在各个平台上编译TensorFlow的详细说明以及脚本文件。编译成功后会生成一个`benchmark`用于检测整个系统是否搭建成功。
由于测试的系统是Ubuntu的，这里还是给出Ubuntu下的编译说明。不管是哪个平台上，编译前需要下载一些依赖项。这个脚本只需要运行一次，下载完成即可。
(**所有命令需要在根目录下运行**)
```shell
$ tensorflow/contrib/makefile/download_dependencies.sh
```
相关的库文件会保存在`tensorflow/contrib/makefile/downloads/`文件夹内。另外，需要下载一个测试用的graph: 
[inception](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip)

在Ubuntu上编译前需确保相关的软件包已经安装好，可先运行：
```shell
$ sudo apt-get install autoconf automake libtool curl make g++ unzip zlib1g-dev \
git python
```
然后运行`build_all_linux.sh`脚本：
```shell
$ tensorflow/contrib/makefile/build_all_linux.sh
```
编译完成后，会生成可执行文件`tensorflow/contrib/makefile/gen/bin/benchmark`。下载graph后，需解压缩：
```shell
$ mkdir -p ~/graphs
$ curl -o ~/graphs/inception.zip \
 https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip \
 && unzip ~/graphs/inception.zip -d ~/graphs/inception
```
运行`benchmark`：
```shell
$ tensorflow/contrib/makefile/gen/bin/benchmark \
 --graph=$HOME/graphs/inception/tensorflow_inception_graph.pb
```

## 编译Android平台上可运行的label_image
本项目的平台是Ｖ２上的Android，识别电线使用的是深度学习图片分类的功能。
TensorFlow的GitHub上已经提供了在Raspberry上运行图片分类的Ｃ＋＋代码label_image，在`tensorflow/contrib/pi_examples`目录下可找到详细说明。
而在Android上运行需要对应把一些文件换成Android版的。编译label_image之前需要得到一个TensorFlow的静态库。
上面运行`build_all_linux.sh`的时候已经生成了一个Linux下的库，Android下的需要对应的运行`build_all_android.sh`。
载入图片需要的libjpeg.so，在Android上也需要单独下载安装(忘了链接，需要的话可以找我拷)。Ubuntu下可直接运行：
```shell
$ sudo apt-get install -y libjpeg-dev
```
为了生成可以在Android上运行的label_image，这里的makefile参考了生成Android benchmark的makefile。＠祖国将整理后的文件上传到了GitHub，
[链接](https://github.com/yuzuguo/tensorflow/tree/master/tensorflow/contrib/pi_examples/input_image)。可直接克隆＠祖国的Repo运行,
或者根据他的文件进行修改。主要注意makefile以及编译时的参数，如`arm64-v8a`。运行前Android版需要V2上对应的NDK(未上传)，并设置本地的路径：
```shell
$ export NDK_ROOT=/absolute/path/to/NDK/android-ndk-rxxx/
```


## 获取MobileNet模型
编译好可执行图像分类代码后，最重要的是找到一个合适的深度学习模型。
[tensorflow/models](https://github.com/tensorflow/models)主要是提供各种基于TensorFlow实现的深度学习模型。
这次识别电线用到的MobileNet也是从这里获取的。MobileNet是Google提出的一个针对移动端的轻量级深度学习模型。
除了现成的模型，这里还包含了整个基于TensorFlow的深度学习训练工具以及说明文档。之后项目的训练也会基于这些工具进行。
建议将这个Repo克隆到本地。

```shell
$ git checkout https://github.com/tensorflow/models.git
```
### TensorFlow-Slim
[TensorFlow-Slim](https://github.com/tensorflow/models/tree/master/research/slim)是一个基于TensorFlow的用于图片分类的轻量级API。
文件夹里提供了包含定义、训练、评估复杂模型的脚本及实例。此外，还有很多经过预训练的复杂模型，这些模型可以直接使用或者加上自己的数据集用于微调模型。
本项目使用的MobileNet_v1_128就是通过这个API生成的。这里只提供了用ImageNet预训练的checkpoint,还需要根据脚本生成对应的pb文件用于运行。
之后的训练也是准备基于这些预训练的模型进行微调(fine-tunning)。

### MobileNet_v1
根据参数不同，[这里](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)提供了１６种不同的MobileNet_v1的模型。
针对每一个模型都有准确率和大小的数据，可以满足不同的需求。在计算能力允许的情况下，我们尽量选择准确率高的模型。本项目在前四个模型中进行了评估，
加上每个对应的量化模型，一共是八个模型。具体在V2和ZULUKO上的测试情况已经上传到Google Drive:[MobileNet性能评估](https://docs.google.com/a/perceptin.io/spreadsheets/d/1T6trlSegiXijToxLqcKRkkhcakmS0sVPTplmE_Ge_LM/edit?usp=drive_web)。

前面提到的量化模型也是TensorFlow-Slim API里提供的将模型参数由32bit转换为8bit的功能。这样可以通过减少数据精度到达更快的速度以及相近的准确度。
在识别电线时，使用的是用现有模型分类映射的方案，很多时候识别出的准确率很低。如果在量化模型中，这个结果会被输出为零。
不能满足我们映射的要求。但在自己训练的模型中，量化模型在保证准确率的情况下有一定的优势。

生成一个可运行的pb文件的流程可以参考`Exporting the Inference Graph`。

以MobileNet_v1_1.0_224为例，生成一个模型结构:
```shell
$ python export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v1 \
  --image_size=224 \
  --output_file=/tmp/mobilenet_v1_1.0_224.pb
```
将checkpoint导入到模型中:
```shell
bazel build tensorflow/python/tools:freeze_graph

bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=/tmp/mobilenet_v1_1.0_224.pb \
  --input_checkpoint=/tmp/checkpoints/mobilenet_v1_1.0_224.ckpt \
  --input_binary=true --output_graph=/tmp/mobilenet_v1_1.0_224_frozen.pb \
  --output_node_names=MobilenetV1/Predictions/Reshape_1
```
到这一步已经生成了一个可使用的模型pb文件，针对MobileNet_v1还需要两个参数input_layer和output_layer分别是`input`和`MobilenetV1/Predictions/Reshape_1`。接下来可以用一个C++代码测试新生成模型的性能。
```shell
bazel build tensorflow/examples/label_image:label_image

bazel-bin/tensorflow/examples/label_image/label_image \
  --image=${HOME}/Pictures/flowers.jpg \
  --input_layer=input \
  --output_layer=MobilenetV1/Predictions/Reshape_1 \
  --graph=/tmp/mobilenet_v1_1.0_224_frozen.pb \
  --labels=/tmp/imagenet_slim_labels.txt \
  --input_width=224 \
  --input_height=224
```

### 量化模型
这部分在GitHub的README里没有说明，但在文件夹里可以找到实现这个功能的脚本文件。其中[`scripts`](https://github.com/tensorflow/models/tree/master/research/slim/scripts)文件夹内的`export_mobilenet.sh`是用来直接生成任意MobileNet_v1模型的frozen和quantized文件的脚本文件。
但我没有运行成功，这个是基于`bazel`编译的。编译很久后提示出错，目前没有解决。但是这个可以通过单独运行Python脚本编译。

在TensorFlow根目录下运行：
```shell
$ bazel-bin/tensorflow/tools/quantization/quantize_graph \
--input=/tmp/mobilenet_v1_1.0_224_freeze.pb \
--output_node_names="MobilenetV1/Predictions/Reshape_1" --print_nodes --output=/tmp/quantized_graph.pb \
--mode=eightbit
```
生成的量化模型可以跟之前的frozen模型一样进行测试，将输出尺寸改成对应的大小即可。

## 在V2上运行label_image
到这一步已经生成了所有需要的文件，包括可执行`label_image`文件、不同参数的MobileNet_v1模型、待测试的图片以及标签文件`imagenet_slim_labels.txt`。
接下来需要把这些文件拷贝到V2上进行运行测试。如果用Raspberry pi转接，需要先用`ssh`登录到Raspberry pi上。
```shell
$ ssh pi@192.168.0.1
```
这里需要输入密码: `raspberry`。登录以后用`adb`连接到V2。前提是已经将USB3.0的数据线连接Raspberry pi和V2。
```shell
$ sudo adb shell
```
这样转接可以实现PC和V2上无线连接，但数据传输也需要转一次。首先，将数据从PC传到Raspberry pi:
```shell
$ scp <your files> pi@192.168.0.1:/home/pi/{your path}
```
反向传输的话，将收发路径位置反过来即可。接下来是将数据发送到V2：
```shell
adb push <your files> /slam/{your path}
```
一般V2上都把数据保存在`slam`路径下，可以在下面建一个自己的目录。这里的反向传输数据需要用到`adb pull`，后面就是文件加路径。


获取系统`root`权限后就可以运行`label_image`进行测试了，之后可以自行修改源代码加入自己想要的功能，然后按原来的流程编译通过即可。
```shell
$ su
$ ./label_image \
  --image=flowers.jpg \
  --input_layer=input \
  --output_layer=MobilenetV1/Predictions/Reshape_1 \
  --graph=/tmp/mobilenet_v1_1.0_224_frozen.pb \
  --labels=/tmp/imagenet_slim_labels.txt \
  --input_width=224 \
  --input_height=224
```

