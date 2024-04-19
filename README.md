# YOLOv8 on Jetson Nano

## 简介

本项目使用TensorRT在Jetson Nano运行YOLOv8，具体的效果如下：

![](https://github-production-user-asset-6210df.s3.amazonaws.com/53283758/324040992-f592af7d-87d6-43d8-a6f5-dbd4345426d1.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240419%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240419T164456Z&X-Amz-Expires=300&X-Amz-Signature=5b10702bc45426f4d940d1c9c27ab4795059304350306ed4672f774f1843f136&X-Amz-SignedHeaders=host&actor_id=53283758&key_id=0&repo_id=644678708)

## 需要的依赖

若想要运行代码，需要有一个Jetson Nano单板计算机（2GB或4GB的均可），其中的软件配置如下：

- Jetpack 4.6.1
- TensorRT 8.2.1.8
- OpenCV 4.1.1

## 使用方法

### 模型转换

1、导出ONNX模型

通过以下代码导出YOLOv8的ONNX文件：

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(imgsz=320, format='onnx')
```

2、转换为TensorRT的引擎文件

由于TensorRT的优化与硬件相关，不同硬件中转换的模型可能会出现错误。因此，首先将ONNX模型上传至Jetson Nano当中，然后通过trtexec进行转换。

```shell
trtexec --onnx=<ONNX file> --saveEngine=<output file>
```

### 运行代码

将模型存放至工程目录当中，修改代码的模型路径，然后编译并运行，即可得到对应结果。

```c++
YOLO model("../model/yolov8n.engine", logger);
```

