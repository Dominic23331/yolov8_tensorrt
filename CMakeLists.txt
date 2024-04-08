cmake_minimum_required(VERSION 3.10)
project(yolov8_tensorrt)

set(CMAKE_CXX_STANDARD 14)

#如果不需要指定OpenCV版本，代码如下
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
        ./src/opencv4/)
#添加OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# 添加CUDA10.2头文件
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)
# tensorRT
include_directories(/usr/local/cuda/include)
link_directories(/usr/lib/arrch64-linux-gnu)
#显示OpenCV_INCLUDE_DIRS的值
message(${OpenCV_INCLUDE_DIRS})

add_executable(yolov8_tensorrt main.cpp)
