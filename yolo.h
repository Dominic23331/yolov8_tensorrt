//
// Created by DELL on 2024/4/8.
//

#ifndef YOLOV8_TENSORRT_YOLO_H
#define YOLOV8_TENSORRT_YOLO_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include "utils.h"


class YOLO
{
public:
    YOLO(const std::string& model_path, nvinfer1::ILogger &logger, float conf_threshold=0.6, float iou_threshold=0.65);
    ~YOLO();

    static int input_w;
    static int input_h;
    static float conf_threshold;
    static float iou_threshold;

    void show();
    void run(cv::Mat& img);

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;

    void *buffer[2];
    int offset[2];
    int out_dim_2;

    std::vector<float> preprocess(cv::Mat& image);
    std::vector<Box> poseprocess(std::vector<float> tensor, int img_w, int img_h);
};


#endif //YOLOV8_TENSORRT_YOLO_H
