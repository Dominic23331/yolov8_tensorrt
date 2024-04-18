#include <iostream>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include "yolo.h"

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

int main() {
    YOLO model("../model/yolov8n.engine", logger);
    model.show();

    cv::Mat img = cv::imread("../img/tennis.jpeg");
    model.run(img);

    return 0;
}
