#include <iostream>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include "yolo.h"
#include "utils.h"


class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;


int main()
{
    // init model
    YOLO model("../model/yolov8n.engine", logger);
    // show model
    model.show();
    // warmup
    model.warmup(30);
    // get benchmark
    model.benchmark();

    // inference model
    cv::Mat img = cv::imread("../img/zidane.jpeg");
    cv::Mat show_img;
    img.copyTo(show_img);
    std::vector<Box> boxes = model.run(img);
    draw_boxes(show_img, boxes);

    for (const auto & box : boxes)
        std::cout << box << std::endl;

    cv::imwrite("../result.jpg", show_img);

    return 0;
}
