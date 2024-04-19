//
// Created by DELL on 2024/4/10.
//

#ifndef YOLOV8_TENSORRT_UTILS_H
#define YOLOV8_TENSORRT_UTILS_H
#include <iostream>
#include <ostream>
#include <vector>
#include <tuple>
#include <string>
#include <opencv2/opencv.hpp>

struct Box
{
    float x1;
    float y1;
    float x2;
    float y2;
    std::string cls;
    float conf;

    Box()
    {
        x1 = 0;
        y1 = 0;
        x2 = 0;
        y2 = 0;
        cls = "";
        conf = 0;
    }
};


std::ostream &operator<<(std::ostream& cout, const Box& b);
bool MixImage(cv::Mat& srcImage, cv::Mat mixImage, cv::Point startPoint);
std::tuple<cv::Mat, int, int> resize(cv::Mat& img, int w, int h);
std::vector<float> decode_cls(std::vector<float>& box);
bool compare_boxes(const Box& b1, const Box& b2);
float intersection_over_union(const Box& b1, const Box& b2);
std::vector<Box> non_maximum_suppression(std::vector<Box> boxes, float iou_thre);
void draw_boxes(cv::Mat& img,
                const std::vector<Box>& boxes,
                const cv::Scalar& color=cv::Scalar(255, 0, 0),
                int thickness=2);
#endif //YOLOV8_TENSORRT_UTILS_H
