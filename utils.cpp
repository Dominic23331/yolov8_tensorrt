//
// Created by DELL on 2024/4/10.
//
#include "utils.h"


/**
 * @brief Mix two images
 * @param srcImage Original image
 * @param mixImage Past image
 * @param startPoint Start point
 * @return Success or not
*/
bool MixImage(cv::Mat& srcImage, cv::Mat mixImage, cv::Point startPoint)
{

    if (!srcImage.data || !mixImage.data)
    {
        return false;
    }

    int addCols = startPoint.x + mixImage.cols > srcImage.cols ? 0 : mixImage.cols;
    int addRows = startPoint.y + mixImage.rows > srcImage.rows ? 0 : mixImage.rows;
    if (addCols == 0 || addRows == 0)
    {
        return false;
    }

    cv::Mat roiImage = srcImage(cv::Rect(startPoint.x, startPoint.y, addCols, addRows));

    mixImage.copyTo(roiImage, mixImage);
    return true;
}


/**
 * @brief Resize image
 * @param img Input image
 * @param w Resized width
 * @param h Resized height
 * @return Resized image and offset
*/
std::tuple<cv::Mat, int, int> resize(cv::Mat& img, int w, int h)
{
    cv::Mat result;

    int ih = img.rows;
    int iw = img.cols;

    float scale = MIN(float(w) / float(iw), float(h) / float(ih));
    int nw = iw * scale;
    int nh = ih * scale;

    cv::resize(img, img, cv::Size(nw, nh));
    result = cv::Mat::ones(cv::Size(w, h), CV_8UC1) * 128;
    cv::cvtColor(result, result, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    bool ifg = MixImage(result, img, cv::Point((w - nw) / 2, (h - nh) / 2));
    if (!ifg)
    {
        std::cerr << "MixImage failed" << std::endl;
        abort();
    }

    std::tuple<cv::Mat, int, int> res_tuple = std::make_tuple(result, (w - nw) / 2, (h - nh) / 2);

    return res_tuple;
}


std::vector<float> decode_cls(std::vector<float>& box)
{
    std::vector<float> cls_list(box.begin() + 4, box.end());
    float conf = *std::max_element(cls_list.begin(), cls_list.end());
    float cls = std::max_element(cls_list.begin(), cls_list.end()) - cls_list.begin();

    std::vector<float> result(box.begin(), box.begin() + 4);
    result.push_back(conf);
    result.push_back(cls);

    return result;
}


bool compare_boxes(const Box& b1, const Box& b2)
{
    return b1.conf < b2.conf;
}


float intersection_over_union(const Box& b1, const Box& b2)
{
    float x1 = std::max(b1.x1, b2.x1);
    float y1 = std::max(b1.y1, b2.y1);
    float x2 = std::min(b1.x2, b2.x2);
    float y2 = std::min(b1.y2, b2.y2);

    // get intersection
    float box_intersection = std::max((float)0, x2 - x1) * std::max((float)0, y2 - y1);

    // get union
    float area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
    float area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);
    float box_union = area1 + area2 - box_intersection;

    // To prevent the denominator from being zero, add a very small numerical value to the denominator
    float iou = box_intersection / (box_union + 0.0001);

    return iou;
}


std::vector<Box> non_maximum_suppression(std::vector<Box> boxes, float iou_thre)
{
    // Sort boxes based on confidence
    std::sort(boxes.begin(), boxes.end(), compare_boxes);

    std::vector<Box> result;
    std::vector<Box> temp;
    while (!boxes.empty())
    {
        temp.clear();

        Box chosen_box = boxes.back();
        boxes.pop_back();
        for (int i = 0; i < boxes.size(); i++)
        {
            if (boxes[i].cls != chosen_box.cls || intersection_over_union(boxes[i], chosen_box) < iou_thre)
                temp.push_back(boxes[i]);
        }

        boxes = temp;
        result.push_back(chosen_box);
    }
    return result;
}
