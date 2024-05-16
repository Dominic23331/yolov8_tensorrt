//
// Created by DELL on 2024/4/8.
//
#include "yolo.h"


// set COCO classes
std::vector<std::string> classes = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
        "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

// set YOLO params
int YOLO::input_h = 320;
int YOLO::input_w = 320;
float YOLO::conf_threshold = 0.3;
float YOLO::iou_threshold = 0.65;

/**
 * YOLO constructor
 * @param model_path Engine file path
 * @param logger tensorrt logger
 */
YOLO::YOLO(const std::string &model_path, nvinfer1::ILogger &logger) {
    // load engine file
    std::ifstream engineStream(model_path, std::ios::binary);

    // check engine file exist.
    if (!engineStream.is_open()) {
        std::cout << "Cannot find model from: " + model_path << std::endl;
        exit(0);
    }

    engineStream.seekg(0, std::ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

    context->setBindingDimensions(0, nvinfer1::Dims4(1, 3, input_h, input_w));

    cudaStreamCreate(&stream);

    offset[0] = 0;
    offset[1] = 0;

    // Calculate output dimension by inputting image size
    out_dim_2 = (input_w / 8) * (input_h / 8) + (input_w / 16) * (input_h / 16) + (input_w / 32) * (input_h / 32);
    boxes_result.resize(out_dim_2 * 84);
}

/**
 * YOLO Destructor
 */
YOLO::~YOLO() {
    cudaFree(stream);
    cudaFree(buffer[0]);
    cudaFree(buffer[1]);
}


/**
 * Display model structure
 */
void YOLO::show() {
    for (int i = 0; i < engine->getNbBindings(); i++) {
        std::cout << "node: " << engine->getBindingName(i) << ", ";
        if (engine->bindingIsInput(i)) {
            std::cout << "type: input" << ", ";
        } else {
            std::cout << "type: output" << ", ";
        }
        nvinfer1::Dims dim = engine->getBindingDimensions(i);
        std::cout << "dimensions: ";
        for (int d = 0; d < dim.nbDims; d++) {
            std::cout << dim.d[d] << " ";
        }
        std::cout << "\n";
    }
}


/**
 * YOLO`s preprocess function
 * @param image input image
 * @return tensor
 */
std::vector<float> YOLO::preprocess(cv::Mat &image)
{
    // get resized image
    std::tuple<cv::Mat, int, int> resized = resize(image, input_w, input_h);
    cv::Mat resized_image = std::get<0>(resized);

    // get resize offset
    offset[0] = std::get<1>(resized);
    offset[1] = std::get<2>(resized);

    resized_image.convertTo(resized_image, CV_32F, 1. / 255.);

    std::vector<cv::Mat> channels(3);
    cv::split(resized_image, channels);

    std::vector<float> input_tensor;
    for (int i = 2; i >= 0; i--)
    {
        std::vector<float> tmp = channels[i].reshape(1, 1);
        input_tensor.insert(input_tensor.end(), tmp.begin(), tmp.end());
    }

    return input_tensor;
}


/**
 * YOLO`s postprocess
 * @param tensor output result, shape: [1, 84, output_dim_2]
 * @param img_w input image`s width
 * @param img_h input image`s height
 * @return boxes
 */
std::vector<Box> YOLO::postprocess(std::vector<float> tensor, int img_w, int img_h)
{
    // decode the boxes
    std::vector<Box> boxes;

    // [1, 84, out_dim_2]
    std::vector<float> b;
    for (int i=0; i < out_dim_2; i++)
    {
        b.clear();
        for (int j=0; j < 84; j++)
        {
            b.push_back(tensor[j * out_dim_2 + i]);
        }

        b = decode_cls(b);
        if (b[4] < conf_threshold)
            continue;

        Box db;
        db.x1 = b[0] - b[2] / 2;
        db.y1 = b[1] - b[3] / 2;
        db.x2 = b[0] + b[2] / 2;
        db.y2 = b[1] + b[3] / 2;
        db.conf = b[4];
        db.cls = classes[(int) b[5]];

        boxes.push_back(db);
    }

    // nms
    boxes = non_maximum_suppression(boxes, iou_threshold);

    // resize boxes
    for (auto & boxe : boxes)
    {
        boxe.x1 = MAX((boxe.x1 - offset[0]) * img_w / (input_w - 2 * offset[0]), 0);
        boxe.y1 = MAX((boxe.y1 - offset[1]) * img_h / (input_h - 2 * offset[1]), 0);
        boxe.x2 = MIN((boxe.x2 - offset[0]) * img_w / (input_w - 2 * offset[0]), img_w);
        boxe.y2 = MIN((boxe.y2 - offset[1]) * img_h / (input_h - 2 * offset[1]), img_h);
    }

    return boxes;
}


/**
 * inference the model
 * @param img input image
 * @return boxes
 */
std::vector<Box> YOLO::run(cv::Mat &img) {
    // get input image`s shape
    int img_h = img.rows;
    int img_w = img.cols;

    // preprocess
    auto input = preprocess(img);

    // upload to cuda
    cudaMalloc(&buffer[0], 3 * input_h * input_w * sizeof(float));
    cudaMalloc(&buffer[1], out_dim_2 * 84 * sizeof(float));
    cudaMemcpyAsync(buffer[0], input.data(), 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, stream);

    // inference
    context->enqueueV2(buffer, stream, nullptr);
    cudaStreamSynchronize(stream);

    // download from cuda
    cudaMemcpyAsync(boxes_result.data(), buffer[1], out_dim_2 * 84 * sizeof(float), cudaMemcpyDeviceToHost);

    // postprocess
    return postprocess(boxes_result, img_w, img_h);
}


/**
 * Warmup
 * @param epoch warmup epoch
 */
void YOLO::warmup(int epoch)
{
    std::cout << "Warm up." << std::endl;

    for (int step = 0; step <= epoch; ++step)
    {
        // use a random tensor to warmup
        cv::Mat randomImage(input_h, input_w, CV_8UC3);
        cv::randu(randomImage, cv::Scalar::all(0), cv::Scalar::all(255));
        run(randomImage);

        printProgressBar(step, epoch);
    }
    std::cout << std::endl;
}


/**
 * test YOLO speed benchmark
 */
void YOLO::benchmark()
{
    // set epoch
    int epoch = 100;
    // init benchmark vector
    std::vector<std::vector<double>> benchmarks(6);

    std::cout << "Start running benchmark..." << std::endl;

    for (int step = 0; step <= epoch; ++step)
    {
        cv::Mat randomImage(input_h, input_w, CV_8UC3);
        cv::randu(randomImage, cv::Scalar::all(0), cv::Scalar::all(255));

        std::clock_t t1 = std::clock();

        int img_h = randomImage.rows;
        int img_w = randomImage.cols;

        auto input = preprocess(randomImage);

        std::clock_t t2 = std::clock();

        cudaMalloc(&buffer[0], 3 * input_h * input_w * sizeof(float));
        cudaMalloc(&buffer[1], out_dim_2 * 84 * sizeof(float));

        cudaMemcpyAsync(buffer[0], input.data(), 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice, stream);

        std::clock_t t3 = std::clock();

        context->enqueueV2(buffer, stream, nullptr);

        std::clock_t t4 = std::clock();

        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(boxes_result.data(), buffer[1], out_dim_2 * 84 * sizeof(float), cudaMemcpyDeviceToHost);

        std::clock_t t5 = std::clock();

        postprocess(boxes_result, img_w, img_h);

        std::clock_t t6 = std::clock();

        benchmarks[0].push_back(double (t6 - t1) / (double) CLOCKS_PER_SEC);
        benchmarks[1].push_back(double (t2 - t1) / (double) CLOCKS_PER_SEC);
        benchmarks[2].push_back(double (t3 - t2) / (double) CLOCKS_PER_SEC);
        benchmarks[3].push_back(double (t4 - t3) / (double) CLOCKS_PER_SEC);
        benchmarks[4].push_back(double (t5 - t4) / (double) CLOCKS_PER_SEC);
        benchmarks[5].push_back(double (t6 - t5) / (double) CLOCKS_PER_SEC);

        printProgressBar(step, epoch);
    }
    std::cout << std::endl;

    double all_time, preprocess_time, cuda_upload_time, model_run_time, cuda_download_time, postprocess_time, fps;

    // Calculate the running time
    all_time = average(benchmarks[0]);
    preprocess_time = average(benchmarks[1]);
    cuda_upload_time = average(benchmarks[2]);
    model_run_time = average(benchmarks[3]);
    cuda_download_time = average(benchmarks[4]);
    postprocess_time = average(benchmarks[5]);
    fps = 1 / all_time;

    // show benchmark
    std::cout << "Waste time: " << all_time << std::endl;
    std::cout << "Preprocess time: " << preprocess_time << std::endl;
    std::cout << "Cuda upload time: " << cuda_upload_time << std::endl;
    std::cout << "Model inference time: " << model_run_time << std::endl;
    std::cout << "Cuda download time: " << cuda_download_time << std::endl;
    std::cout << "Postprocess time: " << postprocess_time << std::endl;
    std::cout << "FPS: " << fps << std::endl;
}
