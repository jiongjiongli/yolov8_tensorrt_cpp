#include "yolo_detector.h"
#include "YoloConfig.h"
#include "./logging.h"
#define INPUT_NAME "images"
#define OUTPUT_NAME "output0"
#define TRT_FILE_SUFFIX ".cpp_trt"
using namespace nvinfer1;


static bool ifFileExists(const char *fileName)
{
    struct stat my_stat;
    return (stat(fileName, &my_stat) == 0);
}

Yolo::Yolo()
{

}

bool Yolo::Init(const std::string& strModelName, float conf_thresh)
{
    mConfThresh = conf_thresh;
    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + TRT_FILE_SUFFIX;
    if(ifFileExists(strTrtName.c_str()))
    {
        loadTrt(strTrtName);
    }
    else
    {
        onnxToTrt(strModelName);
    }
}

void Yolo::onnxToTrt(const std::string strModelName)
{
    std::cout << "Start onnxToTrt ..." << std::endl;
    Logger gLogger;
    //根据tensorrt pipeline 构建网络
    IBuilder* builder = createInferBuilder(gLogger);
    builder->setMaxBatchSize(1);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(strModelName.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(4ULL << 30);
    config->setFlag(BuilderFlag::kFP16);
    m_CudaEngine = builder->buildEngineWithConfig(*network, *config);

    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + TRT_FILE_SUFFIX;
    IHostMemory *gieModelStream = m_CudaEngine->serialize();
    std::string serialize_str;
    std::ofstream serialize_output_stream;
    serialize_str.resize(gieModelStream->size());
    memcpy((void*)serialize_str.data(),gieModelStream->data(),gieModelStream->size());
    serialize_output_stream.open(strTrtName.c_str());
    serialize_output_stream<<serialize_str;
    serialize_output_stream.close();
    m_CudaContext = m_CudaEngine->createExecutionContext();
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    std::cout << "Completed onnxToTrt !" << std::endl;
}

void Yolo::loadTrt(const std::string strName)
{
    std::cout << "Start loadTrt ..." << std::endl;
    Logger gLogger;
    IRuntime* runtime = createInferRuntime(gLogger);
    std::ifstream fin(strName);
    std::string cached_engine = "";
    while (fin.peek() != EOF)
    {
        std::stringstream buffer;
        buffer << fin.rdbuf();
        cached_engine.append(buffer.str());
    }
    fin.close();
    m_CudaEngine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    m_CudaContext = m_CudaEngine->createExecutionContext();
    runtime->destroy();
    std::cout << "Completed loadTrt !" << std::endl;
}

void Yolo::preprocessImage(const std::string& image_path,
                           float* gpu_input,
                           const Dims& dims)
{
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty())
    {
        std::cerr << "Input image " << image_path << " load failed\n";
        return;
    }
    cv::cuda::GpuMat gpu_frame;
    // upload image to GPU
    gpu_frame.upload(frame);
    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[0];
    auto input_size = cv::Size(input_width, input_height);
    // resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    std::vector< cv::cuda::GpuMat > chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
}

void Yolo::postprocessResults(float *gpu_output,
                        const Dims &dims,
                        int batch_size)
{
    // copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(dims) * batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    int num_channels = dims_i.d[1];
    int num_anchors = dims_i.d[2];

    float *max_pred_conf_pos = std::max_element(cpu_output + 4 * num_anchors, cpu_output + num_channels * num_anchors);
    std::cout << "max_pred_conf_pos:" << (max_pred_conf_pos - feat_blob) << endl;
    std::cout << "max_pred_conf_score:" << (*max_pred_conf_pos) << endl;
}

void Yolo::infer(const std::string& image_path)
{
    int batch_size = 1;
    std::vector<Dims> input_dims; // we expect only one input
    std::vector<Dims> output_dims; // and one output
    std::vector<void*> buffers(m_CudaEngine->getNbBindings()); // buffers for input and output data
    for (size_t i = 0; i < m_CudaEngine->getNbBindings(); ++i)
    {
        Dims dims_i = m_CudaEngine->getBindingDimensions(i);

        std::cout << "Dims at " << i << " " << "number_dims:" << dims_i.nbDims << " ";
        std::cout << "Dims values:"

        for (size_t dim_index = 0; dim_index < dims.nbDims; ++dim_index)
        {
            std::cout << dims_i.d[i] << " ";
        }

        auto binding_size = getSizeByDim(dims_i) * batch_size * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (m_CudaEngine->bindingIsInput(i))
        {
            input_dims.emplace_back(dims_i);
            std::cout << "input";
        }
        else
        {
            output_dims.emplace_back(dims_i);
            std::cout << "output";
        }

        std::cout << endl;
    }

    std::cout << "input_dims size:" << input_dims.size() << " " << "output_dims size:" << output_dims.size() << endl;

    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Error: Expect at least one input and one output for network!" << endl;
        return -1;
    }

    // preprocess input data
    preprocessImage(image_path,
                    (float*)buffers[0],
                    input_dims[0]);
    // inference
    m_CudaContext->enqueue(batch_size, buffers.data(), 0, nullptr);
    // post-process results
    postprocessResults((float*)buffers[1], output_dims[0], batch_size);

    for (void* buf : buffers)
    {
        cudaFree(buf);
    }
}

bool Yolo::UnInit()
{
    m_CudaContext->destroy();
    m_CudaEngine->destroy();
}

Yolo::~Yolo()
{
    UnInit();
}

int main(int argc, char *argv[])
{
    std::string strModelName = "/project/ev_sdk/model/best.onnx";
    float conf_thresh = 0.5;

    Yolo detector = Yolo();
    detector.Init(strModelName, conf_thresh);

    return 0;
}
