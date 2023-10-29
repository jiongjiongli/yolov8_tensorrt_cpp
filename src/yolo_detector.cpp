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
