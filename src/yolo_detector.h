
#ifndef COMMON_DET_INFER_H
#define COMMON_DET_INFER_H

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include "opencv2/core.hpp"
#include <opencv2/freetype.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

class Yolo
{
    public:
        Yolo();
        ~Yolo();
        bool Init(const std::string& strModelName, float thresh);
        bool UnInit();
    private:
        void onnxToTrt(const std::string strModelName);
        void loadTrt(const std::string strName);
        void preprocessImage(const std::string& image_path,
            float* gpu_input,
            const Dims& dims);
        void postprocessResults(float *gpu_output,
                        const nvinfer1::Dims &dims,
                        int batch_size);
        void infer(const std::string& image_path);

    private:
        nvinfer1::ICudaEngine *m_CudaEngine;
        nvinfer1::IExecutionContext *m_CudaContext;
        cudaStream_t m_CudaStream;

    private:
        float mConfThresh;
};

#endif
