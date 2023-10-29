
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

    private:
        nvinfer1::ICudaEngine *m_CudaEngine;
        nvinfer1::IExecutionContext *m_CudaContext;
        cudaStream_t m_CudaStream;

    private:
        float mConfThresh;
};

#endif
