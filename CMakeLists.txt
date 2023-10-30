# TODO 1: Set the minimum required version of CMake to be 3.5
cmake_minimum_required(VERSION 3.5)

# TODO 2: Create a project named Tutorial
# set the project name
# TODO 7: Set the project version number as 1.0 in the above project command
project(yolo8_trt_cpp VERSION 1.0)

# TODO 6: Set the variable CMAKE_CXX_STANDARD to 11
#         and the variable CMAKE_CXX_STANDARD_REQUIRED to True
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# TODO 8: Use configure_file to configure and copy TutorialConfig.h.in to
#         TutorialConfig.h
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/YoloConfig.h.in ${PROJECT_BINARY_DIR}/YoloConfig.h)


# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)

include_directories(/usr/local/cuda-11.1/targets/x86_64-linux/include/)

# TODO 3: Add an executable called Tutorial to the project
# Hint: Be sure to specify the source file as tutorial.cxx
# add the executable
add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo_detector.cpp)

# TODO 9: Use target_include_directories to include ${PROJECT_BINARY_DIR}
target_include_directories(${PROJECT_NAME} PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    # glog
    # WKTParser
    # jsoncpp
    nvinfer
    nvonnxparser
    -L/usr/local/cuda/lib64/ -lcudart
    )
