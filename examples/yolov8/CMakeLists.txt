cmake_minimum_required(VERSION 3.10)

project(yolov8 C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}.")

# tpu sdk
set(TPU_SDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../tpu_sdk/${TARGET_ARCH}/)
include_directories(${TPU_SDK_PATH}/include)
link_directories(${TPU_SDK_PATH}/lib)

# opencv
set(OPENCV_PATH ${TPU_SDK_PATH}/opencv)
include_directories(${OPENCV_PATH}/include)
link_directories(${OPENCV_PATH}/lib)

set(CVI_LIBS ${CVI_LIBS} cviruntime cvikernel)
if(NOT CMAKE_CROSSCOMPILING)
  set(CVI_LIBS ${CVI_LIBS} cvicmodel)
endif()

set(OPENCV_LIBS ${OPENCV_LIBS} opencv_core opencv_imgcodecs opencv_imgproc)
if(NOT CMAKE_CROSSCOMPILING)
  set(OPENCV_LIBS ${OPENCV_LIBS} opencv_highgui)
endif()

set(EXTRA_LIBS ${EXTRA_LIBS} dl stdc++ pthread z)

add_executable(yolov8 yolov8.cpp)
target_link_libraries(yolov8
    ${CVI_LIBS}
    ${OPENCV_LIBS}
    ${EXTRA_LIBS})
install(TARGETS yolov8  DESTINATION .)

