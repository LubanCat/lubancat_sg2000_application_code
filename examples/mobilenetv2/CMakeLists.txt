cmake_minimum_required(VERSION 3.10)

project(mobilenetv2 C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# tpu sdk
set(TPU_SDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../tpu_sdk/${TARGET_ARCH}/)
include_directories(${TPU_SDK_PATH}/include)
link_directories(${TPU_SDK_PATH}/lib)

# opencv
set(OPENCV_PATH ${TPU_SDK_PATH}/opencv)
include_directories(${OPENCV_PATH}/include)
link_directories(${OPENCV_PATH}/lib)

# Libs
set(CVI_LIBS ${CVI_LIBS} cviruntime cvikernel)
if(NOT CMAKE_CROSSCOMPILING)
  set(CVI_LIBS ${CVI_LIBS} cvicmodel)
endif()

set(OPENCV_LIBS ${OPENCV_LIBS} opencv_core opencv_imgcodecs opencv_imgproc)
if(NOT CMAKE_CROSSCOMPILING)
  set(OPENCV_LIBS ${OPENCV_LIBS} opencv_highgui)
endif()

set(EXTRA_LIBS ${EXTRA_LIBS} dl stdc++ pthread z)

# target
add_executable(mobilenetv2 mobilnetv2.cpp)
target_link_libraries(mobilenetv2
    ${CVI_LIBS}
    ${OPENCV_LIBS}
    ${EXTRA_LIBS})

add_executable(mobilenetv2_fused_preprocess mobilnetv2_fused_preprocess.cpp)
target_link_libraries(mobilenetv2_fused_preprocess
    ${CVI_LIBS}
    ${OPENCV_LIBS}
    ${EXTRA_LIBS})

# install
file(GLOB IMAGE_FILES "../data/cat.jpg")
file(GLOB LABEL_FILE "../data/synset_words.txt")
install(FILES ${IMAGE_FILES} DESTINATION .)
install(FILES ${LABEL_FILE} DESTINATION .)
install(TARGETS mobilenetv2  DESTINATION .)
install(TARGETS mobilenetv2_fused_preprocess  DESTINATION .)