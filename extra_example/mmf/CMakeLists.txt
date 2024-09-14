cmake_minimum_required(VERSION 3.10)

project(video_dec_display C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}.")

# tpu sdk
set(TPU_SDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../tpu_sdk/${TARGET_ARCH}/)
include_directories(${TPU_SDK_PATH}/include)
link_directories(${TPU_SDK_PATH}/lib)

# middleware
set(MIDDLEWARE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../middleware/v2/${TARGET_ARCH})
include_directories(${MIDDLEWARE_PATH}/include)
include_directories(${MIDDLEWARE_PATH}/include/isp/cv181x)
link_directories(${MIDDLEWARE_PATH}/lib)
link_directories(${MIDDLEWARE_PATH}/lib/3rd)
set(MIDDLEWARE_COMMON_LIBS ${MIDDLEWARE_COMMON_LIBS} -lcli -lsys -lvi -lvo -lvpss -lrgn -lgdc -lvenc -lcvi_bin -lcvi_bin_isp -lisp -lisp_algo -lae -laf -lawb -lsns_full -latomic -lvdec -lmisc)
set(MIDDLEWARE_AUDIO_LIBS ${MIDDLEWARE_AUDIO_LIBS} -ldl  -lcvi_audio -lcvi_vqe -lcvi_VoiceEngine -lcvi_RES1 -ltinyalsa -lrt -lpthread -lcvi_ssp)
set(MIDDLEWARE_IVE_LIBS ${MIDDLEWARE_SAMPLE_LIBS} -lsys -lcvi_ive -latomic)
set(MIDDLEWARE_SAMPLE_LIBS ${MIDDLEWARE_SAMPLE_LIBS} -lsample -lini)

# libs
set(CVI_LIBS ${CVI_LIBS} cviruntime cvikernel)
if(NOT CMAKE_CROSSCOMPILING)
  set(CVI_LIBS ${CVI_LIBS} cvicmodel)
endif()

set(EXTRA_LIBS ${EXTRA_LIBS} dl stdc++ pthread)

add_executable(video_dec_display video_dec_display.c)
target_link_libraries(video_dec_display
    ${MIDDLEWARE_COMMON_LIBS}
    ${MIDDLEWARE_SAMPLE_LIBS}
    ${EXTRA_LIBS})

install(TARGETS video_dec_display  DESTINATION .)
