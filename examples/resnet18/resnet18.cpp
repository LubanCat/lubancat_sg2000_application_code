#include <stdio.h>
#include <fstream>
#include <string>
#include <numeric>
#include <cviruntime.h>
#include <opencv2/opencv.hpp>

#define IMG_RESIZE_DIMS 256

void softmax(float *prob, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        prob[i] = exp(prob[i]);
        sum += prob[i];
    }
    for (int i = 0; i < n; ++i) {
        prob[i] /= sum;
    }
}

static void dump_tensors(CVI_TENSOR *tensors, int32_t num) {
  for (int32_t i = 0; i < num; i++) {
    printf("  [%d] %s, shape (%d,%d,%d,%d), count %zu, fmt %d\n",
        i,
        tensors[i].name,
        tensors[i].shape.dim[0],
        tensors[i].shape.dim[1],
        tensors[i].shape.dim[2],
        tensors[i].shape.dim[3],
        tensors[i].count,
        tensors[i].fmt);
  }
}

static void usage(char **argv) {
  printf("Usage:\n");
  printf("   %s cvimodel image.jpg label_file\n", argv[0]);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    usage(argv);
    exit(-1);
  }

  // load model file
  const char *model_file = argv[1];
  CVI_MODEL_HANDLE model = nullptr;
  int ret = CVI_NN_RegisterModel(model_file, &model);
  if (CVI_RC_SUCCESS != ret) {
    printf("CVI_NN_RegisterModel failed, err %d\n", ret);
    exit(1);
  }
  printf("CVI_NN_RegisterModel succeeded\n");

  // get input output tensors
  CVI_TENSOR *input_tensors;
  CVI_TENSOR *output_tensors;
  int32_t input_num;
  int32_t output_num;
  CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors,
                               &output_num);
  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
  assert(input);
  CVI_TENSOR *output = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, output_tensors, output_num);
  assert(output);

  printf("Input Tensor Number  : %d\n", input_num);
  dump_tensors(input_tensors, input_num);
  printf("Output Tensor Number : %d\n", output_num);
  dump_tensors(output_tensors, output_num);

  // nchw
  CVI_SHAPE shape = CVI_NN_TensorShape(input);
  int32_t height = shape.dim[2];
  int32_t width = shape.dim[3];

  // imread
  cv::Mat image;
  image = cv::imread(argv[2]);
  if (!image.data) {
    printf("Could not open or find the image\n");
    return -1;
  }

  // resize
  cv::resize(image, image, cv::Size(IMG_RESIZE_DIMS, IMG_RESIZE_DIMS)); // linear is default

  //Packed2Planar
  cv::Mat channels[3];
  for (int i = 0; i < 3; i++) {
    channels[i] = cv::Mat(image.rows, image.cols, CV_8SC1);
  }
  cv::split(image, channels);

  int8_t *ptr = (int8_t *)CVI_NN_TensorPtr(input);
  int channel_size = height * width;
  for (int i = 0; i < 3; ++i) {
    memcpy(ptr + i * channel_size, channels[i].data, channel_size);
  }

  // memcpy(ptr, image.data, height * width * 3);
  // run inference
  CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
  printf("CVI_NN_Forward succeeded\n");

  std::vector<std::string> labels;
  std::ifstream file(argv[3]);
  if (!file) {
    printf("Didn't find synset_words file\n");
    exit(1);
  } else {
    std::string line;
    while (std::getline(file, line)) {
      labels.push_back(std::string(line));
    }
  }

  int32_t top_num = 5;
  float *prob = (float *)CVI_NN_TensorPtr(output);
  int32_t count = CVI_NN_TensorCount(output);

  softmax(prob, count);

  // find top-k prob and cls
  std::vector<size_t> idx(count);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&prob](size_t idx_0, size_t idx_1) {return prob[idx_0] > prob[idx_1];});

  // show results.
  printf("------\n");
  for (size_t i = 0; i < top_num; i++) {
    int top_k_idx = idx[i];
    printf("  %f, idx %d", prob[top_k_idx], top_k_idx);
    if (!labels.empty())
      printf(", %s", labels[top_k_idx].c_str());
    printf("\n");
  }
  printf("------\n");
  CVI_NN_CleanupModel(model);
  printf("CVI_NN_CleanupModel succeeded\n");
  return 0;
}
