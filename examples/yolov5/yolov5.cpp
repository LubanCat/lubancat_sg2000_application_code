#include <stdio.h>
#include <fstream>
#include <string>
#include <numeric>
#include <cviruntime.h>
#include <opencv2/opencv.hpp>
#include <set>

#define IMG_RESIZE_DIMS 640
#define BGR_MEAN        0,0,0
#define INPUT_SCALE     0.0039216

#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 80
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

const int anchors[3][6] = {{10, 13, 16, 30, 33, 23},
                          {30, 61, 62, 45, 59, 119},
                          {116, 90, 156, 198, 373, 326}};

static const char* class_names[] = {
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
  "hair drier", "toothbrush"
};

typedef struct {
  CVI_TENSOR *input_tensors;
  CVI_TENSOR *output_tensors;
  int32_t input_num;
  int32_t output_num;

  int model_width;
  int model_height;
  bool is_quant;
  int batch;

  int padd_w;
  int padd_h;
  float scale;
} cvi_app_context_t;

typedef struct {
  cv::Rect_<float> rect;
  float prop;
  int cls_id;
} object_detect_result;

typedef struct {
  int id;
  int count;
  object_detect_result results[128];
} object_detect_result_list;

static void usage(char **argv) {
  printf("Usage:\n");
  printf("   %s cvimodel image.jpg \n", argv[0]);
  printf("ex: %s yolov5n_int8_sym.cvimodel cat.jpg \n", argv[0]);
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

cv::Mat letterbox(cvi_app_context_t *cvi_yolov5_ctx, cv::Mat &src, int h, int w)
{
  cv::Mat resize_img;

  int in_w = src.cols; // width
  int in_h = src.rows; // height
  int tar_w = w;
  int tar_h = h;

  float scale = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
  int inside_w = round(in_w * scale);
  int inside_h = round(in_h * scale);
  int padd_w = tar_w - inside_w;
  int padd_h = tar_h - inside_h;

  cv::resize(src, resize_img, cv::Size(inside_w, inside_h));

  padd_w = padd_w / 2;
  padd_h = padd_h / 2;

  int top = int(round(padd_h - 0.1));
  int bottom = int(round(padd_h + 0.1));
  int left = int(round(padd_w - 0.1));
  int right = int(round(padd_w + 0.1));
  cv::copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));

  cvi_yolov5_ctx->padd_w = padd_w;
  cvi_yolov5_ctx->padd_h = padd_h;
  cvi_yolov5_ctx->scale = scale;

  return resize_img;
}

static inline int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static inline float sigmoid(float x)
{
  return static_cast<float>(1.f / (1.f + exp(-x)));
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        if (order[i] == -1 || classIds[i] != filterId)
        {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static int postprocess(cvi_app_context_t *cvi_yolov5_ctx, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
{
  int validCount = 0;
  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;

  // (1*255*80*80) (1*255*40*40) (1*255*20*20)
  for (int n = 0; n < cvi_yolov5_ctx->output_num; n++) {
    if (!cvi_yolov5_ctx->output_tensors[n].name) {
      continue;
    }
    float *data = (float *)CVI_NN_TensorPtr(&cvi_yolov5_ctx->output_tensors[n]);
    CVI_SHAPE shape = CVI_NN_TensorShape(&cvi_yolov5_ctx->output_tensors[n]);
    int32_t grid_h = shape.dim[2];  // 80 40 20
    int32_t grid_w = shape.dim[3];
    printf("grid_h:%d  grid_w:%d \n", grid_h, grid_w);

    int grid_len = grid_h * grid_w;    // 80*80  40*40  20*20
    int stride = cvi_yolov5_ctx->model_height / grid_h; // 8、16、32

    int *anchor = (int *)anchors[n];

    for (int a = 0; a < 3; a++)
    {
      for (int i = 0; i < grid_h; i++)
      {
        for (int j = 0; j < grid_w; j++)
        {
          float *obj = data + (PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j;
          float box_confidence = sigmoid(obj[0]);
          if (box_confidence >= conf_threshold)
          {
            int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
            float *in_ptr = data + offset;
            float box_x = sigmoid(*in_ptr) * 2.0 - 0.5;
            float box_y = sigmoid(in_ptr[grid_len]) * 2.0 - 0.5;
            float box_w = sigmoid(in_ptr[2 * grid_len]) * 2.0;
            float box_h = sigmoid(in_ptr[3 * grid_len]) * 2.0;
            box_x = (box_x + j) * (float)stride;
            box_y = (box_y + i) * (float)stride;
            box_w = box_w * box_w * (float)anchor[a * 2];
            box_h = box_h * box_h * (float)anchor[a * 2 + 1];
            box_x -= (box_w / 2.0);
            box_y -= (box_h / 2.0);

            // find class index with max class score
            float maxClassProbs = in_ptr[5 * grid_len];
            int maxClassId = 0;
            for (int k = 1; k < OBJ_CLASS_NUM; ++k)
            {
              float prob = in_ptr[(5 + k) * grid_len];
              if (prob > maxClassProbs)
              {
                maxClassId = k;
                maxClassProbs = prob;
              }
            }
            if (maxClassProbs > conf_threshold)
            {
              validCount++;
              objProbs.push_back(sigmoid(maxClassProbs) * box_confidence);
              classId.push_back(maxClassId);
              filterBoxes.push_back(box_x);
              filterBoxes.push_back(box_y);
              filterBoxes.push_back(box_w);
              filterBoxes.push_back(box_h);
            }
          }
        }
      }
    }

  }

  printf("get detection nums: %d\n", validCount);
  if (validCount <= 0)
  {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i)
  {
      indexArray.push_back(i);
  }
  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set)
  {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  od_results->count = 0;

  /* box valid detect target */
  for (int i = 0; i < validCount; ++i)
  {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
    {
      continue;
    }
    int n = indexArray[i];

    float x1 = filterBoxes[n * 4 + 0] - cvi_yolov5_ctx->padd_w;
    float y1 = filterBoxes[n * 4 + 1] - cvi_yolov5_ctx->padd_h;
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];

    od_results->results[last_count].rect.x = (int)(clamp(x1, 0, cvi_yolov5_ctx->model_width) / cvi_yolov5_ctx->scale);
    od_results->results[last_count].rect.y = (int)(clamp(y1, 0, cvi_yolov5_ctx->model_height) / cvi_yolov5_ctx->scale);
    od_results->results[last_count].rect.width = (int)(clamp(x2, 0, cvi_yolov5_ctx->model_width) / cvi_yolov5_ctx->scale - od_results->results[last_count].rect.x);
    od_results->results[last_count].rect.height = (int)(clamp(y2, 0, cvi_yolov5_ctx->model_height) / cvi_yolov5_ctx->scale - od_results->results[last_count].rect.y);

    od_results->results[last_count].prop = obj_conf;
    od_results->results[last_count].cls_id = id;
    last_count++;
  }
  od_results->count = last_count;
  return 0;

}

int main(int argc, char **argv) {
  if (argc != 3) {
    usage(argv);
    exit(-1);
  }

  cvi_app_context_t cvi_yolov5_ctx;
  memset(&cvi_yolov5_ctx, 0, sizeof(cvi_app_context_t));

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
  CVI_NN_GetInputOutputTensors(model, &cvi_yolov5_ctx.input_tensors, &cvi_yolov5_ctx.input_num, &cvi_yolov5_ctx.output_tensors,
                               &cvi_yolov5_ctx.output_num);
  printf("Input Tensor Number  : %d\n", cvi_yolov5_ctx.input_num);
  dump_tensors(cvi_yolov5_ctx.input_tensors, cvi_yolov5_ctx.input_num);
  printf("Output Tensor Number : %d\n", cvi_yolov5_ctx.output_num);
  dump_tensors(cvi_yolov5_ctx.output_tensors, cvi_yolov5_ctx.output_num);

  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, cvi_yolov5_ctx.input_tensors, cvi_yolov5_ctx.input_num);
  assert(input);

  CVI_SHAPE shape = CVI_NN_TensorShape(input);
  cvi_yolov5_ctx.model_height = shape.dim[2];
  cvi_yolov5_ctx.model_width = shape.dim[3];
  printf("model_height:%d  model_width:%d \n", cvi_yolov5_ctx.model_height, cvi_yolov5_ctx.model_width);

  // imread
  cv::Mat image;
  image = cv::imread(argv[2]);
  if (!image.data) {
    printf("Could not open or find the image\n");
    return -1;
  }

  // preprocess: letterbox,resize,bgr2rgb
  cv::Mat resized_img = letterbox(&cvi_yolov5_ctx, image, IMG_RESIZE_DIMS, IMG_RESIZE_DIMS);
  cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
  printf("resized_img: %d, %d \n",resized_img.cols, resized_img.rows);

  // Packed2Planar (rgb nchw)
  cv::Mat channels[3];
  for (int i = 0; i < 3; i++) {
    channels[i] = cv::Mat(resized_img.rows, resized_img.cols, CV_8SC1);
  }
  cv::split(resized_img, channels);

  int8_t *ptr = (int8_t *)CVI_NN_TensorPtr(input);
  int channel_size = cvi_yolov5_ctx.model_height * cvi_yolov5_ctx.model_width;
  for (int i = 0; i < 3; ++i) {
    memcpy(ptr + i * channel_size, channels[i].data, channel_size);
  }

  // run inference
  CVI_NN_Forward(model, cvi_yolov5_ctx.input_tensors, cvi_yolov5_ctx.input_num, cvi_yolov5_ctx.output_tensors, cvi_yolov5_ctx.output_num);
  printf("CVI_NN_Forward succeeded\n");

  // postprocess
  const float nms_threshold = 0.45;      // 默认的NMS阈值
  const float box_conf_threshold = 0.25; // 默认的置信度阈值
  object_detect_result_list od_results;
  postprocess(&cvi_yolov5_ctx, box_conf_threshold, nms_threshold, &od_results);

  // draw box on image
  char text[256];
  printf("%d objects are detected: \n", od_results.count);
  for (int i = 0; i < od_results.count; i++)
  {
    object_detect_result *det_result = &(od_results.results[i]);
    printf(" %s @ (%f %f %f %f) %.3f\n", class_names[det_result->cls_id],
            det_result->rect.x, det_result->rect.y,
            det_result->rect.width, det_result->rect.height,
            det_result->prop);
    int x1 = det_result->rect.x;
    int y1 = det_result->rect.y;

    cv::rectangle(image, det_result->rect, cv::Scalar(255, 0, 0));
    sprintf(text, "%s %.1f%%", class_names[det_result->cls_id], det_result->prop * 100);
    cv::putText(image, text, cv::Point(x1, y1 - 6), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0,0,255), 1, cv::LINE_AA);
  }

  // save
  cv::imwrite("./out.jpg", image);

  CVI_NN_CleanupModel(model);
  return 0;
}
