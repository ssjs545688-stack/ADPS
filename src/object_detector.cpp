#include <algorithm>
#include <array>
#include <cjson/cJSON.h>
#include <cmath>   // math.h 대신 C++ 래퍼 사용 (중복 포함 제거)
#include <cstdio>  // stdio.h 중복 제거
#include <cstdlib> // stdlib.h 중복 제거
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>
 
#ifdef USE_ONNX_RUNTIME
#include <onnxruntime/onnxruntime_cxx_api.h>
#endif

#include "object_detector.h"

// =============================================================================
// object_detector.cpp
// 목적 : 색상 기반 객체 검출·추적·충돌 위험도 평가 파이프라인 구현
//
// §1  : 구조체 / 매크로 (object_detector.h 참조)
// §2  : HSV 색상 후보 검출 + 모폴로지(Open/Close)
// §3  : 윤곽선 검출 + 형태 분류 (삼각형/사각형/원)
// §4  : Depth 처리 – 내부 60% 영역 + 30분위수 강건 거리
// §5  : 구역(LEFT/CENTER/RIGHT) 분류 + CentroidTracker
// §6  : Closing Speed, TTC, 위험 단계(SAFE/WARNING/STOP)
// §7  : EMA 거리 필터 + 카메라-LiDAR 구역 융합
// §8  : CSV 로깅(객체·프레임 단위) + 연속성 검증
// =============================================================================

// =============================================================================
// §5: CentroidTracker 정적 멤버 정의
// =============================================================================
const ObjectDetectorConfig *CentroidTracker::cfg_ptr_ = nullptr;
double CentroidTracker::dt_ = 1.0 / 30.0;
double CentroidTracker::cfg_ema_alpha_ = 0.25;

// =============================================================================
// §8: 로그 파일 핸들 (정적 전역)
// =============================================================================
static FILE *g_obj_log = nullptr;   // 객체 단위 상세 CSV
static FILE *g_frame_log = nullptr; // 프레임 단위 요약 CSV

// =============================================================================
// §1: 기본 설정값 초기화
// =============================================================================
void init_detector_config(ObjectDetectorConfig *cfg) {
  if (!cfg)
    return;

  // §2: HSV – 빨강(이중 구간) / 노랑
  const int rl1[3] = {0, 120, 70};
  const int ru1[3] = {10, 255, 255};
  const int rl2[3] = {170, 120, 70};
  const int ru2[3] = {180, 255, 255};
  const int yl[3] = {18, 100, 100};
  const int yu[3] = {35, 255, 255};
  for (int i = 0; i < 3; ++i) {
    cfg->red_lower1[i] = rl1[i];
    cfg->red_upper1[i] = ru1[i];
    cfg->red_lower2[i] = rl2[i];
    cfg->red_upper2[i] = ru2[i];
    cfg->yellow_lower[i] = yl[i];
    cfg->yellow_upper[i] = yu[i];
  }

  // 새 파라미터 기본값 (그림자/엣지/ROI)
  cfg->shadow_hsv_upper[0] = 255;
  cfg->shadow_hsv_upper[1] = 40;
  cfg->shadow_hsv_upper[2] = 50;
  cfg->edge_density_threshold = 0.15; // 15% 이상
  cfg->obj_roi_top_ratio = 0.45;
  cfg->obj_roi_bottom_ratio = 0.85;
  cfg->max_object_width_ratio = 0.5;
  cfg->min_object_area_px2 = 500.0;
  cfg->max_object_area_ratio = 0.25;
  cfg->min_object_aspect_ratio = 0.5;
  cfg->max_object_aspect_ratio = 2.5;

  cfg->enable_deep_inference = false;
  cfg->deep_model_path[0] = '\0';
  cfg->deep_confidence_threshold = 0.5;

  cfg->black_car_enable = true;
  cfg->black_car_hsv_upper[0] = 180;
  cfg->black_car_hsv_upper[1] = 80;
  cfg->black_car_hsv_upper[2] = 80;

  cfg->enable_clahe = true;
  cfg->clahe_clip_limit = 2;
  cfg->clahe_tile_grid_size = 16;

  cfg->enable_shadow_texture = true;
  cfg->shadow_texture_window = 15;
  cfg->shadow_texture_variance_threshold = 0.12;

  cfg->enable_multichannel_edge_fusion = true;

  // §2: 모폴로지
  cfg->morph_open_size = 5;
  cfg->morph_close_size = 9;

  // §3: 윤곽선 필터
  cfg->min_contour_area = 500.0;
  cfg->max_contour_area = 50000.0;
  cfg->poly_epsilon_ratio = 0.04;
  cfg->circle_threshold = 0.75;

  // §4: Depth (기본 비활성화 – Depth 카메라 없는 환경 대응)
  cfg->use_depth = false;
  cfg->depth_scale = 0.001;
  cfg->depth_inner_ratio = 0.6;
  cfg->depth_percentile = 30.0;
  cfg->max_valid_depth_m = 15.0;
  cfg->min_depth_samples = 10;

  // [신규] §4b: 모노큘러 거리 추정 기본값
  // focal_length_px 기본값 900은 1280px 폭, 수평화각(HFOV) 약 70도인
  // 일반적인 대시캠 기준 근사치(f_px = width / (2*tan(HFOV/2)))일 뿐이라,
  // 실제 카메라와 다르면 거리 오차가 클 수 있음. 정확도가 중요하면
  // 아래 estimate_monocular_distance() 주석의 실측 보정 절차를 따를 것.
  cfg->mono_depth_enable = true;
  cfg->focal_length_px = 900.0;
  cfg->mono_object_width_m = 1.8;

  // §5: 추적기
  cfg->max_match_dist_px = 80.0;
  cfg->max_missing_frames = 15;

  // §6: 위험도
  cfg->warn_dist_m = 4.0;
  cfg->stop_dist_m = 2.0;
  cfg->warn_ttc_s = 3.0;
  cfg->stop_ttc_s = 1.5;
  cfg->min_danger_frames = 3;
  cfg->max_distance_jump_ratio = 0.5;
  cfg->speed_ema_alpha = 0.3;

  // §7: EMA
  cfg->ema_alpha = 0.25;

  // §8: 로깅
  cfg->enable_logging = true;
  std::strncpy(cfg->obj_log_path, "logs/objects_detail.csv", 255);
  std::strncpy(cfg->frame_log_path, "logs/frame_summary.csv", 255);
}

// =============================================================================
// JSON 파일로부터 설정 로드
// =============================================================================
bool load_detector_config_json(const char *filepath,
                               ObjectDetectorConfig *cfg) {
  init_detector_config(cfg); // 먼저 기본값 초기화

  FILE *fp = std::fopen(filepath, "r");
  if (!fp) {
    std::printf("[ObjectDetector] 설정 파일 없음: %s → 기본값 사용\n",
                filepath);
    return false;
  }
  std::fseek(fp, 0, SEEK_END);
  long len = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  char *buf = static_cast<char *>(std::malloc(len + 1));
  if (!buf) {
    std::fclose(fp);
    return false;
  }
  size_t read_len =
      std::fread(buf, 1, len, fp); // [수정] 반환값 체크: 부분 읽기 방지
  buf[read_len] = '\0';
  std::fclose(fp);

  cJSON *root = cJSON_Parse(buf);
  std::free(buf);
  if (!root) {
    std::printf("[ObjectDetector] JSON 파싱 오류: %s\n", filepath);
    return false;
  }

  // ── 헬퍼 람다 ─────────────────────────────────────────────────────────
  auto arr3i = [&](cJSON *parent, const char *key, int dst[3]) {
    cJSON *arr = cJSON_GetObjectItem(parent, key);
    if (!arr)
      return;
    for (int i = 0; i < 3 && i < cJSON_GetArraySize(arr); ++i)
      dst[i] = cJSON_GetArrayItem(arr, i)->valueint;
  };
  auto geti = [](cJSON *p, const char *k, int def) -> int {
    cJSON *v = cJSON_GetObjectItem(p, k);
    return v ? v->valueint : def;
  };
  auto getd = [](cJSON *p, const char *k, double def) -> double {
    cJSON *v = cJSON_GetObjectItem(p, k);
    return v ? v->valuedouble : def;
  };
  auto getb = [](cJSON *p, const char *k, bool def) -> bool {
    cJSON *v = cJSON_GetObjectItem(p, k);
    return v ? (cJSON_IsTrue(v) != 0) : def;
  };
  auto gets = [](cJSON *p, const char *k, char *dst, int sz) {
    cJSON *v = cJSON_GetObjectItem(p, k);
    if (v && v->valuestring)
      std::strncpy(dst, v->valuestring, sz - 1);
  };

  // ── §2: HSV ────────────────────────────────────────────────────────────
  cJSON *hsv = cJSON_GetObjectItem(root, "hsv");
  if (hsv) {
    arr3i(hsv, "red_lower1", cfg->red_lower1);
    arr3i(hsv, "red_upper1", cfg->red_upper1);
    arr3i(hsv, "red_lower2", cfg->red_lower2);
    arr3i(hsv, "red_upper2", cfg->red_upper2);
    arr3i(hsv, "yellow_lower", cfg->yellow_lower);
    arr3i(hsv, "yellow_upper", cfg->yellow_upper);
    arr3i(hsv, "shadow_upper", cfg->shadow_hsv_upper);
  }

  // ── 새 파라미터 (Edge, ROI) ───────────────────────────────────────────
  cJSON *edg = cJSON_GetObjectItem(root, "hybrid");
  if (edg) {
    cfg->edge_density_threshold =
        getd(edg, "edge_density", cfg->edge_density_threshold);
    cfg->obj_roi_top_ratio = getd(edg, "roi_top", cfg->obj_roi_top_ratio);
    cfg->obj_roi_bottom_ratio =
        getd(edg, "roi_bottom", cfg->obj_roi_bottom_ratio);
    cfg->max_object_width_ratio =
        getd(edg, "max_object_width_ratio", cfg->max_object_width_ratio);
    cfg->enable_clahe = getb(edg, "enable_clahe", cfg->enable_clahe);
    cfg->clahe_clip_limit =
        geti(edg, "clahe_clip_limit", cfg->clahe_clip_limit);
    cfg->clahe_tile_grid_size =
        geti(edg, "clahe_tile_grid_size", cfg->clahe_tile_grid_size);
    cfg->enable_shadow_texture =
        getb(edg, "enable_shadow_texture", cfg->enable_shadow_texture);
    cfg->shadow_texture_window =
        geti(edg, "shadow_texture_window", cfg->shadow_texture_window);
    cfg->shadow_texture_variance_threshold =
        getd(edg, "shadow_texture_variance_threshold",
             cfg->shadow_texture_variance_threshold);
    cfg->enable_multichannel_edge_fusion =
        getb(edg, "enable_multichannel_edge_fusion",
             cfg->enable_multichannel_edge_fusion);
  }

  cJSON *bc = cJSON_GetObjectItem(root, "black_car");
  if (bc) {
    cfg->black_car_enable = getb(bc, "enable", cfg->black_car_enable);
    arr3i(bc, "hsv_upper", cfg->black_car_hsv_upper);
  }

  cJSON *dl = cJSON_GetObjectItem(root, "deep");
  if (dl) {
    cfg->enable_deep_inference = getb(dl, "enable", cfg->enable_deep_inference);
    gets(dl, "model_path", cfg->deep_model_path, 256);
    cfg->deep_confidence_threshold =
        getd(dl, "confidence_threshold", cfg->deep_confidence_threshold);
  }

  // ── §2: 모폴로지 ───────────────────────────────────────────────────────
  cJSON *mo = cJSON_GetObjectItem(root, "morphology");
  if (mo) {
    cfg->morph_open_size = geti(mo, "open_kernel_size", cfg->morph_open_size);
    cfg->morph_close_size =
        geti(mo, "close_kernel_size", cfg->morph_close_size);
  }

  // ── §3: 윤곽선 ─────────────────────────────────────────────────────────
  cJSON *co = cJSON_GetObjectItem(root, "contour");
  if (co) {
    cfg->min_contour_area = getd(co, "min_area_px2", cfg->min_contour_area);
    cfg->max_contour_area = getd(co, "max_area_px2", cfg->max_contour_area);
    cfg->min_object_area_px2 =
        getd(co, "min_object_area_px2", cfg->min_object_area_px2);
    cfg->max_object_area_ratio =
        getd(co, "max_object_area_ratio", cfg->max_object_area_ratio);
    cfg->min_object_aspect_ratio =
        getd(co, "min_aspect_ratio", cfg->min_object_aspect_ratio);
    cfg->max_object_aspect_ratio =
        getd(co, "max_aspect_ratio", cfg->max_object_aspect_ratio);
    cfg->poly_epsilon_ratio =
        getd(co, "poly_epsilon_ratio", cfg->poly_epsilon_ratio);
    cfg->circle_threshold = getd(co, "circle_threshold", cfg->circle_threshold);
  }

  // ── §4: Depth ──────────────────────────────────────────────────────────
  cJSON *dp = cJSON_GetObjectItem(root, "depth");
  if (dp) {
    cfg->use_depth = getb(dp, "use_depth", cfg->use_depth);
    cfg->depth_scale = getd(dp, "depth_scale", cfg->depth_scale);
    cfg->depth_inner_ratio = getd(dp, "inner_ratio", cfg->depth_inner_ratio);
    cfg->depth_percentile = getd(dp, "percentile", cfg->depth_percentile);
    cfg->max_valid_depth_m =
        getd(dp, "max_valid_depth_m", cfg->max_valid_depth_m);
    cfg->min_depth_samples =
        geti(dp, "min_depth_samples", cfg->min_depth_samples);
    // [신규] 모노큘러 거리 추정 파라미터 (없으면 기본값 유지 - 하위호환)
    cfg->mono_depth_enable =
        getb(dp, "mono_depth_enable", cfg->mono_depth_enable);
    cfg->focal_length_px = getd(dp, "focal_length_px", cfg->focal_length_px);
    cfg->mono_object_width_m =
        getd(dp, "mono_object_width_m", cfg->mono_object_width_m);
  }

  // ── §5: 추적기 ─────────────────────────────────────────────────────────
  cJSON *tr = cJSON_GetObjectItem(root, "tracker");
  if (tr) {
    cfg->max_match_dist_px =
        getd(tr, "max_match_dist_px", cfg->max_match_dist_px);
    cfg->max_missing_frames =
        geti(tr, "max_missing_frames", cfg->max_missing_frames);
  }

  // ── §6: 위험도 ─────────────────────────────────────────────────────────
  cJSON *ri = cJSON_GetObjectItem(root, "risk");
  if (ri) {
    cfg->warn_dist_m = getd(ri, "warn_dist_m", cfg->warn_dist_m);
    cfg->stop_dist_m = getd(ri, "stop_dist_m", cfg->stop_dist_m);
    cfg->warn_ttc_s = getd(ri, "warn_ttc_s", cfg->warn_ttc_s);
    cfg->stop_ttc_s = getd(ri, "stop_ttc_s", cfg->stop_ttc_s);
    cfg->min_danger_frames =
        geti(ri, "min_danger_frames", cfg->min_danger_frames);
    cfg->max_distance_jump_ratio =
        getd(ri, "max_distance_jump_ratio", cfg->max_distance_jump_ratio);
    cfg->speed_ema_alpha = getd(ri, "speed_ema_alpha", cfg->speed_ema_alpha);
  }

  // ── §7: EMA ────────────────────────────────────────────────────────────
  cJSON *em = cJSON_GetObjectItem(root, "ema");
  if (em)
    cfg->ema_alpha = getd(em, "alpha", cfg->ema_alpha);

  // ── §8: 로깅 ───────────────────────────────────────────────────────────
  cJSON *lg = cJSON_GetObjectItem(root, "logging");
  if (lg) {
    cfg->enable_logging = getb(lg, "enable", cfg->enable_logging);
    gets(lg, "obj_log_path", cfg->obj_log_path, 256);
    gets(lg, "frame_log_path", cfg->frame_log_path, 256);
  }

  cJSON_Delete(root);
  std::printf("[ObjectDetector] 설정 로드 완료: %s\n", filepath);
  return true;
}

#ifdef USE_ONNX_RUNTIME
DeepObjectDetector::DeepObjectDetector(const std::string &model_path,
                                       int num_threads, bool use_cuda)
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                      "DeepObjectDetector")),
      session_options_(), session_(nullptr), input_names_storage_(),
      output_names_storage_(), input_names_(), output_names_(), input_shape_(),
      input_size_(640, 640), initialized_(false) {
  initialize_model(model_path, use_cuda, num_threads);
}

void DeepObjectDetector::initialize_model(const std::string &model_path,
                                          bool use_cuda, int num_threads) {
  try {
    session_options_.SetIntraOpNumThreads(num_threads);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#ifdef USE_CUDA
    if (use_cuda) {
      Ort::ThrowOnError(
          OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0));
    }
#endif
#ifdef _WIN32
    std::wstring w_model_path(model_path.begin(), model_path.end());
    session_ = std::make_unique<Ort::Session>(*env_, w_model_path.c_str(),
                                              session_options_);
#else
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(),
                                              session_options_);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_->GetInputCount();
    input_names_storage_.reserve(num_inputs);
    input_names_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      Ort::AllocatedStringPtr name_ptr =
          session_->GetInputNameAllocated(i, allocator);
      input_names_storage_.emplace_back(name_ptr.get()); // string으로 복사
      input_names_.push_back(input_names_storage_.back().c_str());
      Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      input_shape_ = tensor_info.GetShape();
      if (input_shape_.size() >= 4) {
        input_size_.width = static_cast<int>(input_shape_[3]);
        input_size_.height = static_cast<int>(input_shape_[2]);
      }
    }

    size_t num_outputs = session_->GetOutputCount();
    output_names_storage_.reserve(num_outputs);
    output_names_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
      Ort::AllocatedStringPtr name_ptr =
          session_->GetOutputNameAllocated(i, allocator);
      output_names_storage_.emplace_back(name_ptr.get()); // string으로 복사
      output_names_.push_back(output_names_storage_.back().c_str());
    }

    initialized_ = true;
  } catch (const Ort::Exception &e) {
    std::fprintf(stderr, "[DeepObjectDetector] ONNXRuntime 초기화 실패: %s\n",
                 e.what());
    initialized_ = false;
  }
}

std::vector<float> DeepObjectDetector::preprocess(const cv::Mat &bgr,
                                                  const cv::Size &target_size,
                                                  float &scale,
                                                  cv::Point &pad) const {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  int orig_w = rgb.cols;
  int orig_h = rgb.rows;
  int target_w = target_size.width;
  int target_h = target_size.height;
  scale = std::min(static_cast<float>(target_w) / orig_w,
                   static_cast<float>(target_h) / orig_h);
  int resized_w = static_cast<int>(std::round(orig_w * scale));
  int resized_h = static_cast<int>(std::round(orig_h * scale));

  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(resized_w, resized_h));

  pad.x = (target_w - resized_w) / 2;
  pad.y = (target_h - resized_h) / 2;
  cv::Mat letterbox = cv::Mat::zeros(target_h, target_w, CV_8UC3);
  resized.copyTo(letterbox(cv::Rect(pad.x, pad.y, resized_w, resized_h)));
  letterbox.convertTo(letterbox, CV_32F, 1.0f / 255.0f);

  std::vector<float> tensor;
  tensor.resize(3 * target_h * target_w);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < target_h; ++y) {
      for (int x = 0; x < target_w; ++x) {
        tensor[c * target_h * target_w + y * target_w + x] =
            letterbox.at<cv::Vec3f>(y, x)[c];
      }
    }
  }
  return tensor;
}

std::vector<DetectedObject> DeepObjectDetector::postprocess(
    const std::vector<float> &output, float scale, const cv::Point &pad,
    double confidence_threshold, const cv::Size &orig_size) const {
  std::vector<DetectedObject> detections;
  if (output.empty())
    return detections;

  // YOLOv8 ONNX output shape is typically [1, 84, 8400] (flattened)
  // 84 = 4 (cx, cy, w, h) + 80 class scores
  // 8400 = number of anchors for 640x640 input
  size_t num_anchors = 8400;
  size_t num_channels = 84;

  if (output.size() > 0) {
    if (output.size() % num_anchors == 0) {
      num_channels = output.size() / num_anchors;
    } else {
      num_anchors = output.size() / num_channels;
    }
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (size_t i = 0; i < num_anchors; ++i) {
    float max_score = 0.0f;
    int class_id = -1;

    // 1. 클래스별 점수 중 최대값 찾기 (인덱스 4부터 클래스 스코어)
    for (size_t c = 4; c < num_channels; ++c) {
      float score = output[c * num_anchors + i];
      if (score > max_score) {
        max_score = score;
        class_id = static_cast<int>(c - 4);
      }
    }

    if (max_score < confidence_threshold)
      continue;

    // 2. 차량 관련 클래스(2:car, 3:motorcycle, 5:bus, 7:truck)만 필터링
    bool is_vehicle =
        (class_id == 2 || class_id == 3 || class_id == 5 || class_id == 7);
    if (!is_vehicle)
      continue;

    // 3. 바운딩 박스 복원 (cx, cy, w, h)
    float cx = output[0 * num_anchors + i];
    float cy = output[1 * num_anchors + i];
    float w = output[2 * num_anchors + i];
    float h = output[3 * num_anchors + i];

    // [핵심 수정] preprocess()는 Letterbox 방식으로 동작함
    // scale = min(640/orig_w, 640/orig_h), pad = 검은 여백 오프셋
    // 따라서 역연산: (텐서 좌표 - 패딩) / scale = 원본 좌표
    // 단순 비율 곱셈(prev 방식)은 패딩을 무시하여 박스가 하늘로 뜨는 원인이었음
    cx = (cx - pad.x) / scale;
    cy = (cy - pad.y) / scale;
    w = w / scale;
    h = h / scale;

    float left = cx - (w * 0.5f);
    float top = cy - (h * 0.5f);

    cv::Rect rect(
        static_cast<int>(std::round(left)), static_cast<int>(std::round(top)),
        static_cast<int>(std::round(w)), static_cast<int>(std::round(h)));

    // 원본 이미지 크기(orig_size)를 벗어나는 박스 클리핑
    rect &= cv::Rect(0, 0, orig_size.width, orig_size.height);
    if (rect.area() <= 0)
      continue;

    boxes.push_back(rect);
    scores.push_back(max_score);
    class_ids.push_back(class_id);
  }

  std::vector<int> indices;
  const float nms_threshold =
      0.20f; // [핵심 수정] IOU 임계값을 낮춰 중복된 찌꺼기 박스 완벽 제거
  cv::dnn::NMSBoxes(boxes, scores, static_cast<float>(confidence_threshold),
                    nms_threshold, indices);

  for (int idx : indices) {
    const cv::Rect &rect = boxes[idx];
    DetectedObject obj;
    std::memset(&obj, 0, sizeof(obj));
    obj.id = -1;
    obj.class_id = (idx < static_cast<int>(class_ids.size())
                        ? class_ids[idx]
                        : -1); // [수정] DetectedObject에 class_id 필드 추가로
                               // 컴파일 오류 해소
    const char *class_name = "vehicle";
    switch (obj.class_id) {
    case 2:
      class_name = "car";
      break;
    case 3:
      class_name = "motorcycle";
      break;
    case 5:
      class_name = "bus";
      break;
    case 7:
      class_name = "truck";
      break;
    default:
      class_name = "vehicle";
      break;
    }
    std::strncpy(obj.class_name, class_name, 31);
    obj.x = rect.x;
    obj.y = rect.y;
    obj.w = rect.width;
    obj.h = rect.height;
    obj.center_x = BB_CX(obj);
    obj.center_y = BB_CY(obj);
    obj.area = rect.area();
    obj.confidence = scores[idx];
    obj.shape = SHAPE_RECT;
    obj.circularity = 0.0;
    obj.sector = SECTOR_CENTER;
    obj.raw_distance_m = -1.0;
    obj.distance_m = -1.0;
    obj.prev_distance_m = -1.0;
    obj.closing_speed = 0.0;
    obj.ttc = 9999.0;
    obj.risk = RISK_SAFE;
    obj.danger_frames = 0;
    obj.missing_frames = 0;
    detections.push_back(obj);
  }

  return detections;
}

std::vector<DetectedObject>
DeepObjectDetector::infer(const cv::Mat &bgr, double confidence_threshold,
                          const cv::Size &input_size) {
  std::vector<DetectedObject> result;
  if (!initialized_ || !session_)
    return result;

  float scale;
  cv::Point pad;
  std::vector<float> input_tensor = preprocess(bgr, input_size, scale, pad);
  size_t tensor_size = input_tensor.size();

  std::array<int64_t, 4> tensor_shape = {1, 3, input_size.height,
                                         input_size.width};
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
      mem_info, input_tensor.data(), tensor_size, tensor_shape.data(), 4);

  auto output_tensors = session_->Run(
      Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor_ort, 1,
      output_names_.data(), output_names_.size());
  if (output_tensors.empty())
    return result;

  const Ort::Value &predictions = output_tensors[0];
  const float *out_data = predictions.GetTensorData<float>();
  auto output_info = predictions.GetTensorTypeAndShapeInfo();
  size_t output_size = static_cast<size_t>(output_info.GetElementCount());
  std::vector<float> output_vector(out_data, out_data + output_size);

  return postprocess(output_vector, scale, pad, confidence_threshold,
                     bgr.size());
}
#endif

// =============================================================================
// §5: dt 설정 (외부에서 fps 결정 후 주입)
// =============================================================================
void set_tracker_dt(double dt_seconds) {
  CentroidTracker::dt_ = (dt_seconds > 1e-6) ? dt_seconds : (1.0 / 30.0);
}

// =============================================================================
// §8: 로그 파일 열기 / 닫기
// =============================================================================
void open_logs(const ObjectDetectorConfig &cfg) {
  if (!cfg.enable_logging)
    return;

  const std::filesystem::path obj_log_path(cfg.obj_log_path);
  const std::filesystem::path frame_log_path(cfg.frame_log_path);
  const std::filesystem::path log_dir = obj_log_path.parent_path().empty()
                                            ? frame_log_path.parent_path()
                                            : obj_log_path.parent_path();

  if (!log_dir.empty() && !std::filesystem::exists(log_dir)) {
    std::error_code ec;
    std::filesystem::create_directories(log_dir, ec);
    if (ec) {
      std::fprintf(stderr,
                   "[Error] 로그 디렉터리를 생성할 수 없습니다: %s (%s)\n",
                   log_dir.string().c_str(), ec.message().c_str());
      return;
    }
  }

  g_obj_log = std::fopen(obj_log_path.string().c_str(), "w");
  if (g_obj_log)
    std::fprintf(g_obj_log,
                 "frame,track_id,class_name,x,y,w,h,cx,cy,area,confidence,"
                 "shape,circularity,sector,raw_dist_m,dist_m,closing_speed_ms,"
                 "ttc_s,risk,danger_frames\n");
  else
    std::fprintf(stderr, "[Error] 객체 로그를 열 수 없습니다: %s\n",
                 obj_log_path.string().c_str());

  g_frame_log = std::fopen(frame_log_path.string().c_str(), "w");
  if (g_frame_log)
    std::fprintf(g_frame_log,
                 "frame,n_objects,n_warning,n_stop,min_dist_m,mode\n");
  else
    std::fprintf(stderr, "[Error] 프레임 로그를 열 수 없습니다: %s\n",
                 frame_log_path.string().c_str());
}

void close_logs() {
  if (g_obj_log) {
    std::fclose(g_obj_log);
    g_obj_log = nullptr;
  }
  if (g_frame_log) {
    std::fclose(g_frame_log);
    g_frame_log = nullptr;
  }
}

// =============================================================================
// §8: 프레임 요약 로그 기록
// =============================================================================
void write_frame_log(int frame_idx, int n_objects, int n_warning, int n_stop,
                     double min_dist, const char *mode_str) {
  if (!g_frame_log)
    return;
  std::fprintf(g_frame_log, "%d,%d,%d,%d,%.3f,%s\n", frame_idx, n_objects,
               n_warning, n_stop, min_dist, mode_str ? mode_str : "N/A");
  std::fflush(g_frame_log);
}

// =============================================================================
// §2: 차량 하부 그림자 마스크 생성
//   1) BGR → HSV 변환
//   2) S < shadow_upper[1], V < shadow_upper[2] 조건의 어두운 영역 추출
//   3) 모폴로지 Close 로 구멍 메우기 및 연결
// =============================================================================
// [기술] CLAHE (Contrast Limited Adaptive Histogram Equalization)
//   - 어두운/불균일한 조도 환경에서 로컬 대비를 향상시켜 엣지와 윤곽 검출력을
//   높임
static void apply_clahe_to_bgr(const cv::Mat &src, cv::Mat &dst,
                               const ObjectDetectorConfig &cfg) {
  if (!cfg.enable_clahe || src.empty()) {
    dst = src;
    return;
  }
  cv::Mat lab;
  cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
  std::vector<cv::Mat> lab_channels;
  cv::split(lab, lab_channels);

  cv::Ptr<cv::CLAHE> clahe =
      cv::createCLAHE(cfg.clahe_clip_limit, cv::Size(cfg.clahe_tile_grid_size,
                                                     cfg.clahe_tile_grid_size));
  cv::Mat l_clahe;
  clahe->apply(lab_channels[0], l_clahe);
  l_clahe.copyTo(lab_channels[0]);
  cv::merge(lab_channels, lab);
  cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
}

// [기술] Local Standard Deviation Texture Analysis
//   - 슬라이딩 윈도우 내 국소 분산/표준편차를 계산해 그림자와 구조물 텍스처를
//   구분
static cv::Mat compute_local_stddev(const cv::Mat &gray, int window) {
  cv::Mat f;
  gray.convertTo(f, CV_32F, 1.0 / 255.0);
  cv::Mat mean, mean_sq;
  cv::Mat sq;
  cv::multiply(f, f, sq);
  cv::blur(f, mean, cv::Size(window, window));
  cv::blur(sq, mean_sq, cv::Size(window, window));
  cv::Mat var = mean_sq - mean.mul(mean);
  cv::Mat stddev;
  cv::sqrt(cv::max(var, 0.0f), stddev);
  return stddev;
}

// [기술] HSV Thresholding + Texture Filtering + Morphology
//   - HSV 색상 마스킹, 로컬 텍스처 분석, 모폴로지 Close로 그림자 후보를 정제
static cv::Mat build_shadow_mask(const cv::Mat &hsv, const cv::Mat &gray,
                                 const ObjectDetectorConfig &cfg, int roi_top,
                                 int roi_bottom) {
  cv::Mat shadow_mask;
  cv::inRange(hsv, cv::Scalar(0, 0, 0),
              cv::Scalar(cfg.shadow_hsv_upper[0], cfg.shadow_hsv_upper[1],
                         cfg.shadow_hsv_upper[2]),
              shadow_mask);

  if (cfg.black_car_enable) {
    cv::Mat black_car_mask;
    cv::inRange(hsv, cv::Scalar(0, 0, 0),
                cv::Scalar(cfg.black_car_hsv_upper[0],
                           cfg.black_car_hsv_upper[1],
                           cfg.black_car_hsv_upper[2]),
                black_car_mask);
    cv::bitwise_or(shadow_mask, black_car_mask, shadow_mask);
  }

  if (cfg.enable_shadow_texture && cfg.shadow_texture_window >= 3) {
    cv::Mat stddev = compute_local_stddev(gray, cfg.shadow_texture_window);
    cv::Mat texture_mask;
    cv::threshold(stddev, texture_mask, cfg.shadow_texture_variance_threshold,
                  255, cv::THRESH_BINARY);
    texture_mask.convertTo(texture_mask, CV_8U, 255.0);
    // 어두운 영역이더라도 텍스처가 높으면 그림자에서 제외
    shadow_mask.setTo(0, texture_mask);
  }

  // ROI 외곽 영역 제거 (하늘 및 내 차 후드 제거)
  if (roi_top > 0)
    shadow_mask(cv::Rect(0, 0, shadow_mask.cols, roi_top)).setTo(0);
  if (roi_bottom < shadow_mask.rows)
    shadow_mask(cv::Rect(0, roi_bottom, shadow_mask.cols,
                         shadow_mask.rows - roi_bottom))
        .setTo(0);

  // 모폴로지 Close (가로로 긴 커널을 사용하여 끊어진 그림자 연결)
  int ck = cfg.morph_close_size + (cfg.morph_close_size % 2 == 0 ? 1 : 0);
  cv::Mat k_close =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ck * 2, ck));
  cv::morphologyEx(shadow_mask, shadow_mask, cv::MORPH_CLOSE, k_close);

  return shadow_mask;
}

// =============================================================================
// §3: 형태 분류
//   circularity = 4π·Area / Perimeter²  (원: ~1.0)
//   꼭짓점 수 → 삼각형(3) / 사각형(4~6) / 미분류
// =============================================================================
// [기술] Geometry-based Shape Classification
//   - approxPolyDP와 circularity를 이용해 윤곽선을 삼각형/사각형/원으로 분류
static ShapeType classify_shape(const std::vector<cv::Point> &contour,
                                double area, double perimeter,
                                double epsilon_ratio, double circ_thresh,
                                double &out_circularity) {
  out_circularity =
      (perimeter > 1e-3) ? (4.0 * CV_PI * area) / (perimeter * perimeter) : 0.0;

  if (out_circularity >= circ_thresh)
    return SHAPE_CIRCLE;

  std::vector<cv::Point> approx;
  cv::approxPolyDP(contour, approx, epsilon_ratio * perimeter, true);
  int n = static_cast<int>(approx.size());

  if (n == 3)
    return SHAPE_TRIANGLE;
  if (n >= 4 && n <= 6)
    return SHAPE_RECT;
  return SHAPE_UNKNOWN;
}

// =============================================================================
// §4: Depth Map에서 강건한 거리 계산
//   - BBox 내부 inner_ratio% 영역에서 유효 Depth 수집
//   - percentile 분위수를 대표 거리로 채택 (이상치·배경 억제)
//   - RGB/Depth 해상도 차이 스케일 보정 포함
// =============================================================================
static double compute_robust_depth(const cv::Mat &depth_map,
                                   const DetectedObject &obj,
                                   const ObjectDetectorConfig &cfg, int rgb_w,
                                   int rgb_h) {
  if (depth_map.empty() || !cfg.use_depth)
    return -1.0;
  if (depth_map.cols <= 0 || depth_map.rows <= 0)
    return -1.0;

  // §4: 해상도 차이 스케일 변환 비율
  //   scale_x = depth_cols / rgb_cols
  //   scale_y = depth_rows / rgb_rows
  double sx = (rgb_w > 0) ? (double)depth_map.cols / rgb_w : 1.0;
  double sy = (rgb_h > 0) ? (double)depth_map.rows / rgb_h : 1.0;

  // 내부 inner_ratio% 영역 (BBox 중앙 부분)
  double half_w = obj.w * 0.5 * cfg.depth_inner_ratio;
  double half_h = obj.h * 0.5 * cfg.depth_inner_ratio;

  int dx1 = std::max(0, static_cast<int>((obj.center_x - half_w) * sx));
  int dy1 = std::max(0, static_cast<int>((obj.center_y - half_h) * sy));
  int dx2 = std::min(depth_map.cols - 1,
                     static_cast<int>((obj.center_x + half_w) * sx));
  int dy2 = std::min(depth_map.rows - 1,
                     static_cast<int>((obj.center_y + half_h) * sy));

  if (dx2 <= dx1 || dy2 <= dy1)
    return -1.0;

  // 유효 Depth 값 수집
  std::vector<double> vals;
  vals.reserve(static_cast<size_t>((dx2 - dx1 + 1) * (dy2 - dy1 + 1)));

  for (int r = dy1; r <= dy2; ++r) {
    for (int c = dx1; c <= dx2; ++c) {
      double raw_d = 0.0;
      switch (depth_map.type()) {
      case CV_16U:
        raw_d =
            static_cast<double>(depth_map.at<uint16_t>(r, c)) * cfg.depth_scale;
        break;
      case CV_32F:
        raw_d = static_cast<double>(depth_map.at<float>(r, c));
        break;
      case CV_64F:
        raw_d = depth_map.at<double>(r, c);
        break;
      default:
        break;
      }
      // NaN, 0, 너무 먼 값 제외
      if (raw_d > 0.01 && raw_d < cfg.max_valid_depth_m && std::isfinite(raw_d))
        vals.push_back(raw_d);
    }
  }

  if (static_cast<int>(vals.size()) < cfg.min_depth_samples)
    return -1.0;

  // §4: 낮은 percentile 채택 (30% 분위수 → 가장 가까운 쪽 대표)
  std::sort(vals.begin(), vals.end());
  int idx = static_cast<int>((cfg.depth_percentile / 100.0) *
                             static_cast<double>(vals.size() - 1));
  idx = std::max(0, std::min(idx, static_cast<int>(vals.size()) - 1));
  return vals[static_cast<size_t>(idx)];
}

// =============================================================================
// [신규] §4b: 모노큘러 거리 추정 (Depth 카메라 없을 때의 폴백)
//
//   핀홀 카메라 모델: 이미지 위 물체의 폭(px)은 실제 폭(m)에 비례하고
//   거리(m)에 반비례한다.
//       bbox_width_px = (real_width_m * focal_length_px) / distance_m
//   ->  distance_m    = (real_width_m * focal_length_px) / bbox_width_px
//
//   [focal_length_px 실측 보정 방법]
//   1) 차량(또는 기준 물체)을 정확히 아는 거리 D_known(m)에 세워두고 촬영
//   2) 그 프레임에서 해당 물체의 바운딩박스 폭 W_known(px)을 측정
//   3) focal_length_px = (W_known * D_known) / real_width_m 으로 계산해서
//      object_detector_config.json 의 depth.focal_length_px 에 반영
//   캘리브레이션 없이 기본값(900)을 쓰면 카메라 화각/렌즈에 따라
//   실제 거리와 오차가 날 수 있음 (참고용 근사치로 사용할 것).
// =============================================================================
// [기술] Monocular Distance Estimation
//   - 핀홀 카메라 모델로 바운딩박스 폭과 초점거리 기반 거리를 역산
static double estimate_monocular_distance(const DetectedObject &obj,
                                          const ObjectDetectorConfig &cfg) {
  if (obj.w <= 0)
    return -1.0;
  if (cfg.focal_length_px <= 0.0 || cfg.mono_object_width_m <= 0.0)
    return -1.0;

  double dist = (cfg.mono_object_width_m * cfg.focal_length_px) /
                static_cast<double>(obj.w);

  if (!std::isfinite(dist) || dist <= 0.0 || dist > cfg.max_valid_depth_m)
    return -1.0;
  return dist;
}

// =============================================================================
// §5: 구역 분류 (화면 폭 기준 좌/중/우 3분할)
// =============================================================================
static SectorType assign_sector(double center_x, int frame_width) {
  double ratio = center_x / static_cast<double>(frame_width);
  if (ratio < 0.33)
    return SECTOR_LEFT;
  if (ratio < 0.67)
    return SECTOR_CENTER;
  return SECTOR_RIGHT;
}

static const char *shape_to_string(ShapeType shape) {
  switch (shape) {
  case SHAPE_TRIANGLE:
    return "triangle";
  case SHAPE_RECT:
    return "rect";
  case SHAPE_CIRCLE:
    return "circle";
  default:
    return "unknown";
  }
}

static const char *sector_to_string(SectorType sector) {
  switch (sector) {
  case SECTOR_LEFT:
    return "LEFT";
  case SECTOR_CENTER:
    return "CENTER";
  case SECTOR_RIGHT:
    return "RIGHT";
  default:
    return "UNKNOWN";
  }
}

static const char *risk_to_string(RiskLevel risk) {
  switch (risk) {
  case RISK_STOP:
    return "STOP";
  case RISK_WARNING:
    return "WARNING";
  default:
    return "SAFE";
  }
}

static cv::Scalar risk_color(RiskLevel risk) {
  switch (risk) {
  case RISK_STOP:
    return cv::Scalar(0, 0, 255);
  case RISK_WARNING:
    return cv::Scalar(0, 140, 255);
  default:
    return cv::Scalar(0, 220, 60);
  }
}

static double clamp(double value, double low, double high) {
  return std::max(low, std::min(high, value));
}

// [기술] TTC + Risk Scoring
//   - 거리, 접근 속도, TTC, 구역 정보를 종합해 위험도 점수를 계산
static double compute_risk_score(const DetectedObject &obj,
                                 const ObjectDetectorConfig &cfg) {
  if (obj.distance_m <= 0.0)
    return 0.0;

  double warn_dist = std::max(1.0, cfg.warn_dist_m);
  double stop_dist = std::max(1.0, cfg.stop_dist_m);
  double warn_ttc = std::max(0.1, cfg.warn_ttc_s);

  double dist_score = 1.0 - clamp(obj.distance_m / warn_dist, 0.0, 1.0);
  double ttc_score = 0.0;
  if (obj.ttc > 0.0) {
    ttc_score = 1.0 - clamp(obj.ttc / warn_ttc, 0.0, 1.0);
  }

  double closing_score = clamp(obj.closing_speed / 4.0, 0.0, 1.0);
  double position_score = (obj.sector == SECTOR_CENTER) ? 0.2 : 0.0;
  double urgency_score = (obj.distance_m < stop_dist) ? 0.25 : 0.0;

  double class_weight = 1.0;
  if (std::strcmp(obj.class_name, "vehicle") == 0) {
    class_weight = 1.10;
  } else if (std::strcmp(obj.class_name, "pedestrian") == 0) {
    class_weight = 1.25;
  } else if (std::strcmp(obj.class_name, "cyclist") == 0) {
    class_weight = 1.15;
  }

  double confidence_bonus = clamp(obj.confidence * 0.1, 0.0, 0.15);

  double score = 0.40 * dist_score + 0.25 * ttc_score + 0.15 * closing_score +
                 position_score + urgency_score + confidence_bonus;
  score *= class_weight;

  return clamp(score, 0.0, 1.0);
}

// =============================================================================
// §6: 위험 단계 평가 (Closing Speed + TTC + 구역 복합 판정)
// =============================================================================
// [기술] Risk Level Evaluation
//   - SAFE / WARNING / STOP 상태를 TTC와 거리 기반으로 판정
static RiskLevel evaluate_risk(const DetectedObject &obj,
                               const ObjectDetectorConfig &cfg) {
  if (obj.distance_m <= 0.0)
    return RISK_SAFE; // 거리 무효

  bool in_center = (obj.sector == SECTOR_CENTER);
  bool dist_stop = (obj.distance_m < cfg.stop_dist_m);
  bool dist_warn = (obj.distance_m < cfg.warn_dist_m);
  bool ttc_stop = (obj.ttc > 0.0 && obj.ttc < cfg.stop_ttc_s);
  bool ttc_warn = (obj.ttc > 0.0 && obj.ttc < cfg.warn_ttc_s);

  bool closing = (obj.closing_speed > 0.2);
  bool strong_closure = (obj.closing_speed > 1.0 && obj.distance_m < 8.0);
  
  // [신규] 바운딩 박스 면적 급증률 및 하단 진입 조건 추가
  bool fast_growing = (obj.area_growth_rate > 1.15); // 면적이 15% 이상 커짐
  bool at_bottom_warn = obj.is_at_bottom;

  double score = compute_risk_score(obj, cfg);

  // 중앙 구역에서 정지거리 진입, TTC 위험, 또는 (경고거리 내 + 면적급증) 시 STOP
  if (in_center && (dist_stop || (ttc_stop && closing) || (dist_warn && fast_growing)))
    return RISK_STOP;

  // WARNING 조건에 면적 급증, 하단 진입 등 추가
  if ((dist_warn && closing) || (ttc_warn && closing) || strong_closure || 
      (in_center && fast_growing) || (in_center && at_bottom_warn) || 
      score >= 0.75)
    return RISK_WARNING;

  if (score >= 0.40)
    return RISK_WARNING;

  return RISK_SAFE;
}

// =============================================================================
// §7: LiDAR 구역 융합 – 카메라 거리와 비교해 더 가까운 값 채택
// =============================================================================
// [기술] Camera-LiDAR Sensor Fusion
//   - 카메라 기반 거리와 LiDAR 거리 중 더 신뢰도 높은 값을 선택해 융합
static double fuse_lidar(double camera_dist, SectorType sector,
                         const LidarSector &lidar) {
  if (!lidar.valid)
    return camera_dist;

  double lidar_dist = 9999.0;
  switch (sector) {
  case SECTOR_LEFT:
    lidar_dist = lidar.dist_left;
    break;
  case SECTOR_CENTER:
    lidar_dist = lidar.dist_center;
    break;
  case SECTOR_RIGHT:
    lidar_dist = lidar.dist_right;
    break;
  }

  if (camera_dist <= 0.0)
    return (lidar_dist < 9999.0) ? lidar_dist : -1.0;
  if (lidar_dist >= 9999.0)
    return camera_dist;
  return std::min(camera_dist, lidar_dist);
}

// =============================================================================
// §2+§3: 그림자 및 엣지 밀도 기반 DetectedObject 후보 추출
// =============================================================================
static std::vector<DetectedObject>
extract_candidates(const cv::Mat &bgr, const cv::Mat &depth_map,
                   const ObjectDetectorConfig &cfg, cv::Mat &out_debug_mask) {
  std::vector<DetectedObject> results;

  cv::Mat processed;
  apply_clahe_to_bgr(bgr, processed, cfg);

  cv::Mat hsv, gray;
  cv::cvtColor(processed, hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);

  int roi_top = static_cast<int>(bgr.rows * cfg.obj_roi_top_ratio);
  int roi_bottom = static_cast<int>(bgr.rows * cfg.obj_roi_bottom_ratio);

  // §2: 하부 그림자 마스크 생성
  cv::Mat shadow_mask = build_shadow_mask(hsv, gray, cfg, roi_top, roi_bottom);
  out_debug_mask = shadow_mask.clone(); // 디버그 창에 출력할 마스크

  // 엣지 맵 생성 (차체 후보 검증용)
  cv::Mat edges;
  if (cfg.enable_multichannel_edge_fusion) {
    cv::Mat lab;
    cv::cvtColor(processed, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_ch;
    cv::split(lab, lab_ch);

    cv::Mat edge_l, edge_a, edge_b;
    cv::Canny(lab_ch[0], edge_l, 40, 120);
    cv::Canny(lab_ch[1], edge_a, 50, 130);
    cv::Canny(lab_ch[2], edge_b, 50, 130);

    cv::Mat edge_l_f, edge_a_f, edge_b_f;
    edge_l.convertTo(edge_l_f, CV_32F, 1.0 / 255.0);
    edge_a.convertTo(edge_a_f, CV_32F, 1.0 / 255.0);
    edge_b.convertTo(edge_b_f, CV_32F, 1.0 / 255.0);

    // L 채널에 높은 가중치를 주고 a/b 채널은 낮추어 색상 노이즈 영향 최소화
    cv::Mat fused = 0.7f * edge_l_f + 0.15f * edge_a_f + 0.15f * edge_b_f;
    fused = cv::min(fused, 1.0f);
    fused.convertTo(edges, CV_8U, 255.0);
    cv::threshold(edges, edges, 50, 255, cv::THRESH_BINARY);
  } else {
    cv::Canny(gray, edges, 50, 150);
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(shadow_mask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  for (const auto &cnt : contours) {
    cv::Rect shadow_rect = cv::boundingRect(cnt);

    // 그림자는 옆으로 퍼진 형태여야 함 (폭이 일정 수준 이상, 가로세로 비율
    // > 1.5)
    if (shadow_rect.width < 30 || shadow_rect.height < 5)
      continue;
    if (shadow_rect.width < shadow_rect.height * 1.5)
      continue;

    // [신규] 여러 차량의 그림자가 하나로 뭉친 것으로 의심되는 비정상적으로
    // 큰 후보는 제외. (실제 사례: 다리 위 정체 구간에서 근처 차량 여러 대의
    // 그림자가 모폴로지 Close 과정에서 하나로 이어 붙어, 나무/배경까지 포함한
    // 거대한 박스 하나가 "차량 1대"로 오검출되고 거기서 나온 잘못된 폭이
    // 모노큘러 거리 추정을 왜곡시켜 TTC가 튀는 문제가 있었음)
    if (shadow_rect.width > bgr.cols * cfg.max_object_width_ratio)
      continue;

    // 그림자 바로 윗부분(차체 뒷면)을 ROI로 설정
    int body_w = shadow_rect.width;
    int body_h =
        static_cast<int>(shadow_rect.width * 0.8); // 일반적 차량 후면 비율
    int body_x = shadow_rect.x;
    int body_y = std::max(roi_top, shadow_rect.y - body_h);

    // 차체가 잘렸다면 보정
    body_h = shadow_rect.y - body_y;
    if (body_h <= 10)
      continue;

    cv::Rect body_rect(body_x, body_y, body_w, body_h);

    // §3: 엣지 밀도 검사
    cv::Mat body_edges = edges(body_rect);
    int edge_pixels = cv::countNonZero(body_edges);
    double edge_density = static_cast<double>(edge_pixels) / (body_w * body_h);

    if (edge_density < cfg.edge_density_threshold)
      continue;

    // 검출 확정 시 바운딩 박스를 차체 + 그림자 크기로 병합
    cv::Rect final_rect = body_rect | shadow_rect;

    const double final_area = static_cast<double>(final_rect.area());
    const double frame_area = static_cast<double>(bgr.cols) * bgr.rows;
    const double aspect_ratio = final_rect.height > 0
                                    ? static_cast<double>(final_rect.width) /
                                          static_cast<double>(final_rect.height)
                                    : 0.0;
    if (final_area < cfg.min_object_area_px2)
      continue;
    if (final_area > cfg.max_object_area_ratio * frame_area)
      continue;
    if (aspect_ratio < cfg.min_object_aspect_ratio ||
        aspect_ratio > cfg.max_object_aspect_ratio)
      continue;

    DetectedObject obj;
    std::memset(&obj, 0, sizeof(obj));
    obj.id = -1;
    std::strncpy(obj.class_name, "vehicle", 31);

    obj.x = final_rect.x;
    obj.y = final_rect.y;
    obj.w = final_rect.width;
    obj.h = final_rect.height;
    obj.center_x = BB_CX(obj);
    obj.center_y = BB_CY(obj);
    obj.area = final_rect.area();
    obj.confidence = std::min(1.0, edge_density * 2.0);

    obj.shape = SHAPE_RECT; // 차체는 기본적으로 사각형 처리
    obj.circularity = 0.0;

    obj.sector = assign_sector(obj.center_x, bgr.cols);
    obj.raw_distance_m =
        compute_robust_depth(depth_map, obj, cfg, bgr.cols, bgr.rows);

    // [신규] Depth 카메라 결과가 없거나(use_depth=false) 유효 샘플이 부족하면
    // 모노큘러(폭 기반) 거리 추정으로 폴백. 이게 없으면 use_depth=false인
    // 환경에서는 distance_m이 항상 -1이라 위험도 판정 자체가 절대 발동하지
    // 않음.
    if (obj.raw_distance_m <= 0.0 && cfg.mono_depth_enable) {
      obj.raw_distance_m = estimate_monocular_distance(obj, cfg);
    }
    obj.distance_m = obj.raw_distance_m;

    obj.prev_distance_m = -1.0;
    obj.closing_speed = 0.0;
    obj.ttc = 9999.0;
    obj.risk = RISK_SAFE;
    obj.danger_frames = 0;
    obj.missing_frames = 0;

    results.push_back(obj);
  }

  return results;
}

// =============================================================================
// §5: CentroidTracker 구현
// =============================================================================
CentroidTracker::CentroidTracker(int max_missing, double max_dist_px)
    : next_id_(0), max_missing_(max_missing), max_dist_(max_dist_px) {}

void CentroidTracker::reset() {
  tracks_.clear();
  next_id_ = 0;
}

// =============================================================================
// [기술] Kalman Filter
//   - 등속도(Constant Velocity) 모델로 객체의 위치와 속도를 예측·보정해 추적
//   연속성 확보
// =============================================================================
//   상태벡터 x = [cx, cy, vx, vy]^T
//   전이행렬 F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
//   관측행렬 H = [[1,0,0,0],[0,1,0,0]]  (중심좌표만 관측 가능, 속도는 추정치)
// =============================================================================
cv::KalmanFilter CentroidTracker::makeKalman(double cx, double cy, double dt) {
  cv::KalmanFilter kf(4, 2, 0, CV_32F);
  float dtf = static_cast<float>(dt > 1e-6 ? dt : (1.0 / 30.0));

  kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, dtf, 0, 0, 1, 0, dtf, 0,
                         0, 1, 0, 0, 0, 0, 1);

  cv::setIdentity(
      kf.measurementMatrix); // 2x4에 setIdentity -> [[1,0,0,0],[0,1,0,0]]
  // processNoiseCov: 모델(등속도 가정)이 실제와 얼마나 어긋날 수 있는지.
  // 값을 키우면 예측보다 최신 검출을 더 신뢰, 낮추면 예측(관성)을 더 신뢰.
  cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
  // measurementNoiseCov: 검출 좌표 자체의 노이즈(픽셀 흔들림) 크기.
  cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
  cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1.0));

  kf.statePost = (cv::Mat_<float>(4, 1) << static_cast<float>(cx),
                  static_cast<float>(cy), 0.0f, 0.0f);
  return kf;
}

std::vector<DetectedObject>
CentroidTracker::update(std::vector<DetectedObject> &detections, int frame_height) {
  // ── 검출 없음: Kalman prediction으로 코스팅 + missing 카운트 증가 후 소멸
  // 처리 ──
  if (detections.empty()) {
    std::vector<int> to_erase;
    for (auto &kv : tracks_) {
      // 검출이 아예 없는 프레임에도 등속도 모델로 위치를 계속 예측해둬야
      // 나중에 재검출됐을 때 정지된 옛 위치가 아니라 그럴듯한 위치와 매칭됨
      cv::Mat pred = kv.second.kf.predict();
      kv.second.cx = pred.at<float>(0);
      kv.second.cy = pred.at<float>(1);
      kv.second.missing++;
      if (kv.second.missing > max_missing_)
        to_erase.push_back(kv.first);
    }
    for (int id : to_erase)
      tracks_.erase(id);
    return {};
  }

  // ── 기존 트랙 없음: 전부 신규 등록 ──────────────────────────────────
  if (tracks_.empty()) {
    for (auto &det : detections) {
      TrackState ts;
      ts.cx = det.center_x;
      ts.cy = det.center_y;
      ts.kf = makeKalman(det.center_x, det.center_y, dt_);
      ts.missing = 0;
      ts.dist_ema = det.raw_distance_m;
      ts.speed_ema = 0.0;
      ts.danger_frames = 0;
      ts.prev_dist = -1.0;
      ts.prev_area = det.w * det.h; // [신규]
      std::strncpy(ts.class_name, det.class_name, 31);
      
      det.area_growth_rate = 1.0; // [신규]
      det.is_at_bottom = (frame_height > 0) && ((det.y + det.h) > frame_height * 0.85); // [신규] 하위 15% 진입 여부
      det.id = next_id_++;
      tracks_[det.id] = ts;
    }
    return detections;
  }

  // ── §5: 칼만 예측 + 유클리드 거리 기반 그리디 매칭 ──────────────────
  std::vector<int> track_ids;
  track_ids.reserve(tracks_.size());
  for (auto &kv : tracks_) {
    // [칼만 예측] 등속도 모델로 이번 프레임의 위치를 미리 예측.
    // 이후 매칭은 "마지막 관측 위치"가 아니라 이 "예측 위치" 기준으로 수행돼서
    // 객체가 빠르게 움직여도 같은 트랙에 안정적으로 붙는다.
    cv::Mat pred = kv.second.kf.predict();
    kv.second.cx = pred.at<float>(0);
    kv.second.cy = pred.at<float>(1);
    track_ids.push_back(kv.first);
  }

  int nT = static_cast<int>(track_ids.size());
  int nD = static_cast<int>(detections.size());

  // cost 행렬 [트랙][검출]
  std::vector<std::vector<double>> cost(
      static_cast<size_t>(nT),
      std::vector<double>(static_cast<size_t>(nD), 1e9));

  for (int t = 0; t < nT; ++t) {
    const TrackState &ts = tracks_[track_ids[t]];
    for (int d = 0; d < nD; ++d) {
      double dx = ts.cx - detections[d].center_x;
      double dy = ts.cy - detections[d].center_y;
      cost[t][d] = std::sqrt(dx * dx + dy * dy);
    }
  }

  // 그리디 최소 거리 할당 (소규모에서 헝가리안 알고리즘과 유사)
  std::vector<bool> det_used(static_cast<size_t>(nD), false);
  std::vector<bool> trk_used(static_cast<size_t>(nT), false);
  std::vector<std::pair<int, int>> matches; // (track_idx, det_idx)

  for (int iter = 0; iter < std::min(nT, nD); ++iter) {
    double best = 1e9;
    int bt = -1, bd = -1;
    for (int t = 0; t < nT; ++t) {
      if (trk_used[t])
        continue;
      for (int d = 0; d < nD; ++d) {
        if (det_used[d])
          continue;
        if (cost[t][d] < best) {
          best = cost[t][d];
          bt = t;
          bd = d;
        }
      }
    }
    if (bt < 0 || best > max_dist_)
      break;
    matches.emplace_back(bt, bd);
    trk_used[bt] = det_used[bd] = true;
  }

  // ── 매칭 성공 트랙: 상태 업데이트 ───────────────────────────────────
  std::vector<DetectedObject> output;
  output.reserve(static_cast<size_t>(nD));

  const ObjectDetectorConfig *cfg = cfg_ptr_;

  for (auto &[ti, di] : matches) {
    int id = track_ids[ti];
    TrackState &ts = tracks_[id];
    DetectedObject &det = detections[di];
    det.id = id;

    // [기술] EMA Filter (distance smoothing)
    //   - 거리와 closing speed의 노이즈를 지수 이동 평균으로 완화
    // [신규] 이전 EMA 거리 대비 이번 프레임 원시 거리가 너무 크게 튀면
    // (그림자 뭉침 등으로 인한 순간적 오검출 가능성) EMA에 반영하지 않고
    // 이전 값을 그대로 유지. 실제로 급접근 중이라면 다음 프레임들에서도
    // 계속 가까운 값이 들어오면서 자연스럽게 EMA가 따라감.
    double raw = det.raw_distance_m;
    double jump_ratio = cfg ? cfg->max_distance_jump_ratio : 0.5;
    bool is_outlier_jump =
        (raw > 0.0) && (ts.dist_ema > 0.0) &&
        (std::fabs(raw - ts.dist_ema) / ts.dist_ema > jump_ratio);

    if (raw > 0.0 && !is_outlier_jump) {
      if (ts.dist_ema <= 0.0)
        ts.dist_ema = raw;
      else
        ts.dist_ema =
            cfg_ema_alpha_ * raw + (1.0 - cfg_ema_alpha_) * ts.dist_ema;
      det.distance_m = ts.dist_ema;
    } else {
      det.distance_m = (ts.dist_ema > 0.0) ? ts.dist_ema : -1.0;
    }

    // §6: Closing Speed = (prev_dist - curr_dist) / dt  (양수 = 접근 중)
    // [신규] closing_speed도 한 번 더 EMA로 스무딩해서 TTC 스파이크 완화.
    // 거리(det.distance_m)는 이미 EMA된 값이지만, 그 프레임간 미분은
    // 잔여 노이즈가 dt로 나뉘며 증폭되어 TTC가 순간적으로 튀는 경우가 있음.
    if (ts.prev_dist > 0.0 && det.distance_m > 0.0 && dt_ > 1e-6) {
      double raw_speed = (ts.prev_dist - det.distance_m) / dt_;
      double speed_alpha = cfg ? cfg->speed_ema_alpha : 0.3;
      ts.speed_ema =
          speed_alpha * raw_speed + (1.0 - speed_alpha) * ts.speed_ema;
      det.closing_speed = ts.speed_ema;
      det.prev_distance_m = ts.prev_dist;
      // §6: TTC = curr_dist / closing_speed
      det.ttc = (det.closing_speed > 0.01)
                    ? (det.distance_m / det.closing_speed)
                    : 9999.0;
    } else {
      det.closing_speed = 0.0;
      det.prev_distance_m = ts.prev_dist;
      det.ttc = 9999.0;
    }

    // [기술] Kalman Correction
    //   - 예측값을 실제 검출 중심점으로 보정해 다음 프레임 예측을 안정화
    // 이 과정에서 자동으로 갱신됨 (다음 프레임 예측에 반영)
    cv::Mat measurement =
        (cv::Mat_<float>(2, 1) << static_cast<float>(det.center_x),
         static_cast<float>(det.center_y));
    cv::Mat corrected = ts.kf.correct(measurement);
    ts.cx = corrected.at<float>(0);
    ts.cy = corrected.at<float>(1);
    ts.prev_dist = det.distance_m;
    ts.missing = 0;

    // [신규] 면적 급증률 및 하단 진입 여부 계산 (FCW 보조 지표)
    double curr_area = det.w * det.h;
    det.area_growth_rate = (ts.prev_area > 0) ? (curr_area / ts.prev_area) : 1.0;
    ts.prev_area = curr_area;
    det.is_at_bottom = (frame_height > 0) && ((det.y + det.h) > frame_height * 0.85);

    // §8: 위험 단계 + 연속성 검증 (danger_frames 누적)
    RiskLevel raw_risk = (cfg) ? evaluate_risk(det, *cfg) : RISK_SAFE;

    if (raw_risk == RISK_STOP)
      ts.danger_frames += 2;
    else if (raw_risk == RISK_WARNING)
      ts.danger_frames += 1; // [수정] WARNING도 위험 누적 카운터 증가 (이전에는
                             // -1로 역전되어 WARNING 판정 불가)
    else
      ts.danger_frames = std::max(0, ts.danger_frames - 2);

    // [기술] Hysteresis / Debouncing
    //   - 위험 상태 전환 시 연속 프레임을 누적해 순간 노이즈에 의한 채터링을
    //   방지
    int min_df = cfg ? cfg->min_danger_frames : 3;
    if (raw_risk == RISK_STOP && ts.danger_frames >= min_df)
      det.risk = RISK_STOP;
    else if (raw_risk == RISK_WARNING && ts.danger_frames >= 2)
      det.risk = RISK_WARNING;
    else if (cfg && raw_risk == RISK_WARNING &&
             det.distance_m <
                 cfg->warn_dist_m * 0.7) // [수정] cfg nullptr 역참조 방지
      det.risk = RISK_WARNING;
    else
      det.risk = RISK_SAFE;

    det.danger_frames = ts.danger_frames;
    det.missing_frames = 0;
    output.push_back(det);
  }

  // ── 비매칭 기존 트랙: Kalman prediction으로 코스팅하며 missing 카운트 증가
  // ──
  std::vector<int> to_erase;
  for (int t = 0; t < nT; ++t) {
    if (trk_used[t])
      continue;
    // correct()를 호출하지 않으므로 위 predict() 루프에서 구한 예측 위치가
    // 그대로 ts.cx/cy에 유지된 상태 -> 몇 프레임 동안 가려짐을 자연스럽게 버팀
    tracks_[track_ids[t]].missing++;
    if (tracks_[track_ids[t]].missing > max_missing_)
      to_erase.push_back(track_ids[t]);
  }
  for (int id : to_erase)
    tracks_.erase(id);

  // ── 비매칭 신규 검출: 새 트랙 등록 ──────────────────────────────────
  for (int d = 0; d < nD; ++d) {
    if (det_used[d])
      continue;
    TrackState ts;
    ts.cx = detections[d].center_x;
    ts.cy = detections[d].center_y;
    ts.kf = makeKalman(detections[d].center_x, detections[d].center_y, dt_);
    ts.missing = 0;
    ts.dist_ema = detections[d].raw_distance_m;
    ts.speed_ema = 0.0;
    ts.danger_frames = 0;
    ts.prev_dist = -1.0;
    ts.prev_area = detections[d].w * detections[d].h; // [신규]
    std::strncpy(ts.class_name, detections[d].class_name, 31);

    detections[d].area_growth_rate = 1.0; // [신규]
    detections[d].is_at_bottom = (frame_height > 0) && ((detections[d].y + detections[d].h) > frame_height * 0.85); // [신규]

    detections[d].id = next_id_++;
    detections[d].risk = RISK_SAFE;
    tracks_[detections[d].id] = ts;
    output.push_back(detections[d]);
  }

  return output;
}

// =============================================================================
// §1~§8: 메인 검출 파이프라인
// =============================================================================
std::vector<DetectedObject>
detect_objects(const cv::Mat &rgb_frame, const cv::Mat &depth_map,
               const ObjectDetectorConfig &cfg, CentroidTracker &tracker,
               const LidarSector &lidar, DeepObjectDetector *deep_detector,
               cv::Mat &out_debug_mask) {
  // 트래커에 설정 주입
  CentroidTracker::cfg_ptr_ = &cfg;
  CentroidTracker::cfg_ema_alpha_ = cfg.ema_alpha;

  std::vector<DetectedObject> candidates;
#ifdef USE_ONNX_RUNTIME
  // [기술] ONNX Runtime 기반 딥러닝 추론 경로
  //   - 딥 모델이 사용 가능하면 하이브리드 파이프라인으로 객체를 추론
  if (deep_detector && deep_detector->is_initialized() &&
      cfg.enable_deep_inference) {
    candidates = deep_detector->infer(rgb_frame, cfg.deep_confidence_threshold);
    out_debug_mask = cv::Mat::zeros(rgb_frame.size(), CV_8UC1);
  } else {
    candidates = extract_candidates(rgb_frame, depth_map, cfg, out_debug_mask);
  }
#else
  candidates = extract_candidates(rgb_frame, depth_map, cfg, out_debug_mask);
#endif

  // §5+§6+§7(EMA): 추적기 업데이트
  std::vector<DetectedObject> tracked = tracker.update(candidates, rgb_frame.rows);

  // §7: LiDAR 구역 융합 (카메라 거리 + LiDAR 구역 최소 거리)
  for (auto &obj : tracked) {
    if (lidar.valid) {
      double fused = fuse_lidar(obj.distance_m, obj.sector, lidar);
      if (fused > 0.0)
        obj.distance_m = fused;
    }
  }

  // §8: 객체 단위 상세 로그 기록
  static int s_frame_idx = 0;
  if (g_obj_log) {
    for (const auto &obj : tracked) {
      const char *shape_str = shape_to_string(obj.shape);
      const char *sect_str = sector_to_string(obj.sector);
      const char *risk_str = risk_to_string(obj.risk);

      std::fprintf(g_obj_log,
                   "%d,%d,%s,%d,%d,%d,%d,%.1f,%.1f,%.1f,%.3f,"
                   "%s,%.3f,%s,%.3f,%.3f,%.3f,%.3f,%s,%d\n",
                   s_frame_idx, obj.id, obj.class_name, obj.x, obj.y, obj.w,
                   obj.h, obj.center_x, obj.center_y, obj.area, obj.confidence,
                   shape_str, obj.circularity, sect_str, obj.raw_distance_m,
                   obj.distance_m, obj.closing_speed, obj.ttc, risk_str,
                   obj.danger_frames);
    }
    std::fflush(g_obj_log);
  }

  ++s_frame_idx;
  return tracked;
}

// =============================================================================
// 시각화: 검출 결과 오버레이 (위험도 색상 + 거리 + TTC + 형태 + 구역)
// =============================================================================
void draw_detections(cv::Mat &display,
                     const std::vector<DetectedObject> &objects) {
  for (const auto &obj : objects) {
    // 위험 단계별 색상: STOP=빨강, WARNING=주황, SAFE=녹색
    cv::Scalar color = risk_color(obj.risk);
    int thickness = (obj.risk == RISK_STOP) ? 3 : 2;

    // §1: BB_X2 / BB_Y2 매크로로 우하단 좌표 계산
    cv::rectangle(display, cv::Rect(obj.x, obj.y, obj.w, obj.h), color,
                  thickness);

    // 중심점 마커
    cv::circle(display,
               cv::Point(static_cast<int>(obj.center_x),
                         static_cast<int>(obj.center_y)),
               4, color, -1);

    // 상단 레이블 (Track ID, 형태, 구역, 위험 단계)
    const char *shape_str = (obj.shape == SHAPE_TRIANGLE) ? "TRI"
                            : (obj.shape == SHAPE_RECT)   ? "RECT"
                            : (obj.shape == SHAPE_CIRCLE) ? "CIR"
                                                          : "UNK";
    const char *sect_str = (obj.sector == SECTOR_LEFT)     ? "L"
                           : (obj.sector == SECTOR_CENTER) ? "C"
                                                           : "R";
    const char *risk_tag = (obj.risk == RISK_STOP)      ? " !!STOP!!"
                           : (obj.risk == RISK_WARNING) ? " !WARN!"
                                                        : "";

    char label[128];
    std::snprintf(label, sizeof(label), "ID:%d %s [%s]%s", obj.id, shape_str,
                  sect_str, risk_tag);
    cv::putText(display, label, cv::Point(obj.x, std::max(obj.y - 8, 10)),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv::LINE_AA);

    // 하단 레이블 (거리 + TTC + 접근 속도)
    if (obj.distance_m > 0.0) {
      char dist_label[96];
      if (obj.ttc < 9000.0) {
        std::snprintf(dist_label, sizeof(dist_label),
                      "%.2fm  TTC:%.1fs  Closing:%.2fm/s", obj.distance_m,
                      obj.ttc, obj.closing_speed);
      } else {
        std::snprintf(dist_label, sizeof(dist_label), "%.2fm  Closing:%.2fm/s",
                      obj.distance_m, obj.closing_speed);
      }
      cv::putText(display, dist_label, cv::Point(obj.x, BB_Y2(obj) + 16),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 100), 1,
                  cv::LINE_AA);
    }
  }
}
