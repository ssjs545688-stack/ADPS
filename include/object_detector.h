#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <algorithm>
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

// =============================================================================
// object_detector.h
// 목적: 색상 기반 객체 검출, 추적, 충돌 위험도 평가 파이프라인 헤더
// 요구사항: §1(구조체) §2(HSV+모폴로지) §3(윤곽선) §4(Depth)
//           §5(구역+추적) §6(TTC) §7(EMA+LiDAR) §8(로깅)
// =============================================================================

// -----------------------------------------------------------------------------
// §1: 바운딩 박스 편의 매크로
//   BB_X2 : 우하단 x = x + width
//   BB_Y2 : 우하단 y = y + height
//   BB_CX : 중심 x   = x + width  / 2.0
//   BB_CY : 중심 y   = y + height / 2.0
// -----------------------------------------------------------------------------
#define BB_X2(obj) ((obj).x + (obj).w)
#define BB_Y2(obj) ((obj).y + (obj).h)
#define BB_CX(obj) ((obj).x + (obj).w * 0.5)
#define BB_CY(obj) ((obj).y + (obj).h * 0.5)

// -----------------------------------------------------------------------------
// §1 / §3 / §5 / §6: 열거형
// -----------------------------------------------------------------------------
typedef enum {
  SHAPE_UNKNOWN = 0,
  SHAPE_TRIANGLE = 3,
  SHAPE_RECT = 4,
  SHAPE_CIRCLE = 99
} ShapeType;

typedef enum {
  SECTOR_LEFT = 0,
  SECTOR_CENTER = 1,
  SECTOR_RIGHT = 2
} SectorType;

typedef enum { RISK_SAFE = 0, RISK_WARNING = 1, RISK_STOP = 2 } RiskLevel;

// -----------------------------------------------------------------------------
// §1: 핵심 객체 구조체
// -----------------------------------------------------------------------------
typedef struct {
  int id;
  int class_id; // [수정] 딥러닝 모델 클래스 인덱스 (색상 기반 검출 시 -1)
  char class_name[32];

  // 2D 바운딩 박스 (§1)
  //   우하단: x2 = BB_X2(obj), y2 = BB_Y2(obj)
  //   중심  : cx = BB_CX(obj), cy = BB_CY(obj)
  int x, y, w, h;
  double center_x, center_y;
  double area;
  double confidence;

  // §4: 거리 (raw = 원시값, distance_m = EMA 필터 후)
  double raw_distance_m;
  double distance_m;

  // §6: 충돌 평가
  double prev_distance_m;
  double closing_speed; // 양수 = 접근 중 (m/s)
  double ttc;           // Time-To-Collision (s)

  // §3: 형태
  ShapeType shape;
  double circularity; // 4π·Area / Perimeter²

  // §5: 구역/추적
  SectorType sector;
  int missing_frames;

  // §6/§8: 위험도 및 연속성 카운터
  RiskLevel risk;
  int danger_frames; // 연속 위험 프레임 수
  
  // [신규] FCW 개선 보조 지표
  double area_growth_rate; // 픽셀 면적 급증률 (1.0 = 유지, >1.0 = 커짐)
  bool is_at_bottom;       // 화면 하단 진입 여부
} DetectedObject;

// -----------------------------------------------------------------------------
// §7: LiDAR 구역별 데이터
// -----------------------------------------------------------------------------
typedef struct {
  double dist_left;
  double dist_center;
  double dist_right;
  bool valid;
} LidarSector;

// -----------------------------------------------------------------------------
// §2~§8: 시스템 설정 구조체
// -----------------------------------------------------------------------------
typedef struct {
  // 기존 색상(보조용)
  int red_lower1[3];
  int red_upper1[3];
  int red_lower2[3];
  int red_upper2[3];
  int yellow_lower[3];
  int yellow_upper[3];

  // 새 파라미터: 그림자/엣지 및 ROI
  int shadow_hsv_upper[3];       // 그림자 검출용 상한 (H, S, V)
  double edge_density_threshold; // 바운딩 박스 내 엣지 밀도 임계값
  double obj_roi_top_ratio;      // 객체 검출 ROI 상단 (예: 0.3)
  double obj_roi_bottom_ratio;   // 객체 검출 ROI 하단 (예: 0.8)

  // 검은 차량 대응: 어두운 차체를 추가로 후보에 포함
  bool black_car_enable;
  int black_car_hsv_upper[3];
  // [신규] 그림자 후보의 폭이 프레임 폭 대비 이 비율을 넘으면 "여러 차량이
  // 뭉친 오검출"로 간주하고 제외 (기본 0.5 = 프레임 폭의 50%)
  double max_object_width_ratio;
  // [신규] 객체 크기/비율 필터링
  double min_object_area_px2;
  double max_object_area_ratio;
  double min_object_aspect_ratio;
  double max_object_aspect_ratio;

  // [신규] 딥러닝 기반 객체 검출 옵션
  bool enable_deep_inference;
  char deep_model_path[256];
  double deep_confidence_threshold;

  // [신규] CLAHE 기반 대비 보정
  bool enable_clahe;
  int clahe_clip_limit;
  int clahe_tile_grid_size;

  // [신규] 그림자 텍스처 기반 보호
  bool enable_shadow_texture;
  int shadow_texture_window;
  double shadow_texture_variance_threshold;

  // [신규] 다중 채널 엣지 퓨전
  bool enable_multichannel_edge_fusion;

  // §2: 모폴로지 커널 크기 (홀수 권장)
  int morph_open_size;
  int morph_close_size;

  // §3: 윤곽선 필터
  double min_contour_area;
  double max_contour_area;
  double poly_epsilon_ratio; // approxPolyDP epsilon = ratio * arcLength
  double circle_threshold;   // 원형도 임계값 (0.75 이상 → 원)

  // §4: Depth 처리
  bool use_depth;
  double depth_scale;       // raw depth 단위 변환 (예: mm→m: 0.001)
  double depth_inner_ratio; // 내부 샘플 영역 비율 (0.6 = 안쪽 60%)
  double depth_percentile;  // 대표 거리로 사용할 분위수 (예: 30.0)
  double max_valid_depth_m; // 유효 Depth 최대값 (m)
  int min_depth_samples;    // 유효 Depth 최소 샘플 수

  // [신규] §4b: Depth 카메라 없을 때(use_depth=false) 쓰는 모노큘러 거리 추정
  //   원리(Width-based Distance Estimation, 핀홀 카메라 모델):
  //     distance_m = (실제_폭_m * 초점거리_px) / 바운딩박스_폭_px
  //   focal_length_px는 카메라 캘리브레이션 값이 없으면 부정확할 수 있어
  //   실측으로 보정 필요 (아래 estimate_monocular_distance 주석 참고)
  bool mono_depth_enable; // true면 use_depth=false거나 depth 샘플 부족 시 폴백
                          // 사용
  double focal_length_px; // 카메라 초점거리 (픽셀 단위)
  double mono_object_width_m; // 검출 대상의 가정 실제 폭 (m). 예: 승용차 후면
                              // 폭 ~1.8m

  // §5: 추적기
  double max_match_dist_px; // 트랙 매칭 허용 최대 유클리드 거리 (px)
  int max_missing_frames;   // 소멸 전 허용 최대 비검출 프레임 수

  // §6: 위험도 임계값
  double warn_dist_m;    // 경고 거리 (m)
  double stop_dist_m;    // 정지 거리 (m)
  double warn_ttc_s;     // 경고 TTC (s)
  double stop_ttc_s;     // 정지 TTC (s)
  int min_danger_frames; // 연속 위험 프레임 최소값 (오검출 방지)

  // [신규] 거리값 급변(오검출/노이즈) 방지
  //   이전 EMA 거리 대비 이번 프레임 원시 거리가 이 비율 이상 튀면
  //   (예: 0.5 = 50% 이상 변화) 노이즈로 간주하고 EMA 갱신에서 제외
  double max_distance_jump_ratio;
  // [신규] closing_speed(그리고 그로부터 계산되는 TTC)를 한 번 더 EMA로
  // 스무딩. 거리값 자체는 이미 EMA된 값이지만, 그 미분(속도)은 residual
  // 노이즈가 dt로 나뉘며 증폭돼 TTC가 순간적으로 튀는 경우가 있어서 추가.
  double speed_ema_alpha;

  // §7: EMA 계수
  double ema_alpha; // 0에 가까울수록 부드러움 (0~1)

  // §8: 로깅
  bool enable_logging;
  char obj_log_path[256];   // 객체 단위 상세 CSV 경로
  char frame_log_path[256]; // 프레임 단위 요약 CSV 경로
} ObjectDetectorConfig;

#ifdef USE_ONNX_RUNTIME
// -----------------------------------------------------------------------------
// §Deep: ONNXRuntime 기반 딥러닝 추론 클래스
//   - 생성자에서 세션을 한 번만 초기화
//   - 이미지 전처리, 텐서 생성, Run(), NMS 후처리
// -----------------------------------------------------------------------------
class DeepObjectDetector {
public:
  DeepObjectDetector(const std::string &model_path, int num_threads = 1,
                     bool use_cuda = false);
  bool is_initialized() const noexcept { return initialized_; }

  std::vector<DetectedObject>
  infer(const cv::Mat &bgr, double confidence_threshold,
        const cv::Size &input_size = cv::Size(640, 640));

private:
  std::vector<float> preprocess(const cv::Mat &bgr, const cv::Size &target_size,
                                float &scale, cv::Point &pad) const;
  std::vector<DetectedObject> postprocess(const std::vector<float> &output,
                                          float scale, const cv::Point &pad,
                                          double confidence_threshold,
                                          const cv::Size &orig_size) const;
  void initialize_model(const std::string &model_path, bool use_cuda,
                        int num_threads);

  std::unique_ptr<Ort::Env> env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_storage_;
  std::vector<std::string> output_names_storage_;
  std::vector<const char *> input_names_;
  std::vector<const char *> output_names_;
  std::vector<int64_t> input_shape_;
  cv::Size input_size_;
  bool initialized_;
};
#else
struct DeepObjectDetector {
  bool is_initialized() const noexcept { return false; }
};
#endif

// -----------------------------------------------------------------------------
// §5: CentroidTracker
//   - 프레임 간 유클리드 거리 비교로 동일 객체에 일관된 track_id 부여
//   - max_missing_frames 초과 시 트랙 소멸
// -----------------------------------------------------------------------------
class CentroidTracker {
public:
  explicit CentroidTracker(int max_missing, double max_dist_px);

  std::vector<DetectedObject> update(std::vector<DetectedObject> &detections, int frame_height = -1);
  void reset();

  // detect_objects()에서 주입하는 공유 설정 (정적 멤버)
  static const ObjectDetectorConfig *cfg_ptr_;
  static double dt_; // 프레임 간격 (s)
  static double cfg_ema_alpha_;

private:
  int next_id_;
  int max_missing_;
  double max_dist_;

  // [칼만 필터 도입] 상태벡터 [cx, cy, vx, vy] (등속도 모델), 관측값 [cx, cy]
  // - predict(): 이전 속도로 현재 위치를 미리 예측 -> 매칭 기준점으로 사용
  // - correct(): 실제 검출된 중심점으로 예측을 보정
  // - 검출이 잠깐 끊겨도(occlusion) predict()만으로 몇 프레임 코스팅 가능
  static cv::KalmanFilter makeKalman(double cx, double cy, double dt);

  struct TrackState {
    double cx,
        cy; // 현재 신뢰 위치 (매칭/그리기에 사용, predict/correct 결과 반영)
    cv::KalmanFilter kf;
    int missing;
    double dist_ema;   // §7: EMA 필터 누적값
    double speed_ema;  // [신규] closing_speed EMA (TTC 노이즈 완화)
    int danger_frames; // §8: 연속 위험 프레임 카운터
    double prev_dist;  // §6: 이전 프레임 거리 (TTC 계산용)
    double prev_area;  // [신규] 이전 프레임 픽셀 면적 (급증률 계산용)
    char class_name[32];
  };
  std::map<int, TrackState> tracks_;
};

// -----------------------------------------------------------------------------
// 공개 함수 원형
// -----------------------------------------------------------------------------

// §1: 기본값으로 설정 초기화
void init_detector_config(ObjectDetectorConfig *cfg);

// JSON 파일에서 설정 로드 (실패 시 기본값 유지, false 반환)
bool load_detector_config_json(const char *filepath, ObjectDetectorConfig *cfg);

// §5+§6: 프레임 간격(dt) 설정 (TTC 계산 정확도에 영향)
void set_tracker_dt(double dt_seconds);

// §1~§8: 메인 검출 파이프라인
std::vector<DetectedObject>
detect_objects(const cv::Mat &rgb_frame, const cv::Mat &depth_map,
               const ObjectDetectorConfig &cfg, CentroidTracker &tracker,
               const LidarSector &lidar, DeepObjectDetector *deep_detector,
               cv::Mat &out_debug_mask);

// §8: 로그 파일 열기 / 닫기
void open_logs(const ObjectDetectorConfig &cfg);
void close_logs();

// §8: 프레임 요약 로그 기록
void write_frame_log(int frame_idx, int n_objects, int n_warning, int n_stop,
                     double min_dist, const char *mode_str);

// 시각화: 검출 결과 오버레이
void draw_detections(cv::Mat &display,
                     const std::vector<DetectedObject> &objects);

#endif
