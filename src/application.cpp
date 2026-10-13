#include "application.h"

#include "lane_detector.h"
#include "object_detector.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>


static const double BRIGHTNESS_NIGHT_ENTER =
    25.0; // 30 미만 야간 진입 (안전 마진 포함 25)
static const double BRIGHTNESS_NIGHT_EXIT = 30.0;  // 30 이상이면 흐림/비로 복귀
static const double BRIGHTNESS_RAINY_ENTER = 70.0; // 30~70 사이는 악천후/비
static const double BRIGHTNESS_SUNNY_ENTER =
    70.0; // 70 이상이면 무조건 맑은 주간 (그림자 강건성 확보)
static const double SATURATION_OVERCAST_MAX = 35.0; // 고조도 저채도 안개
static const double SATURATION_SUNNY_MIN = 25.0; // SUNNY 판정 채도 하한선 완화

static const int STABLE_FRAME_THRESHOLD =
    10; // 최소 10프레임 이상 연속 충족 시 전환 (채터링 완벽 차단)

struct ProgramOptions {
  std::filesystem::path video_path = "good/1251.mp4";
  std::filesystem::path output_path;
  std::filesystem::path sunny_config_path = "configs/lane_config_sunny.json";
  std::filesystem::path overcast_config_path =
     "configs/lane_config_overcast.json";
  std::filesystem::path rainy_config_path = "configs/lane_config_rainy.json";
  std::filesystem::path night_config_path = "configs/lane_config_night.json";
};

struct LaneConfigSet {
  LaneConfig sunny;
  LaneConfig overcast;
  LaneConfig rainy;
  LaneConfig night;
};

struct ObjectDetectorConfigSet {
  ObjectDetectorConfig sunny;
  ObjectDetectorConfig overcast;
  ObjectDetectorConfig rainy;
  ObjectDetectorConfig night;
};



struct AutoLightState {
  LightMode current = MODE_SUNNY;
  LightMode pending = MODE_SUNNY;
  int pending_frames = 0;
};

ProgramOptions parse_program_options(int argc, char **argv) {
  ProgramOptions options;
  if (argc > 1)
    options.video_path = argv[1];
  if (argc > 2)
    options.output_path = argv[2];
  if (argc > 3)
    options.sunny_config_path = argv[3];
  if (argc > 4)
    options.overcast_config_path = argv[4];
  if (argc > 5)
    options.rainy_config_path = argv[5];
  if (argc > 6)
    options.night_config_path = argv[6];
  return options;
}

bool load_lane_config_set(const ProgramOptions &options,
                         LaneConfigSet &configs) {
  const struct {
    const char *name;
    const std::filesystem::path *path;
    LaneConfig *config;
  } entries[] = {
     {"sunny", &options.sunny_config_path, &configs.sunny},
     {"overcast", &options.overcast_config_path, &configs.overcast},
     {"rainy", &options.rainy_config_path, &configs.rainy},
     {"night", &options.night_config_path, &configs.night},
  };

  for (const auto &entry : entries) {
    const std::string path = entry.path->string();
    if (!load_config(path.c_str(), entry.config)) {
     std::printf("[Error] %s 설정 파일을 열 수 없습니다: %s\n", entry.name,
                 path.c_str());
     return false;
    }
    std::printf("[Info] %s 설정 로드: %s\n", entry.name, path.c_str());
  }
  return true;
}

bool load_object_detector_config_set(ObjectDetectorConfigSet &configs) {
  const char *default_path = "configs/object_detector_config.json";
  const struct {
    const char *path;
    ObjectDetectorConfig *config;
  } entries[] = {
     {"configs/object_detector_config_sunny.json", &configs.sunny},
     {"configs/object_detector_config_overcast.json", &configs.overcast},
     {"configs/object_detector_config_rainy.json", &configs.rainy},
     {"configs/object_detector_config_night.json", &configs.night},
  };

  for (const auto &entry : entries) {
    if (!load_detector_config_json(entry.path, entry.config) &&
       !load_detector_config_json(default_path, entry.config)) {
     std::printf("[Error] 객체 검출기 설정을 불러오지 못했습니다: %s\n",
                 entry.path);
     return false;
    }
  }
  return true;
}

bool open_input_video(const std::filesystem::path &video_path,
                     cv::VideoCapture &capture, int &width, int &height,
                     double &fps) {
  capture.open(video_path.string());
  if (!capture.isOpened()) {
    std::printf("[Error] 비디오 파일을 열 수 없습니다: %s\n",
               video_path.string().c_str());
    return false;
  }

  width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  fps = capture.get(cv::CAP_PROP_FPS);
  if (fps <= 0)
    fps = 10.0;
  return true;
}

std::filesystem::path resolve_output_path(
    const std::filesystem::path &requested_path) {
  if (!requested_path.empty())
    return std::filesystem::absolute(requested_path);

  std::filesystem::path output_path =
     std::filesystem::current_path() / "output_result.avi";
  const std::filesystem::path build_file =
     std::filesystem::current_path() / ".." / "CMakeLists.txt";
  if (std::filesystem::exists(build_file))
    output_path =
       std::filesystem::current_path().parent_path() / "output_result.avi";
  return std::filesystem::absolute(output_path);
}

bool open_output_video(const std::filesystem::path &output_path, int width,
                      int height, double fps, cv::VideoWriter &writer) {
  const std::filesystem::path output_dir = output_path.parent_path();
  if (!output_dir.empty() && !std::filesystem::exists(output_dir)) {
    std::error_code error;
    std::filesystem::create_directories(output_dir, error);
    if (error) {
     std::printf("[Error] 출력 디렉터리를 생성할 수 없습니다: %s (%s)\n",
                 output_dir.string().c_str(), error.message().c_str());
     return false;
    }
  }

  const std::vector<int> backends = {
#ifdef CV_CAP_FFMPEG
     cv::CAP_FFMPEG,
#endif
#ifdef CV_CAP_MSMF
     cv::CAP_MSMF,
#endif
#ifdef CV_CAP_DSHOW
     cv::CAP_DSHOW,
#endif
     cv::CAP_ANY,
  };
  const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  for (int backend : backends) {
    if (backend == cv::CAP_ANY)
     writer.open(output_path.string(), fourcc, fps,
                 cv::Size(width * 2, height), true);
    else
     writer.open(output_path.string(), backend, fourcc, fps,
                 cv::Size(width * 2, height), true);

    if (writer.isOpened()) {
     std::printf("[Info] VideoWriter opened with backend %d\n", backend);
     return true;
    }
    std::printf("[Warn] VideoWriter backend %d 실패: %s\n", backend,
               output_path.string().c_str());
  }

  std::printf("[Error] 비디오 writer를 초기화하지 못했습니다: %s\n",
             output_path.string().c_str());
  return false;
}

LightingMetrics measure_lighting(const cv::Mat &frame) {
  const int width = frame.cols;
  const int height = frame.rows;
  const cv::Rect sky_roi(0, 0, width, height / 2);
  const cv::Rect road_roi(0, height / 2, width, height / 2);

  cv::Mat sky_gray, sky_hsv, road_gray, road_hsv;
  cv::cvtColor(frame(sky_roi), sky_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(frame(sky_roi), sky_hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(frame(road_roi), road_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(frame(road_roi), road_hsv, cv::COLOR_BGR2HSV);

  std::vector<cv::Mat> sky_channels, road_channels;
  cv::split(sky_hsv, sky_channels);
  cv::split(road_hsv, road_channels);
  const double brightness =
     0.4 * cv::mean(sky_gray)[0] + 0.6 * cv::mean(road_gray)[0];
  const double saturation =
     0.4 * cv::mean(sky_channels[1])[0] +
     0.6 * cv::mean(road_channels[1])[0];
  return {brightness, saturation};
}

LightMode classify_lighting(const LightingMetrics &metrics,
                           LightMode current_mode) {
  if (metrics.brightness < BRIGHTNESS_NIGHT_ENTER)
    return MODE_NIGHT;
  if (metrics.brightness < BRIGHTNESS_RAINY_ENTER)
    return MODE_RAINY;
  if (metrics.saturation <= SATURATION_OVERCAST_MAX)
    return MODE_OVERCAST;
  if (metrics.saturation >= SATURATION_SUNNY_MIN &&
     metrics.brightness >= BRIGHTNESS_SUNNY_ENTER)
    return MODE_SUNNY;
  return current_mode;
}

bool update_auto_light_mode(const LightingMetrics &metrics,
                           AutoLightState &state,
                           const LaneConfigSet &lane_configs,
                           const ObjectDetectorConfigSet &object_configs,
                           LaneConfig &current_lane_config,
                           const ObjectDetectorConfig *&current_object_config) {
  const LightMode target_mode = classify_lighting(metrics, state.current);
  if (target_mode == state.current) {
    state.pending_frames = 0;
    return false;
  }

  if (target_mode == state.pending) {
    ++state.pending_frames;
  } else {
    state.pending = target_mode;
    state.pending_frames = 1;
  }
  if (state.pending_frames < STABLE_FRAME_THRESHOLD)
    return false;

  state.current = target_mode;
  state.pending_frames = 0;
  close_logs();

  const char *mode_name = "";
  switch (state.current) {
  case MODE_SUNNY:
    current_lane_config = lane_configs.sunny;
    current_object_config = &object_configs.sunny;
    mode_name = "SUNNY";
    break;
  case MODE_OVERCAST:
    current_lane_config = lane_configs.overcast;
    current_object_config = &object_configs.overcast;
    mode_name = "OVERCAST/FOG";
    break;
  case MODE_RAINY:
    current_lane_config = lane_configs.rainy;
    current_object_config = &object_configs.rainy;
    mode_name = "RAINY";
    break;
  case MODE_NIGHT:
    current_lane_config = lane_configs.night;
    current_object_config = &object_configs.night;
    mode_name = "NIGHT";
    break;
  }

  std::printf("[AutoLight] %d프레임 연속 충족 → %s 모드로 전환 "
             "(Bright=%.1f, Sat=%.1f)\n",
             STABLE_FRAME_THRESHOLD, mode_name, metrics.brightness,
             metrics.saturation);
  open_logs(*current_object_config);
  return true;
}

void write_object_frame_summary(int &frame_index,
                               const std::vector<DetectedObject> &objects,
                               const char *mode_name) {
  int warning_count = 0;
  int stop_count = 0;
  double min_distance = 9999.0;
  for (const auto &object : objects) {
    if (object.risk == RISK_WARNING)
     ++warning_count;
    if (object.risk == RISK_STOP)
     ++stop_count;
    if (object.distance_m > 0 && object.distance_m < min_distance)
     min_distance = object.distance_m;
  }

  write_frame_log(frame_index++, static_cast<int>(objects.size()),
                 warning_count, stop_count,
                 min_distance < 9999.0 ? min_distance : -1.0, mode_name);
}

std::unique_ptr<DeepObjectDetector>
initialize_deep_detector(const ObjectDetectorConfig &config) {
  std::unique_ptr<DeepObjectDetector> detector;
#ifdef USE_ONNX_RUNTIME
  if (config.enable_deep_inference && config.deep_model_path[0] != '\0') {
    const int thread_count =
        std::max(1u, std::thread::hardware_concurrency());
    detector = std::make_unique<DeepObjectDetector>(
        config.deep_model_path, thread_count, false);
    if (!detector->is_initialized()) {
      std::printf("[Warning] Deep inference initialization failed, fallback "
                  "to hybrid pipeline\n");
      detector.reset();
    }
  }
#else
  if (config.enable_deep_inference) {
    std::printf("[Warn] ONNXRuntime 빌드가 없으므로 deep inference가 "
                "비활성화됩니다.\n");
  }
#endif
  return detector;
}

std::vector<DetectedObject>
detect_and_track_objects(const cv::Mat &frame,
                         const ObjectDetectorConfig &config,
                         CentroidTracker &tracker,
                         DeepObjectDetector *deep_detector,
                         cv::Mat &debug_mask) {
  const cv::Mat depth_map;
  const LidarSector lidar = {99.0, 99.0, 99.0, false};
  return detect_objects(frame, depth_map, config, tracker, lidar,
                        deep_detector, debug_mask);
}

int run_application(int argc, char **argv) {
  const ProgramOptions options = parse_program_options(argc, argv);
  LaneConfigSet lane_configs;
  if (!load_lane_config_set(options, lane_configs))
    return -1;

  AutoLightState light_state;
  LaneConfig cfg = lane_configs.sunny;

  cv::VideoCapture cap;
  int width = 0;
  int height = 0;
  double fps = 0.0;
  if (!open_input_video(options.video_path, cap, width, height, fps))
    return -1;

  ObjectDetectorConfigSet object_configs;
  if (!load_object_detector_config_set(object_configs)) {
    cap.release();
    return -1;
  }

  const ObjectDetectorConfig *current_obj_cfg = &object_configs.sunny;
  open_logs(*current_obj_cfg);

  std::unique_ptr<DeepObjectDetector> deep_detector =
      initialize_deep_detector(*current_obj_cfg);

  CentroidTracker tracker(current_obj_cfg->max_missing_frames,
                          current_obj_cfg->max_match_dist_px);
  set_tracker_dt(1.0 / fps);
  // ──────────────────────────────────────────────────────────────────────

  const std::filesystem::path output_path =
      resolve_output_path(options.output_path);
  cv::VideoWriter writer;
  if (!open_output_video(output_path, width, height, fps, writer)) {
    cap.release();
    return -1;
  }
  LaneState state = {{0, 0, 0, false}, {0, 0, 0, false}, 0.0, 0.0, 0, 0, false};
  cv::Mat frame;
  int frame_index = 0;

  while (cap.read(frame)) {
    if (frame.empty())
      break;

    cv::Mat obj_debug_mask;
    std::vector<DetectedObject> tracked_objects =
        detect_and_track_objects(frame, *current_obj_cfg, tracker,
                                 deep_detector.get(), obj_debug_mask);

    const LightingMetrics lighting = measure_lighting(frame);
    update_auto_light_mode(lighting, light_state, lane_configs, object_configs,
                           cfg, current_obj_cfg);
    const LightMode current_mode = light_state.current;
    const double mean_brightness = lighting.brightness;
    const double mean_saturation = lighting.saturation;

    cv::Mat roi_mask;
    cv::Mat roi_edges =
        create_lane_edge_map(frame, cfg, current_mode, mean_brightness,
                             roi_mask);
    const LaneSegmentPair detected_lanes =
        detect_lane_segments(roi_edges, cfg, width, height);
    update_lane_tracking(state, detected_lanes, cfg.temporal_alpha);
    const LanePosition lane_position =
        calculate_lane_position(state, cfg, width, height);

    cv::Mat display;
    draw_lane_guidance(frame, display, state, cfg, lane_position, current_mode,
                       lighting);

    const char *mask_mode_str = light_mode_name(current_mode);
    cv::Mat roi_overlay = create_lane_debug_overlay(
        roi_edges, roi_mask, obj_debug_mask, current_mode);

    // ── [객체 검출 시각화 오버레이] ──────────────────────────────────────
    draw_detections(display, tracked_objects);

    // ── [프레임 단위 로깅] ───────────────────────────────────────────────
    write_object_frame_summary(frame_index, tracked_objects, mask_mode_str);
    // ──────────────────────────────────────────────────────────────────────

    // 왼쪽(원본+결과)과 오른쪽(마스크)을 수평 합성하여 출력
    cv::Mat side_by_side(height, width * 2, CV_8UC3);
    display.copyTo(side_by_side(cv::Rect(0, 0, width, height)));
    roi_overlay.copyTo(side_by_side(cv::Rect(width, 0, width, height)));

    writer.write(side_by_side);
    // ─────────────────────────────────────────────────────────────
  }

  close_logs();
  cap.release();
  writer.release();
  return 0;
}
