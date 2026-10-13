#include "lane_detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <cjson/cJSON.h>

bool load_config(const char *filepath, LaneConfig *config) {
  if (!config)
    return false;

  const std::filesystem::path path(filepath);
  if (!std::filesystem::exists(path))
    return false;

  std::ifstream file(path, std::ios::binary);
  if (!file.is_open())
    return false;

  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  if (content.empty())
    return false;

  cJSON *json = cJSON_Parse(content.c_str());
  if (!json)
    return false;

  auto get_double = [](cJSON *obj, const char *key, double fallback) -> double {
    cJSON *item = cJSON_GetObjectItem(obj, key);
    return item && cJSON_IsNumber(item) ? item->valuedouble : fallback;
  };
  auto get_int = [](cJSON *obj, const char *key, int fallback) -> int {
    cJSON *item = cJSON_GetObjectItem(obj, key);
    return item && cJSON_IsNumber(item) ? item->valueint : fallback;
  };
  auto get_array3i = [](cJSON *obj, const char *key, int out[3]) -> bool {
    cJSON *arr = cJSON_GetObjectItem(obj, key);
    if (!arr || !cJSON_IsArray(arr) || cJSON_GetArraySize(arr) < 3)
      return false;

    for (int i = 0; i < 3; ++i) {
      cJSON *elem = cJSON_GetArrayItem(arr, i);
      if (!elem || !cJSON_IsNumber(elem))
        return false;
      out[i] = elem->valueint;
    }
    return true;
  };

  config->roi_top_ratio = get_double(json, "roi_top_ratio", 0.0);
  config->roi_bottom_ratio = get_double(json, "roi_bottom_ratio", 0.0);
  config->roi_left_ratio = get_double(json, "roi_left_ratio", 0.05);
  config->roi_right_ratio = get_double(json, "roi_right_ratio", 0.95);
  config->roi_top_left_ratio = get_double(json, "roi_top_left_ratio", 0.35);
  config->roi_top_right_ratio = get_double(json, "roi_top_right_ratio", 0.65);

  int white_lower[3] = {0, 0, 0};
  int white_upper[3] = {0, 0, 0};
  int yellow_lower[3] = {0, 0, 0};
  int yellow_upper[3] = {0, 0, 0};
  if (get_array3i(json, "white_hsv_lower", white_lower))
    std::memcpy(config->white_hsv_lower, white_lower, sizeof(white_lower));
  if (get_array3i(json, "white_hsv_upper", white_upper))
    std::memcpy(config->white_hsv_upper, white_upper, sizeof(white_upper));
  if (get_array3i(json, "yellow_hsv_lower", yellow_lower))
    std::memcpy(config->yellow_hsv_lower, yellow_lower, sizeof(yellow_lower));
  if (get_array3i(json, "yellow_hsv_upper", yellow_upper))
    std::memcpy(config->yellow_hsv_upper, yellow_upper, sizeof(yellow_upper));

  config->gaussian_kernel_size = get_int(json, "gaussian_kernel_size", 3);
  config->sobel_thresh_min = get_int(json, "sobel_thresh_min", 35);
  config->canny_low_threshold = get_double(json, "canny_low_threshold", 50.0);
  config->canny_high_threshold =
      get_double(json, "canny_high_threshold", 150.0);
  config->hough_rho = get_double(json, "hough_rho", 1.0);
  config->hough_theta_deg = get_double(json, "hough_theta_deg", 1.0);
  config->hough_threshold = get_int(json, "hough_threshold", 50);
  config->hough_min_line_length =
      get_double(json, "hough_min_line_length", 30.0);
  config->hough_max_line_gap = get_double(json, "hough_max_line_gap", 20.0);
  config->minimum_absolute_slope =
      get_double(json, "minimum_absolute_slope", 0.1);
  config->maximum_absolute_slope =
      get_double(json, "maximum_absolute_slope", 2.0);
  config->lane_width_pixels = get_double(json, "lane_width_pixels", 200.0);
  config->lane_width_meters = get_double(json, "lane_width_meters", 3.5);
  config->lookahead_y_ratio = get_double(json, "lookahead_y_ratio", 0.8);
  config->departure_warning_m = get_double(json, "departure_warning_m", 0.3);
  config->departure_stop_m = get_double(json, "departure_stop_m", 0.7);
  config->temporal_alpha = get_double(json, "temporal_alpha", 0.2);
  config->minimum_confidence = get_double(json, "minimum_confidence", 0.5);
  config->capture_frequency_hz = get_double(json, "capture_frequency_hz", 10.0);
  config->video_fps = get_double(json, "video_fps", 30.0);

  cJSON_Delete(json);
  return true;
}

cv::Mat create_roi_mask(int width, int height, const LaneConfig *config) {
  cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);

  // ROI 사다리꼴 최대 확장 설정
  // - 하단: 화면 맨 왼쪽(0) ~ 맨 오른쪽(width), 맨 아래(height) 전부 포함
  // - 상단: 좌측 0.20, 우측 0.80 으로 대폭 확장하여 급커브/교차로 차선 손실 방지
  // - 상단 Y: config->roi_top_ratio (JSON 파일로 날씨별 조정 가능)
  std::vector<cv::Point> pts = {
      cv::Point(0,                                                 height),                                        // 좌측 하단: 화면 끝
      cv::Point(static_cast<int>(width * config->roi_top_left_ratio),  static_cast<int>(height * config->roi_top_ratio)), // 좌측 상단
      cv::Point(static_cast<int>(width * config->roi_top_right_ratio), static_cast<int>(height * config->roi_top_ratio)), // 우측 상단
      cv::Point(width,                                             height)};                                       // 우측 하단: 화면 끝
  cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{pts}, cv::Scalar(255));

  // 하단 대시보드 및 후드 차단 (하위 5% 마스킹 - 최소한만 제거)
  int dashboard_cut_y = static_cast<int>(height * 0.95);
  cv::rectangle(mask, cv::Point(0, dashboard_cut_y), cv::Point(width, height),
                cv::Scalar(0), -1);

  return mask;
}


cv::Mat create_hsv_mask(const cv::Mat &frame, const LaneConfig *config) {
  // ── [1] CLAHE 전처리: 그림자/저조도 영역 로컬 대비 극대화 ──────────────
  // L 채널(명도)에 한정 적용 → 하이라이트 날아감(White-out) 방지
  cv::Mat hls_pre;
  cv::cvtColor(frame, hls_pre, cv::COLOR_BGR2HLS);
  std::vector<cv::Mat> hls_pre_ch;
  cv::split(hls_pre, hls_pre_ch);  // ch[0]=H, ch[1]=L, ch[2]=S

  auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  cv::Mat l_enhanced;
  clahe->apply(hls_pre_ch[1], l_enhanced);
  hls_pre_ch[1] = l_enhanced;

  cv::Mat hls_enhanced;
  cv::merge(hls_pre_ch, hls_enhanced);

  // CLAHE 적용된 이미지를 BGR로 역변환하여 이후 처리에 활용
  cv::Mat frame_enhanced;
  cv::cvtColor(hls_enhanced, frame_enhanced, cv::COLOR_HLS2BGR);

  // ── [2] HSV 기반 컬러 마스크 (기존 로직, CLAHE 보정 이미지 입력) ─────
  cv::Mat hsv, mask_white_hsv, mask_yellow_hsv;
  cv::cvtColor(frame_enhanced, hsv, cv::COLOR_BGR2HSV);
  cv::inRange(hsv,
              cv::Scalar(config->white_hsv_lower[0], config->white_hsv_lower[1],
                         config->white_hsv_lower[2]),
              cv::Scalar(config->white_hsv_upper[0], config->white_hsv_upper[1],
                         config->white_hsv_upper[2]),
              mask_white_hsv);
  cv::inRange(
      hsv,
      cv::Scalar(config->yellow_hsv_lower[0], config->yellow_hsv_lower[1],
                 config->yellow_hsv_lower[2]),
      cv::Scalar(config->yellow_hsv_upper[0], config->yellow_hsv_upper[1],
                 config->yellow_hsv_upper[2]),
      mask_yellow_hsv);

  // ── [3] HLS 기반 보조 마스크 (그림자 속 차선 보강) ─────────────────────
  // 흰 차선: HLS L채널 고명도 영역 → 그림자 속에서도 L값 150 이상이면 흰 차선
  // 노란 차선: HLS H채널(15~40) + S채널(80 이상) 조합 → 그림자에도 채도 유지
  cv::Mat hls_full;
  cv::cvtColor(frame_enhanced, hls_full, cv::COLOR_BGR2HLS);

  cv::Mat mask_white_hls, mask_yellow_hls;
  // 흰 차선: H 무관, L >= 150, S 낮음(무채색)
  cv::inRange(hls_full,
              cv::Scalar(0,   150,  0),
              cv::Scalar(180, 255, 60),
              mask_white_hls);
  // 노란 차선: H 15~40 (노란색 대역), L 50 이상, S 80 이상 (그림자에도 채도 잔존)
  cv::inRange(hls_full,
              cv::Scalar(15,  50,  80),
              cv::Scalar(40, 220, 255),
              mask_yellow_hls);

  // HSV 마스크와 HLS 마스크를 OR 융합 (그림자에서 어느 한쪽이라도 검출되면 인정)
  cv::Mat combined;
  cv::Mat mask_white, mask_yellow;
  cv::bitwise_or(mask_white_hsv,  mask_white_hls,  mask_white);
  cv::bitwise_or(mask_yellow_hsv, mask_yellow_hls, mask_yellow);
  cv::bitwise_or(mask_white, mask_yellow, combined);

  // ── [4] Sobel X 수직 엣지 AND 필터 (기존 로직 유지) ────────────────────
  cv::Mat gray, blur, sobel_x, sobel_mask;
  cv::cvtColor(frame_enhanced, gray, cv::COLOR_BGR2GRAY);

  int ksize = config->gaussian_kernel_size;
  if (ksize % 2 == 0)
    ksize += 1;
  cv::GaussianBlur(gray, blur, cv::Size(ksize, ksize), 0);

  cv::Sobel(blur, sobel_x, CV_32F, 1, 0, 3);
  sobel_x = cv::abs(sobel_x);

  double minVal, maxVal;
  cv::minMaxLoc(sobel_x, &minVal, &maxVal);
  if (maxVal > 0)
    sobel_x = sobel_x / maxVal * 255.0;
  sobel_x.convertTo(sobel_x, CV_8U);

  cv::threshold(sobel_x, sobel_mask, config->sobel_thresh_min, 255,
                cv::THRESH_BINARY);

  cv::Mat sobel_dilated;
  cv::dilate(sobel_mask, sobel_dilated, cv::Mat(), cv::Point(-1, -1), 3);
  cv::Mat final_mask;
  cv::bitwise_and(combined, sobel_dilated, final_mask);

  return final_mask;
}


double calc_x_at_y(double slope, double intercept, double y) {
  if (fabs(slope) < 1e-5)
    return 0.0;
  return (y - intercept) / slope;
}

// ── 양선 거리 및 평행도 검증 ──

// ─────────────────────────────────────────────────────────────────
// 자동 조도 및 기상 감지 임계값 및 상태 정의 (Hysteresis & N-Frame Debounce)
// ─────────────────────────────────────────────────────────────────

cv::Mat create_lane_edge_map(const cv::Mat &frame, const LaneConfig &config,
                            LightMode mode, double brightness,
                            cv::Mat &roi_mask) {
  const int width = frame.cols;
  const int height = frame.rows;
  roi_mask = create_roi_mask(width, height, &config);
  cv::Mat color_mask = create_hsv_mask(frame, &config);

  const int sky_height = static_cast<int>(height * config.roi_top_ratio);
  if (sky_height > 0 && sky_height < height)
    color_mask(cv::Rect(0, 0, width, sky_height)).setTo(0);

  int kernel_size = config.gaussian_kernel_size;
  if (kernel_size % 2 == 0)
    ++kernel_size;

  constexpr double bright_tight = 120.0;
  constexpr double bright_loose = 60.0;
  double canny_scale = 1.0;
  if (brightness <= bright_loose) {
    canny_scale = 0.45;
  } else if (brightness < bright_tight) {
    canny_scale =
        0.45 + 0.55 * (brightness - bright_loose) /
                  (bright_tight - bright_loose);
  }
  const double canny_low = config.canny_low_threshold * canny_scale;
  const double canny_high = config.canny_high_threshold * canny_scale;

  cv::Mat blurred_color, color_edges;
  cv::GaussianBlur(color_mask, blurred_color,
                  cv::Size(kernel_size, kernel_size), 0);
  cv::Canny(blurred_color, color_edges, canny_low, canny_high);

  cv::Mat target_edges;
  if (mode == MODE_SUNNY) {
    target_edges = color_edges;
  } else {
    cv::Mat gray_frame, blurred_gray, gray_edges;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray_frame, blurred_gray,
                    cv::Size(kernel_size, kernel_size), 0);
    cv::Canny(blurred_gray, gray_edges, canny_low, canny_high);
    cv::bitwise_or(color_edges, gray_edges, target_edges);
  }

  cv::Mat roi_edges;
  cv::bitwise_and(target_edges, target_edges, roi_edges, roi_mask);
  return roi_edges;
}



LaneSegmentPair detect_lane_segments(const cv::Mat &roi_edges,
                                    const LaneConfig &config, int width,
                                    int height) {
  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(roi_edges, lines, config.hough_rho,
                 config.hough_theta_deg * CV_PI / 180.0,
                 config.hough_threshold, config.hough_min_line_length,
                 config.hough_max_line_gap);

  double left_slope_sum = 0.0;
  double left_intercept_sum = 0.0;
  double left_weight_sum = 0.0;
  double right_slope_sum = 0.0;
  double right_intercept_sum = 0.0;
  double right_weight_sum = 0.0;

  for (const auto &line : lines) {
    const double x1 = line[0];
    const double y1 = line[1];
    const double x2 = line[2];
    const double y2 = line[3];
    if (x1 == x2)
      continue;

    const double slope = (y2 - y1) / (x2 - x1);
    if (fabs(slope) < config.minimum_absolute_slope ||
        fabs(slope) > config.maximum_absolute_slope)
      continue;

    const double intercept = y1 - slope * x1;
    const double length = std::sqrt((x2 - x1) * (x2 - x1) +
                                   (y2 - y1) * (y2 - y1));
    const double proximity = std::max(y1, y2) / static_cast<double>(height);
    const double weight = length * proximity * proximity * proximity;

    if (slope < 0 && x1 < width * 0.55 && x2 < width * 0.55) {
      left_slope_sum += slope * weight;
      left_intercept_sum += intercept * weight;
      left_weight_sum += weight;
    } else if (slope > 0 && x1 > width * 0.45 && x2 > width * 0.45) {
      right_slope_sum += slope * weight;
      right_intercept_sum += intercept * weight;
      right_weight_sum += weight;
    }
  }

  LaneSegmentPair segments = {
      {0, 0, left_weight_sum, left_weight_sum > 0},
      {0, 0, right_weight_sum, right_weight_sum > 0}};
  if (segments.left.valid) {
    segments.left.slope = left_slope_sum / left_weight_sum;
    segments.left.intercept = left_intercept_sum / left_weight_sum;
  }
  if (segments.right.valid) {
    segments.right.slope = right_slope_sum / right_weight_sum;
    segments.right.intercept = right_intercept_sum / right_weight_sum;
  }
  return segments;
}

void update_lane_tracking(LaneState &state, const LaneSegmentPair &current,
                         double alpha) {
  constexpr int max_coast_frames = 8;
  if (!state.initialized) {
    state.left = current.left;
    state.right = current.right;
    state.left_coast_count = 0;
    state.right_coast_count = 0;
    state.initialized = true;
    return;
  }

  const auto update_side = [alpha](LineSegment &tracked,
                                  const LineSegment &detected,
                                  double &confidence, int &coast_count) {
    if (detected.valid) {
      tracked.slope = alpha * detected.slope + (1 - alpha) * tracked.slope;
      tracked.intercept =
          alpha * detected.intercept + (1 - alpha) * tracked.intercept;
      confidence = std::min(1.0, confidence + 0.2);
      coast_count = 0;
    } else if (coast_count < max_coast_frames) {
      ++coast_count;
      confidence = std::max(0.0, confidence - 0.04);
    } else {
      confidence = std::max(0.0, confidence - 0.15);
    }
  };

  update_side(state.left, current.left, state.conf_left,
              state.left_coast_count);
  update_side(state.right, current.right, state.conf_right,
              state.right_coast_count);
}



LanePosition calculate_lane_position(const LaneState &state,
                                    const LaneConfig &config, int width,
                                    int height) {
  const double lookahead_y = height * config.lookahead_y_ratio;
  const double bottom_y = height * config.roi_bottom_ratio;
  const double left_x =
      calc_x_at_y(state.left.slope, state.left.intercept, lookahead_y);
  const double right_x =
      calc_x_at_y(state.right.slope, state.right.intercept, lookahead_y);
  const bool left_reliable = state.conf_left >= config.minimum_confidence;
  const bool right_reliable = state.conf_right >= config.minimum_confidence;

  const double image_center_x = width / 2.0;
  double lane_center_x = image_center_x;
  if (left_reliable && right_reliable)
    lane_center_x = (left_x + right_x) / 2.0;
  else if (left_reliable)
    lane_center_x = left_x + config.lane_width_pixels / 2.0;
  else if (right_reliable)
    lane_center_x = right_x - config.lane_width_pixels / 2.0;

  const double offset_pixels = image_center_x - lane_center_x;
  const double meter_per_pixel =
      config.lane_width_meters / config.lane_width_pixels;
  return {lookahead_y, bottom_y, left_x, right_x,
          offset_pixels * meter_per_pixel, left_reliable, right_reliable};
}

void draw_lane_guidance(const cv::Mat &frame, cv::Mat &display,
                        const LaneState &state, const LaneConfig &config,
                        const LanePosition &position, LightMode mode,
                        const LightingMetrics &lighting) {
  display = frame.clone();
  if (position.left_reliable) {
    const double x1 = calc_x_at_y(state.left.slope, state.left.intercept,
                                  position.bottom_y);
    cv::line(display, cv::Point(x1, position.bottom_y),
             cv::Point(position.left_x, position.lookahead_y),
             cv::Scalar(255, 0, 0), 3);
  }
  if (position.right_reliable) {
    const double x1 = calc_x_at_y(state.right.slope, state.right.intercept,
                                  position.bottom_y);
    cv::line(display, cv::Point(x1, position.bottom_y),
             cv::Point(position.right_x, position.lookahead_y),
             cv::Scalar(0, 0, 255), 3);
  }

  cv::line(display, cv::Point(0, position.lookahead_y),
           cv::Point(frame.cols, position.lookahead_y),
           cv::Scalar(0, 255, 255), 1);
  char text[128];
  std::snprintf(text, sizeof(text),
                "Offset: %.2fm | Conf (L:%.1f, R:%.1f)", position.offset_m,
                state.conf_left, state.conf_right);
  cv::putText(display, text, cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0, 255, 0), 2);

  const char *mode_label = "MODE: SUNNY (Auto)";
  cv::Scalar mode_color(0, 220, 255);
  if (mode == MODE_OVERCAST) {
    mode_label = "MODE: OVERCAST/FOG (Auto)";
    mode_color = cv::Scalar(220, 220, 0);
  } else if (mode == MODE_RAINY) {
    mode_label = "MODE: RAINY (Auto)";
    mode_color = cv::Scalar(255, 180, 0);
  } else if (mode == MODE_NIGHT) {
    mode_label = "MODE: NIGHT (Auto)";
    mode_color = cv::Scalar(255, 50, 150);
  }
  std::snprintf(text, sizeof(text), "%s | Bright: %.1f | Sat: %.1f",
                mode_label, lighting.brightness, lighting.saturation);
  cv::putText(display, text, cv::Point(30, 70), cv::FONT_HERSHEY_SIMPLEX, 0.65,
              mode_color, 2);

  if (!position.left_reliable && !position.right_reliable) {
    cv::putText(display, "[WARNING] LANE TRACKING LOST!", cv::Point(30, 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2);
  } else if (fabs(position.offset_m) > config.departure_stop_m) {
    cv::putText(display, "[WARNING] STOP / CRITICAL DEPARTURE!",
                cv::Point(30, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 0, 255), 2);
  } else if (fabs(position.offset_m) > config.departure_warning_m) {
    cv::putText(display, "[WARNING] LANE DEPARTURE!", cv::Point(30, 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2);
  }
}

const char *light_mode_name(LightMode mode) {
  switch (mode) {
  case MODE_SUNNY:
    return "SUNNY";
  case MODE_OVERCAST:
    return "OVERCAST";
  case MODE_RAINY:
    return "RAINY";
  case MODE_NIGHT:
    return "NIGHT";
  }
  return "UNKNOWN";
}

cv::Mat create_lane_debug_overlay(const cv::Mat &roi_edges,
                                  const cv::Mat &roi_mask,
                                  const cv::Mat &object_debug_mask,
                                  LightMode mode) {
  cv::Mat edge_overlay, roi_mask_bgr, output;
  cv::cvtColor(roi_edges, edge_overlay, cv::COLOR_GRAY2BGR);
  cv::cvtColor(roi_mask, roi_mask_bgr, cv::COLOR_GRAY2BGR);
  cv::addWeighted(edge_overlay, 1.0, roi_mask_bgr, 0.25, 0, output);
  cv::putText(output, "Lane (White/Yellow) + Shadow (Red)", cv::Point(10, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 128), 2);

  if (!object_debug_mask.empty() && object_debug_mask.size() == output.size()) {
    const cv::Mat red_overlay(output.size(), CV_8UC3, cv::Scalar(0, 0, 255));
    red_overlay.copyTo(output, object_debug_mask);
  }
  char mode_text[64];
  std::snprintf(mode_text, sizeof(mode_text), "Mode: %s",
                light_mode_name(mode));
  cv::putText(output, mode_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
              0.6, cv::Scalar(0, 220, 255), 2);
  return output;
}
