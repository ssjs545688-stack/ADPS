#ifndef LANE_DETECTOR_H
#define LANE_DETECTOR_H

#include <opencv2/opencv.hpp>

typedef struct {
  double roi_top_ratio;
  double roi_bottom_ratio;
  double roi_left_ratio;
  double roi_right_ratio;
  double roi_top_left_ratio;   // 상단 좌측 꼭짓점 X 비율 (소실점 부근 좌측)
  double roi_top_right_ratio;  // 상단 우측 꼭짓점 X 비율 (소실점 부근 우측)
  int white_hsv_lower[3];
  int white_hsv_upper[3];
  int yellow_hsv_lower[3];
  int yellow_hsv_upper[3];
  int gaussian_kernel_size;
  int sobel_thresh_min;
  double canny_low_threshold;
  double canny_high_threshold;
  double hough_rho;
  double hough_theta_deg;
  int hough_threshold;
  double hough_min_line_length;
  double hough_max_line_gap;
  double minimum_absolute_slope;
  double maximum_absolute_slope;
  double lane_width_pixels;
  double lane_width_meters;
  double lookahead_y_ratio;
  double departure_warning_m;
  double departure_stop_m;
  double temporal_alpha;
  double minimum_confidence;
  double capture_frequency_hz;
  double video_fps;
} LaneConfig;

typedef struct {
  double slope;
  double intercept;
  double length;
  bool valid;
} LineSegment;

typedef struct {
  LineSegment left;
  LineSegment right;
  double conf_left;
  double conf_right;
  int left_coast_count;  // 점선 공백 구간 관성 메모리 유지 프레임 카운트
  int right_coast_count; // 점선 공백 구간 관성 메모리 유지 프레임 카운트
  bool initialized;
} LaneState;

enum LightMode { MODE_SUNNY, MODE_OVERCAST, MODE_RAINY, MODE_NIGHT };

struct LightingMetrics {
  double brightness;
  double saturation;
};

struct LaneSegmentPair {
  LineSegment left;
  LineSegment right;
};

struct LanePosition {
  double lookahead_y;
  double bottom_y;
  double left_x;
  double right_x;
  double offset_m;
  bool left_reliable;
  bool right_reliable;
};

typedef struct {
    double lateral_error_m;
    double lateral_error_px;
    double confidence;
    bool left_detected;
    bool right_detected;
    cv::Point2f left_lookahead;
    cv::Point2f right_lookahead;
    cv::Point2f center_lookahead;
} LaneResult;

bool load_config(const char *filepath, LaneConfig *config);
cv::Mat create_roi_mask(int width, int height, const LaneConfig *config);
cv::Mat create_hsv_mask(const cv::Mat &frame, const LaneConfig *config);
double calc_x_at_y(double slope, double intercept, double y);
cv::Mat create_lane_edge_map(const cv::Mat &frame, const LaneConfig &config,
                             LightMode mode, double brightness,
                             cv::Mat &roi_mask);
LaneSegmentPair detect_lane_segments(const cv::Mat &roi_edges,
                                     const LaneConfig &config, int width,
                                     int height);
void update_lane_tracking(LaneState &state, const LaneSegmentPair &current,
                          double alpha);
LanePosition calculate_lane_position(const LaneState &state,
                                     const LaneConfig &config, int width,
                                     int height);
void draw_lane_guidance(const cv::Mat &frame, cv::Mat &display,
                        const LaneState &state, const LaneConfig &config,
                        const LanePosition &position, LightMode mode,
                        const LightingMetrics &lighting);
const char *light_mode_name(LightMode mode);
cv::Mat create_lane_debug_overlay(const cv::Mat &roi_edges,
                                  const cv::Mat &roi_mask,
                                  const cv::Mat &object_debug_mask,
                                  LightMode mode);
LaneResult process_lane_pipeline(const cv::Mat &frame, const LaneConfig *config,
                                 cv::Mat &display_frame);

#endif
