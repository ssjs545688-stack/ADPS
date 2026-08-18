#ifndef LANE_DETECTOR_H
#define LANE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <cjson/cJSON.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// lane_config.json과 1:1 매핑되는 구조체
typedef struct {
    float roi_top_ratio;
    float roi_bottom_ratio;
    float roi_left_ratio;
    float roi_right_ratio;

    int white_hsv_lower[3];
    int white_hsv_upper[3];
    int yellow_hsv_lower[3];
    int yellow_hsv_upper[3];

    int gaussian_kernel_size;
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

// 차선 인식 최종 결과 구조체
typedef struct {
    double lateral_error_m;      // 횡오차 (m)
    double lateral_error_px;     // 횡오차 (pixel)
    double confidence;           // 신뢰도 (0.0 ~ 1.0)
    bool left_detected;          // 좌측 차선 검출 여부
    bool right_detected;         // 우측 차선 검출 여부
    cv::Point2f left_lookahead;  // Lookahead Y 지점의 좌측 차선 좌표
    cv::Point2f right_lookahead; // Lookahead Y 지점의 우측 차선 좌표
    cv::Point2f center_lookahead;// Lookahead Y 지점의 중심 좌표
} LaneResult;

// 주요 함수 원형
bool load_config(const char* filename, LaneConfig* config);
LaneResult process_lane_pipeline(const cv::Mat& frame, const LaneConfig* config, cv::Mat& display_frame);

#endif