# ADAS 객체 검출 파이프라인 구현 계획 (§1~§8)

기존 `main.cpp`(차선 검출 전용)에 완전한 객체 검출·추적·충돌 위험도 평가 파이프라인을 모듈로 추가합니다.  
`object_detector.h`의 뼈대가 이미 있으므로, 해당 인터페이스에 맞는 구현체(`object_detector.cpp`)와 JSON 설정(`object_detector_config.json`)을 신규 생성하고, `main.cpp`를 수정하여 두 파이프라인을 통합합니다.

---

## 제안 변경사항

### 1. 신규: `configs/object_detector_config.json`
§2~§8 전 파라미터를 JSON으로 외부화하여 재컴파일 없이 튜닝 가능하게 합니다.

```
{
  "hsv": { "red_lower1/upper1/lower2/upper2", "yellow_lower/upper" },
  "morph": { "open_size", "close_size" },
  "contour": { "min_area", "max_area", "poly_epsilon_ratio", "circle_threshold" },
  "depth": { "use_depth", "depth_scale", "inner_ratio", "percentile", ... },
  "tracker": { "max_match_dist_px", "max_missing_frames" },
  "risk": { "warn_dist_m", "stop_dist_m", "warn_ttc_s", "stop_ttc_s", "min_danger_frames" },
  "ema": { "alpha" },
  "logging": { "enable", "obj_log_path", "frame_log_path" }
}
```

### 2. 신규: `src/object_detector.cpp`
`object_detector.h`의 모든 함수·클래스를 구현합니다.

| 섹션 | 구현 내용 |
|------|-----------|
| §1 | `BB_X2`, `BB_CX` 등 매크로 활용, `init_detector_config()` |
| §2 | `detect_color_candidates()` – HSV 변환, 빨강 이중 구간 OR, Open/Close 모폴로지 |
| §3 | `classify_contours()` – `RETR_EXTERNAL`, 면적/둘레 필터, `approxPolyDP` + 원형도 |
| §4 | `compute_robust_depth()` – 내부 60% ROI 수집, 30분위수, 해상도 스케일 |
| §5 | `CentroidTracker::update()` – 유클리드 매칭, missing 관리, `assign_sector()` |
| §6 | `evaluate_risk()` – Closing Speed, TTC, 복합 위험 단계 판정 |
| §7 | EMA 필터(`apply_ema()`), `fuse_lidar()` – 카메라·LiDAR 구역 비교 융합 |
| §8 | `open_logs/close_logs/write_frame_log()`, danger_frames 연속성 검증 |

### 3. 수정: `src/main.cpp`
- `#include "object_detector.h"` 추가
- `main()` 루프 내에 `detect_objects()` 호출 및 `draw_detections()` 삽입
- LiDAR 더미 데이터 또는 실제 데이터 연결 포인트 추가

### 4. 수정: `CMakeLists.txt`
- `cJSON` 라이브러리 링크 추가 (`cjson`)
- `object_detector.cpp`는 `GLOB_RECURSE`로 자동 수집됨

---

## 오픈 질문

> [!IMPORTANT]
> **LiDAR 연동 방식**: 현재 `main.cpp`에 LiDAR 코드가 없습니다.  
> `LidarSector`를 (A) 더미 고정값 `{99.0, 99.0, 99.0, false}`로 초기화하고 실제 데이터 연결 포인트만 남길까요, 아니면 (B) 실제 LiDAR 드라이버/시리얼 연동까지 구현할까요?

> [!IMPORTANT]
> **Depth 카메라**: 현재 영상이 단일 RGB AVI 파일입니다.  
> Depth Map은 (A) 빈 Mat(`cv::Mat()`)으로 처리하여 `use_depth=false`로 작동, 아니면 (B) 별도 파일(`.bin`/`.exr`) 경로를 지정하는 방식으로 할까요?

> [!NOTE]
> 위 두 질문에 대한 기본 방향: **더미/비활성화** 방식으로 구현하고 실제 연동 주석·TODO를 남겨둡니다.

---

## 검증 계획
1. `cmake --build build` 컴파일 성공 확인
2. 기존 `good/sample_vid.mp4` 재생 시 객체 검출 오버레이 확인
3. CSV 로그 파일 생성 확인
