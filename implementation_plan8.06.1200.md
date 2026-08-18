# [이슈 디버깅] ADAS 딥러닝 및 차선 인식 파이프라인 수정 계획

사용자님의 피드백을 바탕으로 3가지 크리티컬 이슈의 원인을 분석하였으며, 아래와 같이 수정을 진행하고자 합니다.

## User Review Required

> [!WARNING]
> 객체 검출 좌표 복원 로직을 기존의 `pad/scale` 역연산 방식에서, 사용자님이 제안하신 "가로세로 독립 스케일링(`x * (orig_width / tensor_width)`)" 방식으로 전면 교체합니다. 이 변경으로 인해 화면 비율 왜곡이 즉시 해소될 것입니다.

> [!IMPORTANT]
> 다각형 `roi_mask`가 메모리에 생성만 되고 실제 엣지 이미지에는 씌워지지(bitwise_and) 않는 누락 버그를 발견했습니다. 이 부분을 수정하여 하늘 및 가로수 노이즈를 완벽하게 차단하겠습니다.

## Proposed Changes

### 객체 검출 후처리 로직 (Bounding Box, NMS)

#### [MODIFY] [main.cpp](file:///c:/work/adas_telos/src/main.cpp)
- **Bounding Box 스케일링 수정**:
  - `src/main.cpp:817-820` 구간의 비율 역연산 공식을 사용자 제안 수식으로 교체합니다.
  - 수정 전: `cx = (cx - pad.x) / scale;`
  - 수정 후: `cx = cx * (static_cast<float>(orig_size.width) / 640.0f);` (y, w, h 동일 적용)
- **NMS IOU 임계값 최적화**:
  - `src/main.cpp:841`
  - IOU 임계값이 높아 겹친 박스가 제거되지 않는 현상(트럭에 여러 박스가 생기는 등)을 해결하기 위해 `nms_threshold`를 `0.35f`에서 `0.20f`로 하향 조정합니다. (조금만 겹쳐도 중복 제거됨)

### 차선 인식 (Lane Detection) 파이프라인

#### [MODIFY] [main.cpp](file:///c:/work/adas_telos/src/main.cpp)
- **다각형 ROI 마스크 적용 누락 수정**:
  - `src/main.cpp:2462`에서 `roi_mask`를 선언 및 생성하지만, 이후 Hough 변환 전까지 엣지 이미지와 융합되지 않는 버그 수정.
  - 컬러/그레이스케일 엣지가 모두 융합되어 `target_edges`가 최종 도출된 직후, `cv::bitwise_and(target_edges, roi_mask, target_edges);` 연산을 추가 적용하여 상단 노이즈 원천 차단.

## Verification Plan

### Automated Tests
- `make -j$(nproc)` 명령어를 통해 `main.cpp`의 문법 및 타입 에러 여부를 검증합니다.

### Manual Verification
- 수정된 `dms_node` 실행 후 `image_df9a00.jpg` (Bounding Box 크기 정상화 확인), `image_df99a5.jpg` / `image_df8ebf.jpg` (하늘, 가로수 등 배경 노이즈 제거 확인) 등 디버그 창 결과를 교차 검증하시길 권장합니다.
