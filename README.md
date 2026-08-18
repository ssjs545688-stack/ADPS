# ADAS_Telos

ADAS_Telos는 영상 및 센서 융합 기반의 Advanced Driver Assistance System(첨단 운전자 보조 시스템) 프로젝트입니다. 현재 차선 검출 파이프라인이 구현되어 있으며, 차량 등 전방 객체 탐지 및 충돌 위험 평가(FCW) 파이프라인의 통합이 계획되어 진행 중입니다.

## 🚀 주요 기능 (구현 및 계획 포함)

### 1. 환경 맞춤형 차선 검출 (Lane Detection)
- 다양한 날씨 및 조도 환경에 대응하기 위해 모듈화된 설정을 제공합니다.
- `configs/lane_config_night.json`, `lane_config_rainy.json` 등의 외부 설정을 통해 알고리즘 파라미터를 동적으로 제어합니다.

### 2. 하이브리드 객체 탐지 (Object Detection)
- **전통적 컴퓨터 비전**: HSV 임계값 처리(Thresholding), 모폴로지(Morphology), 윤곽선(Contour) 분석을 통한 차량 및 객체 후보 영역 검출.
- **딥러닝 추론**: ONNX Runtime 기반 경량 모델(YOLOv8n)을 활용한 신속하고 정확한 객체 인식 및 NMS(Non-Maximum Suppression) 처리.

### 3. 안정적인 객체 추적 (Tracking)
- **Centroid Tracker & Kalman Filter**: 프레임 간 객체의 매칭 및 연속성을 유지하며, 일시적 가려짐 현상을 보정합니다.

### 4. 깊이 추정 및 센서 융합 (Depth Estimation & Fusion)
- **Monocular Distance Estimation**: 단안 카메라의 바운딩 박스 크기 비율을 이용한 1차적 거리 역산.
- **Camera-LiDAR Sensor Fusion**: ROI(관심 영역) 내 카메라 깊이 데이터와 LiDAR 센서 데이터를 융합하여 정밀한 위치 파악.
- 노이즈 저감을 위해 지수 이동 평균(EMA, Exponential Moving Average) 필터 적용.

### 5. 다단계 충돌 위험 평가 (Risk Evaluation)
- 거리와 상대 접근 속도(Closing Speed)를 바탕으로 **TTC(Time-To-Collision)** 를 산출합니다.
- 상태(SAFE, WARNING, STOP)를 평가하고 Hysteresis(이력 현상) 및 Debouncing을 적용하여 경고의 신뢰도를 높입니다.

## 📁 프로젝트 구조

```text
adas_telos/
├── configs/            # 차선 및 객체 탐지 환경 설정 (JSON)
├── include/            # C++ 헤더 파일
├── src/                # C++ 소스 코드 (main.cpp, object_detector.cpp 등)
├── models/             # 딥러닝 가중치 모델 폴더 (yolov8n.pt 등)
├── CMakeLists.txt      # CMake 빌드 설정 파일
├── Dockerfile          # 컨테이너화 지원 파일
└── docker-compose.yml  # 다중 컨테이너 실행 스크립트
```

## 🛠️ 빌드 및 실행 (Build & Run)

본 프로젝트는 CMake를 이용하여 빌드할 수 있습니다.

```bash
# 1. 도커환경 진입

docker-compose build
docker-compose up 
docker-compose exec dms-app bash

# 2. 빌드 디렉토리 생성 및 이동
mkdir build
cd build

# 3. CMake 설정 및 빌드
cmake ..
make -j$(nproc) .

# 4. 실행
cd ..
./build/dms_node
```

## 💡 향후 계획 (TODO)
- `object_detector` 모듈의 `main.cpp` 완전 통합.
- 실제 LiDAR 드라이버/시리얼 연동 및 테스트.
- 다중 차선 검출 고도화 및 야간 환경에서의 탐지율 향상.
- 탑승자 인식 후 구별, 맞춤 음성 등 제공 
