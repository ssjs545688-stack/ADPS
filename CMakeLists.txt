cmake_minimum_required(VERSION 3.16)
project(dms_project VERSION 1.0.0 LANGUAGES CXX C)

# 1. C++ 표준 설정 (OpenCV 및 최신 C++ 기능 사용을 위해 C++17 권장)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 빌드 타입 설정 (기본값: Release / 온디바이스 성능 최적화)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 2. 필수 패키지 검색 (OpenCV, Threads, cJSON)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(cJSON CONFIG REQUIRED)

# ONNX Runtime 검색 (공식 Prebuilt 릴리즈는 CMake Config 파일 대신 헤더/소스를 직접 제공함)
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h PATHS /usr/local/include /usr/include)
find_library(ONNXRUNTIME_LIB NAMES onnxruntime PATHS /usr/local/lib /usr/lib)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    message(STATUS "Found ONNXRuntime library: ${ONNXRUNTIME_LIB}")
    message(STATUS "Found ONNXRuntime include: ${ONNXRUNTIME_INCLUDE_DIR}")

    # <onnxruntime/onnxruntime_cxx_api.h> 호환성을 위한 하위 디렉터리 생성 및 헤더 복사 (캐시/기존 환경 호환)
    if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/onnxruntime_cxx_api.h")
        file(MAKE_DIRECTORY "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime")
        file(GLOB ONNX_HEADERS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime*.h")
        foreach(HEADER_PATH ${ONNX_HEADERS})
            file(COPY "${HEADER_PATH}" DESTINATION "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime")
        endforeach()
    endif()

    if(NOT TARGET onnxruntime::onnxruntime)
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR};${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime"
        )
    endif()
else()
    message(FATAL_ERROR "ONNXRuntime not found (missing onnxruntime_cxx_api.h or libonnxruntime.so)")
endif()

# 3. 헤더 파일 경로 포함 (include 디렉터리 및 OpenCV 헤더)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# 4. 소스 파일 수집 (src 디렉터리 내부의 모든 .cpp 파일)
file(GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# 소스 파일이 없을 경우 대비 (src/main.cpp가 없으면 빌드 오류 방지 안내)
if(NOT SOURCES)
    message(WARNING "src 디렉터리에 .cpp 소스 파일이 없습니다. src/main.cpp를 작성해주세요.")
endif()

# 5. 실행 바이너리(dms_node) 생성 Target 설정
add_executable(dms_node ${SOURCES})

# 6. 라이브러리 링크 (OpenCV, Threads, 시스템 라이브러리)
target_link_libraries(dms_node
    PRIVATE
        ${OpenCV_LIBS}
        Threads::Threads
        cjson
)

if(TARGET onnxruntime::onnxruntime)
    target_compile_definitions(dms_node PRIVATE USE_ONNX_RUNTIME=1)
    target_link_libraries(dms_node PRIVATE onnxruntime::onnxruntime)
    message(STATUS "ONNXRuntime support enabled")
else()
    message(FATAL_ERROR "ONNXRuntime not found; deep inference module requires it.")
endif()

# 7. 빌드 정보 출력
message(STATUS "----------------------------------------------------")
message(STATUS "Project Name       : ${PROJECT_NAME}")
message(STATUS "Build Type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard       : ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV Version     : ${OpenCV_VERSION}")
message(STATUS "----------------------------------------------------")