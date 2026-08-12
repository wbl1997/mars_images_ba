cmake_minimum_required(VERSION 3.10)
project(Mars)

# 单配置生成器下默认 Release（可用 -DCMAKE_BUILD_TYPE=Debug 覆盖）
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 源文件
file(GLOB SRC_FILES
    src/*.cpp
)

# 头文件目录
include_directories(
    include
    include/myinc
    Mars
)

# 查找常用库
set(OpenCV_DIR "/home/wbl/pkgs/opencv455/opencv/install/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(Glog REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG REQUIRED libglog)

pkg_check_modules(GFLAGS libgflags)

find_package(Ceres REQUIRED)
find_package(GDAL REQUIRED)
find_library(PROJ_LIB proj)

set(CSPICE_ROOT "/home/wbl/pkgs/cspice")
if(NOT EXISTS ${CSPICE_ROOT})
    message(FATAL_ERROR "CSPICE directory not found: ${CSPICE_ROOT}")
endif()

# 查找 CSPICE 库文件
find_library(CSPICE_LIB 
    NAMES cspice cspice.a
    PATHS ${CSPICE_ROOT}/lib
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(CSUPPORT_LIB 
    NAMES csupport csupport.a
    PATHS ${CSPICE_ROOT}/lib
    NO_DEFAULT_PATH
    REQUIRED
)

# 检查库文件是否找到
if(NOT CSPICE_LIB)
    message(FATAL_ERROR "CSPICE library not found in ${CSPICE_ROOT}/lib")
endif()

if(NOT CSUPPORT_LIB)
    message(FATAL_ERROR "CSUPPORT library not found in ${CSPICE_ROOT}/lib")
endif()

message(STATUS "Found CSPICE: ${CSPICE_LIB}")
message(STATUS "Found CSUPPORT: ${CSUPPORT_LIB}")

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    include
    ${CSPICE_ROOT}/include
)

add_executable(Mars Mars/Mars.cpp ${SRC_FILES})

# 链接库
target_link_libraries(Mars
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${GLOG_LIBRARIES}
    Threads::Threads
    GDAL::GDAL
    ${CSPICE_LIB}
    ${CSUPPORT_LIB}
    OpenMP::OpenMP_CXX
    m  # 数学库，CSPICE 需要
    # glog
)

add_executable(draw_match Mars/draw_match.cpp src/ImageMatch.cpp)

# 链接库（ImageMatch 使用 OpenMP，需与 Mars 一样链接）
target_link_libraries(draw_match 
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${GLOG_LIBRARIES}
    Threads::Threads
    GDAL::GDAL
    ${CSPICE_LIB}
    ${CSUPPORT_LIB}
    OpenMP::OpenMP_CXX
    m  # 数学库，CSPICE 需要
)

add_executable(scoreb_robust_test tests/scoreb_robust_test.cpp)
target_link_libraries(scoreb_robust_test
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${GLOG_LIBRARIES}
    Threads::Threads
    GDAL::GDAL
    ${CSPICE_LIB}
    ${CSUPPORT_LIB}
    OpenMP::OpenMP_CXX
    m
)

add_executable(scoreb_config_test tests/scoreb_config_test.cpp)

add_executable(local_prediction_test tests/local_prediction_test.cpp)
target_link_libraries(local_prediction_test
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${GLOG_LIBRARIES}
    Threads::Threads
    GDAL::GDAL
    ${CSPICE_LIB}
    ${CSUPPORT_LIB}
    OpenMP::OpenMP_CXX
    m
)


if(GFLAGS_FOUND)
    target_link_libraries(Mars ${GFLAGS_LIBRARIES})
endif()

# proj4
if(PROJ_LIB)
    target_link_libraries(Mars ${PROJ_LIB})
endif()

# 可选：设置输出目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
