cmake_minimum_required(VERSION 3.16)
project(lunar_horizon CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

find_path(EIGEN3_INCLUDE_DIR
  NAMES Eigen/Dense
  HINTS ${EIGEN3_ROOT} /usr/include/eigen3 /usr/local/include/eigen3)
if(NOT EIGEN3_INCLUDE_DIR)
  message(FATAL_ERROR "Eigen not found; set EIGEN3_ROOT")
endif()

find_package(OpenMP)

add_executable(horizon src/horizon.cpp)
target_include_directories(horizon PRIVATE ${EIGEN3_INCLUDE_DIR} src)
if(OpenMP_CXX_FOUND)
  target_link_libraries(horizon PRIVATE OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found; the horizon precompute will be single threaded")
endif()

add_executable(edges src/edges.cpp)
target_include_directories(edges PRIVATE ${EIGEN3_INCLUDE_DIR} src)
if(OpenMP_CXX_FOUND)
  target_link_libraries(edges PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(planner src/planner.cpp)
target_include_directories(planner PRIVATE src)

add_executable(replay src/replay.cpp)
target_include_directories(replay PRIVATE src)
