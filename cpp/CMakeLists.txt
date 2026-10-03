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

# The executive of phase 6. BehaviorTree.CPP 4.9 ships in the ROS distribution
# at /opt/ros/jazzy. Its own CMake package config is an ament one and pulls in
# ament_cmake, which needs a Python module that is not installed here, so the
# header and the library are found directly in the manner this file already
# uses for Eigen. Nothing is installed and no ROS node is involved.
find_path(BTCPP_INCLUDE_DIR
  NAMES behaviortree_cpp/bt_factory.h
  HINTS ${BTCPP_ROOT} /opt/ros/jazzy/include /usr/local/include /usr/include)
find_library(BTCPP_LIBRARY
  NAMES behaviortree_cpp
  HINTS ${BTCPP_ROOT} /opt/ros/jazzy/lib /usr/local/lib /usr/lib)
get_filename_component(BTCPP_RPATH ${BTCPP_LIBRARY} DIRECTORY)
if(BTCPP_INCLUDE_DIR AND BTCPP_LIBRARY)
  add_executable(executive src/executive.cpp)
  target_include_directories(executive PRIVATE src ${BTCPP_INCLUDE_DIR})
  target_link_libraries(executive PRIVATE ${BTCPP_LIBRARY})
  # The ROS-built library carries no runpath of its own and expects
  # LD_LIBRARY_PATH to be set by a ROS setup script. A RUNPATH on this
  # executable would not help, because RUNPATH is not consulted when resolving
  # a dependency's own dependencies; the old RPATH tag is, so the binary is
  # self-contained and needs no environment to run.
  set_target_properties(executive PROPERTIES BUILD_RPATH "${BTCPP_RPATH}")
  target_link_options(executive PRIVATE "-Wl,--disable-new-dtags")
else()
  message(WARNING "BehaviorTree.CPP not found; the executive will not be built")
endif()
