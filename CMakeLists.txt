cmake_minimum_required(VERSION 3.25)
project(sparse_direct_vo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8 /bigobj /arch:AVX2)
  add_compile_definitions(_USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra -march=native)
endif()

find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs calib3d videoio)
find_package(GTest CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(tinyply CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

add_library(sdv STATIC
  src/sdv/camera.h
  src/sdv/image_pyramid.h
  src/sdv/image_pyramid.cpp
  src/sdv/immature_point.h
  src/sdv/immature_point.cpp
  src/sdv/mono_initializer.h
  src/sdv/mono_initializer.cpp
  src/sdv/odometry.h
  src/sdv/odometry.cpp
  src/sdv/parallel.h
  src/sdv/photometric.h
  src/sdv/photometric.cpp
  src/sdv/point_selector.h
  src/sdv/point_selector.cpp
  src/sdv/profiler.h
  src/sdv/rig.h
  src/sdv/stereo_depth.h
  src/sdv/stereo_depth.cpp
  src/sdv/tracker.h
  src/sdv/tracker.cpp
  src/sdv/undistort.h
  src/sdv/undistort.cpp
  src/sdv/validity_mask.h
  src/sdv/validity_mask.cpp
  src/sdv/window_optimizer.h
  src/sdv/window_optimizer.cpp
  src/sdv/io/kitti.h
  src/sdv/io/colmap.h
  src/sdv/io/colmap.cpp
  src/sdv/io/camera_config.h
  src/sdv/io/camera_config.cpp
  src/sdv/io/frame_source.h
  src/sdv/io/frame_source.cpp
  src/sdv/io/kitti.cpp
  src/sdv/io/ply.h
  src/sdv/io/ply.cpp
  src/sdv/io/rig_config.h
  src/sdv/io/rig_config.cpp
  src/sdv/eval/trajectory.h
  src/sdv/eval/trajectory.cpp
)
target_include_directories(sdv PUBLIC src)
target_link_libraries(sdv PUBLIC Eigen3::Eigen Sophus::Sophus spdlog::spdlog tinyply
  opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_videoio yaml-cpp::yaml-cpp)

add_executable(run_kitti apps/run_kitti.cpp)
target_link_libraries(run_kitti PRIVATE sdv Boost::program_options)

add_executable(track_stereo apps/track_stereo.cpp)
target_link_libraries(track_stereo PRIVATE sdv Boost::program_options)

add_executable(depth_test apps/depth_test.cpp)
target_link_libraries(depth_test PRIVATE sdv Boost::program_options)

add_executable(eval_traj apps/eval_traj.cpp)
target_link_libraries(eval_traj PRIVATE sdv Boost::program_options)

add_executable(mono_init apps/mono_init.cpp)
target_link_libraries(mono_init PRIVATE sdv Boost::program_options)

add_executable(window_stereo apps/window_stereo.cpp)
target_link_libraries(window_stereo PRIVATE sdv Boost::program_options)

add_executable(video_frames apps/video_frames.cpp)
target_link_libraries(video_frames PRIVATE sdv Boost::program_options)

add_executable(rig_from_sensorconfig apps/rig_from_sensorconfig.cpp)
target_link_libraries(rig_from_sensorconfig PRIVATE sdv Boost::program_options)

add_executable(estimate_sync apps/estimate_sync.cpp)
target_link_libraries(estimate_sync PRIVATE sdv Boost::program_options)

add_executable(run_vo apps/run_vo.cpp)
target_link_libraries(run_vo PRIVATE sdv Boost::program_options)

enable_testing()
add_executable(sdv_tests
  tests/synthetic_scene.h
  tests/test_camera.cpp
  tests/test_colmap.cpp
  tests/test_camera_config.cpp
  tests/test_immature_point.cpp
  tests/test_mono_initializer.cpp
  tests/test_odometry.cpp
  tests/test_photometric.cpp
  tests/test_ply.cpp
  tests/test_point_selector.cpp
  tests/test_pyramid.cpp
  tests/test_tracker.cpp
  tests/test_trajectory.cpp
  tests/test_window_optimizer.cpp
)
target_include_directories(sdv_tests PRIVATE tests)
target_link_libraries(sdv_tests PRIVATE sdv GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(sdv_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60)
