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
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs calib3d videoio features2d)
find_package(GTest CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(tinyply CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(fbow CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)
find_package(libzip CONFIG REQUIRED)
find_package(FFmpegLibs CONFIG REQUIRED COMPONENTS avcodec avutil swscale PATHS ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  NO_DEFAULT_PATH)

# aiMotive recording reader (camera streams, sensorconfig calibration); independent of the odometry library.
add_library(sdv_aimrec STATIC
  src/sdv/aimrec/calibration.h
  src/sdv/aimrec/calibration.cpp
  src/sdv/aimrec/camera_meta.h
  src/sdv/aimrec/camera_meta.cpp
  src/sdv/aimrec/camera_stream.h
  src/sdv/aimrec/camera_stream.cpp
  src/sdv/aimrec/recording.h
  src/sdv/aimrec/recording.cpp
)
target_include_directories(sdv_aimrec PUBLIC src)
target_link_libraries(sdv_aimrec PUBLIC Eigen3::Eigen Sophus::Sophus opencv_core
  PRIVATE opencv_imgcodecs opencv_imgproc spdlog::spdlog yaml-cpp::yaml-cpp libzip::zip
  FFmpeg::avcodec FFmpeg::avutil FFmpeg::swscale)

add_library(sdv STATIC
  src/sdv/camera.h
  src/sdv/image_pyramid.h
  src/sdv/image_pyramid.cpp
  src/sdv/immature_point.h
  src/sdv/immature_point.cpp
  src/sdv/keyframe_features.h
  src/sdv/keyframe_features.cpp
  src/sdv/loop_detector.h
  src/sdv/loop_detector.cpp
  src/sdv/mono_initializer.h
  src/sdv/mono_initializer.cpp
  src/sdv/map_point.h
  src/sdv/odometry.h
  src/sdv/odometry.cpp
  src/sdv/brightness_fit.h
  src/sdv/brightness_fit.cpp
  src/sdv/semi_dense_mapper.h
  src/sdv/semi_dense_mapper.cpp
  src/sdv/parallel.h
  src/sdv/global_ba.h
  src/sdv/global_ba.cpp
  src/sdv/pose_graph.h
  src/sdv/pose_graph.cpp
  src/sdv/place_database.h
  src/sdv/place_database.cpp
  src/sdv/photometric_ba.h
  src/sdv/photometric_ba.cpp
  src/sdv/photometric_ba_cost.h
  src/sdv/photometric_ba_cost.cpp
  src/sdv/photometric.h
  src/sdv/photometric.cpp
  src/sdv/point_filter.h
  src/sdv/point_filter.cpp
  src/sdv/point_merge.h
  src/sdv/point_merge.cpp
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
  src/sdv/io/colmap_export.h
  src/sdv/io/colmap_export.cpp
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
  opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_videoio opencv_features2d yaml-cpp::yaml-cpp Ceres::ceres
  PRIVATE fbow::fbow sdv_aimrec)
# libstdc++ implements std::execution::par on top of oneTBB.
if(NOT MSVC)
  find_package(TBB CONFIG REQUIRED)
  target_link_libraries(sdv PUBLIC TBB::tbb)
endif()

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
target_link_libraries(rig_from_sensorconfig PRIVATE sdv sdv_aimrec Boost::program_options)

add_executable(aimrec_extract apps/aimrec_extract.cpp)
target_link_libraries(aimrec_extract PRIVATE sdv_aimrec spdlog::spdlog opencv_imgcodecs Boost::program_options)

add_executable(estimate_sync apps/estimate_sync.cpp)
target_link_libraries(estimate_sync PRIVATE sdv Boost::program_options)

add_executable(run_vo apps/run_vo.cpp)
target_link_libraries(run_vo PRIVATE sdv Boost::program_options)

add_executable(render_cloud apps/render_cloud.cpp)
target_link_libraries(render_cloud PRIVATE sdv Boost::program_options)

add_executable(train_vocabulary apps/train_vocabulary.cpp)
target_link_libraries(train_vocabulary PRIVATE sdv Boost::program_options)

add_executable(place_recall apps/place_recall.cpp)
target_link_libraries(place_recall PRIVATE sdv Boost::program_options)

add_executable(detect_loops apps/detect_loops.cpp)
target_link_libraries(detect_loops PRIVATE sdv Boost::program_options)

add_executable(close_loops apps/close_loops.cpp)
target_link_libraries(close_loops PRIVATE sdv Boost::program_options)

add_executable(export_colmap apps/export_colmap.cpp)
target_link_libraries(export_colmap PRIVATE sdv Boost::program_options)

enable_testing()
add_executable(sdv_tests
  tests/synthetic_scene.h
  tests/test_aimrec.cpp
  tests/test_camera.cpp
  tests/test_colmap.cpp
  tests/test_camera_config.cpp
  tests/test_immature_point.cpp
  tests/test_keyframe_features.cpp
  tests/test_loop_detector.cpp
  tests/test_pose_graph.cpp
  tests/test_global_ba.cpp
  tests/test_photometric_ba.cpp
  tests/test_photometric_ba_cost.cpp
  tests/test_place_database.cpp
  tests/test_mono_initializer.cpp
  tests/test_odometry.cpp
  tests/test_photometric.cpp
  tests/test_ply.cpp
  tests/test_point_filter.cpp
  tests/test_point_merge.cpp
  tests/test_point_selector.cpp
  tests/test_pyramid.cpp
  tests/test_brightness_fit.cpp
  tests/test_semi_dense_mapper.cpp
  tests/test_tracker.cpp
  tests/test_trajectory.cpp
  tests/test_window_optimizer.cpp
)
target_include_directories(sdv_tests PRIVATE tests)
target_link_libraries(sdv_tests PRIVATE sdv sdv_aimrec libzip::zip GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(sdv_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60)
