cmake_minimum_required(VERSION 3.8)
project(ekf_se3_localization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake   REQUIRED)
find_package(rclcpp        REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs      REQUIRED)
find_package(sensor_msgs   REQUIRED)
find_package(tf2           REQUIRED)
find_package(tf2_ros       REQUIRED)
find_package(tf2_eigen     REQUIRED)
find_package(Eigen3        REQUIRED)
find_package(Sophus        REQUIRED)

# ---------------------------------------------------------------------------
# Pure-math EKF library — no ROS headers, independently unit-testable
# ---------------------------------------------------------------------------
add_library(ekf_se3_lib STATIC src/ekf_se3.cpp)
target_include_directories(ekf_se3_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(ekf_se3_lib Eigen3::Eigen Sophus::Sophus)

# ---------------------------------------------------------------------------
# EKF localization node
# ---------------------------------------------------------------------------
add_executable(ekf_node src/ekf_node.cpp)
target_include_directories(ekf_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(ekf_node ekf_se3_lib)
ament_target_dependencies(ekf_node
  rclcpp geometry_msgs nav_msgs tf2 tf2_ros tf2_eigen)

# ---------------------------------------------------------------------------
# LiDAR beacon detector node
# ---------------------------------------------------------------------------
add_executable(lidar_beacon_detector_node src/lidar_beacon_detector_node.cpp)
target_include_directories(lidar_beacon_detector_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(lidar_beacon_detector_node Eigen3::Eigen Sophus::Sophus)
ament_target_dependencies(lidar_beacon_detector_node
  rclcpp geometry_msgs nav_msgs sensor_msgs)

# ---------------------------------------------------------------------------
# Figure-8 open-loop path generator
# ---------------------------------------------------------------------------
add_executable(figure8_node src/figure8_node.cpp)
ament_target_dependencies(figure8_node rclcpp geometry_msgs nav_msgs)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS ekf_se3_lib ekf_node lidar_beacon_detector_node figure8_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include/)

install(PROGRAMS scripts/noisy_odom_node.py scripts/gz_ground_truth.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config/ launch/
  DESTINATION share/${PROJECT_NAME}/)

install(DIRECTORY models worlds
  DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
