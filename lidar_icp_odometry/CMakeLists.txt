cmake_minimum_required(VERSION 3.8)
project(lidar_icp_odometry)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(ament_cmake   REQUIRED)
find_package(rclcpp        REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs      REQUIRED)
find_package(sensor_msgs   REQUIRED)
find_package(tf2           REQUIRED)
find_package(tf2_ros       REQUIRED)
find_package(tf2_eigen     REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(Eigen3            REQUIRED)
find_package(Sophus            REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs           REQUIRED)

# ---------------------------------------------------------------------------
# icp_solver_lib — pure-math ICP (no rclcpp, no PCL)
#   nanoflann is vendored as a single header in include/third_party/
# ---------------------------------------------------------------------------
add_library(icp_solver_lib STATIC src/icp_solver.cpp)
target_include_directories(icp_solver_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(icp_solver_lib
  Eigen3::Eigen
  Sophus::Sophus)
ament_target_dependencies(icp_solver_lib sensor_msgs)   # PointCloud2ConstIterator

# ---------------------------------------------------------------------------
# slam_lib — Scan Context + Pose Graph Optimizer (no rclcpp, no PCL)
# ---------------------------------------------------------------------------
add_library(slam_lib STATIC
  src/scan_context.cpp
  src/pose_graph_optimizer.cpp)
target_include_directories(slam_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(slam_lib
  Eigen3::Eigen
  Sophus::Sophus
  icp_solver_lib)

# ---------------------------------------------------------------------------
# icp_odometry_node — raw ICP odometry (no loop closure)
# ---------------------------------------------------------------------------
add_executable(icp_odometry_node src/icp_odometry_node.cpp)
target_include_directories(icp_odometry_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(icp_odometry_node icp_solver_lib)
ament_target_dependencies(icp_odometry_node
  rclcpp geometry_msgs nav_msgs sensor_msgs tf2 tf2_ros tf2_eigen)

# ---------------------------------------------------------------------------
# slam_node — full LiDAR SLAM: ICP + Scan Context + Gauss-Newton PGO
# ---------------------------------------------------------------------------
add_executable(slam_node src/slam_node.cpp)
target_include_directories(slam_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(slam_node slam_lib icp_solver_lib)
ament_target_dependencies(slam_node
  rclcpp geometry_msgs nav_msgs sensor_msgs tf2 tf2_ros tf2_eigen
  visualization_msgs std_msgs)

# ---------------------------------------------------------------------------
# Circular trajectory generator (identical to ekf_se3_localization)
# ---------------------------------------------------------------------------
add_executable(figure8_node src/figure8_node.cpp)
ament_target_dependencies(figure8_node rclcpp geometry_msgs nav_msgs)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS
    icp_solver_lib
    slam_lib
    icp_odometry_node
    slam_node
    figure8_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include/)

install(DIRECTORY config/ launch/ worlds models
  DESTINATION share/${PROJECT_NAME}/)

install(PROGRAMS
  scripts/gz_ground_truth.py
  scripts/analyze_slam_bag.py
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
