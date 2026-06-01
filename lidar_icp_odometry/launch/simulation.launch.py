import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg        = get_package_share_directory('lidar_icp_odometry')
    params     = os.path.join(pkg, 'params.yaml')
    rviz_cfg   = os.path.join(pkg, 'slam_viz.rviz')
    world_file = os.path.join(pkg, 'worlds', 'icp_world.sdf')
    model_sdf  = os.path.join(pkg, 'models', 'lidar_car', 'model.sdf')

    sim_time = {'use_sim_time': True}

    # -------------------------------------------------------------------------
    # Launch argument: slam_mode
    #   false (default) → run icp_odometry_node  (§8.2 raw ICP odometry only)
    #   true            → run slam_node           (§8.4 full SLAM pipeline)
    # -------------------------------------------------------------------------
    slam_mode_arg = DeclareLaunchArgument(
        'slam_mode',
        default_value='false',
        description=(
            'false = raw ICP odometry (icp_odometry_node, §8.2); '
            'true  = full SLAM with loop closure + PGO (slam_node, §8.4)'))

    slam_mode = LaunchConfiguration('slam_mode')

    # ---- 1. Gz Fortress simulator --------------------------------------------
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-r {world_file}',
            'on_exit_shutdown': 'true',
        }.items())

    # ---- 2. Spawn car model --------------------------------------------------
    spawn_car = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'lidar_car',
            '-file', model_sdf,
            '-x', '0', '-y', '0', '-z', '0.0',
        ],
        output='screen')

    # ---- 3. ROS-Gz bridge ----------------------------------------------------
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/world/icp_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        output='screen')

    # ---- 4. Static TFs ----------------------------------------------------------
    # world → odom (coincide at t=0)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom'])

    # base_link → lidar sensor frame (Ignition scoped name: lidar_car/base_link/lidar)
    # Needed so RViz can transform /lidar/points (frame_id=lidar_car/base_link/lidar)
    # into the odom fixed frame.  Pose from model.sdf: 0 0 0.8 above base_link.
    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0.8', '0', '0', '0',
                   'base_link', 'lidar_car/base_link/lidar'])

    # ---- 5. Ground truth publisher -------------------------------------------
    ground_truth = Node(
        package='lidar_icp_odometry',
        executable='gz_ground_truth.py',
        parameters=[sim_time],
        output='screen')

    # ---- 6. Circle trajectory generator -------------------------------------
    figure8 = Node(
        package='lidar_icp_odometry',
        executable='figure8_node',
        parameters=[params, sim_time],
        output='screen')

    # ---- 7a. Raw ICP odometry node (slam_mode=false) ------------------------
    icp_node = Node(
        package='lidar_icp_odometry',
        executable='icp_odometry_node',
        parameters=[params, sim_time],
        output='screen',
        condition=UnlessCondition(slam_mode))

    # ---- 7b. Full SLAM node (slam_mode=true) ---------------------------------
    slam_node = Node(
        package='lidar_icp_odometry',
        executable='slam_node',
        parameters=[params, sim_time],
        output='screen',
        condition=IfCondition(slam_mode))

    # ---- 8. RViz -------------------------------------------------------------
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        parameters=[sim_time],
        output='screen')

    # Delay all nodes 5 s to give Gazebo time to fully start
    delayed_nodes = TimerAction(
        period=5.0,
        actions=[spawn_car, bridge, static_tf, static_tf_lidar, ground_truth,
                 figure8, icp_node, slam_node, rviz])

    return LaunchDescription([slam_mode_arg, gz_sim, delayed_nodes])
