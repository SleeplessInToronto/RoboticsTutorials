import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg           = get_package_share_directory('ekf_se3_localization')
    params_file   = os.path.join(pkg, 'params.yaml')
    world_file    = os.path.join(pkg, 'worlds', 'beacons_world.sdf')
    model_sdf     = os.path.join(pkg, 'models', 'ekf_car', 'model.sdf')
    rviz_config   = os.path.join(pkg, 'ekf_rviz.rviz')

    # ---- 1. Gz Fortress simulator ----
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-r {world_file}',
            'on_exit_shutdown': 'true',
        }.items())

    # ---- 2. Spawn car model ----
    spawn_car = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'ekf_car',
            '-file', model_sdf,
            '-x', '0', '-y', '0', '-z', '0.0',
        ],
        output='screen')

    # ---- 3. ROS-Gz topic bridge ----
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/lidar/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/world/beacons_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        output='screen')

    # ---- 4. Static TF: world → map (they coincide in this sim) ----
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'map'])

    sim_time = {'use_sim_time': True}

    # ---- 5. Ground truth publisher (Gz world pose → /ground_truth/odom) ----
    ground_truth = Node(
        package='ekf_se3_localization',
        executable='gz_ground_truth.py',
        parameters=[sim_time],
        output='screen')

    # ---- 6. Noisy odometry (simulates real-world wheel-encoder drift) ----
    noisy_odom = Node(
        package='ekf_se3_localization',
        executable='noisy_odom_node.py',
        parameters=[sim_time],
        output='screen')

    # ---- 7. Circle path generator ----
    figure8 = Node(
        package='ekf_se3_localization',
        executable='figure8_node',
        parameters=[params_file, sim_time],
        output='screen')

    # ---- 8. LiDAR beacon detector ----
    detector = Node(
        package='ekf_se3_localization',
        executable='lidar_beacon_detector_node',
        parameters=[params_file, sim_time],
        output='screen')

    # ---- 9. EKF node ----
    ekf = Node(
        package='ekf_se3_localization',
        executable='ekf_node',
        parameters=[params_file, sim_time],
        output='screen')

    # ---- 10. RViz ----
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[sim_time],
        output='screen')

    delayed_nodes = TimerAction(
        period=5.0,
        actions=[spawn_car, bridge, static_tf, ground_truth,
                 noisy_odom, figure8, detector, ekf, rviz])

    return LaunchDescription([gz_sim, delayed_nodes])
