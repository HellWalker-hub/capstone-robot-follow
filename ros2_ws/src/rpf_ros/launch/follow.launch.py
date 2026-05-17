"""
RPF follow launch file — starts all three nodes on the Mac.

Usage:
    source /opt/ros/humble/setup.bash
    cd ros2_ws && colcon build --symlink-install && source install/setup.bash
    ros2 launch rpf_ros follow.launch.py

Optional overrides:
    ros2 launch rpf_ros follow.launch.py camera_index:=1 reid_threshold:=0.60
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Repo root — perception package lives here
RPF_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))


def generate_launch_description():
    return LaunchDescription([

        # ------------------------------------------------------------------
        # Make the perception package importable from any node in this launch
        # ------------------------------------------------------------------
        SetEnvironmentVariable(
            name='PYTHONPATH',
            value=RPF_ROOT + ':' + os.environ.get('PYTHONPATH', ''),
        ),

        # ------------------------------------------------------------------
        # Launch arguments
        # ------------------------------------------------------------------
        DeclareLaunchArgument('camera_index',   default_value='0'),
        DeclareLaunchArgument('reid_threshold', default_value='0.55'),
        DeclareLaunchArgument('reid_every_n',   default_value='3'),

        # ------------------------------------------------------------------
        # Perception node (Mac) — detector + tracker + ReID + state machine
        # ------------------------------------------------------------------
        Node(
            package='rpf_ros',
            executable='perception_node',
            name='rpf_perception',
            output='screen',
            parameters=[{
                'camera_index':   LaunchConfiguration('camera_index'),
                'reid_threshold': LaunchConfiguration('reid_threshold'),
                'reid_every_n':   LaunchConfiguration('reid_every_n'),
            }],
        ),

        # ------------------------------------------------------------------
        # UKF node (Mac) — metric distance/velocity from /rpf/tracks
        # ------------------------------------------------------------------
        Node(
            package='rpf_ros',
            executable='ukf_node',
            name='ukf_tracker',
            output='screen',
        ),

        # ------------------------------------------------------------------
        # Controller node (Mac) — /rpf/target + /tracked_persons → /cmd_vel
        # ------------------------------------------------------------------
        Node(
            package='rpf_ros',
            executable='controller_node',
            name='rpf_controller',
            output='screen',
        ),
    ])
