from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_size',
            default_value='small',
            description='Depth Anything model size: small/base/large'
        ),
        
        # RealSense节点 - 仅启用彩色摄像头，无remap
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense_camera',
            parameters=[{
                'camera_name': 'd435i',
                'enable_gyro': False,
                'enable_accel': False,
                'enable_infra': False,
                'enable_depth': False,
                'enable_color': True,
                'color.width': 640,
                'color.height': 480,
                'color_fps': 15.0,
                'infra1.enable_compressed': False,
                'infra2.enable_compressed': False,
                'color.enable_compressed': False,
                'unite_imu_method': 0,
                'initial_reset': True
            }]
        ),
        
        # Depth Anything节点，订阅realsense原始彩色话题
        Node(
            package='depth_anything_ros',
            executable='depth_anything_node',
            name='depth_anything',
            parameters=[{
                'model_size': LaunchConfiguration('model_size'),
                'input_topic': '/camera/realsense_camera/color/image_raw',
                'depth_topic': '/depth_map',
                'color_topic': '/depth_map_color'
            }],
            output='screen'
        ),
        
        # 伪彩色深度图可视化
        Node(
            package='image_view',
            executable='image_view',
            name='depth_color_view',
            parameters=[{'autosize': True}],
            remappings=[('image', '/depth_map_color')]
        ),
        
        # RGB图像可视化，订阅realsense彩色话题
        Node(
            package='image_view',
            executable='image_view',
            name='rgb_view',
            parameters=[{'autosize': True}],
            remappings=[('image', '/camera/realsense_camera/color/image_raw')]
        )
    ])
