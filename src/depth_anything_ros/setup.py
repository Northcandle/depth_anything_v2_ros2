from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'depth_anything_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['depth_anything_ros', 'depth_anything_ros.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 添加这行来包含 launch 文件
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
    'setuptools',
    'torch',
    'numpy',
    'opencv-python'
    ],
    zip_safe=True,
    maintainer='li',
    maintainer_email='li@todo.todo',
    description='Depth Anything V2 ROS2 integration',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'depth_anything_node = depth_anything_ros.depth_anything_node:main',
        ],
    },
)
