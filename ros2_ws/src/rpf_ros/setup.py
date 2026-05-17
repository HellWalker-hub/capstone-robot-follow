from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rpf_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rpf',
    maintainer_email='iampersonal13@gmail.com',
    description='ROS2 nodes for the occlusion-resistant person follower pipeline.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'perception_node = rpf_ros.perception_node:main',
            'ukf_node        = rpf_ros.ukf_node:main',
            'controller_node = rpf_ros.controller_node:main',
        ],
    },
)
