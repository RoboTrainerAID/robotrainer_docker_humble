from setuptools import setup
import os
from glob import glob


package_name = 'polar_oh1'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Wonse Jo',
    maintainer_email='wonsu0513@gmail.com',
    description='ROS 2 drivers for the Polar OH1+ and Polar H10 heart rate sensors.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'polar_oh1_node = polar_oh1.polar_oh1_node:main',
            'polar_h10_node = polar_oh1.polar_h10_node:main',
        ],
    },
)
