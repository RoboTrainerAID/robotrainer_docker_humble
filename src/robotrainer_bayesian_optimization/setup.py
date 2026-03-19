from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robotrainer_bayesian_optimization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='docker',
    maintainer_email='docker@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robotrainer_bayesian_optimization = robotrainer_bayesian_optimization.bayesian_optimization_node:main',
            'user_data_test_node = robotrainer_bayesian_optimization.user_data_test_node:main', 
        ],
    },
)
