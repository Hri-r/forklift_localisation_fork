from setuptools import find_packages, setup

package_name = 'my_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/sim_launch.launch.py']),
        ('share/' + package_name + '/launch', ['my_gazebo.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='monisha',
    maintainer_email='monisha@todo.todo',
    description='Camera follower and ArUco detector node for simulation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_follower_node = my_simulation.camera_follower_node:main',
            'aruco_live_detector = my_simulation.aruco_live_detector:main',
            'controller = my_simulation.controller:main',
        ],
    },
)

