from setuptools import find_packages, setup

package_name = 'my_aruco_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name + '/launch', ['launch/localization_bayes.launch.py']),
        ('share/' + package_name + '/launch', ['launch/localization_kalman.launch.py']),
        ('share/' + package_name + '/launch', ['launch/localization_ekf.launch.py']),
        ('share/' + package_name + '/launch', ['launch/localization_kf.launch.py']),
        ('share/' + package_name + '/launch', ['launch/localisation_bayes_prefuckup.launch.py']),
        ('share/' + package_name + '/launch', ['launch/trial.launch.py']),
        ('share/' + package_name + '/' + package_name, ['my_aruco_detector/aruco_marker_map.yaml']),
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='monisha',
    maintainer_email='monisha@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'markwe_pose_detect = my_aruco_detector.markwe_pose_detect:main',
        	'markwe_pose_localise = my_aruco_detector.markwe_pose_localise:main',
        	'marker_pose_localise = my_aruco_detector.marker_pose_localise:main',
        	'marker_pose_localise_new = my_aruco_detector.marker_pose_localise_new:main',
            	'sensor_fusion_node = my_aruco_detector.sensor_fusion_node:main',
            	'sensor_fusion_node_drift = my_aruco_detector.sensor_fusion_node_drift:main',
            	'sensor_fusion_node_fixed = my_aruco_detector.sensor_fusion_node_fixed:main',
            	'simple_kalman_fusion = my_aruco_detector.simple_kalman_fusion:main',
            	'forklift_display= my_aruco_detector.forklift_display:main' ,
            	'new_forklift_display=my_aruco_detector.new_forklift_display:main',
            	'imu_calibration_node=my_aruco_detector.imu_calibration_node:main',
            	'full_sensor_fusion_node=my_aruco_detector.full_sensor_fusion_node:main',
            	'full_sensor_fusion_fixing=my_aruco_detector.full_sensor_fusion_fixing:main',
            	'pf_full_sensor_fusion=my_aruco_detector.pf_full_sensor_fusion:main',
            	'real_ekf=my_aruco_detector.real_ekf:main',
            	'kf_full_sensor_fusion_m=my_aruco_detector.kf_full_sensor_fusion_m:main',
            	'kf_full_sensor_fusion=my_aruco_detector.kf_full_sensor_fusion:main',
        ],
    },
)
