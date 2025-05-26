#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import transforms3d
import yaml
from ament_index_python.packages import get_package_share_directory
import os


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.got_camera_info = False

        # Subscriptions
        self.create_subscription(CameraInfo, '/my_camera/camera/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Image, '/my_camera/camera/image_raw', self.image_callback, 10)

        # Publisher
        self.pose_pub = self.create_publisher(PoseStamped, '/marker_pose', 10)

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # Marker size (side length in meters)
        self.marker_length = 0.125

        # Load marker poses from YAML
        pkg_path = get_package_share_directory('my_simulation')
        yaml_path = os.path.join(pkg_path, 'config', 'marker_poses.yaml')
        self.marker_world_poses = self.load_marker_world_poses(yaml_path)

        self.get_logger().info(f"Using OpenCV version: {cv2.__version__}")

    def load_marker_world_poses(self, filepath):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        poses = {}
        for marker_id, info in data['markers'].items():
            pos = np.array(info['position'])  # [x, y, z]
            quat = np.array(info['orientation'])  # [x, y, z, w]
            quat = np.roll(quat, 1)
            poses[int(marker_id)] = (pos, quat)
        return poses

    def camera_info_callback(self, msg):
        if not self.got_camera_info:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d).reshape((1, -1))
            self.got_camera_info = True
            self.get_logger().info("Camera info received.")

    def image_callback(self, msg):
        if not self.got_camera_info:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)

            if ids is not None:
                for i, corner in enumerate(corners):
                    marker_id = int(ids[i][0])
                    if marker_id not in self.marker_world_poses:
                        self.get_logger().warn(f"Marker ID {marker_id} not found in world YAML.")
                        continue

                    # Estimate pose of the marker w.r.t. camera frame
                    marker_half = self.marker_length / 2.0
                    obj_points = np.array([
                        [-marker_half,  marker_half, 0],
                        [ marker_half,  marker_half, 0],
                        [ marker_half, -marker_half, 0],
                        [-marker_half, -marker_half, 0]
                    ], dtype=np.float32)

                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        corner[0],
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    self.get_logger().info(f"tvec: {tvec.flatten()} rvec: {rvec.flatten()}")

                    if not success:
                        self.get_logger().warn("Pose estimation failed.")
                        continue

                    # Pose estimation
                    R_mc, _ = cv2.Rodrigues(rvec)
                    T_mc = np.eye(4)
                    T_mc[:3, :3] = R_mc
                    T_mc[:3, 3] = tvec.flatten()

                    # OpenCV to Gazebo transform
                    T_cv_to_gazebo = np.array([
                        [ 0,  0,  -1, 0],
                        [ 1,  0,  0, 0],
                        [ 0, -1,  0, 0],
                        [ 0,  0,  0, 1]
                    ])

                    # Get marker's world pose
                    marker_pos, marker_quat = self.marker_world_poses[marker_id]
                    R_wm = transforms3d.quaternions.quat2mat(marker_quat)
                    T_wm = np.eye(4)
                    T_wm[:3, :3] = R_wm
                    T_wm[:3, 3] = marker_pos

                    # Camera pose in world frame
                    T_wc = T_wm @ T_cv_to_gazebo @ np.linalg.inv(T_mc)

                    # Extract position and quaternion
                    camera_pos = T_wc[:3, 3]
                    camera_rot = T_wc[:3, :3]
                    camera_quat = transforms3d.quaternions.mat2quat(camera_rot)

                    axes_length = 0.2

                    # Publish pose
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = 'world'
                    pose_msg.pose.position.x = float(camera_pos[0])
                    pose_msg.pose.position.y = float(camera_pos[1])
                    pose_msg.pose.position.z = float(camera_pos[2])
                    pose_msg.pose.orientation.x = float(camera_quat[1])
                    pose_msg.pose.orientation.y = float(camera_quat[2])
                    pose_msg.pose.orientation.z = float(camera_quat[3])
                    pose_msg.pose.orientation.w = float(camera_quat[0])
                    self.pose_pub.publish(pose_msg)

                    # Visual feedback
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length)

            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Detection error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
