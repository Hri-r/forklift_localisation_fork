import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import tf_transformations
import transforms3d
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import yaml

class MarkerDetector(Node):
    def __init__(self):
        super().__init__('marker_detector')
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/marker_pose',
            self.marker_pose_callback,
            10
        )
        self.br = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)

        
        # Static transform from camera to robot (camera in front of robot)
        self.T_cam_to_robot = np.eye(4)
        self.T_cam_to_robot[0, 3] = 0.055  # 10 cm behind camera, adjust as needed
        self.T_cam_to_robot[2, 3] = -0.105


    def marker_pose_callback(self, msg):
    # Camera pose in world frame
        cam_pos = np.array([msg.pose.position.x,
                            msg.pose.position.y,
                            msg.pose.position.z])
        cam_quat = np.array([msg.pose.orientation.w,
                            msg.pose.orientation.x,
                            msg.pose.orientation.y,
                            msg.pose.orientation.z])  # Note order: [w, x, y, z]

        # Build T_w_c (camera pose in world)
        R_w_c = transforms3d.quaternions.quat2mat(cam_quat)
        T_w_c = np.eye(4)
        T_w_c[:3, :3] = R_w_c
        T_w_c[:3, 3] = cam_pos

        # Robot pose in world: T_w_r = T_w_c @ T_cam_to_robot
        T_w_r = T_w_c @ self.T_cam_to_robot

        robot_pos = T_w_r[:3, 3]
        rot_mat = T_w_r[:3, :3]
        quat = transforms3d.quaternions.mat2quat(rot_mat)  # [w, x, y, z]

        # Publish robot pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = float(robot_pos[0])
        pose_msg.pose.position.y = float(robot_pos[1])
        pose_msg.pose.position.z = float(robot_pos[2])
        pose_msg.pose.orientation.x = quat[1]
        pose_msg.pose.orientation.y = quat[2]
        pose_msg.pose.orientation.z = quat[3]
        pose_msg.pose.orientation.w = quat[0]
        self.pose_pub.publish(pose_msg)

        # Broadcast TF transform
        t_robot = TransformStamped()
        t_robot.header.stamp = self.get_clock().now().to_msg()
        t_robot.header.frame_id = 'world'
        t_robot.child_frame_id = 'base_link'
        t_robot.transform.translation.x = float(robot_pos[0])
        t_robot.transform.translation.y = float(robot_pos[1])
        t_robot.transform.translation.z = float(robot_pos[2])
        t_robot.transform.rotation.x = quat[1]
        t_robot.transform.rotation.y = quat[2]
        t_robot.transform.rotation.z = quat[3]
        t_robot.transform.rotation.w = quat[0]
        self.br.sendTransform(t_robot)

def main(args=None):
    rclpy.init(args=args)
    node = MarkerDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

