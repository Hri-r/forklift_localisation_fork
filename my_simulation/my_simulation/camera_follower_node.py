#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Quaternion
from rclpy.qos import qos_profile_sensor_data
from tf_transformations import euler_from_quaternion
import math

class CameraFollower(Node):
    def __init__(self):
        super().__init__('camera_follower')
        
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile_sensor_data
        )
        self.cli = self.create_client(SetEntityState, '/set_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /set_entity_state service...')
        
        self.latest_odom = None
        self.timer = self.create_timer(0.001, self.update_camera_pose)  # 10 Hz

    def odom_callback(self, msg):
        self.latest_odom = msg

    def update_camera_pose(self):
        if self.latest_odom is None:
            return

        msg = self.latest_odom
        robot_pos = msg.pose.pose.position
        robot_ori = msg.pose.pose.orientation

        # Extract yaw from quaternion
        _, _, yaw = euler_from_quaternion([
            robot_ori.x, robot_ori.y, robot_ori.z, robot_ori.w
        ])

        # Compute camera pose
        camera_pose = Pose()

        off_length = 0.055

        offset_z = 0.105         # height above robot

        camera_pose.position.x = robot_pos.x + off_length*math.cos(yaw)
        camera_pose.position.y = robot_pos.y + off_length*math.sin(yaw)
        camera_pose.position.z = robot_pos.z + offset_z

        # Match orientation
        camera_pose.orientation = self.yaw_to_quaternion(yaw)

        state = EntityState()
        state.name = 'camera'
        state.pose = camera_pose

        req = SetEntityState.Request()
        req.state = state
        self.cli.call_async(req)

    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

def main(args=None):
    rclpy.init(args=args)
    node = CameraFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()