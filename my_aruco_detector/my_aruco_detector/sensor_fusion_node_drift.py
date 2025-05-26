import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
import numpy as np
import transforms3d.euler

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Pose state: x, y, theta
        self.state = np.array([0.0, 0.0, 0.0])
        self.last_time = self.get_clock().now()

        # Motion estimates
        self.ax = 0.0
        self.omega = 0.0
        self.v = 0.0  # velocity estimate

        # Fusion weight (0=marker only, 1=IMU only)
        self.alpha = 0
        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)

        # Subscriptions
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(PoseStamped, '/robot_pose', self.marker_callback, 10)

        # Timer for prediction loop
        self.timer = self.create_timer(0.05, self.update_state)  # 20 Hz

    def imu_callback(self, msg):
        self.ax = msg.linear_acceleration.x
        self.omega = msg.angular_velocity.z

    def update_state(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0 or dt > 1:
            return  # Ignore bad timing

        x, y, theta = self.state

        # Update velocity and state
        self.v += self.ax * dt
        dx = self.v * np.cos(theta) * dt
        dy = self.v * np.sin(theta) * dt
        dtheta = self.omega * dt

        self.state += np.array([dx, dy, dtheta])
        self.publish_pose()

    def marker_callback(self, msg):
        # Convert quaternion to yaw (theta)
        q = msg.pose.orientation
        quat = [q.w, q.x, q.y, q.z]
        _, _, measured_theta = transforms3d.euler.quat2euler(quat, axes='sxyz')

        measured_x = msg.pose.position.x
        measured_y = msg.pose.position.y
        measured_pose = np.array([measured_x, measured_y, measured_theta])

        # Complementary filter update
        self.state = self.alpha * self.state + (1 - self.alpha) * measured_pose

        self.get_logger().info(f"[FUSION] Corrected pose → x: {self.state[0]:.2f}, y: {self.state[1]:.2f}, yaw: {self.state[2]:.2f}")

    def publish_pose(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = float(self.state[0])
        pose_msg.pose.position.y = float(self.state[1])
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.z = np.sin(self.state[2] / 2)
        pose_msg.pose.orientation.w = np.cos(self.state[2] / 2)
        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
