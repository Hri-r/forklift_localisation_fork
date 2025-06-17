import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import time
from collections import deque

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # State: [x, y, theta, v, ax_bias]
        self.x = np.zeros((5, 1))
        self.P = np.eye(5) * 0.01

        self.Q = np.diag([0.01, 0.01, 0.01, 0.001, 0.01])
        self.R = np.diag([0.1, 0.1, 0.01])  # Measurement noise for marker pose

        self.last_time = self.get_clock().now()
        
        self.bias_estimates = []
        self.plotacc = 0.0

        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)
        self.create_subscription(Imu, '/calibrated_imu', self.imu_callback, 10)
        self.create_subscription(PoseStamped, '/robot_pose', self.marker_callback, 10)
        self.omega = 0.0
        self.a_true = []
        self.a_means_ = 0.0
        self.alpha =  0.9
        self.last_a = 0.0
        self.a = 0.0
        self.last_pos = np.zeros((2, 1))

        # self.timer = self.create_timer(0.001, self.ekf_predict)
        
        self.initializing = True

        self.bias_history = []
        self.raw_acc = []
        self.raw_biased = []
        self.time_history = []
        self.plot_start_time = time.time()


    def imu_callback(self, msg):
        acc = msg.linear_acceleration.x
        ader = acc - self.last_a

        self.plotacc = acc
        # self.a_means = acc

        if abs(ader) > 0.06:
             self.alpha = 0.0
        elif abs(ader) > 0.02:
            self.alpha = 0.9
        else:
            self.alpha = 0.99

        self.a_means_ = self.alpha * self.a_means_ + (1 - self.alpha) * acc
        

        self.last_a = self.a_means_

        # self.medarr.append(acc)
        # if len(self.medarr) == self.medarr.maxlen:
        #     self.a_means_ = np.median(self.medarr)

        current_time = time.time() - self.plot_start_time
        self.time_history.append(current_time)
        self.raw_acc.append(self.a_means_)
        self.bias_history.append(self.x[4,0])
        self.a_true.append(self.a)
        pointoone = 0.01 * np.ones((len(self.time_history),))

        if(current_time > 30 and current_time < 31):
            fig, ax = plt.subplots()
            ax.plot(self.time_history[-10000:], self.raw_acc[-10000:], label='filtered acc (x)', lw=0.5, color='red')
            ax.plot(self.time_history[-10000:], self.bias_history[-10000:], label='bias (x)', lw=0.5, color='blue')
            ax.plot(self.time_history[-10000:], pointoone[-10000:], label='vel_0.01 (x)', lw=0.5, color='black')
            ax.plot(self.time_history[-10000:], self.a_true[-10000:], label='acc true (x)', lw=0.5, color='green')
            ax.set_title("Raw IMU Acceleration (X-axis)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Acceleration (m/s²)")
            ax.legend()
            fig.tight_layout()
            fig.savefig("raw_acc_plot.png")
            self.get_logger().info("Saved raw acceleration plot.")

        self.ekf_predict()

        # delta = acc - self.a_means_
        # if abs(delta) > 0.03:
        #     self.a_means_ += 0.01 * np.sign(delta)
        # else:
        #     self.a_means_ += 0.001 * delta


        if(self.initializing):
                self.bias_estimates.append(acc)
                if len(self.bias_estimates) >= 500:
                    self.x[4][0] = np.mean(self.bias_estimates)
                    self.initializing = False
                    self.get_logger().info(f'Bias initialized: {self.x[4][0]}')
                    self.plot_start_time = time.time()
                self.a_means_ = 0

        self.omega = msg.angular_velocity.z

        if not self.initializing and abs(self.x[3,0])<0.005 and abs(self.a_means_) < 0.025:
        # EKF update step using acceleration to refine bias
            H = np.zeros((1, 5))
            H[0, 4] = 1  # We observe bias directly through measured acc
            v = self.x[3, 0]
            bias = self.x[4, 0]
            z = np.array([[self.a_means_]])
            y = z - H @ self.x
            R_imu = np.array([[0.008]])  # You can tune this

            self.P[4, :] = 0.0
            self.P[:, 4] = 0.0
            self.P[4, 4] = 0.01  # Small variance for bias

            S = H @ self.P @ H.T + R_imu

            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y

            if self.x[4,0] > 0.5:
                self.get_logger().info(f"{self.P[4,4]}, {self.x[4,0]}")

            if(self.a == 0.0):
                self.x[3, 0] = v

            self.P = (np.eye(5) - K @ H) @ self.P

    def marker_callback(self, msg):
        z = np.zeros((3, 1))

        alpha_pos = 0.9

        msg.pose.position.x = alpha_pos * msg.pose.position.x + (1 - alpha_pos) * self.last_pos[0, 0]
        msg.pose.position.y = alpha_pos * msg.pose.position.y + (1 - alpha_pos) * self.last_pos[1, 0]

        self.last_pos[0, 0] = msg.pose.position.x
        self.last_pos[1, 0] = msg.pose.position.y

        z[0, 0] = msg.pose.position.x
        z[1, 0] = msg.pose.position.y

        v = self.x[3,0]

        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        yaw = 2 * np.arctan2(qz, qw)
        z[2, 0] = yaw

        H = np.zeros((3, 5))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # theta

        y = z - H @ self.x
        y[2, 0] = self.normalize_angle(y[2, 0])

        error_norm = np.linalg.norm(y[:2])
        if error_norm > 0.1 and abs(self.x[3, 0]) < 0.05:
            self.x[0, 0] = z[0, 0]
            self.x[1, 0] = z[1, 0]
            self.x[2, 0] = z[2, 0]
            self.P[:3, :3] = np.diag([0.01, 0.01, 0.01])
            return

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        if(self.a == 0.0):
            self.x[3, 0] = v
        self.P = (np.eye(5) - K @ H) @ self.P

    def ekf_predict(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0 or dt > 1:
            return
        alpha_ = 0.8
        x, y, theta, v, b = self.x.flatten()
        a = self.a_means_ - b
        a = np.clip(a, -0.5, 0.5)
        if abs(a) < 0.025 :
            a = 0.0
        else:
            self.get_logger().info(f"acc = {a}, bias = {b}, a_means_ = {self.a_means_}")
        self.a = a
        # else:
        #      self.get_logger().info(f"acc = {a}")

        self.raw_biased.append(self.a)

        theta_new = theta + self.omega * dt
        v_new = v + a * dt
        x_new = x + v_new * np.cos(theta_new) * dt
        y_new = y + v_new * np.sin(theta_new) * dt

        self.x = np.array([[x_new], [y_new], [self.normalize_angle(theta_new)], [v_new], [b]])

        # self.get_logger().info(f"\n\n State: {self.x.flatten()} \n\n acc = {a}")

        F = np.eye(5)
        F[0, 2] = -v_new * np.sin(theta_new) * dt
        F[0, 3] = np.cos(theta_new) * dt
        F[1, 2] = v_new * np.cos(theta_new) * dt
        F[1, 3] = np.sin(theta_new) * dt
        F[3, 4] = -dt

        self.P = F @ self.P @ F.T + self.Q

        self.P[4, :] = 0.0
        self.P[:, 4] = 0.0
        self.P[4, 4] = 0.01  # Small variance for bias

        self.publish_pose()

    def publish_pose(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(self.x[0])
        msg.pose.position.y = float(self.x[1])
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = float(np.sin(self.x[2] / 2.0))
        msg.pose.orientation.w = float(np.cos(self.x[2] / 2.0))
        self.pose_pub.publish(msg)

    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
	main()