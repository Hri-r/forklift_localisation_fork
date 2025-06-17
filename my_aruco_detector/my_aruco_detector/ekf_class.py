import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
import numpy as np

class EKF:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros((4,1))
        self.P = np.eye(4) * 0.1
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1])
        self.R = np.diag([0.05, 0.05, 0.1])
    
    def predict(self, ax, omega):
        x, y, theta, vx = self.x.flatten()
        dt = self.dt

        vx += ax * dt
        theta += omega * dt
        x += vx * np.cos(theta) * dt
        y += vx * np.sin(theta) * dt

        self.x = np.array([[x], [y], [theta], [vx]])

        F = np.array([
            [1, 0, -vx * np.sin(theta) * dt, np.cos(theta) * dt],
            [0, 1, vx * np.cos(theta) * dt, np.sin(theta) * dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, z):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])
        
        y = z.reshape(3,1) - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def get_state(self):
        return self.x.flatten()