import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import yaml

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')

        self.calibrated_imu_pub = self.create_publisher(Imu, '/calibrated_imu', 10)

        # Load calibration from YAML
        self.load_calibration_from_yaml('/home/obi/forklift/forklift_localisation/my_aruco_detector/my_aruco_detector/calibration.yaml')

        # Subscribe to raw IMU
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)

    def load_calibration_from_yaml(self, yaml_file):
        with open(yaml_file, 'r') as f:
            calib = yaml.safe_load(f)

        self.acceleration_offset = calib['acceleration_offset']
        self.angular_velocity_offset = calib['angular_velocity_offset']

        self.get_logger().info(f"Loaded calibration offsets from {yaml_file}")

    def imu_callback(self, msg):
        calibrated_imu = Imu()
        calibrated_imu.header = msg.header

        calibrated_imu.linear_acceleration.x = msg.linear_acceleration.x - self.acceleration_offset['x']
        calibrated_imu.linear_acceleration.y = msg.linear_acceleration.y - self.acceleration_offset['y']
        calibrated_imu.linear_acceleration.z = msg.linear_acceleration.z - self.acceleration_offset['z']

        calibrated_imu.angular_velocity.x = msg.angular_velocity.x - self.angular_velocity_offset['x']
        calibrated_imu.angular_velocity.y = msg.angular_velocity.y - self.angular_velocity_offset['y']
        calibrated_imu.angular_velocity.z = msg.angular_velocity.z - self.angular_velocity_offset['z']

        self.calibrated_imu_pub.publish(calibrated_imu)

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

