import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import tf_transformations
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import yaml

class MarkerDetector(Node):
    def __init__(self):
        super().__init__('marker_detector')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.listener_callback,
            10
        )
        self.pose_pub = self.create_publisher(PoseStamped, '/marker_pose', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.br = TransformBroadcaster(self)

        # Camera parameters
        self.camera_matrix = np.array([
            [920.7894, 0, 589.1690],
            [0, 919.1768, 541.4353],
            [0, 0, 1.0000]
        ], dtype=np.float32)
        self.dist_coeffs = np.array([[0.205245304646941], [-0.332544106955044], [0], [0], [0]], dtype=np.float32)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.parameters = cv2.aruco.DetectorParameters_create()

        # Load marker positions and goals
        yaml_path = '/home/monisha/turtlebot3_ws/src/my_aruco_detector/my_aruco_detector/aruco_marker_map.yaml'
        with open(yaml_path, 'r') as f:
            marker_map = yaml.safe_load(f)

        self.marker_world_positions = {}
        self.goal_markers = {}
        for marker_id, data in marker_map.items():
            if 'position' in data and 'orientation' in data:
                self.marker_world_positions[int(marker_id)] = {
                    'position': data['position'],
                    'orientation': data['orientation']
                }

            if 'goal' in data:
                self.goal_markers[int(marker_id)] = data['goal']

        self.marker_length = 0.03  # marker side length in meters

    def listener_callback(self, msg):
        try:
            # Decode the compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

                for i in range(len(ids)):
                    marker_id = int(ids[i][0])

                    # Set axis color and label color
                    is_goal = marker_id in self.goal_markers
                    axis_length = 0.02  # 2 cm
                    color = (0, 255, 0) if is_goal else (255, 0, 0)  # Green for goal, Blue for regular

                    # Draw the axis (OpenCV uses internal color logic, draw with default for all)
                    cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], axis_length)

                    # Draw the marker ID
                    c = corners[i][0].mean(axis=0).astype(int)
                    label = f"Goal ID: {marker_id}" if is_goal else f"ID: {marker_id}"
                    cv2.putText(frame, label, (c[0], c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


                    # First, if it is a goal marker
                    if marker_id in self.goal_markers:
                        goal_x, goal_y = self.goal_markers[marker_id]
                        goal_msg = PoseStamped()
                        goal_msg.header.stamp = self.get_clock().now().to_msg()
                        goal_msg.header.frame_id = 'map'
                        goal_msg.pose.position.x = goal_x
                        goal_msg.pose.position.y = goal_y
                        goal_msg.pose.position.z = 0.0
                        goal_msg.pose.orientation.w = 1.0  # no orientation for goal
                        self.goal_pub.publish(goal_msg)
                        continue  # DO NOT reset pose with goal markers

                    if marker_id not in self.marker_world_positions:
                        continue

                    # For pillar markers: do localization reset
                    tvec = tvecs[i].reshape((3, 1))
                    rvec = rvecs[i].reshape((3, 1))

                    R_m_c, _ = cv2.Rodrigues(rvec)
                    T_m_c = np.eye(4)
                    T_m_c[:3, :3] = R_m_c
                    T_m_c[:3, 3] = tvec[:, 0]

                    # Convert marker pose from camera frame to robot frame
                    # T_cam_to_robot = np.array([
                    #     [ 0,  0, 1, 0],  # x_robot = z_cam
                    #     [-1,  0, 0, 0],  # y_robot = -x_cam
                    #     [ 0, -1, 0, 0],  # z_robot = -y_cam
                    #     [ 0,  0, 0, 1]
                    # ])
                    
                    # T_m_r = T_cam_to_robot @ T_m_c

                    # T_r_m = np.linalg.inv(T_m_r)

                    T_c_m = np.linalg.inv(T_m_c)

                    marker_data = self.marker_world_positions[marker_id]
                    pos = marker_data['position']
                    quat = marker_data['orientation']  # [x, y, z, w]

                    T_w_m = tf_transformations.quaternion_matrix(quat)
                    T_w_m[:3, 3] = [pos[0], pos[1], pos[2]]  # include Z from YAML

                    # T_w_r = T_w_m @ T_r_m
                    T_w_c = T_w_m @ T_c_m
                    
                    # print("marker wrt camera", T_m_c)
                    # print("camera wrt marker", T_c_m)
                    # print("marker wrt world", T_w_m)
                    
                    # print("camera wrt world", T_w_c)
                    
                    cam_pos = T_w_c[:3, 3]
                    roll, pitch, yaw_robot = tf_transformations.euler_from_matrix(T_w_c)

                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = 'map'
                    pose_msg.pose.position.x = float(cam_pos[0])
                    pose_msg.pose.position.y = float(cam_pos[1])
                    pose_msg.pose.position.z = 0.0
                    pose_msg.pose.orientation.z = np.sin(yaw_robot / 2)
                    pose_msg.pose.orientation.w = np.cos(yaw_robot / 2)
                    self.pose_pub.publish(pose_msg)

                    # Broadcast robot transform
                    t_robot = TransformStamped()
                    t_robot.header.stamp = self.get_clock().now().to_msg()
                    t_robot.header.frame_id = 'map'
                    t_robot.child_frame_id = 'base_link'
                    t_robot.transform.translation.x = float(cam_pos[0])
                    t_robot.transform.translation.y = float(cam_pos[1])
                    t_robot.transform.translation.z = 0.0
                    quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw_robot)
                    t_robot.transform.rotation.x = quat[0]
                    t_robot.transform.rotation.y = quat[1]
                    t_robot.transform.rotation.z = quat[2]
                    t_robot.transform.rotation.w = quat[3]
                    self.br.sendTransform(t_robot)

            cv2.imshow("ArUco Marker Detection", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    marker_detector = MarkerDetector()
    rclpy.spin(marker_detector)
    marker_detector.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

