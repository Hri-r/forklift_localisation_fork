import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import pygame
import math
import sys

WIDTH, HEIGHT = 600, 600
MAP_SIZE = 2.0  # meters
SCALE = WIDTH / MAP_SIZE  # pixels per meter

class ForkliftDisplay(Node):
    def __init__(self):
        super().__init__('forklift_display')

        self.forklift_pos = None
        self.forklift_yaw = 0.0
        self.goal_pos = None
        self.path_history = []

        self.landmarks = [
            [0.5, 0.5, "Pillar A"],
            [1.2, 0.4, "Box"],
            [1.6, 1.5, "Ramp"],
            [1.8, 0.7, "Dock"]
        ]

        self.create_subscription(PoseStamped, '/robot_pose', self.robot_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Forklift Navigation Display")
        self.font = pygame.font.SysFont(None, 24)

    def robot_callback(self, msg):
        self.forklift_pos = [msg.pose.position.x, msg.pose.position.y]

        self.get_logger().info(f"Robot position: {self.forklift_pos}")

        if not self.path_history or self.forklift_pos != list(self.path_history[-1]):
            self.path_history.append(tuple(self.forklift_pos))

        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        self.forklift_yaw = -math.atan2(2.0 * qz * qw, 1.0 - 2.0 * qz * qz)

    def goal_callback(self, msg):
        self.goal_pos = [msg.pose.position.x, msg.pose.position.y]

    def world_to_screen(self, x, y):
        return int(WIDTH/2 + x * SCALE), int(HEIGHT/2- y * SCALE)

    def draw_grid(self, spacing=0.1):
        for x in range(int(MAP_SIZE / spacing) + 1):
            screen_x = int(x * spacing * SCALE)
            pygame.draw.line(self.screen, (50, 50, 50), (screen_x, 0), (screen_x, HEIGHT))
        for y in range(int(MAP_SIZE / spacing) + 1):
            screen_y = HEIGHT - int(y * spacing * SCALE)
            pygame.draw.line(self.screen, (50, 50, 50), (0, screen_y), (WIDTH, screen_y))

    def draw_landmarks(self):
        for x, y, label in self.landmarks:
            pos = self.world_to_screen(x, y)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 6)
            self.screen.blit(self.font.render(label, True, (200, 200, 200)), (pos[0] + 8, pos[1] - 8))

    def draw_arrow(self, start, angle, color=(255, 0, 0), size=20):
        end = (
            start[0] + size * math.cos(angle),
            start[1] + size * math.sin(angle)
        )
        pygame.draw.line(self.screen, color, start, end, 3)
        for offset in [math.pi / 6, -math.pi / 6]:
            tip = (
                end[0] - 10 * math.cos(angle - offset),
                end[1] - 10 * math.sin(angle - offset)
            )
            pygame.draw.line(self.screen, color, end, tip, 3)

    def draw_arrow_between(self, start, end):
        pygame.draw.line(self.screen, (0, 255, 0), start, end, 4)
        dx, dy = end[0] - start[0], end[1] - start[1]
        angle = math.atan2(dy, dx)
        for offset in [math.pi / 6, -math.pi / 6]:
            tip_x = end[0] - 10 * math.cos(angle - offset)
            tip_y = end[1] - 10 * math.sin(angle - offset)
            pygame.draw.line(self.screen, (0, 255, 0), end, (tip_x, tip_y), 4)

    def draw_fov_cone(self, origin, angle, fov=math.pi / 3, length=40):
        left_angle = angle - fov / 2
        right_angle = angle + fov / 2

        left = (
            origin[0] + length * math.cos(left_angle),
            origin[1] + length * math.sin(left_angle)
        )
        right = (
            origin[0] + length * math.cos(right_angle),
            origin[1] + length * math.sin(right_angle)
        )

        pygame.draw.polygon(self.screen, (100, 100, 255, 100), [origin, left, right], 1)

    def draw_compass(self):
        compass_center = (50, 50)
        compass_radius = 30
        pygame.draw.circle(self.screen, (80, 80, 80), compass_center, compass_radius, 2)
        self.draw_arrow(compass_center, self.forklift_yaw, color=(255, 255, 0), size=compass_radius - 5)
        label = self.font.render("N", True, (255, 255, 255))
        self.screen.blit(label, (compass_center[0] - 8, compass_center[1] - compass_radius - 12))

    def draw_display(self):
        self.screen.fill((30, 30, 30))
        self.draw_grid(spacing=1.0)
        self.draw_landmarks()

        if self.forklift_pos:
            forklift_screen = self.world_to_screen(*self.forklift_pos)
            pygame.draw.circle(self.screen, (255, 0, 0), forklift_screen, 8)
            self.draw_arrow(forklift_screen, self.forklift_yaw)
            self.draw_fov_cone(forklift_screen, self.forklift_yaw)

            for pos in self.path_history:
                trail = self.world_to_screen(*pos)
                pygame.draw.circle(self.screen, (100, 100, 100), trail, 2)

        if self.goal_pos:
            goal_screen = self.world_to_screen(*self.goal_pos)
            pygame.draw.circle(self.screen, (0, 150, 255), goal_screen, 8)

            if self.forklift_pos:
                forklift_screen = self.world_to_screen(*self.forklift_pos)
                self.draw_arrow_between(forklift_screen, goal_screen)

        self.draw_compass()

        pygame.draw.rect(self.screen, (50, 50, 50), pygame.Rect(0, HEIGHT - 40, WIDTH, 40))
        pos_text = "Forklift: N/A" if not self.forklift_pos else f"Forklift: ({self.forklift_pos[0]:.2f}, {self.forklift_pos[1]:.2f})"
        goal_text = "Goal: N/A" if not self.goal_pos else f"Goal: ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f})"
        status = self.font.render(f"{pos_text} | {goal_text}", True, (255, 255, 0))
        self.screen.blit(status, (10, HEIGHT - 30))

        pygame.display.flip()

def main(args=None):
    rclpy.init(args=args)
    node = ForkliftDisplay()
    clock = pygame.time.Clock()

    try:
        while rclpy.ok():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            rclpy.spin_once(node, timeout_sec=0.01)
            node.draw_display()
            clock.tick(30)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        pygame.quit()

if __name__ == '__main__':
    main()

