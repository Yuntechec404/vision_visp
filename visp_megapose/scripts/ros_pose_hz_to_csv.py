#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import csv
from datetime import datetime

class PoseFrequencyLoggerROS2(Node):
    def __init__(self, topic, output_csv):
        super().__init__('pose_hz_logger')
        self.topic = topic
        self.output_csv = output_csv
        self.last_time = None
        self.hz = 0

        self.create_subscription(Pose, topic, self.callback, 10)
        self.start_logging()
        self.get_logger().info(f"Started logging {self.topic} frequency to {self.output_csv}")

    def callback(self, msg):
        current_time = self.get_clock().now().to_msg().sec + \
                       self.get_clock().now().to_msg().nanosec * 1e-9
        if self.last_time is not None:
            self.hz = 1.0 / (current_time - self.last_time)
            self.log_to_csv(self.hz)
        self.last_time = current_time

    def log_to_csv(self, hz):
        with open(self.output_csv, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([datetime.now().isoformat(), hz])

    def start_logging(self):
        with open(self.output_csv, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "Hz"])


def main(args=None):
    rclpy.init(args=args)
    topic = '/cube'
    output_csv = 'ros2_pose_hz.csv'
    pose_logger = PoseFrequencyLoggerROS2(topic, output_csv)
    rclpy.spin(pose_logger)
    pose_logger.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
