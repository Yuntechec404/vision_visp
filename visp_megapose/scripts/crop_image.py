#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

class ImageCropperNode(Node):
    def __init__(self):
        super().__init__('image_cropper_node')
        self.bridge = CvBridge()

        # 订阅 RealSense 相机的影像主题
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # 根据实际的影像主题修改
            self.image_callback,
            10
        )

        # 订阅 CameraInfo 主题
        self.info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',  # 根据实际的 CameraInfo 主题修改
            self.info_callback,
            10
        )

        # 发布裁切后的影像
        self.image_publisher = self.create_publisher(
            Image,
            '/camera/color/image_cropped_raw',
            10
        )

        # 发布裁切后的 CameraInfo
        self.info_publisher = self.create_publisher(
            CameraInfo,
            '/camera/color/image_cropped_info',
            10
        )

        # 保存最近的 CameraInfo
        self.camera_info = None

        self.get_logger().info("ImageCropperNode has started.")

    def info_callback(self, msg):
        # 保存接收到的 CameraInfo
        self.camera_info = msg

    def image_callback_1(self, msg):
        try:
            if self.camera_info is None:
                self.get_logger().warn("Waiting for CameraInfo message...")
                return

            # 转换 ROS2 的 Image 消息到 OpenCV 图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Convert to RGBA format
            cv_image_rgba = cv2.cvtColor(cv_image, cv2.COLOR_RGB2RGBA)

            # 获取图像高度和宽度
            height, width, _ = cv_image.shape

            # 裁切下半部分
            cropped_image = cv_image[height // 2:, :, :]
            
            # 转换裁切后的图像为 ROS2 Image 消息
            cropped_msg = self.bridge.cv2_to_imgmsg(cropped_image, encoding='rgb8')
            cropped_msg.header.stamp = msg.header.stamp
            cropped_msg.header.frame_id = msg.header.frame_id

            # 发布裁切后的影像
            self.image_publisher.publish(cropped_msg)

            # 调整 CameraInfo
            cropped_camera_info = self.adjust_camera_info(self.camera_info, height // 2)
            cropped_camera_info.header.stamp = msg.header.stamp
            cropped_camera_info.header.frame_id = msg.header.frame_id

            # 发布裁切后的 CameraInfo
            self.info_publisher.publish(cropped_camera_info)

            self.get_logger().info(f"Published image and CameraInfo with size: {cv_image.shape}")
            self.get_logger().info(f"Published cropped image and CameraInfo with size: {cropped_image.shape}")

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def image_callback(self, msg):
        try:
            if self.camera_info is None:
                self.get_logger().warn("Waiting for CameraInfo message...")
                return

            # 转换 ROS2 的 Image 消息到 OpenCV 图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # 获取图像高度和宽度
            height, width, _ = cv_image.shape

            # 将上半部分涂黑（填充零）
            cv_image[:height // 2, :, :] = 0  # 将上半部分的像素值设为零 (黑色)

            # 转换处理后的图像为 ROS2 Image 消息
            blacked_out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')
            blacked_out_msg.header.stamp = msg.header.stamp
            blacked_out_msg.header.frame_id = msg.header.frame_id

            # 发布处理后的影像
            self.image_publisher.publish(blacked_out_msg)

            # CameraInfo 不需要调整，因为尺寸未变
            self.info_publisher.publish(self.camera_info)

            self.get_logger().info(f"Published processed image with blacked-out upper half.")

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def adjust_camera_info(self, camera_info, offset):
        """
        调整 CameraInfo 数据以适应裁切后的图像。
        """
        cropped_info = CameraInfo()
        cropped_info.header = camera_info.header
        cropped_info.height = camera_info.height // 2  # 调整高度
        cropped_info.width = camera_info.width  # 宽度保持不变
        cropped_info.distortion_model = camera_info.distortion_model
        cropped_info.d = camera_info.d
        cropped_info.k = list(camera_info.k)
        cropped_info.r = list(camera_info.r)
        cropped_info.p = list(camera_info.p)

        # 调整投影矩阵中的垂直平移部分
        cropped_info.p[5] -= offset

        return cropped_info

def main(args=None):
    rclpy.init(args=args)
    node = ImageCropperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ImageCropperNode.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
