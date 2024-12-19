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

        self.image_subscription = self.create_subscription(Image,'/camera/camera/color/image_raw',self.image_callback,10)

        self.info_subscription = self.create_subscription(CameraInfo,'/camera/camera/color/camera_info',self.info_callback,10)

        self.image_publisher = self.create_publisher(Image,'/camera/color/image_cropped_raw',10)

        self.info_publisher = self.create_publisher(CameraInfo,'/camera/color/image_cropped_info',10)

        self.camera_info = None

        self.get_logger().info("ImageCropperNode has started.")

    def info_callback(self, msg):
        self.camera_info = msg

    def image_callback_1(self, msg):
        try:
            if self.camera_info is None:
                self.get_logger().warn("Waiting for CameraInfo message...")
                return

            # 轉換 ROS2 的 Image 訊息到 OpenCV 映像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Convert to RGBA format
            cv_image_rgba = cv2.cvtColor(cv_image, cv2.COLOR_RGB2RGBA)

            # 取得影像高度和寬度
            height, width, _ = cv_image.shape

            # 裁切下半部分
            cropped_image = cv_image[height // 2:, :, :]
            
            # 轉換裁切後的影像為 ROS2 Image 訊息
            cropped_msg = self.bridge.cv2_to_imgmsg(cropped_image, encoding='rgb8')
            cropped_msg.header.stamp = msg.header.stamp
            cropped_msg.header.frame_id = msg.header.frame_id

            # 發布裁切後的影像
            self.image_publisher.publish(cropped_msg)

            # 調整 CameraInfo
            cropped_camera_info = self.adjust_camera_info(self.camera_info, height // 2)
            cropped_camera_info.header.stamp = msg.header.stamp
            cropped_camera_info.header.frame_id = msg.header.frame_id

            # 發布裁切後的 CameraInfo
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

            # 轉換 ROS2 的 Image 訊息到 OpenCV 映像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # 取得影像高度和寬度
            height, width, _ = cv_image.shape

            # 將上半部塗黑（填充零）
            cv_image[:height // 2, :, :] = 0

            # 轉換處理後的影像為 ROS2 Image 訊息
            blacked_out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')
            blacked_out_msg.header.stamp = msg.header.stamp
            blacked_out_msg.header.frame_id = msg.header.frame_id

            # 發布處理後的影像
            self.image_publisher.publish(blacked_out_msg)

            # CameraInfo 不需要調整，因為尺寸不變
            self.info_publisher.publish(self.camera_info)

            self.get_logger().info(f"Published processed image with blacked-out upper half.")

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def adjust_camera_info(self, camera_info, offset):
        """
        調整 CameraInfo 資料以適應裁切後的影像。
        """
        cropped_info = CameraInfo()
        cropped_info.header = camera_info.header
        cropped_info.height = camera_info.height // 2  # 調整高度
        cropped_info.width = camera_info.width
        cropped_info.distortion_model = camera_info.distortion_model
        cropped_info.d = camera_info.d
        cropped_info.k = list(camera_info.k)
        cropped_info.r = list(camera_info.r)
        cropped_info.p = list(camera_info.p)

        # 調整投影矩陣中的垂直平移部分
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
