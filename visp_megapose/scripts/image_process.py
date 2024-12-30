#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

class ImageCropperNode:
    def __init__(self):
        # 初始化 ROS 節點
        rospy.init_node('image_cropper_node', anonymous=True)
        self.bridge = CvBridge()

        # 訂閱影像和相機資訊
        self.image_subscription = rospy.Subscriber(
            '/camera/camera/color/image_raw', Image, self.image_callback)
        self.info_subscription = rospy.Subscriber(
            '/camera/camera/color/camera_info', CameraInfo, self.info_callback)

        # 發佈處理後的影像和相機資訊
        self.image_publisher = rospy.Publisher(
            '/camera/color/image_cropped_raw', Image, queue_size=10)
        self.info_publisher = rospy.Publisher(
            '/camera/color/image_cropped_info', CameraInfo, queue_size=10)

        self.camera_info = None

        rospy.loginfo("ImageCropperNode has started.")

    def info_callback(self, msg):
        self.camera_info = msg

    def image_callback(self, msg):
        try:
            if self.camera_info is None:
                rospy.logwarn("Waiting for CameraInfo message...")
                return

            # 轉換 ROS 的 Image 訊息到 OpenCV 映像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # 取得影像高度和寬度
            height, width, _ = cv_image.shape

            # 將上半部塗黑（填充零）
            cv_image[:height // 2, :, :] = 0

            # 轉換處理後的影像為 ROS Image 訊息
            blacked_out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')
            blacked_out_msg.header.stamp = msg.header.stamp
            blacked_out_msg.header.frame_id = msg.header.frame_id

            # 發布處理後的影像
            self.image_publisher.publish(blacked_out_msg)

            # CameraInfo 不需要調整，因為尺寸不變
            self.info_publisher.publish(self.camera_info)

            rospy.loginfo("Published processed image with blacked-out upper half.")

        except Exception as e:
            rospy.logerr(f"Failed to process image: {e}")

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

def main():
    node = ImageCropperNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down ImageCropperNode.")

if __name__ == '__main__':
    main()
