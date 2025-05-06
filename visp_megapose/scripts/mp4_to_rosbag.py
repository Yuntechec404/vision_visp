#!/usr/bin/env python3
import argparse
import cv2
import rosbag
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge

def main():
    # 解析命令行参数
    video_path = rospy.get_param(rospy.get_name() + "/video_path", "/home/user/catkin_ws/Produce2.mp4")
    bag_path = rospy.get_param(rospy.get_name() + "/bag_path", "/home/user/catkin_ws/video.bag")
    fps_override = rospy.get_param(rospy.get_name() + "/fps", 30.0)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file:", video_path)
        return

    # 获取视频帧率和分辨率
    fps = fps_override
    if fps <= 0 or fps != fps:  # 检查 NaN 或非正数
        fps = 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 如果需要固定为 640x480，可在此处调整
    width = 640
    height = 480

    # 准备 rosbag 和 cv_bridge
    bag = rosbag.Bag(bag_path, 'w')
    bridge = CvBridge()

    # 相机内参（请替换为实际的标定值）
    # 例如使用 plumb_bob 模型的五参数畸变
    camera_info_msg = CameraInfo()
    camera_info_msg.header.frame_id = "camera_color_optical_frame"
    camera_info_msg.width = width
    camera_info_msg.height = height
    camera_info_msg.distortion_model = "plumb_bob"

    # 相機內參和畸變參數 (請用實際值替換以下示例值)
    # 焦距和主點坐標 (內參)
    fx = 600.0  # 焦距 x (像素)
    fy = 600.0  # 焦距 y (像素)
    cx = 320.0  # 主點 x 坐標 (像素)
    cy = 240.0  # 主點 y 坐標 (像素)

    # 畸變係數 (k1, k2, p1, p2, k3) 假設使用 plumb_bob 模型
    k1 = 0.0
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0
    k3 = 0.0

    # 这里示例值，应使用实际标定结果
    camera_info_msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    camera_info_msg.K = [fx, 0.0, cx,
                         0.0, fy, cy,
                         0.0, 0.0, 1.0]
    camera_info_msg.R = [1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0]
    camera_info_msg.P = [fx, 0.0, cx, 0.0,
                         0.0, fy, cy, 0.0,
                         0.0, 0.0, 1.0, 0.0]

    # 帧时间控制
    timestamp = 0.0
    duration = 1.0 / fps
    frame_id = "camera_color_optical_frame"

    # 循环读取视频帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 如果不是 640x480，则调整尺寸
        frame = cv2.resize(frame, (width, height))

        # 转换为 ROS 图像消息
        img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.frame_id = frame_id
        # 设置时间戳
        timestamp += duration
        img_msg.header.stamp = rospy.Time.from_sec(timestamp)

        # 同步设置 CameraInfo 时间戳
        camera_info_msg.header.stamp = img_msg.header.stamp

        # 写入 rosbag
        bag.write("/camera/color/image_raw", img_msg, img_msg.header.stamp)
        bag.write("/camera/color/camera_info", camera_info_msg, camera_info_msg.header.stamp)

        frame_count += 1

    # 关闭资源
    cap.release()
    bag.close()
    print(f"Written {frame_count} frames to {bag_path}")

if __name__ == "__main__":
    main()
