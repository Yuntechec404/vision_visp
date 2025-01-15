#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/opencv.hpp>

void callback(const sensor_msgs::ImageConstPtr &image_msg, const sensor_msgs::CameraInfoConstPtr &camera_info_msg)
{
  try
  {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat input_image = cv_ptr->image;

    if (input_image.empty())
    {
      ROS_WARN("Received an empty image!");
      return;
    }

    ROS_INFO("Camera Info - Width: %d, Height: %d", camera_info_msg->width, camera_info_msg->height);
    cv::imshow("Synced Image", input_image);
    cv::waitKey(1);
  }
  catch (const cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vision_node");
  ros::NodeHandle nh;

  std::string image_topic, camera_info_topic;
  nh.param<std::string>("image_topic", image_topic, "/usb_cam/image_raw");
  nh.param<std::string>("camera_info_topic", camera_info_topic, "/usb_cam/camera_info");

  message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, image_topic, 10);
  message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub(nh, camera_info_topic, 10);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo> MySyncPolicy;
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(20), image_sub, camera_info_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  ROS_INFO("Synchronizing image and camera info...");
  ros::spin();

  return 0;
}
