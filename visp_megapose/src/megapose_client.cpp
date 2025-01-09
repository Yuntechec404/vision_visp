#define ROS_NO_DEPRECATED_API
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Transform.h>
#include <iostream>


#include <ros/ros.h>
#include <deque>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

// ViSP includes
#include <visp3/core/vpTime.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/detection/vpDetectorDNNOpenCV.h>

// OpenCV/ViSP bridge includes
#include <visp_bridge/3dpose.h>
#include <visp_bridge/camera.h>
#include <visp_bridge/image.h>
#include <cv_bridge/cv_bridge.h>

// ROS 1 includes
#include <tf/transform_broadcaster.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Pose.h>
#include <visual_servoing/Detection.h>

// ROS 1 message filters
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// ROS 1 custom messages and services
#include <visp_megapose/Confidence.h>
#include <visp_megapose/Init.h>
#include <visp_megapose/Track.h>
#include <visp_megapose/Render.h>

using namespace std::chrono_literals; // For using time literals like 1s

enum DetectionMethod
{
  UNKNOWN,
  CLICK,
  DNN
};

struct Detection_allowed
{
  bool detection_allowed;
  float layer;
};

std::map<std::string, DetectionMethod> stringToDetectionMethod = {
  {"UNKNOWN", UNKNOWN},
  {"CLICK", CLICK},
  {"DNN", DNN}};

bool fileExists(const std::string &path)
{
  std::ifstream file(path);
  return file.good();
}

class MegaPoseClient
{
public:
  MegaPoseClient();
  ~MegaPoseClient();
  void spin();
};

MegaPoseClient::MegaPoseClient()
{
  ROS_INFO("MegaPoseClient initialized!");
}

MegaPoseClient::~MegaPoseClient()
{
  ROS_INFO("Shutting down MegaPoseClient");
  ros::shutdown();
}

void MegaPoseClient::spin()
{
  while (ros::ok())
  {
    ros::spinOnce(); // 確保回呼函數可以執行
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "megapose_client");
  ros::NodeHandle n;

  MegaPoseClient client;
  client.spin();
  return 0;
}
