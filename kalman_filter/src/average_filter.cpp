#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "std_msgs/msg/string.hpp"
#include "visp_megapose/msg/confidence.hpp"  // Correct include for Confidence.msg
#include <deque>
#include <numeric>  // For calculating averages
#include <fstream>
#include <iomanip>  // For formatting output
#include <chrono>
#include <sstream>

class AverageFilter : public rclcpp::Node {
private:
    bool csv_save;
    std::string object;
    double confidence_minimum;
    bool object_confidence = false;  // 控制物件信心狀態
    // bool flag = false;  // 用來檢查 Kalman Filter 是否需要重新初始化
    std::deque<double> buffer_x, buffer_y, buffer_z,buffer_qw, buffer_qx, buffer_qy, buffer_qz;
    size_t buffer_size;
    double orig_x = 0.0, orig_y = 0.0, orig_z = 0.0;
    double orig_qw = 0.0, orig_qx = 0.0, orig_qy = 0.0, orig_qz = 0.0;
    double filt_x = 0.0, filt_y = 0.0, filt_z = 0.0;
    double filt_qw = 0.0, filt_qx = 0.0, filt_qy = 0.0, filt_qz = 0.0;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr sub_orig;
    rclcpp::Subscription<visp_megapose::msg::Confidence>::SharedPtr sub_conf;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pub_filt;
    std::ofstream data_file_;

public:
    AverageFilter();
    ~AverageFilter();
    void confidenceCallback(const visp_megapose::msg::Confidence::SharedPtr confidence_msg);
    void trackingCallback(const geometry_msgs::msg::Pose::SharedPtr msg);
    void spin();
};
AverageFilter::~AverageFilter() {
    if(csv_save)
        data_file_.close();
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Shutting down AverageFilter");
}
AverageFilter::AverageFilter() : Node("AverageFilter") {
    csv_save = this->declare_parameter<bool>("csv_save",false);
    object = this->declare_parameter<std::string>("object","/cube");
    confidence_minimum = this->declare_parameter<double>("confidence_minimum",0.5);
    buffer_size = this->declare_parameter<int>("buffer_size",5);
    RCLCPP_INFO(this->get_logger(), "object: %s", object.c_str());
    RCLCPP_INFO(this->get_logger(), "csv_save: %s", csv_save ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "confidence_minimum: %f", confidence_minimum);
    RCLCPP_INFO(this->get_logger(), "buffer_size: %ld", buffer_size);
    
    if(csv_save){
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_time), "%Y%m%d%H%M");
        std::string filename = "filter_" + ss.str() + ".csv";
        data_file_.open(filename);
        data_file_ << "orig_x, orig_y, orig_z, orig_qw, orig_qx, orig_qy, orig_qz, "
                << "filt_x, filt_y, filt_z, filt_qw, filt_qx, filt_qy, filt_qz\n";
    }
    std::string sub_conf_name = object + "_confidence";
    sub_conf = this->create_subscription<visp_megapose::msg::Confidence>(
        sub_conf_name, 1, std::bind(&AverageFilter::confidenceCallback, this, std::placeholders::_1));

    sub_orig = this->create_subscription<geometry_msgs::msg::Pose>(
        object, 1, std::bind(&AverageFilter::trackingCallback, this, std::placeholders::_1));

    std::string pub_filt_name = object + "_filter";
    pub_filt = this->create_publisher<geometry_msgs::msg::Pose>(pub_filt_name, 1);
}
// Handle confidence updates
void AverageFilter::confidenceCallback(const visp_megapose::msg::Confidence::SharedPtr confidence_msg) {
    object_confidence = confidence_msg->object_confidence > confidence_minimum && confidence_msg->model_detection;
}
// Calculate moving average for a deque of values
double calculateMovingAverage(const std::deque<double>& buffer) {
    if (buffer.size() < 1) return 0.0;  // Avoid division by zero
    return std::accumulate(buffer.begin(), buffer.end(), 0.0) / buffer.size();
}
// Handle pose updates
void AverageFilter::trackingCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
    orig_x = msg->position.x;
    orig_y = msg->position.y;
    orig_z = msg->position.z;
    orig_qw = msg->orientation.w;
    orig_qx = msg->orientation.x;
    orig_qy = msg->orientation.y;
    orig_qz = msg->orientation.z;
    // Update the buffer with the new pose data
    if (buffer_x.size() >= buffer_size) {
        buffer_x.pop_front();
        buffer_y.pop_front();
        buffer_z.pop_front();
        buffer_qw.pop_front();
        buffer_qx.pop_front();
        buffer_qy.pop_front();
        buffer_qz.pop_front();
    }
    buffer_x.push_back(msg->position.x);
    buffer_y.push_back(msg->position.y);
    buffer_z.push_back(msg->position.z);
    buffer_qw.push_back(msg->orientation.w);
    buffer_qx.push_back(msg->orientation.x);
    buffer_qy.push_back(msg->orientation.y);
    buffer_qz.push_back(msg->orientation.z);

    // Compute the moving average for position and orientation
    filt_x = calculateMovingAverage(buffer_x);
    filt_y = calculateMovingAverage(buffer_y);
    filt_z = calculateMovingAverage(buffer_z);
    filt_qw = calculateMovingAverage(buffer_qw);
    filt_qx = calculateMovingAverage(buffer_qx);
    filt_qy = calculateMovingAverage(buffer_qy);
    filt_qz = calculateMovingAverage(buffer_qz);
}
// Main loop
void AverageFilter::spin() {
    while (rclcpp::ok()) {
        rclcpp::spin_some(this->get_node_base_interface());
        rclcpp::Rate loop_rate(10);

        if (!object_confidence) {
            // flag = false;
            RCLCPP_WARN(this->get_logger(), "%s confidence is false, waiting for re-initialization...",object.c_str());
            // Clear the buffers
            buffer_x.clear();
            buffer_y.clear();
            buffer_z.clear();
            buffer_qw.clear();
            buffer_qx.clear();
            buffer_qy.clear();
            buffer_qz.clear();

            loop_rate.sleep();
            continue;
        }

        if(csv_save) {  // Only save when averaging is done
            data_file_ << std::fixed << std::setprecision(6)
                       << orig_x << ", " << orig_y << ", " << orig_z << ", "
                       << orig_qw << ", " << orig_qx << ", " << orig_qy << ", " << orig_qz << ", "
                       << filt_x << ", " << filt_y << ", " << filt_z << ", "
                       << filt_qw << ", " << filt_qx << ", " << filt_qy << ", " << filt_qz << "\n";
        }
        geometry_msgs::msg::Pose filtered_pose;
        filtered_pose.position.x = filt_x;
        filtered_pose.position.y = filt_y;
        filtered_pose.position.z = filt_z;
        filtered_pose.orientation.w = filt_qw;
        filtered_pose.orientation.x = filt_qx;
        filtered_pose.orientation.y = filt_qy;
        filtered_pose.orientation.z = filt_qz;

        RCLCPP_INFO(this->get_logger(),
                    "Filtered Pose: [x: %f, y: %f, z: %f, qw: %f, qx: %f, qy: %f, qz: %f]",
                    filtered_pose.position.x, filtered_pose.position.y, filtered_pose.position.z,
                    filtered_pose.orientation.w, filtered_pose.orientation.x,
                    filtered_pose.orientation.y, filtered_pose.orientation.z);

        pub_filt->publish(filtered_pose);
        loop_rate.sleep();
    }
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    AverageFilter().spin();
    rclcpp::shutdown();
    return 0;
}
