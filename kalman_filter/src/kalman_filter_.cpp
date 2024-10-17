#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "std_msgs/msg/string.hpp"
#include "kalman_filter.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>  // For formatting output
#include "visp_megapose/msg/confidence.hpp"  // Correct include for Confidence.msg
#include <chrono>
#include <sstream>

using Eigen::VectorXd;

class KalmanFilter : public rclcpp::Node {
private:
    bool csv_save;
    std::string object;
    double confidence_minimum;
    bool object_confidence = false;  // 控制物件信心狀態
    bool flag = false;  // 用來檢查 Kalman Filter 是否需要重新初始化
    double orig_x = 0.0, orig_y = 0.0, orig_z = 0.0;
    double orig_qw = 0.0, orig_qx = 0.0, orig_qy = 0.0, orig_qz = 0.0;
    double filt_x = 0.0, filt_y = 0.0, filt_z = 0.0;
    double filt_qw = 0.0, filt_qx = 0.0, filt_qy = 0.0, filt_qz = 0.0;
    std::shared_ptr<KalmanFilter_hpp> kalman_filter_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr sub_orig;
    rclcpp::Subscription<visp_megapose::msg::Confidence>::SharedPtr sub_conf;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pub_filt;
    std::ofstream data_file_;
    rclcpp::Time last_update_time_;

public:
    KalmanFilter();
    ~KalmanFilter();
    void confidenceCallback(const visp_megapose::msg::Confidence::SharedPtr confidence_msg);
    void trackingCallback(const geometry_msgs::msg::Pose::SharedPtr msg);
    void spin();
};
KalmanFilter::~KalmanFilter() {
    if(csv_save)
        data_file_.close();
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Shutting down KalmanFilter");
}
KalmanFilter::KalmanFilter() : Node("KalmanFilter") {
    csv_save = this->declare_parameter<bool>("csv_save",false);
    object = this->declare_parameter<std::string>("object","/cube");
    confidence_minimum = this->declare_parameter<double>("confidence_minimum",0.5);
    RCLCPP_INFO(this->get_logger(), "object: %s", object.c_str());
    RCLCPP_INFO(this->get_logger(), "csv_save: %s", csv_save ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "confidence_minimum: %f", confidence_minimum);

    kalman_filter_ = std::make_shared<KalmanFilter_hpp>();
    
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
        sub_conf_name, 1, std::bind(&KalmanFilter::confidenceCallback, this, std::placeholders::_1));

    sub_orig = this->create_subscription<geometry_msgs::msg::Pose>(
        object, 1, std::bind(&KalmanFilter::trackingCallback, this, std::placeholders::_1));

    std::string pub_filt_name = object + "_filter";
    pub_filt = this->create_publisher<geometry_msgs::msg::Pose>(pub_filt_name, 1);
}
// Handle confidence updates
void KalmanFilter::confidenceCallback(const visp_megapose::msg::Confidence::SharedPtr confidence_msg) {
    object_confidence = confidence_msg->object_confidence > confidence_minimum && confidence_msg->model_detection;
}

// Handle pose updates
void KalmanFilter::trackingCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
    orig_x = msg->position.x;
    orig_y = msg->position.y;
    orig_z = msg->position.z;
    orig_qw = msg->orientation.w;
    orig_qx = msg->orientation.x;
    orig_qy = msg->orientation.y;
    orig_qz = msg->orientation.z;
}
// Main loop
void KalmanFilter::spin() {
    last_update_time_ = this->get_clock()->now();  // Get the current time
    rclcpp::Rate loop_rate(10);

    while (rclcpp::ok()) {
        rclcpp::spin_some(this->get_node_base_interface());

        if (!object_confidence) {
            flag = false;
            RCLCPP_WARN(this->get_logger(), "%s confidence is false, waiting for re-initialization...",object.c_str());
            loop_rate.sleep();
            continue;
        }

        if (object_confidence && !flag) {
            kalman_filter_ = std::make_shared<KalmanFilter_hpp>();
            flag = true;
            RCLCPP_INFO(this->get_logger(), "Kalman Filter re-initialized");
        }

        // Calculate dt based on real time
        rclcpp::Time current_time = this->get_clock()->now();
        double dt = (current_time - last_update_time_).seconds();
        last_update_time_ = current_time;

        VectorXd z(7);
        z << orig_x, orig_y, orig_z, orig_qw, orig_qx, orig_qy, orig_qz;

        kalman_filter_->predict(dt);
        kalman_filter_->update(z);

        VectorXd state = kalman_filter_->getState();
        filt_x = state(0);
        filt_y = state(1);
        filt_z = state(2);
        filt_qw = state(6);
        filt_qx = state(7);
        filt_qy = state(8);
        filt_qz = state(9);
        if(csv_save){
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
    KalmanFilter().spin();
    rclcpp::shutdown();
    return 0;
}
