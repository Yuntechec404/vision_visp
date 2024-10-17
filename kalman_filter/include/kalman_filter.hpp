#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include <Eigen/Dense>

using namespace Eigen;

class KalmanFilter_hpp {
public:
    KalmanFilter_hpp() {
        // Initializing state vectors and matrices
        state = VectorXd::Zero(10);  // [x, y, z, vx, vy, vz, qw, qx, qy, qz]
        P = MatrixXd::Identity(10, 10);  // Covariance matrix
        F = MatrixXd::Identity(10, 10);  // State transition matrix
        H = MatrixXd::Zero(7, 10);       // Measurement matrix for position and orientation
        R = MatrixXd::Identity(7, 7);    // Measurement noise (set appropriate values)
        Q = MatrixXd::Identity(10, 10) * 0.01;  // Process noise (tune this based on your system)

        // Mapping from state to measurements (position and orientation)
        // Position part
        H(0, 0) = 1;  // x
        H(1, 1) = 1;  // y
        H(2, 2) = 1;  // z

        // Orientation part (qw, qx, qy, qz)
        H(3, 6) = 1;  // qw
        H(4, 7) = 1;  // qx
        H(5, 8) = 1;  // qy
        H(6, 9) = 1;  // qz
    }

    void predict(float dt) {
        // Update F matrix for time delta (dt)
        F(0, 3) = dt;  // x += vx * dt
        F(1, 4) = dt;  // y += vy * dt
        F(2, 5) = dt;  // z += vz * dt

        // Predict step
        state = F * state;
        P = F * P * F.transpose() + Q;
    }

    void update(const VectorXd& z) {
        if (z.size() != 7) {
            throw std::runtime_error("Measurement vector z must have 7 elements.");
        }

        // Kalman gain
        MatrixXd S = H * P * H.transpose() + R;
        MatrixXd K = P * H.transpose() * S.inverse();

        // Update state and covariance
        VectorXd y = z - H * state;  // Residual (innovation)
        state = state + K * y;
        P = (MatrixXd::Identity(10, 10) - K * H) * P;

        // Normalize quaternion part of state (orientation)
        normalizeQuaternion(state);
    }

    VectorXd getState() const {
        return state;
    }

private:
    void normalizeQuaternion(VectorXd& state) {
        double norm = std::sqrt(
            state(6) * state(6) +
            state(7) * state(7) +
            state(8) * state(8) +
            state(9) * state(9)
        );
        if (norm > 1e-6) {  // Avoid division by zero
            state(6) /= norm;  // qw
            state(7) /= norm;  // qx
            state(8) /= norm;  // qy
            state(9) /= norm;  // qz
        }
    }

    VectorXd state;    // State vector [x, y, z, vx, vy, vz, qw, qx, qy, qz]
    MatrixXd F;        // State transition matrix
    MatrixXd P;        // Covariance matrix
    MatrixXd Q;        // Process noise covariance
    MatrixXd R;        // Measurement noise covariance
    MatrixXd H;        // Measurement matrix
};
