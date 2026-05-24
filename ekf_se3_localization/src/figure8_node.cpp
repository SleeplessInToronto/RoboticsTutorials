#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>

/**
 * Closed-loop circle velocity controller.
 *
 * Desired: linear.x = speed, angular.z = speed / radius
 * Actual:  read from /odom twist (wheel-encoder integrated)
 *
 * A proportional controller drives actual velocity → desired, so that the
 * cmd_vel published to /cmd_vel closely matches what the car actually does.
 * The EKF uses /cmd_vel as its prediction input u_i; when actual ≈ commanded
 * the motion model error is small and Q can be honest rather than inflated.
 */
class CircleNode : public rclcpp::Node
{
public:
  CircleNode()
  : Node("figure8_node")
  {
    declare_parameter<double>("speed",  1.0);
    declare_parameter<double>("radius", 7.0);
    declare_parameter<double>("rate",   50.0);
    declare_parameter<double>("kp_v",   1.0);  // proportional gain — linear velocity
    declare_parameter<double>("kp_w",   2.0);  // proportional gain — angular velocity

    const double speed  = get_parameter("speed").as_double();
    const double radius = get_parameter("radius").as_double();
    const double rate   = get_parameter("rate").as_double();
    kp_v_ = get_parameter("kp_v").as_double();
    kp_w_ = get_parameter("kp_w").as_double();

    desired_vx_  = speed;
    desired_wz_  = speed / radius;

    pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
        actual_vx_ = msg->twist.twist.linear.x;
        actual_wz_ = msg->twist.twist.angular.z;
        odom_received_ = true;
      });

    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / rate));
    timer_ = create_wall_timer(dt, [this]() { publish_cmd(); });

    RCLCPP_INFO(get_logger(),
      "Circle controller: v_des=%.2f m/s, r=%.1f m, w_des=%.3f rad/s, kp_v=%.1f kp_w=%.1f",
      desired_vx_, 1.0 / (desired_wz_ / desired_vx_), desired_wz_, kp_v_, kp_w_);
  }

private:
  void publish_cmd()
  {
    geometry_msgs::msg::Twist msg;
    if (odom_received_) {
      msg.linear.x  = desired_vx_ + kp_v_ * (desired_vx_ - actual_vx_);
      msg.angular.z = desired_wz_ + kp_w_ * (desired_wz_ - actual_wz_);
    } else {
      msg.linear.x  = desired_vx_;
      msg.angular.z = desired_wz_;
    }
    pub_->publish(msg);
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  double desired_vx_{1.0};
  double desired_wz_{0.143};
  double actual_vx_{0.0};
  double actual_wz_{0.0};
  double kp_v_{1.0};
  double kp_w_{2.0};
  bool   odom_received_{false};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CircleNode>());
  rclcpp::shutdown();
  return 0;
}
