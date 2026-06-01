#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cmath>

/**
 * Closed-loop circle velocity controller.
 *
 * Drives a circle of the configured radius at the configured speed.
 * Stops automatically after max_laps full laps by publishing zero
 * velocity and shutting down — prevents the repeated-traversal
 * pose-graph degradation described in the SLAM Handbook (Ch.1 p.22).
 */
class CircleNode : public rclcpp::Node
{
public:
  CircleNode()
  : Node("figure8_node")
  {
    declare_parameter<double>("speed",    1.0);
    declare_parameter<double>("radius",   7.0);
    declare_parameter<double>("rate",    50.0);
    declare_parameter<double>("kp_v",     0.0);
    declare_parameter<double>("kp_w",     0.0);
    declare_parameter<double>("max_laps", 1.3);

    const double speed    = get_parameter("speed").as_double();
    const double radius   = get_parameter("radius").as_double();
    const double rate     = get_parameter("rate").as_double();
    kp_v_     = get_parameter("kp_v").as_double();
    kp_w_     = get_parameter("kp_w").as_double();
    max_angle_ = get_parameter("max_laps").as_double() * 2.0 * M_PI;

    desired_vx_ = speed;
    desired_wz_ = speed / radius;
    dt_         = 1.0 / rate;

    pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
        actual_vx_    = msg->twist.twist.linear.x;
        actual_wz_    = msg->twist.twist.angular.z;
        odom_received_ = true;
      });

    const auto dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(dt_));
    timer_ = create_wall_timer(dt_ns, [this]() { publish_cmd(); });

    RCLCPP_INFO(get_logger(),
      "Circle controller: v=%.2f m/s  r=%.1f m  ω=%.3f rad/s  max_laps=%.2f",
      desired_vx_, radius, desired_wz_,
      get_parameter("max_laps").as_double());
  }

private:
  void publish_cmd()
  {
    if (done_) return;

    total_angle_ += desired_wz_ * dt_;

    if (total_angle_ >= max_angle_) {
      // Send zero velocity and stop
      pub_->publish(geometry_msgs::msg::Twist{});
      done_ = true;
      RCLCPP_INFO(get_logger(),
        "Completed %.2f laps — stopping simulation.",
        get_parameter("max_laps").as_double());
      rclcpp::shutdown();
      return;
    }

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

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr    pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr   odom_sub_;
  rclcpp::TimerBase::SharedPtr                               timer_;

  double desired_vx_{1.0};
  double desired_wz_{0.143};
  double actual_vx_{0.0};
  double actual_wz_{0.0};
  double kp_v_{0.0};
  double kp_w_{0.0};
  double dt_{0.02};
  double max_angle_{1.3 * 2.0 * M_PI};
  double total_angle_{0.0};
  bool   odom_received_{false};
  bool   done_{false};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CircleNode>());
  rclcpp::shutdown();
  return 0;
}
