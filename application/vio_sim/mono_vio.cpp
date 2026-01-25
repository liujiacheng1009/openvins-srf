#include <memory>
#include <atomic>
#include <fstream>
#include <mutex>
#include <Eigen/Eigen>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <tf/transform_broadcaster.h>
#include <deque>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "bag_reader.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/FilterWrapper.h"
#include "utils/dataset_reader.h"
#include "utils/print.h"
#include "utils/sensor_data.h"
#include "utils/trajectory_logger.h"
#include <glog/logging.h>

using namespace core;
using namespace type;
using namespace msckf;

std::shared_ptr<VioManager> sys;
std::atomic<bool> thread_update_running{false};
std::mutex camera_queue_mtx;
std::deque<core::CameraData> camera_queue;
std::map<int, double> camera_last_timestamp;
TrajectoryLogger traj_logger_;

struct InputParameter
{
  std::string config_path;
  std::string bag_path;
  std::string traj_path;
  std::string log_path = "";
};

bool parse_arguments(int argc, char **argv, InputParameter &param)
{
  int opt;             /* arguments */
  extern char *optarg; /* current returned argument */
  if (argc < 3)
  {
    std::cout << "input params : -c config_file -d dataset_bag -t save_traj_path -l save_log_path" << std::endl;
    return -1;
  }

  while ((opt = getopt(argc, argv, "c:d:t:l:")) != -1)
  {
    switch (opt)
    {
    case 'c':
      param.config_path = optarg;
      std::cout << "config file: " << optarg << std::endl;
      break;
    case 'd':
      param.bag_path = optarg;
      std::cout << "dataset bag : " << optarg << std::endl;
      break;
    case 't':
      param.traj_path = optarg;
      std::cout << "save traj file: " << optarg << std::endl;
      break;
    case 'l':
      param.log_path = optarg;
      std::cout << "save log file in dir: " << optarg << std::endl;
      break;
    default:
      std::cout << "wrong params: " << optarg << std::endl;
      return false;
    }
  }

  if (param.config_path.empty() || param.bag_path.empty())
  {
    std::cout << "config_file and bag_path must set !! " << std::endl;
    return false;
  }

  return true;
}

void callback_inertial(sensor_msgs::Imu const *imu_msg_ptr)
{

  // convert into correct format
  core::ImuData message;
  message.timestamp = imu_msg_ptr->header.stamp.toSec();
  message.wm = Eigen::Matrix<number_t, 3, 1>(imu_msg_ptr->angular_velocity.x, imu_msg_ptr->angular_velocity.y,
                                             imu_msg_ptr->angular_velocity.z);
  message.am = Eigen::Matrix<number_t, 3, 1>(imu_msg_ptr->linear_acceleration.x,
                                             imu_msg_ptr->linear_acceleration.y,
                                             imu_msg_ptr->linear_acceleration.z);

  // send it to our VIO system
  sys->feed_measurement_imu(message);

  // If the processing queue is currently active / running just return so we can keep getting measurements
  // Otherwise create a second thread to do our update in an async manor
  // The visualization of the state, images, and features will be synchronous with the update!
  if (thread_update_running)
    return;
  thread_update_running = true;
  std::thread thread([&]
                     {
    // Lock on the queue (prevents new images from appending)
    std::lock_guard<std::mutex> lck(camera_queue_mtx);

    // Count how many unique image streams
    std::map<int, bool> unique_cam_ids;
    for (const auto &cam_msg : camera_queue) {
      unique_cam_ids[cam_msg.sensor_ids.at(0)] = true;
    }

    // If we do not have enough unique cameras then we need to wait
    // We should wait till we have one of each camera to ensure we propagate in the correct order
    auto params = sys->get_params();
    size_t num_unique_cameras = (params.state_options.num_cameras == 2) ? 1 : params.state_options.num_cameras;
    if (unique_cam_ids.size() == num_unique_cameras) {

      // Loop through our queue and see if we are able to process any of our camera measurements
      // We are able to process if we have at least one IMU measurement greater than the camera time
      double timestamp_imu_inC = message.timestamp - sys->get_state()->_calib_dt_CAMtoIMU->value()(0);
      while (!camera_queue.empty() && camera_queue.at(0).timestamp < timestamp_imu_inC) {
        sys->feed_measurement_camera(camera_queue.at(0));
        camera_queue.pop_front();
        if (sys->initialized())
        {
          double timestamp_s = sys->get_state()->_timestamp;
          Eigen::Matrix<number_t, 3, 1> bias_a = sys->get_state()->_imu->bias_a();
          Eigen::Matrix<number_t, 3, 1> bias_g = sys->get_state()->_imu->bias_g();
          LOG(INFO) << "vio_ba " << bias_a(0) << " " << bias_a(1) << " " << bias_a(2);
          LOG(INFO) << "vio_bg " << bias_g(0) << " " << bias_g(1) << " " << bias_g(2);
          Eigen::Matrix<number_t, 3, 1> vel = sys->get_state()->_imu->vel();
          LOG(INFO) << "vio_vel " << vel(0) << " " << vel(1) << " " << vel(2);
          Eigen::Matrix<number_t, 3, 1> p_IinG = sys->get_state()->_imu->pos();
          Eigen::Matrix<number_t, 4, 1> q_GtoI = sys->get_state()->_imu->quat(); // jpl表示，等价于hamilton表示的q_ItoG
          Eigen::Quaternion<number_t> vio_quat(q_GtoI[3], q_GtoI[0], q_GtoI[1], q_GtoI[2]);
          Eigen::Matrix<number_t, 3, 1> vio_euler = rot2rpy(vio_quat.toRotationMatrix());
          LOG(INFO) << "vio_pos " << p_IinG[0] << " " << p_IinG[1] << " " << p_IinG[2];
          LOG(INFO) << "vio_euler " << 57.3 * vio_euler[0] << " " << 57.3 * vio_euler[1] << " " << 57.3 * vio_euler[2];
          traj_logger_.log(timestamp_s, p_IinG[0], p_IinG[1], p_IinG[2],
                           q_GtoI[0], q_GtoI[1], q_GtoI[2], q_GtoI[3]);
        }
      }
    }
    thread_update_running = false; });

  // If we are single threaded, then run single threaded
  // Otherwise detach this thread so it runs in the background!
  if (!sys->get_params().use_multi_threading_subs)
  {
    thread.join();
  }
  else
  {
    thread.detach();
  }
  return;
}

void callback_monocular(const sensor_msgs::Image *const &msg0)
{
  int cam_id0 = 0;
  // Check if we should drop this image
  double timestamp = msg0->header.stamp.toSec();
  double time_delta = 1.0 / sys->get_params().track_frequency;
  if (camera_last_timestamp.find(cam_id0) != camera_last_timestamp.end() && timestamp < camera_last_timestamp.at(cam_id0) + time_delta)
  {
    return;
  }
  camera_last_timestamp[cam_id0] = timestamp;

  // Get the image
  cv::Mat img = cv::Mat_<uchar>(msg0->height, msg0->width,
                                const_cast<uchar *>(msg0->data.data())); // XXX: not safe

  // Create the measurement
  core::CameraData message;
  message.timestamp = timestamp;
  message.sensor_ids.push_back(cam_id0);
  message.images.push_back(img.clone());

  // Load the mask if we are using it, else it is empty
  // TODO: in the future we should get this from external pixel segmentation
  if (sys->get_params().use_mask)
  {
    message.masks.push_back(sys->get_params().masks.at(cam_id0));
  }
  else
  {
    message.masks.push_back(cv::Mat::zeros(img.rows, img.cols, CV_8UC1));
  }

  // append it to our queue of images
  std::lock_guard<std::mutex> lck(camera_queue_mtx);
  camera_queue.push_back(message);
  std::sort(camera_queue.begin(), camera_queue.end());
  return;
}

void callback_gtpose(const geometry_msgs::PoseStamped *const &msg0)
{
  // log gt pose for debug
  static bool has_aligned{false};
  static Eigen::Matrix<number_t, 3, 3> R_GtoV = Eigen::Matrix<number_t, 3, 3>::Identity();
  static Eigen::Matrix<number_t, 3, 1> t_GinV = Eigen::Matrix<number_t, 3, 1>::Zero();
  static std::deque<std::shared_ptr<geometry_msgs::PoseStamped>> msg_queue;
  if (!has_aligned)
  {
    std::shared_ptr<geometry_msgs::PoseStamped> msg_cpy(
        new geometry_msgs::PoseStamped(*msg0));
    msg_queue.push_back(msg_cpy);
    if (sys->initialized())
    {
      double timestamp_s = sys->get_state()->_timestamp;
      for (auto &msg : msg_queue)
      {
        constexpr double max_align_dt = 5.e-2; // 适应lidar的帧率，绝大部分场景对齐发生在静止阶段
        if (std::fabs(msg->header.stamp.toSec() - timestamp_s) < max_align_dt) 
        {
          Eigen::Quaternion<number_t> gt_quat(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
          Eigen::Matrix<number_t, 3, 1> p_IinG = sys->get_state()->_imu->pos();
          Eigen::Matrix<number_t, 4, 1> q_GtoI = sys->get_state()->_imu->quat(); // jpl表示，等价于hamilton表示的q_ItoG
          Eigen::Quaternion<number_t> vio_quat(q_GtoI[3], q_GtoI[0], q_GtoI[1], q_GtoI[2]);
          R_GtoV = vio_quat.toRotationMatrix() * gt_quat.toRotationMatrix().transpose();
          t_GinV = p_IinG - R_GtoV * Eigen::Matrix<number_t, 3, 1>(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
          has_aligned = true;
          while (!msg_queue.empty())
            msg_queue.pop_front();
          break;
        }
      }
    }
    if (msg_queue.size() > 100)
      msg_queue.pop_front();
    return;
  }

  Eigen::Matrix<number_t, 3, 1> gt_pos = R_GtoV * Eigen::Matrix<number_t, 3, 1>(msg0->pose.position.x, msg0->pose.position.y, msg0->pose.position.z) + t_GinV;
  LOG(INFO) << "gt_pos " << gt_pos(0) << " " << gt_pos(1) << " " << gt_pos(2);
  Eigen::Quaternion<number_t> gt_quat(msg0->pose.orientation.w, msg0->pose.orientation.x, msg0->pose.orientation.y, msg0->pose.orientation.z);
  Eigen::Matrix<number_t, 3, 3> gt_rot = R_GtoV * gt_quat.toRotationMatrix();
  Eigen::Matrix<number_t, 3, 1> gt_euler = rot2rpy(gt_rot);
  LOG(INFO) << "gt_euler " << 57.3 * gt_euler[0] << " " << 57.3 * gt_euler[1] << " " << 57.3 * gt_euler[2];
  return;
}

int main(int argc, char **argv)
{
  InputParameter input_param;
  if (!parse_arguments(argc, argv, input_param))
  {
    return -1;
  }

  google::InitGoogleLogging(argv[0]);
  if (!input_param.log_path.empty()) {
    FLAGS_log_dir = input_param.log_path;
    FLAGS_alsologtostderr = true;
  }
  if (!input_param.traj_path.empty())
  {
    traj_logger_.open(input_param.traj_path, false);
  }

  auto parser = std::make_shared<core::YamlParser>(input_param.config_path);
  if (!parser->successful())
  {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }
  std::string verbosity = "DEBUG";
  parser->parse_config("verbosity", verbosity);
  core::Printer::setPrintLevel(verbosity);

  VioManagerOptions params;
  params.print_and_load(parser);
  params.num_opencv_threads = 0; // for repeatability
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;
  sys = std::make_shared<VioManager>(params);

  BagReader bag_player;
  if (!bag_player.load(input_param.bag_path))
  {
    PRINT_ERROR(RED "读取bag包错误 !!\n");
    std::exit(EXIT_FAILURE);
  }
  std::string imu_topic;
  parser->parse_external("relative_config_imu", "imu0", "rostopic", imu_topic);
  std::function<void(sensor_msgs::Imu const *const)> imu_cb = callback_inertial;
  bag_player.register_callback(imu_topic, imu_cb);

  std::string cam_topic;
  parser->parse_external("relative_config_imucam", "cam" + std::to_string(0), "rostopic", cam_topic);
  std::function<void(sensor_msgs::Image const *const)> mono_img_cb = callback_monocular;
  bag_player.register_callback(cam_topic, mono_img_cb);

  std::string gtpose_topic = "/gt_pose";
  std::function<void(geometry_msgs::PoseStamped const *const)> gtpose_cb = callback_gtpose;
  bag_player.register_callback(gtpose_topic, gtpose_cb);

  bag_player.play();

  if (traj_logger_.isOpen())
  {
    traj_logger_.flush();
  }
  return EXIT_SUCCESS;
}