#include <csignal>
#include <memory>

#include "core/VioManager.h"
#include "sim/Simulator.h"
#include "utils/colors.h"
#include "utils/dataset_reader.h"
#include "utils/print.h"
#include "utils/sensor_data.h"
#include "state/State.h"
#include <glog/logging.h>

using namespace core;
using namespace type;
using namespace msckf;

std::shared_ptr<Simulator> sim;
std::shared_ptr<VioManager> sys;

// Define the function to be called when ctrl-c (SIGINT) is sent to process
void signal_callback_handler(int signum) { std::exit(signum); }

// Main function
int main(int argc, char **argv) {

  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path_to_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  google::InitGoogleLogging(argv[0]);
  if (!log_path.empty()) {
    FLAGS_log_dir = log_path;
    FLAGS_alsologtostderr = true;
  }
  // Load the config
  auto parser = std::make_shared<core::YamlParser>(config_path);

  // Verbosity
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  core::Printer::setPrintLevel(verbosity);

  // Create our VIO system
  VioManagerOptions params;
  params.print_and_load(parser);
  params.print_and_load_simulation(parser);
  params.num_opencv_threads = 0; // for repeatability
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;
  sim = std::make_shared<Simulator>(params);
  sys = std::make_shared<VioManager>(params);

  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Get initial state
  // NOTE: we are getting it at the *next* timestep so we get the first IMU message
  double next_imu_time = sim->current_timestamp() + 1.0 / params.sim_freq_imu;
  Eigen::Matrix<double, 17, 1> imustate;
  bool success = sim->get_state(next_imu_time, imustate);
  if (!success) {
    PRINT_ERROR(RED "[SIM]: Could not initialize the filter to the first state\n" RESET);
    PRINT_ERROR(RED "[SIM]: Did the simulator load properly???\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Since the state time is in the camera frame of reference
  // Subtract out the imu to camera time offset
  imustate(0, 0) -= sim->get_true_parameters().calib_camimu_dt;

  // Initialize our filter with the groundtruth
  sys->initialize_with_gt(imustate);

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Buffer our camera image
  double buffer_timecam = -1;
  std::vector<int> buffer_camids;
  std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> buffer_feats;

  // Step through the rosbag
  signal(SIGINT, signal_callback_handler);
  while (sim->ok()) {

    // IMU: get the next simulated IMU measurement if we have it
    core::ImuData message_imu;
    bool hasimu = sim->get_next_imu(message_imu.timestamp, message_imu.wm, message_imu.am);
    if (hasimu) {
      sys->feed_measurement_imu(message_imu);
    }

    // CAM: get the next simulated camera uv measurements if we have them
    double time_cam;
    std::vector<int> camids;
    std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> feats;
    bool hascam = sim->get_next_cam(time_cam, camids, feats);
    if (hascam) {
      if (buffer_timecam != -1) {
        sys->feed_measurement_simulation(buffer_timecam, buffer_camids, buffer_feats);
      }
      buffer_timecam = time_cam;
      buffer_camids = camids;
      buffer_feats = feats;

      {
        Eigen::Matrix<number_t, 3, 1> vel = sys->get_state()->_imu->vel();
        LOG(INFO) << "vio_vel " << vel(0) << " " << vel(1) << " " << vel(2);
        Eigen::Matrix<number_t, 3, 1> p_IinG = sys->get_state()->_imu->pos();
        Eigen::Matrix<number_t, 4, 1> q_GtoI = sys->get_state()->_imu->quat(); // jpl表示，等价于hamilton表示的q_ItoG
        Eigen::Quaternion<number_t> vio_quat(q_GtoI[3], q_GtoI[0], q_GtoI[1], q_GtoI[2]);
        Eigen::Matrix<number_t, 3, 1> vio_euler = rot2rpy(vio_quat.toRotationMatrix());
        LOG(INFO) << "vio_pos " << p_IinG[0] << " " << p_IinG[1] << " " << p_IinG[2];
        LOG(INFO) << "vio_euler " << 57.3 * vio_euler[0] << " " << 57.3 * vio_euler[1] << " " << 57.3 * vio_euler[2];
      }

      {
        Eigen::Matrix<double, 17, 1> imustate;
        sim->get_state(message_imu.timestamp, imustate); // need to be earlier than the current time
        Eigen::Matrix<number_t, 3, 1> vel = imustate.block(8, 0, 3, 1).cast<number_t>();
        LOG(INFO) << "gt_vel " << vel(0) << " " << vel(1) << " " << vel(2);
        Eigen::Matrix<number_t, 3, 1> p_IinG = imustate.block(5, 0, 3, 1).cast<number_t>();
        Eigen::Matrix<number_t, 4, 1> q_GtoI = imustate.block(1, 0, 4, 1).cast<number_t>(); // jpl表示，等价于hamilton表示的q_ItoG
        Eigen::Quaternion<number_t> vio_quat(q_GtoI[3], q_GtoI[0], q_GtoI[1], q_GtoI[2]);
        Eigen::Matrix<number_t, 3, 1> vio_euler = rot2rpy(vio_quat.toRotationMatrix());
        LOG(INFO) << "gt_pos " << p_IinG[0] << " " << p_IinG[1] << " " << p_IinG[2];
        LOG(INFO) << "gt_euler " << 57.3 * vio_euler[0] << " " << 57.3 * vio_euler[1] << " " << 57.3 * vio_euler[2];
      }
    }
  }

  // Done!
  return EXIT_SUCCESS;
}
