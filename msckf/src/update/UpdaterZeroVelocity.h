/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_UPDATER_ZEROVELOCITY_H
#define OV_MSCKF_UPDATER_ZEROVELOCITY_H

#include <memory>

#include "utils/sensor_data.h"
#include "utils/constant.h"
#include "UpdaterOptions.h"
#include "core/VioManagerOptions.h"
#include "utils/NoiseManager.h"
#include "lowpass_filter/lowpass_filter.h"

namespace core {
class Feature;
class FeatureDatabase;
} // namespace core
namespace type {
class Landmark;
} // namespace type

namespace msckf {

class State;
class Propagator;

/**
 * @brief Will try to *detect* and then update using zero velocity assumption.
 *
 * Consider the case that a VIO unit remains stationary for a period time.
 * Typically this can cause issues in a monocular system without SLAM features since no features can be triangulated.
 * Additional, if features could be triangulated (e.g. stereo) the quality can be poor and hurt performance.
 * If we can detect the cases where we are stationary then we can leverage this to prevent the need to do feature update during this period.
 * The main application would be using this on a **wheeled vehicle** which needs to stop (e.g. stop lights or parking).
 */
class UpdaterZeroVelocity {

public:
  /**
   * @brief Default constructor for our zero velocity detector and updater.
   * @param options Updater options (chi2 multiplier)
   * @param noises imu noise characteristics (continuous time)
   * @param db Feature tracker database with all features in it
   * @param prop Propagator class object which can predict the state forward in time
   * @param gravity_mag Global gravity magnitude of the system (normally 9.81)
   * @param zupt_max_velocity Max velocity we should consider to do a update with
   * @param zupt_noise_multiplier Multiplier of our IMU noise matrix (default should be 1.0)
   * @param zupt_max_disparity Max disparity we should consider to do a update with
   */
  UpdaterZeroVelocity(UpdaterOptions &options, NoiseManager &noises, std::shared_ptr<core::FeatureDatabase> db,
                      std::shared_ptr<Propagator> prop, double gravity_mag, double zupt_max_velocity, double zupt_noise_multiplier,
                      double zupt_max_disparity, ZUPT_CHECK_METHOD zupt_check_method, double zupt_check_acc_threshold, double zupt_check_gyro_threshold, int zupt_check_imu_cnt);

  /**
   * @brief Feed function for inertial data
   * @param message Contains our timestamp and inertial information
   * @param oldest_time Time that we can discard measurements before
   */
  void feed_imu(const core::ImuData &message, double oldest_time = -1) {

    // Append it to our vector
    imu_data.emplace_back(message);

    // Sort our imu data (handles any out of order measurements)
    // std::sort(imu_data.begin(), imu_data.end(), [](const IMUDATA i, const IMUDATA j) {
    //    return i.timestamp < j.timestamp;
    //});

    _acc_lowpass_filter.addSample(message.am, message.timestamp * kSecToNsec);
    _acc_static_counter.appendFrame(
        (message.am - _acc_lowpass_filter.getFilteredData()).norm() < _zupt_check_acc_threshold,
        message.timestamp);
    _gyro_lowpass_filter.addSample(message.wm, message.timestamp * kSecToNsec);
    _gyro_static_counter.appendFrame(
        (message.wm - _gyro_lowpass_filter.getFilteredData()).norm() < _zupt_check_gyro_threshold,
        message.timestamp);
    // Clean old measurements
    // std::cout << "ZVUPT: imu_data.size() " << imu_data.size() << std::endl;
    clean_old_imu_measurements(oldest_time - 0.10);
  }

  /**
   * @brief This will remove any IMU measurements that are older then the given measurement time
   * @param oldest_time Time that we can discard measurements before (in IMU clock)
   */
  void clean_old_imu_measurements(double oldest_time) {
    if (oldest_time < 0)
      return;
    auto it0 = imu_data.begin();
    while (it0 != imu_data.end()) {
      if (it0->timestamp < oldest_time) {
        it0 = imu_data.erase(it0);
      } else {
        it0++;
      }
    }
  }

  /**
   * @brief Will first detect if the system is zero velocity, then will update.
   * @param state State of the filter
   * @param timestamp Next camera timestamp we want to see if we should propagate to.
   * @return True if the system is currently at zero velocity
   */
  bool try_update(std::shared_ptr<State> state, double timestamp);

protected:
  LowpassFilter _acc_lowpass_filter;
  LowpassFilter _gyro_lowpass_filter;

  IsStaticCounter _acc_static_counter;
  IsStaticCounter _gyro_static_counter;
  /// Options used during update (chi2 multiplier)
  UpdaterOptions _options;

  /// Container for the imu noise values
  NoiseManager _noises;

  /// Feature tracker database with all features in it
  std::shared_ptr<core::FeatureDatabase> _db;

  /// Our propagator!
  std::shared_ptr<Propagator> _prop;

  /// Gravity vector
  Eigen::Matrix<number_t, 3, 1> _gravity;

  /// Max velocity (m/s) that we should consider a zupt with
  double _zupt_max_velocity = 1.0;

  /// Multiplier of our IMU noise matrix (default should be 1.0)
  double _zupt_noise_multiplier = 1.0;

  /// Max disparity (pixels) that we should consider a zupt with
  double _zupt_max_disparity = 1.0;

  ZUPT_CHECK_METHOD _zupt_check_method = ZUPT_CHECK_METHOD::USE_IMU_AND_DISPARITY;

  double _zupt_check_gyro_threshold = 0.05;

  double _zupt_check_acc_threshold = 0.5;

  /// Chi squared 95th percentile table (lookup would be size of residual)
  std::map<int, double> chi_squared_table;

  /// Our history of IMU messages (time, angular, linear)
  std::vector<core::ImuData> imu_data;

  /// Estimate for time offset at last propagation time
  double last_prop_time_offset = 0.0;
  bool have_last_prop_time_offset = false;

  /// Last timestamp we did zero velocity update with
  double last_zupt_state_timestamp = 0.0;

  /// Number of times we have called update
  int last_zupt_count = 0;
};

} // namespace msckf

#endif // OV_MSCKF_UPDATER_ZEROVELOCITY_H
