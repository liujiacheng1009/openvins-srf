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

#include "Feature.h"

using namespace core;

void Feature::clean_old_measurements(const std::vector<double> &valid_times) {

  // Loop through each of the cameras we have
  for (auto const &pair : timestamps) {

    // Assert that we have all the parts of a measurement
    assert(timestamps[pair.first].size() == uvs[pair.first].size());
    assert(timestamps[pair.first].size() == uvs_norm[pair.first].size());

    // Our iterators
    auto it1 = timestamps[pair.first].begin();
    auto it2 = uvs[pair.first].begin();
    auto it3 = uvs_norm[pair.first].begin();

    // Loop through measurement times, remove ones that are not in our timestamps
    while (it1 != timestamps[pair.first].end()) {
      if (std::find(valid_times.begin(), valid_times.end(), *it1) == valid_times.end()) {
        it1 = timestamps[pair.first].erase(it1);
        it2 = uvs[pair.first].erase(it2);
        it3 = uvs_norm[pair.first].erase(it3);
      } else {
        ++it1;
        ++it2;
        ++it3;
      }
    }
  }
}

void Feature::clean_invalid_measurements(const std::vector<double> &invalid_times) {

  // Loop through each of the cameras we have
  for (auto const &pair : timestamps) {

    // Assert that we have all the parts of a measurement
    assert(timestamps[pair.first].size() == uvs[pair.first].size());
    assert(timestamps[pair.first].size() == uvs_norm[pair.first].size());

    // Our iterators
    auto it1 = timestamps[pair.first].begin();
    auto it2 = uvs[pair.first].begin();
    auto it3 = uvs_norm[pair.first].begin();

    // Loop through measurement times, remove ones that are in our timestamps
    while (it1 != timestamps[pair.first].end()) {
      if (std::find(invalid_times.begin(), invalid_times.end(), *it1) != invalid_times.end()) {
        it1 = timestamps[pair.first].erase(it1);
        it2 = uvs[pair.first].erase(it2);
        it3 = uvs_norm[pair.first].erase(it3);
      } else {
        ++it1;
        ++it2;
        ++it3;
      }
    }
  }
}

void Feature::clean_older_measurements(double timestamp) {

  // Loop through each of the cameras we have
  for (auto const &pair : timestamps) {

    // Assert that we have all the parts of a measurement
    assert(timestamps[pair.first].size() == uvs[pair.first].size());
    assert(timestamps[pair.first].size() == uvs_norm[pair.first].size());

    // Our iterators
    auto it1 = timestamps[pair.first].begin();
    auto it2 = uvs[pair.first].begin();
    auto it3 = uvs_norm[pair.first].begin();

    // Loop through measurement times, remove ones that are older then the specified one
    while (it1 != timestamps[pair.first].end()) {
      if (*it1 <= timestamp) {
        it1 = timestamps[pair.first].erase(it1);
        it2 = uvs[pair.first].erase(it2);
        it3 = uvs_norm[pair.first].erase(it3);
      } else {
        ++it1;
        ++it2;
        ++it3;
      }
    }
  }
}

float Feature::calc_parallax(std::vector<Eigen::Matrix<number_t, 3, 3>> &cam_orents_history)
{
  float parallax = -1.;
  auto orien = [](const Eigen::VectorXf &pt, Eigen::Vector3f &e)
  {
    float phi = atan2(pt[1], sqrt(pow(pt[0], 2) + 1));
    float psi = atan2(pt[0], 1);
    e << cos(phi) * sin(psi), sin(phi), cos(phi) * cos(psi);
  };
  assert(timestamps.size() > 0);
  for (auto const &pair : timestamps)
  {
    assert(timestamps[pair.first].size() >= 2);
    int idk = cam_orents_history.size() - 1;
    int id0 = std::max((int)(cam_orents_history.size() - uvs_norm[pair.first].size()), 0); // some feat has meas more than max clones while not update yet
    Eigen::Vector3f e0, ek;
    orien(uvs_norm[pair.first].front(), e0);
    orien(uvs_norm[pair.first].back(), ek);
    Eigen::Matrix<number_t, 3, 3> mRk0 = cam_orents_history[idk].transpose() * cam_orents_history[id0];
    float theta = std::fabs(std::acos(ek.dot(mRk0.cast<float>() * e0)));
    parallax = std::max(parallax, (float)(40 * sin(theta) > 1 ? theta * 180 / M_PI : 0));
  }
  return parallax;
}