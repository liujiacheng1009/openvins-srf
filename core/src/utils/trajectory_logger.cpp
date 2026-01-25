#include "trajectory_logger.h"
#include <glog/logging.h>

namespace core
{

  void TrajectoryLogger::open(const std::string &filename, bool use_self_format)
  {
    if (traj_file_.is_open())
    {
      traj_file_.close();
    }

    traj_file_.open(filename, std::ios::out);
    if (!traj_file_.is_open())
    {
      LOG(WARNING) << "failed to open " << filename;
      return;
    }
    else
    {
      LOG(WARNING) << "success opened " << filename << " for trajectory logging";
    }
    use_self_format_ = use_self_format;

    if (use_self_format)
    {
      traj_file_ << "# timestamp_in_s tx ty tz qx qy qz qw vx vy vz total_cost system_status "
                    "cpu_load mem_usage vel_quality";
    }
    else
    {
      traj_file_ << "# timestamp_in_s tx ty tz qx qy qz qw";
    }
  }

  bool TrajectoryLogger::isOpen() { return traj_file_.is_open(); }

  void TrajectoryLogger::close()
  {
    if (traj_file_.is_open())
    {
      traj_file_.close();
    }
  }

  int TrajectoryLogger::log(double timestamp_s, float px, float py, float pz, float qx, float qy,
                            float qz, float qw)
  {
    if (use_self_format_)
    {
      return log_self_format(timestamp_s, px, py, pz, qx, qy, qz, qw, 0, 0, 0, 0, 0, 0, -1, -1);
    }

    return log_std_format(timestamp_s, px, py, pz, qx, qy, qz, qw);
  }

  int TrajectoryLogger::log(double timestamp_s, float px, float py, float pz, float qx, float qy,
                            float qz, float qw, float vx, float vy, float vz, float time_cost,
                            int status, float quality, float cpu_usage, float mem_usage,
                            float recall_ratio, float global_time)
  {
    if (use_self_format_)
    {
      return log_self_format(timestamp_s, px, py, pz, qx, qy, qz, qw, vx, vy, vz, time_cost, status,
                             quality, cpu_usage, mem_usage, recall_ratio, global_time);
    }

    return log_std_format(timestamp_s, px, py, pz, qx, qy, qz, qw);
  }

  void TrajectoryLogger::flush()
  {
    if (traj_file_.is_open())
    {
      traj_file_.flush();
    }
  }

  int TrajectoryLogger::log_std_format(double timestamp_s, float px, float py, float pz, float qx,
                                       float qy, float qz, float qw)
  {
    if (!traj_file_.is_open())
    {
      return -1;
    }
    snprintf(line_text_, sizeof(line_text_), "%f %f %f %f %f %f %f %f", timestamp_s, px, py, pz, qx,
             qy, qz, qw);
    traj_file_ << std::endl
               << line_text_;
    return 0;
  }

  int TrajectoryLogger::log_self_format(double timestamp_s, float px, float py, float pz, float qx,
                                        float qy, float qz, float qw, float vx, float vy, float vz,
                                        float time_cost, int status, float quality, float cpu_usage,
                                        float mem_usage, float recall_ratio, float global_time)
  {
    if (!traj_file_.is_open())
    {
      return -1;
    }
    // clang-format off
	snprintf(line_text_, sizeof(line_text_),
			 "\n%f %f %f %f %f %f %f %f "
			 "%f %f %f %f %d "
			 "%f %f %f %f %f",
			 timestamp_s, px, py, pz, qx, qy, qz, qw,
			 vx, vy, vz, time_cost, status,
			 cpu_usage, mem_usage, quality, recall_ratio, global_time);
    // clang-format on

    traj_file_ << line_text_;
    return 0;
  }

} // namespace core
