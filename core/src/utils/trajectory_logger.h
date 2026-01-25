#pragma once

#include <cstdio>
#include <fstream>
#include <type_traits>
#include <Eigen/Eigen>

namespace core
{

    /**
     * @brief The TrajectoryLogger class: used to log vio trajectory
     */
    class TrajectoryLogger
    {
    protected:
        std::fstream traj_file_;
        bool use_self_format_ = false;
        char line_text_[256]; /*< internal line buffer, a single line should be short that 256 bytes*/

    public:
        TrajectoryLogger() = default;

        void open(std::string const &filename, bool use_self_format);

        bool isOpen();

        void close();

        template <typename T>
        int log(double timestamp_s, Eigen::Matrix<T, 3, 1> const &p_w_b, Eigen::Quaternion<T> const &q_w_b)
        {
            static_assert(std::is_floating_point<T>::value, "only support floating points");
            return log(timestamp_s, p_w_b(0), p_w_b(1), p_w_b(2), q_w_b.x(), q_w_b.y(), q_w_b.z(),
                       q_w_b.w());
        }

        template <typename T>
        int log(double timestamp_s, Eigen::Matrix<T, 3, 1> const &p_w_b, Eigen::Quaternion<T> const &q_w_b,
                Eigen::Matrix<T, 3, 1> const &v_w_b, float time_cost, int status, float quality,
                float cpu_usage = -1.0f, float mem_usage = -1.0f, float recall_ratio = -1.0f,
                float global_time = -1.0f)
        {
            static_assert(std::is_floating_point<T>::value, "only support floating points");
            return log(timestamp_s, p_w_b(0), p_w_b(1), p_w_b(2), q_w_b.x(), q_w_b.y(), q_w_b.z(),
                       q_w_b.w(), v_w_b(0), v_w_b(1), v_w_b(2), time_cost, status, quality, cpu_usage,
                       mem_usage, recall_ratio, global_time);
        }

        int log(double timestamp_s, float px, float py, float pz, float qx, float qy, float qz, float qw);

        int log(double timestamp_s, float px, float py, float pz, float qx, float qy, float qz, float qw,
                float vx, float vy, float vz, float time_cost, int status, float quality,
                float cpu_usage = -1.0f, float mem_usage = -1.0f, float recall_ratio = -1.0f,
                float global_time = -1.0f);

        void flush();

    protected:
        inline int log_std_format(double timestamp_s, float px, float py, float pz, float qx, float qy,
                                  float qz, float qw);

        inline int log_self_format(double timestamp_s, float px, float py, float pz, float qx, float qy,
                                   float qz, float qw, float vx, float vy, float vz, float time_cost,
                                   int status, float quality, float cpu_usage = -1.0f,
                                   float mem_usage = -1.0f, float recall_ratio = -1.0f,
                                   float global_time = -1.0f);
    };

} // namespace core
