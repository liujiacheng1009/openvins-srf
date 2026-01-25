#pragma once

#include <vector>
#include <Eigen/Core>
#include "types/Type.h"

class LowpassFilter
{
public:
    explicit LowpassFilter(number_t cutoff_freq_hz);

    void addSample(const Eigen::Matrix<number_t, 3,1> &sample, int64_t timestamp_ns);

    void addWeightedSample(const Eigen::Matrix<number_t, 3,1> &sample, int64_t timestamp_ns, number_t weight);

    Eigen::Matrix<number_t, 3,1> getFilteredData() const { return filtered_data_; }

    int64_t getMostRecentTimestampNs() const { return timestamp_most_recent_update_ns_; }

    bool isInitialized() const { return initialized_; }

    void reset();

private:
    number_t cutoff_time_constant_ = 1.0 / (2.0 * M_PI);
    int64_t timestamp_most_recent_update_ns_ = 0;
    bool initialized_ = false;

    static constexpr number_t kMinTimestepS = 0.001f;

    static constexpr number_t kMaxTimestepS = 1.00f;

    Eigen::Matrix<number_t, 3,1> filtered_data_ = Eigen::Matrix<number_t, 3,1>(0, 0, 0);
};

class IsStaticCounter
{
public:
    explicit IsStaticCounter(int min_static_frames_threshold)
        : min_static_frames_threshold_(min_static_frames_threshold), consecutive_static_frames_(0) {}

    void appendFrame(bool is_static, int64_t timestamp_ns)
    {
        if (is_static)
        {
            ++consecutive_static_frames_;
            if (consecutive_static_frames_ > 1e5)
                consecutive_static_frames_ = 2 * min_static_frames_threshold_;
        }
        else
        {
            consecutive_static_frames_ = 0;
        }
        const number_t delta_s =
            static_cast<number_t>(timestamp_ns - timestamp_most_recent_update_ns_) * 1e-9;
        if (delta_s > kMaxTimeResetSec)
        {
            consecutive_static_frames_ = 0;
        }
        timestamp_most_recent_update_ns_ = timestamp_ns;
    }

    int getStaticFrames() { return consecutive_static_frames_; }

    bool isRecentlyStatic() const
    {
        return consecutive_static_frames_ >= min_static_frames_threshold_;
    }

    void reset()
    {
        consecutive_static_frames_ = 0;
        timestamp_most_recent_update_ns_ = 0;
    }

private:
    int min_static_frames_threshold_ = 50;
    int consecutive_static_frames_ = 0;
    int64_t timestamp_most_recent_update_ns_ = 0;
    static constexpr number_t kMaxTimeResetSec = 1.00f;
};