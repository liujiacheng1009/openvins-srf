#include "lowpass_filter.h"
#include <cmath>

LowpassFilter::LowpassFilter(number_t cutoff_freq_hz)
    : cutoff_time_constant_(1.0 / (2.0 * M_PI * cutoff_freq_hz)), initialized_(false)
{
    reset();
}

void LowpassFilter::addSample(const Eigen::Matrix<number_t, 3,1> &sample, int64_t timestamp_ns)
{
    addWeightedSample(sample, timestamp_ns, 1.0);
}

void LowpassFilter::addWeightedSample(const Eigen::Matrix<number_t, 3,1> &sample, int64_t timestamp_ns,
                                      number_t weight)
{
    if (!initialized_)
    {
        // Initialize filter state
        filtered_data_ = {sample[0], sample[1], sample[2]};
        timestamp_most_recent_update_ns_ = timestamp_ns;
        initialized_ = true;
        return;
    }

    if (timestamp_ns < timestamp_most_recent_update_ns_)
    {
        timestamp_most_recent_update_ns_ = timestamp_ns;
        return;
    }

    const number_t delta_s =
        static_cast<number_t>(timestamp_ns - timestamp_most_recent_update_ns_) * 1e-9;
    if (delta_s <= kMinTimestepS || delta_s > kMaxTimestepS)
    {
        timestamp_most_recent_update_ns_ = timestamp_ns;
        return;
    }

    const number_t weighted_delta_secs = static_cast<number_t>(weight) * delta_s;

    const number_t alpha =
        weighted_delta_secs / static_cast<number_t>(cutoff_time_constant_ + weighted_delta_secs);

    for (int i = 0; i < 3; ++i)
    {
        filtered_data_[i] = (1.0 - alpha) * filtered_data_[i] + alpha * sample[i];
    }
    timestamp_most_recent_update_ns_ = timestamp_ns;
}

void LowpassFilter::reset()
{
    initialized_ = false;
    filtered_data_ = Eigen::Matrix<number_t, 3,1>(0, 0, 0);
}