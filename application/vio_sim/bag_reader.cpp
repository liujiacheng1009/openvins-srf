#include "bag_reader.h"

BagReader::~BagReader()
{
    if (bag_.isOpen())
    {
        bag_.close();
    }
}

bool BagReader::load(const std::string &bag_path)
{
    bag_.open(bag_path, rosbag::bagmode::Read);
    return true;
}

void BagReader::play(double skip_time)
{
    std::vector<std::string> topics_to_query(subscribed_topics_.begin(), subscribed_topics_.end());
    rosbag::View bag_view(bag_, rosbag::TopicQuery(topics_to_query));

    double bag_start_time = -1;
    for (rosbag::MessageInstance &msg : bag_view)
    {
        if (bag_start_time < 0)
        {
            if (skip_time < 0)
            {
                bag_start_time = 0;
            }
            else
            {
                bag_start_time = skip_time + msg.getTime().toSec();
            }
        }
        if (msg.getTime().toSec() < bag_start_time)
        {
            continue;
        }

        auto iter_cb = mp_topic_2_callback_.find(msg.getTopic());
        if (iter_cb != mp_topic_2_callback_.end())
        {
            iter_cb->second->callback_bag_msg(msg);
        }
    }
}