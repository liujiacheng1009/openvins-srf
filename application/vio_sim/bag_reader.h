#pragma once
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <unordered_map>
#include <unordered_set>
class BagReader
{
protected:
    struct CallBackInterface
    {
        virtual ~CallBackInterface() = default;

        virtual void callback(void *msg) = 0;

        virtual void callback_bag_msg(rosbag::MessageInstance &ros_msg) = 0;
    };

    template <typename Function, typename MsgType>
    struct CallBackBasePtr : public CallBackInterface
    {
        Function f;
        virtual void callback(void *msg) { return f(reinterpret_cast<MsgType *>(msg)); }

        virtual void callback_bag_msg(rosbag::MessageInstance &ros_msg)
        {
            auto msg = ros_msg.instantiate<MsgType>();
            if (msg == nullptr)
                return;
            return f(&(*msg));
        }
    };

    template <typename Function, typename MsgType>
    struct CallBackBaseRef : public CallBackInterface
    {
        Function f;
        virtual void callback(void *msg) { return f(*(reinterpret_cast<MsgType *>(msg))); }

        virtual void callback_bag_msg(rosbag::MessageInstance &ros_msg)
        {
            auto msg = ros_msg.instantiate<MsgType>();
            if (msg == nullptr)
                return;
            return f(*msg);
        }
    };

protected:
    rosbag::Bag bag_;
    std::unordered_map<std::string, std::shared_ptr<CallBackInterface>> mp_topic_2_callback_;
    std::unordered_set<std::string> subscribed_topics_;

public:
    BagReader() = default;

    ~BagReader();

    bool load(std::string const &bag_name);

    template <typename MsgType>
    void register_callback(std::string const &topic,
                           std::function<void(MsgType const *const)> callback)
    {
        std::shared_ptr<CallBackBasePtr<decltype(callback), MsgType>> cb(
            new CallBackBasePtr<decltype(callback), MsgType>());
        cb->f = callback;

        mp_topic_2_callback_[topic] = cb;
        subscribed_topics_.insert(topic);
    }

    void play(double start_time = 0);

protected:
    template <typename MsgType>
    void register_callback(std::string const &topic, void (*callback)(MsgType const *const))
    {
        std::shared_ptr<CallBackBasePtr<decltype(callback), MsgType>> cb(
            new CallBackBasePtr<decltype(callback), MsgType>());
        cb->f = callback;

        mp_topic_2_callback_[topic] = cb;
        subscribed_topics_.insert(topic);
    }

    template <typename MsgType>
    void register_callback(std::string const &topic,
                           std::function<void(std::shared_ptr<MsgType>)> callback)
    {
        std::shared_ptr<CallBackBaseRef<decltype(callback), std::shared_ptr<MsgType>>> cb(
            new CallBackBaseRef<decltype(callback), std::shared_ptr<MsgType>>());
        cb->f = callback;

        mp_topic_2_callback_[topic] = cb;
        subscribed_topics_.insert(topic);
    }
};