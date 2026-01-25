#pragma once
#include <atomic>
#include <type_traits>

struct GlobalFlagPool
{
    GlobalFlagPool() = delete;

    static void setJointUpdate(const bool use_joint_update)
    {
        use_joint_update_.store(use_joint_update, std::memory_order_seq_cst);
    }
    static bool getJointUpdate() { return use_joint_update_.load(std::memory_order_seq_cst); }

    static void setJointAnchorChange(const bool use_joint_anchor_change)
    {
        use_joint_anchor_change_.store(use_joint_anchor_change, std::memory_order_seq_cst);
    }
    static bool getJointAnchorChange() { return use_joint_anchor_change_.load(std::memory_order_seq_cst); }

protected:
    static inline std::atomic<bool> use_joint_update_{false};
    static inline std::atomic<bool> use_joint_anchor_change_{false};
};