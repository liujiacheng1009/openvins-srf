#include "state/FilterWrapper.h"
#include "types/PoseJPL.h"
#include "types/IMU.h"
#include "types/Landmark.h"
#include "state/StateOptions.h"
#include "state/State.h"
#include "update/UpdaterZeroVelocity.h"

#include <Eigen/src/Core/Matrix.h>
#include <gtest/gtest.h>
#include <memory>

using namespace core;
using namespace type;
using namespace msckf;

class FilterSRFTest : public ::testing::Test
{
protected:
    FilterSRFTest() {}
    ~FilterSRFTest() {}

    MatrixX getRandomPositiveDefMat(int k)
    {
        MatrixX P_k = MatrixX::Random(k, k);
        MatrixX P_k_u = P_k.llt().matrixU();
        MatrixX P_k_l = P_k.llt().matrixL();
        return P_k_u * P_k_l;
    }

    bool IsStateAndCovNear(std::shared_ptr<State> state_ekf, std::shared_ptr<State> state_srf)
    {
        constexpr number_t val_tol_thre = std::is_same_v<number_t, float> ? 5.e-6 : 1.e-14;
        constexpr number_t cov_tol_thre = std::is_same_v<number_t, float> ? 1.e-3 : 5.e-13; // float has only 7 digits, and the initial slam feat cov can be larger than 1e3 ,,,
        size_t var_size = FilterWrapper::get_state_variables(state_ekf).size();
        for (size_t i = 0; i < var_size; i++)
        {
            auto &var1 = FilterWrapper::get_state_variables(state_ekf).at(i);
            auto &var2 = FilterWrapper::get_state_variables(state_srf).at(i);
            if (var1->id() != var2->id())
                return false;
            if (var1->size() != var2->size())
                return false;
            if ((var1->value() - var2->value()).norm() > val_tol_thre){
                return false;
            }          
        }
        auto srf_cov = FilterWrapper::get_state_cov(state_srf);
        srf_cov = (srf_cov.transpose() * srf_cov).eval();
        if (((FilterWrapper::get_state_cov(state_ekf) - srf_cov).array().abs() > cov_tol_thre).any())
            return false;
        return true;
    }

    virtual void SetUp()
    {
        imu1 = std::shared_ptr<IMU>(new IMU());
        imu1->set_local_id(0);
        imu1->set_value(VectorX::Random(16, 1));
        imu1_clone = std::shared_ptr<IMU>(new IMU());
        imu1_clone->set_local_id(imu1->id());
        imu1_clone->set_value(imu1->value());
        pose1 = std::shared_ptr<PoseJPL>(new PoseJPL());
        pose1->set_local_id(15);
        pose1->set_value(VectorX::Random(7, 1));
        pose1_clone = std::shared_ptr<PoseJPL>(new PoseJPL());
        pose1_clone->set_local_id(pose1->id());
        pose1_clone->set_value(pose1->value());
        pose2 = std::shared_ptr<PoseJPL>(new PoseJPL());
        pose2->set_local_id(21);
        pose2->set_value(VectorX::Random(7, 1));
        pose2_clone = std::shared_ptr<PoseJPL>(new PoseJPL());
        pose2_clone->set_local_id(pose2->id());
        pose2_clone->set_value(pose2->value());
        lm1 = std::shared_ptr<Landmark>(new Landmark(3));
        lm1->set_local_id(27);
        lm1->set_value(VectorX::Random(3, 1));
        lm1_clone = std::shared_ptr<Landmark>(new Landmark(3));
        lm1_clone->set_local_id(lm1->id());
        lm1_clone->set_value(lm1->value());
        lm2 = std::shared_ptr<Landmark>(new Landmark(3));
        lm2->set_local_id(30);
        lm2->set_value(VectorX::Random(3, 1));
        lm2_clone = std::shared_ptr<Landmark>(new Landmark(3));
        lm2_clone->set_local_id(lm2->id());
        lm2_clone->set_value(lm2->value());
        imu1_dt = std::shared_ptr<type::Vec>(new Vec(1));
        imu1_dt->set_local_id(33);
        imu1_dt->set_value(VectorX::Random(1, 1));
        imu1_dt_clone = std::shared_ptr<type::Vec>(new Vec(1));
        imu1_dt_clone->set_local_id(imu1_dt->id());
        imu1_dt_clone->set_value(imu1_dt->value());
        order.emplace_back(imu1);
        order.emplace_back(pose1);
        order.emplace_back(pose2);
        order.emplace_back(lm1);
        order.emplace_back(lm2);
        order.emplace_back(imu1_dt);
        order_clone.emplace_back(imu1_clone);
        order_clone.emplace_back(pose1_clone);
        order_clone.emplace_back(pose2_clone);
        order_clone.emplace_back(lm1_clone);
        order_clone.emplace_back(lm2_clone);
        order_clone.emplace_back(imu1_dt_clone);
        StateOptions opt;
        opt.do_calib_camera_timeoffset = true;
        state_ekf = std::shared_ptr<State>(new State(opt));
        state_ekf->_imu = imu1;
        state_ekf->_calib_dt_CAMtoIMU = imu1_dt;
        state_srf = std::shared_ptr<State>(new State(opt));
        state_srf->_imu = imu1_clone;
        state_srf->_calib_dt_CAMtoIMU = imu1_dt_clone;
        FilterWrapper::get_state_cov(state_ekf).resizeLike(MatrixX::Zero(34, 34));
        FilterWrapper::get_state_cov(state_srf).resizeLike(MatrixX::Zero(34, 34));
        FilterWrapper::get_state_variables(state_ekf) = order;
        FilterWrapper::get_state_variables(state_srf) = order_clone;
        MatrixX covariance = MatrixX::Zero(34,34);
        covariance.diagonal() = getRandomPositiveDefMat(34).diagonal().array();
        FilterEKF::set_initial_covariance(state_ekf, covariance, order);
        FilterSRF::set_initial_covariance(state_srf, covariance, order_clone);
    }
    virtual void TearDown() {}

public:
    std::shared_ptr<State> state_ekf, state_srf;
    std::shared_ptr<IMU> imu1, imu1_clone;
    std::shared_ptr<Vec> imu1_dt, imu1_dt_clone;
    std::shared_ptr<PoseJPL> pose1, pose2, pose1_clone, pose2_clone;
    std::shared_ptr<Landmark> lm1, lm2, lm1_clone, lm2_clone;
    std::vector<std::shared_ptr<Type>> order, order_clone;
};

TEST_F(FilterSRFTest, Update)
{
    std::vector<std::shared_ptr<Type>> H_order;
    H_order.emplace_back(pose1);
    H_order.emplace_back(pose2);
    H_order.emplace_back(lm1);
    VectorX res = VectorX::Random(30);
    MatrixX H = MatrixX::Random(30, 15);
    MatrixX R = MatrixX::Identity(30, 30);
    FilterEKF::Update(state_ekf, H_order, H, res, R);
    FilterSRF::Update(state_srf, H_order, H, res, R);
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srf), true);
}

TEST_F(FilterSRFTest, InitializeInvertible)
{
    std::shared_ptr<Landmark> new_landmark = std::shared_ptr<Landmark>(new Landmark(3));
    new_landmark->set_value(VectorX::Random(3, 1));
    std::vector<std::shared_ptr<Type>> new_H_order;
    new_H_order.emplace_back(pose1);
    new_H_order.emplace_back(pose2);
    MatrixX H_R_init = MatrixX::Random(3, 12);
    MatrixX H_L_init = MatrixX::Random(3, 3).llt().matrixU();
    MatrixX R = Eigen::Matrix<number_t, 3, 3>::Identity();
    VectorX res = Eigen::Matrix<number_t, 3, 1>::Random();

    FilterEKF::initialize_invertible(state_ekf, new_landmark,
                                     new_H_order, H_R_init, H_L_init,
                                     R, res);
    FilterSRF::initialize_invertible(state_srf, new_landmark,
                                        new_H_order, H_R_init, H_L_init,
                                        R, res);
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srf), true);
}

TEST_F(FilterSRFTest, Propagation)
{
    std::vector<std::shared_ptr<Type>> order_NEW, order_OLD;
    order_NEW.push_back(pose2);
    order_NEW.push_back(lm1);
    order_OLD.push_back(pose1);
    order_OLD.push_back(pose2);
    order_OLD.push_back(lm2);
    MatrixX Phi = MatrixX::Random(9, 15);
    MatrixX Q = MatrixX::Identity(9, 9);
    FilterEKF::Propagation(state_ekf, order_NEW, order_OLD, Phi, Q);
    FilterSRF::Propagation(state_srf, order_NEW, order_OLD, Phi, Q);
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srf), true);
}

TEST_F(FilterSRFTest, Marginalize)
{
    FilterEKF::marginalize(state_ekf, lm1);
    FilterSRF::marginalize(state_srf, lm1_clone);
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srf), true);
}

TEST_F(FilterSRFTest, Clone)
{
    FilterEKF::clone(state_ekf, imu1->pose());
    FilterSRF::clone(state_srf, imu1_clone->pose());
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srf), true);
}

TEST_F(FilterSRFTest, AugmentClone)
{
    Eigen::Matrix<number_t, 3, 1> last_w = Eigen::Matrix<number_t, 3, 1>::Random(3, 1);
    FilterEKF::augment_clone(state_ekf, last_w);
    FilterSRF::augment_clone(state_srf, last_w);
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srf), true);
}

TEST_F(FilterSRFTest, PropagationAndClone)
{
    std::vector<std::shared_ptr<Type>> order_NEW, order_OLD;
    order_NEW.push_back(pose2);
    order_NEW.push_back(lm1);
    order_OLD.push_back(pose1);
    order_OLD.push_back(pose2);
    order_OLD.push_back(lm2);
    MatrixX Phi = MatrixX::Random(9, 15);
    MatrixX Q = MatrixX::Identity(9, 9);
    Eigen::Matrix<number_t, 3, 1> last_w = Eigen::Matrix<number_t, 3, 1>::Random(3, 1);
    FilterEKF::Propagation(state_ekf, order_NEW, order_OLD, Phi, Q);
    FilterEKF::augment_clone(state_ekf, last_w);
    FilterSRF::PropagationAndClone(state_srf, order_NEW, order_OLD, Phi, Q, last_w);
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srf), true);
}

TEST_F(FilterSRFTest, GetMarginalCovariance)
{
    std::vector<std::shared_ptr<Type>> small_variables1;
    small_variables1.emplace_back(pose1);
    small_variables1.emplace_back(lm1);
    std::vector<std::shared_ptr<Type>> small_variables2;
    small_variables2.emplace_back(pose1_clone);
    small_variables2.emplace_back(lm1_clone);
    MatrixX small_cov1 = FilterEKF::get_marginal_covariance(state_ekf, small_variables1);
    MatrixX small_cov2 = FilterSRF::get_marginal_covariance(state_srf, small_variables2);
    EXPECT_EQ(((small_cov1 - small_cov2).array().abs() > 1e-5).any(), false);
}