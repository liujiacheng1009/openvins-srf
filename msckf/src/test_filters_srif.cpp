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
using namespace type;

class FilterSRIFTest : public ::testing::Test
{
protected:
    FilterSRIFTest() {}
    ~FilterSRIFTest() {}
    //@liujiacheng need double check
    constexpr static number_t val_tol_thre = std::is_same_v<number_t, float> ? 1.e-5 : 2.e-14;
    constexpr static number_t cov_tol_thre = std::is_same_v<number_t, float> ? 5.e-2 : 5.e-11; // float has only 7 digits, and the initial slam feat cov can be larger than 1e3 ,,,

    MatrixX getRandomPositiveDefMat(int k)
    {
        MatrixX P_k = MatrixX::Random(k, k);
        MatrixX P_k_u = P_k.llt().matrixU();
        MatrixX P_k_l = P_k.llt().matrixL();
        return P_k_u * P_k_l;
    }

    bool CompareVariable(std::shared_ptr<Type> var1, std::shared_ptr<Type> var2){
        if(var1->id() != var2->id()) return false;
        if(var1->size() != var2->size()) return false;
        if(var1->value() != var2->value()) return false;
        return true;
    }

    bool CompareMat(MatrixX mat1, MatrixX mat2, number_t thre = val_tol_thre)
    {
        if (((mat1 - mat2).array().abs() > thre).any())
        {
            std::cout << (mat1 - mat2).array().abs().maxCoeff() << " > " << thre << "!! \n";
            return false;
        }
        return true;
    }

    bool IsStateAndCovNear(std::shared_ptr<State> state_ekf, std::shared_ptr<State> state_srif)
    {
        size_t var_size = FilterWrapper::get_state_variables(state_ekf).size();
        for (size_t i = 0; i < var_size; i++)
        {
            auto &var1 = FilterWrapper::get_state_variables(state_ekf).at(i);
            auto &var2 = FilterWrapper::get_state_variables(state_srif).at(i);
            if (var1->id() != var2->id())
                return false;
            if (var1->size() != var2->size())
                return false;
            if ((var1->value() - var2->value()).norm() > val_tol_thre){
                std::cout << (var1->value() - var2->value()).norm() << " > " << val_tol_thre <<"!! \n";
                return false;
            }          
        }
        auto srif_cov = FilterWrapper::get_state_cov(state_srif);
        srif_cov = srif_cov.transpose() * srif_cov;
        if (((FilterWrapper::get_state_cov(state_ekf).inverse() - srif_cov).array().abs() > cov_tol_thre).any()){
            std::cout << (FilterWrapper::get_state_cov(state_ekf).inverse() - srif_cov).array().abs().maxCoeff() << " > " << cov_tol_thre <<" !! \n";
            return false;
        }
            
        return true;
    }

    virtual void SetUp()
    {
        lm1 = std::shared_ptr<Landmark>(new Landmark(3));
        lm1->set_local_id(0);
        lm1->set_value(VectorX::Random(3, 1));
        lm1_clone = std::shared_ptr<Landmark>(new Landmark(3));
        lm1_clone->set_local_id(lm1->id());
        lm1_clone->set_value(lm1->value());
        lm2 = std::shared_ptr<Landmark>(new Landmark(3));
        lm2->set_local_id(3);
        lm2->set_value(VectorX::Random(3, 1));
        lm2_clone = std::shared_ptr<Landmark>(new Landmark(3));
        lm2_clone->set_local_id(lm2->id());
        lm2_clone->set_value(lm2->value());
        imu1_dt = std::shared_ptr<type::Vec>(new Vec(1));
        imu1_dt->set_local_id(6);
        imu1_dt->set_value(VectorX::Random(1, 1));
        imu1_dt_clone = std::shared_ptr<type::Vec>(new Vec(1));
        imu1_dt_clone->set_local_id(imu1_dt->id());
        imu1_dt_clone->set_value(imu1_dt->value());
        pose1 = std::shared_ptr<PoseJPL>(new PoseJPL());
        pose1->set_local_id(7);
        pose1->set_value(VectorX::Random(7, 1));
        pose1_clone = std::shared_ptr<PoseJPL>(new PoseJPL());
        pose1_clone->set_local_id(pose1->id());
        pose1_clone->set_value(pose1->value());
        pose2 = std::shared_ptr<PoseJPL>(new PoseJPL());
        pose2->set_local_id(13);
        pose2->set_value(VectorX::Random(7, 1));
        pose2_clone = std::shared_ptr<PoseJPL>(new PoseJPL());
        pose2_clone->set_local_id(pose2->id());
        pose2_clone->set_value(pose2->value());
        imu1 = std::shared_ptr<IMU>(new IMU());
        imu1->set_local_id(19);
        imu1->set_value(VectorX::Random(16, 1));
        imu1_clone = std::shared_ptr<IMU>(new IMU());
        imu1_clone->set_local_id(imu1->id());
        imu1_clone->set_value(imu1->value());

        order.emplace_back(lm1);
        order.emplace_back(lm2);
        order.emplace_back(imu1_dt);
        order.emplace_back(pose1);
        order.emplace_back(pose2);
        order.emplace_back(imu1);
        order_clone.emplace_back(lm1_clone);
        order_clone.emplace_back(lm2_clone);
        order_clone.emplace_back(imu1_dt_clone);
        order_clone.emplace_back(pose1_clone);
        order_clone.emplace_back(pose2_clone);
        order_clone.emplace_back(imu1_clone);

        StateOptions opt;
        opt.do_calib_camera_timeoffset = false;
        state_ekf = std::shared_ptr<State>(new State(opt));
        state_ekf->_imu = imu1;
        state_ekf->_calib_dt_CAMtoIMU = imu1_dt;
        state_srif = std::shared_ptr<State>(new State(opt));
        state_srif->_imu = imu1_clone;
        state_srif->_calib_dt_CAMtoIMU = imu1_dt_clone;
        FilterWrapper::get_state_cov(state_ekf).resizeLike(MatrixX::Zero(34, 34));
        FilterWrapper::get_state_cov(state_srif).resizeLike(MatrixX::Zero(34, 34));
        FilterWrapper::get_state_variables(state_ekf) = order;
        FilterWrapper::get_state_variables(state_srif) = order_clone;
        MatrixX covariance = MatrixX::Zero(34,34);
        covariance.diagonal() = getRandomPositiveDefMat(34).diagonal().array();
        FilterEKF::set_initial_covariance(state_ekf, covariance, order);
        FilterSRIF::set_initial_covariance(state_srif, covariance, order_clone);
        state_srif->_clones_IMU[state_srif->_timestamp] = std::dynamic_pointer_cast<PoseJPL>(state_srif->_imu->pose()->clone());
    }
    virtual void TearDown() {}

public:
    std::shared_ptr<State> state_ekf, state_srif;
    std::shared_ptr<IMU> imu1, imu1_clone;
    std::shared_ptr<Vec> imu1_dt, imu1_dt_clone;
    std::shared_ptr<PoseJPL> pose1, pose2, pose1_clone, pose2_clone;
    std::shared_ptr<Landmark> lm1, lm2, lm1_clone, lm2_clone;
    std::vector<std::shared_ptr<Type>> order, order_clone;
};

TEST_F(FilterSRIFTest, SetInitialCovariance)
{
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srif), true);
}

TEST_F(FilterSRIFTest, Marginalize)
{
    FilterEKF::marginalize(state_ekf, lm2);
    FilterSRIF::marginalize(state_srif, lm2_clone);
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srif), true);
    FilterEKF::marginalize(state_ekf, imu1);
    FilterSRIF::marginalize(state_srif, imu1_clone);
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srif), true);
    EXPECT_EQ(CompareVariable(state_srif->_clones_IMU[state_srif->_timestamp], state_srif->_imu->pose()), true);
}

TEST_F(FilterSRIFTest, PropagationAndClone)
{
    std::vector<std::shared_ptr<Type>> order_NEW, order_OLD;
    order_NEW.push_back(imu1);
    order_OLD.push_back(pose1);
    order_OLD.push_back(pose2);
    order_OLD.push_back(imu1);

    std::vector<std::shared_ptr<Type>> order_NEW_clone, order_OLD_clone;
    order_NEW_clone.push_back(imu1_clone);
    order_OLD_clone.push_back(pose1_clone);
    order_OLD_clone.push_back(pose2_clone);
    order_OLD_clone.push_back(imu1_clone);

    MatrixX Phi = MatrixX::Random(15, 27);
    MatrixX Q = MatrixX::Identity(15, 15);
    Eigen::Matrix<number_t, 3, 1> last_w = Eigen::Matrix<number_t, 3, 1>::Random(3, 1);
    FilterEKF::Propagation(state_ekf, order_NEW, order_OLD, Phi, Q);
    FilterEKF::augment_clone(state_ekf, last_w);
    std::shared_ptr<Type> aug_pose = FilterWrapper::get_state_variables(state_ekf).back();
    FilterEKF::marginalize(state_ekf, aug_pose);
    std::shared_ptr<PoseJPL>& last_clone_pose = FilterWrapper::get_last_clone_pose(state_srif);
    last_clone_pose = std::dynamic_pointer_cast<PoseJPL>(imu1_clone->pose()->clone());
    std::shared_ptr<PoseJPL> last_clone_pose_tmp = last_clone_pose;
    FilterSRIF::PropagationAndClone(state_srif, order_NEW_clone, order_OLD_clone, Phi, Q, last_w);
    FilterSRIF::marginalize(state_srif, last_clone_pose_tmp);
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srif), true);
    EXPECT_EQ(CompareVariable(state_srif->_clones_IMU[state_srif->_timestamp], state_srif->_imu->pose()), true);
    EXPECT_EQ(CompareVariable(FilterWrapper::get_last_clone_pose(state_srif), state_srif->_imu->pose()), true);
    EXPECT_EQ(FilterWrapper::get_state_cov(state_srif).isUpperTriangular(), true);
}

TEST_F(FilterSRIFTest, Update)
{
    std::vector<std::shared_ptr<Type>> H_order, H_order_clone;
    H_order.emplace_back(pose1);
    H_order.emplace_back(pose2);
    H_order.emplace_back(lm1);
    H_order_clone.emplace_back(pose1_clone);
    H_order_clone.emplace_back(pose2_clone);
    H_order_clone.emplace_back(lm1_clone);
    VectorX res = VectorX::Random(50);
    MatrixX H = MatrixX::Random(50, 15);
    MatrixX R = MatrixX::Identity(50, 50);
    FilterEKF::Update(state_ekf, H_order, H, res, R);
    MatrixX H1 = H.topRows(30);
    MatrixX H2 = H.bottomRows(20);
    MatrixX res1 = res.topRows(30);
    MatrixX res2 = res.bottomRows(20);
    MatrixX R1 = MatrixX::Identity(30, 30);
    MatrixX R2 = MatrixX::Identity(20, 20);
    FilterSRIF::Update(state_srif, H_order_clone, H1, res1, R1);
    FilterSRIF::Update(state_srif, H_order_clone, H2, res2, R2);
    FilterSRIF::Update(state_srif);
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srif), true);
    EXPECT_EQ(FilterWrapper::get_state_cov(state_srif).isUpperTriangular(), true);
}

TEST_F(FilterSRIFTest, AnchorChange)
{
    std::vector<std::shared_ptr<Type>> order_NEW, order_OLD;
    order_NEW.push_back(pose1);
    order_NEW.push_back(pose2);
    order_OLD.push_back(lm2);
    order_OLD.push_back(pose1);
    order_OLD.push_back(pose2);
    MatrixX Phi = MatrixX::Random(12, 15);
    MatrixX Q = MatrixX::Zero(12, 12);
    FilterEKF::Propagation(state_ekf, order_NEW, order_OLD, Phi, Q);
    MatrixX Phi_clone = Phi;
    MatrixX Phi_newvar_inv = (Phi.rightCols(12)).colPivHouseholderQr().solve(Eigen::Matrix<number_t, 12, 12>::Identity());
    Phi_clone.rightCols(12) = Phi_newvar_inv;
    Phi_clone.leftCols(3) = -Phi_newvar_inv * Phi_clone.leftCols(3);
    FilterSRIF::AnchorChange(state_srif, order_NEW, order_OLD, Phi_clone);
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srif), true);
    EXPECT_EQ(FilterWrapper::get_state_cov(state_srif).isUpperTriangular(), true);
}

TEST_F(FilterSRIFTest, Initialize)
{
    std::shared_ptr<Landmark> new_landmark = std::shared_ptr<Landmark>(new Landmark(3));
    new_landmark->set_value(VectorX::Random(3, 1));
    std::shared_ptr<Landmark> new_landmark_clone = std::shared_ptr<Landmark>(new Landmark(3));
    new_landmark_clone->set_value(new_landmark->value());
    std::vector<std::shared_ptr<Type>> new_H_order, new_H_order_clone;
    new_H_order.emplace_back(pose1);
    new_H_order.emplace_back(pose2);
    new_H_order_clone.emplace_back(pose1_clone);
    new_H_order_clone.emplace_back(pose2_clone);
    MatrixX H_R = MatrixX::Random(20, 12);
    MatrixX H_L = MatrixX::Random(20, 3);
    MatrixX R = Eigen::Matrix<number_t, 20, 20>::Identity();
    VectorX res = Eigen::Matrix<number_t, 20, 1>::Random();
    FilterEKF::initialize(state_ekf, new_landmark, new_H_order, H_R, H_L, R, res, 1.0);
    auto &lms = state_srif->_features_SLAM;
    lms.insert({1, lm1_clone});
    lms.insert({2, lm2_clone});
    FilterSRIF::initialize(state_srif, new_landmark_clone, new_H_order_clone, H_R, H_L, R, res, 1.0);
    FilterSRIF::Update(state_srif);
    auto srif_cov = FilterWrapper::get_state_cov(state_srif);
    auto ekf_cov = FilterWrapper::get_state_cov(state_ekf);
    auto ekf_cov_inv = ekf_cov.inverse();
    MatrixX ekf_cov_inv_shuffle = MatrixX::Zero(ekf_cov_inv.rows(), ekf_cov_inv.cols());
    int krows = ekf_cov_inv.rows() - new_landmark_clone->id() - LDim::Landmark;
    ekf_cov_inv_shuffle.topLeftCorner(new_landmark_clone->id(), new_landmark_clone->id()) = ekf_cov_inv.topLeftCorner(new_landmark_clone->id(), new_landmark_clone->id());
    ekf_cov_inv_shuffle.bottomLeftCorner(krows, new_landmark_clone->id()) = ekf_cov_inv.block(new_landmark_clone->id(), 0, krows, new_landmark_clone->id());
    ekf_cov_inv_shuffle.topRightCorner(new_landmark_clone->id(), krows) = ekf_cov_inv.block(0, new_landmark_clone->id(), new_landmark_clone->id(), krows);
    ekf_cov_inv_shuffle.bottomRightCorner(krows, krows) = ekf_cov_inv.block(new_landmark_clone->id(), new_landmark_clone->id(), krows, krows);
    ekf_cov_inv_shuffle.block(new_landmark_clone->id(), 0, LDim::Landmark, new_landmark_clone->id()) = ekf_cov_inv.bottomLeftCorner(LDim::Landmark, new_landmark_clone->id());
    ekf_cov_inv_shuffle.block(0, new_landmark_clone->id(), new_landmark_clone->id(), LDim::Landmark) = ekf_cov_inv.topRightCorner(new_landmark_clone->id(), LDim::Landmark);
    ekf_cov_inv_shuffle.block(new_landmark_clone->id(), new_landmark_clone->id(), LDim::Landmark, LDim::Landmark) = ekf_cov_inv.bottomRightCorner(LDim::Landmark, LDim::Landmark);
    ekf_cov_inv_shuffle.block(new_landmark_clone->id() + LDim::Landmark, new_landmark_clone->id(), krows, LDim::Landmark) = ekf_cov_inv.rightCols(LDim::Landmark).middleRows(new_landmark_clone->id(), krows);
    ekf_cov_inv_shuffle.block(new_landmark_clone->id(), new_landmark_clone->id() + LDim::Landmark, LDim::Landmark, krows) = ekf_cov_inv.bottomRows(LDim::Landmark).middleCols(new_landmark_clone->id(), krows);
    number_t tol = std::is_same_v<number_t, float> ? 6.e-4 : 2.e-12;
    EXPECT_EQ(CompareMat(srif_cov.transpose() * srif_cov, ekf_cov_inv_shuffle, tol), true);
    EXPECT_EQ(FilterWrapper::get_state_cov(state_srif).isUpperTriangular(), true);
}

TEST_F(FilterSRIFTest, InitializeInvertible)
{
    std::shared_ptr<Landmark> new_landmark = std::shared_ptr<Landmark>(new Landmark(3));
    new_landmark->set_value(VectorX::Random(3, 1));
    std::shared_ptr<Landmark> new_landmark_clone = std::shared_ptr<Landmark>(new Landmark(3));
    new_landmark_clone->set_value(new_landmark->value());
    std::vector<std::shared_ptr<Type>> new_H_order, new_H_order_clone;
    new_H_order.emplace_back(pose1);
    new_H_order.emplace_back(pose2);
    new_H_order_clone.emplace_back(pose1_clone);
    new_H_order_clone.emplace_back(pose2_clone);
    MatrixX H_R_init = MatrixX::Random(3, 12);
    MatrixX H_L_init = MatrixX::Random(3, 3).llt().matrixU();
    MatrixX R = Eigen::Matrix<number_t, 3, 3>::Identity();
    VectorX res = Eigen::Matrix<number_t, 3, 1>::Random();
    FilterEKF::initialize_invertible(state_ekf, new_landmark,
                                     new_H_order, H_R_init, H_L_init,
                                     R, res);
    auto &lms = state_srif->_features_SLAM;
    lms.insert({1, lm1_clone});
    lms.insert({2, lm2_clone});
    FilterSRIF::initialize_invertible(state_srif, new_landmark_clone,
                                      new_H_order_clone, H_R_init, H_L_init,
                                      R, res);
    auto srif_cov = FilterWrapper::get_state_cov(state_srif);
    auto ekf_cov = FilterWrapper::get_state_cov(state_ekf);
    auto ekf_cov_inv = ekf_cov.inverse();
    MatrixX ekf_cov_inv_shuffle = MatrixX::Zero(ekf_cov_inv.rows(), ekf_cov_inv.cols());
    int krows = ekf_cov_inv.rows() - new_landmark_clone->id() - LDim::Landmark;
    ekf_cov_inv_shuffle.topLeftCorner(new_landmark_clone->id(), new_landmark_clone->id()) = ekf_cov_inv.topLeftCorner(new_landmark_clone->id(), new_landmark_clone->id());
    ekf_cov_inv_shuffle.bottomLeftCorner(krows, new_landmark_clone->id()) = ekf_cov_inv.block(new_landmark_clone->id(), 0, krows, new_landmark_clone->id());
    ekf_cov_inv_shuffle.topRightCorner(new_landmark_clone->id(), krows) = ekf_cov_inv.block(0, new_landmark_clone->id(), new_landmark_clone->id(), krows);
    ekf_cov_inv_shuffle.bottomRightCorner(krows, krows) = ekf_cov_inv.block(new_landmark_clone->id(), new_landmark_clone->id(), krows, krows);
    ekf_cov_inv_shuffle.block(new_landmark_clone->id(), 0, LDim::Landmark, new_landmark_clone->id()) = ekf_cov_inv.bottomLeftCorner(LDim::Landmark, new_landmark_clone->id());
    ekf_cov_inv_shuffle.block(0, new_landmark_clone->id(), new_landmark_clone->id(), LDim::Landmark) = ekf_cov_inv.topRightCorner(new_landmark_clone->id(), LDim::Landmark);
    ekf_cov_inv_shuffle.block(new_landmark_clone->id(), new_landmark_clone->id(), LDim::Landmark, LDim::Landmark) = ekf_cov_inv.bottomRightCorner(LDim::Landmark, LDim::Landmark);
    ekf_cov_inv_shuffle.block(new_landmark_clone->id() + LDim::Landmark, new_landmark_clone->id(), krows, LDim::Landmark) = ekf_cov_inv.rightCols(LDim::Landmark).middleRows(new_landmark_clone->id(), krows);
    ekf_cov_inv_shuffle.block(new_landmark_clone->id(), new_landmark_clone->id() + LDim::Landmark, LDim::Landmark, krows) = ekf_cov_inv.bottomRows(LDim::Landmark).middleCols(new_landmark_clone->id(), krows);
    number_t tol = std::is_same_v<number_t, float> ? 1.e-4 : 1.e-12;
    EXPECT_EQ(CompareMat(srif_cov.transpose() * srif_cov, ekf_cov_inv_shuffle, tol), true);
    EXPECT_EQ(FilterWrapper::get_state_cov(state_srif).isUpperTriangular(), true);
}

TEST_F(FilterSRIFTest, GetMarginalCovariance)
{
    std::vector<std::shared_ptr<Type>> small_variables1;
    small_variables1.emplace_back(imu1);
    small_variables1.emplace_back(imu1_dt);
    std::vector<std::shared_ptr<Type>> small_variables2;
    small_variables2.emplace_back(imu1_clone);
    small_variables2.emplace_back(imu1_dt_clone);
    MatrixX small_cov1 = FilterEKF::get_marginal_covariance(state_ekf, small_variables1);
    auto &lms = state_srif->_features_SLAM;
    lms.insert({1, lm1_clone});
    lms.insert({2, lm2_clone});
    FilterSRIF::get_chicheck_sqrtcov(state_srif, true);
    MatrixX small_cov2 = FilterSRIF::get_marginal_covariance(state_srif, small_variables2);
    EXPECT_EQ(CompareMat(small_cov1, small_cov2), true);
}

TEST_F(FilterSRIFTest, GetHxPmargHxT)
{
    std::vector<std::shared_ptr<Type>> Hx_order, Hx_order_clone;
    Hx_order.push_back(pose2);
    Hx_order.push_back(imu1_dt);
    Hx_order.push_back(imu1);
    Hx_order_clone.push_back(pose2_clone);
    Hx_order_clone.push_back(imu1_dt_clone);
    Hx_order_clone.push_back(imu1_clone);
    MatrixX H_x = MatrixX::Random(30, 22);
    MatrixX H_x_clone = H_x;
    auto &lms = state_srif->_features_SLAM;
    lms.insert({1, lm1_clone});
    lms.insert({2, lm2_clone});
    MatrixX P_marg = FilterEKF::get_marginal_covariance(state_ekf, Hx_order);
    MatrixX S_ekf = H_x * P_marg * H_x.transpose();
    FilterSRIF::get_chicheck_sqrtcov(state_srif, true);
    MatrixX S_srif = FilterSRIF::get_HxPmargHxT(state_srif, Hx_order_clone, H_x_clone);
    EXPECT_EQ(CompareMat(S_ekf, S_srif), true);
}

TEST_F(FilterSRIFTest, MarginalizeSlam)
{
    auto &lms = state_ekf->_features_SLAM;
    lms.insert({10001, lm1});
    lms.insert({10002, lm2});
    lm2->should_marg = true;
    auto &lms_clone = state_srif->_features_SLAM;
    lms_clone.insert({10001, lm1_clone});
    lms_clone.insert({10002, lm2_clone});
    lm2_clone->should_marg = true;
    FilterEKF::marginalize_slam(state_ekf);
    FilterSRIF::marginalize_slam(state_srif);
    EXPECT_EQ(IsStateAndCovNear(state_ekf, state_srif), true);
    EXPECT_EQ(CompareVariable(state_srif->_clones_IMU[state_srif->_timestamp], state_srif->_imu->pose()), true);
}