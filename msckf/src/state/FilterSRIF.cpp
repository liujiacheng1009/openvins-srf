#include "FilterSRIF.h"

#include "state/State.h"

#include "types/Landmark.h"
#include "utils/colors.h"
#include "utils/print.h"

#include <boost/math/distributions/chi_squared.hpp>

using namespace core;
using namespace type;
using namespace msckf;

void FilterSRIF::set_initial_last_clone_pose(std::shared_ptr<State> state)
{
    std::shared_ptr<PoseJPL> new_clone = std::dynamic_pointer_cast<PoseJPL>(state->_imu->pose()->clone());
    state->last_clone_pose = new_clone;
}

void FilterSRIF::PropagationAndClone(std::shared_ptr<State> state, const std::vector<std::shared_ptr<type::Type>> &order_NEW,
                                     const std::vector<std::shared_ptr<type::Type>> &order_OLD, const MatrixX &Phi,
                                     const MatrixX &Q, Eigen::Matrix<number_t, 3, 1> &last_w)
{
    // We need at least one old and new variable
    if (order_NEW.empty() || order_OLD.empty())
    {
        PRINT_ERROR(RED "FilterSRIF::Propagation() - Called with empty variable arrays!\n" RESET);
        std::exit(EXIT_FAILURE);
    }
    MatrixX &state_U = state->_Cov;
    MatrixX &state_res = state->_res;
    int size_order_NEW = order_NEW.at(0)->size();
    for (size_t i = 0; i < order_NEW.size() - 1; i++)
    {
        size_order_NEW += order_NEW.at(i + 1)->size();
    }
    // Size of the old phi matrix
    int size_order_OLD = order_OLD.at(0)->size();
    for (size_t i = 0; i < order_OLD.size() - 1; i++)
    {
        size_order_OLD += order_OLD.at(i + 1)->size();
    }
    assert(size_order_NEW == Phi.rows());
    assert(size_order_OLD == Phi.cols());
    assert(size_order_NEW == Q.cols());
    assert(size_order_NEW == Q.rows());
    MatrixX Q_sqrt_inv = MatrixX::Zero(Q.rows(), Q.cols());
    VectorX Q_sqrt_inv_diag = Q.diagonal().array().sqrt().inverse();
    Q_sqrt_inv.diagonal() = Q_sqrt_inv_diag;
    int old_U_size = state_U.rows();
    int new_U_size = old_U_size + LDim::Pose;
    state_U.conservativeResize(old_U_size + size_order_NEW, old_U_size + size_order_NEW);
    state_U.bottomRows(size_order_NEW).setZero();
    state_U.rightCols(size_order_NEW).setZero();
    int index_i = 0;
    for (int i = 0; i < order_OLD.size(); ++i)
    {
        auto &var_old = order_OLD.at(i);
        state_U.block(old_U_size, var_old->id(), size_order_NEW, var_old->size()) = Q_sqrt_inv_diag.asDiagonal() * Phi.middleCols(index_i, var_old->size());
        index_i += var_old->size();
    }
    state_U.bottomRightCorner(size_order_NEW, size_order_NEW) = -Q_sqrt_inv;

    // If we are doing time calibration, then our clones are a function of the time offset
    // Logic is based on Mingyang Li and Anastasios I. Mourikis paper:
    // http://journals.sagepub.com/doi/pdf/10.1177/0278364913515286
    if (0)
    {
        // Jacobian to augment by
        Eigen::Matrix<number_t, LDim::Pose, 1> dnc_dt = MatrixX::Zero(LDim::Pose, 1);
        dnc_dt.block(0, 0, LDim::Ori, 1) = last_w;
        dnc_dt.block(LDim::Ori, 0, LDim::Tvec, 1) = state->_imu->vel();
        state_U.block(old_U_size, state->_calib_dt_CAMtoIMU->id(), LDim::Pose, 1) = Q_sqrt_inv.topLeftCorner(LDim::Pose, LDim::Pose) * dnc_dt;
    }
    FilterUtils::flipToHead(state_U.leftCols(old_U_size), LDim::Vel + LDim::Bias);
    if (state->_clones_IMU.size() > state->_options.max_clone_size)
    {
        int marg_pose_id = old_U_size - state->_clones_IMU.size() * LDim::Pose;
        FilterUtils::flipToHead(state_U.leftCols(marg_pose_id + LDim::Pose), LDim::Pose);
        double marginal_time = state->margtimestep();
        // Lock the mutex to avoid deleting any elements from _clones_IMU while accessing it from other threads
        std::lock_guard<std::mutex> lock(state->_mutex_state);
        assert(marginal_time != INFINITY);
        std::shared_ptr<Type> marg = state->_clones_IMU[marginal_time];
        std::vector<std::shared_ptr<Type>> remaining_variables;
        for (size_t i = 0; i < state->_variables.size(); i++)
        {
            // Only keep non-marginal states
            if (state->_variables.at(i) != marg)
            {
                if (state->_variables.at(i)->id() > marg->id())
                {
                    // If the variable is "beyond" the marginal one in ordering, need to "move it forward"
                    state->_variables.at(i)->set_local_id(state->_variables.at(i)->id() - marg->size());
                }
                remaining_variables.push_back(state->_variables.at(i));
            }
        }
        marg->set_local_id(-1);
        state->_variables = remaining_variables;
        state->_clones_IMU.erase(marginal_time);
        new_U_size -= LDim::Pose;
    }

    // Create clone from the type being cloned
    std::shared_ptr<PoseJPL> new_clone = std::dynamic_pointer_cast<PoseJPL>(state->_imu->pose()->clone());
    // Append the new clone to our clone vector
    state->_clones_IMU[state->_timestamp] = new_clone;
    state->_imu->set_local_id(new_U_size - LDim::PoseVelBias);
    new_clone->set_local_id(state->_imu->pose()->id());
    state->last_clone_pose->set_local_id(new_U_size - LDim::PoseVelBias - LDim::Pose);
    state->_variables.push_back(state->last_clone_pose);
    state->last_clone_pose = new_clone;

    FilterUtils::performQRGivens(state_U, 0);
    state_U = state_U.bottomRightCorner(new_U_size, new_U_size).eval();
    state_res = MatrixX::Zero(state_U.rows(), 1);
    return;
}

void FilterSRIF::marginalize(std::shared_ptr<State> state, std::shared_ptr<Type> marg)
{
    // Check if the current state has the element we want to marginalize
    if (std::find(state->_variables.begin(), state->_variables.end(), marg) == state->_variables.end())
    {
        PRINT_ERROR(RED "FilterSRIF::marginalize() - Called on variable that is not in the state\n" RESET);
        PRINT_ERROR(RED "FilterSRIF::marginalize() - Marginalization, does NOT work on sub-variables yet...\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    int marg_size = marg->size();
    int marg_id = marg->id();
    MatrixX &state_U = state->_Cov;
    MatrixX &state_res = state->_res;
    MatrixX state_U_new = state_U;
    if (marg_id > 0)
    {
        state_U_new.middleCols(0, marg_size) = state_U.middleCols(marg_id, marg_size);
        state_U_new.middleCols(marg_size, marg_id) = state_U.middleCols(0, marg_id);
    }
    FilterUtils::performQRGivens(state_U_new, 0);
    state_U = state_U_new.bottomRightCorner(state_U_new.rows() - marg_size, state_U_new.cols() - marg_size);
    state_res = MatrixX::Zero(state_U.rows(), 1);
    // Now we keep the remaining variables and update their ordering
    // Note: DOES NOT SUPPORT MARGINALIZING SUBVARIABLES YET!!!!!!!
    std::vector<std::shared_ptr<Type>> remaining_variables;
    for (size_t i = 0; i < state->_variables.size(); i++)
    {
        // Only keep non-marginal states
        if (state->_variables.at(i) != marg)
        {
            if (state->_variables.at(i)->id() > marg_id)
            {
                // If the variable is "beyond" the marginal one in ordering, need to "move it forward"
                state->_variables.at(i)->set_local_id(state->_variables.at(i)->id() - marg_size);
            }
            remaining_variables.push_back(state->_variables.at(i));
        }
    }

    // Delete the old state variable to free up its memory
    // NOTE: we don't need to do this any more since our variable is a shared ptr
    // NOTE: thus this is automatically managed, but this allows outside references to keep the old variable
    // delete marg;
    marg->set_local_id(-1);

    // Now set variables as the remaining ones
    state->_variables = remaining_variables;
    state->_clones_IMU[state->_timestamp]->set_local_id(state->_imu->pose()->id());
}

void FilterSRIF::set_initial_covariance(std::shared_ptr<State> state, const MatrixX &covariance,
                                        const std::vector<std::shared_ptr<type::Type>> &order)
{

    // We need to loop through each element and overwrite the current covariance values
    // For example consider the following:
    // x = [ ori pos ] -> insert into -> x = [ ori bias pos ]
    // P = [ P_oo P_op ] -> P = [ P_oo  0   P_op ]
    //     [ P_po P_pp ]        [  0    P*    0  ]
    //                          [ P_po  0   P_pp ]
    // The key assumption here is that the covariance is block diagonal (cross-terms zero with P* can be dense)
    // This is normally the care on startup (for example between calibration and the initial state

    // For each variable, lets copy over all other variable cross terms
    // Note: this copies over itself to when i_index=k_index
    int i_index = 0;
    for (size_t i = 0; i < order.size(); i++)
    {
        int k_index = 0;
        for (size_t k = 0; k < order.size(); k++)
        {
            state->_Cov.block(order[i]->id(), order[k]->id(), order[i]->size(), order[k]->size()) =
                covariance.block(i_index, k_index, order[i]->size(), order[k]->size());
            k_index += order[k]->size();
        }
        i_index += order[i]->size();
    }
    state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
    VectorX R_sqrt_diag = state->_Cov.diagonal().array().sqrt().inverse();
    state->_Cov.diagonal() = R_sqrt_diag;
    state->_res = MatrixX::Zero(state->_Cov.rows(), 1);
}

void FilterSRIF::Update(std::shared_ptr<State> state, const std::vector<std::shared_ptr<type::Type>> &H_order, const MatrixX &H,
                        const VectorX &res, const MatrixX &R)
{
    assert(res.rows() == R.rows());
    assert(H.rows() == res.rows());
    assert(H_order.size() > 0);
    MatrixX &state_U = state->_Cov;
    MatrixX &state_res = state->_res;
    VectorX R_inv_sqrt_diag = R.diagonal().array().sqrt().inverse();
    MatrixX H_big = MatrixX::Zero(res.rows(), state_U.cols());
    int current_id = 0;
    int min_var_id = state_U.cols();
    for (const auto &meas_var : H_order)
    {
        H_big.middleCols(meas_var->id(), meas_var->size()) = H.middleCols(current_id, meas_var->size());
        current_id += meas_var->size();
        min_var_id = std::min(meas_var->id(), min_var_id);
    }
    int valid_cols = state_U.cols() - min_var_id;
    MatrixX tempQR = MatrixX::Zero(res.rows() + valid_cols, valid_cols + 1);
    tempQR.topLeftCorner(valid_cols, valid_cols) = state_U.bottomRightCorner(valid_cols, valid_cols);
    tempQR.topRightCorner(valid_cols, 1) = state_res.bottomRows(valid_cols);
    tempQR.bottomRightCorner(res.rows(), 1) = R_inv_sqrt_diag.asDiagonal() * res;
    tempQR.bottomLeftCorner(res.rows(), valid_cols) = R_inv_sqrt_diag.asDiagonal() * H_big.rightCols(valid_cols);

    Eigen::VectorXi seq(valid_cols);
    seq.setLinSpaced(valid_cols, 0, valid_cols - 1);
    std::vector<int> first_cols(&seq[0], seq.data() + valid_cols);
    std::vector<int> Hx_first_cols = FilterUtils::getFirstColsOfMat(tempQR.bottomRows(res.rows()));
    first_cols.insert(first_cols.end(), Hx_first_cols.begin(), Hx_first_cols.end());
    FilterUtils::performPermutationQR(tempQR, std::move(first_cols), 0, valid_cols);
    state_U.bottomRightCorner(valid_cols, valid_cols) = tempQR.topLeftCorner(valid_cols, valid_cols);
    state_res.bottomRows(valid_cols) = tempQR.topRightCorner(valid_cols, 1);
    return;
}

void FilterSRIF::Update(std::shared_ptr<State> state)
{
    MatrixX &state_U = state->_Cov;
    MatrixX &state_res = state->_res;
    VectorX dx = state_U.triangularView<Eigen::Upper>().solve(state_res);
    state_res = MatrixX::Zero(state_U.rows(), 1);
    for (size_t i = 0; i < state->_variables.size(); i++)
    {
        state->_variables.at(i)->update(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
    }
    // If we are doing online intrinsic calibration we should update our camera objects
    // NOTE: is this the best place to put this update logic??? probably..
    if (state->_options.do_calib_camera_intrinsics)
    {
        for (auto const &calib : state->_cam_intrinsics)
        {
            state->_cam_intrinsics_cameras.at(calib.first)->set_value(calib.second->value());
        }
    }
    return;
}

void FilterSRIF::AnchorChange(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &order_NEW,
                              const std::vector<std::shared_ptr<Type>> &order_OLD, const MatrixX &Phi)
{
    // We need at least one old and new variable
    if (order_NEW.empty() || order_OLD.empty())
    {
        PRINT_ERROR(RED "FilterSRF::Propagation() - Called with empty variable arrays!\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    MatrixX &state_U = state->_Cov;
    // Loop through our Phi order and ensure that they are continuous in memory
    int size_order_NEW = order_NEW.at(0)->size();
    for (size_t i = 0; i < order_NEW.size() - 1; i++)
    {
        if (order_NEW.at(i)->id() + order_NEW.at(i)->size() != order_NEW.at(i + 1)->id())
        {
            PRINT_ERROR(RED "FilterSRF::Propagation() - Called with non-contiguous state elements!\n" RESET);
            PRINT_ERROR(
                RED "FilterSRF::Propagation() - This code only support a state transition which is in the same order as the state\n" RESET);
            std::exit(EXIT_FAILURE);
        }
        size_order_NEW += order_NEW.at(i + 1)->size();
    }
    // Size of the old phi matrix
    int size_order_OLD = order_OLD.at(0)->size();
    for (size_t i = 0; i < order_OLD.size() - 1; i++)
    {
        size_order_OLD += order_OLD.at(i + 1)->size();
    }

    // Assert that we have correct sizes
    assert(size_order_NEW == Phi.rows());
    assert(size_order_OLD == Phi.cols());
    int new_start_id = order_NEW.at(0)->id();
    int index_i = 0;
    MatrixX temp_info = state_U.middleCols(new_start_id, size_order_NEW);
    state_U.middleCols(new_start_id, size_order_NEW).setZero();
    for (int i = 0; i < order_OLD.size(); ++i)
    {
        auto &var_old = order_OLD.at(i);
        int var_id = var_old->id();
        int var_size = var_old->size();
        state_U.middleCols(var_id, var_size).noalias() += temp_info * Phi.middleCols(index_i, var_old->size());
        index_i += var_old->size();
    }
    FilterUtils::performQRGivens(state_U, 0);
    return;
}

bool FilterSRIF::initialize(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                             const std::vector<std::shared_ptr<Type>> &H_order, MatrixX &H_R, MatrixX &H_L,
                             MatrixX &R, VectorX &res, number_t chi_2_mult) {
  // Check that this new variable is not already initialized
  if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
    PRINT_ERROR("FilterSRIF::initialize() - Called on variable that is already in the state\n");
    PRINT_ERROR("FilterSRIF::initialize() - Found this variable at %d in covariance\n", new_variable->id());
    std::exit(EXIT_FAILURE);
  }

  // Check that we have isotropic noise (i.e. is diagonal and all the same value)
  // TODO: can we simplify this so it doesn't take as much time?
  assert(R.rows() == R.cols());
  assert(R.rows() > 0);
  for (int r = 0; r < R.rows(); r++) {
    for (int c = 0; c < R.cols(); c++) {
      if (r == c && R(0, 0) != R(r, c)) {
        PRINT_ERROR(RED "FilterSRIF::initialize() - Your noise is not isotropic!\n" RESET);
        PRINT_ERROR(RED "FilterSRIF::initialize() - Found a value of %.2f verses value of %.2f\n" RESET, R(r, c), R(0, 0));
        std::exit(EXIT_FAILURE);
      } else if (r != c && R(r, c) != 0.0) {
        PRINT_ERROR(RED "FilterSRIF::initialize() - Your noise is not diagonal!\n" RESET);
        PRINT_ERROR(RED "FilterSRIF::initialize() - Found a value of %.2f at row %d and column %d\n" RESET, R(r, c), r, c);
        std::exit(EXIT_FAILURE);
      }
    }
  }

  //==========================================================
  //==========================================================
  // First we perform QR givens to seperate the system
  // The top will be a system that depends on the new state, while the bottom does not
  size_t new_var_size = new_variable->size();
  assert((int)new_var_size == H_L.cols());

  Eigen::JacobiRotation<number_t> tempHo_GR;
  for (int n = 0; n < H_L.cols(); ++n) {
    for (int m = (int)H_L.rows() - 1; m > n; m--) {
      // Givens matrix G
      tempHo_GR.makeGivens(H_L(m - 1, n), H_L(m, n));
      // Multiply G to the corresponding lines (m-1,m) in each matrix
      // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
      //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
      (H_L.block(m - 1, n, 2, H_L.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
      (H_R.block(m - 1, 0, 2, H_R.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
    }
  }

  // Separate into initializing and updating portions
  // 1. Invertible initializing system
  MatrixX Hxinit = H_R.block(0, 0, new_var_size, H_R.cols());
  MatrixX H_finit = H_L.block(0, 0, new_var_size, new_var_size);
  VectorX resinit = res.block(0, 0, new_var_size, 1);
  MatrixX Rinit = R.block(0, 0, new_var_size, new_var_size);

  // 2. Nullspace projected updating system
  MatrixX Hup = H_R.block(new_var_size, 0, H_R.rows() - new_var_size, H_R.cols());
  VectorX resup = res.block(new_var_size, 0, res.rows() - new_var_size, 1);
  MatrixX Rup = R.block(new_var_size, new_var_size, R.rows() - new_var_size, R.rows() - new_var_size);

  //==========================================================
  //==========================================================

  // Do mahalanobis distance testing
  MatrixX S = FilterWrapper::get_HxPmargHxT(state, H_order, Hup);
  S += Rup;
  number_t chi2 = resup.dot(S.llt().solve(resup));

  // Get what our threshold should be
  boost::math::chi_squared chi_squared_dist(res.rows());
  number_t chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
  if (chi2 > chi_2_mult * chi2_check) {
    return false;
  }

  //==========================================================
  //==========================================================
  // Finally, initialize it in our state
  FilterSRIF::initialize_invertible(state, new_variable, H_order, Hxinit, H_finit, Rinit, resinit);
  
  // Update with updating portion
  if (Hup.rows() > 0)
  {
      FilterSRIF::Update(state, H_order, Hup, resup, Rup);
  }
  return true;
}

void FilterSRIF::initialize_invertible(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                                       const std::vector<std::shared_ptr<Type>> &H_order, const MatrixX &H_R,
                                       const MatrixX &H_L, const MatrixX &R, const VectorX &res)
{
    // Check that this new variable is not already initialized
    if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end())
    {
        PRINT_ERROR("FilterSRIF::initialize_invertible() - Called on variable that is already in the state\n");
        PRINT_ERROR("FilterSRIF::initialize_invertible() - Found this variable at %d in covariance\n", new_variable->id());
        std::exit(EXIT_FAILURE);
    }

    // Check that we have isotropic noise (i.e. is diagonal and all the same value)
    // TODO: can we simplify this so it doesn't take as much time?
    assert(R.rows() == R.cols());
    assert(R.rows() > 0);
    for (int r = 0; r < R.rows(); r++)
    {
        for (int c = 0; c < R.cols(); c++)
        {
            if (r == c && R(0, 0) != R(r, c))
            {
                PRINT_ERROR(RED "FilterSRIF::initialize_invertible() - Your noise is not isotropic!\n" RESET);
                PRINT_ERROR(RED "FilterSRIF::initialize_invertible() - Found a value of %.2f verses value of %.2f\n" RESET, R(r, c), R(0, 0));
                std::exit(EXIT_FAILURE);
            }
            else if (r != c && R(r, c) != 0.0)
            {
                PRINT_ERROR(RED "FilterSRIF::initialize_invertible() - Your noise is not diagonal!\n" RESET);
                PRINT_ERROR(RED "FilterSRIF::initialize_invertible() - Found a value of %.2f at row %d and column %d\n" RESET, R(r, c), r, c);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    assert(res.rows() == R.rows());
    assert(H_L.rows() == res.rows());
    assert(H_L.rows() == H_R.rows());
    MatrixX H_R_big = MatrixX::Zero(H_R.rows(), state->_Cov.cols());
    int current_it = 0;
    for (const auto &meas_var : H_order)
    {
        H_R_big.block(0, meas_var->id(), H_R.rows(), meas_var->size()) = H_R.block(0, current_it, H_R.rows(), meas_var->size());
        current_it += meas_var->size();
    }
    int new_id = state->_features_SLAM.size() * LDim::Landmark;
    int new_size = new_variable->size();
    MatrixX &state_U = state->_Cov;
    MatrixX &state_res = state->_res;
    MatrixX state_U_new = MatrixX::Zero(state_U.rows() + R.rows(), state_U.cols() + R.cols());
    int remaining_cols = state_U.cols() - new_id;
    assert(remaining_cols > 0);
    if (new_id > 0)
    {
        state_U_new.topLeftCorner(new_id, new_id) = state_U.topLeftCorner(new_id, new_id);
        state_U_new.topRightCorner(new_id, remaining_cols) = state_U.topRightCorner(new_id, remaining_cols);
    }
    state_U_new.bottomRightCorner(remaining_cols, remaining_cols) = state_U.bottomRightCorner(remaining_cols, remaining_cols);
    VectorX R_sqrtinv_diag = R.diagonal().array().sqrt().inverse();
    state_U_new.block(new_id, new_id, H_L.rows(), H_L.cols()) = R_sqrtinv_diag.asDiagonal() * H_L;
    state_U_new.block(new_id, new_id + H_L.cols(), H_L.rows(), state_U.cols() - new_id) = R_sqrtinv_diag.asDiagonal() * H_R_big.rightCols(state_U.cols() - new_id);
    state_res.conservativeResize(state_U_new.rows(), 1);
    state_res.bottomRows(remaining_cols) = state_res.middleRows(new_id, remaining_cols).eval();
    state_res.middleRows(new_id, new_size) = MatrixX::Zero(H_L.rows(), 1);
    state_U = state_U_new;
    new_variable->update(H_L.inverse() * res);
    new_variable->set_local_id(new_id);
    for (auto &var : state->_variables)
    {
        if (var->id() >= new_id)
        {
            var->set_local_id(var->id() + new_size);
        }
    }
    state->_clones_IMU[state->_timestamp]->set_local_id(state->_imu->pose()->id());
    state->_variables.push_back(new_variable);
}

void FilterSRIF::marginalize_slam(std::shared_ptr<State> state)
{
    int ct_marginalized = 0;
    std::vector<std::shared_ptr<type::Type>> marg_slam_feats;
    int kslam_feats = state->_features_SLAM.size();
    auto it0 = state->_features_SLAM.begin();
    while (it0 != state->_features_SLAM.end())
    {
        if ((*it0).second->should_marg)
        {
            marg_slam_feats.emplace_back((*it0).second);
            it0 = state->_features_SLAM.erase(it0);
            ct_marginalized++;
        }
        else
        {
            it0++;
        }
    }
    if (marg_slam_feats.size() == 0)
        return;
    MatrixX &state_U = state->_Cov;
    MatrixX &state_res = state->_res;
    int ncols = state_U.cols();
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm;
    perm.resize(ncols);
    std::vector<int> col_values(ncols, 0);
    Eigen::VectorXi indices(ncols);
    indices.setLinSpaced(ncols, 0, ncols - 1);
    int marg_sum = 0;
    int k = -1;
    for (const auto &var : marg_slam_feats)
    {
        marg_sum += var->size();
        std::fill(col_values.begin() + var->id(), col_values.begin() + var->id() + var->size(), k);
        k--;
    }
    std::stable_sort(indices.data(), indices.data() + indices.size(),
                     [&col_values](int a, int b)
                     { return col_values[a] < col_values[b]; });
    perm.indices() = indices;
    state_U = state_U * perm;
    FilterUtils::performQRGivens(state_U,0);
    state_U = state_U.bottomRightCorner(ncols - marg_sum, ncols - marg_sum).eval();
    state_res = MatrixX::Zero(ncols - marg_sum, 1);
    std::vector<std::shared_ptr<Type>> remaining_variables;
    for (size_t i = 0; i < state->_variables.size(); i++)
    {
        if (std::find(marg_slam_feats.begin(), marg_slam_feats.end(), state->_variables.at(i)) != marg_slam_feats.end())
        {
            state->_variables.at(i)->set_local_id(-1);
            continue;
        }
        remaining_variables.push_back(state->_variables.at(i));
    }
    std::stable_sort(remaining_variables.begin(), remaining_variables.end(),
                     [](std::shared_ptr<Type> a, std::shared_ptr<Type> b)
                     { return a->id() < b->id(); });
    int curr_id = 0;
    for (auto &var : remaining_variables)
    {
        var->set_local_id(curr_id);
        curr_id += var->size();
    }
    state->_variables = remaining_variables;
    state->_clones_IMU[state->_timestamp]->set_local_id(state->_imu->pose()->id());
    return;
}

MatrixX FilterSRIF::get_chicheck_sqrtcov(std::shared_ptr<State> state, bool enforce)
{
    if (!enforce)
    {
        return state->_chiCheckCov;
    }
    MatrixX &state_marg_cov = state->_chiCheckCov;
    MatrixX &state_U = state->_Cov;
    int n = state_U.cols() - LDim::Landmark * state->_features_SLAM.size();
    state_marg_cov.setIdentity(n, n);
    state_U.bottomRightCorner(n, n).triangularView<Eigen::Upper>().solveInPlace(state_marg_cov);
    return state_marg_cov;
}

MatrixX FilterSRIF::get_marginal_covariance(std::shared_ptr<State> state,
                                            const std::vector<std::shared_ptr<type::Type>> &small_variables)
{
    MatrixX marg_U = FilterSRIF::get_chicheck_sqrtcov(state);
    int kslam_feats = state->_features_SLAM.size();
    int cov_size = 0;
    for (size_t i = 0; i < small_variables.size(); i++)
    {
        assert(small_variables[i]->id() >= kslam_feats * LDim::Landmark);
        cov_size += small_variables[i]->size();
    }
    MatrixX Small_U = MatrixX::Zero(marg_U.rows(), cov_size);
    int i_index = 0;
    int krow = 0;
    for (size_t i = 0; i < small_variables.size(); i++)
    {
        auto &var = small_variables[i];
        Small_U.middleCols(i_index, var->size()) = marg_U.middleCols(var->id() - kslam_feats * LDim::Landmark, var->size());
        i_index += var->size();
        krow = std::max(krow, var->id() - kslam_feats * LDim::Landmark + var->size());
    }
    Small_U = Small_U.topRows(krow).eval();
    return Small_U.transpose() * Small_U;
}

MatrixX FilterSRIF::get_HxPmargHxT(std::shared_ptr<State> state,
                                   const std::vector<std::shared_ptr<type::Type>> &small_variables, const MatrixX &Hx)
{
    MatrixX marg_U = FilterSRIF::get_chicheck_sqrtcov(state);
    MatrixX Hx_big = MatrixX::Zero(Hx.rows(), state->_Cov.cols());
    int index_i = 0;
    for (auto &var : small_variables)
    {
        Hx_big.middleCols(var->id(), var->size()) = Hx.middleCols(index_i, var->size());
        index_i += var->size();
    }
    MatrixX U_HxT = marg_U * Hx_big.rightCols(marg_U.cols()).transpose();
    return U_HxT.transpose() * U_HxT;
}