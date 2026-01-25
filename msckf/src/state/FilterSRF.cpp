#include "FilterSRF.h"

#include "state/State.h"

#include "types/Landmark.h"
#include "utils/colors.h"
#include "utils/print.h"

#include <boost/math/distributions/chi_squared.hpp>

using namespace core;
using namespace type;
using namespace msckf;



void FilterSRF::PropagationAndClone(std::shared_ptr<State> state, const std::vector<std::shared_ptr<type::Type>> &order_NEW,
                                    const std::vector<std::shared_ptr<type::Type>> &order_OLD, const MatrixX &Phi,
                                    const MatrixX &Q, Eigen::Matrix<number_t, 3, 1> &last_w)
{
  // We need at least one old and new variable
  if (order_NEW.empty() || order_OLD.empty())
  {
    PRINT_ERROR(RED "FilterSRF::Propagation() - Called with empty variable arrays!\n" RESET);
    std::exit(EXIT_FAILURE);
  }
  MatrixX &state_U = state->_Cov;
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
  MatrixX Q_sqrt = MatrixX::Zero(Q.rows(), Q.cols());
  VectorX Q_sqrt_diag = Q.diagonal().array().sqrt();
  Q_sqrt.diagonal() = Q_sqrt_diag;
  int aug_size = state->_imu->pose()->size();
  MatrixX tempQR = MatrixX::Zero(state->_Cov.rows() + size_order_NEW, state->_Cov.cols() + aug_size);
  int new_start_id = order_NEW.at(0)->id();
  int old_start_id = order_OLD.at(0)->id();
  int index_i = 0;
  MatrixX UPhiT = MatrixX::Zero(state->_Cov.rows(), size_order_NEW);
  for (int i = 0; i < order_OLD.size(); ++i)
  {
    auto &var_old = order_OLD.at(i);
    UPhiT.topRows(var_old->id() + var_old->size()).noalias() += state_U.block(0, var_old->id(), var_old->id() + var_old->size(), var_old->size()) * Phi.middleCols(index_i, var_old->size()).transpose();
    index_i += var_old->size();
  }
  state_U.middleCols(new_start_id, size_order_NEW) = UPhiT;
  tempQR.block(0, new_start_id, size_order_NEW, size_order_NEW) = Q_sqrt;
  tempQR.bottomLeftCorner(state_U.rows(), state_U.cols()) = state_U;

  int new_loc = state_U.cols();
  if (state->_clones_IMU.size() > 0)
  {
    for (auto &var : state->_clones_IMU)
    {
      new_loc = std::min(new_loc, var.second->id());
    }
  }
  int old_loc = state->_imu->pose()->id();

  tempQR.rightCols(state->_Cov.cols() - new_loc) = tempQR.middleCols(new_loc, state->_Cov.cols() - new_loc).eval();
  tempQR.middleCols(new_loc, aug_size) = tempQR.middleCols(old_loc, aug_size).eval();
  // Create clone from the type being cloned
  std::shared_ptr<Type> new_clone = nullptr;
  new_clone = state->_imu->pose()->clone();
  new_clone->set_local_id(new_loc);
  for (size_t k = 0; k < state->_variables.size(); k++)
  {
    auto &var = state->_variables.at(k);
    if (var->id() >= new_loc)
    {
      var->set_local_id(var->id() + aug_size);
    }
  }
  state->_variables.push_back(new_clone);
  // Cast to a JPL pose type, check if valid
  std::shared_ptr<PoseJPL> pose = std::dynamic_pointer_cast<PoseJPL>(new_clone);
  if (pose == nullptr)
  {
    PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
    std::exit(EXIT_FAILURE);
  }
  // Append the new clone to our clone vector
  state->_clones_IMU[state->_timestamp] = pose;

  // If we are doing time calibration, then our clones are a function of the time offset
  // Logic is based on Mingyang Li and Anastasios I. Mourikis paper:
  // http://journals.sagepub.com/doi/pdf/10.1177/0278364913515286
  if (state->_options.do_calib_camera_timeoffset)
  {
    // Jacobian to augment by
    Eigen::Matrix<number_t, 6, 1> dnc_dt = MatrixX::Zero(6, 1);
    dnc_dt.block(0, 0, 3, 1) = last_w;
    dnc_dt.block(3, 0, 3, 1) = state->_imu->vel();
    tempQR.middleCols(pose->id(), pose->size()) += tempQR.middleCols(state->_calib_dt_CAMtoIMU->id(), state->_calib_dt_CAMtoIMU->size()) * dnc_dt.transpose();
  }

  FilterUtils::performQRGivens(tempQR, 0);
  state_U = tempQR.topLeftCorner(state->_Cov.rows() + aug_size, state->_Cov.cols() + aug_size);
  return;
}

void FilterSRF::Propagation(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &order_NEW,
                            const std::vector<std::shared_ptr<Type>> &order_OLD, const MatrixX &Phi,
                            const MatrixX &Q, bool use_joint_prop)
{
  // We need at least one old and new variable
  if (order_NEW.empty() || order_OLD.empty())
  {
    PRINT_ERROR(RED "FilterSRF::Propagation() - Called with empty variable arrays!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  MatrixX& state_U = state->_Cov;
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
  assert(size_order_NEW == Q.cols());
  assert(size_order_NEW == Q.rows());
  MatrixX Q_sqrt = MatrixX::Zero(Q.rows(), Q.cols());
  VectorX Q_sqrt_diag = Q.diagonal().array().sqrt();
  Q_sqrt.diagonal() = Q_sqrt_diag;
  MatrixX tempQR = MatrixX::Zero(state->_Cov.rows() + size_order_NEW, state->_Cov.cols());
  int new_start_id = order_NEW.at(0)->id();
  int old_start_id = order_OLD.at(0)->id();
  int index_i = 0;
  MatrixX UPhiT = MatrixX::Zero(state->_Cov.rows(), size_order_NEW);
  for (int i = 0; i < order_OLD.size(); ++i)
  {
    auto &var_old = order_OLD.at(i);
    UPhiT.topRows(var_old->id() + var_old->size()).noalias() += state_U.block(0, var_old->id(), var_old->id() + var_old->size(), var_old->size()) * Phi.middleCols(index_i, var_old->size()).transpose();
    index_i += var_old->size();
  }

  if (Q_sqrt_diag == VectorX::Zero(size_order_NEW))
  {
    if (use_joint_prop) // only for slam feat anchor change
    {
      state->joint_order_new_pairs.emplace_back(std::make_pair(new_start_id, size_order_NEW));
      state->joint_U_PhiT.emplace_back(UPhiT);
      return;
    }
    state_U.middleCols(new_start_id, size_order_NEW) = UPhiT;
    tempQR.topLeftCorner(state_U.rows(), state_U.cols()) = state_U;
    FilterUtils::performQRGivens(tempQR, new_start_id);
  }
  else
  {
    state_U.middleCols(new_start_id, size_order_NEW) = UPhiT;
    tempQR.block(0, new_start_id, size_order_NEW, size_order_NEW) = Q_sqrt;
    tempQR.bottomLeftCorner(state_U.rows(), state_U.cols()) = state_U;
    FilterUtils::performQRGivens(tempQR, 0);
  }
  state_U = tempQR.topLeftCorner(state->_Cov.rows(), state->_Cov.cols());
  return;
}

void FilterSRF::JointPropagation(std::shared_ptr<State> state)
{
  MatrixX &state_U = state->_Cov;
  int new_start_id = state_U.cols();
  int nlandmark = state->joint_U_PhiT.size();
  if (nlandmark < 1)
    return;
  for (int i = 0; i < nlandmark; ++i)
  {
    auto &idx = state->joint_order_new_pairs[i];
    state_U.middleCols(idx.first, idx.second) = state->joint_U_PhiT[i];
    new_start_id = std::min(new_start_id, idx.first);
  }
  FilterUtils::performQRGivens(state_U, new_start_id);
  state->joint_order_new_pairs.clear();
  state->joint_U_PhiT.clear();
  return;
}

void FilterSRF::StackMeasures(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &H_order, const MatrixX &H, const VectorX &res, const MatrixX &R)
{
  assert(res.rows() == R.rows());
  assert(H.rows() == res.rows());
  VectorX R_inv_diag = R.diagonal().array().inverse();
  VectorX R_inv_sqrt_diag = R.diagonal().array().sqrt().inverse();
  state->joint_H_orders.emplace_back(H_order);
  state->joint_Hs.emplace_back(H);
  state->joint_res.conservativeResize(state->joint_res.size() + res.size());
  state->joint_res.tail(res.size()) = res;
  state->joint_R_inv_diag.conservativeResize(state->joint_R_inv_diag.size() + R_inv_diag.size());
  state->joint_R_inv_diag.tail(R_inv_diag.size()) = R_inv_diag;
  state->joint_R_inv_sqrt_diag.conservativeResize(state->joint_R_inv_sqrt_diag.size() + R_inv_sqrt_diag.size());
  state->joint_R_inv_sqrt_diag.tail(R_inv_sqrt_diag.size()) = R_inv_sqrt_diag;
  return;
}

void FilterSRF::Update(std::shared_ptr<State> state, const std::vector<std::shared_ptr<Type>> &H_order, const MatrixX &H,
                       const VectorX &res, const MatrixX &R)
{
  assert(res.rows() == R.rows());
  assert(H.rows() == res.rows());
  MatrixX &state_U = state->_Cov;
  VectorX R_inv_diag = R.diagonal().array().inverse();
  VectorX R_inv_sqrt_diag = R.diagonal().array().sqrt().inverse();
  MatrixX tempHUT = MatrixX::Zero(res.rows(), state_U.rows());
  MatrixX H_big = MatrixX::Zero(res.rows(), state_U.cols());
  int current_id = 0;
  for (const auto &meas_var : H_order)
  {
    H_big.middleCols(meas_var->id(), meas_var->size()) = H.middleCols(current_id, meas_var->size());
    tempHUT.leftCols(meas_var->id() + meas_var->size()).noalias() += H.middleCols(current_id, meas_var->size()) * state_U.middleCols(meas_var->id(), meas_var->size()).topRows(meas_var->id() + meas_var->size()).transpose();
    current_id += meas_var->size();
  }
  MatrixX tempQR = MatrixX::Zero(res.size() + state_U.rows(), state_U.cols());
  tempQR.bottomLeftCorner(res.size(), state_U.cols()) = R_inv_sqrt_diag.asDiagonal() * tempHUT;
  tempQR.topLeftCorner(state_U.rows(), state_U.cols()) = MatrixX::Identity(state_U.rows(), state_U.cols());
  Eigen::JacobiRotation<number_t> GR;
  for (int col = tempQR.cols() - 1; col >= 0; --col)
  {
    for (int row = 0; row < tempQR.rows() - tempQR.cols() + col; ++row)
    {
      if (tempQR(row, col) != 0)
      {
        GR.makeGivens(tempQR(row + 1, col), tempQR(row, col));
        tempQR.block(row, 0, 2, col + 1).applyOnTheLeft(1, 0, GR.adjoint());
        tempQR(row, col) = 0;
      }
    }
  }
  MatrixX FT = tempQR.bottomLeftCorner(state_U.rows(), state_U.cols()).transpose();
  FilterSRF::Update(state, FT, H_big, res, R_inv_diag);
}

void FilterSRF::Update(std::shared_ptr<State> state, const MatrixX &FT, const MatrixX &H_big,
                       const VectorX &res, const VectorX &R_inv_diag)
{
  MatrixX &state_U = state->_Cov;
  for (int j = 0; j < FT.cols(); ++j)
  {
    VectorX v1 = state_U.block(0, j, j + 1, 1);
    FT.topLeftCorner(j + 1, j + 1).triangularView<Eigen::Upper>().solveInPlace(v1);
    state_U.block(0, j, j + 1, 1) = v1;
  }
  VectorX dx = state_U.triangularView<Eigen::Upper>().transpose() * (state_U * (H_big.transpose() * (R_inv_diag.asDiagonal() * res)));
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
}

void FilterSRF::JointUpdate(std::shared_ptr<State> state)
{
  MatrixX &state_U = state->_Cov;
  int nmeas = state->joint_H_orders.size();
  if (nmeas < 1)
    return;
  MatrixX tempHUT_all = MatrixX::Zero(state->joint_res.size(), state_U.rows());
  MatrixX H_big_all = MatrixX::Zero(state->joint_res.size(), state_U.cols());
  int H_meas_id = 0;
  for (int i = 0; i < nmeas; ++i)
  {
    int current_id = 0;
    auto &H_order = state->joint_H_orders[i];
    auto &H = state->joint_Hs[i];
    MatrixX tempHUT = MatrixX::Zero(H.rows(), state_U.rows());
    MatrixX H_big = MatrixX::Zero(H.rows(), state_U.cols());
    for (const auto &meas_var : H_order)
    {
      H_big.middleCols(meas_var->id(), meas_var->size()) = H.middleCols(current_id, meas_var->size());
      tempHUT.leftCols(meas_var->id() + meas_var->size()).noalias() += H.middleCols(current_id, meas_var->size()) * state_U.middleCols(meas_var->id(), meas_var->size()).topRows(meas_var->id() + meas_var->size()).transpose();
      current_id += meas_var->size();
    }
    tempHUT_all.middleRows(H_meas_id, tempHUT.rows()) = tempHUT;
    H_big_all.middleRows(H_meas_id, H_big.rows()) = H_big;
    H_meas_id += H_big.rows();
  }
  MatrixX tempQR = MatrixX::Zero(state->joint_res.size() + state_U.rows(), state_U.cols());
  tempQR.topRows(state->joint_res.size()) = state->joint_R_inv_sqrt_diag.asDiagonal() * tempHUT_all;
  tempQR.bottomRows(state_U.rows()) = MatrixX::Identity(state_U.rows(), state_U.cols());
  auto perm = FilterUtils::genReversePermMat(tempQR.cols());
  tempQR = tempQR * perm;
  int num_cols = tempQR.cols();
  Eigen::VectorXi seq(num_cols);
  seq.setLinSpaced(num_cols, num_cols - 1, 0);
  std::vector<int> first_cols(&seq[0], seq.data() + num_cols);
  std::vector<int> Hx_first_cols = FilterUtils::getFirstColsOfMat(tempQR.topRows(tempQR.rows() - num_cols));
  Hx_first_cols.insert(Hx_first_cols.end(), first_cols.begin(), first_cols.end());
  FilterUtils::performPermutationQR(tempQR, std::move(Hx_first_cols), 0, num_cols);
  tempQR.conservativeResize(tempQR.cols(), tempQR.cols());
  tempQR = perm.transpose() * tempQR * perm;
  MatrixX FT = tempQR.bottomLeftCorner(state_U.rows(), state_U.cols()).transpose();
  FilterSRF::Update(state, FT, H_big_all, state->joint_res, state->joint_R_inv_diag);
  state->joint_H_orders.clear();
  state->joint_Hs.clear();
  state->joint_res.conservativeResize(0);
  state->joint_R_inv_diag.conservativeResize(0);
  state->joint_R_inv_sqrt_diag.conservativeResize(0);
}

void FilterSRF::set_initial_covariance(std::shared_ptr<State> state, const MatrixX &covariance,
                                         const std::vector<std::shared_ptr<type::Type>> &order) {

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
  for (size_t i = 0; i < order.size(); i++) {
    int k_index = 0;
    for (size_t k = 0; k < order.size(); k++) {
      state->_Cov.block(order[i]->id(), order[k]->id(), order[i]->size(), order[k]->size()) =
          covariance.block(i_index, k_index, order[i]->size(), order[k]->size());
      k_index += order[k]->size();
    }
    i_index += order[i]->size();
  }
  state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
  VectorX R_sqrt_diag = state->_Cov.diagonal().array().sqrt();
  state->_Cov.diagonal() = R_sqrt_diag;
}

MatrixX FilterSRF::get_marginal_covariance(std::shared_ptr<State> state,
                                                        const std::vector<std::shared_ptr<Type>> &small_variables)
{
  MatrixX marg_U = get_marginal_sqrtcov(state, small_variables);
  return marg_U.transpose() * marg_U;
}

MatrixX FilterSRF::get_marginal_sqrtcov(std::shared_ptr<State> state,
                                                const std::vector<std::shared_ptr<Type>> &small_variables)
{
  // Calculate the marginal covariance size we need to make our matrix
  int cov_size = 0;
  for (size_t i = 0; i < small_variables.size(); i++)
  {
    cov_size += small_variables[i]->size();
  }

  // For each variable, lets copy over all other variable cross terms
  // Note: this copies over itself to when i_index=k_index
  MatrixX &state_U = state->_Cov;
  MatrixX Small_U = MatrixX::Zero(state_U.rows(), cov_size);
  int i_index = 0;
  int krow = 0;
  for (size_t i = 0; i < small_variables.size(); i++)
  {
    auto &var = small_variables[i];
    Small_U.middleCols(i_index, var->size()) = state_U.middleCols(var->id(), var->size());
    i_index += var->size();
    krow = std::max(krow, var->id() + var->size());
  }
  Small_U = Small_U.topRows(krow).eval();
  return Small_U;
}

MatrixX FilterSRF::get_full_covariance(std::shared_ptr<State> state) {

  // Size of the covariance is the active
  int cov_size = (int)state->_Cov.rows();

  // Construct our return covariance
  MatrixX full_cov = MatrixX::Zero(cov_size, cov_size);

  // Copy in the active state elements
  full_cov.block(0, 0, state->_Cov.rows(), state->_Cov.rows()) = state->_Cov.transpose() * state->_Cov;

  // Return the covariance
  return full_cov;
}

void FilterSRF::marginalize(std::shared_ptr<State> state, std::shared_ptr<Type> marg) {

  // Check if the current state has the element we want to marginalize
  if (std::find(state->_variables.begin(), state->_variables.end(), marg) == state->_variables.end()) {
    PRINT_ERROR(RED "FilterSRF::marginalize() - Called on variable that is not in the state\n" RESET);
    PRINT_ERROR(RED "FilterSRF::marginalize() - Marginalization, does NOT work on sub-variables yet...\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  int marg_size = marg->size();
  int marg_id = marg->id();
  MatrixX& state_U = state->_Cov;
  MatrixX state_U_new = MatrixX::Zero(state_U.rows(), state_U.cols() - marg_size);
  if (marg_id > 0)
    state_U_new.topLeftCorner(state_U.rows(), marg_id) = state_U.topLeftCorner(state_U.rows(), marg_id);
  int r_size = (int)state_U.cols() - marg_id - marg_size;
  if (r_size > 0)
    state_U_new.topRightCorner(state_U.rows(), r_size) = state_U.topRightCorner(state_U.rows(), r_size);
  FilterUtils::performQRGivens(state_U_new, marg_id);

  state_U = state_U_new.topLeftCorner(state_U_new.cols(), state_U_new.cols());
  // Now we keep the remaining variables and update their ordering
  // Note: DOES NOT SUPPORT MARGINALIZING SUBVARIABLES YET!!!!!!!
  std::vector<std::shared_ptr<Type>> remaining_variables;
  for (size_t i = 0; i < state->_variables.size(); i++) {
    // Only keep non-marginal states
    if (state->_variables.at(i) != marg) {
      if (state->_variables.at(i)->id() > marg_id) {
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
}

std::shared_ptr<Type> FilterSRF::clone(std::shared_ptr<State> state, std::shared_ptr<Type> variable_to_clone)
{
  // Get total size of new cloned variables, and the old covariance size
  int total_size = variable_to_clone->size();
  int old_size = (int)state->_Cov.rows();
  int new_loc = (int)state->_Cov.rows();
  if (state->_clones_IMU.size() > 0)
  {
    for (auto &var : state->_clones_IMU)
    {
      new_loc = std::min(new_loc, var.second->id());
    }
  }
  MatrixX& state_U = state->_Cov;
  // Resize both our covariance to the new size
  state_U.conservativeResizeLike(MatrixX::Zero(old_size + total_size, old_size + total_size));
  // What is the new state, and variable we inserted
  const std::vector<std::shared_ptr<Type>> new_variables = state->_variables;
  std::shared_ptr<Type> new_clone = nullptr;

  // Loop through all variables, and find the variable that we are going to clone
  for (size_t k = 0; k < state->_variables.size(); k++)
  {

    // Skip this if it is not the same
    // First check if the top level variable is the same, then check the sub-variables
    std::shared_ptr<Type> type_check = state->_variables.at(k)->check_if_subvariable(variable_to_clone);
    if (state->_variables.at(k) == variable_to_clone)
    {
      type_check = state->_variables.at(k);
    }
    else if (type_check != variable_to_clone)
    {
      continue;
    }

    // So we will clone this one
    int old_loc = type_check->id();

    // Copy the covariance elements
    state_U.rightCols(old_size - new_loc) = state_U.middleCols(new_loc, old_size - new_loc).eval();
    state_U.middleCols(new_loc, total_size) = state_U.middleCols(old_loc, total_size).eval();
    // Create clone from the type being cloned
    new_clone = type_check->clone();
    new_clone->set_local_id(new_loc);
    break;
  }

  // Check if the current state has this variable
  if (new_clone == nullptr)
  {
    PRINT_ERROR(RED "FilterSRF::clone() - Called on variable is not in the state\n" RESET);
    PRINT_ERROR(RED "FilterSRF::clone() - Ensure that the variable specified is a variable, or sub-variable..\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Add to variable list and return
  for (size_t k = 0; k < state->_variables.size(); k++)
  {
    auto &var = state->_variables.at(k);
    if (var->id() >= new_loc)
    {
      var->set_local_id(var->id() + total_size);
    }
  }
  state->_variables.push_back(new_clone);
  return new_clone;
}

bool FilterSRF::initialize(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                             const std::vector<std::shared_ptr<Type>> &H_order, MatrixX &H_R, MatrixX &H_L,
                             MatrixX &R, VectorX &res, number_t chi_2_mult) {

  // Check that this new variable is not already initialized
  if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
    PRINT_ERROR("FilterSRF::initialize_invertible() - Called on variable that is already in the state\n");
    PRINT_ERROR("FilterSRF::initialize_invertible() - Found this variable at %d in covariance\n", new_variable->id());
    std::exit(EXIT_FAILURE);
  }

  // Check that we have isotropic noise (i.e. is diagonal and all the same value)
  // TODO: can we simplify this so it doesn't take as much time?
  assert(R.rows() == R.cols());
  assert(R.rows() > 0);
  for (int r = 0; r < R.rows(); r++) {
    for (int c = 0; c < R.cols(); c++) {
      if (r == c && R(0, 0) != R(r, c)) {
        PRINT_ERROR(RED "FilterSRF::initialize() - Your noise is not isotropic!\n" RESET);
        PRINT_ERROR(RED "FilterSRF::initialize() - Found a value of %.2f verses value of %.2f\n" RESET, R(r, c), R(0, 0));
        std::exit(EXIT_FAILURE);
      } else if (r != c && R(r, c) != 0.0) {
        PRINT_ERROR(RED "FilterSRF::initialize() - Your noise is not diagonal!\n" RESET);
        PRINT_ERROR(RED "FilterSRF::initialize() - Found a value of %.2f at row %d and column %d\n" RESET, R(r, c), r, c);
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
  MatrixX U_HupT = FilterSRF::get_marginal_sqrtcov(state, H_order) * Hup.transpose();
  MatrixX S = U_HupT.transpose() * U_HupT + Rup;
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
  FilterSRF::initialize_invertible(state, new_variable, H_order, Hxinit, H_finit, Rinit, resinit);
  
  // Update with updating portion
  if (Hup.rows() > 0)
  {
    if (GlobalFlagPool::getJointUpdate())
    {
      FilterSRF::StackMeasures(state, H_order, Hup, resup, Rup);
    }
    else
    {
      FilterSRF::Update(state, H_order, Hup, resup, Rup);
    }
  }
  return true;
}

void FilterSRF::initialize_invertible(std::shared_ptr<State> state, std::shared_ptr<Type> new_variable,
                                           const std::vector<std::shared_ptr<Type>> &H_order, const MatrixX &H_R,
                                           const MatrixX &H_L, const MatrixX &R, const VectorX &res)
{
  // Check that this new variable is not already initialized
  if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end())
  {
    PRINT_ERROR("FilterSRF::initialize_invertible() - Called on variable that is already in the state\n");
    PRINT_ERROR("FilterSRF::initialize_invertible() - Found this variable at %d in covariance\n", new_variable->id());
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
        PRINT_ERROR(RED "FilterSRF::initialize_invertible() - Your noise is not isotropic!\n" RESET);
        PRINT_ERROR(RED "FilterSRF::initialize_invertible() - Found a value of %.2f verses value of %.2f\n" RESET, R(r, c), R(0, 0));
        std::exit(EXIT_FAILURE);
      }
      else if (r != c && R(r, c) != 0.0)
      {
        PRINT_ERROR(RED "FilterSRF::initialize_invertible() - Your noise is not diagonal!\n" RESET);
        PRINT_ERROR(RED "FilterSRF::initialize_invertible() - Found a value of %.2f at row %d and column %d\n" RESET, R(r, c), r, c);
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

  MatrixX& state_U = state->_Cov;
  MatrixX state_U_new = MatrixX::Zero(state_U.rows() + R.rows(), state_U.cols() + R.cols());
  state_U_new.topLeftCorner(state_U.rows(), state_U.cols()) = state_U;
  MatrixX H_L_inv = H_L.inverse();
  state_U_new.topRightCorner(state_U.rows(), R.cols()) = -state_U * H_R_big.transpose() * H_L_inv.transpose();
  VectorX R_sqrt_diag = R.diagonal().array().sqrt();
  MatrixX R_sqrt;
  R_sqrt = MatrixX::Zero(R.rows(), R.cols());
  R_sqrt.diagonal() = R_sqrt_diag;
  state_U_new.bottomRightCorner(R.rows(), R.cols()) = R_sqrt * H_L_inv.transpose();
  FilterUtils::performQRGivens(state_U_new, state_U_new.cols()-R.cols());
  size_t oldSize = state_U.rows();
  state_U.conservativeResizeLike(state_U_new);
  state_U = state_U_new;

  new_variable->update(H_L_inv * res);
  // Now collect results, and add it to the state variables
  new_variable->set_local_id(oldSize);
  state->_variables.push_back(new_variable);
}

void FilterSRF::augment_clone(std::shared_ptr<State> state, Eigen::Matrix<number_t, 3, 1> last_w) {

  // We can't insert a clone that occured at the same timestamp!
  if (state->_clones_IMU.find(state->_timestamp) != state->_clones_IMU.end()) {
    PRINT_ERROR(RED "TRIED TO INSERT A CLONE AT THE SAME TIME AS AN EXISTING CLONE, EXITING!#!@#!@#\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our cloner and add it to our vector of types
  // NOTE: this will clone the clone pose to the END of the covariance...
  std::shared_ptr<Type> posetemp = FilterSRF::clone(state, state->_imu->pose());

  // Cast to a JPL pose type, check if valid
  std::shared_ptr<PoseJPL> pose = std::dynamic_pointer_cast<PoseJPL>(posetemp);
  if (pose == nullptr) {
    PRINT_ERROR(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Append the new clone to our clone vector
  state->_clones_IMU[state->_timestamp] = pose;

  // If we are doing time calibration, then our clones are a function of the time offset
  // Logic is based on Mingyang Li and Anastasios I. Mourikis paper:
  // http://journals.sagepub.com/doi/pdf/10.1177/0278364913515286
  if (state->_options.do_calib_camera_timeoffset)
  {
    // Jacobian to augment by
    Eigen::Matrix<number_t, 6, 1> dnc_dt = MatrixX::Zero(6, 1);
    dnc_dt.block(0, 0, 3, 1) = last_w;
    dnc_dt.block(3, 0, 3, 1) = state->_imu->vel();
    state->_Cov.middleCols(pose->id(), pose->size()) += state->_Cov.middleCols(state->_calib_dt_CAMtoIMU->id(), state->_calib_dt_CAMtoIMU->size()) * dnc_dt.transpose();
    FilterUtils::performQRGivens(state->_Cov, pose->id(), pose->size());
  }
}

void FilterSRF::marginalize_old_clone(std::shared_ptr<State> state) {
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    double marginal_time = state->margtimestep();
    // Lock the mutex to avoid deleting any elements from _clones_IMU while accessing it from other threads
    std::lock_guard<std::mutex> lock(state->_mutex_state);
    assert(marginal_time != INFINITY);
    FilterSRF::marginalize(state, state->_clones_IMU.at(marginal_time));
    // Note that the marginalizer should have already deleted the clone
    // Thus we just need to remove the pointer to it from our state
    state->_clones_IMU.erase(marginal_time);
  }
}

void FilterSRF::marginalize_slam(std::shared_ptr<State> state) {
  // Remove SLAM features that have their marginalization flag set
  // We also check that we do not remove any aruoctag landmarks
  int ct_marginalized = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((*it0).second->should_marg) {
      FilterSRF::marginalize(state, (*it0).second);
      it0 = state->_features_SLAM.erase(it0);
      ct_marginalized++;
    } else {
      it0++;
    }
  }
}

