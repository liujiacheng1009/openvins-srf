#pragma once

#include <Eigen/Eigen>
#include <memory>

#include "types/Type.h"
#include "FilterUtils.h"

namespace type {
class Type;
} // namespace type

namespace msckf {

class State;

/**
 * @brief Helper which manipulates the State and its covariance.
 *
 * In general, this class has all the core logic for an Extended Kalman Filter (EKF)-based system.
 * This has all functions that change the covariance along with addition and removing elements from the state.
 * All functions here are static, and thus are self-contained so that in the future multiple states could be tracked and updated.
 * We recommend you look directly at the code for this class for clarity on what exactly we are doing in each and the matching documentation
 * pages.
 */
class FilterSRIF {

public:
  static void PropagationAndClone(std::shared_ptr<State> state, const std::vector<std::shared_ptr<type::Type>> &order_NEW,
                                  const std::vector<std::shared_ptr<type::Type>> &order_OLD, const MatrixX &Phi,
                                  const MatrixX &Q, Eigen::Matrix<number_t, 3, 1> &last_w);

  static void set_initial_last_clone_pose(std::shared_ptr<State> state);

  static void marginalize(std::shared_ptr<State> state, std::shared_ptr<type::Type> marg);

  static void marginalize_slam(std::shared_ptr<State> state);

  static void set_initial_covariance(std::shared_ptr<State> state, const MatrixX &covariance,
                                     const std::vector<std::shared_ptr<type::Type>> &order);

  static void Update(std::shared_ptr<State> state, const std::vector<std::shared_ptr<type::Type>> &H_order, const MatrixX &H,
                     const VectorX &res, const MatrixX &R);

  static void Update(std::shared_ptr<State> state);

  static void AnchorChange(std::shared_ptr<State> state, const std::vector<std::shared_ptr<type::Type>> &order_NEW,
                           const std::vector<std::shared_ptr<type::Type>> &order_OLD, const MatrixX &Phi);

  static bool initialize(std::shared_ptr<State> state, std::shared_ptr<type::Type> new_variable,
                         const std::vector<std::shared_ptr<type::Type>> &H_order, MatrixX &H_R, MatrixX &H_L,
                         MatrixX &R, VectorX &res, number_t chi_2_mult);

  static void initialize_invertible(std::shared_ptr<State> state, std::shared_ptr<type::Type> new_variable,
                                    const std::vector<std::shared_ptr<type::Type>> &H_order, const MatrixX &H_R,
                                    const MatrixX &H_L, const MatrixX &R, const VectorX &res);

  static MatrixX get_chicheck_sqrtcov(std::shared_ptr<State> state, bool enforce = false);

  static MatrixX get_marginal_covariance(std::shared_ptr<State> state,
                                         const std::vector<std::shared_ptr<type::Type>> &small_variables);

  static MatrixX get_HxPmargHxT(std::shared_ptr<State> state,
                                const std::vector<std::shared_ptr<type::Type>> &small_variables, const MatrixX &Hx);

private:
  /**
   * All function in this class should be static.
   * Thus an instance of this class cannot be created.
   */
  FilterSRIF() {}
};

} // namespace msckf
