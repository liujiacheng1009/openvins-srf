#pragma once
#include "FilterEKF.h"
#include "FilterSRF.h"
#include "FilterSRIF.h"
#include "state/State.h"
#include "types/Type.h"

#include <Eigen/Eigen>
#include <memory>

namespace type
{
  class Type;
  class PoseJPL;
} // namespace type

namespace msckf
{

  class FilterWrapper
  {

  public:
    static void PropagationAndClone(std::shared_ptr<State> state, const std::vector<std::shared_ptr<type::Type>> &order_NEW,
                                    const std::vector<std::shared_ptr<type::Type>> &order_OLD, const MatrixX &Phi,
                                    const MatrixX &Q, Eigen::Matrix<number_t, 3, 1> &last_w);

    static void Propagation(std::shared_ptr<State> state, const std::vector<std::shared_ptr<type::Type>> &order_NEW,
                            const std::vector<std::shared_ptr<type::Type>> &order_OLD, const MatrixX &Phi,
                            const MatrixX &Q, bool use_joint_prop = false);

    static void Update(std::shared_ptr<State> state, const std::vector<std::shared_ptr<type::Type>> &H_order, const MatrixX &H,
                       const VectorX &res, const MatrixX &R, bool use_joint_update = false);

    static void JointUpdate(std::shared_ptr<State> state);

    static void JointPropagation(std::shared_ptr<State> state);

    static void set_initial_covariance(std::shared_ptr<State> state, const MatrixX &covariance,
                                       const std::vector<std::shared_ptr<type::Type>> &order);

    static void set_initial_last_clone_pose(std::shared_ptr<State> state);

    static MatrixX get_marginal_covariance(std::shared_ptr<State> state,
                                                   const std::vector<std::shared_ptr<type::Type>> &small_variables);

    static MatrixX get_HxPmargHxT(std::shared_ptr<State> state,
                                          const std::vector<std::shared_ptr<type::Type>> &small_variables, const MatrixX &Hx);

    static MatrixX get_full_covariance(std::shared_ptr<State> state);

    static void marginalize(std::shared_ptr<State> state, std::shared_ptr<type::Type> marg);

    static std::shared_ptr<type::Type> clone(std::shared_ptr<State> state, std::shared_ptr<type::Type> variable_to_clone);

    static bool initialize(std::shared_ptr<State> state, std::shared_ptr<type::Type> new_variable,
                           const std::vector<std::shared_ptr<type::Type>> &H_order, MatrixX &H_R, MatrixX &H_L,
                           MatrixX &R, VectorX &res, number_t chi_2_mult);

    static void augment_clone(std::shared_ptr<State> state, Eigen::Matrix<number_t, 3, 1> last_w);

    static void marginalize_old_clone(std::shared_ptr<State> state);

    static void marginalize_slam(std::shared_ptr<State> state);

    static MatrixX &get_state_cov(std::shared_ptr<State> state);

    static std::vector<std::shared_ptr<type::Type>> &get_state_variables(std::shared_ptr<State> state);

    static std::shared_ptr<type::PoseJPL>& get_last_clone_pose(std::shared_ptr<State> state);

  private:
    /**
     * All function in this class should be static.
     * Thus an instance of this class cannot be created.
     */
    FilterWrapper() {}
  };

} // namespace msckf
