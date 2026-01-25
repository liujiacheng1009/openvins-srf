#include "FilterWrapper.h"

using namespace core;
using namespace type;
using namespace msckf;

void FilterWrapper::PropagationAndClone(std::shared_ptr<State> state, const std::vector<std::shared_ptr<type::Type>> &order_NEW,
                                        const std::vector<std::shared_ptr<type::Type>> &order_OLD, const MatrixX &Phi,
                                        const MatrixX &Q, Eigen::Matrix<number_t, 3, 1> &last_w)
{
#ifdef VIO_USE_SRF
    FilterSRF::PropagationAndClone(state, order_NEW, order_OLD, Phi, Q, last_w);
#elif defined(VIO_USE_SRIF)
    FilterSRIF::PropagationAndClone(state, order_NEW, order_OLD, Phi, Q, last_w);
#else
    FilterEKF::Propagation(state, order_NEW, order_OLD, Phi, Q);
    FilterEKF::augment_clone(state, last_w);
#endif
}

void FilterWrapper::Propagation(std::shared_ptr<State> state, const std::vector<std::shared_ptr<type::Type>> &order_NEW,
                                const std::vector<std::shared_ptr<type::Type>> &order_OLD, const MatrixX &Phi,
                                const MatrixX &Q, bool use_joint_prop)
{
#ifdef VIO_USE_SRF
    FilterSRF::Propagation(state, order_NEW, order_OLD, Phi, Q, use_joint_prop);
#elif defined (VIO_USE_SRIF)
    assert((Q.array() == 0).all());
    FilterSRIF::AnchorChange(state, order_NEW, order_OLD, Phi);
#else
    FilterEKF::Propagation(state, order_NEW, order_OLD, Phi, Q);
#endif
}

void FilterWrapper::Update(std::shared_ptr<State> state, const std::vector<std::shared_ptr<type::Type>> &H_order, const MatrixX &H,
                           const VectorX &res, const MatrixX &R, bool use_joint_update)
{
#ifdef VIO_USE_SRF
    if (use_joint_update)
    {
        FilterSRF::StackMeasures(state, H_order, H, res, R);
    }
    else
    {
        FilterSRF::Update(state, H_order, H, res, R);
    }
#elif defined (VIO_USE_SRIF)
    FilterSRIF::Update(state, H_order, H, res, R);
#else
    FilterEKF::Update(state, H_order, H, res, R);
#endif
}

void FilterWrapper::JointPropagation(std::shared_ptr<State> state)
{
#ifdef VIO_USE_SRF
    FilterSRF::JointPropagation(state);
#else
    //
#endif
}

void FilterWrapper::JointUpdate(std::shared_ptr<State> state)
{
#ifdef VIO_USE_SRF
    if (GlobalFlagPool::getJointUpdate())
        FilterSRF::JointUpdate(state);
#elif defined(VIO_USE_SRIF)
    FilterSRIF::Update(state);
#else
    // 
#endif
}

void FilterWrapper::set_initial_covariance(std::shared_ptr<State> state, const MatrixX &covariance,
                                           const std::vector<std::shared_ptr<type::Type>> &order)
{
#ifdef VIO_USE_SRF
    FilterSRF::set_initial_covariance(state, covariance, order);
#elif defined(VIO_USE_SRIF)
    FilterSRIF::set_initial_covariance(state, covariance, order);
#else
    FilterEKF::set_initial_covariance(state, covariance, order);
#endif
}

void FilterWrapper::set_initial_last_clone_pose(std::shared_ptr<State> state)
{
#ifdef VIO_USE_SRIF
    FilterSRIF::set_initial_last_clone_pose(state);
#else
    //
#endif
}

MatrixX FilterWrapper::get_marginal_covariance(std::shared_ptr<State> state,
                                               const std::vector<std::shared_ptr<type::Type>> &small_variables)
{
#ifdef VIO_USE_SRF
    return FilterSRF::get_marginal_covariance(state, small_variables);
#elif defined (VIO_USE_SRIF)
    return FilterSRIF::get_marginal_covariance(state,small_variables);
#else
    return FilterEKF::get_marginal_covariance(state, small_variables);
#endif
}

MatrixX FilterWrapper::get_HxPmargHxT(std::shared_ptr<State> state,
                                              const std::vector<std::shared_ptr<type::Type>> &small_variables, const MatrixX &Hx)
{
#ifdef VIO_USE_SRF
    MatrixX U_HxT = FilterSRF::get_marginal_sqrtcov(state, small_variables) * Hx.transpose();
    return U_HxT.transpose() * U_HxT;
#elif defined(VIO_USE_SRIF)
    return FilterSRIF::get_HxPmargHxT(state, small_variables, Hx);
#else
    MatrixX P_marg = FilterEKF::get_marginal_covariance(state, small_variables);
    return Hx * P_marg * Hx.transpose();
#endif
}

MatrixX FilterWrapper::get_full_covariance(std::shared_ptr<State> state)
{
#ifdef VIO_USE_SRF
    return FilterSRF::get_full_covariance(state);
#elif defined(VIO_USE_SRIF)
    std::cerr << "srif not implement get_full_covariance !!" << std::endl;
    exit(-1);
#else
    return FilterEKF::get_full_covariance(state);
#endif
}

void FilterWrapper::marginalize(std::shared_ptr<State> state, std::shared_ptr<type::Type> marg)
{
#ifdef VIO_USE_SRF
    FilterSRF::marginalize(state, marg);
#elif defined(VIO_USE_SRIF)
    FilterSRIF::marginalize(state, marg);
#else
    FilterEKF::marginalize(state, marg);
#endif
}

std::shared_ptr<type::Type> FilterWrapper::clone(std::shared_ptr<State> state, std::shared_ptr<type::Type> variable_to_clone)
{
#ifdef VIO_USE_SRF
    return FilterSRF::clone(state, variable_to_clone);
#elif defined(VIO_USE_SRIF)
    std::cerr << "srif not implement clone !!" << std::endl;
    exit(-1);
#else
    return FilterEKF::clone(state, variable_to_clone);
#endif
}

bool FilterWrapper::initialize(std::shared_ptr<State> state, std::shared_ptr<type::Type> new_variable,
                               const std::vector<std::shared_ptr<type::Type>> &H_order, MatrixX &H_R, MatrixX &H_L,
                               MatrixX &R, VectorX &res, number_t chi_2_mult)
{
#ifdef VIO_USE_SRF
    return FilterSRF::initialize(state, new_variable, H_order, H_R, H_L, R, res, chi_2_mult);
#elif defined (VIO_USE_SRIF)
    return FilterSRIF::initialize(state, new_variable, H_order, H_R, H_L, R, res, chi_2_mult);
#else
    return FilterEKF::initialize(state, new_variable, H_order, H_R, H_L, R, res, chi_2_mult);
#endif
}

void FilterWrapper::augment_clone(std::shared_ptr<State> state, Eigen::Matrix<number_t, 3, 1> last_w)
{
#ifdef VIO_USE_SRF
    FilterSRF::augment_clone(state, last_w);
#elif defined(VIO_USE_SRIF)
    std::cerr << "srif not implement augment_clone !!" << std::endl;
    exit(-1);
#else
    FilterEKF::augment_clone(state, last_w);
#endif
}

void FilterWrapper::marginalize_old_clone(std::shared_ptr<State> state)
{
#ifdef VIO_USE_SRF
    FilterSRF::marginalize_old_clone(state);
#elif defined (VIO_USE_SRIF)
    // has implement in FilterSRIF::PropagationAndClone
#else
    FilterEKF::marginalize_old_clone(state);
#endif
}

void FilterWrapper::marginalize_slam(std::shared_ptr<State> state)
{
#ifdef VIO_USE_SRF
    FilterSRF::marginalize_slam(state);
#elif defined (VIO_USE_SRIF)
    FilterSRIF::marginalize_slam(state);
#else
    FilterEKF::marginalize_slam(state);
#endif
}

MatrixX &FilterWrapper::get_state_cov(std::shared_ptr<State> state)
{
    return state->_Cov;
}

std::vector<std::shared_ptr<type::Type>> &FilterWrapper::get_state_variables(std::shared_ptr<State> state)
{
    return state->_variables;
}

std::shared_ptr<type::PoseJPL>& FilterWrapper::get_last_clone_pose(std::shared_ptr<State> state)
{
    return state->last_clone_pose;
}