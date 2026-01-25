#include "FilterUtils.h"



void FilterUtils::flipToHead(Eigen::Ref<MatrixX> Mat, const int dim)
{
    int cols = Mat.cols();
    MatrixX tempM1 = Mat.rightCols(dim);
    MatrixX tempM2 = Mat.leftCols(cols - dim);
    Mat.leftCols(dim).swap(tempM1);
    Mat.rightCols(cols - dim).swap(tempM2);
    return;
}

void FilterUtils::performQRGivens(Eigen::Ref<MatrixX> tempQR, int start_col, int num_cols)
{
  constexpr number_t tol_thre = std::is_same_v<number_t, float> ? 5e-8 : 1e-13;
  const size_t total_cols = tempQR.cols();
  if (num_cols == -1)
    num_cols = total_cols - start_col;
  Eigen::JacobiRotation<number_t> GR;
  for (int col = start_col; col < start_col + num_cols; ++col)
  {
    const size_t remaining_cols = total_cols - col;
    for (int row = (int)tempQR.rows() - 1; row > col; --row)
    {
      if (__builtin_expect(std::abs(tempQR(row, col)) < tol_thre, 0))
      {
        tempQR(row, col) = 0;
        continue;
      }
      GR.makeGivens(tempQR(row - 1, col), tempQR(row, col));
      tempQR.block(row - 1, col, 2, remaining_cols).applyOnTheLeft(0, 1, GR.adjoint());
    }
  }
  return;
}

void FilterUtils::performQRHouseholder(Eigen::Ref<MatrixX> tempQR, int start_col, int num_cols)
{
  const size_t total_cols = tempQR.cols();
  const size_t total_rows = tempQR.rows();
  if (num_cols == -1)
  {
    num_cols = tempQR.cols() - start_col;
  }
  size_t total_rank = start_col;
  VectorX tempVector(total_cols + 1);
  number_t *tempData = tempVector.data();
  for (size_t k = start_col; k < start_col + num_cols && total_rank < total_rows; ++k)
  {
    size_t remainingRows = total_rows - total_rank;
    size_t remainingCols = total_cols - k - 1;
    number_t beta;
    number_t hCoeff;
    tempQR.col(k).tail(remainingRows).makeHouseholderInPlace(hCoeff, beta);
    if (std::abs(beta) > 1e-6)
    {
      tempQR.coeffRef(total_rank, k) = beta;
      tempQR.bottomRightCorner(remainingRows, remainingCols).applyHouseholderOnTheLeft(tempQR.col(k).tail(remainingRows - 1), hCoeff, tempData + k + 1);
      total_rank += 1;
    }
    else
    {
      tempQR.coeffRef(total_rank, k) = 0;
    }
    tempQR.col(k).tail(remainingRows - 1).setZero();
  }
  return;
}

Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> FilterUtils::genReversePermMat(int n)
{
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm;
  perm.resize(n);
  Eigen::VectorXi new_order(n);
  for (int i = 0; i < n; ++i)
  {
    new_order(i) = n - 1 - i;
  }
  perm.indices() = new_order;
  return perm;
}

Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> FilterUtils::genLadderPermMat(std::vector<int> &first_cols)
{
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm;
  int n = first_cols.size();
  perm.resize(n);
  Eigen::VectorXi indices(n);
  indices.setLinSpaced(n, 0, n - 1);
  std::sort(indices.data(), indices.data() + indices.size(),
            [&first_cols](int a, int b)
            { return first_cols[a] < first_cols[b]; });
  perm.indices() = indices;
  return perm.transpose();
}

void FilterUtils::performPermutationQR(Eigen::Ref<MatrixX> tempQR, std::vector<int> &&first_cols, int start_col, int num_cols)
{
  if (num_cols == -1)
  {
    num_cols = tempQR.cols() - start_col;
  }
  tempQR = FilterUtils::genLadderPermMat(first_cols) * tempQR;
  performQRGivens(tempQR, start_col, num_cols);
  return;
}

std::vector<int> FilterUtils::getFirstColsOfMat(const Eigen::Ref<MatrixX> tempQR)
{
  int num_rows = tempQR.rows();
  int num_cols = tempQR.cols();
  std::vector<int> Hx_first_cols(num_rows, 0);
  constexpr number_t tol_thre = std::is_same_v<number_t, float> ? 5e-8 : 1e-13;
  for (int i = 0; i < num_rows; ++i)
  {
    for (int j = 0; j < num_cols; ++j)
    {
      if (std::abs(tempQR(i, j)) > tol_thre)
      {
        Hx_first_cols[i] = j;
        break;
      }
    }
  }
  return Hx_first_cols;
}