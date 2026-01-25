
#pragma once

#include <Eigen/Eigen>
#include <memory>
#include "types/Type.h"

class FilterUtils
{

public:
    static void performQRGivens(Eigen::Ref<MatrixX> tempQR, int start_col, int num_cols = -1);

    static void performQRHouseholder(Eigen::Ref<MatrixX> tempQR, int start_col, int num_cols = -1);

    static Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> genReversePermMat(int n);

    static Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> genLadderPermMat(std::vector<int> &first_cols);

    static void performPermutationQR(Eigen::Ref<MatrixX> tempQR, std::vector<int> &&first_cols, int start_col = 0, int num_cols = -1);

    static std::vector<int> getFirstColsOfMat(const Eigen::Ref<MatrixX> tempQR);

    static void flipToHead(Eigen::Ref<MatrixX> Mat, const int dim);

private:
    /**
     * All function in this class should be static.
     * Thus an instance of this class cannot be created.
     */
    FilterUtils() {}
};