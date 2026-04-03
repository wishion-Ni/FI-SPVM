#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Config.h"

namespace trspv {

class ParamSelector {
public:
    ParamSelector(const Eigen::MatrixXcd& A,
                  const Eigen::VectorXcd& b,
                  const Eigen::SparseMatrix<double>& D,
                  const ADMMConfig& baseCfg,
                  const ParamSelectionConfig& psc);

    ADMMConfig select() const;

private:
    Eigen::MatrixXcd A_;
    Eigen::VectorXcd b_;
    Eigen::SparseMatrix<double> D_;
    ADMMConfig baseCfg_;
    ParamSelectionConfig psc_;
};

}  // namespace trspv
