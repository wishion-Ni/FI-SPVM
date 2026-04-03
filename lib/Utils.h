// Utils.h
#pragma once

#include <Eigen/Sparse>

namespace trspv {

// Build the 1D difference matrix D with shape (n - 1) x n.
Eigen::SparseMatrix<double> build1DDiff(int n);

// Build the 2D TV matrix D2D = [ kron(Dtau, Ibeta) ; kron(Itau, Dbeta) ].
// Rows = (Ntau - 1) * Nbeta + Ntau * (Nbeta - 1), cols = Ntau * Nbeta.
Eigen::SparseMatrix<double> build2DTV(int Ntau, int Nbeta);

void scaleTVBySteps(Eigen::SparseMatrix<double>& D2D,
    int Ntau, int Nbeta,
    double dlogtau, double dbeta);

} // namespace trspv
