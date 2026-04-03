// Utils.cpp
#include "Utils.h"

#include <unsupported/Eigen/KroneckerProduct>

#include <cmath>
#include <vector>

namespace trspv {

Eigen::SparseMatrix<double> build1DDiff(int n) {
    Eigen::SparseMatrix<double> D(n > 1 ? n - 1 : 0, n);
    if (n < 2) {
        return D;
    }

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(2 * (n - 1));
    for (int i = 0; i < n - 1; ++i) {
        trips.emplace_back(i, i, -1.0);
        trips.emplace_back(i, i + 1, +1.0);
    }
    D.setFromTriplets(trips.begin(), trips.end());
    return D;
}

Eigen::SparseMatrix<double> build2DTV(int Ntau, int Nbeta) {
    auto Dtau = build1DDiff(Ntau);
    auto Dbeta = build1DDiff(Nbeta);

    Eigen::SparseMatrix<double> Itau(Ntau, Ntau), Ibeta(Nbeta, Nbeta);
    Itau.setIdentity();
    Ibeta.setIdentity();

    Eigen::SparseMatrix<double> Dtop = Eigen::kroneckerProduct(Ibeta, Dtau).eval();
    Eigen::SparseMatrix<double> Dbottom = Eigen::kroneckerProduct(Dbeta, Itau).eval();

    const int rows = Dtop.rows() + Dbottom.rows();
    const int cols = Ntau * Nbeta;
    Eigen::SparseMatrix<double> D2D(rows, cols);
    D2D.reserve(Dtop.nonZeros() + Dbottom.nonZeros());

    for (int outer = 0; outer < Dtop.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(Dtop, outer); it; ++it) {
            D2D.insert(it.row(), it.col()) = it.value();
        }
    }

    const int rowOffset = Dtop.rows();
    for (int outer = 0; outer < Dbottom.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(Dbottom, outer); it; ++it) {
            D2D.insert(it.row() + rowOffset, it.col()) = it.value();
        }
    }

    D2D.makeCompressed();
    return D2D;
}

void scaleTVBySteps(Eigen::SparseMatrix<double>& D2D,
    int Ntau, int Nbeta,
    double dlogtau, double dbeta) {
    auto safe = [](double x) { return (x > 0 ? x : 1.0); };
    dlogtau = std::abs(dlogtau);
    dbeta = std::abs(dbeta);

    const int rowsTau = Nbeta * std::max(0, Ntau - 1);

    for (int k = 0; k < D2D.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(D2D, k); it; ++it) {
            const double scale = (it.row() < rowsTau)
                ? (1.0 / safe(dlogtau))
                : (1.0 / safe(dbeta));
            it.valueRef() *= scale;
        }
    }
}

} // namespace trspv
