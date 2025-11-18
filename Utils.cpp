// Utils.cpp
#include "Utils.h"
#include <unsupported/Eigen/KroneckerProduct>
#include <vector>

namespace trspv {

    Eigen::SparseMatrix<double> build1DDiff(int n) {
        Eigen::SparseMatrix<double> D(n > 1 ? n - 1 : 0, n);
        if (n < 2) return D;
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
        // 1) build 1D diffs
        auto Dtau = build1DDiff(Ntau);
        auto Dbeta = build1DDiff(Nbeta);

        // 2) identity mats
        Eigen::SparseMatrix<double> Itau(Ntau, Ntau), Ibeta(Nbeta, Nbeta);
        Itau.setIdentity();
        Ibeta.setIdentity();

        // 3) evaluate Kron products into real SparseMatrices
        Eigen::SparseMatrix<double> Dtop = Eigen::kroneckerProduct(Ibeta, Dtau).eval(); // ¦Ó-TV
        Eigen::SparseMatrix<double> Dbottom = Eigen::kroneckerProduct(Dbeta, Itau).eval(); // ¦Â-TV

        // 4) allocate the big D2D
        int rows = Dtop.rows() + Dbottom.rows();
        int cols = Ntau * Nbeta;
        Eigen::SparseMatrix<double> D2D(rows, cols);
        D2D.reserve(Dtop.nonZeros() + Dbottom.nonZeros());

        // 5) insert Dtop
        for (int outer = 0; outer < Dtop.outerSize(); ++outer) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Dtop, outer); it; ++it) {
                D2D.insert(it.row(), it.col()) = it.value();
            }
        }

        // 6) insert Dbottom with row offset
        int roff = Dtop.rows();
        for (int outer = 0; outer < Dbottom.outerSize(); ++outer) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(Dbottom, outer); it; ++it) {
                D2D.insert(it.row() + roff, it.col()) = it.value();
            }
        }

        D2D.makeCompressed();
        return D2D;
    }

    void scaleTVBySteps(Eigen::SparseMatrix<double>& D2D,
        int Ntau, int Nbeta,
        double dlogtau, double dbeta)
    {
        auto safe = [](double x) { return (x > 0 ? x : 1.0); };
        dlogtau = std::abs(dlogtau);
        dbeta = std::abs(dbeta);

        const int rowsTau = Nbeta * std::max(0, Ntau - 1);
        const int rowsBeta = Ntau * std::max(0, Nbeta - 1);

        for (int k = 0; k < D2D.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(D2D, k); it; ++it) {
                double s = (it.row() < rowsTau) ? (1.0 / safe(dlogtau))
                    : (1.0 / safe(dbeta));
                it.valueRef() *= s;
            }
        }
    }


} // namespace trspv
