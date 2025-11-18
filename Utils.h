// Utils.h
#pragma once

#include <Eigen/Sparse>

namespace trspv {

	// 构造一维差分矩阵 D (n-1)×n
	Eigen::SparseMatrix<double> build1DDiff(int n);

	// 构造二维 TV 差分矩阵 D2D = [ kron(Dτ, Iβ) ; kron(Iτ, Dβ) ]
	// 返回行数 = (Ntau-1)*Nbeta + Ntau*(Nbeta-1)，列数 = Ntau*Nbeta
	Eigen::SparseMatrix<double> build2DTV(int Ntau, int Nbeta);

	void scaleTVBySteps(Eigen::SparseMatrix<double>& D2D,
		int Ntau, int Nbeta,
		double dlogtau, double dbeta);

} // namespace trspv

