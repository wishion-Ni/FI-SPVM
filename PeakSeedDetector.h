// === PeakSeedDetector.h ===
#pragma once
#include <vector>
#include <complex>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>

/**
 * Detect dominant kinetic processes by locating local extrema on the
 * Im(SPV)‑vs‑log‑frequency curve.  Designed as a lightweight pre‑processing
 * step before DynRec inversion.
 */
class PeakSeedDetector {
public:
    struct Options {
        std::string interp_type = "akima";  //!< "akima" | "pchip"
        int   smooth_window = 5;         //!< odd ≥3; 0 → no smoothing
        double peak_prominence = 0.02;      //!< relative threshold (0‑1)
        double peak_dist_dec = 0.15;      //!< min distance between peaks (decades)
        size_t interp_factor = 4;         //!< up‑sample ×factor
    };

    explicit PeakSeedDetector(const Options& opt = Options{}) : opt_(opt) {}

    /**
     * @param freq_Hz  modulation frequencies (Hz)
     * @param spv      complex SPV, same length
     * @return         τ seeds (s) in ascending order
     */
    std::vector<double> operator()(const std::vector<double>& freq_Hz,
        const std::vector<std::complex<double>>& spv) const;
private:
    Options opt_;

    // helpers -------------------------------------------------------------
    std::vector<double> logspace(double log10_start, double log10_end, size_t n) const;
    std::vector<double> movingAverage(const std::vector<double>& v, int w) const;

    // Akima / PCHIP interpolation utilities ------------------------------
    void computeDifferences(const std::vector<double>& x,
        const std::vector<double>& y,
        std::vector<double>& d) const;
    void computeAkimaSlope(const std::vector<double>& x,
        const std::vector<double>& y,
        std::vector<double>& m) const;
    void computePCHIPSlope(const std::vector<double>& x,
        const std::vector<double>& y,
        std::vector<double>& m) const;
    double evalHermite(double xi,
        double x0, double x1,
        double y0, double y1,
        double m0, double m1) const;
    std::vector<double> akimaInterp(const std::vector<double>& x,
        const std::vector<double>& y,
        const std::vector<double>& xi) const;
    std::vector<double> pchipInterp(const std::vector<double>& x,
        const std::vector<double>& y,
        const std::vector<double>& xi) const;
};
