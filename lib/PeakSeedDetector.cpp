// === PeakSeedDetector.cpp ===
#include "PeakSeedDetector.h"
#include <iterator>

// ----------- utilities -------------
std::vector<double> PeakSeedDetector::logspace(double a, double b, size_t n) const {
    std::vector<double> v(n);
    double step = (b - a) / static_cast<double>(n - 1);
    for (size_t i = 0; i < n; ++i) v[i] = std::pow(10.0, a + step * i);
    return v;
}
std::vector<double> PeakSeedDetector::movingAverage(const std::vector<double>& v, int w) const {
    if (w < 3 || w % 2 == 0) return v;
    int h = w / 2; std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        size_t from = (i < h ? 0 : i - h); size_t to = std::min(v.size() - 1, i + h);
        double s = 0; for (size_t j = from; j <= to; ++j) s += v[j];
        out[i] = s / static_cast<double>(to - from + 1);
    }
    return out;
}
void PeakSeedDetector::computeDifferences(const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& d) const {
    d.resize(x.size() - 1);
    for (size_t i = 0; i + 1 < x.size(); ++i) d[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
}
void PeakSeedDetector::computeAkimaSlope(const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& m) const {
    std::vector<double> d; computeDifferences(x, y, d); size_t n = x.size();
    std::vector<double> dd(n + 3);
    dd[0] = d[1]; dd[1] = d[0]; std::copy(d.begin(), d.end(), dd.begin() + 2);
    dd[n + 1] = d[n - 2]; dd[n + 2] = d[n - 3]; m.resize(n);
    for (size_t i = 0; i < n; ++i) {
        double w1 = std::fabs(dd[i + 3] - dd[i + 2]); double w2 = std::fabs(dd[i + 1] - dd[i]);
        m[i] = (w1 + w2 > 0) ? (w1 * dd[i + 1] + w2 * dd[i + 2]) / (w1 + w2) : 0.5 * (dd[i + 2] + dd[i + 1]);
    }
}
void PeakSeedDetector::computePCHIPSlope(const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& m) const {
    std::vector<double> d; computeDifferences(x, y, d); size_t n = x.size(); m.resize(n);
    m[0] = d[0]; m[n - 1] = d[n - 2];
    for (size_t i = 1; i < n - 1; ++i) { if (d[i - 1] * d[i] > 0) { double w1 = 2 * (x[i] - x[i - 1]) + (x[i + 1] - x[i]); double w2 = (x[i] - x[i - 1]) + 2 * (x[i + 1] - x[i]); m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]); } else m[i] = 0.0; }
}
inline double PeakSeedDetector::evalHermite(double xi, double x0, double x1, double y0, double y1, double m0, double m1) const {
    double dx = x1 - x0; double t = (xi - x0) / dx; double t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * dx * m0 + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * dx * m1;
}
std::vector<double> PeakSeedDetector::akimaInterp(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& xi) const {
    std::vector<double> m; computeAkimaSlope(x, y, m); std::vector<double> out(xi.size());
    for (size_t k = 0; k < xi.size(); ++k) { double xv = xi[k]; auto it = std::upper_bound(x.begin(), x.end(), xv); size_t i = (it == x.begin() ? 0 : std::min<size_t>(it - x.begin() - 1, x.size() - 2)); out[k] = evalHermite(xv, x[i], x[i + 1], y[i], y[i + 1], m[i], m[i + 1]); }
    return out;
}
std::vector<double> PeakSeedDetector::pchipInterp(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& xi) const {
    std::vector<double> m; computePCHIPSlope(x, y, m); std::vector<double> out(xi.size());
    for (size_t k = 0; k < xi.size(); ++k) { double xv = xi[k]; auto it = std::upper_bound(x.begin(), x.end(), xv); size_t i = (it == x.begin() ? 0 : std::min<size_t>(it - x.begin() - 1, x.size() - 2)); out[k] = evalHermite(xv, x[i], x[i + 1], y[i], y[i + 1], m[i], m[i + 1]); }
    return out;
}

// ---------------- main operator() ----------------
std::vector<double> PeakSeedDetector::operator()(const std::vector<double>& freq_Hz, const std::vector<std::complex<double>>& spv) const {
    if (freq_Hz.size() != spv.size() || freq_Hz.size() < 3) return {};
    // 1. log‑f grid
    double fmin = *std::min_element(freq_Hz.begin(), freq_Hz.end()); double fmax = *std::max_element(freq_Hz.begin(), freq_Hz.end());
    auto f_log = logspace(std::log10(fmin), std::log10(fmax), freq_Hz.size() * opt_.interp_factor);
    // 2. Im part & sort
    std::vector<double> im_in(spv.size()); for (size_t i = 0; i < spv.size(); ++i) im_in[i] = std::imag(spv[i]);
    std::vector<size_t> order(freq_Hz.size()); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {return freq_Hz[a] < freq_Hz[b]; });
    std::vector<double> f_sorted(freq_Hz.size()), im_sorted(freq_Hz.size());
    for (size_t i = 0; i < order.size(); ++i) { f_sorted[i] = freq_Hz[order[i]]; im_sorted[i] = im_in[order[i]]; }
    // 3. interpolation
    std::vector<double> im_intp = (opt_.interp_type == "akima" ? akimaInterp(f_sorted, im_sorted, f_log) : pchipInterp(f_sorted, im_sorted, f_log));
    if (opt_.smooth_window >= 3) im_intp = movingAverage(im_intp, opt_.smooth_window);
    // 4. peak detection (max & min)
    double maxAbs = *std::max_element(im_intp.begin(), im_intp.end(), [](double a, double b) {return std::fabs(a) < std::fabs(b); }); maxAbs = std::fabs(maxAbs);
    std::vector<double> candidates;
    for (size_t i = 1; i + 1 < im_intp.size(); ++i) {
        double prevDiff = im_intp[i] - im_intp[i - 1]; double nextDiff = im_intp[i + 1] - im_intp[i];
        bool isMax = (prevDiff > 0 && nextDiff < 0); bool isMin = (prevDiff < 0 && nextDiff>0);
        if (!(isMax || isMin)) continue;
        if (std::fabs(im_intp[i]) < opt_.peak_prominence * maxAbs) continue;
        candidates.push_back(f_log[i]);
    }
    if (candidates.empty()) return {};
    // 5. rank by |Im| & distance filter
    std::vector<std::pair<double, double>> pk; for (double f : candidates) { auto it = std::lower_bound(f_log.begin(), f_log.end(), f); size_t idx = std::distance(f_log.begin(), it); pk.emplace_back(std::fabs(im_intp[idx]), f); } std::sort(pk.begin(), pk.end(), [](auto a, auto b) {return a.first > b.first; });
    std::vector<double> sel;
    for (auto& p : pk) { double f = p.second; bool close = false; for (double sf : sel) { if (std::fabs(std::log10(f) - std::log10(sf)) < opt_.peak_dist_dec) { close = true; break; } } if (!close) sel.push_back(f); }
    std::sort(sel.begin(), sel.end());
    constexpr double twoPi = 6.283185307179586; std::vector<double> tau(sel.size()); for (size_t i = 0; i < sel.size(); ++i) tau[i] = 1.0 / (twoPi * sel[i]);
    return tau;
}

// === end PeakSeedDetector ===
