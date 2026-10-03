// Lit and dark intervals per cell, shared by the planner and the executive.
//
// Moved out of planner.cpp unchanged on 2026-10-03, so the executive's monitor
// uses the planner's own illumination model rather than a second
// implementation of it. That is deliberate: the monitor is the planner side of
// the comparison and the replay checker is the independent judge, so a second
// interval implementation here would test nothing and could disagree with the
// plan being monitored. The move was verified by rerunning a cached planner
// instance and asserting its outputs are byte-identical.
//
// Mirrors src/intervals.py exactly, including the guard in dark_spans that
// stops a zero-length step from looping.

#ifndef INTERVALS_FLAT_HPP
#define INTERVALS_FLAT_HPP

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace ivflat {

struct Intervals {
    std::vector<double> times;
    std::vector<int64_t> offsets;
    std::vector<uint8_t> lit0;
    double horizon_h = 0.0;

    const double* begin_of(int c) const { return times.data() + offsets[c]; }
    const double* end_of(int c) const { return times.data() + offsets[c + 1]; }

    bool is_lit(int c, double t) const {
        const size_t k = static_cast<size_t>(
            std::upper_bound(begin_of(c), end_of(c), t) - begin_of(c));
        return (lit0[c] != 0) != (k % 2 == 1);
    }
    double next_change(int c, double t) const {
        const double* p = std::upper_bound(begin_of(c), end_of(c), t);
        return p < end_of(c) ? *p : horizon_h;
    }
    void transitions_in(int c, double t0, double t1, std::vector<double>& out) const {
        const double* lo = std::upper_bound(begin_of(c), end_of(c), t0);
        const double* hi = std::lower_bound(begin_of(c), end_of(c), t1);
        for (const double* p = lo; p < hi; ++p) out.push_back(*p);
    }

    void dark_spans(int a, int b, double t0, double t1,
                    std::vector<std::pair<double, double>>& merged) const {
        static thread_local std::vector<std::pair<double, double>> spans;
        spans.clear();
        const int cs[2] = {a, b};
        for (int i = 0; i < 2; ++i) {
            double t = t0;
            while (t < t1) {
                const double nxt = std::min(next_change(cs[i], t), t1);
                if (!is_lit(cs[i], t)) spans.emplace_back(t, nxt);
                t = nxt > t ? nxt : t1;
            }
        }
        merged.clear();
        if (spans.empty()) return;
        std::sort(spans.begin(), spans.end());
        merged.push_back(spans[0]);
        for (size_t i = 1; i < spans.size(); ++i) {
            if (spans[i].first <= merged.back().second)
                merged.back().second = std::max(merged.back().second, spans[i].second);
            else
                merged.push_back(spans[i]);
        }
    }

    void shadow_along(int a, int b, double t_dep, double t_arr, double s_dep,
                      double& s_arr, double& s_max) const {
        static thread_local std::vector<std::pair<double, double>> spans;
        dark_spans(a, b, t_dep, t_arr, spans);
        double s = s_dep;
        s_max = s_dep;
        s_arr = 0.0;
        for (const auto& sp : spans) {
            s = (sp.first <= t_dep ? s_dep : 0.0) + (sp.second - sp.first);
            s_max = std::max(s_max, s);
            s_arr = sp.second >= t_arr ? s : 0.0;
        }
        if (spans.empty()) s_arr = 0.0;
    }
};

}  // namespace ivflat

#endif  // INTERVALS_FLAT_HPP
