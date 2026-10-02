// Time-dependent planner: the label-setting search, in C++.
//
// A port of src/planner.py, not a reimplementation: every expression is in the
// same order so the two carry the same doubles and can be held to exact
// agreement. The Python stays as the cross-check, in the manner of check 1c.
// PLAN.md, "Planners", registers the tolerance.
//
// State is (cell, arrival time, continuous shadow time). Waiting is allowed in
// lit cells only, which is what makes the dominance rule sound; see PLAN.md for
// the reachability clause and why it is not optional.
//
// Usage: planner <params.json>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "json_flat.hpp"

namespace {

template <typename T>
std::vector<T> read_raw(const std::string& path, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(T)));
    if (f.gcount() != static_cast<std::streamsize>(n * sizeof(T)))
        throw std::runtime_error("short read on " + path);
    return v;
}

template <typename T>
void write_raw(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(T)));
}

// Lit and dark intervals per cell. Mirrors src/intervals.py exactly, including
// the guard in dark_spans that stops a zero-length step from looping.
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

struct Label {
    double t, s;
    int cell;
    int64_t lid;
    bool operator>(const Label& o) const {
        if (t != o.t) return t > o.t;
        if (s != o.s) return s > o.s;
        if (cell != o.cell) return cell > o.cell;
        return lid > o.lid;
    }
};

struct Result {
    double arrival = std::numeric_limits<double>::quiet_NaN();
    int32_t path_len = 0;
    uint8_t aborted = 0;
    int64_t pushed = 0, popped = 0, touched = 0, max_labels = 0;
    double seconds = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <params.json>\n", argv[0]);
        return 2;
    }
    const json_flat::Object p = json_flat::load(argv[1]);
    const int n_cells = p.integer("n_cells");
    const int n_edges = p.integer("n_edges");
    const int n_trans = p.integer("n_transitions");
    const int n_inst = p.integer("n_instances");
    const double tol = p.num("shadow_tol_h");
    const double t_start = p.num("t_start_h");
    const double limit_s = p.num("time_limit_s");
    // Both approximations are off by default and both are pessimistic: they
    // lose feasible plans and cannot invent them. See src/planner.py.
    const double grid_h = p.num("shadow_grid_h", 0.0);
    const bool relax = p.num("relax_dominance", 0.0) != 0.0;

    const auto indptr = read_raw<int64_t>(p.str("indptr_path"), n_cells + 1);
    const auto indices = read_raw<int32_t>(p.str("indices_path"), n_edges);
    const auto travel = read_raw<double>(p.str("travel_path"), n_edges);
    Intervals iv;
    iv.times = read_raw<double>(p.str("itimes_path"), n_trans);
    iv.offsets = read_raw<int64_t>(p.str("ioffsets_path"), n_cells + 1);
    iv.lit0 = read_raw<uint8_t>(p.str("ilit0_path"), n_cells);
    iv.horizon_h = p.num("horizon_h");
    const auto pairs = read_raw<int32_t>(p.str("pairs_path"), 2u * n_inst);

    // Two exact prunings, both optional so the engine can be run without them
    // and the answers compared. See src/planner.py for why each is exact.
    const bool use_component = p.strings.count("component_path") != 0;
    const double v_cap = p.num("v_cap", 0.0);
    const bool use_bound = v_cap > 0.0 && p.strings.count("position_path") != 0;
    std::vector<int32_t> component;
    std::vector<double> position;
    if (use_component) component = read_raw<int32_t>(p.str("component_path"), n_cells);
    if (use_bound) position = read_raw<double>(p.str("position_path"), 3u * n_cells);

    std::vector<Result> out(n_inst);
    const bool want_paths = p.strings.count("out_path_cells") != 0;
    std::vector<int32_t> path_cells;
    std::vector<double> path_times;
    std::vector<int64_t> path_off(n_inst + 1, 0);
    std::vector<std::vector<std::pair<double, double>>> labels(n_cells);
    std::vector<std::pair<int64_t, int>> parent;
    std::vector<int> touched_cells;
    std::vector<double> trans;

    for (int q = 0; q < n_inst; ++q) {
        const int start = pairs[2 * q], goal = pairs[2 * q + 1];
        if (use_component && component[start] != component[goal]) {
            Result skip;
            skip.seconds = 0.0;
            out[q] = skip;
            continue;
        }
        std::vector<double> to_goal;
        if (use_bound) {
            to_goal.resize(n_cells);
            const double gx = position[3 * goal], gy = position[3 * goal + 1];
            const double gz = position[3 * goal + 2];
            for (int c = 0; c < n_cells; ++c) {
                const double dx = position[3 * c] - gx, dy = position[3 * c + 1] - gy;
                const double dz = position[3 * c + 2] - gz;
                to_goal[c] = std::sqrt(dx * dx + dy * dy + dz * dz) / v_cap;
            }
        }
        for (int c : touched_cells) labels[c].clear();
        touched_cells.clear();
        parent.clear();
        std::vector<double> label_t;
        std::priority_queue<Label, std::vector<Label>, std::greater<Label>> heap;
        Result r;
        const auto wall0 = std::chrono::steady_clock::now();

        auto admit = [&](int cell, double t, double s, int64_t par) {
            if (s > tol) return;
            if (t > iv.horizon_h) return;
            if (use_bound && t + to_goal[cell] > iv.horizon_h) return;
            for (const auto& L : labels[cell]) {
                if (L.first <= t && L.second <= s) {
                    if (L.first == t) return;
                    if (relax) return;
                    if (iv.is_lit(cell, L.first) && iv.next_change(cell, L.first) > t)
                        return;
                }
            }
            if (labels[cell].empty()) touched_cells.push_back(cell);
            labels[cell].emplace_back(t, s);
            const int64_t lid = static_cast<int64_t>(parent.size());
            parent.emplace_back(par, cell);
            label_t.push_back(t);
            heap.push(Label{t, s, cell, lid});
            ++r.pushed;
        };

        admit(start, t_start, 0.0, -1);
        int64_t checks = 0;
        while (!heap.empty()) {
            const Label top = heap.top();
            heap.pop();
            ++r.popped;
            if (++checks % 512 == 0 && limit_s > 0.0) {
                const double el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - wall0).count();
                if (el > limit_s) { r.aborted = 1; break; }
            }
            if (top.cell == goal) {
                r.arrival = top.t;
                std::vector<int32_t> rc;
                std::vector<double> rt;
                int64_t node = top.lid;
                while (node >= 0) {
                    ++r.path_len;
                    rc.push_back(static_cast<int32_t>(parent[node].second));
                    rt.push_back(label_t[node]);
                    node = parent[node].first;
                }
                if (want_paths) {
                    for (size_t i = rc.size(); i-- > 0;) {
                        path_cells.push_back(rc[i]);
                        path_times.push_back(rt[i]);
                    }
                }
                break;
            }
            const bool lit_here = iv.is_lit(top.cell, top.t);
            const double window_end = lit_here ? iv.next_change(top.cell, top.t) : top.t;
            for (int64_t e = indptr[top.cell]; e < indptr[top.cell + 1]; ++e) {
                const int nb = indices[e];
                const double tau = travel[e];
                std::set<double> cands;
                if (lit_here) {
                    const double hi = window_end + tau;
                    trans.clear();
                    iv.transitions_in(top.cell, top.t, hi, trans);
                    iv.transitions_in(nb, top.t, hi, trans);
                    cands.insert(top.t);
                    for (double tr : trans) {
                        cands.insert(tr);
                        cands.insert(tr - tau);
                        cands.insert(tr - tol);
                        cands.insert(tr + tol - tau);
                    }
                } else {
                    cands.insert(top.t);
                }
                for (double d : cands) {
                    if (!(d >= top.t && d < window_end) && !(!lit_here && d == top.t))
                        continue;
                    const double arrive = d + tau;
                    const double s_dep = lit_here ? 0.0 : top.s;
                    double s_arr, s_max;
                    iv.shadow_along(top.cell, nb, d, arrive, s_dep, s_arr, s_max);
                    if (grid_h > 0.0) {
                        s_arr = std::ceil(s_arr / grid_h) * grid_h;
                        s_max = std::ceil(s_max / grid_h) * grid_h;
                    }
                    if (s_max > tol) continue;
                    admit(nb, arrive, s_arr, top.lid);
                }
            }
        }
        r.seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall0).count();
        for (int c : touched_cells) {
            ++r.touched;
            r.max_labels = std::max<int64_t>(r.max_labels,
                                             static_cast<int64_t>(labels[c].size()));
        }
        out[q] = r;
        path_off[q + 1] = static_cast<int64_t>(path_cells.size());
    }

    std::vector<double> arr(n_inst), secs(n_inst);
    std::vector<uint8_t> flags(n_inst);
    std::vector<int32_t> plen(n_inst);
    std::vector<int64_t> stats(4u * n_inst);
    for (int q = 0; q < n_inst; ++q) {
        arr[q] = out[q].arrival;
        secs[q] = out[q].seconds;
        flags[q] = out[q].aborted;
        plen[q] = out[q].path_len;
        stats[4 * q] = out[q].pushed;
        stats[4 * q + 1] = out[q].popped;
        stats[4 * q + 2] = out[q].touched;
        stats[4 * q + 3] = out[q].max_labels;
    }
    write_raw(p.str("out_arrival"), arr);
    write_raw(p.str("out_seconds"), secs);
    write_raw(p.str("out_flags"), flags);
    write_raw(p.str("out_pathlen"), plen);
    write_raw(p.str("out_stats"), stats);
    if (want_paths) {
        write_raw(p.str("out_path_cells"), path_cells);
        write_raw(p.str("out_path_times"), path_times);
        write_raw(p.str("out_path_offsets"), path_off);
    }
    std::printf("planner: %d instances\n", n_inst);
    return 0;
}
