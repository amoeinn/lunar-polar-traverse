// Lit intervals and shadow-edge speeds for the lunar polar illumination model.
//
// For every cell of the planning region and every 15 minute epoch of a year,
// g = (sun elevation) - (that cell's horizon at the sun's azimuth). A cell is
// lit where g > 0, so a first-light transition is a sign change from negative
// to non-negative; it is refined by bisection on g, which is continuous in
// time because the horizon lookup interpolates between rays.
//
// At each transition the shadow edge's level-set normal speed is
// |dg/dt| / |grad_x g|, with grad_x g by central differences over the four
// axis neighbours at the transition time. See PLAN.md, P1, for why that is the
// registered estimator and tests/test_edge_speed_spike.py for its validation.
//
// This file carries a third copy of the sun convention, after horizon.cpp and
// src/frames.py. The probe mode exists so a Python cross-check can hold it to
// the other two; PLAN.md registers that check's tolerance.
//
// Usage: edges <params.json>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "json_flat.hpp"

using Eigen::Vector3d;

namespace {

constexpr double kDeg = 57.295779513082320876798154814105;

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

// Identical to horizon.cpp's: Snyder south polar aspect, the cosine with x.
inline Vector3d map_to_unit(double x, double y, double R) {
    const double r = std::hypot(x, y);
    const double t = r / (2.0 * R);
    const double den = 1.0 + t * t;
    const double sin_theta = 2.0 * t / den;
    const double cos_theta = (1.0 - t * t) / den;
    const double sl = (r > 0.0) ? x / r : 0.0;
    const double cl = (r > 0.0) ? y / r : 1.0;
    return Vector3d(sin_theta * cl, sin_theta * sl, -cos_theta);
}

// Per-cell geometry, built once. Mirrors src.illumination.RegionCells.
struct Cell {
    Vector3d origin, up, east, north;
};

inline Cell make_cell(double x, double y, double h, double R, double obs_h) {
    Cell c;
    const Vector3d u = map_to_unit(x, y, R);
    const double rr = std::hypot(x, y);
    c.up = u;
    c.east = (rr > 0.0) ? Vector3d(-x / rr, y / rr, 0.0) : Vector3d(1.0, 0.0, 0.0);
    c.north = u.cross(c.east);
    c.origin = (R + h + obs_h) * u;
    return c;
}

// Azimuth from local north toward local east, and elevation, of a target.
inline void az_el(const Cell& c, const Vector3d& target, double& az, double& el) {
    const Vector3d d = (target - c.origin).normalized();
    az = std::atan2(d.dot(c.east), d.dot(c.north)) * kDeg;
    if (az < 0.0) az += 360.0;
    el = std::asin(std::max(-1.0, std::min(1.0, d.dot(c.up)))) * kDeg;
}

// Linear interpolation between azimuth rays, as src.frames.horizon_lookup.
inline double horizon_at(const float* prof, int n_az, double az_deg) {
    double a = std::fmod(az_deg, 360.0);
    if (a < 0.0) a += 360.0;
    a *= n_az / 360.0;
    const double fi = std::floor(a);
    const double frac = a - fi;
    int i0 = static_cast<int>(fi) % n_az;
    const int i1 = (i0 + 1) % n_az;
    return (1.0 - frac) * prof[i0] + frac * prof[i1];
}

// The sun at an arbitrary time, linearly interpolated in Cartesian Moon-fixed
// coordinates between the tabulated samples. PLAN.md bounds the chord error.
struct SunTable {
    const double* p;      // 3 doubles per sample
    int n;
    double dt_hours;

    Vector3d at(double t_hours) const {
        double u = t_hours / dt_hours;
        int k = static_cast<int>(std::floor(u));
        if (k < 0) k = 0;
        if (k > n - 2) k = n - 2;
        const double f = u - k;
        const double* a = p + 3 * static_cast<size_t>(k);
        const double* b = a + 3;
        return Vector3d((1.0 - f) * a[0] + f * b[0], (1.0 - f) * a[1] + f * b[1],
                        (1.0 - f) * a[2] + f * b[2]);
    }
};

inline double g_of(const Cell& c, const float* prof, int n_az, const Vector3d& sun,
                   double* az_out = nullptr, double* el_out = nullptr) {
    double az, el;
    az_el(c, sun, az, el);
    if (az_out) *az_out = az;
    if (el_out) *el_out = el;
    return el - horizon_at(prof, n_az, az);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <params.json>\n", argv[0]);
        return 2;
    }
    const json_flat::Object p = json_flat::load(argv[1]);
    const double R = p.num("radius_m");
    const double obs_h = p.num("observer_height_m");
    const int nx = static_cast<int>(p.num("region_nx"));
    const int ny = static_cast<int>(p.num("region_ny"));
    const double rox = p.num("region_ox"), roy = p.num("region_oy");
    const double rpx = p.num("region_px");
    const int n_az = static_cast<int>(p.num("n_azimuth"));
    const double pin_deg = p.num("pin_horizon_deg");
    const size_t ncell = static_cast<size_t>(nx) * ny;

    const auto horizon = read_raw<float>(p.str("horizon_path"), ncell * n_az);
    const auto hdist = read_raw<float>(p.str("horizon_dist_path"), ncell * n_az);
    const auto height = read_raw<float>(p.str("height_path"), ncell);
    const int n_sun = static_cast<int>(p.num("n_sun"));
    const auto sunbuf = read_raw<double>(p.str("sun_path"), 3u * n_sun);
    const SunTable sun{sunbuf.data(), n_sun, p.num("sun_dt_hours")};

    std::vector<Cell> cells(ncell);
    for (int r = 0; r < ny; ++r)
        for (int c = 0; c < nx; ++c) {
            const size_t m = static_cast<size_t>(r) * nx + c;
            cells[m] = make_cell(rox + (c + 0.5) * rpx, roy - (r + 0.5) * rpx,
                                 height[m], R, obs_h);
        }

    // Probe mode: g, azimuth and elevation at listed (cell, time) records.
    if (p.strings.count("probe_path")) {
        const int n_probe = static_cast<int>(p.num("n_probe"));
        const auto idx = read_raw<int32_t>(p.str("probe_path"), n_probe);
        const auto tim = read_raw<double>(p.str("probe_time_path"), n_probe);
        std::vector<double> out(3u * n_probe);
        for (int i = 0; i < n_probe; ++i) {
            const size_t m = static_cast<size_t>(idx[i]);
            double az, el;
            const double g = g_of(cells[m], &horizon[m * n_az], n_az, sun.at(tim[i]),
                                  &az, &el);
            out[3 * i] = g;
            out[3 * i + 1] = az;
            out[3 * i + 2] = el;
        }
        write_raw(p.str("probe_out_path"), out);
        std::printf("probe: %d records written\n", n_probe);
        return 0;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const double dt = sun.dt_hours;
    const double h_der = 1.0e-3;          // hours, for the time derivative
    std::vector<uint32_t> counts(ncell, 0);
    std::vector<std::vector<float>> sp_t, rl_t;
    std::vector<std::vector<uint8_t>> fl_t;
    std::vector<std::vector<uint32_t>> cl_t;
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    sp_t.resize(nthreads); rl_t.resize(nthreads); fl_t.resize(nthreads);
    cl_t.resize(nthreads);

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& sp = sp_t[tid];
        auto& rl = rl_t[tid];
        auto& fl = fl_t[tid];
        auto& cl = cl_t[tid];
#pragma omp for schedule(dynamic, 8)
        for (int r = 0; r < ny; ++r) {
            for (int c = 0; c < nx; ++c) {
                const size_t m = static_cast<size_t>(r) * nx + c;
                const float* prof = &horizon[m * n_az];
                const bool interior = (r > 0 && r < ny - 1 && c > 0 && c < nx - 1);
                const size_t nb[4] = {m - nx, m + nx, m - 1, m + 1};   // N, S, W, E
                double gprev = g_of(cells[m], prof, n_az, sun.at(0.0));
                for (int k = 1; k < n_sun; ++k) {
                    const double tk = k * dt;
                    const double gk = g_of(cells[m], prof, n_az, sun.at(tk));
                    if (!(gprev < 0.0 && gk >= 0.0)) { gprev = gk; continue; }
                    double a = tk - dt, b = tk;
                    for (int it = 0; it < 40; ++it) {
                        const double mid = 0.5 * (a + b);
                        if (g_of(cells[m], prof, n_az, sun.at(mid)) < 0.0) a = mid;
                        else b = mid;
                    }
                    const double ts = 0.5 * (a + b);
                    ++counts[m];
                    if (!interior) { gprev = gk; continue; }

                    double az_s, el_s;
                    g_of(cells[m], prof, n_az, sun.at(ts), &az_s, &el_s);
                    const double dgdt =
                        (g_of(cells[m], prof, n_az, sun.at(ts + h_der)) -
                         g_of(cells[m], prof, n_az, sun.at(ts - h_der))) / (2.0 * h_der);
                    double gn[4];
                    bool pinned = false;
                    for (int q = 0; q < 4; ++q) {
                        const float* pq = &horizon[nb[q] * n_az];
                        gn[q] = g_of(cells[nb[q]], pq, n_az, sun.at(ts));
                        if (std::fabs(horizon_at(pq, n_az, az_s) -
                                      horizon_at(prof, n_az, az_s)) > pin_deg)
                            pinned = true;
                    }
                    const double gx = (gn[3] - gn[2]) / (2.0 * rpx);   // E - W
                    const double gy = (gn[0] - gn[1]) / (2.0 * rpx);   // N - S
                    const double grad = std::hypot(gx, gy);
                    const double speed = (grad > 0.0) ? std::fabs(dgdt) / grad
                                                      : std::numeric_limits<double>::infinity();

                    int bin = static_cast<int>(std::lround(az_s)) % n_az;
                    if (bin < 0) bin += n_az;
                    const double s_arc = hdist[m * n_az + bin];
                    const double th = horizon[m * n_az + bin];
                    const double psi = s_arc / R;
                    const double tt = std::tan(th / kDeg);
                    const double cc = 1.0 - std::cos(psi), ss = std::sin(psi);
                    const double den = 1.0 - cc - tt * ss;
                    const double relief =
                        (s_arc > 0.0 && den > 0.0)
                            ? (R * (tt * ss + cc) + height[m]) / den - height[m]
                            : 0.0;

                    sp.push_back(static_cast<float>(speed));
                    rl.push_back(static_cast<float>(relief));
                    fl.push_back(static_cast<uint8_t>(pinned ? 1 : 0));
                    cl.push_back(static_cast<uint32_t>(m));
                    gprev = gk;
                }
            }
        }
    }

    std::vector<float> speeds, reliefs;
    std::vector<uint8_t> flags;
    std::vector<uint32_t> owners;
    for (int i = 0; i < nthreads; ++i) {
        speeds.insert(speeds.end(), sp_t[i].begin(), sp_t[i].end());
        reliefs.insert(reliefs.end(), rl_t[i].begin(), rl_t[i].end());
        flags.insert(flags.end(), fl_t[i].begin(), fl_t[i].end());
        owners.insert(owners.end(), cl_t[i].begin(), cl_t[i].end());
    }
    write_raw(p.str("out_speed"), speeds);
    write_raw(p.str("out_relief"), reliefs);
    write_raw(p.str("out_flags"), flags);
    write_raw(p.str("out_cell"), owners);
    write_raw(p.str("out_counts"), counts);

    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    uint64_t total = 0;
    for (auto v : counts) total += v;
    json_flat::Writer w;
    w.add("n_cells", static_cast<double>(ncell));
    w.add("n_epochs", static_cast<double>(n_sun));
    w.add("cadence_hours", dt);
    w.add("transitions_total", static_cast<double>(total));
    w.add("edges_recorded", static_cast<double>(speeds.size()));
    w.add("pin_horizon_deg", pin_deg);
    w.add("bisection_iterations", 40.0);
    w.add("time_derivative_hours", h_der);
    w.add("sun_interpolation", "linear in Cartesian Moon-fixed coordinates");
    w.add("estimator", "|dg/dt| / |grad_x g|, central differences over four axis neighbours");
    w.add("threads", static_cast<double>(nthreads));
    w.add("seconds", secs);
    w.add("source_commit", p.str("source_commit"));
    w.save(p.str("out_sidecar"));
    std::printf("%zu cells, %d epochs, %llu transitions, %zu edges, %.1f s\n",
                ncell, n_sun, static_cast<unsigned long long>(total), speeds.size(), secs);
    return 0;
}
