// Horizon precompute for the lunar polar illumination model.
//
// For each cell of the planning region, the maximum terrain elevation angle
// per 1 degree azimuth bin, ray marched on great circles through a near-field
// 20 m DEM and a far-field 80 m DEM. Every elevation angle comes from exact 3D
// positions on the reference sphere; there is no d^2/2R anywhere. See PLAN.md,
// "Horizon model", for the design and for why each constant is what it is.
//
// Usage: horizon <params.json>
// Writes horizon.f32, horizon_dist.f32 and a sidecar JSON, paths from params.

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "json_flat.hpp"

using Eigen::Vector3d;

namespace {

// A north-up raster of float32 heights on the polar stereographic grid.
struct Grid {
    std::vector<float> h;
    int width = 0, height = 0;
    double ox = 0.0;   // map x of the left edge of column 0
    double oy = 0.0;   // map y of the top edge of row 0
    double px = 0.0;   // pixel size, metres

    void load(const std::string& path, int w, int ht) {
        width = w;
        height = ht;
        h.resize(static_cast<size_t>(w) * ht);
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open raster " + path);
        f.read(reinterpret_cast<char*>(h.data()),
               static_cast<std::streamsize>(h.size() * sizeof(float)));
        if (f.gcount() != static_cast<std::streamsize>(h.size() * sizeof(float)))
            throw std::runtime_error("short read on raster " + path);
    }

    // Bilinear sample at a map coordinate. False if outside, or if any of the
    // four contributing cells is not finite.
    inline bool sample(double x, double y, double& out) const {
        const double cf = (x - ox) / px - 0.5;
        const double rf = (oy - y) / px - 0.5;
        const double fi = std::floor(cf), fj = std::floor(rf);
        const int i = static_cast<int>(fi), j = static_cast<int>(fj);
        if (i < 0 || j < 0 || i + 1 >= width || j + 1 >= height) return false;
        const double a = cf - fi, b = rf - fj;
        const float* r0 = &h[static_cast<size_t>(j) * width + i];
        const float* r1 = &h[static_cast<size_t>(j + 1) * width + i];
        const double h00 = r0[0], h01 = r0[1], h10 = r1[0], h11 = r1[1];
        if (!(std::isfinite(h00) && std::isfinite(h01) &&
              std::isfinite(h10) && std::isfinite(h11))) return false;
        out = (1.0 - b) * ((1.0 - a) * h00 + a * h01) + b * ((1.0 - a) * h10 + a * h11);
        return true;
    }
};

// Moon-fixed unit vector of a map point. Inverse south polar stereographic on
// the sphere: colatitude from the south pole is 2 atan(r / 2R).
inline Vector3d map_to_unit(double x, double y, double R) {
    const double r = std::hypot(x, y);
    const double t = r / (2.0 * R);
    const double den = 1.0 + t * t;
    const double sin_theta = 2.0 * t / den;
    const double cos_theta = (1.0 - t * t) / den;
    const double sl = (r > 0.0) ? x / r : 0.0;   // sin(LON), LON = atan2(x, y)
    const double cl = (r > 0.0) ? y / r : 1.0;   // cos(LON)
    // Snyder south polar aspect: x = rho sin(LON), y = rho cos(LON), so the
    // cosine goes with the x component. Interchanging the two is a reflection
    // rather than a rotation, it passed every check internal to this project,
    // and an external landmark is what caught it. README, Results, "The check
    // that mattered most".
    return Vector3d(sin_theta * cl, sin_theta * sl, -cos_theta);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <params.json>\n", argv[0]);
        return 2;
    }
    const json_flat::Object p = json_flat::load(argv[1]);

    const double R = p.num("radius_m");
    const double observer_height = p.num("observer_height_m");
    const double reach = p.num("reach_m");
    const double switch_m = p.num("switch_m");
    const double near_step = p.num("near_step_m");
    const double far_step = p.num("far_step_m");
    const int n_az = p.integer("n_azimuth");

    const int nx = p.integer("region_nx"), ny = p.integer("region_ny");
    const double rox = p.num("region_ox"), roy = p.num("region_oy");
    const double rpx = p.num("region_px");

    Grid near_grid, far_grid;
    near_grid.ox = p.num("near_ox"); near_grid.oy = p.num("near_oy"); near_grid.px = p.num("near_px");
    near_grid.load(p.str("near_path"), p.integer("near_width"), p.integer("near_height"));
    far_grid.ox = p.num("far_ox"); far_grid.oy = p.num("far_oy"); far_grid.px = p.num("far_px");
    far_grid.load(p.str("far_path"), p.integer("far_width"), p.integer("far_height"));

    // Step schedule. Fixed for the whole run, so the central angle's cosine and
    // sine are precomputed once and shared by every ray.
    std::vector<double> s_arc;
    std::vector<uint8_t> use_far;
    {
        const double near_end = std::min(switch_m, reach);
        const int n_near = static_cast<int>(std::floor(near_end / near_step + 1e-9));
        for (int k = 1; k <= n_near; ++k) { s_arc.push_back(k * near_step); use_far.push_back(0); }
        if (reach > switch_m) {
            const int n_far = static_cast<int>(std::floor((reach - switch_m) / far_step + 1e-9));
            for (int k = 1; k <= n_far; ++k) {
                s_arc.push_back(switch_m + k * far_step);
                use_far.push_back(1);
            }
        }
    }
    const int n_steps = static_cast<int>(s_arc.size());
    if (n_steps == 0) throw std::runtime_error("empty step schedule");
    std::vector<double> cpsi(n_steps), spsi(n_steps), shalf2(n_steps);
    for (int k = 0; k < n_steps; ++k) {
        const double psi = s_arc[k] / R;
        cpsi[k] = std::cos(psi);
        spsi[k] = std::sin(psi);
        // (R+h)cos(psi) - R_obs cancels catastrophically at short range: at the
        // first 20 m step the two terms agree to ten digits. Rewritten with
        // cos(psi) - 1 = -2 sin^2(psi/2) it is exact, and the height difference
        // is then a difference of heights rather than of two radii.
        const double sh = std::sin(0.5 * psi);
        shalf2[k] = sh * sh;
    }

    const size_t n_cells = static_cast<size_t>(nx) * ny;
    std::vector<float> horizon(n_cells * n_az);
    std::vector<float> horizon_dist(n_cells * n_az);

    const double two_R = 2.0 * R;
    const double deg = 180.0 / M_PI;
    long long misses = 0;

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    // Tiles are the unit of parallel work as well as the unit of cache reuse,
    // so a tile size fixed at 32 starves the threads on any region smaller than
    // about 128 cells square: a 32 x 32 region is one tile and one busy thread.
    // Aim for a few tiles per thread, and never below 8 so that neighbouring
    // cells still share their far-field cache lines.
    int TILE = 32;
    {
        const double want = std::sqrt(static_cast<double>(nx) * ny / (4.0 * n_threads));
        TILE = std::max(8, std::min(32, static_cast<int>(want)));
    }
    const auto t0 = std::chrono::steady_clock::now();

    // Tiled so that the rays of neighbouring cells, traced consecutively at the
    // same azimuth, reuse the same far-field cache lines.
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic) reduction(+ : misses)
#endif
    for (int tr = 0; tr < ny; tr += TILE) {
        for (int tc = 0; tc < nx; tc += TILE) {
            const int r1 = std::min(tr + TILE, ny), c1 = std::min(tc + TILE, nx);
            const int nt = (r1 - tr) * (c1 - tc);

            std::vector<Vector3d> up(nt), east(nt), north(nt);
            std::vector<double> r_obs(nt);
            std::vector<size_t> base(nt);
            int m = 0;
            for (int r = tr; r < r1; ++r) {
                for (int c = tc; c < c1; ++c, ++m) {
                    const double x = rox + (c + 0.5) * rpx;
                    const double y = roy - (r + 0.5) * rpx;
                    const Vector3d u = map_to_unit(x, y, R);
                    const double rr = std::hypot(x, y);
                    // east = normalize(z_axis x up). With up = (sin t cos L,
                    // sin t sin L, -cos t) that is (-sin L, cos L, 0), and
                    // sin L = x/r, cos L = y/r, so east = (-x, y, 0)/|(x, y)|.
                    const Vector3d e = (rr > 0.0) ? Vector3d(-x / rr, y / rr, 0.0)
                                                  : Vector3d(1.0, 0.0, 0.0);
                    up[m] = u;
                    east[m] = e;
                    north[m] = u.cross(e);
                    double h0 = 0.0;
                    if (!near_grid.sample(x, y, h0)) throw std::runtime_error(
                        "region cell falls outside the near grid");
                    r_obs[m] = h0 + observer_height;   // height of the eye, not a radius
                    base[m] = (static_cast<size_t>(r) * nx + c) * n_az;
                }
            }

            for (int a = 0; a < n_az; ++a) {
                const double az = a * (2.0 * M_PI / n_az);
                const double ca = std::cos(az), sa = std::sin(az);
                for (int q = 0; q < nt; ++q) {
                    const Vector3d& u = up[q];
                    const Vector3d t = ca * north[q] + sa * east[q];
                    const double h_eye = r_obs[q];
                    // Track the maximum of A/B with B > 0 by cross-multiplying,
                    // which avoids a division per step. elev = atan2(A, B).
                    double bestA = -1.0e300, bestB = 1.0, bestS = 0.0;
                    for (int k = 0; k < n_steps; ++k) {
                        const double cp = cpsi[k], sp = spsi[k];
                        const double dx = cp * u.x() + sp * t.x();
                        const double dy = cp * u.y() + sp * t.y();
                        const double dz = cp * u.z() + sp * t.z();
                        const double inv = two_R / (1.0 - dz);
                        double h;
                        const Grid& g = use_far[k] ? far_grid : near_grid;
                        // map x comes from the 3D y component and vice versa,
                        // the inverse of map_to_unit above.
                        if (!g.sample(dy * inv, dx * inv, h)) { ++misses; continue; }
                        const double Rh = R + h;
                        const double A = (h - h_eye) - 2.0 * Rh * shalf2[k];
                        const double B = Rh * sp;
                        if (A * bestB > bestA * B) { bestA = A; bestB = B; bestS = s_arc[k]; }
                    }
                    horizon[base[q] + a] = static_cast<float>(std::atan2(bestA, bestB) * deg);
                    horizon_dist[base[q] + a] = static_cast<float>(bestS);
                }
            }
        }
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    {
        std::ofstream f(p.str("out_horizon"), std::ios::binary);
        f.write(reinterpret_cast<const char*>(horizon.data()),
                static_cast<std::streamsize>(horizon.size() * sizeof(float)));
    }
    {
        std::ofstream f(p.str("out_horizon_dist"), std::ios::binary);
        f.write(reinterpret_cast<const char*>(horizon_dist.data()),
                static_cast<std::streamsize>(horizon_dist.size() * sizeof(float)));
    }

    json_flat::Writer w;
    w.add("azimuth_convention",
          "from local north, positive toward local east, in each cell's own local "
          "tangent frame; ray k is exactly k degrees, k = 0..n_azimuth-1");
    w.add("array_layout", "float32, C order, (region_ny, region_nx, n_azimuth)");
    w.add("projection", "Snyder south polar: x = rho sin(LON), y = rho cos(LON)");
    w.add("horizon_units", "degrees");
    w.add("horizon_dist_units", "metres of arc along the ray");
    w.add("radius_m", R);
    w.add("observer_height_m", observer_height);
    w.add("reach_m", reach);
    w.add("switch_m", switch_m);
    w.add("near_step_m", near_step);
    w.add("far_step_m", far_step);
    w.add("n_azimuth", n_az);
    w.add("n_steps_per_ray", n_steps);
    w.add("region_nx", nx);
    w.add("region_ny", ny);
    w.add("region_ox", rox);
    w.add("region_oy", roy);
    w.add("region_px", rpx);
    w.add("near_path", p.str("near_path"));
    w.add("near_px", near_grid.px);
    w.add("far_path", p.str("far_path"));
    w.add("far_px", far_grid.px);
    w.add("dem_sampling", "bilinear, near and far");
    w.add("ray_path", "great circle, advanced incrementally in 3D");
    w.add("samples_total", static_cast<long long>(n_cells) * n_az * n_steps);
    w.add("samples_outside_grid", misses);
    w.add("threads", n_threads);
    w.add("seconds", secs);
    if (p.strings.count("source_commit")) w.add("source_commit", p.str("source_commit"));
    w.save(p.str("out_sidecar"));

    std::printf("cells %zu, azimuths %d, steps/ray %d, samples %.3e\n",
                n_cells, n_az, n_steps, static_cast<double>(n_cells) * n_az * n_steps);
    std::printf("threads %d, %.3f s, %.3e samples/s, outside-grid %lld\n",
                n_threads, secs,
                static_cast<double>(n_cells) * n_az * n_steps / std::max(secs, 1e-9),
                misses);
    return 0;
}
