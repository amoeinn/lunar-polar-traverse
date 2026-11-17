// Replay checker: does a plan actually work when the rover is integrated through it?
//
// Check 2 of the independent check. It shares the DEM, the projection constants,
// the sphere radius and the rover parameters with the planner, and nothing else:
// not the horizon arrays, not the interval tables, not src/cost.py, not
// src/frames.py. See PLAN.md, "Replay checker, built out of phase order", for
// what it necessarily shares and why that is bounded.
//
// Its own derivations: position integrated continuously along the path; pitch and
// roll from the normal of a least-squares plane fit over a disc at that position;
// sun visibility by marching the great circle at the sun's exact azimuth and time.
//
// Usage: replay <params.json>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "json_flat.hpp"

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

struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double a, double b, double c) : x(a), y(b), z(c) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double norm() const { return std::sqrt(dot(*this)); }
    Vec3 unit() const { const double n = norm(); return n > 0 ? *this * (1.0 / n) : *this; }
};

// A north-up float32 raster, bilinearly sampled. Written here rather than shared:
// bilinear is the DEM convention stated in PLAN.md, not a borrowed routine.
struct Raster {
    std::vector<float> v;
    int w = 0, h = 0;
    double ox = 0, oy = 0, px = 0;

    void load(const std::string& path, int width, int height) {
        w = width; h = height;
        v = read_raw<float>(path, static_cast<size_t>(w) * h);
    }
    bool at(double x, double y, double& out) const {
        const double c = (x - ox) / px - 0.5, r = (oy - y) / px - 0.5;
        const double fc = std::floor(c), fr = std::floor(r);
        const int i = static_cast<int>(fc), j = static_cast<int>(fr);
        if (i < 0 || j < 0 || i + 1 >= w || j + 1 >= h) return false;
        const double a = c - fc, b = r - fr;
        const double v00 = v[static_cast<size_t>(j) * w + i];
        const double v01 = v[static_cast<size_t>(j) * w + i + 1];
        const double v10 = v[static_cast<size_t>(j + 1) * w + i];
        const double v11 = v[static_cast<size_t>(j + 1) * w + i + 1];
        if (!std::isfinite(v00) || !std::isfinite(v01) || !std::isfinite(v10)
            || !std::isfinite(v11)) return false;
        out = (1 - a) * (1 - b) * v00 + a * (1 - b) * v01
            + (1 - a) * b * v10 + a * b * v11;
        return true;
    }
};

// Snyder south polar aspect, written from the statement in PLAN.md: the map
// point (x, y) has colatitude 2 atan(r / 2R) and longitude atan2(x, y), so the
// Moon-fixed unit vector is (sin t cos L, sin t sin L, -cos t).
inline Vec3 map_to_unit(double x, double y, double R) {
    const double r = std::hypot(x, y);
    const double q = r / (2.0 * R);
    const double den = 1.0 + q * q;
    const double st = 2.0 * q / den, ct = (1.0 - q * q) / den;
    const double sl = (r > 0) ? x / r : 0.0, cl = (r > 0) ? y / r : 1.0;
    return {st * cl, st * sl, -ct};
}

inline void local_frame(double x, double y, double R, Vec3& up, Vec3& east, Vec3& north) {
    up = map_to_unit(x, y, R);
    const double r = std::hypot(x, y);
    east = (r > 0) ? Vec3(-x / r, y / r, 0.0) : Vec3(1.0, 0.0, 0.0);
    north = up.cross(east);
}

struct Rover {
    double v_flat, mu, cap_ratio, roll_ref, roll_exponent, pitch_max, roll_max;
    double speed(double pitch_deg, double roll_deg) const {
        const double th = pitch_deg / kDeg;
        const double drive = std::sin(th) + mu * std::cos(th);
        double v = drive > 0.0 ? v_flat * mu / drive
                               : std::numeric_limits<double>::infinity();
        v = std::min(v, cap_ratio * v_flat);
        return v / (1.0 + std::pow(std::fabs(roll_deg) / roll_ref, roll_exponent));
    }
};

}  // namespace

// The sun table arrives in METRES. az_el subtracts c.origin, built from
// radius_m on the 1737400 m sphere, so a kilometre table mixes units and drops
// the sun about 0.67 deg in elevation while leaving the azimuth exact, which is
// why it survived every azimuth check in this project's record. Checked on the
// magnitude because units are not carried in the file: kilometres come in three
// orders low, a double conversion three orders high.
inline void check_sun_metres(const double* sun, int n_sun, const char* where) {
    constexpr double kMinM = 1.3e11, kMaxM = 1.7e11;
    for (int k = 0; k < n_sun; ++k) {
        const double x = sun[3 * k], y = sun[3 * k + 1], z = sun[3 * k + 2];
        const double d = std::sqrt(x * x + y * y + z * z);
        if (!(d >= kMinM && d <= kMaxM)) {
            std::fprintf(stderr,
                "%s: sun sample %d has magnitude %.6e m, outside [%.3e, %.3e]. "
                "SPICE spkpos returns kilometres; this project works in metres.\n",
                where, k, d, kMinM, kMaxM);
            std::exit(2);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <params.json>\n", argv[0]);
        return 2;
    }
    const json_flat::Object p = json_flat::load(argv[1]);
    const double R = p.num("radius_m");
    const double rox = p.num("region_ox"), roy = p.num("region_oy");
    const double rpx = p.num("region_px");
    const int rnx = p.integer("region_nx");
    Raster near_g, far_g;
    near_g.ox = p.num("near_ox"); near_g.oy = p.num("near_oy"); near_g.px = p.num("near_px");
    near_g.load(p.str("near_path"), p.integer("near_width"), p.integer("near_height"));
    far_g.ox = p.num("far_ox"); far_g.oy = p.num("far_oy"); far_g.px = p.num("far_px");
    far_g.load(p.str("far_path"), p.integer("far_width"), p.integer("far_height"));

    const Rover rover{p.num("v_flat"), p.num("mu"), p.num("cap_ratio"),
                      p.num("roll_ref"), p.num("roll_exponent"),
                      p.num("pitch_max"), p.num("roll_max")};
    const double reach = p.num("reach_m"), switch_m = p.num("switch_m");
    const double near_step = p.num("near_step_m"), far_step = p.num("far_step_m");
    const double step_m = p.num("integration_step_m");
    const double vis_cadence_h = p.num("visibility_cadence_s") / 3600.0;
    const double disc_r = p.num("disc_radius_m");
    const double shadow_tol = p.num("shadow_tol_h");
    const double limit_tol = p.num("limit_tolerance_deg", 1.0e9);

    const int n_sun = p.integer("n_sun");
    const double sun_dt = p.num("sun_dt_hours");
    const auto sun = read_raw<double>(p.str("sun_path"), 3u * n_sun);
    check_sun_metres(sun.data(), n_sun, "replay");
    auto sun_at = [&](double t) {
        double u = t / sun_dt;
        int k = static_cast<int>(std::floor(u));
        k = std::max(0, std::min(k, n_sun - 2));
        const double f = u - k;
        return Vec3((1 - f) * sun[3 * k] + f * sun[3 * k + 3],
                    (1 - f) * sun[3 * k + 1] + f * sun[3 * k + 4],
                    (1 - f) * sun[3 * k + 2] + f * sun[3 * k + 5]);
    };

    const int n_plans = p.integer("n_plans");
    const auto offsets = read_raw<int64_t>(p.str("plan_offsets_path"), n_plans + 1);
    const auto cells = read_raw<int32_t>(p.str("plan_cells_path"),
                                         static_cast<size_t>(offsets[n_plans]));
    const auto starts = read_raw<double>(p.str("plan_start_path"), n_plans);
    // The planner's waiting decisions, one per path node. The checker integrates
    // its own travel times and inserts these holds, so its arrival differs from
    // the planner's only by the integration, which is what the timing check is.
    const bool have_waits = p.strings.count("plan_wait_path") != 0;
    std::vector<double> waits;
    if (have_waits)
        waits = read_raw<double>(p.str("plan_wait_path"),
                                 static_cast<size_t>(offsets[n_plans]));

    auto cell_xy = [&](int c, double& x, double& y) {
        x = rox + (c % rnx + 0.5) * rpx;
        y = roy - (c / rnx + 0.5) * rpx;
    };

    // Pitch and roll from a disc fit: sample the DEM on rings about the
    // position, express each sample in the local tangent frame, and least
    // squares fit u = a e + b n. The gradient (a, b) is in the local frame,
    // never in map axes, which is the planner's route.
    auto disc_fit = [&](double x, double y, double h0, const Vec3& up, const Vec3& east,
                        const Vec3& north, double& a, double& b) {
        const Vec3 origin = up * (R + h0);
        double see = 0, snn = 0, sen = 0, seu = 0, snu = 0;
        int used = 0;
        for (int ring = 1; ring <= 3; ++ring) {
            const double rad = disc_r * ring / 3.0;
            for (int k = 0; k < 12; ++k) {
                const double ang = 2.0 * M_PI * k / 12.0;
                const double dx = rad * std::sin(ang), dy = rad * std::cos(ang);
                double hs;
                if (!near_g.at(x + dx, y + dy, hs)) continue;
                const Vec3 ps = map_to_unit(x + dx, y + dy, R) * (R + hs) - origin;
                const double e = ps.dot(east), nn = ps.dot(north), u = ps.dot(up);
                see += e * e; snn += nn * nn; sen += e * nn;
                seu += e * u; snu += nn * u;
                ++used;
            }
        }
        const double det = see * snn - sen * sen;
        if (used < 6 || std::fabs(det) < 1e-9) { a = b = 0.0; return false; }
        a = (seu * snn - snu * sen) / det;
        b = (snu * see - seu * sen) / det;
        return true;
    };

    // Sun visibility by direct ray march at the sun's exact azimuth.
    auto lit_at = [&](double x, double y, double h0, double t) {
        Vec3 up, east, north;
        local_frame(x, y, R, up, east, north);
        const Vec3 eye = up * (R + h0);
        const Vec3 d = (sun_at(t) - eye).unit();
        const double se = d.dot(east), sn = d.dot(north);
        const double sun_el = std::asin(std::max(-1.0, std::min(1.0, d.dot(up)))) * kDeg;
        const Vec3 tang = (north * sn + east * se).unit();
        for (double s = near_step; s <= reach; s += (s < switch_m ? near_step : far_step)) {
            const double psi = s / R;
            const Vec3 dir = up * std::cos(psi) + tang * std::sin(psi);
            const double q = 2.0 * R / (1.0 - dir.z);
            double hs;
            const Raster& g = (s < switch_m) ? near_g : far_g;
            if (!g.at(dir.y * q, dir.x * q, hs)) continue;
            const double Rh = R + hs;
            const double sh = std::sin(0.5 * psi);
            const double A = (hs - h0) - 2.0 * Rh * sh * sh;
            const double B = Rh * std::sin(psi);
            if (std::atan2(A, B) * kDeg > sun_el) return false;
        }
        return true;
    };

    std::vector<double> out_arrival(n_plans), out_shadow(n_plans);
    std::vector<uint8_t> out_verdict(n_plans);
    std::vector<int32_t> out_illegal(n_plans);
    std::vector<float> step_pitch, step_roll, step_pitch_mean, step_roll_mean;
    std::vector<int64_t> step_off(n_plans + 1, 0);
    // Optional trajectory dump, written only when the caller asks for it. The
    // executive of phase 6 consumes this as its world: it is the checker's own
    // integration of where the rover actually is and whether it is lit, which
    // is the one execution-time source in this project that does not share the
    // planner's time model. Kinds: 0 departure, 1 cell arrival, 2 lit/dark
    // transition seen during an edge or a hold, 3 end of a planned hold.
    const bool want_traj = p.strings.count("out_traj_t") > 0;
    std::vector<double> traj_t, traj_shadow;
    std::vector<int32_t> traj_index;
    std::vector<uint8_t> traj_kind, traj_lit;
    std::vector<int64_t> traj_off(n_plans + 1, 0);
    const auto emit = [&](double tt, double sh, int32_t idx, uint8_t kind,
                          bool lit) {
        if (!want_traj) return;
        traj_t.push_back(tt);
        traj_shadow.push_back(sh);
        traj_index.push_back(idx);
        traj_kind.push_back(kind);
        traj_lit.push_back(lit ? 1 : 0);
    };
    const auto wall0 = std::chrono::steady_clock::now();

    for (int q = 0; q < n_plans; ++q) {
        const int64_t lo = offsets[q], hi = offsets[q + 1];
        double t = starts[q], shadow = 0.0, shadow_max = 0.0;
        int illegal = 0;
        double next_vis = t;
        bool lit_now = true;
        // Transition emission needs a previous state to compare against, and
        // lit_now starts at a placeholder that the first visibility check
        // always overwrites. This flag says when a real previous state exists,
        // so a transition on the very first edge is emitted like any other
        // while the placeholder never produces a spurious one.
        bool traj_have_lit = false;
        if (want_traj && hi > lo) {
            double sx, sy, sh0;
            cell_xy(cells[lo], sx, sy);
            const bool oks = near_g.at(sx, sy, sh0);
            emit(t, 0.0, 0, 0, oks ? lit_at(sx, sy, sh0, t) : true);
        }
        for (int64_t k = lo; k + 1 < hi; ++k) {
            double x0, y0, x1, y1;
            cell_xy(cells[k], x0, y0);
            cell_xy(cells[k + 1], x1, y1);
            const double seg = std::hypot(x1 - x0, y1 - y0);
            const int nsub = std::max(1, static_cast<int>(std::ceil(seg / step_m)));
            double worst_pitch = 0.0, worst_roll = 0.0;
            double sum_pitch = 0.0, sum_roll = 0.0;
            int counted = 0;
            for (int i = 0; i < nsub; ++i) {
                const double f = (i + 0.5) / nsub;
                const double x = x0 + f * (x1 - x0), y = y0 + f * (y1 - y0);
                double h0;
                if (!near_g.at(x, y, h0)) { ++illegal; continue; }
                Vec3 up, east, north;
                local_frame(x, y, R, up, east, north);
                double a, b;
                disc_fit(x, y, h0, up, east, north, a, b);
                // heading in the local frame, from the map-space direction
                const Vec3 hdir = (map_to_unit(x1, y1, R) * (R + h0)
                                   - map_to_unit(x0, y0, R) * (R + h0));
                const double he = hdir.dot(east), hn = hdir.dot(north);
                const double hl = std::hypot(he, hn);
                const double ue = he / hl, un = hn / hl;
                const double pitch = std::atan(a * ue + b * un) * kDeg;
                const double roll = std::atan(-a * un + b * ue) * kDeg;
                if (std::fabs(pitch) > worst_pitch) worst_pitch = std::fabs(pitch);
                if (std::fabs(roll) > worst_roll) worst_roll = std::fabs(roll);
                sum_pitch += pitch; sum_roll += roll; ++counted;
                if (std::fabs(pitch) > rover.pitch_max + limit_tol
                    || std::fabs(roll) > rover.roll_max + limit_tol) ++illegal;
                const double v = rover.speed(pitch, roll);
                const double dt = (seg / nsub) / v;
                if (t >= next_vis) {
                    const bool was = lit_now;
                    lit_now = lit_at(x, y, h0, t);
                    next_vis = t + vis_cadence_h;
                    if (want_traj && traj_have_lit && lit_now != was)
                        emit(t, shadow, static_cast<int32_t>(k - lo), 2, lit_now);
                    traj_have_lit = true;
                }
                if (lit_now) shadow = 0.0;
                else { shadow += dt; shadow_max = std::max(shadow_max, shadow); }
                t += dt;
            }
            step_pitch.push_back(static_cast<float>(worst_pitch));
            step_roll.push_back(static_cast<float>(worst_roll));
            step_pitch_mean.push_back(static_cast<float>(counted ? sum_pitch / counted : 0.0));
            step_roll_mean.push_back(static_cast<float>(counted ? sum_roll / counted : 0.0));
            // hold here for as long as the planner's schedule says, accruing
            // shadow by the checker's own visibility rather than the planner's
            emit(t, shadow, static_cast<int32_t>(k + 1 - lo), 1, lit_now);
            if (have_waits && waits[k + 1] > 0.0) {
                const double hold_end = t + waits[k + 1];
                double hx, hy;
                cell_xy(cells[k + 1], hx, hy);
                double hh;
                const bool okh = near_g.at(hx, hy, hh);
                while (t < hold_end) {
                    const double dt = std::min(vis_cadence_h, hold_end - t);
                    const bool was = lit_now;
                    if (okh) lit_now = lit_at(hx, hy, hh, t);
                    if (want_traj && traj_have_lit && lit_now != was)
                        emit(t, shadow, static_cast<int32_t>(k + 1 - lo), 2,
                             lit_now);
                    if (lit_now) shadow = 0.0;
                    else { shadow += dt; shadow_max = std::max(shadow_max, shadow); }
                    t += dt;
                }
                next_vis = t;
                emit(t, shadow, static_cast<int32_t>(k + 1 - lo), 3, lit_now);
            }
        }
        step_off[q + 1] = static_cast<int64_t>(step_pitch.size());
        traj_off[q + 1] = static_cast<int64_t>(traj_t.size());
        out_arrival[q] = t;
        out_shadow[q] = shadow_max;
        out_illegal[q] = illegal;
        out_verdict[q] = (illegal == 0 && shadow_max <= shadow_tol) ? 1 : 0;
    }

    write_raw(p.str("out_arrival"), out_arrival);
    write_raw(p.str("out_shadow"), out_shadow);
    write_raw(p.str("out_verdict"), out_verdict);
    write_raw(p.str("out_illegal"), out_illegal);
    if (want_traj) {
        write_raw(p.str("out_traj_t"), traj_t);
        write_raw(p.str("out_traj_shadow"), traj_shadow);
        write_raw(p.str("out_traj_index"), traj_index);
        write_raw(p.str("out_traj_kind"), traj_kind);
        write_raw(p.str("out_traj_lit"), traj_lit);
        write_raw(p.str("out_traj_offsets"), traj_off);
    }
    write_raw(p.str("out_step_pitch"), step_pitch);
    write_raw(p.str("out_step_roll"), step_roll);
    write_raw(p.str("out_step_pitch_mean"), step_pitch_mean);
    write_raw(p.str("out_step_roll_mean"), step_roll_mean);
    write_raw(p.str("out_step_offsets"), step_off);
    std::printf("replay: %d plans, %zu steps, %.2f s\n", n_plans, step_pitch.size(),
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - wall0).count());
    return 0;
}
