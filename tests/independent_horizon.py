"""Independent horizon ray march, for PLAN.md check 1c.

Written from the specification in PLAN.md Method, "Horizon model". It shares
no code with cpp/src/horizon.cpp and imports nothing from src/. Where the same
quantity can be formed more than one way, this implementation deliberately
takes the other route, so that agreement between the two is evidence rather
than a transcription:

  local frame     closed form in colatitude and longitude, rather than
                  numerical cross products
  3D -> map       via arccos, arctan2 and tan(theta/2), rather than the
                  algebraic identity x = 2R dx / (1 - dz)
  elevation       arcsin of the along-up component over a chord length from
                  the law of cosines, rather than atan2 of the two components
  bilinear        scipy.ndimage.map_coordinates, order 1

The cancellation-free rearrangement is derived here rather than copied. For an
observer of radius rho_o and a target of radius rho_t separated by central
angle psi, the component of the separation along the observer's up is
rho_t cos(psi) - rho_o, which at psi of order 1e-5 subtracts two numbers near
1.7e6 that differ by 1e-4. Using cos(psi) - 1 = -2 sin^2(psi/2) and splitting
off the radius difference gives

    A = (h_t - h_o - observer_height) - 2 rho_t sin^2(psi/2)

in which the first term is a difference of heights and the second is formed
from sin(psi/2) directly. The chord follows the same way, from
rho_t^2 + rho_o^2 - 2 rho_t rho_o cos(psi) = (rho_t - rho_o)^2
+ 4 rho_t rho_o sin^2(psi/2).
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

AZIMUTH_CHUNK = 180          # rays per pass, to bound peak memory


def load_params(work_dir):
    return json.loads((Path(work_dir) / "params.json").read_text())


def load_grid(params, which):
    """Load a raw float32 window, promoted to float64.

    The promotion is not cosmetic. scipy.ndimage.map_coordinates returns the
    input dtype, so interpolating a float32 grid rounds every sampled height
    to float32, discarding up to half an ulp, about 6e-5 m at 1365 m. At the
    20 m first step that is worth 1.7e-4 deg of elevation, which is two orders
    above a float32 ulp of the answer and would have been misread as a
    disagreement with the model. The C++ reads float32 and promotes to double
    before interpolating; this matches it.
    """
    arr = np.fromfile(params[f"{which}_path"], dtype=np.float32)
    arr = arr.reshape(params[f"{which}_height"], params[f"{which}_width"]).astype(np.float64)
    return arr, params[f"{which}_ox"], params[f"{which}_oy"], params[f"{which}_px"]


def unit_from_map(x, y, radius):
    """Moon-fixed unit vector of a map point: colatitude 2 atan(r / 2R) from
    the south pole, longitude atan2(x, y)."""
    r = np.hypot(x, y)
    theta = 2.0 * np.arctan2(r, 2.0 * radius)
    lam = np.arctan2(x, y)
    st, ct = np.sin(theta), np.cos(theta)
    # Snyder south polar aspect: the cosine of the longitude goes with x.
    return np.array([st * np.cos(lam), st * np.sin(lam), -ct]), theta, lam


def frame_from_map(x, y, radius):
    """(up, east, north) in closed form.

    east = normalize(z_axis x up). With up = (sin t cos l, sin t sin l, -cos t)
    the cross product is (-sin t sin l, sin t cos l, 0), whose norm is sin t, so
    east = (-sin l, cos l, 0). north = up x east works out to
    (cos t cos l, cos t sin l, sin t). These are the textbook local frame.
    """
    up, theta, lam = unit_from_map(x, y, radius)
    sl, cl = np.sin(lam), np.cos(lam)
    east = np.array([-sl, cl, 0.0])
    north = np.array([np.cos(theta) * cl, np.cos(theta) * sl, np.sin(theta)])
    # internal consistency of the closed forms, not a second implementation
    assert abs(up @ east) < 1e-12 and abs(up @ north) < 1e-12 and abs(east @ north) < 1e-12
    assert np.allclose(np.cross(up, east), north, atol=1e-12)
    return up, east, north


def map_from_unit(d, radius):
    """Map coordinates of a Moon-fixed unit vector, via colatitude and
    longitude rather than the algebraic shortcut.

    Colatitude comes from atan2 of the horizontal magnitude against -dz, not
    from arccos(-dz). arccos has derivative 1/sin(theta), which is 8.7e4 at
    the 20 m step, so it loses about 7e-13 rad there, worth a micron of map
    position and 2e-5 deg of elevation at that range. That is a weakness of
    this reference, not of the model under test, and atan2 removes it while
    remaining a different route from x = 2R dx / (1 - dz).
    """
    theta = np.arctan2(np.hypot(d[0], d[1]), -d[2])
    lam = np.arctan2(d[1], d[0])          # LON from the 3D y and x components
    r = 2.0 * radius * np.tan(0.5 * theta)
    return r * np.sin(lam), r * np.cos(lam)


def step_schedule(params):
    reach = params["reach_m"]
    switch = params["switch_m"]
    near_step = params["near_step_m"]
    far_step = params["far_step_m"]
    near_end = min(switch, reach)
    n_near = int(np.floor(near_end / near_step + 1e-9))
    s = [np.arange(1, n_near + 1) * near_step]
    is_far = [np.zeros(n_near, dtype=bool)]
    if reach > switch:
        n_far = int(np.floor((reach - switch) / far_step + 1e-9))
        s.append(switch + np.arange(1, n_far + 1) * far_step)
        is_far.append(np.ones(n_far, dtype=bool))
    return np.concatenate(s), np.concatenate(is_far)


class Reference:
    def __init__(self, work_dir):
        self.params = load_params(work_dir)
        self.radius = self.params["radius_m"]
        self.observer_height = self.params["observer_height_m"]
        self.near = load_grid(self.params, "near")
        self.far = load_grid(self.params, "far")
        self.s_arc, self.is_far = step_schedule(self.params)
        self.psi = self.s_arc / self.radius
        self.sin_half2 = np.sin(0.5 * self.psi) ** 2
        self.cos_psi = np.cos(self.psi)
        self.sin_psi = np.sin(self.psi)

    def cell_xy(self, row, col):
        p = self.params
        return (p["region_ox"] + (col + 0.5) * p["region_px"],
                p["region_oy"] - (row + 0.5) * p["region_px"])

    def _sample(self, grid_tuple, x, y):
        grid, ox, oy, px = grid_tuple
        col = (x - ox) / px - 0.5
        row = (oy - y) / px - 0.5
        return map_coordinates(grid, np.stack([row.ravel(), col.ravel()]),
                               order=1, mode="constant", cval=np.nan,
                               output=np.float64).reshape(x.shape)

    def march(self, row, col, azimuths_deg):
        """Horizon angle and the arc distance that sets it, per azimuth."""
        x0, y0 = self.cell_xy(row, col)
        up, east, north = frame_from_map(x0, y0, self.radius)
        h_obs = float(self._sample(self.near, np.array([[x0]]), np.array([[y0]]))[0, 0])
        rho_o = self.radius + h_obs + self.observer_height

        az = np.asarray(azimuths_deg, dtype=float)
        out_h = np.empty(az.size)
        out_d = np.empty(az.size)
        for lo in range(0, az.size, AZIMUTH_CHUNK):
            chunk = az[lo:lo + AZIMUTH_CHUNK]
            a = np.radians(chunk)[:, None]
            t = np.cos(a) * north[None, :] + np.sin(a) * east[None, :]   # (m, 3)
            cp = self.cos_psi[None, :]
            sp = self.sin_psi[None, :]
            d = (cp[:, :, None] * up[None, None, :] + sp[:, :, None] * t[:, None, :])
            xs, ys = map_from_unit([d[..., 0], d[..., 1], d[..., 2]], self.radius)

            h = np.empty(xs.shape)
            nf = ~self.is_far
            h[:, nf] = self._sample(self.near, xs[:, nf], ys[:, nf])
            h[:, self.is_far] = self._sample(self.far, xs[:, self.is_far], ys[:, self.is_far])
            assert np.isfinite(h).all(), "reference march sampled outside a grid"

            rho_t = self.radius + h
            A = (h - h_obs - self.observer_height) - 2.0 * rho_t * self.sin_half2[None, :]
            chord = np.sqrt((rho_t - rho_o) ** 2
                            + 4.0 * rho_t * rho_o * self.sin_half2[None, :])
            elev = np.degrees(np.arcsin(A / chord))

            best = np.argmax(elev, axis=1)
            out_h[lo:lo + chunk.size] = elev[np.arange(chunk.size), best]
            out_d[lo:lo + chunk.size] = self.s_arc[best]
        return out_h, out_d


def select_cells(horizon, dist, switch_m, seed=20260921):
    """The sampled cells, by the rule registered in PLAN.md check 1c."""
    n = horizon.shape[0]
    far_frac = (dist > switch_m).mean(axis=2)
    near_frac = 1.0 - far_frac
    peak = horizon.max(axis=2) - horizon.min(axis=2)

    def top4(field):
        flat = np.argsort(field, axis=None, kind="stable")[::-1][:4]
        return [tuple(int(v) for v in np.unravel_index(i, field.shape)) for i in flat]

    picks, why = [], {}

    def add(cells, label):
        for c in cells:
            if c not in why:
                why[c] = label
                picks.append(c)

    add(top4(near_frac), "most near-set")
    add(top4(far_frac), "most far-set")
    add([(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)], "region corner")
    add(top4(peak), "largest peak-to-peak")
    add([(417, 367), (418, 367)], "largest adjacent jump")
    rng = np.random.default_rng(seed)
    add([tuple(int(v) for v in rng.integers(0, n, size=2)) for _ in range(2)], "random seed 20260921")
    return picks, why, near_frac, far_frac, peak
