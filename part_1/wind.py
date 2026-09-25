"""
Wind template

Students should compute generalized BODY-frame wind loads:
    tau_w6 = [Fx, Fy, Fz, Mx, My, Mz]

The simulator uses the 3-DOF subset [Fx, Fy, Mz] = tau_w6 indices [0, 1, 5]
and calls, once per step:

    wind.step(t, dt, eta, nu) -> (tau_w6, info)

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (gust spectra, time variation)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5])
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (RELATIVE wind: compute the loads from V_rw = V_wind - V_vessel,
            using the horizontal components nu[0], nu[1])

Outputs:
    tau_w6 : (6,) BODY loads
    info   : optional dict for logging, e.g.
             {"U": ambient speed, "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad)}
             Return {} (or None) if you do not need it.
             NOTE: "beta_ned" is always the direction the wind blows
             TOWARDS, even when the constructor semantics is "from" —
             convert before logging, do not log the raw constructor value.

Wind coefficient data
---------------------
The vessel wind coefficients C(alpha) = [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] are
provided in `data/wind_coeff.csv` (repository root), tabulated against the relative
wind angle alpha in degrees (0..360). Load them with:

    alpha_deg, C6 = load_wind_coefficients()

The wind loads are then computed as F_wind = U_rw^2 * C(alpha_rw), where U_rw
and alpha_rw are the relative wind speed and angle in the BODY frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from simulation.utils import Rz, wrap_angle_2pi

_WIND_COEFF_FILE = Path(__file__).resolve().parent.parent / "data" / "wind_coeff.csv"


def load_wind_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the vessel wind coefficient table.

    Returns
    -------
    alpha_deg : (M,) ndarray
        Relative wind angle grid [deg], from 0 to 360.
    C6 : (M, 6) ndarray
        Coefficients [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] at each angle.
    """
    table = np.loadtxt(_WIND_COEFF_FILE, delimiter=",", skiprows=1)
    return table[:, 0], table[:, 1:]


class Wind:
    """
    Wind load generator on the vessel hull.

    Parameters
    ----------
    mean_speed : float
        Mean wind speed [m/s].
    beta : float
        Wind direction [rad] in NED (0 = North, pi/2 = East).
    semantics : str, optional
        Direction convention:
        - "from" (default): direction the wind blows from.
        - "towards": direction the wind blows towards.
    sigma_slow : float, optional
        Standard deviation of the slowly varying speed component [m/s].
    tau_slow : float, optional
        Time constant of the 1st-order Gauss-Markov slow variation [s].
    seed : int | None, optional
        Random seed for reproducible slow variations.
    """

    def __init__(
        self,
        mean_speed: float = 0.0,
        beta: float = 0.0,
        *,
        semantics: str = "from",
        sigma_slow: float = 0.0,
        tau_slow: float = 120.0,
        seed: int | None = None,
    ) -> None:
        self.mean_speed = float(mean_speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)
        self.seed = seed

        # Load aerodynamic coefficient table
        self.alpha_grid_deg, self.C6_tab = load_wind_coefficients()

        # State for the 1st-order Gauss-Markov process (slowly varying component)
        self.u_slow = 0.0
        self.rng = np.random.default_rng(seed)

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Advance one step and compute generalized BODY-frame wind loads.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        dt : float
            Simulation time step [s].
        eta : np.ndarray
            6-DOF vessel state in NED [N, E, z, phi, theta, psi].
        nu : np.ndarray
            6-DOF vessel velocities in BODY [u, v, w, p, q, r].

        Returns
        -------
        tau_w6 : np.ndarray
            (6,) generalized BODY loads [Fx, Fy, Fz, Mx, My, Mz] (N, Nm).
        info : dict
            Diagnostic metrics {"U", "beta_ned", "alpha_body"}.
        """
        # 1. Update the slowly-varying wind speed using a Gauss-Markov process
        if self.sigma_slow > 0.0 and self.tau_slow > 0.0:
            # Discrete-time Gauss-Markov update
            variance_input = (self.sigma_slow ** 2) * (1.0 - np.exp(-2.0 * dt / self.tau_slow))
            w_k = self.rng.normal(0.0, np.sqrt(max(variance_input, 1e-12)))
            self.u_slow = np.exp(-dt / self.tau_slow) * self.u_slow + w_k
        else:
            self.u_slow = 0.0

        u_ambient = max(0.0, self.mean_speed + self.u_slow)

        # 2. Determine wind direction in NED (towards convention)
        if self.semantics == "from":
            beta_towards = self.beta + np.pi
        else:
            beta_towards = self.beta

        # 3. Ambient wind velocity in NED
        v_w_ned = np.array([
            u_ambient * np.cos(beta_towards),
            u_ambient * np.sin(beta_towards),
            0.0,
        ], dtype=float)

        # 4. Project ambient wind into BODY frame: V_w_b = Rz(psi)^T @ V_w_ned
        psi = float(eta[5])
        R = Rz(psi)
        v_w_body = R.T @ v_w_ned

        # 5. Compute relative wind velocity in horizontal BODY plane:
        # V_rw = V_wind_body - V_vessel_body
        v_rw_x = v_w_body[0] - float(nu[0])
        v_rw_y = v_w_body[1] - float(nu[1])

        # Relative wind speed and angle
        u_rw = float(np.hypot(v_rw_x, v_rw_y))
        alpha_rw = float(np.arctan2(v_rw_y, v_rw_x))  # in (-pi, pi]

        # Convert relative wind angle to degrees in [0, 360) for table lookup
        alpha_rw_2pi = wrap_angle_2pi(alpha_rw)
        alpha_deg = float(np.rad2deg(alpha_rw_2pi))

        # 6. Interpolate aerodynamic coefficient vector C(alpha_rw)
        c_interp = np.zeros(6, dtype=float)
        for i in range(6):
            c_interp[i] = np.interp(
                alpha_deg,
                self.alpha_grid_deg,
                self.C6_tab[:, i],
                period=360.0,
            )

        # 7. Compute BODY forces and moments: tau_wind = U_rw^2 * C(alpha_rw)
        tau_w6 = (u_rw ** 2) * c_interp

        info = {
            "U": u_ambient,
            "beta_ned": (beta_towards + np.pi) % (2.0 * np.pi) - np.pi,  # TOWARDS in (-pi, pi]
            "alpha_body": alpha_rw,
        }

        return tau_w6, info