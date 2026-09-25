"""
Controller template

Students should implement a controller that maps the vessel state and the
full reference to a body-frame wrench. The simulator calls, once per step:

    controller.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the remaining components are
zero on input and ignored on output.

Inputs (full loop state and full reference):
    t       : current simulation time [s]
    dt      : time step [s]
    eta     : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta[0], E = eta[1], psi = eta[5])
    nu      : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu[0], v = nu[1], r = nu[5])
    eta_ref : (6,) NED reference state
              (use N_d = eta_ref[0], E_d = eta_ref[1], psi_d = eta_ref[5])
    nu_ref  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref[0], Edot_d = nu_ref[1], psidot_d = nu_ref[5])
    acc_ref : (6,) NED-frame reference accelerations, same layout as nu_ref
              (use for model-based / inertia feedforward)

Output:
    tau_d   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d[0], Fy = tau_d[1], Mz = tau_d[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model (ideal in Part 1)
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py``, ``pytest``,
``notebooks/part_1_demo.ipynb``) construct your controller as
``DPController()`` with NO arguments, so your final tuned gains must be the
constructor defaults. Tuning only inside ``run_case_part1.py`` will pass your
own runs but fail the checks.
"""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from part_1.config import PIDGains
from simulation.utils import wrap_angle_pi


class DPController:
    """
    3-DOF Dynamic Positioning PID Controller.

    Calculates the desired 3-DOF generalized BODY wrench [Fx, Fy, Mz] based on:
      - Position error: e_pos_body = Rz(psi)^T * (eta_ref[:2] - eta[:2])
      - Heading error:  e_psi = wrap_angle_pi(eta_ref[5] - eta[5])
      - Velocity error: e_nu = nu_ref_body - nu (with damping)
      - Integrated tracking errors with clamping anti-windup.
    """

    def __init__(self, gains: Optional[PIDGains] = None, *args, **kwargs) -> None:
        """
        Initialize the controller with gains from config or custom instance.
        """
        self.gains = gains if gains is not None else PIDGains()

        self.Kp = np.asarray(self.gains.Kp, dtype=float)
        self.Kd = np.asarray(self.gains.Kd, dtype=float)
        self.Ki = np.asarray(self.gains.Ki, dtype=float)
        self.int_limit = np.asarray(self.gains.int_limit, dtype=float)

        # Integrator states in BODY frame: [int_ex_b, int_ey_b, int_epsi]
        self.int_err = np.zeros(3, dtype=float)

        # Diagnostic logging hooks for simulation engine / plotters
        self.last_pid_body: Dict[str, np.ndarray] = {
            "P": np.zeros(6, dtype=float),
            "I": np.zeros(6, dtype=float),
            "D": np.zeros(6, dtype=float),
        }

    @property
    def int_ned(self) -> np.ndarray:
        """Return the horizontal integrated error for logger compatibility."""
        return self.int_err[:2]

    @property
    def int_psi(self) -> float:
        """Return the heading integrated error for logger compatibility."""
        return float(self.int_err[2])

    def reset(self, *args, **kwargs) -> None:
        """Reset internal integrator states and logging buffers."""
        self.int_err = np.zeros(3, dtype=float)
        self.last_pid_body = {
            "P": np.zeros(6, dtype=float),
            "I": np.zeros(6, dtype=float),
            "D": np.zeros(6, dtype=float),
        }

    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        nu_ref: Optional[np.ndarray] = None,
        acc_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the desired generalized BODY wrench tau_d.
        """
        psi = float(eta[5])

        # 1. Position error in NED, rotated to BODY frame using Rz(psi)^T
        pos_err_ned = np.asarray(eta_ref[:2], dtype=float) - np.asarray(eta[:2], dtype=float)
        c, s = np.cos(psi), np.sin(psi)
        pos_err_body = np.array([
            c * pos_err_ned[0] + s * pos_err_ned[1],
            -s * pos_err_ned[0] + c * pos_err_ned[1],
        ], dtype=float)

        # 2. Shortest-path heading error wrapping to (-pi, pi]
        e_psi = wrap_angle_pi(float(eta_ref[5]) - psi)

        # 3-DOF error vector: [e_x_b, e_y_b, e_psi]
        e = np.array([pos_err_body[0], pos_err_body[1], e_psi], dtype=float)

        # 3. Integrator update with anti-windup clamping
        self.int_err += e * dt
        self.int_err = np.clip(self.int_err, -self.int_limit, self.int_limit)

        # 4. Velocity error (damping + reference feedforward if present)
        if nu_ref is not None and not np.all(np.isnan(nu_ref)):
            nu_ref_ned_xy = np.asarray(nu_ref[:2], dtype=float)
            u_d = c * nu_ref_ned_xy[0] + s * nu_ref_ned_xy[1]
            v_d = -s * nu_ref_ned_xy[0] + c * nu_ref_ned_xy[1]
            r_d = float(nu_ref[5])
            nu_d = np.array([u_d, v_d, r_d], dtype=float)
            e_dot = nu_d - np.asarray(nu[[0, 1, 5]], dtype=float)
        else:
            e_dot = -np.asarray(nu[[0, 1, 5]], dtype=float)

        # 5. PID component calculations
        tau_P_3 = self.Kp * e
        tau_I_3 = self.Ki * self.int_err
        tau_D_3 = self.Kd * e_dot

        tau_3 = tau_P_3 + tau_I_3 + tau_D_3

        # 6. Populate 6-DOF BODY wrench
        tau_d = np.zeros(6, dtype=float)
        tau_d[0] = tau_3[0]  # Surge force Fx [N]
        tau_d[1] = tau_3[1]  # Sway force Fy [N]
        tau_d[5] = tau_3[2]  # Yaw moment Mz [Nm]

        # Log breakdown for plotters
        self.last_pid_body["P"] = np.array([tau_P_3[0], tau_P_3[1], 0.0, 0.0, 0.0, tau_P_3[2]], dtype=float)
        self.last_pid_body["I"] = np.array([tau_I_3[0], tau_I_3[1], 0.0, 0.0, 0.0, tau_I_3[2]], dtype=float)
        self.last_pid_body["D"] = np.array([tau_D_3[0], tau_D_3[1], 0.0, 0.0, 0.0, tau_D_3[2]], dtype=float)

        return tau_d

    def apply_external_aw(
        self, tau_applied: np.ndarray, psi: float, dt: float
    ) -> None:
        """
        Anti-windup back-calculation scheme when actuators saturate.
        """
        tau_applied = np.asarray(tau_applied, dtype=float)
        tau_req = self.last_pid_body["P"] + self.last_pid_body["I"] + self.last_pid_body["D"]
        delta_tau = tau_applied[[0, 1, 5]] - tau_req[[0, 1, 5]]

        Kaw = 0.05
        self.int_err += Kaw * delta_tau / (self.Kp + 1e-6) * dt
        self.int_err = np.clip(self.int_err, -self.int_limit, self.int_limit)