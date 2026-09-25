"""
Reference template

Students should filter or shape commanded setpoints before they are sent to
the controller. The simulator calls, once per step:

    ref.step(t, dt, eta_cmd) -> (eta_ref, nu_ref, acc_ref)

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; leave the rest zero.

Inputs:
    t       : current simulation time [s]
    dt      : time step [s]
    eta_cmd : (6,) commanded setpoint
              (use N_cmd = eta_cmd[0], E_cmd = eta_cmd[1], psi_cmd = eta_cmd[5])

Outputs (all NED-frame, (6,) each):
    eta_ref : filtered reference
              (fill in N_ref = [0], E_ref = [1], psi_ref = [5])
    nu_ref  : reference velocities
              (fill in Ndot_ref = [0], Edot_ref = [1], psidot_ref = [5])
    acc_ref : reference accelerations
              (fill in Nddot_ref = [0], Eddot_ref = [1], psiddot_ref = [5])

The simulator forwards all three to the controller, so a smooth reference
model here directly enables velocity/acceleration feedforward there.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np

from part_1.config import RefAxisConfig
from simulation.utils import wrap_angle_pi


class ReferenceModel:
    """
    3-DOF reference model shaping step commands into smooth reference trajectories.

    Implements a 2nd-order low-pass filter per controlled coordinate:
        eta_ddot = -2 * zeta * wn * eta_dot - wn^2 * (eta_ref - eta_cmd)
    with proper shortest-path angle wrapping for yaw.
    """

    def __init__(
        self,
        dt: float,
        cfg_xy: RefAxisConfig | None = None,
        cfg_psi: RefAxisConfig | None = None,
    ):
        self.dt = float(dt)
        self.cfg_xy = cfg_xy if cfg_xy is not None else RefAxisConfig()
        self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig()

        # Generalized 6-DOF reference states in NED (only indices [0, 1, 5] are used)
        self.eta_ref = np.zeros(6, dtype=float)
        self.nu_ref = np.zeros(6, dtype=float)
        self.acc_ref = np.zeros(6, dtype=float)

    def reset(self, eta0: np.ndarray) -> None:
        """
        Initialize the reference model at the vessel's current (6,) state.
        
        Parameters
        ----------
        eta0 : np.ndarray
            Initial 6-DOF NED vessel pose [N, E, z, phi, theta, psi].
        """
        self.eta_ref = np.asarray(eta0, dtype=float).reshape(6).copy()
        self.eta_ref[5] = wrap_angle_pi(self.eta_ref[5])
        self.nu_ref = np.zeros(6, dtype=float)
        self.acc_ref = np.zeros(6, dtype=float)

    def step(
        self, t: float, dt: float, eta_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance the reference model by dt towards eta_cmd.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        dt : float
            Time step [s].
        eta_cmd : np.ndarray
            Commanded setpoint (6,) in NED.

        Returns
        -------
        eta_ref : np.ndarray
            Filtered reference positions and heading (6,) in NED.
        nu_ref : np.ndarray
            Reference velocities (6,) in NED.
        acc_ref : np.ndarray
            Reference accelerations (6,) in NED.
        """
        cmd = np.asarray(eta_cmd, dtype=float).reshape(6)

        # Coordinate axes: (index, config, is_angle)
        axes = [
            (0, self.cfg_xy, False),   # North [m]
            (1, self.cfg_xy, False),   # East [m]
            (5, self.cfg_psi, True),   # Yaw (psi) [rad]
        ]

        for idx, cfg, is_angle in axes:
            wn = cfg.wn
            zeta = cfg.zeta
            rate_limit = cfg.rate_limit

            # Compute tracking displacement (ensure shortest path across +-pi for yaw)
            if is_angle:
                delta = wrap_angle_pi(self.eta_ref[idx] - cmd[idx])
            else:
                delta = self.eta_ref[idx] - cmd[idx]

            # 2nd-order ODE acceleration command: eta_ddot = -2*zeta*wn*eta_dot - wn^2 * delta
            acc = -2.0 * zeta * wn * self.nu_ref[idx] - (wn**2) * delta

            # Forward Euler integration for velocity
            vel = self.nu_ref[idx] + dt * acc

            # Apply velocity saturation (rate limiting) if configured
            if rate_limit is not None and rate_limit > 0.0:
                vel = float(np.clip(vel, -rate_limit, rate_limit))
                acc = (vel - self.nu_ref[idx]) / dt

            # Forward Euler integration for position
            pos = self.eta_ref[idx] + dt * vel

            # Keep heading reference normalized within (-pi, pi]
            if is_angle:
                pos = wrap_angle_pi(pos)

            self.acc_ref[idx] = acc
            self.nu_ref[idx] = vel
            self.eta_ref[idx] = pos

        return self.eta_ref.copy(), self.nu_ref.copy(), self.acc_ref.copy()