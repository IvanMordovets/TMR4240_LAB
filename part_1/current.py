"""
Current template

Students should provide the ambient current as a generalized NED velocity
vector. The simulator calls, once per step:

    current.step(t, dt, eta, nu) -> nu_c_ned

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (time-varying currents)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5]; position for spatially varying fields)
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (only indices [0, 1, 5] are nonzero in the 3-DOF model)

Output:
    nu_c_ned : (6,) generalized NED current velocity [m/s]
               [V_N, V_E, V_D, 0, 0, 0]
               Only the horizontal components are used by the 3-DOF model.
               Direction convention is 'towards' (the direction the current
               flows to): a current with V_N > 0, V_E = 0 pushes the vessel
               North.
"""
from __future__ import annotations

import numpy as np


class Current:
    """
    Ocean current velocity generator in the Earth-fixed NED frame.

    Parameters
    ----------
    speed : float
        Current speed [m/s].
    beta : float
        Current direction [rad] in NED (0 = North, pi/2 = East).
    semantics : str, optional
        Direction convention:
        - "towards" (default): beta is the direction the water flows towards.
        - "from": beta is the direction the water comes from (flow points to beta + pi).
    beta_end : float | None, optional
        Target direction [rad] after duration seconds (Simulation 2).
    duration : float, optional
        Time window [s] over which the direction rotates from beta to beta_end.
    """

    def __init__(
        self,
        speed: float = 0.0,
        beta: float = 0.0,
        *,
        semantics: str = "towards",
        beta_end: float | None = None,
        duration: float = 0.0,
    ) -> None:
        self.speed = float(speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.beta_end = None if beta_end is None else float(beta_end)
        self.duration = float(duration)

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> np.ndarray:
        """
        Advance one step and return the NED current velocity vector.

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
        nu_c_ned : np.ndarray
            (6,) vector [V_N, V_E, 0, 0, 0, 0] in m/s.
        """
        # Determine the instantaneous direction in the constructor convention
        if self.beta_end is not None and self.duration > 0.0:
            fraction = float(np.clip(t / self.duration, 0.0, 1.0))
            current_beta = self.beta + fraction * (self.beta_end - self.beta)
        else:
            current_beta = self.beta

        # Convert direction to "towards" (the direction the flow moves towards)
        if self.semantics == "from":
            beta_towards = current_beta + np.pi
        else:
            beta_towards = current_beta

        # Calculate horizontal NED velocity components
        v_n = self.speed * np.cos(beta_towards)
        v_e = self.speed * np.sin(beta_towards)

        # Generalized 6-DOF NED current velocity vector
        nu_c_ned = np.zeros(6, dtype=float)
        nu_c_ned[0] = v_n
        nu_c_ned[1] = v_e

        return nu_c_ned