"""
Thrust Allocation template

Students should implement an algorithm that maps the desired body-frame
wrench to individual thruster commands. The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state — use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (rate-aware/dynamic allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (rate-aware allocation)
    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Students may implement, for example:
    - pseudo-inverse allocation,
    - weighted least-squares allocation,
    - optimization-based allocation,
    - power-minimizing allocation.

Contract with the checks, exactly as the public checks test it: ``allocate()``
must never command more thrust than a thruster can deliver, so for an
infeasible demand (the check uses 400 kN of sway) every |u_cmd| <= u_max.
*How* you handle an infeasible demand beyond that limit is your design
decision, not a checked criterion — state your rule and justify it in the
report.  The check also calls ``allocate()`` repeatedly from the initial
actuator state, applies each command through IDEAL actuators and feeds the
resulting actuator state back as ``u_now``/``alpha_now``; after 30 s of such
calls the achieved wrench must equal the requested one.  A steady-state
allocator therefore passes on its first call; a rate-aware allocator
(optional extension, and optional in the project text too) passes once it has
converged.  In the simulation the transient (thrust ramps, 2 rpm azimuth
slews) is shaped by the actuator model in ``models/thruster_dynamics.py``
whatever you command, so rate limiting your own commands is optional, not
required.  ``u_now``/``alpha_now`` are passed in because a thruster produces
the same force as (u, alpha) or as (-u, alpha + pi): which of the two you
command, and on what basis, is part of your allocation design.
"""
from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig


class ThrustAllocator:
    """Template for student thrust allocation."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters

        # Build B_e column by column, and remember which z-index(es) belong
        # to which physical thruster (self.thrusters[i] <-> self.layout[i]),
        # since a tunnel takes 1 column of z but an azimuth takes 2.
        columns = []
        self.layout: list[tuple] = []
        for th in self.thrusters:
            if th.kind == "tunnel":
                col = np.array([np.cos(th.alpha0),
                                np.sin(th.alpha0),
                                th.x*np.sin(th.alpha0) - th.y*np.cos(th.alpha0)])
                self.layout.append(("tunnel", len(columns)))
                columns.append(col)
            elif th.kind == "azimuth":
                colx = np.array([1.0, 0.0, -th.y])
                coly = np.array([0.0, 1.0, th.x])
                self.layout.append(("azimuth", len(columns), len(columns) + 1))
                columns.append(colx)
                columns.append(coly)

        self.B_e = np.column_stack(columns)
        
        def get_B_e(self) -> np.ndarray:
            return self.B_e

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.thrusters)
        tau_c = tau_d[[0, 1, 5]]          # (3,) — the 3-DOF wrench: [Fx, Fy, Mz]
        
        M = self.B_e @ self.B_e.T         # (3,3) — always square, invertible if rank(B_e)=3
        y = np.linalg.solve(M, tau_c)     # solve rather than invert (numerically preferred)
        z = self.B_e.T @ y                # (5,) — [u_T, Fx1, Fy1, Fx2, Fy2]

        # Unpack z back into per-thruster (u_i, alpha_i) using self.layout.
        u_cmd = np.zeros(n)
        alpha_cmd = np.zeros(n)
        for i, entry in enumerate(self.layout):
            if entry[0] == "tunnel":
                _, col = entry
                u_cmd[i] = z[col]
                alpha_cmd[i] = self.thrusters[i].alpha0
            else:
                _, colx, coly = entry
                Fx, Fy = z[colx], z[coly]
                u_cmd[i] = np.hypot(Fx, Fy)
                alpha_cmd[i] = np.arctan2(Fy, Fx)

        return u_cmd, alpha_cmd
