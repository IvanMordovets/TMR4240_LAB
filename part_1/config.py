# part_1/config.py
# -----------------------------------------------------------------------------
# TMR4240 Marine Control Systems I
# Project – Design of Dynamic Positioning System
#
# Copyright (C) 2026: NTNU, Trondheim
# License: GPL-3.0-or-later
# -----------------------------------------------------------------------------
"""
Project Part 1 configuration — all tunable parameters in one place.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from models.thruster_dynamics import ThrusterConfig


@dataclass
class SimConfig:
    """Part 1 simulation clock and options (ideal thrusters by default)."""
    dt: float = 0.05
    T: float = 1000.0
    method: str = "Euler"
    use_reference: bool = True
    thruster_dynamics: bool = False  # Part 1: ideal actuators (no rate limits, no saturation)
    bypass_actuators: bool = False   # apply tau_d directly (debug)


@dataclass
class RefAxisConfig:
    """Reference-model configuration for one axis (see part_1/reference.py)."""
    wn: float = 0.4                     # natural frequency [rad/s] (tuned for Gunnerus: soft start & settling < 600 s)
    zeta: float = 1.0                   # damping ratio [-] (critical damping = 0% overshoot)
    rate_limit: Optional[float] = None  # max |x_dot| (m/s or rad/s); None = off


@dataclass
class PIDGains:
    """
    PID Controller Gains for R/V Gunnerus 3-DOF DP system.
    Ordered as [Surge, Sway, Yaw].
    """
    # Proportional gains: [N/m, N/m, Nm/rad]
    Kp: np.ndarray = field(
        default_factory=lambda: np.array([45000, 35000, 900000], dtype=float)
    )
    # Derivative gains: [N*s/m, N*s/m, Nm*s/rad] (fortement augmentés pour freiner l'inertie)
    Kd: np.ndarray = field(
        default_factory=lambda: np.array([700000, 700000, 5500000], dtype=float)
    )
    # Integral gains: [N/(m*s), N/(m*s), Nm/(rad*s)]
    Ki: np.ndarray = field(
        default_factory=lambda: np.array([700, 500, 15000], dtype=float)
    )
    # Clamping limits resserrées pour éviter l'effet de traîne : [m*s, m*s, rad*s]
    int_limit: np.ndarray = field(
        default_factory=lambda: np.array([12.0, 12.0, np.deg2rad(8.0)], dtype=float)
    )


def default_thrusters_gunnerus3() -> list[ThrusterConfig]:
    """Three-thruster Gunnerus layout from the project description (Table 3)."""
    return [
        ThrusterConfig("Tunnel_Bow", "tunnel",  x=+12.0, y=0.0,
                       u_max=32000,  u_rate=4000,  rot_speed=0.0,    alpha0=np.pi / 2),
        ThrusterConfig("Azimuth_1",  "azimuth", x=-13.0, y=+3.0,
                       u_max=80000,  u_rate=10000, rot_speed=0.2094, alpha0=0.0),
        ThrusterConfig("Azimuth_2",  "azimuth", x=-13.0, y=-3.0,
                       u_max=80000,  u_rate=10000, rot_speed=0.2094, alpha0=0.0),
    ]
