# -*- coding: utf-8 -*-
"""
Global configuration for the proposed personal space field motion planner.
"""

import math
import numpy as np


class P:
    # System config
    Nx = 3  # state vector: zeta = [x, y, phi]
    Ny = Nx
    Nu = 4  # input vector: u = [acceleration, steer]
    T = 28  # finite time horizon length
    sim_Ts = 0.02
    Ts = 0.1
    total_simTime = 50.0
    ref_sampleTime = Ts * 1.5

    # MPC config
    Np = 20
    Nc = 20

    obs_weight = 2.5
    obs_derv = 1.1

    boun_weight = 0.1
    boun_derv = 1.1
    Obs_stepTime = Ts * 1.0
    nn = 0.9
    Row = 100

    PS_weight = 50
    sigma_yy_1 = 0.165
    # Front-lobe stretch ratio: sigma_xx_1 = sigma_yy_1 * sigma_xx_multiplier.
    # Shared by the MPC personal space field gradient (Proposed) and the contour visualization.
    sigma_xx_multiplier = 1.5

    Q = np.kron(np.eye(Np), np.diag([3, 3, 6]))
    R = 1 * np.kron(np.eye(Nc), np.diag([1, 1, 2, 2]))
    S = 1 * np.kron(np.eye(Nc), np.diag([1, 1]))

    # vehicle config
    L = 2.4  # [m]
    D = 0.9  # [m]
    lf = L / 2  # [m]
    lr = lf  # [m]

    # Physical Limitations of Robot
    maxWheelVel = 1.3  # [m/s]
    minWheelVel = -0.01  # [m/s]
    maxWheelAccl = 3 / 3.6 / 10  # [m/s]
    minWheelAccl = -6 / 3.6 / 10  # [m/s]
    maxSteering = math.pi / 2  # [rad]
    minSteering = -math.pi / 2  # [rad]
    maxDeltaSteering = math.pi / 3 / 10  # [rad/s]
    minDeltaSteering = -math.pi / 3 / 10  # [rad/s]

    wheelVel_ref = 1.0

    interval = .15
    n = 10

    PS_H_MAX = 1.0e2
    PS_F_MAX = 1.0e2
