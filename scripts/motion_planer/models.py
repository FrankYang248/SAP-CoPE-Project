# -*- coding: utf-8 -*-
"""
Kinematic models used together with the proposed personal space field motion planner:
the four-wheel-steer robot model and the constant-velocity obstacle
prediction model used over the MPC horizon.
"""

import math
import numpy as np

from config import P


class motion_model:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw

        self.dx = 0.0
        self.dy = 0.0
        self.dyaw = 0.0

    def update(self, vf, vr, delta_f, delta_r):
        if vf > P.maxWheelVel:
            vf = P.maxWheelVel
        elif vf < P.minWheelVel:
            vf = P.minWheelVel
        if vr > P.maxWheelVel:
            vr = P.maxWheelVel
        elif vr < P.minWheelVel:
            vr = P.minWheelVel
        if delta_f > P.maxSteering:
            delta_f = P.maxSteering
        elif delta_f < P.minSteering:
            delta_f = P.minSteering
        if delta_r > P.maxSteering:
            delta_r = P.maxSteering
        elif delta_r < P.minSteering:
            delta_r = P.minSteering

        beta = math.atan2((P.lr * math.tan(delta_f) + P.lf * math.tan(delta_r)), (P.lf + P.lr))
        vc = (vf * math.cos(delta_f) + vr * math.cos(delta_r)) / (2 * math.cos(beta))
        self.dyaw = vc * math.cos(beta) * (math.tan(delta_f) - math.tan(delta_r)) / (P.lf + P.lr)
        self.dx = math.cos(self.yaw + beta) * vc
        self.dy = math.sin(self.yaw + beta) * vc

        self.yaw += self.dyaw * P.sim_Ts
        self.x += self.dx * P.sim_Ts
        self.y += self.dy * P.sim_Ts
        return np.array([self.x, self.y, self.yaw])


class obs_model_pred:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw

    def update(self, x_dot, y_dot, yaw_dot):
        if self.x <= 0.0 or self.y <= 0.0:
            self.x = self.x
            self.y = self.y
            self.yaw = self.yaw

        else:
            self.x += x_dot * P.Obs_stepTime
            self.y += y_dot * P.Obs_stepTime
            self.yaw += yaw_dot * P.Obs_stepTime

        return np.array([self.x, self.y, self.yaw])
