# -*- coding: utf-8 -*-
"""
Proposed personal space field.

`ps_derivation` builds the symbolic two-Gaussian directional potential and
returns CasADi functions for its gradient and Hessian. `PS_obs` evaluates
those functions for a single obstacle.
"""

import numpy as np
import casadi as ca

from config import P
from geometry_utils import vertices_coord_set, circle_coord_set, find_min_dis


def ps_derivation():
    # Define symbolic variables
    PS_weight = ca.SX.sym('PS_weight')
    X_pred = ca.SX.sym('X_pred')
    Y_pred = ca.SX.sym('Y_pred')
    xx = ca.SX.sym('xx')
    yy = ca.SX.sym('yy')
    Xi = ca.SX.sym('Xi')
    Yi = ca.SX.sym('Yi')
    Theta_i = ca.SX.sym('Theta_i')
    sigma_xx_1 = ca.SX.sym('sigma_xx_1')
    sigma_xx_2 = ca.SX.sym('sigma_xx_2')
    sigma_yy_1 = ca.SX.sym('sigma_yy_1')
    sigma_yy_2 = ca.SX.sym('sigma_yy_2')

    # Calculate robot position
    X_rob = X_pred + xx
    Y_rob = Y_pred + yy

    # Transform to obstacle's coordinate frame
    X_rob_prim = (X_rob - Xi) * ca.cos(Theta_i) + (Y_rob - Yi) * ca.sin(Theta_i)
    Y_rob_prim = -(X_rob - Xi) * ca.sin(Theta_i) + (Y_rob - Yi) * ca.cos(Theta_i)

    # Lambda function (sigmoid for directional weighting)
    k = 1000
    lamda = 0.5 * (ca.tanh(k * X_rob_prim / 2) + 1)

    # Two Gaussian components
    PS1 = ca.exp(-0.5 * (X_rob_prim**2 / sigma_xx_1**2 + Y_rob_prim**2 / sigma_yy_1**2))
    PS2 = ca.exp(-0.5 * (X_rob_prim**2 / sigma_xx_2**2 + Y_rob_prim**2 / sigma_yy_2**2))

    # Combined personal space field
    PS = PS_weight * (lamda * PS1 + (1 - lamda) * PS2)

    # Calculate gradient with respect to X_pred and Y_pred
    gradient_X_pred = ca.jacobian(PS, X_pred)
    gradient_Y_pred = ca.jacobian(PS, Y_pred)

    # Combine gradients into a single vector
    gradient = ca.vertcat(gradient_X_pred, gradient_Y_pred)

    # Calculate Hessian matrix with respect to [X_pred, Y_pred]
    # Create state vector
    state_vars = ca.vertcat(X_pred, Y_pred)

    # Hessian is the Jacobian of the gradient
    hessian = ca.jacobian(gradient, state_vars)

    input_var = ca.vertcat(
        PS_weight, X_pred, Y_pred, xx, yy, Xi, Yi,
        Theta_i, sigma_xx_1, sigma_xx_2, sigma_yy_1, sigma_yy_2
    )

    ps_func = ca.Function('ps_func',
                          [input_var],
                          [PS])

    grad_func = ca.Function('grad_func',
                            [input_var],
                            [gradient.T])
    hess_func = ca.Function('hess_func',
                            [input_var],
                            [hessian])
    return grad_func, hess_func


def PS_obs(grad_func, hess_func, X, Y, theta, obs_list_pred, option="Proposed"):
    L = P.L
    D = P.D
    sigma_yy_1 = P.sigma_yy_1
    PS_weight = P.PS_weight

    # obs_list_pred = [radius, Xi, Yi, theta]; the obstacle is treated as a
    # circle of the given radius rather than an L x D rectangle.
    radius = obs_list_pred[0]
    Xi = obs_list_pred[1]
    Yi = obs_list_pred[2]
    Theta_i = obs_list_pred[3]

    obs_list = circle_coord_set(radius, Xi, Yi)

    sigma_xx_1 = sigma_yy_1 * P.sigma_xx_multiplier
    sigma_xx_2 = sigma_yy_1
    sigma_yy_2 = sigma_yy_1

    X_pred, Y_pred, theta_pred = X, Y, theta

    rob_coord = vertices_coord_set(L, D, X, Y, theta)
    rob_short, obs_short = find_min_dis(rob_coord, obs_list, 1)

    xx = rob_short[0] - X
    yy = rob_short[1] - Y

    if option == "Proposed":
        input_val = np.array([
            PS_weight, X_pred, Y_pred, xx, yy, Xi, Yi, Theta_i,
            sigma_xx_1, sigma_xx_2, sigma_yy_1, sigma_yy_2
        ])
    else:
        raise ValueError("PS_obs only supports the 'Proposed' option")

    curr_ps_grad = np.array(grad_func(input_val))
    curr_ps_hess = np.array(hess_func(input_val))

    return curr_ps_grad, curr_ps_hess

