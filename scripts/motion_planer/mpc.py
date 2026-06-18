# -*- coding: utf-8 -*-
"""
Model Predictive Control for the proposed personal space field motion planner.

`MPC_PS_formulation` stacks the personal space field gradient/Hessian over the
prediction horizon, `matrix_batch_formulation` builds the batched QP cost,
`MPC_Constraint_formulation` builds the input/rate constraints, and
`MPC_formulation` assembles and solves the OSQP problem.
"""

import math
import time
import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import eigh
from sklearn.neighbors import KDTree

from config import P
from potential_field import PS_obs


def MPC_PS_formulation(grad_func, hess_func, X, Y, theta, U, obs_list, Obs_vel, options="Proposed"):
    inteval = P.interval
    weight_a = P.obs_weight
    weight_b = P.obs_derv
    boun_weight = P.boun_weight
    boun_derv = P.boun_derv

    PS_f_cell = [None] * P.Np
    PS_h = np.zeros((2 * P.Np, 2 * P.Np))

    obs_list_pred = obs_list
    X_pred = X
    Y_pred = Y
    theta_pred = theta

    # obs_list has shape (5, N_obstacles); iterate over columns, not rows.
    Nobs = obs_list.shape[1] if obs_list.ndim == 2 else 0

    Vf_pred = U[0, 0]
    Vr_pred = U[1, 0]
    delta_f = U[2, 0]
    delta_r = U[3, 0]
    theta_pred_array = np.array([])

    start_time = time.time()

    # Personal space field construction
    for j in range(P.Np):

        curr_ps_f_sum = np.zeros((1, 2))
        curr_ps_h_sum = np.zeros((2, 2))

        # Roll the obstacle [Xi, Yi, theta] rows forward by their velocity.
        obs_list_pred[1:4, :] = obs_list_pred[1:4, :] + Obs_vel * P.Obs_stepTime

        beta_pred = math.atan2((P.lr * math.tan(delta_f) + P.lf * math.tan(delta_r)), (P.lf + P.lr))
        X_pred = X_pred + math.cos(theta_pred + beta_pred) * (P.nn * Vf_pred * math.cos(delta_f) + P.nn * Vr_pred * math.cos(delta_r)) / 2 / math.cos(beta_pred) * P.Ts
        Y_pred = Y_pred + math.sin(theta_pred + beta_pred) * (P.nn * Vf_pred * math.cos(delta_f) + P.nn * Vr_pred * math.cos(delta_r)) / 2 / math.cos(beta_pred) * P.Ts
        theta_pred = theta_pred + (math.tan(delta_f) - math.tan(delta_r)) * (P.nn * Vf_pred * math.cos(delta_f) + P.nn * Vr_pred * math.cos(delta_r)) / 2 / (P.lf + P.lr) * P.Ts
        theta_pred_array = np.vstack((theta_pred_array, theta_pred)) if theta_pred_array.size else theta_pred
        theta_pred_array = theta_pred_array.reshape(-1, 1)

        for i in range(Nobs):
            curr_ps_f, curr_ps_h = PS_obs(grad_func, hess_func, X_pred, Y_pred, theta_pred, obs_list_pred[:, i], options)

            curr_ps_f_sum = curr_ps_f_sum + curr_ps_f
            curr_ps_h_sum = curr_ps_h_sum + curr_ps_h
        PS_f_cell[j] = curr_ps_f_sum
        PS_h[2 * j:2 * j + 2, 2 * j:2 * j + 2] = curr_ps_h_sum

    # end_time = time.time()
    # print("personal space field computation time:", end_time - start_time)

    PS_f = np.block(PS_f_cell)

    # Magnitude caps to prevent near-collision Hessian / gradient blow-up.
    # The Gaussian personal space field derivatives scale as PS_weight/sigma^2 (~920 per
    # pedestrian at peak with sigma_yy=0.36, PS_weight=120); stacking N_obs
    # pedestrians or having one inside the sigma-band can push these to 1e6+.
    # Clipping element-wise bounds the QP cost so OSQP can solve it cleanly
    # while still leaving plenty of headroom for "strong repulsion".
    PS_H_MAX = P.PS_H_MAX
    PS_F_MAX = P.PS_F_MAX
    PS_h = np.clip(PS_h, -PS_H_MAX, PS_H_MAX)
    PS_f = np.clip(PS_f, -PS_F_MAX, PS_F_MAX)
    # Replace any NaN/Inf that slipped through (e.g. obstacle directly on the
    # predicted robot point) with zero — the conservative "no contribution".
    PS_h = np.nan_to_num(PS_h, nan=0.0, posinf=PS_H_MAX, neginf=-PS_H_MAX)
    PS_f = np.nan_to_num(PS_f, nan=0.0, posinf=PS_F_MAX, neginf=-PS_F_MAX)

    return PS_h, PS_f, theta_pred_array


def ref_gen(ref_defined, X, Y):
    Np = P.Np

    rob_curr = np.array([X, Y])
    rob_curr = rob_curr.reshape(-1, 1)

    tree = KDTree((ref_defined[0:2, :]).T)
    dist, ind = tree.query(rob_curr.T, k=1)
    indd = ind.flatten()
    Yita_ref_cell = [None] * Np
    for p in range(1, Np + 1):
        if indd + p - 1 >= len(ref_defined[0]):
            if p == 1:
                dummy = ref_defined[:, -1]
                Yita_ref_cell[p - 1] = dummy
            else:
                dummy2 = Yita_ref_cell[p - 2]
                Yita_ref_cell[p - 1] = dummy2
        else:
            Yita_ref_cell[p - 1] = ref_defined[:, indd + p - 1]

    Yita_ref = np.column_stack(Yita_ref_cell)
    Yita_ref = Yita_ref.reshape(-1, 1, order='F')

    return Yita_ref, indd


def matrix_batch_formulation(X, Y, theta, U, PS_h, PS_f, ref_defined):
    Np = P.Np
    Nc = P.Nc
    Ny = P.Ny
    Ts = P.Ts
    la = P.lf
    lb = P.lf
    Nx = P.Nx
    Nu = P.Nu
    Q = P.Q
    R = P.R
    G = P.G
    wheelVel_ref = P.wheelVel_ref

    kesi = np.zeros((Nx + Nu, 1))

    # Assign values to kesi array
    kesi[0] = X
    kesi[1] = Y
    kesi[2] = theta

    Vf = U[0, 0]
    Vr = U[1, 0]
    delta_f = U[2, 0]
    delta_r = U[3, 0]

    # Assign control inputs to kesi array
    kesi[Nx:] = U

    # Define matrix 'a'
    A = np.array([
                    [1.0, 0, -0.5*Ts*np.sin(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2)*(Vf*np.cos(delta_f) + Vr*np.cos(delta_r))],
                    [0, 1.0, 0.5*Ts*np.cos(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2)*(Vf*np.cos(delta_f) + Vr*np.cos(delta_r))],
                    [0, 0, 1.0]
                ])

    # Define matrix 'b'
    B = np.array([
                        [0.5*Ts*np.cos(delta_f)*np.cos(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2),
                        0.5*Ts*np.cos(delta_r)*np.cos(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2),
                        -1.0*Ts*(0.5*Vf*np.sin(delta_f)*np.cos(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2) + (0.5*np.sin(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*(Vf*np.cos(delta_f) + Vr*np.cos(delta_r))*(0.5*np.tan(delta_f)**2 + 0.5))/((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2) - (0.5*np.cos(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))*(Vf*np.cos(delta_f) + Vr*np.cos(delta_r))*(0.5*np.tan(delta_f)**2 + 0.5))/((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2)),
                        -1.0*Ts*(0.5*Vr*np.sin(delta_r)*np.cos(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2) + (0.5*np.sin(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*(Vf*np.cos(delta_f) + Vr*np.cos(delta_r))*(0.5*np.tan(delta_r)**2 + 0.5))/((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2) - (0.5*np.cos(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))*(Vf*np.cos(delta_f) + Vr*np.cos(delta_r))*(0.5*np.tan(delta_r)**2 + 0.5))/((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2))],

                        [0.5*Ts*np.cos(delta_f)*np.sin(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2),
                        0.5*Ts*np.cos(delta_r)*np.sin(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2),
                        Ts*((0.5*np.cos(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*(Vf*np.cos(delta_f) + Vr*np.cos(delta_r))*(0.5*np.tan(delta_f)**2 + 0.5))/((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2) - 0.5*Vf*np.sin(delta_f)*np.sin(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2) + (0.5*np.sin(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))*(Vf*np.cos(delta_f) + Vr*np.cos(delta_r))*(0.5*np.tan(delta_f)**2 + 0.5))/((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2)),
                        Ts*((0.5*np.cos(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*(Vf*np.cos(delta_f) + Vr*np.cos(delta_r))*(0.5*np.tan(delta_r)**2 + 0.5))/((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2) - 0.5*Vr*np.sin(delta_r)*np.sin(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2) + (0.5*np.sin(theta + np.arctan(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r)))*(0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))*(Vf*np.cos(delta_f) + Vr*np.cos(delta_r))*(0.5*np.tan(delta_r)**2 + 0.5))/((0.5*np.tan(delta_f) + 0.5*np.tan(delta_r))**2 + 1.0)**(1/2))],

                        [0.208*Ts*np.cos(delta_f)*(np.tan(delta_f) - 1.0*np.tan(delta_r)),
                        0.208*Ts*np.cos(delta_r)*(np.tan(delta_f) - 1.0*np.tan(delta_r)),
                        Ts*(0.208*(Vf*np.cos(delta_f) + Vr*np.cos(delta_r))*(np.tan(delta_f)**2 + 1.0) - 0.208*Vf*np.sin(delta_f)*(np.tan(delta_f) - 1.0*np.tan(delta_r))),
                        -1.0*Ts*(0.208*(Vf*np.cos(delta_f) + Vr*np.cos(delta_r))*(np.tan(delta_r)**2 + 1.0) + 0.208*Vr*np.sin(delta_r)*(np.tan(delta_f) - 1.0*np.tan(delta_r)))]
                    ])

    beta = math.atan2((lb*math.tan(delta_f)+la*math.tan(delta_r)), (la+lb))  # vehicle slip angle
    Vc = (Vf*math.cos(delta_f)+Vr*math.cos(delta_r))/(2*math.cos(beta))  # vehicle speed
    state_k1 = np.zeros((3, 1))
    state_k1[0, 0] = X + Ts * (np.cos(theta + beta) * Vc)
    state_k1[1, 0] = Y + Ts * (np.sin(theta + beta) * Vc)
    state_k1[2, 0] = theta + Ts * ((Vc*np.cos(beta)*(np.tan(delta_f)-np.tan(delta_r))) / (la + lb))

    # Calculate d_k
    misc1 = np.matmul(A, kesi[0:Nx])
    misc2 = np.matmul(B, kesi[Nx:Nx+Nu])
    d_piao_k = state_k1 - misc1.reshape(-1, 1) - misc2.reshape(-1, 1)

    # Assign values to matrix C
    C = np.eye(Ny)

    GAMMA_cell = [[np.dot(C, np.linalg.matrix_power(A, max(p - q, 0))) if q <= p else np.zeros((Ny, Nx)) for q in range(Np)] for p in range(Np)]
    THETA_cell = [[np.dot(np.dot(C, np.linalg.matrix_power(A, max(j - k, 0))), B) if k <= j else np.zeros((Ny, Nu)) for k in range(Nc)] for j in range(Np)]

    GAMMA = np.block(GAMMA_cell)
    THETA = np.block(THETA_cell)
    PSI = []
    PHI = []
    for j in range(Np):
        if j == 0:
            PSI = C@(np.linalg.matrix_power(A, j+1))
            PHI = d_piao_k
        else:
            midStep = C@(np.linalg.matrix_power(A, j+1))
            PSI = np.concatenate((PSI, midStep), axis=0)
            PHI = np.concatenate((PHI, d_piao_k), axis=0)

    # Calculate Cps and Z0
    Cps = np.kron(np.eye(Np), np.array([[1, 0, 0], [0, 1, 0]]))
    Z0 = np.tile(np.array([[X], [Y]]), (Np, 1))

    CC = np.kron(np.eye(Nc), [[1, 0, 0, 0], [0, 1, 0, 0]])

    # Construct H
    H = 2 * (THETA.T @ Q @ THETA) + 2 * R + THETA.T @ Cps.T @ PS_h @ Cps @ THETA + 2 * CC.T @ G @ CC

    # Ensure symmetry
    H = (H + H.T) / 2

    # Compute eigendecomposition
    eigenvalues, eigenvectors = eigh(H)

    # Clip eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0)

    # Reconstruct
    H = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    Yita_ref, ind_ref = ref_gen(ref_defined, X, Y)

    error_1 = Yita_ref - PSI @ kesi[0:Nx] - GAMMA @ PHI

    U_ref = wheelVel_ref*np.ones(2*P.Nc).reshape(-1, 1)
    f = -2 * error_1.T @ Q @ THETA - 2*U_ref.T @ G @ CC + PS_f @ Cps @ THETA + ((Cps @ PSI @ kesi[0:Nx] + Cps @ GAMMA @ PHI - Z0).T @ PS_h @ Cps) @ THETA

    return H, f, THETA, GAMMA, PSI, PHI, kesi, ind_ref


def MPC_Constraint_formulation(U, THETA, GAMMA, PSI, PHI, kesi):
    Np = P.Np
    Nc = P.Nc
    Nu = P.Nu

    Vf = U[0, 0]
    Vr = U[1, 0]
    delta_f = U[2, 0]
    delta_r = U[3, 0]

    A_I = np.kron(-1*np.eye(Nc, k=-1)+np.eye(Nc), np.eye(Nu))
    U_t = np.kron(np.ones((P.Nc, 1)), U)
    U_t[P.Nu:P.Nu*P.Nc, :] = 0

    umin = np.array([P.minWheelVel, P.minWheelVel, P.minSteering, P.minSteering])
    umin = umin.reshape(-1, 1)
    umax = np.array([P.maxWheelVel, P.maxWheelVel, P.maxSteering, P.maxSteering])
    umax = umax.reshape(-1, 1)
    delta_umin = np.array([P.minWheelAccl, P.minWheelAccl, P.minDeltaSteering, P.minDeltaSteering])
    delta_umin = delta_umin.reshape(-1, 1)
    delta_umax = np.array([P.maxWheelAccl, P.maxWheelAccl, P.maxDeltaSteering, P.maxDeltaSteering])
    delta_umax = delta_umax.reshape(-1, 1)
    Umin = np.kron(np.ones((Nc, 1)), umin)
    Umax = np.kron(np.ones((Nc, 1)), umax)
    delta_Umin = np.kron(np.ones((Nc, 1)), delta_umin)
    delta_Umax = np.kron(np.ones((Nc, 1)), delta_umax)

    diff = 0.1
    F = np.array([np.cos(delta_f), -np.cos(delta_r), -Vf*np.sin(delta_f), Vr*np.sin(delta_r)]).reshape(1, -1)
    con_current = Vf*np.cos(delta_f) - Vr*np.cos(delta_r)
    Aa = np.kron(np.eye(P.Nc), F)
    Bb_upper = np.kron(np.ones((P.Nc, 1)), diff-con_current+F@U)
    Bb_lower = np.kron(np.ones((P.Nc, 1)), -diff-con_current+F@U)
    # Construct matrices for constraints
    aa_cell = [
        [np.kron(np.eye(Nc), np.eye(Nu))],
        [A_I],
        [Aa]
    ]

    u_cell = [
        [Umax],
        [delta_Umax+U_t],
        [Bb_upper]
    ]

    l_cell = [
        [Umin],
        [delta_Umin+U_t],
        [Bb_lower]
    ]

    A_cons = np.vstack([np.hstack(cell) for cell in aa_cell])
    l = np.vstack([np.hstack(cell) for cell in l_cell])
    u = np.vstack([np.hstack(cell) for cell in u_cell])
    return A_cons, l, u


def MPC_formulation(X, Y, theta, U, PS_h, PS_f, ref_defined):
    H, f, THETA, GAMMA, PSI, PHI, kesi, ind_ref = matrix_batch_formulation(X, Y, theta, U, PS_h, PS_f, ref_defined)
    f = f.T

    A_cons, l, u = MPC_Constraint_formulation(U, THETA, GAMMA, PSI, PHI, kesi)

    sH = sparse.csr_matrix(H)
    sA_cons = sparse.csr_matrix(A_cons)

    prob = osqp.OSQP()
    prob.setup(sH, f, sA_cons, l, u, eps_abs=1e-5,
                                     eps_rel=1e-5,
                                     eps_prim_inf=1e-6,
                                     eps_dual_inf=1.0e-6,
                                     max_iter=1000,
                                     verbose=False)
    Sol = prob.solve()
    state = Sol.info.status_val
    return Sol.x, state, ind_ref, THETA, GAMMA, PSI, PHI, kesi, H
