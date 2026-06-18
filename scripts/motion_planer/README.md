# Motion Planer — Proposed Personal Space Field MPC Controller

Essential functions for the **proposed motion planner**, extracted from
`Supporting_Functions_conventional_MPC_PF.py`.

The controller tracks a reference path with a **Model Predictive Controller
(MPC)** while avoiding dynamic obstacles (pedestrians) through a **personal
space field** — a directional, two-Gaussian repulsion potential whose gradient
and Hessian are folded into the MPC cost. The robot is a four-wheel-steer
vehicle (independent front/rear wheel speeds and steering angles).

> The legacy `SFM` (Social Force Model) naming used in the original code has
> been renamed to `PS` (personal space field) throughout this package.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | array maths |
| `scipy` | sparse matrices, `cKDTree`, eigen-decomposition |
| `scikit-learn` | `KDTree` for reference look-ahead |
| `casadi` | symbolic gradient/Hessian of the personal space field |
| `osqp` | quadratic-program solver for the MPC step |

Python 3.8+ is recommended. Install everything with:

```bash
pip install numpy scipy scikit-learn casadi osqp
```

---

## Package layout

| File | Contents |
|---|---|
| `config.py` | `P` — all tuning/vehicle/MPC parameters |
| `geometry_utils.py` | robot rectangle vertices, circular-obstacle boundary, boundary discretisation, minimum-distance search |
| `models.py` | `motion_model` (robot kinematics), `obs_model_pred` (constant-velocity obstacle predictor) |
| `potential_field.py` | `ps_derivation()` (symbolic field), `PS_obs()` (per-obstacle gradient/Hessian) |
| `mpc.py` | `MPC_PS_formulation()`, `MPC_formulation()`, and the batched QP helpers |
| `__init__.py` | re-exports the public API |

---

## Data conventions

- **Robot state** — `X, Y, theta` (position [m] and heading [rad]).
- **Control input** `U` — a `(4, 1)` column `[Vf, Vr, delta_f, delta_r]`
  (front/rear wheel speed, front/rear steering angle).
- **Obstacles** `obs_list` — shape `(4, N)`, one column per obstacle:
  rows are `[radius, Xi, Yi, theta]` (each obstacle is modelled as a circle of
  the given radius at position `(Xi, Yi)` with heading `theta`).
- **Obstacle velocities** `Obs_vel` — shape `(3, N)`, rows `[vx, vy, vyaw]`,
  used to roll obstacles forward over the prediction horizon.
- **Reference path** `ref_defined` — shape `(3, M)`, rows `[x; y; theta]`.

If no obstacle is visible, pass a single far-away dummy column, e.g.
`obs_list = np.array([[0.25], [-1000.0], [-1000.0], [0.0]])` and
`Obs_vel = np.zeros((3, 1))`.

---

## Usage

```python
import numpy as np
from config import P
from models import motion_model
from potential_field import ps_derivation
from mpc import MPC_PS_formulation, MPC_formulation

# 1. Build the personal space field gradient/Hessian functions once.
grad_func, hess_func = ps_derivation()

# 2. Initialise the robot and its control input.
robot = motion_model(x=0.0, y=0.0, yaw=0.0)
U = np.zeros((P.Nu, 1))          # [Vf, Vr, delta_f, delta_r]

# 3. Your reference path (3 x M) and obstacle data (4 x N), (3 x N).
ref_defined = ...                # [x; y; theta]
obs_list    = ...                # [radius; Xi; Yi; theta]
Obs_vel     = ...                # [vx; vy; vyaw]

X, Y, theta = robot.x, robot.y, robot.yaw

# 4. Stack the personal space field over the horizon and solve the MPC QP.
PS_h, PS_f, theta_pred = MPC_PS_formulation(
    grad_func, hess_func, X, Y, theta, U, obs_list, Obs_vel)

sol, status, ref_ind, THETA, GAMMA, PSI, PHI, kesi, H = MPC_formulation(
    X, Y, theta, U, PS_h, PS_f, ref_defined)

# 5. Apply the first control move (OSQP status 1 = solved, 2 = solved-inaccurate).
if status in (1, 2):
    U = sol[0:P.Nu].reshape(-1, 1)
    robot.update(vf=U[0, 0], vr=U[1, 0], delta_f=U[2, 0], delta_r=U[3, 0])
else:
    pass  # solver failed — hold the previous U
```

Run this inside a simulation loop: re-read the robot pose and obstacle data each
step, then call `MPC_PS_formulation` + `MPC_formulation` and apply the result.
The robot integrates at `P.sim_Ts`; the MPC is typically re-solved every few
simulation steps rather than every step.

---

## Tuning

All knobs live in `config.py` (`class P`). The most relevant for the personal
space field controller:

| Parameter | Meaning |
|---|---|
| `Np`, `Nc` | prediction / control horizon length |
| `Ts`, `sim_Ts` | MPC step time / simulation step time [s] |
| `Q`, `R`, `G` | tracking, input, and reference-speed weights |
| `PS_weight` | overall strength of the personal space field |
| `sigma_yy_1`, `sigma_xx_multiplier` | lateral / longitudinal extent of the field lobes |
| `PS_H_MAX`, `PS_F_MAX` | element-wise caps on the field Hessian/gradient (numerical safety near collisions) |
| vehicle limits | `maxWheelVel`, `maxSteering`, `maxWheelAccl`, `maxDeltaSteering`, ... |
