# -*- coding: utf-8 -*-
"""
Motion Planer
=============

Essential functions for the proposed personal space field motion planner,
extracted from ``Supporting_Functions_conventional_MPC_PF.py``.

The previous "SFM" naming has been renamed to "PS" (personal space field)
throughout the planner's functions, variables and comments.

Pipeline
--------
1. ``ps_derivation()``      -> symbolic gradient / Hessian of the personal space field.
2. ``MPC_PS_formulation()`` -> stack personal space field gradient/Hessian over the horizon.
3. ``MPC_formulation()``    -> assemble and solve the OSQP problem.
"""

from config import P
from geometry_utils import (
    vertices_coord_set,
    circle_coord_set,
    discrete_boundary,
    discrete_boundary2,
    compute_distance,
    compute_line_coordinates,
    find_min_dis,
)
from models import motion_model, obs_model_pred
from potential_field import ps_derivation, PS_obs
from mpc import (
    MPC_PS_formulation,
    ref_gen,
    matrix_batch_formulation,
    MPC_Constraint_formulation,
    MPC_formulation,
)

__all__ = [
    "P",
    "vertices_coord_set",
    "circle_coord_set",
    "discrete_boundary",
    "discrete_boundary2",
    "compute_distance",
    "compute_line_coordinates",
    "find_min_dis",
    "motion_model",
    "obs_model_pred",
    "ps_derivation",
    "PS_obs",
    "MPC_PS_formulation",
    "ref_gen",
    "matrix_batch_formulation",
    "MPC_Constraint_formulation",
    "MPC_formulation",
]
