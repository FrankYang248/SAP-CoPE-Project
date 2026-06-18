# -*- coding: utf-8 -*-
"""
Geometry helpers used by the proposed personal space field motion planner:
rectangle vertices, boundary discretisation and minimum-distance search.
"""

import math
import numpy as np
from scipy.spatial import cKDTree

from config import P


def vertices_coord_set(L, D, xc, yc, theta):
    L1 = -L / 2
    D1 = -D / 2
    D2 = -D / 2
    L2 = L / 2
    L3 = L / 2
    D3 = D / 2
    D4 = D / 2
    L4 = -L / 2

    x = np.array([L1, L2, L3, L4, L1])
    y = np.array([D1, D2, D3, D4, D1])

    cth = np.cos(theta)
    sth = np.sin(theta)
    Xrot = x * cth - y * sth
    Yrot = x * sth + y * cth

    X_corn = Xrot + xc
    Y_corn = Yrot + yc

    Coord = np.column_stack((X_corn, Y_corn))
    return Coord


def circle_coord_set(radius, xc, yc, n_points=12):
    """Boundary points of a circular obstacle of given radius, centred at
    (xc, yc), returned as a closed polygon (first point repeated at the end)."""
    angles = np.linspace(0, 2 * np.pi, n_points + 1)
    x = xc + radius * np.cos(angles)
    y = yc + radius * np.sin(angles)
    return np.column_stack((x, y))


def discrete_boundary(pos, n):
    X = pos[:, 0]
    Y = pos[:, 1]
    inter = []
    for i in range(len(pos) - 1):
        if inter:
            inter.pop()
        inter.extend(np.column_stack((np.linspace(X[i], X[i + 1], n), np.linspace(Y[i], Y[i + 1], n))))

    coord = np.array(inter)
    return coord


def compute_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def compute_line_coordinates(start, end, interval_length):
    # Calculate the total distance between the points
    total_distance = compute_distance(start, end)

    # Calculate the unit vector from start to end
    unit_vector = ((end[0] - start[0]) / total_distance, (end[1] - start[1]) / total_distance)

    num_intervals = total_distance / interval_length

    # Generate the coordinates
    coordinates = [start]
    interval_multiple = interval_length
    for i in range(1, int(num_intervals + 1)):
        new_x = start[0] + unit_vector[0] * interval_multiple
        new_y = start[1] + unit_vector[1] * interval_multiple
        coordinates.append((new_x, new_y))
        interval_multiple += interval_length

    return coordinates


def discrete_boundary2(pos, interval_length):
    X = pos[:, 0]
    Y = pos[:, 1]
    inter = np.array([])
    for i in range(len(pos) - 1):
        coordinates = np.array(compute_line_coordinates(np.array([X[i], Y[i]]), np.array([X[i + 1], Y[i + 1]]), interval_length))
        inter = np.vstack([inter, coordinates]) if inter.size else coordinates

    coord = np.array(inter)
    return coord


def find_min_dis(obs1, obs2, if_obs):
    if if_obs == 1:
        discrete_points_obs1 = discrete_boundary2(obs1, P.interval)
        discrete_points_obs2 = discrete_boundary2(obs2, P.interval)
    else:
        discrete_points_obs1 = obs1
        discrete_points_obs2 = obs2

    min_dists, min_dist_idx = cKDTree(discrete_points_obs1).query(discrete_points_obs2, 1)
    ind = np.argmin(min_dists, axis=None)
    min_point_obs1 = discrete_points_obs1[min_dist_idx[ind]]
    min_point_obs2 = discrete_points_obs2[ind]

    min_point_obs1 = np.array(min_point_obs1)
    min_point_obs2 = np.array(min_point_obs2)

    min_point_obs1.reshape(-1, 1)
    min_point_obs2.reshape(-1, 1)

    return min_point_obs1, min_point_obs2
