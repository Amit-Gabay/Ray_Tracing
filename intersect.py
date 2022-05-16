import numpy as np
from math import sqrt


EPSILON = 10**-9


def find_sphere_intersect(ray, sphere):
    L = sphere.center_pos - ray.orig
    t_ca = np.dot(L, ray.dir)
    # If sphere is behind us:
    if t_ca < 0:
        return -1

    r_squared = sphere.radius_squared

    d_squared = np.dot(L, L) - t_ca * t_ca
    # If the ray is outside of sphere:
    if d_squared > r_squared:
        return -1
    t_hc = sqrt(r_squared - d_squared)
    t = t_ca - t_hc
    return t


def find_plane_intersect(ray, plane):
    return (-1 * (np.dot(ray.orig, plane.normal_vector.dir) - plane.offset)) / np.dot(ray.dir, plane.normal_vector.dir)


def find_box_intersect(ray, box):
    x_min = (box.min_extent[0] - ray.orig[0]) / ray.dir[0]
    x_max = (box.max_extent[0] - ray.orig[0]) / ray.dir[0]

    if x_min > x_max:
        # Swap:
        tmp = x_min
        x_min = x_max
        x_max = tmp

    y_min = (box.min_extent[1] - ray.orig[1]) / ray.dir[1]
    y_max = (box.max_extent[1] - ray.orig[1]) / ray.dir[1]

    if y_min > y_max:
        # Swap:
        tmp = y_min
        y_min = y_max
        y_max = tmp

    if x_min > y_max or y_min > x_max:
        return -1

    if y_min > x_min:
        x_min = y_min
    if y_max < x_max:
        x_max = y_max

    z_min = (box.min_extent[2] - ray.orig[2]) / ray.dir[2]
    z_max = (box.max_extent[2] - ray.orig[2]) / ray.dir[2]

    if z_min > z_max:
        # Swap:
        tmp = z_min
        z_min = z_max
        z_max = tmp

    if x_min > z_max or z_min > x_max:
        return -1

    if z_min > x_min:
        x_min = z_min
    if z_max < x_max:
        x_max = z_max
    return x_min


def find_intersect(scene, ray, find_all=True):
    # Initialize intersection's distance and surface:
    min_dist = -1
    min_surface = None
    surfaces = []

    # Find nearest intersection with the scene surfaces:
    for sphere in scene.sphere_list:
        dist = find_sphere_intersect(ray, sphere)
        if find_all and dist >= EPSILON:
            surfaces.append((sphere, dist))
        elif (min_dist == -1 and dist >= EPSILON) or EPSILON <= dist < min_dist:
            min_dist = dist
            min_surface = sphere

    for plane in scene.plane_list:
        dist = find_plane_intersect(ray, plane)
        if find_all and dist >= EPSILON:
            surfaces.append((plane, dist))
        elif (min_dist == -1 and dist >= EPSILON) or EPSILON <= dist < min_dist:
            min_dist = dist
            min_surface = plane

    for box in scene.box_list:
        dist = find_box_intersect(ray, box)
        if find_all and dist >= EPSILON:
            surfaces.append((box, dist))
        elif (min_dist == -1 and dist >= EPSILON) or EPSILON <= dist < min_dist:
            min_dist = dist
            min_surface = box

    if find_all:
        surfaces.sort(key=lambda tuple: tuple[1])
        return surfaces

    min_intersect = ray.at(min_dist)
    return min_surface, min_intersect


