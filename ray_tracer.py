import scene_utils
import intersect

import time
import sys
import numpy as np
import random
from scene_utils import *
from PIL import Image


def construct_pixel_ray(camera, screen, i, j):
    pixel_center = screen.corner_pixel + i * screen.horizontal.dir + j * screen.vertical.dir
    ray_direction = Vector(np.array(pixel_center - camera.pos))
    pixel_ray = Ray(camera.pos, ray_direction.dir)
    return pixel_ray


def calc_pixel_color(scene, ray, eye_pos, recursion_depth):
    if recursion_depth == scene.settings.max_recursion:
        return np.array(scene.settings.bg_color)
    surfaces = intersect.find_intersect(scene, ray, find_all=True)
    if len(surfaces) == 0:
        return np.array(scene.settings.bg_color)
    output_color = calc_surface_color(scene, ray, eye_pos, surfaces, 0, recursion_depth)
    return np.array(output_color)


def calc_surface_normal(surface, min_intersect):
    if type(surface) == Sphere:
        return Vector(min_intersect - surface.center_pos)

    elif type(surface) == Plane:
        return surface.normal_vector

    else:
        return calc_box_normal(surface, min_intersect)


def calc_box_normal(box, min_intersect):
    intersect_x = min_intersect[0]
    intersect_y = min_intersect[1]
    intersect_z = min_intersect[2]
    center_x = box.center_pos[0]
    center_y = box.center_pos[1]
    center_z = box.center_pos[2]
    edge_len = box.edge_len

    # intersection is on the upper x-parallel plane
    if intersect_x - center_x == edge_len/2:
        return np.array((1, 0, 0))
    # intersection is on the lower x-parallel plane
    elif center_x - intersect_x == edge_len/2:
        return np.array((-1, 0, 0))
    # intersection is on the upper y-parallel plane
    elif intersect_y - center_y == edge_len/2:
        return np.array((0, 1, 0))
    # intersection is on the lower y-parallel plane
    elif center_y - intersect_y == edge_len/2:
        return np.array((0, -1, 0))
    # intersection is on the upper z-parallel plane
    elif intersect_z - center_z == edge_len/2:
        return np.array((0, 0, 1))
    # intersection is on the lower z-parallel plane
    else:
        return np.array((0, 0, -1))


def calc_light_intensity(scene, light, min_intersect, surface):
    N = scene.settings.N
    light_ray = Vector(min_intersect - light.pos)

    # Create perpendicular plane x,y to ray
    x = light_ray.perpendicular_vector()
    y = np.cross(light_ray.dir, x)
    y /= np.linalg.norm(y)

    # Create rectangle
    left_bottom_cell = light.pos - (light.light_radius/2) * x - (light.light_radius/2) * y

    # Normalize rectangle directions by cell size:
    cell_length = light.light_radius / N
    x *= cell_length
    y *= cell_length

    # Cast ray from cell to point and see if intersect with our point first
    intersect_counter = 0.
    for i in range(N):
        for j in range(N):
            cell_pos = left_bottom_cell + (i + random.random()) * x + (j + random.random()) * y
            ray_vector = min_intersect - cell_pos
            ray_vector /= np.linalg.norm(ray_vector)
            cell_light_ray = Ray(cell_pos, ray_vector)
            cell_surface, cell_min_intersect = intersect.find_intersect(scene, cell_light_ray, find_all=False)
            if cell_surface == surface:
                intersect_counter += 1.
    fraction = float(intersect_counter) / float(N * N)
    return (1 - light.shadow_intens) + (light.shadow_intens * fraction)


def calc_specular_color(light, eye_pos, light_intens, min_intersect, normal, phong_coeff):
    L = min_intersect - light.pos
    L /= np.linalg.norm(L)
    R = L - (2 * np.dot(L, normal.dir) * normal.dir)
    R /= np.linalg.norm(R)
    V = eye_pos - min_intersect
    V /= np.linalg.norm(V)
    dot_product = np.dot(R, V)
    if dot_product < 0:
        return np.zeros(3, dtype=float)
    specular = light.color * light_intens * (dot_product ** phong_coeff) * light.specular_intens
    return specular


def calc_diffuse_color(light, light_intens, min_intersect, normal):
    light_vector = light.pos - min_intersect
    light_vector /= np.linalg.norm(light_vector)
    dot_product = np.dot(normal.dir, light_vector)
    if dot_product < 0:
        return np.zeros(3, dtype=float)
    diffuse = light.color * light_intens * dot_product
    return diffuse


def calc_surface_color(scene, ray, eye_pos, surfaces, curr_surface, recursion_depth):
    min_surface = surfaces[curr_surface][0]
    bg_color = np.array(scene.settings.bg_color)
    trans_value = (scene.material_list[min_surface.material_idx]).transparent_val
    mat_diffuse = (scene.material_list[min_surface.material_idx]).diffuse_color
    mat_specular = (scene.material_list[min_surface.material_idx]).specular_color
    mat_reflection = (scene.material_list[min_surface.material_idx]).reflection_color
    phong_coeff = (scene.material_list[min_surface.material_idx]).phong_coeff

    diffuse_color = np.zeros(3, dtype=float)
    specular_color = np.zeros(3, dtype=float)

    min_intersect = ray.at(surfaces[curr_surface][1])
    normal = calc_surface_normal(min_surface, min_intersect)
    for light in scene.light_list:
        # Calculate the light intensity
        light_intens = calc_light_intensity(scene, light, min_intersect, min_surface)
        # Compute light effect on diffuse color
        diffuse_color += calc_diffuse_color(light, light_intens, min_intersect, normal)
        # Compute light effect on specular color
        specular_color += calc_specular_color(light, eye_pos, light_intens, min_intersect, normal, phong_coeff)

    reflection_vector = ray.dir - (2 * np.dot(ray.dir, normal.dir) * normal.dir)
    reflection_vector /= np.linalg.norm(reflection_vector)
    reflection_ray = Ray(min_intersect, reflection_vector)
    reflection_color = calc_pixel_color(scene, reflection_ray, reflection_ray.orig, recursion_depth+1)

    if trans_value > 0. and curr_surface < len(surfaces)-1:
        bg_color *= calc_surface_color(scene, ray, eye_pos, surfaces, curr_surface+1, 0)

    diffuse_color *= mat_diffuse
    specular_color *= mat_specular
    reflection_color *= mat_reflection

    return bg_color * trans_value + (diffuse_color + specular_color) * (1 - trans_value) + reflection_color


def save_image(image_array, output_path):
    image_array = np.clip(image_array, 0., 1.)
    image_array *= 255.
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    image.save(output_path)


def ray_tracing(scene, img_width, img_height, output_path):
    image_array = np.zeros((img_width, img_height, 3), dtype=float)
    screen = scene_utils.represent_screen(scene.camera, img_width, img_height)
    for i in range(img_width):
        for j in range(img_height):
            ray = construct_pixel_ray(scene.camera, screen, i, j)
            output_color = calc_pixel_color(scene, ray, scene.camera.pos, 0)
            image_array[j, i] = output_color
    save_image(image_array, output_path)


def main(scene_path, output_path, img_width=500, img_height=500):
    start = time.time()
    aspect_ratio = img_height/img_width
    scene = scene_utils.parse_scene(scene_path, aspect_ratio)
    ray_tracing(scene, img_width, img_height, output_path)
    end = time.time()
    print(f'Total time: {end-start}')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])

    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
