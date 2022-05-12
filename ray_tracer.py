import scene_utils
import intersect

import sys
import numpy as np
import random
from scene_utils import *
from PIL import Image


def represent_screen(camera, width_pixels, height_pixels):
    # Determine screen's horizontal, vertical vectors:
    horizontal = Vector(np.cross(camera.up_vector.dir, camera.towards.dir))
    vertical = Vector(np.cross(horizontal.dir, camera.towards.dir))

    # Fix camera's up vector:
    camera.up_vector = vertical.dir

    # Determine screen's leftmost bottom pixel (corner pixel):
    screen_center = camera.pos + camera.towards.dir * camera.screen_dist
    left_bottom_pixel = screen_center - (camera.screen_width/2 * horizontal.dir) - (camera.screen_height/2 * vertical.dir)

    # Normalize screen's horizontal, vertical vectors by pixel's width / height:
    pixel_width = camera.screen_width / width_pixels
    pixel_height = camera.screen_height / height_pixels
    horizontal.dir = pixel_width * horizontal.dir
    vertical.dir = pixel_height * vertical.dir

    # Align to the left bottom pixel's center:
    left_bottom_pixel += 0.5 * horizontal.dir + 0.5 * vertical.dir

    # Represent the screen:
    screen = Screen(left_bottom_pixel, horizontal, vertical)
    return screen


def construct_pixel_ray(camera, screen, i, j):
    pixel_center = screen.corner_pixel + i * screen.horizontal.dir + j * screen.vertical.dir
    ray_direction = pixel_center - camera.pos
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    pixel_ray = Ray(camera.pos, ray_direction)
    return pixel_ray


def calc_pixel_color(scene, ray, recursion_depth=0):
    if recursion_depth == scene.settings.max_recursion:
        return scene.settings.bg_color
    surface, min_intersect = intersect.find_min_intersect(scene, ray)
    if surface == None:
        return scene.settings.bg_color
    output_color = calc_surface_color(scene, surface, min_intersect, recursion_depth)
    return output_color


def calc_surface_normal(surface, min_intersect):
    if type(surface) == Sphere:
        return np.array(min_intersect - surface.center_pos)

    elif type(surface) == Plane:
        return np.array(surface.normal_vector)

    else:
        pass  # TODO fill


def calc_soft_shadow_fraction(scene, N, light, min_intersect):
    light_ray = Vector(np.array(min_intersect - light.pos))

    # Create perpendicular plane x,y to ray
    x = np.array((1., 1., 1.))
    x -= (np.dot(x, light_ray.dir) * light_ray.dir)  # make it orthogonal to ray
    x = Vector(x)
    y = Vector(np.cross(light_ray.dir, x.dir))

    # Create rectangle
    left_bottom_cell = light.pos - light.light_radius * x.dir - light.light_radius * y.dir

    # Normalize rectangle directions by cell size:
    cell_width = light.light_radius * 2 / N
    cell_height = light.light_radius * 2 / N
    x.dir *= cell_width
    y.dir *= cell_height
    intersect_counter = 0

    # Cast ray from cell to point and see if intersect with our point first
    for i in range(N):
        for j in range(N):
            cell_pos = left_bottom_cell + i * x.dir + j * y.dir
            cell_pos += random.random() * x.dir + random.random() * y.dir
            ray_direction = min_intersect - cell_pos
            cell_light_ray = Ray(light.pos, ray_direction / np.linalg.norm(ray_direction))
            cell_surface, cell_min_intersect = intersect.find_min_intersect(scene, cell_light_ray)
            if cell_min_intersect.all() == min_intersect.all():
                intersect_counter += 1
    return intersect_counter / (N * N)


def calc_surface_color(scene, surface, min_intersect, recursion_depth):
    bg_col = np.array(scene.settings.bg_color)
    trans_val = (scene.material_list[surface.material_idx]).transparent_val
    material_diffuse = np.array((scene.material_list[surface.material_idx]).diffuse_color)
    material_specular = np.array((scene.material_list[surface.material_idx]).specular_color)
    material_reflection = np.array((scene.material_list[surface.material_idx]).reflection_color)

    return bg_col * trans_val + (material_diffuse + material_specular) * (1 - trans_val)

    diffuse_col = np.array((0., 0., 0.), dtype=float)
    specular_col = np.array((0., 0., 0.), dtype=float)

    normal = Vector(calc_surface_normal(surface, min_intersect))
    for light in scene.light_list:
        # Calc light effect on diffuse color
        light_direction = min_intersect - light.pos
        soft_shadow_fraction = calc_soft_shadow_fraction(scene, scene.settings.N, light, min_intersect)
        light_intensity = (1 - light.shadow_intens) + (light.shadow_intens * soft_shadow_fraction)
        diffuse_col += (light_intensity * np.dot(normal.dir, light_direction)) / len(scene.light_list)

        # Calc light effect on specular color
        L = np.array(min_intersect - light.pos)
        R = L - 2 * (np.dot(L, normal.dir)) * normal.dir
        V = np.array(scene.camera.pos - min_intersect)
        surface_material = scene.material_list[surface.material_idx]
        n = surface_material.phong_coeff
        specular_col += (light_intensity * ((R.dot(V))**n) * light.specular_intens) / len(scene.light_list)

        # Recursively calc reflection color
        #reflection_ray = Ray(min_intersect, R)
        # reflection_col += (calc_pixel_color(scene, reflection_ray, recursion_depth+1) * material_reflection) / len(scene.light_list)

    diffuse_col *= material_diffuse
    specular_col *= material_specular

    output_color = bg_col * trans_val + (diffuse_col + specular_col) * (1 - trans_val) #+ reflection_col
    return output_color


def save_image(image_array, output_path):
    image_array = 255.*image_array
    print(image_array)
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    image.save(output_path)


def ray_tracing(scene, img_width, img_height, output_path):
    image_array = np.zeros((img_width, img_height, 3), dtype=float)
    screen = represent_screen(scene.camera, img_width, img_height)
    for i in range(img_width):
        for j in range(img_height):
            ray = construct_pixel_ray(scene.camera, screen, i, j)
            output_color = calc_pixel_color(scene, ray)
            image_array[j, i] = output_color
    save_image(image_array, output_path)


def main(scene_path, output_path, img_width=500, img_height=500):
    aspect_ratio = img_height/img_width
    scene = scene_utils.parse_scene(scene_path, aspect_ratio)
    ray_tracing(scene, img_width, img_height, output_path)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])

    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
