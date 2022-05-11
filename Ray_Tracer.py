import sys
import numpy as np
import random
from Classes import *
from PIL import Image


def parse_scene(scene_path, asp_ratio):
    with open(scene_path, 'rb') as f:
        content = f.read()

    material_list = []
    sphere_list = []
    plane_list = []
    light_list = []
    box_list = []
    camera = None
    settings = None
    for line in content.splitlines():
        if len(line) == 0 or line[0] == b'#':
            continue
        line = line.split()
        obj_name = line[0]

        if obj_name == b'cam':
            pos = np.array((float(line[1]), float(line[2]), float(line[3])))
            look_at = np.array((float(line[4]), float(line[5]), float(line[6])))
            up_vector = np.array((float(line[7]), float(line[8]), float(line[9])))
            screen_dist = float(line[10])
            screen_width = float(line[11])
            screen_height = screen_width * asp_ratio
            camera = Camera(pos, look_at, up_vector, screen_dist, screen_width, screen_height)

        elif obj_name == b'set':
            bg_color = (float(line[1]), float(line[2]), float(line[3]))
            N = int(line[4])
            max_recursion = int(line[5])
            settings = Settings(bg_color, N, max_recursion)

        elif obj_name == b'lgt':
            pos = (float(line[1]), float(line[2]), float(line[3]))
            color = (float(line[4]), float(line[5]), float(line[6]))
            specular_intens = float(line[7])
            shadow_intens = float(line[8])
            light_radius = float(line[9])
            light_list.append(Light(pos, color, specular_intens, shadow_intens, light_radius))

        elif obj_name == b'sph':
            center_pos = (float(line[1]), float(line[2]), float(line[3]))
            radius = float(line[4])
            material_idx = int(line[5])
            sphere_list.append(Sphere(center_pos, radius, material_idx))

        elif obj_name == b'pln':
            normal_vector = (float(line[1]), float(line[2]), float(line[3]))
            offset = float(line[4])
            material_idx = int(line[5])
            plane_list.append(Plane(normal_vector, offset, material_idx))

        elif obj_name == b'box':
            center_pos = (float(line[1]), float(line[2]), float(line[3]))
            edge_len = float(line[4])
            material_idx = int(line[5])
            box_list.append(Box(center_pos, edge_len, material_idx))

        elif obj_name == b'mtl':
            diffuse_col = (float(line[1]), float(line[2]), float(line[3]))
            specular_col = (float(line[4]), float(line[5]), float(line[6]))
            reflection_col = (float(line[7]), float(line[8]), float(line[9]))
            phong_coeff = float(line[10])
            transparent_val = float(line[11])
            material_list.append(Material(diffuse_col, specular_col, reflection_col, phong_coeff, transparent_val))

    scene = Scene(camera, settings, sphere_list, plane_list, box_list, light_list, material_list)
    return scene


def represent_screen(camera, width_pixels, height_pixels):
    # Determine screen's horizontal, vertical vectors:
    horizontal = np.cross(camera.towards, camera.up_vector)
    # Fix the up vector:
    camera.up_vector = np.cross(horizontal, camera.towards)
    horizontal = horizontal / np.linalg.norm(horizontal)
    vertical = np.cross(horizontal, camera.towards)
    vertical = vertical / np.linalg.norm(vertical)
    # Determine screen's leftmost bottom pixel (corner pixel):
    left_bottom_pixel = camera.towards * camera.screen_dist - camera.screen_width/2 * horizontal - camera.screen_height/2 * vertical
    # Normalize screen's horizontal, vertical vectors by pixel's width / height:
    pixel_width = camera.screen_width / width_pixels
    pixel_height = camera.screen_height / height_pixels
    horizontal = pixel_width * horizontal
    vertical = pixel_height * vertical
    # Align 'left_bottom_pixel' to the pixel's center:
    left_bottom_pixel += 0.5*horizontal + 0.5*vertical
    # Represent the screen:
    screen = Screen(left_bottom_pixel, horizontal, vertical)
    return screen


def construct_pixel_ray(camera, screen, i, j):
    pixel_center = screen.corner_pixel + i * screen.horizontal + j * screen.vertical
    ray_vector = pixel_center - camera.pos
    pixel_ray = Ray(camera.pos, ray_vector)
    return pixel_ray
    #towards = camera.look_at - camera.pos
    #up = camera.up_vector
    #right = np.cross(towards, up)
    #P0 = camera.pos
    #d = camera.screen_dist
    #w = camera.screen_width
    #h = camera.screen_height

    #P_left = P0 + d * towards - (w * right) / 2
    #P_down = P0 + d * towards - (h * up) / 2
    #P = P_left + (j/w + 0.5) * w * right + P_down + (i/h + 0.5) * h * up
    #V = (P-P0) / np.linalg.norm(P-P0)
    #return V


def calc_pixel_color(scene, ray, recursion_depth=0):
    if recursion_depth == scene.settings.max_recursion:
        return scene.settings.bg_color
    surface, min_intersect = find_min_intersect(scene, ray)
    output_color = calc_surface_color(scene, surface, min_intersect, recursion_depth)
    return output_color


def ray_tracing(scene, img_width, img_height, output_path):
    image = np.zeros((img_width, img_height, 3), dtype=float)
    screen = represent_screen(scene.camera, img_width, img_height)
    for i in range(img_width):
        for j in range(img_height):
            ray = construct_pixel_ray(scene.camera, screen, i, j)
            output_color = calc_pixel_color(scene, ray)
            #surface, min_intersect = find_min_intersect(scene, ray)
            #output_color = calc_surface_color(scene, surface, min_intersect)
            image[i, j] = output_color
    save_image(image, output_path)


def find_sphere_intersect(scene, ray, sphere):
    O = sphere.center_pos
    P0 = scene.camera.pos
    L = O - P0
    V = ray.direction
    r = sphere.radius
    r_squared = r**2

    tca = np.dot(L, V)
    if tca < 0:  # sphere is in the back
        return -1
    d_squared = np.dot(L, L) - tca**2
    if d_squared > r_squared:  # outside of sphere
        return -1
    thc = (r_squared-d_squared)**0.5
    t = tca - thc
    return t


def find_plane_intersect(scene, ray, plane):
    P0 = scene.camera.pos
    N = plane.normal_vector
    d = -1*plane.offset
    V = ray.direction
    t = -1 * (np.dot(P0, N) + d) / np.dot(V, N)
    return t


def find_box_intersect(scene, ray, box):
    pass  # TODO fill


def find_min_intersect(scene, ray):
    min_t = -1
    min_surface = None
    for sphere in scene.sphere_list:
        t = find_sphere_intersect(scene, ray, sphere)
        if min_t == -1 or 0 <= t <= min_t:
            min_t = t
            min_surface = sphere

    for plane in scene.plane_list:
        t = find_plane_intersect(scene, ray, plane)
        if min_t == -1 or 0 <= t <= min_t:
            min_t = t
            min_surface = plane

    for box in scene.box_list:
        t = find_box_intersect(scene, ray, box)
        if min_t == -1 or 0 <= t <= min_t:
            min_t = t
            min_surface = box

    min_intersect = ray.compute_point(min_t)
    return min_surface, min_intersect


def calc_surface_normal(surface, min_intersect):
    if type(surface) == Sphere:
        return np.array(min_intersect - surface.center_pos)

    elif type(surface) == Plane:
        return np.array(surface.normal)

    else:
        pass # TODO fill


def calc_soft_shadow_fraction(scene, N, light, min_intersect):
    light_ray_direct = min_intersect - light.pos
    light_ray_direct /= np.linalg.norm(light_ray_direct)

    # Create perpendicular plane x,y to ray
    x = np.array((1., 1., 1.))
    x -= (np.dot(x, light_ray_direct) * np.array(light_ray_direct)) # make it orthogonal to ray
    x /= np.linalg.norm(x)  # normalize x
    y = np.cross(light_ray_direct, x)

    # Create rectangle
    left_bottom_cell = light.pos - light.light_radius * x - light.light_radius * y

    # Normalize rectangle directions by cell size:
    cell_width = light.light_radius * 2 / N
    cell_height = light.light_radius * 2 / N
    x *= cell_width
    y *= cell_height
    intersect_counter = 0

    # Cast ray from cell to point and see if intersect with our point first
    for i in range(N):
        for j in range(N):
            cell_pos = left_bottom_cell + i * x + j * y
            cell_pos += random.random() * x + random.random() * y
            ray_direction = min_intersect - cell_pos
            cell_light_ray = Ray(light.pos, ray_direction / np.linalg.norm(ray_direction))
            cell_surface, cell_min_intersect = find_min_intersect(scene, cell_light_ray)
            if cell_min_intersect.all() == min_intersect.all():
                intersect_counter += 1
    return intersect_counter / (N * N)


def calc_surface_color(scene, surface, min_intersect, recursion_depth):
    bg_col = np.array(scene.settings.bg_color)
    trans_val = np.array((scene.material_list[surface.material_idx]).transparent_val)
    diffuse_col = np.array((scene.material_list[surface.material_idx]).diffuse_color)
    specular_col = np.array((scene.material_list[surface.material_idx]).specular_color)
    reflection_col = np.array((scene.material_list[surface.material_idx]).reflection_color)

    normal = calc_surface_normal(surface, min_intersect)
    normal /= np.linalg.norm(normal)
    for light in scene.light_list:
        # Calc light effect on diffuse color
        light_direction = min_intersect - light.pos
        soft_shadow_fraction = calc_soft_shadow_fraction(scene, scene.settings.N, light, min_intersect)
        light_intensity = (1 - light.shadow_intens) + (light.shadow_intens * soft_shadow_fraction)
        diffuse_col *= light_intensity * normal.dot(light_direction)

        # Calc light effect on specular color
        L = np.array(min_intersect - light.pos)
        R = L - 2 * (L.dot(normal)) * normal
        V = np.array(scene.camera.pos - min_intersect)
        surface_material = scene.material_list[surface.material_idx]
        n = surface_material.phong_coeff
        specular_col *= light_intensity * ((R.dot(V))**n) * light.specular_intens

        # Recursively calc reflection color
        reflection_ray = Ray(min_intersect, R)
        reflection_col *= calc_pixel_color(scene, reflection_ray, recursion_depth+1)

    output_color = bg_col * trans_val + (diffuse_col + specular_col) * (1 - trans_val) + reflection_col
    return output_color


def save_image(image, output_path):
    img = Image.fromarray(image)
    img.save(output_path)


def main(scene_path, output_path, img_width=500, img_height=500):
    aspect_ratio = img_height/img_width
    scene = parse_scene(scene_path, aspect_ratio)
    ray_tracing(scene, img_width, img_height, output_path)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])

    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
