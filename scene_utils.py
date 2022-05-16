import numpy as np


class Scene:
    __slots__ = ('camera', 'settings', 'sphere_list', 'plane_list', 'box_list', 'light_list', 'material_list')

    def __init__(self, camera, settings, sphere_list, plane_list, box_list, light_list, material_list):
        self.camera = camera
        self.settings = settings
        self.sphere_list = sphere_list
        self.plane_list = plane_list
        self.box_list = box_list
        self.light_list = light_list
        self.material_list = material_list


class Camera:
    __slots__ = ('pos', 'look_at', 'towards', 'up_vector', 'screen_dist', 'screen_width', 'screen_height')

    def __init__(self, pos, look_at, up_vector, screen_dist, screen_width, screen_height):
        self.pos = pos
        self.look_at = look_at
        self.towards = Vector(look_at - pos)
        self.up_vector = up_vector
        self.screen_dist = screen_dist
        self.screen_width = screen_width
        self.screen_height = screen_height


class Settings:
    __slots__ = ('bg_color', 'N', 'max_recursion')

    def __init__(self, bg_color, N, max_recursion):
        self.bg_color = bg_color
        self.N = N
        self.max_recursion = max_recursion


class Material:
    __slots__ = ('diffuse_color', 'specular_color', 'reflection_color', 'phong_coeff', 'transparent_val')

    def __init__(self, diffuse_color, specular_color, reflection_color, phong_coeff, transparent_val):
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.reflection_color = reflection_color
        self.phong_coeff = phong_coeff
        self.transparent_val = transparent_val


class Plane:
    __slots__ = ('normal_vector', 'offset', 'material_idx')

    def __init__(self, normal_vector, offset, material_idx):
        self.normal_vector = normal_vector
        self.offset = offset
        self.material_idx = material_idx


class Sphere:
    __slots__ = ('center_pos', 'radius', 'radius_squared', 'material_idx')

    def __init__(self, center_pos, radius, material_idx):
        self.center_pos = center_pos
        self.radius = radius
        self.radius_squared = radius * radius
        self.material_idx = material_idx


class Box:
    __slots__ = ('center_pos', 'edge_len', 'min_extent', 'max_extent', 'material_idx')

    def __init__(self, center_pos, edge_len, material_idx):
        self.center_pos = center_pos
        self.edge_len = edge_len
        self.min_extent = self.center_pos - (self.edge_len/2, self.edge_len/2, self.edge_len/2)
        self.max_extent = self.center_pos + (self.edge_len/2, self.edge_len/2, self.edge_len/2)
        self.material_idx = material_idx


class Light:
    __slots__ = ('pos', 'color', 'specular_intens', 'shadow_intens', 'light_radius')

    def __init__(self, pos, color, specular_intens, shadow_intens, light_radius):
        self.pos = pos
        self.color = color
        self.specular_intens = specular_intens
        self.shadow_intens = shadow_intens
        self.light_radius = light_radius


class Ray:
    __slots__ = ('orig', 'dir')

    def __init__(self, orig, dir):
        self.orig = orig
        self.dir = dir

    def at(self, distance):
        return self.orig + distance * self.dir


class Screen:
    __slots__ = ('corner_pixel', 'horizontal', 'vertical')

    def __init__(self, corner_pixel, horizontal, vertical):
        self.corner_pixel = corner_pixel
        self.horizontal = horizontal
        self.vertical = vertical


class Vector:
    __slots__ = ('dir')

    def __init__(self, dir):
        self.dir = dir / np.linalg.norm(dir)

    def perpendicular_vector(self):
        vector = np.cross(self.dir, np.array([1, 0, 0]))
        if (vector == 0).all():
            vector = np.cross(self.dir, np.array([0, 1, 0]))
        vector /= np.linalg.norm(vector)
        return vector


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
        if len(line) == 0:
            continue
        obj_name = line[0]

        if obj_name == b'cam':
            pos = np.array((float(line[1]), float(line[2]), float(line[3])), dtype=float)
            look_at = np.array((float(line[4]), float(line[5]), float(line[6])), dtype=float)
            up_vector = Vector(np.array((float(line[7]), float(line[8]), float(line[9])), dtype=float))
            screen_dist = float(line[10])
            screen_width = float(line[11])
            screen_height = screen_width * asp_ratio
            camera = Camera(pos, look_at, up_vector, screen_dist, screen_width, screen_height)

        elif obj_name == b'set':
            bg_color = np.array((float(line[1]), float(line[2]), float(line[3])), dtype=float)
            N = int(line[4])
            max_recursion = int(line[5])
            settings = Settings(bg_color, N, max_recursion)

        elif obj_name == b'lgt':
            pos = np.array((float(line[1]), float(line[2]), float(line[3])), dtype=float)
            color = np.array((float(line[4]), float(line[5]), float(line[6])), dtype=float)
            specular_intens = float(line[7])
            shadow_intens = float(line[8])
            light_radius = float(line[9])
            light_list.append(Light(pos, color, specular_intens, shadow_intens, light_radius))

        elif obj_name == b'sph':
            center_pos = np.array((float(line[1]), float(line[2]), float(line[3])), dtype=float)
            radius = float(line[4])
            material_idx = int(line[5]) - 1
            sphere_list.append(Sphere(center_pos, radius, material_idx))

        elif obj_name == b'pln':
            normal_vector = Vector(np.array((float(line[1]), float(line[2]), float(line[3])), dtype=float))
            offset = float(line[4])
            material_idx = int(line[5]) - 1
            plane_list.append(Plane(normal_vector, offset, material_idx))

        elif obj_name == b'box':
            center_pos = np.array((float(line[1]), float(line[2]), float(line[3])), dtype=float)
            edge_len = float(line[4])
            material_idx = int(line[5]) - 1
            box_list.append(Box(center_pos, edge_len, material_idx))

        elif obj_name == b'mtl':
            diffuse_col = np.array((float(line[1]), float(line[2]), float(line[3])), dtype=float)
            specular_col = np.array((float(line[4]), float(line[5]), float(line[6])), dtype=float)
            reflection_col = np.array((float(line[7]), float(line[8]), float(line[9])), dtype=float)
            phong_coeff = float(line[10])
            transparent_val = float(line[11])
            material_list.append(Material(diffuse_col, specular_col, reflection_col, phong_coeff, transparent_val))

    scene = Scene(camera, settings, sphere_list, plane_list, box_list, light_list, material_list)
    return scene


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