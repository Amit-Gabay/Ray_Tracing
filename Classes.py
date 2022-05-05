class Scene:
    def __init__(self, camera, settings, sphere_list, plane_list, box_list, light_list, material_list):
        self.camera = camera
        self.settings = settings
        self.sphere_list = sphere_list
        self.plane_list = plane_list
        self.box_list = box_list
        self.light_list = light_list
        self.material_list = material_list


class Camera:
    def __init__(self, pos, look_at, up_vector, screen_dist, screen_width, screen_height):
        self.pos = pos
        self.look_at = look_at
        self.towards = look_at - pos
        self.up_vector = up_vector
        self.screen_dist = screen_dist
        self.screen_width = screen_width
        self.screen_height = screen_height


class Settings:
    def __init__(self, bg_color, N, max_recursion):
        self.bg_color = bg_color
        self.N = N
        self.max_recursion = max_recursion


class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, phong_coeff, transparent_val):
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.reflection_color = reflection_color
        self.phong_coeff = phong_coeff
        self.transparent_val = transparent_val


class Plane:
    def __init__(self, normal_vector, offset, material_idx):
        self.normal_vector = normal_vector
        self.offset = offset
        self.material_idx = material_idx


class Sphere:
    def __init__(self, center_pos, radius, material_idx):
        self.center_pos = center_pos
        self.radius = radius
        self.material_idx = material_idx


class Box:
    def __init__(self, center_pos, edge_len, material_idx):
        self.center_pos = center_pos
        self.edge_len = edge_len
        self.material_idx = material_idx


class Light:
    def __init__(self, pos, color, specular_intens, shadow_intens, light_radius):
        self.pos = pos
        self.color = color
        self.specular_intens = specular_intens
        self.shadow_intens = shadow_intens
        self.light_radius = light_radius


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def compute_point(self, scalar):
        return self.origin + scalar * self.direction


class Screen:
    def __init__(self, corner_pixel, horizontal, vertical):
        self.corner_pixel = corner_pixel
        self.horizontal = horizontal
        self.vertical = vertical
