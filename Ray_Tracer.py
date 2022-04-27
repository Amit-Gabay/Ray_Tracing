import sys


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
    def __init__(self, pos, look_up, up_vector, screen_dist, screen_width):
        self.pos = pos
        self.look_up = look_up
        self.up_vector = up_vector
        self.screen_dist = screen_dist
        self.screen_width = screen_width


class Settings:
    def __init__(self, bg_color, N, max_recursion):
        self.bg_color = bg_color
        self.N = N
        self.max_recursion = max_recursion


class Material:
    def __init__(self, diffuse_col, specular_col, reflection_col, phong_coeff, transparent_val):
        self.diffuse_col = diffuse_col
        self.specular_col = specular_col
        self.reflection_col = reflection_col
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


def parse_scene(scene_path):
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
            pos = (float(line[1]), float(line[2]), float(line[3]))
            look_up = (float(line[4]), float(line[5]), float(line[6]))
            up_vector = (float(line[7]), float(line[8]), float(line[9]))
            screen_dist = float(line[10])
            screen_width = float(line[11])
            camera = Camera(pos, look_up, up_vector, screen_dist, screen_width)

        elif obj_name == b'set':
            bg_color = (float(line[1]), float(line[2]), float(line[3]))
            N = float(line[4])
            max_recursion = float(line[5])
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
            material_idx = float(line[5])
            sphere_list.append(Sphere(center_pos, radius, material_idx))

        elif obj_name == b'pln':
            normal_vector = (float(line[1]), float(line[2]), float(line[3]))
            offset = float(line[4])
            material_idx = float(line[5])
            plane_list.append(Plane(normal_vector, offset, material_idx))

        elif obj_name == b'box':
            center_pos = (float(line[1]), float(line[2]), float(line[3]))
            edge_len = float(line[4])
            material_idx = float(line[5])
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


def main(scene_path, output_path, img_width=500, img_height=500):
    scene = parse_scene(scene_path)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])

    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])