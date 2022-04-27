import sys

def main(scene_path, output_path, img_width=500, img_height=500):
    scene = parse_scene(scene_path)


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
        if len(line) == 0 or line[0] == '#':
            pass
        line = line.split()
        obj_name = line[0]

        if obj_name == 'cam':
            pos = (line[1], line[2], line[3])
            look_up = (line[4], line[5], line[6])
            up_vector = (line[7], line[8], line[9])
            screen_dist = line[10]
            screen_width = line[11]
            camera = Camera(pos, look_up, up_vector, screen_dist, screen_width)

        elif obj_name == 'set':
            bg_color = (line[1], line[2], line[3])
            N = line[4]
            max_recursion = line[5]
            settings = Settings(bg_color, N, max_recursion)

        elif obj_name == 'lgt':
            pos = (line[1], line[2], line[3])
            color = (line[4], line[5], line[6])
            specular_intens = line[7]
            shadow_intens = line[8]
            light_radius = line[9]
            light_list.append(Light(pos, color, specular_intens, shadow_intens, light_radius))

        elif obj_name == 'sph':
            center_pos = (line[1], line[2], line[3])
            radius = line[4]
            material_idx = line[5]
            sphere_list.append(Sphere(center_pos, radius, material_idx))

        elif obj_name == 'pln':
            normal_vector = (line[1], line[2], line[3])
            offset = line[4]
            material_idx = line[5]
            plane_list.append(Plane(normal_vector, offset, material_idx))

        elif obj_name == 'box':
            center_pos = (line[1], line[2], line[3])
            edge_len = line[4]
            material_idx = line[5]
            box_list.append(Box(center_pos, edge_len, material_idx))

        elif obj_name == 'mtl':
            diffuse_col = (line[1], line[2], line[3])
            specular_col = (line[4], line[5], line[6])
            reflection_col = (line[7], line[8], line[9])
            phong_coeff = line[10]
            transparent_val = line[11]
            material_list.append(Material(diffuse_col, specular_col, reflection_col, phong_coeff, transparent_val))

    scene = Scene(camera, settings, sphere_list, plane_list, box_list, light_list, material_list)
    return scene


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])

    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


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















