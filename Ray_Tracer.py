import sys

def main(scene_path, output_path, img_width=500, img_height=500):
    scene = parse_scene(scene_path)


def parse_scene(scene_path):
    with open(scene_path, 'rb') as f:
        content = f.read()

    for line in content.splitlines():
        if len(line) == 0 or line[0] == '#':
            pass
        line = line.split()
        obj_name = line[0]



if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])

    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


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















