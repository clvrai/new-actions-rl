import os
import envs.block_stack.utils as utils
import numpy as np
from collections import defaultdict

FREEZE_Z = -10.0
SCALE_MODIFIER = 0.8

class EnvShape(object):
    def __init__(self, ply, angle, scale, rgba, pos, subtract_dist=0.1):
        self.ply = ply
        self.angle = angle
        self.scale = scale
        self.rgba = rgba
        self.pos = pos
        self.type = ply
        self.subtract_dist = subtract_dist * scale     # Safe distance to subtract from its bounding


ply_variations = {
    'middle_triangle': (0.1, 0.5, 6, 0., np.pi, 10),
    'hat':  (0.1, 0.5, 6, 0., np.pi, 10),
    'triangle':  (0.1, 0.5, 6, 0., 2 * np.pi, 10),
    'tall_triangle':  (0.1, 0.3, 6, 0., 2 * np.pi, 10),
    'horizontal_rectangle':  (0.1, 0.5, 6, 0., np.pi, 10),
    'dome': (0.2, 0.6, 6, 0., np.pi, 10),
    'cone-hires': (0.1, 0.6, 30, 0., np.pi, 1), # 30
    'cone': (0.1, 0.6, 30, 0., np.pi, 1), # 30
    'tetrahedron': (0.1, 0.6, 10, 0., np.pi / 4, 6),
    'arch':  (0.1, 0.5, 6, 0., np.pi, 10),
    'rectangle':  (0.1, 0.3, 10, 0., np.pi, 6),
    'half_rectangle':  (0.1, 0.6, 10, 0., np.pi, 6),
    'cube': (0.1, 0.6, 10, 0., np.pi / 4, 6),
    'sphere': (0.1, 0.8, 30, 0., np.pi, 1), # 30
    'capsule': (0.1, 0.4, 30, 0., np.pi, 1), # 30
    'cylinder': (0.1, 0.6, 30, 0., np.pi, 1), # 30
}


def gen_polys(asset_path):
    stl_files = os.listdir(asset_path)
    polygons = [f.split('.')[0] for f in stl_files]
    shape_names = ['sphere', 'capsule', 'cylinder']
    polygons.extend(shape_names)

    bounds = {
        'pos':   [[-.5, .5], [-.5, 0]],
        'hsv': [[0, 1], [0.5, 1], [0.5, 1]],
        'scale': [0.1, 0.3],
        'force': [[0, 0], [0, 0], [0, 0]]
    }

    # This gives 900 shapes in total. This number is then divided in two
    # between train and test.

    # So we generate the same set of things every time.
    rng = np.random.RandomState(42)
    shape_count = 0
    axis = [0,0,1]

    all_polygons = []
    for ply in polygons:
        min_scale, max_scale, num_scale, min_angle, max_angle, num_angle = ply_variations[ply]
        for scale in np.linspace(min_scale, max_scale, num_scale, endpoint=True):
            for angle in np.linspace(min_angle, max_angle, num_angle, endpoint=False):

                axangle = utils.fixed_axangle(theta=angle, axis=axis)
                rgba = utils.sample_rgba_from_hsv(rng, *bounds['hsv'])
                freeze_z = FREEZE_Z * (shape_count + 1)
                rnd_pos = utils.uniform(rng, *bounds['pos'])
                mod_scale = scale * SCALE_MODIFIER
                all_polygons.append(EnvShape(ply, axangle, mod_scale, rgba,
                                             [*rnd_pos, freeze_z]))
                shape_count += 1

    return all_polygons, shape_names, polygons


def rnd_train_test_split(all_polygons, polygon_types):
    train_shapes = []
    test_shapes = []
    for ply in polygon_types:
        min_scale, max_scale, num_scale, min_angle, max_angle, num_angle = ply_variations[ply]
        scale_vals = np.linspace(min_scale, max_scale, num_scale, endpoint=True)
        scale_vals = scale_vals * SCALE_MODIFIER
        np.random.shuffle(scale_vals)
        mid_ind = (len(scale_vals) + 1) // 2

        # Randomly split each group.
        train_vals = scale_vals[:mid_ind]
        test_vals = scale_vals[mid_ind:]

        for i, shape in enumerate(all_polygons):
            if shape.ply == ply:
                if any([abs(shape.scale - y) < 1e-5 for y in train_vals]):
                    train_shapes.append(i)
                elif any([abs(shape.scale - y) < 1e-5 for y in test_vals]):
                    test_shapes.append(i)

    print('# train tools', len(train_shapes))
    print('# test tools', len(test_shapes))
    return train_shapes, test_shapes


def full_train_test_split(polys):
    poly_groups = defaultdict(list)

    for i, p in enumerate(polys):
        poly_groups[p.ply].append(i)

    train_names = [
        'dome',
        'rectangle',
        'half_rectangle',
        'capsule',
        'triangle',
        'arch',
        'tall_triangle',
        'sphere',
        ]

    test_names = [
        'hat',
        'cylinder',
        'tetrahedron',
        'horizontal_rectangle',
        'middle_triangle',
        'cube',
        'cone-hires',
        'cone',
        ]

    train_polys = [i for name in train_names for i in poly_groups[name] ]
    test_polys = [i for name in test_names for i in poly_groups[name] ]

    return train_polys, test_polys
