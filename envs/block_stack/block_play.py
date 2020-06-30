import gym
import os.path as osp
import numpy as np
import gym.spaces as spaces
from envs.block_stack.stack_xml import XML
import mujoco_py as mjc

from envs.block_stack.stack_env import SMALL_RENDER_DIM
from envs.block_stack.poly_gen import gen_polys


class BlockPlay(gym.Env):
    def __init__(self, asset_path='assets/stl'):
        self.asset_path = osp.join(osp.dirname(
            osp.realpath(__file__)), asset_path)
        self.all_polygons, self.shape_names, self.polygon_types = gen_polys(self.asset_path)
        self.observation_space = spaces.Box(low=0.0, high=255.0,
                                            shape=(SMALL_RENDER_DIM, SMALL_RENDER_DIM, 3), dtype=np.float32)

    def update_args(self, args):
        new_dir = osp.join(args.action_seg_loc, 'stack_%s' % args.exp_type)

        self.polys = self.all_polygons

        self.action_space = spaces.Discrete(len(self.polys))

        self.rngs = {}

        for i in range(len(self.polys)):
            self.rngs[i] = np.random.RandomState(42)

    def seed(self, seed_val):
        np.random.seed(seed_val)

class BlockPlayImg(BlockPlay):
    def __init__(self):
        super().__init__()

    def step(self, a):
        rng = self.rngs[a]
        # point camera at [0, 0, 0] with random position around circle.
        z_pos = rng.uniform(0.5, 2.0)
        radius = 2.0
        angle = rng.uniform(0, 2 * np.pi)

        pos = [radius * np.cos(angle), radius * np.sin(angle), z_pos]

        xml = XML(self.asset_path, self.shape_names, cam_pos=pos,
                no_ground=True)

        poly = self.polys[a]
        spawn_pos = [0, 0, 0]

        xml.add_mesh(poly.ply, pos=spawn_pos, axangle=poly.angle,
                     scale=poly.scale, rgba=poly.rgba)

        xml_str = xml.instantiate()
        model = mjc.load_model_from_xml(xml_str)
        self.sim = mjc.MjSim(model)
        self.sim.step()

        render_dim = SMALL_RENDER_DIM
        image = self.sim.render(render_dim, render_dim, camera_name='custom', depth=False)
        image = np.flipud(image)
        image = image.astype(np.float32)
        # Not subtracting 0.5 here because I forgot when I initially generated
        # the data. Instead, the postprocessing step is done in `option.py`.
        image = (image / 255.0)

        info = {
            # No post-processing will be applied to the states
            'states': image,
            # Just a garbage value
            'actions': [0]
            }
        return image, 0.0, False, info

    def reset(self):
        return np.zeros(self.observation_space.shape)

    def render(self):
        pass
