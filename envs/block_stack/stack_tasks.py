from envs.block_stack.stack_env_simplest import StackEnvSimplest
from envs.block_stack.stack_env_simplest_all import StackEnvAll
import numpy as np


class RndShapes(StackEnvSimplest):
    def get_place_polys(self):
        return [
                (np.random.randint(len(self.all_polygons)), np.random.uniform(-0.6, 0.6))
                ]



class TaskAShapes(StackEnvSimplest):
    def get_place_polys(self):
        return [
                (643, 0.6),
                (656, -0.9)
                ]


class MovingShapes(StackEnvSimplest):
    def get_place_polys(self):
        x = np.random.uniform(-0.2, 0.2)
        fixed_gap = np.random.uniform(1.0, 1.6)
        cylinder = np.random.choice([798, 799, 800])
        return [
                (cylinder, x - fixed_gap / 2),
                (cylinder, x + fixed_gap / 2)
                ]

class MovingShapesAll(StackEnvAll):
    def get_place_polys(self):
        x = np.random.uniform(-0.2, 0.2)
        fixed_gap = np.random.uniform(1.0, 1.6)
        cylinder = np.random.choice([798, 799, 800])
        return [
                (cylinder, x - fixed_gap / 2),
                (cylinder, x + fixed_gap / 2)
                ]
