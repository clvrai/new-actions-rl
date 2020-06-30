from .segment import Segment, LINE_THICKNESS
from .floor import ELASTICITY
import pygame as pg
import numpy as np

class Wall(Segment):
    def __init__(self, pos, length=16.0, elasticity=ELASTICITY, color='salmon', friction=1.0, sensor=False, thickness=None):
        start_y = pos[1] - (length/2)
        end_y = pos[1] + (length/2)
        self.sensor = sensor

        super().__init__([pos[0], start_y], [pos[0], end_y], pos, friction=friction,
            elasticity=elasticity, color=color, thickness=thickness)

    def add_to_space(self, space, use_friction=True):
        super().add_to_space(space, use_friction=use_friction)
        self.shape.sensor = self.sensor


class WallElastic(Wall):
    def __init__(self, pos, thickness=None, color='pink'):
        super().__init__(pos, length=16, elasticity=0.8, color=color, thickness=thickness)
