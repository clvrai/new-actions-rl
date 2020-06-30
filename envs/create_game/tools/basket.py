from pymunk import Body
import pymunk
import pygame as pg
import numpy as np
from .img_tool import ImageTool
from .fixed_box import FixedBox
from .gravity_obj import MOVING_OBJ_COLLISION_TYPE
from .goal import GOAL_OBJ_COLLISION_TYPE, goal_target_begin_handler
from .basic_obj import BasicObj, LINE_THICKNESS
from .gravity_obj import MOVING_OBJ_COLLISION_TYPE

class Basket(FixedBox):
    def __init__(self, pos, size=10.0, color='black'):
        super().__init__(pos, size=size)
        self.pos = pos
        self.size = size
        self.collision_type = GOAL_OBJ_COLLISION_TYPE


    def add_to_space(self, space):
        super().add_to_space(space)
        self.img = ImageTool('basket.png', 0.0,
                self.pos[:],
                use_shape=self.shape,
                debug_render=False)
        basket = self.img.get_shape()
        basket.sensor=True
        basket.collision_type = GOAL_OBJ_COLLISION_TYPE
        basket.is_goal = True
        h = space.add_collision_handler(GOAL_OBJ_COLLISION_TYPE, MOVING_OBJ_COLLISION_TYPE)
        h.begin = goal_target_begin_handler

    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        self.img.render(screen, scale, self.flipy)
