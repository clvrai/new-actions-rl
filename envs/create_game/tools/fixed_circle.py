from .fixed_obj import FixedObj
import pymunk
import pygame as pg
from pygame import gfxdraw



class FixedCircle(FixedObj):
    def __init__(self, pos, radius=5.0, elasticity=0.4, color='blue'):
        super().__init__(pos)
        mass = 1.0
        self.radius = radius
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
        self.body = self._create_body(mass, inertia)
        self.body.position = self.pos
        self.shape = pymunk.Circle(self.body, radius, pymunk.Vec2d(0,0))
        self.elasticity = elasticity
        self.shape.elasticity = elasticity
        # Default friction is 0
        self.color = color


    def get_body(self):
        return self.body

    def get_shape(self):
        return self.shape

    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1
        draw_pos = scale * self.flipy(self.pos)
        draw_pos[0] = int(draw_pos[0])
        draw_pos[1] = int(draw_pos[1])

        if anti_alias:
            gfxdraw.filled_circle(screen, draw_pos[0], draw_pos[1], int(self.radius * scale), pg.Color(self.color))
            gfxdraw.aacircle(screen, draw_pos[0], draw_pos[1], int(self.radius * scale), pg.Color(self.color))
        else:
            pg.draw.circle(screen, pg.Color(self.color), draw_pos,
                int(self.radius * scale))


class BouncyCircle(FixedCircle):
    def __init__(self, pos, radius=5.0, elasticity=1.2, color='blue'):
        super().__init__(pos, radius=radius, elasticity=elasticity, color=color)
