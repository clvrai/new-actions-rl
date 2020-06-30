from .fixed_obj import FixedObj
import pymunk as pm
import pygame as pg
from pygame import gfxdraw
import numpy as np
from ..constants import fixed_color, bouncy_color


class FixedPoly(FixedObj):
    def __init__(self, pos, n_sides=3, angle = 0.0, size=10.0, friction=1.0, elasticity = 0.5, color='slategray'):
        super().__init__(pos)
        mass = 1.0
        moment = pm.moment_for_box(mass, (size, size))
        self.body = self._create_body(mass, moment)
        self.body.position = pm.Vec2d(pos[0], pos[1])

        self.vertices = self.get_vertices(pm.Vec2d([0., 0.]), n_sides, angle, size)

        self.shape = pm.Poly(self.body, self.vertices)
        self.shape.friction = friction
        self.shape.elasticity = elasticity

        self.color = color
        self.size = size

    def get_body(self):
        return self.body

    def get_shape(self):
        return self.shape

    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        pointlist = []
        for v in self.shape.get_vertices():
            x, y = v.rotated(self.shape.body.angle) + self.shape.body.position
            point = scale * self.flipy([x, y])
            pointlist.append([int(point[0]), int(point[1])])

        if anti_alias:
            gfxdraw.filled_polygon(screen, pointlist, pg.Color(self.color))
            gfxdraw.aapolygon(screen, pointlist, pg.Color(self.color))
        else:
            pg.draw.polygon(screen, pg.Color(self.color), pointlist)


    def get_vertices(self, pos, n_vertices, angle, size):

        theta = angle
        vertices = []
        for point in range(n_vertices):
            theta += 2 * np.pi / n_vertices
            vertices.append([pos[0] + np.cos(theta) * size, pos[1] + np.sin(theta) * size])
        return vertices

# Star
class Star(FixedPoly):
     def get_vertices(self, pos, n_vertices, angle, size):
         theta = angle + np.pi / (2 * n_vertices)
         vertices = []
         for point in range(n_vertices):
             theta += 2 * np.pi / n_vertices
             vertices.append([pos[0] + np.cos(theta) * size, pos[1] + np.sin(theta) * size])
             vertices.append([pos[0] + np.cos(theta + np.pi / n_vertices) * size / 2, pos[1] + np.sin(theta + np.pi / n_vertices) * size / 2])
         return vertices

     def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1

        pointlist = []
        for v in self.vertices:
            x, y = pm.Vec2d(v).rotated(self.shape.body.angle) + self.shape.body.position
            point = scale * self.flipy([x, y])
            pointlist.append([int(point[0]), int(point[1])])

        if anti_alias:
            gfxdraw.filled_polygon(screen, pointlist, pg.Color(self.color))
            gfxdraw.aapolygon(screen, pointlist, pg.Color(self.color))
        else:
            pg.draw.polygon(screen, pg.Color(self.color), pointlist)


# Triangle
class FixedTriangle(FixedPoly):
    def __init__(self, pos, angle = 0.0, size=10.0, friction=1.0, elasticity = 0.5,
        color=fixed_color):
        super().__init__(pos, n_sides=3, angle=angle, size=size,
            elasticity=elasticity, color=color)



class BouncyTriangle(FixedPoly):
    def __init__(self, pos, angle = 0.0, size=10.0, friction=1.0, elasticity=1.2,
        color=bouncy_color):
        super().__init__(pos, n_sides=3, angle=angle, size=size,
            elasticity=elasticity, color=color)




# Square
class FixedSquare(FixedPoly):
    def __init__(self, pos, angle = 0.0, size=10.0, friction=1.0, elasticity = 0.5,
        color=fixed_color):
        super().__init__(pos, n_sides=4, angle=angle, size=size,
            elasticity=elasticity, color=color)


class BouncySquare(FixedPoly):
    def __init__(self, pos, angle = 0.0, size=10.0, friction=1.0, elasticity=1.2,
        color=bouncy_color):
        super().__init__(pos, n_sides=4, angle=angle, size=size,
            elasticity=elasticity, color=color)



# Pentagon
class FixedPentagon(FixedPoly):
    def __init__(self, pos, angle = 0.0, size=10.0, friction=1.0, elasticity = 0.5,
        color=fixed_color):
        super().__init__(pos, n_sides=5, angle=angle, size=size,
            elasticity=elasticity, color=color)


class BouncyPentagon(FixedPoly):
    def __init__(self, pos, angle = 0.0, size=10.0, friction=1.0, elasticity=1.2,
        color=bouncy_color):
        super().__init__(pos, n_sides=5, angle=angle, size=size,
            elasticity=elasticity, color=color)



# Hexagon
class FixedHexagon(FixedPoly):
    def __init__(self, pos, angle = 0.0, size=10.0, friction=1.0, elasticity = 0.5,
        color=fixed_color):
        super().__init__(pos, n_sides=6, angle=angle, size=size,
            elasticity=elasticity, color=color)


class BouncyHexagon(FixedPoly):
    def __init__(self, pos, angle = 0.0, size=10.0, friction=1.0, elasticity=1.2,
        color=bouncy_color):
        super().__init__(pos, n_sides=6, angle=angle, size=size,
            elasticity=elasticity, color=color)

