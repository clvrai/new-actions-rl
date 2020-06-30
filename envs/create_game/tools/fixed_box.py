from .fixed_obj import FixedObj
import pymunk
import pygame as pg
from .img_tool import ImageTool

class FixedBox(FixedObj):
    def __init__(self, pos, size=10.0, friction=1.0, elasticity = 0.4, color='slategray'):
        super().__init__(pos)
        mass = 1.0
        moment = pymunk.moment_for_box(mass, (size, size))
        self.body = self._create_body(mass, moment)
        self.body.position = pymunk.Vec2d(pos[0], pos[1])

        self.shape = pymunk.Poly.create_box(self.body, (size, size))
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

        draw_pos = scale * self.flipy(self.body.position)

        draw_size = scale * self.size
        draw_rect = (draw_pos[0] - (draw_size / 2),
            draw_pos[1] - (draw_size / 2),
            draw_size,
            draw_size)

        pg.draw.rect(screen, pg.Color(self.color), draw_rect)


class BouncyBox(FixedBox):
    def __init__(self, pos, size=10.0, friction=1.0, elasticity=1.2, color='blue'):
        super().__init__(pos, size=size, friction=friction, elasticity=elasticity, color=color)



class FixedRect(FixedObj):
     def __init__(self, pos, angle = 0.0, width=10.0, height=10.0, friction=1.0, elasticity = 0.4, color='slategray'):
         super().__init__(pos)
         mass = 1.0
         moment = pymunk.moment_for_box(mass, (width, height))
         self.body = self._create_body(mass, moment)
         self.body.position = pymunk.Vec2d(pos[0], pos[1])

         self.shape = pymunk.Poly.create_box(self.body, (width, height))
         self.shape.body.angle = angle
         self.shape.friction = friction
         self.shape.elasticity = elasticity
         self.color = color
         self.width = width
         self.height = height

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
