from .basic_obj import BasicObj
import pymunk
import pygame as pg

BOX_SIZE = 5.0

class Puncher(BasicObj):
    def __init__(self, pos, length=10.0, color='black'):
        super().__init__(pos)

        start_pos = pos
        end_pos = pos[:]
        end_pos[0] -= BOX_SIZE

        pop_end_pos = end_pos[:]
        pop_end_pos[0] -= length

        self.start_pos = start_pos
        self.end_pos = end_pos
        self.color = color
        self.pop_end_pos = pop_end_pos
        self.cur_iter = 0

    def add_to_space(self, space):
        self.space = space
        orig_box = pymunk.Segment(space.static_body, self.start_pos, self.end_pos,
                radius=BOX_SIZE)
        self.shape = orig_box
        space.add(orig_box)
        self.attached_shapes.append(orig_box)


    def activate(self):
        print('Activating')
        new_shape = pymunk.Segment(self.space.static_body, self.start_pos,
                self.pop_end_pos,
                radius=BOX_SIZE)
        self.space.add(new_shape)
        #self.shape.unsafe_set_endpoints(self.start_pos, self.pop_end_pos)
        self.end_pos = self.pop_end_pos


    def render(self, screen, scale=None, anti_alias=False):
        if scale is None:
            scale = 1
        p1 = scale * self.flipy(self.start_pos)
        p2 = scale * self.flipy(self.end_pos)
        pg.draw.lines(screen, pg.Color(self.color), False, (p1,
            p2), int(scale * BOX_SIZE))

        self.cur_iter += 1
        if self.cur_iter == 10:
            self.activate()

