from .basic_obj import BasicObj
from pymunk import Body


class FixedObj(BasicObj):
    def __init__(self, pos):
        super().__init__(pos)

    def _create_body(self, mass, inertia):
        return Body(mass, inertia, Body.STATIC)
