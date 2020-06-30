from .basic_obj import BasicObj
from pymunk import Body


MOVING_OBJ_COLLISION_TYPE = 1

class GravityObj(BasicObj):
    def __init__(self, pos):
        super().__init__(pos)
        self.collision_type = MOVING_OBJ_COLLISION_TYPE

    def _create_body(self, mass, inertia):
        return Body(mass, inertia)
