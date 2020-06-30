from envs.gym_minigrid.minigrid import *
from envs.gym_minigrid.register import register

import itertools as itt


class CrossingEnv(MiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(self, size=9, num_crossings=1, obstacle_type=Lava, seed=None):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        self.safe_wall_gen = False
        self.use_subgoal_reward = False
        super().__init__(
            grid_size=size,
            max_steps=size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None,
            use_subgoal_reward=False
        )

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (1, 1)
        self.start_dir = 0

        # Place a goal square in the bottom-right corner
        # self.grid.set(width - 2, height - 2, Goal())
        self.put_obj(Goal(), width - 2, height - 2)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        if self.safe_wall_gen:
            wall_offset = 3
            stride = 1
        else:
            wall_offset = 2
            stride = 2

        rivers = [(v, i) for i in range(wall_offset, height - wall_offset, stride)]
        rivers += [(h, j) for j in range(wall_offset, width - wall_offset, stride)]

        rng = np.random.RandomState(123) if self.fixed_rivers else self.np_random

        rng.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_v = sorted([pos for direction, pos in rivers if direction is v])
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)
            # self.grid.set(i, j, self.obstacle_type())

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        rng.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = rng.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1]))
                room_i += 1
            elif direction is v:
                i = rng.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1]))
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)
            if self.use_subgoal_reward:
                self.put_obj(SubGoal(), i, j)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

        # If using subgoals, initialize agent on top left
        if not self.use_subgoal_reward:
            self.place_agent()

class LavaCrossingEnv(CrossingEnv):
    def __init__(self):
        super().__init__(size=9, num_crossings=1)

class LavaCrossingS9N2Env(CrossingEnv):
    def __init__(self):
        super().__init__(size=9, num_crossings=2)

class LavaCrossingS9N3Env(CrossingEnv):
    def __init__(self):
        super().__init__(size=9, num_crossings=3)

class LavaCrossingS11N5Env(CrossingEnv):
    def __init__(self):
        super().__init__(size=11, num_crossings=5)

register(
    id='MiniGrid-LavaCrossingS9N1-v0',
    entry_point='envs.gym_minigrid.envs:LavaCrossingEnv'
)

register(
    id='MiniGrid-LavaCrossingS9N2-v0',
    entry_point='envs.gym_minigrid.envs:LavaCrossingS9N2Env'
)

register(
    id='MiniGrid-LavaCrossingS9N3-v0',
    entry_point='envs.gym_minigrid.envs:LavaCrossingS9N3Env'
)

register(
    id='MiniGrid-LavaCrossingS11N5-v0',
    entry_point='envs.gym_minigrid.envs:LavaCrossingS11N5Env'
)

class SimpleCrossingEnv(CrossingEnv):
    def __init__(self):
        super().__init__(size=9, num_crossings=1, obstacle_type=Wall)

class SimpleCrossingS9N2Env(CrossingEnv):
    def __init__(self):
        super().__init__(size=9, num_crossings=2, obstacle_type=Wall)

class SimpleCrossingS9N3Env(CrossingEnv):
    def __init__(self):
        super().__init__(size=9, num_crossings=3, obstacle_type=Wall)

class SimpleCrossingS11N5Env(CrossingEnv):
    def __init__(self):
        super().__init__(size=11, num_crossings=5, obstacle_type=Wall)

register(
    id='MiniGrid-SimpleCrossingS9N1-v0',
    entry_point='envs.gym_minigrid.envs:SimpleCrossingEnv'
)

register(
    id='MiniGrid-SimpleCrossingS9N2-v0',
    entry_point='envs.gym_minigrid.envs:SimpleCrossingS9N2Env'
)

register(
    id='MiniGrid-SimpleCrossingS9N3-v0',
    entry_point='envs.gym_minigrid.envs:SimpleCrossingS9N3Env'
)

register(
    id='MiniGrid-SimpleCrossingS11N5-v0',
    entry_point='envs.gym_minigrid.envs:SimpleCrossingS11N5Env'
)
