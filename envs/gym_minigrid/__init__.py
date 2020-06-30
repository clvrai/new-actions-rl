# Import the envs module so that envs register themselves
import envs.gym_minigrid.envs

# Import wrappers so it's accessible when installing with pip
import envs.gym_minigrid.wrappers

from rlf import register_env_interface
from envs.gym_minigrid.minigrid_interface import MiniGridInterface


register_env_interface('^MiniGrid(.*?)$', MiniGridInterface)
