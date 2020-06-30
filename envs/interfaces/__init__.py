
from gym.envs.registration import register
from rlf import register_env_interface

from envs.interfaces.create_env_interface import CreateGameInterface, CreatePlayInterface

register_env_interface('^Create((?!Play).)*$', CreateGameInterface)
register_env_interface('^Create(.*?)Play(.*)?$', CreatePlayInterface)
register_env_interface('^StateCreate(.*?)Play(.*)?$', CreatePlayInterface)

register(
    id='CreateGamePlay-v0',
    entry_point='envs.interfaces.create_play:CreatePlay',
)

register(
    id='StateCreateGamePlay-v0',
    entry_point='envs.interfaces.create_play:StateCreatePlay',
)