from gym.envs.registration import register
from rlf import register_env_interface
from envs.recogym.reco_interface import RecoInterface

register_env_interface('^RecoEnv-v0$', RecoInterface)
register(
    id = 'RecoEnv-v0',
    entry_point = 'envs.recogym.reco_env:RecoEnv'
)

