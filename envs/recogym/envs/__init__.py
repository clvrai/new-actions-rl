from .reco_env_v0 import RecoEnv0
from .reco_env_v1 import RecoEnv1

from .observation import Observation
from .configuration import Configuration
from .context import Context, DefaultContext
from .session import Session

from .reco_env_v0 import env_0_args
from .reco_env_v1 import env_1_args
from gym.envs.registration import register

register(
    id = 'reco-gym-v0',
    entry_point = 'envs.recogym.envs:RecoEnv0'
)

register(
    id = 'reco-gym-v1',
    entry_point = 'envs.recogym.envs:RecoEnv1'
)
