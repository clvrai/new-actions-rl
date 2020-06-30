import gym
from envs.recogym.envs import env_1_args
import envs.recogym.envs
import gym.spaces as spaces
import os.path as osp
import numpy as np

from envs.action_env import ActionEnv


PROD_GEN_SEED = 41

def get_env(args):
    env_1_args['num_products'] = args.reco_n_prods  # 10,000 by default
    env_1_args['K'] = args.reco_prod_dim
    env_1_args['reco_max_steps'] = args.reco_max_steps  # 100 by default
    env_1_args['random_seed'] = PROD_GEN_SEED   # This will be reset externally by gym env.set_seed()

    env_1_args['normalize_beta'] = args.reco_normalize_beta # True by default
    env_1_args['number_of_flips'] = args.reco_num_flips # 1000 by default
    env_1_args['change_omega_for_bandits'] = args.reco_change_omega
    env_1_args['normal_time_generator'] = args.reco_normal_time_generator

    env_1_args['deterministic_env'] = args.reco_deterministic
    env_1_args['random_product_view'] = args.reco_random_product_view

    env = gym.make('reco-gym-v1')
    env.init_gym(env_1_args)
    return env

class RecoEnv(ActionEnv):
    def __init__(self):
        super().__init__()
        self.env = None
        self.debug_log = False
        self.env_args = None
        self.last_viewed_item_idx = None
        self.use_items = None
        self.set_seed = None
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                shape=(1,), dtype=np.float32)
        self.count = 0
        self.rewarded_actions = []

    def set_args(self, args, set_eval):
        # Set up env
        self.env = get_env(args)

        self.observation_space = spaces.Box(low=0.0, high=1.0,
                shape=(args.reco_prod_dim,), dtype=np.float32)
        super().set_args(args, set_eval)

        if self.set_seed is not None:
            self.env.reset_random_seed(self.set_seed)

    def _set_action_space(self, sub_split):
        super()._set_action_space(sub_split)
        self.env.aval_idx = self.aval_idx

    def get_env_name(self):
        return 'reco'


    def seed(self, seed_val):
        if self.env is not None:
            self.env.reset_random_seed(seed_val)
        else:
            self.set_seed = seed_val

    def step(self, a):
        if len(a.shape) > 0:
            a = a[0]

        obs, reward, done, _ = self.env.step(a)

        if a in self.rewarded_actions and self.args.reco_no_repeat:
            reward = 0.0

        obs = self._extract_obs(obs, reward)
        if reward != 0.0:
            self._log('Got reward: %.2f' % reward)
        self.did_clicks.append(reward)

        info = {
                'aval': self.aval_idx,
                }
        self.count += 1
        if done:
            info['ep_len'] = self.count
            info['ep_avg_ctr'] = np.mean(self.did_clicks)
            self.count = 0

        if reward > 0:
            self.rewarded_actions.append(a)

        return obs, reward, done, info

    def _get_item_repr(self, prod_i):
        item_repr = self.env.beta[prod_i]
        return item_repr

    def _extract_obs(self, obs, reward):
        cur_user_repr = self.env.omega[:, 0]

        # # This can be used to use the last (or entire list of) viewed item representations
        # sess = obs.current_sessions
        # if len(sess) > 0:
        #    self.last_viewed_item_idx = sess[-1]['v']
        #    last_item_repr = self._get_item_repr(self.last_viewed_item_idx)
        norm_user_repr = cur_user_repr / np.sqrt((cur_user_repr ** 2).sum())
        return norm_user_repr

    def reset(self):
        super().reset()
        self._log('Reset')
        self.did_clicks = []
        self.env.reset()
        self.rewarded_actions = []
        obs, reward, done, info = self.env.step(None)
        return self._extract_obs(obs, 0.0)

    def render(self, mode='rgb_array'):
        # Nothing to render now
        return [0]

    def set_fixed_action_space(self, args, action_set):
        sub_split = np.copy(action_set)
        if args.reco_special_fixed_action_set_size is not None:
            rng = np.random.RandomState(42)
            sub_split = rng.choice(sub_split,
                args.reco_special_fixed_action_set_size,
                replace=False)
        return sub_split
