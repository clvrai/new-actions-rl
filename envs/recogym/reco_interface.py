from envs.action_env import get_aval_actions
from envs.action_env_interface import ActionEnvInterface
from envs.recogym.reco_env import get_env
import torch

class RecoInterface(ActionEnvInterface):
    def __init__(self, args):
        self.args = args

    def setup(self, args, task_id):
        super().setup(args, task_id)
        args.overall_aval_actions = get_aval_actions(args, 'reco')


    def env_trans_fn(self, env, set_eval):
        env = super()._generic_setup(env, set_eval)
        return env

    def get_gt_embs(self):
        env = get_env(self.args)
        if self.args.reco_include_mu:
            return torch.cat([torch.Tensor(env.beta), torch.Tensor(env.mu_bandit).unsqueeze(-1)], -1)
        else:
            return torch.Tensor(env.beta)

    def get_id(self):
        return 'RE'

    def get_special_stat_names(self):
        return ['ep_avg_ctr']

    def get_env_option_names(self):
        indv_labels = [('Train' if ind in self.train_action_set else 'Test') for ind in self.train_test_action_set]
        label_list = sorted(list(set(indv_labels)))
        return indv_labels, label_list
