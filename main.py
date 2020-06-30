import copy
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from method.dist_mem import DistributionMemory
from method.emb_mem import EmbeddingMemory
from method.embedder.embedder import extract_dists
from method.models.main_method import MainMethod
from method.models.distance_method import DistanceMethod

import pymunkoptions
pymunkoptions.options["debug"] = False

from arguments import get_args
from rlf import get_env_interface, RunSettings, run_policy

import envs.recogym
import envs.block_stack
import envs.interfaces
import envs.create_game

class ExpRunSettings(RunSettings):
    def __init__(self):
        self.args = get_args()
        self.set_args = copy.copy(self.args)


    def init(self):
        emb_mem, dist_mem = get_embs(self.args, self)

        self.args.emb_mem = emb_mem
        self.args.dist_mem = dist_mem

    def get_args(self):
        return self.args

    def get_policy_type(self):
        if args.distance_based:
            return DistanceMethod
        else:
            return MainMethod

    def get_test_args(self):
        test_args = copy.copy(self.args)
        test_args.eval_split = True
        test_args.test_split = False
        test_args.train_split = False
        test_args.sample_clusters = False

        if test_args.env_name.startswith('Create') and test_args.split_type == 'full_clean':
            test_args.gt_clusters = True
            test_args.half_tools = True
            test_args.half_tool_ratio = 0.5
        else:
            test_args.gt_clusters = False

        test_args.action_random_sample = True
        test_args.action_set_size = self.args.test_action_set_size if self.args.test_action_set_size is not None else self.args.action_set_size

        if self.args.fixed_action_set:
            # This converts discrete to Nearest Neighbor during evaluation
            test_args.fixed_action_set = False
            test_args.load_fixed_action_set = self.args.nearest_neighbor
            test_args.training_action_set_size = None

        self.test_args = test_args
        return self.test_args


    def get_set_args(self):
        return self.set_args

    def get_env_trans_fn(self, args, task_id=None):
        env_interface = get_env_interface(args.env_name)(args)
        env_interface.setup(args, task_id)
        args.env_interface = env_interface
        return env_interface.env_trans_fn


    def get_train_env_trans(self):
        return self.get_env_trans_fn(self.args)

    def get_test_env_trans(self):
        return self.get_env_trans_fn(self.test_args)




def get_embs(args, run_settings):
    emb_mem = EmbeddingMemory(cuda=args.cuda, args=args)
    # exec_fns = []
    if args.env_name.startswith('Create'):
        emb_mem.extract_disc = True

    dist_mem = DistributionMemory(args.cuda,
                                  args.n_distributions, args)

    # We must load the play data set
    copy_args = copy.copy(args)
    copy_args.both_train_test = True
    copy_args.eval_only = False

    if not args.gt_embs and not args.use_random_embeddings:
        copy_args.env_name = get_env_interface(args.env_name)(args).get_play_env_name()

    env_trans_fn = run_settings.get_env_trans_fn(copy_args)
    extract_dists(copy_args, dist_mem, emb_mem,
            env_trans_fn,
            args.env_name, load_all=True)

    if emb_mem.mem_keys is not None:
        dist_mem.store_embs(emb_mem.mem_keys)

    if args.gt_embs:
        env_interface = get_env_interface(args.env_name)(args)
        gt_embs = env_interface.get_gt_embs()
        dist_mem.load_gt(args.env_name, args.cuda, gt_embs)
        emb_mem.load_gt(args.env_name, args.cuda, args, gt_embs)
        dist_mem.store_embs(emb_mem.mem_keys)

        # Override the z dim
        args.z_dim = dist_mem.mem_keys.shape[-1]
        args.o_dim = emb_mem.get_key_dim()
        print('GT embs')
    elif args.use_random_embeddings:
        emb_mem.randomize_embeddings()
        dist_mem.randomize_embs()
        dist_mem.store_embs(emb_mem.mem_keys)
        print('Randomized embeddings')
    elif args.distance_effect:
        emb_mem.replace(dist_mem)

    if args.no_var:
        dist_mem.no_var()

    if args.discrete_beta and (args.distance_based or (args.use_ours_categorical and (args.distance_inside_cat or args.action_distribution))):
        emb_mem.set_max_range(args)
    elif (args.discrete_beta and (args.use_main_model or args.use_ours_categorical or args.categorical_nn)) or args.bound_effect:
        dist_mem.set_max_range(args)

    if args.normalize_embs:
        dist_mem.normalize_mem()

    return emb_mem, dist_mem


if __name__ == '__main__':
    run_settings = ExpRunSettings()
    run_settings.init()

    args = run_settings.get_args()

    if args.only_vis_embs:
        args.env_trans_fn = run_settings.get_env_trans_fn(args)
        from method.embedder.vis import vis_embs
        args.load_emb_model_file = 'eval'
        import gym; temp_env = gym.make(args.env_name)
        args.env_interface.env_trans_fn(temp_env, set_eval=False)
        if args.env_name.startswith('Reco'):
            save_prefix = 'reco'
        else:
            save_prefix = args.load_embeddings_file if args.load_embeddings_file is not None else ''
        if args.gt_embs:
            save_prefix = save_prefix.split('_')[0] + '_engineered'

        vis_embs(args.dist_mem, args.emb_mem, args.num_distributions, args.exp_type,
                 True, save_prefix=save_prefix, args=args,
                 use_idx=None)
        vis_embs(args.dist_mem, args.emb_mem, args.num_distributions, args.exp_type,
                 True, save_prefix=save_prefix + '_train',
                 args=args,
                 use_idx=args.env_interface.train_action_set)
        vis_embs(args.dist_mem, args.emb_mem, args.num_distributions, args.exp_type,
                 True, save_prefix=save_prefix + '_test', args=args,
                 use_idx=args.env_interface.test_action_set)
    else:
        run_policy(run_settings)
