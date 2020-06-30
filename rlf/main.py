from rlf.rl.rl import load_from_checkpoint, train, init_torch, create_algo, create_rollout_buffer
import torch
from rlf.rl.logger import Logger
from rlf.rl.checkpointer import Checkpointer
from rlf.rl.envs import make_vec_envs
from rlf.rl.evaluation import full_eval


class RunSettings(object):
    def __init__(self):
        pass

    def get_policy_type(self):
        raise NotImplemented()

    def get_train_env_trans(self, args, task_id=None):
        pass

    def get_test_env_trans(self, args, task_id=None):
        pass

    def get_args(self):
        raise ValueError('not implemented')

    def get_test_args(self):
        raise ValueError('not implemented')

    def get_set_args(self):
        raise ValueError('not implemented')


def get_num_updates(args):
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    if args.lr_env_steps is None:
        args.lr_env_steps = args.num_env_steps
    lr_updates = int(
        args.lr_env_steps) // args.num_steps // args.num_processes
    return num_updates, lr_updates


def get_fine_tune_run(load_name, env_name):
    from rlf.exp_mgr.eval import get_max_run
    import wandb
    import subprocess
    import os
    import os.path as osp
    env_id, prefix = load_name.split('/')
    api = wandb.Api()
    best_run = get_max_run(env_id, api, prefix, False)

    if env_name == 'StackEnvSimplestMovingAll-v0' and '-m' in load_name:
        env_name = 'StackEnvSimplestMoving-v0'

    load_folder_name = osp.join('data', 'trained_models', env_name, prefix)
    if not osp.exists(load_folder_name):
        os.makedirs(load_folder_name)

    run_cmd = (
        f"scp aszot@lim-b.usc.edu:~/nips2019_backup/{load_folder_name}/model_{best_run}.pt"
        f" {os.getcwd()}/{load_folder_name}/"
    )
    processes = subprocess.run(run_cmd.split(' '))

    run_cmd = (
        f"scp aszot@lim-b.usc.edu:/home/ayush/nips2019_backup/{load_folder_name}/model_{best_run}.pt"
        f" {os.getcwd()}/{load_folder_name}/"
    )
    processes = subprocess.run(run_cmd.split(' '))

    return osp.join(load_folder_name, 'model_%i.pt' % best_run)


def run_policy(run_settings):
    args = run_settings.get_args()
    test_args = run_settings.get_test_args()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    test_args.device = args.device

    if args.load_best_name is not None:
        args.load_file = get_fine_tune_run(args.load_best_name, args.env_name)

    log = Logger(run_settings.get_set_args())
    log.set_prefix(args)
    log.set_prefix(test_args)

    checkpointer = Checkpointer(args)

    init_torch(args)
    args.env_trans_fn = run_settings.get_train_env_trans()
    args.test_env_trans_fn = run_settings.get_test_env_trans()

    test_args.env_trans_fn = run_settings.get_train_env_trans()
    test_args.test_env_trans_fn = run_settings.get_test_env_trans()
    args.trajectory_len = None
    test_args.trajectory_len = None

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, args.device,
                         False, args.env_trans_fn, args)

    policy_class = run_settings.get_policy_type()

    policy = policy_class(args, envs.observation_space, envs.action_space)

    action_space = envs.action_space

    if checkpointer.should_load():
        load_from_checkpoint(policy, envs, checkpointer)

    updater = create_algo(policy, args)

    rollouts = create_rollout_buffer(policy, envs,
                                     action_space,
                                     args)

    if args.eval_only:
        full_eval(envs, policy, log, checkpointer, args)
        return

    log.watch_model(policy)

    start_update = 0
    if args.resume:
        updater.load_resume(checkpointer)
        policy.load_resume(checkpointer)
        start_update = checkpointer.get_key('step')

    num_updates, lr_updates = get_num_updates(args)

    train(envs, rollouts, policy, updater, log, start_update,
          num_updates, lr_updates, args, test_args, checkpointer)
