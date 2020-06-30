import os.path as osp
from tensorboardX import SummaryWriter
import os
from six.moves import shlex_quote
from rlf.rl import utils
import sys
import pipes
import time
import numpy as np
import random
import datetime
import string
import copy
from rlf.exp_mgr import config_mgr

from collections import deque, defaultdict



class Logger(object):
    def __init__(self, args, log_dir='./data/logs/'):
        if not osp.exists(log_dir + args.env_name):
            os.makedirs(log_dir + args.env_name)

        if args.eval_only:
            args.wand = False

        self.wandb = None
        self._create_prefix(args)
        if args.wand:
            self.wandb = self._create_wandb(args)

        self.writer = None
        if not args.eval_only:
            self.writer = self._create_writer(args, log_dir)

        # log_dir default is '/tmp/gym/'
        log_dir = os.path.expanduser(args.log_dir)
        self.eval_log_dir = log_dir + "_eval"
        self.trial_log_dir = log_dir + "_trial"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(self.eval_log_dir)
        utils.cleanup_log_dir(self.trial_log_dir)

        self._reset_ep_stats()
        self.alg_add_info = defaultdict(lambda: deque(maxlen=100))

    def backup(self, args, global_step=-1):
        log_dir = self.backup_log_dir
        model_dir = osp.join(args.save_dir, args.env_name, args.prefix)
        vid_dir = osp.join(args.vid_dir, args.env_name, args.prefix)

        log_base_dir = log_dir.rsplit('/', 1)[0]
        model_base_dir = model_dir.rsplit('/', 1)[0]
        vid_base_dir = vid_dir.rsplit('/', 1)[0]
        proj_name = config_mgr.get_prop('proj_name')
        sync_host = config_mgr.get_prop('sync_host')
        cmds = [
            "ssh -i ~/.ssh/id_open_rsa/id -p {} {}@{} 'mkdir -p ~/{}_backup/{}'".format(
                args.sync_port, args.backup, sync_host, proj_name, log_dir),
            "ssh -i ~/.ssh/id_open_rsa/id -p {} {}@{} 'mkdir -p ~/{}_backup/{}'".format(
                args.sync_port, args.backup, sync_host, proj_name, model_dir),
            "ssh -i ~/.ssh/id_open_rsa/id -p {} {}@{} 'mkdir -p ~/{}_backup/{}'".format(
                args.sync_port, args.backup, sync_host, proj_name, vid_dir),
            'rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p {}" {} {}@{}:~/{}_backup/{}'.format(
                args.sync_port, log_dir, args.backup, sync_host, proj_name, log_base_dir),
            'rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p {}" {} {}@{}:~/{}_backup/{}'.format(
                args.sync_port, model_dir, args.backup, sync_host, proj_name, model_base_dir),
            'rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p {}" {} {}@{}:~/{}_backup/{}'.format(
                args.sync_port, vid_dir, args.backup, sync_host, proj_name, vid_base_dir),
        ]
        os.system("\n".join(cmds))
        print('\n' + '*' * 50)
        print('*' * 5 + ' backup at global step {}'.format(global_step))
        print('*' * 50 + '\n')
        print('')



    def _reset_ep_stats(self):
        self.ep_stats = defaultdict(list)

    def log_alg_extra(self, extra):
        for k, v in extra.items():
            if 'alg_add' in k:
                self.alg_add_info[k].append(v)

    def log_ep_stats(self, info):
        for key in info:
            if 'ep_' in key:
                self.ep_stats[key].append(info[key])

    def wandb_log(self, d, ns):
        if self.wandb is not None:
            self.wandb.log(d, step=ns)

    def write_scalar(self, full_k, v, ns):
        self.writer.add_scalar(full_k, v, ns)
        k = full_k.split('/')[-1]
        self.wandb_log({k: v}, ns)

    def _get_env_id(self, args):
        if args.env_name.startswith('CreateLevel'):
            lvl_name = args.env_name.split('CreateLevel')[1].split('-')[0]
            if lvl_name == '%s':
                # forgot to format it
                raise ValueError('Must specify level name')
            return lvl_name
        elif args.env_name.startswith('Create'):
            return 'CP'
        elif args.env_name.startswith('StateCreate'):
            return 'SCP'    # State Create Play
        elif args.env_name.startswith('Stack') or args.env_name.startswith('Block'):
            return 'BS'
        elif args.env_name.startswith('SimpleStack'):
            return 'SS'
        elif args.env_name.startswith('MiniGrid'):
            return 'GW'
        elif args.env_name.startswith('Reco'):
            return 'RE'
        else:
            raise ValueError('Could not recognize env')

    def _create_prefix(self, args):
        assert args.prefix is not None and args.prefix != '', 'Must specify a prefix'
        d = datetime.datetime.today()
        date_id = '%i%i' % (d.month, d.day)
        env_id = self._get_env_id(args)
        rnd_id = ''.join(random.sample(
            string.ascii_uppercase + string.digits, k=2))
        before = ('%s-%s-%s-%s-' %
                  (date_id, env_id, args.seed, rnd_id))

        if args.prefix != 'debug' and args.prefix != 'NONE':
            self.prefix = before + args.prefix
            print('Assigning full prefix %s' % self.prefix)
        else:
            self.prefix = args.prefix


    def _create_wandb(self, args):
        import wandb
        args.prefix = self.prefix
        wandb.init(project="functional-rl", name=self.prefix)
        wandb.config.update(args)
        return wandb

    def set_prefix(self, args):
        args.prefix = self.prefix

    def _create_writer(self, args, log_dir):
        log_dir = osp.join(log_dir, args.env_name, args.prefix)
        writer = SummaryWriter(log_dir)

        # cmd
        train_cmd = 'python3 main.py ' + \
            ' '.join([pipes.quote(s) for s in sys.argv[1:]])
        with open(osp.join(log_dir, "cmd.txt"), "a+") as f:
            f.write(train_cmd)

        # git diff
        print('Save git commit and diff to {}/git.txt'.format(log_dir))
        cmds = ["echo `git rev-parse HEAD` >> {}".format(
            shlex_quote(osp.join(log_dir, 'git.txt'))),
            "git diff >> {}".format(
            shlex_quote(osp.join(log_dir, 'git.txt')))]
        os.system("\n".join(cmds))

        args_lines = "Date and Time:\n"
        args_lines += time.strftime("%d/%m/%Y\n")
        args_lines += time.strftime("%H:%M:%S\n\n")
        arg_dict = args.__dict__
        for k in sorted(arg_dict.keys()):
            args_lines += "{}: {}\n".format(k, arg_dict[k])

        with open(osp.join(log_dir, "args.txt"), "w") as f:
            f.write(args_lines)

        self.backup_log_dir = log_dir

        return writer

    def watch_model(self, policy):
        actor_critic = policy.get_actor_critic()
        if self.wandb is not None:
            for ac in actor_critic.get_policies():
                if hasattr(ac.dist, 'disc') and hasattr(ac.dist.disc, 'linear'):
                    self.wandb.watch((ac.dist.disc.linear, ac.base))
                else:
                    self.wandb.watch(ac.base)
                break

    def start_interval_log(self):
        self.start = time.time()

    def interval_log(self, j, total_num_steps, episode_rewards, log_vals, args):
        end = time.time()
        print(
            "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
            .format(j, total_num_steps,
                    int(args.num_processes * args.num_steps / (end - self.start)),
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards)))
        for k, v in self.ep_stats.items():
            self.writer.add_scalar(
                'ep/{}'.format(k[3:]), np.mean(v), total_num_steps)
        self.wandb_log({k: np.mean(v)
                        for k, v in self.ep_stats.items()}, total_num_steps)

        # TODO: This is specific for lgoic game. Change this.
        if 'ep_goal_hit' in self.ep_stats:
            print(
                "Mean hit target {}, Mean hit Goal {}"
                .format(np.mean(self.ep_stats['ep_target_hit']),
                        np.mean(self.ep_stats['ep_goal_hit'])))

        for k, v in log_vals.items():
            self.writer.add_scalar('data/' + k, v, total_num_steps)
        self.writer.add_scalar(
            'data/avg_reward', np.mean(episode_rewards), total_num_steps)
        self.writer.add_scalar(
            'data/max_reward', np.max(episode_rewards), total_num_steps)
        self.writer.add_scalar(
            'data/min_reward', np.min(episode_rewards), total_num_steps)

        self.wandb_log({
            **log_vals,
            'avg_reward': np.mean(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards),
        }, total_num_steps)

        reg_add_info = {k: v for k,
                        v in self.alg_add_info.items() if 'hist' not in k}
        self.wandb_log({k: np.mean(v)
                        for k, v in reg_add_info.items()}, total_num_steps)

        self._reset_ep_stats()

    def close(self):
        self.writer.close()
