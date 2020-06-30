import random
from collections import deque

import numpy as np
import torch
from tqdm import tqdm

from rlf.rl import algo, utils
from rlf.rl.evaluation import full_eval, train_eval
from rlf.rl.storage import RolloutStorage
from rlf.rl.utils import get_vec_normalize
import time

def train(envs, rollouts, policy, updater, log, start_update,
          end_update, lr_updates, args, test_args, checkpointer):

    print('RL Training (%d/%d)' % (start_update, end_update))
    episode_rewards = deque(maxlen=100)

    test_eval_envs = None
    train_eval_envs = None

    for j in range(start_update, end_update):
        log.start_interval_log()

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                updater.optimizer, j, lr_updates,
                updater.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            cur_num_steps = (j * args.num_steps + step) * args.num_processes

            # Sample actions
            ac_outs, q_outs = policy.get_action(
                rollouts.obs[step],
                rollouts.add_input[step],
                rollouts.recurrent_hidden_states[step] if args.recurrent_policy else None,
                rollouts.masks[step],
                args,
                network='critic',
                num_steps=cur_num_steps)

            value, action, action_log_prob, recurrent_hidden_states = ac_outs
            take_action, add_reward, extra = q_outs

            obs, reward, done, infos = envs.step(take_action)
            reward += add_reward

            log.log_alg_extra(extra)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                if done[i]:
                    log.log_ep_stats(info)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            aval = np.array(envs.get_aval())
            add_input = torch.FloatTensor(aval)

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks,
                            add_input)

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        with torch.no_grad():
            next_value = policy.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[step] if args.recurrent_policy else None,
                rollouts.masks[-1], rollouts.actions[-1],
                rollouts.add_input[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        if args.decay_clipping:
            updater.clip_param = args.clip_param * (1. - j / end_update)

        log_vals = updater.update(rollouts)

        rollouts.after_update()

        if ((j+1) % args.save_interval == 0) and checkpointer.should_save():
            save_agent(policy, envs, j, updater, args, checkpointer, log)

        if (j+1) % args.log_interval == 0 and len(episode_rewards) > 1:
            log.interval_log(j, total_num_steps,
                             episode_rewards, log_vals, args)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and (j+1) % args.eval_interval == 0):
            test_eval_envs, train_eval_envs = train_eval(envs, policy, args,
                       test_args, log, j, total_num_steps, test_eval_envs,
                       train_eval_envs)

    log.close()



def create_algo(policy, args):
    # only use the different dist LR for our method
    if args.algo == 'ppo':
        updater = algo.PPO(
            policy,
            args)
    else:
        raise ValueError('RL Algorithm not present')

    return updater


def init_torch(args):
    # Set all seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        print('Making deterministic!')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.use_double:
        print('Setting default to double tensor')
        torch.set_default_tensor_type(torch.DoubleTensor)


def create_rollout_buffer(policy, envs, action_space, args):
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, action_space, args,
                              policy.get_actor_critic_count(),
                              policy.get_dim_add_input())

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)

    add_input = torch.FloatTensor(envs.get_aval())
    if add_input is not None:
        rollouts.add_input[0].copy_(add_input)

    rollouts.to(args.device)
    return rollouts


def load_from_checkpoint(policy, envs, checkpointer):
    policy.load_actor_from_checkpoint(checkpointer)

    ob_rms = checkpointer.get_key('ob_rms')
    vec_norm = get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

def save_agent(policy, envs, j, updater, args, checkpointer, log):
    checkpointer.save_key('ob_rms', getattr(utils.get_vec_normalize(envs), 'ob_rms', None))
    checkpointer.save_key('step', j)

    policy.save_actor_to_checkpointer(checkpointer)

    updater.save(checkpointer)

    checkpointer.flush(num_updates=j)
    if args.backup:
        log.backup(args, j + 1)



