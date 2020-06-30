from collections import defaultdict

import numpy as np
import torch
import os
import os.path as osp

from rlf.rl import utils
from rlf.rl.envs import make_vec_envs
from rlf.rl.utils import save_mp4
from rlf.baselines.common.tile_images import tile_images

import cv2

VID_DIR = './vids'

def train_eval(envs, policy, args, test_args, log, j,
        total_num_steps, test_eval_envs, train_eval_envs):
    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm != None:
        ob_rms = vec_norm.ob_rms
    else:
        ob_rms = None

    args.evaluation_mode = True
    test_args.evaluation_mode = True

    print('Evaluating train')
    eval_train_reward, eval_train_info, train_eval_envs = evaluate(args, policy, ob_rms,
            log, args.env_trans_fn, j+1, num_render=args.num_render, train_mode='train',
            eval_envs=train_eval_envs)

    if not args.no_test_eval:
        assert args.test_env_trans_fn is not None
        print('Evaluating test')
        eval_test_reward, eval_test_info, test_eval_envs = evaluate(test_args,
                policy, ob_rms,
                log, args.test_env_trans_fn, j+1, num_render=args.num_render, train_mode='test',
                eval_envs=test_eval_envs)

    include_keys = args.env_interface.get_special_stat_names()
    if not args.no_test_eval:
        log.wandb_log({'eval_test_' + k: np.mean(v)
            for k, v in eval_test_info.items()
            if k in include_keys}, ns=total_num_steps)
    log.wandb_log({'eval_train_' + k: np.mean(v)
        for k, v in eval_train_info.items()
        if k in include_keys}, ns=total_num_steps)

    if not args.no_test_eval:
        log.wandb_log({'eval_test_reward' : np.mean(eval_test_reward)},  ns=total_num_steps)
    log.wandb_log({'eval_train_reward'  : np.mean(eval_train_reward)},  ns=total_num_steps)

    args.evaluation_mode = False
    test_args.evaluation_mode = False
    return test_eval_envs, train_eval_envs

def full_eval(envs, policy, log, checkpointer, args):
    assert checkpointer.should_load()
    ob_rms = None
    if utils.get_vec_normalize(envs) is not None:
        ob_rms = utils.get_vec_normalize(envs).ob_rms
    args.evaluation_mode = True

    evaluate(args, policy, ob_rms, log, args.env_trans_fn,
            0, verbose=args.verbose_eval,
            num_render=args.num_render, train_mode='test' if args.test_split else 'train')
    envs.close()
    args.evaluation_mode = False

    return


def evaluate(args, policy, ob_rms, log, env_trans_fn, num_iters=0,
             verbose=False, num_render=None, train_mode='train', eval_envs=None):

    if 'Reco' in args.env_name:
        num_render = 0

    if args.eval_num_processes is None:
        num_processes = args.num_processes
    else:
        num_processes = args.eval_num_processes
    seed = args.seed
    num_eval = args.num_eval
    env_name = args.env_name
    eval_log_dir = log.eval_log_dir

    if eval_envs is None:
        eval_envs = make_vec_envs(env_name, seed + num_iters, num_processes,
                                  None, eval_log_dir, args.device, True,
                                  env_trans_fn, args, set_eval=True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []
    ep_stats = defaultdict(list)

    obs = eval_envs.reset()

    if args.recurrent_policy:
        eval_recurrent_hidden_states = torch.zeros(
            num_processes, policy.actor_critic.recurrent_hidden_state_size, device=args.device)
    eval_masks = torch.zeros(num_processes, 1, device=args.device)

    frames = []

    cur_episode_frames = []
    success_frames = []
    failure_frames = []

    policy.eval()
    im_counter = 0
    start_frame = None
    prev_frame = None
    total_sim = num_processes * num_eval
    env_interface = args.env_interface
    iter_i = 0

    while len(eval_episode_rewards) < total_sim:
        iter_i += 1
        with torch.no_grad():
            add_input = torch.FloatTensor(eval_envs.get_aval())
            ac_outs, q_outs = policy.get_action(obs,
                    add_input,
                    eval_recurrent_hidden_states if args.recurrent_policy else None,
                    eval_masks,
                    args,
                    network='critic', num_steps=None)

            value, action, action_log_prob, eval_recurrent_hidden_states = ac_outs
            take_action, reward_effect, extra = q_outs

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(take_action)

        cur_frame = None
        if (num_render is not None and len(eval_episode_rewards) < num_render) or num_render is None:
            if args.should_render_obs:
                utils.render_obs(obs, 'obs_%i' % iter_i, args)
            cur_frame = eval_envs.render(mode=env_interface.get_render_mode())

            env_interface.mod_render_frames(frames, infos, cur_frame)
            if isinstance(cur_frame, list):
                frames.extend(cur_frame)
            else:
                frames.append(cur_frame)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=args.device)

        save_dir = osp.join(args.result_dir, args.env_name)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        # Result Figures
        if args.env_name.startswith('Create') and args.render_result_figures and \
            args.render_ball_traces and args.eval_only and num_processes == 1:
            #draw_frame = cur_frame
            #if len(np.array(draw_frame).shape) == 4:
            #    draw_frame = draw_frame[-1]
            if start_frame is None:
                start_frame = cur_frame
                save_name = '%s_%s-%i-a.png' % (args.env_name, args.prefix, im_counter)
                cv2.imwrite(osp.join(save_dir, save_name), cv2.cvtColor(np.float32(start_frame), cv2.COLOR_RGB2BGR))

            elif done[0] and prev_frame is not None:
                if infos[0]['ep_goal_hit']:
                    bordercolor = [0, 255, 0]
                else:
                    bordercolor = [255, 0, 0]
                bordersize=5

                #draw_prev_frame = prev_frame
                #if len(np.array(draw_prev_frame).shape) == 4:
                #    draw_prev_frame = draw_prev_frame[-1]
                border=cv2.copyMakeBorder(
                    prev_frame, top=bordersize,
                    bottom=bordersize, left=bordersize, right=bordersize,
                    borderType= cv2.BORDER_CONSTANT, value=bordercolor )

                save_path = osp.join(save_dir, '%s_%s-%i-b.png' % (args.env_name, args.prefix, im_counter))
                cv2.imwrite(save_path, cv2.cvtColor(np.float32(border if args.render_borders else prev_frame), cv2.COLOR_RGB2BGR))
                print('Wrote to %s' % save_path)
                im_counter += 1

                save_path = osp.join(save_dir, '%s_%s-%i-a.png' % (args.env_name, args.prefix, im_counter))
                cv2.imwrite(save_path,
                        cv2.cvtColor(np.float32(cur_frame), cv2.COLOR_RGB2BGR))
                print('Wrote to %s' % save_path)


        # Success & Failure Cases
        if args.env_name.startswith('Create') and num_processes == 1 and args.eval_only \
            and args.success_failures:

            if done[0]:
                if infos[0]['ep_goal_hit']:
                    success_frames.extend(cur_episode_frames)
                else:
                    failure_frames.extend(cur_episode_frames)
                cur_episode_frames = []

            cur_episode_frames.append(cur_frame)

        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                er = info['episode']['r']
                eval_episode_rewards.append(er)
                if verbose:
                    if len(eval_episode_rewards) % (total_sim / 10) == 0:
                        print(100.0 * (len(eval_episode_rewards) / total_sim), '%')
            if done[i]:
                for key in info:
                    if 'ep_' in key:
                        ep_stats[key].append(info[key])

        if cur_frame is not None:
            prev_frame = cur_frame

    if verbose:
        # We are in full evaluation mode, job is going to end after this
        # evaluation function has ended.
        eval_envs.close()

    mean_reward = np.mean(eval_episode_rewards)

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), mean_reward))

    ret_info = {}

    for k, v in ep_stats.items():
        ret_info[k] = np.mean(v)
        print('{}: mean = {}, std = {}'.format(k, np.mean(v), np.std(v)))

    if not osp.exists(args.vid_dir):
        os.makedirs(args.vid_dir)

    # 60 is super fast for grid world
    add = ''
    if args.load_file != '':
        add = args.load_file.split('/')[-2]
        add += '_'

    save_name = '%s%s_%s' % (add,
            str(num_iters), train_mode)

    save_dir = osp.join(args.vid_dir, args.env_name, args.prefix)

    if args.render_mega_res:
        fps = args.vid_fps * args.large_steps / args.mega_res_interval
    else:
        fps = args.vid_fps

    if len(frames) > 0 and not args.success_only:
        save_mp4(frames, save_dir, save_name, fps=fps, no_frame_drop=True)
        print('Rendered frames to %s' % osp.join(save_dir, save_name))

    if args.success_failures:
        # Success Frames
        if len(success_frames) > 0:
            success_name = '%s%s_%s-success' % (add, str(num_iters), train_mode)
            save_mp4(success_frames, save_dir, success_name, fps=fps, no_frame_drop=True)
            print('Rendered Success Frames')

        # Failure Frames
        if len(failure_frames) > 0 and not args.success_only:
            failure_name = '%s%s_%s-failure' % (add, str(num_iters), train_mode)
            save_mp4(failure_frames, save_dir, failure_name, fps=fps, no_frame_drop=True)
            print('Rendered Failure Frames')

    policy.train()

    return mean_reward, ret_info, eval_envs
