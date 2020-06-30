import argparse
import os.path as osp
import os
import uuid
from collections import defaultdict
import numpy as np
import pickle
import yaml
import wandb
import sys
sys.path.insert(0, '.')
from rlf.exp_mgr import config_mgr

settings = {
        'lg': {
            'metric': 'ep_goal_hit',
            'look_for': 'ep_goal_hit',
            'split_by': 'ep_goal_hit: mean = ',
            },
        'gw': {
            'metric': 'ep_success',
            'look_for': 'ep_success',
            'split_by': 'ep_success: mean = ',
            },
        'bs': {
            'metric': 'ep_final_height',
            'look_for': 'ep_final_height',
            'split_by': 'ep_final_height: mean = ',
            },
        'reco': {
            'metric': 'ep_avg_ctr',
            'look_for': 'ep_avg_ctr',
            'split_by': 'ep_avg_ctr: mean = ',
            }

        }

def get_max_run(env, api, exp_name, eval_train):
    runs = api.runs(config_mgr.get_prop('w_b_proj'), {"config.prefix": exp_name})
    assert len(runs) == 1
    sel_col = settings[env]['metric']
    if eval_train:
        sel_col = sel_col.replace('test', 'train')

    for wbrun in runs:
        hist = wbrun.history(samples=15000)
        dat = hist[sel_col].fillna(0.0).to_numpy()
        best_run = np.argmax(dat, axis=0)
        best_run_str = str(best_run)
        if best_run_str[-2:] != '99' and best_run_str[-2:] != '49':
            best_run_str = best_run_str[:-2] + '49'
            best_run = int(best_run_str)
        return best_run



def load_seed(seed, lvl_name, cd, num_eval, high_render, env_name,
        vid_render=False, train_eval=False, server_only=False):
    locs = config_mgr.get_prop('backup_locs')
    api = wandb.Api()

    found_loc = None

    if not server_only:
        for loc in locs:
            open_cmd_f = osp.join(loc, 'data/logs', lvl_name, seed, 'cmd.txt')
            if osp.exists(open_cmd_f):
                found_loc = loc
                break
        if found_loc is None:
            raise ValueError('Could not find seed %s' % (seed))
    else:
        open_cmd_f = osp.join('/home/ayush/icml/data/logs', lvl_name, seed, 'cmd.txt')
        if not osp.exists(open_cmd_f):
            return None
        found_loc = '/home/ayush/icml/'

    with open(open_cmd_f) as f:
        cmd = f.readlines()[0]

    cmd_repeats = cmd.split('python')
    cmd = 'python'.join(cmd_repeats[:2])

    cmd = 'CUDA_VISIBLE_DEVICES=' + str(cd) + ' ' + cmd
    cmd += ' --eval-only'
    cmd += ' --num-eval %i' % num_eval
    cmd += ' --prefix "eval"'

    cmd = cmd.replace(
            '--load-embeddings-file create_all ',
            '--load-embeddings-file create_all_len7 ')
    cmd = cmd.replace('--log-dir /home/ayush/tmp/gym', '--log-dir ~/tmp/gym')
    cmd = cmd.replace('--log-dir /home/aszot/tmp/gym', '--log-dir ~/tmp/gym')

    if '--nearest-neighbor' in cmd:
        cmd += ' --load-fixed-action-set'

    if high_render:
        cmd += ' --num-processes 1'
        cmd += ' --success-failures'
        cmd += ' --render-high-res'
        cmd += ' --render-ball-traces'
        cmd += ' --render-mega-static-res'
        cmd += ' --render-result-figures'
        cmd = cmd.replace('--render-changed-colors', '')
        cmd = cmd.replace('--render-borders', '')
    elif vid_render:
        cmd += ' --num-processes 1'
        cmd += ' --eval-num-processes 1'
        cmd += ' --success-failures'
        cmd += ' --num-eval 20'
        cmd += ' --render-high-res'
        cmd += ' --render-mega-res'
        cmd += ' --image-resolution 1024'
        cmd = cmd.replace('--render-changed-colors', '')
        cmd = cmd.replace('--render-borders', '')
        if 'StackEnv' in cmd or 'MiniGrid' in cmd:
            cmd += ' --large-steps 1'
            cmd += ' --mega-res-interval 1'
    else:
        cmd += ' --num-render 0'

    cmd = cmd.replace('--action-random-sample False', '--action-random-sample True')
    if not (vid_render or high_render):
        cmd += ' --num-processes 40'
        cmd += ' --eval-num-processes 40'

    search_dir = osp.join(found_loc, 'data/trained_models', lvl_name, seed)

    best_count = get_max_run(env_name, api, seed, train_eval)

    print('Loading %s for %s' % (best_count, seed))
    use_model_file = osp.join(search_dir, 'model_%i.pt' % best_count)
    cmd += ' --load-file "%s"' % use_model_file
    return cmd


def get_results(cmds, is_test, verbose, env, args=None):
    results = defaultdict(list)
    for method, seed, cmd in cmds:
        use_cmd = cmd
        fixed_as = '--fixed-action-set' in use_cmd
        if is_test:
            use_cmd += ' --test-split'
            if args is None or not args.acfull:
                use_cmd = use_cmd.replace('--fixed-action-set', '')
        else:
            if fixed_as:
                use_cmd = use_cmd.replace(
                        '--action-random-sample True',
                        '--action-random-sample False')
                use_cmd = use_cmd.replace('--load-fixed-action-set', '')
        print('Executing ', use_cmd )
        tmp_name = 'tmp_' + seed + '_' + str(uuid.uuid4()) + '.txt'

        if verbose:
            symb = ' | tee'
        else:
            symb = ' > '

        os.system(use_cmd + symb + tmp_name)
        mean_reward = -1.0
        with open(tmp_name, 'r') as f:
            for l in f.readlines():
                if settings[env]['look_for'] in l:
                    mean_reward = l.split(settings[env]['split_by'])[1].split(',')[0]
                    mean_reward = float(mean_reward.rstrip())
                    print('%s-%s: %.2f' % (method, seed, mean_reward))

                    results[method].append(mean_reward)
                if 'Rendered frames' in l:
                    print('')
                    print(l)
                    print('')
                if 'Wrote to ' in l:
                    print('')
                    print(l)
                    print('')
        print('')
        print('-' * 20)
        print('')
        #if osp.exists(tmp_name):
        #    os.remove(tmp_name)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
            help='')
    parser.add_argument('--out-f', type=str, default=None, help='')
    parser.add_argument('--in-f', type=str, default=None,
            help='')
    parser.add_argument('--env', type=str, default=None,
            help='')

    parser.add_argument('--cd', type=int, default=0,
            help='gpu to use')

    parser.add_argument('--num-eval', type=int, default=125,
            help='Number of evaluation episodes per run')

    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--acfull', action='store_true', default=False)
    parser.add_argument('--high-render', action='store_true', default=False)
    parser.add_argument('--vid-render', action='store_true', default=False)
    parser.add_argument('--eval-train', action='store_true', default=False)
    parser.add_argument('--server-only', action='store_true', default=False)

    args = parser.parse_args()

    with open('analysis/eval_cmds/%s.yaml' % args.cfg) as f:
        methods = yaml.load(f)
    lvl_name = methods['task_name']

    results_dir = config_mgr.get_prop('results_dir')

    if args.in_f is None:
        cmds = []
        for method, seeds in methods.items():
            if method.startswith('task'):
                continue
            for seed in seeds:
                if lvl_name == 'infer':
                    use_task = 'CreateLevel' + seed.split('-')[1] + '-v0'
                else:
                    use_task = lvl_name
                cmd = load_seed(seed, use_task, args.cd, args.num_eval,
                        args.high_render, args.env, args.vid_render,
                        args.eval_train, args.server_only)
                if args.acfull and cmd is not None:
                    if 'Reco' in cmd:
                        cmd += ' --reco-special-fixed-action-set-size 500'
                    if 'StackEnv' in cmd:
                        cmd = cmd.replace('--env-name StackEnvSimplestMoving-v0',
                                '--env-name StackEnvSimplestMovingAll-v0')
                    cmd += ' --action-random-sample False --fixed-action-set'

                if cmd is None:
                    print('%s is not on this machine, skipping' % seed)
                    continue

                cmds.append((method, seed, cmd))

        if args.eval_train:
            results_train = get_results(cmds, False, args.verbose, args.env,
                    args)
            results_test = None
        else:
            results_train = None
            results_test = get_results(cmds, True, args.verbose, args.env, args)
        if args.out_f is None:
            name = args.cfg + '.o'
            i = 0
            while osp.exists(name):
                name = args.cfg + ('%i.o' % i)
            args.out_f = name
        save_name = osp.join(results_dir, args.out_f)
        with open(save_name, 'wb') as f:
            pickle.dump((results_train, results_test), f)
            print('saved to %s' % save_name)

    else:
        with open(osp.join(results_dir, args.in_f), 'rb') as f:
            results_train, results_test = pickle.load(f)

        if results_train is not None:
            print('Train')
            for method, result in results_train.items():
                for x in result:
                    print('%s: %.8f' % (method, x))
        if results_test is not None:
            print('Test')
            for method, result in results_test.items():
                for x in result:
                    print('%s: %.8f' % (method, x))



