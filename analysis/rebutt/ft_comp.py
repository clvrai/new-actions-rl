import os.path as osp
import argparse
import numpy as np
import pickle
from collections import defaultdict
import yaml
import wandb


def_ordering = [
		'main',
		'rnd_sample', 'rs',
		'no_ent', 'ne',
		'ac_fixed', 'fixed', 'fx',
		'tvae', 'vae',
		'gt', 'dist', 'nn',
		'im', 'main2', 'all',

		'funnel',
		'basket',
		'ladder',
		'moving',
		'collide',
		'cannon',
		'buckets',
		'belt',
		'navigate',

        'gw',
        'obstacle',
        'push',
        'seesaw',
        'reco',
        'bs',

		'dim0',
		'dim2',
		'full',
		]


settings = {
        'lg': {
            'metric': 'eval_train_ep_goal_hit',
            'look_for': 'ep_goal_hit',
            'split_by': 'ep_goal_hit: mean = ',
            'factor': 100.0,
            },
        'gw': {
            'metric': 'eval_train_ep_success',
            'look_for': 'ep_success',
            'split_by': 'ep_success: mean = ',
            'factor': 100.0,
            },
        'bs': {
            'metric': 'eval_train_ep_final_height',
            'look_for': 'ep_final_height',
            'split_by': 'ep_final_height: mean = ',
            'factor': 1.0,
            },
        'reco': {
            'metric': 'eval_train_ep_avg_ctr',
            'look_for': 'ep_avg_ctr',
            'split_by': 'ep_avg_ctr: mean = ',
            'factor': 1.0,
            }

        }



def load_data(result_name, ret_train=False, scale=1.0):
    full_path = osp.join('/home/aszot/functional-rl/analysis/outcomes/', result_name + '.o')
    with open(full_path, 'rb') as f:
        results_train, results_test = pickle.load(f)
    if ret_train:
        ret_val = results_train
    else:
        ret_val = results_test
    if ret_val is None or len(ret_val) == 0:
        return [], []

    first_ele = ret_val[list(ret_val.keys())[0]][0]
    if isinstance(first_ele, tuple):
        for k in ret_val:
            ret_val[k] = [x[1] for x in ret_val[k]]
    return extract_data(ret_val, scale=scale)

def load_datas(result_names, ret_train=False, scale=1.0, replace=False):
    d = defaultdict(list)
    for result_name in result_names:
        dat, names = load_data(result_name, ret_train, scale)
        for i, n in enumerate(names):
            if replace:
                d[n] = list(dat[i])
            else:
                d[n].extend(list(dat[i]))

    for n in d:
        d[n] = np.array(d[n])

    vn_pairs = list([(k, d[k]) for k in def_ordering if k in d])
    return [x[1] for x in vn_pairs], [x[0] for x in vn_pairs]

def extract_data(dat, scale=1.0):
    names = []
    results = []

    for k in def_ordering:
        if k in dat:
            names.append(k)
            results.append(dat[k])
    for k, v in dat.items():
        if k not in def_ordering:
            names.append(k)
            results.append(v)
    results = [np.array(x) * scale for x in results]
    return results, names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, help='')
    args = parser.parse_args()

    ours = [
            ['rebutt_ftcmp_gw'],
            ['rebutt_ftcmp_reco'],
            ['rebutt_ftcmp_create'],
            ['rebutt_ftcmp_bs1', 'rebutt_ftcmp_bs5'],
            ]

    add_dat = {
            # Additional result from limd
            'bs': [1.6748110463650676]
            }

    ours_dat = {}

    for methods in ours:
        dat, names = load_datas(methods)
        for i, n in enumerate(names):
            ours_dat[n] = np.mean([*dat[i], *add_dat.get(n, [])])

    CMP_FILE_PATH = osp.join('/home/aszot/functional-rl/analysis/rebutt/ft/',
            args.f + '.yaml')
    with open(CMP_FILE_PATH) as f:
        ft_methods = yaml.load(f)
    api = wandb.Api()

    results = defaultdict(dict)

    USE_METHODS = ['discscratch','discft','mainscratch','mainft',]
    #assert len(ft_methods) == len(USE_METHODS)

    for method, seeds in ft_methods.items():
        method_name, comp_item = method.split('_')
        cmp_val = ours_dat[comp_item]
        all_dats = []
        steps = None
        max_step_len = 0
        print('For ' + method_name)
        for seed in seeds:
            print('Fetching ', seed)
            runs = api.runs('clvr/functional-rl', {"config.prefix": seed})
            assert len(runs) == 1
            wbrun = next(runs)
            if comp_item not in settings:
                setting_option = settings['lg']
            else:
                setting_option = settings[comp_item]

            hist = wbrun.history(samples=15000)
            sel_col = setting_option['look_for']
            dat = hist[sel_col].fillna(0.0).to_numpy()
            cur_steps = hist['_step'].to_numpy()
            all_dats.append(dat)
            if len(cur_steps) > max_step_len:
                max_step_len = len(cur_steps)
                steps = cur_steps

        max_len = max(map(len, all_dats))
        all_dats = [np.pad(x, (0, max_len - len(x)), 'edge')
                for x in all_dats]

        all_dats = np.mean(all_dats, axis=0)
        assert len(all_dats) == len(steps)

        found_idx = None
        print('Comparing to %.2f' % cmp_val)
        print('With maximum %.2f' % max(all_dats))
        for i in range(len(steps)):
            if all_dats[i] > cmp_val:
                found_idx = steps[i]
                break

        if found_idx is not None:
            print('Found idx', found_idx)
        else:
            print('NOT FOUND?!?!?')

        results[comp_item][method_name] = found_idx

    env_name_map = {
            'obstacle': 'Obstacle',
            'push': 'Push',
            'seesaw': 'Seesaw',
            'gw': 'Grid World',
            'reco': 'Reco',
            'bs': 'Shape Stack'
            }

    for env_name in results:
        all_n_steps = [results[env_name][n]
                for n in USE_METHODS
                if n in results[env_name]]

        def format_step(x):
            if x is None:
                return 'NA'
            else:
                return "%.1f" % (float(x) / 100000)

        all_n_steps = list(map(format_step, all_n_steps))

        if comp_item not in settings:
            setting_option = settings['lg']
        else:
            setting_option = settings[env_name]
        factor = setting_option['factor']
        cmp_val *= factor

        print(f"{env_name_map[env_name]} {cmp_val:.1f}: {','.join(all_n_steps)}")
