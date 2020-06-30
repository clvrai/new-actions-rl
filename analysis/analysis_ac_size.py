import sys
sys.path.insert(0, '.')
import sys
from rlf.exp_mgr.eval import get_results, load_seed
import os.path as osp
import pickle
import argparse
import yaml
import numpy as np

def get_cmds():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--env')
    parser.add_argument('--cd', type=int)
    parser.add_argument('--cfg', type=str, help='')
    parser.add_argument('--num-eval', type=int, default=25)
    args = parser.parse_args()

    with open('analysis/eval_cmds/%s.yaml' % args.cfg) as f:
        methods = yaml.load(f)

    cmds = []
    seed_names = []
    method_names = []
    for method, seeds in methods.items():
            if method.startswith('task'):
                continue
            for seed in seeds:
                cmd = load_seed(seed, args.env, args.cd, args.num_eval, False, 'lg')

                method_names.append(method)
                seed_names.append(seed)
                cmds.append(cmd)

    return cmds, seed_names, method_names

if __name__ == '__main__':
    cmds, names, methods = get_cmds()
    final_results = {}

    for ac_size in np.arange(10, 65, 5):
        use_cmds = [cmd + ' --action-set-size %i' % ac_size for cmd in cmds]

        combined = list(zip(methods, names, use_cmds))

        results = get_results(combined, True, False, 'lg')
        print('ac size %i results' % ac_size)
        print(results)

        final_results[ac_size] = results

    #RESULTS_DIR = './exp-base/show/results/'
    RESULTS_DIR = './analysis/outcomes/'
    path = osp.join(RESULTS_DIR, 'seesaw_ac_size_analysis.o')
    with open(path, 'wb') as f:
        pickle.dump(final_results, f)



