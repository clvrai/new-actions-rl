import sys
sys.path.insert(0, '.')
from rlf.exp_mgr.eval import get_results, load_seed
import os.path as osp
import pickle
from analysis.analysis_dist import get_cmds
import os

if __name__ == '__main__':
    cmds, names, methods, env = get_cmds()
    final_results = {}

    for ratio in [0.0, 0.2, 0.4, 0.6, 0.8]:
        use_cmds = [cmd + ' --train-mix-ratio %.5f --num-processes 1 --eval-num-processes 1' % ratio for cmd in cmds]
        # use_cmds = [cmd + ' --train-mix-ratio %.5f' % ratio for cmd in cmds]

        combined = list(zip(methods, names, use_cmds))

        results = get_results(combined, True, False, 'lg')
        print('ratio %.5f results' % ratio)
        print(results)

        final_results[ratio] = results

    RESULTS_DIR = osp.join('./analysis/outcomes/', env)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    path = osp.join(RESULTS_DIR, 'analysis_ratio.o')
    print(final_results)
    with open(path, 'wb') as f:
        pickle.dump(final_results, f)



