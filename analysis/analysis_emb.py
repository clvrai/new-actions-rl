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

    # 5.0 - 379
    # 4.0 - 532
    # 3.0 - 813
    # 2.0 - 1446
    # 0.9 - 2702
    # 0.4 - 3772
    # 0.3 - 4037
    # 0.2 - 4463 
    # 0.1 - 5276
    # 0.02 - 6007

    # Need to num-eval 5000 for num-processes 1

    for emb in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        use_cmds = [cmd + ' --analysis-emb %.5f --num-processes 1 --eval-num-processes 1' % emb for cmd in cmds]

        combined = list(zip(methods, names, use_cmds))

        results = get_results(combined, True, False, 'lg')
        print('emb %.5f results' % emb)
        print(results)

        final_results[emb] = results

    RESULTS_DIR = osp.join('./analysis/outcomes/', env)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    path = osp.join(RESULTS_DIR, 'analysis_emb.o')
    print(final_results)
    with open(path, 'wb') as f:
        pickle.dump(final_results, f)

