import sys
sys.path.insert(0, '.')
import sys
from rlf.exp_mgr.eval import get_results, load_seed
import os.path as osp
import pickle
import argparse
import os

seed_names = {
    'CreateLevelPush-v0': [
        '626-Push-31-94-main_analysis',
        '626-Push-41-29-main_analysis',
        '626-Push-51-4C-main_analysis',
        '626-Push-61-Q0-main_analysis',
        '626-Push-71-EG-main_analysis',
        ],
    'CreateLevelObstacle-v0': [
        '626-Obstacle-31-KE-main_analysis',
        '626-Obstacle-41-P8-main_analysis',
        '626-Obstacle-51-BY-main_analysis',
        '626-Obstacle-61-DL-main_analysis',
        '626-Obstacle-71-RW-main_analysis',
        ],
    'CreateLevelSeesaw-v0': [
        '626-Seesaw-31-CW-main_analysis',
        '626-Seesaw-41-GP-main_analysis',
        '626-Seesaw-51-NG-main_analysis',
        '626-Seesaw-61-A8-main_analysis',
        '626-Seesaw-71-LT-main_analysis',
        ]
    }

def get_cmds():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--env')
    parser.add_argument('--cd', type=int)
    parser.add_argument('--num-eval', type=int, default=125)
    args = parser.parse_args()


    methods = ['analysis' for _ in range(len(seed_names[args.env]))]
    cmds = [load_seed(seed, args.env, args.cd, args.num_eval, False, 'lg')
            for seed in seed_names[args.env]]
    return cmds, seed_names[args.env], methods, args.env

if __name__ == '__main__':
    cmds, names, methods, env = get_cmds()
    final_results = {}

    for angle in [15, 12, 9, 6, 3]:
        use_cmds = [cmd + ' --analysis-angle %i' % angle for cmd in cmds]

        combined = list(zip(methods, names, use_cmds))

        results = get_results(combined, True, False, 'lg')
        print('angle %i results' % angle)
        print(results)

        final_results[angle] = results

    RESULTS_DIR = osp.join('./analysis/outcomes/', env)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    path = osp.join(RESULTS_DIR, 'analysis_angle.o')
    print(final_results)
    with open(path, 'wb') as f:
        pickle.dump(final_results, f)
