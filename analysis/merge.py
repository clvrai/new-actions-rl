import argparse
import os.path as osp
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--cfgs', type=str, default=None,
        help='comma seperated config names')
parser.add_argument('--out', type=str, default=None,
        help='output name')
args = parser.parse_args()

assert args.out is not None

cfgs = args.cfgs.split(',')


all_results_train = {}
all_results_test = {}

RESULTS_DIR = './exp-base/show/results/'
def merge(d1, d2):
    for k, v in d2.items():
        if k in d1:
            raise ValueError('%s already exists' % k)
        d1[k] = v
    return d1


for cfg in cfgs:
    with open(osp.join(RESULTS_DIR, cfg + '.o'), 'rb') as f:
        results_train, results_test = pickle.load(f)

    if results_train is not None:
        all_results_train = merge(all_results_train, results_train)

    all_results_test = merge(all_results_test, results_test)

save_name = osp.join(RESULTS_DIR, args.out + '_final.o')
with open(save_name, 'wb') as f:
    pickle.dump((all_results_train, all_results_test), f)
    print('saved to %s' % save_name)





