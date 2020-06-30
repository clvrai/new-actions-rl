import numpy as np
import os.path as osp
from envs.create_game import ToolGenerator
from collections import defaultdict
import random
from envs.action_env import split_action_set
import copy
from envs.create_game import UseSplit


def split_action_set(test_set, validation_ratio):
    # split in a predictable way
    rng = np.random.RandomState(41)

    ret_set = test_set[:]
    rng.shuffle(ret_set)
    N = int(len(ret_set) * validation_ratio)
    eval_set = ret_set[:N]
    test_set = ret_set[N:]
    return eval_set, test_set


def load_training_fixed_set(args):
    add_str = ('_' + args.split_type) if (args.split_type is not None and 'New' in args.exp_type) else ''

    load_dir = osp.join(args.action_seg_loc, 'create_' + args.exp_type + add_str)
    train_filename = osp.join(load_dir, 'set_train.npy')
    with open(train_filename, 'rb') as f:
        train_set = np.load(f)
    return train_set


def gen_action_set(settings, tool_gen, allowed_actions, rng):
    """
    From an overall set of actions sample some sub-actions.

    - settings: Create run settings
    - tool_gen: Object that contains all possible tools
    - allowed_actions: List of indices that indicate valid actions to be
      sampled from.
    - rng: Random number generator to be used.
    """

    args = settings.action_extra

    sets = []
    no_op_tool = len(tool_gen.tools) - 1

    allowed_actions = np.setdiff1d(allowed_actions, [no_op_tool])

    if args.create_len_filter is not None:
        def should_keep(x):
            if x.length is None:
                return True
            if x.tool_type == 'Bouncy_Ball':
                return (x.length * 2) < args.create_len_filter
            elif x.tool_type == 'Bucket':
                return (x.length * 2) < args.create_len_filter
            elif x.tool_type == 'Fixed_Ball':
                return (x.length * 2) < args.create_len_filter
            elif x.tool_type == 'Funnel':
                return (x.length * 2) < args.create_len_filter
            else:
                return x.length < args.create_len_filter

        allowed_actions = np.array([idx
                                  for idx in allowed_actions
                                  if should_keep(tool_gen.tools[idx])])

    ########################################################
    # Computing various parameter splits
    ########################################################

    if args.sample_clusters and hasattr(args, 'emb_mem'):
        if args.emb_mem.should_compute_clusterings():
            args.emb_mem.compute_clusterings(allowed_actions)

        tool_groups = args.emb_mem.action_groups
    elif args.gt_clusters:
        tool_groups = defaultdict(list)
        for tool_idx in allowed_actions:
            tool_obj = tool_gen.tools[tool_idx]
            tool_groups[tool_obj.tool_type].append(tool_idx)

        if args.strict_gt_clusters:
            #print('strictly using gt clusters')
            ret_tools = strict_gt_hack(tool_groups,
                    settings.action_set_size,
                    no_op_tool)
            #print(['%s: %s' % (x.tool_type, str(x.length))
            #    for x in [tool_gen.tools[t] for t in ret_tools]])
            return ret_tools
    else:
        use_set = rng.choice(allowed_actions, settings.action_set_size, replace=False)
        if not settings.separate_skip:
            use_set[0] = no_op_tool
        return use_set

    group_keys = list(tool_groups.keys())

    set_arr = []
    assert settings.action_set_size <= len(allowed_actions)
    while(len(set_arr) < settings.action_set_size):
        if args.half_tools and args.half_tool_ratio is not None:
            group_keys_samples = random.sample(
                group_keys, int(args.half_tool_ratio * len(group_keys)))
        else:
            group_keys_samples = group_keys[:]
        type_selections = rng.choice(
            group_keys_samples, settings.action_set_size, replace=True)
        for type_sel in type_selections:
            for trial in range(10):
                idx = rng.choice(tool_groups[type_sel])
                if idx not in set_arr:
                    break
            set_arr.append(idx)
            if len(set_arr) == settings.action_set_size:
                break
    tool_set = np.array(set_arr)
    if not settings.separate_skip:
        tool_set[0] = no_op_tool
    return tool_set


def strict_gt_hack(tool_groups, set_size, no_op_tool):
    obj_types = list(tool_groups.keys())
    poly_names = ['Ball', 'Box', 'Triangle', 'Square', 'Pentagon', 'Hexagon']
    poly_types = []
    for obj_type in obj_types:
        found = False
        for poly_name in poly_names:
            if poly_name in obj_type:
                found = True
                break
        if found:
            poly_types.append(obj_type)

    for poly_type in poly_types:
        del obj_types[obj_types.index(poly_type)]

    # CONTROLLS HOW MANY POLYGONS ARE SAMPLED PER COMPLETE SET OF OTHER TOOLS
    for i in range(2):
        obj_types.append('poly')

    use_poly_types = poly_types[:]

    set_arr = [no_op_tool]
    while len(set_arr) < (set_size - 1):
        random.shuffle(obj_types)
        for use_key in obj_types:
            if use_key == 'poly':
                use_key = random.choice(use_poly_types)
                del use_poly_types[use_poly_types.index(use_key)]
                if len(use_poly_types) == 0:
                    use_poly_types = poly_types[:]

            use_idx = random.choice(tool_groups[use_key])
            set_arr.append(use_idx)

    set_arr = np.array(set_arr)
    return set_arr


def get_allowed_actions(settings):
    args = settings.action_extra

    tool_gen = ToolGenerator(settings.gran_factor)

    add_str = ('_' + args.split_type) if (args.split_type is not None and 'New' in args.exp_type) else ''
    load_dir = osp.join(settings.action_seg_loc, 'create_' + args.exp_type + add_str)

    # Load in the test and train splits
    train_filename = osp.join(load_dir, 'set_train.npy')
    with open(train_filename, 'rb') as f:
        train_set = np.load(f)

    test_filename = osp.join(load_dir, 'set_test.npy')
    with open(test_filename, 'rb') as f:
        test_set = np.load(f)

    overall_aval_actions = None

    if settings.split_type == UseSplit.TRAIN_TEST:
        tool_ids = [t.tool_id for t in tool_gen.tools]
        overall_aval_actions = np.sort(np.unique(tool_ids))
        if not (overall_aval_actions == np.arange(len(overall_aval_actions))).all():
            raise ValueError('must include all actions')
    elif settings.split_type == UseSplit.TRAIN:
        overall_aval_actions = train_set
    elif settings.split_type == UseSplit.TEST:
        if args.eval_only and args.split_type == 'analysis':
            # Get a custom test set
            if args.analysis_angle is not None:
                overall_aval_actions = extract_n(
                    test_set, tool_gen, args, accept_angle, get_angles)
            elif args.train_mix_ratio is not None:
                cp_args = copy.copy(args)
                cp_args.analysis_emb = 3.0
                real_test = extract_n(test_set, tool_gen,
                                      cp_args, accept_emb, get_embs)
                test_ratio = args.train_mix_ratio
                train_ratio = 1.0 - test_ratio

                test_len = int(len(real_test) * test_ratio)
                train_len = int(len(train_set) * train_ratio)
                set_acs = [*real_test[:test_len], *train_set[:train_len]]
                overall_aval_actions = set_acs
            else:
                assert args.analysis_emb is not None
                overall_aval_actions = extract_n(test_set, tool_gen, args,
                        accept_emb, get_embs)
        else:
            _, overall_aval_actions = split_action_set(test_set,
                    settings.validation_ratio)
    elif settings.split_type == UseSplit.VALIDATION:
        overall_aval_actions, _ = split_action_set(test_set,
                settings.validation_ratio)
    else:
        raise ValueError('Invalid split type')

    return overall_aval_actions



def get_train_test_action_sets(settings):

    args = settings.action_extra

    tool_gen = ToolGenerator(settings.gran_factor)

    add_str = ('_' + args.split_type) if (args.split_type is not None and 'New' in args.exp_type) else ''
    load_dir = osp.join(settings.action_seg_loc, 'create_' + args.exp_type + add_str)

    # Load in the test and train splits
    train_filename = osp.join(load_dir, 'set_train.npy')
    with open(train_filename, 'rb') as f:
        train_set = np.load(f)

    test_filename = osp.join(load_dir, 'set_test.npy')
    with open(test_filename, 'rb') as f:
        test_set = np.load(f)

    tool_ids = [t.tool_id for t in tool_gen.tools]
    train_test_set = np.sort(np.unique(tool_ids))
    if not (train_test_set == np.arange(len(train_test_set))).all():
        raise ValueError('must include all actions')

    train_set = np.sort(train_set)
    test_set = np.sort(test_set)

    return train_set, test_set, train_test_set


def get_angles(ts, args, tool_gen):
    def get_angle(x):
        if x.angle is None:
            ret = x.extra_info['max_angle']
        else:
            ret = x.angle
        return ret * (180.0 / np.pi)
    return np.array([get_angle(tool_gen.tools[i]) for i in ts])


def get_embs(ts, args, tool_gen):
    ac_embs = args.dist_mem.option_embs

    def get_emb(x):
        return ac_embs[x]

    return np.array([get_emb(i).cpu().numpy() for i in ts])


def accept_emb(diffs, args):
    diffs = np.square(diffs)
    diffs = np.sum(diffs, axis=-1)
    return diffs > args.analysis_emb


def accept_angle(diffs, args):
    return diffs >= args.analysis_angle


def extract_n(train_set, tool_gen, args, get_accept, get_prop):
    """
    Used for the analysis scripts in creating splits on certain property types.
    """
    all_names = {
        'Ramp': {},
        'Trampoline': {},
        'See_Saw': {},
        'Hinge_Constrained': {},
        'Cannon': {},
        'Bucket': {},
        'Fixed_Triangle': {},
        'Bouncy_Triangle': {},
        'Hinge': {},
        'Fan': {},
        'Funnel': {},
        'no_op': {},
    }
    from envs.create_game.tool_gen_filters import get_tools_from_filters

    all_tools, _ = get_tools_from_filters(all_names, all_names, tool_gen.tools)
    all_tools = tool_gen.sub_filter_gran_factor(
        sub_gran_factor=5.0, tool_ids=all_tools)

    tls = all_tools[:]

    np.random.shuffle(tls)

    batch_size = 200

    num_iters = len(tls) // batch_size

    def filter_no_op(ts):
        return [x for x in ts if tool_gen.tools[x].tool_type != 'no_op']

    use_train_set = filter_no_op(train_set)
    tls = filter_no_op(tls)

    if args.analysis_emb is not None:
        train_props = np.expand_dims(
            get_prop(use_train_set, args, tool_gen), -2)
    else:
        train_props = np.expand_dims(
            get_prop(use_train_set, args, tool_gen), -1)

    valid_tools = []
    for i in range(num_iters):
        batch = np.array(tls[i * batch_size: (i+1) * batch_size])
        batch_props = get_prop(batch, args, tool_gen)

        diffs = np.abs(batch_props - train_props)
        keep = get_accept(diffs, args)
        keep = keep.all(axis=0)
        idx = np.nonzero(keep)

        keep_batch = batch[idx]
        valid_tools.extend(keep_batch)

    print('-' * 20)
    print('GOT %i TOOLS' % len(valid_tools))
    print('-' * 20)
    return valid_tools
