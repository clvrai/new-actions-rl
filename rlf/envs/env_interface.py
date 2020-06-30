import numpy as np
import os.path as osp
import re


class EnvInterface(object):
    def __init__(self, args):
        self.args = args

    def setup(self, args, task_id):
        self.task_id = task_id

    def env_trans_fn(self, env, set_eval):
        return env

    def get_special_stat_names(self):
        return []

    def get_render_mode(self):
        return 'rgb_array'

    def mod_render_frames(self, frames, infos, cur_frame):
        pass

    def get_env_option_names(self):
        indv_labels = self.args.action_bank
        if isinstance(indv_labels, list) and len(indv_labels) > 0 and isinstance(indv_labels[0], list):
            indv_labels = [tuple(x) for x in indv_labels]
        label_list = sorted(list(set(indv_labels)))

        return indv_labels, label_list


g_env_interface = {}

def get_module(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def register_env_interface(name, env_interface):
    global g_env_interface
    g_env_interface[name] = env_interface

def get_env_interface(name, verbose=False):
    global g_env_interface
    for k in g_env_interface:
        if re.match(re.compile(k), name):
            class_ = g_env_interface[k]
            if verbose:
                print('Found env interface %s for %s' % (class_, name))
            return class_

    if verbose:
        print('Getting default env interface for ', name)
    return EnvInterface


