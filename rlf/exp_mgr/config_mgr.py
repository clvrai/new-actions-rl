import yaml
import os.path as osp

g_settings = None

def get_cached_settings():
    global g_settings
    if g_settings is None:
        cur_dir = osp.dirname(osp.abspath(__file__))
        config_name = osp.join(cur_dir, 'config.yaml')
        with open(config_name) as f:
            g_settings = yaml.load(f)
    return g_settings

def get_prop(name):
    return get_cached_settings()[name]

