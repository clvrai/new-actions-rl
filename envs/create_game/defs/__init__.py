from ..levels.lvl_config import register_json_folder, setup_class_lvls
import os.path as osp

def setup_def_lvls():
    register_json_folder(osp.dirname(osp.abspath(__file__)))
setup_def_lvls()
