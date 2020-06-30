# Register all the tasks

from .levels.lvl_config import register_json_folder, register_json_str
from .defs import setup_def_lvls

from .settings import CreateGameSettings
from .create_action_set import UseSplit
from .levels.create_level_file import CreateLevelFile
from .tool_gen import ToolGenerator

