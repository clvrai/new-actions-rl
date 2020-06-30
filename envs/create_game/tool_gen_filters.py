import numpy as np
from functools import partial

def get_tool_prop(tool, prop):
    try:
        val = getattr(tool, prop)
    except:
        val = tool.extra_info[prop]
    if 'angle' in prop:
        val = val * 180 / np.pi
    return val



def tool_check(tool, filter_dict):
    if tool.tool_type in filter_dict:
        allowed_props = filter_dict[tool.tool_type]
        for prop, filter_list in allowed_props.items():
            val = get_tool_prop(tool, prop)
            if all([abs(val - y) > 1e-5 for y in filter_list]):
                return False
        # Tool exists. All properties look good!
            else:
                # print(tool.tool_type, prop, val)
                pass

        return True
    return False


def get_tools_from_filters(train_filter_dict, test_filter_dict, all_tools):
    train_tools = map(lambda x: x.tool_id,
            filter(partial(tool_check, filter_dict=train_filter_dict), all_tools))
    test_tools = map(lambda x: x.tool_id,
            filter(partial(tool_check, filter_dict=test_filter_dict), all_tools))

    train_tools = list(train_tools)
    test_tools = list(test_tools)
    return train_tools, test_tools
