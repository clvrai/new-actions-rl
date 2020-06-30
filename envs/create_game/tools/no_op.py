"""
    No opertation is taken when this action is called
"""
class NoOp(object):
    def __init__(self, start_pos=None, end_pos=None, friction=None):
        self.shape = None
        self.body = None

    def get_body(self):
        return None

    def add_to_space(self, space):
        return

    def render(self, space, scale=1, anti_alias=False):
        pass

    def set_settings(self, settings):
        self.settings = settings
