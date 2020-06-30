from rlf import EnvInterface


class ActionEnvInterface(EnvInterface):
    def get_play_env_name(self):
        return 'Not existent'

    def setup(self, args, task_id):
        self.task_id = task_id
        self.args = args
        self.args.overall_aval_actions = None

    def env_trans_fn(self, env, set_eval):
        env.update_args(self.args)
        return env

    def get_gt_embs(self):
        raise NotImplemented('GT embs not implemented for this environment')

    def _generic_setup(self, env, set_eval):
        sets = env.get_train_test_action_sets(self.args, env.get_env_name())
        self.train_action_set, self.test_action_set, self.train_test_action_set = sets
        env.set_args(self.args, set_eval)

        self.args.aval_actions = env.get_aval()

        if self.args.fixed_action_set:
            sub_split = env.set_fixed_action_space(self.args, self.args.overall_aval_actions)
            env.is_fixed_action_space = True
            env._set_action_space(sub_split)
            self.args.aval_actions = env.get_aval()
        elif self.args.load_fixed_action_set:
            env.load_training_fixed_set(self.args)
        return env
