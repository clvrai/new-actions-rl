from rlf.rl.distributions import Beta, DiagGaussian, DiagGaussianVariance, MixedDist, ConditionedAuxDist, Categorical
from rlf.rl.model import ActorCritic
from gym.spaces import Discrete, Box
import copy

class BasePolicy(object):
    def __init__(self, args, obs_space, action_space):
        self._set_modified_action_space(action_space)
        self.obs_space = obs_space
        self.args = args
        self.ignore_layers = []

        self._create_actor_critic()

    def _set_modified_action_space(self, action_space):
        self.action_space = copy.deepcopy(action_space)
        if self.args.load_fixed_action_set and self.args.nearest_neighbor:
            train_set_len = len(self.args.training_fixed_action_set)
            if action_space.__class__.__name__ == 'Discrete':
                self.action_space.n = train_set_len
            elif action_space.__class__.__name__ == 'Dict' or action_space.__class__.__name__ == 'SampleDict':
                self.action_space.spaces['index'].n = train_set_len

    def _create_actor_critic(self):
        actor_critic = ActorCritic(
            self.obs_space.shape,
            self.action_space,
            self.args,
            base=None,
            add_input_dim=0)
        self.actor_critic = actor_critic
        actor_critic.dist = self._get_dist(actor_critic.base.output_size)
        if self.args.cuda:
            self.actor_critic = self.actor_critic.cuda()

    def _get_disc_policy(self, num_outputs):
        dist = Categorical(self.actor_critic.base.output_size, num_outputs, self.args)
        return dist

    def _get_cont_policy(self, num_outputs, use_gaussian=False):
        if self.args.use_beta and not use_gaussian:
            # Assumes action space is [-1, 1]
            return Beta(self.actor_critic.base.output_size,
                            num_outputs, self.args.softplus,
                            use_double=self.args.use_dist_double or self.args.use_double)
        elif self.args.fixed_variance:
            return DiagGaussian(self.actor_critic.base.output_size, num_outputs,
                            softplus=self.args.softplus,
                            use_double=self.args.use_dist_double or self.args.use_double,
                            use_mean_entropy=self.args.use_mean_entropy)
        else:
            return DiagGaussianVariance(
                            self.actor_critic.base.output_size, num_outputs,
                            softplus=self.args.softplus,
                            use_double=self.args.use_dist_double or self.args.use_double,
                            use_mean_entropy=self.args.use_mean_entropy)

    def _get_dist(self, hidden_state_dim):
        if self.action_space.__class__.__name__ == "Discrete":
            return self._get_disc_policy(self.action_space.n)

        elif self.action_space.__class__.__name__ == "Box":
            return self._get_cont_policy(self.action_space.shape[0])

        elif self.action_space.__class__.__name__ == 'Dict' or self.action_space.__class__.__name__ == 'SampleDict':
            keys = list(self.action_space.spaces.keys())

            def get_dist(ac):
                if isinstance(ac, Discrete):
                    return self._get_disc_policy(ac.n)
                elif isinstance(ac, Box):
                    return self._get_cont_policy(ac.shape[0]),
                else:
                    raise ValueError('Cannot have nested Dict action spaces')

            ac_values = [self.action_space.spaces[k] for k in sorted(self.action_space.spaces)]

            disc_parts = [self._get_disc_policy(ac.n) for ac in ac_values if isinstance(ac, Discrete)]

            if self.args.conditioned_aux:
                return ConditionedAuxDist(
                        state_size=self.actor_critic.base.output_size,
                        cont_output_size=[ac.shape[0] for ac in ac_values if isinstance(ac, Box)][0],
                        dist_mem=self.args.dist_mem,
                        args=self.args,
                        use_double=self.args.use_dist_double or self.args.use_double)
            else:
                cont_parts = [self._get_cont_policy(ac.shape[0]) for ac in ac_values if isinstance(ac, Box)]
                return MixedDist(
                        disc_parts,
                        cont_parts,
                        self.args)
        else:
            raise NotImplemented('Unrecognized environment action space')

    def get_init_add_input(self, args, evaluate=False):
        pass

    def eval(self):
        self.actor_critic.eval()

    def train(self):
        self.actor_critic.train()


    def get_dim_add_input(self):
        return 0

    def load_actor_from_checkpoint(self, checkpointer):
        def ignore_contains(x):
            for ignore in self.ignore_layers:
                if ignore in x:
                    return True
            return False

        for i, p in enumerate(self.actor_critic.get_policies()):
            if len(self.actor_critic.get_policies()) != 1:
                actor_state = checkpointer.get_key('actor_%i' % i)
            else:
                actor_state = checkpointer.get_key('actor')
            own_state = self.actor_critic.state_dict()
            for name, param in actor_state.items():
                if ignore_contains(name):
                    print('Skipping ', name)
                    continue
                own_state[name].copy_(param)
            print('Copied Model for %i' % i)

    def save_actor_to_checkpointer(self, checkpointer):
        for p in self.actor_critic.get_policies():
            if len(self.actor_critic.get_policies()) == 1:
                suffix = ''
            else:
                suffix = '_%i'
            checkpointer.save_key('actor' + suffix, p.state_dict())

    def load_resume(self, checkpointer):
        pass

    def get_action(self, state, add_input, recurrent_hidden_state,
                   mask, args, network=None, num_steps=None):
        pass

    def get_value(self, inputs, rnn_hxs, masks, action, add_input):
        return self.actor_critic.get_value(inputs, rnn_hxs, masks, action, add_input)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, add_input):
        return self.actor_critic.evaluate_actions(inputs, rnn_hxs, masks,
                action, add_input)

    def get_actor_critic_params(self):
        return self.actor_critic.parameters()

    def get_actor_critic_count(self):
        return len(self.actor_critic.get_policies())


    def get_actor_critic(self):
        return self.actor_critic

    def is_recurrent(self):
        return self.actor_critic.is_recurrent

