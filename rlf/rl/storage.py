import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from gym.spaces import Discrete, Box

def get_shape_for_ac_dist(action_space, args):
    if action_space.__class__.__name__ == 'Discrete':
        action_shape = args.o_dim
    elif action_space.__class__.__name__ == 'Dict' or action_space.__class__.__name__ == 'SampleDict':
        def get_count_for(key, space):
            if isinstance(space, Discrete) and key == 'skip':
                return 1
            elif isinstance(space, Discrete) and key == 'index':
                return args.o_dim
            elif isinstance(space, Box):
                return space.shape[0]
            else:
                raise ValueError('Unrecognized space')
        action_shape = sum([get_count_for(key, space) for (key, space) in zip(action_space.spaces.keys(), action_space.spaces.values())])
    else:
        action_shape = action_space.shape[0]

    return action_shape

def get_shape_for_ac(action_space):
    if action_space.__class__.__name__ == 'Discrete':
        action_shape = 1
    elif action_space.__class__.__name__ == 'Dict' or action_space.__class__.__name__ == 'SampleDict':
        def get_count_for(x):
            if isinstance(x, Discrete):
                return 1
            elif isinstance(x, Box):
                return x.shape[0]
            else:
                raise ValueError()
        # Another example of being hardcoded to logic game.
        # 1 for the discrete action space index selection
        action_shape = sum([get_count_for(x) for x in action_space.spaces.values()])
    else:
        action_shape = action_space.shape[0]

    return action_shape


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


def to_double(inp):
    return inp.double() if (inp is not None and inp.dtype == torch.float32) else inp


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, args,
                 value_dim, add_input_dim):
        self.value_dim = value_dim
        self.args = args

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.add_input = torch.zeros(num_steps + 1, num_processes,
                                     add_input_dim)

        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes,
                                       self.value_dim)
        self.returns = torch.zeros(num_steps + 1, num_processes,
                                   self.value_dim)
        self.action_log_probs = torch.zeros(num_steps, num_processes,
                                            self.value_dim)

        if self.args.recurrent_policy:
            self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes,
                                                       args.state_encoder_hidden_size)
        else:
            self.recurrent_hidden_states = None

        self.dummy_rnn_hs = None

        if args.distance_based:
            action_shape = get_shape_for_ac_dist(action_space, args)
        else:
            action_shape = get_shape_for_ac(action_space)

        self.actions = torch.zeros(num_steps, num_processes, action_shape)

        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
            ind_n = action_space.n

        if args.load_fixed_action_set and 'Create' in args.env_name:
            ind_n = action_space.spaces['index'].n
        elif action_space.__class__.__name__ == 'Dict' or action_space.__class__.__name__ == 'SampleDict':
            ind_n = action_space.spaces['index'].n
        else:
            if action_space.__class__.__name__ == 'Dict' or action_space.__class__.__name__ == 'SampleDict':
                ind_n = action_space.spaces['index'].n
            else:
                ind_n = action_space.n

        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.add_input = self.add_input.to(device)
        if self.args.recurrent_policy:
            self.recurrent_hidden_states = self.recurrent_hidden_states.to(
                device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, add_input):

        self.obs[self.step + 1].copy_(obs)

        self.actions[self.step].copy_(actions)
        self.add_input[self.step + 1].copy_(add_input)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        if self.args.recurrent_policy:
            self.recurrent_hidden_states[self.step +
                                         1].copy_(recurrent_hidden_states)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        if self.args.recurrent_policy:
            self.recurrent_hidden_states[0].copy_(
                self.recurrent_hidden_states[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        # Need to expand the rewards properly for the number of policies that
        # are learning.
        exp_rewards = self.rewards.repeat(1, 1, self.value_dim)

        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = exp_rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] *
                                          gamma * self.masks[step + 1] + exp_rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]
                           ) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = exp_rewards[step] \
                        + gamma * self.value_preds[step + 1] * self.masks[step + 1] \
                        - self.value_preds[step]

                    gae = delta + gamma * \
                        gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + exp_rewards[step]

    def compute_advantages_base(self):
        advantages = self.returns[:-1] - self.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)
        return advantages

    def compute_advantages(self):
        advantages = self.compute_advantages_base()
        return advantages.reshape(-1, self.value_dim)

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            if self.args.recurrent_policy:
                recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                    -1, self.recurrent_hidden_states.size(-1))[indices]
            else:
                recurrent_hidden_states_batch = None

            if self.add_input.shape[-1] > 0:
                add_input_batch = self.add_input[:-1].view(
                    -1, *self.add_input.size()[2:])[indices]
            else:
                add_input_batch = None

            value_preds_batch = self.value_preds[:-
                                                 1].view(-1, self.value_dim)[indices]
            return_batch = self.returns[:-1].view(-1, self.value_dim)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    self.value_dim)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, self.value_dim)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, \
                old_action_log_probs_batch, adv_targ, \
                add_input_batch

    # Only called if args.recurrent_policy is True

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            add_input_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

                if self.add_input.shape[-1] > 0:
                    add_input_batch.append(self.add_input[:-1, ind])
                else:
                    add_input_batch = None

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)
            add_input_batch = None if add_input_batch is None else torch.stack(add_input_batch, 1)


            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N,
                                                         old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            if self.add_input.shape[-1] > 0:
                add_input_batch = _flatten_helper(T, N, add_input_batch)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, add_input_batch

    def get_actions(self):
        actions = self.actions.view(-1, self.actions.size(-1))
        return actions

    def get_masks(self):
        masks = self.masks[:-1].view(-1, 1)
        return masks
