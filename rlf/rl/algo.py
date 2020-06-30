import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from method.embedder.utils import tensor_kl_diagnormal_stdnormal
import traceback
from method.radam import RAdam
import math

class pdb_anomaly(torch.autograd.detect_anomaly):
  def __init__(self):
    super().__init__()
  def __enter__(self):
    super().__enter__()
    return self
  def __exit__(self, type, value, trace):
    super().__exit__()
    if isinstance(value, RuntimeError):
      traceback.print_tb(trace)
      print(str(value))
      import ipdb; ipdb.set_trace()

class PPO():
    def __init__(self,
                 policy,
                 args,
                 use_clipped_value_loss=True):

        self.policy = policy
        self.args = args

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch

        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

        self.max_grad_norm = args.max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        if self.args.rl_use_radam:
            self.optimizer = RAdam([
                {'params': policy.get_actor_critic_params()}
                ],
                lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)
        else:
            self.optimizer = optim.Adam([
                {'params': policy.get_actor_critic_params()}
                ],
                lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)

        if self.args.pac_bayes:
            self.a_space = self.policy.get_actor_critic().action_space
            if self.a_space.__class__.__name__ == 'Discrete':
                num_actions = self.a_space.n
            elif self.a_space.__class__.__name__ == 'Dict' or self.a_space.__class__.__name__ == 'SampleDict':
                num_actions = self.a_space.spaces['index'].n
            probs = torch.ones(num_actions, dtype=torch.float64) / num_actions
            probs = probs.cuda() if args.cuda else probs
            self.prior_distribution = torch.distributions.Categorical(probs)

    def load_resume(self, checkpointer):
        self.optimizer.load_state_dict(
            checkpointer.get_key('actor_opt'))

    def save(self, checkpointer):
        checkpointer.save_key('actor_opt', self.optimizer.state_dict())



    def update(self, rollouts):
        advantages = rollouts.compute_advantages_base()

        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_loss_epoch = 0
        dist_entropy_epoch = 0

        log_vals = defaultdict(lambda: 0)

        for e in range(self.ppo_epoch):
            if self.policy.is_recurrent():
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                # Get all the data from our batch sample
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, \
                   old_action_log_probs_batch, adv_targ, \
                   add_input_batch = sample

                eval_part = self.policy.evaluate_actions(obs_batch,
                        recurrent_hidden_states_batch, masks_batch,
                        actions_batch, add_input_batch)

                values, action_log_probs, dist_entropy, _, dists = eval_part

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean(0)

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean(0)
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean(0)

                self.optimizer.zero_grad()


                if self.args.pac_bayes:
                    n_samples = obs_batch.shape[0] * self.args.num_mini_batch
                    if self.a_space.__class__.__name__ == 'Discrete':
                        disc_dist = dists
                    elif self.a_space.__class__.__name__ == 'Dict' or self.a_space.__class__.__name__ == 'SampleDict':
                        disc_dist = dists.discs[0]
                    kld = torch.distributions.kl_divergence(disc_dist, self.prior_distribution).mean()
                    complexity_loss = torch.sqrt((1 / (2 * (n_samples-1))) * (kld + math.log(2 * n_samples / self.args.pac_bayes_delta))).mean(0)
                    complexity_loss = complexity_loss.float()
                else:
                    complexity_loss = 0

                with pdb_anomaly():
                    loss = (value_loss * self.value_loss_coef + action_loss -
                         dist_entropy.mean() * self.entropy_coef + self.args.complexity_scale * complexity_loss)

                    loss = loss.sum()

                    loss.backward()

                nn.utils.clip_grad_norm_(self.policy.get_actor_critic_params(),
                                         self.max_grad_norm)
                self.optimizer.step()

                log_vals['value_loss'] += value_loss.sum().item()
                log_vals['action_loss'] += action_loss.sum().item()
                log_vals['dist_entropy'] += dist_entropy.mean().item()
                if self.args.pac_bayes:
                    log_vals['complexity_loss'] += complexity_loss.mean().item()
                log_vals['overall_loss'] += loss.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in log_vals:
            log_vals[k] /= num_updates

        return log_vals
