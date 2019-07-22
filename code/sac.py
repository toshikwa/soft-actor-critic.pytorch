import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork


class SAC(object):

    def __init__(self, observation_space, action_space, args):
        # device
        self.device = torch.device("cuda" if args.cuda else "cpu")

        # discount factor
        self.gamma = args.gamma
        # target smoothing coefficient
        self.tau = args.tau
        # entropy temprature
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        # ---- critic ---- #
        # network
        self.critic = QNetwork(
            observation_space.shape[0], action_space.shape[0], args.hidden_size
            ).to(device=self.device)
        # target network
        self.critic_target = QNetwork(
            observation_space.shape[0], action_space.shape[0], args.hidden_size
            ).to(self.device)
        # optimizer
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        # copy parameters to the target network
        hard_update(self.critic_target, self.critic)

        # ---- entropy ---- #
        if self.automatic_entropy_tuning is True:
            # target entropy: -|A|
            self.target_entropy = -torch.prod(torch.Tensor(
                action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        # ---- actor ---- #
        # network
        self.policy = GaussianPolicy(
            observation_space.shape[0],
            action_space.shape[0],
            args.hidden_size,
            action_space).to(self.device)
        # optimizer
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, eval=False):
        """ Select the action. """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # stochastic
        if eval is False:
            action, _, _ = self.policy.sample(state)
        # deterministic
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # sample the batch from memory
        state_batch, action_batch, reward_batch,\
            next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        # current state
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next state
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        # reward
        reward_batch = torch.FloatTensor(
            reward_batch).to(self.device).unsqueeze(1)
        # mask
        mask_batch = torch.FloatTensor(
            mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ =\
                self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target =\
                self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)\
                - self.alpha * next_state_log_pi
            next_q_value = reward_batch +\
                mask_batch * self.gamma * (min_qf_next_target)

        # ---- loss of critics ---- #
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        # ---- loss of the actor ---- #
        # re-sample the action
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # update Q1
        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()
        # update Q2
        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        # update pi
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (
                log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(),\
            alpha_loss.item(), alpha_tlogs.item()

    # save model parameters
    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.policy.save(os.path.join(save_dir, 'actor.pth'))
        self.critic.save(os.path.join(save_dir, 'critic.pth'))

    # load model parameters
    def load_model(self, save_dir):
        self.policy.load(os.path.join(save_dir, 'actor.pth'))
        self.critic.load(os.path.join(save_dir, 'critic.pth'))
