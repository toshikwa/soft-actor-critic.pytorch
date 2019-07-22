import os
import datetime
import itertools
import numpy as np
import gym
import torch
from tensorboardX import SummaryWriter

from sac import SAC
from memory import ReplayMemory
from vis import plot_return_history


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.dirname(CODE_DIR)


class Trainer():

    def __init__(self, args):
        # environment
        self.env = gym.make(args.env_name)

        # seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.env.seed(args.seed)

        # agent
        self.agent = SAC(
            self.env.observation_space, self.env.action_space, args)

        # logdir
        if args.logdir == "":
            today = datetime.datetime.now().strftime("%Y%m%d")
            self.logdir = os.path.join(
                HOME_DIR, 'logs', args.env_name,
                today, args.tag)
        else:
            self.logdir = args.logdir

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        # writer
        self.writer = SummaryWriter(logdir=os.path.join(self.logdir, 'tb'))
        # replay memory
        self.memory = ReplayMemory(args.replay_size)

        # return history
        self.mean_return_history = np.array([], dtype=np.float)
        self.std_return_history = np.array([], dtype=np.float)

        # train configs
        self.args = args
        # training steps
        self.total_numsteps = 0
        # update counts
        self.updates = 0

    def update(self):
        if len(self.memory) > self.args.batch_size:
            # number of updates per step
            for _ in range(self.args.updates_per_step):
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha =\
                    self.agent.update_parameters(
                        self.memory, self.args.batch_size, self.updates)

                self.writer.add_scalar(
                    'loss/critic_1', critic_1_loss, self.updates)
                self.writer.add_scalar(
                    'loss/critic_2', critic_2_loss, self.updates)
                self.writer.add_scalar(
                    'loss/policy', policy_loss, self.updates)
                self.writer.add_scalar(
                    'loss/entropy_loss', ent_loss, self.updates)
                self.writer.add_scalar(
                    'entropy_temprature/alpha', alpha, self.updates)
                self.updates += 1

    def train_episode(self, episode):
        # rewards
        episode_reward = 0
        # steps
        episode_steps = 0
        # done
        done = False
        # initial state
        state = self.env.reset()

        while not done:
            if self.args.vis:
                self.env.render()

            # take the random action
            if self.args.start_steps > self.total_numsteps:
                action = self.env.action_space.sample()
            # sample from the policy
            else:
                action = self.agent.select_action(state)
            # act
            next_state, reward, done, _ = self.env.step(action)
            episode_steps += 1
            episode_reward += reward
            self.total_numsteps += 1

            # ignore the "done" if it comes from hitting the time horizon.
            mask = 1 if episode_steps == self.env._max_episode_steps\
                else float(not done)

            # store in the replay memory
            self.memory.push(state, action, reward, next_state, mask)
            state = next_state

            self.update()
            if self.total_numsteps % self.args.eval_per_steps == 0:
                self.evaluate()

        self.writer.add_scalar('reward/train', episode_reward, episode)
        print(f"Episode: {episode}, "
              f"total numsteps: {self.total_numsteps}, "
              f"episode steps: {episode_steps}, "
              f"total updates: {self.updates}, "
              f"reward: {round(episode_reward, 2)}")

    def evaluate(self):
        # evaluate
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            while not done:
                if self.args.vis:
                    self.env.render()
                action = self.agent.select_action(state, eval=True)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

            returns[i] = episode_reward

        # mean return
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        self.writer.add_scalar(
            'avg_reward/test', mean_return, self.total_numsteps)

        print("----------------------------------------")
        print(f"Num steps: {self.total_numsteps}, "
              f"Test Return: {round(mean_return, 2)}"
              f" +/- {round(std_return, 2)}")
        print("----------------------------------------")

        # save return
        self.mean_return_history = np.append(
            self.mean_return_history, mean_return)
        self.std_return_history = np.append(
            self.std_return_history, std_return)

        # plot
        plot_return_history(
            self.mean_return_history, self.std_return_history,
            os.path.join(self.logdir, 'test_rewards.png'),
            self.args.env_name, self.args.eval_per_steps)

    def train(self):
        # iterate until convergence
        for episode in itertools.count(1):
            # train
            self.train_episode(episode)
            if self.total_numsteps > self.args.num_steps:
                break

        self.agent.save_model(
            os.path.join(self.logdir, "model"))

    def test(self):
        self.agent.load_model(os.path.join(self.logdir, "model"))
        state = self.env.reset()
        done = False
        while not done:
            self.env.render()
            action = self.agent.select_action(state, eval=True)
            next_state, _, done, _ = self.env.step(action)
            state = next_state

    def __del__(self):
        self.env.close()
