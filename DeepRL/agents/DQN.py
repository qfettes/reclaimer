import sys
from collections import deque
from timeit import default_timer as timer

import numpy as np
import torch
import torch.optim as optim
from networks.network_bodies import AtariBody, SimpleBody
from networks.networks import DQN, DuelingDQN
from utils import ExponentialSchedule, LinearSchedule, PiecewiseSchedule
from utils.ReplayMemory import ExperienceReplayMemory, GPUExperienceReplayMemory, PrioritizedReplayMemory

from agents.BaseAgent import BaseAgent

np.set_printoptions(threshold=sys.maxsize)


class Agent(BaseAgent):
    def __init__(self, env=None, config=None, log_dir='/tmp/gym', tb_writer=None,
        valid_arguments=set(), default_arguments={}):

        super(Agent, self).__init__(env=env, config=config, log_dir=log_dir, 
            tb_writer=tb_writer, valid_arguments=valid_arguments,
            default_arguments=default_arguments)
            
        self.config = config
        self.num_feats = env.observation_space.shape
        self.num_actions = env.action_space.n * len(config.adaptive_repeat)
        self.envs = env

        self.declare_networks()

        self.optimizer = optim.Adam(self.q_net.parameters(
        ), lr=self.config.lr, eps=self.config.optim_eps)

        self.loss_fun = torch.nn.SmoothL1Loss(reduction='none')
        # self.loss_fun = torch.nn.MSELoss(reduction='mean')

        # move to correct device
        self.q_net = self.q_net.to(self.device)
        self.target_q_net.to(self.device)

        if self.config.inference:
            self.q_net.eval()
            self.target_q_net.eval()
        else:
            self.q_net.train()
            self.target_q_net.train()

        self.declare_memory()
        self.update_count = 0
        self.nstep_buffer = []

        self.training_priors()

    def declare_networks(self):
        if self.config.dueling_dqn:
            self.q_net = DuelingDQN(self.num_feats, self.num_actions, noisy_nets=self.config.noisy_nets,
                                    noisy_sigma=self.config.noisy_sigma, body=AtariBody)
            self.target_q_net = DuelingDQN(
                self.num_feats, self.num_actions, noisy_nets=self.config.noisy_nets, noisy_sigma=self.config.noisy_sigma, body=AtariBody)
        else:
            self.q_net = DQN(self.num_feats, self.num_actions, noisy_nets=self.config.noisy_nets,
                             noisy_sigma=self.config.noisy_sigma, body=AtariBody)
            self.target_q_net = DQN(self.num_feats, self.num_actions, noisy_nets=self.config.noisy_nets,
                                    noisy_sigma=self.config.noisy_sigma, body=AtariBody)

        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def declare_memory(self):
        if not self.config.gpu_replay:
            if self.config.priority_replay:
                self.memory = PrioritizedReplayMemory(
                    self.config.replay_size, self.config.priority_alpha, self.config.priority_beta_start, self.config.priority_beta_steps)
            else:
                self.memory = ExperienceReplayMemory(self.config.replay_size)
        else:
            assert(not self.config.priority_replay), "Priority Replay not supported with GPU replay"
            self.memory = GPUExperienceReplayMemory(self.config.replay_size, self.device)

    def training_priors(self):
        self.episode_rewards = np.zeros(self.config.nenvs)
        self.last_100_rewards = deque(maxlen=100)

        if len(self.config.eps_end) == 1:
            if self.config.eps_decay[0] > 1.0:
                self.anneal_eps = ExponentialSchedule(
                    self.config.eps_start, self.config.eps_end[0], self.config.eps_decay[0], self.config.max_tsteps)
            else:
                self.anneal_eps = LinearSchedule(
                    self.config.eps_start, self.config.eps_end[0], self.config.eps_decay[0], self.config.max_tsteps)
        else:
            self.anneal_eps = PiecewiseSchedule(
                self.config.eps_start, self.config.eps_end, self.config.eps_decay, self.config.max_tsteps)

        self.prev_observations, self.actions, self.rewards, self.dones = None, None, None, None,
        self.observations = self.envs.reset()

    def append_to_replay(self, s, a, r, s_, t):
        assert(all(s.shape[0] == other.shape[0] for other in (a, r, s_, t))), \
            f"First dim of s {s.shape}, a {a.shape}, r {r.shape}, s_ {s_.shape}, t {t.shape} must be equal."

        # TODO: Naive. This is implemented like rainbow; however, true nstep
        # q learning requires off-policy correction
        self.nstep_buffer.append([s, a, r, s_, t])

        if(len(self.nstep_buffer) < self.config.n_steps):
            return

        T = np.zeros_like(t)
        if self.config.gpu_replay:
            R = torch.zeros_like(r)
        else:
            R = np.zeros_like(r)

        for idx, transition in enumerate(reversed(self.nstep_buffer)):
            exp_ = len(self.nstep_buffer) - idx - 1
            R *= (1.0 - transition[4][0])
            R += (self.config.gamma**exp_) * transition[2]

            T += transition[4]

        S, A, _, _, _ = self.nstep_buffer.pop(0)

        for state, action, reward, next_state, terminal in zip(S, A, R, s_, T):
            next_state = None if terminal >= 1 else next_state
            self.memory.push((state, action, reward, next_state))

    def prep_minibatch(self, tstep):
        # random transition batch is taken from experience replay memory
        data, indices, weights = self.memory.sample(
            self.config.batch_size, tstep)
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = data


        if not self.config.gpu_replay:
            batch_state = torch.from_numpy(batch_state).to(self.device).to(torch.float)
            # don't declare type of actions: this method is re-used for disrete and continuous action spaces
            #   it needs to be flexible to both types
            batch_action = torch.from_numpy(batch_action).to(self.device)
            batch_action = batch_action.unsqueeze(dim=1) if len(batch_action.shape) == 1 else batch_action
            batch_action = batch_action.to(torch.float)
            batch_reward = torch.from_numpy(batch_reward).to(torch.float).to(self.device).unsqueeze(dim=1)

            if not empty_next_state_values:
                non_final_next_states = torch.from_numpy(
                    non_final_next_states).to(self.device).to(torch.float)
        else:
            batch_action = batch_action.unsqueeze(dim=1) if len(batch_action.shape) == 1 else batch_action
            batch_reward = batch_reward.unsqueeze(dim=1)

        non_final_mask = torch.from_numpy(non_final_mask).to(
            self.device).to(torch.bool)
        

        if self.config.priority_replay:
            weights = torch.from_numpy(weights).to(
                self.device).to(torch.float).view(-1, 1)

        batch_state /= self.config.state_norm
        non_final_next_states /= self.config.state_norm

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def compute_loss(self, batch_vars, tstep):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # estimate
        self.q_net.sample_noise()
        current_q_values = self.q_net(batch_state).gather(1, batch_action)

        # target
        with torch.no_grad():
            next_q_values = torch.zeros(
                self.config.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            self.target_q_net.sample_noise()
            if not empty_next_state_values:
                if self.config.double_dqn:
                    max_next_actions = torch.argmax(self.q_net(
                        non_final_next_states), dim=1).view(-1, 1)
                    next_q_values[non_final_mask] = (self.config.gamma**self.config.n_steps) * self.target_q_net(
                        non_final_next_states).gather(1, max_next_actions)
                else:
                    next_q_values[non_final_mask] = (
                        self.config.gamma**self.config.n_steps) * self.target_q_net(non_final_next_states).max(dim=1)[0].view(-1, 1)
            target = batch_reward + next_q_values

        loss = self.loss_fun(current_q_values, target)
        if self.config.priority_replay:
            with torch.no_grad():
                diff = torch.abs(
                    target - current_q_values).squeeze().cpu().numpy().tolist()
                self.memory.update_priorities(indices, diff)
            loss *= weights
        loss = loss.mean()

        # log val estimates
        if self.tb_writer:
            with torch.no_grad():
                self.tb_writer.add_scalar(
                    'Policy/Value Estimate', current_q_values.detach().mean().item(), tstep)
                self.tb_writer.add_scalar(
                    'Policy/Next Value Estimate', target.detach().mean().item(), tstep)

        return loss

    def update_(self, tstep=0):
        batch_vars = self.prep_minibatch(tstep)

        loss = self.compute_loss(batch_vars, tstep)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(), self.config.grad_norm_max)
        self.optimizer.step()

        self.update_target_model()

        # more logging
        if self.tb_writer:
            with torch.no_grad():
                self.tb_writer.add_scalar('Loss/Total Loss', loss.item(), tstep)
                self.tb_writer.add_scalar('Learning/Learning Rate', np.mean(
                    [param_group['lr'] for param_group in self.optimizer.param_groups]), tstep)

                # log weight norm
                weight_norm = 0.
                for p in self.q_net.parameters():
                    param_norm = p.data.norm(2)
                    weight_norm += param_norm.item() ** 2
                weight_norm = weight_norm ** (1./2.)
                self.tb_writer.add_scalar(
                    'Learning/Weight Norm', weight_norm, tstep)

                # log grad_norm
                grad_norm = 0.
                for p in self.q_net.parameters():
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** (1./2.)
                self.tb_writer.add_scalar('Learning/Grad Norm', grad_norm, tstep)

                # log sigma param norm
                if self.config.noisy_nets:
                    sigma_norm = 0.
                    for name, p in self.q_net.named_parameters():
                        if p.requires_grad and 'sigma' in name:
                            param_norm = p.data.norm(2)
                            sigma_norm += param_norm.item() ** 2
                    sigma_norm = sigma_norm ** (1./2.)
                    self.tb_writer.add_scalar(
                        'Policy/Sigma Norm', sigma_norm, tstep)

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if np.random.random() > eps or self.config.noisy_nets:
                X = torch.from_numpy(s).to(self.device).to(
                    torch.float).view((-1,)+self.num_feats)
                X /= self.config.state_norm

                self.q_net.sample_noise()
                return torch.argmax(self.q_net(X), dim=1).cpu().numpy()
            else:
                return np.random.randint(0, self.num_actions, (s.shape[0]))

    def update_target_model(self):
        self.update_count += 1
        self.update_count = int(self.update_count) % int(
            self.config.tnet_update)
        if self.update_count == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def reset_hx(self, idx):
        pass

    def step(self, current_tstep, step=0):
        epsilon = self.anneal_eps(current_tstep)
        if self.tb_writer:
            self.tb_writer.add_scalar('Policy/Epsilon', epsilon, current_tstep)

        self.actions = self.get_action(self.observations, epsilon)

        self.prev_observations = self.observations
        self.observations, self.rewards, self.dones, self.infos = self.envs.step(
            self.actions)

        self.append_to_replay(self.prev_observations, self.actions,
                              self.rewards, self.observations, self.dones.astype(int))

        self.episode_rewards += self.rewards

        for idx, done in enumerate(self.dones):
            if done:
                self.reset_hx(idx)

                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        'Performance/Agent Reward', self.episode_rewards[idx], current_tstep+idx)
                self.episode_rewards[idx] = 0

        for idx, info in enumerate(self.infos):
            if 'episode' in info.keys():
                self.last_100_rewards.append(info['episode']['r'])
                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        'Performance/Environment Reward', info['episode']['r'], current_tstep+idx)
                    self.tb_writer.add_scalar(
                        'Performance/Episode Length', info['episode']['l'], current_tstep+idx)

    def update(self, current_tstep):
        self.update_(current_tstep)
