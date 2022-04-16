from __future__ import absolute_import

import itertools
import os
from collections import deque
from copy import deepcopy
from matplotlib.pyplot import colorbar

import numpy as np
from numpy.ma.extras import mask_cols
import torch
import torch.optim as optim
from networks.networks import DQN_SAC, Actor_SAC, Actor_SAC_DSB_NoTrans, DQN_SAC_DSB_NoTrans

from agents.BaseAgent import BaseAgent
from agents.DQN import Agent as DQN_Agent

from timeit import default_timer as timer
import pickle
import shap

class Agent(DQN_Agent):
    def __init__(self, env=None, config=None, log_dir='/tmp/gym', tb_writer=None,
        valid_arguments=set(), default_arguments={}):
        # NOTE: Calling BaseAgent init instead of DQN. Weird
        # pylint: disable=bad-super-call
        super(Agent.__bases__[0], self).__init__(env=env, config=config,
                         log_dir=log_dir, tb_writer=tb_writer,
                         valid_arguments=valid_arguments,
                         default_arguments=default_arguments)

        self.config = config
        self.computed_normalizing_constants = False

        self.continousActionSpace = False
        if env.action_space.__class__.__name__ == 'Discrete':
            # self.action_space = env.action_space.n * \
            #     len(config.adaptive_repeat)
            ValueError("Discrete Action Spaces are not supported with SAC")
        elif env.action_space.__class__.__name__ == 'Box':
            self.action_space = env.action_space
            self.continousActionSpace = True
        else:
            ValueError('[ERROR] Unrecognized Action Space Type')

        self.num_feats = env.observation_space.shape
        self.envs = env

        self.declare_networks()

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr, eps=self.config.optim_eps)
        self.value_optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr, eps=self.config.optim_eps)

        if self.config.entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).item()
            self.log_entropy_coef = torch.tensor(np.log(self.config.entropy_coef), requires_grad=True, device=self.device)
            self.config.entropy_coef = self.log_entropy_coef.exp().detach()
            self.entropy_optimizer = optim.Adam([self.log_entropy_coef], lr=self.config.lr)

        self.value_loss_fun = torch.nn.MSELoss(reduction='none')

        self.declare_memory()
        self.update_count = 0
        self.nstep_buffer = []

        self.training_priors()

        self.mean = torch.zeros((env.observation_space.shape[-1],), dtype=torch.float, device=self.device)
        self.std = torch.ones((env.observation_space.shape[-1],), dtype=torch.float, device=self.device)
        self.update_times = deque(maxlen=10)  
        self.update_times.append(0.)

    def declare_networks(self):
        if 'dsb' in self.config.env_id:
            # self.policy_net = Actor_SAC_DSB(self.num_feats, self.action_space, hidden_dim=self.config.hidden_units, noisy_nets=self.config.noisy_nets, 
            #     noisy_sigma=self.config.noisy_sigma, nhead=self.config.transformer_heads,
            #     transformer_layers=self.config.transformer_layers, transformer_dropout=self.config.transformer_dropout,
            #     dvfs=self.config.dvfs
            # )
            # self.q_net = DQN_SAC_DSB(self.num_feats, self.action_space, hidden_dim=self.config.hidden_units, noisy_nets=self.config.noisy_nets, 
            #     noisy_sigma=self.config.noisy_sigma, nhead=self.config.transformer_heads,
            #     transformer_layers=self.config.transformer_layers, transformer_dropout=self.config.transformer_dropout
            # )
            self.policy_net = Actor_SAC_DSB_NoTrans(self.num_feats, self.action_space, hidden_dim=self.config.hidden_units, noisy_nets=self.config.noisy_nets, 
                noisy_sigma=self.config.noisy_sigma, nhead=self.config.transformer_heads,
                transformer_layers=self.config.transformer_layers, transformer_dropout=self.config.transformer_dropout,
                dvfs=self.config.dvfs
            )
            self.q_net = DQN_SAC_DSB_NoTrans(self.num_feats, self.action_space, hidden_dim=self.config.hidden_units, noisy_nets=self.config.noisy_nets, 
                noisy_sigma=self.config.noisy_sigma, nhead=self.config.transformer_heads,
                transformer_layers=self.config.transformer_layers, transformer_dropout=self.config.transformer_dropout
            )
            self.target_q_net = deepcopy(self.q_net)
        else:
            self.policy_net = Actor_SAC(self.num_feats, self.action_space, hidden_dim=256, noisy_nets=self.config.noisy_nets, noisy_sigma=self.config.noisy_sigma)
            self.q_net = DQN_SAC(self.num_feats, self.action_space, hidden_dim=256, noisy_nets=self.config.noisy_nets, noisy_sigma=self.config.noisy_sigma)
            self.target_q_net = deepcopy(self.q_net)

        # First layer of protection. Don't compute gradient for target networks
        for p in self.target_q_net.parameters():
            p.requires_grad = False

        # move to correct device
        self.policy_net.to(self.device)
        self.q_net.to(self.device)
        self.target_q_net.to(self.device)

        if self.config.inference:
            self.policy_net.eval()
            self.q_net.eval()
            self.target_q_net.eval()
        else:
            self.policy_net.train()
            self.q_net.train()
            self.target_q_net.train()

        # self.all_named_params = itertools.chain(self.policy_net.named_parameters(), self.q_net.named_parameters())

    def compute_value_loss(self, batch_vars, tstep):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # estimate
        self.q_net.sample_noise()

        # for masking diff length input sequences
        mask = None
        if 'dsb' in self.config.env_id:
            mask = indices
        current_q_values_1, current_q_values_2 = self.q_net(batch_state, batch_action, mask=mask)

        # target
        with torch.no_grad():
            next_action_log_probs = torch.zeros(self.config.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            next_q_values_1 = torch.zeros(self.config.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            next_q_values_2 = torch.zeros(self.config.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)

            self.target_q_net.sample_noise()

            # for masking diff length input sequences
            target_mask = mask[non_final_mask] if mask is not None else None

            next_actions, next_action_log_probs[non_final_mask] = self.policy_net(non_final_next_states, with_logprob=True, mask=target_mask)

            if not empty_next_state_values:
                next_q_values_1[non_final_mask], next_q_values_2[non_final_mask] = self.target_q_net(non_final_next_states, next_actions, mask=target_mask)
                next_q_values = torch.min(next_q_values_1, next_q_values_2)

            target = batch_reward + self.config.gamma**self.config.n_steps * (next_q_values - self.config.entropy_coef * next_action_log_probs)

        loss_q1 = self.value_loss_fun(current_q_values_1, target)
        loss_q2 = self.value_loss_fun(current_q_values_2, target)

        if self.config.priority_replay:
            with torch.no_grad():
                diff = torch.abs(loss_q1 + loss_q2).squeeze().cpu().numpy().tolist()
                self.memory.update_priorities(indices, diff)
            loss_q1 *= weights
            loss_q2 *= weights

        value_loss = loss_q1.mean() + loss_q2.mean()
        if self.config.transfer and tstep > self.config.ewc_start and False:
            # print("value ewc penalty: ", self.config.ewc_lambda / 2. * self.q_ewc.penalty(self.q_net), value_loss)
            value_loss += self.config.ewc_lambda / 2. * self.q_ewc.penalty(self.q_net)

        # log val estimates
        if self.tb_writer:
            with torch.no_grad():
                self.tb_writer.add_scalar('Policy/Value Estimate', torch.cat(
                    (current_q_values_1, current_q_values_2)).detach().mean().item(), tstep)
                self.tb_writer.add_scalar(
                    'Policy/Next Value Estimate', target.detach().mean().item(), tstep)
                self.tb_writer.add_scalar(
                    'Policy/Entropy Coefficient', self.config.entropy_coef, tstep)
                self.tb_writer.add_scalar(
                    'Loss/Value Loss', value_loss.detach().item(), tstep)

        return value_loss

    def compute_policy_loss(self, batch_vars, tstep):
        batch_state, _, _, _, _, _, mask, weights = batch_vars

        # Compute policy loss
        self.policy_net.sample_noise()
        args = (batch_state, False, True, self.tb_writer, tstep, mask) if 'dsb' in self.config.env_id else (batch_state, False, True)
        actions, log_probs = self.policy_net(*args)


        self.q_net.sample_noise()
        q_val1, q_val2 = self.q_net(batch_state, actions, mask=mask)
        q_val = torch.min(q_val1, q_val2)

        policy_loss = (self.config.entropy_coef * log_probs - q_val)
        if self.config.priority_replay:
            policy_loss *= weights
        policy_loss = policy_loss.mean() 
        # if self.config.transfer and tstep > self.config.ewc_start and False:
        #     # print("Policy EWC penalty: ", self.config.ewc_lambda / 2. * self.policy_ewc.penalty(self.policy_net), policy_loss)
        #     policy_loss += self.config.ewc_lambda / 2. * self.policy_ewc.penalty(self.policy_net)

        # log val estimates
        if self.tb_writer:
            with torch.no_grad():
                self.tb_writer.add_scalar(
                    'Loss/Policy Loss', policy_loss.detach().item(), tstep
                )

        return policy_loss, log_probs

    def compute_entropy_loss(self, action_log_probs, weights, tstep):
        entropy_loss = -(self.log_entropy_coef * (action_log_probs + self.target_entropy).detach())
        if self.config.priority_replay:
            entropy_loss *= weights
        entropy_loss = entropy_loss.mean()

        if self.tb_writer:
            with torch.no_grad():
                self.tb_writer.add_scalar(
                    'Loss/Entropy Loss', entropy_loss.detach().item(), tstep)

        return entropy_loss

    def compute_loss(self, batch_vars, tstep):
        #TODO: handle this more elegantly
        if self.mean is not None or self.std is not None:
            batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, mask, weights = batch_vars

            if mask is not None:
                batch_state[mask] = (batch_state[mask] - self.mean) / self.std
                non_final_next_states[ mask[non_final_mask] ] = (non_final_next_states[ mask[non_final_mask] ] - self.mean) / self.std
            else:
                batch_state = (batch_state - self.mean) / self.std
                non_final_next_states = (non_final_next_states - self.mean) / self.std

            batch_vars = batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, mask, weights

        # First run one gradient descent step for Q1 and Q2
        self.value_optimizer.zero_grad()
        loss_q = self.compute_value_loss(batch_vars, tstep)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(), self.config.grad_norm_max)
        self.value_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_net.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.policy_optimizer.zero_grad()
        loss_pi, action_log_probs = self.compute_policy_loss(batch_vars, tstep)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.config.grad_norm_max)
        self.policy_optimizer.step()

        loss_entropy = torch.zeros((1,)).to(self.device)
        if self.config.entropy_tuning:
            self.entropy_optimizer.zero_grad()
            loss_entropy = self.compute_entropy_loss(action_log_probs, batch_vars[-1], tstep)
            loss_entropy.backward()
            torch.nn.utils.clip_grad_norm_(
                [self.log_entropy_coef], self.config.grad_norm_max)
            self.entropy_optimizer.step()

            self.config.entropy_coef = self.log_entropy_coef.exp().detach()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_net.parameters():
            p.requires_grad = True

        # print(loss_pi.detach(), loss_q.detach(), loss_entropy.detach())

        return loss_pi.detach().cpu().item() + loss_q.detach().cpu().item() + loss_entropy.detach().cpu().item()

    def update_(self, tstep=0):
        update_start = timer()

        loss = []
        for _ in range(self.config.update_freq):
            batch_vars = self.prep_minibatch(tstep)

            # NOTE: compute the mask once here for dsb, replaces indices bc we don't do prioritized replay for DSB
            if 'dsb' in self.config.env_id:
                batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars
                if indices is not None:
                    mask = torch.from_numpy(indices).to(self.device).to(torch.bool)
                    batch_vars = batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, mask, weights

            loss.append(self.compute_loss(batch_vars, tstep))

            self.update_target_model()

        # more logging
        if self.tb_writer:
            with torch.no_grad():
                self.tb_writer.add_scalar('Loss/Total Loss', np.mean(loss), tstep)
                self.tb_writer.add_scalar('Learning/Policy Learning Rate', np.mean([param_group['lr'] for param_group in self.policy_optimizer.param_groups]), tstep)
                self.tb_writer.add_scalar('Learning/Value Learning Rate', np.mean([param_group['lr'] for param_group in self.value_optimizer.param_groups]), tstep)

                # log weight norm
                weight_norm = 0.
                for _, p in itertools.chain(self.policy_net.named_parameters(), self.q_net.named_parameters()):
                    param_norm = p.data.norm(2)
                    weight_norm += param_norm.item() ** 2
                weight_norm = weight_norm ** (1./2.)
                self.tb_writer.add_scalar(
                    'Learning/Weight Norm', weight_norm, tstep)

                # log grad_norm
                grad_norm = 0.
                for _, p in itertools.chain(self.policy_net.named_parameters(), self.q_net.named_parameters()):
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** (1./2.)
                self.tb_writer.add_scalar('Learning/Grad Norm', grad_norm, tstep)
                # print(grad_norm)
                # print('*'*20)

                # log sigma param norm
                if self.config.noisy_nets:
                    sigma_norm = 0.
                    for name, p in itertools.chain(self.policy_net.named_parameters(), self.q_net.named_parameters()):
                        if p.requires_grad and 'sigma' in name:
                            param_norm = p.data.norm(2)
                            sigma_norm += param_norm.item() ** 2
                    sigma_norm = sigma_norm ** (1./2.)
                    self.tb_writer.add_scalar(
                        'Policy/Sigma Norm', sigma_norm, tstep)

            update_total_time = timer() - update_start
            self.update_times.append(update_total_time)
            if self.tb_writer:
                self.tb_writer.add_scalar('Learning/Update Time (s)', update_total_time, tstep)

    def get_action(self, obs, deterministic=False, tstep=0):
        self.policy_net.sample_noise()
        X = torch.from_numpy(obs).to(self.device).to(torch.float).view((-1,)+self.num_feats) / self.config.state_norm

        # TODO: handle this more elegantly
        if self.mean is not None or self.std is not None:
            X = (X - self.mean) / self.std

        return self.policy_net.act(X, deterministic).reshape(self.envs.action_space.shape)

    def update_target_model(self):
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.mul_(self.config.polyak_coef)
            target_param.data.add_((1 - self.config.polyak_coef) * param.data)

    # NOTE: Not used for DSB, only used in Atari/Mujoco
    def step(self, current_tstep, step=0):
        if current_tstep < self.config.random_act:
            self.actions = self.envs.action_space.sample()
        else:
            self.actions = self.get_action(self.observations, deterministic=self.config.inference)

        self.prev_observations = self.observations
        self.observations, self.rewards, self.dones, self.infos = self.envs.step(self.actions)

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
            
            # ignore time limit done signal in updates
            if self.config.correct_time_limits and 'bad_transition' in info.keys() and info['bad_transition']:
                self.dones[idx] = False

        self.append_to_replay(self.prev_observations, self.actions.reshape((self.config.nenvs, -1)),
                              self.rewards, self.observations, self.dones.astype(int))

    # NOTE: Not used for DSB, only used in Atari/Mujoco
    def update(self, current_tstep):
        self.update_(current_tstep)

    def save_w(self):
        torch.save(self.policy_net.state_dict(), os.path.join(
            self.log_dir, 'saved_model', 'policy_model.dump'))
        torch.save(self.policy_optimizer.state_dict(), os.path.join(
            self.log_dir, 'saved_model', 'policy_optim.dump'))

        torch.save(self.q_net.state_dict(), os.path.join(
            self.log_dir, 'saved_model', 'value_model.dump'))
        torch.save(self.value_optimizer.state_dict(), os.path.join(
            self.log_dir, 'saved_model', 'value_optim.dump'))

        if self.config.entropy_tuning:
            torch.save(self.log_entropy_coef, os.path.join(
            self.log_dir, 'saved_model', 'entropy_model.dump'))
            torch.save(self.entropy_optimizer, os.path.join(
                self.log_dir, 'saved_model', 'entropy_optim.dump'))

        if 'dsb' in self.config.env_id:
            with open(os.path.join(self.log_dir, 'saved_model', 'mean.dump'), 'wb') as f:
                pickle.dump(
                    self.mean.cpu().numpy(),  f
                )
            
            with open(os.path.join(self.log_dir, 'saved_model', 'std.dump'), 'wb') as f:
                pickle.dump(
                    self.std.cpu().numpy(),  f
                )
            
    def load_w(self, logdir):
        fname_model = os.path.join(logdir, 'policy_model.dump')
        fname_optim = os.path.join(logdir,  'policy_optim.dump')
        if os.path.isfile(fname_model):
            print("Loading Policy Net...")
            self.policy_net.load_state_dict(torch.load(fname_model))
        if os.path.isfile(fname_optim):
            print("Loading Policy Optimizer...")
            self.policy_optimizer.load_state_dict(torch.load(fname_optim))

        fname_model = os.path.join(logdir, 'value_model.dump')
        fname_optim = os.path.join(logdir, 'value_optim.dump')
        if os.path.isfile(fname_model):
            print("Loading Value Net...")
            self.q_net.load_state_dict(torch.load(fname_model))
            self.target_q_net = deepcopy(self.q_net)
        if os.path.isfile(fname_optim):
            print("Loading Value Optimizer...")
            self.value_optimizer.load_state_dict(torch.load(fname_optim))

        if self.config.entropy_tuning:
            fname_model = os.path.join(logdir, 'entropy_model.dump')
            fname_optim = os.path.join(logdir, 'entropy_optim.dump')
            if os.path.isfile(fname_model):
                print("Loading Entropy Net...")
                self.log_entropy_coef = torch.load(fname_model)
                self.config.entropy_coef = self.log_entropy_coef.exp().detach()
            if os.path.isfile(fname_optim): # not exactly right
                print("Loading Entropy Optimizer...")
                # self.entropy_optimizer.load_state_dict(torch.load(fname_optim))
                self.entropy_optimizer = optim.Adam([self.log_entropy_coef], lr=self.config.lr)

        if 'dsb' in self.config.env_id:
            self.computed_normalizing_constants = True

            fname = os.path.join(logdir, 'mean.dump')
            if os.path.isfile(fname):
                print("Loading standardization mean...")
                with open(fname, 'rb') as f:
                    self.mean = torch.as_tensor(pickle.load(f)).to(torch.float).to(self.device)
                    self.mean = self.mean.view(-1)
            else:
                self.computed_normalizing_constants = False
            
            fname = os.path.join(logdir, 'std.dump')
            if os.path.isfile(fname):
                print("Loading standardization std...")
                with open(fname, 'rb') as f:
                    self.std = torch.as_tensor(pickle.load(f)).to(torch.float).to(self.device)
                    self.std = self.std.view(-1)
            else:
                self.computed_normalizing_constants = False

    def dsb_recompute_normalizing_constants(self, prune_features=True):
        self.computed_normalizing_constants = True

        # data, mask = self.memory.sample(min(len(self.memory), 100000), states_only=True)
        # data = data[mask]
        data = self.memory.sample(min(len(self.memory), 100000), states_only=True)

        # NOTE: This changes to all-microservices norm 
        #   instead of per microservice
        data = data.reshape((-1, data.shape[-1]))

        mean = np.mean(data, axis=0).reshape((1, -1))

        if self.envs.dvfs:
            last_element = 21 if prune_features else 28
            mean_0_values = list(range(last_element, mean.shape[-1], last_element)) # util / allocation
            mean_0_values = mean_0_values + [idx-1 for idx in mean_0_values] # util / cap
            mean_0_values = mean_0_values + [idx-2 for idx in mean_0_values] # action 2
            # mean_0_values = mean_0_values + [idx-3 for idx in mean_0_values] # action 1
            # mean_0_values = mean_0_values + [idx-4 for idx in mean_0_values] #core cap
        else:
            last_element = 20 if prune_features else 27
            mean_0_values = list(range(last_element, mean.shape[-1], last_element)) # util / allocation
            mean_0_values = mean_0_values + [idx-1 for idx in mean_0_values] # util / cap
            # mean_0_values = mean_0_values + [idx-2 for idx in mean_0_values] # action 1
            # mean_0_values = mean_0_values + [idx-3 for idx in mean_0_values] #core cap
        
        #TODO: compat mode should addres this
        #   The +1 is to deal with the new QoS feature
        # mean_0_values = mean_0_values + list(range(-(self.envs.controller.num_microservices), 0))
        mean_0_values = mean_0_values + list(range(-(29+19+1), 0))

        mean[:, mean_0_values] = 0.

        std = np.std(data, axis=0).reshape((1, -1))
        std[:, mean_0_values] = 1.

        std[std == 0] = 1.
        mean[std == 0] = 0. 

        self.mean = torch.from_numpy(mean).to(torch.float).to(self.device).unsqueeze(dim=0)
        self.std = torch.from_numpy(std).to(torch.float).to(self.device).unsqueeze(dim=0)

        self.mean = self.mean.view(-1)
        self.std = self.std.view(-1)

    def SHAP(self, ms=0):
        import matplotlib.pyplot as plt
        plt.clf()
        self.policy_optimizer.zero_grad()

        # Make an array of the feature names
        feature_names = [
            'RX Packets', 'RX Bytes', 'TX Packets', 'TX Bytes',
            'RSS', 'Cache Memory', 'Page Faults', 
            'CPU time',
            'IO Bytes', 'IO Services', 
            'RPS', 'FPS', 'Requests', 'Failures',
            '$P_{50\%}$ Latency', '$P_{66\%}$ Latency','$P_{75\%}$ Latency','$P_{80\%}$ Latency','$P_{90\%}$ Latency',
            '$P_{95\%}$ Latency','$P_{98\%}$ Latency','$P_{99\%}$ Latency','$P_{99.9\%}$ Latency','$P_{99.99\%}$ Latency',
            '$P_{99.999\%}$ Latency','Max Latency', 
            'Core Cap', 
            'Previous Action',
            'QoS',
        ]
        for i in range(29+19):
            feature_names.append(f'Microservice {i} Unique ID')
        old_feature_names = [x for x in feature_names]
        feature_names = feature_names + [x+'-3' for x in old_feature_names]
        feature_names = feature_names + [x+'-2' for x in old_feature_names]
        feature_names = feature_names + [x+'-1' for x in old_feature_names]
        feature_names = feature_names + [x for x in old_feature_names]
        for i in range(len(old_feature_names)):
            feature_names[i] = feature_names[i] + '-4'

        old_batch = self.config.batch_size
        self.config.batch_size = 1000
        batch_vars = self.prep_minibatch(0)
        self.config.batch_size = old_batch

        # NOTE: compute the mask once here for dsb
        if 'dsb' in self.config.env_id:
            batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars
            if indices is not None:
                mask = torch.from_numpy(indices).to(self.device).to(torch.bool)
                batch_vars = batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, mask, weights

        #TODO: handle this more elegantly
        if self.mean is not None or self.std is not None:
            batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, mask, weights = batch_vars

            if mask is not None:
                batch_state[mask] = (batch_state[mask] - self.mean) / self.std
                non_final_next_states[ mask[non_final_mask] ] = (non_final_next_states[ mask[non_final_mask] ] - self.mean) / self.std
            else:
                batch_state = (batch_state - self.mean) / self.std
                non_final_next_states = (non_final_next_states - self.mean) / self.std

            batch_vars = batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, mask, weights

        batch_state, _, _, _, _, _, mask, weights = batch_vars
        original_shape = batch_state.shape
        batch_state = batch_state.view(batch_state.shape[:2]+(-1,))

        onems_batch_state = batch_state[:,ms,:]
        
        self.policy_net.sample_noise()
        explainer = shap.DeepExplainer(self.policy_net, onems_batch_state)

        # Compute policy loss
        shap_values = explainer.shap_values(onems_batch_state)

        # ignore some features
        # shap_values = shap_values[:,-len(old_feature_names):]
        # onems_batch_state = onems_batch_state.cpu().numpy()[:,-len(old_feature_names):]
        shap_values = shap_values.reshape((-1, original_shape[-2], original_shape[-1]))

        onems_batch_state = onems_batch_state.cpu().numpy()
        onems_batch_state = onems_batch_state.reshape((-1, original_shape[-2], original_shape[-1]))

        feature_names = feature_names[-len(old_feature_names):]
        shap_values = np.delete(shap_values, (12, 13), axis=2)
        onems_batch_state = np.delete(onems_batch_state, (12, 13), axis=2)
        feature_names = np.delete(feature_names, (12, 13))

        shap.summary_plot(np.sum(shap_values, axis=-2), feature_names=feature_names, plot_size=(5, 5), show=False, plot_type='bar')
        plt.savefig(f'results/feature_analysis/plots/ms{ms}_group_bar.png')

        plt.clf()
        colorbar = True if ms==0 else False
        shap.summary_plot(shap_values[:,-1,:], onems_batch_state[:,-1,:], feature_names=feature_names, plot_size=(5, 5), show=False, max_display=11, color_bar=colorbar)
        plt.savefig(f'results/feature_analysis/plots/ms{ms}_beeswarm.png', dpi=200, bbox_inches='tight')
        
        # plt.clf()
        # shap.bar_plot(shap_values, onems_batch_state, feature_names=feature_names, show=False)
        # plt.savefig(f'results/feature_analysis/plots/ms{ms}_bar.png')

    def SHAP_blackbox(self, input, output, name):
        import matplotlib.pyplot as plt
        plt.clf()
        self.policy_optimizer.zero_grad()

        # Make an array of the feature names
        feature_names = [
            'rx_packets', 'rx_bytex', 'tx_packets', 'tx_bytes',
            'rss', 'cache memory', 'page faults', 
            'cpu time',
            'io bytes', 'ret io serviced', 
            'rps', 'fps', 'request', 'failure',
            '50.0', '66.0','75.0','80.0','90.0',
            '95.0','98.0','99.0','99.9','99.99',
            '99.999','100.0', 
            'core cap', 
            'previous action',
            'QoS',
        ]
        for i in range(29+19):
            feature_names.append(f'Microservice {i} Unique ID')
        old_feature_names = [x for x in feature_names]
        feature_names = feature_names + [x+'-3' for x in old_feature_names]
        feature_names = feature_names + [x+'-2' for x in old_feature_names]
        feature_names = feature_names + [x+'-1' for x in old_feature_names]
        feature_names = feature_names + [x for x in old_feature_names]
        for i in range(len(old_feature_names)):
            feature_names[i] = feature_names[i] + '-4'

        #TODO: handle this more elegantly
        if self.mean is not None or self.std is not None:
            input = (input - self.mean) / self.std
        
        input = input.view((input.shape[0],)+(-1,))
        explainer = shap.DeepExplainer(self.policy_net, input)

        # Compute policy loss
        shap_values = explainer.shap_values(input)

        # ignore some features
        shap_values = shap_values[:,-len(old_feature_names):]
        input = input.cpu().numpy()[:,-len(old_feature_names):]
        feature_names = feature_names[-len(old_feature_names):]
        shap_values = np.delete(shap_values, (12, 13), axis=1)
        input = np.delete(input, (12, 13), axis=1)
        feature_names = np.delete(feature_names, (12, 13))

        shap.summary_plot(shap_values, input, feature_names=feature_names, plot_size=(14, 13))
        plt.savefig(name)


# def init_ewc(self):
#         return
#         old_batch = self.config.batch_size
#         self.config.batch_size = 10000

#         # Get the data an dcompute the masks
#         batch_vars = self.prep_minibatch(0)
#         batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, mask, weights = batch_vars
#         if mask is not None:
#             mask = torch.from_numpy(mask).to(self.device).to(torch.bool)
#             batch_vars = batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, mask, weights
        
#         target = self.compute_q_target_ewc(batch_vars)
#         self.q_ewc = EWC_ActionValue(deepcopy(self.q_net), batch_state, batch_action, target, mask)
#         self.policy_ewc = EWC_Policy(deepcopy(self.policy_net), deepcopy(self.q_net), batch_state, mask, self.config.entropy_coef.detach())
#         self.memory.clear()

#         self.config.batch_size = old_batch

#     def compute_q_target_ewc(self, batch_vars):
#         # Redundant, but I'm just trying to mirror the original code so unpack data you just packed
#         batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, mask, weights = batch_vars

#         # Compute target. This will be called immediately after loading, so target model is identical to 
#         # online network. No need to worry about that part
#         with torch.no_grad():
#             next_action_log_probs = torch.zeros(self.config.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
#             next_q_values_1 = torch.zeros(self.config.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
#             next_q_values_2 = torch.zeros(self.config.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)

#             self.target_q_net.sample_noise()

#             # for masking diff length input sequences
#             target_mask = mask[non_final_mask] if mask is not None else None

#             next_actions, next_action_log_probs[non_final_mask] = self.policy_net(non_final_next_states, with_logprob=True, mask=target_mask)

#             if not empty_next_state_values:
#                 next_q_values_1[non_final_mask], next_q_values_2[non_final_mask] = self.target_q_net(non_final_next_states, next_actions, mask=target_mask)
#                 next_q_values = torch.min(next_q_values_1, next_q_values_2)

#             target = batch_reward + self.config.gamma**self.config.n_steps * (next_q_values - self.config.entropy_coef * next_action_log_probs)

#         return target