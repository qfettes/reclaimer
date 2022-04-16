from __future__ import absolute_import
from math import log

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from networks.layers import NoisyLinear
from networks.network_bodies import (AtariBody, AtariBodyAC, SimpleBody,
                                     SimpleBodyAC)


### Atari/Mujoco Nets ###
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy_nets=False, noisy_sigma=0.5, body=SimpleBody):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy_nets

        self.body = body(input_shape, num_actions, self.noisy, noisy_sigma)

        self.fc1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, noisy_sigma)
        self.fc2 = nn.Linear(512, self.num_actions) if not self.noisy else NoisyLinear(
            512, self.num_actions, noisy_sigma)

    def forward(self, x):
        x = self.body(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy_nets=False, noisy_sigma=0.5, body=SimpleBody):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.noisy = noisy_nets

        self.body = body(input_shape, num_outputs, noisy, noisy_sigma)

        self.adv1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, noisy_sigma)
        self.adv2 = nn.Linear(512, self.num_actions) if not self.noisy else NoisyLinear(
            512, self.num_actions, noisy_sigma)

        self.val1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, noisy_sigma)
        self.val2 = nn.Linear(512, 1) if not self.noisy else NoisyLinear(
            512, 1, noisy_sigma)

    def forward(self, x):
        x = self.body(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()


LOG_STD_MAX = 2
LOG_STD_MIN = -20
class Actor_SAC(nn.Module):
    def __init__(self, input_shape, action_space, hidden_dim=256, noisy_nets = False, noisy_sigma=0.5):
        super().__init__()

        self.noisy = noisy_nets
        num_outputs = action_space.shape[0]

        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2., dtype=torch.float)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2., dtype=torch.float)

        self.fc1 = nn.Linear(input_shape[0], hidden_dim) if not self.noisy else NoisyLinear(input_shape[0], hidden_dim, noisy_sigma)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) if not self.noisy else NoisyLinear(hidden_dim, hidden_dim, noisy_sigma)

        self.actor_mean = nn.Linear(hidden_dim, num_outputs) if not self.noisy else NoisyLinear(hidden_dim, num_outputs, noisy_sigma)
        self.actor_log_std = nn.Linear(hidden_dim, num_outputs) if not self.noisy else NoisyLinear(hidden_dim, num_outputs, noisy_sigma)

        if self.noisy:
            self.sample_noise()

    def forward(self, obs, deterministic=False, with_logprob=True):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mean, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            action = mean
        else:
            action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # See appendix C of original paper
            logprob = pi_distribution.log_prob(action).sum(axis=-1, keepdim=True)
            logprob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1, keepdim=True)
        else:
            logprob = None

        action = torch.tanh(action)
        action = self.action_scale * action + self.action_bias

        return action, logprob

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.forward(obs, deterministic, False)
            return a.cpu().numpy()

    def sample_noise(self):
        if self.noisy:
            self.fc1.sample_noise()
            self.fc2.sample_noise()
            self.actor_mean.sample_noise()
            self.actor_log_std.sample_noise()

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        super().to(device)

class DQN_SAC(nn.Module):
    def __init__(self, input_shape, action_space, hidden_dim=256, noisy_nets=False, noisy_sigma=0.5):
        super().__init__()

        self.noisy = noisy_nets

        num_inputs = input_shape[0] + action_space.shape[0]

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim) if not self.noisy else NoisyLinear(num_inputs, hidden_dim, noisy_sigma)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) if not self.noisy else NoisyLinear(hidden_dim, hidden_dim, noisy_sigma)
        self.linear3 = nn.Linear(hidden_dim, 1) if not self.noisy else NoisyLinear(hidden_dim, 1, noisy_sigma)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim) if not self.noisy else NoisyLinear(num_inputs, hidden_dim, noisy_sigma)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim) if not self.noisy else NoisyLinear(hidden_dim, hidden_dim, noisy_sigma)
        self.linear6 = nn.Linear(hidden_dim, 1) if not self.noisy else NoisyLinear(hidden_dim, 1, noisy_sigma)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

    def sample_noise(self):
        if self.noisy:
            self.linear1.sample_noise()
            self.linear2.sample_noise()
            self.linear3.sample_noise()
            self.linear4.sample_noise()
            self.linear5.sample_noise()
            self.linear6.sample_noise()


### DSB Nets ###
#tmp: keep as transformer
class Actor_SAC_DSB(nn.Module):
    def __init__(self, input_shape, action_space, hidden_dim=256, nhead=4, transformer_layers=1, transformer_dropout=0.1, noisy_nets = False, noisy_sigma=0.5, dvfs=False):
        super().__init__()

        self.noisy = noisy_nets
        num_outputs = action_space.shape[0]
        self.dvfs = dvfs
        conv_out=8

        # self.action_scale = torch.tensor((1.0 - 0.0) / 2., dtype=torch.float)
        # self.action_bias = torch.tensor((1.0 + 0.0) / 2., dtype=torch.float)
        self.action_scale = torch.tensor((1.0 - 0.0) / 2., dtype=torch.float)
        self.action_bias = torch.tensor((1.0 + 0.0) / 2., dtype=torch.float)

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('relu'))

        # self.conv1 = init_(nn.Conv1d(input_shape[1], 8, kernel_size=1, stride=1))
        # self.conv2 = init_(nn.Conv1d(8, conv_out, kernel_size=1, stride=1))
        # self.conv1 = init_(nn.Conv1d(input_shape[1], 8, kernel_size=input_shape[-1], stride=1))
        # self.conv2 = init_(nn.Conv1d(8, conv_out, kernel_size=input_shape[-1], stride=1))

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('relu'),
                                             noisy_layer=self.noisy)

        # self.cpu_fc1 = init_(nn.Linear(conv_out*input_shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(input_shape[0], hidden_dim, noisy_sigma))
        self.cpu_fc1 = init_(nn.Linear(input_shape[-2]*input_shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(input_shape[0], hidden_dim, noisy_sigma)) 
        self.cpu_fc2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
        self.cpu_fc3 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
        self.cpu_fc4 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
        
        # final_norm = nn.LayerNorm(hidden_dim)
        # TODO:  the layer_norm operations after self attention and the first 
        #   fc layer are still in here. This includes 2 feedforward layers AFTER self attenction
        #   Also include the residual connection. No ff layers before attn
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=transformer_dropout, batch_first=True)
        # TODO: Include a normalization layer after encoder?
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0), gain=0.01,
                                             noisy_layer=self.noisy)

        # self.cpu_log_alpha = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))
        # self.cpu_log_beta = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))

        self.actor_mean = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))
        self.actor_log_std = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))

        if self.dvfs:
            def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('relu'),
                                             noisy_layer=self.noisy)

            self.dvfs_embedding = init_(nn.Linear(1, hidden_dim)) if not self.noisy else init_(NoisyLinear(1, hidden_dim, noisy_sigma))
            self.dvfs_fc1 = init_(nn.Linear(hidden_dim+1, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim+1, hidden_dim, noisy_sigma))

            def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0), gain=0.01,
                                             noisy_layer=self.noisy)

            self.dvfs_mean = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))
            self.dvfs_log_std = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))

        if self.noisy:
            self.sample_noise()

    def forward(self, obs, deterministic=False, with_logprob=True, writer=None, tstep=0, mask=None):
        inp_shape = obs.shape

        if mask is not None:
            mask = mask.to(torch.float)

        # #convolve over the time dimension)
        # emb = F.relu(self.conv1(obs.reshape((-1,)+inp_shape[2:])))
        # emb = F.relu(self.conv2(emb))
        # emb = emb.reshape((inp_shape[:2])+(-1,))

        emb = obs.view(inp_shape[:-2]+(-1,))

        # embedding network
        emb = F.relu(self.cpu_fc1(emb))
        emb = F.relu(self.cpu_fc2(emb)) + emb
        emb = F.relu(self.cpu_fc3(emb)) + emb
        emb = F.relu(self.cpu_fc4(emb)) + emb

        if self.dvfs:
            dvfs_emb = torch.ones(emb.shape[0:1]+(1,1), dtype=torch.float, device=obs.device)
            dvfs_emb = self.dvfs_embedding(dvfs_emb)
            emb = torch.cat([emb, dvfs_emb], dim=-2)

        # self attention
        attn_out = F.relu(self.self_attn(emb))

        if self.dvfs:
            dvfs_attn_out = attn_out[:,-1,:]
            attn_out = attn_out[:,:-1,:]

        mean = self.actor_mean(attn_out)
        
        log_std = self.actor_log_std(attn_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mean, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            action = mean
        else:
            action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # See appendix C of original paper
            logprob = pi_distribution.log_prob(action).squeeze(dim=-1)
            logprob -= (2*(np.log(2) - action - F.softplus(-2*action))).squeeze(dim=-1)

            if mask is not None:
                logprob = logprob * mask
                logprob = logprob.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
            else:
                logprob = logprob.mean(dim=1, keepdim=True)
        else:
            logprob = None

        #append the dvfs action as a new column
        if self.dvfs:
            dvfs_inp = torch.cat(
                [
                    dvfs_attn_out,
                    torch.sum(mean.squeeze(dim=-1), dim=-1, keepdim=True) / 200.
                ],
                dim=-1
            )
            dvfs_inp = F.relu(self.dvfs_fc1(dvfs_inp))

            dvfs_mu = self.dvfs_mean(dvfs_inp)
            dvfs_log_sigma = self.dvfs_log_std(dvfs_inp)
            dvfs_log_sigma = torch.clamp(dvfs_log_sigma, LOG_STD_MIN, LOG_STD_MAX)
            dvfs_sigma = torch.exp(dvfs_log_sigma)

            dvfs_pi_distribution = Normal(dvfs_mu, dvfs_sigma)
            if deterministic:
                # Only used for evaluating policy at test time.
                dvfs_action = dvfs_mu
            else:
                dvfs_action = dvfs_pi_distribution.rsample()

            if with_logprob:
                # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
                # See appendix C of original paper
                dvfs_logprob = dvfs_pi_distribution.log_prob(dvfs_action)
                dvfs_logprob -= (2*(np.log(2) - dvfs_action - F.softplus(-2*dvfs_action)))

                logprob += dvfs_logprob

            action = torch.cat(
                (
                    action, 
                    dvfs_action.expand(-1, action.shape[1]).reshape(-1, action.shape[1], 1)
                ),
                dim=-1
            )

        action = torch.tanh(action)
        action = self.action_scale * action + self.action_bias
        
        # TODO: temporary fix to prevent hotel exploit
        action = action * 0.99 + 0.01

        return action, logprob

        # # compute cpu_action distribution parameters
        # cpu_log_alpha = self.cpu_log_alpha(attn_out)
        # cpu_log_alpha = torch.clamp(cpu_log_alpha, -20, 2)
        # cpu_alpha = torch.exp(cpu_log_alpha)

        # cpu_log_beta = self.cpu_log_beta(attn_out)
        # cpu_log_beta = torch.clamp(cpu_log_beta, -20, 2)
        # cpu_beta = torch.exp(cpu_log_beta)

        # # define distribution and use to select cpu_action
        # cpu_distribution = torch.distributions.beta.Beta(cpu_alpha, cpu_beta)
        # if deterministic:
        #     cpu_action = cpu_distribution.mean
        # else:
        #     cpu_action = cpu_distribution.rsample()
        
        # # get the log_prob of the cpu_action
        # log_prob = None
        # if with_logprob:
        #     log_prob = cpu_distribution.log_prob(cpu_action).squeeze(dim=-1).mean(dim=-1, keepdim=True)
        
        # return cpu_action, log_prob

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.forward(obs, deterministic, False)
            return a.cpu().numpy()
    
    def layer_init(self, module, weight_init, bias_init, gain=1, noisy_layer=False):
        if not noisy_layer:
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
        else:
            weight_init(module.weight_mu.data, gain=gain)
            bias_init(module.bias_mu.data)
        return module

    def sample_noise(self):
        if self.noisy:
            self.cpu_fc1.sample_noise()
            self.cpu_fc2.sample_noise()
            self.actor_concentration.sample_noise()

            self.freq_fc1.sample_noise()
            self.freq_fc2.sample_noise()
            self.actor_alpha.sample_noise()
            self.actor_beta.sample_noise()

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        super().to(device)

#tmp: keep as regular net
class Actor_SAC_DSB_NoTrans(nn.Module):
    def __init__(self, input_shape, action_space, hidden_dim=256, nhead=4, transformer_layers=1, transformer_dropout=0.1, noisy_nets = False, noisy_sigma=0.5, dvfs=False):
        super().__init__()

        self.noisy = noisy_nets
        num_outputs = action_space.shape[0]
        self.dvfs = dvfs

        self.action_scale = torch.tensor((1.0 - 0.0) / 2., dtype=torch.float)
        self.action_bias = torch.tensor((1.0 + 0.0) / 2., dtype=torch.float)

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('relu'))

        # self.conv1 = init_(nn.Conv1d(input_shape[1], 8, kernel_size=1, stride=1))
        # self.conv2 = init_(nn.Conv1d(8, conv_out, kernel_size=1, stride=1))
        # self.conv1 = init_(nn.Conv1d(input_shape[1], 8, kernel_size=input_shape[-1], stride=1))
        # self.conv2 = init_(nn.Conv1d(8, conv_out, kernel_size=input_shape[-1], stride=1))

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('relu'),
                                             noisy_layer=self.noisy)

        # self.cpu_fc1 = init_(nn.Linear(conv_out*input_shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(input_shape[0], hidden_dim, noisy_sigma))
        self.cpu_fc1 = init_(nn.Linear(input_shape[-2]*input_shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(input_shape[0], hidden_dim, noisy_sigma)) 
        self.cpu_fc2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
        self.cpu_fc3 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
        self.cpu_fc4 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
        

        self.attn_replace = nn.Sequential(
            init_(nn.Linear(hidden_dim, hidden_dim*4)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim*4, hidden_dim*3)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim*3, hidden_dim)),
            nn.ReLU()
        )


        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0), gain=0.01,
                                             noisy_layer=self.noisy)

        # self.cpu_log_alpha = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))
        # self.cpu_log_beta = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))

        self.actor_mean = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))
        self.actor_log_std = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))

        if self.dvfs:
            def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('relu'),
                                             noisy_layer=self.noisy)

            self.dvfs_embedding = init_(nn.Linear(1, hidden_dim)) if not self.noisy else init_(NoisyLinear(1, hidden_dim, noisy_sigma))
            self.dvfs_fc1 = init_(nn.Linear(hidden_dim+1, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim+1, hidden_dim, noisy_sigma))

            def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0), gain=0.01,
                                             noisy_layer=self.noisy)

            self.dvfs_mean = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))
            self.dvfs_log_std = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))

        if self.noisy:
            self.sample_noise()

    def forward(self, obs, deterministic=False, with_logprob=False, writer=None, tstep=0, mask=None):
        inp_shape = obs.shape

        if mask is not None:
            mask = mask.to(torch.float)

        # #convolve over the time dimension)
        # emb = F.relu(self.conv1(obs.reshape((-1,)+inp_shape[2:])))
        # emb = F.relu(self.conv2(emb))
        # emb = emb.reshape((inp_shape[:2])+(-1,))

        if len(inp_shape) == 4:
            emb = obs.view(inp_shape[:-2]+(-1,))
        elif len(inp_shape) == 3:
            emb = obs.view(inp_shape[:1]+(-1,))
        elif len(inp_shape) == 2:
            emb = obs

        # embedding network
        emb = F.relu(self.cpu_fc1(emb))
        emb = F.relu(self.cpu_fc2(emb)) + emb
        emb = F.relu(self.cpu_fc3(emb)) + emb
        emb = F.relu(self.cpu_fc4(emb)) + emb

        if self.dvfs:
            dvfs_emb = torch.ones(emb.shape[0:1]+(1,1), dtype=torch.float, device=obs.device)
            dvfs_emb = self.dvfs_embedding(dvfs_emb)
            emb = torch.cat([emb, dvfs_emb], dim=-2)

        # self attention replacement
        attn_out = self.attn_replace(emb) + emb

        if self.dvfs:
            dvfs_attn_out = attn_out[:,-1,:]
            attn_out = attn_out[:,:-1,:]

        mean = self.actor_mean(attn_out)
        
        log_std = self.actor_log_std(attn_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mean, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            action = mean
        else:
            action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # See appendix C of original paper
            logprob = pi_distribution.log_prob(action).squeeze(dim=-1)
            logprob -= (2*(np.log(2) - action - F.softplus(-2*action))).squeeze(dim=-1)

            if mask is not None:
                logprob = logprob * mask
                logprob = logprob.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
            else:
                logprob = logprob.mean(dim=1, keepdim=True)
        else:
            logprob = None

        #append the dvfs action as a new column
        if self.dvfs:
            dvfs_inp = torch.cat(
                [
                    dvfs_attn_out,
                    torch.sum(mean.squeeze(dim=-1), dim=-1, keepdim=True) / 200.
                ],
                dim=-1
            )
            dvfs_inp = F.relu(self.dvfs_fc1(dvfs_inp))

            dvfs_mu = self.dvfs_mean(dvfs_inp)
            dvfs_log_sigma = self.dvfs_log_std(dvfs_inp)
            dvfs_log_sigma = torch.clamp(dvfs_log_sigma, LOG_STD_MIN, LOG_STD_MAX)
            dvfs_sigma = torch.exp(dvfs_log_sigma)

            dvfs_pi_distribution = Normal(dvfs_mu, dvfs_sigma)
            if deterministic:
                # Only used for evaluating policy at test time.
                dvfs_action = dvfs_mu
            else:
                dvfs_action = dvfs_pi_distribution.rsample()

            if with_logprob:
                # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
                # See appendix C of original paper
                dvfs_logprob = dvfs_pi_distribution.log_prob(dvfs_action)
                dvfs_logprob -= (2*(np.log(2) - dvfs_action - F.softplus(-2*dvfs_action)))

                logprob += dvfs_logprob

            action = torch.cat(
                (
                    action, 
                    dvfs_action.expand(-1, action.shape[1]).reshape(-1, action.shape[1], 1)
                ),
                dim=-1
            )

        action = torch.tanh(action)
        action = self.action_scale * action + self.action_bias
        
        # TODO: temporary fix to prevent hotel exploit
        action = action * 0.99 + 0.01

        if logprob is not None:
            return action, logprob
        else:
            return action

        # # compute cpu_action distribution parameters
        # cpu_log_alpha = self.cpu_log_alpha(attn_out)
        # cpu_log_alpha = torch.clamp(cpu_log_alpha, -20, 2)
        # cpu_alpha = torch.exp(cpu_log_alpha)

        # cpu_log_beta = self.cpu_log_beta(attn_out)
        # cpu_log_beta = torch.clamp(cpu_log_beta, -20, 2)
        # cpu_beta = torch.exp(cpu_log_beta)

        # # define distribution and use to select cpu_action
        # cpu_distribution = torch.distributions.beta.Beta(cpu_alpha, cpu_beta)
        # if deterministic:
        #     cpu_action = cpu_distribution.mean
        # else:
        #     cpu_action = cpu_distribution.rsample()
        
        # # get the log_prob of the cpu_action
        # log_prob = None
        # if with_logprob:
        #     log_prob = cpu_distribution.log_prob(cpu_action).squeeze(dim=-1).mean(dim=-1, keepdim=True)
        
        # return cpu_action, log_prob

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a = self.forward(obs, deterministic, False)
            return a.cpu().numpy()
    
    def layer_init(self, module, weight_init, bias_init, gain=1, noisy_layer=False):
        if not noisy_layer:
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
        else:
            weight_init(module.weight_mu.data, gain=gain)
            bias_init(module.bias_mu.data)
        return module

    def sample_noise(self):
        if self.noisy:
            self.cpu_fc1.sample_noise()
            self.cpu_fc2.sample_noise()
            self.actor_concentration.sample_noise()

            self.freq_fc1.sample_noise()
            self.freq_fc2.sample_noise()
            self.actor_alpha.sample_noise()
            self.actor_beta.sample_noise()

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        super().to(device)


class DQN_SAC_DSB(nn.Module):
    def __init__(self, input_shape, action_space, hidden_dim=256, nhead=4, transformer_layers=1, transformer_dropout=0.1, noisy_nets=False, noisy_sigma=0.5):
        super().__init__()

        self.noisy = noisy_nets
        conv_out = 8

        # def init_(m): return self.layer_init(m, nn.init.orthogonal_,
        #                                      lambda x: nn.init.constant_(x, 0),
        #                                      nn.init.calculate_gain('relu'),
        #                                      noisy_layer=self.noisy)

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('relu'))

        # self.conv1 = init_(nn.Conv1d(input_shape[1], 8, kernel_size=1, stride=1))
        # self.conv2 = init_(nn.Conv1d(8, conv_out, kernel_size=1, stride=1))
        # self.conv1 = init_(nn.Conv1d(input_shape[1], 8, kernel_size=input_shape[-1], stride=1))
        # self.conv2 = init_(nn.Conv1d(8, conv_out, kernel_size=input_shape[-1], stride=1))

        init_ = lambda x: x

        # self.emb1 = init_(nn.Linear(conv_out*input_shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(input_shape[0], hidden_dim, noisy_sigma))
        self.emb1 = init_(nn.Linear(input_shape[-2]*input_shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(input_shape[0], hidden_dim, noisy_sigma))
        self.emb2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

        ###### Q network 1 ######
        self.fc1 = init_(nn.Linear(hidden_dim+action_space.shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim+2, hidden_dim, noisy_sigma))
        self.fc2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

        #TODO: dropout is disabled, but the layer_norm operations after self attention and the first 
        #   fc layer are still in here. This includes 2 feedforward layers AFTER self attenction
        #   Also include the residual connection. No ff layers before attn
        encoder_layer1 = nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=transformer_dropout, batch_first=True)
        # TODO: Include a normalization layer after encoder?
        self.self_attn1 = nn.TransformerEncoder(encoder_layer1, num_layers=transformer_layers)

        self.fc3 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

        ##### Q network 2 #####
        self.fc5 = init_(nn.Linear(hidden_dim+action_space.shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim+2, hidden_dim, noisy_sigma))
        self.fc6 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

        #TODO: dropout is disabled, but the layer_norm operations after self attention and the first 
        #   fc layer are still in here. This includes 2 feedforward layers AFTER self attenction
        #   Also include the residual connection. No ff layers before attn
        encoder_layer2 = nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=transformer_dropout, batch_first=True)
        # TODO: Include a normalization layer after encoder?
        self.self_attn2 = nn.TransformerEncoder(encoder_layer2, num_layers=transformer_layers)

        self.fc7 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

        ##### Output Layers ######
        # def init_(m): return self.layer_init(m, nn.init.orthogonal_,
        #                                      lambda x: nn.init.constant_(x, 0), gain=1.,
        #                                      noisy_layer=self.noisy)

        self.fc4 = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
        self.fc8 = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

    def forward(self, obs, action, mask=None):
        inp_shape = obs.shape

        if mask is not None:
            mask = mask.to(torch.float).unsqueeze(dim=-1)

        # emb = F.relu(self.conv1(obs.reshape( (-1,)+inp_shape[2:] )))
        # emb = F.relu(self.conv2(emb))
        # emb = emb.reshape((inp_shape[:2])+(-1,))

        emb = obs.view(inp_shape[:-2]+(-1,))

        emb = F.relu(self.emb1(emb))
        emb = F.relu(self.emb2(emb)) + emb
        emb = torch.cat((emb, action), dim=-1)

        x1 = F.relu(self.fc1(emb))
        x1 = F.relu(self.fc2(x1)) + x1
        x1 = F.relu(self.self_attn1(x1))
        x1 = F.relu(self.fc3(x1)) + x1
        x1 = self.fc4(x1)
        
        if mask is not None:
            x1 = x1 * mask
            x1 = torch.sum(x1, axis=-2)
            x1 = x1 / mask.squeeze(dim=-1).sum(dim=-1, keepdim=True)
        else:
            x1 = torch.mean(x1, axis=-2)

        x2 = F.relu(self.fc5(emb))
        x2 = F.relu(self.fc6(x2)) + x2
        x2 = F.relu(self.self_attn2(x2))
        x2 = F.relu(self.fc7(x2)) + x2
        x2 = self.fc8(x2)
        
        if mask is not None:
            x2 = x2 * mask
            x2 = torch.sum(x2, axis=-2)
            x2 = x2 / mask.squeeze(dim=-1).sum(dim=-1, keepdim=True)
        else:
            x2 = torch.mean(x2, axis=-2)

        return x1, x2

    def sample_noise(self):
        if self.noisy:
            self.emb1.sample_noise()
            self.emb2.sample_noise()

            self.fc1.sample_noise()
            self.fc2.sample_noise()
            self.fc3.sample_noise()
            self.fc4.sample_noise()

            self.fc5.sample_noise()
            self.fc6.sample_noise()
            self.fc7.sample_noise()
            self.fc8.sample_noise()

    def layer_init(self, module, weight_init, bias_init, gain=1, noisy_layer=False):
        if not noisy_layer:
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
        else:
            weight_init(module.weight_mu.data, gain=gain)
            bias_init(module.bias_mu.data)
        return module


class DQN_SAC_DSB_NoTrans(nn.Module):
    def __init__(self, input_shape, action_space, hidden_dim=256, nhead=4, transformer_layers=1, transformer_dropout=0.1, noisy_nets=False, noisy_sigma=0.5):
        super().__init__()

        self.noisy = noisy_nets
        conv_out = 8

        # def init_(m): return self.layer_init(m, nn.init.orthogonal_,
        #                                      lambda x: nn.init.constant_(x, 0),
        #                                      nn.init.calculate_gain('relu'),
        #                                      noisy_layer=self.noisy)

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('relu'))

        # self.conv1 = init_(nn.Conv1d(input_shape[1], 8, kernel_size=1, stride=1))
        # self.conv2 = init_(nn.Conv1d(8, conv_out, kernel_size=1, stride=1))
        # self.conv1 = init_(nn.Conv1d(input_shape[1], 8, kernel_size=input_shape[-1], stride=1))
        # self.conv2 = init_(nn.Conv1d(8, conv_out, kernel_size=input_shape[-1], stride=1))

        init_ = lambda x: x

        # self.emb1 = init_(nn.Linear(conv_out*input_shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(input_shape[0], hidden_dim, noisy_sigma))
        self.emb1 = init_(nn.Linear(input_shape[-2]*input_shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(input_shape[0], hidden_dim, noisy_sigma))
        self.emb2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

        ###### Q network 1 ######
        self.fc1 = init_(nn.Linear(hidden_dim+action_space.shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim+2, hidden_dim, noisy_sigma))
        self.fc2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

        self.attn_replace_1 = nn.Sequential(
            init_(nn.Linear(hidden_dim, hidden_dim*4)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim*4, hidden_dim*3)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim*3, hidden_dim)),
            nn.ReLU()
        )

        self.fc3 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

        ##### Q network 2 #####
        self.fc5 = init_(nn.Linear(hidden_dim+action_space.shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim+2, hidden_dim, noisy_sigma))
        self.fc6 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

        self.attn_replace_2 = nn.Sequential(
            init_(nn.Linear(hidden_dim, hidden_dim*4)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim*4, hidden_dim*3)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim*3, hidden_dim)),
            nn.ReLU()
        )

        self.fc7 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

        ##### Output Layers ######
        # def init_(m): return self.layer_init(m, nn.init.orthogonal_,
        #                                      lambda x: nn.init.constant_(x, 0), gain=1.,
        #                                      noisy_layer=self.noisy)

        self.fc4 = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
        self.fc8 = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

    def forward(self, obs, action, mask=None):
        inp_shape = obs.shape

        if mask is not None:
            mask = mask.to(torch.float).unsqueeze(dim=-1)

        # emb = F.relu(self.conv1(obs.reshape( (-1,)+inp_shape[2:] )))
        # emb = F.relu(self.conv2(emb))
        # emb = emb.reshape((inp_shape[:2])+(-1,))

        emb = obs.view(inp_shape[:-2]+(-1,))

        emb = F.relu(self.emb1(emb))
        emb = F.relu(self.emb2(emb)) + emb
        emb = torch.cat((emb, action), dim=-1)

        x1 = F.relu(self.fc1(emb))
        x1 = F.relu(self.fc2(x1)) + x1
        x1 = self.attn_replace_1(x1) + x1
        x1 = F.relu(self.fc3(x1)) + x1
        x1 = self.fc4(x1)
        
        if mask is not None:
            x1 = x1 * mask
            x1 = torch.sum(x1, axis=-2)
            x1 = x1 / mask.squeeze(dim=-1).sum(dim=-1, keepdim=True)
        else:
            x1 = torch.mean(x1, axis=-2)

        x2 = F.relu(self.fc5(emb))
        x2 = F.relu(self.fc6(x2)) + x2
        x2 = self.attn_replace_2(x2) + x2
        x2 = F.relu(self.fc7(x2)) + x2
        x2 = self.fc8(x2)
        
        if mask is not None:
            x2 = x2 * mask
            x2 = torch.sum(x2, axis=-2)
            x2 = x2 / mask.squeeze(dim=-1).sum(dim=-1, keepdim=True)
        else:
            x2 = torch.mean(x2, axis=-2)

        return x1, x2

    def sample_noise(self):
        if self.noisy:
            self.emb1.sample_noise()
            self.emb2.sample_noise()

            self.fc1.sample_noise()
            self.fc2.sample_noise()
            self.fc3.sample_noise()
            self.fc4.sample_noise()

            self.fc5.sample_noise()
            self.fc6.sample_noise()
            self.fc7.sample_noise()
            self.fc8.sample_noise()

    def layer_init(self, module, weight_init, bias_init, gain=1, noisy_layer=False):
        if not noisy_layer:
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
        else:
            weight_init(module.weight_mu.data, gain=gain)
            bias_init(module.bias_mu.data)
        return module


# # NOTE: This version uses a dirichlet and normal to select actions
# #   the normal is squashed and the logprob is adjusted accordingly
# class Actor_SAC_DSB(nn.Module):
#     def __init__(self, input_shape, action_space, hidden_dim=256, nhead=4, transformer_layers=1, transformer_dropout=0.1, noisy_nets = False, noisy_sigma=0.5):
#         super().__init__()

#         self.noisy = noisy_nets
#         num_outputs = action_space.shape[0]

#         def init_(m): return self.layer_init(m, nn.init.orthogonal_,
#                                              lambda x: nn.init.constant_(x, 0),
#                                              nn.init.calculate_gain('relu'),
#                                              noisy_layer=self.noisy)


#         # init_ = lambda x: x

#         self.action_scale = torch.tensor((1.0 - 0.0) / 2., dtype=torch.float)
#         self.action_bias = torch.tensor((1.0 + 0.0) / 2., dtype=torch.float)

#         self.cpu_fc1 = init_(nn.Linear(input_shape[0], hidden_dim)) if not self.noisy else init_(NoisyLinear(input_shape[0], hidden_dim, noisy_sigma))
#         self.cpu_fc2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
#         self.cpu_fc3 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
#         self.cpu_fc4 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
        
        
#         # final_norm = nn.LayerNorm(hidden_dim)
#         #TODO:  the layer_norm operations after self attention and the first 
#         #   fc layer are still in here. This includes 2 feedforward layers AFTER self attenction
#         #   Also include the residual connection. No ff layers before attn
#         encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=transformer_dropout, batch_first=True)
#         # TODO: Include a normalization layer after encoder?
#         self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

#         self.freq_fc1 = init_(nn.Linear(hidden_dim+1, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim+1, hidden_dim, noisy_sigma))
#         self.freq_fc2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
#         def init_(m): return self.layer_init(m, nn.init.orthogonal_,
#                                              lambda x: nn.init.constant_(x, 0), gain=0.01,
#                                              noisy_layer=self.noisy)

#         self.cpu_log_alpha = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))

#         self.freq_mean = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))
#         self.freq_log_std = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))


#         if self.noisy:
#             self.sample_noise()

#     def forward(self, obs, deterministic=False, with_logprob=True):

#         # embedding network
#         emb = F.relu(self.cpu_fc1(obs))
#         emb = F.relu(self.cpu_fc2(emb))
#         emb = F.relu(self.cpu_fc3(emb))
#         emb = F.relu(self.cpu_fc4(emb))

#         # self attention
#         attn_out = F.relu(self.self_attn(emb))

#         # compute cpu_action distribution parameters
#         cpu_log_alpha = self.cpu_log_alpha(attn_out)
#         cpu_log_alpha = torch.clamp(cpu_log_alpha, -20, 2)
#         cpu_alpha = torch.exp(cpu_log_alpha).squeeze()

#         # define distribution and use to select cpu_action
#         cpu_distribution = torch.distributions.dirichlet.Dirichlet(cpu_alpha)
#         if deterministic:
#             cpu_action = cpu_distribution.mean
#         else:
#             cpu_action = cpu_distribution.rsample()
        
#         # get the log_prob of the cpu_action
#         log_prob = None
#         if with_logprob:
#             log_prob = cpu_distribution.log_prob(cpu_action)
#             # scale by the number of microservices, really this is scaling
#             #   the entropy-coef parameter, I'm just doing it here
#             log_prob = log_prob / obs.shape[1] 
#             log_prob = log_prob.unsqueeze(dim=-1)

#         cpu_action = cpu_action.unsqueeze(dim=-1)

#         # define input for freq_action input
#         freq_emb = F.relu(self.freq_fc1(torch.cat((attn_out, cpu_action), dim=-1)))
#         freq_emb = F.relu(self.freq_fc2(freq_emb))

#         freq_mean = self.freq_mean(freq_emb)

#         freq_log_std = self.freq_log_std(freq_emb)
#         freq_log_std = torch.clamp(freq_log_std, -20, 2)
#         freq_std = torch.exp(freq_log_std)


#         freq_distribution = Normal(freq_mean, freq_std)
#         if deterministic:
#             freq_action = freq_distribution.mean()
#         else:
#             freq_action = freq_distribution.rsample()

#         if with_logprob:
#             freq_logprob = freq_distribution.log_prob(freq_action).squeeze().mean(dim=-1, keepdim=True)
#             freq_logprob -= (2*(np.log(2) - freq_action - F.softplus(-2*freq_action))).squeeze().mean(dim=-1, keepdim=True)
#             log_prob += freq_logprob

#         freq_action = torch.tanh(freq_action) * self.action_scale + self.action_bias
#         action = torch.cat((cpu_action, freq_action), dim=-1)
        
#         return action, log_prob

#     def act(self, obs, deterministic=False):
#         with torch.no_grad():
#             a, _ = self.forward(obs, deterministic, False)
#             return a.cpu().numpy()
    
#     def layer_init(self, module, weight_init, bias_init, gain=1, noisy_layer=False):
#         if not noisy_layer:
#             weight_init(module.weight.data, gain=gain)
#             bias_init(module.bias.data)
#         else:
#             weight_init(module.weight_mu.data, gain=gain)
#             bias_init(module.bias_mu.data)
#         return module

#     def sample_noise(self):
#         if self.noisy:
#             self.cpu_fc1.sample_noise()
#             self.cpu_fc2.sample_noise()
#             self.actor_concentration.sample_noise()

#             self.freq_fc1.sample_noise()
#             self.freq_fc2.sample_noise()
#             self.actor_alpha.sample_noise()
#             self.actor_beta.sample_noise()

#     def to(self, device):
#         self.action_scale = self.action_scale.to(device)
#         self.action_bias = self.action_bias.to(device)
#         super().to(device)


# NOTE: This version samples actions from dirichlet and beta distributions
#   Using exponential or relu + const on distribution inputs is a bit tricky,
#   and it seems extra sensitive to the entropy coef. No logprob correction 
#   is needed, though
# class Actor_SAC_DSB(nn.Module):
#     def __init__(self, input_shape, action_space, hidden_dim=256, nhead=4, transformer_layers=1, transformer_dropout=0.1, noisy_nets = False, noisy_sigma=0.5):
#         super().__init__()

#         self.noisy = noisy_nets
#         num_outputs = action_space.shape[0]

#         def init_(m): return self.layer_init(m, nn.init.orthogonal_,
#                                              lambda x: nn.init.constant_(x, 0),
#                                              nn.init.calculate_gain('relu'),
#                                              noisy_layer=self.noisy)


#         # init_ = lambda x: x

#         self.action_scale = torch.tensor((1.0 - 0.0) / 2., dtype=torch.float)
#         self.action_bias = torch.tensor((1.0 + 0.0) / 2., dtype=torch.float)

#         self.cpu_fc1 = init_(nn.Linear(input_shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(input_shape[0], hidden_dim, noisy_sigma))
#         self.cpu_fc2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
#         self.cpu_fc3 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
#         self.cpu_fc4 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
        
        
#         # final_norm = nn.LayerNorm(hidden_dim)
#         #TODO:  the layer_norm operations after self attention and the first 
#         #   fc layer are still in here. This includes 2 feedforward layers AFTER self attenction
#         #   Also include the residual connection. No ff layers before attn
#         encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=transformer_dropout, batch_first=True)
#         # TODO: Include a normalization layer after encoder?
#         self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

#         self.freq_fc1 = init_(nn.Linear(hidden_dim+1, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim+1, hidden_dim, noisy_sigma))
#         self.freq_fc2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

#         self.freq_log_alpha = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))
#         self.freq_log_beta = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))

#         def init_(m): return self.layer_init(m, nn.init.orthogonal_,
#                                              lambda x: nn.init.constant_(x, 0), gain=0.01,
#                                              noisy_layer=self.noisy)

#         self.cpu_log_alpha = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))

#         if self.noisy:
#             self.sample_noise()

#     def forward(self, obs, deterministic=False, with_logprob=True, writer=None, tstep=0):

#         # embedding network
#         emb = F.relu(self.cpu_fc1(obs))
#         emb = F.relu(self.cpu_fc2(emb))
#         emb = F.relu(self.cpu_fc3(emb))
#         emb = F.relu(self.cpu_fc4(emb))

#         # self attention
#         attn_out = F.relu(self.self_attn(emb))

#         # compute cpu_action distribution parameters
#         cpu_log_alpha = self.cpu_log_alpha(attn_out)
#         cpu_log_alpha = torch.clamp(cpu_log_alpha, -20, 2)
#         cpu_alpha = torch.exp(cpu_log_alpha).squeeze(dim=-1)

#         # define distribution and use to select cpu_action
#         cpu_distribution = torch.distributions.dirichlet.Dirichlet(cpu_alpha)
#         if deterministic:
#             cpu_action = cpu_distribution.mean
#         else:
#             cpu_action = cpu_distribution.rsample()
        
#         # get the log_prob of the cpu_action
#         log_prob = None
#         if with_logprob:
#             log_prob = cpu_distribution.log_prob(cpu_action)
#             # scale by the number of microservices, really this is scaling
#             #   the entropy-coef parameter, I'm just doing it here
#             log_prob = log_prob / obs.shape[1] 
#             log_prob = log_prob.unsqueeze(dim=-1)

#         cpu_action = cpu_action.unsqueeze(dim=-1)

#         # define input for freq_action input
#         freq_emb = F.relu(self.freq_fc1(torch.cat((attn_out, cpu_action), dim=-1)))
#         freq_emb = F.relu(self.freq_fc2(freq_emb))

#         freq_log_alpha = self.freq_log_alpha(freq_emb)
#         freq_log_alpha = torch.clamp(freq_log_alpha, -20, 2)
#         freq_alpha = torch.exp(freq_log_alpha)

#         freq_log_beta = self.freq_log_beta(freq_emb)
#         freq_log_beta = torch.clamp(freq_log_beta, -20, 2)
#         freq_beta = torch.exp(freq_log_beta)

#         # freq_alpha = F.relu(self.freq_log_alpha(freq_emb)) + 1e-8
#         # freq_beta = F. relu(self.freq_log_beta(freq_emb)) + 1e-8

#         freq_distribution = torch.distributions.beta.Beta(freq_alpha, freq_beta)
#         if deterministic:
#             freq_action = freq_distribution.mean()
#         else:
#             freq_action = freq_distribution.rsample()

#         if with_logprob:
#             freq_logprob = freq_distribution.log_prob(freq_action).squeeze().mean(dim=-1, keepdim=True)
#             log_prob += freq_logprob
#             if not writer is None:
#                 with torch.no_grad():
#                     writer.add_scalar(
#                         'Policy/Mean Action Prob', torch.exp(log_prob.detach()).mean(), tstep
#                     )

#         action = torch.cat((cpu_action, freq_action), dim=-1)
        
#         return action, log_prob

#     def act(self, obs, deterministic=False):
#         with torch.no_grad():
#             a, _ = self.forward(obs, deterministic, False)
#             return a.cpu().numpy()
    
#     def layer_init(self, module, weight_init, bias_init, gain=1, noisy_layer=False):
#         if not noisy_layer:
#             weight_init(module.weight.data, gain=gain)
#             bias_init(module.bias.data)
#         else:
#             weight_init(module.weight_mu.data, gain=gain)
#             bias_init(module.bias_mu.data)
#         return module

#     def sample_noise(self):
#         if self.noisy:
#             self.cpu_fc1.sample_noise()
#             self.cpu_fc2.sample_noise()
#             self.actor_concentration.sample_noise()

#             self.freq_fc1.sample_noise()
#             self.freq_fc2.sample_noise()
#             self.actor_alpha.sample_noise()
#             self.actor_beta.sample_noise()

#     def to(self, device):
#         self.action_scale = self.action_scale.to(device)
#         self.action_bias = self.action_bias.to(device)
#         super().to(device)

# # NOTE: This version samples both actions from multivariate normal
# #   i.e. linear dependence between action selection. The logprob
# #   is never corrected for the softmax and tanh operations, though.
# class Actor_SAC_DSB(nn.Module):
#     def __init__(self, input_shape, action_space, hidden_dim=256, nhead=4, transformer_layers=1, transformer_dropout=0.1, noisy_nets = False, noisy_sigma=0.5):
#         super().__init__()

#         self.noisy = noisy_nets
#         num_outputs = action_space.shape[0]

#         def init_(m): return self.layer_init(m, nn.init.orthogonal_,
#                                              lambda x: nn.init.constant_(x, 0),
#                                              nn.init.calculate_gain('relu'),
#                                              noisy_layer=self.noisy)


#         # init_ = lambda x: x

#         self.action_scale = torch.tensor((1.0 - 0.0) / 2., dtype=torch.float)
#         self.action_bias = torch.tensor((1.0 + 0.0) / 2., dtype=torch.float)

#         self.cpu_fc1 = init_(nn.Linear(input_shape[-1], hidden_dim)) if not self.noisy else init_(NoisyLinear(input_shape[0], hidden_dim, noisy_sigma))
#         self.cpu_fc2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
#         self.cpu_fc3 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
#         self.cpu_fc4 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
        
        
#         # final_norm = nn.LayerNorm(hidden_dim)
#         #TODO:  the layer_norm operations after self attention and the first 
#         #   fc layer are still in here. This includes 2 feedforward layers AFTER self attenction
#         #   Also include the residual connection. No ff layers before attn
#         encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=transformer_dropout, batch_first=True)
#         # TODO: Include a normalization layer after encoder?
#         self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

#         def init_(m): return self.layer_init(m, nn.init.orthogonal_,
#                                              lambda x: nn.init.constant_(x, 0), gain=0.01,
#                                              noisy_layer=self.noisy)

#         self.cpu_mu = init_(nn.Linear(hidden_dim, 2)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))
#         self.cpu_log_std = init_(nn.Linear(hidden_dim, 4)) if not self.noisy else init_(NoisyLinear(hidden_dim, 1, noisy_sigma))

#         if self.noisy:
#             self.sample_noise()

#     def forward(self, obs, deterministic=False, with_logprob=True, writer=None, tstep=0):
#         # # make the sequence length the first dimension
#         # obs = torch.transpose(obs, 0, 1)

#         # embedding network
#         emb = F.relu(self.cpu_fc1(obs))
#         emb = F.relu(self.cpu_fc2(emb))
#         emb = F.relu(self.cpu_fc3(emb))
#         emb = F.relu(self.cpu_fc4(emb))

#         # self attention
#         attn_out = F.relu(self.self_attn(emb))

#         # compute cpu_action distribution parameters
#         cpu_mean = torch.exp(self.cpu_mu(attn_out))
#         cpu_log_std = self.cpu_log_std(attn_out)

#         cpu_log_std = torch.clamp(cpu_log_std, -20, 2)
#         cpu_std = torch.exp(cpu_log_std)
#         cpu_std = torch.tril(cpu_std.view(cpu_std.shape[:2]+(2,2)))

#         # define distribution and use to select cpu_action
#         cpu_distribution = torch.distributions.multivariate_normal.MultivariateNormal(cpu_mean, scale_tril=cpu_std)
#         if deterministic:
#             cpu_action = cpu_distribution.mean
#         else:
#             cpu_action = cpu_distribution.rsample()
        
#         # get the log_prob of the cpu_action
#         log_prob = None
#         if with_logprob:
#             log_prob = cpu_distribution.log_prob(cpu_action)
#             if not writer is None:
#                 writer.add_scalar(
#                     'Policy/Mean Action Prob', torch.exp(log_prob.detach()).mean(), tstep
#                 )
#             log_prob = log_prob.mean(dim=-1, keepdim=True)

#         action_1 = cpu_action[:,:,0:1]
#         action_2 = cpu_action[:,:,1:2]
#         action = torch.cat(
#             (
#                 F.softmax(action_1, dim=1),
#                 torch.tanh(action_2) * self.action_scale + self.action_bias
#             ),
#             dim=-1
#         )
        
#         return action, log_prob
        

#     def act(self, obs, deterministic=False):
#         with torch.no_grad():
#             a, _ = self.forward(obs, deterministic, False)
#             return a.cpu().numpy()
    
#     def layer_init(self, module, weight_init, bias_init, gain=1, noisy_layer=False):
#         if not noisy_layer:
#             weight_init(module.weight.data, gain=gain)
#             bias_init(module.bias.data)
#         else:
#             weight_init(module.weight_mu.data, gain=gain)
#             bias_init(module.bias_mu.data)
#         return module

#     def sample_noise(self):
#         if self.noisy:
#             self.cpu_fc1.sample_noise()
#             self.cpu_fc2.sample_noise()
#             self.cpu_fc3.sample_noise()
#             self.cpu_fc4.sample_noise()
#             self.cpu_mu.sample_noise()
#             self.cpu_log_std.sample_noise()

#     def to(self, device):
#         self.action_scale = self.action_scale.to(device)
#         self.action_bias = self.action_bias.to(device)
#         super().to(device)


# class DQN_SAC_DSB(nn.Module):
#     def __init__(self, input_shape, action_space, hidden_dim=256, nhead=4, transformer_layers=1, transformer_dropout=0.1, noisy_nets=False, noisy_sigma=0.5):
#         super().__init__()

#         self.noisy = noisy_nets
#         num_inputs = input_shape[0] + action_space.shape[0]

#         # def init_(m): return self.layer_init(m, nn.init.orthogonal_,
#         #                                      lambda x: nn.init.constant_(x, 0),
#         #                                      nn.init.calculate_gain('relu'),
#         #                                      noisy_layer=self.noisy)

#         init_ = lambda x: x

#         self.emb1 = init_(nn.Linear(input_shape[0], hidden_dim)) if not self.noisy else init_(NoisyLinear(input_shape[0], hidden_dim, noisy_sigma))
#         self.emb2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

#         ###### Q network 1 ######
#         self.fc1 = init_(nn.Linear(hidden_dim+2, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim+2, hidden_dim, noisy_sigma))
#         self.fc2 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

#         #TODO: dropout is disabled, but the layer_norm operations after self attention and the first 
#         #   fc layer are still in here. This includes 2 feedforward layers AFTER self attenction
#         #   Also include the residual connection. No ff layers before attn
#         encoder_layer1 = nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=transformer_dropout, batch_first=True)
#         # TODO: Include a normalization layer after encoder?
#         self.self_attn1 = nn.TransformerEncoder(encoder_layer1, num_layers=transformer_layers)

#         self.fc3 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

#         ##### Q network 2 #####
#         self.fc5 = init_(nn.Linear(hidden_dim+2, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim+2, hidden_dim, noisy_sigma))
#         self.fc6 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

#         #TODO: dropout is disabled, but the layer_norm operations after self attention and the first 
#         #   fc layer are still in here. This includes 2 feedforward layers AFTER self attenction
#         #   Also include the residual connection. No ff layers before attn
#         encoder_layer2 = nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=transformer_dropout, batch_first=True)
#         # TODO: Include a normalization layer after encoder?
#         self.self_attn2 = nn.TransformerEncoder(encoder_layer2, num_layers=transformer_layers)

#         self.fc7 = init_(nn.Linear(hidden_dim, hidden_dim)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

#         ##### Output Layers ######
#         # def init_(m): return self.layer_init(m, nn.init.orthogonal_,
#         #                                      lambda x: nn.init.constant_(x, 0), gain=1.,
#         #                                      noisy_layer=self.noisy)

#         self.fc4 = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))
#         self.fc8 = init_(nn.Linear(hidden_dim, 1)) if not self.noisy else init_(NoisyLinear(hidden_dim, hidden_dim, noisy_sigma))

#     def forward(self, state, action):        
#         emb = F.relu(self.emb1(state))
#         emb = F.relu(self.emb2(emb))
#         emb = torch.cat((emb, action), dim=-1)

#         x1 = F.relu(self.fc1(emb))
#         x1 = F.relu(self.fc2(x1))
#         x1 = F.relu(self.self_attn1(x1))
#         x1 = F.relu(self.fc3(x1))
#         x1 = torch.mean(self.fc4(x1), axis=-2)

#         x2 = F.relu(self.fc5(emb))
#         x2 = F.relu(self.fc6(x2))
#         x2 = F.relu(self.self_attn2(x2))
#         x2 = F.relu(self.fc7(x2))
#         x2 = torch.mean(self.fc8(x2), axis=-2)

#         return x1, x2

#     def sample_noise(self):
#         if self.noisy:
#             self.linear1.sample_noise()
#             self.linear2.sample_noise()
#             self.linear3.sample_noise()
#             self.linear4.sample_noise()
#             self.linear5.sample_noise()
#             self.linear6.sample_noise()

#     def layer_init(self, module, weight_init, bias_init, gain=1, noisy_layer=False):
#         if not noisy_layer:
#             weight_init(module.weight.data, gain=gain)
#             bias_init(module.bias.data)
#         else:
#             weight_init(module.weight_mu.data, gain=gain)
#             bias_init(module.bias_mu.data)
#         return module
