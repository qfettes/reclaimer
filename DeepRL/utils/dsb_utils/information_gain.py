import sys
from collections import deque
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn.modules import transformer
import torch.optim as optim

np.set_printoptions(threshold=sys.maxsize)

import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import itertools
import os
import warnings

class ViolPredictor(nn.Module):
    def __init__(self, input_shape, action_space, hidden_dim=256, nhead=4, transformer_layers=1, transformer_dropout=0.1):
        super().__init__()

        conv_out = 16

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('relu'))

        self.drop = nn.Dropout(p=transformer_dropout)

        # self.conv1 = init_(nn.Conv1d(input_shape[1], 16, kernel_size=1, stride=1))
        # self.conv2 = init_(nn.Conv1d(16, conv_out, kernel_size=1, stride=1))
        # self.conv1 = init_(nn.Conv1d(input_shape[1], 8, kernel_size=input_shape[-1], stride=1))
        # self.conv2 = init_(nn.Conv1d(8, conv_out, kernel_size=input_shape[-1], stride=1))

        # self.emb1 = init_(nn.Linear(conv_out*input_shape[-1], hidden_dim))
        self.emb1 = init_(nn.Linear(input_shape[-2]*input_shape[-1], hidden_dim))
        self.emb2 = init_(nn.Linear(hidden_dim, hidden_dim))

        self.emb3 = init_(nn.Linear(hidden_dim, hidden_dim)) 
        self.emb4 = init_(nn.Linear(hidden_dim, hidden_dim)) 
        

        encoder_layer1 = nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=transformer_dropout, batch_first=True)
        self.self_attn1 = nn.TransformerEncoder(encoder_layer1, num_layers=transformer_layers)

        self.fc1 = init_(nn.Linear(hidden_dim+action_space.shape[-1], hidden_dim))
        self.fc2 = init_(nn.Linear(hidden_dim, hidden_dim))

        def init_(m): return self.layer_init(m, nn.init.orthogonal_,
                                             lambda x: nn.init.constant_(x, 0),
                                             nn.init.calculate_gain('sigmoid'))

        self.fc3 = init_(nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        inp_shape = obs.shape
        
        # #convolve over time dim
        # emb = self.drop(F.relu(self.conv1(obs.reshape( (-1,)+inp_shape[2:] ))))
        # emb = self.drop(F.relu(self.conv2(emb)))
        # emb = emb.reshape((inp_shape[:2])+(-1,))
        emb = obs.reshape(inp_shape[:-2]+(-1,))
        
        emb = self.drop(emb)

        #embedding
        emb = self.drop(F.relu(self.emb1(emb)))
        emb = self.drop(F.relu(self.emb2(emb)) + emb)

        #new
        self.drop(F.relu(self.emb3(emb)) + emb)
        self.drop(F.relu(self.emb4(emb)) + emb)

        #self attention
        x1 = F.relu(self.self_attn1(emb))

        #append the proposed action after attention
        x1 = torch.cat([x1, action], dim=-1)
        x1 = self.drop(F.relu(self.fc1(x1)))
        x1 = self.drop(F.relu(self.fc2(x1)) + x1)
        x1 = torch.sigmoid(self.fc3(x1))

        return x1

    def layer_init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
       
        return module

class ViolationReplayMemory(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._eval_storage = []
        self._train_storage = []
        self._maxsize = int(size)

        self._next_idx = 0
        self._train_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        if self._train_idx >= len(self._train_storage):
            self._train_storage.append(data)
        else:
            self._train_storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._train_idx = (self._train_idx + 1) % self._maxsize

    def _encode_sample(self, idxes, train=False, eval=False):
        obses_t, actions, labels = [], [], []
        for i in idxes:
            if train:
                data = self._train_storage[i]
            elif eval:
                data = self._eval_storage[i]
            else:
                data = self._storage[i]
            obs_t, action, label = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            labels.append(label)

        return (np.array(obses_t), np.array(actions), np.array(labels))

    def sample(self, batch_size, train=False, eval=False):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if (train or eval) and not self._eval_storage:
            self.split_train_test()

        idxes = []
        if train:
            assert(not eval), "You set train and eval to True!"
            idxes = np.random.randint(0, len(self._train_storage), size=batch_size)
        elif eval:
            assert(not train), "You set train and eval to True!"
            idxes = np.random.randint(0, len(self._eval_storage), size=batch_size)
        else:
            idxes = np.random.randint(0, len(self._storage), size=batch_size)
        return self._encode_sample(idxes, train=train, eval=eval)

    def split_train_test(self, batch_size=5000):
        idxes = np.random.randint(0, len(self._storage), size=batch_size)
        self._eval_storage = [self._storage[idx] for idx in idxes]

        self._train_storage = [self._storage[idx] for idx in range(len(self._storage)) if idx not in idxes]

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

class InfoGain():
    def __init__(self, experience_replay_buffer, env=None, config=None, log_dir=None, tb_writer=None):

        self.config = config
        self.log_dir = log_dir
        self.num_feats = env.observation_space.shape
        self.action_space = env.action_space
        self.envs = env
        self.tb_writer = tb_writer
        self.device = torch.device(config.device)

        self.init_networks()

        self.loss_fun = torch.nn.BCELoss()

        self.pending_samples = []
        self.declare_memory(experience_replay_buffer)
        
        # self.modifiers = np.ones(self.envs.action_space.shape[0:1] + (13,), dtype=np.float32) \
        #     + np.array([-0.8, -0.4, -0.2, -0.1, -0.05, -0.01, 0.0, 1/(1.-0.01), 1/(1.-0.05), 1/(1.-0.1), 1/(1.-0.2), 1/(1.-0.4), 1/(1.-0.8)]).reshape(1, -1)
        # self.modifiers = np.ones(self.envs.action_space.shape[0:1] + (13,), dtype=np.float32) \
        #     + np.array([-0.8, -0.4, -0.2, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8]).reshape(1, -1)
        self.modifiers = np.ones(self.envs.action_space.shape[0:1] + (9,), dtype=np.float32) \
            + np.array([-0.8, -0.4, -0.2, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05]).reshape(1, -1)

    def init_networks(self):
        self.declare_networks()

        self.optimizer = optim.Adam(self.predictor.parameters(
        ), lr=self.config.lr, eps=self.config.optim_eps, weight_decay=1e-4)

        # move to correct device
        self.predictor = self.predictor.to(self.device)

        if self.config.inference:
            self.predictor.eval()
        else:
            self.predictor.train()

    def declare_networks(self):
        self.predictor = ViolPredictor(self.num_feats, self.action_space, hidden_dim=self.config.hidden_units, 
            nhead=self.config.transformer_heads, transformer_layers=self.config.transformer_layers, transformer_dropout=0.2)

    def declare_memory(self, experience_replay_buffer):
        self.memory = ViolationReplayMemory(self.config.replay_size)
        for s, a, r, s_ in reversed(experience_replay_buffer._storage):
            self.add_sample(s, a, r, s_)
        self.append_to_replay()

    def add_sample(self, s, a, r, s_):
        self.pending_samples.append((s, a, r, s_))

    def append_to_replay(self):
        threshold = 3
        last_violation = 30
        label = 1

        for s, a, r, s_ in reversed(self.pending_samples):
            last_violation += 1

            if s_ is None:
                last_violation = threshold + 1
                label = 1

            if r < 0:
                last_violation = 0
                label = 0

            if last_violation >= threshold:
                label = 1

            self.memory.push((s, a, label))

    def prep_minibatch(self, mean, std, train=False, eval=False):
        # random transition batch is taken from experience replay memory
        batch_state, batch_action, batch_labels = self.memory.sample(self.config.batch_size, train=train, eval=eval)

        batch_state = torch.from_numpy(batch_state).to(self.device).to(torch.float)

        batch_action = torch.from_numpy(batch_action).to(self.device).to(torch.float)
        batch_action = batch_action.unsqueeze(dim=1) if len(batch_action.shape) == 1 else batch_action

        batch_labels = np.repeat(batch_labels.reshape(-1, 1), batch_state.shape[1], axis=1).reshape(batch_state.shape[0:2]+(1,))
        batch_labels = torch.from_numpy(batch_labels).to(self.device).to(torch.float)

        return (batch_state-mean)/std, batch_action, batch_labels

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_labels = batch_vars

        # estimate
        prediction = self.predictor(batch_state, batch_action)

        loss = self.loss_fun(prediction.view(-1), batch_labels.view(-1))

        return loss

    def update(self, mean, std, train=False):
        batch_vars = self.prep_minibatch(mean, std, train=train)

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self, mean, std, max_updates=10000, from_scratch=True):
        # put the model into training mode
        self.predictor.train()

        offset = 500
        if from_scratch:
            self.memory.split_train_test()
            self.init_networks()
            offset = 1000

        
        eval_vars = self.prep_minibatch(mean, std, eval=True)
        loss_vs_time = deque(maxlen=offset*2)

        progress = tqdm.tqdm(range(0, max_updates), dynamic_ncols=True)
        progress.set_description(f"Update Violation Predictor, i = {0}, Loss (last 1000): {0.0}, Loss (last 10000): {0.0}")

        for prog in progress:
            _ = self.update(mean, std, train=True)

            with torch.no_grad():
                self.predictor.eval()
                eval_loss = self.compute_loss(eval_vars).item()
                self.predictor.train()
            loss_vs_time.append(eval_loss)
            
            if prog % 100 == 0:
                last_2000 = np.mean(loss_vs_time)
                last_1000 = last_2000
                if len(loss_vs_time) > offset-1:
                    last_1000 = np.mean(list(itertools.islice(loss_vs_time, len(loss_vs_time)-offset-1, len(loss_vs_time)-1)))
                    
                progress.set_description(f"Update Violation Predictor, i = {prog}, Loss (last {offset}): {last_1000:.4f}, Loss (last {offset*2}): {last_2000:.4f}")

                if ((last_2000 - last_1000) < 0.0001 and len(loss_vs_time) > int(offset*1.1)):
                    # put the model back in evaluation mode
                    self.predictor.eval()
                    return
        
        # put the model back in evaluation mode
        self.predictor.eval()

    def get_action(self, obs, prev_action, mean, std, explore=True, recovery=False):
        with torch.no_grad():
            X = torch.from_numpy(obs).to(self.device).to(torch.float).view((-1,)+self.num_feats)
            X = (X - mean) / std

            if explore:
                all_predictions = []
                threshold = 1.0 if recovery else 0.5
                for col in range(self.modifiers.shape[1]):
                    test_action = np.clip(prev_action * self.modifiers[:,col:col+1], 0.0, 1.0)
                    test_action = torch.from_numpy(test_action).to(torch.float).to(self.device).view(1, -1, 1)
                    all_predictions.append(self.predictor(X, test_action))
                all_predictions = torch.abs(torch.cat(all_predictions, dim=-1).squeeze() - threshold)
                _, best_mod = torch.min(all_predictions, dim=-1)
                best_mod = best_mod.cpu().numpy().reshape(-1, 1)
                best_mod = np.take_along_axis(self.modifiers, best_mod, axis=1)
                return np.clip(prev_action * best_mod, 0.0, 1.0)
            else:
                self.predictor.eval()

                all_predictions = []
                threshold = 0.9
                for col in range(self.modifiers.shape[1]):
                    test_action = np.clip(prev_action * self.modifiers[:,col:col+1], 0.0, 1.0)
                    test_action = torch.from_numpy(test_action).to(torch.float).to(self.device).view(1, -1, 1)
                    all_predictions.append(self.predictor(X, test_action))
                
                # Get the most likely to meet QoS
                all_predictions = torch.cat(all_predictions, dim=-1).squeeze()
                # _, best_mod = torch.max(all_predictions, dim=-1)
                best_mod = np.zeros((all_predictions.shape[0], 1), dtype=int) + all_predictions.shape[-1] - 1

                # Get the first to meet threshold
                for col in reversed(range(all_predictions.shape[-1])):
                    for row in range(all_predictions.shape[0]):
                        if all_predictions[row, col] >= threshold:
                            best_mod[row, 0] = col

                best_mod = np.take_along_axis(self.modifiers, best_mod, axis=1)

                return np.clip(prev_action * best_mod, 0.0, 1.0)

    def save_w(self):
        torch.save(self.predictor.state_dict(), os.path.join(
            self.log_dir, 'saved_model', 'info_gain_model.dump'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.log_dir, 'saved_model', 'info_gain_optim.dump'))
            

    def load_w(self, logdir):
        fname_model = os.path.join(logdir, 'info_gain_model.dump')
        fname_optim = os.path.join(logdir,  'info_gain_optim.dump')
        if os.path.isfile(fname_model):
            print("Loading Info Gain Net...")
            self.predictor.load_state_dict(torch.load(fname_model))
        if os.path.isfile(fname_optim):
            print("Loading Info Gain Optimizer...")
            self.optimizer.load_state_dict(torch.load(fname_optim))

        
