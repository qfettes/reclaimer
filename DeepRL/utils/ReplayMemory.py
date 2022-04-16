from utils.data_structures import MinSegmentTree, SegmentTree, SumSegmentTree
import random

import numpy as np
import torch

class ExperienceReplayMemory(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1 = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1 = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(obs_tp1)

        non_final_mask = np.array(
            tuple(map(lambda s: s is not None, obses_tp1))).astype(bool)
        try:
            non_final_next_states = np.array([s for s in obses_tp1 if s is not None])
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return (np.array(obses_t), np.array(actions), np.array(rewards), non_final_next_states, non_final_mask, empty_next_state_values), None, None

    def _encode_states(self, idxes):
        obses_t = []
        for i in idxes:
            data = self._storage[i]
            obs_t, _, _, _ = data
            if obs_t is not None:
                obses_t.append(np.array(obs_t, copy=False))

        return np.array(obses_t)

    def sample(self, batch_size, tstep=1, states_only=False):
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
        idxes = np.random.randint(0, len(self._storage), size=batch_size)
        if states_only:
            return self._encode_states(idxes)
        else:
            return self._encode_sample(idxes)

class GPUExperienceReplayMemory(object):
    def __init__(self, size, device):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self._device = device

    def __len__(self):
        return len(self._storage)

    def push(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1 = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1 = data
            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)

        non_final_mask = np.array(
            tuple(map(lambda s: s is not None, obses_tp1))).astype(bool)
        try:
            non_final_next_states = torch.stack([s for s in obses_tp1 if s is not None])
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return (torch.stack(obses_t), torch.stack(actions), torch.as_tensor(rewards, device=rewards[0].device, dtype=rewards[0].dtype), non_final_next_states, non_final_mask, empty_next_state_values), None, None

    def sample(self, batch_size, tstep=1):
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
        idxes = np.random.randint(0, len(self._storage), size=batch_size)
        return self._encode_sample(idxes)

# class DSBExperienceReplayMemory(ExperienceReplayMemory):
#     def __init__(self, size):
#         """Create Replay buffer.

#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         """
#         super().__init__(size)

#     def _encode_sample(self, idxes):
#         obses_t, actions, rewards, obses_tp1 = [], [], [], []
#         for i in idxes:
#             data = self._storage[i]
#             obs_t, action, reward, obs_tp1 = data
#             obses_t.append(np.array(obs_t, copy=False))
#             actions.append(np.array(action, copy=False))
#             rewards.append(reward)
#             obses_tp1.append(obs_tp1)

#         non_final_mask = np.array(
#             tuple(map(lambda s: s is not None, obses_tp1))).astype(bool)
#         try:
#             non_final_next_states = np.stack([s for s in obses_tp1 if s is not None], axis=1)
#             empty_next_state_values = False
#         except:
#             non_final_next_states = None
#             empty_next_state_values = True

#         return (np.stack(obses_t, axis=1), np.array(actions), np.array(rewards), non_final_next_states, non_final_mask, empty_next_state_values), None, None

class PrioritizedReplayMemory(object):
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayMemory, self).__init__()
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        assert alpha >= 0
        self._alpha = alpha

        self.beta_start = beta_start
        self.beta_frames = beta_frames

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def __len__(self):
        return len(self._storage)

    def beta_by_step(self, tstep):
        return min(1.0, self.beta_start + tstep * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, data):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1 = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1 = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(obs_tp1)

        non_final_mask = np.array(
            tuple(map(lambda s: s is not None, obses_tp1))).astype(bool)
        try:
            non_final_next_states = np.array(
                [s for s in obses_tp1 if s is not None], dtype=int)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return np.array(obses_t), np.array(actions), np.array(rewards), non_final_next_states, non_final_mask, empty_next_state_values
    
    def _encode_states(self, idxes):
        obses_t = []
        for i in idxes:
            data = self._storage[i]
            obs_t, _, _, _ = data
            obses_t.append(np.array(obs_t, copy=False))

        return np.array(obses_t)

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, tstep=1, states_only=False):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
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
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        if states_only:
            idxes = np.random.randint(0, len(self._storage), size=batch_size)
            return self._encode_states(idxes)

        idxes = self._sample_proportional(batch_size)

        weights = []

        # find smallest sampling prob: p_min = smallest priority^alpha / sum of priorities^alpha
        p_min = self._it_min.min() / self._it_sum.sum()

        beta = self.beta_by_step(tstep)

        # max_weight given to smallest prob
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)

        encoded_sample = self._encode_sample(idxes)

        return encoded_sample, idxes, np.array(weights, copy=False)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = (priority+1e-5) ** self._alpha
            self._it_min[idx] = (priority+1e-5) ** self._alpha

            self._max_priority = max(self._max_priority, (priority+1e-5))


class RecurrentExperienceReplayMemory:
    def __init__(self, capacity, sequence_length=10):
        self.capacity = capacity
        self.memory = []
        self.seq_length = sequence_length

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        finish = random.sample(range(0, len(self.memory)), batch_size)
        begin = [x-self.seq_length for x in finish]
        samp = []
        for start, end in zip(begin, finish):
            # correct for sampling near beginning
            final = self.memory[max(start+1, 0):end+1]

            # correct for sampling across episodes
            for i in range(len(final)-2, -1, -1):
                if final[i][3] is None:
                    final = final[i+1:]
                    break

            # pad beginning to account for corrections
            while(len(final) < self.seq_length):
                final = [(np.zeros_like(self.memory[0][0]), 0, 0,
                          np.zeros_like(self.memory[0][3]))] + final

            samp += final

        # returns flattened version
        return samp, None, None

    def __len__(self):
        return len(self.memory)


'''class PrioritizedReplayMemory(object):  
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.prob_alpha = alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1
        self.beta_start = beta_start
        self.beta_frames = beta_frames

    def beta_by_step(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0**self.prob_alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        total = len(self.buffer)

        probs = prios / prios.sum()

        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_step(self.frame)
        self.frame+=1

        #min of ALL probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min*total)**(-beta)

        weights  = (total * probs[indices]) ** (-beta)
        weights /= max_weight
        weights  = torch.tensor(weights, device=device, dtype=torch.float)
        
        return samples, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + 1e-5)**self.prob_alpha

    def __len__(self):
        return len(self.buffer)'''
