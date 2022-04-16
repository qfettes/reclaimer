# TODO: include wrapper to log original returns

import os
from collections import deque

import gym
import numpy as np
import torch
from baselines import bench
from baselines.common.atari_wrappers import (ClipRewardEnv, EpisodicLifeEnv,
                                             FireResetEnv, FrameStack,
                                             NoopResetEnv, ScaledFloatFrame,
                                             TimeLimit, WarpFrame, make_atari,
                                             wrap_deepmind)
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
from gym import spaces, wrappers
from gym.spaces.box import Box

from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

from utils.dsb_utils.dsb_ctrl import compute_reward


class MaxAndSkipEnv_custom(gym.Wrapper):
    def __init__(self, env, skip=[4], sticky_actions=0.0):
        """
        Return only every `skip`-th frame
        Adds support for adaptive repeat and sticky actions
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

        self.num_actions = env.action_space.n
        self.sticky = sticky_actions
        self.prev_action = None

    def step(self, a):
        """Repeat action, sum reward, and max over last observations."""
        repeat_len = a // self.num_actions
        action = a % self.num_actions

        total_reward = 0.0
        done = None
        for i in range(self._skip[repeat_len]):
            is_sticky = np.random.rand()
            if is_sticky >= self.sticky or self.prev_action is None:
                self.prev_action = action

            obs, reward, done, info = self.env.step(self.prev_action)
            if i == self._skip[repeat_len] - 2:
                self._obs_buffer[0] = obs
            if i == self._skip[repeat_len] - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_atari_custom(env_id, max_episode_steps=None, skip=[4], sticky_actions=0.0):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv_custom(env, skip, sticky_actions)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_custom(env, episode_life=True, clip_rewards=True, frame_stack=4, scale=False):
    """
    Configure environment for DeepMind-style Atari.
    Adds support for custom # of frames stacked
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    return env


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        # return observation.transpose(2, 0, 1)
        return np.array(observation).transpose(2, 0, 1)

class WrapOneHot(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(self.observation_space.n,),
            dtype=np.float)

    def observation(self, observation):
        obs = np.zeros(self.observation_space.shape[0])
        obs[observation] = 1.
        return obs

class WrapSignal(gym.Wrapper):
    def step(self, a):
        obs, reward, done, info = self.env.step(a)
        if done and reward == 0.0:
            reward = -1.0

        return obs, reward, done, info

def make_envs_general(env_id, seed, log_dir, num_envs, stack_frames=4, adaptive_repeat=[4], 
    sticky_actions=0.0, clip_rewards=True, qos=24., meet_qos_weight=0.5, 
    device=None, use_manual_caps=True, no_unique_id=False, save_data=True,
    social=True, random_action_lower_bound=0.0, dvfs=False, dvfs_weight=1.0,
    prune_features=True):
    # for env in gym.envs.registry.env_specs: #hack
    #     # print(env)
    #     if 'gym' in env:
    #         print("Remove {} from registry".format(env))
    #         del gym.envs.registration.registry.env_specs[env]

    if 'dsb' not in env_id:
        env = gym.make(env_id)
        toy = True if env.observation_space.__class__.__name__ == 'Discrete' else False
        atari = True if env.action_space.__class__.__name__ == 'Discrete' else False
        env.close()

        if toy:
            envs = [make_one_toy(env_id, seed, i, log_dir) for i in range(num_envs)]
        elif atari:
            envs = [make_one_atari(env_id, seed, i, log_dir, stack_frames=stack_frames, adaptive_repeat=adaptive_repeat,
                            sticky_actions=sticky_actions, clip_rewards=True) for i in range(num_envs)]
        else:
            envs = [make_one_continuous(env_id, seed, i, log_dir)for i in range(num_envs)]
    else:
        envs = [
            make_one_dsb(
                env_id, seed, i, log_dir, stack_frames=stack_frames, 
                no_unique_id=no_unique_id, qos=qos, meet_qos_weight=meet_qos_weight, 
                use_manual_caps=use_manual_caps, save_data=save_data, social=social,
                random_action_lower_bound=random_action_lower_bound, dvfs=dvfs,
                dvfs_weight=dvfs_weight, prune_features=prune_features
            ) 
        for i in range(num_envs)]

    if 'dsb' in env_id:
        envs = DSBDummyVecEnv(envs)
        if device:
            envs = VecPyTorchDSB(envs, device=device)
    else:
        envs = DummyVecEnv(envs) if len(envs) == 1 else SubprocVecEnv(envs)

    # if not toy and not atari:
    #     if len(envs.observation_space.shape) == 1:
    #         if gamma is None:
    #             envs = VecNormalize(envs, ret=False)
    #         else:
    #             envs = VecNormalize(envs, gamma=gamma)

    #     envs = VecPyTorch(envs, device)

    #     if frame_stack is not None:
    #         envs = VecPyTorchFrameStack(envs, frame_stack, device)
    #     elif len(envs.observation_space.shape) == 3:
    #         envs = VecPyTorchFrameStack(envs, 4, device)

    return envs

def make_one_toy(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)

        if seed:
            env.seed(seed + rank)
            env.action_space.seed(seed + rank)
        else:
            env.seed(np.random.randint(10000000000))
            env.action_space.seed(np.random.randint(10000000000))

        env = WrapOneHot(env)
        # env = WrapSignal(env)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        return env
    return _thunk

def make_one_atari(env_id, seed, rank, log_dir, stack_frames=4, adaptive_repeat=[4], sticky_actions=0.0, clip_rewards=True):
    def _thunk():
        env = make_atari_custom(env_id, max_episode_steps=None,
                                skip=adaptive_repeat, sticky_actions=sticky_actions)

        if seed:
            env.seed(seed + rank)
            env.action_space.seed(seed + rank)
        else:
            env.seed(np.random.randint(10000000000))
            env.action_space.seed(np.random.randint(10000000000))

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        env = wrap_deepmind_custom(
            env, episode_life=True, clip_rewards=clip_rewards, frame_stack=stack_frames, scale=False)
        env = WrapPyTorch(env)

        return env
    return _thunk

def make_one_continuous(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)

        if seed:
            env.seed(seed + rank)
            env.action_space.seed(seed + rank)
        else:
            env.seed(np.random.randint(10000000000))
            env.action_space.seed(np.random.randint(10000000000))

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        return env
    return _thunk

def make_one_dsb(env_id, seed, rank, log_dir, stack_frames=1, no_unique_id=False, qos=24., 
    meet_qos_weight=0.5, use_manual_caps=True, save_data=True, social=True,
    random_action_lower_bound=0.0, dvfs=False, dvfs_weight=1.0,
    prune_features=True):
    def _thunk():
        env = gym.make(env_id, social=social, save_data=save_data, use_manual_caps=use_manual_caps, 
            random_action_lower_bound=random_action_lower_bound, dvfs=dvfs,
            prune_features=prune_features)

        env = qosRewardWrapper(env, qos, meet_qos_weight, dvfs_weight)

        env = ObsStackConv(env, stack_frames)

        if not no_unique_id:
            env = appendUniqueID(env)
        
        env = appendQOSRequirement(env, qos)

        if seed:
            env.seed(seed + rank)
            env.action_space.seed(seed + rank)
        else:
            env.seed(np.random.randint(10000000000))
            env.action_space.seed(np.random.randint(10000000000))

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)

        return env
    return _thunk

class ObsStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(shp[0], shp[1] * k))

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def measure(self):
        ob, reward, info = self.env.measure()
        self.frames.append(ob)
        return self._get_ob(), reward, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=-1)

class ObsStackConv(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(shp[0], k, shp[1]))

    def reset(self):
        ob = np.expand_dims(self.env.reset(), axis=1)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def measure(self):
        ob, reward, info = self.env.measure()
        self.frames.append(np.expand_dims(ob, axis=1))
        return self._get_ob(), reward, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=1)

class appendUniqueID(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        shp = env.observation_space.shape
        self.env = env
        #self.unique_id = np.eye(shp[0])
        
        #TODO: Address this in compatibility mode
        unique_id = np.eye(29 + 19)
        beg = 0
        if not env.social:
            beg = 29
        self.unique_id = unique_id[beg:(beg+shp[0])]
        new_shp = (shp[0], shp[1] + (29+19))

        self.unique_id = np.expand_dims(self.unique_id, axis=1)
        self.unique_id = np.repeat(self.unique_id, shp[1], axis=1)
        new_shp = (shp[0], shp[1], shp[2] + (29+19))

        #TODO: address this in compatibility mode
        # self.observation_space = spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(shp[0], shp[1] + shp[0]))
        self.observation_space = spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=new_shp)

    def reset(self):
        ob = self.env.reset()
        return np.concatenate((ob, self.unique_id), axis=-1)

    def measure(self):
        ob, reward, info = self.env.measure()
        return np.concatenate((ob, self.unique_id), axis=-1), reward, info

class qosRewardWrapper(gym.Wrapper):
    def __init__(self, env, qos_target, meet_qos_weight, dvfs_weight):
        gym.Wrapper.__init__(self, env)
        self.qos_target = qos_target
        self.meet_qos_weight = meet_qos_weight
        self.dvfs_weight = dvfs_weight

    def measure(self):
        ob, reward, info = self.env.measure()
        return ob, \
            compute_reward(reward, self.qos_target, 
                self.meet_qos_weight, action=self.env.controller.most_recent_action, 
                core_caps=self.env.controller.np_manual_caps, dvfs=self.env.dvfs,
                dvfs_weight=self.dvfs_weight), \
                    info

class appendQOSRequirement(gym.Wrapper):
    def __init__(self, env, QoS):
        gym.Wrapper.__init__(self, env)
        shp = env.observation_space.shape
        self.env = env
        self.qos_vec = np.array(
            [QoS/500. for _ in range(shp[0])],
            dtype=np.float
        ).reshape(-1, 1)
        new_shp = (shp[0], shp[1] + 1)
        
        self.qos_vec = self.qos_vec.reshape(-1, 1, 1)
        self.qos_vec = np.repeat(self.qos_vec, shp[1], axis=1)
        new_shp = (shp[0], shp[1], shp[2] + 1)

        self.observation_space = spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=new_shp)

    def reset(self):
        ob = self.env.reset()
        return np.concatenate((ob, self.qos_vec), axis=-1)

    def measure(self):
        ob, reward, info = self.env.measure()
        return np.concatenate((ob, self.qos_vec), axis=-1), reward, info

class DSBDummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)
        self.dvfs = env.dvfs

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.spec = self.envs[0].spec

        self.controller = self.envs[0].controller

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            # if isinstance(self.envs[e].action_space, spaces.Discrete):
            #    action = int(action)

            _, _, self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)

            # TODO: right now this is always false, but watch out.
            # if self.buf_dones[e]:
            #     obs = self.envs[e].reset()
            # self._save_obs(e, obs)
        return (None, None, np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def measure(self):
        for e in range(self.num_envs):
            obs, self.buf_rews[e], self.buf_infos[e] = self.envs[e].measure()
            self._save_obs(e, obs)
        return self._obs_from_buf(), np.copy(self.buf_rews), self.buf_infos.copy()

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
class VecPyTorchDSB(VecEnvWrapper):
    def __init__(self, venv, device=None):
        self.device = torch.device(device) if device else None
        self.venv = venv
        self.controller = venv.controller
        VecEnvWrapper.__init__(self, venv, observation_space=venv.observation_space)

    def measure(self):
        obs, reward, info = self.venv.measure()
        obs = torch.from_numpy(obs).to(torch.float)
        reward = torch.from_numpy(reward).to(torch.float)
        if self.device:
            obs, reward = obs.to(self.device), reward.to(self.device)
        return obs, reward, info

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).to(torch.float)
        if self.device:
            obs = obs.to(self.device)
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()




# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail


# class VecNormalize(VecNormalize_):
#     def __init__(self, *args, **kwargs):
#         super(VecNormalize, self).__init__(*args, **kwargs)
#         self.training = True

#     def _obfilt(self, obs, update=True):
#         if self.ob_rms:
#             if self.training and update:
#                 self.ob_rms.update(obs)
#             obs = np.clip((obs - self.ob_rms.mean) /
#                           np.sqrt(self.ob_rms.var + self.epsilon),
#                           -self.clipob, self.clipob)
#             return obs
#         else:
#             return obs

#     def train(self):
#         self.training = True

#     def eval(self):
#         self.training = False

# # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
# # Checks whether done was caused my timit limits or not


class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# # Adapted from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
# class VecPyTorch(VecEnvWrapper):
#     def __init__(self, venv, device):
#         """Return only every `skip`-th frame"""
#         super(VecPyTorch, self).__init__(venv)
#         self.device = device
#         # TODO: Fix data types

#     def reset(self):
#         obs = self.venv.reset()
#         obs = torch.from_numpy(obs).float().to(self.device)
#         return obs

#     def step_async(self, actions):
#         if isinstance(actions, torch.LongTensor):
#             # Squeeze the dimension for discrete actions
#             actions = actions.squeeze(1)
#         actions = actions.cpu().numpy()
#         self.venv.step_async(actions)

#     def step_wait(self):
#         obs, reward, done, info = self.venv.step_wait()
#         obs = torch.from_numpy(obs).float().to(self.device)
#         reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
#         done = torch.from_numpy(done).unsqueeze(dim=1).to(torch.bool)
#         return obs, reward, done, info

# # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
# class VecPyTorchFrameStack(VecEnvWrapper):
#     def __init__(self, venv, nstack, device=None):
#         self.venv = venv
#         self.nstack = nstack

#         wos = venv.observation_space  # wrapped ob space
#         self.shape_dim0 = wos.shape[0]

#         low = np.repeat(wos.low, self.nstack, axis=0)
#         high = np.repeat(wos.high, self.nstack, axis=0)

#         if device is None:
#             device = torch.device('cpu')
#         self.stacked_obs = torch.zeros((venv.num_envs, ) +
#                                        low.shape).to(device)

#         observation_space = gym.spaces.Box(
#             low=low, high=high, dtype=venv.observation_space.dtype)
#         VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

#     def step_wait(self):
#         obs, rews, news, infos = self.venv.step_wait()
#         self.stacked_obs[:, :-self.shape_dim0] = \
#             self.stacked_obs[:, self.shape_dim0:]
#         for (i, new) in enumerate(news):
#             if new:
#                 self.stacked_obs[i] = 0
#         self.stacked_obs[:, -self.shape_dim0:] = obs
#         return self.stacked_obs, rews, news, infos

#     def reset(self):
#         obs = self.venv.reset()
#         if torch.backends.cudnn.deterministic:
#             self.stacked_obs = torch.zeros(self.stacked_obs.shape)
#         else:
#             self.stacked_obs.zero_()
#         self.stacked_obs[:, -self.shape_dim0:] = obs
#         return self.stacked_obs

#     def close(self):
#         self.venv.close()