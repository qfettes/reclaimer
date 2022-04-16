import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from utils.dsb_utils.dsb_ctrl import dsb_controller, form_obs, compute_reward

class DSBSocialMediaService(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, social=True, dvfs=False, save_data=True, 
        db_file='./results/train/sac/gym_dsb-dsb-social-media-v0/saved_model/microserviceDB.h5', 
        use_manual_caps=True, random_action_lower_bound=0.0, prune_features=True):

        # Not sure about the high value here
        self.dvfs = dvfs
        self.controller = dsb_controller(social=social, dvfs=self.dvfs, db_file=db_file, use_manual_caps=use_manual_caps)
        self.social = social
        self.prune_features = prune_features

        num_feats = 20 if self.prune_features else 27
        if self.dvfs:
            self.action_space = spaces.Box(low=random_action_lower_bound, high=1.0, shape=(self.controller.num_microservices, 2))
        else:
            self.action_space = spaces.Box(low=random_action_lower_bound, high=1.0, shape=(self.controller.num_microservices, 1))
        self.observation_space = spaces.Box(low=0.0, high=np.finfo(np.float32).max, shape=(self.controller.num_microservices, num_feats+self.action_space.shape[1]))

        self.save_data = save_data

    def step(self, action):
        self.controller.execute_action(action)

        # next_state, reward, done, info
        return None, None, False, {}

    def reset(self):
        _ = self.controller.get_features()
        network_features, memory_features, cpu_features, io_features, latency_data = self.controller.get_features()
        action_data = self.controller.most_recent_action

        if self.save_data:
            self.controller.store_features(network_features, memory_features, cpu_features, io_features, latency_data, action_data)

        # # NOTE: correction to put cpu time in cpus, not 1000*cpus
        # # done here to keep compat with old data
        # for key, val in cpu_features[0].items():
        #     cpu_features[0][key] = val / 1000.0

        all_vals = self.controller.raw_data_to_numpy(
            network_features, memory_features, cpu_features, 
            io_features, latency_data
        )

        obs = form_obs(all_vals[:-1], all_vals[-1], action_data, self.controller.np_manual_caps, prune_features=self.prune_features)

        return obs        


    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def measure(self):
        '''
        This is a non-standard function to get the state vector
        and reward information
        '''
        network_features, memory_features, cpu_features, io_features, latency_data = self.controller.get_features()
        action_data = self.controller.most_recent_action

        if self.save_data:
            self.controller.store_features(network_features, memory_features, cpu_features, io_features, latency_data, action_data)

        # # NOTE: correction to put cpu time in cpus, not 1000*cpus
        # # done here to keep compat with old data
        # for key, val in cpu_features[0].items():
        #     cpu_features[0][key] = val / 1000.0
        
        all_vals = self.controller.raw_data_to_numpy(
            network_features, memory_features, cpu_features, 
            io_features, latency_data
        )

        obs = form_obs(all_vals[:-1], all_vals[-1], action_data, self.controller.np_manual_caps, prune_features=self.prune_features)

        reward = float(latency_data['99.0'])
        info = {'latency': latency_data['99.0']}

        return obs, reward, info