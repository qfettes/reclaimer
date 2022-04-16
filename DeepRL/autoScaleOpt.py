# TODO: add arg to control type of noise in noisy nets
# TODO: efficiency for priority replay functions
# TODO: add random act for all algos
# TODO: remove baselines dependency
# TODO: change target network copying to use deepcopy everywhere
# TODO: move/add parameter freezing to declare network for target nets for all algorithms
# TODO: fix inference hparam
# TODO: fix render hparam

from collections import deque
from baselines.common.vec_env import util
from utils.wrappers import make_envs_general
from utils.plot import plot_reward
from utils.hyperparameters import Config
from utils import create_directory, load_config, save_config, update_linear_schedule
from torch.utils.tensorboard import SummaryWriter
import tqdm
import torch
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import random
import os
import argparse
import gym
import gym_dsb
import pybulletgym
import time
import shutil
from shutil import copyfile

import ray, docker

from utils.dsb_utils.dsb_ctrl import read_to_replay_buffer

gym.logger.set_level(40)

def auto_scale(config, valid_arguments, default_arguments):    
    if not config.conservative:
        increase_bounds = (0.6, 0.7)
        # decrease_bounds = (0.3, 0.4)
        decrease_bounds = (0.3, 0.5)
    else:
        increase_bounds = (0.3, 0.5)
        decrease_bounds = (0.0, 0.1)
    # config.no_manual_caps = True
    last_100_rewards = deque(maxlen=100)

    # make/clear directories for logging
    base_dir = os.path.join(config.logdir, 'evaluate/', config.algo, config.env_id.replace(':', '-'))
    log_dir = os.path.join(base_dir, 'logs/')
    eval_dir = os.path.join(base_dir, 'locust_results/')
    tb_dir = os.path.join(base_dir, 'runs/')
    create_directory(base_dir)
    create_directory(log_dir)
    create_directory(eval_dir)
    create_directory(tb_dir)

    # save configuration for later reference
    save_config(config, base_dir, valid_arguments)

    # set seeds
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
    
    device = config.device if config.gpu_replay else None
    envs = make_envs_general(config.env_id, config.seed, log_dir,
                             config.nenvs, stack_frames=config.stack_frames, clip_rewards=True,
                             qos=config.qos_target, meet_qos_weight=config.meet_qos_weight,
                             device=device, use_manual_caps=(not config.no_manual_caps),
                             no_unique_id=config.no_unique_id, save_data=False, social=(not config.hotel),
                             dvfs=config.dvfs, dvfs_weight=config.dvfs_weight)
        
    user_counts = range(config.min_users, config.max_users, config.user_step)
    progress = tqdm.tqdm(user_counts, dynamic_ncols=True)
    progress.set_description(f"Users {0}, Tsteps {0}, mean/median R {0.0:.3f}/{0.0:.3f}, min/max R {0.0:.3f}/{0.0:.3f}")

    previous_cpu, previous_system = np.zeros_like(envs.controller.np_manual_caps), np.zeros_like(envs.controller.np_manual_caps)


    docker_stats_actor = None

    start = timer()
    current_tstep = 0
    for current_users in progress:
        envs.controller.reset_cores_and_frequency(use_max_freq=True)

        fname = './results/locust_log/core_allocation_sum.txt'
        if os.path.isfile(fname):
            os.remove(fname)
        fname = './results/locust_log/frequency.txt'
        if os.path.isfile(fname):
            os.remove(fname)
        fname = './results/locust_log/rewards.txt'
        if os.path.isfile(fname):
            os.remove(fname)

        if not (docker_stats_actor is None):
            del docker_stats_actor

        exp_proc = envs.controller.run_exp(
            duration=config.exp_time, warmup_time=config.warmup_time, 
            users=current_users, workers=0, 
            quiet=True
        )
        
        prev_obs = envs.reset()
        #initialize the actions to the cap, used during reset
        action = np.ones_like(envs.controller.most_recent_action)

        #start timing the first timestep
        exp_start = timer()
        timestep_start = timer()

        while (timer() - exp_start) < (config.exp_time):
            current_tstep += 1

            action = autoscale_action(envs, action, previous_cpu, previous_system, decrease_bounds, increase_bounds, dvfs=config.dvfs)
            
            _, _, done, _ = envs.step(action)

            # log total cpu actions
            with open('./results/locust_log/core_allocation_sum.txt', 'a') as f:
                f.write(str(np.dot(action[:,0], envs.controller.np_manual_caps))+',')
            
            if config.dvfs:
                with open('./results/locust_log/frequency.txt', 'a') as f:
                    f.write(str(action[0,1])+',')

            try:
                time.sleep(
                    config.tstep_real_time - (timer() - timestep_start)
                )
            except ValueError:
                if config.tstep_real_time - (timer() - timestep_start) > 0.2:
                    print(f"Exceeded tstep length {current_tstep} by {config.tstep_real_time - (timer() - timestep_start)}s")

            # roll in everything to next timestep
            timestep_start = timer()

            _, reward, _ = envs.measure()

            last_100_rewards.append(reward[0])
            with open('./results/locust_log/rewards.txt', 'a') as f:
                f.write(str(reward[0])+',')


            if current_tstep % config.print_threshold == 0 and last_100_rewards:
                progress.set_description(
                    f"Users {current_users}, Tsteps {current_tstep}, "
                    f"mean/median R {np.mean(last_100_rewards):.3f}/"
                    f"{np.median(last_100_rewards):.3f}, "
                    f"min/max R {np.min(last_100_rewards):.3f}/"
                    f"{np.max(last_100_rewards):.3f}"
                )

        exp_proc.kill()
        copy_locust_stats(eval_dir, current_users)

    ray.shutdown()
    envs.close()

def autoscale_action(envs, action, previous_cpu, previous_system, decrease_bounds, increase_bounds, noise=False, dvfs=False):
    assert(len(action.shape)==2), f"Expecting 2d tensor for action, got shape {action.shape}"
    
    cpu_allocation = action[:,0] * envs.controller.np_manual_caps
    containers = envs.controller.get_containers()
    new_containers = [x for x in containers if x.name not in ('socialnetwork_jaeger_1', 'hotel_reserv_jaeger', 'socialnetwork-ml-swarm_jaeger_1', 'social-network_jaeger_1', 'resource-sink')]
    # new_containers = list(sorted(new_containers, key=lambda x: envs.controller.positional_encoding[x.name]))
    client = docker.from_env()
    container = client.containers.get(new_containers[0].id)
    
    util_result = ray.get(
        [
            calculate_cpu_usage.remote(
                c.id,
                previous_cpu[ envs.controller.positional_encoding[c.name] ],
                previous_system[ envs.controller.positional_encoding[c.name] ], 
                cpu_allocation[ envs.controller.positional_encoding[c.name] ],
                decrease_bounds,
                increase_bounds
            ) for c in new_containers
        ]
    )
    
    modifier = np.ones_like(envs.controller.most_recent_action)
    mod_idx = 0

    condition = dvfs and np.random.random() > 0.5
    if condition:
        mod_idx = 1

    for r in util_result:
        service_idx = envs.controller.positional_encoding[ r[0] ]

        previous_cpu[service_idx] = r[2]
        previous_system[service_idx] = r[3]

        modifier[service_idx, mod_idx] = r[1]

        if noise and mod_idx==0:
            # try scaling the modifier randomly from 50% to 150% its normal value
            #   adds a little variety to the data collection
            modifier[service_idx, mod_idx] = 1. + (modifier[service_idx, mod_idx] - 1.0) \
                * (np.random.random() + .5)
            if abs(modifier[service_idx, mod_idx] -  1.) <= 1e-8:
                modifier[service_idx, mod_idx] += np.random.random() * 0.1 - 0.05
    
    if condition:
        for r in util_result:
            service_idx = envs.controller.positional_encoding[ r[0] ]
        avg_modifier = np.median(modifier[:,mod_idx])
        if noise:
            avg_modifier += np.random.random() * 0.1 - 0.05
        for r in util_result:
            service_idx = envs.controller.positional_encoding[ r[0] ]
            modifier[service_idx, mod_idx] = avg_modifier

    return np.clip(action * modifier, 0.0, 1.0)

@ray.remote
def calculate_cpu_usage(container_id, previous_cpu, previous_system, cpu_allocation, decrease_bounds, increase_bounds):
    client = docker.from_env()
    container = client.containers.get(container_id)
    stats = container.stats(stream=True, decode=True)

    for d in stats:
        try:
            cpu_usage = 0.0
            cpu_total = float(d["cpu_stats"]["cpu_usage"]["total_usage"])
            cpu_delta = cpu_total - previous_cpu
            previous_cpu = cpu_total

            cpu_system = float(d["cpu_stats"]["system_cpu_usage"])
            system_delta = cpu_system - previous_system
            previous_system = cpu_system

            online_cpus = d["cpu_stats"].get("online_cpus")
            if system_delta > 0.0:
                cpu_usage = (cpu_delta / system_delta) * online_cpus

            relative_utilization = cpu_usage / cpu_allocation
            modifier = 1.0

            # TODO: Change these back
            if relative_utilization < decrease_bounds[0]:
                modifier = 0.7
            elif relative_utilization < decrease_bounds[1]: 
                modifier = 0.9
            elif relative_utilization > increase_bounds[1]:
                modifier =  1.3
            elif relative_utilization > increase_bounds[0]:
                modifier = 1.1
            
            break
        except KeyError:
            modifier = 1.0

    return container.name, modifier, previous_cpu, previous_system

def copy_locust_stats(final_path, user_count):
    locust_path = "./results/locust_log/"
    destination_path = os.path.join(final_path, str(user_count))
    if os.path.isdir(destination_path):
        shutil.rmtree(str(destination_path))
    shutil.copytree(locust_path, destination_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL')
    # dsb-specific parameters
    parser.add_argument('--qos-target', type=float, default=500.,
                        help='QoS latency target (ms) (default: 500)')
    parser.add_argument('--meet-qos-weight', type=float, default=0.1,
                        help='weight on meeting the QoS target (default: 0.1)')
    parser.add_argument('--min-users', type=int, default=10,
                        help='Minimum simulated users (default: 10)')
    parser.add_argument('--max-users', type=int, default=500,
                        help='Maximumum simulated users (default: 500)')
    parser.add_argument('--user-step', type=int, default=20,
                        help='steps size from min to max users (default: 20)')
    parser.add_argument('--exp-time', type=int, default=300,
                        help='Time (seconds) to simulate each user # (default: 300)')
    parser.add_argument('--warmup-time', type=int, default=60,
                        help='Warmup period for new # of users (default: 60)')
    parser.add_argument('--tstep-real-time', type=float, default=1.0,
                        help='Real time (seconds) per timestep (default: 1.0)')
    parser.add_argument('--diurnal-load', action='store_true', default=False,
                        help='Change load in diurnal pattern (default: False)')
    parser.add_argument('--no-manual-caps', action='store_true', default=False,
                        help='do not use manual core caps when True (default: False)')
    parser.add_argument('--hotel', action='store_true', default=False,
                        help='Use hotel reservation benchmark')
    parser.add_argument('--conservative', action='store_true', default=False,
                        help='Use conservative utilization bounds (default: False)')

    # Meta Info
    parser.add_argument('--env-id', default='gym_dsb:dsb-social-media-v0',
                        help='environment to train on (default: gymDSB:dsb-v0)')
    parser.add_argument('--print-threshold', type=int, default=10,
                        help='print progress and plot every print-threshold timesteps (default: 10)')
    parser.add_argument('--logdir', default='./results/',
                                            help='algorithm to use (default: ./results/train)')
    parser.add_argument('--dvfs', action='store_true', default=False,
                        help='Use dvfs (default: False)')
    parser.add_argument('--dvfs-weight', type=float, default=1.0,
                        help='weight for dvfs (exponential) relative to \
                            cpu scaling. < 1 for less priority, > 1 for more (default: 1.0)')

    args = parser.parse_args()

    # TODO: resume here. Log default parameters
    #   record which algorithms are relevant to each parameter
    #   for each algorithm, throw error when irrelevant parameter
    #   is changed from default

    # get all default arguments
    default_arguments = {}
    for key in vars(args):
        default_arguments[key] = parser.get_default(key)

    # Keep track of valid arguments for each algorithm
    #   as library grows, we can throw errors when invalid
    #   args are changed from default values; this is likely
    #   unintended by the user
    universal_arguments = {
        'device', 'algo', 'env_id', 'seed', 'inference',
        'print_threshold', 'save_threshold', 
        'logdir', 'correct_time_limits', 
        'max_tsteps', 'learn_start', 'random_act', 'nenvs',
        'update_freq', 'lr', 'anneal_lr', 'grad_norm_max',
        'gamma', 'optim_eps', 'noisy_nets', 'noisy_sigma'
    }
    dqn_arguments = {
        'stack_frames', 'sticky_actions',
        'replay_size', 'batch_size', 'tnet_update',
        'eps_start', 'eps_end', 'eps_decay', 'n_steps',
        'priority_replay', 'priority_alpha',
        'priority_beta_start', 'priority_beta_steps',
        'gpu_replay'
    }
    sac_arguments = {
        'random_act', 'polyak_coef', 'entropy_coef',
        'entropy_tuning', 'transformer_heads', 'transformer_layers',
        'transformer_dropout', 'hidden_units', 'pretrain_tsteps',
        'min_users', 'user_step', 'max_users', 'exp_time', 'warmup_time',
        'tstep_real_time', 'diurnal_load', 'meet_qos_weight',
        'load_model', 'load_replay', 'recompute_normalizing_const_freq',
        'no_manual_caps', 'no_unique_id', 'hotel', 'qos_target', 
        'conservative', 'dvfs', 'dvfs_weight'
    }

    forbidden_arguments = {
        'tnet_update', 'eps_start', 'eps_decay', 'eps_end', 'nenvs',
        'correct_time_limits', 'max_tsteps', 'learn_start', 'random_act',
        'nenvs', 'update_freq', 'lr', 'anneal_lr', 'grad_norm_max', 'gamma',
        'gamma', 'optim_eps', 'replay_size', 'batch_size', 'tnet_update',
        'eps_start', 'eps_end', 'eps_decay', 'n_steps', 'priority_replay',
        'priority_alpha', 'priority_beta_start', 'priority_beta_steps',
        'gpu_replay', 'polyak_coef', 'entropy_coef', 'entropy_tuning', 
        'pretrain_tsteps', 'recompute_normalizing_const_freq'
    }
    # Import Correct Agent
    valid_arguments = universal_arguments | dqn_arguments | sac_arguments
    valid_arguments = valid_arguments - forbidden_arguments

    # training params
    config = Config()

    # dsb-specific args
    config.qos_target = args.qos_target
    config.meet_qos_weight = args.meet_qos_weight
    config.min_users = args.min_users
    config.max_users = args.max_users
    config.user_step = args.user_step
    config.exp_time = args.exp_time
    config.warmup_time = args.warmup_time
    config.tstep_real_time = args.tstep_real_time
    config.diurnal_load = args.diurnal_load
    config.no_manual_caps = args.no_manual_caps
    config.hotel = args.hotel
    config.conservative = args.conservative
    config.dvfs = args.dvfs
    config.dvfs_weight = args.dvfs_weight

    # meta info
    config.env_id = args.env_id
    config.print_threshold = int(args.print_threshold)#
    config.logdir = args.logdir#

    auto_scale(config, valid_arguments, default_arguments)