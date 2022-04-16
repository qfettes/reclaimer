# TODO: add arg to control type of noise in noisy nets
# TODO: efficiency for priority replay functions
# TODO: add random act for all algos
# TODO: remove baselines dependency
# TODO: change target network copying to use deepcopy everywhere
# TODO: move/add parameter freezing to declare network for target nets for all algorithms
# TODO: fix inference hparam
# TODO: fix render hparam

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

from utils.dsb_utils.information_gain import InfoGain
from utils.dsb_utils.dsb_ctrl import read_to_replay_buffer

gym.logger.set_level(40)


parser = argparse.ArgumentParser(description='RL')
# dsb-specific parameters
parser.add_argument('--qos-target', type=float, default=500.,
                    help='QoS latency target (ms) (default: 500)')
parser.add_argument('--meet-qos-weight', type=float, default=0.1,
                    help='weight on meeting the QoS target (default: 0.1)')
parser.add_argument('--transformer-heads', type=int, default=8,
                    help='Number of heads in the transformer layers (default: 8)')
parser.add_argument('--transformer-layers', type=int, default=3,
                    help='Number of transformer layers (default: 3)')
parser.add_argument('--transformer-dropout', type=float, default=0.0,
                    help='Dropout prob in transformer layers (default: 0.0)')
parser.add_argument('--hidden-units', type=int, default=256,
                    help='Hidden layer units (default: 256)')
parser.add_argument('--min-users', type=int, default=10,
                    help='Minimum simulated users (default: 10)')
parser.add_argument('--max-users', type=int, default=500,
                    help='Maximumum simulated users (default: 500)')
parser.add_argument('--user-step', type=int, default=20,
                    help='steps size from min to max users (default: 20)')
parser.add_argument('--exp-time', type=int, default=600,
                    help='Time (seconds) to simulate each user # (default: 600)')
parser.add_argument('--warmup-time', type=int, default=120,
                    help='Warmup period for new # of users (default: 120)')
parser.add_argument('--tstep-real-time', type=float, default=1.0,
                    help='Real time (seconds) per timestep (default: 1.0)')
parser.add_argument('--diurnal-load', action='store_true', default=False,
                    help='Change load in diurnal pattern (default: False)')
parser.add_argument('--load-model', action='store_true', default=False,
                    help='Load previousely trained model (default: False)')
parser.add_argument('--load-replay', action='store_true', default=False,
                    help='Load previousely trained replay buffer (default: False)')
parser.add_argument('--no-manual-caps', action='store_true', default=False,
                    help='do not use manual core caps when True (default: False)')
parser.add_argument('--no-unique-id', action='store_true', default=False,
                    help='do not use unique-id feature (default: False)')
parser.add_argument('--hotel', action='store_true', default=False,
                    help='Use hotel reservation benchmark')
parser.add_argument('--dvfs', action='store_true', default=False,
                    help='Use dvfs (default: False)')
parser.add_argument('--dvfs-weight', type=float, default=1.0,
                    help='weight for dvfs (exponential) relative to \
                        cpu scaling. < 1 for less priority, > 1 for more (default: 1.0)')
parser.add_argument('--no-conv', action='store_true', default=False,
                    help='Don\'t use conv for time dimension (FC LAyers handle instead) (default: False)')
parser.add_argument('--no-prune', action='store_true', default=False,
                    help='Don\'t prune latency features  (default: False)')
parser.add_argument('--use-info-gain', action='store_true', default=False,
                    help='Collect data via information gain model (default: False)')

#evaluation params


# Meta Info
parser.add_argument('--device', default='cuda',
                    help='device to train on (default: cuda)')
parser.add_argument('--algo', default='sac',
                    help='algorithm to use: sac')
parser.add_argument('--env-id', default='gym_dsb:dsb-social-media-v0',
                    help='environment to train on (default: gymDSB:dsb-v0)')
parser.add_argument('--seed', type=int, default=None, help='random seed. \
                        Note if seed is None then it will be randomly \
                        generated (default: None)')
parser.add_argument('--inference', action='store_true', default=False,
                    help='[NOT WORKING] Inference saved model.')
parser.add_argument('--print-threshold', type=int, default=10,
                    help='print progress and plot every print-threshold timesteps (default: 10)')
parser.add_argument('--save-threshold', type=int, default=300,
                    help='save nn params every save-threshold timesteps (default: 300)')
parser.add_argument('--logdir', default='./results/',
                                        help='algorithm to use (default: ./results/train)')
# Preprocessing
parser.add_argument('--stack-frames', type=int, default=3,
                    help='[Atari Only] Number of frames to stack (default: 3)')

# Noisy Nets
parser.add_argument('--noisy-nets', action='store_true', default=False,
                    help='Use noisy networks for exploration (all algorithms)')
parser.add_argument('--noisy-sigma', type=float, default=0.5,
                    help='Initial sigma value for noisy networks (default: 0.5)')


def evaluate(config, Agent, valid_arguments, default_arguments):
    # TODO: this is wrong. Needs to allow certain variables to change
    #   and also it's never actually set to the used config file
    if config.load_model:
        config = load_config(os.path.join(config.logdir, 'test/', config.algo, config.env_id.replace(':', '-')), config, valid_arguments)
    config.hotel = False #TODO: fix this
    config.qos_target = 500.0 # TODO: fix this
    config.user_step = 20 #TODO: fix this
    config.exp_time = 300 #TODO: fix this
    config.min_users = 20 #TODO: fix this
    config.max_users = 201 #TODO: fix this
    config.use_info_gain = False #TODO: fix this
    config.inference = True


    # make/clear directories for logging
    base_dir = os.path.join(config.logdir, 'evaluate/', config.algo, config.env_id.replace(':', '-'))
    log_dir = os.path.join(base_dir, 'logs/')
    eval_dir = os.path.join(base_dir, 'locust_results/')
    try:
        shutil.rmtree(eval_dir)
    except FileNotFoundError:
        pass
    tb_dir = os.path.join(base_dir, 'runs/')
    create_directory(base_dir)
    create_directory(log_dir)
    create_directory(eval_dir)
    create_directory(tb_dir)

    # Tensorboard writer
    writer = SummaryWriter(log_dir=tb_dir, comment='stuff')

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
                             no_unique_id=config.no_unique_id, social=(not config.hotel),
                             random_action_lower_bound=config.random_action_lower_bound,
                             dvfs=config.dvfs, dvfs_weight=config.dvfs_weight, conv=(not config.no_conv),
                             prune_features=(not config.no_prune))

    agent = Agent(env=envs, config=config, log_dir=base_dir, tb_writer=writer,
        valid_arguments=valid_arguments, default_arguments=default_arguments,)

    if config.use_info_gain:
        info_gain = InfoGain(agent.memory, env=envs, config=config, log_dir=agent.log_dir, tb_writer=agent.tb_writer)

        if config.load_model:
            fpath = os.path.join(config.logdir, 'test', config.algo, config.env_id.replace(':', '-'), 'saved_model')
            info_gain.load_w(fpath)

    # check for existing database and load in information to the agent replay buffer        
    if config.load_model:
        fpath = os.path.join(config.logdir, 'test', config.algo, config.env_id.replace(':', '-'), 'saved_model')
        agent.load_w(fpath)

    # # TODO: adding back in learning components
    # if config.load_replay:
    #     fpath = os.path.join(config.logdir, 'test', config.algo, config.env_id.replace(':', '-'), 'saved_model')
    #     if os.path.isfile(os.path.join(fpath, 'microserviceDB.h5')):
    #         assert(not config.transfer), "Just turn off eval at testing time"

    #         copyfile(os.path.join(fpath, 'microserviceDB.h5'), envs.controller.db_file)
    #         read_to_replay_buffer(agent.memory, envs.controller.db_file, 
    #             config.qos_target, config.meet_qos_weight, stack_frames=args.stack_frames,
    #             device=device, dvfs=config.dvfs, core_caps=envs.controller.np_manual_caps,
    #             no_unique_id=config.no_unique_id, dvfs_weight=config.dvfs_weight, 
    #             conv=(not config.no_conv), prune_features=(not config.no_prune)
    #         )                
    #     else:
    #         print(f"[ALERT] Loaded existing model, but there was no DB file at {os.path.join(fpath, 'microserviceDB.h5')}")
        
    user_counts = range(config.min_users, config.max_users, config.user_step)
    progress = tqdm.tqdm(user_counts, dynamic_ncols=True)
    progress.set_description(f"Users {0}, Tsteps {0}, mean/median R {0.0:.3f}/{0.0:.3f}, min/max R {0.0:.3f}/{0.0:.3f}")

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

        exp_proc = envs.controller.run_exp(
            duration=agent.config.exp_time, warmup_time=agent.config.warmup_time, 
            users=current_users, workers=0, 
            quiet=True
        )
        
        prev_obs = agent.envs.reset()

        #start timing the first timestep
        exp_start = timer()
        timestep_start = timer()

        while (timer() - exp_start) < (agent.config.exp_time):
            current_tstep += 1

            action = agent.get_action(prev_obs, deterministic=agent.config.inference, tstep=current_tstep)
            if config.use_info_gain:
                action = info_gain.get_action(prev_obs, action, agent.mean, agent.std, recovery=False, keep_prob=0.0)
            
            _, _, done, _ = agent.envs.step(action)

            # log total cpu actions
            with open('./results/locust_log/core_allocation_sum.txt', 'a') as f:
                f.write(str(np.dot(action[:,0], agent.envs.controller.np_manual_caps))+',')
            if config.dvfs:
                with open('./results/locust_log/frequency.txt', 'a') as f:
                    f.write(str(action[0,1])+',')

            try:
                time.sleep(
                    agent.config.tstep_real_time - (timer() - timestep_start)
                )
            except ValueError:
                print(f"Exceeded tstep length {current_tstep}")

            # roll in everything to next timestep
            timestep_start = timer()

            obs, reward, info = agent.envs.measure()
            
            # record 99th latency
            if agent.tb_writer and 'latency' in info[0]:
                agent.tb_writer.add_scalar(
                    'Policy/99th Percentile Latency', info[0]['latency'], 
                    current_tstep
                )
                agent.tb_writer.add_scalar(
                    'Policy/QoS Violation', int(info[0]['latency'] > agent.config.qos_target), 
                    current_tstep
                )

            agent.last_100_rewards.append(reward[0])
            with open('./results/locust_log/rewards.txt', 'a') as f:
                f.write(str(reward[0])+',')

            # Add a done signal if the experiment is ending
            if (timer() - exp_start) < (agent.config.exp_time + agent.config.warmup_time):
                pass
            else:
                done = np.array([True])

            # # TODO: Adding back in learning components
            # agent.append_to_replay(
            #     prev_obs, action.reshape((1,)+action.shape), 
            #     reward, obs, done.astype(int)
            # )

            if agent.tb_writer:
                agent.tb_writer.add_scalar(
                    'Policy/Reward', reward, current_tstep)

            # #TODO: adding back in learning components
            # agent.update(current_tstep)

            prev_obs = obs

            if current_tstep % config.print_threshold == 0 and agent.last_100_rewards:
                progress.set_description(
                    f"Users {current_users}, Tsteps {current_tstep}, "
                    f"mean/median R {np.mean(agent.last_100_rewards):.3f}/"
                    f"{np.median(agent.last_100_rewards):.3f}, "
                    f"min/max R {np.min(agent.last_100_rewards):.3f}/"
                    f"{np.max(agent.last_100_rewards):.3f}"
                )

        exp_proc.kill()
        copy_locust_stats(eval_dir, current_users)

    agent.envs.close()

def copy_locust_stats(final_path, user_count):
    locust_path = "./results/locust_log/"
    destination_path = os.path.join(final_path, str(user_count))
    if os.path.isdir(destination_path):
        shutil.rmtree(str(destination_path))
    shutil.copytree(locust_path, destination_path)

if __name__ == '__main__':
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
        'no_manual_caps', 'no_unique_id', 'hotel', 'qos_target', 'dvfs',
        'dvfs_weight', 'no_conv', 'no_prune', 'use_info_gain'
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

    assert(args.algo == 'sac'), f'Only SAC is supported. You picked {args.algo}'

    # Import Correct Agent
    from agents.SAC import Agent
    valid_arguments = universal_arguments | dqn_arguments | sac_arguments
    valid_arguments = valid_arguments - forbidden_arguments

    # training params
    config = Config()

    # dsb-specific args
    config.qos_target = args.qos_target
    config.meet_qos_weight = args.meet_qos_weight
    config.transformer_heads = args.transformer_heads
    config.transformer_layers = args.transformer_layers
    config.transformer_dropout = args.transformer_dropout
    config.hidden_units = args.hidden_units
    config.min_users = args.min_users
    config.max_users = args.max_users
    config.user_step = args.user_step
    config.exp_time = args.exp_time
    config.warmup_time = args.warmup_time
    config.tstep_real_time = args.tstep_real_time
    config.diurnal_load = args.diurnal_load
    config.load_model = args.load_model
    config.load_replay = args.load_replay
    config.no_manual_caps = args.no_manual_caps
    config.no_unique_id = args.no_unique_id
    config.hotel = args.hotel
    config.dvfs = args.dvfs
    config.dvfs_weight = args.dvfs_weight
    config.no_conv = args.no_conv
    config.no_prune = args.no_prune
    config.use_info_gain = args.use_info_gain

    # meta info
    config.device = args.device#
    config.algo = args.algo#
    config.env_id = args.env_id#
    config.seed = args.seed#
    config.inference = args.inference#
    config.print_threshold = int(args.print_threshold)#
    config.save_threshold = int(args.save_threshold)#
    config.logdir = args.logdir#

    # preprocessing
    config.stack_frames = int(args.stack_frames)#

    # Noisy Nets
    config.noisy_nets = args.noisy_nets#
    config.noisy_sigma = args.noisy_sigma#

    evaluate(config, Agent, valid_arguments, default_arguments)