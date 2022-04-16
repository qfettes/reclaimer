from collections import deque
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
from shutil import copyfile

from utils.dsb_utils.dsb_ctrl import analyze_replay_buffer, read_to_replay_buffer

from autoScaleOpt import autoscale_action
from utils.dsb_utils.information_gain import InfoGain

gym.logger.set_level(40)


parser = argparse.ArgumentParser(description='RL')
# dsb-specific parameters
parser.add_argument('--qos-target', type=float, default=500.,
                    help='QoS latency target (ms) (default: 500)')
parser.add_argument('--meet-qos-weight', type=float, default=0.1,
                    help='weight on meeting the QoS target (default: 0.1)')
parser.add_argument('--gpu-replay', action='store_true', default=False,
                    help='Keep Replay Buffer in GPU memory (default: False)')
parser.add_argument('--transformer-heads', type=int, default=4,
                    help='Number of heads in the transformer layers (default: 4)')
parser.add_argument('--transformer-layers', type=int, default=1,
                    help='Number of transformer layers (default: 1)')
parser.add_argument('--transformer-dropout', type=float, default=0.0,
                    help='Dropout prob in transformer layers (default: 0.0)')
parser.add_argument('--hidden-units', type=int, default=256,
                    help='Hidden layer units (default: 256)')
parser.add_argument('--pretrain-tsteps', type=int, default=0,
                    help='Maximimum number of pretrain timsteps to train (default: 0)')
parser.add_argument('--min-users', type=int, default=10,
                    help='Minimum simulated users (default: 10)')
parser.add_argument('--max-users', type=int, default=320,
                    help='Maximumum simulated users (default: 320)')
parser.add_argument('--user-step', type=int, default=100,
                    help='gap between user counts (default: 100)')
parser.add_argument('--exp-time', type=int, default=480,
                    help='Time (seconds) to simulate each user # (default: 480)')
parser.add_argument('--warmup-time', type=int, default=300,
                    help='Warmup period for new # of users (default: 300)')
parser.add_argument('--tstep-real-time', type=float, default=1.0,
                    help='Real time (seconds) per timestep (default: 1.0)')
parser.add_argument('--diurnal-load', action='store_true', default=False,
                    help='Change load in diurnal pattern (default: False)')
parser.add_argument('--load-model', action='store_true', default=False,
                    help='Load previousely trained model (default: False)')
parser.add_argument('--load-replay', action='store_true', default=False,
                    help='Load previousely trained replay buffer (default: False)')
parser.add_argument('--recompute-normalizing-const-freq', type=int, default=1000000,
                    help='How often to recompute normalizing constants (default: 1000000)')
parser.add_argument('--no-manual-caps', action='store_true', default=False,
                    help='do not use manual core caps when True (default: False)')
parser.add_argument('--no-unique-id', action='store_true', default=False,
                    help='do not use unique-id feature (default: False)')
parser.add_argument('--hotel', action='store_true', default=False,
                    help='Use hotel reservation benchmark')
parser.add_argument('--random-action-lower-bound', type=float, default=0.5,
                    help='Lower bound of random actions [0.0 - 1.0] (default: 0.5)')
parser.add_argument('--bootstrap-autoscale', action='store_true', default=False,
                    help='bootstrap from autoscale during random action collection (default: False)')
parser.add_argument('--conservative', action='store_true', default=False,
                    help='Use conservative utilization bounds (default: False)')
parser.add_argument('--dvfs', action='store_true', default=False,
                    help='Use dvfs (default: False)')
parser.add_argument('--dvfs-weight', type=float, default=1.0,
                    help='weight for dvfs (exponential) relative to \
                        cpu scaling. < 1 for less priority, > 1 for more (default: 1.0)')
parser.add_argument('--no-prune', action='store_true', default=False,
                    help='Don\'t prune latency features  (default: False)')
parser.add_argument('--viol-timeout', dest='viol_timeout', type=int, default=30,
                    help='Kill experiments that are really bad')
parser.add_argument('--updates-per-tstep', type=int, default=10,
                    help='Number of updates per timestep (default: 10)')
parser.add_argument('--use-info-gain', action='store_true', default=False,
                    help='Collect data via information gain model (default: False)')
parser.add_argument('--transfer', action='store_true', default=False,
                    help='transfer a model from hotel to social (default: False)')
parser.add_argument('--ewc-lambda', type=float, default=400.,
                    help='Weight on EWC penalty. Default comes from RL exp in paper (default: 400.)')
parser.add_argument('--ewc-start', type=int, default=10000,
                    help='First step to apply ewc penalty')
parser.add_argument('--info-gain-anneal-steps', type=float, default=15000.0,
                    help='number of steps over which info-gain prob is annealed to 0 during RL training (default: 15000')





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
parser.add_argument('--correct-time-limits', action='store_true', default=False,
                    help='Ignore time-limit end of episode when updating (default: False')

# Preprocessing
parser.add_argument('--stack-frames', type=int, default=3,
                    help='[Atari Only] Number of frames to stack (default: 3)')

# Learning Control Variables
parser.add_argument('--max-tsteps', type=int, default=1e6,
                    help='Maximimum number of timsteps to train (default: 1e6)')
parser.add_argument('--learn-start', type=int, default=0,
                    help='tstep to start updating (default: 0)')
parser.add_argument('--random-act', type=int, default=0,
                    help='[SAC Only] Take uniform random actions until this tstep (default: 0)')
parser.add_argument('--nenvs', type=int, default=1,
                    help='number of parallel environments executing (default: 1)')
parser.add_argument('--update-freq', type=int, default=1,
                    help='frequency (tsteps) to perform updates (default: 1)')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='learning rate (default: 3e-4)')
parser.add_argument('--anneal-lr', action='store_true', default=False,
                    help='anneal lr from start value to 0 throught training')
parser.add_argument('--grad-norm-max', type=float, default=40.0,
                    help='max norm of gradients (default: 40.0)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')

# Optimizer Parameters
parser.add_argument('--optim-eps', type=float, default=1e-8,
                    help='epsilon param of optimizer (default: 1e-8)')

# Replay Memory
parser.add_argument('--replay-size', type=int, default=1e5,
                    help='!!!WATCH YOUR RAM USAGE!!! Size of replay buffer (default: 100000)')
parser.add_argument('--batch-size', type=int, default=100,
                    help='Size of minibatches drawn from replay buffer (default: 32)')
parser.add_argument('--polyak-coef', type=float, default=0.995,
                    help='[SAC ONLY] \theta_targ <- polyak_coef*\theta_targ + (1.-polyak_coef)*\theta\
                         while using polyak averaging in SAC (default: 0.995)')

# Nstep
parser.add_argument('--n-steps', type=int, default=1,
                    help='[Exp-Replay Only] Value of N used in N-Step Q-Learning (default: 1)')

# Noisy Nets
parser.add_argument('--noisy-nets', action='store_true', default=False,
                    help='Use noisy networks for exploration (all algorithms)')
parser.add_argument('--noisy-sigma', type=float, default=0.5,
                    help='Initial sigma value for noisy networks (default: 0.5)')

# policy
parser.add_argument('--entropy-coef', type=float, default=1e-6,
                    help='Weight on the entropy term in PG methods (default: 1e-6)')
parser.add_argument('--entropy-tuning', action='store_true', default=False,
                    help='[SAC ONLY] Automatically tune entropy-coef. Uses input to --entropy-coef as start value (default: False)')


# Priority Replay
parser.add_argument('--priority-replay', action='store_true', default=False,
                    help='[Replay Only] Use prioritized replay with dqn')
parser.add_argument('--priority-alpha', type=float, default=0.6,
                    help='[Replay Only] Alpha value of prioritized replay (default: 0.6)')
parser.add_argument('--priority-beta-start', type=float, default=0.4,
                    help='[Replay Only] starting value of beta in prioritized replay (default: 0.4)')
parser.add_argument('--priority-beta-steps', type=int, default=2e7,
                    help='[Replay Only] steps over which to anneal priority beta to 1 (default: 2e7)')
def pretrain(config, Agent, valid_arguments, default_arguments):
    # # Tensorboard writer
    # writer = None
    
    # device = config.device if config.gpu_replay else None
    # envs = make_envs_general(config.env_id, config.seed, '',
    #                          config.nenvs, stack_frames=config.stack_frames, clip_rewards=True,
    #                          qos=config.qos_target, meet_qos_weight=config.meet_qos_weight,
    #                          device=device, use_manual_caps=(not config.no_manual_caps),
    #                          no_unique_id=config.no_unique_id, social=(not config.hotel),
    #                          random_action_lower_bound=config.random_action_lower_bound,
    #                          dvfs=config.dvfs, dvfs_weight=config.dvfs_weight,
    #                          prune_features=(not config.no_prune))

    # agent = Agent(env=envs, config=config, log_dir='', tb_writer=writer,
    #     valid_arguments=valid_arguments, default_arguments=default_arguments,)

    # fpath = os.path.join(config.logdir, 'test', config.algo, config.env_id.replace(':', '-'), 'saved_model')
    # if os.path.isfile(os.path.join(fpath, 'microserviceDB.h5')):
    #     read_to_replay_buffer(agent.memory, os.path.join(fpath, 'microserviceDB.h5'), 
    #         config.qos_target, config.meet_qos_weight, stack_frames=args.stack_frames,
    #         device=device, dvfs=config.dvfs, core_caps=envs.controller.np_manual_caps,
    #         no_unique_id=config.no_unique_id, dvfs_weight=config.dvfs_weight, 
    #         prune_features=(not config.no_prune)
    #     )
    # agent.dsb_recompute_normalizing_constants()

    # manual_caps = [
    #     8., #'social-ml-nginx-thrift': 0,
    #     2., #'social-ml-text-service': 1,
    #     8., #'social-ml-home-timeline-service': 2,
    #     2., #'social-ml-write-home-timeline-service': 3,
    #     2., #'social-ml-user-service': 4, 
    #     2., #'social-ml-user-timeline-service': 5,
    #     2., #'social-ml-write-user-timeline-service': 6,
    #     2., #'social-ml-compose-post-service': 7,
    #     16., #'social-ml-post-storage-service': 8,
    #     2., #'social-ml-url-shorten-service': 9,
    #     2., #'social-ml-compose-post-redis': 10,
    #     2., #'social-ml-social-graph-redis': 11,
    #     4., #'social-ml-media-service': 12,
    #     2., #'social-ml-user-timeline-mongodb': 13,
    #     2., #'social-ml-post-storage-memcached': 14,
    #     24., #'social-ml-media-filter-service': 15,
    #     2., #'social-ml-social-graph-mongodb': 16,
    #     2., #'social-ml-user-mention-service': 17,
    #     2., #'social-ml-user-mongodb': 18,
    #     2., #'social-ml-social-graph-service': 19,
    #     2., #'social-ml-user-memcached': 20,
    #     2., #'social-ml-write-user-timeline-rabbitmq': 21,
    #     2., #'social-ml-post-storage-mongodb': 22,
    #     2., #'social-ml-home-timeline-redis': 23,
    #     2., #'social-ml-user-timeline-redis': 24,
    #     2., #'social-ml-write-home-timeline-rabbitmq': 25,
    #     2., #'social-ml-unique-id-service': 26,
    #     2., #'social-ml-text-filter-service': 27,
    # ] # baseline: 250-270 after 2 min with max for all
    # np_manual_caps = np.array(manual_caps, dtype=float)

    # fpath = os.path.join(config.logdir, 'train', config.algo, config.env_id.replace(':', '-'), 'saved_model', 'microserviceDB.h5')

    # analyze_replay_buffer(fpath, 
    #     config.qos_target, config.meet_qos_weight, stack_frames=args.stack_frames,
    #     dvfs=config.dvfs, core_caps=np_manual_caps,
    #     no_unique_id=config.no_unique_id, dvfs_weight=config.dvfs_weight, 
    #     prune_features=(not config.no_prune),
    #     mean=agent.mean.cpu().numpy(), std=agent.std.cpu().numpy(),
    # )

    # exit()

    if config.load_model:
        load_config(os.path.join(config.logdir, 'test/', config.algo, config.env_id.replace(':', '-')))
    
    # make/clear directories for logging
    base_dir = os.path.join(config.logdir, 'train/', config.algo, config.env_id.replace(':', '-'))
    log_dir = os.path.join(base_dir, 'logs/')
    model_dir = os.path.join(base_dir, 'saved_model/')
    tb_dir = os.path.join(base_dir, 'runs/')
    create_directory(base_dir)
    create_directory(log_dir)
    create_directory(model_dir)
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
                             dvfs=config.dvfs, dvfs_weight=config.dvfs_weight, 
                             prune_features=(not config.no_prune))

    agent = Agent(env=envs, config=config, log_dir=base_dir, tb_writer=writer,
        valid_arguments=valid_arguments, default_arguments=default_arguments,)


    # check for existing database and load in information to the agent replay buffer
    if 'dsb' in config.env_id:
        if config.load_model:
            fpath = os.path.join(config.logdir, 'test', config.algo, config.env_id.replace(':', '-'), 'saved_model')
            agent.load_w(fpath)

        if config.load_replay:
            fpath = os.path.join(config.logdir, 'test', config.algo, config.env_id.replace(':', '-'), 'saved_model')
            if os.path.isfile(os.path.join(fpath, 'microserviceDB.h5')):
                # just copy over existing file if we are keeping data

                if not config.transfer:
                    copyfile(os.path.join(fpath, 'microserviceDB.h5'), envs.controller.db_file)
                    read_to_replay_buffer(agent.memory, envs.controller.db_file, 
                        config.qos_target, config.meet_qos_weight, stack_frames=args.stack_frames,
                        device=device, dvfs=config.dvfs, core_caps=envs.controller.np_manual_caps,
                        no_unique_id=config.no_unique_id, dvfs_weight=config.dvfs_weight, 
                        prune_features=(not config.no_prune)
                    )
                else:
                    original_qos = config.qos_target

                    # config.qos_target = 200.
                    # manual_caps = [
                    #     12, #'hotelreservation_frontend': 7,
                    #     4, #'hotelreservation_profile': 4,
                    #     8, #'hotelreservation_search': 6,
                    #     4, #'hotelreservation_rate': 1,
                    #     4, #'hotelreservation_recommendation': 5,
                    #     4, #'hotelreservation_user': 0,
                    #     4, #'hotelreservation_reservation': 3,
                    #     4, #'hotelreservation_memcached-rate': 13,
                    #     2, #'hotelreservation_memcached-profile': 12,
                    #     2, #'hotelreservation_memcached-reserve_1': 11,
                    #     8, #'hotelreservation_mongodb-profile': 14,
                    #     8, #'hotelreservation_mongodb-rate_1': 8,
                    #     8, #'hotelreservation_mongodb-recommendation': 17
                    #     8, #'hotelreservation_mongodb-reservation': 15,
                    #     8, #'hotelreservation_mongodb-user': 10,
                    # ] # baseline: 250-270 after 2 min with max for all
                    # manual_caps = np.array(manual_caps, dtype=float)

                    # fpath = './results/evaluation_models/cpu/hotel/retrain_350k/sac/gym_dsb-dsb-social-media-v0/saved_model/microserviceDB.h5'
                    # read_to_replay_buffer(agent.memory, fpath, 
                    #     config.qos_target, config.meet_qos_weight, stack_frames=args.stack_frames,
                    #     device=device, dvfs=config.dvfs, core_caps=manual_caps,
                    #     no_unique_id=config.no_unique_id, dvfs_weight=config.dvfs_weight, 
                    #     prune_features=(not config.no_prune)
                    # )

                    # # TODO: control this via a CLI arg
                    # agent.init_ewc()

                    manual_caps = [
                        8., #'social-ml-nginx-thrift': 0,
                        2., #'social-ml-text-service': 1,
                        8., #'social-ml-home-timeline-service': 2,
                        2., #'social-ml-write-home-timeline-service': 3,
                        2., #'social-ml-user-service': 4, 
                        2., #'social-ml-user-timeline-service': 5,
                        2., #'social-ml-write-user-timeline-service': 6,
                        2., #'social-ml-compose-post-service': 7,
                        16., #'social-ml-post-storage-service': 8,
                        2., #'social-ml-url-shorten-service': 9,
                        2., #'social-ml-compose-post-redis': 10,
                        2., #'social-ml-social-graph-redis': 11,
                        4., #'social-ml-media-service': 12,
                        2., #'social-ml-user-timeline-mongodb': 13,
                        2., #'social-ml-post-storage-memcached': 14,
                        24., #'social-ml-media-filter-service': 15,
                        2., #'social-ml-social-graph-mongodb': 16,
                        2., #'social-ml-user-mention-service': 17,
                        2., #'social-ml-user-mongodb': 18,
                        2., #'social-ml-social-graph-service': 19,
                        2., #'social-ml-user-memcached': 20,
                        2., #'social-ml-write-user-timeline-rabbitmq': 21,
                        2., #'social-ml-post-storage-mongodb': 22,
                        2., #'social-ml-home-timeline-redis': 23,
                        2., #'social-ml-user-timeline-redis': 24,
                        2., #'social-ml-write-home-timeline-rabbitmq': 25,
                        2., #'social-ml-unique-id-service': 26,
                        2., #'social-ml-text-filter-service': 27,
                    ] # baseline: 250-270 after 2 min with max for all
                    manual_caps = np.array(manual_caps, dtype=float)

                    config.qos_target = 500.
                    fpath = './results/evaluation_models/cpu/socialMedia/eval_notransformer/sac/gym_dsb-dsb-social-media-v0/saved_model/microserviceDB.h5'
                    read_to_replay_buffer(agent.memory, fpath, 
                        config.qos_target, config.meet_qos_weight, stack_frames=args.stack_frames,
                        device=device, dvfs=config.dvfs, core_caps=manual_caps,
                        no_unique_id=config.no_unique_id, dvfs_weight=config.dvfs_weight, 
                        prune_features=(not config.no_prune)
                    )

                    config.qos_target = original_qos
                    # agent.dsb_recompute_normalizing_constants(prune_features=(not config.no_prune))

                if not agent.computed_normalizing_constants:
                    agent.dsb_recompute_normalizing_constants(prune_features=(not config.no_prune))
            else:
                print(f"[ALERT] Loaded existing model, but there was no DB file at {os.path.join(fpath, 'microserviceDB.h5')}")

    # # SHAP stuff
    # fpath = os.path.join(config.logdir, 'feature_analysis', 'social_allusercounts_1episode', config.algo, config.env_id.replace(':', '-'), 'saved_model', 'microserviceDB.h5')
    # read_to_replay_buffer(agent.memory, fpath, 
    #     config.qos_target, config.meet_qos_weight, stack_frames=args.stack_frames,
    #     device=device, dvfs=config.dvfs, core_caps=envs.controller.np_manual_caps,
    #     no_unique_id=config.no_unique_id, dvfs_weight=config.dvfs_weight, 
    #     prune_features=(not config.no_prune)
    # )
    
    # subset = [0, 15]
    # # progress = tqdm.tqdm(range(envs.observation_space.shape[0]), dynamic_ncols=True)
    # progress = tqdm.tqdm(subset, dynamic_ncols=True)
    # progress.set_description('Computing Beeswarm Graphs...')
    # for ms in progress:
    #     agent.SHAP(ms)

    # states, actions = analyze_replay_buffer(fpath, 
    #     config.qos_target, config.meet_qos_weight, stack_frames=args.stack_frames,
    #     dvfs=config.dvfs, core_caps=envs.controller.np_manual_caps,
    #     no_unique_id=config.no_unique_id, dvfs_weight=config.dvfs_weight, 
    #     prune_features=(not config.no_prune),
    #     mean=agent.mean.cpu().numpy(), std=agent.std.cpu().numpy(),
    # )

    # progress = tqdm.tqdm(range(len(states)), dynamic_ncols=True)
    # progress.set_description('Computing Beeswarm Graphs for Critical Sections...')

    # for ms in progress:
    #     state = torch.from_numpy(states[ms]).to(torch.float).to(agent.device)
    #     action = torch.from_numpy(actions[ms]).to(torch.float).to(agent.device)
    #     agent.SHAP_blackbox(state, action, f'results/feature_analysis/plots/ms{ms}_critical_beeswarm.png')
    # exit()
        
    max_epochs = int(config.pretrain_tsteps / config.nenvs / config.update_freq)

    progress = range(1, max_epochs + 1)
    progress = tqdm.tqdm(range(1, max_epochs + 1), dynamic_ncols=True)
    progress.set_description("Updates %d, Tsteps %d, FPS %d, mean/median R %.1f/%.1f, min/max R %.1f/%.1f" %
                                (0, 0, 0, 0, 0.0, 0.0, 0.0))

    start = timer()
    for epoch in progress:
        if config.anneal_lr:
            update_linear_schedule(
                agent.optimizer, epoch-1, max_epochs, config.lr)

        for step in range(config.update_freq):
            # current step for env 0
            current_tstep = (epoch-1)*config.update_freq * \
                config.nenvs + step*config.nenvs

            # agent.step(current_tstep, step)

            current_tstep = (epoch) * config.nenvs * config.update_freq

            if current_tstep % config.save_threshold == 0:
                agent.save_w()

            if current_tstep % config.print_threshold == 0 and agent.last_100_rewards:
                update_progress(config, progress, agent, current_tstep, start)

        agent.update(current_tstep)

    # if(agent.last_100_rewards):
    #     update_progress(config, progress, agent, config.max_tsteps, start, log_dir, ipynb)

    agent.save_w()
    # envs.close()

    return agent

def train(agent):
    # set seeds
    if agent.config.seed is not None:
        random.seed(agent.config.seed)
        np.random.seed(agent.config.seed)
        torch.manual_seed(agent.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(agent.config.seed)

    envs = agent.envs
    envs.controller.db_file = os.path.join(agent.log_dir, 'saved_model', 'microserviceDB.h5')

    valid_user_counts = list(range(config.min_users, config.max_users, config.user_step))
    random.shuffle(valid_user_counts)

    ##### INFO GAIN STUFF #####
    # info_gain_user_step = 1
    # if config.random_act > 0:
    #     info_gain_user_step = max(int((config.max_users - config.min_users) * config.exp_time / config.random_act), 1)
    # info_gain_start_offset = random.randint(0, info_gain_user_step-1)
    # valid_user_counts = list(range(config.min_users+info_gain_start_offset, config.max_users+1, info_gain_user_step))
    
    collected_tsteps = 0
    user_count_idx = 0

    if config.use_info_gain:
        info_gain = InfoGain(agent.memory, env=envs, config=config, log_dir=agent.log_dir, tb_writer=agent.tb_writer)

        if config.load_model:
            fpath = os.path.join(config.logdir, 'test', config.algo, config.env_id.replace(':', '-'), 'saved_model')
            info_gain.load_w(fpath)
    ##### END INFO GAIN STUFF #####   

    if config.bootstrap_autoscale or config.use_info_gain:
        previous_cpu, previous_system = np.zeros_like(envs.controller.np_manual_caps), np.zeros_like(envs.controller.np_manual_caps)
        
        if not config.conservative:
            increase_bounds = (0.6, 0.7)
            decrease_bounds = (0.3, 0.4)
        else:
            increase_bounds = (0.3, 0.5)
            decrease_bounds = (0.0, 0.1)

    max_epochs = int(
        agent.config.max_tsteps * agent.config.tstep_real_time // 
        agent.config.exp_time // agent.config.nenvs
    )

    progress = range(1, max_epochs + 1)
    progress = tqdm.tqdm(range(1, max_epochs + 1), dynamic_ncols=True)
    progress.set_description("Updates %d, Tsteps %d, FPS %d, mean/median R %.1f/%.1f, min/max R %.1f/%.1f" %
                                (0, 0, 0, 0, 0.0, 0.0, 0.0))


    current_users = valid_user_counts[user_count_idx]
    user_count_idx += 1

    # for diurnal load
    increase_users = True

    ##### INFO GAIN STUFF #####
    if config.use_info_gain:
        get_keep_prob = lambda x: x / config.info_gain_anneal_steps

        if config.load_replay:
            info_gain.fit(agent.mean, agent.std)
    ##### END INFO GAIN STUFF #####

    start = timer()

    current_tstep = agent.config.pretrain_tsteps
    num_updates = agent.config.pretrain_tsteps + agent.config.learn_start

    for epoch in progress:
        if config.anneal_lr:
            update_linear_schedule(
                agent.optimizer, epoch-1, max_epochs, config.lr)


        envs.controller.reset_cores_and_frequency(use_max_freq=True)

        exp_proc = envs.controller.run_exp(
            duration=agent.config.exp_time, warmup_time=agent.config.warmup_time, 
            users=current_users, workers=0, 
            quiet=True
        )
        
        prev_obs = agent.envs.reset()

        #start timing the first timestep
        exp_start = timer()
        timestep_start = timer()
        consecutive_violations = 0
        exp_failed = False
        recovery = False

        while (timer() - exp_start) < (agent.config.exp_time) and (not exp_failed):
            current_tstep += 1

            if (current_tstep - config.pretrain_tsteps) < agent.config.random_act:
                ##### INFO GAIN STUFF #####
                # if config.use_info_gain:
                #     action = info_gain.get_action(prev_obs, 
                #         envs.controller.most_recent_action, agent.mean, agent.std,
                #         recovery=recovery)
                # elif agent.config.bootstrap_autoscale:
                if agent.config.bootstrap_autoscale:
                    ##### END INFO GAIN STUFF #####
                    action = autoscale_action(
                        envs, envs.controller.most_recent_action,
                        previous_cpu, previous_system,
                        decrease_bounds, increase_bounds,
                        noise=True, dvfs=agent.config.dvfs
                    ).astype(np.float32)
                else:
                    action = envs.action_space.sample()
            else:
                action = agent.get_action(prev_obs, deterministic=agent.config.inference, tstep=current_tstep)

            if config.use_info_gain and (current_tstep - config.pretrain_tsteps) >= agent.config.random_act \
                and (current_tstep - config.pretrain_tsteps - config.random_act) < config.info_gain_anneal_steps:

                action = info_gain.get_action(prev_obs, 
                    action, agent.mean, agent.std,
                    recovery=recovery, 
                    keep_prob=get_keep_prob(current_tstep-config.pretrain_tsteps-config.random_act))

            _, _, done, _ = agent.envs.step(action)

            try:
                time.sleep(
                    agent.config.tstep_real_time - (timer() - timestep_start)
                )
            except ValueError:
                if config.tstep_real_time - (timer() - timestep_start) > 0.2:
                    print(f"Exceeded tstep length {current_tstep} by {config.tstep_real_time - (timer() - timestep_start)}s")

            # roll in everything to next timestep
            timestep_start = timer()

            obs, reward, info = agent.envs.measure()

            if reward >= 0.0:
                consecutive_violations = 0
                recovery = False
            else:
                consecutive_violations += 1
                if consecutive_violations >= 3:
                    recovery = True

            if consecutive_violations >= config.viol_timeout:
                exp_failed = True
            
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

            # Add a done signal if the experiment is ending
            if (timer() - exp_start) < (agent.config.exp_time + agent.config.warmup_time) and (not exp_failed):
                pass
            else:
                done = np.array([True])
            
            agent.append_to_replay(
                prev_obs, action.reshape((1,)+action.shape), 
                reward, obs, done.astype(int)
            )

            agent.update(current_tstep)

            ##### INFO GAIN STUFF #####
            if config.use_info_gain and (current_tstep - config.pretrain_tsteps - config.random_act) < config.info_gain_anneal_steps:
                done_sig = obs if not done[0] else None
                info_gain.add_sample(prev_obs.reshape(envs.observation_space.shape), action, reward[0], done_sig)
            ##### END INFO GAIN STUFF #####

            collected_tsteps += 1

            if agent.tb_writer:
                agent.tb_writer.add_scalar(
                    'Policy/Reward', reward, current_tstep)

            prev_obs = obs

            if current_tstep % config.print_threshold == 0 and agent.last_100_rewards:
                update_progress(config, progress, agent, current_tstep, start)

            if current_tstep % config.save_threshold == 0:
                agent.save_w()

                if config.use_info_gain and (current_tstep - config.pretrain_tsteps - config.random_act) < config.info_gain_anneal_steps:
                    info_gain.save_w()

        # modify the user count in diurnal fashion
        if config.diurnal_load:
            if current_users == agent.config.max_users or current_users == agent.config.min_users:
                increase_users = not increase_users
            if increase_users and current_users < agent.config.max_users:
                current_users += 10
            else:
                current_users -= 10
        else:
            # select user count, add a little randomness, and squash to valid range
            if (not exp_failed) or (collected_tsteps >= int(config.exp_time*0.9)):
                collected_tsteps = 0

                current_users = valid_user_counts[user_count_idx] + random.randint(-5, 5)
                current_users = min(current_users, config.max_users)
                current_users = max(current_users, config.min_users)

                user_count_idx += 1

            if user_count_idx == len(valid_user_counts):
                user_count_idx = 0
                random.shuffle(valid_user_counts)

                if not agent.computed_normalizing_constants:
                    agent.dsb_recompute_normalizing_constants(prune_features=(not config.no_prune))

        exp_proc.kill()

        #max freq for updates and whatnot
        envs.controller.reset_cores_and_frequency(use_max_freq=True)

        ##### INFO GAIN STUFF #####
        if config.use_info_gain and (current_tstep - config.pretrain_tsteps - config.random_act) < config.info_gain_anneal_steps:
            info_gain.append_to_replay()

        # if config.use_info_gain and current_tstep < config.random_act:
        #     info_gain.fit(agent.mean, agent.std, from_scratch=True)
        # elif config.use_info_gain and (current_tstep - config.pretrain_tsteps) < config.info_gain_anneal_steps:
        if config.use_info_gain and (current_tstep - config.pretrain_tsteps >= config.random_act) \
            and (current_tstep - config.pretrain_tsteps - config.random_act < config.info_gain_anneal_steps):
            
            info_gain.fit(agent.mean, agent.std, from_scratch=True)
        ##### END INFO GAIN STUFF #####
        
        
        # # offline updates
        # if (current_tstep - config.pretrain_tsteps) > agent.config.learn_start:
        #     update_progress_bar = range(num_updates, num_updates+config.exp_time)
        #     update_progress_bar = tqdm.tqdm(update_progress_bar, dynamic_ncols=True)
        #     update_progress_bar.set_description("Offline Updates...")

        #     if not agent.computed_normalizing_constants or (num_updates % agent.config.recompute_normalizing_const_freq == 0):
        #         agent.dsb_recompute_normalizing_constants(prune_features=(not config.no_prune))

        #     for _ in update_progress_bar:
        #         for _ in range(config.updates_per_tstep): 
        #             agent.update(num_updates)
        #         num_updates += 1


    if(agent.last_100_rewards):
        update_progress(config, progress, agent, config.max_tsteps, start)

    envs.controller.reset_cores_and_frequency(use_max_freq=True)
    agent.save_w()
    if config.use_info_gain and (current_tstep - config.pretrain_tsteps - config.random_act) < config.info_gain_anneal_steps:
        info_gain.save_w()
    agent.envs.close()

def update_progress(config, progress, agent, current_tstep, start):
    end = timer()
    progress.set_description("Upd. %d, Tsteps %d, FPS %d, mean/median R %.3f/%.3f, min/max R %.3f/%.3f" %
        (
        int(np.max([(current_tstep-config.learn_start)/config.update_freq, 0])),
        current_tstep,
        int(current_tstep  / (end - start)),
        np.mean(agent.last_100_rewards),
        np.median(agent.last_100_rewards),
        np.min(agent.last_100_rewards),
        np.max(agent.last_100_rewards))
        )

if __name__ == '__main__':
    args = parser.parse_args()

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
        'min_users', 'max_users', 'user_step', 'exp_time', 'warmup_time',
        'tstep_real_time', 'diurnal_load', 'meet_qos_weight',
        'load_model', 'load_replay', 'recompute_normalizing_const_freq',
        'no_manual_caps', 'no_unique_id', 'hotel', 'qos_target',
        'random_action_lower_bound', 'bootstrap_autoscale', 'conservative',
        'dvfs', 'dvfs_weight', 'no_prune', 'viol_timeout',
        'updates_per_tstep', 'use_info_gain', 'transfer', 'ewc_lambda',
        'ewc_start', 'info_gain_anneal_steps'
    }

    assert(args.algo == 'sac'), f'Only SAC is supported. You picked {args.algo}'

    # Import Correct Agent
    from agents.SAC import Agent
    valid_arguments = universal_arguments | dqn_arguments
    valid_arguments = valid_arguments - {
        'tnet_update', 'eps_start', 'eps_decay', 'eps_end', 'nenvs',
    }
    valid_arguments = valid_arguments | sac_arguments

    # training params
    config = Config()

    # dsb-specific args
    config.qos_target = args.qos_target
    config.meet_qos_weight = args.meet_qos_weight
    config.gpu_replay = args.gpu_replay
    config.transformer_heads = args.transformer_heads
    config.transformer_layers = args.transformer_layers
    config.transformer_dropout = args.transformer_dropout
    config.hidden_units = args.hidden_units
    config.pretrain_tsteps = args.pretrain_tsteps
    config.min_users = args.min_users
    config.max_users = args.max_users
    config.user_step = args.user_step
    config.exp_time = args.exp_time
    config.warmup_time = args.warmup_time
    config.tstep_real_time = args.tstep_real_time
    config.diurnal_load = args.diurnal_load
    config.load_model = args.load_model
    config.load_replay = args.load_replay
    config.recompute_normalizing_const_freq = args.recompute_normalizing_const_freq
    config.no_manual_caps = args.no_manual_caps
    config.no_unique_id = args.no_unique_id
    config.hotel = args.hotel
    config.random_action_lower_bound = args.random_action_lower_bound
    config.bootstrap_autoscale = args.bootstrap_autoscale
    config.conservative = args.conservative
    config.dvfs = args.dvfs
    config.dvfs_weight = args.dvfs_weight
    config.no_prune = args.no_prune
    config.viol_timeout = args.viol_timeout
    config.updates_per_tstep = args.updates_per_tstep
    config.use_info_gain = args.use_info_gain
    config.transfer = args.transfer
    config.ewc_lambda = args.ewc_lambda
    config.ewc_start = args.ewc_start
    config.info_gain_anneal_steps = args.info_gain_anneal_steps

    # meta info
    config.device = args.device#
    config.algo = args.algo#
    config.env_id = args.env_id#
    config.seed = args.seed#
    config.inference = args.inference#
    config.print_threshold = int(args.print_threshold)#
    config.save_threshold = int(args.save_threshold)#
    config.logdir = args.logdir#
    config.correct_time_limits = args.correct_time_limits#

    # preprocessing
    config.stack_frames = int(args.stack_frames)#

    # Learning Control Variables
    config.max_tsteps = int(args.max_tsteps)#
    config.learn_start = int(args.learn_start)#
    config.random_act = int(args.random_act)#
    config.nenvs = int(args.nenvs)#
    config.update_freq = int(args.update_freq)#
    config.lr = args.lr#
    config.anneal_lr = args.anneal_lr#
    config.grad_norm_max = args.grad_norm_max#
    config.gamma = args.gamma#

    # Optimizer params
    config.optim_eps = args.optim_eps#

    # memory
    config.replay_size = int(args.replay_size)#
    config.batch_size = int(args.batch_size)#
    config.polyak_coef = float(args.polyak_coef)#

    # Multi-step returns
    config.n_steps = int(args.n_steps)#

    # Noisy Nets
    config.noisy_nets = args.noisy_nets#
    config.noisy_sigma = args.noisy_sigma#

    # A2C Controls
    config.entropy_coef = args.entropy_coef
    config.entropy_tuning = args.entropy_tuning #SAC only

    # Priority Replay
    config.priority_replay = args.priority_replay#
    config.priority_alpha = args.priority_alpha#
    config.priority_beta_start = args.priority_beta_start#
    config.priority_beta_steps = args.priority_beta_steps#

    agent = pretrain(config, Agent, valid_arguments, default_arguments)
    train(agent)
