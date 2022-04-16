import math

import torch


class Config(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #dsb variables
        self.qos_target = 24.
        self.meet_qos_weight = 0.5
        self.gpu_replay = False
        self.transformer_heads = 4
        self.transformer_layers = 1
        self.transformer_dropout = 0.1
        self.hidden_units = 256
        self.pretrain_tsteps = int(2e5)
        self.min_users = 1
        self.max_users = 52
        self.user_step = 1
        self.exp_time = 600
        self.warmup_time = 120
        self.tstep_real_time = 1.
        self.diurnal_load = False
        self.load_model = False
        self.load_replay = False
        self.recompute_normalizing_const_freq = 1000000
        self.no_manual_caps = False
        self.no_unique_id = False
        self.hotel = False
        self.random_action_lower_bound = 0.0
        self.conservative = False
        self.bootstrap_autoscale = False
        self.dvfs = False
        self.dvfs_weight = 1.0
        self.no_conv=False
        self.viol_timeout = 30
        self.updates_per_tstep = 10
        self.use_info_gain = False

        # meta infor
        self.algo = 'sac'
        self.env_id = 'HalfCheetahPyBulletEnv-v0'
        self.seed = None
        self.inference = False
        self.print_threshold = 10
        self.save_threshold = 300
        self.logdir = './results/train/'
        self.correct_time_limits = False

        # preprocessing
        self.stack_frames = 4
        self.state_norm = 1.0 #not changeable

        # Learning control variables
        self.max_tsteps = int(1e6)
        self.learn_start = int(1000)
        self.random_act = int(1e4)
        self.nenvs = 1
        self.update_freq = 50
        self.lr = 3e-3
        self.anneal_lr = False
        self.grad_norm_max = 40.0
        self.gamma = 0.99

        # Optimizer params
        self.optim_eps = 1e-8

        # Replay memory
        self.replay_size = int(1e6)
        self.batch_size = 100
        self.polyak_coef = 0.995

        # Multi-step returns
        self.n_steps = 1

        # Noisy Nets
        self.noisy_nets = False
        self.noisy_sigma = 0.5

        # a2c controls
        self.entropy_coef = 0.2
        self.entropy_tuning = False

        # priority replay
        self.priority_replay = False
        self.priority_alpha = 0.6
        self.priority_beta_start = 0.4
        self.priority_beta_steps = int(2e7)


        #THESE AREN'T CHANGABLE BY THE TRAINING SCRIPT
        # epsilon variables
        self.eps_start = 1.0
        self.eps_end = [0.1, 0.01]
        self.eps_decay = [0.05, 0.5]
