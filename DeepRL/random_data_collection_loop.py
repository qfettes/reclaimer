# from utils.dsb_utils.dsb_trace import captureRPS
from utils.dsb_utils.dsb_ctrl import dsb_controller, check_users
from utils.dsb_utils.dsb_plot import plot_stress_test_data
import numpy as np
import time
import os
import random


from timeit import default_timer as timer

import json

if __name__ == '__main__':
    min_users = 100
    max_users = 330
    exp_time = 60 * 5 #5 minutes
    warmup = 60 # 1 minute
    timestep_length = 1. #seconds

    controller = dsb_controller()
    check_users(300)

    #get initial state with totals set to 0
    _ = controller.get_features()
    network_features, memory_features, cpu_features, io_features, latency_data = controller.get_features()
    controller.store_features(network_features, memory_features, cpu_features, io_features, latency_data, controller.most_recent_action)

    # start timing the timesteps
    timestep_start = timer()

    while True:
        exp_start = timer()

        exp_proc = controller.run_exp(duration=exp_time, warmup_time=warmup, users=random.randint(min_users, max_users), workers=0, quiet=True)

        while (timer() - exp_start) < (exp_time + warmup):
            #you have state

            # select action
            cpu_action = np.random.randn(27, 1)
            top = np.exp(cpu_action)
            bottom = np.sum(top)
            cpu_action = top / bottom
            freq_action = np.random.rand(27, 1)
            action = np.concatenate((cpu_action, freq_action), axis=1)

            # execute action
            controller.execute_action(action)

            #wait
            if (timer() - timestep_start) < 1.:
                try:
                    time.sleep(
                        1. - (timer() - timestep_start)
                    )
                except ValueError:
                    pass

            #observe reward and next state
            network_features, memory_features, cpu_features, io_features, latency_data = controller.get_features()

            #store action, next_state, reward
            controller.store_features(network_features, memory_features, cpu_features, io_features, latency_data, controller.most_recent_action)

            timestep_start = timer()

        exp_proc.kill()


    #################### DVFS ACTION TEST ####################
    # import time

    # for i in range(5):
    #     dummy_action = np.random.random((27, 2))
    #     dummy_action[np.random.randint(0, 27, 5), 0] += 5
    #     dummy_action[:,0] = np.exp(dummy_action[:,0]) / np.sum(np.exp(dummy_action[:,0]))

    #     controller.execute_action(dummy_action)
    #     time.sleep(0.4)

    # #################### RESET TEST ####################

    # controller.reset_cores_and_frequency(use_max_freq=False)

    # from utils.dsb_utils.dsb_ctrl import containers

    # for container in containers:
    #     if container.name != 'resource-sink':
    #         cpu_str = container.attrs['HostConfig']['CpusetCpus']
    #         print(container.name, cpu_str)
    # print('*'*20)

    # for core in range(os.cpu_count()):
    #     f_name = os.path.join(controller.freq_base_path, f"cpu{core}/cpufreq/scaling_setspeed")

    #     assert(os.path.exists(f_name)), f"The scaling_setspeed does not exist at the following path: {f_name}"
    #     with open(f_name, 'r') as f:
    #         print(f.readline().strip())
    # print('*'*20)

    #################### capture traces test ####################

    # start = timer()

    # # 100ms of latency; nearly all is from the curl call
    # # this is fine, we can just call it 100 ms early in its own process,
    # #   do stuff for 100ms, then use the state/reward accordingly
    # # NOTE: for a frame stack, we will get the time-offset most recent frame
    # #   then n-1 time-offset-i*interval_length frames for all i = 1...n-1
    # #   total of n frames
    # #   Thus, we should request all frames from [now-offset-interval_length*n, now-offset]
    # #   i.e. start of frame 1 to end of frame n (the n in the above formula is intentional)
    # #   After getting this whole interval, chop up into the discrete frames we need
    # # NOTE: We also need to queue up frames until the correct reward is available. E.g.
    # #   if we are colleting frames every interval, and we have an offset, a frame should
    # #   not be added to the replay buffer for offset/interval tsteps; this is when the correct
    # #   latency information will be available to calculate a reward
    # # captureTraces(int(2 * 1000 * 1000) ,int(2 * 1000 * 1000))
    # # captureTraces(0 ,int(2.0 * 1000 * 1000))

    # captureRPS(int(2 * 1000 * 1000))
    # captureRPS(int(2 * 1000 * 1000))

    # print(timer() - start)

    #################### feature collection test ####################
    # controller.get_features()
    # print('Q'*200)
    # controller.get_features()

    #################### feature collection test ####################
    # import time
    # seconds = 10
    # controller.db_file = 'tmp.h5' # don't mess with the real database if it exists

    # start = timer()
    # p_locust = run_locust_docker_compose('utils/locust/docker-compose-socialml.yml', seconds, 10, workers=0, quiet=True)

    # while timer() - start < seconds:
    #     now = timer()

    #     controller.get_features()

    #     time.sleep(1.0 - (timer() - now))

    # p_locust.kill()

    # import h5py
    # with h5py.File('tmp.h5', 'r') as f:
    #     print(f['microserviceDB']['cpu_time'].shape)
    #     print(f['microserviceDB'])

    # #################### Reset Cores and Frequency ####################
    # controller.reset_cores_and_frequency(use_max_freq=True)

    # #################### Plot stress data ####################
    # rootdir = '/home/qfettes/code/microservices/Data/stress_test_data_socialrps10/'
    # plot_stress_test_data(rootdir)