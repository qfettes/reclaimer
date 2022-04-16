import numpy as np
import torch
import docker
import os
import multiprocessing
# from multiprocessing import Pool
import ray
from ray.util.multiprocessing import Pool
import h5py
import sys 
import subprocess
import threading
import time
import tqdm
import tempfile
import csv
import re

from timeit import default_timer as timer
from random import shuffle

from collections import deque

# sys.path.insert(0, '../DeathStarBench/socialNetwork/scripts/')
sys.path.insert(0, '../Sinan/benchmarks/socialNetwork-ml-swarm/scripts/')
from init_social_graph import init_social_graph

banned_containers = ('socialnetwork_jaeger_1', 'hotel_reserv_jaeger', 
    'socialnetwork-ml-swarm_jaeger_1', 'social-network_jaeger_1',
    'hotel_reserv_geo', 'hotelreservation_mongodb-geo', 
    'hotelreservation_consul_1')

# super hacky and bad but we need it for multiprocessing
#   I will fix if I ever have time
containers = []
 
class dsb_controller(object):
    def __init__(self, social=True, dvfs=False, privilaged=True, restart_containers=True, db_file='../Data/microserviceDB_socialNetwork.h5', use_manual_caps=True):
        super().__init__()

        self.max_cpu_utilization = {}
        self.previous_cpu = {}
        self.previous_system = {}

        self.dvfs = dvfs
        self.privilaged = privilaged
        self.restart_containers = restart_containers
        self.use_manual_caps = use_manual_caps
        self.social = social

        if self.privilaged:
            self.__clear_ray_files()
        ray.shutdown()
        ray.init()

        # get docker
        self.client = docker.from_env()

        #workload
        # this compose file should be selected via logic based on specific benchmark
        self.all_workload_files = (
            './utils/locust/docker-compose-socialml.yml',
            './utils/locust/docker-compose-hotel.yml'
        )
        if self.social:
            self.workload_compose_file = './utils/locust/docker-compose-socialml.yml'
        else:
            self.workload_compose_file = './utils/locust/docker-compose-hotel.yml'
        assert os.path.isfile(self.workload_compose_file)

        # this compose file should be selected via logic based on specific benchmark
        self.all_benchmark_files = (
            '../Sinan/benchmarks/socialNetwork-ml-swarm/docker-compose.yml',
            '../DeathStarBench/hotelReservation/docker-compose.yml'
        )
        if self.social:
            self.benchmark_compose_file = '../Sinan/benchmarks/socialNetwork-ml-swarm/docker-compose.yml'
        else:
            self.benchmark_compose_file = '../DeathStarBench/hotelReservation/docker-compose.yml'
        self.num_microservices = 0
        self.changed_container_idxes = []
        self.positional_encoding = {}
        self.name_by_position = {}
        self.valid_containers = {
            'social-ml-nginx-thrift', 'social-ml-text-service', 'social-ml-home-timeline-service',
            'social-ml-write-home-timeline-service', 'social-ml-user-service', 'social-ml-user-timeline-service',
            'social-ml-write-user-timeline-service', 'social-ml-compose-post-service', 'social-ml-post-storage-service',
            'social-ml-url-shorten-service', 'social-ml-compose-post-redis', 'social-ml-social-graph-redis',
            'social-ml-media-service', 'social-ml-user-timeline-mongodb', 'social-ml-post-storage-memcached',
            'social-ml-media-filter-service', 'social-ml-social-graph-mongodb','social-ml-user-mention-service',
            'social-ml-user-mongodb', 'social-ml-social-graph-service', 'social-ml-user-memcached',
            'social-ml-write-user-timeline-rabbitmq', 'social-ml-post-storage-mongodb', 'social-ml-home-timeline-redis', 
            'social-ml-user-timeline-redis', 'social-ml-write-home-timeline-rabbitmq', 'social-ml-unique-id-service',
            'social-ml-text-filter-service', 'hotelreservation_user', 'hotelreservation_rate',
            'hotelreservation_reservation', 'hotelreservation_profile', 'hotelreservation_recommendation',
            'hotelreservation_search', 'hotelreservation_frontend', 'hotelreservation_mongodb-rate_1', 
            'hotelreservation_mongodb-user', 'hotelreservation_memcached-reserve_1', 'hotelreservation_memcached-profile', 
            'hotelreservation_memcached-rate', 'hotelreservation_mongodb-profile', 'hotelreservation_mongodb-reservation', 
            'hotelreservation_mongodb-recommendation', 'socialnetwork_jaeger_1', 
            'hotel_reserv_jaeger', 'socialnetwork-ml-swarm_jaeger_1', 'social-network_jaeger_1', 'resource-sink'
        }
        self.init_containers()
        self.init_container_info()


        self.num_physical_cores = 0
        self.num_wildcard_physical_cores = 0
        self.all_physical_cores = set()
        self.always_occupied_physical_cores = set()
        self.init_core_information()

        # some frequency constants
        # Real min/max is 1000000/2300000 but scaling to these values
        #   gives each discrete mode equal prob when rounding
        self.max_freq = 2349999 #kHz
        self.min_freq = 950000 #kHz
        self.valid_frequencies = tuple(range(1000000, 2400000, 100000))
        self.freq_base_path = "/sys/devices/system/cpu/"
        print(f'CPU Frequency files at {self.freq_base_path}')

        # locust variables
        self.locust_log_path = './results/locust_log/'
        print(f'Locust logs at {self.locust_log_path}')

        # database storage
        self.db_file = db_file
        print(f'Transition file at {self.db_file}')
        self.__add_done_signal_to_db()

        # keeping track of actions
        self.most_recent_action = None

        self.reset_cores_and_frequency(use_max_freq=True)
        self.get_features() # Initializes the global counters

    def __add_done_signal_to_db(self):
        if os.path.isfile(self.db_file):
            with h5py.File(self.db_file, 'r+') as h5f:
                tmp = h5f['microserviceDB'][-1]
                tmp['done_signal'][0] = True
                h5f['microserviceDB'][-1] = tmp

    def init_containers(self):
        global containers

        if self.privilaged:
            self.__stop_docker()
            time.sleep(5)

            self.__start_docker()
            time.sleep(5)

        for workload_f in self.all_workload_files:
            _ = self.__stop_deathstarbench(workload_f, quiet=True)
        
        if self.restart_containers:
            for benchmark_f in self.all_benchmark_files:
                _ = self.__stop_deathstarbench(benchmark_f, quiet=True)

            if self.privilaged:
                self.__stop_docker()
                time.sleep(5)

                self.__start_docker()
                time.sleep(5)

            self.__start_deathstarbench(self.benchmark_compose_file, quiet=True)

            # This is high overhead. Get the list of containers
            #   ahead of time. We will reload as needed
            containers = self.client.containers.list()
            containers = [x for x in containers if x.name in self.valid_containers]

            #check that everyone is here
            container_names = set([x.name for x in containers])
            for name in self.positional_encoding.keys():
                if name not in container_names:
                    print(f"Failed Init: {name} not initialized!")
                    return False

            if self.social:
                time.sleep(1)
                return init_social_graph()
            else:
                return True
        else:
            return True
            
    def init_container_info(self):
        global containers

        # Init feature structures
        self.total_rx_packets = {}
        self.total_rx_bytes = {}
        self.total_tx_packets = {}
        self.total_tx_bytes = {}
        self.total_page_faults = {}
        self.total_cpu_time = {}
        self.total_ret_io_bytes = {}
        self.total_ret_io_serviced = {}
        for container in containers:
            if container.name in banned_containers+('resource-sink',):
                continue
            self.total_rx_packets[container.name] = 0
            self.total_rx_bytes[container.name]   = 0
            self.total_tx_packets[container.name] = 0
            self.total_tx_bytes[container.name]   = 0
            self.total_page_faults[container.name] = 0
            self.total_cpu_time[container.name] = 0
            self.total_ret_io_bytes[container.name] = 0
            self.total_ret_io_serviced[container.name] = 0

        # get a list of containers pids for later feature collection
        self.containers_pids = {}
        self.get_container_pids()

        # keep track of which containers change
        self.changed_container_idxes = []

        # get number of real microservices, excluding jaeger
        self.num_microservices = len(containers) - 1

        if self.social:
            self.positional_encoding = {
                'social-ml-nginx-thrift': 0,
                'social-ml-text-service': 1,
                'social-ml-home-timeline-service': 2,
                'social-ml-write-home-timeline-service': 3,
                'social-ml-user-service': 4, 
                'social-ml-user-timeline-service': 5,
                'social-ml-write-user-timeline-service': 6,
                'social-ml-compose-post-service': 7,
                'social-ml-post-storage-service': 8,
                'social-ml-url-shorten-service': 9,
                'social-ml-compose-post-redis': 10,
                'social-ml-social-graph-redis': 11,
                'social-ml-media-service': 12,
                'social-ml-user-timeline-mongodb': 13,
                'social-ml-post-storage-memcached': 14,
                'social-ml-media-filter-service': 15,
                'social-ml-social-graph-mongodb': 16,
                'social-ml-user-mention-service': 17,
                'social-ml-user-mongodb': 18,
                'social-ml-social-graph-service': 19,
                'social-ml-user-memcached': 20,
                'social-ml-write-user-timeline-rabbitmq': 21,
                'social-ml-post-storage-mongodb': 22,
                'social-ml-home-timeline-redis': 23,
                'social-ml-user-timeline-redis': 24,
                'social-ml-write-home-timeline-rabbitmq': 25,
                'social-ml-unique-id-service': 26,
                'social-ml-text-filter-service': 27,
            }
        else:
            self.positional_encoding = {
                'hotelreservation_frontend': 0,
                'hotelreservation_profile': 1,
                'hotelreservation_search': 2,
                'hotelreservation_rate': 3,
                'hotelreservation_recommendation': 4,
                'hotelreservation_user': 5,
                'hotelreservation_reservation': 6,
                'hotelreservation_memcached-rate': 7,
                'hotelreservation_memcached-profile': 8,
                'hotelreservation_memcached-reserve_1': 9,
                'hotelreservation_mongodb-profile': 10,
                'hotelreservation_mongodb-rate_1': 11,
                'hotelreservation_mongodb-recommendation': 12,
                'hotelreservation_mongodb-reservation': 13,
                'hotelreservation_mongodb-user': 14
            }

        self.manual_caps = [float(os.cpu_count()) for _ in self.positional_encoding.keys()]

        if self.use_manual_caps:
            if self.social:
                self.manual_caps = [
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
                ] 
            else:
                self.manual_caps = [
                    12, #'hotelreservation_frontend': 7,
                    4, #'hotelreservation_profile': 4,
                    8, #'hotelreservation_search': 6,
                    4, #'hotelreservation_rate': 1,
                    4, #'hotelreservation_recommendation': 5,
                    4, #'hotelreservation_user': 0,
                    4, #'hotelreservation_reservation': 3,
                    4, #'hotelreservation_memcached-rate': 13,
                    2, #'hotelreservation_memcached-profile': 12,
                    2, #'hotelreservation_memcached-reserve_1': 11,
                    8, #'hotelreservation_mongodb-profile': 14,
                    8, #'hotelreservation_mongodb-rate_1': 8,
                    8, #'hotelreservation_mongodb-recommendation': 17
                    8, #'hotelreservation_mongodb-reservation': 15,
                    8, #'hotelreservation_mongodb-user': 10,
                ]
                self.manual_caps = [x for x in self.manual_caps]

        
        for key, val in self.positional_encoding.items():
            self.name_by_position[val] = key
        
        self.np_manual_caps = np.array(self.manual_caps, dtype=float)

    def __start_docker(self, quiet=True):
        cmd = 'sudo systemctl start docker'
        self.__docker_compose_helper(cmd, quiet, blocking=True)

    def __stop_docker(self, quiet=True):
        cmd = 'sudo systemctl stop docker'
        self.__docker_compose_helper(cmd, quiet, blocking=True)

    def __start_deathstarbench(self, docker_compose_file, quiet=True):
        cmd = f'docker-compose -f {str(docker_compose_file)} up -d'
        self.__docker_compose_helper(cmd, quiet, blocking=True)

    def __stop_deathstarbench(self, docker_compose_file, quiet=True):
        cmd = f'docker-compose -f {str(docker_compose_file)} down -v --remove-orphans'
        self.__docker_compose_helper(cmd, quiet, blocking=True)

    def get_container_pids(self):
        global containers
        for container in containers:
            if container.name in banned_containers+('resource-sink',):
                continue
            self.containers_pids[container.name] = [process[1] for process in container.top()['Processes']]   

    def close_containers(self):
        self.__stop_deathstarbench(self.benchmark_compose_file, quiet=True)

    def run_exp(self, duration, warmup_time, users, workers=0, quiet=False):
        # Reinit everything to prevent weird behavior?
        if not self.restart_containers: # still stop old exp if not reinit
            _ = self.__stop_deathstarbench(self.workload_compose_file, quiet=True)

        success = False
        while self.restart_containers and not success:
            success = self.init_containers()
        self.init_container_info()

        # append done signal to the end of the last exp
        self.__add_done_signal_to_db()

        #refresh container pids
        self.get_container_pids()

        locust_proc = None
        if workers == 0:
            workers = max(1, users // 2)
            if not self.social:
                workers = max(1, users // 10)

        # env variables
        cmd = 'USERS=' + str(users) + ' EXP_TIME=' + str(duration + warmup_time) + 's '
        cmd += 'docker-compose -f ' + str(self.workload_compose_file) + \
            ' up --scale worker=' + str(workers)
        
        locust_proc = self.__docker_compose_helper(cmd, quiet=quiet, blocking=False)
        
        if warmup_time > 0:
            # assert(False), "Don\'t use warmup times, the logic is bad right now"
            time.sleep(warmup_time)

        assert locust_proc != None

        # verify that we are actually warmed up
        initialized_users = self.__get_user_count()
        if abs(users - initialized_users) > (0.1 * users):
            print(f"Failed Experiment Warmup. Expected {users} users, got {initialized_users} users")
            locust_proc = self.run_exp(duration, warmup_time, users, workers, quiet) 

        return locust_proc

    def init_core_information(self):
        self.num_logical_cores = os.cpu_count()

        # / 2 because we have hyperthreading
        # 0-39 are the first logical cores, 40-79 are the second
        #   run cpupower monitor for more info
        self.num_physical_cores = os.cpu_count() // 2

        # each ms gets at least 1 CPU
        # self.num_wildcard_physical_cores = self.num_physical_cores - self.num_microservices
        self.all_physical_cores = set(range(self.num_physical_cores))
        self.always_occupied_physical_cores = set(range(0, self.num_microservices))

        if self.privilaged:
            self.__set_cpu_governor()

    def reload_container_attrs(self, container_idxes):
        global containers

        # NOTE: doing container.reload() in parallel with multiprocessing pools
        #   does not work for some reason
        for idx in container_idxes:
            container = containers[idx]
            if container.name not in banned_containers+('resource-sink',):
                containers[idx].reload()

    def reset_cores_and_frequency(self, all_cores_all_containers=True, use_max_freq=True):
        if not self.privilaged:
            return

        global containers

        reset_idxes = [idx for idx, c in enumerate(containers) if c.name not in banned_containers+('resource-sink',)]

        # set all microservices to run on all cores
        most_recent_core_action = None
        assigned_containers = {}
        for c in containers:
            if c.name in banned_containers+('resource-sink',):
                continue
            assigned_containers[c.name] = [self.positional_encoding[c.name]] + [i for i in range(self.num_logical_cores) if i != self.positional_encoding[c.name]]

        # allow execution on all cores
        _ = [
                cpu_affinity_update_one_container.remote(
                    containers[idx].id,
                    assigned_containers[containers[idx].name],
                    self.num_physical_cores
                ) for idx in reset_idxes
            ]

        #make sure max utility
        _ = [
                cpu_utility_update_one_container.remote(
                    self.name_by_position[idx],
                    self.manual_caps[idx]
                ) for idx in range(self.num_microservices)
            ]

        # reset all frequencies to max
        if self.dvfs:
            if use_max_freq:
                freq = self.max_freq
            else:
                freq = self.min_freq

            for core in range(os.cpu_count()):
                set_core_frequency(core, return_closest(freq, self.valid_frequencies), self.freq_base_path)

        if self.dvfs:
            self.most_recent_action = np.ones((len(containers)-1, 2), dtype=float)
        else:
            self.most_recent_action = np.ones((len(containers)-1, 1), dtype=float)

        #update the containers to reflect change
        self.reload_container_attrs(reset_idxes)
    
    def execute_action(self, actions):
        """
        Takes action array then computes and assigns approprate resource
        allocations (CPUs and Frequency)

        args:
            actions (np.ndarray): (27, 2) element tensor of floating point values [0-1]
                denoting the approapriate resource allocation. The order is cpu_cores,
                frequncy

        """
        assert(isinstance(actions, np.ndarray)), f"Actions should be given as a np.ndarray, got {type(actions)}"
        assert(actions.shape == self.most_recent_action.shape), f"Your action shape changed from {self.most_recent_action.shape} to {actions.shape}"

        if self.dvfs:
            assert(actions.size == (self.num_microservices)*2), f"Expected {(self.num_microservices)*2} actions, but got {actions.size}"
            self.__allocate_cpu(actions[:,0])
            self.set_all_frequency(actions[:,1])
        else:
            assert(actions.size == self.num_microservices), f"Expected {self.num_microservices} actions, but got {actions.size}"
            self.__allocate_cpu(actions[:,0])

        self.most_recent_action = actions

    # Just allocate cpu utilization cap
    def __allocate_cpu(self, actions):
        _ = [
            cpu_utility_update_one_container.remote(
                self.name_by_position[idx],
                actions[idx] * self.manual_caps[idx]
            ) for idx in range(self.num_microservices)
        ]

    def set_all_frequency(self, freq_actions):
        #assumes actions are [0, 1]
        global containers

        # set the correct driver with: 
        #   https://unix.stackexchange.com/questions/153693/cant-use-userspace-cpufreq-governor-and-set-cpu-frequency
        #   https://silvae86.github.io/2020/06/13/switching-to-acpi-power/#changing-to-acpi-cpufreq-cpu-management-driver
        # set the correct governor with: sudo cpupower frequency-set -g userspace
        # frequency files are found at /sys/devices/system/cpu/cpu<0-79>/cpufreq/scaling_setspeed
        #   - values are written in kHz. 1000000 kHz = 1 GHz
        #   - valid values found with: cpupower frequency-info
        #   2.30 GHz, 2.30 GHz, 2.20 GHz, 2.10 GHz, 2.00 GHz, 1.90 GHz, 1.80 GHz, 1.70 GHz, 1.60 GHz,
        #   1.50 GHz, 1.40 GHz, 1.30 GHz, 1.20 GHz, 1.10 GHz, 1000 MHz
        frequencies = freq_actions * (self.max_freq - self.min_freq) + self.min_freq

        for container in containers:
            if container.name not in banned_containers:
                all_cores = list(range(self.num_logical_cores))

                idx = self.positional_encoding[container.name]

                a = None
                if container.name != 'resource-sink':
                    a = return_closest(frequencies[idx], self.valid_frequencies)
                else:
                    a = return_closest(0 + self.min_freq, self.valid_frequencies)
                
                for core in all_cores:
                    set_core_frequency(core, a, self.freq_base_path)

    def get_features(self):
        epoch_rx_packets, epoch_rx_bytes, epoch_tx_packets, epoch_tx_bytes = self.__get_network_features()
        epoch_rss, epoch_cache_memory, epoch_page_faults = self.__get_memory_features()
        epoch_cpu_time = self.__get_cpu_features()
        epoch_ret_io_bytes, epoch_ret_io_serviced = self.__get_io_features()
        end_to_end_lat = self.__get_latency_data()

        return (epoch_rx_packets, epoch_rx_bytes, epoch_tx_packets, epoch_tx_bytes), \
            (epoch_rss, epoch_cache_memory, epoch_page_faults), (epoch_cpu_time,), \
                (epoch_ret_io_bytes, epoch_ret_io_serviced), end_to_end_lat

    # TODO: Rework this feature to store in batches rather than single data points
    def store_features(self, network_features, memory_features, cpu_features, io_features, latency_data, action_data):
        np_epoch_rx_packets, np_epoch_rx_bytes, np_epoch_tx_packets, \
            np_epoch_tx_bytes, np_epoch_rss, np_epoch_cache_memory, \
            np_epoch_page_faults, np_epoch_cpu_time, np_epoch_ret_io_bytes, \
            np_epoch_ret_io_serviced, np_latency_data \
            = self.raw_data_to_numpy(
                network_features, memory_features, cpu_features, 
                io_features, latency_data
            )

        if self.dvfs:
            action_desc = ('action_data', float, (self.num_microservices, 2))
        else:
            action_desc = ('action_data', float, (self.num_microservices, 1))

        # create np rec array from data
        ds_dtype = [
            ('timestamp', int, (1,)),
            ('rx_packets', int, (self.num_microservices, )),
            ('rx_bytes', int, (self.num_microservices, )),
            ('tx_packets', int, (self.num_microservices, )),
            ('tx_bytes', int, (self.num_microservices, )),

            ('rss', float, (self.num_microservices, )),
            ('cache_memory', float, (self.num_microservices, )),
            ('page_faults', int, (self.num_microservices, )),

            ('cpu_time', int, (self.num_microservices, )),

            ('ret_io_bytes', int, (self.num_microservices, )),
            ('ret_io_serviced', int, (self.num_microservices, )),

            ('latency_data', float, np_latency_data.shape),

            action_desc,

            ('done_signal', bool, (1,))
        ]

        # create recarray to append
        ds_arr = np.recarray((1,), dtype=ds_dtype)
        ds_arr['timestamp'] = np.array([int(time.time())], dtype=int)
        ds_arr['rx_packets'] = np_epoch_rx_packets
        ds_arr['rx_bytes'] = np_epoch_rx_bytes
        ds_arr['tx_packets'] = np_epoch_tx_packets
        ds_arr['tx_bytes'] = np_epoch_tx_bytes
        ds_arr['rss'] = np_epoch_rss
        ds_arr['cache_memory'] = np_epoch_cache_memory
        ds_arr['page_faults'] = np_epoch_page_faults
        ds_arr['cpu_time'] = np_epoch_cpu_time
        ds_arr['ret_io_bytes'] = np_epoch_ret_io_bytes
        ds_arr['ret_io_serviced'] = np_epoch_ret_io_serviced
        ds_arr['latency_data'] = np_latency_data
        ds_arr['action_data'] = action_data
        ds_arr['done_signal'] = np.asarray([False], dtype=bool).reshape((1,))

        if os.path.exists(self.db_file):
            dset = h5f = h5py.File(self.db_file, 'a')

            idx = dset['microserviceDB'].shape[0]

            # extend dataset
            dset['microserviceDB'].resize(idx+1, axis=0)

            dset['microserviceDB'][idx:idx+1] = ds_arr
        else:
            h5f = h5py.File(self.db_file, 'w')
            dset = h5f.create_dataset('microserviceDB', data=ds_arr, maxshape=(None,), chunks=True)

        h5f.close()

    def raw_data_to_numpy(self, network_features, memory_features, cpu_features, io_features, latency_data):
        epoch_rx_packets, epoch_rx_bytes, epoch_tx_packets, epoch_tx_bytes = network_features
        epoch_rss, epoch_cache_memory, epoch_page_faults = memory_features
        epoch_cpu_time = cpu_features[0]
        epoch_ret_io_bytes, epoch_ret_io_serviced = io_features

        np_epoch_rx_packets = np.zeros((self.num_microservices, ), dtype=int)
        np_epoch_rx_bytes = np.zeros((self.num_microservices, ), dtype=int)
        np_epoch_tx_packets = np.zeros((self.num_microservices, ), dtype=int)
        np_epoch_tx_bytes = np.zeros((self.num_microservices, ), dtype=int)

        np_epoch_rss = np.zeros((self.num_microservices, ), dtype=float)
        np_epoch_cache_memory = np.zeros((self.num_microservices, ), dtype=float)
        np_epoch_page_faults = np.zeros((self.num_microservices, ), dtype=int)

        np_epoch_cpu_time = np.zeros((self.num_microservices, ), dtype=int)

        np_epoch_ret_io_bytes = np.zeros((self.num_microservices, ), dtype=int)
        np_epoch_ret_io_serviced = np.zeros((self.num_microservices, ), dtype=int)

        for name, pos in self.positional_encoding.items():
            if name == 'resource-sink':
                continue

            try:
                np_epoch_rx_packets[pos] = epoch_rx_packets[name]
                np_epoch_rx_bytes[pos] = epoch_rx_bytes[name]
                np_epoch_tx_packets[pos] = epoch_tx_packets[name]
                np_epoch_tx_bytes[pos] = epoch_tx_bytes[name]
            except:
                np_epoch_rx_packets[pos] = 0
                np_epoch_rx_bytes[pos] = 0
                np_epoch_tx_packets[pos] = 0
                np_epoch_tx_bytes[pos] = 0

            try:
                np_epoch_rss[pos] = epoch_rss[name]
                np_epoch_cache_memory[pos] = epoch_cache_memory[name]
                np_epoch_page_faults[pos] = epoch_page_faults[name]
            except:
                np_epoch_rss[pos] = 0
                np_epoch_cache_memory[pos] = 0
                np_epoch_page_faults[pos] = 0

            try:
                np_epoch_cpu_time[pos] = epoch_cpu_time[name]
            except:
                np_epoch_cpu_time[pos] = 0

            try:
                np_epoch_ret_io_bytes[pos] = epoch_ret_io_bytes[name]
                np_epoch_ret_io_serviced[pos] = epoch_ret_io_serviced[name]
            except:
                np_epoch_ret_io_bytes[pos] = 0
                np_epoch_ret_io_serviced[pos] = 0

        np_latency_data = np.asarray(
            [
                latency_data['timestamp'],
                latency_data['rps'],
                latency_data['fps'],
                latency_data['request'],
                latency_data['failure'],
                latency_data['50.0'],
                latency_data['66.0'],
                latency_data['75.0'],
                latency_data['80.0'],
                latency_data['90.0'],
                latency_data['95.0'],
                latency_data['98.0'],
                latency_data['99.0'],
                latency_data['99.9'],
                latency_data['99.99'],
                latency_data['99.999'],
                latency_data['100.0']
            ],
            dtype=float
        )
    
        return np_epoch_rx_packets, np_epoch_rx_bytes, np_epoch_tx_packets, \
            np_epoch_tx_bytes, np_epoch_rss, np_epoch_cache_memory, \
            np_epoch_page_faults, np_epoch_cpu_time, np_epoch_ret_io_bytes, \
            np_epoch_ret_io_serviced, np_latency_data

    def __get_network_features(self):
        global containers
        
        rx_packets = {}
        rx_bytes = {}
        tx_packets = {}
        tx_bytes = {}

        epoch_rx_packets = {}
        epoch_rx_bytes = {}
        epoch_tx_packets = {}
        epoch_tx_bytes = {}

        for container in containers:
            if container.name in banned_containers+('resource-sink',):
                continue

            rx_packets[container.name] = 0
            rx_bytes[container.name]   = 0
            tx_packets[container.name] = 0
            tx_bytes[container.name]   = 0

            for pid in self.containers_pids[container.name]:
                fname = f'/proc/{pid}/net/dev'

                if os.path.exists(fname):
                    try:
                        with open(fname, 'r') as f:
                            lines = f.readlines()

                            for line in lines:
                                if 'Inter-|   Receive' in line or 'face |bytes    packets errs' in line:
                                    continue
                                else:
                                    data = line.split(' ')
                                    data = [d for d in data if (d != '' and '#' not in d and ":" not in d)]
                                    rx_packets[container.name] += int(data[1])
                                    rx_bytes[container.name] += int(data[0])
                                    tx_packets[container.name] += int(data[9])
                                    tx_bytes[container.name] += int(data[8])
                    except (OSError, FileNotFoundError):
                        rx_packets[container.name] = 0
                        rx_bytes[container.name] = 0
                        tx_packets[container.name] = 0
                        tx_bytes[container.name] = 0

            epoch_rx_packets[container.name] = rx_packets[container.name] - self.total_rx_packets[container.name]
            epoch_rx_bytes[container.name] = rx_bytes[container.name] - self.total_rx_bytes[container.name]
            epoch_tx_packets[container.name] = tx_packets[container.name] - self.total_tx_packets[container.name]
            epoch_tx_bytes[container.name] = tx_bytes[container.name] - self.total_tx_bytes[container.name]

            self.total_rx_packets[container.name] = rx_packets[container.name]
            self.total_rx_bytes[container.name] = rx_bytes[container.name]
            self.total_tx_packets[container.name] = tx_packets[container.name]
            self.total_tx_bytes[container.name] = tx_bytes[container.name]

            if epoch_rx_packets[container.name] <= 0:
                epoch_rx_packets[container.name] = 0
            if epoch_rx_bytes[container.name] <= 0:
                epoch_rx_bytes[container.name] = 0
            if epoch_tx_packets[container.name] <= 0:
                epoch_tx_packets[container.name] = 0
            if epoch_tx_bytes[container.name] <= 0:
                epoch_tx_bytes[container.name] = 0

        return epoch_rx_packets, epoch_rx_bytes, epoch_tx_packets, epoch_tx_bytes

    def __get_memory_features(self):
        global containers

        epoch_rss = {}
        epoch_cache_memory = {}
        epoch_page_faults = {}

        for container in containers:
            if container.name in banned_containers+('resource-sink',):
                continue

            fname = f'/sys/fs/cgroup/memory/docker/{container.id}/memory.stat'
            if os.path.exists(fname):

                try:
                    with open(fname, 'r') as f:
                        lines = f.readlines()

                        for line in lines:
                            if 'total_cache' in line:
                                epoch_cache_memory[container.name] = round(int(line.split(' ')[1])/(1024.0**2), 3)	# turn byte to mb
                            elif 'total_rss' in line and 'total_rss_huge' not in line:
                                epoch_rss[container.name] = round(int(line.split(' ')[1])/(1024.0**2), 3)
                            elif 'total_pgfault' in line:
                                pf = int(line.split(' ')[1])
                                epoch_page_faults[container.name] = pf - self.total_page_faults[container.name]
                                self.total_page_faults[container.name] = pf
                except (OSError, FileNotFoundError):
                    epoch_cache_memory[container.name] = 0
                    epoch_rss[container.name] = 0
                    epoch_page_faults[container.name] = 0
            else:
                epoch_cache_memory[container.name] = 0
                epoch_rss[container.name] = 0
                epoch_page_faults[container.name] = 0
                self.total_page_faults[container.name] = 0

            if epoch_rss[container.name] <= 0:
                epoch_rss[container.name] = 0
            if epoch_cache_memory[container.name] <= 0:
                epoch_cache_memory[container.name] = 0
            if epoch_page_faults[container.name] <= 0:
                epoch_page_faults[container.name] = 0
                self.total_page_faults[container.name] = 0
            

        return epoch_rss, epoch_cache_memory, epoch_page_faults

    def __get_cpu_features(self):
        global containers

        epoch_cpu_time = {}

        for container in containers:
            if container.name in banned_containers+('resource-sink',):
                continue

            fname = f'/sys/fs/cgroup/cpuacct/docker/{container.id}/cpuacct.usage'
            if os.path.exists(fname):
                try:
                    with open(fname, 'r') as f:
                        cum_cpu_time = int(f.readlines()[0])/1000000.0	# turn ns to ms
                        epoch_cpu_time[container.name] = max(cum_cpu_time - self.total_cpu_time[container.name], 0)
                        self.total_cpu_time[container.name] = cum_cpu_time
                except (OSError, FileNotFoundError):
                    epoch_cpu_time[container.name] = 0
            else:
                cum_cpu_time = 0	# turn ns to ms
                epoch_cpu_time[container.name] = max(cum_cpu_time - self.total_cpu_time[container.name], 0)
                self.total_cpu_time[container.name] = 0
            
            if epoch_cpu_time[container.name] <= 0:
                epoch_cpu_time[container.name] = 0

        return epoch_cpu_time

    def __get_io_features(self):
        global containers

        epoch_ret_io_bytes = {}
        epoch_ret_io_serviced = {}

        for container in containers:
            if container.name in banned_containers+('resource-sink',):
                continue

            fname = f'/sys/fs/cgroup/blkio/docker/{container.id}/blkio.throttle.io_service_bytes_recursive'
            if os.path.exists(fname):
                try:
                    with open(fname, 'r') as f:
                        lines = f.readlines()
                        
                        if len(lines) > 0:
                            sector_num = int(lines[0].split(' ')[-1])
                            epoch_ret_io_bytes[container.name] = sector_num - self.total_ret_io_bytes[container.name]
                            if epoch_ret_io_bytes[container.name] < 0:
                                epoch_ret_io_bytes[container.name] = 0
                        else:
                            sector_num = 0
                            epoch_ret_io_bytes[container.name] = 0
                        
                        self.total_ret_io_bytes[container.name] = sector_num
                except (OSError, FileNotFoundError):
                    epoch_ret_io_bytes[container.name] = 0
            else:
                epoch_ret_io_bytes[container.name] = 0
                self.total_ret_io_bytes[container.name] = 0
            
            fname = f'/sys/fs/cgroup/blkio/docker/{container.id}/blkio.throttle.io_serviced_recursive'
            if os.path.exists(fname):
                try:
                    with open(fname, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if 'Total' in line:
                                serv_num = int(line.split(' ')[-1])
                                epoch_ret_io_serviced[container.name] = serv_num - self.total_ret_io_serviced[container.name]
                                if epoch_ret_io_serviced[container.name] < 0:
                                    epoch_ret_io_serviced[container.name] = serv_num
                                self.total_ret_io_serviced[container.name] = serv_num
                except (OSError, FileNotFoundError):
                    epoch_ret_io_serviced[container.name] = 0
            else:
                epoch_ret_io_serviced[container.name] = 0
                self.total_ret_io_serviced[container.name] = 0

        return epoch_ret_io_bytes, epoch_ret_io_serviced

    def __get_latency_data(self):
        # NOTE: There is more information in this file
        #   consider using it later.
        end_to_end_lat = {}

        try:
            name = 'social_stats_history.csv'
            if not self.social:
                name = 'hotel_stats_history.csv'

            with open(os.path.join(self.locust_log_path, name), 'r') as f:
                lines = f.readlines()
                assert len(lines) > 1
                fields = lines[0].split(',')

                # "Timestamp","User Count","Type","Name","Requests/s","Failures/s","50%","66%","75%","80%","90%","95%","98%","99%","99.9%","99.99%","99.999%","100%","Total Request Count","Total Failure Count"
                pos = {}
                pos['timestamp'] = None
                pos['50%'] = None
                pos['66%'] = None
                pos['75%'] = None
                pos['80%'] = None
                pos['90%'] = None
                pos['95%'] = None
                pos['98%'] = None
                pos['99%'] = None
                pos['99.9%'] = None
                pos['99.99%'] = None
                pos['99.999%'] = None
                pos['100%'] = None
                pos['rps'] = None
                pos['fps'] = None
                pos['request'] = None
                pos['failure'] = None
                for i, k in enumerate(fields):
                    k = k.replace('\"', '').strip()
                    if k == '50%':
                        pos['50%'] = i
                    elif k == '66%':
                        pos['66%'] = i
                    elif k == '75%':
                        pos['75%'] = i
                    elif k == '80%':
                        pos['80%'] = i
                    elif k == '90%':
                        pos['90%'] = i
                    elif k == '95%':
                        pos['95%'] = i
                    elif k == '98%':
                        pos['98%'] = i
                    elif k == '99%':
                        pos['99%'] = i
                    elif k == '99.9%':
                        pos['99.9%'] = i
                    elif k == '99.99%':
                        pos['99.99%'] = i
                    elif k == '99.999%':
                        pos['99.999%'] = i
                    elif k == '100%':
                        pos['100%'] = i
                    elif k == 'Timestamp':
                        pos['timestamp'] = i
                    elif k == 'Requests/s':
                        pos['rps'] = i
                    elif k == 'Failures/s':
                        pos['fps'] = i
                    elif k == 'Total Request Count':
                        pos['request'] = i
                    elif k == 'Total Failure Count':
                        pos['failure'] = i

                data = lines[-1].split(',')
                try:
                    end_to_end_lat['timestamp'] = _get_int_val(data[ pos['timestamp'] ])
                except:
                    end_to_end_lat['timestamp'] = 0

                try:
                    end_to_end_lat['fps'] = _get_float_val(data[ pos['fps'] ]) # failures/s
                    end_to_end_lat['rps'] = _get_float_val(data[ pos['rps'] ]) # requests/s
                except:
                    end_to_end_lat['fps'] = 0
                    end_to_end_lat['rps'] = 0

                try:
                    end_to_end_lat['50.0'] = _get_int_val(data[ pos['50%'] ])
                    end_to_end_lat['66.0'] = _get_int_val(data[ pos['66%'] ])
                    end_to_end_lat['75.0'] = _get_int_val(data[ pos['75%'] ])
                    end_to_end_lat['80.0'] = _get_int_val(data[ pos['80%'] ])
                    end_to_end_lat['90.0'] = _get_int_val(data[ pos['90%'] ])
                    end_to_end_lat['95.0'] = _get_int_val(data[ pos['95%'] ])
                    end_to_end_lat['98.0'] = _get_int_val(data[ pos['98%'] ])
                    end_to_end_lat['99.0'] = _get_int_val(data[ pos['99%'] ])
                    end_to_end_lat['99.9'] = _get_int_val(data[ pos['99.9%'] ])
                    end_to_end_lat['99.99'] = _get_int_val(data[ pos['99.99%'] ])
                    end_to_end_lat['99.999'] = _get_int_val(data[ pos['99.999%'] ])
                    end_to_end_lat['100.0'] = _get_int_val(data[ pos['100%'] ])
                except:
                    end_to_end_lat['50.0'] = 0
                    end_to_end_lat['66.0'] = 0
                    end_to_end_lat['75.0'] = 0
                    end_to_end_lat['80.0'] = 0
                    end_to_end_lat['90.0'] = 0
                    end_to_end_lat['95.0'] = 0
                    end_to_end_lat['98.0'] = 0
                    end_to_end_lat['99.0'] = 0
                    end_to_end_lat['99.9'] = 0
                    end_to_end_lat['99.99'] = 0
                    end_to_end_lat['99.999'] = 0
                    end_to_end_lat['100.0'] = 0

                try:
                    end_to_end_lat['request'] = _get_int_val(data[ pos['request'] ])
                    end_to_end_lat['failure'] = _get_int_val(data[ pos['failure'] ])
                except:
                    end_to_end_lat['request'] = 0
                    end_to_end_lat['failure'] = 0
        except (OSError, FileNotFoundError):
            end_to_end_lat['timestamp'] = 0
            end_to_end_lat['fps'] = 0
            end_to_end_lat['rps'] = 0
            end_to_end_lat['50.0'] = 0
            end_to_end_lat['66.0'] = 0
            end_to_end_lat['75.0'] = 0
            end_to_end_lat['80.0'] = 0
            end_to_end_lat['90.0'] = 0
            end_to_end_lat['95.0'] = 0
            end_to_end_lat['98.0'] = 0
            end_to_end_lat['99.0'] = 0
            end_to_end_lat['99.9'] = 0
            end_to_end_lat['99.99'] = 0
            end_to_end_lat['99.999'] = 0
            end_to_end_lat['100.0'] = 0
            end_to_end_lat['request'] = 0
            end_to_end_lat['failure'] = 0

        return end_to_end_lat

    def __get_user_count(self):
        users = 0

        try:
            name = 'social_stats_history.csv'
            if not self.social:
                name = 'hotel_stats_history.csv'

            with open(os.path.join(self.locust_log_path, name), 'r') as f:
                lines = f.readlines()
                assert len(lines) > 1
                fields = lines[0].split(',')

                # "Timestamp","User Count","Type","Name","Requests/s","Failures/s","50%","66%","75%","80%","90%","95%","98%","99%","99.9%","99.99%","99.999%","100%","Total Request Count","Total Failure Count"
                pos = {}
                pos['users'] = None
                for i, k in enumerate(fields):
                    k = k.replace('\"', '').strip()
                    if k == 'User Count':
                        pos['users'] = i

                data = lines[-1].split(',')
                try:
                    users = _get_int_val(data[ pos['users'] ])
                except:
                    users = 0

        except (OSError, FileNotFoundError):
            users = 0

        return users

    def __docker_compose_helper(self, cmd, quiet=True, blocking=False):
        _stdout = sys.stdout
        _stderr = sys.stderr
        if quiet:
            _stdout = subprocess.DEVNULL
            _stderr = subprocess.DEVNULL

        print(cmd)

        if blocking:
            dsb_docker_compose_proc = subprocess.call(cmd, shell=True,
                stdout=_stdout, stderr=_stderr)
        else:
            dsb_docker_compose_proc = subprocess.Popen(cmd, shell=True,
                stdout=_stdout, stderr=_stderr)

        assert dsb_docker_compose_proc != None
        return dsb_docker_compose_proc

    def __set_cpu_governor(self):
        # With dvfs, set gov to userspace and disable boosting
        if self.dvfs:
            cmd = 'sudo cpupower frequency-set -g userspace && echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost > /dev/null'
        # When no dvfs, set gov to performance and enable turbo boost
        else:
            cmd = 'sudo cpupower frequency-set -g performance && echo 1 | sudo tee /sys/devices/system/cpu/cpufreq/boost > /dev/null'
        _stdout = subprocess.DEVNULL
        _stderr = sys.stderr

        print(cmd)
        proc = subprocess.call(cmd, shell=True, stdout=_stdout, stderr=_stderr)

        assert proc != None

    def __clear_ray_files(self):
        _stdout = subprocess.DEVNULL
        _stderr = sys.stderr

        cmd = 'sudo rm -rf /tmp/ray/'
        print(cmd)
        proc = subprocess.call(cmd, shell=True, stdout=_stdout, stderr=_stderr)
        assert proc != None

        cmd = 'sudo ps aux | grep ray::IDLE | grep -v grep | awk \'{print $2}\' | xargs kill -9'
        print(cmd)
        proc = subprocess.call(cmd, shell=True, stdout=_stdout, stderr=_stderr)
        assert proc != None

    def get_containers(self):
        return containers


@ray.remote
def cpu_utility_update_one_container(container_name, cpus, blkio_weight=None): 
    _stdout = subprocess.DEVNULL
    # _stdout = sys.stdout
    _stderr = sys.stderr
    # _stderr = subprocess.DEVNULL

    if blkio_weight is None:
        cmd = f'docker update --cpus={round(cpus, ndigits=2)} {container_name}'
    else:
        cmd = f'docker update --cpus={round(cpus, ndigits=2)} --blkio-weight={blkio_weight} {container_name}'
    proc = subprocess.call(cmd, shell=True, stdout=_stdout, stderr=_stderr)
    assert proc != None

    return proc
            
def check_users(interval_s):
    cmd = 'who'
    _stdout = subprocess.PIPE

    proc = subprocess.Popen(cmd, shell=True, stdout=_stdout)

    users = proc.stdout.read().decode('utf-8')
    with open('../Data/user_history.txt', 'a') as f:
        f.write(f"{int(time.time())}\n" + users + '-')
    
    threading.Timer(interval_s,check_users, args=(interval_s)).start()

def set_core_frequency(core, frequency, freq_base_path):
    f_name = os.path.join(freq_base_path, f"cpu{core}/cpufreq/scaling_setspeed")

    assert(os.path.exists(f_name)), f"The scaling_setspeed does not exist at the following path: {f_name}"
    with open(f_name, 'w') as f:
        f.write(str(frequency) + '\n')

def return_closest(n, value_set):
    return min(value_set, key=lambda x: abs(x-n))

@ray.remote
def cpu_affinity_update_one_container(container_id, cores, num_physical_cores): 
    client = docker.from_env()
    container = client.containers.get(container_id)
    
    cpu_str = str(cores[0]) if cores else ''
    for cpu_id in cores[1:]:
        cpu_str += ','
        cpu_str += str(cpu_id)

    _ = container.update(
        cpuset_cpus=cpu_str
    )      

def _get_int_val(str_val):
    str_val = str_val.replace('\"', '')
    if 'N/A' in str_val:
        return 0
    else:
        return int(str_val)

def _get_float_val(str_val):
    str_val = str_val.replace('\"', '')
    if 'N/A' in str_val:
        return 0.0
    else:
        return float(str_val)
        
def read_to_replay_buffer(replay_buffer, db_file, QoS, meet_QoS_weight, 
    core_caps, no_unique_id=False, stack_frames=1, device=None, 
    dvfs=False, social=True, dvfs_weight=1.0, conv=True,
    prune_features=True):

    device = torch.device(device) if device else device
    # all_data = deque(maxlen=1000000)
    with h5py.File(db_file, 'r+') as h5f:
        tmp = h5f['microserviceDB'][-1]
        tmp['done_signal'][0] = True
        h5f['microserviceDB'][-1] = tmp

        buffer = deque(maxlen=stack_frames)
        prev_state = None

        # num_read = min(replay_buffer._maxsize, 1000)
        if replay_buffer._maxsize < h5f['microserviceDB'].shape[0]:
            progress = tqdm.tqdm(h5f['microserviceDB'][-replay_buffer._maxsize:], dynamic_ncols=True)
        else:
            # # Some code to eliminate samples from the DB permananetly
            # h5f2 = h5py.File(os.path.join(*db_file.split('/')[:-1], 'microserviceDB2.h5'), 'w')
            # dset = h5f2.create_dataset('microserviceDB', data=h5f['microserviceDB'][:-5900], maxshape=(None,), chunks=True)
            # exit()
            progress = tqdm.tqdm(h5f['microserviceDB'], dynamic_ncols=True) 
        progress.set_description(f"Loading Data from {db_file} into Replay Buffer...")

        for tstep in progress:
            ######## We already have the previous state #########

            ######## action ########
            # TODO: This is a temporary fix to the hotel exploit
            action = tstep['action_data'].astype(np.float32)
            if np.sum(action) < 0.001:
                print("ignoring transition")
                continue
                # print("BAD ACTION ", idx, np.sum(action))

            ######## reward ########

            reward = compute_reward(tstep['latency_data'][-5], QoS, meet_QoS_weight, 
                action=action, core_caps=core_caps, dvfs=dvfs, dvfs_weight=dvfs_weight)

            reward = np.array(reward, dtype=np.float32)

            # first collect microservice specific features
            all_vals = []
            for name in tstep.dtype.names[1:-3]:
                # # NOTE: correction to put cpu time in cpus, not 1000*cpus
                # # done here to keep compat with old data
                # if 'cpu_time' in name:
                #     all_vals.append(tstep[name]/1000.0)
                #     continue

                all_vals.append(tstep[name])

            next_state = form_obs(all_vals, tstep['latency_data'], tstep['action_data'], core_caps, prune_features=prune_features)
            if conv:
                next_state = np.expand_dims(next_state, axis=1)


            if stack_frames > 1:
                if len(buffer) == 0:
                    for _ in range(stack_frames-1):
                        buffer.append(np.copy(next_state))
                buffer.append(next_state)
                next_state = np.concatenate(buffer, axis=1)

            qos_vec = np.array(
                [QoS/500. for _ in range(next_state.shape[0])],
                dtype=np.float
            ).reshape(-1, 1)
            if conv:
                qos_vec = qos_vec.reshape(-1, 1, 1)
                qos_vec = np.repeat(qos_vec, next_state.shape[1], axis=1)

            if not no_unique_id:
                # generate the unique id for each microservice
                # unique_id = np.eye(next_state.shape[0])
                # (num social-media + 1 resource sink) + (num hotel + 1 resource sink)
                
                #TODO: Fix this for compatibility mode
                unique_id = np.eye(29 + 19)
                beg = 0
                if not social:
                    beg = 29
                unique_id = unique_id[beg:(beg+next_state.shape[0])]
                if conv:
                    unique_id = np.expand_dims(unique_id, axis=1)
                    unique_id = np.repeat(unique_id, next_state.shape[1], axis=1)

                # append unique ids to next state
                next_state = np.concatenate((next_state, unique_id, qos_vec), axis=-1)

                # create a dummy initial state if we don't have one
                if prev_state is None:
                    prev_state = np.copy(next_state)
            elif prev_state is None:
                next_state = np.concatenate((next_state, qos_vec), axis=1)
                prev_state = np.copy(next_state)

            if tstep['done_signal'][0]:
                next_state = None
                buffer.clear()

            # convert to pytorch buffer
            if device:
                if type(prev_state) == np.ndarray:
                    prev_state = torch.from_numpy(prev_state).to(torch.float).to(device)
                action = torch.from_numpy(action).to(torch.float).to(device)
                reward = torch.from_numpy(reward).to(torch.float).to(device)
                if next_state is not None:
                    next_state = torch.from_numpy(next_state).to(torch.float).to(device)

            # store features
            replay_buffer.push((prev_state, action, reward, next_state))
                
            prev_state = next_state


def form_obs(all_vals, latency_data, action_data, core_caps, prune_features=False):
    next_state = np.stack(all_vals, axis=1) # these are the ms specific features

    extra_features = [next_state]

    new_lat_data = latency_data[1:]
    if prune_features:
        new_lat_data = new_lat_data[[0, 1, 2, 3, 8, 9, 10, 11, 12]]

    # append latency data to all microservice features
    extra_features.append(
        np.repeat(
            new_lat_data.reshape(1, -1), 
            next_state.shape[0], 
            axis=0
        )
    )

    extra_features.append(core_caps.reshape(-1, 1) / os.cpu_count())

    # add action as a feature for this data
    extra_features.append(action_data)

    # # Utilization / cap
    # extra_features.append(next_state[:,7:8] / core_caps.reshape(-1, 1))

    # # Utilization / allocation
    # extra_features.append(next_state[:,7:8] / ((action_data[:,0:1]+1e-8) * core_caps.reshape(-1, 1)))

    # put it all together
    next_state = np.concatenate(extra_features, axis=1)

    return next_state

def compute_reward(tail_latency, QoS, meet_QoS_weight, action, core_caps, dvfs, dvfs_weight):
    assert((type(tail_latency) == np.float32 or type(tail_latency) == np.float64 or type(tail_latency) == float) and type(QoS) == float), f"Wrong type {type(tail_latency)} {type(QoS)}"
    reward = -1.
    # reward = -(np.abs(QoS - tail_latency) / QoS)

    if tail_latency <= QoS:
        reward = 0.
        if dvfs:
            high = 1.
            # The min frequency is actually 1000MhZ
            #   rescale the reward such that you can only get
            #   1Ghz/2.3GhZ = low/high modifier to reward at best
            low = 1./2.3
            reward += (meet_QoS_weight * (1. - np.dot(action[:,0]*(action[:,1]*(high-low)+low)**dvfs_weight, core_caps) / np.sum(core_caps)))
        else:
            reward += (meet_QoS_weight * (1. - np.dot(action[:,0], core_caps) / np.sum(core_caps)))
    
    #clip the reward, could have been < -1.0 before
    if dvfs:
        reward = np.clip(reward, -1.0, 1.0)

    # return reward / 10.
    return reward / 10.
            
def analyze_replay_buffer(db_file, QoS, meet_QoS_weight, 
    core_caps, no_unique_id=False, stack_frames=1,
    dvfs=False, social=True, dvfs_weight=1.0, conv=True,
    prune_features=True, mean=None, std=None):

    action_diff = []
    states = []
    actions = []
    q_prev_action = None
    done_signal = True

    # all_data = deque(maxlen=1000000)
    with h5py.File(db_file, 'r+') as h5f:
        tmp = h5f['microserviceDB'][-1]
        tmp['done_signal'][0] = True
        h5f['microserviceDB'][-1] = tmp

        buffer = deque(maxlen=stack_frames)
        prev_state = None

        # num_read = min(replay_buffer._maxsize, 1000)
        progress = tqdm.tqdm(h5f['microserviceDB'][-3000:], dynamic_ncols=True) 
        progress.set_description(f"Loading Data from {db_file} into Replay Buffer...")

        for idx, tstep in enumerate(progress):
            ######## We already have the previous state #########

            ######## action ########
            # TODO: This is a temporary fix to the hotel exploit
            action = tstep['action_data'].astype(np.float32)
            if np.sum(action) < 0.001:
                print("ignoring transition")
                continue
                # print("BAD ACTION ", idx, np.sum(action))
            actions.append(action)

            
            if done_signal:
                action_diff.append(np.zeros_like(action))
                done_signal = False
            else:
                action_diff.append(np.abs(action - q_prev_action))
            q_prev_action = action

            # first collect microservice specific features
            all_vals = []
            for name in tstep.dtype.names[1:-3]:
                # # NOTE: correction to put cpu time in cpus, not 1000*cpus
                # # done here to keep compat with old data
                # if 'cpu_time' in name:
                #     all_vals.append(tstep[name]/1000.0)
                #     continue

                all_vals.append(tstep[name])

            next_state = form_obs(all_vals, tstep['latency_data'], tstep['action_data'], core_caps, prune_features=prune_features)
            if conv:
                next_state = np.expand_dims(next_state, axis=1)


            if stack_frames > 1:
                if len(buffer) == 0:
                    for _ in range(stack_frames-1):
                        buffer.append(np.copy(next_state))
                buffer.append(next_state)
                next_state = np.concatenate(buffer, axis=1)

            qos_vec = np.array(
                [QoS/500. for _ in range(next_state.shape[0])],
                dtype=np.float
            ).reshape(-1, 1)
            if conv:
                qos_vec = qos_vec.reshape(-1, 1, 1)
                qos_vec = np.repeat(qos_vec, next_state.shape[1], axis=1)

            if not no_unique_id:
                # generate the unique id for each microservice
                # unique_id = np.eye(next_state.shape[0])
                # (num social-media + 1 resource sink) + (num hotel + 1 resource sink)
                
                #TODO: Fix this for compatibility mode
                unique_id = np.eye(29 + 19)
                beg = 0
                if not social:
                    beg = 29
                unique_id = unique_id[beg:(beg+next_state.shape[0])]
                if conv:
                    unique_id = np.expand_dims(unique_id, axis=1)
                    unique_id = np.repeat(unique_id, next_state.shape[1], axis=1)

                # append unique ids to next state
                next_state = np.concatenate((next_state, unique_id, qos_vec), axis=-1)

                # create a dummy initial state if we don't have one
                if prev_state is None:
                    prev_state = np.copy(next_state)
            elif prev_state is None:
                next_state = np.concatenate((next_state, qos_vec), axis=1)
                prev_state = np.copy(next_state)

            if tstep['done_signal'][0]:
                next_state = None
                buffer.clear()
                done_signal = True

            states.append(prev_state)
                
            prev_state = next_state

    all_states = []
    all_actions = []
    states = np.array(states)[10:]
    actions = np.array(actions)[10:]
    action_diff = np.array(action_diff)[10:]
    action_diff = np.transpose(action_diff.reshape(action_diff.shape[:2]))

    k_actions = 300
    for ms in range(states.shape[1]):
        state = states[:,ms,:,:]
        action = actions[:,ms,:].reshape(-1)
        single_action_diff = action_diff[ms]
        
        top_k_action_diffs = np.argpartition(single_action_diff, -k_actions)[-k_actions:]
        state = state[top_k_action_diffs]
        action = action[top_k_action_diffs]
        all_states.append(state)
        all_actions.append(action)
    return all_states, all_actions