# Foreword
The following code depends on old versions of DSB. Docker-compose, as it is used in these scripts, will attempt to pull docker images which may not be hosted at some point in the future. For preservation, we include the version of DeathStarBench which we used for these experiments (all credit to the original repository, https://github.com/delimitrou/DeathStarBench). The original images will be possible to build manually from the code provided (instructions below). Docker images from other open-source projects are also used by DeathStarBench (e.g. mongoDB). The code for those is not available in this repository, and if they become unavailable in the future, we recommend using a newer version of these images.

To modify the docker images used by the applications, refer to:
- Social Media Service: ./Sinan/benchmarks/socialNetwork-ml-swarm/docker-compose.yml
- Hotel Reservation Service: ./DeathStarBench/hotelReservation/docker-compose.yml

## Building DSB 
The two application from DeathStarBench used by Recalimer are located at the following paths:
- ./Sinan/benchmarks/socialNetwork-ml-swarm/
- ./DeathStarBench/hotelReservation/

The original authors of DeathStarBench provide a DOCKERFILE and docker-compose files to build and deploy those application. To build an application, use the following command:

`docker build -t qfettes/social-network-microservices`

Follow up by replacing the images in docker-compose.yml with the image you built, for each application you rebuild them for. 

## Using a newer version of DSB
Reclaimer should function as intended with newer versions of DSB. However, the current names of docker containers are hardcoded throughout DeepRL/utils/dsb_utils/dsb_ctrl.py, and the number of containers are hardcoded throughout DeepRL/utils/dsb_utils/dsb_ctrl.py, /home/qfettes/code/reclaimer/DeepRL/customEnvs/gym-dsb/gym_dsb/envs/dsb_social_media.py, DeepRL/utils/wrappers.py, and DeepRL/agents/SAC.py. 

# System Requirements
All experiments are performed on a server with dual socket, Intel Xeon Gold 6230N 20-core, 40-thread CPUs, and two Tesla V100 GPUs. The operating system is Ubuntu 20.04.4 LTS. The CPU driver is set to acpi-cpufreq, and is set to use the performance governor with CPU frequency boosting enabled. The GPU driver is 510.54, with CUDA version 11.6. The docker version is 20.10.13, build a224086, and the docker-compose version is 1.29.2, build 5becea4c. we used gcc 9.4.0, cmake version 3.16.3, and GNU Make 4.2.1.

## Changing your CPU driver
While instructions may change, at the time of writing [this](https://silvae86.github.io/2020/06/13/switching-to-acpi-power/#changing-to-acpi-cpufreq-cpu-management-driver) guide was used to set the appropriate CPU DRIVE

## Installing DeathStarBench
- Navigate to the official [DeathStarBench repository](https://github.com/delimitrou/DeathStarBench) and clone the repository to ./DeathStarBench
- Follow all install instructions given in their readme

### Add your username to the docker group (avoids repeated use of sudo)
`sudo groupadd docker`

`sudo gpasswd -a <your username> docker`

`sudo service docker restart`

## Packages to install via apt
- cpufrequtils==008-1.1 (used to set cpugovernor)
- docker==1.5-2
- docker-compose==1.25.0-1
- python3
- libssl-dev==1.1.1f-1ubuntu2.12
- zlib1g-dev==1:1.2.11.dfsg-2ubuntu1.2
- luarocks==2.4.2+dfsg-1

## Python Packages
- See requirements.txt
- Additionally, openai baselines (tf2 branch) is required to provide some openai gym environment wrappers. The commit version is at the following URL (https://github.com/openai/baselines/tree/tf2)
- Finally traverse to DeepRL/customEnvs/gym-dsb and execute `pip install -e .`

## luarocks packages
`luarocks install luasocket`

# After Installation
`cd DeepRL`
- Follow the README in that directory to execute experiments

# Additional Notes
- TO view the mechanisms for interacting with DSB and modifying this code to work with other applications, see code/reclaimer/DeepRL/utils/dsb_utils/dsb_ctrl.py 
- To view the wrapper as a gym environment for ease of use with RL algorithms, see code/reclaimer/DeepRL/customEnvs/gym-dsb


