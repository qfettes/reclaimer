# Foreward
Note that these scripts require superuser access to execute, and assume a NOPSSWD policy, so that the scripts themselves may execute commands with `sudo`. I highly envourage you to carefully read DeepRL/utils/dsb_utils/dsb_ctrl.py before execution to understand how root access will be used. Expect over 4 days of data collection and training with overhead for each application.

# Command for running dsb for Hotel reservation
```bash
# step 1, end after random-act
sudo /home/qfettes/miniconda3/envs/ms/bin/python3 train.py --pretrain 0 --replay-size 1000000 --learn-start 57600 --random-act 57600 --lr 1e-4 --gamma 0.9 --entropy-coef 1 --entropy-tuning --stack-frames 3 --exp-time 300 --meet-qos-weight 1.0 --warmup 60 --transformer-layers 3 --transformer-heads 8 --no-prune --viol-timeout 30 --min-users 1000 --max-users 4500 --user-step 100 --qos-target 200 --hotel --bootstrap-autoscale --conservative --max-tsteps 57600

#step 1a: sudo mv results/train results/test

# step 2, end after random-act
sudo /home/qfettes/miniconda3/envs/ms/bin/python3 train.py --pretrain 0 --replay-size 1000000 --learn-start 57600 --random-act 57600 --lr 1e-4 --gamma 0.9 --entropy-coef 1 --entropy-tuning --stack-frames 3 --exp-time 300 --meet-qos-weight 1.0 --warmup 60 --transformer-layers 3 --transformer-heads 8 --no-prune --viol-timeout 30 --min-users 1000 --max-users 4500 --user-step 100 --qos-target 200 --load-replay --hotel --bootstrap-autoscale --conservative --use-info-gain  --max-tsteps 57600

#step 2a: sudo mv results/train results/test

# step 3, go until 260000 total steps
sudo /home/qfettes/miniconda3/envs/ms/bin/python3 train.py --pretrain 0 --replay-size 1000000 --learn-start 0 --random-act 0 --lr 1e-4 --gamma 0.9 --entropy-coef 1 --entropy-tuning --stack-frames 3 --exp-time 300 --meet-qos-weight 1.0 --warmup 60 --transformer-layers 3 --transformer-heads 8 --no-prune --viol-timeout 30 --min-users 1000 --max-users 4500 --user-step 100 --qos-target 200 --load-replay --hotel --bootstrap-autoscale --conservative --use-info-gain --info-gain-anneal-steps 57600

#step 3a: sudo mv results/train results/test

# Command Testing dsb
```bash
sudo /home/qfettes/miniconda3/envs/ms/bin/python3 test.py --qos-target 200 --min-users 20 --max-users 200 --exp-time 300 --meet-qos-weight 0.5 --warmup 60 --hotel --inference --load-model
```
# Command for testing dsb for Social Media Service
```bash
# step 1, end after random-act
sudo /home/qfettes/miniconda3/envs/ms/bin/python3 train.py --pretrain 0 --replay-size 1000000 --learn-start 57600 --random-act 57600 --lr 1e-4 --gamma 0.9 --entropy-coef 1 --entropy-tuning --stack-frames 3 --exp-time 300 --meet-qos-weight 1.0 --warmup 60 --transformer-layers 3 --transformer-heads 8 --no-prune --viol-timeout 30 --min-users 20 --max-users 200 --user-step 100 --qos-target 500 --bootstrap-autoscale --conservative --max-tsteps 57600

#step 1a: sudo mv results/train results/test

# step 2, end after random-act
sudo /home/qfettes/miniconda3/envs/ms/bin/python3 train.py --pretrain 0 --replay-size 1000000 --learn-start 57600 --random-act 57600 --lr 1e-4 --gamma 0.9 --entropy-coef 1 --entropy-tuning --stack-frames 3 --exp-time 300 --meet-qos-weight 1.0 --warmup 60 --transformer-layers 3 --transformer-heads 8 --no-prune --viol-timeout 30 --min-users 20 --max-users 200 --user-step 100 --qos-target 500 --load-replay --bootstrap-autoscale --conservative --use-info-gain  --max-tsteps 57600

#step 2a: sudo mv results/train results/test

# step 3, go until 260000 total steps
sudo /home/qfettes/miniconda3/envs/ms/bin/python3 train.py --pretrain 0 --replay-size 1000000 --learn-start 0 --random-act 0 --lr 1e-4 --gamma 0.9 --entropy-coef 1 --entropy-tuning --stack-frames 3 --exp-time 300 --meet-qos-weight 1.0 --warmup 60 --transformer-layers 3 --transformer-heads 8 --no-prune --viol-timeout 30 --min-users 20 --max-users 200 --user-step 100 --qos-target 500 --load-replay --bootstrap-autoscale --conservative --use-info-gain --info-gain-anneal-steps 57600

#step 3a: sudo mv results/train results/test

# Command Testing dsb
```bash
sudo /home/qfettes/miniconda3/envs/ms/bin/python3 test.py --qos-target 500 --min-users 20 --max-users 200 --exp-time 300 --meet-qos-weight 0.5 --warmup 60 --inference --load-model
```
    
## Requirements: 

* Python>=3.6
* numpy
* scipy
* matplotlib
* notebook
* gym 
* pytorch>=1.3.0
* openCV 
* baselines
* tensorboard
* pytest
* plotly
* pandas
* kaleido
* docker
* ray
* aiohttp
* pybulletgym