import time, subprocess, sys, argparse, os, ray

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--rmi', action='store_true', default=False,
                    help='remove images after down (default: False)')

def stop_deathstarbench(docker_compose_file, quiet=True, rmi=False):
    cmd = f'docker-compose -f {str(docker_compose_file)} down -v --remove-orphans'
    if rmi:
        cmd += ' --rmi all'
    __docker_compose_helper(cmd, quiet, blocking=True)

def __docker_compose_helper(cmd, quiet=True, blocking=False):
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

def start_docker(quiet=True):
    cmd = 'sudo systemctl start docker'
    __docker_compose_helper(cmd, quiet, blocking=True)

def stop_docker(quiet=True):
    cmd = 'sudo systemctl stop docker'
    __docker_compose_helper(cmd, quiet, blocking=True)

def __clear_ray_files():
    ray.shutdown()

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

if __name__ == '__main__':
    cd = os.path.join('/', *__file__.split('/')[:-1])
    args = parser.parse_args()

    # all_workload_files = (
    #     os.path.join(cd, 'DeepRL/utils/locust/docker-compose-socialml.yml'),
    #     os.path.join(cd, 'DeepRL/utils/locust/docker-compose-hotel.yml'),
    # )

    # all_benchmark_files = (
    #     os.path.join(cd, 'Sinan/benchmarks/socialNetwork-ml-swarm/docker-compose.yml'),
    #     os.path.join(cd, 'DeathStarBench/hotelReservation/docker-compose.yml')
    # )
    all_workload_files = (
        os.path.join('.', 'DeepRL/utils/locust/docker-compose-socialml.yml'),
        os.path.join('.', 'DeepRL/utils/locust/docker-compose-hotel.yml'),
    )

    all_benchmark_files = (
        os.path.join('.', 'Sinan/benchmarks/socialNetwork-ml-swarm/docker-compose.yml'),
        os.path.join('.', 'DeathStarBench/hotelReservation/docker-compose.yml')
    )

    for workload_f in all_workload_files:
        _ = stop_deathstarbench(workload_f, quiet=False, rmi=args.rmi)
    
    for benchmark_f in all_benchmark_files:
        _ = stop_deathstarbench(benchmark_f, quiet=False, rmi=args.rmi)

    __clear_ray_files()

    stop_docker()
    time.sleep(1)

    start_docker()