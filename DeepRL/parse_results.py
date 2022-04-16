import glob
import os
import numpy as np
import scipy

from utils.plot import plot_reward
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.font_manager import FontProperties

def read_data(rootdir, qos, hotel=False):
    mean_cpu = {}
    max_cpu = {}

    tail_latencies = {}
    violation_rates = {}

    for _, dirs, _ in os.walk(rootdir):
        for subdir in sorted(dirs, key=lambda x: (len(x), x)):
            key = int(subdir)

            # get mean/max action
            action_file = os.path.join(rootdir, subdir, 'core_allocation_sum.txt')
            if not os.path.isfile(action_file): #compat with old naming
                action_file = os.path.join(rootdir, subdir, 'action_sum.txt')


            with open(action_file, 'r') as f:
                all_cpu_allocations = [float(x) for x in f.readline().strip().split(',') if x]

                mean_cpu[key] = np.mean(all_cpu_allocations)
                max_cpu[key] = np.max(all_cpu_allocations)

            # get violation rate
            single_exp_tail_latencies = []

            if hotel:
                name = 'hotel_stats_history.csv'
            else:
                name = 'social_stats_history.csv'
                
            with open(os.path.join(rootdir, subdir, name), 'r') as f:
                lines = f.readlines()
                assert len(lines) > 1
                fields = lines[0].split(',')

                # "Timestamp","User Count","Type","Name","Requests/s","Failures/s","50%","66%","75%","80%","90%","95%","98%","99%","99.9%","99.99%","99.999%","100%","Total Request Count","Total Failure Count"
                pos = {}
                tail_key = '99%'
                pos[tail_key] = None
                for i, k in enumerate(fields):
                    k = k.replace('\"', '').strip()
                    if k == tail_key:
                        pos[tail_key] = i

                # TODO: warmup is hard-coded here
                count = 0
                violations = 0

                beg = 61
                for line in lines[beg:]:
                    data = line.strip().split(',')
                    single_exp_tail_latencies.append(int(data[pos[tail_key]]))

                    count += 1
                    if single_exp_tail_latencies[-1] > qos:
                        violations +=1
            
            print(np.std(single_exp_tail_latencies))
            tail_latencies[key] = np.mean(single_exp_tail_latencies)
            violation_rates[key] = float(violations)/float(count)


    print(os.path.join(rootdir, subdir, name))
    print('8'*30)
    return mean_cpu, max_cpu, tail_latencies, violation_rates

def read_data_detailed(rootdir):
    tail_latencies = {}

    for _, dirs, _ in os.walk(rootdir):
        for subdir in sorted(dirs, key=lambda x: (len(x), x)):
            key = int(subdir)

            # get latency
            single_exp_tail_latencies = []

            name = 'hotel_stats_history.csv'
                
            with open(os.path.join(rootdir, subdir, name), 'r') as f:
                lines = f.readlines()
                assert len(lines) > 1
                fields = lines[0].split(',')

                # "Timestamp","User Count","Type","Name","Requests/s","Failures/s","50%","66%","75%","80%","90%","95%","98%","99%","99.9%","99.99%","99.999%","100%","Total Request Count","Total Failure Count"
                pos = {}
                tail_key = '99%'
                pos[tail_key] = None
                for i, k in enumerate(fields):
                    k = k.replace('\"', '').strip()
                    if k == tail_key:
                        pos[tail_key] = i

                # TODO: warmup is hard-coded here
                beg = 53
                for line in lines[beg:]:
                    data = line.strip().split(',')
                    single_exp_tail_latencies.append(int(data[pos[tail_key]]))

            tail_latencies[key] = single_exp_tail_latencies

    return tail_latencies[4000][:41]

def read_sinan_data(rootdir, hotel=False, name=None):
    if name is None:
        fname = os.path.join(rootdir, 'results.txt')
    else:
        fname = os.path.join(rootdir, name)

    mean_cpu = {}
    max_cpu = {}
    tail_latencies = {}
    violation_rates = {}
    counts = {}

    with open(fname, 'r') as f:
        for line in f.readlines()[1:]:
            if not line.strip().split():
                break
            vals = line.strip().split()
            key = int(vals[0])
            mean_cpu[key] = float(vals[-2])
            max_cpu[key] = float(vals[-1])
            tail_latencies[key] = float(vals[2])
            violation_rates[key] = float(vals[1])
            counts[key] = 1
    
    return mean_cpu, max_cpu, tail_latencies, violation_rates

def figure1(data, fname='results.png', title='', ylabel='', ylim=None, plot_QoS=None):
    
    fontsize = 14


    plt.style.use(['seaborn-whitegrid'])
    matplotlib.rc("legend", frameon=True)

    colors = sns.color_palette(palette='colorblind')

    fig = plt.figure(figsize=(9,6))
    fig, ax1 = plt.subplots()

    if plot_QoS:
        qos_line = plt.axhline(plot_QoS, linestyle='--', color=colors[-1], linewidth=4)

    ax1.plot(data, color=colors[1], linewidth=3)

    ax1.set_xlabel('Timestep', fontsize=fontsize)
    ax1.set_ylabel(ylabel, color=colors[1])
    ax1.set_ylim((0, 1500))
    ax1.set_xticks(list(range(5,41,5)))
    ax1.tick_params(axis='y', colors=colors[1])

    ax2 = ax1.twinx()
    core_allocation = [80 for _ in range(10)]
    core_allocation += [0.5 for _ in range(10)]
    core_allocation += [80 for _ in range(len(data) - len(core_allocation))]
    ax2.plot(core_allocation, color=colors[2], linewidth=3)
    ax2.set_ylabel('Core Allocation', color=colors[2])
    ax2.set_yscale('log')
    ax2.set_ylim((0.1, 100))
    ax2.tick_params(axis='y', colors=colors[2])


    beta = scipy.stats.beta(2.3,3.5)
    x_axis = [x/100. for x in range(100)]
    x_axis = list(range(9)) + [x * 30. + 10 for x in x_axis]
    beta_samples = [0. for _ in range(9)] + [beta.pdf(x/100.) for x in range(100)]
    ax3 = ax1.twinx()
    ax3.set_ylim(-0.1, 2.0)
    ax3.axis('off')
    ax3.plot(x_axis, beta_samples, color=colors[0], linewidth=3)

    if title:
        plt.title(title, fontsize=fontsize)

    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.clf()

def bar_plot_paper_quality(data, append_mean=False, fname='results.png', title='', ylabel='', hotel=False, ylim=None,
    plot_QoS=None, num_methods=3, transfer=False):
    
    fontsize = 20

    plt.style.use(['seaborn-whitegrid'])
    matplotlib.rc("font", size=fontsize)

    labels = list(data[0].keys())
    all_data = []
    for d in data:
        single_method_data = []
        for key in labels:
            single_method_data.append(d[key])
        
        if append_mean:
            single_method_data.append(np.mean(single_method_data))
        
        all_data.append(single_method_data)
    
    if append_mean:
        labels += ['Mean']
    columns = [str(x) for x in labels]
    columns[0] += ' U'
    if not transfer:
        rows = ('RECLAIMER', 'Sinan', 'AutoScale',)
        rows = rows[:num_methods]
    else:
        rows = ('Transfer', 'Random Init')
    
    if transfer:
        colors = sns.color_palette(palette='colorblind')[4:6]
    else:
        colors = sns.color_palette(palette='colorblind')
        colors[:3] = colors[:3][::-1]

    bar_width = 0.15

    # Initialize the vertical-offset for the stacked bar chart.
    index = np.arange(len(labels))
    fig = plt.figure(figsize=(15,6))

    cell_text = []
    for idx, d in enumerate(all_data):
        plt.bar(index+bar_width*(idx-((len(all_data)-1)/2)), d, bar_width, color=colors[idx])
        if np.max(d) >= 100:
            cell_text.append(['%1.0f' % (x) for x in d])
        elif np.max(d) >= 10:
            cell_text.append(['%1.1f' % (x) for x in d])
        else:
            cell_text.append(['%1.2f' % (x) for x in d])
            
    bbox = [0.03, -0.35, 0.94, 0.35]
    if hotel:
        bbox = [0.02, -0.35, 0.96, 0.35]
    tab = plt.table(cellText=cell_text,
            rowLabels=rows,
            rowColours=colors,
            colLabels=columns,
            loc='bottom',
            cellLoc='center',
            bbox=bbox
            )
    tab.auto_set_font_size(False)
    tab.set_fontsize(fontsize)

    for (row, col), cell in tab.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'), fontsize=fontsize)

    fig.subplots_adjust(left=0.2, bottom=0.1)

    if plot_QoS:
        plt.axhline(plot_QoS, linestyle='--', color=colors[-1])

    plt.ylim(ylim)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks([])
    if title:
        plt.title(title, fontsize=fontsize)

    fig = plt.gcf()
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.clf()

if __name__=='__main__':
    qos = 500

    violation_ylim = (0, 0.042)

    all_mean_cpu, all_max_cpu, all_tail, all_violation = [], [], [], []

    rootdir = './results/evaluation_results/cpu/socialNetwork/us/eval_notransformer/sac/gym_dsb-dsb-social-media-v0/locust_results/'
    mean_cpu, max_cpu, tail_latencies, violation_rates = read_data(rootdir, qos)
    all_mean_cpu.append(mean_cpu)
    all_max_cpu.append(max_cpu)
    all_tail.append(tail_latencies)
    all_violation.append(violation_rates)

    rootdir = './results/evaluation_results/cpu/socialNetwork/sinan/'
    mean_cpu, max_cpu, tail_latencies, violation_rates = read_sinan_data(rootdir, hotel=False)
    all_mean_cpu.append(mean_cpu)
    all_max_cpu.append(max_cpu)
    all_tail.append(tail_latencies)
    all_violation.append(violation_rates)

    rootdir = './results/evaluation_results/cpu/socialNetwork/conservative/evaluate/sac/gym_dsb-dsb-social-media-v0/locust_results/'
    mean_cpu, max_cpu, tail_latencies, violation_rates = read_data(rootdir, qos)
    all_mean_cpu.append(mean_cpu)
    all_max_cpu.append(max_cpu)
    all_tail.append(tail_latencies)
    all_violation.append(violation_rates)

    bar_plot_paper_quality(all_mean_cpu, append_mean=True, fname='social-mean_cpu.png',
        ylabel='Mean Allocated CPUs')
    
    bar_plot_paper_quality(all_max_cpu, append_mean=True, fname='social-max_cpu.png',
        ylabel='Maximum Allocated CPUs')

    bar_plot_paper_quality(all_tail, append_mean=True, fname='social-tail_latency.png',
        ylabel='Tail Latency (ms)',
        plot_QoS=qos, ylim=(0, qos*1.05))
    
    bar_plot_paper_quality(all_violation, append_mean=True, fname='social-violation_rate.png',
        ylabel='Violation Rate', ylim=violation_ylim)


    ###################### HOTEL ######################


    qos = 200

    all_mean_cpu, all_max_cpu, all_tail, all_violation = [], [], [], []

    rootdir = './results/evaluation_results/cpu/hotel/us/eval_noexploit_publish2_nohat/sac/gym_dsb-dsb-social-media-v0/locust_results/'
    rootdir = './results/evaluation_results/cpu/hotel/us/eval_retrain/sac/gym_dsb-dsb-social-media-v0/locust_results/'
    mean_cpu, max_cpu, tail_latencies, violation_rates = read_data(rootdir, qos, hotel=True)
    del_keys = []
    for key, val in mean_cpu.items():
        if int(key) > 3500:
            del_keys.append(key)
    for key in del_keys:
        mean_cpu.pop(key, None)
        max_cpu.pop(key, None)
        tail_latencies.pop(key, None)
        violation_rates.pop(key, None)


    all_mean_cpu.append(mean_cpu)
    all_max_cpu.append(max_cpu)
    all_tail.append(tail_latencies)
    all_violation.append(violation_rates)

    rootdir = './results/evaluation_results/cpu/hotel/sinan/'
    mean_cpu, max_cpu, tail_latencies, violation_rates = read_sinan_data(rootdir, hotel=True, name='results8.txt')
    all_mean_cpu.append(mean_cpu)
    all_max_cpu.append(max_cpu)
    all_tail.append(tail_latencies)
    all_violation.append(violation_rates)

    rootdir = './results/evaluation_results/cpu/hotel/conservative/eval_noexploit_publish/sac/gym_dsb-dsb-social-media-v0/locust_results/'
    mean_cpu, max_cpu, tail_latencies, violation_rates = read_data(rootdir, qos, hotel=True)
    all_mean_cpu.append(mean_cpu)
    all_max_cpu.append(max_cpu)
    all_tail.append(tail_latencies)
    all_violation.append(violation_rates)

    bar_plot_paper_quality(all_mean_cpu, append_mean=True, fname='hotel-mean_cpu.png',
        ylabel='Mean Allocated CPUs', hotel=True)
    
    bar_plot_paper_quality(all_max_cpu, append_mean=True, fname='hotel-max_cpu.png',
        ylabel='Maximum Allocated CPUs', hotel=True)

    bar_plot_paper_quality(all_tail, append_mean=True, fname='hotel-tail_latency.png',
        ylabel='Tail Latency (ms)',  hotel=True, plot_QoS=qos, ylim=(0, qos*1.05))
    
    bar_plot_paper_quality(all_violation, append_mean=True, fname='hotel-violation_rate.png',
        ylabel='Violation Rate', hotel=True, ylim=violation_ylim)

    ###################### Figure 1 ######################

    rootdir = './results/evaluation_results/figure1/sac/gym_dsb-dsb-social-media-v0/locust_results'
    tail_latencies= read_data_detailed(rootdir)
    figure1(tail_latencies, fname="figure1.png", ylabel='Tail Latency (ms)', plot_QoS=200)

    ###################### Transfer Hotel -> Social ######################


    qos = 500

    all_mean_cpu, all_max_cpu, all_tail, all_violation = [], [], [], []

    rootdir = './results/evaluation_results/transfer/h_to_s_publish2/sac/gym_dsb-dsb-social-media-v0/locust_results'
    mean_cpu, max_cpu, tail_latencies, violation_rates = read_data(rootdir, qos, hotel=False)
    all_mean_cpu.append(mean_cpu)
    all_max_cpu.append(max_cpu)
    all_tail.append(tail_latencies)
    all_violation.append(violation_rates)

    rootdir = './results/evaluation_results/transfer/baseline2/sac/gym_dsb-dsb-social-media-v0/locust_results'
    mean_cpu, max_cpu, tail_latencies, violation_rates = read_data(rootdir, qos, hotel=False)
    all_mean_cpu.append(mean_cpu)
    all_max_cpu.append(max_cpu)
    all_tail.append(tail_latencies)
    all_violation.append(violation_rates)

    bar_plot_paper_quality(all_mean_cpu, append_mean=True, fname='transfer-mean_cpu.png',
        ylabel='Mean Allocated CPUs', hotel=True, num_methods=1, transfer=True)
    
    bar_plot_paper_quality(all_max_cpu, append_mean=True, fname='transfer-max_cpu.png',
        ylabel='Maximum Allocated CPUs', hotel=True, num_methods=1, transfer=True)

    bar_plot_paper_quality(all_tail, append_mean=True, fname='transfer-tail_latency.png',
        ylabel='Tail Latency (ms)',  hotel=True, plot_QoS=qos, ylim=(0, qos*1.05), num_methods=1, transfer=True)
    
    bar_plot_paper_quality(all_violation, append_mean=True, fname='transfer-violation_rate.png',
        ylabel='Violation Rate', hotel=True, ylim=violation_ylim, num_methods=1, transfer=True)