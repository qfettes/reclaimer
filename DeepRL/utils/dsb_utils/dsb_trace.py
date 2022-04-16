from utils.dsb_utils.dsb_const import *
from utils.dsb_utils.dsb_action_exec import update_containers

import datetime, pycurl, json
from io import BytesIO
import threading
import numpy as np
import multiprocessing

from timeit import default_timer as timer

span_interval_start = None
microseconds_per_second = 1000000
spanCounts = {} # Dictionary to capture operations count after end of each trace interval
# trace_counts = []

absolute_start = int(datetime.datetime.now().timestamp() * microseconds_per_second)

statsDict={}

def captureRPS(interval_length_micros):
    global containers
    global changed_container_idxes

    # NOTE: Kind of slow. See if you can come back to this
    #   one day to save 10-30 ms
    # This works because the order of containers has not
    #   changed since changed_container_idxes was populated
    update_containers(changed_container_idxes)
    changed_container_idxes = [] # empty this
    
    occupied_cores = set()

    current_physical_cpu_affinity = {}
    new_physical_cpu_affinity = {}

    # Use this loop to give containers priority in keeping previous cores
    #   this will avoid thrashing due to thread constantly changing cores
    #   they have a 'budget' of cores they can keep, which is defined by
    #   the action and the total number of wildcard cores
    # for container in containers:
    #     if container.name not in ('socialnetwork_jaeger_1', 'resource-sink'):
    #         idx = positional_encoding[container.name]

    #         # make an empty list for the new cpu affinities
    #         #   always append its default core
    #         new_physical_cpu_affinity[container.name] = [positional_encoding[container.name]]

    #         # get the current cpu affinity for the container as a list of strings
    #         received_packets = container.stats(stream=False)['networks']['eth0']['rx_packets']
    #         sent_packets = container.stats(stream=False)['networks']['eth0']['tx_packets']

    #         print("done")
    start = timer()
    for container in containers:
        if container.name in ('socialnetwork_jaeger_1', 'resource-sink'):
            continue
            
        container_pids = []
        for process in container.top()['Processes']:
            container_pids.append(process[1])
        
        for pid in container
        print(container.top()['Processes'])
    print(timer() - start)
    exit()


def get_container_stats(idx):
    start = None
    for stats in containers[idx].stats(stream=True, decode=True):
        statsDict[containers[idx].name] = (stats['networks']['eth0']['rx_packets'], stats['networks']['eth0']['tx_packets'])


        if idx==0:
            if start:
                print(timer() - start)
            start = timer()

# def captureTraces(interval_end_offset_micros, interval_length_micros):
#     global span_interval_start

#     latencyStatistics = [] # Capturing the latency information of the operations (List of Lists) 
#     processList = {} # Dictionary to capture the intermediate processes to service names after each trace interval

#     # datetime.datetime.now().timestamp() yields a time in seconds
#     span_interval_end = int(datetime.datetime.now().timestamp() * microseconds_per_second) - interval_end_offset_micros
#     if not span_interval_start:
#         span_interval_start = span_interval_end - interval_length_micros

#     c = pycurl.Curl() # Using Pycurl library of python to perform curl operations
#     data = BytesIO()

#     # Below url to be used for CURL in capturing of the traces every 1 second 
#     url = f"http://localhost:16686/api/traces?service=nginx-web-server&start={span_interval_start}&end={span_interval_end}&limit=10000"
#     # print(span_interval_end, span_interval_start, span_interval_end - span_interval_start)
#     start = timer()
#     c.setopt(c.URL, url)
#     c.setopt(c.WRITEFUNCTION, data.write)
#     c.perform()
#     traces = json.loads(data.getvalue()) # capturing the traces as json object in dictionary format

#     # form a dictionary
    
#     # Parsing the traces dictionary object to capture the span(operation) counts and latency information
#     max_start = 0
#     min_start = float('inf')
#     for trace in traces['data']: #Iterating over the traces
#         processIDMap = {}

#         for process in trace['processes'].items(): # Iterating over the process list of each trace 
#             if process[0] in processIDMap:
#                 assert(processIDMap[process[0]] == process[1]['serviceName']), f"Process ID {process[0]} has alredy been mapped to a different service name"
#             else:
#                 processIDMap[process[0]] = process[1]['serviceName']

#             # TODO: remove this
#             processList[process[0]] = process[1]['serviceName'] 

#         for span in trace['spans']: 
#             pid = span['processID']
#             print(processIDMap[pid])
#             spanCounts[processList[span['processID']]] = spanCounts.get(processList[span['processID']],0)+1	
#             # print(span)
#             # exit()
#             latencyStatistics.append([i+1,item['traceID'],span['spanID'],span['operationName'],span['startTime'],span['processID'],float(span['duration'])/1000])
#             # print(latencyStatistics[-1])
#             # exit()
#             max_start = max(span['startTime'], max_start)
#             min_start = min(span['startTime'], min_start)
#         exit()

#     trace_counts.append(span_interval_end/microseconds_per_second - max_start/microseconds_per_second)

#     print(f"End - Max: {span_interval_end/microseconds_per_second - max_start/microseconds_per_second}")
#     print(f"Min - Start: {min_start/microseconds_per_second - span_interval_start/microseconds_per_second}")
#     print(f"Mean E-M: {np.mean(trace_counts)}")
#     print('*'*20)

#     # set start as the end of the previous measure
#     span_interval_start = span_interval_end

#     # threading.Timer(interval_length_micros/microseconds_per_second, captureTraces, args=[interval_end_offset_micros, interval_length_micros]).start() # Timer to invoke captureTraces() function every 1 second