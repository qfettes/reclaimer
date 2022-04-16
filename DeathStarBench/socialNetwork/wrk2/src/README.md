# Multiple Versions of wrk.c files 

We have added two different versions of wrk.c files named 
1. wrk_Mar5_10SecondsTraces_Using-TimeStamps
2. wrk_Mar8_10SecondsTraces_Using-limit

Initially, before developing this multiple versions, we tried to use CURL command via shell script to retreive InProgress traces when workload Generator is live and running.
But there was significant delay of 3 or 4 seconds in retrieval of traces,i.e, In 10 second interval, the traces which get generated in final 3 or 4 seconds were not returned when called via shell script. 

So, we expected calling CURL command directly from wrk.c file by integrating the LIBCURL library in the C program may help in faster retrieval of traces.

The first modified version 'wrk_Mar5_10SecondsTraces_Using-TimeStamps' captures traces every 10 seconds using CURL command with the help of timestamps in the 10 second interval. 
Here the timestamps used will be that of the first and last trace which get generated in the 10 second interval. So, if we run workload generator for 60 seconds, we do get 6 trace files in json format which contain the traces that occured in each intervals of 10 seconds.

The second modified version 'wrk_Mar8_10SecondsTraces_Using-limit' captures traces every 10 seconds using CURL command with the help of limit parameter in the 10 second interval. The traces which get generated in the 10 second interval will be counted and used in the CURL command with limit = count as parameter.  So, if we run workload generator for 60 seconds, we do get 6 trace files in json format which contain the traces that occured in each intervals of 10 seconds.
 
 
### Observations - 

Both the modified versions of wrk.c files above did not work as expected and do have latency issues where the traces generated in the final seconds do not get reflected in the trace files.

1. In the first modified version, the CURL request sent after every 10th second when called after the workload generator has completed running, does give all the traces in the json file in that specific interval. But the same CURL request if called when the workload generator is live and running after every 10th second, does not give all the traces which get generated in that specific interval.

2. Similarly, we expected different approach using limit paramater may help in better retrieval of traces. But even CURL command using limit paramater during live workload Generator did not help in better retrieval of traces.


In order to test using these modified files, we have to install LIBCURL library using the command below - 

```
sudo apt install libcurl4-openssl-dev
```

Then navigate back to the wrk2 folder, and update the makeFile with the -lcurl flag in the LIBS. The changes in makeFile will come in effect using the make command.

```
make
```


The trace files in json format will be generated in wrk2 folder.








