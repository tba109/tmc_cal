import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import sys

f = open('../../tmc_cal_data/keysight_34470a_Mar24_2017.csv','r')
time_meas = []
v_meas = []
for line in f:
    strtime = line.split(',')[0]
    tstamp = time.mktime(datetime.datetime.strptime(strtime,'%Y-%m-%d %H:%M:%S.%f').timetuple())
    time_meas.append(tstamp)
    v_meas.append(float(line.split(',')[1]))

for x,y in zip(time_meas,v_meas):
    print x,y

time_meas_start = time_meas[0]
time_meas_hr = [(x - time_meas_start)/3600. for x in time_meas]
v_meas = [x*1.E6 for x in v_meas]

plt.plot(time_meas_hr,v_meas)
plt.xlabel('Time (hr)')
plt.ylabel('Reference ($\mu$V)')
plt.show()

plt.plot(time_meas_hr,v_meas)
plt.xlabel('Time (hr)')
plt.ylabel('Reference ($\mu$V)')
# Find a set to find the average value that isn't skewed by outliers
v_meas_mean = np.mean(v_meas)
v_meas_std = np.std(v_meas)
print v_meas_mean, v_meas_std
v_meas_no = [x for x in v_meas if (x > (v_meas_mean - v_meas_std) and x < (v_meas_mean + v_meas_std))] # "_no" stands for "no outliers"
print len(v_meas), len(v_meas_no)
v_meas_mean_no = np.mean(v_meas_no)
print v_meas_mean, v_meas_mean_no 
plt.ylim(v_meas_mean_no-2.,v_meas_mean_no+2.)
plt.show()

f.close()

