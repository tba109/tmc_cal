import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import sys

f = open('tmeas_2016-09-20_13_15_40_177697.txt','r')
time_meas = []
sensor_meas = []

for line in f:
    if len(line.split()) == 14:
        if line.split()[1] == 'TSIG0':
            strtime = line.split()[0]
            tstamp = time.mktime(datetime.datetime.strptime(strtime,'%Y-%m-%d_%H:%M:%S:%f').timetuple())
            sensor_meas_single = float(line.split()[2])
            time_meas.append(tstamp)
            sensor_meas.append(float(line.split()[2]))

# These cut the data on the window
time_meas_2 = []
sensor_meas_2 = []

for i in range(len(sensor_meas)):
    if i-4 >= 0 and i+4 <= len(sensor_meas)-1:
        if sensor_meas[i-4] < 1.3E7 and sensor_meas[i-4] > 9.45866E6 and sensor_meas[i+4] < 1.3E7 and sensor_meas[i+4] > 9.45866E6:
            time_meas_2.append(time_meas[i])
            sensor_meas_2.append(sensor_meas[i])

for x,y in zip(time_meas_2,sensor_meas_2):
    print x,y

plt.plot(time_meas,sensor_meas,marker='.',color='r')
plt.plot(time_meas_2,sensor_meas_2,marker='.')
plt.show()

f.close()
