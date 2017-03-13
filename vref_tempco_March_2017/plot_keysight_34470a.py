import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import sys

f = open('../../tmc_cal_data/keysight_34470a_March7_2017.csv','r')
time_meas = []
v_meas = []
for line in f:
    strtime = line.split(',')[0]
    tstamp = time.mktime(datetime.datetime.strptime(strtime,'%Y-%m-%d %H:%M:%S.%f').timetuple())
    time_meas.append(tstamp)
    v_meas.append(float(line.split(',')[1]))

for x,y in zip(time_meas,v_meas):
    print x,y

plt.plot(time_meas,v_meas)
plt.show()

f.close()

