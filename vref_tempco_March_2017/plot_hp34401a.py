import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

# f = open('hp34401a_2016-09-20_13_15_05_748130.txt','r')
f = open('../../tmc_cal_data/hp34401a_2017-03-07_08_35_53_600208.txt','r')
time_meas = []
v_meas = []
for line in f:
    strtime = line.split()[0]
    tstamp = time.mktime(datetime.datetime.strptime(strtime,'%Y-%m-%d_%H:%M:%S:%f').timetuple())
    time_meas.append(tstamp)
    v_meas.append(float(line.split()[1]))

for x,y in zip(time_meas,v_meas):
    print x,y

plt.plot(time_meas,v_meas)
plt.show()

f.close()

