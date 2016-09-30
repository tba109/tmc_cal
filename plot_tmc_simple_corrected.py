import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import sys

# Here is the TMC data
f = open('tmeas_2016-09-20_13_15_40_177697.txt','r')
sensor_time = []
sensor_meas = []

for line in f:
    if len(line.split()) == 14:
        if line.split()[1] == 'TSIG0':
            strtime = line.split()[0]
            tstamp = time.mktime(datetime.datetime.strptime(strtime,'%Y-%m-%d_%H:%M:%S:%f').timetuple())
            sensor_meas_single = float(line.split()[2])
            sensor_time.append(tstamp)
            sensor_meas.append(float(line.split()[2]))

# These cut the data on the window
sensor_time_2 = []
sensor_meas_2 = []

for i in range(len(sensor_meas)):
    if i-4 >= 0 and i+4 <= len(sensor_meas)-1:
        if sensor_meas[i-4] < 1.3E7 and sensor_meas[i-4] > 9.45866E6 and sensor_meas[i+4] < 1.3E7 and sensor_meas[i+4] > 9.45866E6:
            sensor_time_2.append(sensor_time[i])
            sensor_meas_2.append(sensor_meas[i])

for x,y in zip(sensor_time_2,sensor_meas_2):
    print x,y

plt.plot(sensor_time,sensor_meas,marker='.',color='r')
plt.plot(sensor_time_2,sensor_meas_2,marker='.')
plt.show()

f.close()

# These are the reference measurements (drifty, ugh!)

f = open('hp34401a_2016-09-20_13_15_05_748130.txt','r')
ref_time = []
ref_meas = []
for line in f:
    strtime = line.split()[0]
    tstamp = time.mktime(datetime.datetime.strptime(strtime,'%Y-%m-%d_%H:%M:%S:%f').timetuple())
    ref_time.append(tstamp)
    ref_meas.append(float(line.split()[1]))

for x,y in zip(ref_time,ref_meas):
    print x,y

plt.plot(ref_time,ref_meas)
plt.show()

f.close()

# This is the interpolated reference measurements
from scipy import interpolate

f_interp = interpolate.interp1d(ref_time,ref_meas)

ref_interp = f_interp(sensor_time_2)

plt.plot(ref_interp,sensor_meas_2,marker='.')

from scipy.optimize import curve_fit
def f_line(x,A,B):
    return A*x+B

A,B = curve_fit(f_line,ref_interp,sensor_meas_2)[0]
plt.plot(ref_interp,f_line(ref_interp,A,B),color='r')
plt.show()

print A,B

# Finally, perform the correction:
sensor_meas_3 = []
for x,r in zip(sensor_meas_2,ref_interp):
    y = x - (A*r+B)
    print y
    sensor_meas_3.append(y)
    
sensor_time_3 = sensor_time_2

# First pass at corrected data
plt.plot(sensor_time_3,sensor_meas_3,marker='.')
plt.show()


# Now get a better correction for the correlation, not biased by the jumps
sensor_meas_4 = []
sensor_time_4 = []
for x,y in zip(sensor_meas_3,sensor_time_3):
    if x < 75 and x > -75:
        sensor_meas_4.append(x)
        sensor_time_4.append(y)

ref_interp_4 = f_interp(sensor_time_4)        
A2,B2 = curve_fit(f_line,ref_interp_4,sensor_meas_4)[0]
plt.plot(ref_interp_4,sensor_meas_4,'.')
plt.plot(ref_interp_4,f_line(ref_interp_4,A2,B2),color='r')

plt.show()

# Plot the corrected data, still include the big jumps
sensor_meas_5 = []

ref_interp_5 = f_interp(sensor_time_3)
for x,r in zip(sensor_meas_3,ref_interp_5):
    sensor_meas_5.append(x - (A2*r + B2))
sensor_time_5 = sensor_time_3

# Test
voltage_5 = []
for x in sensor_meas_5:
    voltage_5.append(1.E6*x*0.625/8388608.)

plt.plot(sensor_time_5,voltage_5,marker='.')
plt.show()
