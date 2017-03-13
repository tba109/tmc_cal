#######################################################################################################
# Tyler Anderson Fri Mar 10 10:28:14 EST 2017
# Find the tempco of the data using interpolating functions
#######################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from scipy import interpolate

# Read in the temperature data
f = open('../../tmc_cal_data/hp34401a_2017-03-07_08_35_53_600208.txt','r')
temp_time_meas = []
temp_v_meas = []
for line in f:
    strtime = line.split()[0]
    tstamp = time.mktime(datetime.datetime.strptime(strtime,'%Y-%m-%d_%H:%M:%S:%f').timetuple())
    temp_time_meas.append(tstamp)
    temp_v_meas.append(float(line.split()[1]))

for x,y in zip(temp_time_meas,temp_v_meas):
    print x,y


temp_degc_meas = [x*100. - 273.15 for x in temp_v_meas]
temp_hours = [(ts - np.min(temp_time_meas))/3600. for ts in temp_time_meas]
plt.plot(temp_hours,temp_degc_meas)
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (degrees C)')
plt.show()

f.close()

# Read in the VREF data 
f = open('../../tmc_cal_data/keysight_34470a_March7_2017.csv','r')
vref_time_meas = []
vref_v_meas = []
for line in f:
    strtime = line.split(',')[0]
    tstamp = time.mktime(datetime.datetime.strptime(strtime,'%Y-%m-%d %H:%M:%S.%f').timetuple())
    vref_time_meas.append(tstamp)
    vref_v_meas.append(float(line.split(',')[1]))

for x,y in zip(vref_time_meas,vref_v_meas):
    print x,y

vref_hours = [(ts - np.min(vref_time_meas))/3600. for ts in vref_time_meas]
plt.plot(vref_hours,vref_v_meas)
plt.xlabel('Time (hours)')
plt.ylabel('Reference Voltage (V)')
plt.show()

f.close()

# What are the ranges for both datasets? If same, we can just interpolate from that. 
print '%d %d' % (np.min(temp_time_meas),np.max(temp_time_meas))
print '%d %d' % (np.min(vref_time_meas),np.max(vref_time_meas))

# Looks good, so now just interpolate
f_temp_interp = interpolate.interp1d(temp_time_meas,temp_degc_meas)
temp_degc_meas_b = f_temp_interp(vref_time_meas)

# Finally, plot the result
plt.plot(temp_degc_meas_b,vref_v_meas)
plt.xlabel('Temperature (degrees C)')
plt.ylabel('Reference Voltage (V)')
plt.show()
