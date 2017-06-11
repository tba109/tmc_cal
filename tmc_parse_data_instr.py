import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import sys

def tmc_parse_data(filename,row_str,col_str):
    
    col_str_to_idx = {'ADC0' : 2, 
                      'ADC1' : 3, 
                      'ADC2' : 4, 
                      'ADC3' : 5, 
                      'ADC4' : 6, 
                      'ADC5' : 7, 
                      'ADC6' : 8, 
                      'ADC7' : 9, 
                      'ADC8' : 10, 
                      'ADC9' : 11, 
                      'ADC10': 12, 
                      'ADC11': 13} 

    sensor_time = []
    sensor_meas = []

    try:
        f = open(filename,'r')
    
    except IOError:
        print "Cold not read file: ", filename
        sys.exit()
        

    for line in f:
        if len(line.split()) == 14: # This is a line with data in it"
            if line.split()[1] == row_str:
                strtime = line.split()[0]
                tstamp = time.mktime(datetime.datetime.strptime(strtime,'%Y/%m/%d-%H:%M:%S').timetuple())
                sensor_meas_single = float(line.split()[col_str_to_idx[col_str]])
                sensor_time.append(tstamp)
                # sensor_meas.append(float(line.split()[2]))
                sensor_meas.append(sensor_meas_single)
                
    f.close()

    return sensor_time,sensor_meas


def tmc_find_valid_meas(sensor_time,sensor_meas,lim=6.E6,mid=9.45866E6):
    # Find time where we are within +/- 100 counts of the mean
    # sensor_mean = np.mean(sensor_meas)
    # upper_lim = sensor_mean + lim
    # lower_lim = sensor_mean - lim
    upper_lim = mid + lim
    lower_lim = mid - lim
    sensor_time_2 = []
    sensor_meas_2 = []
    for i in range(len(sensor_meas)):
        if i-4 >= 0 and i+4 <= len(sensor_meas)-1:
            if sensor_meas[i-4] < upper_lim and sensor_meas[i-4] > lower_lim:
                if sensor_meas[i+4] < upper_lim and sensor_meas[i+4] > lower_lim:
                    sensor_time_2.append(sensor_time[i])
                    sensor_meas_2.append(sensor_meas[i])
                
    return sensor_time_2,sensor_meas_2
