import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import sys
from scipy import interpolate

def dmm_interp(filename,interp_time):
    ref_time = []
    ref_meas = []

    try:
        f = open(filename,'r')
        
    except IOError:
        print "Cold not read file: ", filename
        sys.exit()        

    for line in f:
        strtime = line.split()[0]
        tstamp = time.mktime(datetime.datetime.strptime(strtime,'%Y-%m-%d_%H:%M:%S:%f').timetuple())
        ref_time.append(tstamp)
        ref_meas.append(float(line.split()[1]))

    
    f_interp = interpolate.interp1d(ref_time,ref_meas)
    ref_interp = f_interp(interp_time)

    return ref_interp


def dmm_read(filename):
    ref_time = []
    ref_meas = []

    try:
        f = open(filename,'r')
        
    except IOError:
        print "Cold not read file: ", filename
        sys.exit()        

    for line in f:
        strtime = line.split()[0]
        tstamp = time.mktime(datetime.datetime.strptime(strtime,'%Y-%m-%d_%H:%M:%S:%f').timetuple())
        ref_time.append(tstamp)
        ref_meas.append(float(line.split()[1]))

    
    return ref_meas

