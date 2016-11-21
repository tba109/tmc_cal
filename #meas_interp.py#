import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import sys
from scipy import interpolate

def meas_interp(meas_time,meas,interp_time):
    f_interp = interpolate.interp1d(meas_time,meas)
    return f_interp(interp_time)

