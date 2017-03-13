import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import sys
sys.path.append('../')
import tmc_parse_data

# 100kOhm || 1nF
# filename = '../../tmc_cal_data/tmeas_2017-03-13_14_52_38_967326.txt'

# 100kOhm || 1nF 
# filename = '../../tmc_cal_data/tmeas_2017-03-13_15_06_09_006481.txt'

# 10MegOhm || 1nF
# filename = '../../tmc_cal_data/tmeas_2017-03-13_15_45_12_007292.txt'

# No connection between signal ground and earth 
# filename = '../../tmc_cal_data/tmeas_2017-03-13_16_58_33_448961.txt'

# 100kOhm || 0.1uF
filename = '../../tmc_cal_data/tmeas_2017-03-13_17_23_32_258037.txt'

sensor_time,sensor_meas = tmc_parse_data.tmc_parse_data(filename,'TSIG1','ADC0')
print np.std(sensor_meas)
plt.plot(sensor_meas)
plt.show()
