import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import dmm_interp
import numpy as np

dmm_file = '../../tmc_cal_data/hp34401a_2016-12-13_09_13_13_564462.txt'
# dmm_file = '../../tmc_cal_data/hp34401a_2016-12-13_09_40_08_773484.txt'
# dmm_file = '../../tmc_cal_data/hp34401a_2016-12-13_10_23_05_042283.txt'
dmm_meas = dmm_interp.dmm_read(dmm_file)
print 'Mean = %f, Stderr = %f' % (np.mean(dmm_meas),np.std(dmm_meas)*1.E6/(np.sqrt(len(dmm_meas))))
plt.ylabel("DMM Data (Volts)")
plt.plot(dmm_meas)
plt.show()
