# Testing 1, 2, 3
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import time
import datetime
import sys
from scipy.optimize import curve_fit
import meas_interp

def f_line(x,A,B):
    return A*x+B


# Return datasets cut so that x1 and x2 are withing +/-n*sigma of the mean of x1
def cut_outliers(t,x1,x2,n=3.):
    t_ret = []
    x1_ret = []
    x2_ret = []
    x1_mean = np.mean(x1)
    x1_std = np.std(x1)
    for ti,x1i,x2i in zip(t,x1,x2):
        if x1i < (x1_mean + n*x1_std) and x1i > (x1_mean - n*x1_std):
            t_ret.append(ti)
            x1_ret.append(x1i)
            x2_ret.append(x2i)
    return t_ret,x1_ret,x2_ret

# Linear correction to data in terms of a (slope) and b (intercept)
def lin_corr(x,y,a,b):
    y_ret = []
    for xi,yi in zip(x,y):
        y_ret.append(yi-(a*xi + b))
    return y_ret
        
# Same as lin_corr, but only correct for variation around mean
def lin_corr_ac(x,y,a,b):
    y_ret = []
    x_mean = np.mean(x)
    for xi,yi in zip(x,y):
        y_ret.append(yi-(a*(xi-x_mean)))
    return y_ret

def tmc_tsig_dmm_corr(tsig_time,tsig_meas,dmm_meas):
    plt.plot(dmm_meas,tsig_meas)
    A,B = curve_fit(f_line,dmm_meas,tsig_meas)[0]
    # print "First Pass Correction parameters: A=%f, B=%f" % (A,B)
    plt.plot(dmm_meas,f_line(dmm_meas,A,B),color='r')
    plt.show()

    # Do the first pass correction to remove the coarse effects of reference drift
    sensor_meas_mean_sub = []
    for x,r in zip(tsig_meas,dmm_meas):
        y = x - (A*r+B)
        # print y
        sensor_meas_mean_sub.append(y)
    
    # Redo the sensor vs. dmm analysis, this time removing the outliers/jumps
    sensor_meas_corr_1_stdev = np.std(sensor_meas_mean_sub)
    sensor_meas_corr_1 = []
    dmm_meas_corr_1 = []
    tsig_meas_corr_1 = []
    for x,r,x2 in zip(sensor_meas_mean_sub,dmm_meas,tsig_meas):
        if x < 3.*sensor_meas_corr_1_stdev and x > -3.*sensor_meas_corr_1_stdev:
            sensor_meas_corr_1.append(x)
            dmm_meas_corr_1.append(r)
            tsig_meas_corr_1.append(x2)

    plt.plot(dmm_meas,sensor_meas_mean_sub,color='b',marker='.')
    plt.plot(dmm_meas_corr_1,sensor_meas_corr_1,color='g',marker='.')
    plt.axhline(y=3.*sensor_meas_corr_1_stdev,color='r')
    plt.axhline(y=-3.*sensor_meas_corr_1_stdev,color='r')
    plt.show()

    # Corrected data
    plt.plot(dmm_meas,tsig_meas,color='b',marker='.')
    plt.plot(dmm_meas_corr_1,tsig_meas_corr_1,color='g',marker='.')
    A2,B2 = curve_fit(f_line,dmm_meas_corr_1,tsig_meas_corr_1)[0]
    plt.plot(dmm_meas_corr_1,f_line(dmm_meas_corr_1,A2,B2),color='r')
    plt.show()

    # But we actually want Volts in terms of ADC counts, so....
    plt.plot(tsig_meas_corr_1,dmm_meas_corr_1,color='g',marker='.')
    B3 = 0 # I got a somewhat unexpected slope when I let the intercept be free
    A3, = curve_fit(lambda x, a : f_line(x,a,B3),tsig_meas_corr_1,dmm_meas_corr_1)[0]
    dumb_line = []
    for x in tsig_meas_corr_1:
        dumb_line.append(x*A3+B3)
    plt.plot(tsig_meas_corr_1,dumb_line,color='r')
    plt.show()
    print "ADC channels to voltage calibration: Slope = %g (Volts/count), Intercept = %g V" %(A3,B3)

    # Now, do the final, full correction
    sensor_volts = [x*A3 for x in tsig_meas] # convert the ADC counts to Volts
    dmm_mean = np.mean(dmm_meas)
    dmm_mean_sub = [x-dmm_mean for x in dmm_meas] # Just get the AC variations
    sensor_volts_dmm_corr = [x-y for x,y in zip(sensor_volts,dmm_mean_sub)] # subtract them off
    return sensor_volts_dmm_corr

# Try to correlate residual ADC motion with board temperature
def tmc_tsig_btemp_corr(tsig_time,tsig_meas,btemp_time,btemp_meas):
    
    # Note: I am having a problem because we are slightly out of bounds with sensor time. Fix by truncating the first
    # and last measurement
    sensor_time = tsig_time[1:-1]
    sensor_volts = tsig_meas[1:-1]
    btemp_interp = meas_interp.meas_interp(btemp_time,btemp_meas,sensor_time)
    # remove the outliers
    sensor_time_2, sensor_volts_2, btemp_2 = cut_outliers(sensor_time,sensor_volts,btemp_interp)

    # Find the tempco
    plt.plot(btemp_2,sensor_volts_2,'.')
    A,B = curve_fit(f_line,btemp_2,sensor_volts_2)[0]
    print "Temperature Correction parameters: A=%g, B=%f" % (A,B)
    fit_line = [A*x + B for x in btemp_2]
    plt.plot(btemp_2,fit_line,color='r')
    plt.show()

    # Perform the temperature correction
    tsig_btemp_corr = lin_corr_ac(btemp_interp,sensor_volts,A,B)
    plt.plot(tsig_time,tsig_meas,'.',color='b')
    plt.plot(sensor_time,tsig_btemp_corr,'.',color='g')
    plt.show()
    return sensor_time_2, sensor_volts_2, btemp_2


import tmc_parse_data
import dmm_interp
def main():
    # Get the sensor data
    sensor_time,sensor_meas = tmc_parse_data.tmc_parse_data('../tmc_cal_data/tmeas_2016-09-20_13_15_40_177697.txt','TSIG0','ADC0')
    plt.plot(sensor_time,sensor_meas)
    plt.show()

    # Cut out the sections where the reference was not connected
    sensor_time_2,sensor_meas_2 = tmc_parse_data.tmc_find_valid_meas(sensor_time,sensor_meas)
    plt.plot(sensor_time,sensor_meas)
    plt.plot(sensor_time_2,sensor_meas_2,color='r',marker='.')
    plt.show()

    # Get the DMM data
    dmm_meas_2 = dmm_interp.dmm_interp('../tmc_cal_data/hp34401a_2016-09-20_13_15_05_748130.txt',sensor_time_2)
    plt.plot(sensor_time_2,dmm_meas_2)
    plt.show()
    
    # Do the reference drift correction
    sensor_volts_dmm_corr = tmc_tsig_dmm_corr(sensor_time_2,sensor_meas_2,dmm_meas_2)
    
    # TSIG
    tsig_dates = [datetime.datetime.fromtimestamp(ts) for ts in sensor_time_2]
    
    # CURR
    curr_time,curr_meas = tmc_parse_data.tmc_parse_data('tmeas_2016-09-20_13_15_40_177697.txt','CURR0','ADC0')
    curr_dates = [datetime.datetime.fromtimestamp(ts) for ts in curr_time]
    curr_ma = [(x/8388608.-1.)*62.5 for x in curr_meas]

    # ATEMP
    atemp_time,atemp_meas = tmc_parse_data.tmc_parse_data('tmeas_2016-09-20_13_15_40_177697.txt','ATEMP','ADC0')
    atemp_dates = [datetime.datetime.fromtimestamp(ts) for ts in atemp_time]
    atemp_degc = [(x-8388608.)/13584. - 272.5 for x in atemp_meas]

    # BTEMP
    btemp_time,btemp_meas = tmc_parse_data.tmc_parse_data('tmeas_2016-09-20_13_15_40_177697.txt','BTEMP','ADC0')
    btemp_dates = [datetime.datetime.fromtimestamp(ts) for ts in btemp_time]
    btemp_degc = [625*x/8388608. - 625 -273. for x in btemp_meas]

    # BSLN
    bsln_time,bsln_meas = tmc_parse_data.tmc_parse_data('tmeas_2016-09-20_13_15_40_177697.txt','BSLN','ADC0')
    bsln_dates = [datetime.datetime.fromtimestamp(ts) for ts in bsln_time]
    # bsln_volts = [(x/8388608.-1)*0.625 + 0.625 for x in bsln_meas]
    # bsln_volts = [(x/8388608.-1)*1.25 + 1.25 for x in bsln_meas]
    bsln_volts = [(x/8388608.-1)*1.25 for x in bsln_meas]

    # Make a pretty plot
    ax1 = plt.subplot(515)
    plt.ylabel('TSIG (V)')
    plt.setp(ax1.get_xticklabels(), fontsize=8)
    plt.xticks(rotation=25)
    plt.plot(tsig_dates,sensor_volts_dmm_corr)
    
    ax2 = plt.subplot(514, sharex=ax1)
    plt.ylabel('Current (uA)')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.plot(curr_dates,curr_ma)
    
    ax3 = plt.subplot(513, sharex=ax1)
    plt.ylabel('ATemp (degC)')
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.plot(atemp_dates,atemp_degc)
    
    ax4 = plt.subplot(512, sharex=ax1)
    plt.ylabel('BTemp (degC)')
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.plot(btemp_dates,btemp_degc)
    
    ax5 = plt.subplot(511, sharex=ax1)
    plt.ylabel('Baseline (V)')
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.plot(bsln_dates,bsln_volts)

    # # mng = plt.get_current_fig_manager()
    # # mng.full_screen_toggle()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 12)
    # # when saving, specify the DPI
    # if args.save_dir is not None:
    #     figure.savefig("./"+args.save_dir+"/"+sch+".png", dpi = 100)
    #     print "./"+args.save_dir+sch+".png"
    # else:
    #     figure.savefig(sch+".png", dpi = 100)
    plt.show()

    sensor_time_3, sensor_volts_3, btemp_3 = tmc_tsig_btemp_corr(sensor_time_2,sensor_volts_dmm_corr,btemp_time,btemp_degc)

if __name__ == "__main__":
    main()
    
