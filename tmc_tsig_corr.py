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
def lin_corr_ac(x,y,a=1.):
    y_ret = []
    x_mean = np.mean(x)
    for xi,yi in zip(x,y):
        y_ret.append(yi-(a*(xi-x_mean)))
    return y_ret

# Interpolate data at a value

# Correct/remove AC variations from a channel
def tmc_tsig_ac_corr(tsig_time,tsig_meas,ac_meas):
    # Convert to volts
    plt.plot(tsig_time, tsig_meas)
    plt.plot(tsig_time,ac_meas,color='r')
    plt.show()

    # Do a straight ac subtraction
    ac_mean = np.mean(ac_meas)
    ac_mean_sub = [x - ac_mean for x in ac_meas]
    tsig_meas_volts_ac_corr = [x-y for x,y in zip(tsig_meas,ac_mean_sub)]
    plt.plot(tsig_time,tsig_meas_volts_ac_corr)
    plt.show()

    return tsig_meas_volts_ac_corr

# def tmc_tsig_cal(tsig_time,tsig_meas,tsig_dmm_meas):

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


def plot_all(tsig_dates,tsig_volts,
             curr_dates,curr_ma,
             atemp_dates,atemp_degc,
             btemp_dates,btemp_degc,
             bsln_dates,bsln_volts):

    # Make a pretty plot
    ax1 = plt.subplot(515)
    plt.ylabel('TSIG (V)')
    plt.setp(ax1.get_xticklabels(), fontsize=8)
    plt.xticks(rotation=25)
    plt.plot(tsig_dates,tsig_volts)
    
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


import tmc_parse_data
import dmm_interp
def main():
        
    # Do the reference drift correction
    sensor_time,sensor_meas = tmc_parse_data.tmc_parse_data('../tmc_cal_data/tmeas_2016-09-20_13_15_40_177697.txt','TSIG0','ADC0')
    # Cut out the sections where the reference was not connected
    sensor_time_2,sensor_meas_2 = tmc_parse_data.tmc_find_valid_meas(sensor_time,sensor_meas)
    plt.ylabel("TSIG data with Cuts")
    plt.plot(sensor_time,sensor_meas)
    plt.plot(sensor_time_2,sensor_meas_2,color='r',marker='.')
    plt.show()

    # Get the DMM data
    dmm_meas_2 = dmm_interp.dmm_interp('../tmc_cal_data/hp34401a_2016-09-20_13_15_05_748130.txt',sensor_time_2)
    plt.ylabel("DMM Data (Volts)")
    plt.plot(sensor_time_2,dmm_meas_2)
    plt.show()

    tsig_volts = [((x/8388608.)-1.)*0.625 + 0.625 for x in sensor_meas_2]
    sensor_volts_dmm_corr = lin_corr_ac(dmm_meas_2,tsig_volts)

    # TSIG
    tsig_dates = [datetime.datetime.fromtimestamp(ts) for ts in sensor_time_2]
    
    # CURR
    curr_time,curr_meas = tmc_parse_data.tmc_parse_data('../tmc_cal_data/tmeas_2016-09-20_13_15_40_177697.txt','CURR0','ADC0')
    curr_ma = [(x/8388608.-1.)*62.5 for x in curr_meas]
    curr_time_2, curr_ma_2 = tmc_parse_data.tmc_find_valid_meas(curr_time,curr_ma,2,12.5)
    curr_dates = [datetime.datetime.fromtimestamp(ts) for ts in curr_time_2]

    # ATEMP
    atemp_time,atemp_meas = tmc_parse_data.tmc_parse_data('../tmc_cal_data/tmeas_2016-09-20_13_15_40_177697.txt','ATEMP','ADC0')
    atemp_dates = [datetime.datetime.fromtimestamp(ts) for ts in atemp_time]
    atemp_degc = [(x-8388608.)/13584. - 272.5 for x in atemp_meas]

    # BTEMP
    btemp_time,btemp_meas = tmc_parse_data.tmc_parse_data('../tmc_cal_data/tmeas_2016-09-20_13_15_40_177697.txt','BTEMP','ADC0')
    btemp_dates = [datetime.datetime.fromtimestamp(ts) for ts in btemp_time]
    btemp_degc = [625*x/8388608. - 625 -273. for x in btemp_meas]

    # BSLN
    bsln_time,bsln_meas = tmc_parse_data.tmc_parse_data('../tmc_cal_data/tmeas_2016-09-20_13_15_40_177697.txt','BSLN','ADC0')
    bsln_dates = [datetime.datetime.fromtimestamp(ts) for ts in bsln_time]
    bsln_volts = [(x/8388608.-1)*0.625*2 for x in bsln_meas]

    # Plot up the DMM corrected data
    plot_all(tsig_dates,sensor_volts_dmm_corr,
             curr_dates,curr_ma_2,
             atemp_dates,atemp_degc,
             btemp_dates,btemp_degc,
             bsln_dates,bsln_volts) 

    # Next: do calibration, find tempco

if __name__ == "__main__":
    main()
    
