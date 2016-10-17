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

# Return datasets cut so that x1 and x2 correspond to x1 values within +/-n*sigma of the mean of x1
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

# Try to correlate x with y and find coefficients
def tmc_corr(x_time,x_meas,y_time,y_meas):
    
    # Note: I am having a problem because we are slightly out of bounds with x_time. Fix by truncating the first
    # and last measurement
    x_time_b = x_time[1:-1]
    x_meas_b = x_meas[1:-1]
    y_interp = meas_interp.meas_interp(y_time,y_meas,x_time_b)
    # remove the outliers
    x_time_b_2, x_meas_b_2, y_meas_2 = cut_outliers(x_time_b,x_meas_b,y_interp)

    # Find the tempco
    plt.plot(y_meas_2,x_meas_b_2,'.')
    A,B = curve_fit(f_line,y_meas_2,x_meas_b_2)[0]
    
    fit_line = [A*x + B for x in y_meas_2]
    plt.plot(y_meas_2,fit_line,color='r')
    plt.show()

    # Perform the correction
    x_corr = lin_corr_ac(y_interp,x_meas_b,A)
    plt.plot(x_time,x_meas,'.',color='b')
    plt.plot(x_time_b,x_corr,'.',color='g')
    plt.show()
    return A,B,x_time_b,x_corr


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

    chan = '5'
    adc = '11'
    
    # October 10, 2016 (the shorted board 3 run (ADC 9, 10, and 11)
    # tmc_file = '../tmc_cal_data/tmeas_2016-10-07_18_54_15_041531.txt'
    
    # Long wire twisted pairs on 9-3, 10-3, 11-3
    # tmc_file = '../tmc_cal_data/tmeas_2016-10-11_17_20_59_884324.txt'
    
    # Long wire twisted pair with mid-line bananas on 9-3. 10-3 and 11-3 are long twisted pair, as before
    tmc_file = '../tmc_cal_data/tmeas_2016-10-15_11_49_41_941548.txt'

    # Do the reference drift correction
    sensor_time,sensor_meas = tmc_parse_data.tmc_parse_data(tmc_file,'TSIG'+chan,'ADC'+adc)
    tsig_volts = [((x/8388608.)-1.)*0.625 for x in sensor_meas]
    
    #################################################
    # TSIG
    tsig_dates = [datetime.datetime.fromtimestamp(ts) for ts in sensor_time]
    
    # CURR
    curr_time,curr_meas = tmc_parse_data.tmc_parse_data(tmc_file,'CURR'+chan,'ADC'+adc)
    curr_ma = [(x/8388608.-1.)*62.5 for x in curr_meas]
    curr_time_2, curr_ma_2 = tmc_parse_data.tmc_find_valid_meas(curr_time,curr_ma,2,12.5)
    curr_dates = [datetime.datetime.fromtimestamp(ts) for ts in curr_time_2]

    # ATEMP
    atemp_time,atemp_meas = tmc_parse_data.tmc_parse_data(tmc_file,'ATEMP','ADC'+adc)
    atemp_dates = [datetime.datetime.fromtimestamp(ts) for ts in atemp_time]
    atemp_degc = [(x-8388608.)/13584. - 272.5 for x in atemp_meas]

    # BTEMP
    btemp_time,btemp_meas = tmc_parse_data.tmc_parse_data(tmc_file,'BTEMP','ADC'+'0') # BTEMP doesn't come up every time
    btemp_dates = [datetime.datetime.fromtimestamp(ts) for ts in btemp_time]
    btemp_degc = [625*x/8388608. - 625 -273. for x in btemp_meas]

    # BSLN
    bsln_time,bsln_meas = tmc_parse_data.tmc_parse_data(tmc_file,'BSLN','ADC'+adc)
    bsln_dates = [datetime.datetime.fromtimestamp(ts) for ts in bsln_time]
    bsln_volts = [(x/8388608.-1)*0.625*2 for x in bsln_meas]

    # Plot up the DMM corrected data
    plot_all(tsig_dates,tsig_volts,
             curr_dates,curr_ma_2,
             atemp_dates,atemp_degc,
             btemp_dates,btemp_degc,
             bsln_dates,bsln_volts) 
    
    ################################################
    # Plot the sum of BSLN and TSIG
    bsln_interp = meas_interp.meas_interp(bsln_time,bsln_volts,sensor_time)
    tsig_pure = bsln_interp + tsig_volts # remove baseline
    plt.plot(sensor_time,tsig_pure,'.')
    plt.show()

    print 'ADC %d, CHAN %d: standard deviation is %f uV' % (int(adc),int(chan),np.std(tsig_pure*1.E6))

    plt.plot(bsln_interp,tsig_volts,'.')
    plt.show()

if __name__ == "__main__":
    main()
    
