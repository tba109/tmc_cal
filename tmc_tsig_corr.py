import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import time
import datetime
import sys
from scipy.optimize import curve_fit
import meas_interp

# Take the moving average
def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    result = np.convolve(interval, window, 'same')
    # print 'Result is length = %d' % len(result)
    for i in range(len(result)):
        if i < window_size:
            #print "%d: replacing %f with %f" % (i,result[i],np.sum(interval[:i+1])/float(i+1))
            result[i] = np.sum(interval[:i+1])/float(i+1)
        elif i > (len(result)-window_size-1):
            #print "%d: replacing %f with %f" % (i,result[i],np.sum(interval[i:])/float(len(result)-i))
            result[i] = np.sum(interval[i:])/float(len(result)-i)
            
    
    # print result[0],result[len(result)-1]
    return result


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


def tmc_gain_corr(sensor_time,tsig_volts,zero_time,zero_volts,bsln_time,bsln_volts):
    print len(sensor_time)
    print len(tsig_volts)
    print len(zero_time)
    print len(zero_volts)
    print len(bsln_time)
    print len(bsln_volts)
    # Create interpolating function for zero
    sensor_time_2 = sensor_time[1:-1]
    zero_volts_2 = meas_interp.meas_interp(zero_time,zero_volts,sensor_time_2)
    bsln_volts_2 = meas_interp.meas_interp(bsln_time,bsln_volts,sensor_time_2)
    tsig_volts_2 = tsig_volts[1:-1]


    y_tsig = [y-z for y,z in zip(tsig_volts_2,zero_volts_2)]
    y_bsln = [y-z for y,z in zip(bsln_volts_2,zero_volts_2)]
    g_corr = [y/0.625 for y in y_bsln]
    
    print 'This is the gain correction'
    plt.plot(sensor_time_2,g_corr)
    plt.show()

    print 'This is the gain corrected tsig'
    x_tsig = [y/g for y,g in zip(y_tsig,g_corr)]
    plt.plot(sensor_time_2,x_tsig)
    plt.show()

    return sensor_time_2, x_tsig

import tmc_parse_data
import dmm_interp
def main():

    chan = '0'
    adc = '11'
    
    # September 20, 2016 (the 12 hour run)
    # tmc_file = '../tmc_cal_data/tmeas_2016-09-20_13_15_40_177697.txt'
    # dmm_file = '../tmc_cal_data/hp34401a_2016-09-20_13_15_05_748130.txt'
    
    # September 21, 2016 (the 1 week run)
    # tmc_file = '../tmc_cal_data/tmeas_2016-09-21_15_31_42_428268.txt'
    # dmm_file = '../tmc_cal_data/hp34401a_2016-09-21_15_31_33_706609.txt'

    # October 18, 2016 (the bad HPF board)
    # tmc_file = '../tmc_cal_data/tmeas_2016-10-18_13_24_16_642445.txt'
    # dmm_file = '../tmc_cal_data/hp34401a_2016-10-18_13_24_09_926237.txt'

    # October 24, 2016 (using an external +12V Agilent linear regulating supply)
    # tmc_file = '../tmc_cal_data/tmeas_2016-10-24_14_56_14_908250.txt'
    # dmm_file = '../tmc_cal_data/hp34401a_2016-10-24_14_31_31_869278.txt'

    # October 28. 2016 (switch to LM399A)
    # tmc_file = '../tmc_cal_data/tmeas_2016-10-28_18_30_08_343466.txt'
    # dmm_file = '../tmc_cal_data/hp34401a_2016-10-28_18_28_35_305888.txt'

    # November 4, 2016 (switch to LM399A)
    tmc_file = '../tmc_cal_data/tmeas_2016-10-31_17_57_26_163981.txt'
    dmm_file = '../tmc_cal_data/hp34401a_2016-10-31_17_57_18_843663.txt'


    fcoeffs = open('tmc_coeffs_bd_0_1_2_3.txt','a')

    # This is for writing the gain corrected signals to files
    f_tsig_gain_corr_name = 'f_tsig_gain_corr_ADC' + str(adc) + '_CHAN' + str(chan) + '.txt'
    f_tsig_gain_corr = open(f_tsig_gain_corr_name,'w')


    # Do the reference drift correction
    sensor_time,sensor_meas = tmc_parse_data.tmc_parse_data(tmc_file,'TSIG'+chan,'ADC'+adc)
    # Cut out the sections where the reference was not connected
    sensor_time_2,sensor_meas_2 = tmc_parse_data.tmc_find_valid_meas(sensor_time,sensor_meas)
    plt.ylabel("TSIG data with Cuts")
    plt.plot(sensor_time,sensor_meas)
    plt.plot(sensor_time_2,sensor_meas_2,color='r',marker='.')
    plt.show()

    # Get the DMM data
    dmm_meas_2 = dmm_interp.dmm_interp(dmm_file,sensor_time_2)
    plt.ylabel("DMM Data (Volts)")
    plt.plot(sensor_time_2,dmm_meas_2)
    plt.show()

    # tsig_volts = [((x/8388608.)-1.)*0.625 + 0.625 for x in sensor_meas_2]
    tsig_volts = [((x/8388608.)-1.)*0.625 for x in sensor_meas_2]
    sensor_volts_dmm_corr = lin_corr_ac(dmm_meas_2,tsig_volts)

    # Quick hack for quick-look plots Mon Oct 31 08:45:54 EDT 2016
    
    tsig_hack_meas = [((x/8388608.)-1.)*0.625 + 0.625 for x in sensor_meas_2]
    tsig_hack_mean = np.mean(tsig_hack_meas)
    tsig_hack_meas = [x-tsig_hack_mean for x in tsig_hack_meas]
    dmm_meas_2_mean = np.mean(dmm_meas_2)
    dmm_meas_2_hack = [x-dmm_meas_2_mean for x in dmm_meas_2]
    plt.plot(sensor_time_2,tsig_hack_meas,'.',color='red')
    plt.plot(sensor_time_2,dmm_meas_2_hack,'.',color='blue')
    plt.show()

    # Another quick hack to show that the DMM agrees with the signal
    diff = [x-y for x,y in zip(tsig_hack_meas,dmm_meas_2_hack)]
    plt.plot(sensor_time_2,diff,'.',color='blue')
    mvavg = moving_average(diff,40)
    plt.plot(sensor_time_2,mvavg,color='red')
    plt.show()

    #################################################
    # TSIG
    tsig_dates = [datetime.datetime.fromtimestamp(ts) for ts in sensor_time_2]
    
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
    
    # ZERO
    zero_time,zero_meas = tmc_parse_data.tmc_parse_data(tmc_file,'ZERO','ADC'+adc)
    zero_dates = [datetime.datetime.fromtimestamp(ts) for ts in zero_time]
    zero_volts = [(x/8388608.-1)*0.625*2 for x in zero_meas]

    # Plot up the DMM corrected data
    plot_all(tsig_dates,sensor_volts_dmm_corr,
             curr_dates,curr_ma_2,
             atemp_dates,atemp_degc,
             btemp_dates,btemp_degc,
             bsln_dates,bsln_volts) 

    # Another hack
    plot_all(tsig_dates,mvavg,
             curr_dates,curr_ma_2,
             atemp_dates,atemp_degc,
             btemp_dates,btemp_degc,
             bsln_dates,bsln_volts) 


    ################################################
    # Fri Oct 21 14:40:37 EDT 2016
    # Try to do a gain correction. Maybe it will help
    print 'Gain correction'
    time_gain_corr, tsig_gain_corr = tmc_gain_corr(sensor_time_2,tsig_volts,zero_time,zero_volts,bsln_time,bsln_volts)
    btemp_interp = meas_interp.meas_interp(btemp_time,btemp_degc,time_gain_corr)
    plt.plot(btemp_interp,tsig_gain_corr,'.')
    plt.show()
    
    ################################################
    # Do a dmm correction on the gain corrected data
    print 'DMM correction'
    dmm_meas_3 = dmm_interp.dmm_interp(dmm_file,time_gain_corr)
    sensor_volts_dmm_corr_2 = lin_corr_ac(dmm_meas_3,tsig_gain_corr)
    sensor_volts_dmm_corr_2_mean = np.mean(sensor_volts_dmm_corr_2)
    sensor_volts_dmm_corr_2_mean_sub = [x-sensor_volts_dmm_corr_2_mean for x in sensor_volts_dmm_corr_2]
    sensor_volts_dmm_corr_mean = np.mean(sensor_volts_dmm_corr)
    sensor_volts_dmm_corr_mean_sub = [x - sensor_volts_dmm_corr_mean for x in sensor_volts_dmm_corr]
    plt.plot(sensor_time_2,sensor_volts_dmm_corr_mean_sub,color='r')
    plt.plot(time_gain_corr,sensor_volts_dmm_corr_2_mean_sub)
    plt.show()

    for t,x in zip(time_gain_corr,sensor_volts_dmm_corr_2_mean_sub):
        line = str(t) + ' ' + str(x) + '\n'
        f_tsig_gain_corr.write(line)

    f_tsig_gain_corr.close()

    print 'Correlated to btemp'
    plt.plot(btemp_interp,sensor_volts_dmm_corr_2,'.')
    plt.show()


    ################################################
    # Plot TSIG vs BSLN
    print "TSIG vs BSLN"
    bsln_interp = meas_interp.meas_interp(bsln_time,bsln_volts,sensor_time_2)
    plt.plot(tsig_volts,bsln_interp,'.')
    plt.show()

    # Plot the difference between TSIG and baseline
    tsig_sub_bsln = [x-y for x,y in zip(tsig_volts,bsln_interp)]
    plt.plot(sensor_time_2,tsig_sub_bsln,'.')
    plt.show()

    #################################################
    # Find tempco of TSIG, do it in raw ADC units
    sensor_adu_dmm_corr = [((x-0.625)/0.625 + 1)*8388608 for x in sensor_volts_dmm_corr]
    A_tsig_vs_btemp,B_tsig_vs_btemp,sensor_adu_btemp_time,sensor_adu_btemp_corr = tmc_corr(sensor_time_2,sensor_adu_dmm_corr,btemp_time,btemp_degc)
    print 'TSIG corrected for temperature: Slope = %f, Intercept = %f' %(A_tsig_vs_btemp,B_tsig_vs_btemp)
    print 'Taking -2.5uV/mK as the sensor temperature coefficient, the temperature error for the TMC alone would be %f mK/degC' % (A_tsig_vs_btemp*0.625/(2.5*8388608.)*1.E6)

    ################################################
    # Find the calibration of TSIG to the DMM
    plt.plot(sensor_meas_2,dmm_meas_2)
    plt.show()

    # Find the First Pass correlation
    # A_dmm_tsig, B_dmm_tsig, tsig_time_dmm_corr, tsig_dmm_corr = tmc_corr(sensor_time_2,dmm_meas_2,sensor_time_2,sensor_meas_2)
    A_dmm_tsig, B_dmm_tsig,dum1,dum2 = tmc_corr(sensor_time_2,dmm_meas_2,sensor_time_2,sensor_meas_2)
    print 'First Pass: TSIG calibrated to DMM: Slope = %f microvolts/ADU, Intercept = %f Volts' % (A_dmm_tsig*1.E6,B_dmm_tsig) 

    # Cut the outliers
    tsig_flat = lin_corr(sensor_meas_2,dmm_meas_2,A_dmm_tsig,B_dmm_tsig)
    print 'Here comes the flat'
    plt.plot(sensor_time_2,tsig_flat)
    plt.show()
    time_out_cut,dum_x1,tsig_out_cut = cut_outliers(sensor_time_2,tsig_flat,sensor_meas_2)
    time_out_cut,dum_x1,dmm_out_cut = cut_outliers(sensor_time_2,tsig_flat,dmm_meas_2)
    print 'Here comes the cut values'
    plt.plot(tsig_out_cut,dmm_out_cut,'.')
    
    # Do the correction
    B3 = 0 # I got a somewhat unexpected slope when I let the intercept be free
    A3, = curve_fit(lambda x, a : f_line(x,a,B3),tsig_out_cut,dmm_out_cut)[0]
    dumb_line = []
    for x in tsig_out_cut:
        dumb_line.append(x*A3+B3)
    plt.plot(tsig_out_cut,dumb_line,color='r')
    plt.show()
    print "ADC channels to voltage calibration: Slope = %g (Volts/count), Intercept = %g V" %(A3,B3)

    fcoeffs.write(tmc_file)
    coef_str = '    ADC' + adc + ', TSIG' + chan + ': ' + str(A_tsig_vs_btemp) + ', ' + str(B_tsig_vs_btemp) + '; ' + str(A3) + '\n'
    fcoeffs.write(coef_str)
    fcoeffs.close()


if __name__ == "__main__":
    main()
    
