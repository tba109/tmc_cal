import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import time
import datetime
import sys
from scipy.optimize import curve_fit
import meas_interp
from scipy import interpolate

# Subtract the mean
def mean_sub(x):
    return (x-np.mean(x))


# Subtract the mean and normalize it
def mean_sub_norm(x):
    return ((x-np.mean(x))/np.max(x-np.mean(x)))

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

def plot_all(time_dt,v700m,vbs,vls,vss,ctemp):
    # Make a pretty plot
    ax1 = plt.subplot(515)
    plt.ylabel('v700m (V)')
    plt.setp(ax1.get_xticklabels(), fontsize=8)
    plt.xticks(rotation=25)
    plt.plot(time_dt,v700m)
    
    ax2 = plt.subplot(514, sharex=ax1)
    plt.ylabel('vbs (V)')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.plot(time_dt,vbs)
    
    ax3 = plt.subplot(513, sharex=ax1)
    plt.ylabel('vbsln (V)')
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.plot(time_dt,vls)
    
    ax4 = plt.subplot(512, sharex=ax1)
    plt.ylabel('vss (V)')
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.plot(time_dt,vss)
    
    ax5 = plt.subplot(511, sharex=ax1)
    plt.ylabel('btemp (V)')
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.plot(time_dt,ctemp)

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
    
    # December 12, 2016 (LM399A, only ADC0 CH0 and CH1, remove distribution boards, PGA=2 for all), 700m
    tmc_file = '../../tmc_cal_data/tmeas_2016-12-10_16_49_39_779812.txt'
    dmm_file = '../../tmc_cal_data/hp34401a_2016-12-10_16_49_35_184315.txt'

    ########################################################################################################
    # Plot all of the data

    # Look at the 0.7V data
    chan = '0'
    adc = '0'
    sensor_time_700m,sensor_meas_700m = tmc_parse_data.tmc_parse_data(tmc_file,'TSIG'+chan,'ADC'+adc)
    sensor_mvavg_700m = moving_average(sensor_meas_700m,40)
    v_700m = [(x/8388608. - 1)*1.25 for x in sensor_mvavg_700m]
    # plt.ylabel("TSIG (Volts)")
    # plt.plot(sensor_time_700m,v_700m,color='blue')
    # plt.show()
    
    # Look at the banana short data
    chan = '1'
    adc = '0'
    sensor_time_bs,sensor_meas_bs = tmc_parse_data.tmc_parse_data(tmc_file,'TSIG'+chan,'ADC'+adc)
    sensor_mvavg_bs = moving_average(sensor_meas_bs,40)
    v_bs = [(x/8388608. - 1)*1.25 for x in sensor_mvavg_bs]
    # plt.ylabel("TSIG (Volts)")
    # plt.plot(sensor_time_180m,v_180m,color='blue')
    # plt.show()

    # Look at the long short data
    chan = '2'
    adc = '0'
    sensor_time_ls,sensor_meas_ls = tmc_parse_data.tmc_parse_data(tmc_file,'TSIG'+chan,'ADC'+adc)
    sensor_mvavg_ls = moving_average(sensor_meas_ls,40)
    v_ls = [(x/8388608. - 1)*1.25  for x in sensor_mvavg_ls]
    # plt.ylabel("TSIG (Volts)")
    # plt.plot(sensor_time_180m,v_180m,color='blue')
    # plt.show()

    # Look at the short short data
    chan = '3'
    adc = '0'
    sensor_time_ss,sensor_meas_ss = tmc_parse_data.tmc_parse_data(tmc_file,'TSIG'+chan,'ADC'+adc)
    sensor_mvavg_ss = moving_average(sensor_meas_ss,40)
    v_ss = [(x/8388608. - 1)*1.25 for x in sensor_mvavg_ss]
    # plt.ylabel("TSIG (Volts)")
    # plt.plot(sensor_time_180m,v_180m,color='blue')
    # plt.show()

    # Look at the DMM
    dmm_meas = dmm_interp.dmm_interp(dmm_file,sensor_time_700m)
    dmm_meas_mvavg = moving_average(dmm_meas,400)
    plt.ylabel("DMM Data (Volts)")
    plt.plot(sensor_time_700m,dmm_meas_mvavg)
    plt.show()

    # Baseline
    bsln_time,bsln_meas = tmc_parse_data.tmc_parse_data(tmc_file,'BSLN','ADC'+adc)
    bsln_mvavg = moving_average(bsln_meas,40)
    v_bsln = [(x/8388608. - 1)*1.25 for x in bsln_mvavg]
    # plt.ylabel("Baseline (Volts)")
    # plt.plot(bsln_time,v_bsln)
    # plt.show()

    # Zero 
    zero_time,zero_meas = tmc_parse_data.tmc_parse_data(tmc_file,'ZERO','ADC'+adc)
    zero_mvavg = moving_average(zero_meas,40)
    v_zero = [(x/8388608. - 1)*1.25 for x in zero_meas]
    # plt.ylabel("Zero (Volts)")
    # plt.plot(zero_time,v_zero)
    # plt.show()

    # BTEMP
    btemp_time,btemp_meas = tmc_parse_data.tmc_parse_data(tmc_file,'BTEMP','ADC'+'0') # BTEMP doesn't come up every time
    btemp_dates = [datetime.datetime.fromtimestamp(ts) for ts in btemp_time]
    btemp_degc = [625*x/8388608. - 625 -273. for x in btemp_meas]
    # plt.ylabel("Board Temperature (degC)")
    # plt.plot(btemp_time,btemp_degc)
    # plt.show()

    # ATEMP
    atemp_time,atemp_meas = tmc_parse_data.tmc_parse_data(tmc_file,'ATEMP','ADC'+adc)
    atemp_dates = [datetime.datetime.fromtimestamp(ts) for ts in atemp_time]
    atemp_degc = [(x-8388608.)/13584. - 272.5 for x in atemp_meas]
    # plt.ylabel("ADC Temperature (degC)")
    # plt.plot(atemp_time,atemp_degc)
    # plt.show()

    # Create a time variable and work from here
    time_ts = [x for x in sensor_time_700m if (x > 1.48145E9 and x <1.48151E9)]
    time_dt = [datetime.datetime.fromtimestamp(ts) for ts in time_ts]

    #######################################################################################################
    # Make some interpolating functions
    f_700m = interpolate.interp1d(sensor_time_700m,v_700m)
    f_bs = interpolate.interp1d(sensor_time_bs,v_bs)
    f_ls = interpolate.interp1d(sensor_time_ls,v_ls)
    f_ss = interpolate.interp1d(sensor_time_ss,v_ss)
    f_bsln = interpolate.interp1d(bsln_time,v_bsln)
    f_zero = interpolate.interp1d(zero_time,v_zero)
    f_btemp = interpolate.interp1d(btemp_time,btemp_degc)
    f_atemp = interpolate.interp1d(atemp_time,atemp_degc)
    f_dmm = interpolate.interp1d(sensor_time_700m,dmm_meas_mvavg)

    #######################################################################################################
    # Step 1: Start with the follow raw signals
    v700m_1 = f_700m(time_ts)
    vbs_1 = f_bs(time_ts)
    vls_1 = f_ls(time_ts)
    vss_1 = f_ss(time_ts)
    vbsln_1 = f_bsln(time_ts)
    vzero_1 = f_zero(time_ts)
    atemp_1 = f_atemp(time_ts)
    btemp_1 = f_btemp(time_ts)
    ctemp_1 = btemp_1 # This gets used beyond, so pick atemp or btemp and run with it
    vdmm_1 = f_dmm(time_ts)
    print 'Step 1: raw signals'
    # plot_all(time_dt,v700m_1,vbs_1,vls_1,vss_1,ctemp_1)
    # Wed Mar 15 11:39:45 EDT 2017
    # Let's look at baseline instead
    plot_all(time_dt,v700m_1,vbs_1,vbsln_1,vss_1,ctemp_1)

    ########################################################################################################
    # Step 2: Subtract off the zero point
    v700m_2 = [x-z for x,z in zip(v700m_1,vbs_1)]
    vbs_2 = [x-z for x,z in zip(vbs_1,vbs_1)]
    vbsln_2 = [x-z for x,z in zip(vls_1,vbsln_1)]
    vss_2 = [x-z for x,z in zip(vss_1,vbs_1)]
    ctemp_2 = ctemp_1
    print 'Step 2: zero point subtraction'
    plot_all(time_dt,v700m_2,vbs_2,vbsln_2,vss_2,ctemp_2)

    #######################################################################################################
    # Wed Mar 15 11:43:25 EDT 2017
    # I might expect bsln to do a "decent" job as a gain correction here. Let's try it. 
    # Step 3: 
    bsln_mean = np.mean(vbsln_2)
    gcorr = [x/bsln_mean for x in vbsln_2]
    vbsln_3 = [x/y for x,y in zip(vbsln_2,gcorr)]
    v700m_3 = [x/y for x,y in zip(v700m_2,gcorr)]
    print 'Step 3: Try using baseline as a correction'
    plot_all(time_dt,v700m_3,vbs_2,vbsln_3,vss_2,ctemp_2)

    
    #######################################################################################################
    # Let's try a slightly different approach: plot v700m_2 versus btemp_1
    print 'Step 3b: Try correlation between v700m_3 and btemp_1?'
    popt, pcov = curve_fit(f_line,btemp_1,v700m_3)
    print 'a = %f uV/degC, b = %f uV' % (popt[0]*1.E6,popt[1]*1.E6)
    plt.plot(btemp_1,v700m_3)
    plt.plot(btemp_1,f_line(btemp_1,popt[0],popt[1]))
    plt.show()

    # Tired of typing these
    a=popt[0]
    b=popt[1]

    #######################################################################################################
    # Let's try a slightly different approach: plot v700m_2 versus btemp_1
    print 'Step 4: Subtract off the dmm measured drift and find tempco again'
    v700m_4 = [x-y for x,y in zip(v700m_3,vdmm_1)]
    popt, pcov = curve_fit(f_line,btemp_1,v700m_4)
    print 'a = %f uV/degC, b = %f uV' % (popt[0]*1.E6,popt[1]*1.E6)
    plt.plot(btemp_1,v700m_4)
    plt.plot(btemp_1,f_line(btemp_1,popt[0],popt[1]))
    plt.show()

    # Tired of typing these
    a=popt[0]
    b=popt[1]

    print 'The final corrections'
    plot_all(time_dt,v700m_4,vbs_2,vbsln_3,vss_2,ctemp_2)


if __name__ == "__main__":
    main()
