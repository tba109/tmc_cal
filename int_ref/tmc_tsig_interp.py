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

def plot_all(time_dt,v700m,v180m,vbsln,vzero,ctemp):
    # Make a pretty plot
    ax1 = plt.subplot(515)
    plt.ylabel('v700m (V)')
    plt.setp(ax1.get_xticklabels(), fontsize=8)
    plt.xticks(rotation=25)
    plt.plot(time_dt,v700m)
    
    ax2 = plt.subplot(514, sharex=ax1)
    plt.ylabel('v180m')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.plot(time_dt,v180m)
    
    ax3 = plt.subplot(513, sharex=ax1)
    plt.ylabel('vbsln (V)')
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.plot(time_dt,vbsln)
    
    ax4 = plt.subplot(512, sharex=ax1)
    plt.ylabel('vzero (V)')
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.plot(time_dt,vzero)
    
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
    
    # November 17, 2016 (LM399A, only ADC0 CH0 and CH1, remove distribution boards, PGA=1 for all)
    tmc_file = '../../tmc_cal_data/tmeas_2016-11-11_17_28_59_569650.txt'
    dmm_file = '../../tmc_cal_data/hp34401a_2016-11-11_17_28_44_915809.txt'

    ########################################################################################################
    # Plot all of the data

    # Look at the 0.7V data
    chan = '0'
    adc = '0'
    sensor_time_700m,sensor_meas_700m = tmc_parse_data.tmc_parse_data(tmc_file,'TSIG'+chan,'ADC'+adc)
    sensor_mvavg_700m = moving_average(sensor_meas_700m,40)
    v_700m = [(x/8388608. - 1)*2.5 for x in sensor_mvavg_700m]
    # plt.ylabel("TSIG (Volts)")
    # plt.plot(sensor_time_700m,v_700m,color='blue')
    # plt.show()
    
    # Look at the 0.18V data
    chan = '1'
    adc = '0'
    sensor_time_180m,sensor_meas_180m = tmc_parse_data.tmc_parse_data(tmc_file,'TSIG'+chan,'ADC'+adc)
    sensor_mvavg_180m = moving_average(sensor_meas_180m,40)
    v_180m = [(x/8388608. - 1)*2.5 for x in sensor_mvavg_180m]
    # plt.ylabel("TSIG (Volts)")
    # plt.plot(sensor_time_180m,v_180m,color='blue')
    # plt.show()

    # Look at the 7V TMC data
    dmm_meas = dmm_interp.dmm_interp(dmm_file,sensor_time_700m)
    dmm_meas_mvavg = moving_average(dmm_meas,400)
    # plt.ylabel("DMM Data (Volts)")
    # plt.plot(sensor_time_700m,dmm_meas_mvavg)
    # plt.show()

    # Baseline
    bsln_time,bsln_meas = tmc_parse_data.tmc_parse_data(tmc_file,'BSLN','ADC'+adc)
    bsln_mvavg = moving_average(bsln_meas,40)
    v_bsln = [(x/8388608. - 1)*2.5 for x in bsln_mvavg]
    # plt.ylabel("Baseline (Volts)")
    # plt.plot(bsln_time,v_bsln)
    # plt.show()

    # Zero 
    zero_time,zero_meas = tmc_parse_data.tmc_parse_data(tmc_file,'ZERO','ADC'+adc)
    zero_mvavg = moving_average(zero_meas,40)
    v_zero = [(x/8388608. - 1)*2.5 for x in zero_meas]
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
    time_ts = sensor_time_700m[10:-10] # This peels off some problematic boundaries
    time_dt = [datetime.datetime.fromtimestamp(ts) for ts in time_ts]

    #######################################################################################################
    # Make some interpolating functions
    f_700m = interpolate.interp1d(sensor_time_700m,v_700m)
    f_180m = interpolate.interp1d(sensor_time_180m,v_180m)
    f_bsln = interpolate.interp1d(bsln_time,v_bsln)
    f_zero = interpolate.interp1d(zero_time,v_zero)
    f_btemp = interpolate.interp1d(btemp_time,btemp_degc)
    f_atemp = interpolate.interp1d(atemp_time,atemp_degc)
    f_dmm = interpolate.interp1d(sensor_time_700m,dmm_meas_mvavg)

    #######################################################################################################
    # Step 1: Start with the follow raw signals
    v700m_1 = f_700m(time_ts)
    v180m_1 = f_180m(time_ts)
    vbsln_1 = f_bsln(time_ts)
    vzero_1 = f_zero(time_ts)
    atemp_1 = f_atemp(time_ts)
    btemp_1 = f_btemp(time_ts)
    ctemp_1 = btemp_1 # This gets used beyond, so pick atemp or btemp and run with it
    vdmm_1 = f_dmm(time_ts)
    print 'Step 1: raw signals'
    plot_all(time_dt,v700m_1,v180m_1,vbsln_1,vzero_1,ctemp_1)

    # print('Does BTEMP follow ATEMP very well?')
    # plt.plot(time_ts,btemp_1 - atemp_1)
    # plt.show()

    ########################################################################################################
    # Step 2: Subtract off the zero point
    v700m_2 = [x-z for x,z in zip(v700m_1,vzero_1)]
    v180m_2 = [x-z for x,z in zip(v180m_1,vzero_1)]
    vbsln_2 = [x-z for x,z in zip(vbsln_1,vzero_1)]
    vzero_2 = [x-z for x,z in zip(vzero_1,vzero_1)]
    ctemp_2 = ctemp_1
    print 'Step 2: zero point subtraction'
    plot_all(time_dt,v700m_2,v180m_2,vbsln_2,vzero_2,ctemp_2)

    ########################################################################################################
    # Step 3a: This is what it looks like with an additive baseline correction
    bsln_mean = np.mean(vbsln_2)
    bsln_corr = [x - bsln_mean for x in vbsln_2] 
    v700m_3a = [x-z for x,z in zip(v700m_2,bsln_corr)]
    v180m_3a = [x-z for x,z in zip(v180m_2,bsln_corr)]
    vbsln_3a = [x-z for x,z in zip(vbsln_2,bsln_corr)]
    vzero_3a = [x-z for x,z in zip(vzero_2,bsln_corr)]
    ctemp_3a = ctemp_2
    print 'Step 3a: This is what it looks like with an additive baseline correction'
    plot_all(time_dt,v700m_3a,v180m_3a,vbsln_3a,vzero_3a,ctemp_3a)
    
    ########################################################################################################
    # Step 3b: This is what it looks like with a gain baseline correction
    mean_bsln = np.mean(vbsln_2)
    gainc_3b = [x/mean_bsln for x in vbsln_2]
    # print 'Here is the gain correction'
    # plt.plot(time_ts,gainc_3b)
    # plt.show()

    v700m_3b = [x/g for x,g in zip(v700m_2,gainc_3b)]
    v180m_3b = [x/g for x,g in zip(v180m_2,gainc_3b)]
    vbsln_3b = [x/g for x,g in zip(vbsln_2,gainc_3b)]
    vzero_3b = [x/g for x,g in zip(vzero_2,gainc_3b)]
    ctemp_3b = ctemp_2
    print 'Step 3b: This is what it looks like with a gain baseline correction'
    plot_all(time_dt,v700m_3b,v180m_3b,vbsln_3b,vzero_3b,ctemp_3b)
    
    ########################################################################################################
    print 'So, let\'s work with that additive offset a little more.'
    print 'What happens if I correlate 700m (red) with vdmm (green)?'
    vdmm_corr = np.mean(v700m_3a)/np.mean(vdmm_1)
    # vdmm_corr = 1.
    print 'vdmm_corr = %f' % vdmm_corr
    vdmm_3a = [x*vdmm_corr for x in vdmm_1]
    # plt.plot(time_ts,v700m_3a)
    # plt.show()
    # plt.plot(time_ts,vdmm_3a)
    # plt.show()
    plt.plot(time_ts,mean_sub(v700m_3a),color='red')
    plt.plot(time_ts,mean_sub(vdmm_3a),color='green')
    plt.plot(time_ts,mean_sub(v700m_3a)-mean_sub(vdmm_3a),color='violet')
    plt.show()

    #######################################################################################################
    # And now, a question: how well do baseline, ADC temp, and board temp match? 
    print 'How well do -1*mean_sub_norm(v700m_2,vbsln_2,btemp_1,atemp_1) all correlate?'
    plt.plot(time_ts,-1.*mean_sub_norm(v700m_2),color='black')
    plt.plot(time_ts,-1.*mean_sub_norm(vbsln_2),color='red')
    plt.plot(time_ts,mean_sub_norm(btemp_1),color='green')
    plt.plot(time_ts,mean_sub_norm(atemp_1),color='blue')
    plt.show()
    
    #######################################################################################################
    # Let's try a slightly different approach: plot v700m_2 versus atemp_1
    print 'Step 3c: Try correlation between v700m_2 and atemp_1?'
    popt, pcov = curve_fit(f_line,atemp_1,v700m_2)
    print 'a = %f, b = %f' % (popt[0],popt[1])
    plt.plot(atemp_1,v700m_2)
    plt.plot(atemp_1,f_line(atemp_1,popt[0],popt[1]))
    plt.show()

    # Tired of typing these
    a=popt[0]
    b=popt[1]

    print 'Now how well do these coefficients do?'
    plt.plot(time_ts,mean_sub(v700m_2),color='blue')
    plt.plot(time_ts,mean_sub(a*atemp_1),color='red')
    plt.plot(time_ts,mean_sub(v700m_2-a*atemp_1),color='green')
    plt.show()

    #######################################################################################################
    # Step 3c: do a correction based on ATEMP. This may be somewhat more robust (I hope!) than subtracting
    # baseline. 
    v700m_3c = [y-a*x-b for x,y in zip(atemp_1,v700m_2)]
    v700m_3c = moving_average(v700m_3c,40)
    plt.plot(time_ts,v700m_3c)
    plt.show()

    ########################################################################################################
    # Let's try to make the comparison between the three methods easier
    print 'Compare the 700m baseline additive (red) and gain (green) corrections and the temperature correction (blue)'
    plt.plot(time_ts,mean_sub(v700m_3a),color='red')
    plt.plot(time_ts,mean_sub(v700m_3b),color='green')
    plt.plot(time_ts,mean_sub(v700m_3c),color='blue')
    plt.show()
    
    print 'Compare the 180m baseline additive (red) and gain (green) corrections'
    plt.plot(time_ts,mean_sub(v180m_3a),color='red')
    plt.plot(time_ts,mean_sub(v180m_3b),color='green')
    plt.show()
    
    print 'So far, consistent with what I saw last year during the stability run.'
    print 'The data is best described as having an additive offset, although the'
    print 'zero point measurement doesn\'t show this.'
    print 'The effect is after the PGA, as it seems to have roughly the same'
    print 'magnitude, regardless of the PGA settings.'
    print 'The only way I see that this can happen is if there is a thermocouple'
    print 'on the reference voltage. Maybe this is the case....I could certainly'
    print 'be believed. That reference voltage connection method I did is maybe a'
    print 'bit drifty'    

    ########################################################################################################
    # Step 4: DMM correlation for the atemp correlated data
    plt.plot(time_ts,mean_sub(v700m_3c),color='red')
    plt.plot(time_ts,mean_sub(vdmm_3a),color='green')
    plt.plot(time_ts,mean_sub(v700m_3c)-mean_sub(vdmm_3a),color='violet')
    plt.show()


if __name__ == "__main__":
    main()
