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

def plot_all(time_dt,tsig,curr,vbsln,vzero,ctemp):
    # Make a pretty plot
    ax1 = plt.subplot(515)
    plt.ylabel('tsig (V)')
    plt.setp(ax1.get_xticklabels(), fontsize=8)
    plt.xticks(rotation=25)
    plt.plot(time_dt,tsig)
    
    ax2 = plt.subplot(514, sharex=ax1)
    plt.ylabel('curr (uA)')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.plot(time_dt,curr)
    
    ax3 = plt.subplot(513, sharex=ax1)
    plt.ylabel('bsln (V)')
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.plot(time_dt,vbsln)
    
    ax4 = plt.subplot(512, sharex=ax1)
    plt.ylabel('zero (V)')
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


import tmc_parse_data_instr
import dmm_interp
def main():
    
    tmc_file = '../../raw/combo_pt100_ctrl_June2_2017/combo.txt'
    # tmc_file = '../../raw/combo_pt100_ctrl_June2_2017/combo_2.txt'
    # tmc_file = '../../raw/RawOutput_2017-06-02.txt'

    ########################################################################################################
    # Plot all of the data

    # Look at the 2n2222 data
    chan = '1'
    adc = '7'
    sensor_time_tsig,sensor_meas_tsig = tmc_parse_data_instr.tmc_parse_data(tmc_file,'TSIG'+chan,'ADC'+adc)
    sensor_mvavg_tsig = moving_average(sensor_meas_tsig,40)
    v_tsig = [(x/8388608. - 1)*1.25 for x in sensor_mvavg_tsig]
    # plt.ylabel("TSIG (Volts)")
    # plt.plot(sensor_time_tsig,v_tsig,color='blue')
    # plt.show()

    calref_time_tsig,calref_meas_tsig = tmc_parse_data_instr.tmc_parse_data(tmc_file,'TSIG'+'5','ADC'+'8')
    calref_mvavg_tsig = moving_average(calref_meas_tsig,40)
    v_calref = [(x/8388608. - 1)*1.25 for x in calref_mvavg_tsig]
    # plt.ylabel("TSIG (Volts)")
    # plt.plot(calref_time_tsig,v_tsig,color='blue')
    # plt.show()
    
    # Look at the current
    sensor_time_curr,sensor_meas_curr = tmc_parse_data_instr.tmc_parse_data(tmc_file,'CURR'+chan,'ADC'+adc)
    sensor_mvavg_curr = moving_average(sensor_meas_curr,40)
    v_curr = [(x/8388608. - 1)*1.25 for x in sensor_mvavg_curr]
    i_curr = [v*100 for v in v_curr]
    # plt.ylabel("TSIG (Volts)")
    # plt.plot(sensor_time_180m,v_180m,color='blue')
    # plt.show()

    # Baseline
    bsln_time,bsln_meas = tmc_parse_data_instr.tmc_parse_data(tmc_file,'BSLN','ADC'+adc)
    bsln_mvavg = moving_average(bsln_meas,40)
    v_bsln = [(x/8388608. - 1)*1.25 for x in bsln_mvavg]
    # plt.ylabel("Baseline (Volts)")
    # plt.plot(bsln_time,v_bsln)
    # plt.show()

    # Zero 
    zero_time,zero_meas = tmc_parse_data_instr.tmc_parse_data(tmc_file,'ZERO','ADC'+adc)
    zero_mvavg = moving_average(zero_meas,40)
    v_zero = [(x/8388608. - 1)*1.25 for x in zero_meas]
    # plt.ylabel("Zero (Volts)")
    # plt.plot(zero_time,v_zero)
    # plt.show()

    # BTEMP
    btemp_time,btemp_meas = tmc_parse_data_instr.tmc_parse_data(tmc_file,'BTEMP','ADC'+'0') # BTEMP doesn't come up every time
    btemp_dates = [datetime.datetime.fromtimestamp(ts) for ts in btemp_time]
    btemp_degc = [625*x/8388608. - 625 -273. for x in btemp_meas]
    # plt.ylabel("Board Temperature (degC)")
    # plt.plot(btemp_time,btemp_degc)
    # plt.show()

    # ATEMP
    atemp_time,atemp_meas = tmc_parse_data_instr.tmc_parse_data(tmc_file,'ATEMP','ADC'+adc)
    atemp_dates = [datetime.datetime.fromtimestamp(ts) for ts in atemp_time]
    atemp_degc = [(x-8388608.)/13584. - 272.5 for x in atemp_meas]
    # plt.ylabel("ADC Temperature (degC)")
    # plt.plot(atemp_time,atemp_degc)
    # plt.show()

    # Create a time variable and work from here
    # time_ts = [x for x in sensor_time_tsig if (x > 1.48145E9 and x <1.48151E9)]
    time_ts = sensor_time_tsig[10:-10]
    time_dt = [datetime.datetime.fromtimestamp(ts) for ts in time_ts]

    #######################################################################################################
    # Make some interpolating functions
    f_tsig = interpolate.interp1d(sensor_time_tsig,v_tsig)
    f_calref = interpolate.interp1d(calref_time_tsig,v_calref)
    f_curr = interpolate.interp1d(sensor_time_curr,i_curr)
    f_bsln = interpolate.interp1d(bsln_time,v_bsln)
    f_zero = interpolate.interp1d(zero_time,v_zero)
    f_btemp = interpolate.interp1d(btemp_time,btemp_degc)
    f_atemp = interpolate.interp1d(atemp_time,atemp_degc)

    #######################################################################################################
    # Step 1: Start with the follow raw signals
    vtsig_1 = f_tsig(time_ts)
    vcalref_1 = f_calref(time_ts)
    curr_1 = f_curr(time_ts)
    vbsln_1 = f_bsln(time_ts)
    vzero_1 = f_zero(time_ts)
    atemp_1 = f_atemp(time_ts)
    btemp_1 = f_btemp(time_ts)
    ctemp_1 = btemp_1 # This gets used beyond, so pick atemp or btemp and run with it
    print 'Step 1: raw signals'
    # plot_all(time_dt,vtsig_1,curr_1,vbsln_1,vzero_1,ctemp_1)
    plot_all(time_dt,vtsig_1,curr_1,vbsln_1,vcalref_1,ctemp_1)

    #######################################################################################################
    # Step 2: 
    bsln_mean = np.mean(vbsln_1)
    vtsig_2 = (vtsig_1-vzero_1)*bsln_mean/vbsln_1
    curr_2 = (curr_1-vzero_1)*bsln_mean/vbsln_1
    vcalref_2 = (vcalref_1-vzero_1)*bsln_mean/vbsln_1
    # vtsig_2 = (vtsig_1)*bsln_mean/vbsln_1
    # curr_2 = (curr_1)*bsln_mean/vbsln_1
    # vcalref_2 = (vcalref_1)*bsln_mean/vbsln_1
    print 'Step 2: baseline normalization of tsig, curr, v'
    plot_all(time_dt,vtsig_2,curr_2,vbsln_1,vcalref_2,ctemp_1)
    
    # mean_vtsig_2 = np.mean(vtsig_2)
    # mean_vcalref_2 = np.mean(vcalref_2)
    # plt.plot(time_dt,vtsig_2 - mean_vtsig_2)
    # # plt.plot(time_dt,(vcalref_2 - mean_vcalref_2)*mean_vtsig_2/mean_vcalref_2)
    # plt.plot(time_dt,(vcalref_2 - mean_vcalref_2))
    # plt.show()

    # mean_sub_tsig_sub_vcalref = vtsig_2 - mean_vtsig_2 - (vcalref_2 - mean_vcalref_2)*3/2.
    # vcalref_mean = np.mean(vcalref_2)
    # plt.plot(time_dt,mean_sub_tsig_sub_vcalref)
    # mean2 = np.mean(vtsig_2*vcalref_mean/vcalref_2)
    # plt.plot(time_dt,vtsig_2*vcalref_mean/vcalref_2 - mean2) 
    # plt.show()

    #######################################################################################################
    # Step 3: 
    vcalref_mean = np.mean(vcalref_2)
    vtsig_3 = vtsig_2*vcalref_mean/vcalref_2
    curr_3 = curr_2*vcalref_mean/vcalref_2
    curr_3 = curr_2*vcalref_mean/vcalref_2*2273/1.E6
    # ctemp_3 = moving_average(ctemp_1,10000)
    print 'Step 3: tsig and curr calibration reference normalization'
    plot_all(time_dt,vtsig_3,curr_3,vbsln_1,vcalref_2,ctemp_1)


    print 'Plottig in mK'
    vtsig_3_mK = vtsig_3*1.E6/-2.5
    vtsig_3_mK_mean = np.mean(vtsig_3_mK)
    vtsig_3_mK = vtsig_3_mK - vtsig_3_mK_mean
    print 'RMS = %f mK' % np.std(vtsig_3_mK)
    plt.plot(time_dt,vtsig_3_mK)
    # plt.ylim(-1,1)
    plt.show()
    
    #######################################################################################################
    # Step 4: 
    vtsig_4 = vtsig_3 - curr_3
    print 'Step 4: tsig current correction'
    plot_all(time_dt,vtsig_4,curr_3,vbsln_1,vcalref_2,ctemp_1)
    
    # plt.plot(time_dt, vtsig_3-np.mean(vtsig_3))
    # plt.plot(time_dt, vtsig_4-np.mean(vtsig_4))
    # plt.show()

if __name__ == "__main__":
    main()
