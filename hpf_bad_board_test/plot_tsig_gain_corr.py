import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import time
import datetime
import sys
from scipy.optimize import curve_fit

# color_list = ['b','g','r','c','m','y','k','w']
color_list = ['black','brown','red','orange','yellow','green','darkgreen','cyan','blue','violet','magenta','gray']

# file_list = [
#     'f_tsig_gain_corr_ADC0_CHAN5.txt',
#     'f_tsig_gain_corr_ADC1_CHAN5.txt',
#     'f_tsig_gain_corr_ADC2_CHAN5.txt',
#     'f_tsig_gain_corr_ADC3_CHAN5.txt',
#     'f_tsig_gain_corr_ADC4_CHAN5.txt',
#     'f_tsig_gain_corr_ADC5_CHAN5.txt',
#     'f_tsig_gain_corr_ADC6_CHAN5.txt',
#     'f_tsig_gain_corr_ADC7_CHAN5.txt',
#     'f_tsig_gain_corr_ADC8_CHAN5.txt',
#     'f_tsig_gain_corr_ADC9_CHAN5.txt',
#     'f_tsig_gain_corr_ADC10_CHAN5.txt',
#     'f_tsig_gain_corr_ADC11_CHAN5.txt',
#     ]

file_list = [
    'f_tsig_gain_corr_ADC0_CHAN0.txt',
    'f_tsig_gain_corr_ADC1_CHAN0.txt',
    'f_tsig_gain_corr_ADC2_CHAN0.txt',
    'f_tsig_gain_corr_ADC3_CHAN0.txt',
    'f_tsig_gain_corr_ADC4_CHAN0.txt',
    'f_tsig_gain_corr_ADC5_CHAN0.txt',
    'f_tsig_gain_corr_ADC6_CHAN0.txt',
    'f_tsig_gain_corr_ADC7_CHAN0.txt',
    'f_tsig_gain_corr_ADC8_CHAN0.txt',
    'f_tsig_gain_corr_ADC9_CHAN0.txt',
    'f_tsig_gain_corr_ADC10_CHAN0.txt',
    'f_tsig_gain_corr_ADC11_CHAN0.txt',
    ]


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

t = []
x = []
n = 0
for fname in file_list:
    f = open(fname,'r')
    t.append([])
    x.append([])
    for line in f:
        ti = float(line.split()[0])
        xi = float(line.split()[1])
        t[n].append(ti)
        x[n].append(xi)
    n+=1

# Check it
for tel, xel in zip(t,x):
    for ti,xi in zip(tel,xel):
        print ti,xi

# Convert timestamps to datetime
dates=[]
n=0
for tel in t:
    dates.append([])
    for ti in tel:
        dates[n].append(datetime.datetime.fromtimestamp(ti))
    n+=1

# Check it
for dtel in dates:
    for di in dtel:
        print di

print len(dates)
print len(x)

# Take the moving average
for i in range(len(x)):
    x[i] = 1.E6*moving_average(x[i],20)

# Plot everything
ax1 = plt.subplot(111)
plt.setp(ax1.get_xticklabels(), fontsize=8)
plt.xticks(rotation=25)

icolor = 0
for dtel, xel in zip(dates,x):
    plt.plot(dtel,xel,color=color_list[icolor])
    plt.ylabel('TSIG Measurement ($\mu$V)')
    icolor=(icolor+1)%len(color_list)
plt.show()

# Plot stuff on the same board
icolor = 0
t_bd0 = t[0:3]
x_bd0 = x[0:3]
for tel, xel in zip(t_bd0,x_bd0):
    plt.plot(tel,xel,color=color_list[icolor])
    icolor=(icolor+1)%len(color_list)
plt.show()

# Plot stuff on the same board
icolor = 3
t_bd1 = t[3:6]
x_bd1 = x[3:6]
for tel, xel in zip(t_bd1,x_bd1):
    plt.plot(tel,xel,color=color_list[icolor])
    icolor=(icolor+1)%len(color_list)
plt.show()

# Plot stuff on the same board
icolor = 6
t_bd2 = t[6:9]
x_bd2 = x[6:9]
for tel, xel in zip(t_bd2,x_bd2):
    plt.plot(tel,xel,color=color_list[icolor])
    icolor=(icolor+1)%len(color_list)
plt.show()
        
# Plot stuff on the same board
icolor = 9
t_bd3 = t[9:12]
x_bd3 = x[9:12]
for tel, xel in zip(t_bd3,x_bd3):
    plt.plot(tel,xel,color=color_list[icolor])
    icolor=(icolor+1)%len(color_list)
plt.show()

# Let's find the mean value for the drift between all boards
for i in range(len(t[0])):
    x_mean = (x[0][i] + x[1][i] + x[2][i] + x[3][i] + 
              x[4][i] + x[5][i] + x[6][i] + x[7][i] + 
              x[8][i] + x[9][i] + x[10][i] + x[11][i])/12.
    for j in range(len(x)):
        x[j][i] = x[j][i] - x_mean
        
# Plot everything
icolor = 0
for tel, xel in zip(t,x):
    plt.plot(tel,xel,color=color_list[icolor])
    icolor=(icolor+1)%len(color_list)
plt.show()

# Plot everything in mK
icolor = 0
for tel, xel in zip(t,x):
    x_mk = [xi/(-2.5E-6) for xi in xel]
    plt.plot(tel,x_mk,color=color_list[icolor])
    icolor=(icolor+1)%len(color_list)
plt.show()

