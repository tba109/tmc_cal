import os.path
import serial
import sys
import datetime

if __name__ == '__main__':

    ser = serial.Serial()
    filename = 'hp34401a_' + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f') + '.txt'
    f = open(filename,'w')
    
    try:
        success = True
        ser = serial.Serial( '/dev/ttyUSB0', 9600, timeout=0.5 )
        cmd = '++mode 0'
        print 'Sending:', cmd        
        ser.write(cmd + '\n')
        while(1):
            s = ser.readline();
            if len(s) > 0:
                print datetime.datetime.now(),s[:-1]
                line = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f') + ' ' + s
                f.write(line)
            
        
    except serial.SerialException, e:
        print e
        f.close()
        
    except KeyboardInterrupt, e:
        ser.close()
        f.close()
