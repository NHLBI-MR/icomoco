import numpy as np


import cmath
import math
import time
import sys

from scipy.fft import fft,ifft,fftshift,ifftshift
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta, lfilter, kaiserord

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def kaiserWindow(beta,lengthF):
    window=np.zeros((lengthF))
    for jj in range(1,lengthF+1):
        denom = I0(20,beta)
        numer = beta*np.sqrt(1 - pow(2*(jj)/(lengthF)-1,2))
        window[jj-1] = I0(20,numer)/denom

    return window

def I0(n,x):

    I0_x = 1.0 
    for ii in range(1,n+1):
        I0_x = I0_x+ pow((x/pow(2,ii)) / n_jiecheng(ii),2)
    return I0_x

def n_jiecheng(n):
    sum = 1
    for ii in range(1,n+1):
        sum = sum * n
    return sum
def kaiser_window_generate(bands,errors,ftype,fs,filterLength):
    nyquist_rate = fs/2
    if(ftype=='highpass' and bands>=nyquist_rate):
        eprint('cutoff < nyquist sampling')
    
    if(ftype=='bandpass'):
        transitionBand = np.min(np.diff([bands]))
        alpha          = np.min(20*abs(math.log10(np.min(errors))))
    

    if(ftype=='highpass'):
        transitionBand = np.min(np.diff(np.array([0,bands,fs])))
        alpha          = np.min(20*abs(math.log10(np.min(errors))))
    


    if(alpha > 50):
        beta = 0.1102*(alpha-8.7)
    else:
        if(alpha >=21 and alpha <=50):
            beta = 0.5842*pow(alpha-21,0.4) + 0.07886*(alpha-21)
        else:
            beta = 0
        
    

    M = math.ceil((alpha - 7.95)/(2.285*transitionBand))

    if(filterLength%2==1 and M%2==0):
        M = M + 1
    
    if(filterLength%2==0 and M%2==1):
        M = M + 1

    window = kaiserWindow(beta, M)
   

    if(filterLength<M):
        filterLength = M
    
    if(ftype == 'bandpass'):
    
        stIn = math.ceil(filterLength*bands[1]/fs)-1
        endIn = math.ceil(filterLength*bands[2]/fs)
   
        rect  = np.zeros((filterLength))
        rect[stIn:(endIn+1)] = 1
     
        
        #plt.plot(np.linspace(0,2*nyquist_rate,filterLength),rect)
        fwindow = np.squeeze(fftshift(fft(fftshift(window.squeeze()))))
        filter = np.convolve(np.squeeze(rect),fwindow,'same')
        #plt.plot(np.linspace(0,2*nyquist_rate,filter.shape[0]),filter)

    if(ftype == 'highpass'):
    
        stIn  = math.ceil(filterLength*bands/fs)-1
        endIn = filterLength
    
        rect  =  np.zeros((filterLength))
        rect[math.floor(stIn):endIn] = 1
     
        #plt.plot(np.linspace(0,2*nyquist_rate,filterLength),rect.squeeze());
       # plt.plot(abs(ifftshift(ifft(ifftshift(rect)))));
        
        xx = rect[::-1]
        rect_filt = np.squeeze(np.concatenate((rect[::-1],rect,rect[::-1])))
        filter = np.convolve(rect_filt,np.squeeze(fftshift(fft(fftshift(window)))),'same');
        filter = filter[filterLength:2*filterLength]


    #plt.plot(np.linspace(0,2*nyquist_rate,filterLength),20*np.log10((filter)));
    #if(ftype == 'bandpass'):
        #plt.plot(np.linspace(0,2*nyquist_rate,filterLength),20*np.log10((filter[0:None])))
        
    #if(ftype == 'highpass'):
        #plt.plot(np.linspace(0,nyquist_rate*2,math.floor(filterLength)),20*np.log10((filter[0:None])))

    return filter
    
    
if __name__ == "__main__":
   # main()
    st = time.time()
    filter = kaiser_window_generate(0.02,[0.01,0.01],'highpass',1.3068,470)
    bpfilter = kaiser_window_generate([0.08,0.1,0.45,0.5],[0.001,0.001,0.001],'bandpass',1/(160*1e-3),1410)
    eprint(filter[0:10])
    eprint(bpfilter[0:10])
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    eprint('Execution time:', elapsed_time, 'seconds')
   

