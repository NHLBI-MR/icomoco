
import gadgetron
import ismrmrd as mrd

import numpy as np
import cupy as cp
import sys
import cmath
import math
import time
import shelve
from scipy.signal import butter, filtfilt
from scipy import signal


import cupyx.scipy

from scipy.fft import fft,ifft,fftshift,ifftshift
from cupyx.scipy.fft import fft as cufft
from cupyx.scipy.fft import ifft as cuifft
from cupyx.scipy.fft import fftshift as cufftshift
from cupyx.scipy.fft import ifftshift as cuifftshift
from cupyx.scipy import ndimage
from cupy.linalg import svd as cusvd

import importlib
import inspect
#import matplotlib.pyplot as plt
import kaiser_window

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def _parse_params(xml):
    return {p.get('name'): p.get('value') for p in xml.iter('property')}

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, cp.asnumpy(data))
    return y

def cufilterData(input, filter):

    
    cuin = cp.asarray(input) 
    if(len(cuin.shape)>2):
        cuin = cp.reshape(cuin,(cuin.shape[0]*cuin.shape[1],cuin.shape[2]))
    filter = cp.asarray(filter)
    
    concat_arr = cp.concatenate(((cuin[:,::-1]),cuin[:,:],(cuin[:,::-1])),axis=1).astype(cp.complex64)
    st = time.time()
    
    temp = cp.zeros([concat_arr.shape[0],max(concat_arr.shape[1],filter.shape[0])],cp.complex64)

    if(cuin.shape[1]>cuin.shape[0]):
        for ii in range(0,cuin.shape[0]):
            temp[ii,:] = cp.convolve(concat_arr[ii,:].squeeze(),cufftshift(cufft(filter)),'same')
    else:
        temp = ndimage.convolve1d(concat_arr,cufftshift(cufft(filter)),axis=1)
    

    cp.cuda.runtime.deviceSynchronize()
    
    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (filtrations):', elapsed_time, 'seconds')
    out = temp[:,int(temp.shape[1]/2-cuin.shape[1]/2):int(temp.shape[1]/2+cuin.shape[1]/2)]

    if(len(input.shape)>2):
        out = cp.reshape(out,(input.shape[0],input.shape[1],input.shape[2]))

    
    
    return out

def filterData(input, filter):
    if(len(input.shape)>2):
            input = np.reshape(input,(input.shape[0]*input.shape[1],input.shape[2]))
    out = input
    if(filter.shape[0] > input.shape[1]):
        for ii in range(0,input.shape[0]):
                concat_arr = np.concatenate(((input[ii,::-1]),input[ii,:],(input[ii,::-1])))
                temp = np.real(np.convolve(concat_arr,fftshift(fft(filter)),'same'))
                out[ii,:] = temp[int(temp.shape[0]/2-input.shape[2]/2):int(temp.shape[0]/2+input.shape[2]/2)]
    else:
        temp = np.zeros((input.shape[0],3*input.shape[1]),dtype=complex)
        for ii in range(0,input.shape[0]):
                temp[ii,:] = (np.convolve(np.concatenate((np.flipud(input[ii,:]),input[ii,:],np.flipud(input[ii,:]))),fftshift(fft((filter))),'same'))
    
        out = temp[:,int(temp.shape[1]/2-input.shape[1]/2):int(temp.shape[1]/2+input.shape[1]/2)]
    return out

def correctTrajectoryFluctuations(data_array,navangles):
    na  = cp.asnumpy(navangles)
    una = np.unique(na)
    idx = np.argsort(-1*na)
    interleaves = len(una)

    factor = (int(math.ceil(data_array.shape[1] / interleaves)) % int(interleaves)) - round(data_array.shape[1] / interleaves)
    # print(na[idx[0:20]])
    # print(len(na))
    # print(np.min(na))
    # print(np.max(na))
    #factor = int(np.prod(data_array.shape) / (data_array.shape[0] * (int(round(data_array.shape[1] / interleaves)) % int(interleaves))) - interleaves)
    if (factor < 0):
        factor = 0

    
    nav_samplingTime = 0
    numNavsPerStack = int(len(navangles) / interleaves)
    # print("numNavsPerStack:",numNavsPerStack)
    # print("len(navangles):",len(navangles))
    # print("interleaves:",interleaves)

    for  ii in range(numNavsPerStack + 1,len(na),numNavsPerStack): # this hardcoding is a potential bug
            nav_samplingTime += na[idx[ii - numNavsPerStack]] - na[idx[ii]] 
    
    # print("nav_samplingTime:",nav_samplingTime)
    # print("data_array:",data_array.shape)

    nav_samplingTime = nav_samplingTime/interleaves
    # print("nav_samplingTime:",nav_samplingTime)
   
    sorted_signal = data_array[:,idx]
    
   
    sorted_signal = cp.reshape(sorted_signal,(data_array.shape[0],int(data_array.shape[1]/interleaves),interleaves+factor))
    
    # print("sorted_signal.shape:",sorted_signal.shape)
    # plt.plot(cp.asnumpy(sorted_signal[0,0,:]).squeeze())
    # plt.show()
    # print("sorted_signal.shape:",sorted_signal.shape)
    #filter = kaiser_window.kaiser_window_generate(0.1,[0.00001,0.00001],'highpass',abs(1/nav_samplingTime),sorted_signal.shape[2])
    #filter = kaiser_window.kaiser_window_generate([0.0001,0.001,abs(1/(nav_samplingTime))-0.001*abs(1/(nav_samplingTime)),abs(1/(nav_samplingTime))],[0.01,0.01,0.01],'bandpass',abs(1/(nav_samplingTime)),sorted_signal.shape[2])
    
    filtered_signal = butter_highpass_filter(sorted_signal.squeeze(), 0.01, abs(1/(nav_samplingTime)), order=5)
    #filtered_signal = cufilterData(sorted_signal.squeeze(),filter)
    # plt.plot(cp.asnumpy(filtered_signal[0,:]).squeeze())
    # plt.show()
    # plt.plot(np.linspace(-1*abs(1/nav_samplingTime),abs(1/nav_samplingTime),math.floor(filtered_signal.shape[2])),ifftshift(ifft(ifftshift(cp.asnumpy(filtered_signal[0,0,:]).squeeze()))))
    # plt.show()
    #filtered_signal = np.transpose(filtered_signal,(1,2,0))

    
    filtered_signalX = cp.reshape(filtered_signal,(data_array.shape[0],data_array.shape[1]))
    filtered_signal = np.copy(filtered_signalX)
    filtered_signal[:,idx] = filtered_signalX
    
    return filtered_signal

def mean2(x):
    y = np.sum(x) / np.size(x)
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r

def findNavAngle(navTimestamps, acqTimestamps, traj_angles):
    orderedInd = np.argsort(acqTimestamps)
    navAngles = []
    
    for jj in range(0,len(navTimestamps)):
        tstamp = navTimestamps[jj]
        size_nA = len(navAngles)

        for ii in range(0,len(acqTimestamps)):
            if (ii > 0 and (int(tstamp) - int(acqTimestamps[orderedInd[ii]])) <= 0 and (int(tstamp) - int(acqTimestamps[orderedInd[ii - 1]])) >= 0):
                navAngles.append(traj_angles[orderedInd[ii - 1]])
                break
            else: 
                if ((int(tstamp) - int(acqTimestamps[orderedInd[ii]])) <= 0 and ii == 0 and jj == 0):
                    navAngles.append(traj_angles[orderedInd[ii]])
                    break
                
    return navAngles

def findNavAngle_fast(traj_angles):
    orderedInd = np.argsort(acqTimestamps)
    navAngles = []
    
    for jj in range(0,len(navTimestamps)):
        tstamp = navTimestamps[jj]
        size_nA = len(navAngles)

        for ii in range(0,len(acqTimestamps)):
            if (ii > 0 and (int(tstamp) - int(acqTimestamps[orderedInd[ii]])) <= 0 and (int(tstamp) - int(acqTimestamps[orderedInd[ii - 1]])) >= 0):
                navAngles.append(traj_angles[orderedInd[ii - 1]])
                break
            else: 
                if ((int(tstamp) - int(acqTimestamps[orderedInd[ii]])) <= 0 and ii == 0 and jj == 0):
                    navAngles.append(traj_angles[orderedInd[ii]])
                    break
                
    return navAngles


def estimateGatingSignal(nav_data,nav_tstamp,navangles,acq_tstamp):


    data_array = cp.asarray(np.concatenate(nav_data,axis=1))    
    data_array= cp.reshape(data_array,(nav_data[0].shape[0],len(nav_data),nav_data[0].shape[1]))
    
    number_channels = data_array.shape[0]
    data_array = cp.abs(cufft(data_array,axis=2))
   
    data_array= cp.transpose(data_array,(0,2,1))
    data_array= cp.reshape(data_array,(nav_data[0].shape[0]*nav_data[0].shape[1],len(nav_data)))
    #data_array= np.transpose(data_array,(1,0))

    data_array = correctTrajectoryFluctuations(data_array,navangles)

    # Bandpass filterations
    samplingTime = 0
    for ii in range(1,len(nav_tstamp)):
        samplingTime += float(nav_tstamp[ii] - nav_tstamp[ii - 1]) * 2.5
    samplingTime = samplingTime/len(nav_tstamp)
    
    max_sampTime = 0
    for ii in range(1,len(nav_tstamp)):
        if(max_sampTime < (nav_tstamp[ii]-nav_tstamp[ii - 1])*2.5):
            max_sampTime = (nav_tstamp[ii] - nav_tstamp[ii - 1]) * 2.5
    if(2*samplingTime > max_sampTime):
        samplingTime = max_sampTime
    
    bpfilter = kaiser_window.kaiser_window_generate([0.08,0.1,0.45,0.50],[0.001,0.001,0.001],'bandpass',1/(samplingTime*1e-3),data_array.shape[1])

    filtered_signal = cufilterData(data_array,bpfilter)

    filtered_signal = cp.asarray(np.reshape(filtered_signal,(nav_data[0].shape[0],nav_data[0].shape[1],len(nav_data))))
    compressed_signal = cp.zeros((filtered_signal.shape[0],filtered_signal.shape[2]),dtype=complex)


    temp = (filtered_signal.transpose((0,2,1))).astype(cp.csingle)
    [u,s,v] = cp.linalg.svd(temp,full_matrices=False)
    compressed_signal = u[:,:,0]


    C=cp.zeros((compressed_signal.shape[0],compressed_signal.shape[0]),dtype=complex)
    G=cp.zeros((compressed_signal.shape[0],compressed_signal.shape[0]),dtype=complex)
    

    threshold = 0.98
    for ii in range(0,compressed_signal.shape[0]):
        for jj in range(0,compressed_signal.shape[0]):
            C[ii,jj]= corr2(cp.real(compressed_signal[ii,:]).squeeze(),cp.real(compressed_signal[jj,:]).squeeze())
            G[ii,jj]= (cp.abs(C[ii,jj])>threshold)
    

    [ug,sg,vg]=cp.linalg.svd(G,full_matrices=False)
    ind_dom_motion = cp.argwhere(cp.abs(cp.sum(ug[:,cp.argwhere(cp.max(cp.diag(sg))==cp.diag(sg))],axis=1))>0.1)
    ind_dom_motion = cp.argwhere(cp.abs((ug[:,0]))>0.1)

    

    dominantM = C[ind_dom_motion,ind_dom_motion]

    negInd = ind_dom_motion[cp.argwhere(dominantM[:,0]<0)]
    negInd = cp.argwhere(dominantM[:,0]<0)
    yfilt1 = compressed_signal[ind_dom_motion,:]


   

    for ii in range(0,negInd.shape[0]):
        maxC= np.max(yfilt1[negInd(ii),:])
        yfilt1[negInd(ii),:] = yfilt1[negInd(ii),:]*-1
    
    yfilt1 = cp.asnumpy(np.real(np.mean(yfilt1,axis=0)))
    
    return yfilt1, samplingTime/2.5

def stable_binning(selectedSig,timestamp,acceptancePercent,bidirectional):

    eprint('Stable Binning')
    st = time.time()   

    if(bidirectional):
        input_diff = np.diff(selectedSig,axis=1)
        input_diff = np.concatenate(([np.asarray(input_diff[0,0])],input_diff[0,:]),axis=0)
    else:
        input_diff = np.ones(selectedSig.shape)
        
    I = np.argsort(selectedSig*np.sign(input_diff)).squeeze()
    sig = np.sort(selectedSig*np.sign(input_diff))
    #print(I.shape)
    accept_perc = acceptancePercent

    lengthData = int(np.floor(acceptancePercent/100 * sig.shape[1]))
    slope = np.zeros(sig.shape[1]-lengthData)

    st = time.time()   

    ss = cp.zeros((lengthData,sig.shape[1]-lengthData))

    cp.cuda.runtime.deviceSynchronize()

    
    scp = cp.asarray(sig).squeeze()
    cp.cuda.runtime.deviceSynchronize()


    numWindows = 100
    stride = int(np.floor((sig.shape[1]-lengthData)/numWindows))
    indexer = cp.asarray(stride*np.arange(numWindows)[None, :] + np.arange(lengthData)[:, None])

    ss = cp.squeeze(scp[indexer])


    b = cp.polyfit(cp.asarray(range(0,lengthData)), ss, deg=1)
        # for ii in range(0,sig.shape[1]-lengthData):
    #     b = cp.polyfit(cp.asarray(range(0,lengthData)), ss[0,range(ii,ii+lengthData)], deg=1)
    #     slope[ii] = b[0]
    slope = cp.asnumpy(b[0,:].squeeze())
    
    slope[slope==0]=[]
    V=np.min(slope)

    I2=np.argmin(slope)*stride
    

    
    Smin   = np.min(sig[0,range(I2,I2+lengthData-1)])
    Smax   = np.max(sig[0,range(I2,I2+lengthData-1)])
    
    indices = np.flatnonzero( (sig[0,:].squeeze() < Smax) & (sig[0,:].squeeze() > Smin) )
    timestamp = np.array(timestamp)
    accepted_times = timestamp[I[indices]]
    

    #plt.plot(range(0,sig.shape[1]),sig.squeeze(),range(I,I+lengthData),sig[0,range(I,I+lengthData)].squeeze(),"r-")
    cp.cuda.runtime.deviceSynchronize()

    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (stable binning function):', elapsed_time, 'seconds')
             
    return I, Smin, Smax, indices, accepted_times
    
    
    
def binning_div(selectedSig,timestamp,numBins,evenBins,bidirectional):
    
    eprint('Binning Division')
    st = time.time()   
    
    indices = []
    accepted_times = []
    if(bidirectional):
        input_diff = np.diff(selectedSig,axis=1)
        input_diff = np.concatenate(([np.asarray(input_diff[0,0])],input_diff[0,:]),axis=0)
    else:
        input_diff = np.ones(selectedSig.shape)
        
    I = np.argsort(selectedSig*np.sign(input_diff))
    sig = np.sort(selectedSig*np.sign(input_diff))
    
    n95 = np.percentile(sig,99)
    n05 = np.percentile(sig,1)
    
    
    if(evenBins):

        low_idx = np.argmin(np.abs(sig-n05))
        high_idx = np.argmin(np.abs(sig-n95))
        
        delta = np.floor((high_idx-low_idx)/(numBins))
        
        indices_sorted_min = np.floor(low_idx + delta*range(0,numBins))
        indices_sorted_max = np.floor(low_idx + delta*range(1,numBins+1))

    else:
        delta = (n95-n05)/numBins
        min_limits = n05 + delta*range(0,numBins)
        max_limits = n05 + delta*range(1,numBins+1)
        
        indices_sorted_min = []
        indices_sorted_max = []
        
        for ii in range(0,numBins):
            indices_sorted_min.append(max(np.flatnonzero(sig<min_limits[ii])))
            indices_sorted_max.append(max(np.flatnonzero(sig<max_limits[ii])))
    
    timestamp = np.array(timestamp)
    I = I.squeeze()
    for ii in range(0,numBins):
        #eprint(int(indices_sorted_max[ii]))
        indices.append(I[np.array(range(int(indices_sorted_min[ii]),int(indices_sorted_max[ii])))])
        accepted_times.append(timestamp[indices[ii]])

    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (Binning Div function):', elapsed_time, 'seconds')
    return indices, accepted_times
        
        

                                         
        
   
    
def binning(selectedSig, timestamp, acceptancePercent, bidirectional, do_stable_binning, evenBins, numBins):
    eprint("Binning")
    st = time.time()   

    selectedSig = selectedSig-np.min(selectedSig)
    selectedSig = selectedSig/np.percentile(np.abs(np.sort(selectedSig)),99)


    if(do_stable_binning):
        I, Smin, Smax, indices, accepted_times = stable_binning(selectedSig,timestamp,acceptancePercent,bidirectional)
        
        max_resp = Smax
        min_resp = Smin
        
        # Flip sign if the most stable phase is >0.5 of the scaled signal
        selectedSig = selectedSig*np.power(-1,(max_resp+min_resp)/2 >0.5)+1*((max_resp+min_resp)/2 >0.5) 
        
        #% Do it again in case of a flip
        I, Smin, Smax, indices, accepted_times = stable_binning(selectedSig,timestamp,acceptancePercent,bidirectional) 
        
        max_resp = Smax
        min_resp = Smin
        
        atimes =[]
        atimes.append(accepted_times)
    else:
        cp.cuda.runtime.deviceSynchronize()

        I, min_resp, max_resp, ind, at = stable_binning(selectedSig,timestamp,acceptancePercent,bidirectional)
        cp.cuda.runtime.deviceSynchronize()

        # Flip sign if the most stable phase is >0.5 of the scaled signal
        selectedSig = selectedSig*np.power(-1,(max_resp+min_resp)/2 > 0.5)+1*((max_resp+min_resp)/2 > 0.5) 
        
        
        indices, atimes = binning_div(selectedSig,timestamp,numBins,evenBins,bidirectional)
    
    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (binning function):', elapsed_time, 'seconds')
        
    return atimes
    
    
# idx_to_send_nav{1} = get_idx_to_send(header.dataacq_time_stamp,nav_gating.accepted_times,nav_gating.sampling_time)
def get_idx_to_send(data_timestamps,accepted_times,sampling_time):
    idx_to_send =[]
    for ii in range(0,len(data_timestamps)):
        if(np.shape(np.flatnonzero(abs((data_timestamps[ii])-(accepted_times.squeeze())) <= sampling_time/2)>0)[0]):
            idx_to_send.append(ii)
        #    eprint(np.shape(np.flatnonzero(abs((data_timestamps[ii])-(accepted_times.squeeze())) <= sampling_time/2)>0)[0])
    return np.array(idx_to_send)

def get_idx_to_send2(data_timestamps,accepted_times,sampling_time):
    st = time.time()
    idx_to_send =[]
    data_timestamps = cp.expand_dims(cp.asarray(data_timestamps),axis=1)
    accepted_times  = cp.expand_dims(cp.asarray(accepted_times),axis=0)


    ar2 = data_timestamps - accepted_times
    x = cp.flatnonzero(cp.sum(abs(ar2) <= sampling_time/2,axis=1))
    
    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (idx to send):', elapsed_time, 'seconds')
    
    return cp.asnumpy(x)

def get_idx_to_send2_np(data_timestamps,accepted_times,sampling_time):
    st = time.time()
    idx_to_send =[]
    data_timestamps = np.expand_dims((data_timestamps),axis=1)
    accepted_times  = np.expand_dims((accepted_times),axis=0)


    ar2 = data_timestamps - accepted_times
    x = np.flatnonzero(np.sum(abs(ar2) <= sampling_time/2,axis=1))
    
    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (idx to send) Numpy:', elapsed_time, 'seconds')
    
    return (x)

def create_ismrmrd_image(data, reference, field_of_view, index):
        return mrd.image.Image.from_array(
            data,
            acquisition=reference,
            image_index=index,
            image_type=mrd.IMTYPE_MAGNITUDE,
            field_of_view=(field_of_view.x, field_of_view.y, field_of_view.z),
            transpose=False
        )
        
def BinningGadget(connection):
    eprint("Ready to do some Binning in Python")

    #connection.filter(lambda acq: acq)
    connection.filter(mrd.Acquisition)
    
    params = _parse_params(connection.config)

    if "numBins" in params:
        numBins = int(params["numBins"])
    else:    
        numBins = 6
        
    if "useDC" in params:
        if params["useDC"] == 'True':
            useDC = True
        else:
            useDC = False    
    else:    
        useDC = False
    
    if "stableBinning" in params:
        if params["stableBinning"] == 'True':
            do_stable_binning = True
        else:
            do_stable_binning = False
            
    else:
        do_stable_binning = False
        
    if "evenbins" in params:
        if params["evenbins"] == 'True':
            evenbins = True
        else:
            evenbins = False
    else:
        evenbins = True
    
    if "bidirectional" in params:
        if params["bidirectional"] == 'True':
            bidirectional = True
        else:
            bidirectional = False
    else:
        bidirectional = False    
    
    
    eprint("bidirectional: ", bidirectional)
    eprint("evenbins: ", evenbins)
    eprint("stableBinning: ", do_stable_binning)
    eprint("Numbins: ", numBins)
    eprint("useDC: ", useDC)

    count = 0
    firstacq=0
    navangles = []
    acq_tstamp = []
    nav_data    = []
    nav_tstamp  = []
    nav_indices = []
    data_indices = []
    kencode_step = []
    mrd_header = connection.header
    
    eprint("zencode/2:", int(mrd_header.encoding[0].encodedSpace.matrixSize.z/2))
    
    field_of_view = mrd_header.encoding[0].reconSpace.fieldOfView_mm

    st = time.time()

    dx = cp.random.rand(2,2).astype(cp.csingle)
    u,s,v = cp.linalg.svd(dx,full_matrices=False)
    
    input = cp.asarray(cp.random.rand(1,5500))
    filterIn = cp.asarray(cp.random.rand(5500))

    cp.convolve(input[0,:].squeeze(),cufftshift(cufft(filterIn)),'same')
    ndimage.convolve1d(input[0,:].squeeze(),cufftshift(cufft(filterIn)),axis=0)

    ss = cp.zeros((int(5500*0.4),5500-int(5500*0.4)))
    cp.polyfit(cp.asarray(range(0,int(5500*0.4))),ss , deg=1)  
    
    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (warmups):', elapsed_time, 'seconds')
    del ss
    cp._default_memory_pool.free_all_blocks()
    st = time.time()
    acquisition_0 = []
    for acq in connection:
        count= count+1
        if not useDC and (acq.isFlagSet(mrd.ACQ_IS_RTFEEDBACK_DATA or mrd.ACQ_IS_HPFEEDBACK_DATA or mrd.ACQ_IS_NAVIGATION_DATA)):
            nav_data.append(np.array(acq.data))
            nav_tstamp.append(acq.acquisition_time_stamp)
            nav_indices.append(count-1)
        else:
            if not (acq.isFlagSet(mrd.ACQ_IS_RTFEEDBACK_DATA or mrd.ACQ_IS_HPFEEDBACK_DATA or mrd.ACQ_IS_NAVIGATION_DATA)):
                if useDC:
                    if(acq.idx.kspace_encode_step_2 == int(mrd_header.encoding[0].encodedSpace.matrixSize.z/2)):
                        nav_data.append(np.array(acq.data[:,0:10]))
                        nav_tstamp.append(acq.acquisition_time_stamp)
                        nav_indices.append(count-1)    
                navangles.append(180*cmath.phase(complex(acq.traj[25,0],acq.traj[25,1]))/math.pi)
                acq_tstamp.append(acq.acquisition_time_stamp)
                data_indices.append(count-1)
                kencode_step.append(acq.idx.kspace_encode_step_1)
                if(len(acquisition_0)<1):
                    acquisition_0.append(acq)
                connection.send(acq)

    eprint("Total Acq:",count)
    et = time.time()
    # # get the execution time
    elapsed_time = et - st
    eprint('Execution time:', elapsed_time, 'seconds')
    traj_angles = cp.asarray(navangles)
    temp_indices = cp.asarray(nav_indices)
    temp_indices = temp_indices[1:None]-1
    kencode_step = cp.asarray(kencode_step)

    kstep_nav   = cp.concatenate((cp.asarray([kencode_step[0]]),kencode_step[temp_indices]),axis=0)
    angles_sorted = cp.zeros(cp.unique(traj_angles).shape)
    angles_sorted[kencode_step] = traj_angles
    navangles    = angles_sorted[kstep_nav]
    
    nav_data_copy   = nav_data.copy()
    nav_tstamp_copy = nav_tstamp.copy()
    nav_angles_copy = navangles.copy()
    acq_tstamp_copy = acq_tstamp.copy()


    st = time.time()   
    respiratory_waveform, samplingTime = estimateGatingSignal(nav_data_copy,nav_tstamp_copy,nav_angles_copy,acq_tstamp_copy) 
    cp.cuda.runtime.deviceSynchronize()

    et = time.time()
    elapsed_time = et - st
    eprint('Execution time (GatingSignal):', elapsed_time, 'seconds')
    st = time.time()   

    filename='/tmp/shelve.out'
    # my_shelf = shelve.open(filename,'n') # 'n' for new

    # for key in dir():
    #     try:
    #         my_shelf[key] = globals()[key]
    #     except TypeError:
    #         #
    #         # __builtins__, my_shelf, and imported modules can not be shelved.
    #         #
    #         eprint('ERROR shelving: {0}'.format(key))
    # my_shelf.close()

    #binning(selectedSig, timestamp, acceptancePercent, bidirectional, do_stable_binning, evenBins, numBins):
    acceptedTimes = binning(respiratory_waveform,nav_tstamp,40,bidirectional, do_stable_binning, evenbins, numBins )
    

    idx_to_send = []
    maxSize = 0 
    for ii in range(0,len(acceptedTimes)):
        try:
            temp = get_idx_to_send2(acq_tstamp,acceptedTimes[ii], samplingTime)
        except:
             eprint("GPUs couldnt do idx to send moving to cpu")
        finally:
             temp = get_idx_to_send2_np(acq_tstamp,acceptedTimes[ii], samplingTime)
      #  temp = get_idx_to_send2(acq_tstamp,acceptedTimes[ii], samplingTime) # Very expensive
        eprint(ii)
        eprint(acceptedTimes[ii].shape)
      #  temp = get_idx_to_send2_np(acq_tstamp,acceptedTimes[ii], samplingTime)
    
        idx_to_send.append(np.concatenate(([np.array(temp.shape[0])],temp)))
        if(idx_to_send[ii].shape[0] > maxSize):
            maxSize = idx_to_send[ii].shape[0]
    
            
    imageSize = pow(2,math.ceil(math.log2(math.sqrt(maxSize))))
    
    for ii in range(0,len(acceptedTimes)):
        data = np.zeros((imageSize*imageSize))
        data[range(0,len(idx_to_send[ii]))] = idx_to_send[ii].squeeze()
        data = np.reshape(data,(imageSize,imageSize))
        image = create_ismrmrd_image(data, acquisition_0[0], field_of_view, ii)
        connection.send(image)
        
        
if __name__ == "__main__":
    gadgetron.external.listen(20000,BinningGadget)
    
    
    
    
    