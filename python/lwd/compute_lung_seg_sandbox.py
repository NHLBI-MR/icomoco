"""
Lung MRI segmentation.
"""

import numpy as np
#import collections
#import os
import sys
#import math
import time
#import random
#import logging
#import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
NoneType = type(None)

import torch
import torchvision
#import scipy.io as sio 
import scipy
from scipy.sparse import coo_matrix, vstack
from skimage import morphology 
from skimage.morphology import cube, binary_erosion, binary_dilation
from skimage.transform import resize
import gadgetron
import ismrmrd as mrd



def lung_segmentation(data, resolution, display=False):

     def prob_thresh(probs, device, p_thresh=0.5, params=None):
        cpu_device = torch.device('cpu')
 
        probs = probs.to(device=cpu_device)
 
        RO = probs.shape[0]
        E1 = probs.shape[1]
     
        number_of_blobs = float("inf")
        blobs = np.zeros((RO,E1))
    
        mask = (probs > p_thresh).float()
     
        return mask
     
     def cpad_2d(data, RO, E1):
         '''
         data: [dRO, dE1, N], padded it round center to [RO, E1, N]
         return value: (s_ro, s_e1), starting point of data in padded array
         '''
 
         dRO, dE1, N = data.shape
 
         s_ro = int((RO-dRO)/2)
         s_e1 = int((E1-dE1)/2)
 
         #print(data.shape, RO, E1, s_ro, s_e1)
         if(s_ro>=0):
             data_padded = np.zeros((RO, dE1, N))
             if(dRO>=RO):
                 data_padded = data[s_ro:s_ro+RO,:,:]
             else:
                 data_padded[s_ro:s_ro+dRO,:,:] = data
             data = data_padded
         else:
             data_padded = np.zeros((RO, dE1, N))
             if(dRO+s_ro+s_ro>RO):
                 data_padded = data[-s_ro:(dRO+s_ro-1),:,:]
             else:
                 data_padded = data[-s_ro:(dRO+s_ro),:,:]
             data = data_padded
 
#         print(data.shape)
 
         if(s_e1>=0):
             data_padded = np.zeros((RO, E1, N))
             if(dE1>=E1):
                 data_padded = data[:,s_e1:s_e1+E1,:]
             else:
                 data_padded[:,s_e1:s_e1+dE1,:] = data
             data = data_padded
         else:
             data_padded = np.zeros((RO, E1, N))
             if(dE1+s_e1+s_e1>E1):
                 data_padded = data[:,-s_e1:(dE1+s_e1-1),:]
             else:
                 data_padded = data[:,-s_e1:(dE1+s_e1),:]
             data = data_padded
 
         return data_padded, s_ro, s_e1
 

     # takes in data in the form of a numpy array [RO E1 N], and returns masks as a numpy array of same dimension
     device = 'cpu'
     try:
        model_file = "/opt/conda/envs/gadgetron/share/gadgetron/python/best_lung_seg_model_2022.pkl"
        model = torch.load(model_file)
     except:
        model_file = "/opt/package/share/gadgetron/python/best_lung_seg_model_2022.pkl"
        model = torch.load(model_file)
     # upsample to 1.5 mm in-plane resolution    
     data_orig=data
     upsampling_factor = (np.round(resolution / 1.5).astype(int))
     data = scipy.ndimage.zoom(data, (upsampling_factor, upsampling_factor, 1), order=3)

     
     data = np.transpose(data, (2,0,1))
     N, orig_RO, orig_E1 = data.shape

     #print(orig_RO, orig_E1)
     RO = 384
     E1 = 384
 
     if torch.cuda.is_available():
         device = torch.device('cuda')
     else:
         device = torch.device('cpu')
 
     print("Lung segmentation, device is ", device, file=sys.stderr)
     
     data_normalized = np.zeros((N, RO, E1), dtype='float32')
     
     #NORMALIZE IMAGE BETWEEN 0-1
     for n in range(N):
         data2D, s_ro, s_e1 = cpad_2d(np.expand_dims(data[n,:,:], axis=2), RO, E1)
         data2D = data2D.squeeze()
         if np.max(data2D) != 0:
             data2D = data2D / np.max(data2D)
         data_normalized[n,:,:] = data2D
 
     im = np.expand_dims(data_normalized, axis=1)
     img_rs = torch.from_numpy(im.astype(np.float32)).to(device=device)
          
     model.to(device=device)  
     model.eval() 
   
     output = np.zeros((384, 384, N), np.float32)
     with torch.no_grad():
         t0 = time.time()
         scores = model(img_rs)

         probs = torch.sigmoid(scores)
         probs = probs.detach().cpu().float().numpy().astype(np.float32)
 
         # Resize output mask to required iutput size 
         output = np.zeros((384, 384, N), np.float32)
         
         #if display:
         #    fig, axes = plt.subplots(nrows=8, ncols=9, figsize=(50,50), sharex=True, sharey=True)
         
         for i in range(N):
             mask = prob_thresh(torch.from_numpy(probs[i,0,:,:]), 'cpu', p_thresh=0.5)
             output[:,:, i] = mask
 
             masked = np.ma.masked_where(mask == 0, mask)

         t1 = time.time()
 
         print("Mask computed in %f seconds" % (t1-t0), file=sys.stderr)
         output, _, _ = cpad_2d(output, orig_RO, orig_E1)
         #downsample to original image size
         output = scipy.ndimage.zoom(output, (1/upsampling_factor, 1/upsampling_factor, 1), order=3)
         output[output>=0.3]=1
         output[output<0.3]=0

         if display:
             fig, axes = plt.subplots(nrows=8, ncols=9, figsize=(50,50), sharex=True, sharey=True)

         for i in range(N):
            if display:
                    y = i//9
                    x = i%9
                    axes[y,x].imshow(data_orig[:,:,i], 'gray', clim=(0.0, np.max(data_orig)*0.4))
                    axes[y,x].grid(False)
                    axes[y,x].imshow(np.ma.masked_where(output[:,:,i] == 0, output[:,:,i]), 'flag', interpolation='none', alpha=0.2)
         if display:
             plt.show()

         return output


def coil_shading_correction(image, lungmask):
    def create_regularization_matrix_torch(nx, ny, ngrid):
        # Difference approximation in x
         
        i, j = torch.meshgrid(torch.arange(2,nx), torch.arange(1, ny+1))
        
        ind = j.reshape((-1, 1))+ ny * (i.reshape((-1, 1)) - 1)
        ind = ind-1
        len_ = len(ind)
         
        data = torch.cat((torch.ones((len_, 1)) * -1, torch.ones((len_, 1)) * 2, torch.ones((len_, 1)) * -1), dim=1)
        row = torch.kron(torch.ones((1,3)),ind)
        col= torch.cat((ind-ny, ind, ind+ny), dim=1)
         
        row=row.flatten().long()
        col=col.flatten().long()
        data=data.flatten()
        T1 = torch.zeros((ngrid,ngrid))
        T1[row,col]=data.flatten()

        # Difference approximation in y
        i, j = torch.meshgrid(torch.arange(1,nx+1), torch.arange(2, ny))

        ind = j.reshape((-1,1))+ ny * (i.reshape((-1,1)) - 1)
        ind = ind-1
        len_ = len(ind)
         
        data = torch.cat((torch.ones((len_, 1)) * -1, torch.ones((len_, 1)) * 2, torch.ones((len_, 1)) * -1), dim=1)
        row = torch.kron(torch.ones((1,3)),ind)
        col= torch.cat((ind-1, ind, ind+1), axis=1)
       
        row=row.flatten().long()
        col=col.flatten().long()
        data=data.flatten()
        T2 = torch.zeros((ngrid,ngrid))
        
        T2[row,col]=data.flatten()
        T_full=torch.vstack((T1, T2))
         
        return T_full

    def tikReg2D_torch(image, lambda_opt, T, nx, ny, ngrid):       
        # Extract non-zero elements and their indices
        image_orig=image
        b = image[image != 0]
        bind = np.flatnonzero(image)
        nb = len(b)

        A= np.zeros((nb,ngrid))
        row=np.arange(nb).flatten().astype(int)
        col=bind.flatten().astype(int)
        A[row,col]=np.ones(nb)

        A = torch.from_numpy(A.astype(np.float32)).to('cuda:0') 
        b = torch.from_numpy(b.astype(np.float32)).to('cuda:0') 
      
        # Append zeros to the rhs
        b = torch.cat((b, torch.zeros(T.shape[0]).to('cuda:0') ))
  
        # Solve the minimization problem using sparse matrices
        AT=torch.vstack((A, lambda_opt*T.to('cuda:0'))).to('cuda:0')
   
        A = AT.T @ AT
        b = AT.T @ b
        t22=time.time()
        X = torch.linalg.lstsq(A.to('cuda:0'),b.to('cuda:0')).solution
        t33=time.time()
        print('Solver: ', t33-t22)

        X = torch.reshape(X,(ny, nx))
       
        return X
  
    #Calculate a body mask using thresholding, excluding any lung tissue
    image_orig=image
    bodymask=np.zeros_like(image)
    mean_intensity = np.mean(image)
    bodymask[image>= mean_intensity] = 1
    bodymask[lungmask==1]=0
   
    #normalize image between 0 and 1
    im_in=np.divide(image-np.min(image),np.max(image)-np.min(image))
    im_in[np.isnan(im_in)]=0
    im_in[np.isinf(im_in)]=0
    im_in[bodymask==0]=0

    #calculate lambda in the middle slice
    midslice=round(im_in.shape[2]/2)
    lambda_opt= 40.75 
   
    # Create the Tikhonov regularization matrix T
    ny,nx,nz= image.shape
    ngrid=np.multiply(ny,nx)
    T = create_regularization_matrix_torch(nx, ny, ngrid)

    shading_map=torch.zeros(image.shape)
    slices = np.where(np.sum(np.sum(lungmask, axis=0), axis=0) > 0)[0]
    for loop in slices.astype(int):
    #for loop in range(im_in.shape[2]):
        shading_map[:,:,loop]=tikReg2D_torch(im_in[:,:,loop],lambda_opt, T, nx, ny, ngrid)

    shading_map = shading_map.cpu().numpy()
    im_norm=np.divide(image_orig.astype(float), shading_map.astype(float))
    im_norm[np.isnan(im_norm)]=0
    im_norm[np.isinf(im_norm)]=0

    return im_norm, shading_map, lambda_opt, bodymask

def body_mask(lungmask):
     #Calculate a body mask by expanding lung mask
     se = cube(8)
     void = binary_dilation(lungmask,se)

     se = cube(16)
     bodymask = binary_dilation(lungmask,se)

     bodymask[void == 1] = 0

     # Define the central slices
     central_slices = np.zeros_like(bodymask)
     central_slices[:,:,np.array(range(round(lungmask.shape[2]/2)-3,round(lungmask.shape[2]/2)+2))] = 1
     bodymask = np.multiply(bodymask,central_slices)
     
     return bodymask

def calc_lwd(image, lungmask, bodymask):
    LWD_map = 70*np.divide(image,np.median(image[bodymask==1])) #assuming 70% body water density
    LWD = np.mean(LWD_map[lungmask==1])
    return LWD_map, LWD

def create_ismrmrd_image(data, field_of_view, index):
    return mrd.image.Image.from_array(data, image_index=1, image_type=mrd.IMTYPE_MAGNITUDE, 
    field_of_view=(field_of_view.x, field_of_view.y, field_of_view.z), transpose=0, 
    repetition=0, image_series_index=index)

def receive_images(connection):
     #receive image code
     hdr=connection.header
     field_of_view = hdr.encoding[0].reconSpace.fieldOfView_mm
     matrixSize = hdr.encoding[0].reconSpace.matrixSize
     resolution = np.array(field_of_view.x/matrixSize.x)
     count = 0
     for acquisition in connection:
        count = count + 1
        print(count)
        
        im=np.copy(acquisition.data)
        im=1000*im/np.max(im)

        image=create_ismrmrd_image(im, field_of_view, count)
        connection.send(image)
        
        if count == 2:           
            im       = np.transpose(np.rot90(np.squeeze(im), 2), [1, 2, 0])
            lungmask = lung_segmentation(im, resolution, display=False)
            bodymask = body_mask(lungmask)
            im_norm, shading_map, lambda_opt, bodymask_norm = coil_shading_correction(im, lungmask)
            LWD_map, LWD = calc_lwd(im_norm, lungmask, bodymask)

            LWD_map = np.rot90(LWD_map,-1, (0,1))
            LWD_map = np.transpose(LWD_map[np.newaxis, :, :, :],[0, 3, 2, 1])
            LWD_map = create_ismrmrd_image(LWD_map, field_of_view, 3)
            connection.send(LWD_map)

            lungmask = 10*lungmask + bodymask
            lungmask = np.rot90(lungmask,-1, (0,1))
            lungmask = np.transpose(lungmask[np.newaxis, :, :, :],[0, 3, 2, 1])
            lungmask = create_ismrmrd_image(lungmask, field_of_view, 4)
            connection.send(lungmask)

            im_norm = np.rot90(im_norm,-1, (0,1))
            im_norm = np.transpose(im_norm[np.newaxis, :, :, :],[0, 3, 2, 1])
            im_norm =create_ismrmrd_image(im_norm, field_of_view, 5)
            connection.send(im_norm)
            
           

#if __name__ == "__main__":

     #gadgetron.external.listen(2000,receive_images)

     #display=True    
     #if display:
         #fig, axes = plt.subplots(nrows=8, ncols=9, figsize=(50,50), sharex=True, sharey=True)

         #for i in range(72):   
            #y = i//9
            #x = i%9
            #axes[y,x].imshow(im_norm[:,:,i], 'gray', clim=(0.0, np.max(im_norm)))
            #axes[y,x].grid(False)
            #axes[y,x].imshow(np.ma.masked_where(lungmask[:,:,i] == 0, lungmask[:,:,i]), 'flag', interpolation='none', alpha=0.2)
            #axes[y,x].imshow(np.ma.masked_where(bodymask[:,:,i] == 0, bodymask[:,:,i]), 'flag', interpolation='none', alpha=0.4)
            #axes[y,x].imshow(np.ma.masked_where(bodymask_norm[:,:,i] == 0, bodymask_norm[:,:,i]), 'flag', interpolation='none', alpha=0.8)
        
         #plt.show()