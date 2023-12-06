import opticalflow3D
import cupy as cp
import cupyx
from skimage.transform import warp
import numpy as np
import matplotlib.pyplot as plt


def register_images(input_images,ref_index):
    #input images 3D nimages, nx,ny,nz
    images = cp.asarray(input_images)
    output = cp.zeros(images.shape,cp.complex64)
    deformation_fields = np.zeros([images.shape[0],3,images.shape[1],images.shape[2],images.shape[3]])
    nimages, nr, nc, nz = images.shape
    
    ref_image = images[ref_index,...].squeeze()
    
    for ind in range(0,nimages):
        
        mov_image = images[ind,...].squeeze()
        farneback = opticalflow3D.Farneback3D(iters = 5,
                                              num_levels = 5,
                                              scale = 0.5,
                                              filter_size= 9,
                                              presmoothing = None,
                                              filter_type="gaussian",
                                              sigma_k = 0.05)
        
        output_vx, output_vy, output_vz, x = farneback.calculate_flow(0.05*cp.abs(ref_image/cp.max(ref_image.ravel())), 0.05*cp.abs(mov_image/cp.max(mov_image.ravel())), 
                                                                total_vol=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                                                                sub_volume=(ref_image.shape[0], ref_image.shape[1], ref_image.shape[2]),
                                                                overlap=(ref_image.shape[0], ref_image.shape[1],  ref_image.shape[2]),
                                                                threadsperblock=(8, 8, 8),
                                                                )
        
        
        

        row_coords, col_coords, slice_coords = np.meshgrid(np.arange(nr), np.arange(nc), np.arange(nz),
                                            indexing='ij')

        rcomp = cp.real(mov_image)
        icomp = cp.imag(mov_image)
        
        #Apply deformation to real and imaginary
        output[ind,:,:,:] = cupyx.scipy.ndimage.map_coordinates(rcomp,
                                            cp.array([row_coords + output_vx, col_coords + output_vy, slice_coords + output_vz]),
                                                                    mode="wrap") + 1j * cupyx.scipy.ndimage.map_coordinates(icomp,
                                            cp.array([row_coords + output_vx, col_coords + output_vy, slice_coords + output_vz]),
                                                                    mode="wrap")
                                                                    
        deformation_fields[ind,...] = np.concatenate([np.expand_dims(output_vy,0),np.expand_dims(output_vx,0),np.expand_dims(output_vz,0)],axis=0)
        deformation_fields[ind,...] = np.nan_to_num( deformation_fields[ind,...])

    display_registered_images(input_images[ref_index,...].squeeze(),
                            np.mean(input_images,axis=0).squeeze(),
                            np.mean(output,axis=0).squeeze(), 
                            8)
        
    return cp.asnumpy(output), deformation_fields
    

def display_registered_images(ref_image,mov_image,warped_image,slice_pos=0):
    
    nr, nc, nz = ref_image.shape
    seq_im = np.zeros((nr, nc, nz,  3))
    seq_im[..., 0] = np.abs(cp.asnumpy(mov_image))
    seq_im[..., 1] = np.abs(cp.asnumpy(ref_image))
    seq_im[..., 2] = np.abs(cp.asnumpy(ref_image))

    # build an RGB image with the registered sequence
    reg_im = np.zeros((nr, nc, nz, 3))
    reg_im[..., 0] = np.abs(cp.asnumpy(warped_image))
    reg_im[..., 1] = np.abs(cp.asnumpy(ref_image))
    reg_im[..., 2] = np.abs(cp.asnumpy(ref_image))

    # build an RGB image with the registered sequence
    target_im = np.zeros((nr, nc, nz, 3))
    target_im[..., 0] = np.abs(cp.asnumpy(ref_image))
    target_im[..., 1] = np.abs(cp.asnumpy(ref_image))
    target_im[..., 2] = np.abs(cp.asnumpy(ref_image))

    # build an RGB image with the registered sequence
    m_im = np.zeros((nr, nc, nz, 3))
    m_im[..., 0] = np.abs(cp.asnumpy(mov_image))
    m_im[..., 1] = np.abs(cp.asnumpy(mov_image))
    m_im[..., 2] = np.abs(cp.asnumpy(mov_image))
    
    # build an RGB image with the registered sequence
    warped_im = np.zeros((nr, nc, nz, 3))
    warped_im[..., 0] = np.abs(cp.asnumpy(warped_image))
    warped_im[..., 1] = np.abs(cp.asnumpy(warped_image))
    warped_im[..., 2] = np.abs(cp.asnumpy(warped_image))

    # --- Show the result

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(10, 20))

    ax0.imshow(np.flipud(4*seq_im[:,:,slice_pos,:]/np.max(seq_im[:,:,slice_pos,:].ravel())))
    ax0.set_title("Unregistered sequence")
    ax0.set_axis_off()

    ax1.imshow(np.flipud(4*reg_im[:,:,slice_pos,:]/np.max(reg_im[:,:,slice_pos,:].ravel())))
    ax1.set_title("Registered sequence")
    ax1.set_axis_off()

    ax2.imshow(np.flipud(4*target_im[:,:,slice_pos,:]/np.max(target_im[:,:,slice_pos,:].ravel())))
    ax2.set_title("Target")
    ax2.set_axis_off()

    ax3.imshow(np.flipud(4*warped_im[:,:,slice_pos,:]/np.max(warped_im[:,:,slice_pos,:].ravel())))
    ax3.set_title("Warped")
    ax3.set_axis_off()
    
    ax4.imshow(np.flipud(4*m_im[:,:,slice_pos,:]/np.max(m_im[:,:,slice_pos,:].ravel())))
    ax4.set_title("Warped")
    ax4.set_axis_off()

    fig.tight_layout()