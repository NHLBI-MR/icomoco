// 
#include "gpuRegistration.cuh"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include <gadgetron/vector_td_utilities.h>
#include "cubicTex3D.cu"

#include "internal/bspline_kernel.cu"
#include "cubicPrefilter3D.cu"
#include "stdio.h"
#include <thrust/extrema.h>
#include "texture_indirect_functions.h"
#include "cuda_texture_types.h"
#include "cuda_utils.h"

using namespace Gadgetron;

__device__ float cubictex(cudaTextureObject_t tex, float3 coord)
{
    // shift the coordinate from [0,extent] to [-0.5, extent-0.5]
    const float3 coord_grid = coord - 0.5f;
    const float3 index = floor(coord_grid);
    const float3 fraction = coord_grid - index;
    float3 w0, w1, w2, w3;
    bspline_weights<float3>(fraction, w0, w1, w2, w3);

    const float3 g0 = w0 + w1;
    const float3 g1 = w2 + w3;
    const float3 h0 = (w1 / g0) - 0.5f + index; // h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
    const float3 h1 = (w3 / g1) + 1.5f + index; // h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

    // fetch the eight linear interpolations
    // weighting and fetching is interleaved for performance and stability reasons
    float tex000 = tex3D<float>(tex, h0.x, h0.y, h0.z);
    float tex100 = tex3D<float>(tex, h1.x, h0.y, h0.z);
    tex000 = g0.x * tex000 + g1.x * tex100; // weigh along the x-direction
    float tex010 = tex3D<float>(tex, h0.x, h1.y, h0.z);
    float tex110 = tex3D<float>(tex, h1.x, h1.y, h0.z);
    tex010 = g0.x * tex010 + g1.x * tex110; // weigh along the x-direction
    tex000 = g0.y * tex000 + g1.y * tex010; // weigh along the y-direction
    float tex001 = tex3D<float>(tex, h0.x, h0.y, h1.z);
    float tex101 = tex3D<float>(tex, h1.x, h0.y, h1.z);
    tex001 = g0.x * tex001 + g1.x * tex101; // weigh along the x-direction
    float tex011 = tex3D<float>(tex, h0.x, h1.y, h1.z);
    float tex111 = tex3D<float>(tex, h1.x, h1.y, h1.z);
    tex011 = g0.x * tex011 + g1.x * tex111; // weigh along the x-direction
    tex001 = g0.y * tex001 + g1.y * tex011; // weigh along the y-direction

    return (g0.z * tex000 + g1.z * tex001); // weigh along the z-direction
}
    // Simple transformation kernel
    __global__ static void  deform_imageKernel(float* output, const float* vector_field, cudaTextureObject_t texObj, int width,
                                       int height, int depth)
    {

        const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
        const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
        const int izo = blockDim.z * blockIdx.z + threadIdx.z;
        if (ixo < width && iyo < height && izo < depth)
        {

            const int idx = ixo + iyo * width + izo * width * height;
            const int elements = width * height * depth;
            float ux = vector_field[idx] + (0.5f + ixo);
            float uy = vector_field[idx + elements] + (0.5f + iyo);
            float uz = vector_field[idx + 2 * elements] + (0.5f + izo);
            // printf("ux: %0.2f, vector_field[%d]: %0.2f, ixo: %d \n",ux,idx,vector_field[idx],ixo);
            // printf("output: %d, input: %d",tex3D<float>(texObj, ux, uy, uz),
            // tex3D<float>(texObj, (0.5f + ixo), (float)(0.5f + iyo), (float)(0.5f + izo)));
            
            float3 cord;
            cord.x = ux, cord.y = uy, cord.z = uz; //(ux,uy,uz);
            output[idx] = cubictex(texObj, cord);
        }
    }
    // Simple transformation kernel
    __global__ static void  deform_imageKernel(float* output, const float* vector_field,
                                       int width, int height,
                                       int depth)
    {

        const int ixo = blockDim.x * blockIdx.x + threadIdx.x;
        const int iyo = blockDim.y * blockIdx.y + threadIdx.y;
        const int izo = blockDim.z * blockIdx.z + threadIdx.z;
        printf("ixo: %d \n", ixo);

        if (ixo < width && iyo < height && izo < depth)
        {

            const int idx = ixo + iyo * width + izo * width * height;
            const int elements = width * height * depth;
            float ux = vector_field[idx] + (0.5f + ixo);
            float uy = vector_field[idx + elements] + (0.5f + iyo);
            float uz = vector_field[idx + 2 * elements] + (0.5f + izo);

            printf("vector_field[%d]: %0.2f, ixo: %d \n", idx, vector_field[idx], ixo);

            //   printf("output: %d, input: %d", cubicTex3D(texObj, ux, uy, uz),
            //        cubicTex3D(texObj, (0.5f + ixo), (float)(0.5f + iyo), (float)(0.5f + izo)));
            //    output[idx] = cubicTex3D(texObj, ux, uy, uz);
        }
    }
    //--------------------------------------------------------------------------
    // Declare the typecast CUDA kernels
    //--------------------------------------------------------------------------
    template <class T>
    __device__ float Multiplier() { return 1.0f; }
    template <>
    __device__ float Multiplier<uchar>() { return 255.0f; }
    template <>
    __device__ float Multiplier<schar>() { return 127.0f; }
    template <>
    __device__ float Multiplier<ushort>() { return 65535.0f; }
    template <>
    __device__ float Multiplier<short>() { return 32767.0f; }
    template <class T>
    __global__ void CopyCast(uchar *destination, const T *source, uint pitch, uint width)
    {
        uint2 index =
            make_uint2(__umul24(blockIdx.x, blockDim.x) + threadIdx.x, __umul24(blockIdx.y, blockDim.y) + threadIdx.y);

        float *dest = (float *)(destination + index.y * pitch) + index.x;
        *dest = (1.0f / Multiplier<T>()) * (float)(source[index.y * width + index.x]);
    }
    //--------------------------------------------------------------------------
    // Declare the typecast templated function
    // This function can be called directly in C++ programs
    //--------------------------------------------------------------------------

    // //! Allocate GPU memory and copy a voxel volume from CPU to GPU memory
    // //! and cast it to the normalized floating point format
    // //! @return the pointer to the GPU copy of the voxel volume
    // //! @param host  pointer to the voxel volume in CPU (host) memory
    // //! @param width   volume width in number of voxels
    // //! @param height  volume height in number of voxels
    // //! @param depth   volume depth in number of voxels
    // template <class T>
    // extern cudaPitchedPtr CastVolumeHostToDevice(const T *host, uint width, uint height, uint depth)
    // {
    //     cudaPitchedPtr device = {0};
    //     const cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
    //     cudaMalloc3D(&device, extent);
    //     const size_t pitchedBytesPerSlice = device.pitch * device.ysize;

    //     T *temp = 0;
    //     const uint voxelsPerSlice = width * height;
    //     const size_t nrOfBytesTemp = voxelsPerSlice * sizeof(T);
    //     cudaMalloc((void **)&temp, nrOfBytesTemp);

    //     uint dimX = min(PowTwoDivider(width), 64);
    //     dim3 dimBlock(dimX, min(PowTwoDivider(height), 512 / dimX));
    //     dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
    //     size_t offsetHost = 0;
    //     size_t offsetDevice = 0;

    //     for (uint slice = 0; slice < depth; slice++)
    //     {
    //         cudaMemcpy(temp, host + offsetHost, nrOfBytesTemp, cudaMemcpyHostToDevice);
    //         CopyCast<T><<<dimGrid, dimBlock>>>((uchar *)device.ptr + offsetDevice, temp, (uint)device.pitch, width);
    //         offsetHost += voxelsPerSlice;
    //         offsetDevice += pitchedBytesPerSlice;
    //     }

    //     cudaFree(temp); // free the temp GPU volume
    //     return device;
    // }
    // //! Copy a voxel volume from a pitched pointer to a texture
    // //! @param tex      [output]  pointer to the texture
    // //! @param texArray [output]  pointer to the texArray
    // //! @param volume   [input]   pointer to the the pitched voxel volume
    // //! @param extent   [input]   size (width, height, depth) of the voxel volume
    // //! @param onDevice [input]   boolean to indicate whether the voxel volume resides in GPU (true) or CPU (false) memory
    // //! @note When the texArray is not yet allocated, this function will allocate it
    // template <class T, enum cudaTextureReadMode mode>
    // void CreateTextureFromVolume(texture<T, 3, mode> *tex, cudaArray **texArray, const cudaPitchedPtr volume,
    //                              cudaExtent extent, bool onDevice)
    // {

    //     cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    //     if (*texArray == 0)
    //         cudaMalloc3DArray(texArray, &channelDesc, extent);
    //     // copy data to 3D array
    //     cudaMemcpy3DParms p = {0};
    //     p.extent = extent;
    //     p.srcPtr = volume;
    //     p.dstArray = *texArray;
    //     p.kind = onDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    //     cudaMemcpy3D(&p);
    //     // bind array to 3D texture
    //     cudaBindTextureToArray(*tex, *texArray, channelDesc);
    //     tex->normalized = false; // access with absolute texture coordinates
    //     tex->filterMode = cudaFilterModeLinear;
    // }

    cuNDArray<complext<float>> gpuRegistration::deform_image(cuNDArray<complext<float>> *image,
                                                             cuNDArray<float> vector_field)
    {

        cuNDArray<float> mir = *real(image);
        cuNDArray<float> mii = *imag(image);

        // clear(&output);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaExtent extent;
        extent.width = image->get_size(0);
        extent.height = image->get_size(1);
        extent.depth = image->get_size(2);

        cudaMemcpy3DParms cpy_params_r = {0};
        cpy_params_r.kind = cudaMemcpyDeviceToDevice;
        cpy_params_r.extent = extent;

        cudaArray *image_array_r;
        cudaMalloc3DArray(&image_array_r, &channelDesc, extent);
        cpy_params_r.dstArray = image_array_r;
        cpy_params_r.srcPtr =
            make_cudaPitchedPtr((void *)mir.data(), extent.width * sizeof(float), extent.width, extent.height);

        cudaMemcpy3DParms cpy_params_i = {0};
        channelDesc = cudaCreateChannelDesc<float>();

        cudaArray *image_array_i;
        cudaMalloc3DArray(&image_array_i, &channelDesc, extent);
        cpy_params_i.kind = cudaMemcpyDeviceToDevice;
        cpy_params_i.extent = extent;
        cpy_params_i.dstArray = image_array_i;
        cpy_params_i.srcPtr =
            make_cudaPitchedPtr((void *)mii.data(), extent.width * sizeof(float), extent.width, extent.height);

        // texture<float, 3, cudaReadModeElementType> coeffsr; // 3D texture
        // texture<float, 3, cudaReadModeElementType> coeffsi; // 3D texture

        cudaExtent volumeExtent = make_cudaExtent(extent.width, extent.height, extent.depth);

        // cudaBindTextureToArray(coeffsr, image_array_r, channelDesc);
        // cudaBindTextureToArray(coeffsi, image_array_i, channelDesc);

        CubicBSplinePrefilter3DTimer((float *)cpy_params_r.srcPtr.ptr, (uint)cpy_params_r.srcPtr.pitch, extent.width,
                                     extent.height, extent.depth);
        CubicBSplinePrefilter3DTimer((float *)cpy_params_i.srcPtr.ptr, (uint)cpy_params_i.srcPtr.pitch, extent.width,
                                     extent.height, extent.depth);

        cudaMemcpy3D(&cpy_params_r);
        cudaMemcpy3D(&cpy_params_i);
        // CreateTextureFromVolume<float,cudaReadModeElementType>(&coeffsr, &coeffArrayr, cpy_params_r.srcPtr, volumeExtent,
        // true); CreateTextureFromVolume<float,cudaReadModeElementType>(&coeffsi, &coeffArrayi, cpy_params_i.srcPtr,
        // volumeExtent, true);
        // create the b-spline coefficients texture
        // CreateTextureFromVolume(&coeffs, &coeffArray, cpy_params.srcPtr, volumeExtent, true);
        // cudaDestroyTextureObject(texObj);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = image_array_r;

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        cudaTextureObject_t texObj_r = 0;
        cudaTextureObject_t texObj_i = 0;
        cudaCreateTextureObject(&texObj_r, &resDesc, &texDesc, NULL);
        resDesc.res.array.array = image_array_i;
        cudaCreateTextureObject(&texObj_i, &resDesc, &texDesc, NULL);

        // coeffsr.normalized = false; // access with absolute texture coordinates
        // coeffsr.filterMode = cudaFilterModeLinear;
        // coeffsi.normalized = false; // access with absolute texture coordinates
        // coeffsi.filterMode = cudaFilterModeLinear;
        // cudaPitchedPtr bsplineCoeffs = make_cudaPitchedPtr((void*)mii.data(), extent.width * sizeof(float), extent.width,
        // extent.height);

        // create the b-spline coefficients texture
        // cudaArray* coeffArray = 0;
        // texture<float, 3, cudaReadModeElementType> coeffs; // 3D texture
        // cudaExtent volumeExtent = make_cudaExtent(extent.width, extent.height, extent.depth);
        // CreateTextureFromVolume(&coeffs, &coeffArray, cpy_params.srcPtr, volumeExtent, true);
        // cudaFree(bsplineCoeffs.ptr);  //they are now in the coeffs texture, we do not need this anymore

        dim3 threads(8, 8, 8);

        dim3 grid((extent.width + threads.x - 1) / threads.x, (extent.height + threads.y - 1) / threads.y,
                  (extent.depth + threads.z - 1) / threads.z);

        cuNDArray<float> outputr(image->get_dimensions());
        cuNDArray<float> outputi(image->get_dimensions());
        
        deform_imageKernel<<<grid, threads>>>(outputr.data(), vector_field.data(), texObj_r, extent.width, extent.height, extent.depth);

        // deform_imageKernel<float>
        //     <<<grid, threads>>>(outputr.data(), vector_field.data(), coeffs, extent.width, extent.height, extent.depth);

        // cudaFreeArray(image_array_r);

        cudaDeviceSynchronize();
        
        deform_imageKernel<<<grid, threads>>>(outputi.data(), vector_field.data(), texObj_i, extent.width, extent.height, extent.depth);

        cudaDeviceSynchronize();
        // cudaFree(&coeffs);
        // Free device memory

        auto output = *cureal_imag_to_complex<float_complext>(&outputr, &outputi);

        cudaFreeArray(image_array_r);
        cudaFreeArray(image_array_i);
        outputr.clear();
        outputi.clear();
        mir.clear();
        mii.clear();

        return output;
    }

