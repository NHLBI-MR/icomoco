#pragma once

#include <gadgetron/cuNDArray.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/cuNDArray_elemwise.h>
#include <gadgetron/cuSenseOperator.h>
#include <gadgetron/cuNFFT.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/cuNDArray_elemwise.h>
#include <gadgetron/hoNDArray.h>


#include <gadgetron/hoNDArray_elemwise.h>
// class gpuRegistration
// {
//     public:

//     gpuRegistration() = default;
//     void applyDeformationbSpline(cuNDArray< complext<float> > *moving_image, cuNDArray<float>  transformation);
//     cuNDArray<complext<float>> deform_image( cuNDArray< complext<float> >* image,  cuNDArray<float> vector_field);
//     float cubictex(cudaTextureObject_t tex, float3 coord);
//     static void deform_imageKernel(float *output, const float *vector_field, cudaTextureObject_t texObj, int width,
//                                           int height, int depth);
//     static  void deform_imageKernel(float *output, const float *vector_field,
//     int width, int height,
//     int depth);
//     private:

// };
namespace Gadgetron{
    class gpuRegistration
    {
    public:
        gpuRegistration() = default;
        cuNDArray<complext<float>> deform_image(cuNDArray<complext<float>> *image, cuNDArray<float> vector_field);
    };
}

