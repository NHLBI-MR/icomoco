
#include "cuda_utils.h"
#include <gadgetron/cuNDArray_elemwise.h>
#include <gadgetron/cuNDArray_operators.h>
#include <gadgetron/cuNDArray_blas.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/complext.h>

#include <complex>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/complex.h>
#include <math.h>
namespace nhlbi_toolbox
{
  namespace cuda_utils
  {
    using namespace Gadgetron;
    template <typename T>
    struct cuNDA_exp : public thrust::unary_function<T, T>
    {
      __device__ T operator()(const T &x) const { return exp(x); }
    };

    template <typename T>
    struct cuNDA_cos : public thrust::unary_function<T, T>
    {
      __device__ T operator()(const T &x) const { return cos(x); }
    };

    template <typename T>
    struct cuNDA_sin : public thrust::unary_function<T, T>
    {
      __device__ T operator()(const T &x) const { return sin(x); }
    };

    template <typename T>
    cuNDArray<T> cuexp(cuNDArray<T> &x)
    {
      cuNDArray<T> results(x.get_dimensions());
      thrust::transform(x.begin(), x.end(), results.begin(), cuNDA_exp<T>());
      return results;
    }

    template <typename T>
    cuNDArray<T> cucos(cuNDArray<T> &x)
    {
      cuNDArray<T> results(x.get_dimensions());
      thrust::transform(x.begin(), x.end(), results.begin(), cuNDA_cos<T>());
      return results;
    }

    template <typename T>
    cuNDArray<T> cusin(cuNDArray<T> &x)
    {
      cuNDArray<T> results(x.get_dimensions());
      thrust::transform(x.begin(), x.end(), results.begin(), cuNDA_sin<T>());
      return results;
    }

  }
}
using namespace Gadgetron;

namespace Gadgetron
{
    template <typename T> struct cuNDA_real_imag_to_complex : public thrust::binary_function<typename realType<T>::Type,typename realType<T>::Type,T>
{
  __device__ T operator()(const typename realType<T>::Type &x,const typename realType<T>::Type &y) const {return T(x,y);}
};

  template <class T>
  boost::shared_ptr<cuNDArray<T>>
  Gadgetron::cureal_imag_to_complex(const cuNDArray<typename realType<T>::Type> *x, const cuNDArray<typename realType<T>::Type> *y)
  {
    if (x == 0x0)
      throw std::runtime_error("Gadgetron::real_imag_to_complex(): Invalid input array");

    boost::shared_ptr<cuNDArray<T>> result(new cuNDArray<T>());
    result->create(x->get_dimensions());
    thrust::device_ptr<T> resPtr = result->get_device_ptr();
    thrust::device_ptr<typename realType<T>::Type> xPtr = x->get_device_ptr();
    thrust::device_ptr<typename realType<T>::Type> yPtr = y->get_device_ptr();
    thrust::transform(xPtr, xPtr + x->get_number_of_elements(), yPtr, resPtr, cuNDA_real_imag_to_complex<T>());
    return result;
  }

  template <typename T> struct cuNDA_imag_to_complex : public thrust::unary_function<typename realType<T>::Type,T>
{
  __device__ float_complext operator()(const typename realType<T>::Type &x) const {return float_complext(0.0f,x);}
};

template<class T> boost::shared_ptr< cuNDArray<T> > 
Gadgetron::imag_to_complex( const cuNDArray<typename realType<T>::Type> *x )
{
  if( x == 0x0 )
    throw std::runtime_error("Gadgetron::imag_to_complex(): Invalid input array");
   
  boost::shared_ptr< cuNDArray<T> > result(new cuNDArray<T>());
  result->create(x->get_dimensions());
  thrust::device_ptr<T> resPtr = result->get_device_ptr();
  thrust::device_ptr<typename realType<T>::Type> xPtr = x->get_device_ptr();
  thrust::transform(xPtr,xPtr+x->get_number_of_elements(),resPtr,cuNDA_imag_to_complex<T>());
  return result;
}


}

template Gadgetron::cuNDArray<float_complext> nhlbi_toolbox::cuda_utils::cuexp<float_complext>(Gadgetron::cuNDArray<float_complext> &);
template Gadgetron::cuNDArray<float> nhlbi_toolbox::cuda_utils::cucos<float>(Gadgetron::cuNDArray<float> &);
template Gadgetron::cuNDArray<float> nhlbi_toolbox::cuda_utils::cusin<float>(Gadgetron::cuNDArray<float> &);
template boost::shared_ptr< cuNDArray<float_complext> > Gadgetron::cureal_imag_to_complex<float_complext>( const cuNDArray<float>*, const cuNDArray<float>* );
template boost::shared_ptr< cuNDArray<float_complext> > Gadgetron::imag_to_complex<float_complext>( const cuNDArray<float>* );
