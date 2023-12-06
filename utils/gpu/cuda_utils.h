#pragma once

#include <gadgetron/cuNDArray.h>

namespace nhlbi_toolbox
{
    namespace cuda_utils
    {
        template <typename T>
        Gadgetron::cuNDArray<T> cuexp(Gadgetron::cuNDArray<T> &x);

        template <typename T>
        Gadgetron::cuNDArray<T> cucos(Gadgetron::cuNDArray<T> &x);

        template <typename T>
        Gadgetron::cuNDArray<T> cusin(Gadgetron::cuNDArray<T> &x);

    } // namespace cuda_util
} // namespace nhlbi_toolbox

namespace Gadgetron
{
    /**
   * @brief Construct a complex array from a real array and imag array.
   * @param[in] x real array, y imag array.
   * @return A new complex array containing the input array in the real component and imag elements in the imaginary component.
   */
  template<class T> boost::shared_ptr< cuNDArray<T> > cureal_imag_to_complex( const cuNDArray<typename realType<T>::Type> *x, const cuNDArray<typename realType<T>::Type> *y );

     /**
 * @brief Construct a complex array from a imag array.
 * @param[in] x Input array.
 * @return A new complex array containing the input array in the imag component and zeros in the imaginary component.
 */

template <class T> boost::shared_ptr<cuNDArray<T>> imag_to_complex(const cuNDArray<typename realType<T>::Type>* x);

}