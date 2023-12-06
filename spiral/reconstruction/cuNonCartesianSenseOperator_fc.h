/** \file cuNonCartesianSenseOperator.h
    \brief Non-Cartesian Sense operator, GPU based.
*/

#pragma once

#include <gadgetron/cuNonCartesianSenseOperator.h>
#include  <gadgetron/cuNFFT.h>
#include  <gadgetron/hoArmadillo.h>
#include "noncartesian_reconstruction_fc.h"

namespace Gadgetron{

  template<class REAL, unsigned int D> class EXPORTGPUPMRI cuNonCartesianSenseOperator_fc : public cuNonCartesianSenseOperator<REAL,D>
  {
  
  public:
  
    typedef typename uint64d<D>::Type _uint64d;
    typedef typename reald<REAL,D>::Type _reald;

    cuNonCartesianSenseOperator_fc(ConvolutionType conv = ConvolutionType::STANDARD):cuNonCartesianSenseOperator<REAL,D>(conv){};
     ~cuNonCartesianSenseOperator_fc() {}
    
     void mult_M( cuNDArray< complext<REAL> >* in, cuNDArray< complext<REAL> >* out, bool accumulate = false );
     void mult_MH( cuNDArray< complext<REAL> >* in, cuNDArray< complext<REAL> >* out, bool accumulate = false );

     void preprocess( cuNDArray<vector_td<REAL,D>> *trajectory );
     void set_fbins( arma::fvec fbins );
     void set_combination_weights( cuNDArray<float_complext> *combinationWeights );
     void set_scaled_time( cuNDArray<REAL> *scaled_time );
     void set_dofield_correction( bool flag);
     void set_recon_params(reconParams rp);

  
  protected:
    cuNDArray<vector_td<REAL,D>> *trajectory_;
    cuNDArray<float_complext> *combinationWeights_;
    cuNDArray<REAL> *scaled_time_;
    arma::fvec fbins_;
    bool doConcomitantFieldCorraction_ = false;
    reconParams recon_params_;
    
  };
  
}
