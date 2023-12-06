/** \file cuNonCartesianSenseOperator.h
    \brief Non-Cartesian Sense operator, GPU based.
*/

#pragma once

#include "cuNonCartesianSenseOperator_fc.h"
#include "cuNonCartesianTSenseOperator.h"
#include "cuNFFT.h"

namespace Gadgetron
{

  template <class REAL, unsigned int D>
  class EXPORTGPUPMRI cuNonCartesianTSenseOperator_fc : public cuNonCartesianTSenseOperator<REAL, D>
  {

  public:
    typedef typename uint64d<D>::Type _uint64d;
    typedef typename reald<REAL, D>::Type _reald;

    cuNonCartesianTSenseOperator_fc(ConvolutionType conv = ConvolutionType::STANDARD) : cuNonCartesianTSenseOperator<REAL, D>(conv){};
    virtual ~cuNonCartesianTSenseOperator_fc() {}

    virtual void mult_M(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out, bool accumulate = false);
    virtual void mult_MH(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out, bool accumulate = false);

    virtual void preprocess(std::vector<cuNDArray<vector_td<REAL, D>>> &trajectory);
    void set_fbins(arma::fvec fbins);
    void set_combination_weights(cuNDArray<float_complext> *combinationWeights);
    void set_scaled_time(std::vector<cuNDArray<REAL>> &scaled_time);
    void set_dofield_correction(bool flag);
    void set_recon_params(reconParams rp);

  protected:
    ConvolutionType convolutionType;
    bool is_preprocessed_;
    std::vector<cuNDArray<vector_td<REAL, D>>> trajectory_;
    cuNDArray<float_complext> *combinationWeights_;
    std::vector<cuNDArray<REAL>> scaled_time_;
    arma::fvec fbins_;
    bool doConcomitantFieldCorraction_ = false;
    reconParams recon_params_;
   std::vector<std::vector<nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc>> nrfc_vector; //gpus<it>
    std::vector<int> eligibleGPUs;

  };

}
