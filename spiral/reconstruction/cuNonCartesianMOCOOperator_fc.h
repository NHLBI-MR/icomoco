/** \file cuNonCartesianSenseOperator.h
    \brief Non-Cartesian Sense operator, GPU based.
*/

#pragma once

#include "cuNonCartesianSenseOperator_fc.h"
#include "cuNonCartesianMOCOOperator.h"
#include "cuNonCartesianTSenseOperator_fc.h"
#include "cuNFFT.h"
#include <gadgetron/vector_td_utilities.h>
namespace Gadgetron
{

  template <class REAL, unsigned int D>
  class EXPORTGPUPMRI cuNonCartesianMOCOOperator_fc : public cuNonCartesianTSenseOperator_fc<REAL, D>
  {

  public:
    typedef typename uint64d<D>::Type _uint64d;
    typedef typename reald<REAL, D>::Type _reald;

    cuNonCartesianMOCOOperator_fc(ConvolutionType conv = ConvolutionType::STANDARD) : cuNonCartesianTSenseOperator_fc<REAL, D>(conv){};
    virtual ~cuNonCartesianMOCOOperator_fc() {}

    virtual void mult_M(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out, bool accumulate = false);
    virtual void mult_MH(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out, bool accumulate = false);

    virtual void preprocess(std::vector<cuNDArray<vector_td<REAL, D>>> &trajectory);
    void set_fbins(arma::fvec fbins);
    void set_combination_weights(cuNDArray<float_complext> *combinationWeights);
    void set_scaled_time(std::vector<cuNDArray<REAL>> &scaled_time);
    void set_dofield_correction(bool flag);
    void set_recon_params(reconParams rp);
    void set_forward_deformation (std::vector<cuNDArray<REAL>> forward_deformation);
    void set_backward_deformation(std::vector<cuNDArray<REAL>> backward_deformation);

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
    std::vector<cuNDArray<REAL> > forward_deformation_;
    std::vector<cuNDArray<REAL> > backward_deformation_;
  
  };

}
