#pragma once
#include <gadgetron/hoNDArray.h>
#include <gadgetron/cuNDArray.h>
#include <ismrmrd/ismrmrd.h>
#include <vector>
#include <armadillo>
#include <reconParams.h>

namespace Gadgetron {
template <template<class> class ARRAY,typename T = float_complext,unsigned int D = 3>
    struct SpiralReconBuffer { 
        ARRAY<T> cuData;
        std::vector<ARRAY<Gadgetron::floatd3>> trajVec;
        std::vector<ARRAY<float>> dcwVec;
        Gadgetron::cuNDArray<Gadgetron::float_complext> csm;
        ARRAY<T> cweights;
        std::vector<ARRAY<float>> sctVector;
        arma::fvec freq_bins; 
        reconParams recon_params;
        ISMRMRD::AcquisitionHeader acqhdr;
    };
}