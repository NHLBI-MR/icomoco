#pragma once

#include "noncartesian_reconstruction.h"
#include "reconParams.h"
#include <gadgetron/cuNDArray_elemwise.h>
using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        class noncartesian_reconstruction_fc : public noncartesian_reconstruction<3>
        {
        public:
            noncartesian_reconstruction_fc(reconParams recon_params) : noncartesian_reconstruction<3>(recon_params){};

            void reconstruct_todevice(cuNDArray<float_complext> &data, cuNDArray<float_complext> &images, cuNDArray<floatd3> &trajectory_in, cuNDArray<float> &dcf_in, cuNDArray<float_complext> &cweights_ho, cuNDArray<float> &scaled_time_in, arma::fvec fbins, int deviceNo, bool forward);
            void reconstruct_parallel(cuNDArray<float_complext> &data, cuNDArray<float_complext> &images, cuNDArray<floatd3> &trajectory_in, cuNDArray<float> &dcf_in, cuNDArray<float_complext> &cweights_ho, cuNDArray<float> &scaled_time_in, arma::fvec fbins, int deviceNo, bool forward);
            void reconstruct(cuNDArray<float_complext> &data, cuNDArray<float_complext> &images, cuNDArray<floatd3> &trajectory_in, cuNDArray<float> &dcf_in, cuNDArray<float_complext> &cweights_ho, cuNDArray<float> &scaled_time_in, arma::fvec fbins);
            void demodulate_kspace(cuNDArray<float_complext> &demodulated_data, const cuNDArray<float> &scaled_time, float demodulation_freq);
            void deconstruct(cuNDArray<float_complext> &images, cuNDArray<float_complext> &data, cuNDArray<floatd3> &trajectory_in, cuNDArray<float> &dcf_in, cuNDArray<float_complext> &cweights, cuNDArray<float> &scaled_time_in, arma::fvec fbins);
            void preprocess(cuNDArray<floatd3> *trajectory);

            std::tuple<std::vector<arma::fvec>, std::vector<cuNDArray<float_complext>>> divide_bins(arma::fvec in, cuNDArray<float_complext> &cuweights, int numGPUs);

        protected:
            bool bins_calculated = false;
            std::vector<arma::fvec> freq_bins;
            std::vector<cuNDArray<float_complext>> cweights_bins;
        private:
        };
    }
}