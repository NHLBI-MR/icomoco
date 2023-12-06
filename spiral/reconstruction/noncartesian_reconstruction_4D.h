#pragma once
#include "noncartesian_reconstruction.h"
#include "cuNonCartesianTSenseOperator_fc.h"
#include "cuNonCartesianMOCOOperator_fc.h"
#include "cuNonCartesianMOCOOperator.h"


#include <gadgetron/cuSbcCgSolver.h>
#include <util_functions.h>
#include <gadgetron/cuNlcgSolver.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        class noncartesian_reconstruction_4D : public noncartesian_reconstruction<3>
        {
        public:
            noncartesian_reconstruction_4D(reconParams recon_params) : noncartesian_reconstruction<3>(recon_params){};

            using noncartesian_reconstruction::organize_data;
            using noncartesian_reconstruction::reconstruct;

            cuNDArray<float_complext> reconstruct(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm);

            cuNDArray<float_complext> reconstruct_nlcg(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm);

            cuNDArray<float_complext> reconstruct_fc(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                cuNDArray<float_complext> *combination_weights,
                std::vector<cuNDArray<float>> &scaled_time,
                arma::fvec fbins);

            cuNDArray<float_complext> reconstructiMOCO_fc(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                cuNDArray<float_complext> *combination_weights,
                std::vector<cuNDArray<float>> &scaled_time,
                arma::fvec fbins,
                cuNDArray<float> *def,
                cuNDArray<float> *invdef);


            cuNDArray<float_complext> reconstructiMOCO(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                cuNDArray<float> *def,
                cuNDArray<float> *invdef);

            cuNDArray<float> padDeformations(cuNDArray<float> deformation, std::vector<size_t> size_deformation);

            std::tuple<cuNDArray<float_complext>,
                       std::vector<cuNDArray<floatd3>>,
                       std::vector<cuNDArray<float>>>
            organize_data(
                hoNDArray<float_complext> *data,
                hoNDArray<floatd3> *traj,
                hoNDArray<float> *dcw);

            std::tuple<cuNDArray<float_complext>,
                       std::vector<cuNDArray<floatd3>>,
                       std::vector<cuNDArray<float>>>
            organize_data(
                std::vector<hoNDArray<float_complext>> *data,
                std::vector<hoNDArray<floatd3>> *traj,
                std::vector<hoNDArray<float>> *dcw);

            std::tuple<std::vector<cuNDArray<floatd3>>,
                       std::vector<cuNDArray<float>>>
            organize_data(
                cuNDArray<floatd3> *traj,
                cuNDArray<float> *dcw,
                std::vector<size_t> number_elements);

            std::tuple<cuNDArray<float>, cuNDArray<float>> register_images_time(hoNDArray<float> images_all, unsigned int referenceIndex, unsigned int level, std::vector<double> iters, std::vector<double> regularization_hilbert_strength, std::vector<double> LocalCCR_sigmaArg, bool BidirectionalReg, bool DivergenceFreeReg, bool verbose, std::string sim_meas, bool useInvDef);
            std::tuple<cuNDArray<float>, cuNDArray<float>> register_images_gpu(cuNDArray<float> images_all, unsigned int referenceIndex, float alpha);

            hoNDArray<std::complex<float>> applyDeformations(hoNDArray<std::complex<float>> images_all, cuNDArray<float> deformation);
        };
    }
}