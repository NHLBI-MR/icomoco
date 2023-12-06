#include "noncartesian_reconstruction.h"
#include <gadgetron/cuNonCartesianSenseOperator.h>
#include <gadgetron/cuSbcCgSolver.h>
#include "cuNonCartesianSenseOperator_fc.h"

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        class noncartesian_reconstruction_3D : public noncartesian_reconstruction<3>
        {
        public:
            noncartesian_reconstruction_3D(reconParams recon_params) : noncartesian_reconstruction<3>(recon_params){};
            
            using noncartesian_reconstruction::reconstruct;
            using noncartesian_reconstruction::organize_data;            

            cuNDArray<float_complext> reconstruct(
                cuNDArray<float_complext> *data,
                cuNDArray<floatd3> *traj,
                cuNDArray<float> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm);

            cuNDArray<float_complext> reconstruct_CGSense(
                cuNDArray<float_complext> *data,
                cuNDArray<floatd3> *traj,
                cuNDArray<float> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm);

            cuNDArray<float_complext> reconstruct_fc(
                cuNDArray<float_complext> *data,
                cuNDArray<floatd3> *traj,
                cuNDArray<float> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                cuNDArray<float_complext> *combination_weights,
                cuNDArray<float> *scaled_time,
                arma::fvec fbins);

            cuNDArray<float_complext> reconstruct_CGSense_fc(
                cuNDArray<float_complext> *data,
                cuNDArray<floatd3> *traj,
                cuNDArray<float> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                cuNDArray<float_complext> *combination_weights,
                cuNDArray<float> *scaled_time,
                arma::fvec fbins);

            std::tuple<cuNDArray<float_complext>,
                       cuNDArray<floatd3>,
                       cuNDArray<float>> organize_data(
                hoNDArray<float_complext> *data,
                hoNDArray<floatd3> *traj,
                hoNDArray<float> *dcw);
    
        };
    }
}