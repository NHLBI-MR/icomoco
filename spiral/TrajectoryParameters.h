#pragma once

#include <ismrmrd/xml.h>
#include <gadgetron/log.h>
#include <gadgetron/Gadget.h>
#include "vds.h"
#include <gadgetron/hoNDArray_fileio.h>
#include <gadgetron/vector_td_utilities.h>
#include <gadgetron/mri_core_utility.h>
#include <gadgetron/mri_core_girf_correction.h>
#include <gadgetron/hoArmadillo.h>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoNDArray_math.h>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoNDArray_elemwise.h>
using namespace Gadgetron;

namespace nhlbi_toolbox
{
    namespace Spiral
    {

        class TrajectoryParameters
        {
        public:
            TrajectoryParameters() = default;
            TrajectoryParameters(const ISMRMRD::IsmrmrdHeader &h);

            std::pair<hoNDArray<floatd2>, hoNDArray<float>>
            calculate_trajectories_and_weight(const ISMRMRD::AcquisitionHeader &acq_header);
            void set_girf_sampling_time(float time);
            void read_girf_kernel(std::string girf_folder);
            hoNDArray<std::complex<float>> get_girf_kernel();

            // NHLBI customisation
            // 2-FOV variable density design
            double  vds_factor_;
            // custom rotation number
            long    spiral_rotations_;

        private:
            Core::optional<hoNDArray<std::complex<float>>> girf_kernel;
            float girf_sampling_time_us;
            long Tsamp_ns_;
            long Nints_;
            double gmax_;
            double smax_;
            double krmax_;
            double fov_;
            float TE_;
            std::string systemModel;

            hoNDArray<floatd2> correct_gradients(const hoNDArray<floatd2> &gradients, float grad_samp_us,
                                                 float girf_samp_us, const float *read_dir, const float *phase_dir,
                                                 const float *slice_dir);
            

        };
    } // namespace Spiral
} // namespace nhlbi_toolbox


