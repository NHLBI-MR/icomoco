#pragma once

#include <gadgetron/vector_td.h>
#include <vector>
#include <complex>
#include <gadgetron/hoNDArray.h>
#include <gadgetron/hoNDFFT.h>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoNDArray_math.h>
#include <boost/optional.hpp>
#include <gadgetron/hoNDArray_fileio.h>
#include <boost/math/constants/constants.hpp>
#include <math.h>
#include <stdio.h>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <boost/filesystem/fstream.hpp>
#include <gadgetron/hoArmadillo.h>
#include <gadgetron/cuSDC.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/cuNDArray.h>
#include <gadgetron/cuNDFFT.h>
#include <gadgetron/mri_core_utility.h> // Added MRD
#include "cuNonCartesianTSenseOperator.h"
#include <gadgetron/cuCgPreconditioner.h>
#include <gadgetron/cuPartialDerivativeOperator2.h>
#include <gadgetron/cuPartialDerivativeOperator.h>
#include <gadgetron/cuNDArray_utils.h>
#include "densityCompensation.h"

#include <gadgetron/cuGpBbSolver.h>
#include <gadgetron/cuSbcCgSolver.h>
#include <ismrmrd/ismrmrd.h>

#include "util_functions.h"

#include "reconParams.h"
using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        template <size_t D = 3> class noncartesian_reconstruction
        {
        public:
            noncartesian_reconstruction(reconParams recon_params);

            void reconstruct(
                cuNDArray<float_complext> *data,
                cuNDArray<float_complext> *image,
                cuNDArray<vector_td<float, D>> *traj,
                cuNDArray<float> *dcw);

            void deconstruct(
                cuNDArray<float_complext> *images,
                cuNDArray<float_complext> *data,
                cuNDArray<vector_td<float, D>> *traj,
                cuNDArray<float> *dcw);

            boost::shared_ptr<cuNDArray<float_complext>> generateCSM(
                cuNDArray<float_complext> *channel_images);

            boost::shared_ptr<cuNDArray<float_complext>> generateMcKenzieCSM(
                cuNDArray<float_complext> *channel_images);

            boost::shared_ptr<cuNDArray<float_complext>> generateRoemerCSM(
                cuNDArray<float_complext> *channel_images);


            // This is the standard method takes in data, traj, and dcw loads things to GPU for recon
            std::tuple<cuNDArray<float_complext>,
                       cuNDArray<vector_td<float, D>>>
            organize_data(
                hoNDArray<float_complext> *data,
                hoNDArray<vector_td<float, D>> *traj);

            // This is the optimized method takes in acq and returns data,traj and dcw for recon - skips accumulate gadget 
            std::tuple<cuNDArray<float_complext>,
                       cuNDArray<vector_td<float, D>>,
                       cuNDArray<float>>
            organize_data(std::vector<Core::Acquisition> *allAcq);

            std::tuple<cuNDArray<float_complext>,
                       cuNDArray<vector_td<float, D>>,
                       cuNDArray<float>,
                       std::vector<size_t>>
            organize_data(std::vector<Core::Acquisition> *allAcq, std::vector<std::vector<size_t>> idx_phases);

            std::vector<size_t> get_recon_dims() { return recon_dims; };

            template <typename T>
            cuNDArray<T> crop_to_recondims(cuNDArray<T> &input);

            boost::shared_ptr<cuNFFT_plan<float, D>> nfft_plan_;
            std::vector<size_t> image_dims_;
            std::vector<size_t> image_dims_os_;

            cuNDArray<float> estimate_dcf(cuNDArray<vector_td<float, 3>> *traj);
            

            cuNDArray<float> estimate_dcf(cuNDArray<vector_td<float, 3>> *traj,std::vector<size_t> number_elements);
            std::vector<cuNDArray<float>> estimate_dcf(std::vector<cuNDArray<vector_td<float, 3>>> *traj);

            cuNDArray<float> estimate_dcf(cuNDArray<vector_td<float, 2>> *traj);
            cuNDArray<float> estimate_dcf(cuNDArray<vector_td<float, 2>> *traj,std::vector<size_t> number_elements);
            std::vector<cuNDArray<float>> estimate_dcf(std::vector<cuNDArray<vector_td<float, 2>>> *traj);
            cuNDArray<float> estimate_dcf(cuNDArray<vector_td<float, 2>> *traj,cuNDArray<float> *dcf_in);
            cuNDArray<float> estimate_dcf(cuNDArray<vector_td<float, 3>> *traj,cuNDArray<float> *dcf_in);
            template <typename T > std::vector<cuNDArray<T>> arraytovector(cuNDArray<T> *inputArray, std::vector<size_t> number_elements);
            std::vector<cuNDArray<float>> estimate_dcf(std::vector<cuNDArray<vector_td<float, 3>>> *traj, std::vector<cuNDArray<float>> *dcf_in);

        protected:
            reconParams recon_params;
            float resx;
            float resy;
            float resz;
            std::vector<size_t> recon_dims;
            std::vector<size_t> recon_dims_encodSpace;
            std::vector<size_t> recon_dims_reconSpace;
            density_compensation dcfO;
            bool isprocessed = false;

        private:
        };
    }
}