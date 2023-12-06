/*
 * SpiraliCoMocoRecon.cpp
 *
 *  Created on: September 17th, 2021
 *      Author: Ahsan Javed
 */

#include <gadgetron/Node.h>
#include <gadgetron/mri_core_grappa.h>
#include <gadgetron/vector_td_utilities.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <gadgetron/cgSolver.h>
#include <gadgetron/hoNDArray_math.h>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoNDArray_elemwise.h>
#include <gadgetron/hoNFFT.h>
#include <gadgetron/hoNDFFT.h>
#include <numeric>
#include <random>
#include <gadgetron/NonCartesianTools.h>
#include <gadgetron/NFFTOperator.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <boost/range/combine.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <gadgetron/GadgetronTimer.h>
#include <gadgetron/mri_core_coil_map_estimation.h>
#include <gadgetron/generic_recon_gadgets/GenericReconBase.h>
#include <boost/hana/functional/iterate.hpp>
#include <gadgetron/cuNDArray_converter.h>
#include <gadgetron/ImageArraySendMixin.h>
#include <gadgetron/cuNFFT.h>
#include <gadgetron/cuNDFFT.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/cuNDArray.h>
#include <gadgetron/b1_map.h>
#include <iterator>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoCgSolver.h>
#include <gadgetron/hoNDImage_util.h>
#include "cuNonCartesianTSenseOperator.h"
#include "cuNonCartesianMOCOOperator.h"
#include <gadgetron/cuNonCartesianSenseOperator.h>
#include <omp.h>
#include <gadgetron/cuCgSolver.h>
#include <gadgetron/cuNlcgSolver.h>
#include <gadgetron/cuCgPreconditioner.h>
#include <gadgetron/cuNDArray_operators.h>
#include <gadgetron/cuImageOperator.h>
#include <gadgetron/cuTvOperator.h>
#include <gadgetron/cuTvPicsOperator.h>
#include "../spiral/SpiralBuffer.h"
#include <gadgetron/mri_core_kspace_filter.h>
#include <gadgetron/ImageIOBase.h>
#include <gadgetron/hoNDArray_fileio.h>
#include <gadgetron/cuSDC.h>
#include "../utils/concomitantFieldCorrection/mri_concomitant_field_correction.h"
#include "../utils/gpu/cuda_utils.h"
// #include "/usr/local/include/cuTVPrimalDualOperator.h"

#include "cuPartialDerivativeOperator.h"
#include "cuPartialDerivativeOperator2.h"
#include "cuGpBbSolver.h"
#include "cuSbcCgSolver.h"
#include <util_functions.h>
#include "noncartesian_reconstruction.h"
#include "noncartesian_reconstruction_4D.h"
#include "noncartesian_reconstruction_3D.h"
#include "reconParams.h"
#include "densityCompensation.h"
#include "SpiralReconBuffer.h"
#include <curand.h>
#include <omp.h>
#include <algorithm>
#include <cmath>

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

class SpiraliCoMocoRecon : public ChannelGadget<Core::variant<Gadgetron::Core::Image<std::complex<float>>, Gadgetron::Core::Image<float>, Gadgetron::SpiralReconBuffer<cuNDArray, float_complext, 3>>>,
                           public ImageArraySendMixin<SpiraliCoMocoRecon>
{
public:
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
    float oversampling_factor_;
    float kernel_width_;
    bool verbose;
    boost::shared_ptr<cuNFFT_plan<float, 3>> nfft_plan_;
    boost::shared_ptr<cuNDArray<float_complext>> csm_;
    hoNDArray<float_complext> hocsm_;

    Gadgetron::ImageIOAnalyze gt_exporter_;
    boost::shared_ptr<cuNDArray<float>> _precon_weights;
    boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

    SpiraliCoMocoRecon(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::variant<Gadgetron::Core::Image<std::complex<float>>, Gadgetron::Core::Image<float>, Gadgetron::SpiralReconBuffer<cuNDArray, float_complext, 3>>>(context, props)
    {
        kernel_width_ = 3;
        oversampling_factor_ = oversampling_factor;
        verbose = false;
    }

    void process(InputChannel<Core::variant<Gadgetron::Core::Image<std::complex<float>>, Gadgetron::Core::Image<float>, Gadgetron::SpiralReconBuffer<cuNDArray, float_complext, 3>>> &in,
                 OutputChannel &out) override
    {

        int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();
        auto matrixSize = this->header.encoding.front().reconSpace.matrixSize;
        auto fov = this->header.encoding.front().encodedSpace.fieldOfView_mm;
        auto resx = fov.x / float(matrixSize.x);
        auto resy = fov.y / float(matrixSize.y);
        auto resz = fov.z / float(matrixSize.z);
        unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();
        size_t RO, E1, E2, CHA, N, S, SLC;

        bool csm_calculated_ = false;
        bool weightsEstimated = false;
        // boost::shared_ptr<cuNDArray<float_complext>> cuData;
        // boost::shared_ptr<hoNDArray<float_complext>> hoData;

        hoNDArray<float> trajectories;

        boost::shared_ptr<cuNDArray<float>> dcw;
        boost::shared_ptr<cuNDArray<floatd3>> traj;

        std::tuple<boost::shared_ptr<hoNDArray<floatd3>>,
                   boost::shared_ptr<hoNDArray<float>>>
            traj_dcw;
        IsmrmrdDataBuffered *buffer;
        IsmrmrdImageArray imarray;
        ISMRMRD::AcquisitionHeader acqhdr;
        std::vector<size_t> recon_dims;
        std::vector<size_t> recon_dims_reconSpace;
        std::vector<size_t> recon_dims_encodSpace;
        hoNDArray<size_t> shots_per_time;
        boost::shared_ptr<cuNDArray<float_complext>> csm;
        hoNDArray<floatd3> gradients;

        cudaSetDevice(selectedDevice);

        cuNDArray<float_complext> cuData;
        std::vector<cuNDArray<Gadgetron::floatd3>> trajVec;
        std::vector<cuNDArray<float>> dcwVec;
        cuNDArray<float_complext> cweights;
        std::vector<cuNDArray<float>> sctVector;
        arma::fvec freq_bins;
        reconParams recon_params;
        bool reconReady = false;
        cuNDArray<float> deformation, invdeformation;
        std::vector<hoNDArray<std::complex<float>>> image_vec;
        std::vector<hoNDArray<float>> deformation_vec;

        nhlbi_toolbox::corrections::mri_concomitant_field_correction field_correction(this->header);

        using namespace Gadgetron::Indexing;

        auto kspace_scaling = 1e-3 * fov.x / matrixSize.x;

        for (auto message : in)
        {

            GadgetronTimer timer("iMoCo Recon Data Recieving");

            if (holds_alternative<SpiralReconBuffer<cuNDArray, float_complext, 3>>(message))
            {
                auto &[cuData_t, trajVec_t, dcwVec_t, csm_t, cweights_t, sctVector_t, freq_bins_t, recon_params_t, acqhdr_t] = Core::get<Gadgetron::SpiralReconBuffer<cuNDArray, float_complext, 3>>(message);
                cuData = std::move(cuData_t);
                trajVec = std::move(trajVec_t);
                dcwVec = std::move(dcwVec_t);
                cweights = std::move(cweights_t);
                csm = boost::make_shared<cuNDArray<float_complext>>(csm_t);
                sctVector = std::move(sctVector_t);
                freq_bins = std::move(freq_bins_t);
                recon_params = std::move(recon_params_t);
                acqhdr = acqhdr_t;
            }
            if (holds_alternative<Gadgetron::Core::Image<float>>(message))
            {
                auto &[head, image, meta] = Core::get<Gadgetron::Core::Image<float>>(message);
                GDEBUG_STREAM("head.image_series_index: " << head.image_series_index);

                if (head.image_series_index == 111)
                {
                    GDEBUG_STREAM("Deformations: " << image.get_number_of_dimensions());

                    deformation_vec.push_back(image);
                }
            }
            if (holds_alternative<Gadgetron::Core::Image<std::complex<float>>>(message))
            {
                auto &[head, image, meta] = Core::get<Gadgetron::Core::Image<std::complex<float>>>(message);
                if (head.image_series_index == 1)
                    image_vec.push_back(image);
                out.push(message);
            }
        }
        cudaSetDevice(selectedDevice);
        auto num_dims_def = (deformation_vec[0].dimensions());
        GDEBUG_STREAM("deformation dimensions: " << deformation_vec[0].get_number_of_dimensions());
        GDEBUG_STREAM("deformation_vec: " << deformation_vec.size());

        num_dims_def.push_back(deformation_vec.size());
        auto def = cuNDArray<float>(num_dims_def);

        size_t stride = std::accumulate(num_dims_def.begin(), num_dims_def.end()-1, 1, std::multiplies<size_t>());

        for (int ii=0;ii<deformation_vec.size();ii++)
        {
            cudaMemcpy(def.get_data_ptr()+ ii*stride,deformation_vec.at(ii).get_data_ptr(),stride * sizeof(float),cudaMemcpyDefault);
        }

        deformation = def;
        deformation.reshape(image_vec[0].get_size(0),
                            image_vec[0].get_size(1),
                            image_vec[0].get_size(2),
                            3, -1);
        invdeformation = deformation;
        invdeformation *= -1.0f;
        reconReady = true;
        if (reconReady)
        {
            GDEBUG_STREAM("image vec size: " << image_vec.size());
            GDEBUG_STREAM("deformation dimensions: " << deformation.get_number_of_dimensions());
            GDEBUG_STREAM("deformation dimensions: " << invdeformation.get_number_of_dimensions());

            this->initialize_encoding_space_limits(this->header);
            recon_params.iterations = iterations_imoco;
            recon_params.lambda_spatial = lambda_imoco;
            recon_params.tolerance = tolerance;
            nhlbi_toolbox::reconstruction::noncartesian_reconstruction_4D reconstruction4D(recon_params);

            hoNDArray<std::complex<float>> images_all(deformation.get_size(0), deformation.get_size(1), deformation.get_size(2), deformation.get_size(4));
            for (auto ii = 0; ii < deformation.get_size(4); ii++)
            {
                images_all(slice, slice, slice, ii) = hoNDArray<std::complex<float>>(image_vec[ii]);
            }
            //     auto [deformation, invdeformation] = reconstruction4D.register_images_time(hoNDArray<float>(abs(images_all)), referencePhase, 4, {moco_iter, moco_iter, moco_iter, moco_iter}, {3, 9, 12, 18}, {1, 1.5, 2, 2}, true, false, false, "LocalCCR", true);

            auto registered_images = images_all;
            IsmrmrdImageArray imarray_sense;
            series_counter = 3;
            //     // Sending respiratory registered images
            for (auto ii = 0; ii < registered_images.get_size(3); ii++)
            {
                auto tmp = hoNDArray<std::complex<float>>(registered_images(slice, slice, slice, ii));
                imarray_sense.data_ = tmp;

                nhlbi_toolbox::utils::attachHeadertoImageArray(imarray_sense, acqhdr, this->header);

                prepare_image_array(imarray_sense, (size_t)0, ((int)series_counter), GADGETRON_IMAGE_REGULAR);

                out.push(imarray_sense);
                imarray_sense.meta_.clear();
                imarray_sense.headers_.clear();
            }

            registered_images = reconstruction4D.applyDeformations(images_all, deformation);

            series_counter = 4;
            //     // Sending respiratory registered images
            for (auto ii = 0; ii < registered_images.get_size(3); ii++)
            {
                auto tmp = hoNDArray<std::complex<float>>(registered_images(slice, slice, slice, ii));
                imarray_sense.data_ = tmp;

                nhlbi_toolbox::utils::attachHeadertoImageArray(imarray_sense, acqhdr, this->header);

                prepare_image_array(imarray_sense, (size_t)0, ((int)series_counter), GADGETRON_IMAGE_REGULAR);

                out.push(imarray_sense);
                imarray_sense.meta_.clear();
                imarray_sense.headers_.clear();
            }
            series_counter++;
            // MOCO recon
            auto moco_images = sum(registered_images, registered_images.get_number_of_dimensions() - 1);
            moco_images /= std::complex<float>((float)registered_images.get_size(3), (float)0);

            // Sending respiratory registered images
            for (auto ii = 0; ii < moco_images.get_size(3); ii++)
            {
                auto tmp = hoNDArray<std::complex<float>>(moco_images(slice, slice, slice, ii));
                imarray_sense.data_ = tmp;

                nhlbi_toolbox::utils::attachHeadertoImageArray(imarray_sense, acqhdr, this->header);

                prepare_image_array(imarray_sense, (size_t)0, ((int)series_counter), GADGETRON_IMAGE_REGULAR);

                out.push(imarray_sense);
                imarray_sense.meta_.clear();
                imarray_sense.headers_.clear();
            }
            series_counter++;
            cuNDArray<float_complext> cuImages_imoco;

            //if(doConcomitantFieldCorrection)
                cuImages_imoco = reconstruction4D.reconstructiMOCO_fc(&cuData, &trajVec, &dcwVec, csm, &cweights, sctVector, freq_bins, &deformation, &invdeformation);
            // else
            //     cuImages_imoco = reconstruction4D.reconstructiMOCO(&cuData, &trajVec, &dcwVec, csm,&deformation, &invdeformation);
                

            auto images_iMOCO = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(cuImages_imoco.to_host())));

            // Sending respiratory registered images
            for (auto ii = 0; ii < images_iMOCO.get_size(3); ii++)
            {
                auto tmp = hoNDArray<std::complex<float>>(images_iMOCO(slice, slice, slice, ii));
                imarray_sense.data_ = tmp;

                nhlbi_toolbox::utils::attachHeadertoImageArray(imarray_sense, acqhdr, this->header);

                prepare_image_array(imarray_sense, (size_t)0, ((int)series_counter), GADGETRON_IMAGE_REGULAR);

                out.push(imarray_sense);
                imarray_sense.meta_.clear();
                imarray_sense.headers_.clear();
            }
        }
        // cuData.clear();
        // trajVec.clear();
        // dcwVec.clear();
        // cweights.clear();
        // csm.get()->clear();
        // sctVector.clear();
    }

protected:
    NODE_PROPERTY(iterations_imoco, size_t, "Number of Iterations iMOCO", 10);
    NODE_PROPERTY(tolerance, float, "Number of Iterations Sense", 300);
    NODE_PROPERTY(Debug, double, "Debug", 0);
    NODE_PROPERTY(lambda_imoco, float, "lambda_imoco", 1e-2);
    NODE_PROPERTY(oversampling_factor, float, "oversampling_factor", 2.1);
    NODE_PROPERTY(norm, float, "norm", 2.0);

    NODE_PROPERTY(doConcomitantFieldCorrection, bool, "doConcomitantFieldCorrection", true);

    NODE_PROPERTY(ftype, std::string, "FilterType", "none");
    NODE_PROPERTY(inftype, std::string, "inftype", "none");
    NODE_PROPERTY(infwidth, double, "infwidth", 0.15);
    NODE_PROPERTY(infsigma, double, "infsigma", 1.0);
    NODE_PROPERTY(fwidth, double, "filterWidth", 0.15);
    NODE_PROPERTY(fsigma, double, "filterSigma", 1.0);
    int series_counter = 0;
};

GADGETRON_GADGET_EXPORT(SpiraliCoMocoRecon)