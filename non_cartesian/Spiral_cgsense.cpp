/*
 * SpiralMocoRecon.cpp
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
#include <curand.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include "SpiralReconBuffer.h"

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

class Spiral_cgsense : public ChannelGadget<Core::variant<Core::Acquisition, std::vector<std::vector<size_t>>>>,
                       public ImageArraySendMixin<Spiral_cgsense>
{
public:
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
    float oversampling_factor_;
    float kernel_width_;
    bool verbose;
    bool csm_calculated_ = false;

    boost::shared_ptr<cuNFFT_plan<float, 3>> nfft_plan_;
    boost::shared_ptr<cuNDArray<float_complext>> csm_;
    hoNDArray<float_complext> hocsm_;
    Gadgetron::ImageIOAnalyze gt_exporter_;
    boost::shared_ptr<cuNDArray<float>> _precon_weights;
    boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

    Spiral_cgsense(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::variant<Core::Acquisition, std::vector<std::vector<size_t>>>>(context, props)
    {
        kernel_width_ = 3;
        oversampling_factor_ = oversampling_factor;
        verbose = false;
    }

    void process(InputChannel<Core::variant<Core::Acquisition, std::vector<std::vector<size_t>>>> &in,
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
        nhlbi_toolbox::utils::enable_peeraccess();
        // boost::shared_ptr<cuNDArray<float_complext>> cuData;
        // boost::shared_ptr<hoNDArray<float_complext>> hoData;

        hoNDArray<float> trajectories;

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

        Gadgetron::reconParams recon_params;

        // Declare vectors for temporalTV
        std::vector<hoNDArray<float_complext>> dataVector;
        std::vector<hoNDArray<float>> dcwVector;
        std::vector<hoNDArray<floatd3>> trajVector;
        std::vector<cuNDArray<float>> sctVector;
        boost::shared_ptr<cuNDArray<float_complext>> cweights;
        arma::fvec freq_bins;
        nhlbi_toolbox::corrections::mri_concomitant_field_correction field_correction(this->header);

        using namespace Gadgetron::Indexing;

        auto kspace_scaling = 1e-3 * fov.x / matrixSize.x;

        auto idx = 0;

        size_t acq_count = 0;
        bool csmSent = false;

        bool weightsEstimated = false;
        auto maxZencode = header.encoding.front().encodingLimits.kspace_encoding_step_2.get().maximum + 1;
        auto maxAcq = ((header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.kspace_encoding_step_2.get().maximum + 1) * (header.encoding[0].encodingLimits.average.get().maximum + 1) * (header.encoding[0].encodingLimits.repetition.get().maximum + 1)); // use -1 for data acquired b/w 12/23 - 01/21
        GDEBUG_STREAM("maxAcq:" << maxAcq);
        std::vector<std::vector<size_t>> idx_phases;
        std::vector<Core::Acquisition> allAcq(maxAcq);
        std::vector<ISMRMRD::AcquisitionHeader> headers(maxAcq);
        
        // Collect all the data -- BEGIN()
        for (auto message : in)
        {
            if (holds_alternative<Core::Acquisition>(message) && idx < maxAcq)
            {
                auto &[head, data, traj] = Core::get<Core::Acquisition>(message);
                allAcq[idx] = std::move(Core::get<Core::Acquisition>(message));
                idx++;
            }
            if (holds_alternative<std::vector<std::vector<size_t>>>(message))
            {
                idx_phases = Core::get<std::vector<std::vector<size_t>>>(message);
            }
        }
        {
            GadgetronTimer timer("Optimized Recon :");

            // Check if binning data was sent -- cannot proceed without it really ! Use different Gadget
            if (idx_phases.empty() || idx_phases[0].size() == 0)
                GERROR("binning was not done \n");
            else
                GDEBUG_STREAM("Idx_phases: " << idx_phases.size());

            // Calculate elements in each bin and sum of all elements
            auto sumall = 0;
            std::vector<size_t> nelem_idx;
            for (auto iph = 0; iph < idx_phases.size(); iph++)
            {
                sumall += idx_phases[iph].size();
                nelem_idx.push_back(idx_phases[iph].size());
                GDEBUG_STREAM("nelem_idx: " << idx_phases[iph].size());

            }

            cudaSetDevice(selectedDevice);
            auto &[headAcq, dataAcq, trajAcq] = allAcq[0];
            acqhdr = headAcq;
            RO = dataAcq.get_size(0);
            CHA = dataAcq.get_size(1);
            E2 = this->header.encoding.front().encodedSpace.matrixSize.z;
            N = dataAcq.get_size(3);
            S = 1;
            SLC = 1;

            recon_params.numberChannels = CHA;
            recon_params.RO = RO;
            recon_params.ematrixSize = this->header.encoding.front().encodedSpace.matrixSize;
            recon_params.rmatrixSize = this->header.encoding.front().reconSpace.matrixSize;
            recon_params.fov = this->header.encoding.front().encodedSpace.fieldOfView_mm;
            recon_params.oversampling_factor_ = oversampling_factor_;
            recon_params.kernel_width_ = kernel_width_;
            recon_params.selectedDevice = selectedDevice;
            recon_params.norm = 2;

            this->initialize_encoding_space_limits(this->header);

            nhlbi_toolbox::reconstruction::noncartesian_reconstruction reconstruction(recon_params);
            allAcq.resize(idx);

            auto [cuData_csm, traj_csm, dcw_csm] = reconstruction.organize_data(&allAcq);
            
            if (!csm_calculated_ || CSMAlways)
            {
                auto dcf_csm = reconstruction.estimate_dcf(&traj_csm);
                cuNDArray<float_complext> channel_images(reconstruction.get_recon_dims());
                {
                    GadgetronTimer timer("Reconstruct CSM");

                    reconstruction.reconstruct(&cuData_csm, &channel_images, &traj_csm, &dcf_csm);
                }

                csm = reconstruction.generateRoemerCSM(&channel_images);

                csm_calculated_ = true;
                channel_images *= *conj(csm.get());
                auto combined = sum(&channel_images, channel_images.get_number_of_dimensions() - 1);

                auto cuimages_all = reconstruction.crop_to_recondims<float_complext>(*combined);
                auto images_all = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(cuimages_all.to_host())));

                nhlbi_toolbox::utils::filterImagealongSlice(images_all, get_kspace_filter_type(ftype), fwidth, fsigma);
                nhlbi_toolbox::utils::filterImage(images_all, get_kspace_filter_type(inftype), infwidth, infsigma);

                imarray.data_ = images_all;
                nhlbi_toolbox::utils::attachHeadertoImageArray(imarray, acqhdr, this->header);
                prepare_image_array(imarray, (size_t)0, ((int)series_counter + 0), GADGETRON_IMAGE_REGULAR);

                out.push(imarray);
                imarray.meta_.clear();
                imarray.headers_.clear();
                cuData_csm.clear();
                channel_images.clear();
                traj_csm.clear();
                dcf_csm.clear();

            }
            if (!NoSense)
            {
                recon_params.iterations = iterations;
                recon_params.iterations_inner = iterations_inner;

                recon_params.tolerance = tolSense;
                recon_params.selectedDevice = selectedDevice;
                recon_params.norm = 2;
                recon_params.lambda_spatial = lambda_spatial;
                nhlbi_toolbox::reconstruction::noncartesian_reconstruction_3D reconstruction3D(recon_params);

                auto [cuData, traj, dcw, num_elements] = reconstruction.organize_data(&allAcq, idx_phases);
                auto dcf_estimated = reconstruction.estimate_dcf(&traj, &dcw);

                // if (doConcomitantFieldCorrection)
                // {
                auto hoTraj = hoNDArray<floatd3>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<floatd3>>(traj.to_host())));
                std::vector<size_t> non_flat_dims = {recon_params.RO, hoTraj.get_number_of_elements() / recon_params.RO};
                auto traj_view = hoNDArray<floatd3>(non_flat_dims, hoTraj.get_data_ptr());
                auto gradients = nhlbi_toolbox::utils::traj2grad(traj_view, kspace_scaling);

                cuNDArray<Gadgetron::float_complext> cuIimages;

                if (doConcomitantFieldCorrection)
                {
                    if (!weightsEstimated)
                    {
                        field_correction.setup(gradients, headAcq);

                        auto cw = hoNDArray<float_complext>(field_correction.combinationWeights);
                        cweights = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(cw));

                        auto x = field_correction.demodulation_freqs;
                        freq_bins = arma::fvec(x.n_elem);
                        for (auto ii = 0; ii < x.n_elem; ii++)
                            freq_bins[ii] = float(x[ii]);

                        weightsEstimated = true;
                    }

                    field_correction.calculate_scaledTime(gradients);
                    auto sct = cuNDArray<float>(field_correction.scaled_time);
                    sctVector.push_back(sct);

                }
                else
                {
                    if (!weightsEstimated)
                    {
                        field_correction.setup(gradients, headAcq);
                        freq_bins = arma::fvec(1);
                        freq_bins = float(0.0);
                        hoNDArray<float_complext> cw({this->header.encoding.front().encodedSpace.matrixSize.x, this->header.encoding.front().encodedSpace.matrixSize.y, this->header.encoding.front().encodedSpace.matrixSize.z, 1});
                        fill(&cw, float_complext(1.0f, 1.0f));
                        cweights = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(cw));

                        weightsEstimated = true;
                    }
                    field_correction.calculate_scaledTime(gradients);
                    auto sct = cuNDArray<float>(field_correction.scaled_time);
                    sctVector.push_back(sct);
                }
                cuIimages = reconstruction3D.reconstruct_CGSense_fc(&cuData, &traj, &dcf_estimated, csm, cweights.get(), &sctVector[0], freq_bins);

                auto images_all = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(cuIimages.to_host())));

                // Sending respiratory resolved images
                using namespace Gadgetron::Indexing;
                IsmrmrdImageArray imarray_sense;
                series_counter = 1;

                for (auto ii = 0; ii < images_all.get_size(3); ii++)
                {
                    auto tmp = hoNDArray<std::complex<float>>(images_all(slice, slice, slice, ii));
                    imarray_sense.data_ = tmp;

                    nhlbi_toolbox::utils::attachHeadertoImageArray(imarray_sense, acqhdr, this->header);

                    prepare_image_array(imarray_sense, (size_t)0, ((int)series_counter), GADGETRON_IMAGE_REGULAR);

                    out.push(imarray_sense);
                    imarray_sense.meta_.clear();
                    imarray_sense.headers_.clear();
                }
                series_counter++;
            }
        }
    }

protected:
    NODE_PROPERTY(useIterativeDCW, bool, "Use Iterative DCW", false);
    NODE_PROPERTY(useIterativeDCWEstimated, bool, "Iterative DCW with Estimates", false);
    NODE_PROPERTY(iterations, size_t, "Number of Iterations", 1);
    NODE_PROPERTY(iterations_inner, size_t, "Number of iterations_inner", 1);
    NODE_PROPERTY(lambda_spatial, float, "lambda_spatial", 0);

    NODE_PROPERTY(tolSense, float, "tolerance", 100);
    NODE_PROPERTY(fwidth, double, "filterWidth", 0.15);
    NODE_PROPERTY(fsigma, double, "filterSigma", 1.0);
    NODE_PROPERTY(Debug, double, "Debug", 0);
    NODE_PROPERTY(NoSense, double, "NoSense", 0);
    NODE_PROPERTY(doConcomitantFieldCorrection, bool, "doConcomitantFieldCorrection", true);

    NODE_PROPERTY(ftype, std::string, "FilterType", "none");
    NODE_PROPERTY(inftype, std::string, "inftype", "none");
    NODE_PROPERTY(infwidth, double, "infwidth", 0.15);
    NODE_PROPERTY(infsigma, double, "infsigma", 1.0);
    NODE_PROPERTY(SOS, bool, "SOS", false);
    NODE_PROPERTY(CSMAlways, bool, "CSMAlways", false);

    NODE_PROPERTY(oversampling_factor, float, "oversampling_factor", 1.5);

    int series_counter = 0;
};

GADGETRON_GADGET_EXPORT(Spiral_cgsense)
