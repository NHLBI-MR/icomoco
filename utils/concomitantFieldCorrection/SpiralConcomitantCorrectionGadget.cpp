#include <gadgetron/Node.h>
#include <gadgetron/mri_core_grappa.h>
#include <gadgetron/vector_td_utilities.h>
#include "../../spiral/SpiralBuffer.h"
#include "../../utils/util_functions.h"
#include "../../utils/concomitantFieldCorrection/mri_concomitant_field_correction.h"
#include "../../utils/gpu/cuda_utils.h"

#include <gadgetron/hoNDArray_fileio.h>
#include <gadgetron/hoNDArray_math.h>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoNDArray_elemwise.h>
#include <gadgetron/hoNFFT.h>
#include <gadgetron/hoNDFFT.h>
#include <gadgetron/cuNDArray_converter.h>
#include <gadgetron/cuNFFT.h>
#include <gadgetron/cuNDFFT.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/cuNDArray.h>

#include <gadgetron/cuNDArray_operators.h>
#include <gadgetron/cuNonCartesianSenseOperator.h>

#include <gadgetron/vector_td.h>

#include <sstream>
#include <fstream>
#include <iostream>

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace nhlbi_toolbox::utils;
class SpiralConcomitantCorrectionGadget : public ChannelGadget<Core::variant<Gadgetron::SpiralBuffer<cuNDArray, float_complext, 2>, Gadgetron::SpiralBuffer<cuNDArray, float_complext, 3>,
                                                                             Gadgetron::SpiralBuffer<hoNDArray, float_complext, 2>, Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>>
{

public:
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
    float oversampling_factor_;
    float kernel_width_;
    bool verbose;
    bool calculated_cw = false;
    bool csm_calculated = false;
    boost::shared_ptr<cuNFFT_plan<float, 3>> nfft_plan_;
    boost::shared_ptr<cuNDArray<float_complext>> csm_;

    SpiralConcomitantCorrectionGadget(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::variant<Gadgetron::SpiralBuffer<cuNDArray, float_complext, 2>, Gadgetron::SpiralBuffer<cuNDArray, float_complext, 3>,
                                                                                                                                       Gadgetron::SpiralBuffer<hoNDArray, float_complext, 2>, Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>>(context, props)
    {
        kernel_width_ = 5.5;
        verbose = false;
    }

    boost::shared_ptr<cuNDArray<float_complext>> deconstruct(
        cuNDArray<float_complext> *images,
        cuNDArray<float> *dcw,
        std::vector<size_t> recon_dims)
    {

        GadgetronTimer timer("Deconstruct");
        auto RO = images->get_size(0);
        auto PE = images->get_size(1);
        auto SL = images->get_size(2);
        auto CHA = images->get_size(3);

        auto result = boost::make_shared<cuNDArray<float_complext>>(recon_dims);
        recon_dims.pop_back();
        recon_dims.pop_back();

        // auto temp = boost::make_shared<cuNDArray<float_complext>>(recon_dims);

        // cuNDArray<float_complext> tempdata(std::vector<size_t>{RO, PE, SL});

        //cuNDArray<float_complext> tempdata_gpu1({RO, E1E2, CHA}, 1); // Tricks to save memory and allow calculations

        // for (int iCHA = 0; iCHA < CHA; iCHA++)
        // {
        //     cudaMemcpy(tempdata.get_data_ptr(),
        //                images->get_data_ptr() + RO * PE * SL * iCHA,
        //                RO * PE * SL * sizeof(float_complext), cudaMemcpyDefault);
        //     nfft_plan_->compute(tempdata, *temp.get(), dcw, NFFT_comp_mode::FORWARDS_C2NC);
        nfft_plan_->compute(images, *result, dcw, NFFT_comp_mode::FORWARDS_C2NC);
        //     cudaMemcpy(result->get_data_ptr() + recon_dims[0] * recon_dims[1] * iCHA,
        //                temp->get_data_ptr(),
        //                recon_dims[0] * recon_dims[1] * sizeof(float_complext), cudaMemcpyDefault);
        // }

        return result;
    }

    boost::shared_ptr<cuNDArray<float_complext>> reconstruct(
        cuNDArray<float_complext> *data,
        cuNDArray<float> *dcw,
        std::vector<size_t> recon_dims)
    {

        GadgetronTimer timer("Reconstruct");
        auto RO = data->get_size(0);
        auto E1E2 = data->get_size(1);
        auto CHA = data->get_size(2);

        auto result = boost::make_shared<cuNDArray<float_complext>>(recon_dims);
        recon_dims.pop_back();

        auto temp = boost::make_shared<cuNDArray<float_complext>>(recon_dims);

        cuNDArray<float_complext> tempdata(std::vector<size_t>{RO, E1E2});

        //cuNDArray<float_complext> tempdata_gpu1({RO, E1E2, CHA}, 1); // Tricks to save memory and allow calculations

        for (int iCHA = 0; iCHA < CHA; iCHA++)
        {
            cudaMemcpy(tempdata.get_data_ptr(),
                       data->get_data_ptr() + RO * E1E2 * iCHA,
                       RO * E1E2 * sizeof(float_complext), cudaMemcpyDefault);
            nfft_plan_->compute(tempdata, *temp.get(), dcw, NFFT_comp_mode::BACKWARDS_NC2C);
            //nfft_plan_->compute(data, *result, dcw, NFFT_comp_mode::BACKWARDS_NC2C);
            cudaMemcpy(result->get_data_ptr() + recon_dims[0] * recon_dims[1] * recon_dims[2] * iCHA,
                       temp->get_data_ptr(),
                       recon_dims[0] * recon_dims[1] * recon_dims[2] * sizeof(float_complext), cudaMemcpyDefault);
        }

        return result;
    }

    cuNDArray<float_complext> demodulate_kspace(
        cuNDArray<float_complext> demodulated_data,
        cuNDArray<float> scaled_time,
        float demodulation_freq)
    {
        GadgetronTimer timer("Demodulation");
        constexpr float PI = boost::math::constants::pi<float>();

        auto recon_dim = demodulated_data.get_dimensions();
        recon_dim->pop_back();
        //hoNDArray<std::complex<float>> phase_term(recon_dim);

        auto val = float(-2.0 * PI * demodulation_freq);
        //  scaled_time *= val;

        scaled_time *= val;

        auto arg_exp = *imag_to_complex<float_complext>(&scaled_time);
        auto phase_term = nhlbi_toolbox::cuda_utils::cuexp<float_complext>(arg_exp);

        demodulated_data *= phase_term;

        return demodulated_data;
    }

    hoNDArray<floatd3> traj2grad(hoNDArray<floatd3> trajectory, float kspace_scaling, float gamma)
    {
        auto gradients = trajectory;

        // GDEBUG_STREAM("Trajectory Size 0 " << trajectory.get_size(0));
        // GDEBUG_STREAM("Trajectory Size 1 " << trajectory.get_size(1));
        // GDEBUG_STREAM("Trajectory All " << trajectory.get_number_of_elements());

        for (auto jj = 0; jj < trajectory.get_size(1); jj++)
        {
            auto traj_ptr = trajectory.get_data_ptr() + jj * trajectory.get_size(0);
            auto grad_ptr = gradients.get_data_ptr() + jj * gradients.get_size(0);
            for (auto ii = 0; ii < trajectory.get_size(0); ii++)
            {
                if (ii > 0)
                {
                    grad_ptr[ii][0] = (traj_ptr[ii][0] - traj_ptr[ii - 1][0]) * 1000000 / (gamma * 10 * 2 * kspace_scaling);
                    grad_ptr[ii][1] = (traj_ptr[ii][1] - traj_ptr[ii - 1][1]) * 1000000 / (gamma * 10 * 2 * kspace_scaling);
                    grad_ptr[ii][2] = (traj_ptr[ii][2] - traj_ptr[ii - 1][2]) * 1000000 / (gamma * 10 * 2 * kspace_scaling);
                }
            }
        }
        return gradients;
        // trajectory_and_weights(0, 0) = (gradients_interpolated(0)[0]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling;
        //trajectory_and_weights(1, 0) = (gradients_interpolated(0)[1]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling;
        // for (int ii = 1; ii < size_gradOVS; ii++)
        // {
        //     trajectory_and_weights(0, ii) = ((gradients_interpolated(ii)[0]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling + trajectory_and_weights(0, ii - 1)); // mT/m * Hz/G * 10G * 2e-6
        //     trajectory_and_weights(1, ii) = ((gradients_interpolated(ii)[1]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling + trajectory_and_weights(1, ii - 1)); // mT/m * Hz/G * 10G * 2e-6
        // }
    }
    void process(InputChannel<Core::variant<Gadgetron::SpiralBuffer<cuNDArray, float_complext, 2>, Gadgetron::SpiralBuffer<cuNDArray, float_complext, 3>,
                                            Gadgetron::SpiralBuffer<hoNDArray, float_complext, 2>, Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>> &in,
                 OutputChannel &out) override
    {
        int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();
        cudaSetDevice(selectedDevice);
        ISMRMRD::AcquisitionHeader acqhdr;
        boost::shared_ptr<cuNDArray<float_complext>> cuData;
        boost::shared_ptr<hoNDArray<float_complext>> hoData;
        boost::shared_ptr<cuNDArray<float_complext>> csm_;

        hoNDArray<float> trajectories;
        boost::shared_ptr<cuNDArray<float>> dcw;
        boost::shared_ptr<cuNDArray<floatd3>> traj;

        auto matrixSize = this->header.encoding.front().reconSpace.matrixSize;
        auto fov = this->header.encoding.front().encodedSpace.fieldOfView_mm;
        oversampling_factor_ = 1.5;
        unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();
        size_t RO, E1, E2, CHA, N, S, SLC;
        auto kspace_scaling = 1e-3 * fov.x / matrixSize.x;
        hoNDArray<floatd3> gradients;
        image_dims_.push_back(matrixSize.x);
        image_dims_.push_back(matrixSize.y);
        image_dims_.push_back(warp_size * (this->header.encoding.front().encodedSpace.matrixSize.z / warp_size + 1));

        constexpr double GAMMA = 4258.0; /* Hz/G */
        nhlbi_toolbox::corrections::mri_concomitant_field_correction field_correction(this->header);

        // make the z dimension be multiple of 32 not sure of this discuss with David it was causing issues with cuda
        //Figure out what the oversampled matrix size should be taking the warp size into consideration.

        image_dims_os_ = uint64d3(((static_cast<size_t>(std::ceil(image_dims_[0] * oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size,
                                  ((static_cast<size_t>(std::ceil(image_dims_[1] * oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size,
                                  ((static_cast<size_t>(std::ceil(image_dims_[2] * 1)) + warp_size - 1) / warp_size) * warp_size); // No oversampling is needed in the z-direction for SOS

        for (auto message : in)
        {

            if (holds_alternative<Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>(message))
            {
                auto &[sp_data, sp_traj, sp_dcw, sp_headers] = Core::get<Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>(message);
                acqhdr = sp_headers[0];

                std::vector<size_t> insize = *sp_data.get_dimensions();
                std::vector<size_t> outsize;
                if (acqhdr.active_channels == insize[insize.size() - 1])
                    outsize = *sp_data.get_dimensions();
                else
                    outsize = {insize[0], insize[2], insize[1]};
                nhlbi_toolbox::utils::set_data<hoNDArray>(sp_data, sp_traj, sp_dcw, sp_headers, cuData, traj, dcw, insize, outsize, selectedDevice);

                float scale_factor = float(prod(image_dims_os_)) / asum((dcw.get()));
                *dcw *= scale_factor;

                RO = (*cuData).get_size(0);
                E1 = (*cuData).get_size(1);
                E2 = this->header.encoding.front().encodedSpace.matrixSize.z;
                CHA = (*cuData).get_size(2);

                std::vector<size_t> recon_dims;

                nfft_plan_ = NFFT<cuNDArray, float, 3>::make_plan(from_std_vector<size_t, 3>(image_dims_), image_dims_os_, kernel_width_, ConvolutionType::ATOMIC);
                nfft_plan_->preprocess(*traj.get(), NFFT_prep_mode::NC2C);
                // if (!csm_calculated)
                // {
                recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], CHA};
                auto channel_images = *reconstruct(&(*cuData.get()), &(*dcw.get()), recon_dims);

                recon_dims = {image_dims_[0], image_dims_[1], E2, CHA}; // Cropped to size of Recon Matrix
                cuNDArray<float_complext> channel_images_cropped(recon_dims);

                crop<float_complext, 4>(uint64d4(0, 0, (image_dims_[2] - E2) / 2, 0),
                                        uint64d4(image_dims_[0], image_dims_[1], E2, CHA),
                                        channel_images,
                                        channel_images_cropped);
                //    recon_dims = {image_dims_[0], image_dims_[1], CHA, E2}; // Cropped to size of Recon Matrix
                // channel_images = pad<float_complext, 4>(uint64d4(recon_dims[0], recon_dims[1], image_dims_[2], CHA),
                //                                         channel_images_cropped, float_complext(0));

                //     auto temp = boost::make_shared<cuNDArray<float_complext>>(estimate_b1_map<float, 3>(channel_images_cropped));
                //     csm_ = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 4>(uint64d4(recon_dims[0], recon_dims[1], image_dims_[2], CHA),
                //                                                                                 *temp, float_complext(0)));

                //     out.push(SpiralBuffer<hoNDArray, float_complext, 3>{std::move(sp_data),
                //                                                         std::move(sp_traj),
                //                                                         std::move(sp_dcw),
                //                                                         std::move(sp_headers)});
                //     csm_calculated = true;

                // }
                // else
                // {
                // gradients = traj2grad(sp_traj, kspace_scaling, GAMMA);
                // field_correction.setup(gradients, acqhdr);
                // calculated_cw = true;

                // recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], field_correction.numfreqbins};
                // cuNDArray<float_complext> image_array(recon_dims);

                // auto E_ = boost::shared_ptr<cuNonCartesianSenseOperator<float, 3>>(new cuNonCartesianSenseOperator<float, 3>(ConvolutionType::ATOMIC));

                // //sqrt_inplace(dcw.get());
                // E_->setup(from_std_vector<size_t, 3>(image_dims_), image_dims_os_, kernel_width_);
                // E_->set_dcw(dcw);
                // E_->set_csm(csm_);
                auto codomain_dims = *cuData.get()->get_dimensions();
                //if (codomain_dims[codomain_dims.size() - 1] != CHA)
                //    codomain_dims.pop_back();

                // recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], CHA};
                // recon_dims.pop_back();

                // E_->set_domain_dimensions(&recon_dims);
                // E_->set_codomain_dimensions(&codomain_dims);
                // E_->preprocess(traj.get());

                // cuNDArray<float_complext> reg_image(E_->get_domain_dimensions());
                //square_inplace(dcw.get());

                // auto scaled_time = boost::make_shared<cuNDArray<float>>(field_correction.scaled_time);

                // for (int ii = 0; ii < field_correction.numfreqbins; ii++)
                // {
                //     GadgetronTimer timer("MFI");

                //     GDEBUG_STREAM("Iteration# " << ii);
                //     GDEBUG_STREAM("Frequency: " << field_correction.demodulation_freqs[ii]);

                //     if (ii == 0)
                //         *cuData = demodulate_kspace(*cuData.get(), *scaled_time, field_correction.demodulation_freqs[ii]);
                //     else
                //         *cuData = demodulate_kspace(*cuData.get(), *scaled_time, field_correction.demodulation_freqs[ii] - field_correction.demodulation_freqs[ii - 1]);

                //     recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], CHA};

                //     auto channel_images_temp = *reconstruct(&(*cuData.get()), &(*dcw.get()), recon_dims);
                //     E_->mult_csm_conj_sum(&channel_images_temp, &reg_image);

                //     cudaMemcpy(image_array.get_data_ptr() + (recon_dims[0] * recon_dims[1] * recon_dims[2]) * ii,
                //                reg_image.get_data_ptr(),
                //                recon_dims[0] * recon_dims[1] * recon_dims[2] * sizeof(float_complext), cudaMemcpyDefault);
                // }

                // auto cw = field_correction.combinationWeights;

                // using namespace Gadgetron::Indexing;
                // auto cweights_cu = cuNDArray<float_complext>(hoNDArray<float_complext>(field_correction.combinationWeights));

                // if (field_correction.numfreqbins > 1)
                // {
                //     // Zero pad weights before multiplication by demodulated images
                //     auto zpadded_cw = pad<float_complext, 4>(uint64d4(image_dims_[0], image_dims_[1], image_dims_[2], field_correction.numfreqbins),
                //                                              &cweights_cu, float_complext(0));

                //     image_array *= zpadded_cw;//*conj(&zpadded_cw);
                // }
                // recon_dims = {image_array.get_size(0), image_array.get_size(1), image_array.get_size(2)};
                // cuNDArray<float_complext> imarr(recon_dims);

                //  cudaMemcpy(imarr.get_data_ptr(),
                //             image_array.get_data_ptr(),
                //             (image_array.get_size(0) * image_array.get_size(1) * image_array.get_size(2)) * sizeof(float_complext), cudaMemcpyDefault);

                // image_array = *sum(&image_array, image_array.get_number_of_dimensions() - 1);
                // sqrt_inplace(dcw.get());

                // cuData.get()->clear();
                // recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], CHA};
                // cuNDArray<float_complext> ch_image(recon_dims);

                // E_->mult_csm(&imarr, &ch_image);
                // codomain_dims.pop_back();
                nfft_plan_->preprocess(*traj.get(), NFFT_prep_mode::C2NC);
                //      sqrt_inplace(dcw.get());
                //square_inplace(dcw.get());
                reciprocal_inplace(dcw.get());
                auto data = deconstruct(&channel_images_cropped, dcw.get(), codomain_dims);
                // auto td1 = *sum(data.get(), data.get()->get_number_of_dimensions() - 1);
                // td1 = *sum(&td1, td1.get_number_of_dimensions() - 1);
                // td1 = *sum(&td1, td1.get_number_of_dimensions() - 1);
                // td1 = *sum(&td1, td1.get_number_of_dimensions() - 1);
                auto td1 = mean(data.get());

                // auto td2 = *sum(cuData.get(), cuData.get()->get_number_of_dimensions() - 1);
                // td2 = *sum(&td1, td1.get_number_of_dimensions() - 1);
                // td2 = *sum(&td1, td1.get_number_of_dimensions() - 1);
                // td2 = *sum(&td1, td1.get_number_of_dimensions() - 1);
                auto td2 = mean(cuData.get());

                *data /= *dcw;

                auto td3 = mean(data.get());

                // auto td3 = *sum(data.get(), data.get()->get_number_of_dimensions() - 1);
                // td3 = *sum(&td3, td3.get_number_of_dimensions() - 1);
                // td3 = *sum(&td3, td3.get_number_of_dimensions() - 1);
                // td3 = *sum(&td3, td3.get_number_of_dimensions() - 1);
                // cuNDArray<float_complext> data(cuData.get()->get_dimensions());

                auto combined_acquisitions = hoNDArray<float_complext>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(data.get()->to_host())));

                traj.get()->reshape(combined_acquisitions.get_size(0), combined_acquisitions.get_size(1));
                // dcw.get()->reshape(combined_acquisitions.get_size(0), combined_acquisitions.get_size(1));

                auto combined_traj = hoNDArray<floatd3>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<floatd3>>(traj.get()->to_host())));
                // auto combined_density = hoNDArray<float>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<float>>(dcw.get()->to_host())));

                out.push(SpiralBuffer<hoNDArray, float_complext, 3>{std::move(combined_acquisitions),
                                                                    std::move(sp_traj),
                                                                    std::move(sp_dcw),
                                                                    std::move(sp_headers)});
                // }
            }
        }
    }
};

GADGETRON_GADGET_EXPORT(SpiralConcomitantCorrectionGadget)
