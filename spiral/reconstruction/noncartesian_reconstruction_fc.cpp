#include "noncartesian_reconstruction_fc.h"
#include "util_functions.h"
using namespace Gadgetron;
namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        void noncartesian_reconstruction_fc::reconstruct_todevice(cuNDArray<float_complext> &data, cuNDArray<float_complext> &images, cuNDArray<floatd3> &trajectory_in, cuNDArray<float> &dcf_in, cuNDArray<float_complext> &cweights, cuNDArray<float> &scaled_time_in, arma::fvec fbins, int deviceNo, bool forward)
        {
            //  GadgetronTimer timer("reconstruct_todevice");

            using namespace nhlbi_toolbox::utils;
            int cur_device;

            if (forward)
            {
                cur_device = images.get_device();
                cudaSetDevice(cur_device);
              //  GadgetronTimer timer("reconstruct_todevice");
                auto tdata = set_device(&data, deviceNo);
                data.clear();
                //auto timage = set_device(&images, deviceNo);
                cuNDArray<float_complext> timage(*images.get_dimensions(), deviceNo);
                auto ttraj = set_device(&trajectory_in, deviceNo);
                auto tdcf = set_device(&dcf_in, deviceNo);
                auto tcweights = set_device(&cweights, deviceNo);
                auto tscaled_time_in = set_device(&scaled_time_in, deviceNo);
                cudaSetDevice(deviceNo);

                //   this->nfft_plan_ = NFFT<cuNDArray, float, 3>::make_plan(from_std_vector<size_t, 3>(image_dims_), image_dims_os_, recon_params.kernel_width_, ConvolutionType::ATOMIC);

                reconstruct(tdata,
                            timage,
                            ttraj,
                            tdcf,
                            tcweights,
                            tscaled_time_in,
                            fbins);

                cudaMemcpy(images.get_data_ptr(), timage.get_data_ptr(),
                           timage.get_number_of_elements() * sizeof(float_complext), cudaMemcpyDefault);
            }
            else
            {
                cur_device = data.get_device();
                cudaSetDevice(cur_device);
            //    GadgetronTimer timer("deconstruct_todevice");
                //auto tdata = set_device(&data, deviceNo);
                cuNDArray<float_complext> tdata(*data.get_dimensions(), deviceNo);

                auto timage = set_device(&images, deviceNo);
                auto ttraj = set_device(&trajectory_in, deviceNo);
                auto tdcf = set_device(&dcf_in, deviceNo);
                auto tcweights = set_device(&cweights, deviceNo);
                auto tscaled_time_in = set_device(&scaled_time_in, deviceNo);
                cudaSetDevice(deviceNo);
                //     this->nfft_plan_ = NFFT<cuNDArray, float, 3>::make_plan(from_std_vector<size_t, 3>(image_dims_), image_dims_os_, recon_params.kernel_width_, ConvolutionType::ATOMIC);

                deconstruct(timage,
                            tdata,
                            ttraj,
                            tdcf,
                            tcweights,
                            tscaled_time_in,
                            fbins);
                cudaMemcpy(data.get_data_ptr(), tdata.get_data_ptr(), tdata.get_number_of_elements() * sizeof(float_complext), cudaMemcpyDefault);
            }
            cudaSetDevice(data.get_device());
        }

        void noncartesian_reconstruction_fc::reconstruct_parallel(cuNDArray<float_complext> &data, cuNDArray<float_complext> &images,
                                                                  cuNDArray<floatd3> &trajectory_in,
                                                                  cuNDArray<float> &dcf_in,
                                                                  cuNDArray<float_complext> &cweights,
                                                                  cuNDArray<float> &scaled_time_in,
                                                                  arma::fvec fbins, int deviceNo, bool forward)
        {
            //  GadgetronTimer timer("reconstruct_todevice");

            using namespace nhlbi_toolbox::utils;
            int cur_device = data.get_device();
            cudaSetDevice(cur_device);

            auto size_data = data.get_number_of_elements() * 2 * 4 + trajectory_in.get_number_of_elements() * 3 * 4 +
                             dcf_in.get_number_of_elements() * 4;

            auto eligibleGPUs = nhlbi_toolbox::utils::FindCudaDevices(size_data);
            if (!bins_calculated)
                auto [freq_bins, cweights_bins] = divide_bins(fbins, cweights, eligibleGPUs.size());

            this->freq_bins = freq_bins;
            this->cweights_bins = cweights_bins;

            cuNDArray<float_complext> timage_all;
            auto rdim = *images.get_dimensions();
            auto stride = std::accumulate(rdim.begin(), rdim.end(), 1,
                                          std::multiplies<size_t>()); // product of X,Y,and Z

            rdim.push_back(this->freq_bins.size());
            if (forward)
                timage_all.create(rdim);

            if (forward)
            {
#pragma omp parallel for num_threads(eligibleGPUs.size())
                for (auto iter = 0; iter < eligibleGPUs.size(); iter++)
                {
                    auto gpuDevice = fbins.n_elem > 1 ? eligibleGPUs[iter % eligibleGPUs.size()] : cur_device;
                    //       GadgetronTimer timer("reconstruct_todevice");
                    auto tdata = set_device(&data, gpuDevice);
                    //auto timage = set_device(&images, deviceNo);
                    cuNDArray<float_complext> timage(*images.get_dimensions(), gpuDevice);
                    auto ttraj = set_device(&trajectory_in, gpuDevice);
                    auto tdcf = set_device(&dcf_in, gpuDevice);
                    auto tcweights = set_device(&(this->cweights_bins[iter]), gpuDevice);
                    auto tscaled_time_in = set_device(&scaled_time_in, gpuDevice);
                    cudaSetDevice(iter);

                    //   this->nfft_plan_ = NFFT<cuNDArray, float, 3>::make_plan(from_std_vector<size_t, 3>(image_dims_), image_dims_os_, recon_params.kernel_width_, ConvolutionType::ATOMIC);

                    reconstruct(tdata,
                                timage,
                                ttraj,
                                tdcf,
                                tcweights,
                                tscaled_time_in,
                                this->freq_bins[iter]);

                    cudaMemcpy(timage_all.get_data_ptr() + stride * iter, timage.get_data_ptr(),
                               timage.get_number_of_elements() * sizeof(float_complext), cudaMemcpyDeviceToDevice);
                }
                cudaSetDevice(cur_device);
                auto timage = *sum(&timage_all, timage_all.get_number_of_dimensions() - 1);
                cudaMemcpy(images.get_data_ptr(), timage.get_data_ptr(),
                           timage.get_number_of_elements() * sizeof(float_complext), cudaMemcpyDeviceToDevice);
            }
            else
            {
                //       GadgetronTimer timer("deconstruct_todevice");
                //auto tdata = set_device(&data, deviceNo);
                cuNDArray<float_complext> tdata(*data.get_dimensions(), deviceNo);

                auto timage = set_device(&images, deviceNo);
                auto ttraj = set_device(&trajectory_in, deviceNo);
                auto tdcf = set_device(&dcf_in, deviceNo);
                auto tcweights = set_device(&cweights, deviceNo);
                auto tscaled_time_in = set_device(&scaled_time_in, deviceNo);
                cudaSetDevice(deviceNo);
                //     this->nfft_plan_ = NFFT<cuNDArray, float, 3>::make_plan(from_std_vector<size_t, 3>(image_dims_), image_dims_os_, recon_params.kernel_width_, ConvolutionType::ATOMIC);

                deconstruct(timage,
                            tdata,
                            ttraj,
                            tdcf,
                            tcweights,
                            tscaled_time_in,
                            fbins);
                cudaMemcpy(data.get_data_ptr(), tdata.get_data_ptr(), tdata.get_number_of_elements() * sizeof(float_complext), cudaMemcpyDeviceToDevice);
            }
            cudaSetDevice(data.get_device());
        }
        void noncartesian_reconstruction_fc::reconstruct(cuNDArray<float_complext> &data, cuNDArray<float_complext> &images, cuNDArray<floatd3> &trajectory_in, cuNDArray<float> &dcf_in, cuNDArray<float_complext> &cweights, cuNDArray<float> &scaled_time_in, arma::fvec fbins)
        {
     //   GadgetronTimer timer("Reconstruct (noncartRecon)");

            fill(&images, float_complext(0, 0));
            auto temp_image = images;
            for (int ii = 0; ii < fbins.n_elem; ii++)
            {
                if (fbins.n_elem > 1)
                {
                    if (ii == 0)
                        demodulate_kspace(data, scaled_time_in, fbins[ii]);
                    else
                        demodulate_kspace(data, scaled_time_in, fbins[ii] - fbins[ii - 1]);
                }

                noncartesian_reconstruction::reconstruct(&data, &temp_image, &trajectory_in, &dcf_in);

                auto cw_dims = *cweights.get_dimensions();
                cw_dims.pop_back();

                auto stride_ch = std::accumulate(cw_dims.begin(), cw_dims.end(), 1, std::multiplies<size_t>());

                if (fbins.n_elem > 1)
                {
                    //   GadgetronTimer timer("PRODUCT");
                    auto cw_view = cuNDArray<complext<float>>(cw_dims, cweights.data() + stride_ch * ii);

                    temp_image *= *conj(&cw_view);
                }
                images += temp_image;
            }
            demodulate_kspace(data, scaled_time_in, -1 * fbins[fbins.n_elem - 1]);
        }

        void noncartesian_reconstruction_fc::deconstruct(cuNDArray<float_complext> &images, cuNDArray<float_complext> &data, cuNDArray<floatd3> &trajectory_in, cuNDArray<float> &dcf_in, cuNDArray<float_complext> &cweights, cuNDArray<float> &scaled_time_in, arma::fvec fbins)
        {
//        GadgetronTimer timer("Deconstruct (noncartRecon)");

            auto cw_dims = *cweights.get_dimensions();
            cw_dims.pop_back();

            auto stride_ch = std::accumulate(cw_dims.begin(), cw_dims.end(), 1, std::multiplies<size_t>());

            fill(&data, float_complext(0, 0));

            auto temp_data = data;
            for (int ii = 0; ii < fbins.n_elem; ii++)
            {
                auto channel_images_temp = images;

                if (fbins.n_elem > 1)
                {
                    //   GadgetronTimer timer("PRODUCT");
                    auto cw_view = cuNDArray<complext<float>>(cw_dims, cweights.data() + stride_ch * ii);

                    channel_images_temp *= cw_view;
                }

                noncartesian_reconstruction::deconstruct(&channel_images_temp, &temp_data, &trajectory_in, &dcf_in);

                if (fbins.n_elem > 1)
                {
                    demodulate_kspace(temp_data, scaled_time_in, -1 * fbins[ii]);
                }

                data += temp_data;
            }
        }

        void noncartesian_reconstruction_fc::demodulate_kspace(
            cuNDArray<float_complext> &demodulated_data,
            const cuNDArray<float> &scaled_time,
            float demodulation_freq)
        {
            //GadgetronTimer timer("Demodulation");
            constexpr float PI = boost::math::constants::pi<float>();

            auto recon_dim = demodulated_data.get_dimensions();
            recon_dim->pop_back();
            //hoNDArray<std::complex<float>> phase_term(recon_dim);

            auto val = float(-2.0 * PI * demodulation_freq);
            //  scaled_time *= val;

            //scaled_time *= val;

            auto arg_exp = std::move(*imag_to_complex<float_complext>(&scaled_time));
            arg_exp *= val;
            auto phase_term = nhlbi_toolbox::cuda_utils::cuexp<float_complext>(arg_exp);
            phase_term.squeeze();
            arg_exp.clear();

            std::vector<size_t> recon_dims({demodulated_data.get_size(0), demodulated_data.get_size(1)});
            cuNDArray<float_complext> temp_data(recon_dims);

            for (auto ich = 0; ich < demodulated_data.get_size(2); ich++)
            {
                cudaMemcpy(temp_data.get_data_ptr(),
                           demodulated_data.get_data_ptr() + demodulated_data.get_size(0) * demodulated_data.get_size(1) * ich,
                           demodulated_data.get_size(0) * demodulated_data.get_size(1) * sizeof(float_complext), cudaMemcpyDefault);

                temp_data *= phase_term;

                cudaMemcpy(demodulated_data.get_data_ptr() + demodulated_data.get_size(0) * demodulated_data.get_size(1) * ich,
                           temp_data.get_data_ptr(),
                           demodulated_data.get_size(0) * demodulated_data.get_size(1) * sizeof(float_complext), cudaMemcpyDefault);
            }

            //demodulated_data *= phase_term;

            //return demodulated_data;
        }
        void noncartesian_reconstruction_fc::preprocess(cuNDArray<floatd3> *trajectory)
        {
            this->nfft_plan_->preprocess(*trajectory, NFFT_prep_mode::ALL);
            this->isprocessed = true;
        }

        std::tuple<std::vector<arma::fvec>, std::vector<cuNDArray<float_complext>>> noncartesian_reconstruction_fc::divide_bins(arma::fvec in, cuNDArray<float_complext> &cuweights, int numGPUs)
        {
            std::vector<arma::fvec> out;
            std::vector<cuNDArray<float_complext>> out_weights;
            auto weights = *cuweights.to_host();
            auto sindex = 0;
            auto stride = floor(in.n_elem / numGPUs) - 1;
            for (auto ii = 0; ii < numGPUs; ii++)
            {
                if (ii == (numGPUs - 1))
                {
                    out.push_back(arma::fvec(in(arma::span(sindex, in.n_elem - 1))));
                    std::vector<size_t> rdims = {weights.get_size(0), weights.get_size(1), weights.get_size(2), out[ii].n_elem};
                    hoNDArray<float_complext> warray(rdims);
                    for (auto jj = 0; jj < out[ii].n_elem; jj++)
                    {
                        warray(slice, slice, slice, jj) = weights(slice, slice, slice, sindex + jj);
                    }
                    out_weights.push_back(cuNDArray<float_complext>(warray));
                }
                else
                {
                    out.push_back(arma::fvec(in(arma::span(sindex, sindex + stride))));
                    std::vector<size_t> rdims = {weights.get_size(0), weights.get_size(1), weights.get_size(2), out[ii].n_elem};
                    hoNDArray<float_complext> warray(rdims);
                    for (auto jj = 0; jj < out[ii].n_elem; jj++)
                    {
                        warray(slice, slice, slice, jj) = weights(slice, slice, slice, sindex + jj);
                    }
                    out_weights.push_back(cuNDArray<float_complext>(warray));

                    sindex += stride + 1;
                }

                //auto nn = arma::vec(n);
            }
            this->bins_calculated = true;
            return std::make_tuple(out, out_weights);
        }
    }
}
