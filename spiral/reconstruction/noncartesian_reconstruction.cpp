#include "noncartesian_reconstruction.h"
using namespace Gadgetron;
namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        template <size_t D>
        noncartesian_reconstruction<D>::noncartesian_reconstruction(reconParams recon_params)
        {
            unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();
            this->recon_params = recon_params;
            resx = recon_params.fov.x / float(recon_params.ematrixSize.x);
            resy = recon_params.fov.y / float(recon_params.ematrixSize.y);
            resz = recon_params.fov.z / float(recon_params.ematrixSize.z);

            image_dims_.push_back(recon_params.ematrixSize.x);
            image_dims_.push_back(recon_params.ematrixSize.y);

            if (D == 3 && recon_params.ematrixSize.z != 1) // 3D imaging
            {
                if (recon_params.ematrixSize.z % warp_size != 0)
                    image_dims_.push_back(warp_size * (recon_params.ematrixSize.z / warp_size + 1));
                else
                    image_dims_.push_back(recon_params.ematrixSize.z);

                image_dims_os_.push_back(((static_cast<size_t>(std::ceil(image_dims_[0] * recon_params.oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size);
                image_dims_os_.push_back(((static_cast<size_t>(std::ceil(image_dims_[1] * recon_params.oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size);
                image_dims_os_.push_back(((static_cast<size_t>(std::ceil(image_dims_[2] * recon_params.oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size); // No oversampling is needed in the z-direction for SOS

                recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.numberChannels};
                recon_dims_encodSpace = {image_dims_[0], image_dims_[1], recon_params.ematrixSize.z, recon_params.numberChannels}; // Cropped to size of Encoded Matrix
                recon_dims_reconSpace = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z, recon_params.numberChannels}; // Cropped to size of Recon Matrix
            }
            if (D == 2 && recon_params.ematrixSize.z == 1) // 2D imaging may look for MRD-Header definitions
            {
                // image_dims_os_ = uint64d<D>(((static_cast<size_t>(std::ceil(image_dims_[0] * recon_params.oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size,
                //                           ((static_cast<size_t>(std::ceil(image_dims_[1] * recon_params.oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size);
                image_dims_os_.push_back(((static_cast<size_t>(std::ceil(image_dims_[0] * recon_params.oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size);
                image_dims_os_.push_back(((static_cast<size_t>(std::ceil(image_dims_[1] * recon_params.oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size);

                recon_dims = {image_dims_[0], image_dims_[1], recon_params.numberChannels};
                recon_dims_encodSpace = {image_dims_[0], image_dims_[1], recon_params.numberChannels}; // Cropped to size of Encoded Matrix
                recon_dims_reconSpace = {image_dims_[0], image_dims_[1], recon_params.numberChannels}; // Cropped to size of Recon Matrix

                // this->nfft_plan_ = NFFT<cuNDArray, float, D>::make_plan(from_std_vector<size_t, D>(image_dims_), from_std_vector<size_t, D>(image_dims_os_), recon_params.kernel_width_, ConvolutionType::ATOMIC);
            }
            this->nfft_plan_ = NFFT<cuNDArray, float, D>::make_plan(from_std_vector<size_t, D>(image_dims_), from_std_vector<size_t, D>(image_dims_os_), recon_params.kernel_width_, ConvolutionType::ATOMIC);
            dcfO.oversampling_factor_ = recon_params.oversampling_factor_dcf_;
            dcfO.kernel_width_ = recon_params.kernel_width_dcf_;
            dcfO.iterations = recon_params.iterations_dcf;
            dcfO.useIterativeDCWEstimated = recon_params.useIterativeDCWEstimated;
        }

        template <size_t D>
        template <typename T>
        std::vector<cuNDArray<T>> noncartesian_reconstruction<D>::arraytovector(cuNDArray<T> *inputArray, std::vector<size_t> number_elements)
        {
            std::vector<cuNDArray<T>> vectorOut;

            for (auto iph = 0; iph < number_elements.size(); iph++)
            {
                auto str_phase = std::accumulate(number_elements.begin(), number_elements.begin() + iph, size_t(0));
                auto array_view = cuNDArray<T>({number_elements[iph]}, (*inputArray).get_data_ptr() + str_phase);
                vectorOut.push_back(std::move(array_view));
            }
            return std::move(vectorOut);
        }

        template <size_t D>
        template <typename T>
        cuNDArray<T> noncartesian_reconstruction<D>::crop_to_recondims(cuNDArray<T> &input)
        {
            cuNDArray<T> output(this->recon_dims_reconSpace);

            if (input.get_number_of_dimensions() > 3)
                crop<T, 4>(uint64d4(0, 0, (image_dims_[2] - this->recon_dims_reconSpace[2]) / 2, 0),
                           uint64d4(image_dims_[0], image_dims_[1], this->recon_dims_reconSpace[2], input.get_size(3)),
                           input,
                           output);
            else
                crop<T, 3>(uint64d3(0, 0, (image_dims_[2] - this->recon_dims_reconSpace[2]) / 2),
                           uint64d3(image_dims_[0], image_dims_[1], this->recon_dims_reconSpace[2]),
                           input,
                           output);
            return output;
        }

        template <size_t D>
        boost::shared_ptr<cuNDArray<float_complext>>
        noncartesian_reconstruction<D>::generateCSM(
            cuNDArray<float_complext> *channel_images)
        {
            auto CHA = channel_images->get_size(channel_images->get_number_of_dimensions() - 1); // Last dimension is chanels

            cuNDArray<float_complext> channel_images_cropped(recon_dims_encodSpace);
            crop<float_complext, 4>(uint64d4(0, 0, (recon_dims[2] - recon_dims_encodSpace[2]) / 2, 0),
                                    uint64d4(recon_dims[0], recon_dims[1], recon_dims_encodSpace[2], CHA),
                                    channel_images,
                                    channel_images_cropped);

            //  recon_dims = {image_dims_[0], image_dims_[1], CHA, E2}; // Cropped to size of Recon Matrix
            *channel_images = pad<float_complext, 4>(uint64d4(recon_dims[0], recon_dims[1], recon_dims[2], CHA),
                                                     channel_images_cropped, float_complext(0));
            // auto temp = boost::make_shared<cuNDArray<float_complext>>(estimate_b1_map<float, 3>(channel_images_cropped));
            auto temp = boost::make_shared<cuNDArray<float_complext>>(nhlbi_toolbox::utils::estimateCoilmaps_slice(channel_images_cropped));

            auto csm = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 4>(uint64d4(recon_dims[0], recon_dims[1], recon_dims[2], CHA),
                                                                                            *temp, float_complext(0)));

            return csm;
        }

        template <size_t D>
        boost::shared_ptr<cuNDArray<float_complext>>
        noncartesian_reconstruction<D>::generateMcKenzieCSM(
            cuNDArray<float_complext> *channel_images)
        {
            // McKenzie et al. (Magn Reson Med2002;47:529-538.)
            cuNDArray<float> scale_image(recon_dims_encodSpace);
            auto tmp_csm = std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*channel_images).to_host()));
            auto temp = tmp_csm;
            auto rsos = *sum(abs_square(&temp).get(), D);
            sqrt_inplace(&rsos);
            auto cuda_rsos = cuNDArray<float>(rsos);

            for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
            {
                cudaMemcpy(scale_image.get_data_ptr() + iCHA * cuda_rsos.get_number_of_elements(), cuda_rsos.get_data_ptr(), (cuda_rsos).get_number_of_elements() * sizeof(float), cudaMemcpyDefault);
            }
            auto tmp_scale_image = std::move(*boost::reinterpret_pointer_cast<hoNDArray<float>>(scale_image.to_host()));
            tmp_csm /= tmp_scale_image;
            auto csm = boost::make_shared<cuNDArray<float_complext>>(hoNDArray<float_complext>(tmp_csm));

            return csm;
        }

        template <size_t D>
        boost::shared_ptr<cuNDArray<float_complext>>
        noncartesian_reconstruction<D>::generateRoemerCSM(
            cuNDArray<float_complext> *channel_images)
        {
            auto CHA = channel_images->get_size(channel_images->get_number_of_dimensions() - 1); // Last dimension is chanels

            cuNDArray<float_complext> channel_images_cropped(recon_dims_encodSpace);
            boost::shared_ptr<cuNDArray<float_complext>> csm_new_pad;
            if (D == 3)
            {
                crop<float_complext, 4>(uint64d4(0, 0, (recon_dims[2] - recon_dims_encodSpace[2]) / 2, 0),
                                        uint64d4(recon_dims[0], recon_dims[1], recon_dims_encodSpace[2], CHA),
                                        channel_images,
                                        channel_images_cropped);

                auto filtered_csm = std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(channel_images_cropped.to_host()));
                nhlbi_toolbox::utils::filterImagealongSlice(filtered_csm, get_kspace_filter_type("gaussian"), 100, filtered_csm.get_size(3) / filtered_csm.get_size(1) * 30);
                nhlbi_toolbox::utils::filterImage(filtered_csm, get_kspace_filter_type("gaussian"), 100, 30);
                auto temp = filtered_csm;
                auto rsos = *sum(abs_square(&temp).get(), 3);
                sqrt_inplace(&rsos);
                filtered_csm /= rsos;
                auto csm_new = cuNDArray<float_complext>(hoNDArray<float_complext>(filtered_csm));

                csm_new_pad = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 4>(uint64d4(recon_dims[0], recon_dims[1], recon_dims[2], filtered_csm.get_size(3)),
                                                                                                   csm_new, float_complext(0)));
                //  recon_dims = {image_dims_[0], image_dims_[1], CHA, E2}; // Cropped to size of Recon Matrix
                *channel_images = pad<float_complext, 4>(uint64d4(recon_dims[0], recon_dims[1], recon_dims[2], CHA),
                                                         channel_images_cropped, float_complext(0));
            }
            else
            {
                channel_images_cropped = channel_images;
                auto filtered_csm = std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(channel_images_cropped.to_host()));
                nhlbi_toolbox::utils::filterImage(filtered_csm, get_kspace_filter_type("gaussian"), 100, 30);
                auto temp = filtered_csm;
                auto rsos = *sum(abs_square(&temp).get(), 2);
                sqrt_inplace(&rsos);
                filtered_csm /= rsos;
                csm_new_pad = boost::make_shared<cuNDArray<float_complext>>(hoNDArray<float_complext>(filtered_csm));
            }
            return csm_new_pad;
        }

        template <size_t D>
        void noncartesian_reconstruction<D>::reconstruct(
            cuNDArray<float_complext> *data,
            cuNDArray<float_complext> *image,
            cuNDArray<vector_td<float, D>> *traj,
            cuNDArray<float> *dcw)
        {

            //  GadgetronTimer timer("Reconstruct");
            auto RO = data->get_size(0);
            auto E1E2 = data->get_size(1);
            auto CHA = data->get_size(2);

            if (!this->isprocessed)
            {
                // GadgetronTimer timer("Preprocess");
                this->nfft_plan_->preprocess(*traj, NFFT_prep_mode::NC2C);
                this->isprocessed = true;
            }
            auto data_dimensions = *data->get_dimensions();
            auto image_dimensions = *image->get_dimensions();
            if (CHA != 1)
            {
                data_dimensions.pop_back();  // remove CHA
                image_dimensions.pop_back(); // remove CHA
            }

            auto stride = std::accumulate(data_dimensions.begin(), data_dimensions.end(), 1,
                                          std::multiplies<size_t>()); // product of X,Y,and Z

            auto stride_results = std::accumulate(image_dimensions.begin(), image_dimensions.end(), 1,
                                                  std::multiplies<size_t>()); // product of X,Y,and Z

            struct cudaDeviceProp properties;
            cudaGetDeviceProperties(&properties, data->get_device());

            if (float(cudaDeviceManager::Instance()->getFreeMemory(data->get_device())) / float(std::pow(1024, 3)) >= float(data->get_number_of_elements() * 4 * 2 * 1.5) / float(std::pow(1024, 3)))
            {
                try
                {
                    this->nfft_plan_->compute(*data, (*image), dcw, NFFT_comp_mode::BACKWARDS_NC2C);
                    //throw 505;
                }
                catch (...)
                {
                    GDEBUG_STREAM("Failed: now running in slower channel by channel mode");
                    for (int iCHA = 0; iCHA < CHA; iCHA++)
                    {
                        auto data_view = cuNDArray<complext<float>>(data_dimensions, data->data() + stride * iCHA);
                        auto results_view = cuNDArray<complext<float>>(image_dimensions, image->data() + stride_results * iCHA);

                        this->nfft_plan_->compute(data_view, results_view, dcw, NFFT_comp_mode::BACKWARDS_NC2C);
                    }
                }
            }
            else
            {
                GDEBUG_STREAM("Failed: now running in slower channel by channel mode");
                for (int iCHA = 0; iCHA < CHA; iCHA++)
                {
                    auto data_view = cuNDArray<complext<float>>(data_dimensions, data->data() + stride * iCHA);
                    auto results_view = cuNDArray<complext<float>>(image_dimensions, image->data() + stride_results * iCHA);

                    this->nfft_plan_->compute(data_view, results_view, dcw, NFFT_comp_mode::BACKWARDS_NC2C);
                }
            }
        }

        template <size_t D>
        void noncartesian_reconstruction<D>::deconstruct(
            cuNDArray<float_complext> *images,
            cuNDArray<float_complext> *data,
            cuNDArray<vector_td<float, D>> *traj,
            cuNDArray<float> *dcw)
        {

            // GadgetronTimer timer("Reconstruct");
            auto RO = data->get_size(0);
            auto E1E2 = data->get_size(1);
            auto CHA = data->get_size(2);

            // GDEBUG_STREAM("Channel: " << CHA);
            // GDEBUG_STREAM("E1E2:    " << E1E2);
            // GDEBUG_STREAM("RO:      " << RO);
            if (!this->isprocessed)
            {
                this->nfft_plan_->preprocess(*traj, NFFT_prep_mode::ALL);
                this->isprocessed = true;
            }
            this->nfft_plan_->compute(*images, *data, dcw, NFFT_comp_mode::FORWARDS_C2NC);
        }

        template <size_t D>
        std::tuple<cuNDArray<float_complext>,
                   cuNDArray<vector_td<float, D>>,
                   cuNDArray<float>,
                   std::vector<size_t>>
        noncartesian_reconstruction<D>::organize_data(
            std::vector<Core::Acquisition> *allAcq, std::vector<std::vector<size_t>> idx_phases)

        {

            auto sumall = 0;
            std::vector<size_t> nelem_idx;
            for (auto iph = 0; iph < idx_phases.size(); iph++)
            {
                sumall += idx_phases[iph].size();
                nelem_idx.push_back(idx_phases[iph].size());
            }
            std::vector<size_t> data_dims = {recon_params.RO, sumall, recon_params.numberChannels};
            std::vector<size_t> traj_dims = {recon_params.RO, sumall};
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            std::vector<size_t> number_elements;
            auto cutraj = cuNDArray<vector_td<float, D>>(traj_dims);
            auto cudcf = cuNDArray<float>(traj_dims);

            auto cuData = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(data_dims));

            for (auto jj = 0; jj < idx_phases.size(); jj++)
            {
                auto str_phase = std::accumulate(nelem_idx.begin(), nelem_idx.begin() + jj, size_t(0));
                number_elements.push_back(0);
                for (auto idx_ph = 0; idx_ph < idx_phases[jj].size(); idx_ph++)
                {
                    auto &[head, data, traj] = allAcq->at(idx_phases[jj][idx_ph]);
                    for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                    {
                        cudaMemcpy(cuData.get()->get_data_ptr() + stride * iCHA + data.get_size(0) * (str_phase + idx_ph),
                                   data.get_data_ptr() + data.get_size(0) * iCHA, // + totalnumInt,
                                   data.get_size(0) * sizeof(complext<float>), cudaMemcpyDefault);
                    }
                    // traj does have a fourth bit but I am ignoring it its the place where initial DCF is stored and
                    auto traj_dcw = nhlbi_toolbox::utils::separate_traj_and_dcw_all<D>(&(*traj), head.idx.kspace_encode_step_2, (float)recon_params.ematrixSize.z);
                    auto traj_sep = std::move(*std::get<0>(traj_dcw).get());
                    auto dcw_sep = std::move(*std::get<1>(traj_dcw).get());

                    number_elements[jj] = number_elements[jj] + data.get_size(0);
                    cudaMemcpy(cutraj.get_data_ptr() + data.get_size(0) * (str_phase + idx_ph),
                               traj_sep.get_data_ptr(), // + totalnumInt,
                               recon_params.RO * sizeof(vector_td<float, D>), cudaMemcpyDefault);

                    cudaMemcpy(cudcf.get_data_ptr() + data.get_size(0) * (str_phase + idx_ph),
                               dcw_sep.get_data_ptr(), // + totalnumInt,
                               data.get_size(0) * sizeof(float), cudaMemcpyDefault);
                }
            }

            cutraj.reshape(recon_params.RO * sumall);
            cudcf.reshape(recon_params.RO * sumall);

            return std::make_tuple(std::move(*cuData), std::move(cutraj), std::move(cudcf), number_elements);
        }

        template <size_t D>
        std::tuple<cuNDArray<float_complext>,
                   cuNDArray<vector_td<float, D>>,
                   cuNDArray<float>>
        noncartesian_reconstruction<D>::organize_data(
            std::vector<Core::Acquisition> *allAcq)

        {
            std::vector<size_t> data_dims = {recon_params.RO, allAcq->size(), recon_params.numberChannels};
            std::vector<size_t> traj_dims = {recon_params.RO, allAcq->size()};
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());

            auto cutraj = cuNDArray<vector_td<float, D>>(traj_dims);
            auto cudcw = cuNDArray<float>(traj_dims);

            auto cuData = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(data_dims));

            for (auto jj = 0; jj < allAcq->size(); jj++)
            {
                auto &[head, data, traj] = allAcq->at(jj);
                for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {

                    cudaMemcpy(cuData.get()->get_data_ptr() + stride * iCHA + data_dims[0] * jj,
                               data.get_data_ptr() + data.get_size(0) * iCHA, // + totalnumInt,
                               data.get_size(0) * sizeof(complext<float>), cudaMemcpyDefault);
                }
                // traj does have a fourth bit but I am ignoring it its the place where initial DCF is stored and

                auto traj_dcw = nhlbi_toolbox::utils::separate_traj_and_dcw_all<D>(&(*traj), head.idx.kspace_encode_step_2, (float)recon_params.ematrixSize.z);
                auto traj_sep = std::move(*std::get<0>(traj_dcw).get());
                auto dcw_sep = std::move(*std::get<1>(traj_dcw).get());

                cudaMemcpy(cutraj.get_data_ptr() + data_dims[0] * jj,
                           traj_sep.get_data_ptr(), // + totalnumInt,
                           recon_params.RO * sizeof(vector_td<float, D>), cudaMemcpyDefault);

                cudaMemcpy(cudcw.get_data_ptr() + data_dims[0] * jj,
                           dcw_sep.get_data_ptr(), // + totalnumInt,
                           recon_params.RO * sizeof(float), cudaMemcpyDefault);
            }

            // inefficiency need to copy traj back for DCF -> AJ promises in third person to fix it :D

            cutraj.reshape(recon_params.RO * allAcq->size());
            cudcw.reshape(recon_params.RO * allAcq->size());

            return std::make_tuple(std::move(*cuData), std::move(cutraj), std::move(cudcw));
        }

        template <size_t D>
        std::tuple<cuNDArray<float_complext>,
                   cuNDArray<vector_td<float, D>>>
        noncartesian_reconstruction<D>::organize_data(
            hoNDArray<float_complext> *data,
            hoNDArray<vector_td<float, D>> *traj)
        {
            size_t totalnumInt;
            if (recon_params.numberChannels == data->get_size(2))
                totalnumInt = data->get_size(1);
            else
                totalnumInt = data->get_size(2);

            std::vector<size_t> data_dims = {data->get_size(0), totalnumInt, recon_params.numberChannels};

            auto cuData = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(data_dims));
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());

            std::vector<size_t> non_flat_dims = {recon_params.RO, totalnumInt};

            auto traj_view = hoNDArray<vector_td<float, D>>(non_flat_dims, traj->get_data_ptr());

            std::vector<size_t> flat_dims = {recon_params.RO * totalnumInt};

            auto cutraj = cuNDArray<vector_td<float, D>>(traj);
            cutraj.reshape(recon_params.RO * totalnumInt);

            // Copy data  correctly
            if (recon_params.numberChannels == data->get_size(1)) // if the data is not permuted to and is of the shape RO CHA INT then use this code
            {
                for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    for (auto jj = 0; jj < totalnumInt; jj++)
                    {
                        cudaMemcpy(cuData.get()->get_data_ptr() + stride * iCHA + data_dims[0] * jj,
                                   data->get_data_ptr() + data->get_size(0) * iCHA + data->get_size(0) * data->get_size(1) * (jj), // + totalnumInt,
                                   data->get_size(0) * sizeof(complext<float>), cudaMemcpyDefault);
                    }
                }
            }
            else
            { // if the data is permuted to be RO INT CHA then use this code
                for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    for (auto jj = 0; jj < totalnumInt; jj++)
                    {
                        cudaMemcpy(cuData.get()->get_data_ptr() + stride * iCHA + data_dims[0] * jj,
                                   data->get_data_ptr() + data->get_size(0) * (jj) + data->get_size(0) * data->get_size(1) * iCHA,
                                   data->get_size(0) * sizeof(complext<float>), cudaMemcpyDefault);
                    }
                }
            }
            cutraj.squeeze();
            return std::make_tuple(std::move(*cuData), std::move(cutraj));
        }

        template <size_t D>
        cuNDArray<float> noncartesian_reconstruction<D>::estimate_dcf(cuNDArray<vector_td<float, 3>> *traj)
        {
            auto dims_traj = *(traj->get_dimensions());

            auto hoTraj = hoNDArray<vector_td<float, 3>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<vector_td<float, 3>>>((*traj).to_host())));

            std::vector<size_t> non_flat_dims = {recon_params.RO, traj->get_number_of_elements() / recon_params.RO};
            auto traj_view = hoNDArray<vector_td<float, 3>>(non_flat_dims, hoTraj.get_data_ptr());

            hoNDArray<float> hoflat_dcw = dcfO.estimate_DCF_slice(traj_view, image_dims_);
            float sum_all = asum(&hoflat_dcw);

            std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
            cuNDArray<float> flat_dcw = cuNDArray<float>(hoflat_dcw);

            // why is this consuming memory !
            float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_))) / sum_all;
            flat_dcw *= scale_factor;
            flat_dcw.reshape(flat_dims);

            return flat_dcw;
        }

        template <size_t D>
        cuNDArray<float> noncartesian_reconstruction<D>::estimate_dcf(cuNDArray<vector_td<float, 3>> *traj, cuNDArray<float> *dcf_in)
        {
            auto dims_traj = *(traj->get_dimensions());

            auto hoTraj = hoNDArray<vector_td<float, 3>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<vector_td<float, 3>>>((*traj).to_host())));
            auto hodcw = hoNDArray<float>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<float>>((*dcf_in).to_host())));
            std::vector<size_t> non_flat_dims = {recon_params.RO, traj->get_number_of_elements() / recon_params.RO};
            auto traj_view = hoNDArray<vector_td<float, 3>>(non_flat_dims, hoTraj.get_data_ptr());
            auto dcw_view = hoNDArray<float>(non_flat_dims, hodcw.get_data_ptr());

            hoNDArray<float> hoflat_dcw = dcfO.estimate_DCF_slice(traj_view, dcw_view, image_dims_);
            float sum_all = asum(&hoflat_dcw);

            std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
            cudaSetDevice(traj->get_device());
            cuNDArray<float> flat_dcw = cuNDArray<float>(hoflat_dcw);

            // why is this consuming memory !
            float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_))) / sum_all;
            flat_dcw *= scale_factor;
            flat_dcw.reshape(flat_dims);

            return flat_dcw;
        }

        template <size_t D>
        cuNDArray<float> noncartesian_reconstruction<D>::estimate_dcf(cuNDArray<vector_td<float, 2>> *traj, cuNDArray<float> *dcf_in)
        {
            auto dims_traj = *(traj->get_dimensions());

            //            auto hoTraj = hoNDArray<vector_td<float, 2>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<vector_td<float, 2>>>((*traj).to_host())));

            std::vector<size_t> non_flat_dims = {recon_params.RO, traj->get_number_of_elements() / recon_params.RO};
            auto traj_view = cuNDArray<vector_td<float, 2>>(non_flat_dims, traj->get_data_ptr());

            auto hoflat_dcw = dcfO.estimate_DCF(*traj, *dcf_in, image_dims_);
            float sum_all = asum(&hoflat_dcw);

            std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
            cudaSetDevice(traj->get_device());

            auto flat_dcw = cuNDArray<float>(hoflat_dcw);
            float scale_factor = float(prod(from_std_vector<size_t, 2>(image_dims_os_))) / sum_all;
            // float scale_factor = float(prod(from_std_vector<size_t, 2>(image_dims_os_)));
            flat_dcw *= scale_factor;
            flat_dcw.reshape(flat_dims);

            return flat_dcw;
        }
        template <size_t D>
        cuNDArray<float> noncartesian_reconstruction<D>::estimate_dcf(cuNDArray<vector_td<float, 2>> *traj)
        {
            auto dims_traj = *(traj->get_dimensions());

            //            auto hoTraj = hoNDArray<vector_td<float, 2>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<vector_td<float, 2>>>((*traj).to_host())));

            std::vector<size_t> non_flat_dims = {recon_params.RO, traj->get_number_of_elements() / recon_params.RO};
            auto traj_view = cuNDArray<vector_td<float, 2>>(non_flat_dims, traj->get_data_ptr());

            auto hoflat_dcw = dcfO.estimate_DCF(*traj, image_dims_);
            float sum_all = asum(&hoflat_dcw);
            cudaSetDevice(traj->get_device());

            std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
            auto flat_dcw = cuNDArray<float>(hoflat_dcw);

            float scale_factor = float(prod(from_std_vector<size_t, 2>(image_dims_os_))) / sum_all;
            flat_dcw *= scale_factor;
            flat_dcw.reshape(flat_dims);

            return flat_dcw;
        }

        template <size_t D>
        cuNDArray<float> noncartesian_reconstruction<D>::estimate_dcf(cuNDArray<vector_td<float, 3>> *traj, std::vector<size_t> number_elements)
        {
            cudaSetDevice(traj->get_device());

            auto cudcw = cuNDArray<float>((traj)->get_number_of_elements());

            auto hoTraj = hoNDArray<vector_td<float, 3>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<vector_td<float, 3>>>(traj->to_host())));

            for (auto ii = 0; ii < number_elements.size(); ii++)
            {
                std::vector<size_t> non_flat_dims = {recon_params.RO, number_elements[ii] / recon_params.RO}; // this is needed because we haven't reworked DCF estimation
                auto str_phase = std::accumulate(number_elements.begin(), number_elements.begin() + ii, size_t(0));

                auto traj_view = hoNDArray<vector_td<float, 3>>(non_flat_dims, hoTraj.get_data_ptr() + str_phase);

                auto hoflat_dcw = dcfO.estimate_DCF_slice(traj_view, image_dims_);
                float sum_all = asum(&hoflat_dcw);

                cudaMemcpy(cudcw.get_data_ptr() + str_phase,
                           hoflat_dcw.get_data_ptr(), // + totalnumInt,
                           number_elements[ii] * sizeof(float), cudaMemcpyDefault);

                auto dcf_view = cuNDArray<float>({number_elements[ii]}, cudcw.get_data_ptr() + str_phase);
                float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_))) / sum_all;
                dcf_view *= scale_factor;
                sqrt_inplace(&dcf_view);
            }

            return cudcw;
        }

        template <size_t D>
        std::vector<cuNDArray<float>> noncartesian_reconstruction<D>::estimate_dcf(std::vector<cuNDArray<vector_td<float, 3>>> *traj)
        {
            std::vector<cuNDArray<float>> cudcw;

            for (auto ii = 0; ii < (traj)->size(); ii++)
            {
                cudcw.push_back(std::move(estimate_dcf(&(traj->at(ii)))));
                auto hoflat_dcw = hoNDArray<float>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<float>>(cudcw.at(ii).to_host())));
                float sum_all = asum(&hoflat_dcw);

                float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_))) / sum_all;
                cudcw[ii] *= scale_factor;
                sqrt_inplace(&cudcw[ii]);
            }

            return cudcw;
        }
        template <size_t D>
        std::vector<cuNDArray<float>> noncartesian_reconstruction<D>::estimate_dcf(std::vector<cuNDArray<vector_td<float, 3>>> *traj, std::vector<cuNDArray<float>> *dcf_in)
        {
            std::vector<cuNDArray<float>> cudcw;

            for (auto ii = 0; ii < (traj)->size(); ii++)
            {
                cudcw.push_back(std::move(estimate_dcf(&(traj->at(ii)), &(dcf_in->at(ii)))));
                auto hoflat_dcw = hoNDArray<float>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<float>>(cudcw.at(ii).to_host())));
                float sum_all = asum(&hoflat_dcw);

                float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_))) / sum_all;
                cudcw[ii] *= scale_factor;
                sqrt_inplace(&cudcw[ii]);
            }

            return cudcw;
        }

        template class noncartesian_reconstruction<2>;
        template class noncartesian_reconstruction<3>;
        template cuNDArray<float_complext> noncartesian_reconstruction<2>::crop_to_recondims(cuNDArray<float_complext> &input);
        template cuNDArray<float_complext> noncartesian_reconstruction<3>::crop_to_recondims(cuNDArray<float_complext> &input);
        template cuNDArray<float> noncartesian_reconstruction<2>::crop_to_recondims(cuNDArray<float> &input);
        template cuNDArray<float> noncartesian_reconstruction<3>::crop_to_recondims(cuNDArray<float> &input);
        template std::vector<cuNDArray<float>> noncartesian_reconstruction<2>::arraytovector(cuNDArray<float> *inputArray, std::vector<size_t> number_elements);
        template std::vector<cuNDArray<floatd2>> noncartesian_reconstruction<2>::arraytovector(cuNDArray<floatd2> *inputArray, std::vector<size_t> number_elements);
        template std::vector<cuNDArray<float>> noncartesian_reconstruction<3>::arraytovector(cuNDArray<float> *inputArray, std::vector<size_t> number_elements);
        template std::vector<cuNDArray<floatd3>> noncartesian_reconstruction<3>::arraytovector(cuNDArray<floatd3> *inputArray, std::vector<size_t> number_elements);
        template std::vector<cuNDArray<float_complext>> noncartesian_reconstruction<2>::arraytovector(cuNDArray<float_complext> *inputArray, std::vector<size_t> number_elements);
    }

}
