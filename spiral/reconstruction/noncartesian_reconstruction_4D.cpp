#include "noncartesian_reconstruction_4D.h"
#include "gadgetron/cuNlcgSolver.h"
#include "gpuRegistration.cuh"
#include <util_functions.h>

using namespace Gadgetron;
namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        cuNDArray<float_complext> noncartesian_reconstruction_4D::reconstruct(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm)
        {
            nhlbi_toolbox::utils::enable_peeraccess();
            auto data_dims = *data->get_dimensions();
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            cudaSetDevice(data->get_device());
            
            // prep data and dcw - doing this data save in memory to prevent data from being affected by recon.
            hoNDArray<float_complext> hodata(*data->get_dimensions());
            cudaMemcpy(hodata.get_data_ptr(),data->get_data_ptr(),data->get_number_of_elements()*sizeof(float_complext),cudaMemcpyDeviceToHost);

            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

                for (auto iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview *= (*dcw)[ii];
                }
            }
            auto E_ = boost::shared_ptr<cuNonCartesianTSenseOperator<float, 3>>(new cuNonCartesianTSenseOperator<float, 3>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            nhlbi_toolbox::cuGpBbSolver<float_complext> solver_;

            solver_.set_max_iterations(recon_params.iterations);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_number_of_elements()};

            // cuNDArray<float_complext> reg_image(recon_dims);

            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(*dcw);
            E_->preprocess(*traj);

            // auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            // E_->mult_MH(data, x0.get());
            solver_.set_encoding_operator(E_);

            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            // solver_.set_x0(x0);

            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

            _precon_weights = sum(abs_square(csm.get()).get(), 3);
            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix
            cuNDArray<float> _precon_weights_cropped = this->crop_to_recondims<float>(*_precon_weights);
            reciprocal_sqrt_inplace(&_precon_weights_cropped);
            // reciprocal_sqrt_inplace(_precon_weights.get());
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                  *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_number_of_elements(), recon_params.numberChannels};
            recon_dims.pop_back();
            if (recon_params.shots_per_time.get_number_of_elements() > 1)
            {
                GDEBUG_STREAM("Setup temporal TV");
                boost::shared_ptr<cuPartialDerivativeOperator2<float_complext, 4>>
                    Rt(new cuPartialDerivativeOperator2<float_complext, 4>());

                Rt->set_weight(recon_params.lambda_time);

                Rt->set_domain_dimensions(&recon_dims);
                Rt->set_codomain_dimensions(&recon_dims);
                solver_.add_regularization_operator(Rt, recon_params.norm);
            }
            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rx(new cuPartialDerivativeOperator<float_complext, 4>(0));
            Rx->set_weight(recon_params.lambda_spatial);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Ry(new cuPartialDerivativeOperator<float_complext, 4>(1));
            Ry->set_weight(recon_params.lambda_spatial);
            Ry->set_domain_dimensions(&recon_dims);
            Ry->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rz(new cuPartialDerivativeOperator<float_complext, 4>(2));
            Rz->set_weight(recon_params.lambda_spatial * (resx * resx) / (resz * resz));
            Rz->set_domain_dimensions(&recon_dims);
            Rz->set_codomain_dimensions(&recon_dims);

            // TV->set_domain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            solver_.add_regularization_operator(Ry, recon_params.norm);
            solver_.add_regularization_operator(Rz, recon_params.norm);

            // auto gpus_input_possible = nhlbi_toolbox::utils::FindCudaDevices(data->get_number_of_elements() * 4 * 2 * 4);

            // gpus_input_possible.erase(std::remove(gpus_input_possible.begin(), gpus_input_possible.end(), data->get_device()), gpus_input_possible.end());

            int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();

            GDEBUG_STREAM("Data_device:" << data->get_device());
            GDEBUG_STREAM("gpus_input_possible[0]:" << selectedDevice);

            if (selectedDevice != data->get_device())
            {
                std::vector<int> gpus_input({data->get_device(), selectedDevice});
                solver_.set_gpus(gpus_input);
            }
            auto reg_image = *solver_.solve(data);

            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z, reg_image.get_size(3)}; // Cropped to size of Recon Matrix
            cuNDArray<float_complext> images_cropped(recon_dims);

            crop<float_complext, 4>(uint64d4(0, 0, (image_dims_[2] - recon_params.rmatrixSize.z) / 2, 0),
                                    uint64d4(image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z, reg_image.get_size(3)),
                                    reg_image,
                                    images_cropped);

            cudaMemcpy(data->get_data_ptr(),hodata.get_data_ptr(),data->get_number_of_elements()*sizeof(float_complext),cudaMemcpyHostToDevice);
            // de-prep data
            // for (auto ii = 0; ii < dcw->size(); ii++)
            // {
            //     auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

            //     for (auto iCHA = 0; iCHA < data->get_size(data->get_number_of_dimensions() - 1); iCHA++)
            //     {
            //         auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
            //         dataview /= (*dcw)[ii];
            //     }
            // }

            return images_cropped;
        }

        cuNDArray<float_complext> noncartesian_reconstruction_4D::reconstruct_nlcg(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm)
        {
            nhlbi_toolbox::utils::enable_peeraccess();
            auto data_dims = *data->get_dimensions();
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            cudaSetDevice(data->get_device());

            // prep data and dcw - doing this data save in memory to prevent data from being affected by recon.
            hoNDArray<float_complext> hodata(*data->get_dimensions());
            cudaMemcpy(hodata.get_data_ptr(),data->get_data_ptr(),data->get_number_of_elements()*sizeof(float_complext),cudaMemcpyDeviceToHost);

            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

                for (auto iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview *= (*dcw)[ii];
                }
            }
            auto E_ = boost::shared_ptr<cuNonCartesianTSenseOperator<float, 3>>(new cuNonCartesianTSenseOperator<float, 3>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            cuNlcgSolver<float_complext> solver_;

            solver_.set_max_iterations(recon_params.iterations);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_number_of_elements()};

            cuNDArray<float_complext> reg_image(recon_dims);

            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(*dcw);
            E_->preprocess(*traj);

            // auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            // E_->mult_MH(data, x0.get());
            solver_.set_encoding_operator(E_);

            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            // solver_.set_x0(x0);

            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

            _precon_weights = sum(abs_square(csm.get()).get(), 3);
            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix
            cuNDArray<float> _precon_weights_cropped = this->crop_to_recondims<float>(*_precon_weights);
            reciprocal_sqrt_inplace(&_precon_weights_cropped);
            // reciprocal_sqrt_inplace(_precon_weights.get());
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                  *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_number_of_elements(), recon_params.numberChannels};
            recon_dims.pop_back();

            boost::shared_ptr<cuPartialDerivativeOperator2<float_complext, 4>>
                Rt(new cuPartialDerivativeOperator2<float_complext, 4>());

            Rt->set_weight(recon_params.lambda_time);

            Rt->set_domain_dimensions(&recon_dims);
            Rt->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rt, recon_params.norm);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rx(new cuPartialDerivativeOperator<float_complext, 4>(0));
            Rx->set_weight(recon_params.lambda_spatial);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Ry(new cuPartialDerivativeOperator<float_complext, 4>(1));
            Ry->set_weight(recon_params.lambda_spatial);
            Ry->set_domain_dimensions(&recon_dims);
            Ry->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rz(new cuPartialDerivativeOperator<float_complext, 4>(2));
            Rz->set_weight(recon_params.lambda_spatial * (resx * resx) / (resz * resz));
            Rz->set_domain_dimensions(&recon_dims);
            Rz->set_codomain_dimensions(&recon_dims);

            // TV->set_domain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            solver_.add_regularization_operator(Ry, recon_params.norm);
            solver_.add_regularization_operator(Rz, recon_params.norm);

            reg_image = *solver_.solve(data);

            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z, reg_image.get_size(3)}; // Cropped to size of Recon Matrix
            cuNDArray<float_complext> images_cropped(recon_dims);

            crop<float_complext, 4>(uint64d4(0, 0, (image_dims_[2] - recon_params.rmatrixSize.z) / 2, 0),
                                    uint64d4(image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z, reg_image.get_size(3)),
                                    reg_image,
                                    images_cropped);

            cudaMemcpy(data->get_data_ptr(),hodata.get_data_ptr(),data->get_number_of_elements()*sizeof(float_complext),cudaMemcpyHostToDevice);
            // de-prep data
            // for (auto ii = 0; ii < dcw->size(); ii++)
            // {
            //     auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

            //     for (auto iCHA = 0; iCHA < data->get_size(data->get_number_of_dimensions() - 1); iCHA++)
            //     {
            //         auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
            //         dataview /= (*dcw)[ii];
            //     }
            // }
            return images_cropped;
        }

        cuNDArray<float_complext> noncartesian_reconstruction_4D::reconstruct_fc(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm,
            cuNDArray<float_complext> *combination_weights,
            std::vector<cuNDArray<float>> &scaled_time_vec,
            arma::fvec fbins)
        {
            nhlbi_toolbox::utils::enable_peeraccess();
            auto data_dims = *data->get_dimensions();
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            cudaSetDevice(data->get_device());

            // prep data and dcw - doing this data save in memory to prevent data from being affected by recon.
            hoNDArray<float_complext> hodata(*data->get_dimensions());
            cudaMemcpy(hodata.get_data_ptr(),data->get_data_ptr(),data->get_number_of_elements()*sizeof(float_complext),cudaMemcpyDeviceToHost);
            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

                for (auto iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview *= (*dcw)[ii];
                }
            }

            std::vector<size_t> cwdims = {image_dims_[0], image_dims_[1], image_dims_[2], fbins.n_elem};
            cuNDArray<float_complext> padded_cw(cwdims);
            if (fbins.n_elem > 1 && combination_weights->get_size(3) > 1)
                padded_cw = pad<float_complext, 4>(uint64d4(image_dims_[0], image_dims_[1], image_dims_[2], fbins.n_elem), combination_weights, float_complext(0));
            else
                padded_cw = pad<float_complext, 3>(uint64d3(image_dims_[0], image_dims_[1], image_dims_[2]), combination_weights, float_complext(0));
            padded_cw.squeeze();

            auto E_ = boost::shared_ptr<cuNonCartesianTSenseOperator_fc<float, 3>>(new cuNonCartesianTSenseOperator_fc<float, 3>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            cuGpBbSolver<float_complext> solver_;

            solver_.set_max_iterations(recon_params.iterations);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_number_of_elements()};

            cuNDArray<float_complext> reg_image(recon_dims);

            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(*dcw);
            E_->set_recon_params(recon_params);
            E_->set_combination_weights(&padded_cw);
            E_->set_scaled_time(scaled_time_vec);
            E_->set_fbins(fbins);

            // do everything before preprocess
            E_->preprocess(*traj);

            auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            E_->mult_MH(data, x0.get());
            solver_.set_encoding_operator(E_);

            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            solver_.set_x0(x0);

            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

            _precon_weights = sum(abs_square(csm.get()).get(), 3);
            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix
            cuNDArray<float> _precon_weights_cropped = this->crop_to_recondims<float>(*_precon_weights);

            // crop<float, 3>(uint64d3(0, 0, (image_dims_[2] - recon_params.ematrixSize.z) / 2),
            //                uint64d3(image_dims_[0], image_dims_[1], recon_params.ematrixSize.z),
            //                *_precon_weights,
            //                _precon_weights_cropped);

            reciprocal_sqrt_inplace(&_precon_weights_cropped);
            // reciprocal_sqrt_inplace(_precon_weights.get());
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                  *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_number_of_elements(), recon_params.numberChannels};
            recon_dims.pop_back();

            boost::shared_ptr<cuPartialDerivativeOperator2<float_complext, 4>>
                Rt(new cuPartialDerivativeOperator2<float_complext, 4>());

            Rt->set_weight(recon_params.lambda_time);

            Rt->set_domain_dimensions(&recon_dims);
            Rt->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rt, recon_params.norm);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rx(new cuPartialDerivativeOperator<float_complext, 4>(0));
            Rx->set_weight(recon_params.lambda_spatial);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Ry(new cuPartialDerivativeOperator<float_complext, 4>(1));
            Ry->set_weight(recon_params.lambda_spatial);
            Ry->set_domain_dimensions(&recon_dims);
            Ry->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rz(new cuPartialDerivativeOperator<float_complext, 4>(2));
            Rz->set_weight(recon_params.lambda_spatial * (resx * resx) / (resz * resz));
            Rz->set_domain_dimensions(&recon_dims);
            Rz->set_codomain_dimensions(&recon_dims);

            // TV->set_domain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            solver_.add_regularization_operator(Ry, recon_params.norm);
            solver_.add_regularization_operator(Rz, recon_params.norm);

            reg_image = *solver_.solve(data);

            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z, reg_image.get_size(3)}; // Cropped to size of Recon Matrix
            cuNDArray<float_complext> images_cropped = this->crop_to_recondims<float_complext>(reg_image);

            cudaMemcpy(data->get_data_ptr(),hodata.get_data_ptr(),data->get_number_of_elements()*sizeof(float_complext),cudaMemcpyHostToDevice);
            // de-prep data
            // for (auto ii = 0; ii < dcw->size(); ii++)
            // {
            //     auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

            //     for (auto iCHA = 0; iCHA < data->get_size(data->get_number_of_dimensions() - 1); iCHA++)
            //     {
            //         auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
            //         dataview /= (*dcw)[ii];
            //     }
            // }
            return images_cropped;
        }

        cuNDArray<float_complext> noncartesian_reconstruction_4D::reconstructiMOCO(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm,
            cuNDArray<float> *def,
            cuNDArray<float> *invdef)
        {
            nhlbi_toolbox::utils::enable_peeraccess();
            auto data_dims = *data->get_dimensions();
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            cudaSetDevice(data->get_device());

            arma::fvec fbins(1);
            fbins.ones();
            std::vector<cuNDArray<float>> scaled_time_vec;

            // prep data and dcw - doing this data save in memory to prevent data from being affected by recon.
            hoNDArray<float_complext> hodata(*data->get_dimensions());
            cudaMemcpy(hodata.get_data_ptr(),data->get_data_ptr(),data->get_number_of_elements()*sizeof(float_complext),cudaMemcpyDeviceToHost);
            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];
                auto tmp = cuNDArray<float>(*(*dcw)[ii].get_dimensions());
                fill(&tmp, float(0.0));
                scaled_time_vec.push_back(tmp);

                for (auto iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview *= (*dcw)[ii];
                }
            }

            std::vector<size_t> cwdims = {image_dims_[0], image_dims_[1], image_dims_[2], fbins.n_elem};
            cuNDArray<float_complext> padded_cw(cwdims);
            fill(&padded_cw, float_complext(1.0, 0.0));
            padded_cw.squeeze();

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2]};

            cudaSetDevice(def->get_device());
            std::vector<cuNDArray<float>> padded_def;
            std::vector<cuNDArray<float>> padded_invdef;

            auto defDims = *def->get_dimensions();
            // if (defDims[defDims.size() - 1] > 1)
            defDims.pop_back(); // remove time

            stride = std::accumulate(defDims.begin(), defDims.end(), 1,
                                     std::multiplies<size_t>());

            recon_dims = {image_dims_[0], image_dims_[1], 3, image_dims_[2]};

            for (auto ii = 0; ii < def->get_size(4); ii++)
            {
                auto defView = cuNDArray<float>(defDims, def->data() + stride * ii);
                auto intdefView = cuNDArray<float>(defDims, invdef->data() + stride * ii);

                padded_def.push_back(padDeformations(defView, recon_dims));
                padded_invdef.push_back(padDeformations(intdefView, recon_dims));
            }
            def->clear();
            invdef->clear();
            cudaSetDevice(data->get_device());
            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2]};

            auto E_ = boost::shared_ptr<cuNonCartesianMOCOOperator_fc<float, 3>>(new cuNonCartesianMOCOOperator_fc<float, 3>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            nhlbi_toolbox::cuGpBbSolver<float_complext> solver_;
            // cuSbcCgSolver<float_complext> solver_;

            solver_.set_max_iterations(recon_params.iterations);
            // solver_.set_max_outer_iterations(recon_params.iterations);
            // solver_.set_max_inner_iterations(recon_params.iterations_inner);
            cuNDArray<float_complext> reg_image(recon_dims);

            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            GDEBUG_STREAM("GPU _4d:" << (*dcw)[0].get_device());

            E_->set_dcw(*dcw);
            E_->set_recon_params(recon_params);
            E_->set_combination_weights(&padded_cw);
            E_->set_scaled_time(scaled_time_vec);
            E_->set_fbins(fbins);

            cudaSetDevice(padded_def[0].get_device());
            E_->set_forward_deformation(&padded_def);
            E_->set_backward_deformation(&padded_invdef);
            cudaSetDevice(data->get_device());
            // do everything before preprocess
            E_->preprocess(*traj);

            auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            E_->mult_MH(data, x0.get());
            solver_.set_encoding_operator(E_);

            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            solver_.set_x0(x0);

            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

            _precon_weights = sum(abs_square(csm.get()).get(), 3);
            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix

            cuNDArray<float> _precon_weights_cropped = this->crop_to_recondims<float>(*_precon_weights);

            // crop<float, 3>(uint64d3(0, 0, (image_dims_[2] - recon_params.ematrixSize.z) / 2),
            //                uint64d3(image_dims_[0], image_dims_[1], recon_params.ematrixSize.z),
            //                *_precon_weights,
            //                _precon_weights_cropped);

            reciprocal_sqrt_inplace(&_precon_weights_cropped);
            // reciprocal_sqrt_inplace(_precon_weights.get());
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                  *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.numberChannels};
            recon_dims.pop_back();

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 3>>
                Rx(new cuPartialDerivativeOperator<float_complext, 3>(0));
            Rx->set_weight(recon_params.lambda_spatial);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 3>>
                Ry(new cuPartialDerivativeOperator<float_complext, 3>(1));
            Ry->set_weight(recon_params.lambda_spatial);
            Ry->set_domain_dimensions(&recon_dims);
            Ry->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 3>>
                Rz(new cuPartialDerivativeOperator<float_complext, 3>(2));
            Rz->set_weight(recon_params.lambda_spatial * (resx * resx) / (resz * resz));
            Rz->set_domain_dimensions(&recon_dims);
            Rz->set_codomain_dimensions(&recon_dims);

            // TV->set_domain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            solver_.add_regularization_operator(Ry, recon_params.norm);
            solver_.add_regularization_operator(Rz, recon_params.norm);
            // solver_.add_regularization_group_operator(Rx);
            // solver_.add_regularization_group_operator(Ry);
            // solver_.add_regularization_group_operator(Rz);
            // solver_.add_group(recon_params.norm);

            GDEBUG_STREAM("Data_device:" << data->get_device());
            GDEBUG_STREAM("padded_def[0].get_device():" << padded_def[0].get_device());

            int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();

            if (selectedDevice != data->get_device())
            {
                std::vector<int> gpus_input({data->get_device(), selectedDevice});
                solver_.set_gpus(gpus_input);
            }
            reg_image = *solver_.solve(data);

            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z, reg_image.get_size(3)}; // Cropped to size of Recon Matrix
            cuNDArray<float_complext> images_cropped = this->crop_to_recondims<float_complext>(reg_image);

            // crop<float_complext, 4>(uint64d4(0, 0, (image_dims_[2] - recon_params.ematrixSize.z) / 2, 0),
            //                         uint64d4(image_dims_[0], image_dims_[1], recon_params.ematrixSize.z, reg_image.get_size(3)),
            //                         reg_image,
            //                         images_cropped);

            cudaMemcpy(data->get_data_ptr(),hodata.get_data_ptr(),data->get_number_of_elements()*sizeof(float_complext),cudaMemcpyHostToDevice);
            // de-prep data
            // stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            // for (auto ii = 0; ii < dcw->size(); ii++)
            // {
            //     auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

            //     for (auto iCHA = 0; iCHA < data->get_size(data->get_number_of_dimensions() - 1); iCHA++)
            //     {
            //         auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
            //         dataview /= (*dcw)[ii];
            //     }
            // }

            return images_cropped;
        }
        cuNDArray<float_complext> noncartesian_reconstruction_4D::reconstructiMOCO_fc(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm,
            cuNDArray<float_complext> *combination_weights,
            std::vector<cuNDArray<float>> &scaled_time_vec,
            arma::fvec fbins,
            cuNDArray<float> *def,
            cuNDArray<float> *invdef)
        {
            nhlbi_toolbox::utils::enable_peeraccess();
            auto data_dims = *data->get_dimensions();
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            cudaSetDevice(data->get_device());

            // prep data and dcw - doing this data save in memory to prevent data from being affected by recon.
            hoNDArray<float_complext> hodata(*data->get_dimensions());
            cudaMemcpy(hodata.get_data_ptr(),data->get_data_ptr(),data->get_number_of_elements()*sizeof(float_complext),cudaMemcpyDeviceToHost);
            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

                for (auto iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview *= (*dcw)[ii];
                }
            }

            std::vector<size_t> cwdims = {image_dims_[0], image_dims_[1], image_dims_[2], fbins.n_elem};
            cuNDArray<float_complext> padded_cw(cwdims);
            if (fbins.n_elem > 1 && combination_weights->get_size(3) > 1)
                padded_cw = pad<float_complext, 4>(uint64d4(image_dims_[0], image_dims_[1], image_dims_[2], fbins.n_elem), combination_weights, float_complext(0));
            else
                padded_cw = pad<float_complext, 3>(uint64d3(image_dims_[0], image_dims_[1], image_dims_[2]), combination_weights, float_complext(0));
            padded_cw.squeeze();

            combination_weights->clear();

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2]};

            std::vector<cuNDArray<float>> padded_def;
            std::vector<cuNDArray<float>> padded_invdef;

            auto defDims = *def->get_dimensions();
            if (defDims[defDims.size() - 1] > 1)
                defDims.pop_back();

            stride = std::accumulate(defDims.begin(), defDims.end(), 1,
                                     std::multiplies<size_t>());

            recon_dims = {image_dims_[0], image_dims_[1], 3, image_dims_[2]};

            for (auto ii = 0; ii < def->get_size(4); ii++)
            {
                auto defView = cuNDArray<float>(defDims, def->data() + stride * ii);
                auto intdefView = cuNDArray<float>(defDims, invdef->data() + stride * ii);

                padded_def.push_back(padDeformations(defView, recon_dims));
                padded_invdef.push_back(padDeformations(intdefView, recon_dims));
            }
            def->clear();
            invdef->clear();

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2]};

            auto E_ = boost::shared_ptr<cuNonCartesianMOCOOperator_fc<float, 3>>(new cuNonCartesianMOCOOperator_fc<float, 3>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            cuGpBbSolver<float_complext> solver_;
            // cuSbcCgSolver<float_complext> solver_;

            solver_.set_max_iterations(recon_params.iterations);
            // solver_.set_max_outer_iterations(recon_params.iterations);
            // solver_.set_max_inner_iterations(recon_params.iterations_inner);
            cuNDArray<float_complext> reg_image(recon_dims);

            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(*dcw);
            E_->set_recon_params(recon_params);
            E_->set_combination_weights(&padded_cw);
            E_->set_scaled_time(scaled_time_vec);
            E_->set_fbins(fbins);
            E_->set_forward_deformation(&padded_def);
            E_->set_backward_deformation(&padded_invdef);
            // do everything before preprocess
            E_->preprocess(*traj);

            auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            E_->mult_MH(data, x0.get());
            solver_.set_encoding_operator(E_);

            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            solver_.set_x0(x0);

            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

            _precon_weights = sum(abs_square(csm.get()).get(), 3);
            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix

            cuNDArray<float> _precon_weights_cropped = this->crop_to_recondims<float>(*_precon_weights);

            // crop<float, 3>(uint64d3(0, 0, (image_dims_[2] - recon_params.ematrixSize.z) / 2),
            //                uint64d3(image_dims_[0], image_dims_[1], recon_params.ematrixSize.z),
            //                *_precon_weights,
            //                _precon_weights_cropped);

            reciprocal_sqrt_inplace(&_precon_weights_cropped);
            // reciprocal_sqrt_inplace(_precon_weights.get());
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                  *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.numberChannels};
            recon_dims.pop_back();

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 3>>
                Rx(new cuPartialDerivativeOperator<float_complext, 3>(0));
            Rx->set_weight(recon_params.lambda_spatial);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 3>>
                Ry(new cuPartialDerivativeOperator<float_complext, 3>(1));
            Ry->set_weight(recon_params.lambda_spatial);
            Ry->set_domain_dimensions(&recon_dims);
            Ry->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 3>>
                Rz(new cuPartialDerivativeOperator<float_complext, 3>(2));
            Rz->set_weight(recon_params.lambda_spatial * (resx * resx) / (resz * resz));
            Rz->set_domain_dimensions(&recon_dims);
            Rz->set_codomain_dimensions(&recon_dims);

            // TV->set_domain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            solver_.add_regularization_operator(Ry, recon_params.norm);
            solver_.add_regularization_operator(Rz, recon_params.norm);
            // solver_.add_regularization_group_operator(Rx);
            // solver_.add_regularization_group_operator(Ry);
            // solver_.add_regularization_group_operator(Rz);
            // solver_.add_group(recon_params.norm);
            GDEBUG_STREAM("Data_device:" << data->get_device());
            GDEBUG_STREAM("padded_def[0].get_device():" << padded_def[0].get_device());

            int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();

            if (selectedDevice != data->get_device())
            {
                std::vector<int> gpus_input({data->get_device(), selectedDevice});
                solver_.set_gpus(gpus_input);
            }

            reg_image = *solver_.solve(data);

            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z, reg_image.get_size(3)}; // Cropped to size of Recon Matrix
            cuNDArray<float_complext> images_cropped = this->crop_to_recondims<float_complext>(reg_image);

            // crop<float_complext, 4>(uint64d4(0, 0, (image_dims_[2] - recon_params.ematrixSize.z) / 2, 0),
            //                         uint64d4(image_dims_[0], image_dims_[1], recon_params.ematrixSize.z, reg_image.get_size(3)),
            //                         reg_image,
            //                         images_cropped);

            cudaMemcpy(data->get_data_ptr(),hodata.get_data_ptr(),data->get_number_of_elements()*sizeof(float_complext),cudaMemcpyHostToDevice);
            // de-prep data
            // stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            // for (auto ii = 0; ii < dcw->size(); ii++)
            // {
            //     auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

            //     for (auto iCHA = 0; iCHA < data->get_size(data->get_number_of_dimensions() - 1); iCHA++)
            //     {
            //         auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
            //         dataview /= (*dcw)[ii];
            //     }
            // }
            return images_cropped;
        }
        cuNDArray<float> noncartesian_reconstruction_4D::padDeformations(cuNDArray<float> deformation, std::vector<size_t> size_deformation)
        {
            cudaSetDevice(deformation.get_device());
            cuNDArray<float> tempDef(size_deformation);

            auto offsetSlice_dest = size_deformation[3] - (size_deformation[3] - deformation.get_size(2)) / 2;
            auto numSlices = (size_deformation[3] - deformation.get_size(2)) / 2;
            auto defTemp = permute(deformation, {0, 1, 3, 2});
            // copy to end the first numslices
            cudaMemcpy(tempDef.get_data_ptr() + defTemp.get_size(0) * defTemp.get_size(1) * defTemp.get_size(2) * offsetSlice_dest,
                       defTemp.get_data_ptr(),
                       defTemp.get_size(0) * defTemp.get_size(1) * defTemp.get_size(2) * numSlices * sizeof(float), cudaMemcpyDefault);
            // copy to middle all the slice
            cudaMemcpy(tempDef.get_data_ptr() + defTemp.get_size(0) * defTemp.get_size(1) * defTemp.get_size(2) * numSlices,
                       defTemp.get_data_ptr(),
                       defTemp.get_size(0) * defTemp.get_size(1) * defTemp.get_size(2) * deformation.get_size(2) * sizeof(float), cudaMemcpyDefault);
            // copy to start last n slices
            cudaMemcpy(tempDef.get_data_ptr(),
                       defTemp.get_data_ptr(),
                       defTemp.get_size(0) * defTemp.get_size(1) * defTemp.get_size(2) * numSlices * sizeof(float), cudaMemcpyDefault);

            return (permute(tempDef, {0, 1, 3, 2}));
        }
        std::tuple<cuNDArray<float_complext>,
                   std::vector<cuNDArray<floatd3>>,
                   std::vector<cuNDArray<float>>>
        noncartesian_reconstruction_4D::organize_data(
            hoNDArray<float_complext> *data,
            hoNDArray<floatd3> *traj,
            hoNDArray<float> *dcw)
        {
            auto totalnumInt = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.end(), size_t(0));
            std::vector<size_t> data_dims = {data->get_size(0), totalnumInt, recon_params.numberChannels};

            auto cuData = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(data_dims));
            std::vector<cuNDArray<floatd3>> cuTrajVec;
            std::vector<cuNDArray<float>> cuDCWVec;

            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());

            auto cutraj = cuNDArray<floatd3>(traj);
            auto cudcw = cuNDArray<float>(dcw);

            for (auto ii = 0; ii < recon_params.shots_per_time.get_number_of_elements(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

                std::vector<size_t> non_flat_dims = {data_dims[0], *(recon_params.shots_per_time.begin() + ii)};

                auto traj_view = hoNDArray<floatd3>(non_flat_dims, traj->get_data_ptr() + inter_acc);
                auto hoflat_dcw = dcfO.estimate_DCF_slice(traj_view, image_dims_,true);
                // auto hoflat_dcw = hoNDArray<float>(non_flat_dims);
                // hoflat_dcw.fill(1.0);
                std::vector<size_t> flat_dims = {data_dims[0] * *(recon_params.shots_per_time.begin() + ii)};

                cuNDArray<floatd3> flat_traj(flat_dims, cutraj.get_data_ptr() + inter_acc);
                // cuNDArray<float> flat_dcw(flat_dims, cudcw.get_data_ptr() + inter_acc);

                cuTrajVec.push_back(flat_traj);
                cuDCWVec.push_back(cuNDArray<float>(hoflat_dcw));
                float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_))) / asum((&cuDCWVec[ii]));
                cuDCWVec[ii] *= scale_factor;
                sqrt_inplace(&cuDCWVec[ii]);
                if (recon_params.numberChannels == data->get_size(1)) // if the data is not permuted to and is of the shape RO CHA INT then use this code
                {
                    for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                    {
                        for (auto jj = 0; jj < *(recon_params.shots_per_time.begin() + ii); jj++)
                        {
                            cudaMemcpy(cuData.get()->get_data_ptr() + inter_acc + stride * iCHA + data_dims[0] * jj,
                                       data->get_data_ptr() + data->get_size(0) * iCHA + data->get_size(0) * data->get_size(1) * (jj) + inter_acc,
                                       data->get_size(0) * sizeof(complext<float>), cudaMemcpyDefault);
                        }
                        // auto dataview = cuNDArray<complext<float>>(cuTrajVec[ii].get_dimensions(), cuData.get()->data() + inter_acc + stride * iCHA);
                        // dataview *= cuDCWVec[ii];
                    }
                }
                else
                { // if the data is permuted to be RO INT CHA then use this code
                    for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                    {
                        for (auto jj = 0; jj < *(recon_params.shots_per_time.begin() + ii); jj++)
                        {
                            cudaMemcpy(cuData.get()->get_data_ptr() + inter_acc + stride * iCHA + data_dims[0] * jj,
                                       data->get_data_ptr() + data->get_size(0) * (jj) + inter_acc + data->get_size(0) * data->get_size(1) * iCHA,
                                       data->get_size(0) * sizeof(complext<float>), cudaMemcpyDefault);
                        }
                        // auto dataview = cuNDArray<complext<float>>(cuTrajVec[ii].get_dimensions(), cuData.get()->data() + inter_acc + stride * iCHA);
                        // dataview *= cuDCWVec[ii];
                    }
                }
            }
            return std::make_tuple(std::move(*cuData), std::move(cuTrajVec), std::move(cuDCWVec));
        }

        std::tuple<std::vector<cuNDArray<floatd3>>,
                   std::vector<cuNDArray<float>>>
        noncartesian_reconstruction_4D::organize_data(
            cuNDArray<floatd3> *traj,
            cuNDArray<float> *dcw,
            std::vector<size_t> number_elements)

        {
            std::vector<cuNDArray<floatd3>> cuTrajVec;
            std::vector<cuNDArray<float>> cuDCWVec;

            for (auto iph = 0; iph < number_elements.size(); iph++)
            {

                auto traj_view = cuNDArray<floatd3>(number_elements, (*traj).get_data_ptr());
                auto dcw_view = cuNDArray<float>(number_elements, (*dcw).get_data_ptr());
                cuTrajVec.push_back(std::move(traj_view));
                cuDCWVec.push_back(std::move(dcw_view));
            }
            return (std::make_tuple(std::move(cuTrajVec), std::move(cuDCWVec)));
        }

        std::tuple<cuNDArray<float_complext>,
                   std::vector<cuNDArray<floatd3>>,
                   std::vector<cuNDArray<float>>>
        noncartesian_reconstruction_4D::organize_data(
            std::vector<hoNDArray<float_complext>> *dataVector,
            std::vector<hoNDArray<floatd3>> *trajVector,
            std::vector<hoNDArray<float>> *dcwVector)
        {
            // find total number of Interleaves
            auto totalnumInt = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.end(), size_t(0));
            std::vector<size_t> data_dims = {recon_params.RO, totalnumInt, recon_params.numberChannels};

            auto cuData = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(data_dims));
            std::vector<cuNDArray<floatd3>> cuTrajVec;
            std::vector<cuNDArray<float>> cuDCWVec;

            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            {
                GadgetronTimer timer("DataCopy:");

                for (auto ii = 0; ii < dataVector->size(); ii++)
                {
                    auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

                    std::vector<size_t> flat_dims = {(*trajVector)[ii].get_number_of_elements()};
                    cuNDArray<floatd3> nonFtraj((*trajVector)[ii]);
                    cuNDArray<floatd3> flat_traj(flat_dims, nonFtraj.get_data_ptr());

                    std::vector<size_t> non_flat_dims = {recon_params.RO, *(recon_params.shots_per_time.begin() + ii)};

                    auto traj_view = hoNDArray<floatd3>(non_flat_dims, (*trajVector)[ii].get_data_ptr());
                    auto hoflat_dcw = dcfO.estimate_DCF_slice(traj_view, image_dims_);

                    cuNDArray<float> nonFdcw(hoflat_dcw);
                    cuNDArray<float> flat_dcw(flat_dims, nonFdcw.get_data_ptr());

                    cuTrajVec.push_back(flat_traj);
                    cuDCWVec.push_back(flat_dcw);

                    float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_))) / asum((&cuDCWVec[ii]));
                    cuDCWVec[ii] *= scale_factor;
                    sqrt_inplace(&cuDCWVec[ii]);
                    if (recon_params.numberChannels == (*dataVector)[ii].get_size(1)) // if the data is not permuted to and is of the shape RO CHA INT then use this code
                    {
                        for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                        {
                            for (auto jj = 0; jj < *(recon_params.shots_per_time.begin() + ii); jj++)
                            {
                                cudaMemcpy(cuData.get()->get_data_ptr() + inter_acc + stride * iCHA + data_dims[0] * jj,
                                           (*dataVector)[ii].get_data_ptr() + (*dataVector)[ii].get_size(0) * iCHA + (*dataVector)[ii].get_size(0) * (*dataVector)[ii].get_size(1) * jj,
                                           (*dataVector)[ii].get_size(0) * sizeof(complext<float>), cudaMemcpyDefault);
                            }
                            // auto dataview = cuNDArray<complext<float>>(trajVector[ii].get_dimensions(), cuData.get()->data() + inter_acc + stride * iCHA);
                            // dataview *= cuDCWVec[ii];
                        }
                    }
                    else
                    { // if the data is permuted to be RO INT CHA then use this code
                        for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                        {
                            for (auto jj = 0; jj < *(recon_params.shots_per_time.begin() + ii); jj++)
                            {
                                cudaMemcpy(cuData.get()->get_data_ptr() + inter_acc + stride * iCHA + data_dims[0] * jj,
                                           (*dataVector)[ii].get_data_ptr() + (*dataVector)[ii].get_size(0) * jj + (*dataVector)[ii].get_size(0) * (*dataVector)[ii].get_size(1) * iCHA,
                                           (*dataVector)[ii].get_size(0) * sizeof(complext<float>), cudaMemcpyDefault);
                            }
                            // auto dataview = cuNDArray<complext<float>>(trajVector[ii].get_dimensions(), cuData.get()->data() + inter_acc + stride * iCHA);
                            // dataview *= cuDCWVec[ii];
                        }
                    }
                }
            }
            return std::make_tuple(std::move(*cuData), std::move(cuTrajVec), std::move(cuDCWVec));
        }

        hoNDArray<std::complex<float>> noncartesian_reconstruction_4D::applyDeformations(hoNDArray<std::complex<float>> images_all, cuNDArray<float> deformation)
        {
            auto registered_images = images_all;
            gpuRegistration gr;

            auto getDefSize = *deformation.get_dimensions();
            getDefSize.pop_back();

            auto stride = std::accumulate(getDefSize.begin(), getDefSize.end(), size_t(1), std::multiplies<size_t>());

            for (auto ii = 0; ii < images_all.get_size(3); ii++)
            {
                auto timage = cuNDArray<float_complext>(hoNDArray<float_complext>(hoNDArray<std::complex<float>>(images_all(slice, slice, slice, ii))));
                auto defView = cuNDArray<float>(getDefSize, deformation.get_data_ptr() + stride * ii);

                auto rimage = gr.deform_image(&timage, defView);
                registered_images(slice, slice, slice, ii) = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(rimage.to_host())));
            }
            return registered_images;
        }

    }
}
