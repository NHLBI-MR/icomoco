
#include "noncartesian_reconstruction_3D.h"

using namespace Gadgetron;
namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        std::tuple<cuNDArray<float_complext>,
                   cuNDArray<floatd3>,
                   cuNDArray<float>>
        noncartesian_reconstruction_3D::organize_data(
            hoNDArray<float_complext> *data,
            hoNDArray<floatd3> *traj,
            hoNDArray<float> *dcw)
        {
            auto [cuData, cutraj] = noncartesian_reconstruction::organize_data(data, traj);
            auto cudcw = noncartesian_reconstruction::estimate_dcf(&cutraj);
            sqrt_inplace(&cudcw);
            return std::make_tuple(std::move(cuData), std::move(cutraj), std::move(cudcw));
        }

        cuNDArray<float_complext> noncartesian_reconstruction_3D::reconstruct(
            cuNDArray<float_complext> *data,
            cuNDArray<floatd3> *traj,
            cuNDArray<float> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm)
        {
            auto data_dims = *data->get_dimensions();
            auto dcwPtr = boost::make_shared<cuNDArray<float>>(*dcw);
            // need to multiply by the weights to correctly to the FWD transform because we did sqrt of dcw
            *data *= *dcw;

            auto E_ = boost::shared_ptr<cuNonCartesianSenseOperator<float, 3>>(new cuNonCartesianSenseOperator<float, 3>(ConvolutionType::ATOMIC));
            // spit0-bergman cannot do precon
            //auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2]};

            // Setup Encoding Operator
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(dcwPtr);
            E_->preprocess(traj);

            auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            E_->mult_MH(data, x0.get());

            // setup solver spit-bergman
            cuSbcCgSolver<float_complext> solver_;
            solver_.set_encoding_operator(E_);
            solver_.set_max_inner_iterations(recon_params.iterations_inner);
            solver_.set_max_outer_iterations(recon_params.iterations);
            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            solver_.set_x0(x0);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.numberChannels};
            recon_dims.pop_back();

            cuNDArray<float_complext> reg_image(recon_dims);

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

            //TV->set_domain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            solver_.add_regularization_operator(Ry, recon_params.norm);
            solver_.add_regularization_operator(Rz, recon_params.norm);

            reg_image = *solver_.solve(data);

            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix
            cuNDArray<float_complext> images_cropped(recon_dims);

            crop<float_complext, 3>(uint64d3(0, 0, (image_dims_[2] - recon_params.rmatrixSize.z) / 2),
                                    uint64d3(image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z),
                                    reg_image,
                                    images_cropped);

            // de-prep data
            *data /= *dcw;

            return images_cropped;
        }

        cuNDArray<float_complext> noncartesian_reconstruction_3D::reconstruct_fc(
            cuNDArray<float_complext> *data,
            cuNDArray<floatd3> *traj,
            cuNDArray<float> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm,
            cuNDArray<float_complext> *combination_weights,
            cuNDArray<float> *scaled_time,
            arma::fvec fbins)
        {
            auto data_dims = *data->get_dimensions();
            auto dcwPtr = boost::make_shared<cuNDArray<float>>(*dcw);
            // need to multiply by the weights to correctly to the FWD transform because we did sqrt of dcw
            *data *= *dcw;

            auto E_ = boost::shared_ptr<cuNonCartesianSenseOperator_fc<float, 3>>(new cuNonCartesianSenseOperator_fc<float, 3>(ConvolutionType::ATOMIC));
            // spit0-bergman cannot do precon
            //auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2]};

            // Setup Encoding Operator
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(dcwPtr);
            E_->preprocess(traj);
            E_->set_combination_weights(combination_weights);
            E_->set_scaled_time(scaled_time);
            E_->set_fbins(fbins);
            E_->set_recon_params(recon_params);

            auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            E_->mult_MH(data, x0.get());

            // setup solver spit-bergman
            cuSbcCgSolver<float_complext> solver_;
            solver_.set_encoding_operator(E_);
            solver_.set_max_inner_iterations(recon_params.iterations_inner);
            solver_.set_max_outer_iterations(recon_params.iterations);
            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            solver_.set_x0(x0);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.numberChannels};
            recon_dims.pop_back();

            cuNDArray<float_complext> reg_image(recon_dims);

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

            //TV->set_domain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            solver_.add_regularization_operator(Ry, recon_params.norm);
            solver_.add_regularization_operator(Rz, recon_params.norm);

            reg_image = *solver_.solve(data);

            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix
            cuNDArray<float_complext> images_cropped(recon_dims);

            crop<float_complext, 3>(uint64d3(0, 0, (image_dims_[2] - recon_params.rmatrixSize.z) / 2),
                                    uint64d3(image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z),
                                    reg_image,
                                    images_cropped);

            // de-prep data
            *data /= *dcw;

            return images_cropped;
        }
        cuNDArray<float_complext> noncartesian_reconstruction_3D::reconstruct_CGSense_fc(
            cuNDArray<float_complext> *data,
            cuNDArray<floatd3> *traj,
            cuNDArray<float> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm,
            cuNDArray<float_complext> *combination_weights,
            cuNDArray<float> *scaled_time,
            arma::fvec fbins)
        {
            auto data_dims = *data->get_dimensions();
            sqrt_inplace(dcw);
            auto dcwPtr = boost::make_shared<cuNDArray<float>>(*dcw);
            // need to multiply by the weights to correctly to the FWD transform because we did sqrt of dcw

            *data *= *dcw;

            auto E_ = boost::shared_ptr<cuNonCartesianSenseOperator_fc<float, 3>>(new cuNonCartesianSenseOperator_fc<float, 3>(ConvolutionType::ATOMIC));
            // spit0-bergman cannot do precon
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2]};

            std::vector<size_t> cwdims = {image_dims_[0], image_dims_[1], image_dims_[2], fbins.n_elem};
            cuNDArray<float_complext> padded_cw(cwdims);
            if (fbins.n_elem > 1 && combination_weights->get_size(3) > 1)
                padded_cw = pad<float_complext, 4>(uint64d4(image_dims_[0], image_dims_[1], image_dims_[2], fbins.n_elem), combination_weights, float_complext(0));
            else
                padded_cw = pad<float_complext, 3>(uint64d3(image_dims_[0], image_dims_[1], image_dims_[2]), combination_weights, float_complext(0));
            padded_cw.squeeze();            

            // Setup Encoding Operator
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(dcwPtr);
            E_->preprocess(traj);
            E_->set_combination_weights(&padded_cw);
            E_->set_scaled_time(scaled_time);
            E_->set_fbins(fbins);
            E_->set_recon_params(recon_params);

            auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            E_->mult_MH(data, x0.get());
            
            // Preconditioner
            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

            _precon_weights = sum(abs_square(csm.get()).get(), 3);
            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix
            cuNDArray<float> _precon_weights_cropped(recon_dims);

            crop<float, 3>(uint64d3(0, 0, (image_dims_[2] - recon_params.rmatrixSize.z) / 2),
                           uint64d3(image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z),
                           *_precon_weights,
                           _precon_weights_cropped);

            reciprocal_sqrt_inplace(&_precon_weights_cropped);
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                  *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

            D_->set_weights(precon_weights);


            // setup solver spit-bergman
            // setup solver spit-bergman
            cuCgSolver<float_complext> solver_;
            solver_.set_encoding_operator(E_);
            solver_.set_max_iterations(recon_params.iterations);
            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            solver_.set_x0(x0);
            solver_.set_preconditioner(D_);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.numberChannels};
            recon_dims.pop_back();

            cuNDArray<float_complext> reg_image(recon_dims);

            reg_image = *solver_.solve(data);

            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix
            cuNDArray<float_complext> images_cropped(recon_dims);

            crop<float_complext, 3>(uint64d3(0, 0, (image_dims_[2] - recon_params.rmatrixSize.z) / 2),
                                    uint64d3(image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z),
                                    reg_image,
                                    images_cropped);

            // de-prep data
            *data /= *dcw;

            return images_cropped;
        }

        cuNDArray<float_complext> noncartesian_reconstruction_3D::reconstruct_CGSense(
            cuNDArray<float_complext> *data,
            cuNDArray<floatd3> *traj,
            cuNDArray<float> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm)
        {
            auto data_dims = *data->get_dimensions();
            auto dcwPtr = boost::make_shared<cuNDArray<float>>(*dcw);
            // need to multiply by the weights to correctly to the FWD transform because we did sqrt of dcw
            *data *= *dcw;

            auto E_ = boost::shared_ptr<cuNonCartesianSenseOperator<float, 3>>(new cuNonCartesianSenseOperator<float, 3>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2]};

            // Setup Encoding Operator
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(dcwPtr);
            E_->preprocess(traj);

            auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            E_->mult_MH(data, x0.get());

            // Preconditioner
            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

            _precon_weights = sum(abs_square(csm.get()).get(), 3);
            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix
            cuNDArray<float> _precon_weights_cropped(recon_dims);

            crop<float, 3>(uint64d3(0, 0, (image_dims_[2] - recon_params.rmatrixSize.z) / 2),
                           uint64d3(image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z),
                           *_precon_weights,
                           _precon_weights_cropped);

            reciprocal_sqrt_inplace(&_precon_weights_cropped);
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                  *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

            D_->set_weights(precon_weights);

            // setup solver spit-bergman
            cuCgSolver<float_complext> solver_;
            solver_.set_encoding_operator(E_);
            solver_.set_max_iterations(recon_params.iterations);
            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            solver_.set_x0(x0);
            solver_.set_preconditioner(D_);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.numberChannels};
            recon_dims.pop_back();

            cuNDArray<float_complext> reg_image(recon_dims);

            reg_image = *solver_.solve(data);

            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix
            cuNDArray<float_complext> images_cropped(recon_dims);

            crop<float_complext, 3>(uint64d3(0, 0, (image_dims_[2] - recon_params.rmatrixSize.z) / 2),
                                    uint64d3(image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z),
                                    reg_image,
                                    images_cropped);

            // de-prep data
            *data /= *dcw;

            return images_cropped;
        }

    }
}