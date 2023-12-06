//
// Created by dchansen on 10/2/18.
// Borrowed to add 2-FOV vds design

#include "TrajectoryParameters.h"
#include "../waveforms/WaveformToTrajectory.h"
#include "../utils/mri_core_girf_correction.h"
namespace nhlbi_toolbox
{
    namespace Spiral
    {

        std::pair<hoNDArray<floatd2>, hoNDArray<float>>
        nhlbi_toolbox::Spiral::TrajectoryParameters::calculate_trajectories_and_weight(const ISMRMRD::AcquisitionHeader &acq_header)
        {
            // Two-fov percentage definition for variable density design
            int nfov = 2; /*  number of fov coefficients.             */
            double fov_vds_[nfov];
            fov_vds_[0] = std::round(fov_*10.0f)/10.0f;
            if (strstr(systemModel.c_str(),"MAGNETOM eMeRge-XL") || strstr(systemModel.c_str(),"MAGNETOM Sola"))
                vds_factor_ = acq_header.user_float[5];
            else
                vds_factor_ = acq_header.user_int[5];
                if(vds_factor_==0)
                    vds_factor_ = acq_header.user_float[5];

            GDEBUG_STREAM("VDS:"<<vds_factor_);
            if (vds_factor_ == 0 || vds_factor_ <1)
                vds_factor_ = 100;
             krmax_ = std::round(krmax_*10000.0f)/10000.0f;

             fov_vds_[1] = std::round((-1 * fov_  * (1.0 - 1.0 * (vds_factor_ / 100.0)))*1000.0f)/1000.0f; //
        //      fov_vds_[1] = -1 * fov_ * (1.0 / krmax_) * (1.0 - 1.0 * (vds_factor_ / 100.0));
            

            int ngmax = 1e5; /*  maximum number of gradient samples      */
            double sample_time = (1.0f * Tsamp_ns_) * 1.0e-9;
            // auto base_gradients = calculate_vds(smax_, gmax_, sample_time, sample_time, Nints_, &fov_, nfov, krmax_, ngmax, acq_header.number_of_samples);
            auto base_gradients = nhlbi_toolbox::Spiral::calculate_vds(smax_, gmax_, sample_time, sample_time, Nints_, &fov_vds_[0], nfov, krmax_, ngmax, acq_header.number_of_samples);
            auto filename = "/opt/data/gt_data/base_gradients.real2";
            nhlbi_toolbox::utils::write_cpu_nd_array<floatd2>(base_gradients, filename);
            int samples_per_interleave_ = base_gradients.get_number_of_elements();
            

            if (spiral_rotations_ == 0)
            { // this is a hack which requires this parameter..
                // normal operation
                base_gradients = nhlbi_toolbox::Spiral::create_rotations(base_gradients, Nints_);
            }
            else
            {
                // Custom spiral rotations
                base_gradients = nhlbi_toolbox::Spiral::create_rotations(base_gradients, spiral_rotations_);
            }

            auto trajectories = nhlbi_toolbox::Spiral::calculate_trajectories(base_gradients, sample_time, krmax_);

            auto weights = nhlbi_toolbox::Spiral::calculate_weights_Hoge(base_gradients, trajectories);

            filename = "/opt/data/gt_data/trajectories.real2";
            nhlbi_toolbox::utils::write_cpu_nd_array<floatd2>(trajectories, filename);
            filename = "/opt/data/gt_data/weights.real";
            
            nhlbi_toolbox::utils::write_cpu_nd_array<float>(weights, filename);
            
            if (this->girf_kernel)
            {
                // base_gradients=nhlbi_toolbox::GIRF::girf_correct(base_gradients, this->girf_kernel, rotation_matrix, 2e-6, 10e-6, 0.85e-6);
                base_gradients = correct_gradients(base_gradients, sample_time, this->girf_sampling_time_us, acq_header.read_dir, acq_header.phase_dir, acq_header.slice_dir);
                auto filename = "/opt/data/gt_data/base_gradients_correct.real2";
                nhlbi_toolbox::utils::write_cpu_nd_array<floatd2>(base_gradients, filename);
                // Weights should be calculated without GIRF corrections according to Hoge et al 2005
                trajectories = nhlbi_toolbox::Spiral::calculate_trajectories(base_gradients, sample_time, krmax_);

             
                filename = "/opt/data/gt_data/trajectories_correct.real2";
                nhlbi_toolbox::utils::write_cpu_nd_array<floatd2>(trajectories, filename);
                weights = calculate_weights_Hoge(base_gradients, trajectories);
            }

            return std::make_pair(std::move(trajectories), std::move(weights));
        }
        void nhlbi_toolbox::Spiral::TrajectoryParameters::read_girf_kernel(std::string girf_folder)
        {

            this->girf_kernel = std::make_optional<hoNDArray<std::complex<float>>>(
                nhlbi_toolbox::corrections::readGIRFKernel(girf_folder)); // AJ fix for now
        }

        void nhlbi_toolbox::Spiral::TrajectoryParameters::set_girf_sampling_time(float time)
        {
            this->girf_sampling_time_us = time;
        }

        hoNDArray<std::complex<float>> nhlbi_toolbox::Spiral::TrajectoryParameters::get_girf_kernel()
        {
            return *this->girf_kernel;
        }

        nhlbi_toolbox::Spiral::TrajectoryParameters::TrajectoryParameters(const ISMRMRD::IsmrmrdHeader &h)
        {
            ISMRMRD::TrajectoryDescription traj_desc;

            if (h.encoding[0].trajectoryDescription)
            {
                traj_desc = *h.encoding[0].trajectoryDescription;
            }
            else
            {
                throw std::runtime_error("Trajectory description missing");
            }

            if (traj_desc.identifier != "HargreavesVDS2000")
            {
                throw std::runtime_error("Expected trajectory description identifier 'HargreavesVDS2000', not found.");
            }

            try
            {
                auto userparam_long = to_map(traj_desc.userParameterLong);
                auto userparam_double = to_map(traj_desc.userParameterDouble);
                Tsamp_ns_ = userparam_long.at("SamplingTime_ns");
                Nints_ = userparam_long.at("interleaves");
                spiral_rotations_ = h.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum + 1;
                gmax_ = userparam_double.at("MaxGradient_G_per_cm");
                smax_ = userparam_double.at("MaxSlewRate_G_per_cm_per_s");
                krmax_ = userparam_double.at("krmax_per_cm");
                fov_ = userparam_double.at("FOVCoeff_1_cm");
                systemModel = (h.acquisitionSystemInformation).get().systemModel->c_str();
            }

            catch (std::out_of_range exception)
            {
                std::string s = "Missing user parameters: " + std::string(exception.what());
                throw std::runtime_error(s);
            }

            TE_ = h.sequenceParameters->TE->at(0);

            if (h.userParameters)
            {
                try
                {
                    auto user_params_string = to_map(h.userParameters->userParameterString);
                    auto user_params_double = to_map(h.userParameters->userParameterDouble);

                    auto girf_kernel_string = user_params_string.at("GIRF_kernel");
                    this->girf_kernel = std::make_optional<hoNDArray<std::complex<float>>>(
                        nhlbi_toolbox::corrections::load_girf_kernel(girf_kernel_string));
                    girf_sampling_time_us = user_params_double.at("GIRF_sampling_time_us");
                }

                catch (std::out_of_range exception)
                {
                }
            }

            GDEBUG("smax:                    %f\n", smax_);
            GDEBUG("gmax:                    %f\n", gmax_);
            GDEBUG("Tsamp_ns:                %d\n", Tsamp_ns_);
            GDEBUG("Nints:                   %d\n", Nints_);
            GDEBUG("fov:                     %f\n", fov_);
            GDEBUG("krmax:                   %f\n", krmax_);
            GDEBUG("GIRF kernel:             %d\n", bool(this->girf_kernel));
            GDEBUG("systemModel:                   %s\n", systemModel);
        }

        hoNDArray<floatd2>
        nhlbi_toolbox::Spiral::TrajectoryParameters::correct_gradients(const hoNDArray<floatd2> &gradients, float grad_samp_us,
                                                                       float girf_samp_us, const float *read_dir, const float *phase_dir,
                                                                       const float *slice_dir)
        {

            arma::fmat33 rotation_matrix;
            rotation_matrix(0, 0) = read_dir[0];
            rotation_matrix(1, 0) = read_dir[1];
            rotation_matrix(2, 0) = read_dir[2];
            rotation_matrix(0, 1) = phase_dir[0];
            rotation_matrix(1, 1) = phase_dir[1];
            rotation_matrix(2, 1) = phase_dir[2];
            rotation_matrix(0, 2) = slice_dir[0];
            rotation_matrix(1, 2) = slice_dir[1];
            rotation_matrix(2, 2) = slice_dir[2];

            return nhlbi_toolbox::corrections::girf_correct(gradients, *girf_kernel, rotation_matrix, grad_samp_us, girf_samp_us, 0.85e-6);
        }
    } // namespace Spiral
} // namespace nhlbi_toolbox