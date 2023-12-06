#include "densityCompensation.h"

namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        std::vector<arma::uvec> density_compensation::extractSlices(hoNDArray<floatd3> sp_traj)
        {
            auto sl_traj = arma::vec(sp_traj.get_size(1));
            for (auto ii = 0; ii < sp_traj.get_size(1); ii++)
                sl_traj[ii] = sp_traj(1, ii)[2];

            auto z_encodes = arma::vec(arma::unique(sl_traj));
            //z_encodes.print();

            std::vector<arma::uvec> slice_indexes;
            for (auto ii = 0; ii < z_encodes.n_elem; ii++)
            {
                arma::uvec temp = (find(sl_traj == z_encodes[ii]));
                slice_indexes.push_back(temp);
                //  slice_indexes[ii].print();
            }

            return slice_indexes;
        }

        template <unsigned int D>
        cuNDArray<float> density_compensation::estimate_DCF(cuNDArray<vector_td<float, D>> &traj, cuNDArray<float> &dcw, std::vector<size_t> image_dims)
        {
            // GadgetronTimer timer("estimate_DCF");

            std::vector<size_t> flat_dims = {traj.get_number_of_elements()};
            cuNDArray<vector_td<float, D>> flat_traj(flat_dims, traj.get_data_ptr());
            traj = cuNDArray<vector_td<float, D>>(flat_traj);
            cuNDArray<float> flat_dcw(flat_dims, dcw.get_data_ptr());
            dcw = cuNDArray<float>(flat_dcw);
            if (D == 2 && image_dims.size() > 2)
                image_dims.pop_back();

            if (useIterativeDCWEstimated)
            {
              //  GDEBUG_STREAM("useIterativeDCWEstimated: " << useIterativeDCWEstimated);

                auto temp = *(Gadgetron::estimate_dcw<float, D>(&traj,
                                                                &dcw,
                                                                from_std_vector<size_t, D>(image_dims),
                                                                oversampling_factor_,
                                                                size_t(iterations), kernel_width_, ConvolutionType::ATOMIC));
                dcw = cuNDArray<float>(temp);
            }
            else
            {
                auto temp = *(Gadgetron::estimate_dcw<float, D>(&traj,
                                                                from_std_vector<size_t, D>(image_dims),
                                                                oversampling_factor_,
                                                                size_t(iterations), kernel_width_, ConvolutionType::ATOMIC));
                dcw = cuNDArray<float>(temp);
            }

            return dcw;
        }
        template <unsigned int D>
        cuNDArray<float> density_compensation::estimate_DCF(cuNDArray<vector_td<float, D>> &traj, std::vector<size_t> image_dims)
        {
            // GadgetronTimer timer("estimate_DCF");

            std::vector<size_t> flat_dims = {traj.get_number_of_elements()};
            cuNDArray<vector_td<float, D>> flat_traj(flat_dims, traj.get_data_ptr());
            traj = cuNDArray<vector_td<float, D>>(flat_traj);

            if (D == 2 && image_dims.size() > 2)
                image_dims.pop_back();

            auto temp = *(Gadgetron::estimate_dcw<float, D>(&traj,
                                                            from_std_vector<size_t, D>(image_dims),
                                                            oversampling_factor_,
                                                            size_t(iterations), kernel_width_, ConvolutionType::STANDARD));
            auto dcw = cuNDArray<float>(temp);

            return dcw;
        }

        hoNDArray<floatd2> density_compensation::traj3Dto2D(hoNDArray<floatd3> &sp_traj)
        {
            auto dims = sp_traj.get_dimensions();
            auto traj = hoNDArray<floatd2>(dims);

            auto traj_ptr = traj.get_data_ptr();
            auto ptr = sp_traj.get_data_ptr();

            for (size_t i = 0; i < sp_traj.get_number_of_elements(); i++)
            {

                traj_ptr[i][0] = ptr[i][0];
                traj_ptr[i][1] = ptr[i][1];
            }

            return traj;
        }
        hoNDArray<float> density_compensation::estimate_DCF_slice(hoNDArray<floatd3> &sp_traj, hoNDArray<float> &sp_dcw, std::vector<size_t> image_dims)
        {
            hoNDArray<float> hodcw(sp_dcw.get_size(0), sp_dcw.get_size(1));
            auto slice_index = extractSlices(sp_traj);
            auto traj_2D = traj3Dto2D(sp_traj);
            bool estimate_oneslice = true;

            auto eligibleGPUs = nhlbi_toolbox::utils::FindCudaDevices(0);
            while (eligibleGPUs.size() > maxDevices)
                eligibleGPUs.pop_back();

            for (auto ii = 1; ii < slice_index.size(); ii++)
            {
                if (slice_index[ii - 1].n_elem != slice_index[ii].n_elem)
                    estimate_oneslice = false;
            }
            std::string pout = estimate_oneslice ? "Fully Sampled Estimating only one slice" : "Estimating DCF for all slices";
            GDEBUG_STREAM(pout);

#pragma omp parallel for num_threads(eligibleGPUs.size())
            for (auto ii = 0; ii < (estimate_oneslice ? 1 : slice_index.size()); ii++)
            {
                auto slvec = slice_index[ii];
                //GDEBUG_STREAM("Selected Device: " << ii % (eligibleGPUs.size()));
                cudaSetDevice(ii % (eligibleGPUs.size()));
                hoNDArray<floatd2> temp_traj({sp_traj.get_size(0), slvec.n_elem});
                hoNDArray<float> temp_dcw({sp_traj.get_size(0), slvec.n_elem});
                for (auto jj = 0; jj < slvec.n_elem; jj++)
                {
                    temp_traj(slice, jj) = traj_2D(slice, slvec[jj]);
                    temp_dcw(slice, jj) = sp_dcw(slice, slvec[jj]);
                }
                auto cutraj = cuNDArray<floatd2>(temp_traj);
                auto cudcw = cuNDArray<float>(temp_dcw);
                cudcw = estimate_DCF<2>(cutraj, cudcw, image_dims);
                temp_dcw = *cudcw.to_host();
                temp_dcw.reshape(sp_dcw.get_size(0), -1);

                for (auto kk = ii; kk < (estimate_oneslice ? slice_index.size() : ii + 1); kk++)
                {
                    auto slvec = slice_index[kk];
                    for (auto jj = 0; jj < slvec.n_elem; jj++)
                    {
                        hodcw(slice, slvec[jj]) = temp_dcw(slice, jj);
                    }
                }
            }
            return hodcw;
        }
        hoNDArray<float> density_compensation::estimate_DCF_slice(hoNDArray<floatd3> &sp_traj, std::vector<size_t> image_dims)
        {
            GadgetronTimer timer("DCF:");

            int origDevice;
            cudaGetDevice(&origDevice);
            hoNDArray<float> hodcw(sp_traj.get_size(0), sp_traj.get_size(1));
            auto slice_index = extractSlices(sp_traj);
            auto traj_2D = traj3Dto2D(sp_traj);
            bool estimate_oneslice = true;

            auto eligibleGPUs = nhlbi_toolbox::utils::FindCudaDevices(0);
            while (eligibleGPUs.size() > maxDevices)
                eligibleGPUs.pop_back();

            for (auto ii = 1; ii < slice_index.size(); ii++)
            {
                if (slice_index[ii - 1].n_elem != slice_index[ii].n_elem)
                    estimate_oneslice = false;
            }
            std::string pout = estimate_oneslice ? "Fully Sampled Estimating only one slice" : "Estimating DCF for all slices";
            GDEBUG_STREAM(pout);

#pragma omp parallel for num_threads(eligibleGPUs.size())
            for (auto ii = 0; ii < (estimate_oneslice ? 1 : slice_index.size()); ii++)
            {
                auto slvec = slice_index[ii];
                //GDEBUG_STREAM("Selected Device: " << ii % (eligibleGPUs.size()));
                cudaSetDevice(ii % (eligibleGPUs.size()));
                hoNDArray<floatd2> temp_traj({sp_traj.get_size(0), slvec.n_elem});
                hoNDArray<float> temp_dcw({sp_traj.get_size(0), slvec.n_elem});
                for (auto jj = 0; jj < slvec.n_elem; jj++)
                {
                    temp_traj(slice, jj) = traj_2D(slice, slvec[jj]);
                }
                auto cutraj = cuNDArray<floatd2>(temp_traj);
                auto cudcw = estimate_DCF<2>(cutraj, image_dims);
                temp_dcw = *cudcw.to_host();
                temp_dcw.reshape(sp_traj.get_size(0), -1);

                for (auto kk = ii; kk < (estimate_oneslice ? slice_index.size() : ii + 1); kk++)
                {
                    auto slvec = slice_index[kk];
                    for (auto jj = 0; jj < slvec.n_elem; jj++)
                    {
                        hodcw(slice, slvec[jj]) = temp_dcw(slice, jj);
                    }
                }
                cutraj.clear();
                cudcw.clear();
            }
            cudaSetDevice(origDevice);
            return hodcw;
        }
    }
}