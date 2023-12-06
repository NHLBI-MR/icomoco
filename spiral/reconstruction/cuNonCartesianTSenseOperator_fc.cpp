#pragma once

#include "cuNonCartesianTSenseOperator_fc.h"
#include "vector_td_utilities.h"

using namespace Gadgetron;

template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator_fc<REAL, D>::mult_M(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out,
                                                      bool accumulate)
{

    if (!in || !out)
    {
        throw std::runtime_error("cuNonCartesianSenseOperator::mult_M : 0x0 input/output not accepted");
    }
    if (!in->dimensions_equal(&this->domain_dims_) || !out->dimensions_equal(&this->codomain_dims_))
    {
        throw std::runtime_error(
            "cuNonCartesianSenseOperator::mult_H: input/output arrays do not match specified domain/codomains");
    }
    // Cart -> noncart
    std::vector<size_t> full_dimensions = *this->get_domain_dimensions();   // cart
    std::vector<size_t> data_dimensions = *this->get_codomain_dimensions(); // Non-cart

    auto timeD = full_dimensions[full_dimensions.size() - 1];
    full_dimensions.pop_back();
    full_dimensions.push_back(this->ncoils_);
    full_dimensions.push_back(timeD);

    int cur_device = in->get_device();
    cudaSetDevice(cur_device);

    // std::iter_swap(full_dimensions.end(), full_dimensions.end() - 1); // swap the coil dimension and time

    full_dimensions.pop_back(); // remove time dimension

    std::vector<size_t> slice_dimensions = *this->get_domain_dimensions();
    slice_dimensions.pop_back(); // remove time
    auto stride = std::accumulate(slice_dimensions.begin(), slice_dimensions.end(), 1,
                                  std::multiplies<size_t>()); // product of X,Y,and Z

    std::vector<size_t> tmp_dims = *this->get_codomain_dimensions();
    auto stride_data = std::accumulate(tmp_dims.begin(), tmp_dims.end() - 1, 1, std::multiplies<size_t>());
    GadgetronTimer timer("Deconstruct");
    GDEBUG_STREAM("NumGPUs(): " << eligibleGPUs.size());
    omp_set_nested(1);
    omp_set_max_active_levels(256);
    data_dimensions.pop_back(); // remove CHA

    auto tmpview_dims = full_dimensions;
    tmpview_dims.pop_back();

#pragma omp parallel for schedule(dynamic) num_threads(eligibleGPUs.size()) shared(in, out, trajectory_, combinationWeights_, scaled_time_, fbins_) ordered
    for (size_t it = 0; it < this->shots_per_time_.size(); it++)
    {
        cudaSetDevice(cur_device);
        //auto gpuDevice = fbins_.n_elem > 1 ? eligibleGPUs[it % eligibleGPUs.size()] : cur_device; // only parallelize over gpus if doing concomitant fc else its too expensive
        auto gpuDevice = eligibleGPUs[it % eligibleGPUs.size()];

        cuNDArray<complext<REAL>> tmp(&full_dimensions);

        auto inter_acc = std::accumulate(this->shots_per_time_.begin(), this->shots_per_time_.begin() + it, size_t(0)); // sum of cum sum shots per time

        auto slice_view_in = cuNDArray<complext<REAL>>(slice_dimensions, in->data() + stride * it);

        this->mult_csm(&slice_view_in, &tmp);

        auto ddims = data_dimensions;

        ddims.pop_back();                           // remove interleave
        ddims.push_back(this->shots_per_time_[it]); // insert correct interleave

        for (size_t iCHA = 0; iCHA < this->ncoils_; iCHA++)
        {
            auto tmp_view = cuNDArray<complext<REAL>>(tmpview_dims, tmp.data() + stride * iCHA);
            auto slice_view_out =
                cuNDArray<complext<REAL>>(ddims, out->data() + inter_acc * data_dimensions[0] + stride_data * iCHA);
            cudaSetDevice(gpuDevice);
            //
            if (cur_device != gpuDevice)
            {
                if (accumulate)
                {
                    cuNDArray<complext<REAL>> tmp_out(&ddims);
                    nrfc_vector[gpuDevice][it].reconstruct_todevice(tmp_out, tmp_view, trajectory_[it], this->dcw_[it], *combinationWeights_, scaled_time_[it], fbins_, gpuDevice, false);
                    slice_view_out += tmp_out;
                }
                else
                {
                    nrfc_vector[gpuDevice][it].reconstruct_todevice(slice_view_out, tmp_view, trajectory_[it], this->dcw_[it], *combinationWeights_, scaled_time_[it], fbins_, gpuDevice, false);
                }
            }
            else
            {
                if (accumulate)
                {
                    cuNDArray<complext<REAL>> tmp_out(&ddims);
                    nrfc_vector[gpuDevice][it].deconstruct(tmp_view, tmp_out, trajectory_[it], this->dcw_[it], *combinationWeights_, scaled_time_[it], fbins_);
                    slice_view_out += tmp_out;
                }
                else
                {
                    nrfc_vector[gpuDevice][it].deconstruct(tmp_view, slice_view_out, trajectory_[it], this->dcw_[it], *combinationWeights_, scaled_time_[it], fbins_);
                }
            }
            cudaSetDevice(cur_device);
        }
    }
}

template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator_fc<REAL, D>::mult_MH(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out,
                                                       bool accumulate)
{

    if (!in || !out)
    {
        throw std::runtime_error("cuNonCartesianSenseOperator::mult_MH : 0x0 input/output not accepted");
    }

    if (!in->dimensions_equal(&this->codomain_dims_) || !out->dimensions_equal(&this->domain_dims_))
    {
        throw std::runtime_error(
            "cuNonCartesianSenseOperator::mult_MH: input/output arrays do not match specified domain/codomains");
    }
    std::vector<size_t> out_dimensions = *this->get_domain_dimensions();
    std::vector<size_t> in_dimensions = *this->get_codomain_dimensions();

    auto RO = in->get_size(0);
    auto E1E2 = in->get_size(1);
    auto CHA = in->get_size(2);

    in_dimensions.pop_back(); // Remove CH dimension

    out_dimensions.pop_back();               // Remove the timeDimension
    out_dimensions.push_back(this->ncoils_); // add coil dimension

    out_dimensions.pop_back(); // rm coil dimension
    // cuNDArray<complext<REAL>> tmp_coilCmb(&out_dimensions);

    auto stride_ch = std::accumulate(in_dimensions.begin(), in_dimensions.end(), 1,
                                     std::multiplies<size_t>()); // product of X,Y,and Z

    auto stride_out = std::accumulate(out_dimensions.begin(), out_dimensions.end(), 1,
                                      std::multiplies<size_t>()); // product of X,Y,and Z
    if (!accumulate)
    {
        clear(out);
    }

    int cur_device = in->get_device();
    cudaSetDevice(cur_device);

    out_dimensions.push_back(this->ncoils_); // add coil dimension
    auto out_dimensions2 = out_dimensions;
    out_dimensions2.pop_back();
    in_dimensions.pop_back(); // Remove INT dimension
    GadgetronTimer timer("Reconstruct Sense");

#pragma omp parallel for schedule(dynamic) num_threads(eligibleGPUs.size()) shared(in, out, trajectory_, combinationWeights_, scaled_time_, fbins_) ordered
    for (size_t it = 0; it < this->shots_per_time_.size(); it++)
    {
        cudaSetDevice(cur_device);
        //auto gpuDevice = fbins_.n_elem > 1 ? eligibleGPUs[it % eligibleGPUs.size()] : cur_device; // only parallelize over gpus if doing concomitant fc else its too expensive
        auto gpuDevice = eligibleGPUs[it % eligibleGPUs.size()];
        cuNDArray<complext<REAL>> tmp(out_dimensions); // x y z coil
        //GDEBUG_STREAM("GPU :" << gpuDevice);

        auto in_dim_t = in_dimensions;
        in_dim_t.push_back(this->shots_per_time_[it]);
        in_dim_t.push_back(CHA);

        auto inter_acc = std::accumulate(this->shots_per_time_.begin(), this->shots_per_time_.begin() + it, 0);

        cuNDArray<complext<REAL>> slice_view(in_dim_t);

        crop<float_complext, 3>(uint64d3(0, inter_acc, 0),
                                uint64d3(RO, this->shots_per_time_[it], this->ncoils_),
                                *in, slice_view);

        cudaSetDevice(gpuDevice);

        if (cur_device != gpuDevice)
            nrfc_vector[gpuDevice][it].reconstruct_todevice(slice_view, tmp, trajectory_[it], (this->dcw_[it]), *combinationWeights_, scaled_time_[it], fbins_, gpuDevice, true);
        else
            nrfc_vector[gpuDevice][it].reconstruct(slice_view, tmp, trajectory_[it], (this->dcw_[it]), *combinationWeights_, scaled_time_[it], fbins_);
        cudaSetDevice(cur_device);

        auto slice_view_output = cuNDArray<complext<REAL>>(out_dimensions2, out->data() + stride_out * it);

        this->mult_csm_conj_sum(&tmp, &slice_view_output);
    }
}

template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator_fc<REAL, D>::preprocess(std::vector<cuNDArray<vector_td<REAL, D>>> &trajectory)
{
    using namespace nhlbi_toolbox::utils;

    if (&(*trajectory.begin()) == 0x0)
    {
        throw std::runtime_error("cuNonCartesianTSenseOperator_fc: cannot preprocess 0x0 trajectory.");
    }

    boost::shared_ptr<std::vector<size_t>> domain_dims = this->get_domain_dimensions();
    if (domain_dims.get() == 0x0 || domain_dims->size() == 0)
    {
        throw std::runtime_error("cuNonCartesianTSenseOperator_fc::preprocess : operator domain dimensions not set");
    }

    auto data_dims = *this->get_codomain_dimensions();
    auto dataSizeT = std::accumulate(data_dims.begin(), data_dims.end(), size_t(1), std::multiplies<size_t>()) * 4 * 2;
    auto dataSize = dataSizeT + dataSizeT / (2 * this->ncoils_) + 3 * dataSizeT / (2 * this->ncoils_); // add dcw and traj

    auto image_dims = *this->get_domain_dimensions();
    auto imageSize = std::accumulate(image_dims.begin(), image_dims.end(), size_t(1), std::multiplies<size_t>()) * 4 * 2;
    dataSize += imageSize;
    this->eligibleGPUs = nhlbi_toolbox::utils::FindCudaDevices(dataSize);

    for (auto ii = 0; ii < eligibleGPUs.size(); ii++)
    {
        cudaSetDevice(eligibleGPUs[ii]);
        std::vector<nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc> temp_vector;
        for (auto it = 0; it < this->shots_per_time_.size(); it++)
        {

            auto ttraj = set_device(&trajectory[it], ii);
            nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc nrfc(this->recon_params_);

            nrfc.preprocess(&ttraj);
            temp_vector.push_back(nrfc);
        }
        nrfc_vector.push_back(temp_vector);
    }
    int cur_device = trajectory[0].get_device();
    cudaSetDevice(cur_device);

    trajectory_ = trajectory;
    is_preprocessed_ = true;
}

template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator_fc<REAL, D>::set_fbins(arma::fvec fbins)
{
    fbins_ = fbins;
}
template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator_fc<REAL, D>::set_combination_weights(cuNDArray<float_complext> *combinationWeights)
{
    combinationWeights_ = combinationWeights;
}
template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator_fc<REAL, D>::set_recon_params(reconParams rp)
{
    recon_params_ = rp;
    // nc_recon_fc_ = new nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc(recon_params);
}
template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator_fc<REAL, D>::set_scaled_time(std::vector<cuNDArray<REAL>> &scaled_time)
{
    scaled_time_ = scaled_time;
}
template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator_fc<REAL, D>::set_dofield_correction(bool flag)
{
    doConcomitantFieldCorraction_ = flag;
    // if(is_preprocessed_)
    // reprocess to spread overGPUs
}

//template class EXPORTGPUPMRI cuNonCartesianTSenseOperator_fc<float, 1>;
//template class EXPORTGPUPMRI cuNonCartesianTSenseOperator_fc<float, 2>;
template class EXPORTGPUPMRI cuNonCartesianTSenseOperator_fc<float, 3>;
//template class EXPORTGPUPMRI cuNonCartesianTSenseOperator_fc<float, 4>;

// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<double, 1>;
// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<double, 2>;
// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<double, 3>;
// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<double, 4>;
