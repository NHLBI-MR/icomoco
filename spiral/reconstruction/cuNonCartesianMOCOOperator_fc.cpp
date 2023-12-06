
#include "cuNonCartesianMOCOOperator_fc.h"
#include "gpuRegistration.cuh"

using namespace Gadgetron;

template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::mult_M(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out,
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

    data_dimensions.pop_back(); // remove coil dimension from tmp_data;

    int dims_orig = full_dimensions.size();
    //GDEBUG_STREAM("dims_orig:" << dims_orig);

    // pop out the dims
    for (int ii = 0; ii < (dims_orig - 3); ii++)
    {
    //    GDEBUG_STREAM("I am poping Full");
        full_dimensions.pop_back();
    }

    full_dimensions.push_back(this->ncoils_);

    int cur_device = in->get_device();
    cudaSetDevice(cur_device);

    std::vector<size_t> slice_dimensions = *this->get_domain_dimensions();
    auto input = cuNDArray<complext<REAL>>(slice_dimensions, in->data());

    dims_orig = slice_dimensions.size();
    for (int ii = 0; ii < (dims_orig - 3); ii++)
        slice_dimensions.pop_back();

    auto stride = std::accumulate(slice_dimensions.begin(), slice_dimensions.end(), 1,
                                  std::multiplies<size_t>()); // product of X,Y,and Z

    std::vector<size_t> tmp_dims = *this->get_codomain_dimensions();
    auto stride_data = std::accumulate(tmp_dims.begin(), tmp_dims.end() - 1, 1, std::multiplies<size_t>());

    auto tmpview_dims = full_dimensions;
    // tmpview_dims.pop_back();

    GadgetronTimer timer("Deconstruct");
    // GDEBUG_STREAM("NumGPUs(): " << eligibleGPUs.size());

    cuNonCartesianMOCOOperator<float, 3> mocoObj(this->convolutionType);
    gpuRegistration gr;

//#pragma omp parallel for num_threads(eligibleGPUs.size()) shared(in, out, trajectory_, combinationWeights_, scaled_time_, fbins_) ordered
    for (size_t it = 0; it < this->shots_per_time_.get_number_of_elements(); it++)
    {
        cudaSetDevice(cur_device);
        // auto gpuDevice = fbins_.n_elem > 1 ? eligibleGPUs[it % eligibleGPUs.size()] : cur_device; // only parallelize over gpus if doing concomitant fc else its too expensive
        auto gpuDevice = eligibleGPUs[it % eligibleGPUs.size()];

        cuNDArray<complext<REAL>> tmp(&full_dimensions);

        auto inter_acc = std::accumulate(this->shots_per_time_.begin(), this->shots_per_time_.begin() + it, size_t(0)); // sum of cum sum shots per time

        // auto slice_view_in = cuNDArray<complext<REAL>>(slice_dimensions, in->data() + stride * it);
        // GDEBUG_STREAM("GPU :" << eligibleGPUs[it % eligibleGPUs.size()]);

        //
        full_dimensions.pop_back(); // remove channel

        auto slice_input = cuNDArray<complext<REAL>>(full_dimensions, input.data() + int(it / this->shots_per_time_.get_size(0)) * stride);

        full_dimensions.push_back(this->ncoils_);

        // Move the image to moving image
        auto slice_view_in = gr.deform_image(&slice_input, backward_deformation_[it]);

        this->mult_csm(&slice_view_in, &tmp);

        auto ddims = data_dimensions;

        ddims.pop_back();                                       // remove interleave
        ddims.push_back(*(this->shots_per_time_.begin() + it)); // insert correct interleave
        ddims.push_back(this->ncoils_);                         // insert channels

        // for (size_t iCHA = 0; iCHA < this->ncoils_; iCHA++)
        {
            // auto tmp_view = cuNDArray<complext<REAL>>(tmpview_dims, tmp.data() + stride * iCHA);
            auto tmp_view = cuNDArray<complext<REAL>>(tmpview_dims, tmp.data());

            // auto slice_view_out =
            //     cuNDArray<complext<REAL>>(ddims, out->data() + inter_acc * data_dimensions[0] + stride_data * iCHA);
            cuNDArray<complext<REAL>> slice_view_out(ddims);
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
                 //   GDEBUG_STREAM("GPU :" << gpuDevice);

                    nrfc_vector[gpuDevice][it].deconstruct(tmp_view, slice_view_out, trajectory_[it], this->dcw_[it], *combinationWeights_, scaled_time_[it], fbins_);
                }
            }
            for (size_t iCHA = 0; iCHA < this->ncoils_; iCHA++)
                cudaMemcpyAsync(out->data() + inter_acc * data_dimensions[0] + stride_data * iCHA, slice_view_out.get_data_ptr() + *(this->shots_per_time_.begin() + it) * data_dimensions[0] * iCHA,
                                *(this->shots_per_time_.begin() + it) * data_dimensions[0] * sizeof(float_complext), cudaMemcpyDefault);

            cudaSetDevice(cur_device);
        }
    }
}

template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::mult_MH(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out,
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

    // out_dimensions.push_back(this->ncoils_); // add coil dimension

    // out_dimensions.pop_back(); // rm coil dimension
    // cuNDArray<complext<REAL>> tmp_coilCmb(&out_dimensions);

    int dims_orig = (out_dimensions.size());
    for (int ii = 0; ii < dims_orig - 3; ii++)
    {
        out_dimensions.pop_back();
    }
    auto out_slice_dims = out_dimensions;
    auto is4D = dims_orig == 4 ? true : false;
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

    out_dimensions.push_back(this->shots_per_time_.get_size(0)); // add time dimension
    cuNDArray<complext<REAL>> moving_images(out_dimensions);
    out_dimensions.pop_back(); // rm time

    out_dimensions.push_back(this->ncoils_); // add coil dimension
    auto out_dimensions2 = out_dimensions;
    out_dimensions2.pop_back();
    in_dimensions.pop_back(); // Remove INT dimension
    GadgetronTimer timer("Reconstruct Sense");

    auto output = cuNDArray<complext<REAL>>(out_dimensions2, out->data());
    fill<complext<REAL>>(&output, complext<REAL>((REAL)0, (REAL)0));

    cuNonCartesianMOCOOperator<float, 3> mocoObj(this->convolutionType);
    gpuRegistration gr;

    size_t time_dims;
    if(is4D)
        time_dims = this->shots_per_time_.get_size(1);
    else
        time_dims = 1;

    auto moco_dims = this->shots_per_time_.get_size(0);
   // GDEBUG_STREAM("time_dims:" << this->shots_per_time_.get_size(1));
   // GDEBUG_STREAM("resp_dims:" << this->shots_per_time_.get_size(0));
    for (size_t ito = 0; ito < time_dims; ito++)
    {
        auto slice_view_output = cuNDArray<complext<REAL>>(out_slice_dims, output.data() + stride_out * ito);
#pragma omp parallel for num_threads(eligibleGPUs.size()) shared(in, out, trajectory_, combinationWeights_, scaled_time_, fbins_) ordered
        for (size_t it = 0; it < this->shots_per_time_.get_size(0); it++)
        {
            cudaSetDevice(cur_device);
            auto gpuDevice = eligibleGPUs[it % eligibleGPUs.size()];
            cuNDArray<complext<REAL>> tmp(out_dimensions); // x y z coil
            // GDEBUG_STREAM("GPU :" << gpuDevice);

            auto in_dim_t = in_dimensions;
            in_dim_t.push_back(*(this->shots_per_time_.begin() + it + ito * moco_dims));
            in_dim_t.push_back(CHA);

            auto inter_acc = std::accumulate(this->shots_per_time_.begin(), this->shots_per_time_.begin() + it + ito * moco_dims, 0);
            //GDEBUG_STREAM("inter_acc:" << inter_acc);

            // cuNDArray<complext<REAL>> slice_view(in_dim_t);

            auto slice_view = crop<float_complext, 3>(uint64d3(0, inter_acc, 0),
                                                      uint64d3(RO, *(this->shots_per_time_.begin() + it + ito * moco_dims), this->ncoils_),
                                                      *in);
            cudaSetDevice(gpuDevice);

            if (cur_device != gpuDevice)
                nrfc_vector[gpuDevice][it + ito * moco_dims].reconstruct_todevice(slice_view, tmp, trajectory_[it + ito * moco_dims], (this->dcw_[it + ito * moco_dims]), *combinationWeights_, scaled_time_[it + ito * moco_dims], fbins_, gpuDevice, true);
            else
                nrfc_vector[gpuDevice][it + ito * moco_dims].reconstruct(slice_view, tmp, trajectory_[it + ito * moco_dims], (this->dcw_[it + ito * moco_dims]), *combinationWeights_, scaled_time_[it + ito * moco_dims], fbins_);
            cudaSetDevice(cur_device);

            auto slice_view_moving_im = cuNDArray<complext<REAL>>(out_dimensions2, moving_images.data() + stride_out * int(it / this->shots_per_time_.get_size(0)));

            this->mult_csm_conj_sum(&tmp, &slice_view_moving_im);

            slice_view_output += gr.deform_image(&slice_view_moving_im, forward_deformation_[it + ito * moco_dims ]);
        }

        nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(slice_view_output, "/opt/data/gt_data/output_moco.complex");

        slice_view_output /= complext<REAL>((REAL)this->shots_per_time_.get_size(0), (REAL)0);
    }
    
    nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(output, "/opt/data/gt_data/output_norm.complex");
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_forward_deformation(std::vector<cuNDArray<REAL>> forward_deformation)
{
    forward_deformation_ = forward_deformation;
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_backward_deformation(std::vector<cuNDArray<REAL>> backward_deformation)
{
    backward_deformation_ = backward_deformation;
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::preprocess(std::vector<cuNDArray<vector_td<REAL, D>>> &trajectory)
{
    using namespace nhlbi_toolbox::utils;
    omp_set_max_active_levels(256);
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

    int cur_device = trajectory[0].get_device();

    eligibleGPUs = nhlbi_toolbox::utils::FindCudaDevices(dataSize);

    for (auto ii = 0; ii < eligibleGPUs.size(); ii++)
    {
        cudaSetDevice(eligibleGPUs[ii]);
        std::vector<nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc> temp_vector;
        for (auto it = 0; it < this->shots_per_time_.get_number_of_elements(); it++)
        {

            auto ttraj = set_device(&trajectory[it], eligibleGPUs[ii]);
            nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc nrfc(this->recon_params_);

            nrfc.preprocess(&ttraj);
            temp_vector.push_back(nrfc);
        }
        nrfc_vector.push_back(temp_vector);
    }
    cudaSetDevice(cur_device);

    // eligibleGPUs.erase(std::remove(eligibleGPUs.begin(), eligibleGPUs.end(), cur_device), eligibleGPUs.end());

    trajectory_ = trajectory;
    is_preprocessed_ = true;
}

template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_fbins(arma::fvec fbins)
{
    fbins_ = fbins;
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_combination_weights(cuNDArray<float_complext> *combinationWeights)
{
    combinationWeights_ = combinationWeights;
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_recon_params(reconParams rp)
{
    recon_params_ = rp;
    // nc_recon_fc_ = new nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc(recon_params);
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_scaled_time(std::vector<cuNDArray<REAL>> &scaled_time)
{
    scaled_time_ = scaled_time;
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_dofield_correction(bool flag)
{
    doConcomitantFieldCorraction_ = flag;
    // if(is_preprocessed_)
    // reprocess to spread overGPUs
}

// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator_fc<float, 1>;
// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator_fc<float, 2>;
template class EXPORTGPUPMRI cuNonCartesianMOCOOperator_fc<float, 3>;
// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator_fc<float, 4>;

// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<double, 1>;
// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<double, 2>;
// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<double, 3>;
// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<double, 4>;
