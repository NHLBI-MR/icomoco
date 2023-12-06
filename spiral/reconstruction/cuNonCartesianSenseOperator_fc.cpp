#include "cuNonCartesianSenseOperator_fc.h"
#include "vector_td_utilities.h"

using namespace Gadgetron;

template <class REAL, unsigned int D>
void cuNonCartesianSenseOperator_fc<REAL, D>::mult_M(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out,
                                                     bool accumulate)
{

    if (!in || !out)
    {
        throw std::runtime_error("cuNonCartesianSenseOperator_fc::mult_M : 0x0 input/output not accepted");
    }
    if (!in->dimensions_equal(&this->domain_dims_) || !out->dimensions_equal(&this->codomain_dims_))
    {
        throw std::runtime_error(
            "cuNonCartesianSenseOperator_fc::mult_H: input/output arrays do not match specified domain/codomains");
    }
    // Cart -> noncart
    std::vector<size_t> full_dimensions = *this->get_domain_dimensions(); // cart
    full_dimensions.push_back(this->ncoils_);

    cuNDArray<complext<REAL>> tmp(&full_dimensions);

    this->mult_csm(in, &tmp);

    nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc nrfc(recon_params_);

    // Forwards NFFT
    if (accumulate)
    {
        cuNDArray<complext<REAL>> tmp_out(out->get_dimensions());
        this->plan_->compute(tmp, tmp_out, this->dcw_.get(), NFFT_comp_mode::FORWARDS_C2NC);
        *out += tmp_out;
    }
    else
        nrfc.deconstruct(tmp, *out, *trajectory_, *this->dcw_, *combinationWeights_, *scaled_time_, fbins_);
    //plan_->compute(tmp, *out, this->dcw_.get(), NFFT_comp_mode::FORWARDS_C2NC);
}

template <class REAL, unsigned int D>
void cuNonCartesianSenseOperator_fc<REAL, D>::mult_MH(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out,
                                                      bool accumulate)
{

    if (!in || !out)
    {
        throw std::runtime_error("cuNonCartesianSenseOperator_fc::mult_MH : 0x0 input/output not accepted");
    }

    if (!in->dimensions_equal(&this->codomain_dims_) || !out->dimensions_equal(&this->domain_dims_))
    {
        throw std::runtime_error(
            "cuNonCartesianSenseOperator_fc::mult_MH: input/output arrays do not match specified domain/codomains");
    }
    std::vector<size_t> out_dimensions = *this->get_domain_dimensions();
    std::vector<size_t> in_dimensions = *this->get_codomain_dimensions();

    auto RO = in->get_size(0);
    auto E1E2 = in->get_size(1);
    auto CHA = in->get_size(2);

    nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc nrfc(recon_params_);

    in_dimensions.pop_back(); // Remove CH dimension

    out_dimensions.push_back(this->ncoils_); // add coil dimension
    cuNDArray<complext<REAL>> tmp(&out_dimensions);
    out_dimensions.pop_back(); // rm coil dimension

    auto stride_ch = std::accumulate(in_dimensions.begin(), in_dimensions.end(), 1, std::multiplies<size_t>());

    auto stride_out = std::accumulate(out_dimensions.begin(), out_dimensions.end(), 1, std::multiplies<size_t>());
    in_dimensions.push_back(CHA);
    // Remove channel dimension if the last dimension is the same as the number of coils
    if (in_dimensions[in_dimensions.size() - 1] == this->ncoils_ && in_dimensions.size() > 2)
    {

        nrfc.reconstruct(*in, tmp, *trajectory_, *this->dcw_, *combinationWeights_, *scaled_time_, fbins_);
        // for (size_t ich = 0; ich < CHA; ich++) {

        //     auto slice_view = cuNDArray<complext<REAL>>(in_dimensions, in->data() + stride_ch * ich);
        //     auto out_view_ch = cuNDArray<complext<REAL>>(out_dimensions, tmp.data() + stride_out * ich);

        //     plan_->compute(slice_view, out_view_ch, dcw_.get(), NFFT_comp_mode::BACKWARDS_NC2C);
        // }
    }
    else
    {
        // throw std::runtime_error("cuNonCartesianSenseOperator_fc::Last dimension is not the coil dimension");
        this->plan_->compute(in, tmp, this->dcw_.get(), NFFT_comp_mode::BACKWARDS_NC2C);
    }

    if (!accumulate)
    {
        clear(out);
    }

    this->mult_csm_conj_sum(&tmp, out);
}

template <class REAL, unsigned int D>
void cuNonCartesianSenseOperator_fc<REAL, D>::preprocess(cuNDArray<vector_td<REAL, D>> *trajectory)
{
    if (trajectory == 0x0)
    {
        throw std::runtime_error("cuNonCartesianSenseOperator_fc: cannot preprocess 0x0 trajectory.");
    }

    boost::shared_ptr<std::vector<size_t>> domain_dims = this->get_domain_dimensions();
    if (domain_dims.get() == 0x0 || domain_dims->size() == 0)
    {
        throw std::runtime_error("cuNonCartesianSenseOperator_fc::preprocess : operator domain dimensions not set");
    }
    this->plan_->preprocess(trajectory, NFFT_prep_mode::ALL);
    trajectory_ = trajectory;
    this->is_preprocessed_ = true;
}

template <class REAL, unsigned int D>
void cuNonCartesianSenseOperator_fc<REAL, D>::set_fbins(arma::fvec fbins)
{
    fbins_ = fbins;
}
template <class REAL, unsigned int D>
void cuNonCartesianSenseOperator_fc<REAL, D>::set_combination_weights(cuNDArray<float_complext> *combinationWeights)
{
    combinationWeights_ = combinationWeights;
}
template <class REAL, unsigned int D>
void cuNonCartesianSenseOperator_fc<REAL, D>::set_recon_params(reconParams rp)
{
    recon_params_ = rp;
    // nc_recon_fc_ = new nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc(recon_params);
}
template <class REAL, unsigned int D>
void cuNonCartesianSenseOperator_fc<REAL, D>::set_scaled_time(cuNDArray<REAL> *scaled_time)
{
    scaled_time_ = scaled_time;
}
template <class REAL, unsigned int D>
void cuNonCartesianSenseOperator_fc<REAL, D>::set_dofield_correction(bool flag)
{
    doConcomitantFieldCorraction_ = flag;
    // if(is_preprocessed_)
    // reprocess to spread overGPUs
}

//
// Instantiations
//

//template class EXPORTGPUPMRI cuNonCartesianSenseOperator_fc<float, 1>;
//template class EXPORTGPUPMRI cuNonCartesianSenseOperator_fc<float, 2>;
template class EXPORTGPUPMRI cuNonCartesianSenseOperator_fc<float, 3>;
//template class EXPORTGPUPMRI cuNonCartesianSenseOperator_fc<float, 4>;

// template class EXPORTGPUPMRI cuNonCartesianSenseOperator_fc<double, 1>;
// template class EXPORTGPUPMRI cuNonCartesianSenseOperator_fc<double, 2>;
// template class EXPORTGPUPMRI cuNonCartesianSenseOperator_fc<double, 3>;
// template class EXPORTGPUPMRI cuNonCartesianSenseOperator_fc<double, 4>;
