#include "cuNonCartesianTSenseOperator.h"
#include <gadgetron/vector_td_utilities.h>
#include <gadgetron/GadgetronTimer.h>

using namespace Gadgetron;

template <class REAL, unsigned int D>
cuNonCartesianTSenseOperator<REAL, D>::cuNonCartesianTSenseOperator(ConvolutionType conv) : cuSenseOperator<REAL, D>()
{

    convolutionType = conv;
    is_preprocessed_ = false;
}

template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator<REAL, D>::mult_M(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out,
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

    // for (float ii = 0; ii < full_dimensions.size() - 3; ii++)
    // {
    //     char str[80];
    //     sprintf(str, "cuNonCartesianSenseOperator::mult_H: %d dimension should be same as %d dim size of shots per time", ii + 3, ii);
    //     if (full_dimensions[ii + 3] != shots_per_time_.get_size(0))
    //         throw std::runtime_error(std::string(str));
    // }

    // if (full_dimensions[3] != shots_per_time_.get_size(0))
    //     throw std::runtime_error(
    //         sprintf("cuNonCartesianSenseOperator::mult_H: %d dimension should be same as %d dim size of shots per time",ii+3,ii));

    // if (shots_per_time_.get_size(1) > 1 && full_dimensions[4] != shots_per_time_.get_size(1))
    //     throw std::runtime_error(
    //         "cuNonCartesianSenseOperator::mult_H: fourth dimension should be same as second dim size of shots per time");
    bool time_dims_2d = shots_per_time_.get_size(1) > 1;
    data_dimensions.pop_back(); // remove coil dimension from tmp_data;

    // auto timeD = full_dimensions[full_dimensions.size() - 1];
    // auto timeD = full_dimensions[full_dimensions.size() - 1];
    auto dims_orig = full_dimensions.size();
    for (auto ii = 0; ii < dims_orig - 3; ii++)
        full_dimensions.pop_back();
    // if (time_dims_2d)
    //     full_dimensions.pop_back();
    // full_dimensions.pop_back();

    full_dimensions.push_back(this->ncoils_);

    // full_dimensions.push_back(shots_per_time_.get_size(0));
    // full_dimensions.push_back(shots_per_time_.get_size(1));

    // std::iter_swap(full_dimensions.end(), full_dimensions.end() - 1); // swap the coil dimension and time

    // full_dimensions.pop_back(); // remove time dimension

    std::vector<size_t> slice_dimensions = *this->get_domain_dimensions();

    auto dims_orig2 = slice_dimensions.size();

    for (auto ii = 0; ii < dims_orig2 - 3; ii++)
        slice_dimensions.pop_back(); // remove time

    // slice_dimensions.pop_back(); // remove time
    auto stride = std::accumulate(slice_dimensions.begin(), slice_dimensions.end(), 1,
                                  std::multiplies<size_t>()); // product of X,Y,and Z

    std::vector<size_t> tmp_dims = *this->get_codomain_dimensions();
    auto stride_data = std::accumulate(tmp_dims.begin(), tmp_dims.end() - 1, 1, std::multiplies<size_t>());
    GadgetronTimer timer("Deconstruct");

    auto tmpview_dims = full_dimensions;

    for (size_t it = 0; it < shots_per_time_.get_number_of_elements(); it++)
    {
        auto inter_acc = std::accumulate(shots_per_time_.begin(),
                                         shots_per_time_.begin() + it, size_t(0)); // sum of cum sum shots per time

        auto slice_view_in = cuNDArray<complext<REAL>>(slice_dimensions, in->data() + stride * it);

        cuNDArray<complext<REAL>> tmp(&full_dimensions);
        this->mult_csm(&slice_view_in, &tmp);

        data_dimensions.pop_back();                                 // remove interleave
        data_dimensions.push_back(*(shots_per_time_.begin() + it)); // insert correct interleave
        // data_dimensions.push_back(this->ncoils_);       // insert coils again

        // cuNDArray<complext<REAL>> tmp_data(&data_dimensions);

        full_dimensions.pop_back(); // remove ch
        try
        {
            
            data_dimensions.push_back(this->ncoils_);       // insert coils again
            cuNDArray<complext<REAL>> tmp_out(&data_dimensions);

            if (accumulate)
            {
                cuNDArray<complext<REAL>> tmp_out2(&data_dimensions);
                plan_[it]->compute(tmp, tmp_out2, &dcw_[it], NFFT_comp_mode::FORWARDS_C2NC);
                tmp_out += tmp_out2;
            }
            else
                plan_[it]->compute(tmp, tmp_out, &dcw_[it], NFFT_comp_mode::FORWARDS_C2NC);
            
            data_dimensions.pop_back();
            
            for (size_t iCHA = 0; iCHA < this->ncoils_; iCHA++)
                cudaMemcpyAsync(out->data() + inter_acc * data_dimensions[0] + stride_data * iCHA, tmp_out.get_data_ptr() + *(shots_per_time_.begin() + it) * data_dimensions[0] * iCHA, 
                *(shots_per_time_.begin() + it) * data_dimensions[0] *sizeof(float_complext),cudaMemcpyDefault);

         //   throw 505;
        }
        catch (...)
        {
            GDEBUG_STREAM("TSENSE - doing channel by channel C2NC");
            for (size_t iCHA = 0; iCHA < this->ncoils_; iCHA++)
            {
                auto tmp_view = cuNDArray<complext<REAL>>(full_dimensions, tmp.data() + stride * iCHA);

                auto slice_view_out =
                    cuNDArray<complext<REAL>>(data_dimensions, out->data() + inter_acc + stride_data * iCHA);

                if (accumulate)
                {
                    cuNDArray<complext<REAL>> tmp_out(&full_dimensions);
                    plan_[it]->compute(tmp_view, tmp_out, &dcw_[it], NFFT_comp_mode::FORWARDS_C2NC);
                    slice_view_out += tmp_out;
                }
                else
                // slice_view_out
                {

                    // cuNDArray<complext<REAL>> tmp_new(std::vector<size_t>({1340,1848}));
                    plan_[it]->compute(tmp_view, slice_view_out, &dcw_[it], NFFT_comp_mode::FORWARDS_C2NC);
                }
            }
        }

        full_dimensions.push_back(this->ncoils_);
        // size_t inter_acc = 0;
        // if (it > 0)

        // This is not correct yet ! -- AJ
        // for (size_t iCHA = 0; iCHA < this->ncoils_; iCHA++)
        //     cudaMemcpy(out->get_data_ptr() + inter_acc + stride_data * iCHA,
        //                tmp_data.get_data_ptr() + tmp_data.get_size(0) * tmp_data.get_size(1) * iCHA,
        //                tmp_data.get_size(0) * tmp_data.get_size(1) * sizeof(complext<REAL>), cudaMemcpyDefault);
    }
}

template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator<REAL, D>::mult_MH(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out,
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

    //GDEBUG_STREAM("out_dimensions.size(): " << out_dimensions.size());
    auto dims_orig = (out_dimensions.size());
    for (auto ii = 0; ii < dims_orig - 3; ii++)
    {
      //  GDEBUG_STREAM("out_dimensions.end(): " << out_dimensions[out_dimensions.size() - 1]);
        out_dimensions.pop_back();
    }
    //GDEBUG_STREAM("out_dimensions.size(): " << out_dimensions.size());

    // out_dimensions.pop_back();               // Remove the timeDimension
    out_dimensions.push_back(this->ncoils_); // add coil dimension
    cuNDArray<complext<REAL>> tmp(&out_dimensions);
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
    GadgetronTimer timer("Reconstruct");

    for (size_t it = 0; it < shots_per_time_.get_number_of_elements(); it++)
    {

        auto inter_acc = std::accumulate(shots_per_time_.begin(), shots_per_time_.begin() + it, 0) ;
        in_dimensions.pop_back(); // Remove INT dimension
        in_dimensions.push_back(*(shots_per_time_.begin() + it));
        // GDEBUG_STREAM("shots_per_time: " << *(shots_per_time_.begin()+it));
        try
        {
            auto slice_view = crop<float_complext, 3>(uint64d3(0, inter_acc, 0),
                                                      uint64d3(RO, *(shots_per_time_.begin() + it), this->ncoils_),
                                                      *in);

            plan_[it]->compute(slice_view, tmp, &dcw_[it], NFFT_comp_mode::BACKWARDS_NC2C);
      //      throw 505;
        }
        catch (...)
        {
            GDEBUG_STREAM("TSENSE - doing channel by channel NC2C");

            for (size_t ich = 0; ich < CHA; ich++)
            {

                auto slice_view = cuNDArray<complext<REAL>>(in_dimensions, in->data() + stride_ch * ich + inter_acc * in_dimensions[0]);
                auto out_view_ch = cuNDArray<complext<REAL>>(out_dimensions, tmp.data() + stride_out * ich);

                plan_[it]->compute(slice_view, out_view_ch, &dcw_[it], NFFT_comp_mode::BACKWARDS_NC2C);
            }
        }
        auto slice_view_output = cuNDArray<complext<REAL>>(out_dimensions, out->data() + stride_out * it);

        this->mult_csm_conj_sum(&tmp, &slice_view_output);
    }
}

template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator<REAL, D>::setup(_uint64d matrix_size, _uint64d matrix_size_os, REAL W)
{
    for (auto ii = 0; ii < shots_per_time_.get_number_of_elements(); ii++)
        plan_.push_back(NFFT<cuNDArray, REAL, D>::make_plan(matrix_size, matrix_size_os, W, convolutionType));
}

template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator<REAL, D>::preprocess(std::vector<cuNDArray<_reald>> &trajectory)
{
    if (&(*trajectory.begin()) == 0x0)
    {
        throw std::runtime_error("cuNonCartesianSenseOperator: cannot preprocess 0x0 trajectory.");
    }

    boost::shared_ptr<std::vector<size_t>> domain_dims = this->get_domain_dimensions();
    if (domain_dims.get() == 0x0 || domain_dims->size() == 0)
    {
        throw std::runtime_error("cuNonCartesianSenseOperator::preprocess : operator domain dimensions not set");
    }
    for (auto ii = 0; ii < shots_per_time_.get_number_of_elements(); ii++)
        plan_[ii]->preprocess(trajectory[ii], NFFT_prep_mode::ALL);
    is_preprocessed_ = true;
}

template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator<REAL, D>::set_dcw(std::vector<cuNDArray<REAL>> dcw)
{
    dcw_ = dcw;
}

template <class REAL, unsigned int D>
void cuNonCartesianTSenseOperator<REAL, D>::set_shots_per_time(hoNDArray<size_t> shots_per_time)
{
    shots_per_time_ = shots_per_time;
}

template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<float, 1>;
template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<float, 2>;
template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<float, 3>;
template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<float, 4>;

// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<double, 1>;
// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<double, 2>;
// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<double, 3>;
// template class EXPORTGPUPMRI cuNonCartesianTSenseOperator<double, 4>;
