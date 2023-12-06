#include <gadgetron/Node.h>
#include <gadgetron/mri_core_utility.h>
#include <gadgetron/mri_core_acquisition_bucket.h>
#include <ismrmrd/xml.h>
#include <gadgetron/gadgetron_mricore_export.h>
#include <gadgetron/mri_core_def.h>
#include <gadgetron/mri_core_data.h>
#include <gadgetron/hoNDArray_math.h>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoNDArray.h>
#include <ismrmrd/ismrmrd.h>
#include <gadgetron/GadgetronTimer.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;

class RemoveSpiralOversampling : public ChannelGadget<Core::Acquisition>
{

public:
    RemoveSpiralOversampling(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Acquisition>(context, props)
    {
    }
    void process(InputChannel<Acquisition> &in, OutputChannel &out) override
    {
        // #pragma omp parallel
        // #pragma omp for
        for (auto message : in)
        {

            using namespace Gadgetron::Indexing;
            auto &[head, data, traj] = message;
            uint16_t extraSamples = 0; // default crop on VDS seq it will be reset with discard pre and or UTE seq
            //GDEBUG_STREAM(" head.discard_pre: " << uint16_t(head.discard_pre));
            if (head.discard_pre > 0)
                extraSamples = head.discard_pre;
            else
                extraSamples = data.get_size(0) - traj->get_size(1);

            if (!(head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_HPFEEDBACK_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_RTFEEDBACK_DATA)))
            {
                if (extraSamples > 0)
                {
                    if (extraSamples < 11)
                    {
                        hoNDArray<std::complex<float>> data_short(data.get_size(0) - extraSamples, head.active_channels);
                        #pragma omp parallel
                        #pragma omp for
                        for (int ii = extraSamples; ii < data.get_size(0); ii++)
                        {
                            data_short(ii - extraSamples, slice) = data(ii, slice);
                        }
                        data = data_short;
                        head.number_of_samples = data.get_size(0);
                    }
                    else
                    {
                        hoNDArray<std::complex<float>> temp{traj->get_size(1) - head.discard_pre, data.get_size(1)};
                        hoNDArray<float> temp_traj{traj->get_size(1) - head.discard_pre, traj->get_size(0)};

                        std::vector<size_t> data_dimensions = *(data.get_dimensions());
                        data_dimensions[0] = head.number_of_samples;

                        auto temp_dims = *(temp.get_dimensions());
                        temp_dims.pop_back();
                        size_t cropSize = 0;
                        if (head.discard_pre > 0)
                            cropSize = head.discard_pre;
                        else
                            cropSize = (extraSamples > crop_index_st) ? 20 : crop_index_st;
                        for (auto ich = 0; ich < data_dimensions[data_dimensions.size() - 1]; ich++)
                        {

                            temp(slice, ich) = hoNDArray<std::complex<float>>(temp_dims, data.data() + data.get_size(0) * ich + cropSize);
                        }

                        if (head.discard_pre > 0)
                        {
                            auto ptraj = permute(*traj, {1, 0});
                            for (auto ich = 0; ich < traj->get_size(0); ich++)
                            {

                                temp_traj(slice, ich) = hoNDArray<float>(temp_dims, ptraj.data() + ptraj.get_size(0) * ich + cropSize);
                            }
                            // crop<std::complex<float>, 2>(vector_td<size_t, 2>(crop_index_st, 0),
                            //                                                vector_td<size_t, 2>(traj->get_size(1), data.get_size(1)),
                            //                                                data,temp);
                            *traj = permute(temp_traj, {1, 0});
                        }
                        data = temp;
                        head.number_of_samples = data.get_size(0);
                    }
                }
            }
            out.push(Core::Acquisition(std::move(head), std::move(data), std::move(traj)));
        }
    }

protected:
    NODE_PROPERTY(crop_index_st, size_t, "start index to crop acquisition data", 20);
};

GADGETRON_GADGET_EXPORT(RemoveSpiralOversampling)