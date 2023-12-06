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

using namespace Gadgetron;
using namespace Gadgetron::Core;

class RemoveAcqsGadget : public ChannelGadget<Core::Acquisition>
{

public:
    RemoveAcqsGadget(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Acquisition>(context, props)
    {
    }
    void process(InputChannel<Acquisition> &in, OutputChannel &out) override
    {
        for (auto message : in)
        {
            auto &[head, data, traj] = message;
            if (!(head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_RTFEEDBACK_DATA)))
            {
                continue;
            }
            else
            {
                out.push(Core::Acquisition(std::move(head), std::move(data), std::move(traj)));
            }
            
        }
    }
};

GADGETRON_GADGET_EXPORT(RemoveAcqsGadget)