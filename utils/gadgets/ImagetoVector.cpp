
#include <gadgetron/GadgetMRIHeaders.h>
#include <gadgetron/Node.h>
#include <gadgetron/Types.h>
#include <gadgetron/hoNDArray.h>
#include <gadgetron/hoNDArray_math.h>
#include <gadgetron/mri_core_data.h>
#include <gadgetron/gadgetron_mricore_export.h>

class ImagetoVector : public Core::ChannelGadget<Core::Image<double>>
{
public:
    using Core::ChannelGadget<Core::Image<double>>::ChannelGadget;

    std::vector<size_t> ImagetoVector_convert(hoNDArray<double> data)
    {
        GDEBUG_STREAM("Indices Number: "<< data(0, 0));
        std::vector<size_t> indices(data.get_data_ptr()+1, data.get_data_ptr() + size_t(1 + data(0, 0)));

        return indices;
    }
    void process(Core::InputChannel<Core::Image<double>> &in, Core::OutputChannel &out) override
    {
        std::vector<hoNDArray<double>> images;
        std::vector<size_t> images_index;
        for (auto message : in)
        {
            auto &[imhead, data, meta] = message;
            // visit([&](auto message)
            //       { splitInputData(message, out); },
            //       msg);
            images.push_back(data);
            images_index.push_back(imhead.image_index);
        }

        std::vector<std::vector<size_t>> idx_to_send;
        for (auto msg : images)
        {
            idx_to_send.push_back(ImagetoVector_convert(msg));
        }
        GDEBUG_STREAM("idx_to_send: "<< idx_to_send.size());
        GDEBUG_STREAM("idx_to_send[0]: "<< idx_to_send[0].size());

        out.push(idx_to_send);
    }
};
GADGETRON_GADGET_EXPORT(ImagetoVector)
