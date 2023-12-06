#include <gadgetron/Node.h>
#include "TrajectoryParameters.h"
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/vector_td_utilities.h>
#include <gadgetron/mri_core_acquisition_bucket.h>
#include <complex>
#include <ismrmrd/ismrmrd.h>
#include <map>
#include <gadgetron/mri_core_data.h>
#include <boost/algorithm/string.hpp>
#include <gadgetron/hoNDArray_elemwise.h>
#include <gadgetron/hoNDArray_reductions.h>
#include <gadgetron/gadgetron_mricore_export.h>
#include <ismrmrd/xml.h>
#include <gadgetron/cuNFFT.h>
#include <gadgetron/cuNDFFT.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/cuNDArray.h>
#include "../utils/util_functions.h"
#include <gadgetron/GadgetronTimer.h>
#include <gadgetron/hoArmadillo.h>

#include "SpiralBuffer.h"
using namespace Gadgetron;
using namespace Gadgetron::Core;

class SpiralAcquisitionCropData : public ChannelGadget<Core::Acquisition>
{

public:
    SpiralAcquisitionCropData(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Acquisition>(context, props), header{context.header}, trajParams{context.header}
    {
    }

    void process(InputChannel<Acquisition> &in, OutputChannel &out) override
    {
        size_t acq_count = 0;
        bool csmSent = false;

        using namespace Gadgetron::Indexing;
        //auto [acq_header, data, traj] = in.pop();

        std::vector<Core::Acquisition> acquisitionsvec;
        //acquisitionsvec.push_back(std::move(Core::Acquisition((acq_header), (data), (traj))));

        auto maxZencode = header.encoding.front().encodingLimits.kspace_encoding_step_2.get().maximum + 1;
        auto maxAcq = ((header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.kspace_encoding_step_2.get().maximum + 1) * (header.encoding[0].encodingLimits.average.get().maximum + 1) * (header.encoding[0].encodingLimits.repetition.get().maximum + 1)); // use -1 for data acquired b/w 12/23 - 01/21

        // auto RO = data.get_size(0);
        // auto CHA = acq_header.active_channels;
        auto numInttotal = maxAcq / maxZencode;
        std::vector<size_t> scanINT;

        uint64_t lastFlags;
        std::vector<size_t> scan_counter;
        auto counterNav = 0;
        auto counterAcq = 0;
        /* Getting data, trajectories and headers and sorting them and pushing them along */
        for (auto [acq_header, data, traj] : in)
        {
            if (acq_header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT))
                lastFlags = acq_header.flags;

            scan_counter.push_back(acq_header.scan_counter);
            if ((acq_header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA) || acq_header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_RTFEEDBACK_DATA)))
                counterNav++;
            acquisitionsvec.push_back((Core::Acquisition(std::move(acq_header), std::move(data), std::move(traj))));
        }
        auto navsPerStack = ceil(float(counterNav) / float(numInttotal));
        GDEBUG_STREAM("numNavsPerStack Cropping:" << navsPerStack);
        GDEBUG_STREAM("counterNav Cropping:" << counterNav);
        GDEBUG_STREAM("numInttotal Cropping:" << numInttotal);

        for (auto ii = 0; ii < fixedINTscan_interval; ii++)
        {
            if (fixedINTscan_interval > 1)
            {
                auto numacq = size_t(fixedINTscan_start +
                                     ((ii * (((numInttotal < fixedINTscan_end) ? numInttotal : fixedINTscan_end) - fixedINTscan_start)) / (fixedINTscan_interval - 1))) *
                              (maxZencode + navsPerStack);
                GDEBUG_STREAM("Data sizes to reconstruct: " << numacq);

                //size_t(fixedINTscan_start +
                //((ii * (((numInttotal < fixedINTscan_end) ? numInttotal : fixedINTscan_end) - fixedINTscan_start)) / (fixedINTscan_interval - 1))) *
                //maxZencode
                scanINT.push_back(numacq);
            }
            else
            {
                auto data_size_tor = size_t(((fixedINTscan_start < numInttotal) ? fixedINTscan_start : numInttotal));
                data_size_tor *= (maxZencode + navsPerStack);
                GDEBUG_STREAM("Data sizes to reconstruct: " << data_size_tor);
                scanINT.push_back(data_size_tor);
            }
        }

        for (auto numSend = 0; numSend < scanINT.size(); numSend++)
        {
            GadgetronTimer timer("Sending less data");

            auto acqC = 0;
            auto acqS = 0;
            while (acqS < scanINT[numSend])
            {
                auto index_send = std::find(scan_counter.begin(), scan_counter.end(), acqS) - scan_counter.begin();
                if (index_send < scan_counter.size())
                {
                    auto &[head, data, traj] = acquisitionsvec[index_send];
                    if (acqS == (scanINT[numSend] - 1))
                    {

                        head.flags = lastFlags;
                        head.setFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT);
                        out.push(std::move(Core::Acquisition(std::move(head), std::move(data), std::move(traj))));
                    }
                    else
                    {
                        // GDEBUG_STREAM("index_send: " << index_send);
                        // GDEBUG_STREAM("acqS: " << acqS);
                        // GDEBUG_STREAM("head.scan_counter: " << head.scan_counter);

                        out.push(std::move(Core::Acquisition(std::move(head), std::move(data), std::move(traj))));
                    }
                }
                acqS++;

                //out.push(std::move(acquisitionsvec[index_send]));

                //  auto &[head, data, traj] = acquisitionsvec[index_send];
                // if (!(head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_RTFEEDBACK_DATA)))
                //     acqC++;
            }
        }
    }

protected:
    ISMRMRD::IsmrmrdHeader header;

    NODE_PROPERTY(fixedINTscan_start, size_t, "fixedINT for reducedScan", 127);
    NODE_PROPERTY(fixedINTscan_end, size_t, "fixedINT for reducedScan max", 411);
    NODE_PROPERTY(fixedINTscan_interval, size_t, "Numer of points", 7); //
private:
    size_t avg = 0;
    size_t phase = 0;
    nhlbi_toolbox::Spiral::TrajectoryParameters trajParams;
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
};

GADGETRON_GADGET_EXPORT(SpiralAcquisitionCropData)