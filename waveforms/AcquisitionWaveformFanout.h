#pragma once
#include <gadgetron/Node.h>

#include <ismrmrd/ismrmrd.h>

#include <gadgetron/Fanout.h>
#include <gadgetron/Types.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;

using AcquisitionWaveformFanout = Gadgetron::Core::Parallel::Fanout<variant<Acquisition, Waveform>>;
