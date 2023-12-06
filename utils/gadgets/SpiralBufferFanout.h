#pragma once
#include <gadgetron/Node.h>

#include <ismrmrd/ismrmrd.h>

#include <gadgetron/Fanout.h>
#include <gadgetron/Types.h>
#include "../spiral/SpiralBuffer.h"
#include <gadgetron/cuNDArray.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;

using SpiralBufferFanout = Gadgetron::Core::Parallel::Fanout<Gadgetron::SpiralBuffer<cuNDArray, float_complext, 2>>;
