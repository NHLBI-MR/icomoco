#pragma once
#include <gadgetron/hoNDArray.h>
#include <ismrmrd/ismrmrd.h>

namespace Gadgetron
{
//#pragma pack(push, 1) // 1-byte alignment
    struct FeedbackData
    {
        bool myBool;
        long myints[2];
        float myFloat;
    };
//#pragma pack(pop) // Restore old alignment
}
