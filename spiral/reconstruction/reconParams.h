#pragma once
#include <gadgetron/hoNDArray.h>
#include <ismrmrd/ismrmrd.h>

namespace Gadgetron
{
    struct reconParams
    {
        ISMRMRD::MatrixSize ematrixSize;
        ISMRMRD::MatrixSize rmatrixSize;
        ISMRMRD::FieldOfView_mm fov;
        hoNDArray<size_t> shots_per_time;
        size_t numberChannels;
        size_t RO;
        float oversampling_factor_;
        float kernel_width_;
        size_t iterations = 10;
        size_t iterations_imoco = 10;
        size_t iterations_inner = 2;
        float tolerance = 1e-3;
        size_t iterations_dcf = 10;
        float kernel_width_dcf_ = 3;
        float oversampling_factor_dcf_ = 5.5;
        int selectedDevice = 0;
        float lambda_spatial = 1e-1;
        float lambda_spatial_imoco = 1e-1;
        float lambda_time = 1e-1;
        float lambda_time2 = 1e-1;
        size_t norm = 2;
        bool useIterativeDCWEstimated = false;
    };
}