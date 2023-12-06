/*
  An example of how to estimate DCF
*/
#pragma once
// Gadgetron includes
#include <gadgetron/cuSDC.h>
#include <gadgetron/hoNDArray_utils.h>

#include <gadgetron/cuNFFT.h>
#include <gadgetron/cuNDFFT.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/cuNDArray.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/cuNDArray_operators.h>
#include <gadgetron/cuNonCartesianSenseOperator.h>
#include <gadgetron/cuCgSolver.h>
#include <gadgetron/cuNlcgSolver.h>
#include <gadgetron/cuCgPreconditioner.h>
#include <gadgetron/cuImageOperator.h>
#include <gadgetron/cuGpBbSolver.h>
#include <gadgetron/cuTvOperator.h>
#include <gadgetron/cuTvPicsOperator.h>
#include <gadgetron/cuNlcgSolver.h>
#include <gadgetron/cuPartialDerivativeOperator.h>
#include <gadgetron/cuSbcCgSolver.h>

#include <gadgetron/hoNDArray_fileio.h>
#include <gadgetron/parameterparser.h>
#include <gadgetron/NFFTOperator.h>

// Std includes
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <gadgetron/GadgetronTimer.h>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <gadgetron/mri_core_coil_map_estimation.h>
#include <gadgetron/hoArmadillo.h>
#include "util_functions.h"
#include "noncartesian_reconstruction.h"
#include "noncartesian_reconstruction_4D.h"
#include "noncartesian_reconstruction_3D.h"
#include "reconParams.h"
#include "densityCompensation.h"
#include <complex>

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace std;
using namespace nhlbi_toolbox::reconstruction;
// Define desired precision
typedef float _real;
namespace po = boost::program_options;
uint64d3 image_dims_os_;
std::vector<size_t> image_dims_;
float kernel_width_;
boost::shared_ptr<cuNFFT_plan<float, 3>> nfft_plan_;
boost::shared_ptr<cuNDArray<float_complext>> cuData;

int main(int argc, char **argv)
{
    std::vector<size_t> recon_dims;

    //
    // Parse command line
    //
    std::string trajectory_file;
    std::string def_file;
    std::string invdef_file;
    std::string data_file;
    std::string dcw_file;
    std::string csm_file;
    std::string out_file;
    std::string recondim_file;
    std::string erecondim_file;
    std::string fov_file;
    std::string out_file_csm;
    std::string shotspertime_file;
    std::string combination_weights_file;
    std::string scaled_time_file;
    std::string fbins_file;

    float oversampling_factor_;
    float lambda_spatial;
    float lambda_time;
    float lambda_time2;
    float tolerance;
    float lF;
    float uF;

    bool calculateCSM;
    bool isCI;

    int iterations_;

    bool pseudoReplica;
    size_t xsize_;
    size_t ysize_;
    size_t ezsize_;
    size_t rzsize_;
    size_t numFbins;
    size_t reconstructionType; // 0-csm + recon, 1-cgsense, 2-3DTV, 3-4DspatialTempTV

    // Initialize to prevent errors
    lF = 0;
    uF = 0;
    numFbins = 1;
    pseudoReplica = false;
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Produce help message")                                                          //
        ("trajectory,t", po::value<std::string>(&trajectory_file), "trajectory file to read trajectory")          //
        ("shortPerTime,spt", po::value<std::string>(&shotspertime_file), "shots per time")                        //
        ("data,d", po::value<std::string>(&data_file), "data file to read data")                                  //
        ("dcw,w", po::value<std::string>(&dcw_file), "dcw file to read dcw")                                      //
        ("csm,c", po::value<std::string>(&csm_file), "csm file to read csm")                                      //
        ("combination_weights,cw", po::value<std::string>(&combination_weights_file), "combination_weights_file") //
        ("scaled_time,sct", po::value<std::string>(&scaled_time_file), "scaled_time_file")                        //
        ("fbins,fb", po::value<std::string>(&fbins_file), "fbins_file")                                           //
        ("out,o", po::value<std::string>(&out_file), "out file to write images")                                  //
        ("csmout,csmout", po::value<std::string>(&out_file_csm), "outputFileCSM")                                 //
        ("reconDims,rd", po::value<std::string>(&recondim_file), "recon dims")                                    //
        ("ereconDims,erd", po::value<std::string>(&erecondim_file), "erecon dims")                                //
        ("fov,fo", po::value<std::string>(&fov_file), "fov dims")                                                 //
        ("oversampling,f", po::value<float>(&oversampling_factor_), "oversampling factor")                        //
        ("iterations,i", po::value<int>(&iterations_), "size of reconst z")                                       //
        ("kwidth,k", po::value<float>(&kernel_width_), "kernel width")                                            //
        ("calculateCSM,s", po::value<bool>(&calculateCSM), "Flag to calculate CSM")                               //
        ("tolerance,tol", po::value<float>(&tolerance), "tolerance_sense")                                        //
        ("lambda_spatial,ls", po::value<float>(&lambda_spatial), "lambda_spatial")                                //
        ("lambda_time,lt", po::value<float>(&lambda_time), "lambda_time")                                         //
        ("lambda_time2,lt", po::value<float>(&lambda_time2), "lambda_time2")                                         //
        ("deformation,def", po::value<std::string>(&def_file), "deformation field")                               //
        ("invdeformation,idef", po::value<std::string>(&invdef_file), "inv deformation field")                    //
        ("reconstructionType,rt", po::value<size_t>(&reconstructionType), "reconstructionType: 0-csm + gridding recon, 1-cgsense, 2-3DTV, 3-4DspatialTempTV, 4- concom Sense spTV3D, 5- concom 4DspatialTempTV, 6- icomoco_fc, 7-4DspatialTempTV_nlcg  ")("pseudoReplica,pr", po::value<bool>(&pseudoReplica), "true/false");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    auto eligibleGPUs = nhlbi_toolbox::utils::FindCudaDevices(0);

    int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();

    if (std::find(eligibleGPUs.begin(), eligibleGPUs.end(), selectedDevice) == eligibleGPUs.end())
        selectedDevice = eligibleGPUs[0];

    GDEBUG_STREAM("Selected Device: " << selectedDevice);
    cudaSetDevice(selectedDevice);

    unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();
    GDEBUG_STREAM("warp_size: " << warp_size);

    reconParams recon_params;

    boost::shared_ptr<hoNDArray<vector_td<float, 3>>> trajectory_in =
        read_nd_array<floatd3>((char *)trajectory_file.c_str());

    boost::shared_ptr<hoNDArray<float_complext>> data_in =
        read_nd_array<float_complext>((char *)data_file.c_str());

    boost::shared_ptr<hoNDArray<float>> dcf_in =
        read_nd_array<float>((char *)dcw_file.c_str());

    boost::shared_ptr<hoNDArray<float>> recon_dims_read =
        read_nd_array<float>((char *)recondim_file.c_str());

    boost::shared_ptr<hoNDArray<float>> fov_file_read =
        read_nd_array<float>((char *)fov_file.c_str());

    boost::shared_ptr<hoNDArray<float>> erecon_dims_read =
        read_nd_array<float>((char *)erecondim_file.c_str());

    boost::shared_ptr<hoNDArray<float_complext>> csm_in =
        read_nd_array<float_complext>((char *)csm_file.c_str());

    boost::shared_ptr<hoNDArray<float>> def_in =
        read_nd_array<float>((char *)def_file.c_str());

    boost::shared_ptr<hoNDArray<float>> invdef_in =
        read_nd_array<float>((char *)invdef_file.c_str());

    hoNDArray<size_t> shots_per_time =
        hoNDArray<size_t>(*read_nd_array<float>((char *)shotspertime_file.c_str()));

    hoNDArray<float_complext> hocw =
        hoNDArray<float_complext>(*read_nd_array<float_complext>((char *)combination_weights_file.c_str()));

    hoNDArray<float> hosct =
        hoNDArray<float>(*read_nd_array<float>((char *)scaled_time_file.c_str()));

    hoNDArray<float> hofbins =
        hoNDArray<float>(*read_nd_array<float>((char *)fbins_file.c_str()));

    std::vector<float> mat_size(recon_dims_read.get()->get_data_ptr(), recon_dims_read.get()->get_data_ptr() + recon_dims_read.get()->get_number_of_elements());
    std::vector<float> emat_size(erecon_dims_read.get()->get_data_ptr(), erecon_dims_read.get()->get_data_ptr() + recon_dims_read.get()->get_number_of_elements());
    std::vector<float> fov_mat(fov_file_read.get()->get_data_ptr(), fov_file_read.get()->get_data_ptr() + fov_file_read.get()->get_number_of_elements());
    // hoNDArray<size_t> shots_per_time(shotspertime_read.get_data_ptr(), shotspertime_read.get_data_ptr() + shotspertime_read.get_number_of_elements());
    std::vector<float> fbins_vec(hofbins.get_data_ptr(), hofbins.get_data_ptr() + hofbins.get_number_of_elements());

    arma::fvec fbins(fbins_vec);

    auto cw = cuNDArray<float_complext>(hocw);
    auto sct = cuNDArray<float>(hosct);

    ISMRMRD::MatrixSize ematsize, rmatsize;
    ISMRMRD::FieldOfView_mm fov;
    ematsize.x = size_t(emat_size[0]);
    ematsize.y = size_t(emat_size[1]);
    ematsize.z = size_t(emat_size[2]);

    rmatsize.x = size_t(mat_size[0]);
    rmatsize.y = size_t(mat_size[1]);
    rmatsize.z = size_t(mat_size[2]);

    fov.x = fov_mat[0];
    fov.y = fov_mat[1];
    fov.z = fov_mat[2];

    recon_params.ematrixSize = ematsize;
    recon_params.rmatrixSize = rmatsize;
    recon_params.fov = fov;
    recon_params.shots_per_time = shots_per_time;
    recon_params.oversampling_factor_ = oversampling_factor_;
    recon_params.kernel_width_ = kernel_width_;
    recon_params.iterations = iterations_;
    recon_params.tolerance = tolerance;
    recon_params.numberChannels = data_in.get()->get_size(data_in.get()->get_number_of_dimensions() - 1);
    recon_params.selectedDevice = selectedDevice;
    recon_params.lambda_spatial = lambda_spatial;
    recon_params.lambda_time2   = lambda_time2;
    recon_params.lambda_spatial_imoco = lambda_spatial;
    recon_params.iterations_imoco = iterations_;
    recon_params.lambda_time = lambda_time;
    recon_params.RO = data_in.get()->get_size(0);
    recon_params.norm = 2;

    boost::shared_ptr<Gadgetron::cuNDArray<Gadgetron::float_complext>> csm;
    noncartesian_reconstruction<3> reconstruction(recon_params);

    noncartesian_reconstruction_3D reconstruction3D(recon_params);
    noncartesian_reconstruction_4D reconstruction4D(recon_params);

    cuNDArray<Gadgetron::float_complext> images;

    std::vector<cuNDArray<float>> scaled_time_vec;

    for (size_t it = 0; it < shots_per_time.get_number_of_elements(); it++)
    {
        std::vector<size_t> sct_dims = {recon_params.RO, *(shots_per_time.begin() + it)};
        size_t inter_acc = std::accumulate(shots_per_time.begin(), shots_per_time.begin() + it, 0) * sct.get_size(0);
        if (sct.get_size(0) < 1)
        {
            cuNDArray<float> temp(sct_dims);
            fill(&temp, float(0.0));
            scaled_time_vec.push_back(temp);
        }
        else
        {
            auto temp = cuNDArray<float>(sct_dims, sct.data() + inter_acc);
            scaled_time_vec.push_back(temp);
        }
    }

    if (calculateCSM)
    {
        auto [cuData, traj_csm] = reconstruction.organize_data(data_in.get(), trajectory_in.get());
        auto dcf_csm = reconstruction.estimate_dcf(&traj_csm);

        if (!pseudoReplica)
        {
            cuNDArray<float_complext> channel_images(reconstruction.get_recon_dims());
            {
                GadgetronTimer timer("Reconstruct");

                reconstruction.reconstruct(&cuData, &channel_images, &traj_csm, &dcf_csm);
            }
            nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(channel_images, (out_file + std::string("_channelImages.complex")));

            csm = reconstruction.generateRoemerCSM(&channel_images);

            channel_images *= *conj(csm.get());
            auto combined = sum(&channel_images, channel_images.get_number_of_dimensions() - 1);
            cuData.clear();
            traj_csm.clear();
            dcf_csm.clear();
            if (reconstructionType == 0)
                images = *combined;

            nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(*combined, (out_file + std::string("_combined.complex")));

            nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(*csm, out_file + std::string("_csm.complex"));
        }
    }
    else
    {
        csm = boost::make_shared<cuNDArray<float_complext>>(*csm_in);
    }

    switch (reconstructionType)
    {
    case 1:
    {
        auto [cuData, traj, dcw] = reconstruction3D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get());
        images = reconstruction3D.reconstruct_CGSense(&cuData, &traj, &dcw, csm);
    }
    break;

    case 2:
    {
        auto [cuData, traj, dcw] = reconstruction3D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get());
        images = reconstruction3D.reconstruct(&cuData, &traj, &dcw, csm);
    }
    break;
    case 3:
    {
        auto [cuData, trajVec, dcwVec] = reconstruction4D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get());
        images = reconstruction4D.reconstruct(&cuData, &trajVec, &dcwVec, csm);
    }
    break;
    case 4:
    {
        auto [cuData, traj, dcw] = reconstruction3D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get());
        images = reconstruction3D.reconstruct_CGSense_fc(&cuData, &traj, &dcw, csm, &cw, &sct, fbins);
    }
    break;
    case 5:
    {
        auto [cuData, trajVec, dcwVec] = reconstruction4D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get());
        images = reconstruction4D.reconstruct_fc(&cuData, &trajVec, &dcwVec, csm, &cw, scaled_time_vec, fbins);
    }
    break;
    case 6:
    {
        auto def = cuNDArray<float>(*def_in);
        auto invdef = cuNDArray<float>(*invdef_in);
        auto [cuData, trajVec, dcwVec] = reconstruction4D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get());

        images = reconstruction4D.reconstructiMOCO_fc(&cuData, &trajVec, &dcwVec, csm, &cw, scaled_time_vec, fbins, &def, &invdef);
    }
    break;
    case 7:
    {
        auto [cuData, trajVec, dcwVec] = reconstruction4D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get());
        images = reconstruction4D.reconstruct_nlcg(&cuData, &trajVec, &dcwVec, csm);
    }
    break;
    case 9:
    {
        auto def = cuNDArray<float>(*def_in);
        auto invdef = cuNDArray<float>(*invdef_in);
        auto [cuData, trajVec, dcwVec] = reconstruction4D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get());


        images = reconstruction4D.reconstructiMOCO(&cuData, &trajVec, &dcwVec, csm, &def, &invdef);
    }
    break;
    }
    nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(images, out_file + std::string("_images.complex"));

    std::exit(0);
    return 0;
}
