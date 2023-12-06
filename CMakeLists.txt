cmake_minimum_required(VERSION 3.16...3.21 FATAL_ERROR)

if (${CMAKE_VERSION} VERSION_LESS 3.17)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else ()
    cmake_policy(VERSION 3.17)
endif()
project(NHLBI_GT_TOOLBOX LANGUAGES CXX C)

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

if (EXISTS $ENV{CMAKE_PREFIX_PATH})
    list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
endif ()
if (EXISTS $ENV{ISMRMRD_HOME})
    list(APPEND CMAKE_PREFIX_PATH $ENV{ISMRMRD_HOME})
endif ()
if (EXISTS $ENV{TORCH_HOME})
    list(APPEND CMAKE_PREFIX_PATH $ENV{TORCH_HOME})
endif ()
if (CMAKE_PREFIX_PATH)
    list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
endif ()
message(STATUS "Looking for packages in : ${CMAKE_PREFIX_PATH}")
# ---

# check the compiler version
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (NOT CMAKE_CXX_COMPILER_VERSION) #Compiler version is not set on Ubuntu 12.02 (gcc 4.6)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERSION)
    endif ()
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
        message("CXX COMPILER VERSION: ${CMAKE_CXX_COMPILER_ID} : ${CMAKE_CXX_COMPILER_VERSION}")
        message(FATAL_ERROR "Gadgetron requires GCC version >= 6.0")
    endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)
        message(FATAL_ERROR "Gadgetron requires Clang version >= 3.5")
    endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
        message(FATAL_ERROR "Gadgetron requires MSVC 2013 or above ")
    endif ()
else ()
    message(WARNING "Unsupported compiler!")
endif ()

set(CMAKE_INSTALL_RPATH ".:${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(GADGETRON_INSTALL_INCLUDE_PATH include/gadgetron)

# build options for OpenMP support
find_package(OpenMP)
option(USE_OPENMP "Use OpenMP" On)
if (OPENMP_FOUND)
    if (USE_OPENMP)
        message("OpenMP multithreading enabled")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DUSE_OMP)
    else ()
        message("OpenMP multithreading is supported, but disabled")
    endif ()
else ()
    message("OpenMP multithreading not supported")
endif ()

# for non-container efforts..
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

find_package(Gadgetron REQUIRED)
add_definitions(-DARMA_DONT_USE_WRAPPER -DARMA_USE_CXX11 -DARMA_64BIT_WORD)
find_package(CUDA_advanced)
find_package(Boost REQUIRED)
find_package(Python3_compat REQUIRED)
find_package(Armadillo REQUIRED )

if (USE_MKL)
    if (LINUX)
        list(APPEND CMAKE_PREFIX_PATH "/opt/intel/oneapi/mkl/latest")
    endif()
    find_package(MKL REQUIRED)
    link_directories(GTBLAS BEFORE INTERFACE "${MKL_ROOT}/lib/intel64/")
    add_library(GTBLAS INTERFACE)
    set_property(TARGET GTBLAS PROPERTY INTERFACE_LINK_LIBRARIES ${MKL_LIBRARIES})
    message("MKL LIB DIR ${MKL_ROOT}/lib/intel64")

    add_definitions(-DUSE_MKL -DARMA_BLAS_LONG_LONG -DARMA_USE_BLAS -DARMA_USE_LAPACK)
else ()
    find_package(CBLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    find_package(LAPACKE REQUIRED)
    add_library(GTBLAS INTERFACE)
    set_property(TARGET GTBLAS PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CBLAS_INCLUDE_DIR} ${LAPACK_INCLUDE_DIR} ${LAPACKE_INCLUDE_DIR})
    set_property(TARGET GTBLAS PROPERTY INTERFACE_LINK_LIBRARIES ${CBLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})

    find_package(PkgConfig)
    pkg_check_modules(PC_lapacke lapacke)

    string(COMPARE EQUAL ${BLA_VENDOR} "OpenBLAS" USING_OPENBLAS)
    if (USING_OPENBLAS)
        message("OpenBLAS detected; setting max number of OpenBLAS pthreads to one.")
        add_definitions(-DFORCE_LIMIT_OPENBLAS_NUM_THREADS)
    endif()
endif ()
#find_package(CBLAS REQUIRED)
#find_package(LAPACK REQUIRED)
#find_package(cpr REQUIRED)
#find_package(ITK REQUIRED)
#include(${ITK_USE_FILE})

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
# necessary for Windows and RHEL <=6 systems

if(WIN32)
    add_definitions( -DHAVE_SNPRINTF) #Python decides to overwrite snprintf if this is not defined, because ofcourse it does.
endif()
# We actually only use system and thread explicitly, but they require linking in date_time and chrono
if (WIN32)
    find_package(Boost 1.71.0 COMPONENTS coroutine system date_time chrono program_options filesystem timer exception REQUIRED )
else()
    find_package(Boost 1.71.0 COMPONENTS coroutine system date_time program_options filesystem timer REQUIRED )
endif()
add_definitions("-DBOOST_BIND_GLOBAL_PLACEHOLDERS") # To suppress messages in later versions of boost, in version 1.76.0 there are internal conflicts in boost
add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS") # To suppress messages in later versions of boost, in version 1.76.0 there are internal conflicts in boost
if (NOT DEFINED Boost_VERSION_STRING)
    set(Boost_VERSION_STRING "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
endif()

find_package(ZFP)


if (ZFP_FOUND)
   add_definitions(-DGADGETRON_COMPRESSION_ZFP)
endif()

if (ZFP_FOUND)
   include_directories(${ZFP_INCLUDE_DIR})
endif()

option(USE_CUDA "Build with CUDA support" On)

if (USE_CUDA)
    find_package(CUDA_advanced)
endif()

if (CUDA_FOUND AND USE_CUDA)

    if (NOT CUDA_cublas_device_LIBRARY)
        list(REMOVE_ITEM CUDA_CUBLAS_LIBRARIES CUDA_cublas_device_LIBRARY-NOTFOUND)
    endif()

    add_definitions(-DUSE_CUDA)
    set( GADGETRON_CUDA_FOUND_BOOL 1 )
    include_directories( ${CUDA_INCLUDE_DIRS} )
    set(CUDA_PROPAGATE_HOST_FLAGS ON)
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++17")

    set(CMAKE_CUDA_STANDARD 17)
    #  endif ()
    #set(CUDA_VERBOSE_BUILD ON)


    # Compile kernels for compute models 1.0 and 2.0 as default for Cuda 4.1
    # Support compute model 3.0 from Cuda 4.2 and up
    # Support compute model 3.5 from Cuda 5 and up

    option(GADGETRON_CUDA_ALL_COMPUTE_MODEL "Build CUDA components for all computing models" OFF)
    if (GADGETRON_CUDA_ALL_COMPUTE_MODEL)
        message("Compiling CUDA components to support all compute model 3.5, 5.0, 5.2, 6.0, 6.1, 7.0")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_35,code=sm_35")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_50,code=sm_50")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_52,code=sm_52")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_60,code=sm_60")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_61,code=sm_61")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_70,code=sm_70")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_75,code=sm_75")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_80,code=sm_80")
    endif ()

    message("Compiling with ${CUDA_NVCC_FLAGS}")
else ()
    message("CUDA not found. CUDA components will not be compiled.")
    set( GADGETRON_CUDA_FOUND_BOOL 0 )
    set( CUDA_NVCC_FLAGS "NA")
endif ()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_subdirectory(sdc)
#set(GADGETRON_BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${GADGETRON_BUILD_RPATH}" PARENT_SCOPE)


set( gadgetron_nhlbi_gt_toolbox_header_files
	waveforms/WaveformToTrajectory.h
	waveforms/AcquisitionWaveformFanout.h
	utils/mri_core_girf_correction.h
	spiral/SpiralBuffer.h
    spiral/vds.h
	spiral/TrajectoryParameters.h
	utils/gadgets/SpiralBufferFanout.h
	utils/concomitantFieldCorrection/mri_concomitant_field_correction.h
)
   
set( gadgetron_nhlbi_gt_toolbox_src_files
	waveforms/WaveformToTrajectory.cpp
	waveforms/AcquisitionWaveformFanout.cpp
	utils/mri_core_girf_correction.cpp
	waveforms/RemoveNavsGadget.cpp
	waveforms/RemoveAcqsGadget.cpp
	spiral/TrajectoryParameters.cpp
    spiral/vds.cpp
	spiral/SpiralAcquisitionCropData.cpp
	non_cartesian/Spiral_cgsense.cpp
	non_cartesian/SpiralTVRecon.cpp
	utils/gadgets/RemoveSpiralOversampling.cpp
	utils/gadgets/SpiralBufferFanout.cpp
	utils/gadgets/removeNavAcq.cpp
	utils/gadgets/ImagetoVector.cpp
	utils/concomitantFieldCorrection/mri_concomitant_field_correction.cpp	
	non_cartesian/SpiraliCoMocoRecon.cpp
    
)
#	utils/concomitantFieldCorrection/SpiralConcomitantCorrectionGadget.cpp
#	non_cartesian/SpiralMocoRecon_new.cpp 


set( gadgetron_nhlbi_gt_toolbox_config_files
	config/imoco_recon.xml
	config/spiral_cgsense.xml
 )

add_library(nhlbi_gt_toolbox SHARED  
${gadgetron_nhlbi_gt_toolbox_header_files} 
${gadgetron_nhlbi_gt_toolbox_src_files}
${gadgetron_nhlbi_gt_toolbox_config_files}
)


target_link_libraries(nhlbi_gt_toolbox gadgetron_core 
                                       gadgetron_toolbox_cpucore_math 
                                       gadgetron_toolbox_spiral 
                                       gadgetron_toolbox_cpunfft 
                                       gadgetron_toolbox_cpusdc 
                                       gadgetron_toolbox_cpureg
                                       gadgetron_mricore
									   gadgetron_toolbox_cpufft
									   gadgetron_toolbox_mri_core
									   gadgetron_toolbox_log
									   gadgetron_toolbox_cpuoperator
									   gadgetron_toolbox_image_analyze_io
									   gadgetron_core_parallel )

# if(CUDA_FOUND)
    # target_link_libraries(nhlbi_gt_toolbox
	# gadgetron_mri_noncartesian
	# gadgetron_toolbox_demons
	# gadgetron_toolbox_gpu
	# )

	target_include_directories(nhlbi_gt_toolbox
    PUBLIC    
	$<INSTALL_INTERFACE:include>
	${CMAKE_PREFIX_PATH}/include/
	${CMAKE_PREFIX_PATH}/include/gadgetron/
	${CMAKE_PREFIX_PATH}/include/gadgetron/io/ 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/non_cartesian/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/waveforms/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/spiral/>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/spiral/reconstruction/>
	${CMAKE_SOURCE_DIR}/standalones/moco/inc/
	/opt/code/cuda-samples/Common
)
include_directories(
	#${CMAKE_SOURCE_DIR}/standalones/moco/inc/
	#Common/
	${CMAKE_SOURCE_DIR}/utils/motionCorrection/
)
# endif()
install(FILES ${gadgetron_nhlbi_gt_toolbox_header_files} DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

add_subdirectory(utils)
#add_subdirectory(gpuregistration)
add_subdirectory(python)

#target_link_libraries(nhlbi_gt_toolbox
#	nhlbi_toolbox_gpureg -lcurand)


target_link_libraries(nhlbi_gt_toolbox
	nhlbi_gt_utils)
 
set( gadgetron_nhlbi_gpu_recon_header_files
spiral/reconstruction/densityCompensation.h
spiral/reconstruction/noncartesian_reconstruction.h
spiral/reconstruction/reconParams.h
spiral/reconstruction/SpiralReconBuffer.h
spiral/reconstruction/noncartesian_reconstruction_4D.h
spiral/reconstruction/noncartesian_reconstruction_3D.h
spiral/reconstruction/noncartesian_reconstruction_fc.h
spiral/reconstruction/cuNonCartesianSenseOperator_fc.h
spiral/reconstruction/cuNonCartesianTSenseOperator_fc.h
spiral/reconstruction/cuNonCartesianTSenseOperator.h
spiral/reconstruction/cuNonCartesianMOCOOperator_fc.h
spiral/reconstruction/cuNonCartesianMOCOOperator.h
)

set( gadgetron_nhlbi_gpu_recon_src_files
spiral/reconstruction/densityCompensation.cpp 
spiral/reconstruction/noncartesian_reconstruction.cpp
spiral/reconstruction/noncartesian_reconstruction_4D.cpp
spiral/reconstruction/noncartesian_reconstruction_3D.cpp
spiral/reconstruction/noncartesian_reconstruction_fc.cpp
spiral/reconstruction/cuNonCartesianSenseOperator_fc.cpp
spiral/reconstruction/cuNonCartesianTSenseOperator_fc.cpp
spiral/reconstruction/cuNonCartesianTSenseOperator.cpp
spiral/reconstruction/cuNonCartesianMOCOOperator_fc.cpp
spiral/reconstruction/cuNonCartesianMOCOOperator.cu
)

	cuda_add_library(nhlbi_toolbox_gpu SHARED
	utils/gpu/cuda_utils.h
	utils/gpu/cuda_utils.cu
	utils/gpu/gpuregistration/gpuRegistration.cu
	utils/gpu/gpuregistration/gpuRegistration.cuh
	${gadgetron_nhlbi_gpu_recon_header_files} 
	${gadgetron_nhlbi_gpu_recon_src_files}
	#utils/gpu/gpuregistration/cubicTex3D.cu
	#utils/gpu/gpuregistration/cubicPrefilter3D.cu
)
set_target_properties(nhlbi_toolbox_gpu PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(nhlbi_toolbox_gpu gadgetron_core gadgetron_toolbox_cpucore_math gadgetron_toolbox_spiral gadgetron_toolbox_cpunfft)

target_link_libraries(nhlbi_toolbox_gpu
gadgetron_toolbox_hostutils 
gadgetron_toolbox_mri_core
gadgetron_toolbox_cpusdc
gadgetron_toolbox_gpu
gadgetron_mri_noncartesian
gadgetron_toolbox_spiral 
gadgetron_toolbox_cpunfft
gadgetron_toolbox_log
gadgetron_mricore
gadgetron_toolbox_cpufft
gadgetron_toolbox_mri_core
gadgetron_toolbox_log
gadgetron_toolbox_cpuoperator
gadgetron_toolbox_image_analyze_io
gadgetron_core_parallel
gadgetron_toolbox_cpusdc
${CUDA_LIBRARIES}
${CUDA_CUFFT_LIBRARIES} 
${CUDA_CUBLAS_LIBRARIES} 
${CUDA_CUSPARSE_LIBRARIES}
${MKL_LIBRARIES}
Boost::program_options
)

target_include_directories(nhlbi_toolbox_gpu
	PUBLIC    
	$<INSTALL_INTERFACE:include>
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/gpu/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/non_cartesian/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/waveforms/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/spiral/>
${CMAKE_PREFIX_PATH}/lib/
${CMAKE_PREFIX_PATH}/include/
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/gpu/gpuregistration/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/gpu/gpuregistration/internal/>
${CMAKE_PREFIX_PATH}/include/
${CMAKE_PREFIX_PATH}/include/gadgetron/
${CMAKE_PREFIX_PATH}/include/gadgetron/io/ 
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/non_cartesian/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/waveforms/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/spiral/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/spiral/reconstruction/>
${CMAKE_SOURCE_DIR}/standalones/moco/inc/
/opt/code/cuda-samples/Common
${CUDA_INCLUDE_DIRS}
)

install(TARGETS nhlbi_toolbox_gpu
LIBRARY DESTINATION lib
ARCHIVE DESTINATION lib
RUNTIME DESTINATION bin
COMPONENT main
)

install(FILES
utils/gpu/cuda_utils.h
utils/gpu/gpuregistration/cubicTex3D.cu
utils/gpu/gpuregistration/cubicPrefilter3D.cu
utils/gpu/gpuregistration/gpuRegistration.cu
utils/gpu/gpuregistration/gpuRegistration.cuh
${gadgetron_nhlbi_gpu_recon_header_files}
${gadgetron_nhlbi_gpu_recon_src_files}
DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH}
COMPONENT main)


target_link_libraries(nhlbi_gt_toolbox
nhlbi_toolbox_gpu -lcurand)

#target_link_libraries(nhlbi_gt_toolbox
#nhlbi_gt_recon -lcurand)



install(FILES ${gadgetron_nhlbi_gt_toolbox_config_files} DESTINATION ${GADGETRON_INSTALL_CONFIG_PATH} COMPONENT main)
install(TARGETS nhlbi_gt_toolbox
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	COMPONENT main
)
add_subdirectory(standalones)
