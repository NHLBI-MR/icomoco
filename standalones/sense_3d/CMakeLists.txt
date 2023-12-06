###############################################################################################################################################################################

add_executable(reconstruction reconstruction.cpp)

target_link_libraries(reconstruction gadgetron_core gadgetron_toolbox_cpucore_math gadgetron_toolbox_spiral gadgetron_toolbox_cpunfft)

target_link_libraries(reconstruction
  gadgetron_toolbox_hostutils 
  gadgetron_toolbox_mri_core
  gadgetron_toolbox_cpusdc
	gadgetron_mri_noncartesian
  gadgetron_toolbox_spiral 
  gadgetron_toolbox_cpunfft
  gadgetron_toolbox_log
  gadgetron_toolbox_gpu
  ${CUDA_LIBRARIES}
  Boost::program_options
  nhlbi_gt_toolbox
  nhlbi_toolbox_gpu
  nhlbi_gt_utils
  )
  
target_include_directories(reconstruction
        PUBLIC    
		$<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/non_cartesian/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/waveforms/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/spiral/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/spiral/reconstruction/>
    ${CONDA_PREFIX}/include
)

install(TARGETS reconstruction DESTINATION bin COMPONENT main)


###############################################################################################################################################################################


