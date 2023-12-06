

include_directories(
	${CMAKE_SOURCE_DIR}/standalones/moco/inc/
	/opt/code/cuda-samples/Common
)
set( gadgetron_nhlbi_gt_utils_header_files
util_functions.h
)

set( gadgetron_nhlbi_gt_utils_src_files
util_functions.cpp 
)

add_library(nhlbi_gt_utils SHARED  
${gadgetron_nhlbi_gt_utils_header_files} 
${gadgetron_nhlbi_gt_utils_src_files}
)

target_link_libraries(nhlbi_gt_utils gadgetron_core gadgetron_toolbox_cpucore_math gadgetron_toolbox_spiral gadgetron_toolbox_cpunfft gadgetron_toolbox_cpusdc gadgetron_toolbox_cpureg )

target_link_libraries(nhlbi_gt_utils gadgetron_mricore
									   gadgetron_toolbox_cpufft
									   gadgetron_toolbox_mri_core
									   gadgetron_toolbox_log
									   gadgetron_toolbox_cpuoperator
									   gadgetron_toolbox_image_analyze_io
									   gadgetron_core_parallel
									   ${CUDA_LIBRARIES}
										)
#${ITK_LIBRARIES}
# GPU libraries - if gpus dont exist look else where. this recons not for you ;)
target_link_libraries(nhlbi_gt_utils
gadgetron_toolbox_gpu
gadgetron_mri_noncartesian
gadgetron_toolbox_demons
)
#nhlbi_toolbox_gpureg
target_include_directories(nhlbi_gt_utils
    PUBLIC
	${CMAKE_PREFIX_PATH}/include/
	${CMAKE_PREFIX_PATH}/include/gadgetron/
	${CMAKE_PREFIX_PATH}/include/gadgetron/io/
	$<INSTALL_INTERFACE:include/>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/gpu/gpuregistration/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/gpu/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/non_cartesian/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/waveforms/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/spiral/>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/spiral/reconstruction>
)

set(gadgetron_python_gadgets_files python/BinningGadget.py
python/kaiser_window.py
python/registration/registration_gadget.py
python/registration/registration_oflow3D.py
) 

install(FILES ${gadgetron_python_gadgets_files}
        DESTINATION ${GADGETRON_INSTALL_PYTHON_MODULE_PATH} COMPONENT main)


install(FILES ${gadgetron_nhlbi_gt_utils_header_files} DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)


install(TARGETS nhlbi_gt_utils
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin  
	COMPONENT main
)
#add_subdirectory(gpu)
