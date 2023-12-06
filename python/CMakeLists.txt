set(gadgetron_python_gadgets_files 
lwd/best_lung_seg_model_2022.pkl
lwd/compute_lung_seg_sandbox.py
) 

install(FILES ${gadgetron_python_gadgets_files}
        DESTINATION ${GADGETRON_INSTALL_PYTHON_MODULE_PATH} COMPONENT main)

