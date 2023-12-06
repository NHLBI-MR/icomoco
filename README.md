# iCoMoCo Resontruction in Gadgetron

checkout and build gadgetron from https://github.com/gadgetron/gadgetron 

update the environment using the environment.yml file in this repository 

`mamba env update -f environment.yml`

build the code using 

`mkdir build &&
cd build &&
cmake ../ -GNinja -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DC_PREFIX_PATH=${CONDA_PREFIX} -DUSE_CUDA=ON -DUSE_MKL=ON` 

once built, the package can be used with gadgetron using the config xml files provided with this repository.
