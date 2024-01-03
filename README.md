# iCoMoCo Resontruction in Gadgetron

checkout and build gadgetron from https://github.com/gadgetron/gadgetron 

update the environment using the environment.yml file in this repository 

`mamba env update -f environment.yml`

build the code using 

`mkdir build &&
cd build &&
cmake ../ -GNinja -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DC_PREFIX_PATH=${CONDA_PREFIX} -DUSE_CUDA=ON -DUSE_MKL=ON` 

once built, the package can be used with gadgetron using the config xml files provided with this repository.


## Alternatively the provided docker image can be used to test the code

the docker image can be pulled using the following command:

`docker pull gadgetronnhlbi/ubuntu_2004_cuda117_public_icomoco:built_rt`

This image can be deployed with: 

`docker run --gpus all  --name=deploy_rt -ti -p 9063:9002 --volume=[LOCAL_DATA_FOLDER]:/opt/data --restart unless-stopped --detach gadgetronnhlbi/ubuntu_2004_cuda117_public_icomoco:built_rt`

where the local data folder is the path to a folder containing raw data that can be used for testing the reconstruction. 

once the docker container is running start a bash terminal inside the container using: 

`docker exec -ti deplot_rt bash` 

then you can simply navigate to `/opt/data/` and test the code using: 

`gadgetron_ismrmrd_client -p 9002 -f DATA_FILE -c imoco_recon.xml -o OUTPUT_FILENAME.h5` 

in another terminal session you can monitor the logs from the container using `docker logs -f deploy_rt`

Please note that if you are using the gadgetron_ismrmrd_client from outside the container then you may need to specify the server address with `-a SERVER_ADDRESS` and the port `-p 9063`

## Dataset

The test data can be downloaded from zenodo: `https://doi.org/10.5281/zenodo.10456573`



