#!/bin/bash

if [ $# -lt 2 ]
then
    base_name=" gadgetronnhlbi/ubuntu_2004_cuda117_lit"
else
    if [ $# -eq 2 ]
    then
       base_name=" $2"
       echo ${base_name}
    fi
fi

#base_name=" gadgetronnhlbi/ubuntu_2004_cuda117_lit"
dev_name="$base_name:$1_dev"
rt_name="$base_name:$1_rt"

echo "image prefix: $rt_name"
echo "image prefix: $dev_name"


docker build --build-arg BUILDKIT_INLINE_CACHE=0 --target gadgetron_cudabuild -t gadgetron_cudabuild -f ../gadgetron/Dockerfile ../gadgetron

docker build --build-arg BUILDKIT_INLINE_CACHE=0 --target gadgetron_nhlbicudabuild -t gadgetron_dev_nhlbi -f Dockerfile ../

docker tag gadgetron_dev_nhlbi ${dev_name}

docker push ${dev_name}

docker build --build-arg BUILDKIT_INLINE_CACHE=0 --target gadgetron_nhlbi_rt_cuda -t gadgetron_rt_nhlbi -f Dockerfile ../

docker tag gadgetron_rt_nhlbi ${rt_name}

docker push ${rt_name}
