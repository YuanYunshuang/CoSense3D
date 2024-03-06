FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel
LABEL hostname="cosense3d-docker"
ENV OMP_NUM_THREADS=16
ENV CUDA_HOME='/usr/local/cuda-11.8'
ENV PATH /opt/conda/bin:/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs/:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/compat/:/usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility

#ENTRYPOINT ["top", "-b"]
WORKDIR /project
COPY requirements.txt /project/requirements.txt
COPY ./cosense3d/ops/ /project/ops/

RUN apt-get update -y && apt-get install git -y && conda update conda -y
RUN apt-get install build-essential python3-dev libopenblas-dev -y
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y

RUN cd ops && pip install . && cd ..

#RUN conda install openblas-devel -y
RUN apt install ninja-build
RUN pip install -U setuptools

RUN pip install -r requirements.txt

RUN conda install pybind11 -y
RUN conda install -c conda-forge libstdcxx-ng -y

RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine \
    -v --no-deps     \
    --global-option="--blas_include_dirs=${CONDA_PREFIX}/include"     \
    --global-option="--blas=openblas" \
    --global-option="--force_cuda"

WORKDIR /workspace
