FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel
LABEL hostname="cosense3d-docker"
ENV OMP_NUM_THREADS=16

#ENTRYPOINT ["top", "-b"]
WORKDIR /project
COPY requirements.txt /project/requirements.txt
COPY ./cosense3d/ops/ /project/ops/

RUN apt-get update -y && apt-get install git -y && conda update conda -y
RUN apt-get install build-essential python3-dev libopenblas-dev -y
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y

RUN cd ops && pip install .

RUN conda install openblas-devel -y
RUN conda install pybind11 -y
RUN conda install -c conda-forge libstdcxx-ng -y

RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine \
    -v --no-deps     \
    --global-option="--blas_include_dirs=${CONDA_PREFIX}/include"     \
    --global-option="--blas=openblas" \
    --install-option="--gpu"


RUN pip install -r requirements.txt

WORKDIR /workspace
