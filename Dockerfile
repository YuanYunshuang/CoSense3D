FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel
LABEL hostname="cosense-docker"
ENV OMP_NUM_THREADS=16

#ENTRYPOINT ["top", "-b"]
WORKDIR /project
COPY requirements.txt /project/requirements.txt
COPY ./cosense3d/ops/ /project/ops/

#RUN apt-get update -y && apt-get install git -y
#RUN apt-get install build-essential python3-dev libopenblas-dev -y
#
#RUN conda update conda -y
#RUN conda install openblas-devel -y
#RUN conda install pybind11 -y
#RUN conda install -c conda-forge libstdcxx-ng -y
#
#RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine \
#    -v --no-deps     \
#    --global-option="--blas_include_dirs=${CONDA_PREFIX}/include"     \
#    --global-option="--blas=openblas"
#
#RUN pip install torch-scatter
#RUN pip install -r requirements.txt
#RUN cd project/ops && pip install .
#WORKDIR /workspace
