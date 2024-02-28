FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel
LABEL authors="Yunshuang Yuan"
LABEL hostname="cosense-docker"

ENTRYPOINT ["top", "-b"]
WORKDIR /workspace

RUN apt-get update && apt-get install -y git
RUN apt-get install build-essential python3-dev libopenblas-dev -y

RUN conda update conda
RUN conda install openblas-devel -y
RUN conda install pybind11 -y
RUN conda install -c conda-forge libstdcxx-ng libffi -y

RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine \
    -v --no-deps     \
    --global-option="--blas_include_dirs=${CONDA_PREFIX}/include"     \
    --global-option="--blas=openblas"


RUN pip install opencv-python==4.2.0.34
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp38-cp38-linux_x86_64.whl
RUN pip install numpy PyYAML Shapely scipy rich tqdm open3d plyfile einops timm flash-attn spconv-cu118 pypcd

COPY ./cosense3d/ops /workspace
RUN cd /workspace/ops && pip install . && cd ..

COPY requirements_cosense_4090.txt /workspace
RUN pip install -r requirements_cosense_4090.txt
