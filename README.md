
## Installation
1. create conda environment
```shell
conda create -n consense3d python=3.8
conda activate cosense3d
```

2. Install pytorch and compile local Pytorch Extensions(CUDA nvcc compiler needed)
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
--extra-index-url https://download.pytorch.org/whl/cu113
cd cosense3d/ops
pip install . && cd ..
```

3. install python packages
```shell
pip install -r reququirements_cosense.txt
```

3. install MinkovskiEngine
```shell
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
    --global-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
    --global-option="--blas=openblas"
export OMP_NUM_THREADS=16
```

## Training
```shell
cd cosense3d
python tools/agent_runner.py -config config/sp3d_cav.yaml --mode train
```