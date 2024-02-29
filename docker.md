```shell
apt-get update && apt-get install libgl1-mesa-glx libglib2.0-0 -y
apt-get install build-essential python3-dev libopenblas-dev -y
conda update conda
conda install -c conda-forge libstdcxx-ng  -y
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine   -v --no-deps  \       
--global-option="--blas_include_dirs=${CONDA_PREFIX}/include"  \
--global-option="--blas=openblas"

cd CoSense3D/cosense3d/ops/
pip install .
cd ..

cd CoSense3D/
pip install -r requirements_cosense_4090.txt

PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS_fpvrcnn_opv2vt.yaml --mode train 

```



