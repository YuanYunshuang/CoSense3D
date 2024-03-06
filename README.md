# CoSense3D
Welcome to the CoSense3D! This is an agent-oriented framework specially designed for cooperative perception for autonomous driving.
The agent-based structure of this framework aims to accelerate the development process of cooperative perception models by 
more efficient and flexible data loading and distributing process, as well as the forward and gradient calculation scheduling.
More details of the structure of this framework can be found in the [documentations](docs%2F_build%2Fhtml%2Findex.html).


This repo contains the official implementation of the paper

__StreamLTS: Query-based Temporal-Spatial LiDAR Fusion for Cooperative Object Detection__

## Installation
You can install the environment with our provided batch script with the following commands. 
For more detailed information about the installation, please refer to [Installation](docs/_build/html/md/installation.html) page.
```bash
conda create -n consense3d python=3.8
conda activate cosense3d
cd OpenCosense3D 
# for Nvidia RTX 3090
bash setup_env_3090.sh
# for Nvidia RTX 4090
bash setup_env_4090.sh
```

## Datasets
> **NOTE:** 
> Since the dataset link is related to the authors' affiliation and personal accounts,  violating the anonymity of the reviewing process.
Therefore, they will be accessible after the publication of this paper.

## Quick start
The main entry of this project is at ```cosense3d/tools/agent_runner.py```. 

Required arguments: 
- ```config```: the yaml configuration file path.
- ```mode```: runner mode. ```vis_train``` and ```vis_test``` for visualizing the training and the testing data, respectively. 
```train``` and ```test``` for training and testing the model.

Optional arguments:
- ```visualize```: visualize the data during the training and the testing process.
- ```resume-from```: resume training from the give checkpoint path.
- ```load-from```: load checkpoint to initialize model for training or testing.
- ```log-dir```: logging path for training output. If not provided, it will log to the 
```cosense3d/logs/default_[Month-Day-Hour-Minute-Second]``` path.
- ```run-name```: if given, the logging path will be formatted as ```cosense3d/logs/[run-name]_[Month-Day-Hour-Minute-Second]```.
- ```seed```: global random seed.
- ```gpus```: number of gpus for training. The default is 0 means no parallel training. This number can only to set to >= 1 
when using ```tochrun``` for parallel training on multiple GPUs.
- ```data-path```: overwrite the data path in the yaml config file.
- ```meta-path```: overwrite the meta path in the yaml config file.
- ```batch-size```: overwrite the training batch size in the yaml config file.
- ```n-workers```: overwrite the number of the workers in the yaml config file.

### GUI
Our framework provides a graphical user interface for interactive visualization of the data and the training and testing process.
To have a quick look into your dataset, run 
```bash
cd OpenCoSense3D 
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config [CONFIG FILE] --mode [vis_train | vis_test]
# check if the OPV2Vt data is correctly loaded during training
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./config/StreamLTS_opv2vt.yaml --mode train --visualize

```
Demos:
```bash
# visualize OPV2Vt dataset test set
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./config/opv2vt.yaml --mode vis_test
```
![DEMO OPV2Vt](docs/_static/opv2vt.gif)

```bash
# visualize DairV2Xt dataset test set
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./config/dairv2xt.yaml --mode vis_test
```
![DEMO DairV2Xt](docs/_static/dairv2xt.gif)

### Train
```bash
# Train on a single GPU
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./config/StreamLTS_opv2vt.yaml --mode train --run-name sLTS-opv2vt
# Parallel training on multiple GPUs
PYTHONPATH=. OMP_NUM_THREADS=16 torchrun \
--nproc_per_node=2 \
cosense3d/tools/agent_runner.py \
--config ./cosense3d/config/stream_lidar_st_v3.yaml \
--mode train \
--gpus 2 \
--batch-size 2
```
### Test
```bash
# Train on a single GPU
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./config/StreamLTS_opv2vt.yaml --mode test --load-from path/to/ckpt.pth
```
## Benchmark and model zoo

| Model      | OPV2Vt AP@0.5 | OPV2Vt AP@0.7 | DairV2Xt AP@0.5 | DairV2Xt AP@0.7 | OPV2Vt ckpt                                                                                      | DairV2Xt ckpt                                                                                     |
|------------|---------------|---------------|-----------------|-----------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Fcooper    | 54.4          | 17.0          | 41.8            | 17.7            | [<img src="./docs/_static/download.png" alt="drawing" width="20"/>](https://will_be_available)   | [<img src="./docs/_static/download.png" alt="drawing" width="20"/>](https://will_be_available)    |
| FPVRCNN    | 70.8          | 41.2          | 51.8            | 23.9            | [<img src="./docs/_static/download.png" alt="drawing" width="20"/>](https://will_be_available)   | [<img src="./docs/_static/download.png" alt="drawing" width="20"/>](https://will_be_available)    |
| AttnFusion | 78.7          | 41.4          | 62.1            | 34.0            | [<img src="./docs/_static/download.png" alt="drawing" width="20"/>](https://will_be_available)   | [<img src="./docs/_static/download.png" alt="drawing" width="20"/>](https://will_be_available)    |
| StreamLTS  | 81.2          | 59.5          | 61.2            | 33.4            | [<img src="./docs/_static/download.png" alt="drawing" width="20"/>](https://will_be_available)   | [<img src="./docs/_static/download.png" alt="drawing" width="20"/>](https://will_be_available)    | 
