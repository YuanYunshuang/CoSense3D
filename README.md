
<p align="center">
  <img src="docs/_static/imgs/cosense_logo.png" />
</p>


Welcome to [CoSense3D](yuanyunshuang.github.io/CoSense3D/)! This is an agent-oriented framework specially designed for cooperative perception for autonomous driving.
The agent-based structure of this framework aims to accelerate the development process of cooperative perception models by 
more efficient and flexible data loading and distributing process, as well as the forward and gradient calculation scheduling.

## Update
[2024/08/23] Add official implementation of [StreamLTS](https://arxiv.org/abs/2407.03825).

## Installation
Quick installation scripts are provided to install the environment with the following commands. 
For more detailed information about the installation, please refer to [Installation](docs/md/installation.md) page.
```bash
conda create -n consense3d python=3.8
conda activate cosense3d
cd Cosense3D 
# for Nvidia RTX 3090
bash setup_env_3090.sh
# for Nvidia RTX 4090
bash setup_env_4090.sh
```

## Datasets
CoSense3D formats meta information of a dataset in a standardized json file, 
including the annotations and the relative path to the image and point cloud data.
For a given new opensource dataset for collective perception, the meta info are parsed to json files, 
the media files are kept with its original folder structure. Currently, the supported datasets are:
- OPV2V
- OPV2Vt
- DairV2Xt

For more details about downloading the datasets, please refer to [Datasets](docs/md/prepare_data.md) page.

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
cd CoSense3D 
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config [CONFIG FILE] --mode [vis_train | vis_test]
# check if the OPV2Vt data is correctly loaded during training
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/cood/sp3d.yaml --mode train --visualize

```
Demos:
```bash
# visualize OPV2Vt dataset test set
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./config/opv2v.yaml --mode vis_test
```
![DEMO OPV2Vt](docs/_static/imgs/opv2vt.gif)

```bash
# visualize DairV2Xt dataset test set
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./config/dairv2x.yaml --mode vis_test
```
![DEMO DairV2Xt](docs/_static/imgs/dairv2xt.gif)

### Train
```bash
# Train on a single GPU
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./cosense3d/config/streamLTS/streamLTS_opv2vt.yaml --mode train --run-name sp3d-opv2vt
# Parallel training on multiple GPUs
PYTHONPATH=. OMP_NUM_THREADS=16 torchrun \
--nproc_per_node=2 \
cosense3d/tools/agent_runner.py \
--config ./cosense3d/config/streamLTS/streamLTS_opv2vt.yaml \
--mode train \
--gpus 2 \
--batch-size 2
```
### Test
```bash
# Test on a single GPU
PYTHONPATH=. python cosense3d/tools/agent_runner.py --config ./path/to/config/file.yaml --mode test --load-from path/to/ckpt.pth
```
## Benchmark and Model Zoo

### Time-Aligned Cooperative Object Detection (TA-COOD)
| Model      | Backbone    | Fusion | OPV2Vt AP@0.5 | OPV2Vt AP@0.7 | OPV2Vt AP@0.5 | OPV2Vt AP@0.7 | OPV2Vt ckpt                                                                                                                                                                                          | DairV2Xt ckpt|
|------------|-------------|--------|---------------|---------------|---------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| Fcooper    | PointPillar | Maxout | 54.4          | 17.0          | 41.8          | 17.7          | [<img src="./docs/_static/imgs/download.png" alt="drawing" width="20" />](https://data.uni-hannover.de:8080/dataset/upload/users/ikg/yuan/cosense3d/model_zoo/StreamLTS/streamLTS_fcooper_opv2vt.pth) |[<img src="./docs/_static/imgs/download.png" alt="drawing" width="20" />](https://data.uni-hannover.de:8080/dataset/upload/users/ikg/yuan/cosense3d/model_zoo/StreamLTS/streamLTS_fcooper_dairv2xt.pth)|
| FPVRCNN    | Spconv      | Attn | 70.8          | 41.2          | 51.8          | 23.9          | [<img src="./docs/_static/imgs/download.png" alt="drawing" width="20" />](https://data.uni-hannover.de:8080/dataset/upload/users/ikg/yuan/cosense3d/model_zoo/StreamLTS/streamLTS_fpvrcnn_opv2vt.pth) |[<img src="./docs/_static/imgs/download.png" alt="drawing" width="20" />](https://data.uni-hannover.de:8080/dataset/upload/users/ikg/yuan/cosense3d/model_zoo/StreamLTS/streamLTS_fpvrcnn_dairv2xt.pth)|
| AttnFusion | VoxelNet    | Attn | 78.7          | 41.4          | 62.1          | 34.0          | [<img src="./docs/_static/imgs/download.png" alt="drawing" width="20" />](https://data.uni-hannover.de:8080/dataset/upload/users/ikg/yuan/cosense3d/model_zoo/StreamLTS/streamLTS_attnfusion_opv2vt.pth) |[<img src="./docs/_static/imgs/download.png" alt="drawing" width="20" />](https://data.uni-hannover.de:8080/dataset/upload/users/ikg/yuan/cosense3d/model_zoo/StreamLTS/streamLTS_attnfusion_dairv2xt.pth)|
| StreamLTS  | MinkUnet    | Attn   | 85.3          | 72.1          | 64.0          | 40.4          | [<img src="./docs/_static/imgs/download.png" alt="drawing" width="20" />](https://data.uni-hannover.de:8080/dataset/upload/users/ikg/yuan/cosense3d/model_zoo/StreamLTS/streamLTS_opv2vt.pth) |[<img src="./docs/_static/imgs/download.png" alt="drawing" width="20" />](https://data.uni-hannover.de:8080/dataset/upload/users/ikg/yuan/cosense3d/model_zoo/StreamLTS/streamLTS_dairv2xt.pth)|

### Object Detection

| Model      | Ngrad | OPV2V AP@0.7 | OPV2V AP@0.5 | OPV2Vt ckpt                                                                                          |
|------------|-------|--------------|--------------|------------------------------------------------------------------------------------------------------|
| Fcooper    | all   | 82.2         | 89.9         | [<img src="./docs/_static/imgs/download.png" alt="drawing" width="20" />](https://seafile.cloud.uni-hannover.de/d/7b1bb0dad44040d68ffc/) |
| Fcooper    | 2     | 68.7         | 83.5         | [<img src="./docs/_static/imgs/download.png" alt="drawing" width="20" />](https://seafile.cloud.uni-hannover.de/d/b7783a8073704b67846e/) |
| FPVRCNN    | all   | 84.0         | 87.3         | [<img src="./docs/_static/imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/d0179efa3b854018947e/)  |
| FPVRCNN    | 1     | 84.9         | 87.8         | [<img src="./docs/_static/imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/29478ff7ea20456a8f36/)  |
| EviBEV     | all   | 84.1         | 91.1         | [<img src="./docs/_static/imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/05c3154d643a49589472/)  |
| EviBEV     | 1     | 79.5         | 89.1         | [<img src="./docs/_static/imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/2d0f6488241244de9d9f/)  |
| AttnFusion | all   | 87.6         | 92.3         | [<img src="./docs/_static/imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/a80c24c303854fa9b382/)  |
| AttnFusion | 1     | 87.1         | 92.6         | [<img src="./docs/_static/imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/3ae798d04a8742ac8661/)  |



## Citations

```
@INPROCEEDINGS{cosense3d,
  author={Yunshuang Yuan and Monika Sester},
  booktitle={2024 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={CoSense3D: an Agent-based Efficient Learning Framework for Collective Perception}, 
  year={2024},
  pages={1-6},
  keywords={Collective Perception, Object Detection, Efficient Training},
  doi={10.1109/IV55152.2023.10186693}}
  
@INPROCEEDINGS{cosense3d,
  author={Yunshuang Yuan and Monika Sester},
  booktitle={The European Conference on Computer Vision Workshop (ECCVW)}, 
  title={StreamLTS: Query-based Temporal-Spatial LiDAR Fusion for Cooperative Object Detection}, 
  year={2024},
  keywords={Collective Perception, Point Cloud, Data Fusion}}

