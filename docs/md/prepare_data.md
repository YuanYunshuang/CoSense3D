# Prepare Datasets
## OPV2V
Please download the official [OPV2V dataset](https://mobility-lab.seas.ucla.edu/opv2v/) and 
the reformatted [meta files](https://data.uni-hannover.de/dataset/678827e9-bb64-44b8-b8fd-e583c740b5f5/resource/eade1879-e67b-4112-a088-2a92ca76e004/download/opv2v_meta.zip). 
Before training or testing your models, you should either set the corresponding `data_path` and `meta_path` 
in `cosense3d/pycfg/base/opv2v.py` or via the arguments `--data-path` and `--meta-path` while
running `cosense3d/tools/agent_runner.py`.

