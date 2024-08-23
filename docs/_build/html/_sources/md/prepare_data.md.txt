# Prepare Datasets
> Check the dataset [page](https://data.uni-hannover.de/dataset/cosense3d) for download links or use the downloading script as following commands.
## OPV2Vt
```shell
cd CoSense3D
bash cosense3d/tools/download.sh OPV2Vt path/to/output_dir
```

## DairV2Xt

Download [DAIR-V2X-C](https://thudair.baai.ac.cn/coop-dtest) dataset and extract it to the following structure.

```shell
├── dair-v2x
│   ├── cooperative-vehicle-infrastructure
|      |── 2021_08_16_22_26_54
|      |── ...
│   ├── cooperative-vehicle-infrastructure-infrastructure-side-image
│   ├── cooperative-vehicle-infrastructure-infrastructure-side-velodyne
│   ├── cooperative-vehicle-infrastructure-vehicle-side-image
│   ├── cooperative-vehicle-infrastructure-vehicle-side-velodyne
```
Then download the meta files with
```shell
bash cosense3d/tools/download.sh DairV2xt /path/to/dair-v2x
```

## OPV2V

```shell
bash cosense3d/tools/download.sh OPV2V path/to/output_dir
```
