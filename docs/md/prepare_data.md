# Prepare Datasets
## OPV2V
Please download the official OPV2V dataset and the reformatted meta files. 
To train or test your models, you can either set the corresponding `data_path` and `meta_path` in 

## OPV2Vt
>  Coming soon!

## DairV2Xt

Download [DAIR-V2X-C](https://thudair.baai.ac.cn/coop-dtest) dataset and the new generated meta data (will be available at the publication) and extract and structure them as following.

```shell
├── dair-v2x
│   ├── cooperative-vehicle-infrastructure
|      |── 2021_08_16_22_26_54
|      |── ...
│   ├── cooperative-vehicle-infrastructure-infrastructure-side-image
│   ├── cooperative-vehicle-infrastructure-infrastructure-side-velodyne
│   ├── cooperative-vehicle-infrastructure-vehicle-side-image
│   ├── cooperative-vehicle-infrastructure-vehicle-side-velodyne
│   ├── meta
```