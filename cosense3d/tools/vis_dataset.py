import torch
from pathlib import Path
from cosense3d.dataset.early_fusion_dataset import EarlyFusionDataset
from cosense3d.dataset.intermediate_fusion_dataset import IntermediateFusionDataset
from cosense3d.dataset.cosense_dataset import CosenseDataset
from cosense3d.dataset import get_dataloader
from cosense3d.config import load_config
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--vis_option", type=str, default="frame")  # frame or seq
    parser.add_argument("--log_dir", type=str, default="../logs")
    args = parser.parse_args()
    args.config = str(Path(__file__).parents[1] / 'config' / 'petr_opv2v_cam.yaml')
    cfgs = load_config(args)

    # cfgs['DATASET']['visualize'] = True
    # cfgs['DATASET']['preprocessors']['train'] = ['GeoAugmentation', 'CropLidarRange']
    # cfgs['DATASET']['preprocessors']['test'] = []
    # cfgs['DATASET']['preprocessors']['train'] = ['ProjectPointsToEgo']
    cfgs['DATASET']['shuffle'] = False
    dataloader = get_dataloader(cfgs['DATASET'], mode='train')
    for data in dataloader.dataset:
        print(data['scenario'], data['frame'])
        getattr(dataloader.dataset, f"visualize_{args.vis_option}")(data)
    dataloader.dataset.visualizer.close()

    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=2,
    #                                          sampler=None, num_workers=4,
    #                                          shuffle=False,
    #                                          collate_fn=dataset.collate_batch)
    # for batch in dataloader:
    #     print(batch.keys())


if __name__ == '__main__':
    main()