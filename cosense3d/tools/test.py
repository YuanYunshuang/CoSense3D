import argparse
import os, tqdm, time, glob

import torch

from cosense3d.model import get_model
from cosense3d.dataset import get_dataloader
from cosense3d.utils import misc, metrics
from cosense3d.utils.train_utils import *
from cosense3d.config import load_config
import cosense3d.dataset.post_processors as PostP


def test(cfgs, args):
    seed_everything(1234)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load checkpoint
    log_dir = cfgs['TRAIN']['log_dir']

    # set paths
    out_img_dir = os.path.join(log_dir, 'test', 'img')
    out_inf_dir = os.path.join(log_dir, 'test', 'inf')
    misc.ensure_dir(out_img_dir)
    misc.ensure_dir(out_inf_dir)
    logger = open(os.path.join(log_dir, 'test', 'result.txt'), mode='a')
    logger.write(f'Ckpt :{args.ckpt}: \n')

    metrics_instances = []
    for metric_name in cfgs['TEST']['metrics']:
        metric_cls = getattr(metrics, metric_name, None)
        if metric_cls is not None:
            metrics_instances.append(
                metric_cls(cfgs['TEST']['metrics'][metric_name],
                os.path.join(log_dir, 'test'), logger)
            )

    # load models
    test_dataloader = get_dataloader(cfgs['DATASET'], mode='test',)
    model = get_model(cfgs['MODEL'], 'test').to(device)
    model.post_process.set_log_dir(log_dir)

    inf_files = glob.glob(os.path.join(out_inf_dir, '*.pth'))
    if len(test_dataloader) == len(inf_files):
        for inf_file in tqdm.tqdm(inf_files):
            out_dict = torch.load(inf_file)
            for metric in metrics_instances:
                metric.add_samples(out_dict)
        for metric in metrics_instances:
            metric.summary()
        return

    # load checkpoint
    ckpt = torch.load(os.path.join(log_dir, f'{args.ckpt}.pth'))
    load_model_dict(model, ckpt['model_state_dict'])

    result = []
    batch_idx = 0
    with torch.no_grad():
        model.eval()
        for batch_data in tqdm.tqdm(test_dataloader):
            batch_idx += 1
            # if batch_idx > 3:
            #     break
            load_tensors_to_gpu(batch_data)
            # Forward pass
            out_dict = model(batch_data)
            # loss_dict = model.loss(batch_data)
            # result.append(loss_dict)
            # if post_processor is not None:
            #     out_dict = post_processor(batch_data)
            torch.save(out_dict, os.path.join(log_dir, 'test', 'inf', f"{batch_idx}.pth"))
            for metric in metrics_instances:
                metric.add_samples(out_dict)

    for metric in metrics_instances:
        metric.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--ckpt", type=str, default="last")
    parser.add_argument("--save-img", action="store_true")
    parser.add_argument("--vis-func", type=str) # , default="vis_semantic_unc"
    args = parser.parse_args()

    setattr(args, 'config', os.path.join(args.log_dir, 'config.yaml'))
    cfgs = load_config(args)

    test(cfgs, args)