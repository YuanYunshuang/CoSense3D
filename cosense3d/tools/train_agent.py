import glob
import os
import torch
import argparse
from datetime import datetime

from cosense3d.model import get_model
from cosense3d.dataset import get_dataloader
from cosense3d.utils.misc import ensure_dir, setup_logger
from cosense3d.config import load_config, save_config
from cosense3d.utils.train_utils import seed_everything
from cosense3d.agents.center_controller import get_controller
from cosense3d.agents.core.train_runner import TrainRunner


def train(cfgs):
    seed_everything(2023)

    train_dataloader = get_dataloader(cfgs['DATASET'])
    center_controller = get_controller(cfgs['CONTROLLER'], 'train')
    train_runner = TrainRunner(train_dataloader, center_controller, **cfgs['TRAIN'])
    train_runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.resume:
        assert os.path.exists(os.path.join(args.log_dir, 'config.yaml')), \
            "config.yaml file not found in the given log_dir."
        setattr(args, 'config', os.path.join(args.log_dir, 'config.yaml'))
    cfgs = load_config(args)
    setup_logger(args.run_name, args.debug)
    # for ME
    os.environ['OMP_NUM_THREADS'] = "16"
    train(cfgs)