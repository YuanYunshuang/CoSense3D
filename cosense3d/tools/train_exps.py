from train import train
from cosense3d.config import load_yaml, load_config
from cosense3d.utils.misc import setup_logger
import argparse, os, shutil


def try_train(args):
    cfgs = load_config(args)
    setup_logger(args.run_name, args.debug)
    # for ME
    os.environ['OMP_NUM_THREADS'] = "16"

    # # exp 1
    # cfgs['MODEL']['heads'][0]['bev']['sampling']['annealing'] = False
    # cfgs['MODEL']['heads'][0]['bev']['sampling']['topk'] = False
    # train(cfgs)
    # # exp 2
    # cfgs['MODEL']['heads'][0]['bev']['sampling']['annealing'] = True
    # cfgs['MODEL']['heads'][0]['bev']['sampling']['topk'] = False
    # train(cfgs)
    # exp 3
    cfgs['MODEL']['heads'][0]['bev']['sampling']['annealing'] = True
    cfgs['MODEL']['heads'][0]['bev']['sampling']['topk'] = True
    train(cfgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try_train(args)
