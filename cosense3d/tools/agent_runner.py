import os, sys
import argparse
import logging

import torch

from cosense3d.dataset import get_dataloader
from cosense3d.utils.misc import setup_logger
from cosense3d.config import load_config, save_config
from cosense3d.utils.train_utils import seed_everything
from cosense3d.agents.center_controller import CenterController
from cosense3d.agents.core.train_runner import TrainRunner
from cosense3d.agents.core.test_runner import TestRunner
from cosense3d.agents.core.vis_runner import VisRunner


def ddp_setup():
    from torch.distributed import init_process_group
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class AgentRunner:
    def __init__(self, args, cfgs):
        self.visualize = args.visualize or 'vis' in args.mode
        self.mode = args.mode
        if args.gpus > 0:
            self.dist = True
            ddp_setup()
        else:
            self.dist = False
        if self.visualize:
            from cosense3d.agents.core.gui import GUI
            from PyQt5.QtWidgets import QApplication
            self.app = QApplication(sys.argv)
            self.gui = GUI(args.mode, cfgs['VISUALIZATION'])

        self.build_runner(args, cfgs)

    def build_runner(self, args, cfgs):
        dataloader = get_dataloader(cfgs['DATASET'],
                                    args.mode.replace('vis_', ''),
                                    self.dist)
        center_controller = CenterController(cfgs['CONTROLLER'], dataloader, self.dist)
        if args.mode == 'train':
            self.runner = TrainRunner(dataloader=dataloader,
                                      controller=center_controller,
                                      **cfgs['TRAIN'])
        elif args.mode == 'test':
            self.runner = TestRunner(dataloader=dataloader,
                                     controller=center_controller,
                                     **cfgs['TEST'])
        else:
            self.runner = VisRunner(dataloader=dataloader,
                                    controller=center_controller,)

    def visible_run(self):
        self.gui.setRunner(self.runner)
        self.app.installEventFilter(self.gui)

        # self.app.setStyle("Fusion")
        from PyQt5.QtWidgets import QDesktopWidget
        desktop = QDesktopWidget().availableGeometry()
        width = (desktop.width() - self.gui.width()) / 2
        height = (desktop.height() - self.gui.height()) / 2

        self.gui.move(int(width), int(height))
        self.gui.initGUI()
        # Start GUI
        self.gui.show()

        logging.info("Showing GUI...")
        sys.exit(self.app.exec_())

    def run(self):
        try:
            if self.visualize:
                self.visible_run()
            else:
                self.runner.run()
        finally:
            if self.dist:
                from torch.distributed import destroy_process_group
                destroy_process_group()


def parse_opv2v_paths(cfgs):
    import socket
    path_map = {
        "ISI": {
            "data": "/home/yuan/data/OPV2V/temporal",
            "meta": "/home/yuan/data/cosense3d/opv2vt"
        },
        "mars": {
            "data": "/koko/OPV2V/temporal",
            "meta": "/koko/cosense3d/opv2vt"
        },
        "ominotago": {
            "data": "/koko/OPV2V/temporal",
            "meta": "/koko/cosense3d/opv2vt"
        },
        "lavander": {
            "data": "/koko/OPV2V/temporal",
            "meta": "/koko/cosense3d/opv2vt"
        },
    }
    name = socket.gethostname()
    cfgs['DATASET']['data_path'] = path_map[name]['data']
    cfgs['DATASET']['meta_path'] = path_map[name]['meta']
    cfgs['DATASET']['enable_split_sub_folder'] = True
    return cfgs


def parse_dairv2x_paths(cfgs):
    import socket
    path_map = {
        "ISI": {
            "data": "/home/yuan/data/DAIR-V2X",
            "meta": "/home/yuan/data/DAIR-V2X/meta_with_pred"
        },
        "mars": {
            "data": "/koko/DAIR-V2X",
            "meta": "/media/yuan/luna/cosense3d/meta_with_pred"
        },
        "ominotago": {
            "data": "/koko/DAIR-V2X",
            "meta": "/koko/cosense3d/meta_with_pred"
        },
        "lavander": {
            "data": "/home/data/DAIR-V2X",
            "meta": "/home/data/DAIR-V2X/meta_with_pred"
        },
    }
    name = socket.gethostname()
    cfgs['DATASET']['data_path'] = path_map[name]['data']
    cfgs['DATASET']['meta_path'] = path_map[name]['meta']
    cfgs['DATASET']['enable_split_sub_folder'] = False
    return cfgs


def parse_paths(cfgs):
    if 'opv2v' in cfgs['DATASET']['data_path'].lower():
        cfgs = parse_opv2v_paths(cfgs)
    elif 'dair' in cfgs['DATASET']['data_path'].lower():
        cfgs = parse_dairv2x_paths(cfgs)
    return cfgs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--mode", type=str, default="test",
                        help="train | test | vis_train | vis_test")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--log-dir", type=str, default=f"{os.path.dirname(__file__)}/../logs")
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--meta-path", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-workers", type=int)
    args = parser.parse_args()

    setup_logger(args.run_name, args.debug)
    # for ME
    os.environ['OMP_NUM_THREADS'] = "16"
    # if 'vis' in args.mode:
    #     args.config = "./config/defaults/base_cav.yaml"

    seed_everything(2023)
    cfgs = load_config(args)
    if args.gpus:
        cfgs['TRAIN']['gpus'] = args.gpus
    if args.batch_size is not None:
        cfgs['DATASET']['batch_size_train'] = args.batch_size
    if args.n_workers is not None:
        cfgs['DATASET']['n_workers'] = args.n_workers
    parse_paths(cfgs)
    agent_runner = AgentRunner(args, cfgs)
    if args.mode == "train":
        save_config(cfgs, agent_runner.runner.logdir)
    agent_runner.run()