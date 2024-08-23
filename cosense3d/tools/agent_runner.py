import os, sys
import argparse
import logging

import numpy as np
import torch

from cosense3d.dataset import get_dataloader
from cosense3d.utils.misc import setup_logger
from cosense3d.config import load_config, save_config
from cosense3d.utils.train_utils import seed_everything
from cosense3d.agents.center_controller import CenterController
from cosense3d.agents.core.train_runner import TrainRunner
from cosense3d.agents.core.test_runner import TestRunner
from cosense3d.agents.core.vis_runner import VisRunner
from cosense3d.tools.path_cfgs import parse_paths


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--mode", type=str, default="test",
                        help="train | test | vis_train | vis_test")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--log-dir", type=str, default=f"{os.path.dirname(__file__)}/../../logs")
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--meta-path", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-workers", type=int)
    parser.add_argument("--data-latency", type=int,
                        help="-1: random latency selected from (0, 1, 2)*100ms;\n"
                             " 0: coop. data has no additional latency relative to ego frame;\n"
                             " n>0: coop. data has n*100ms latency relative to ego frame.")
    parser.add_argument("--loc-err", type=str,
                        help="localization errors for x, y translation "
                             "and rotation angle along z-axis."
                             "example: `0.5,0.5,1` for 0.5m deviation at x and y axis "
                             "and 1 degree rotation angle")
    parser.add_argument("--cnt-cpm-size", action="store_true")
    parser.add_argument("--cpm-thr", type=float, default=0.0)
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
    if args.meta_path is not None:
        cfgs['DATASET']['meta_path'] = args.meta_path
    if args.data_path is not None:
        cfgs['DATASET']['data_path'] = args.data_path
    if args.data_latency is not None:
        cfgs['DATASET']['latency'] = args.data_latency
    if args.loc_err is not None:
        loc_err = [float(x) for x in args.loc_err.split(',')]
        cfgs['DATASET']['loc_err'] = [loc_err[0], loc_err[1], np.deg2rad(loc_err[2])]
    if args.cnt_cpm_size:
        cfgs['TEST']['hooks'].append({'type': 'CPMStatisticHook'})
        cfgs['CONTROLLER']['cav_manager']['cpm_statistic'] = True
    if args.cpm_thr is not None:
        cfgs['CONTROLLER']['cav_manager']['share_score_thr'] = args.cpm_thr

    agent_runner = AgentRunner(args, cfgs)
    if args.mode == "train":
        save_config(cfgs, agent_runner.runner.logdir)
    agent_runner.run()
