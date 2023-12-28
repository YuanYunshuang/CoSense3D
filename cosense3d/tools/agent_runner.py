import os, sys
import argparse
import logging

from cosense3d.dataset import get_dataloader
from cosense3d.utils.misc import setup_logger
from cosense3d.config import load_config, save_config
from cosense3d.utils.train_utils import seed_everything
from cosense3d.agents.center_controller import CenterController
from cosense3d.agents.core.train_runner import TrainRunner
from cosense3d.agents.core.test_runner import TestRunner
from cosense3d.agents.core.vis_runner import VisRunner


class AgentRunner:
    def __init__(self, args, cfgs):
        self.visualize = args.visualize or 'vis' in args.mode
        self.mode = args.mode
        if self.visualize:
            from cosense3d.agents.core.gui import GUI
            from PyQt5.QtWidgets import QApplication
            self.app = QApplication(sys.argv)
            self.gui = GUI(args.mode, cfgs['VISUALIZATION'])

        self.build_runner(args, cfgs)

    def build_runner(self, args, cfgs):
        dataloader = get_dataloader(cfgs['DATASET'], args.mode.replace('vis_', ''))
        center_controller = CenterController(cfgs['CONTROLLER'], dataloader)
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
        if self.visualize:
            self.visible_run()
        else:
            self.runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--mode", type=str, default="test",
                        help="train | test | vis_train | vis_test")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--log_dir", type=str, default=f"{os.path.dirname(__file__)}/../logs")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logger(args.run_name, args.debug)
    # for ME
    os.environ['OMP_NUM_THREADS'] = "16"
    # if 'vis' in args.mode:
    #     args.config = "./config/defaults/base_cav.yaml"

    seed_everything(2023)
    cfgs = load_config(args)
    if not os.path.exists(cfgs['DATASET']['data_path']):
        if 'temporal' in cfgs['DATASET']['data_path']:
            cfgs['DATASET']['data_path'] = "/koko/OPV2V/temporal"
            cfgs['DATASET']['meta_path'] = "/koko/cosense3d/opv2v_temporal"
        else:
            cfgs['DATASET']['data_path'] = "/koko/OPV2V"
            cfgs['DATASET']['meta_path'] = "/koko/cosense3d/opv2v_full"
    agent_runner = AgentRunner(args, cfgs)
    if args.mode == "train":
        save_config(cfgs, agent_runner.runner.logdir)
    agent_runner.run()