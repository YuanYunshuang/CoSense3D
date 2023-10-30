
from cosense3d.agents import core


def get_controller(cfg, mode):
    return CenterController(cfg, mode)


class CenterController:
    def __init__(self, cfg, mode):
        self.mode = mode
        self.setup_core(cfg)

    def setup_core(self, cfg):
        self.data_manager = core.DataManager(cfg[f'pre_process_{self.mode}'],
                                             cfg.get(f'post_process_{self.mode}', None))
        self.cav_manager = core.CAVManager(cfg['cav_manager'])
        self.shared_modules = core.SharedModules(cfg['shared_modules'])
        self.forward_runner = core.ForwardRunner(cfg['forward_runner'])

    def run_iter(self, batch_dict):
        batch_dict = self.data_manager.preP(batch_dict)

        return batch_dict

    def loss(self):
        pass

