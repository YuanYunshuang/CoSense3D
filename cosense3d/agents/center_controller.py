
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
        self.shared_modules = core.SharedModules(cfg['shared_modules'])
        self.cav_manager = core.CAVManager(self.shared_modules.module_keys)
        self.forward_runner = core.ForwardRunner(self.shared_modules)

    def modules(self):
        return self.shared_modules

    def model(self):
        return self.forward_runner

    def run_iter(self, batch_dict):
        self.cav_manager.update_cav_info(batch_dict)
        batch_dict = self.data_manager.pre_process(batch_dict)
        cav_data = self.data_manager.distribute_data(batch_dict)
        tasks = {}
        for cav in self.cav_manager.cavs:
            tasks[cav.id] = cav.forward(cav_data[cav.id])
        tasks = self.task_manager.summarize_tasks(tasks)
        out = self.forward_runner(tasks)

        return batch_dict

    def loss(self):
        pass

