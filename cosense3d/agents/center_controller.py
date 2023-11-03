import sys

import torch

from cosense3d.agents import core


def get_controller(cfg, mode):
    return CenterController(cfg, mode)


class CenterController:
    def __init__(self, cfg, data_loader):
        self.mode = data_loader.dataset.mode
        self.seq_len = data_loader.dataset.seq_len
        self.data_info = data_loader.dataset.cfgs['data_info']
        self.setup_core(cfg)
        self.global_data = {}

    def setup_core(self, cfg):
        self.cav_manager = core.CAVManager(**self.data_info)
        self.data_manager = core.DataManager(self.cav_manager, self.data_info, **cfg['data_manager'])
        self.forward_runner = core.ForwardRunner(cfg['shared_modules'], self.data_manager)
        self.task_manager = core.TaskManager()
        if 'visualizer' in cfg:
            self.visualizer = core.Visualizer(sys.argv)
            self.visualizer.run()

    @property
    def modules(self):
        return self.forward_runner.shared_modules

    @property
    def model(self):
        return self.forward_runner

    @property
    def parameters(self):
        return self.forward_runner.parameters()

    def run_seq(self, batch_dict):
        self.data_manager.generate_augment_params(batch_dict, self.seq_len)
        seq_data = self.data_manager.distribute_to_seq_list(batch_dict, self.seq_len)

        for i in range(self.seq_len):
            self.cav_manager.update_cav_info(**seq_data[i])
            self.global_data = self.data_manager.distribute_to_cav(**seq_data[i])
            # send and receive request
            request = self.cav_manager.send_request()
            self.cav_manager.receive_request(request)
            # pseudo forward
            tasks = self.cav_manager.forward()

            # pseudo fusion
            batched_tasks = self.task_manager.summary(tasks)
            self.forward_runner.eval()
            self.forward_runner(batched_tasks['no_grad'])
            response = self.cav_manager.send_response()
            self.cav_manager.receive_response(response)
            self.forward_runner.train()
            self.forward_runner(batched_tasks['with_grad'])
            self.cav_manager.reset_cpm_memory()

        # send and receive cpms


        return batch_dict

    def loss(self):
        pass

