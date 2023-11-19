import sys

import torch

from cosense3d.agents import core


class CenterController:
    def __init__(self, cfg, data_loader):
        self.mode = data_loader.dataset.mode
        self.seq_len = data_loader.dataset.seq_len
        self.data_info = data_loader.dataset.cfgs['data_info']
        self.num_loss_frame = cfg.get('num_loss_frame', 1)
        self.setup_core(cfg)
        self.global_data = {}

    def setup_core(self, cfg):
        self.cav_manager = core.CAVManager(**self.update_cfg(cfg['cav_manager'], self.data_info))
        self.data_manager = core.DataManager(
            self.cav_manager, **self.update_cfg(cfg['data_manager'][self.mode], self.data_info))
        self.forward_runner = core.ForwardRunner(cfg['shared_modules'], self.data_manager)
        self.task_manager = core.TaskManager()

    def update_cfg(self, cfg, *args):
        for arg in args:
            cfg.update(arg)
        return cfg

    @property
    def modules(self):
        return self.forward_runner.shared_modules

    @property
    def model(self):
        return self.forward_runner

    @property
    def parameters(self):
        return self.forward_runner.parameters()

    def train_forward(self, batch_dict, **kwargs):
        self.data_manager.generate_augment_params(batch_dict, self.seq_len)
        seq_data = self.data_manager.distribute_to_seq_list(batch_dict, self.seq_len)
        self.cav_manager.reset()
        loss = 0
        loss_dict = {}
        for i in range(self.seq_len):
            with_loss = i >= self.seq_len - self.num_loss_frame
            frame_loss_dict = self.run_frame(seq_data[i], with_loss, training_mode=True, **kwargs)
            for k, v in frame_loss_dict.items():
                if 'loss' in k:
                    loss = loss + v
                loss_dict[f'f{i}.{k}'] = v
        loss_dict['total_loss'] = loss
        return loss, loss_dict

    def test_forward(self, batch_dict, **kwargs):
        self.data_manager.generate_augment_params(batch_dict, self.seq_len)
        seq_data = self.data_manager.distribute_to_seq_list(batch_dict, self.seq_len)
        for i in range(self.seq_len):
            self.run_frame(seq_data[i], with_loss=False, training_mode=False, **kwargs)

    def vis_forward(self, batch_dict, **kwargs):
        self.data_manager.generate_augment_params(batch_dict, self.seq_len)
        seq_data = self.data_manager.distribute_to_seq_list(batch_dict, self.seq_len)
        frame_data = seq_data[0]
        self.cav_manager.update_cav_info(**frame_data)
        self.data_manager.distribute_to_cav(**frame_data)
        # send and receive request
        request = self.cav_manager.send_request()
        self.cav_manager.receive_request(request)
        # apply data online transform
        self.cav_manager.forward(False, False)

    def run_frame(self, frame_data, with_loss, training_mode, **kwargs):
        self.cav_manager.update_cav_info(**frame_data)
        self.data_manager.distribute_to_cav(**frame_data)
        self.cav_manager.apply_cav_function('pre_update_memory')
        # send and receive request
        request = self.cav_manager.send_request()
        self.cav_manager.receive_request(request)
        # get pseudo forward tasks
        tasks = self.cav_manager.forward(with_loss, training_mode)
        batched_tasks = self.task_manager.summarize_tasks(tasks)
        # remove empty_boxes after point transformation
        if training_mode:
            self.data_manager.remove_empty_boxes()

        # process local cav data
        if len(batched_tasks[0]['no_grad']) > 0:
            with torch.no_grad():
                self.forward_runner(batched_tasks[0]['no_grad'], **kwargs)
        if not training_mode:
            with torch.no_grad():
                self.forward_runner(batched_tasks[0]['with_grad'], **kwargs)
        else:
            self.forward_runner(batched_tasks[0]['with_grad'], **kwargs)

        # send coop cav feature-level cpm to ego cav
        response = self.cav_manager.send_response()
        self.cav_manager.receive_response(response)

        # process ego cav data and fuse data from coop cav with grad if training
        self.forward_runner(batched_tasks[1]['with_grad'], **kwargs)
        self.cav_manager.apply_cav_function('post_update_memory')

        frame_loss_dict = {}
        if with_loss:
            frame_loss_dict = self.forward_runner.loss(batched_tasks[2]['loss'], **kwargs)
        return frame_loss_dict





