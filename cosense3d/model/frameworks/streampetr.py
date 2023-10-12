from cosense3d.model.frameworks import Model


class StreamPETR(Model):
    def __init__(self, cfgs, mode='train'):
        super().__init__(cfgs, mode)

    def forward(self, batch_dict):
        if getattr(self, 'pre_process', False):
            batch_dict = self.pre_process(batch_dict)
        for module_name in self.modules:
            getattr(self, module_name)(batch_dict)
        if getattr(self, 'post_process', False):
            batch_dict = self.post_process(batch_dict)
        return batch_dict

    def loss(self, batch_dict):
        loss_total = 0
        loss_dict = {}
        for module_name in self.modules:
            module = getattr(self, module_name)
            if hasattr(module, 'loss'):
                loss, ldict = module.loss(batch_dict)
                loss_total = loss_total + loss
                loss_dict.update(ldict)
        loss_dict['total'] = loss_total
        return loss_total, loss_dict