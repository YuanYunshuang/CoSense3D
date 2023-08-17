import importlib, logging
from torch import nn
from cosense3d.model import backbone_3d, submodules, pre_process, post_process


def get_model(cfgs, mode):
    return Model(cfgs, mode)


class Model(nn.Module):
    def __init__(self, cfgs, mode='train'):
        super(Model, self).__init__()
        logging.info("Loading Model ...")
        blk_list = ['backbone_3d', 'backbone_2d', 'submodules', 'heads']

        self.modules = []
        for blk_name in blk_list:
            if blk_name in cfgs:
                modules = self.set_module(blk_name, cfgs[blk_name])
                self.modules.extend(modules)

        if 'pre_process' in cfgs:
            self.pre_process = pre_process.Compose(cfgs['pre_process'])
        if 'post_process' in cfgs and mode == 'test':
            self.eval()
            self.post_process = post_process.Compose(cfgs['post_process'])

    def set_module(self, name, cfgs):
        modules = []
        for module_dict in cfgs:
            for module_name, module_values in module_dict.items():
                m = self.get_object_instance(name, module_name, module_values)
                setattr(self, module_name.lower(), m)
                modules.append(module_name.lower())
        return modules

    def get_object_instance(self, package, module_name, cfgs):
        module = importlib.import_module(f'cosense3d.model.{package}.{module_name}')

        cls_name = ''
        for word in module_name.split('_'):
            cls_name += word[:1].upper() + word[1:]
        cls_obj = getattr(module, cls_name, None)
        assert cls_obj is not None, f'Class \'{cls_name}\' not found.'
        inst = cls_obj(cfgs)
        return inst

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