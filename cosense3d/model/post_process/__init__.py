from importlib import import_module
from cosense3d.model.utils.me_utils import update_me_essentials


class Compose:
    """Composes several pre-processing modules together.
        Take care that these functions modify the input data directly.
    """

    def __init__(self, cfg):
        self.processes = []
        for module_cfg in cfg:
            for k, v in module_cfg.items():
                module = import_module(f"{globals()['__name__']}.{k}")
                cls_name = ''
                for word in k.split('_'):
                    cls_name += word[:1].upper() + word[1:]
                cls_obj = getattr(module, cls_name, None)
                assert cls_obj is not None, f'Class \'{cls_name}\' not found.'
                cls_inst = cls_obj(**v)
                self.processes.append(cls_inst)

    def __call__(self, *args):
        out_dict = {}
        for t in self.processes:
            out = t(*args)
            out_dict.update(out)
        return out_dict

    def set_log_dir(self, log_dir):
        for t in self.processes:
            t.set_log_dir(log_dir)


class PostProcess:
    def __init__(self, data_info, stride=None):
        update_me_essentials(self, data_info, stride)

    def __call__(self, batch_dict):
        raise NotImplementedError

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir