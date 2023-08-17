import logging
from importlib import import_module


class Compose(object):
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

    def __call__(self, data_dict):
        for p in self.processes:
            p(data_dict)
        return data_dict


class PreProcessorBase(object):
    def __init__(self, **kwargs):
        logging.info(f"{self.__class__.__name__}:")
        for k, v in kwargs.items():
            setattr(self, k, v)
            logging.info(f"- {k}: {v}")

    def __call__(self, data_dict):
        raise NotImplementedError