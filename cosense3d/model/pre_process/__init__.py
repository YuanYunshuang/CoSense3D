import logging
from importlib import import_module
from collections import OrderedDict

from .transform import *
from .formatting import *


class PreProcess(object):
    """Composes several pre-processing modules together.
        Take care that these functions modify the input data directly.
    """

    def __init__(self, cfgs):
        self.processes = []
        if isinstance(cfgs, list):
            for module_cfg in cfgs:
                for k, v in module_cfg.items():
                    self.build_process(k, v)
        elif isinstance(cfgs, OrderedDict):
            for k, v in cfgs.items():
                self.build_process(k, v)
        else:
            raise NotImplementedError

    def build_process(self, k, v):
        if '.' in k:
            k, cls_name = k.split('.')
            module = import_module(f"{globals()['__name__']}.{k}")
        elif '_' in k:
            cls_name = ''
            for word in k.split('_'):
                cls_name += word[:1].upper() + word[1:]
            module = import_module(f"{globals()['__name__']}.{k}")
        else:
            cls_name = k
            module = import_module(f"{globals()['__name__']}")
        cls_obj = getattr(module, cls_name, None)
        assert cls_obj is not None, f'Class \'{cls_name}\' not found.'
        cls_inst = cls_obj(**v)
        self.processes.append(cls_inst)

    def __call__(self, data_dict):
        if 'transforms' not in data_dict:
            data_dict['transforms'] = {}
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