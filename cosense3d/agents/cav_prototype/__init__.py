import importlib


def get_prototype(module_full_path):
    module_name, cls_name = module_full_path.rsplit('.', 1)
    module = importlib.import_module(f'cosense3d.agents.cav_prototype.{module_name}')
    cls_obj = getattr(module, cls_name, None)
    assert cls_obj is not None, f'Class \'{module_name}\' not found.'
    return cls_obj