from cosense3d.model import frameworks
import importlib


def get_model(cfgs, mode):
    model_name = cfgs.get('name', 'Model')
    model_cls = getattr(frameworks, model_name)
    return model_cls(cfgs, mode)


def build_module(module_cfg):
    module_full_path=module_cfg['type']
    package, module_name = module_full_path.rsplit('.', 1)
    module = importlib.import_module(f'cosense3d.model.{package}')
    cls_obj = getattr(module, module_name, None)
    assert cls_obj is not None, f'Class \'{module_name}\' not found.'
    inst = cls_obj(**module_cfg)
    return inst


