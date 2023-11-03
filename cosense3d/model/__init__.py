from cosense3d.model import frameworks
import importlib


def get_model(cfgs, mode):
    model_name = cfgs.get('name', 'Model')
    model_cls = getattr(frameworks, model_name)
    return model_cls(cfgs, mode)





