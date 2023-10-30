import os
from importlib import import_module

from cosense3d.utils.misc import load_yaml, save_yaml, update_dict
from cosense3d.config import pycfg


def load_config(args):
    """
    Load yaml config file, merge additional config in args
    and return a dictionary.

    Parameters
    ----------
    args : argparse object or str
        if is str, it should be the yaml config filename
        else args.config indicates config yaml file

    Returns
    -------
    params : dict
        A dictionary that contains defined parameters.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    cfg = {}
    if isinstance(args, str):
        main_cfg = load_yaml(args)
    else:
        # load default
        # modules_default = load_yaml("./config/defaults/modules.yaml")
        # update_dict(cfg, modules_default)
        main_cfg = load_yaml(args.config)
    default_file = f"{path}/defaults/{main_cfg['DATASET']['name']}.yaml"
    if os.path.exists(default_file):
        dataset_default = load_yaml(default_file)
        update_dict(cfg, dataset_default)
    update_dict(cfg, main_cfg)
    parse_pycfg(cfg)

    if isinstance(args, str):
        cfg['TRAIN']['log_dir'] = os.path.dirname(args)
    else:
        # update params
        if hasattr(args, 'run_name'):
            cfg['TRAIN']['run_name'] = args.run_name
        if hasattr(args, 'log_dir'):
            cfg['TRAIN']['log_dir'] = args.log_dir
        if hasattr(args, 'resume'):
            cfg['TRAIN']['resume'] = args.resume

    return cfg


def save_config(config_dict, filename):
    """
    Save config dictionary into yaml file.

    Parameters
    ----------
    config_dict : dict
    filename : str

    Returns
    -------
    """
    config_dict['TRAIN']['save_path'] = filename
    filename = os.path.join(filename, "config.yaml")
    save_yaml(config_dict, filename)


def parse_pycfg(cfg_dict):
    for k, v in cfg_dict.items():
        if isinstance(v, str) and 'pycfg' in v:
            m, n = v.rsplit('.', 1)
            module = import_module(f'cosense3d.config.{m}')
            cfg_dict[k] = getattr(module, n)
        elif isinstance(v, dict):
            parse_pycfg(v)




