import os

from cosense3d.utils.misc import load_yaml, save_yaml, update_dict


def load_config(args):
    """
    Load yaml config file, merge additional config in args
    and return a dictionary.

    Parameters
    ----------
    args : argparse object
        args.config indicates config yaml file

    Returns
    -------
    params : dict
        A dictionary that contains defined parameters.
    """
    cfg = {}
    # load default
    # modules_default = load_yaml("./config/defaults/modules.yaml")
    # update_dict(cfg, modules_default)
    main_cfg = load_yaml(args.config)
    dataset_default = load_yaml(f"./config/defaults/{main_cfg['DATASET']['name']}.yaml")
    update_dict(cfg, dataset_default)
    update_dict(cfg, main_cfg)

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



