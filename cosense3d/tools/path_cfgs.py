
def parse_opv2v_paths(cfgs):
    import socket
    path_map = {
        "ISI": {
            "data": "/home/yuan/data/OPV2Va",
            "meta": "/home/yuan/data/OPV2Va/meta"
        },
        "mars": {
            "data": "/media/yuan/luna/data/OPV2Va",
            "meta": "/media/yuan/luna/data/OPV2Va/meta"
        },
        "ominotago": {
            "data": "/koko/OPV2V/temporal",
            "meta": "/koko/cosense3d/opv2vt"
        },
        "lavander": {
            "data": "/home/data/OPV2Va",
            "meta": "/home/data/OPV2Va/meta",
        },
        "docker": {
            "data": "/data/OPV2Va",
            "meta": "/data/OPV2Va/meta",
        },
        "default": {
            "data": "/data/OPV2Vt",
            "meta": "/data/OPV2Vt/opv2vt"
        }
    }
    name = socket.gethostname()
    if name not in path_map:
        name = "default"
    cfgs['DATASET']['data_path'] = path_map[name]['data']
    cfgs['DATASET']['meta_path'] = path_map[name]['meta']
    cfgs['DATASET']['enable_split_sub_folder'] = True
    return cfgs

def parse_v2vreal_paths(cfgs):
    import socket
    path_map = {
        "ISI": {
            "data": "/home/yuan/data/OPV2Va",
            "meta": "/home/yuan/data/OPV2Va/meta"
        },
        "mars": {
            "data": "/media/yuan/luna/data/OPV2Va",
            "meta": "/media/yuan/luna/data/OPV2Va/meta"
        },
        "ominotago": {
            "data": "/koko/OPV2V/temporal",
            "meta": "/koko/cosense3d/opv2vt"
        },
        "lavander": {
            "data": "/home/data/v2vreal",
            "meta": "/home/data/v2vreal/meta",
        },
        "docker": {
            "data": "/data/OPV2Va",
            "meta": "/data/OPV2Va/meta",
        },
        "default": {
            "data": "/data/OPV2Vt",
            "meta": "/data/OPV2Vt/opv2vt"
        }
    }
    name = socket.gethostname()
    if name not in path_map:
        name = "default"
    cfgs['DATASET']['data_path'] = path_map[name]['data']
    cfgs['DATASET']['meta_path'] = path_map[name]['meta']
    cfgs['DATASET']['enable_split_sub_folder'] = True
    return cfgs


def parse_opv2vt_paths(cfgs):
    import socket
    path_map = {
        "ISI": {
            "data": "/home/yuan/data/OPV2Vt",
            "meta": "/home/yuan/data/OPV2Vt/meta"
        },
        "mars": {
            "data": "/koko/OPV2V/temporal",
            "meta": "/koko/cosense3d/opv2vt"
        },
        "ominotago": {
            "data": "/koko/OPV2V/temporal",
            "meta": "/koko/cosense3d/opv2vt"
        },
        "lavander": {
            "data": "/koko/OPV2V/temporal",
            "meta": "/koko/cosense3d/opv2vt",
            "map": "/koko/OPV2V/maps"
        },
        "default": {
            "data": "/data/OPV2Vt",
            "meta": "/data/OPV2Vt/opv2vt"
        }
    }
    name = socket.gethostname()
    if name not in path_map:
        name = "default"
    cfgs['DATASET']['data_path'] = path_map[name]['data']
    cfgs['DATASET']['meta_path'] = path_map[name]['meta']
    cfgs['DATASET']['map_path'] = path_map[name].get('map', None)
    cfgs['DATASET']['enable_split_sub_folder'] = True
    return cfgs


def parse_dairv2xt_paths(cfgs):
    import socket
    path_map = {
        "ISI": {
            "data": "/home/yuan/data/DairV2Xt",
            "meta": "/home/yuan/data/DairV2Xt/meta"
        },
        "mars": {
            "data": "/koko/DAIR-V2X",
            "meta": "/media/yuan/luna/data/DairV2Xt/meta_with_pred"
        },
        "ominotago": {
            "data": "/koko/yunshuang/DAIR-V2X",
            "meta": "/koko/yunshuang/DAIR-V2X/meta_with_pred"
        },
        "lavander": {
            "data": "/home/data/DAIR-V2X",
            "meta": "/home/data/DAIR-V2X/meta_with_pred"
        },
        "default": {
            "data": "/data/DairV2Xt",
            "meta": "/data/DairV2Xt/meta_with_pred",
        }
    }
    name = socket.gethostname()
    if name not in path_map:
        name = "default"
    cfgs['DATASET']['data_path'] = path_map[name]['data']
    cfgs['DATASET']['meta_path'] = path_map[name]['meta']
    cfgs['DATASET']['enable_split_sub_folder'] = False
    return cfgs


def parse_paths(cfgs):
    if 'opv2v' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_opv2v_paths(cfgs)
    elif 'v2vreal' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_v2vreal_paths(cfgs)
    elif 'opv2vt' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_opv2vt_paths(cfgs)
    elif 'dairv2xt' == cfgs['DATASET']['data_path'].lower():
        cfgs = parse_dairv2xt_paths(cfgs)
    return cfgs