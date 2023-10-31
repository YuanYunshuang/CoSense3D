

class Format2D:
    def __init__(self, ):
        pass

    def __call__(self, data_dict):
        N_cams = len(data_dict['img'])

        def split(v):
            ptr = 0
            v_out = []
            for k, cams in data_dict['chosen_cams'].items():
                n = len(cams)
                v_out.append(v[ptr:ptr+n])
                ptr += n
            return v_out
        for k, v in data_dict.items():
            if 'local' not in k and 'global' not in k and len(v) == N_cams:
                data_dict[k] = split(v)

        return data_dict