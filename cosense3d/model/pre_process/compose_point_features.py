import torch
from cosense3d.model.pre_process import PreProcessorBase


class ComposePointFeatures(PreProcessorBase):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs:
            features: str
        """
        super().__init__(**kwargs)

    def __call__(self, data_dict):
        """
        Update batch_dict with key "features"  in batch_dict["data"]
        Compose cooordinates and features according to the symbols,
        each valid symbol will be mapped to a self.get_feature_[symbol] function
        to get the corresponding feature in lidar. Valid symbols are
        - 'x'(coordinate),
        - 'y'(coordinate),
        - 'z'(coordinate),
        - 'i'(intensity),
        - 't'(theta in degree),
        - 'c'(cos(t)),
        - 's'(sin(t)).

        Parameters
        ----------
        data_dict: dict,
                pcds must be np.ndarray [N, 3+c], columns 1-3 are x, y, z,
                if intensity is availble, it should in the 4th column

        Returns
        -------
        """
        if data_dict['pcds'] is None:
            return
        pcds = data_dict['pcds'][:, 1:5]
        features = getattr(self, 'features', "x,y,z")
        features = features.split(',')
        data = [getattr(self, f'get_feature_{f.strip()}')(pcds) for f in features]
        features = torch.cat(data, dim=1)
        data_dict['features'] = features

        # Feature retrieving functions, input lidar0 columns must be # [x,y,z,i,obj,cls]

    @staticmethod
    def get_feature_x(lidar):
        """x coordinate"""
        return lidar[:, 0].reshape(-1, 1)

    @staticmethod
    def get_feature_y(lidar):
        """y coordinate"""
        return lidar[:, 1].reshape(-1, 1)

    @staticmethod
    def get_feature_z(lidar):
        """z coordinate"""
        return lidar[:, 2].reshape(-1, 1)

    @staticmethod
    def get_feature_i(lidar):
        """intensity"""
        if lidar.shape[1] > 3:
            return lidar[:, 3].reshape(-1, 1)
        else:
            return torch.ones_like(lidar[:, 0]).reshape(-1, 1)

    @staticmethod
    def get_feature_t(lidar):
        """orientation"""
        degs = torch.rad2deg(torch.arctan2(lidar[:, 1], lidar[:, 0]).reshape(-1, 1))
        degs = (degs + 360) % 360
        return degs

    @staticmethod
    def get_feature_d(lidar):
        """distance"""
        return torch.norm(lidar[:, :2], axis=1).reshape(-1, 1)

    @staticmethod
    def get_feature_c(lidar):
        """cosine"""
        return torch.cos(torch.arctan2(lidar[:, 1], lidar[:, 0])).reshape(-1, 1)

    @staticmethod
    def get_feature_s(lidar):
        """sine"""
        return torch.sin(torch.arctan2(lidar[:, 1], lidar[:, 0])).reshape(-1, 1)

    @staticmethod
    def get_feature_cs(lidar):
        """normalized coordinate in euclidian system"""
        x_abs = 1 / (torch.abs(lidar[:, 1] / (lidar[:, 0] +
                                           (lidar[:, 0] == 0) * 1e-6)) + 1)
        y_abs = 1 - x_abs
        x = x_abs * torch.sign(lidar[:, 0])
        y = y_abs * torch.sign(lidar[:, 1])
        return torch.stack([x, y], dim=1)