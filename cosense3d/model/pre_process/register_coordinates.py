import torch
from cosense3d.model.pre_process import PreProcessorBase


class RegisterCoordinates(PreProcessorBase):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs:
            raster_resolution: float, optional
            height: list, optional
        """
        super().__init__(**kwargs)

    def __call__(self, data_dict):
        if data_dict['pcds'] is None:
            return
        pcds = data_dict['pcds']  # in local, global_rot or aug_rot
        objects = data_dict['objects']

        if hasattr(self, 'raster_resolution'):
            tfs = data_dict['tf_cav2ego']
            map_anchors = []
            anchor_offsets = []
            tfs_new = []
            for i, tf in enumerate(tfs):
                pcd_mask = pcds[:, 0] == i
                loc = tf[3, :2]
                # find global registration anchor
                map_anchor = torch.round(loc / self.raster_resolution) \
                             * self.raster_resolution
                # transform pcd and pose to anchor coords
                offset = loc - map_anchor
                pcds[pcd_mask, 1:3] += offset.reshape(1, 2)

                tf[3, :2] -= offset
                tfs_new.append(tf)
                map_anchors.append(map_anchor)
                anchor_offsets.append(offset)

            data_dict['tf_cav2ego'] = tfs_new
            data_dict['map_anchors'] = map_anchors
            data_dict['anchor_offsets'] = anchor_offsets

        if hasattr(self, 'height'):
            pcds[:, 3] -= self.height
            if objects is not None:
                objects[:, 5] -= self.height

        data_dict['pcds'] = pcds
        data_dict['objects'] = objects