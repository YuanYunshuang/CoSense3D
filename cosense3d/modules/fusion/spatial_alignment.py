import torch

from cosense3d.modules import BaseModule
from cosense3d.modules.utils.me_utils import update_me_essentials


class SpatialAlignment(BaseModule):
    def __init__(self, **kwargs):
        super(SpatialAlignment, self).__init__(**kwargs)


    def forward(self, dets_local, cpms=None, **kwargs):
        for ego_det, cpm in zip(dets_local, cpms):
            ego_det_ctr = ego_det['preds']['box'][:, :3]

            for cpm_content in cpm.values():
                coop_det_ctr = cpm_content['coop_det_ctr']

                import matplotlib.pyplot as plt
                pts0 = ego_det_ctr.detach().cpu().numpy()
                pts1 = coop_det_ctr.detach().cpu().numpy()
                plt.plot(pts0[:, 0], pts0[:, 1], 'g.')
                plt.plot(pts1[:, 0], pts1[:, 1], 'r.')
                plt.show()
                plt.close()
