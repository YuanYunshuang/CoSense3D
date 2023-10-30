from mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead
from cosense3d.config.pycfg.stream_petr import pts_bbox_head
from cosense3d.model.heads.sequence_runner import SequenceRunner


class TrackQueryBasedCam(SequenceRunner):
    def __init__(self, cfgs):
        super().__init__(**cfgs)

        self.stream_petr_head = StreamPETRHead(**pts_bbox_head)

    def frame_forward(self,
                      skip=False,
                      seq_len=None,
                      batch_size=None,
                      img_metas=None,
                      img_roi=None,
                      **data):
        location = img_roi['location']
        topk_indices = img_roi['roi']['topk_indexes'] if img_roi['roi'] is not None else None
        out = None
        if not skip:
            out = self.stream_petr_head(location, img_metas, topk_indices, **data)

        return out

    def _loss(self, out_dict,
              gt_bboxes_3d=None,
              gt_labels_3d=None,
              **data):
        loss_dict = self.stream_petr_head.loss(gt_bboxes_3d, gt_labels_3d, out_dict)

        return loss_dict

    def loss(self, batch_dict):
        loss = 0
        for i, ldict in self.loss_dict.items():
            for k, v in ldict.items():
                loss = loss + v
        # only record losses for the newest frame and last layer of decoder
        t = max(list(self.loss_dict.keys()))
        loss_dict = {k: v for k, v in self.loss_dict[t].items() if '.' not in k}

        return loss, loss_dict








