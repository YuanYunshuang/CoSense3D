import torch


class SequenceRunner(torch.nn.Module):
    def __init__(self, name,
                 num_frame_grads=1,
                 num_frame_losses=1,
                 seq_order=1, # 1:( t-T,..., t); -1:(t, ..., t-T)
                 skip_while_no_grad=False,
                 return_result=False,
                 **kwargs):
        super().__init__()
        self.name = name
        self.num_frame_grads = num_frame_grads
        self.num_frame_losses = num_frame_losses
        self.seq_order = seq_order
        self.skip_while_no_grad = skip_while_no_grad
        self.return_result = return_result
        self.loss_dict = {}

    def forward(self, batch_dict):
        seq_len = batch_dict['seq_len']
        seq_out_dict = []

        if self.training:
            if self.seq_order == 1:
                seq_indices = {i: idx for i, idx in enumerate(range(seq_len))}
            elif self.seq_order == -1:
                seq_indices = {i: idx for i, idx in enumerate(range(seq_len - 1, -1, -1))}
            else:
                raise IOError('seq_order should be either 1 or -1.')
            # loop: t-T --> t
            for i, seq_idx in seq_indices.items():
                require_grad = True if i >= seq_len - self.num_frame_grads else False
                return_loss = True if i >= seq_len - self.num_frame_losses else False

                data = self.get_frame_data(batch_dict, seq_idx)

                if not require_grad:
                    self.eval()
                    with torch.no_grad():
                        out_dict = self.frame_forward(skip=self.skip_while_no_grad, **data)
                    self.train()
                else:
                    out_dict = self.frame_forward(**data)

                if return_loss:
                    loss_dict = self._loss(out_dict, **data)
                    self.loss_dict[i] = loss_dict
                seq_out_dict.append(out_dict)

        if self.return_result:
            batch_dict[self.name] = seq_out_dict

    def frame_forward(self, seq_idx, skip=False, **batch_dict):
        raise NotImplementedError

    def _loss(self, out_dict, gt):
        raise NotImplementedError

    def get_frame_data(self, batch_dict, seq_idx):
        data_t = dict()
        for k, v in batch_dict.items():
            if isinstance(v, tuple) or isinstance(v, list):
                data_t[k] = v[seq_idx]
            elif isinstance(v, torch.Tensor):
                data_t[k] = v[:, seq_idx]
            else:
                data_t[k] = v

        return data_t

    def loss(self, batch_dict):
        loss = 0
        for i, ldict in self.loss_dict.items():
            for k, v in ldict.items():
                loss = loss + v
        # only record losses for the newest frame
        t = max(list(self.loss_dict.keys()))

        return loss, self.loss_dict[t]