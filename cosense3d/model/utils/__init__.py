from importlib import import_module

import torch
from torch import nn
import numpy as np

import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator
from torch.distributions.multivariate_normal import _batch_mahalanobis
# TODO move ME relevant functions to me_utils
pi = 3.141592653


def cat_name_str(module_name):
    """

    Parameters
    ----------
    module_name: str, format in xxx_yyy_zzz

    Returns
    -------
    class_name: str, format in XxxYyyZzz
    """
    cls_name = ''
    for word in module_name.split('_'):
        cls_name += word[:1].upper() + word[1:]
    return cls_name


def instantiate(module_name, cls_name=None, module_cfg=None, **kwargs):
    package = import_module(f"cosense3d.model.{module_name}")
    cls_name = cat_name_str(module_name) if cls_name is None else cls_name
    obj_cls = getattr(package, cls_name)
    if module_cfg is None:
        obj_inst = obj_cls(**kwargs)
    else:
        obj_inst = obj_cls(module_cfg)
    return obj_inst


def bias_init_with_prob(prior_prob: float) -> float:
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape

        view_shape = [1 for _ in range(len(feat_shape))]
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)

        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def limit_period(val, offset=0.5, period=2 * pi):
    return val - torch.floor(val / period + offset) * period


def sparse_to_dense(stensor, voxel_size, det_r):
    b = int(stensor.C[:, 0].max()) + 1
    d = stensor.F.shape[-1]
    stride = stensor.tensor_stride
    h = int((det_r['x'][1] - det_r['x'][0]) / voxel_size[0]) // stride[0]
    w = int((det_r['y'][1] - det_r['y'][0]) / voxel_size[1]) // stride[1]
    x_offset = int(det_r['x'][0] / voxel_size[0])
    y_offset = int(det_r['y'][0] / voxel_size[1])
    assert len(stensor.C[:, 3].unique()) == 1
    dtensor = stensor.dense(
        shape=torch.Size((b, d, h, w, 1)),
        min_coordinate=torch.Tensor([x_offset, y_offset, 0]).int())[0].squeeze(dim=-1)

    return dtensor


def prepare_input_data(batch_dict, QMODE):
    in_data = ME.TensorField(
        features=batch_dict.pop("features"),
        coordinates=batch_dict.pop("coords"),
        quantization_mode=QMODE
    )
    # ME rounds to the floor when casting coords to integer
    batch_dict["in_data"] = in_data
    return batch_dict


def voxelize_with_centroids(x: ME.TensorField, enc_mlp):
    cm = x.coordinate_manager
    features = x.F
    coords = x.C[:, 1:]

    out = x.sparse()
    size = torch.Size([len(out), len(x)])
    tensor_map, field_map = cm.field_to_sparse_map(x.coordinate_key, out.coordinate_key)
    coords_p1, count_p1 = downsample_points(coords, tensor_map, field_map, size)
    features_p1, _ = downsample_points(features, tensor_map, field_map, size)
    norm_features = normalize_points(features, features_p1, tensor_map)

    voxel_embs = enc_mlp(torch.cat([features, norm_features], dim=1))
    down_voxel_embs = downsample_embeddings(voxel_embs, tensor_map, size, mode="avg")
    out = ME.SparseTensor(down_voxel_embs,
                          coordinate_map_key=out.coordinate_key,
                          coordinate_manager=cm)

    norm_points_p1 = normalize_centroids(coords_p1, out.C, out.tensor_stride[0])
    return out, norm_points_p1, features_p1, count_p1, voxel_embs


def devoxelize_with_centroids(out: ME.SparseTensor, x: ME.TensorField, h_embs):
    feats = torch.cat([out.slice(x).F, h_embs], dim=1)
    return feats


@torch.no_grad()
def normalize_points(points, centroids, tensor_map):
    tensor_map = tensor_map if tensor_map.dtype == torch.int64 else tensor_map.long()
    norm_points = points - centroids[tensor_map]
    return norm_points


@torch.no_grad()
def normalize_centroids(down_points, coordinates, tensor_stride):
    norm_points = (down_points - coordinates[:, 1:]) / tensor_stride - 0.5
    return norm_points


@torch.no_grad()
def get_kernel_map_and_out_key(stensor, stensor_out=None,
                               kernel_size=3, stride=1, dilation=1,
                               kernel_type='cube', kernel_generator=None):
    """
    Generate kernel maps for the input stensor.
    The hybrid and custom kernel is not implemented in ME v0.5.x,
    this function uses a kernel mask to select the kernel maps for
    the customized kernel shapes.
    :param stensor: ME.SparseTensor, NxC
    :param kernel_type: 'cube'(default) | 'hybrid'
    :return: masked kernel maps
    """
    D = stensor.C.shape[-1] - 1
    if kernel_generator is None:
        kernel_generator = KernelGenerator(kernel_size=kernel_size,
                                           stride=stride,
                                           dilation=dilation,
                                           dimension=D)
    assert D == len(kernel_generator.kernel_stride)
    cm = stensor.coordinate_manager
    in_key = stensor.coordinate_key
    if stensor_out is None:
        out_key = cm.stride(in_key, kernel_generator.kernel_stride)
    else:
        out_key = stensor_out.coordinate_key
    region_type, region_offset, _ = kernel_generator.get_kernel(
        stensor.tensor_stride, False)

    kernel_map = cm.kernel_map(in_key,
                               out_key,
                               kernel_generator.kernel_stride,
                               kernel_generator.kernel_size,
                               kernel_generator.kernel_dilation,
                               region_type=region_type,
                               region_offset=region_offset)
    if kernel_type=='cube':
        kernel_volume = kernel_generator.kernel_volume
    elif kernel_type=='hybrid':
        assert dilation == 1, "currently, hybrid kernel only support dilation=1."
        xx = torch.tensor([-1, 0, 1]).int()
        xx_list = [xx for i in range(D)]
        kernels = torch.meshgrid([*xx_list], indexing='ij')
        kernels = torch.stack([t.flatten() for t in kernels], dim=1)
        kernel_mask = torch.zeros_like(kernels[:, 0]).bool()
        m = torch.logical_or(
            kernels[:, 0] == 0,
            torch.logical_and(kernels[:, 0]==-1, (kernels[:, 1:]==0).all(dim=1))
        )
        kernel_mask[m] = True
        kernel_mask_map = {ic.item(): ih for ih, ic in enumerate(torch.where(kernel_mask)[0])}
        kernel_map = {kernel_mask_map[k]: v for k, v in kernel_map.items() if kernel_mask[k]}
        kernel_volume = kernel_mask.sum().item()
    else:
        raise NotImplementedError

    return kernel_map, out_key, kernel_volume


@torch.no_grad()
def downsample_points(points, tensor_map, field_map, size):
    down_points = ME.MinkowskiSPMMAverageFunction().apply(
        tensor_map, field_map, size, points
    )
    _, counts = torch.unique(tensor_map, return_counts=True)
    return down_points, counts.unsqueeze_(1).type_as(down_points)


@torch.no_grad()
def stride_centroids(points, counts, rows, cols, size):
    stride_centroids = ME.MinkowskiSPMMFunction().apply(rows, cols, counts, size, points)
    ones = torch.ones(size[1], dtype=points.dtype, device=points.device)
    stride_counts = ME.MinkowskiSPMMFunction().apply(rows, cols, ones, size, counts)
    stride_counts.clamp_(min=1)
    return torch.true_divide(stride_centroids, stride_counts), stride_counts


def downsample_embeddings(embeddings, inverse_map, size, mode="avg"):
    assert len(embeddings) == size[1]
    assert mode in ["avg", "max"]
    if mode == "max":
        in_map = torch.arange(size[1], dtype=inverse_map.dtype, device=inverse_map.device)
        down_embeddings = ME.MinkowskiDirectMaxPoolingFunction().apply(
            in_map, inverse_map, embeddings, size[0]
        )
    else:
        cols = torch.arange(size[1], dtype=inverse_map.dtype, device=inverse_map.device)
        down_embeddings = ME.MinkowskiSPMMAverageFunction().apply(
            inverse_map, cols, size, embeddings
        )
    return down_embeddings


def minkconv_layer(in_dim, out_dim, kernel_size, stride, d, bn_momentum, tr=False):
    kernel = [kernel_size] * d
    if tr:
        conv = getattr(ME, 'MinkowskiConvolutionTranspose')
    else:
        conv = getattr(ME, 'MinkowskiConvolution')
    conv_layer = conv(
        in_channels=in_dim,
        out_channels=out_dim,
        kernel_size=kernel,
        stride=stride,
        dilation=1,
        dimension=d
    )
    return conv_layer


def minkconv_conv_block(in_dim, out_dim, kernel, stride, d, bn_momentum,
                        activation='LeakyReLU',
                        tr=False,
                        expand_coordinates=False,
                        norm_before=False):
    if isinstance(kernel, int):
        kernel = [kernel] * d
    if isinstance(stride, int):
        stride = [stride] * d
    if tr:
        conv = getattr(ME, 'MinkowskiConvolutionTranspose')
    else:
        conv = getattr(ME, 'MinkowskiConvolution')
    conv_layer = conv(
        in_channels=in_dim,
        out_channels=out_dim,
        kernel_size=kernel,
        stride=stride,
        dilation=1,
        dimension=d,
        expand_coordinates=expand_coordinates
    )
    activation_fn = getattr(ME, f'Minkowski{activation}')()
    norm_layer = ME.MinkowskiBatchNorm(out_dim, momentum=bn_momentum)
    if norm_before:
        layer = nn.Sequential(conv_layer, norm_layer, activation_fn)
    else:
        layer = nn.Sequential(conv_layer, activation_fn, norm_layer)
    return layer


def get_conv_block(nc, k=3, d=3, tr=False):
    """
    create sparse convolution block
    :param nc: number of channels in each layer in [in_layer, mid_layer, out_layer]
    :param k: kernel size
    :param tr: transposed convolution
    :return: conv block
    """
    if isinstance(k, int):
        k = [k,] * d
    else:
        assert len(k) == d
    bnm = 0.1
    assert len(nc) == 3
    return nn.Sequential(
            minkconv_conv_block(nc[0], nc[1], k, 2, d, bnm, tr=tr),
            minkconv_conv_block(nc[1], nc[1], k, 1, d, bnm, tr=tr),
            minkconv_conv_block(nc[1], nc[2], k, 1, d, bnm, tr=tr),
        )


def get_conv2d_layers(conv_name, in_channels, out_channels, n_layers, kernel_size, stride,
                    padding, relu_last=True, sequential=True, **kwargs):
    """
    Build convolutional layers. kernel_size, stride and padding should be a list with the
    lengths that match n_layers
    """
    seq = []
    if 'bias' in kwargs:
        bias = kwargs.pop('bias')
    else:
        bias = False
    for i in range(n_layers):
        seq.extend([getattr(nn, conv_name)(
            in_channels, out_channels, kernel_size[i], stride=stride[i],
            padding=padding[i], bias=bias, **{k: v[i] for k, v in kwargs.items()}
        ), nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)])
        if i < n_layers - 1 or relu_last:
            seq.append(nn.ReLU())
        in_channels = out_channels
    if sequential:
        return nn.Sequential(*seq)
    else:
        return seq


def get_norm_layer(channels, norm):
    if norm == 'LN':
        norm_layer = nn.LayerNorm(channels)
    elif norm == 'BN':
        norm_layer = nn.BatchNorm1d(channels)
    else:
        raise NotImplementedError
    return norm_layer

def linear_last(in_channels, mid_channels, out_channels, bias=False, norm='BN'):
    return nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=bias),
            get_norm_layer(mid_channels, norm),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, out_channels)
        )


def linear_layers(in_out, activations=None, norm='BN'):
    if activations is None:
        activations = ['ReLU'] * (len(in_out) - 1)
    elif isinstance(activations, str):
        activations = [activations] * (len(in_out) - 1)
    else:
        assert len(activations) == (len(in_out) - 1)
    layers = []
    for i in range(len(in_out) - 1):
        layers.append(nn.Linear(in_out[i], in_out[i+1], bias=False))
        layers.append(get_norm_layer(in_out[i+1], norm))
        layers.append(getattr(nn, activations[i])())
    return nn.Sequential(*layers)


def meshgrid(xmin, xmax, ymin=None, ymax=None, dim=2, n_steps=None, step=None):
    assert dim <= 3, f'dim <= 3, but dim={dim} is given.'
    if ymin is not None and ymax is not None:
        assert dim == 2
        if n_steps is not None:
            x = torch.linspace(xmin, xmax, n_steps)
            y = torch.linspace(ymin, ymax, n_steps)
        elif step is not None:
            x = torch.arange(xmin, xmax, step)
            y = torch.arange(ymin, ymax, step)
        else:
            raise NotImplementedError
        xs = (x, y)
    else:
        if n_steps is not None:
            x = torch.linspace(xmin, xmax, n_steps)
            if ymin is not None and ymax is not None:
                y = torch.linspace(ymin, ymax, n_steps)
        elif step is not None:
            x = torch.arange(xmin, xmax, step)
        else:
            raise NotImplementedError
        xs = (x, ) * dim
    indexing = 'ijk'
    indexing = indexing[:dim]
    coor = torch.stack(
        torch.meshgrid(*xs, indexing=indexing),
        dim=-1
    )
    return coor


def meshgrid_cross(xmins, xmaxs, n_steps=None, steps=None):
    if n_steps is not None:
        assert len(xmins) == len(n_steps)
        xs = [torch.linspace(xmin, xmax + 1, nstp) for xmin, xmax, nstp \
             in zip(xmins, xmaxs, n_steps)]
    elif steps is not None:
        xs = [torch.arange(xmin, xmax + 1, stp) for xmin, xmax, stp \
             in zip(xmins, xmaxs, steps)]
    else:
        raise NotImplementedError
    dim = len(xs)
    indexing = 'ijk'
    indexing = indexing[:dim]
    coor = torch.stack(
        torch.meshgrid(*xs, indexing=indexing),
        dim=-1
    )
    return coor

def pad_r(tensor, value=0):
    tensor_pad = torch.ones_like(tensor[..., :1]) * value
    return torch.cat([tensor, tensor_pad], dim=-1)


def pad_l(tensor, value=0):
    tensor_pad = torch.ones_like(tensor[..., :1]) * value
    return torch.cat([tensor_pad, tensor], dim=-1)


def metric2indices(coor, voxel_size):
    """"Round towards floor"""
    indices = coor.clone()
    indices[:, 1] = indices[:, 1] / voxel_size[0]
    indices[:, 2] = indices[:, 2] / voxel_size[1]
    return torch.floor(indices).long()


def indices2metric(indices, voxel_size):
    """Voxel indices to voxel center in meter"""
    coor = indices.clone().float()
    coor[:, 1] = (coor[:, 1] + 0.5) * voxel_size[0]
    coor[:, 2] = (coor[:, 2] + 0.5) * voxel_size[1]
    return coor


def fuse_batch_indices(coords, num_cav):
    """
    Fusing voxels of CAVs from the same frame
    :param stensor: ME sparse tensor
    :param num_cav: list of number of CAVs for each frame
    :return: fused coordinates and features of stensor
    """

    for i, c in enumerate(num_cav):
        idx_start = sum(num_cav[:i])
        mask = torch.logical_and(
            coords[:, 0] >= idx_start,
            coords[:, 0] < idx_start + c
        )
        coords[mask, 0] = i

    return coords


def weighted_mahalanobis_dists(reg_evi, reg_var, dists, var0):
    log_probs_list = []
    for i in range(reg_evi.shape[1]):
        vars = reg_var[:, i, :] + var0[i]
        covs = torch.diag_embed(vars.squeeze(), dim1=1)
        unbroadcasted_scale_tril = covs.unsqueeze(1)  # N 1 2 2

        # a.shape = (i, 1, n, n), b = (..., i, j, n),
        M = _batch_mahalanobis(unbroadcasted_scale_tril, dists)  # N M
        log_probs = -0.5 * M
        log_probs_list.append(log_probs)

    log_probs = torch.stack(log_probs_list, dim=-1)
    probs = log_probs.exp()  # N M 2
    cls_evi = reg_evi.view(-1, 1, 2)  # N 1 2
    probs_weighted = probs * cls_evi

    return probs_weighted


def draw_sample_prob(centers, reg, samples, res, distr_r, det_r, batch_size, var0):
    # from utils.vislib import draw_points_boxes_plt
    # vis_ctrs = centers[centers[:, 0]==0, 1:].cpu().numpy()
    # vis_sams = samples[samples[:, 0]==0, 1:].cpu().numpy()
    #
    # ax = draw_points_boxes_plt(50, vis_ctrs, points_c='det_r', return_ax=True)
    # draw_points_boxes_plt(50, vis_sams, points_c='b', ax=ax)
    reg_evi = reg[:, :2]
    reg_var = reg[:, 2:].view(-1, 2, 2)

    grid_size = int(det_r / res) * 2
    centers_map = torch.ones((batch_size, grid_size, grid_size),
                              device=reg.device).long() * -1
    ctridx = metric2indices(centers, res).T
    ctridx[1:] += int(grid_size / 2)
    centers_map[ctridx[0], ctridx[1], ctridx[2]] = torch.arange(ctridx.shape[1],
                                                                device=ctridx.device)

    steps = int(distr_r / res)
    offset = meshgrid(-steps, steps, 2, n_steps=steps * 2 + 1).to(samples.device) # s s 2
    samidx = metric2indices(samples, res).view(-1, 1, 3) \
             + pad_l(offset).view(1, -1, 3)  # n s*s 3
    samidx = samidx.view(-1, 3).T  # 3 n*s*s
    samidx[1:] = (samidx[1:] + (det_r / res))
    mask1 = torch.logical_and((samidx[1:] >= 0).all(dim=0),
                             (samidx[1:] < (det_r / res * 2)).all(dim=0))

    inds = samidx[:, mask1].long()
    ctr_idx_of_sam = centers_map[inds[0], inds[1], inds[2]]
    mask2 = ctr_idx_of_sam >= 0
    ctr_idx_of_sam = ctr_idx_of_sam[mask2]
    ns = offset.shape[0]**2
    new_samples = torch.tile(samples.unsqueeze(1),
                             (1, ns, 1)).view(-1, 3)  # n*s*s 3
    new_centers = centers[ctr_idx_of_sam]
    dists_sam2ctr = new_samples[mask1][mask2][:, 1:] - new_centers[:, 1:]

    probs_weighted = weighted_mahalanobis_dists(
        reg_evi[ctr_idx_of_sam],
        reg_var[ctr_idx_of_sam],
        dists_sam2ctr.unsqueeze(1),
        var0=var0
    ).squeeze()

    sample_evis = torch.zeros_like(samidx[:2].T)
    mask = mask1.clone()
    mask[mask1] = mask2
    sample_evis[mask] = probs_weighted
    sample_evis = sample_evis.view(-1, ns, 2).sum(dim=1)

    return sample_evis

