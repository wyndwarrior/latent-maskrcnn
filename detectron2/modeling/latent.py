import torch
import torch.nn as nn


import torch.nn.functional as F
import detectron2.modeling.torch_utils as tu2
from detectron2.layers import roi_align
from detectron2.modeling.poolers import ROIPooler


class Encoder(nn.Module):
    def __init__(self, cfg, input_shape, dim_node=512, n_layers=8, key_dim=16, val_dim=64, z_dim=64):
        super().__init__()
        self.cfg = cfg
        # self.spatial_strides = spatial_strides
        self.mask_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        dim_in = input_shape[self.mask_in_features[0]].channels

        self.dim_in = dim_in
        self.patch_size = s_conv = 64
        s_conv_small = s_conv // 8

        self.n_layers = n_layers
        d_conv, d_box = 64, 64
        self.dim_out = dim_node

        NORM2 = dict(method="group", num_per_group=4, eps=1e-5, affine=False)

        if self.cfg.MODEL.KL.POSITION:
            dim_in += 2

        self.conv = tu2.SpecialSequential(
            tu2.ConvBlock(3 + dim_in, [d_conv // 2], mode="gated_resnet", norm=NORM2, activate_final=True),
            nn.MaxPool2d(2),
            tu2.ConvBlock(d_conv // 2, [d_conv, d_conv], mode="gated_resnet", norm=NORM2, activate_final=True),
            nn.MaxPool2d(2),
            tu2.ConvBlock(d_conv, [d_conv], mode="gated_resnet", norm=NORM2, activate_final=True),
            nn.MaxPool2d(2),
            tu2.FlattenTrailingDimensions(1),
            tu2.MLP(d_conv * s_conv_small * s_conv_small, [dim_node], activate_final=True),
        )

        self.attend = nn.ModuleList(
            [
                tu2.SpecialSequential(
                    tu2.BufferAttend1d(dim_node, key_dim, val_dim),
                    nn.Linear(val_dim, dim_node),
                    tu2.Nonlinearity("leaky_relu"),
                )
                for _ in range(n_layers)
            ]
        )
        self.node_update = nn.ModuleList(
            [tu2.MLP(dim_node, [dim_node, 2 * dim_node], activate_final=True) for _ in range(n_layers)]
        )

        self.z_dim = z_dim
        self.z_pred = nn.Linear(dim_node, z_dim * 2)

        self.pos_channel = nn.Parameter(
            torch.stack(torch.meshgrid(torch.linspace(-1, 1, s_conv), torch.linspace(-1, 1, s_conv)), dim=0)[None],
            requires_grad=False,
        )

        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = self.patch_size#cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        self.mask_pooler = ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )


        tu2.initialize(self)

    def forward(self, features, gt_instances):

        features = [features[f] for f in self.mask_in_features]

        if self.cfg.MODEL.KL.POSITION:
            features = [torch.cat([
                x,
                torch.stack(torch.meshgrid(torch.linspace(-1, 1, x.shape[2]), torch.linspace(-1, 1, x.shape[3])), dim=0)[None].expand(x.shape[0], -1, -1, -1).to(x.device)
                ], dim=1)
                for x in features]

        boxes = [x.gt_boxes for x in gt_instances]
        roi_feats = self.mask_pooler(features, boxes)

        mask_side_len = self.patch_size

        latent_dists = []
        instances_so_far = 0
        for instances_per_image in gt_instances:
            if len(instances_per_image) == 0:
                latent_dists.append([torch.zeros(self.z_dim, device=roi_feats.device)] * 2)
            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.gt_boxes.tensor, mask_side_len
            ).to(device=roi_feats.device)

            image_roi_feats = roi_feats[instances_so_far:instances_so_far + gt_masks_per_image.shape[0]]

            im_feats = torch.cat([image_roi_feats, self.pos_channel.repeat(image_roi_feats.size(0), 1, 1, 1), gt_masks_per_image.unsqueeze(1)],
                                 dim=1)
            node_feats = self.conv(im_feats)

            for i in range(self.n_layers):
                read = self.attend[i](node_feats)
                x = self.node_update[i](node_feats + read)
                x1, x2 = torch.split(x, self.dim_out, dim=1)
                node_feats = torch.addcmul(x1, 1.0, node_feats, torch.sigmoid(x2))

            z_pred = self.z_pred(node_feats).mean(dim=0)
            latents = torch.split(z_pred, self.z_dim, dim=0)

            latent_dists.append(latents)

            instances_so_far += gt_masks_per_image.shape[0]

        return latent_dists

class Prior(nn.Module):
    def __init__(self, cfg, input_shape, z_dim):
        super().__init__()
        self.cfg = cfg
        # self.spatial_strides = spatial_strides
        self.mask_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        dim_in = input_shape[self.mask_in_features[0]].channels
        d_conv = 64
        self.z_dim = z_dim
        NORM2 = dict(method="group", num_per_group=4, eps=1e-5, affine=False)

        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    tu2.ConvBlock(dim_in, [d_conv], mode="resnet", norm=NORM2, activate_final=True),
                    nn.MaxPool2d(2),
                    tu2.ConvBlock(d_conv, [z_dim * 2], mode="resnet", norm=NORM2, activate_final=False),
                )
                for _ in input_shape.keys()
            ]
        )
        self.conv_map = {k: v for k, v in zip(input_shape.keys(), self.conv)}

        tu2.initialize(self)

    def forward(self, conv_blobs):
        im_feats = sum([self.conv_map[i](conv_blobs[i]).mean(dim=(2, 3)) for i in conv_blobs])
        return im_feats.chunk(2, dim=1)


