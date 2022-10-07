# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from collections import defaultdict

import gc
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from torch.nn import Conv2d

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.evaluation.intersection_evaluation import mask_overlap_stats
from detectron2.modeling.latent import Encoder, Prior
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from pycocotools import mask as pcm

from detectron2.utils.memory import retry_if_cuda_oom, _ignore_torch_cuda_oom
from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from torch.nn import functional as F

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
            cfg,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
            **extra,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.cfg = cfg

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "cfg": cfg,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training


        if self.cfg.AGREEMENT.NSAMPLES == 0:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)

            if detected_instances is None:
                if self.proposal_generator is not None:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                results, _ = self.roi_heads(images, features, proposals, None)
            else:
                detected_instances = [x.to(self.device) for x in detected_instances]
                results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

            if do_postprocess:
                assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
                return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
            else:
                return results

        instances = self.augment(batched_inputs[0], self.cfg.AGREEMENT.AUGMENTATIONS)
        ret = post_process_instances(instances, self.cfg, self.device)
        if ret is None:
            return [dict(instances=instances[0])]
        return ret

    def augment(
        self,
        batched_inputs,
        augmentations,
    ):
        images = batched_inputs['image']
        augmentation_map = dict(
            none=lambda x: x,
            flipv=lambda x: x.flip([1]),
            fliph=lambda x: x.flip([2]),
            fliphv=lambda x: x.flip([1, 2]),
        )

        all_instances = []
        for aug in augmentations:
            aug_fn = augmentation_map[aug]
            rgb = aug_fn(images)
            image_list = self.preprocess_image([dict(image=rgb)])
            aug_rgb = image_list.tensor

            features_ = self.backbone(aug_rgb)

            proposals, _ = self.proposal_generator(image_list, features_, None)
            results, _ = self.roi_heads(image_list, features_, proposals, None)

            instances = GeneralizedRCNN._postprocess(results, [batched_inputs], image_list.image_sizes)

            assert len(instances) == 1, instances

            instances = instances[0]['instances']
            masks = aug_fn(instances.pred_masks)
            boxes = Boxes(masks_to_boxes(masks))

            instances = Instances(instances._image_size, pred_boxes=boxes, pred_masks=masks,
                                 pred_classes=instances.pred_classes, scores=instances.scores)

            all_instances.append(instances)

        return all_instances

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results



@META_ARCH_REGISTRY.register()
class LatentRCNN(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        encoder,
        cfg,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        """
        super().__init__(cfg=cfg, **kwargs)
        self.squash_modules = nn.ModuleList(
            [
                Conv2d(encoder.dim_in + encoder.z_dim,
                       encoder.dim_in, kernel_size=1, stride=1, dilation=1, padding=0)
                for _ in self.backbone.output_shape().keys()
            ]
        )
        self.squash_cat = {k: v for k, v in zip(self.backbone.output_shape().keys(), self.squash_modules)}
        self.encoder = encoder
        self.cfg = cfg
        self.kl_scale = 0.0
        if cfg.MODEL.KL.LEARNED_PRIOR:
            self.prior = Prior(cfg, self.backbone.output_shape(), cfg.MODEL.KL.Z_DIM)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "encoder": Encoder(cfg, backbone.output_shape(), z_dim=cfg.MODEL.KL.Z_DIM),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "cfg": cfg,
        }

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        post = self.encoder(features, gt_instances)

        if self.cfg.MODEL.KL.LEARNED_PRIOR:
            prior = self.prior(features)
        else:
            prior = [torch.zeros((images.tensor.shape[0], self.cfg.MODEL.KL.Z_DIM), device=images.tensor.device)] * 2

        all_kls = []
        all_samples = []
        for idx, (mean, logvar) in enumerate(post):
            pmean = prior[0][idx]
            plogvar = prior[1][idx]
            dist = torch.distributions.normal.Normal(mean, torch.exp(0.5 * logvar))
            sample = dist.rsample()

            var, pvar = torch.exp(logvar), torch.exp(plogvar)
            kls = 0.5 * (plogvar - logvar) + (var + (mean - pmean) ** 2) / (2 * pvar) - 0.5
            kl = kls.sum()
            all_kls.append(kl)

            if np.random.rand() < 0.5:
                all_samples.append(sample)
            else:
                all_samples.append(pmean)

        all_samples = torch.stack(all_samples)
        all_kls = torch.stack(all_kls)
        kl = all_kls.mean()
        losses = {"kl": kl.detach()}
        if self.cfg.MODEL.KL.LOSS:
            if self.cfg.MODEL.KL.CONST_SCALE is not None:
                losses['loss_kl'] = kl * self.cfg.MODEL.KL.CONST_SCALE
            else:
                scale = self.cfg.MODEL.KL.TARGET / kl.detach()
                ema = self.cfg.MODEL.KL.EMA
                self.kl_scale = self.kl_scale * ema + (1 - ema) * scale
                losses['loss_kl'] = kl * self.kl_scale
        else:
            losses['loss_kl'] = kl * 0.0

        features = {
            k: self.squash_cat[k](
                torch.cat([x, all_samples[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])], dim=1)
            )
            for k, x in features.items()
        }

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses


    def sample_augment(
        self,
        batched_inputs,
        nsamp,
        augmentations,
    ):
        images = batched_inputs['image']

        augmentation_map = dict(
            none=lambda x: x,
            flipv=lambda x: x.flip([1]),
            fliph=lambda x: x.flip([2]),
            fliphv=lambda x: x.flip([1, 2]),
        )

        all_instances = []
        for aug in augmentations:
            aug_fn = augmentation_map[aug]
            rgb = aug_fn(images)
            image_list = self.preprocess_image([dict(image=rgb)])
            aug_rgb = image_list.tensor

            features_ = self.backbone(aug_rgb)

            if self.cfg.MODEL.KL.LEARNED_PRIOR:
                mean, logvar = self.prior(features_)
                mean, logvar = mean.squeeze(0), logvar.squeeze(0)
            else:
                mean, logvar = [torch.zeros(self.encoder.z_dim, device=aug_rgb.device)] * 2

            dist = torch.distributions.normal.Normal(mean, torch.exp(0.5 * logvar))

            for _ in range(nsamp):
                sample = dist.rsample()
                aug_features = {
                    k: self.squash_cat[k](
                        torch.cat([x, sample[None, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])], dim=1)
                    )
                    for k, x in features_.items()
                }
                proposals, _ = self.proposal_generator(image_list, aug_features, None)
                results, _ = self.roi_heads(image_list, aug_features, proposals, None)

                instances = GeneralizedRCNN._postprocess(results, [batched_inputs], image_list.image_sizes)

                assert len(instances) == 1, instances

                instances = instances[0]['instances']
                masks = aug_fn(instances.pred_masks)
                boxes = Boxes(masks_to_boxes(masks))

                instances = Instances(instances._image_size, pred_boxes=boxes, pred_masks=masks,
                                     pred_classes=instances.pred_classes, scores=instances.scores)

                all_instances.append(instances)

        return all_instances

    def mean_prior(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.cfg.MODEL.KL.LEARNED_PRIOR:
            mu, logvar = self.prior(features)
        else:
            mu = logvar = torch.zeros([images.tensor.size(0), self.encoder.z_dim], device=images.device)

        z = mu

        features = {
            k: self.squash_cat[k](torch.cat([x, z[..., None, None].repeat(1, 1, *x.shape[-2:])], dim=1))
            for k, x in features.items()
        }
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        assert len(batched_inputs) == 1

        if self.cfg.AGREEMENT.NSAMPLES == 0:
            return self.mean_prior(batched_inputs)

        instances = self.sample_augment(batched_inputs[0], self.cfg.AGREEMENT.NSAMPLES, self.cfg.AGREEMENT.AUGMENTATIONS)
        ret = post_process_instances(instances, self.cfg, self.device)
        if ret is None:
            # print(batched_inputs[0]['file_name'])
            return self.mean_prior(batched_inputs)
        return ret


def post_process_instances(instances, cfg, device):
    if cfg.AGREEMENT.MODE == "concat":
        return [dict(instances=Instances.cat(instances))]

    elif cfg.AGREEMENT.MODE in {"union_nms", "mask_nms"}:
        all_instances = Instances.cat(instances).to(torch.device(cfg.AGREEMENT.DEVICE))
        if len(all_instances) == 0:
            # return early if no objects detected
            return [dict(instances=all_instances)]

        permute = all_instances.scores.argsort(descending=True)

        # Take the top K instances if there are too many
        sorted_instances = all_instances[permute[:cfg.AGREEMENT.MAX_INSTANCES]]

        # Filter ones that don't meet score threshold
        sorted_instances = sorted_instances[sorted_instances.scores > cfg.AGREEMENT.SCORE_THRESHOLD]

        # Filter ones that don't meet area threshold
        _, h, w = sorted_instances.pred_masks.shape
        area = sorted_instances.pred_masks.sum((1, 2)) / (h * w)
        sorted_instances = sorted_instances[area > 1e-5]

        # Compute mask iou
        sorted_masks_float = sorted_instances.pred_masks.float()
        intersection = torch.einsum("ihw,jhw->ij", sorted_masks_float, sorted_masks_float)
        area = sorted_masks_float.sum((1, 2)).clamp(min=1e-6)
        union = area[:, None] + area[None] - intersection
        iou = intersection / union

        # for each mask, the index of the suppressor (itself if we keep it)
        sorted_scores = sorted_instances.scores
        sorted_classes = sorted_instances.pred_classes
        sorted_boxes = sorted_instances.pred_boxes.tensor

        suppressed_by = torch.zeros_like(sorted_scores).long().fill_(-1)
        for i, score in enumerate(sorted_scores):
            # if already suppressed, skip
            if suppressed_by[i] >= 0:
                continue

            within_iou = iou[i, i:] >= cfg.AGREEMENT.NMS_THRESHOLD
            same_class = sorted_classes[i:] == sorted_classes[i]
            suppressed_by[i:][within_iou & same_class] = i

        new_scores, new_classes = [], []
        new_masks, new_boxes = [], []
        for i, suppressor in enumerate(suppressed_by):
            if suppressor == i:
                if cfg.AGREEMENT.MODE == "mask_nms":
                    new_mask = sorted_masks_float[i]

                elif cfg.AGREEMENT.MODE == "union_nms":
                    cluster_masks = sorted_masks_float[suppressed_by == i]
                    new_mask_probs = cluster_masks.mean(0)
                    new_mask = new_mask_probs > cfg.AGREEMENT.THRESHOLD

                else:
                    raise NotImplementedError(cfg.AGREEMENT.MODE)

                area_frac = new_mask.float().sum() / cluster_masks.sum((1, 2)).mean()
                if area_frac < cfg.AGREEMENT.MIN_AREA:
                    continue

                new_masks.append(new_mask)
                new_scores.append(sorted_scores[i])
                new_classes.append(sorted_classes[i])
                new_boxes.append(sorted_boxes[i])

        if len(new_scores) == 0:
            return [dict(instances=all_instances)]

        new_scores = torch.stack(new_scores)
        new_classes = torch.stack(new_classes)
        new_masks = torch.stack(new_masks)
        new_boxes = Boxes(torch.stack(new_boxes))

        new_instances = Instances(
            all_instances.image_size,
            scores=new_scores, pred_masks=new_masks, pred_boxes=new_boxes, pred_classes=new_classes,
        ).to(torch.device('cpu'))
        return [dict(instances=new_instances)]

    elif cfg.AGREEMENT.MODE == "agree":
        instances = [inst[inst.scores > cfg.AGREEMENT.MIN_AREA] for inst in instances]
        all_classes = np.concatenate([x.pred_classes.cpu().numpy() for x in instances])

        unique_classes = set(all_classes.tolist())

        if len(unique_classes) == 0:
            return [dict(samples=instances, instances=instances[0])]

        all_mask_occs = [x.pred_masks for x in instances]
        all_scores = [x.scores for x in instances]
        all_classes = [x.pred_classes for x in instances]

        agreement_pred = None

        # mean_area = torch.stack([x.float().mean((1, 2)).min(0).values for x in all_mask_occs if x.shape[0] > 0]).mean()
        mean_area = torch.stack([x.float().mean() for x in all_mask_occs if x.shape[0] > 0]).mean()

        with _ignore_torch_cuda_oom():
            agreement_pred = agreement(instances[0]._image_size,
                                       all_mask_occs,
                                       all_scores,
                                       all_classes,
                                       cfg.AGREEMENT.THRESHOLD,
                                       cfg.AGREEMENT.MIN_AREA,
                                       torch.float32,
                                       device)
        if agreement_pred is None:
            print([x.shape for x in all_mask_occs])
            gc.collect()
            torch.cuda.empty_cache()
            return None

        agreement_instances = Instances(instances[0]._image_size,
                                        **agreement_pred).to(torch.device('cpu'))

        return [dict(samples=[x.to(torch.device('cpu')) for x in instances],
                     instances=agreement_instances)]
    else:
        raise RuntimeError(cfg.AGREEMENT.MODE)


def agreement(image_size, all_mask_occs, all_scores, all_classes,
               min_agreement, min_obj_area, dtype, device):
    all_mask_occs = [x.type(dtype).to(device) for x in all_mask_occs]
    nobj = max([x.shape[0] for x in all_mask_occs])

    unique_classes = list(set(np.concatenate([x.cpu().numpy() for x in all_classes]).tolist()))
    Nclasses = len(unique_classes)

    zero_res = dict(
        pred_masks=torch.zeros((0, image_size[0], image_size[1]), dtype=torch.bool, device=device),
        pred_boxes=Boxes(torch.zeros((0, 4)).to(device)),
        scores=torch.zeros((0,)).to(device),
    )
    if len([x for x in all_mask_occs if len(x) != 0]) == 0:
        return zero_res


    H, W = [x for x in all_mask_occs if len(x) != 0][0][0].shape
    # keep track of the pixels that already have been assigned to an object
    existing = torch.zeros((Nclasses, H, W)).cuda()
    nmaxpool = 8
    processed_masks = []
    processed_boxes = []
    processed_cls = []

    # pool masks save memory, maxp_masks will contain list of [(nobjs, h * w / (nmaxpool ** 2)) for each sample]
    maxp_masks = [
        F.avg_pool2d(x.type(dtype), nmaxpool).view(x.size(0), -1)
        if len(x) != 0
        else torch.zeros((x.shape[0], ((x.shape[1] - nmaxpool) // nmaxpool + 1) *
                          ((x.shape[2] - nmaxpool) // nmaxpool + 1))).type(dtype).to(device)
        for x in all_mask_occs
    ]

    # pad each sample with 0 masks so they have the same number of objects
    # padded_masks has shape (nsamples, max(nobjs in each sample), h * w / (nmaxpool ** 2))
    padded_masks = torch.stack(
        [torch.cat([x, torch.zeros((nobj - x.shape[0],) + x.shape[1:]).type(dtype).to(device)]) for x in maxp_masks]
    )

    padded_scores = torch.stack(
        [torch.cat([x, torch.zeros((nobj - x.shape[0],)).type(dtype).to(device)]) for x in all_scores])

    padded_classes = torch.stack(
        [torch.from_numpy(np.array([unique_classes.index(int(x)) for x in ar] + [-1] * (nobj - ar.shape[0]))).long().cuda() for ar in all_classes])

    # iteratively assign pixels to objects
    while True:

        # find pairwise intersection of all masks ignoring pixels that have already been assigned
        existing_mp = F.avg_pool2d(existing.type(dtype), nmaxpool).view((Nclasses, -1,))
        padded_masks_min_existing = torch.max(padded_masks - existing_mp[padded_classes], torch.zeros([]).type(dtype).to(device))
        overlap = torch.einsum(
            "ijk,xyk->ijxy",
            [padded_masks_min_existing, padded_masks_min_existing],
        )
        overlap = overlap / (padded_masks.sum(2)[:, :, None, None] + 1e-6)
        # overlap has shape (nsamples, max(nobjs), nsamples, max(nobjs))
        # overlap[i,j,k,h] = (area of intersection mask[i,j] with mask[k, h] ignoring already assigned pixels) /
        #                           (area of mask[i, j])

        # for each mask[i,j] and sample k, find the mask with the best overlap in sample k
        best_overlap, best_overlap_idx = overlap.max(3)

        # for each mask[i,j], greedily pick the samples with the best overlap with mask[i, j].
        # consider only the top k=int(min_agreement * len(all_mask_occs))) samples
        # sorted_idx has shape (nsamples, max(nobjs), k)
        sorted_idx = best_overlap.sort(dim=2, descending=True).indices[
            :, :, : max(1, int(min_agreement * len(all_mask_occs)))
        ]

        # index into the top k masks, best_overlap_masks has shape (nobj, nsamp, k, h * w / (nmaxpool ** 2))
        meshA, meshB, meshC = torch.meshgrid(
            torch.arange(sorted_idx.shape[0]), torch.arange(sorted_idx.shape[1]), torch.arange(sorted_idx.shape[2])
        )
        best_overlap_masks = padded_masks_min_existing[sorted_idx, best_overlap_idx[meshA, meshB, sorted_idx]]

        # for each mask, take intersection of top k masks, ignoring already assigned pixels
        # msks has shape (nobj, nsamp, h * w / (nmaxpool ** 2))
        msks = torch.min(torch.cat([best_overlap_masks, padded_masks_min_existing[:, :, None]], dim=2), dim=2).values

        best_scores = padded_scores[sorted_idx,  best_overlap_idx[meshA, meshB, sorted_idx]]

        # msks has shape (nclasses, nobj, nsamp, h * w / (nmaxpool ** 2))
        # msks = torch.max(
        #     msks - existing_mp[padded_classes], torch.zeros([]).type(dtype).to(device)
        # )
        agreement = msks.sum(dim=-1)

        # agreement = agreement / padded_masks.shape[2] * 
        agreement = (agreement[:, :, None] / (padded_masks[sorted_idx, best_overlap_idx[meshA, meshB, sorted_idx]].sum(3) + 1e-6))
        agreement = (agreement * best_scores).mean(2)

        # print(agreement.max())

        if agreement.max() < min_obj_area:
            break

        # greedily pick the object with the biggest area
        r, c = np.unravel_index(agreement.argmax().cpu().numpy(), agreement.shape)

        # recover the indices of the mask with the most intersection in each sample
        # best_idx has shape (nsamples)
        best_idx = overlap[r, c].argmax(dim=1)

        # find which samples were used in the top k, idx_to_consider has shape (k)
        idx_to_consider = sorted_idx[r, c]

        top_class = defaultdict(float)
        for idx in idx_to_consider:
            score = all_scores[idx][best_idx[idx]]
            cls = int(all_classes[idx][best_idx[idx]])
            top_class[cls] += score

        best_cls = 0
        best_score = 0
        for k, v in top_class.items():
            if v > best_score:
                best_score = v
                best_cls = k
        # best_cls = all_classes[r][c]
        best_cls_ind = unique_classes.index(best_cls)

        # intersect all the full-sized masks that were used in the best intersection for mask[r, c]
        stacked_mask_occ = torch.stack([all_mask_occs[idx][best_idx[idx]].type(dtype) for idx in idx_to_consider] + [1-existing[best_cls_ind]])
        inter = torch.min(stacked_mask_occ, dim=0).values.type(dtype)

        # compute mean iop: (area of intersection of agreement mask & all masks used in agreement) / (area mask used in agreement)
        # higher mean iop means mask is confidently small, lower means mask is small due to intersection of unconfident masks
        # inter_overlap = torch.einsum("ijk,jk->i", [stacked_mask_occ, inter])

        inter_ious = inter.sum() / stacked_mask_occ.sum((1, 2))[:-1]
        inter_scores = torch.stack([all_scores[idx][best_idx[idx]] for idx in idx_to_consider])

        new_score = (inter_scores * inter_ious).mean()

        # fp16 will overflow on einsum of the full sized masks
        if not torch.isfinite(inter_ious).all():
            raise RuntimeError("Agreement resulted in a nan.")

        processed_cls.append(best_cls)

        existing[best_cls_ind] = torch.max(inter, existing[best_cls_ind])

        processed_masks.append(inter)
        # rois = all_rois[idx_to_consider[0]][best_idx[idx_to_consider[0]]]
        # processed_boxes.append(torch.cat([rois, mean_iop[None].float()]))
        # processed_boxes.append(new_score)
        processed_boxes.append(new_score)

    if len(processed_boxes) == 0:
        return zero_res

    rois = torch.stack(processed_boxes)
    processed_masks = torch.stack([m >= 0.5 for m in processed_masks])
    boxes = Boxes(masks_to_boxes(processed_masks))

    mean_iop = rois

    return dict(
        pred_masks=processed_masks,
        pred_boxes=boxes,
        scores=mean_iop,
        pred_classes=torch.tensor(processed_cls).int().to(device),
    )

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Get axis aligned boxes from masks.

    Args
    ----
        masks: tensor, dtype uint8, shape(n, h, w)

    Returns
    -------
        boxes: tensor, dtype float32, shape(n, 4) == (n, [w_min, h_min, w_max, h_max])
    """
    boxes = [torch.zeros((0, 4), device=masks.device, dtype=torch.int64)]
    for m in masks:
        if not bool(m.sum()):  # if mask is all zeros
            boxes.append(torch.zeros((1, 4), device=masks.device, dtype=torch.int64))
        else:
            nnz = m.nonzero()
            boxes.append(torch.stack([nnz[:, 1].min(), nnz[:, 0].min(), nnz[:, 1].max(), nnz[:, 0].max()]).unsqueeze(0))
    boxes = torch.cat(boxes, dim=0).float()
    return boxes

@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
