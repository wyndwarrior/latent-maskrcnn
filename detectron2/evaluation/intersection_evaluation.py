import os
import pickle
from collections import defaultdict
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
import numpy as np
import torch.nn.functional as F

from detectron2.structures import PolygonMasks, BitMasks


def to_np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x

    elif torch.is_tensor(x):
        return x.detach().cpu().numpy()

    else:
        return np.asarray(x)


def compute_metrics(mask_occ, gt_mask_occ, pred_scores, metrics=("iou", "iop", "iog")):
    stats = mask_overlap_stats(mask_occ, gt_mask_occ)

    return_dict = defaultdict(list)
    if "iou" in metrics:
        iou = to_np(stats["iou"])
        pred_idx, gt_idx = linear_sum_assignment(1 - iou)
        m = iou[pred_idx, gt_idx] > 1e-2
        pred_idx, gt_idx = pred_idx[m], gt_idx[m]

        for pi, gi in zip(pred_idx, gt_idx):
            return_dict["iou"].append(
                iou[pi, gi]
            )
            return_dict["iou_scores"].append(
                pred_scores[pi].cpu().numpy()
            )

    if "iop" in metrics:
        pred_idx, gt_idx = (
            np.arange(len(mask_occ)),
            to_np(stats["iou"].argmax(1)),
        )
        m = to_np(stats["iou"][pred_idx, gt_idx]) > 1e-2
        pred_idx, gt_idx = pred_idx[m], gt_idx[m]

        for pi, gi in zip(pred_idx, gt_idx):
            return_dict["iop"].append(
                float(stats["intersection"][pi, gi] / stats["area1"][pi])
            )
            return_dict["iop_scores"].append(
                pred_scores[pi].cpu().numpy()
            )

    if "iog" in metrics:
        pred_idx, gt_idx = (
            to_np(stats["iou"].argmax(0)),
            np.arange(len(gt_mask_occ)),
        )
        m = to_np(stats["iou"][pred_idx, gt_idx]) > 1e-2
        pred_idx, gt_idx = pred_idx[m], gt_idx[m]

        for pi, gi in zip(pred_idx, gt_idx):
            return_dict["iog"].append(
                float(stats["intersection"][pi, gi] / stats["area2"][gi])
            )
            return_dict["iog_scores"].append(
                pred_scores[pi].cpu().numpy()
            )

    return return_dict

def mask_overlap_stats(first: torch.Tensor, second: torch.Tensor):
    """Compute masks overlap statistics.

    Inputs
    ------
        first: shape(n, h, w)
        second: shape(m, h, w)

    Returns
    -------
        A dict[str, Tensor] with the following keys (all dtypes are float32):
        * iou: shape(n,m)
        * area1: shape(n)
        * area2: shape(m)
        * intersection: shape(n,m)
    """
    first, second = first.float(), second.float()
    intersection = torch.einsum("ihw,jhw->ij", first, second)
    first_area, second_area = first.sum((1, 2)), second.sum((1, 2))

    union = first_area[:, None] + second_area[None] - intersection
    return dict(iou=intersection / union, area1=first_area, area2=second_area, intersection=intersection)

class IntersectionEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, output_folder):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self.output_folder = output_folder
        self._stats = defaultdict(list)


    def process(self, inputs, outputs):

        assert len(inputs) == 1
        inputs = inputs[0]

        assert len(outputs) == 1
        outputs = outputs[0]

        if len(inputs['instances']) == 0 or len(outputs['instances']) == 0:
            return

        pred_masks = outputs['instances'].pred_masks
        pred_scores = outputs['instances'].scores
        gt_masks = inputs['instances'].gt_masks

        if isinstance(gt_masks, PolygonMasks):
            gt_masks = BitMasks.from_polygon_masks(gt_masks, width=inputs['image'].shape[2], height=inputs['image'].shape[1])

        gt_masks = gt_masks.tensor.to(pred_masks.device)

        pred_masks = F.interpolate(
            pred_masks[None].float(), size=gt_masks.shape[1:], mode="bilinear", align_corners=False
        )[0]

        metrics = compute_metrics(pred_masks, gt_masks, pred_scores)

        for k, v in metrics.items():
            self._stats[k].extend(v)

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self._stats = defaultdict(list)

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """

        for k, v in self._stats.items():
            print(f"mean {k}: {np.mean(v)}")

        os.makedirs(self.output_folder, exist_ok=True)

        with open(os.path.join(self.output_folder, "intersection.pkl"), 'wb') as pfile:
            pickle.dump(self._stats, pfile)
