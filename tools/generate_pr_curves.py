import json

import click
from matplotlib import pyplot as plt
import os.path as osp
import numpy as np
from collections import defaultdict

import pandas as pd

from pycocotools.coco import COCO

from detectron2.data.catalog import MetadataCatalog
from detectron2.evaluation.fast_eval_api import COCOeval_opt

# from pycocotools.cocoeval import COCOeval as COCOeval_opt
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.coco import convert_to_coco_json, convert_to_coco_dict


def _select_thr(thr_lst, thr, atol=1e-8):
    err = np.abs(np.subtract(thr_lst, thr))
    idx = err.argmin()
    assert err[idx] < atol
    return idx

def _mAP(self, iouThr=None, areaRng='all', maxDets=100 ):
    p = self.params
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    # dimension of precision: [TxRxKxAxM]
    s = self.eval['precision']
    # IoU
    if iouThr is not None:
        t = _select_thr(p.iouThrs, iouThr)
        s = s[t]
    s = s[:,:,:,aind,mind]
    if len(s[s>-1])==0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s>-1])
    return mean_s

def _AR(self, iouThr=None, areaRng='all', maxDets=100):
    p = self.params
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    # dimension of recall: [TxKxAxM]
    s = self.eval['recall']
    s = s[:,:,aind,mind]
    if iouThr is not None:
        t = _select_thr(p.iouThrs, iouThr)
        s = s[t]
    if len(s[s>-1])==0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s>-1])
    return mean_s


def _max_recall_at_high_precision(self, pThrs, iouThrs, areaRng="all", maxDets=100):
    p = self.params
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    precision = self.eval['precision'][..., aind, mind]
    recall = p.recThrs
    out = np.zeros([len(pThrs), len(iouThrs), len(p.catIds)])
    for pi, pThr in enumerate(pThrs):
        for ii, iouThr in enumerate(iouThrs):
            t = _select_thr(p.iouThrs, iouThr)
            for ci in range(len(p.catIds)):
                m = (precision[t, :, ci] > pThr).squeeze()
                if m.any():
                    out[pi, ii, ci] = p.recThrs[m].max()

    return out.mean()


predictions = {
    "coco_2017_val": {
      'latent_0.7_fixed': '/nfs/andrew/mrcnn/benchmark/coco/20220118_fix/latent_learned_prior_agree_0.7/2022-01-19-06-40-55/inference',
        **json.load(open('/nfs/andrew/mrcnn/benchmark/coco/20220122_fix/benchmark.json', 'r')),
        **{k+"_new":v for k,v in json.load(open('/nfs/andrew/mrcnn/benchmark/coco/20220118_fix2/benchmark.json', 'r')).items()}
    },
    "polybag_val": {
        **json.load(open('/nfs/andrew/mrcnn/benchmark/polybag/20210521_repro4/benchmark.json', 'r')),
        **{k+"_new":v for k,v in json.load(open('/nfs/andrew/mrcnn/benchmark/polybag/20220119_fix/benchmark.json')).items()},
        },
    # "polybag_val": json.load(open('/nfs/andrew/mrcnn/benchmark/polybag/20220116/benchmark.json', 'r')),
    "cityscapes_in_coco_format_val":
        {
            'new_latent_0.7': '/nfs/andrew/mrcnn/detectron2/benchmarks/results/3_city_latent_agreement/2021-05-25-04-39-18/inference/',
            'new_latent_0.7_minarea': '/nfs/andrew/mrcnn/detectron2/benchmarks/results/3_city_latent_agreement_0.7/2021-05-25-16-39-00/inference/',
            **json.load(open('/nfs/andrew/mrcnn/benchmark/cityscapes/20220122_fix/benchmark.json', 'r')),
            **{k+"_new":v for k,v in json.load(open('/nfs/andrew/mrcnn/benchmark/cityscapes/20220118_fix2/benchmark.json', 'r')).items()},
        }
}


@click.command()
@click.option("-d", "--dset", type=str)
@click.option("-o", "--output-dir", type=str, default="/nfs/benchmark_vis/detectron2/pr_curves")
@click.option("-f", "--fmt", type=str, default="png")
@click.option("-i", "--interpolate", type=bool, is_flag=True)
def main(dset: str, output_dir: str, fmt: str, interpolate: bool):
    gts_file = osp.join(output_dir, dset, "annotations.json")
    convert_to_coco_json(dset, gts_file, allow_cached=False)
    coco_gt = COCO(gts_file)

    metrics = ["IoU", "IoP", "IoG"]
    metricThrs = [0.5, 0.75, 0.9, 0.95]
    nrow, ncol = len(metricThrs) // 2 + 1, 2

    table = defaultdict(dict)
    for metric in metrics:
        fig = plt.figure(figsize=(6 * ncol, 4 * nrow))
        axes = [plt.subplot(nrow, ncol, i + 1) for i in range(len(metricThrs))]
        models = predictions[dset]
        colors = {m: plt.cm.jet(x) for m, x in zip(models, np.linspace(0, 1, len(models)))}
        styles = {m: ['-', '--', '-.', ':'][i%4] for i, m in enumerate(models)}

        for model, dets_file in predictions[dset].items():
            coco_dt = coco_gt.loadRes(
                osp.join(dets_file, "coco_instances_results.json")
            )
            coco_eval = COCOeval_opt(coco_gt, coco_dt, iouType="segm", metric=metric)

            coco_eval.params.iouThrs = metricThrs
            coco_eval.params.recThrs = np.arange(10000) * 0.0001
            coco_eval.params.maxDets = [100]
            coco_eval.params.areaRng = [[0, 1e8]]
            coco_eval.params.interpolateCurve = interpolate

            coco_eval.evaluate()
            coco_eval.accumulate()

            recall = coco_eval.params.recThrs
            for i, metricThr in enumerate(metricThrs):
                precision = coco_eval.eval["precision"][i].squeeze((-2, -1)).mean(-1)
                m = precision > 1e-6
                axes[i].plot(recall[m], precision[m], label=model, color=colors[model], linestyle=styles[model])
                axes[i].set_title(f"PR @ {metric}={metricThr:.2f}")

    
            segm_eval = COCOeval_opt(coco_gt, coco_dt, iouType="segm", metric=metric)
            segm_eval.evaluate()
            segm_eval.accumulate()

            if metric == "IoU":
                # box_eval = COCOeval_opt(coco_gt, coco_dt, iouType="bbox", metric=metric)
                # box_eval.evaluate()
                # box_eval.accumulate()
             
                table[model].update({
                    f"mAP (mask, {metric}=0.5:0.95)": _mAP(segm_eval),
                    # f"mAP (box, {metric}=0.5:0.95)": _mAP(box_eval),
                })

            elif metric == "IoP":
                table[model].update({
                    f"MR @ HP (P=0.75:0.95,{metric}=0.75:0.95)": _max_recall_at_high_precision(
                        segm_eval,
                        pThrs=[0.75, 0.8, 0.85, 0.9, 0.95], 
                        iouThrs=[0.75, 0.8, 0.85, 0.9, 0.95],
                    )
                }) 

            elif metric == "IoG":
                table[model].update({
                    f"AR (mask, {metric}==0.5:0.95)": _AR(segm_eval),
                })

            else:
                raise NotImplementedError(metric)

        plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.15), ncol=5)
        plt.savefig(osp.join(output_dir, dset, f"pr_{metric}.{fmt}"), bbox_inches="tight")
        plt.close(fig)

    df = pd.DataFrame(table).T
    for col in df.columns:
        print(df.sort_values(col).to_string())

    cols = list(df.columns)
    df = df[cols] * 100
    for name, r in df.iterrows():
        print(name, " & ", "  &  ".join([f"{x:.1f}" for x in r]), "\\\\")


if __name__ == "__main__":
    main()
