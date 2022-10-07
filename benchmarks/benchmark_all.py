import json
import os
import subprocess
import sys

import click

dataset_configs = {
    "polybag": [
        dict(
            name="latent",
            base_yaml="configs/latent/latent_polybag.yaml",
            checkpoint="/nfs/andrew/mrcnn/logs/latent_polybag_coco_kl/2022-05-14-21-57-27/model_final.pth",
        ),
        dict(
            name="latent_learned_prior",
            base_yaml="configs/latent/latent_polybag_coco_learned_prior.yaml",
            checkpoint="/nfs/andrew/mrcnn/logs/latent_polybag_coco_learned_prior/2022-05-20-20-40-38/model_final.pth",
        ),
        dict(
            name="mrcnn",
            base_yaml="configs/baselines/mrcnn_polybag.yaml",
            checkpoint="/nfs/nikhil/logs/detectron2/mrcnn_polybag_from_coco/2022-05-14-06-39-44/model_final.pth",
        ),
    ],
    "coco": [
        dict(
            name="latent_learned_prior",
            base_yaml="configs/latent/latent_mrcnn_coco_learned_prior.yaml",
            checkpoint="/nfs/andrew/mrcnn/logs/latent_mrcnn_coco_learned_prior/2022-05-18-19-34-46/model_final.pth",
        ),
        dict(
            name="mrcnn",
            base_yaml="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
            checkpoint="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl",
        ),
    ],
    "cityscapes": [
        dict(
            name="latent_learned_prior",
            base_yaml="configs/latent/latent_cityscapes_coco_learned_prior.yaml",
            checkpoint="/nfs/andrew/mrcnn/logs/latent_cityscapes_coco_learned_prior/2022-05-18-04-59-46/model_final.pth",
        ),
        dict(
            name="latent",
            base_yaml="configs/latent/latent_cityscapes_coco.yaml",
            checkpoint="/nfs/andrew/mrcnn/logs/latent_cityscapes_coco_learned_prior/2022-05-20-20-48-43/model_final.pth",
        ),
        dict(
            name="mrcnn",
            base_yaml="configs/baselines/mrcnn_cityscapes.yaml",
            checkpoint="/nfs/nikhil/logs/detectron2/cityscapes_from_coco/2022-05-09-22-59-59/model_final.pth",
        ),
    ]
}

agree_args = {
    "polybag": ["AGREEMENT.NSAMPLES", "16",
            "AGREEMENT.AUGMENTATIONS", '["none"]',
                      "AGREEMENT.MAX_INSTANCES", "384",],
    "coco": ["AGREEMENT.NSAMPLES", "10",
            "AGREEMENT.AUGMENTATIONS", '["none"]',
                      "AGREEMENT.MAX_INSTANCES", "384",],
    "cityscapes": ["AGREEMENT.NSAMPLES", "8",
            "AGREEMENT.AUGMENTATIONS", '["none"]',
                   "AGREEMENT.DEVICE", "cpu",
                   "AGREEMENT.MAX_INSTANCES", "512"],
}

@click.command()
@click.argument("dataset")
@click.argument("run_name")
@click.option("--recompute", is_flag=True)
@click.option("--agreement", is_flag=True)
def main(dataset, run_name, recompute, agreement):
    benchmark_results = {}
    main_dir = f"/nfs/andrew/mrcnn/benchmark/{dataset}/{run_name}"
    os.makedirs(main_dir, exist_ok=True)
    for model in dataset_configs[dataset]:
        if "latent" in model['name']:
            if agreement:
                configs = [dict(
                    name=f"agree_{i/10:.1f}",
                    args=["AGREEMENT.THRESHOLD", f"{i/10:.1f}",
                          "AGREEMENT.MIN_AREA", "0.1"] + agree_args[dataset],
                ) for i in range(1, 11)]
            else:
                configs = [
                    dict(
                        name="agree_1.0",
                        args=["AGREEMENT.THRESHOLD", "1.0",
                              "AGREEMENT.MIN_AREA", "0.1"] + agree_args[dataset],
                    ),
                    dict(
                        name="agree_0.7",
                        args=["AGREEMENT.THRESHOLD", "0.7",
                              "AGREEMENT.MIN_AREA", "0.1"] + agree_args[dataset],
                    ),
                    dict(
                        name="mean_prior",
                        args=["AGREEMENT.NSAMPLES", "0"],
                    ),
                    dict(
                        name="union_nms",
                        args=["AGREEMENT.THRESHOLD", "0.1",
                              "AGREEMENT.MIN_AREA", "0.1",
                              "AGREEMENT.MODE", "union_nms",
                              "AGREEMENT.NMS_THRESHOLD", "0.3",
                              "AGREEMENT.SCORE_THRESHOLD", "0.01",
                              ] + agree_args[dataset]
                    )
                ]
        else:
            configs = [dict(
                    name="",
                    args=["AGREEMENT.NSAMPLES", "0"],
                )]

        for config in configs:
            variant = f"{model['name']}_{config['name']}"
            print(f"variant {variant}")
            save_dir = os.path.join(main_dir, variant)

            if os.path.exists(save_dir):
                dirs = os.listdir(save_dir)
                if len(dirs) >= 1 and not recompute:
                    inference_path = os.path.join(save_dir, dirs[-1], "inference")
                    benchmark_results[variant] = inference_path
                    print(f"skipping {variant}")
                    continue

            command = "python tools/train_net.py --eval-only --num-gpus 8".split(" ")
            command += ["--config-file", model['base_yaml']]
            command += ["MODEL.WEIGHTS", model['checkpoint']]
            command += ["OUTPUT_DIR", save_dir]
            command += config['args']

            print(" ".join(command))
            subprocess.check_call(command)

            dirs = os.listdir(save_dir)
            if len(dirs) != 1:
                print("not one", dirs)
            inference_path = os.path.join(save_dir, dirs[-1], "inference")
            print(variant, inference_path)

            benchmark_results[variant] = inference_path

            with open(os.path.join(main_dir, "benchmark.json"), 'w') as jfile:
                json.dump(benchmark_results, jfile)

    with open(os.path.join(main_dir, "benchmark.json"), 'w') as jfile:
        json.dump(benchmark_results, jfile)
    print(benchmark_results)


if __name__ == "__main__":
    main()