import json

from detectron2.data import DatasetCatalog, MetadataCatalog


def register_covariant(name, json_file, thing_classes):
    DatasetCatalog.register(name, lambda: json.load(open(json_file)))
    MetadataCatalog.get(name).set(thing_classes=thing_classes, evaluator_type="coco")
