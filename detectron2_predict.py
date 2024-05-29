import torch
import os
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
#import numpy as np

def detectron_init(model='invitro', model_dir='/nemo/lab/briscoej/home/shared/tiagu-models/micropatterns/'):
    if model=='invitro':
        model_dir=model_dir #'/nemo/lab/briscoej/home/shared/tiagu-models/micropatterns/'
        config_fn='Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml'
    elif model=='SC_sections':
        model_dir=model_dir #'/nemo/lab/briscoej/home/shared/tiagu-models/SC_sections/'
        config_fn='COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
    else:
        return("Error: specify model to use by the predictor.")
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_fn))
    cfg.OUTPUT_DIR = model_dir
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_fn)  # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    predictor = DefaultPredictor(cfg)
    return predictor
    

def detectron_predict(img, predictor): 
    img = img.img
    outputs = predictor(img)
    return outputs["instances"].to("cpu")


if __name__ == '__main__':
    main()