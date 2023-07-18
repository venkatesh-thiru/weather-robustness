import os

import torch
from dataset_carla import carlaDataset, RemapBackground
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
from torchmetrics.classification import JaccardIndex
from PIL import Image
import json

validation_image_transform = tvt.Compose([
    tvt.ToTensor(),
    tvt.Normalize(mean = [0.6823, 0.6725, 0.6617], std = [0.2613, 0.2605, 0.2646]),
    tvt.Pad(padding = [0,0,0,16])
])
validation_target_transform = tvt.Compose([
    # tvt.PILToTensor(),
    tvt.Pad(padding = [0,0,0,16]),
    RemapBackground()
])
out_class = 23
encoder = "efficientnet-b6" #encoder
decoder = "UNet" #decoder
model_parms = {"encoder_name":encoder,
                "encoder_depth":5,
                "encoder_weights":"imagenet",
                "in_channels":3,
                "classes":out_class,
                "activation":None}
checkpoint = "checkpoint/CARLA_GEN_SEMANTIC_SEGMENTATION_efficientnet-b6(ENCODER)_UNet(DECODER)" #checkpoint name
metrics_dir = os.path.join("metrics","semantic",decoder,encoder)
os.makedirs(metrics_dir, exist_ok=True)

test_cases = os.listdir("DATA/carla_test/") #test dir
test_cases = [case for case in test_cases if "model3" in case ]
for case in tqdm(test_cases):
# test_case = "test-fog0-rain0-model0-num200-angle-60"
    dataset = carlaDataset(os.path.join("DATA/carla_test/", case), split_type="Test", image_transform=validation_image_transform, target_transform=validation_target_transform, return_image_id=True)
    loader = DataLoader(dataset, batch_size = 1, num_workers=4)

    if decoder == "FPN":
        model = smp.FPN(**model_parms, decoder_dropout=0.3)
    elif decoder == "UNet":
        model = smp.Unet(**model_parms)
    elif decoder == "PSPNet":
        model = smp.PSPNet(**model_parms)
    model = model.cuda()

    weights = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(weights['model_state_dict'])
    model.eval()

    iou_w = JaccardIndex(num_classes=23, average='weighted').cuda()
    iou_pc = JaccardIndex(num_classes=23, average=None).cuda()

    scene_metrics = {}
    for i, batch in enumerate(tqdm(loader, leave= False)):
        x, y = batch['image'], batch['mask']
        x = x.cuda()
        y = y.long().cuda()
        with torch.no_grad():
            logit = model(x)

        iou_weighted = iou_w(logit, y).item()
        iou_per_class = iou_pc(logit, y).tolist()

        id = batch['image_id'][0].split('.')[0]
        scene_metrics[id]={}
        scene_metrics[id]['per_class'] = iou_per_class
        scene_metrics[id]['weighted'] = iou_weighted

    with open(os.path.join(metrics_dir,f"{case}.json"),'w') as out_file:
        json.dump(scene_metrics, out_file)