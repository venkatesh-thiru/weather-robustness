import os
os.environ['CUDA_VISIBLE_DEVICES'] = "MIG-GPU-8ab9a0c8-909c-3f13-97e6-7376d6d4a029/13/0"

import torch
from dataset_carla import carlaDataset, RemapBackground
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvtf
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
from torchmetrics.classification import JaccardIndex
from PIL import Image
import json
import glob
from noise_augmentations import AddGaussianNoise, SaltAndPepperNoise
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2



out_class = 23
encoder = "efficientnet-b6"
decoder = "PSPNet"
model_parms = {"encoder_name":encoder,
                "encoder_depth":3,
                "encoder_weights":"imagenet",
                "in_channels":3,
                "classes":out_class,
                "activation":None}
checkpoint = "checkpoint/CARLA_GEN_SEMANTIC_SEGMENTATION_efficientnet-b6(ENCODER)_PSPNet(DECODER)"
metrics_dir = os.path.join("metrics","robustness",decoder,encoder)
os.makedirs(metrics_dir,exist_ok=True)


print(f"{encoder}_{decoder}")

augmentations = {
                    "gaussian_blur":(tvt.Compose([    
                                    tvt.ToTensor(),
                                    tvt.Normalize(mean = [0.6823, 0.6725, 0.6617], std = [0.2613, 0.2605, 0.2646]),
                                    tvt.Pad(padding = [0,0,0,16]),
                                    tvt.GaussianBlur(kernel_size=[7,7], sigma = (5, 5))
                        ]),False),
                    "gaussian_noise":(tvt.Compose([
                                    tvt.ToTensor(),
                                    tvt.Normalize(mean = [0.6823, 0.6725, 0.6617], std = [0.2613, 0.2605, 0.2646]),
                                    AddGaussianNoise(mean = 0.3, std = 0.1),
                                    tvt.Pad(padding = [0,0,0,16])
                        ]),False),
                    "pixel_dropout":(A.Compose([
                                    A.PixelDropout(dropout_prob=0.05,per_channel=True,p=1),                              
                                    A.Normalize(mean = [0.6823, 0.6725, 0.6617], std = [0.2613, 0.2605, 0.2646]),
                                    A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32,
                                             border_mode=cv2.BORDER_CONSTANT, value=0, position = A.PadIfNeeded.PositionType.TOP_LEFT),
                                    ToTensorV2(),
                        ]),True),
                    "motion_blur":(A.Compose([
                                    A.MotionBlur(blur_limit=37,p=1),                              
                                    A.Normalize(mean = [0.6823, 0.6725, 0.6617], std = [0.2613, 0.2605, 0.2646]),
                                    A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32,
                                             border_mode=cv2.BORDER_CONSTANT, value=0, position = A.PadIfNeeded.PositionType.TOP_LEFT),
                                    ToTensorV2(),
                        ]),True)

}

validation_target_transform = tvt.Compose([
    # tvt.PILToTensor(),
    tvt.Pad(padding = [0,0,0,16]),
    RemapBackground()
])

test_case = "test-fog0-rain0-model3-num200-angle90"

if decoder == "FPN":
    model = smp.FPN(**model_parms, decoder_dropout=0.3)
elif decoder == "UNet":
    model = smp.Unet(**model_parms)
elif decoder == "PSPNet":
    model = smp.PSPNet(**model_parms)
model = model.cuda()
weights = torch.load(checkpoint, map_location="cpu")
model.load_state_dict(weights['model_state_dict'])
model.eval()



for aug in augmentations.keys():
    print(aug)
    augmentation, albumentation = augmentations[aug]
    dataset = carlaDataset(os.path.join("DATA/carla_test/", test_case), split_type="Test", 
                       image_transform=augmentation, target_transform=validation_target_transform, 
                       return_image_id=True, albumentation=albumentation)
    loader = DataLoader(dataset, batch_size = 1, num_workers=4)

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
    
    with open(os.path.join(metrics_dir,f"{aug}.json"),'w') as out_file:
        json.dump(scene_metrics, out_file)
