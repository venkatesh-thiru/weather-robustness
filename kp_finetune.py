import os

import random
import statistics
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader,Dataset, Subset
import matplotlib.pyplot as plt
import torchvision.transforms as tvt
import numpy as np
from torch.cuda.amp import autocast,GradScaler
import json
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import wandb
from dataset_carla import carlaDataset
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from plot_utils import get_carla_plotting_elements
from dataset_carla import carla_kp_finetune,RemapBackground, carlaDataset
import torchvision.transforms as tvt
from torchmetrics.classification import JaccardIndex
import random

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
overflowtest = False

train_image_transform = tvt.Compose([
    tvt.ToTensor(),
    tvt.Normalize(mean = [0.6867, 0.6752, 0.6667], std = [0.2522, 0.2542, 0.2558]),
    tvt.Pad(padding = [0,0,0,16])
])
train_target_transform = tvt.Compose([
    # tvt.PILToTensor(),
    tvt.Pad(padding = [0,0,0,16]),
    RemapBackground()
])

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

def iterate(model, dataloader, criterion, optimizer, mode):
    overall_loss = []
    overall_iou = []
    jaccard_index = torchmetrics.classification.JaccardIndex(task = 'multiclass', num_classes = 23, average = 'weighted').cuda()
    for i, batch in enumerate(tqdm(dataloader, leave=False)):
        x, y = batch['image'], batch['mask']
        x = x.cuda()
        y = y.long().cuda()
        if model == "validation":
            with torch.no_grad():
                logit = model(x)
        else:
            optimizer.zero_grad()
            logit = model(x)
        loss = criterion(logit, y)
        if mode == "training":
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred = logit.argmax(dim = 1)
        
        overall_loss.append(loss.item())
        overall_iou.append(jaccard_index(logit, y).item())
        if overflowtest:
            break
    return statistics.mean(overall_loss), statistics.mean(overall_iou)

# def fine_tune_iteration(model, model_kind, ):

def main():
    encoder = "efficientnet-b6"
    decoder = "FPN"
    checkpoint_dir = f"checkpoint/kp_finetune/{encoder}"#CHANGEHERE
    name = f"CARLA_GEN_SEMANTIC_SEGMENTATION_efficientnet-b6(ENCODER)_{decoder}(DECODER)"#CHANGEHERE
    model_file = f"checkpoint/CARLA_GEN_SEMANTIC_SEGMENTATION_{encoder}(ENCODER)_{decoder}(DECODER)"#CHANGEHERE
    weights = torch.load(model_file, map_location="cpu")
    iterations = [f"iteration-{i}" for i in range(1,27)]

    batch_size = 3

    for iteration in iterations:
        print(f"FINETUNING FOR ITERATION - {iteration}...............................................")
        EPOCHS = 20
        iter_name = f"{name}_{iteration}"
        checkpoint_path = os.path.join(checkpoint_dir, f"{iter_name}.pth")
        wandb.init(project="carla_kp_finetune", name = iter_name, entity="v3nkyc0d3z")
        
        model = smp.FPN(
            encoder_name="efficientnet-b6", #CHANGE HERE
            encoder_depth=5,
            encoder_weights="imagenet",
            decoder_dropout=0.3,
            in_channels=3,
            classes=23,
            activation=None
        ).cuda()
        model.load_state_dict(weights['model_state_dict'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        wandb.watch(model)
        
        train_dataset = carla_kp_finetune(iter=iteration, target_transform=train_target_transform, image_transform=train_image_transform, type="Training")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)

        validation_dataset = carla_kp_finetune(iter=iteration, target_transform=validation_target_transform, image_transform=validation_image_transform, type = "Validation")
        validation_loader = DataLoader(validation_dataset, batch_size=2, num_workers=4)

        old_validation_loss = 0
        for epoch in range(EPOCHS):
            train_loss, train_iou = iterate(model, train_loader,criterion,optimizer,"training")
            validation_loss, validation_iou = iterate(model, validation_loader, criterion, optimizer,"validation")

            wandb.log(
                        {
                            "train loss":train_loss,
                            "validation loss":validation_loss,
                            "train IoU":train_iou,
                            "validation IoU":validation_iou,
                        }
                    )
            if (old_validation_loss == 0) or (old_validation_loss > validation_loss):
                torch.save({
                    "epoch":epoch,
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "loss":validation_loss
                }, checkpoint_path)
                old_validation_loss = validation_loss
                print(f"model saved at EPOCH {epoch}")
        test_iteration(checkpoint_path, iteration)
        wandb.finish()

def test_iteration(checkpoint, iteration, encoder, decoder):
    print(f"Testing {iteration}")
    iteration_metric_path = os.path.join("metrics", "kp_finetune", encoder, iteration)#CHANGEHERE
    os.makedirs(iteration_metric_path, exist_ok=True)
    weights = torch.load(checkpoint, map_location = "cpu")
    model = smp.FPN(
            encoder_name=encoder, #CHANGE HERE
            encoder_depth=5,
            encoder_weights="imagenet",
            decoder_dropout=0.3,
            in_channels=3,
            classes=23,
            activation=None
        ).cuda()
    model.load_state_dict(weights['model_state_dict'])
    test_cases = os.listdir("DATA/carla_test")
    test_cases = [case for case in test_cases if "model3" in case]


    for case in tqdm(test_cases):
        dataset = carlaDataset(os.path.join("DATA/carla_test/", case), split_type="Test", image_transform=validation_image_transform, target_transform=validation_target_transform, return_image_id=True)
        loader = DataLoader(dataset, batch_size = 1, num_workers=8)

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


            
        out_file_path = os.path.join(iteration_metric_path, f"{case}.json")
        with open(out_file_path, "w") as out_file:
            json.dump(scene_metrics, out_file)


if __name__ == "__main__":
    main()

