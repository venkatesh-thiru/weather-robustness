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
from dataset_carla import RemapBackground
import torchvision.transforms as tvt
import random

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
overflowtest = False
# torch.manual_seed(42)
# random.seed(42)


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)




def iterate(model, dataloader, criterion, optimizer, mode):
    overall_loss = []
    overall_iou = []
    jaccard_index = torchmetrics.classification.JaccardIndex(task = 'multiclass', num_classes = 23, average = 'weighted').cuda()
    for i, batch in enumerate(tqdm(dataloader)):
        x, y = batch['image'], batch['mask']
        x = x.cuda()
        y = y.long().cuda()
        if model == "validation":
            model.eval()
            with torch.no_grad():
                logit = model(x)
        else:
            optimizer.zero_grad()
            logit = model(x)
        loss = criterion(logit, y)
        if mode == "training":
            loss.backward()
            optimizer.step()
        
        overall_loss.append(loss.item())
        overall_iou.append(jaccard_index(logit, y).item())
        if overflowtest:
            break
    return statistics.mean(overall_loss), statistics.mean(overall_iou)

def test_model(sample, model, ):
    class_mapping, color_mapping, cmap, norm, patches = get_carla_plotting_elements()
    model.eval()
    image, mask = sample['image'].unsqueeze(dim = 0), sample['mask']
    image_inp = tvt.functional.normalize(image, mean = [0.6867, 0.6752, 0.6667], std = [0.2522, 0.2542, 0.2558])
    image_inp = tvt.functional.pad(image_inp, padding = [0,0,0,16])
    with torch.no_grad():
        prediction = model(image_inp.cuda())
    model.train()
    prediction = F.interpolate(prediction, size = image.shape[2:])
    fig,ax = plt.subplots(2,2, figsize = (20,10))
    ax = ax.flatten()

    ax[0].imshow(to_pil_image(image.squeeze().detach().cpu()))
    ax[0].set_aspect('auto')
    ax[1].imshow(mask.squeeze().numpy(), cmap = cmap, norm = norm)
    ax[1].set_aspect('auto')
    ax[2].imshow(prediction.squeeze().max(dim = 0).indices.detach().cpu().numpy(), cmap = cmap, norm = norm)
    ax[2].set_aspect('auto')
    imconf = ax[3].imshow(prediction.squeeze().max(dim = 0).values.detach().cpu().numpy(), cmap = 'jet')
    ax[3].set_aspect('auto')
    ax[1].legend(handles = patches, loc = 2, bbox_to_anchor = (1.05, 1), ncols = 2)
    fig.colorbar(imconf, ax = ax[3])
    return fig

def main():
    checkpoint_path = "checkpoint" #Checkpoint Directory
    batch_size = 3 #Training&Validation batch size
    Epochs = 100
    out_class = 23 #carla semantic segmentation classes
    root_dir = "DATA" #Data root directory
    encoder = "efficientnet-b6" #Encoder
    decoder = "FPN" # decoder
    training_name = f"CARLA_GEN_SEMANTIC_SEGMENTATION_{encoder}(ENCODER)_{decoder}(DECODER)" #Training name

    wandb.init(project="carla", name = training_name, entity="v3nkyc0d3z") #weights and biases logging

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
    train_dataset = carlaDataset(root_dir= root_dir, split_type = "Train", image_transform=train_image_transform, target_transform=train_target_transform)
    train_loader = DataLoader(train_dataset,batch_size=batch_size, num_workers=4)
    
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
    val_dataset = carlaDataset(root_dir=root_dir, split_type = "Validation", image_transform=validation_image_transform, target_transform=validation_target_transform)
    val_loader = DataLoader(val_dataset,batch_size=2, num_workers=4)

    model_parms = {"encoder_name":encoder,
                   "encoder_depth":5,
                   "encoder_weights":"imagenet",
                   "in_channels":3,
                   "classes":out_class,
                   "activation":None}
    
    if decoder == "FPN":
        model = smp.FPN(**model_parms, decoder_dropout=0.3)

    elif decoder == "UNet":
        model = smp.Unet(**model_parms)

    elif decoder == "PSPNet":
        model = smp.PSPNet(**model_parms)

    model = model.cuda()
    
    wandb.watch(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor = 0.75, patience=5, threshold=0.0001, min_lr=1e-6)

    test_image_transform = tvt.Compose([
        tvt.ToTensor(),
        # tvt.Normalize(mean = [0.6823, 0.6725, 0.6617], std = [0.2613, 0.2605, 0.2646]),
        # tvt.Pad(padding = [0,0,0,16])
    ])
    test_target_transform = tvt.Compose([
        # tvt.PILToTensor(),
        tvt.Pad(padding = [0,0,0,16]),
        RemapBackground()
    ])
    test_dataset = carlaDataset(root_dir=root_dir, split_type = "Validation", image_transform = test_image_transform, target_transform = test_target_transform)

    old_validation_loss = 0

    for epoch in range(Epochs):
        model.train()
        train_loss, train_iou = iterate(
            model, train_loader,criterion,optimizer,"training"
        )
        validation_loss, validation_iou = iterate(
            model, val_loader, criterion, optimizer,"validation"
        )
        scheduler.step(validation_loss)


        sample = random.choice(test_dataset)
        fig = test_model(sample, model)

        wandb.log(
         {
            "train loss":train_loss,
            "validation loss":validation_loss,
            "train IoU":train_iou,
            "validation IoU":validation_iou,
            f"test sample epoch {epoch}":fig,
            "learning_rate":optimizer.param_groups[0]['lr']
         }
        )

        if (old_validation_loss == 0) or (old_validation_loss > validation_loss):
            torch.save({
                "epoch": epoch,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                'loss':validation_loss
                        }, os.path.join(checkpoint_path,training_name))
            old_validation_loss = validation_loss
            print("model saved.")


if __name__ == "__main__":
    main()