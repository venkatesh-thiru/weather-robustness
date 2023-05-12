import torch
import os
import glob
import torchvision.transforms as tvt
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from PIL import Image
import glob
import numpy as np
import random
from os.path import join



class RemapBackground():
    """ Remap background to label 19 """

    def __call__(self, mask):
        mask = torch.where(mask > 22, 0, mask)
        mask = torch.where(mask < 0, 0, mask)
        return mask


class carlaDataset(Dataset):
    def __init__(self,
                 root_dir = "DATA",
                 split_type = "Train",
                 target_transform = None,
                 image_transform = None,
                 return_image_id = False,
                 seed_value = 41,
                 albumentation = False
                 ):
        self.root_dir = root_dir
        self.split_type = split_type       
        self.seed_value = seed_value
        
        if self.split_type == "Train":
            self.datadir = os.path.join(self.root_dir, "carla_train")
        elif self.split_type == "Validation":
            self.datadir = os.path.join(self.root_dir, "carla_validation")
        elif self.split_type == "Test":
            self.datadir = self.root_dir

        if image_transform is None:
            self.image_transform = tvt.Compose([
                tvt.ToTensor()
            ])
        else:
            self.image_transform = image_transform

        if not target_transform is None:
            self.target_transform = target_transform
        else:
            self.target_transform = tvt.Compose([
                RemapBackground()
            ])
        self.images = os.listdir(os.path.join(self.datadir, "rgb_clean"))
        self.return_image_id = return_image_id
        self.albumentation = albumentation

        # ROGUE_BUT I MIGHT NEED THIS LATER
        # self.scene_image_pair = []
        # for scene in self.scenarios:
        #     images = [os.path.split(img_name)[-1].split('.')[0] for img_name in glob.glob(os.path.join(datadir,scene,"rgb_clean","*.png"))]

        #     for image in images:
        #         self.scene_image_pair.append((scene, image))
        
        # https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera ##accessed on 23.03.2023
        self.color_mapping = {
            (0, 0, 0):0,	    #Unlabeled	
            (70, 70, 70):1,	    #Building
            (100, 40, 40):2,	#Fence	
            (55, 90, 80):3,	    #Other	
            (220, 20, 60):4,	#Pedestrian	
            (153, 153, 153):5,	#Pole	
            (157, 234, 50):6,	#RoadLine	
            (128, 64, 128):7,	#Road	
            (244, 35, 232):8,	#SideWalk	
            (107, 142, 35):9,	#Vegetation	
            (0, 0, 142):10,	    #Vehicles	
            (102, 102, 156):11,	#Wall	
            (220, 220, 0):12,	#TrafficSign	
            (70, 130, 180):13,	#Sky	
            (81, 0, 81):14,	    #Ground	
            (150, 100, 100):15,	#Bridge	
            (230, 150, 140):16, #RailTrack	
            (180, 165, 180):17,	#GuardRail	
            (250, 170, 30):18,	#TrafficLight	
            (110, 190, 160):19,	#Static	
            (170, 120, 50):20,	#Dynamic	
            (45, 60, 150):21,	#Water	
            (145, 170, 100):22,	#Terrain	
        }
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        image = Image.open(os.path.join(self.datadir, "rgb_clean", f"{image_id}")).convert('RGB')
        mask  = Image.open(os.path.join(self.datadir, "seg", f"{image_id}")).convert('RGB')
        mask = self.mask_to_class(mask)
        torch.manual_seed(self.seed_value)
        if self.albumentation:
            image = np.array(image)
            image = self.image_transform(image=image)
            image = image['image']
        else:
            image = self.image_transform(image)
        if not self.target_transform is None:
            mask = self.target_transform(mask)
        if self.return_image_id:
            return {'image':image, 'mask':mask, 'image_id':str(image_id)}
        else:
            return {'image':image, 'mask':mask}
    
    def mask_to_class(self, mask):
        mask = torch.from_numpy(np.array(mask))
        mask = torch.squeeze(mask)
        class_mask = mask
        class_mask = class_mask.permute(2,0,1).contiguous()
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.empty(h,w,dtype = torch.long)

        for k in self.color_mapping:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))         
            validx = (idx.sum(0) == 3)          
            mask_out[validx] = torch.tensor(self.color_mapping[k], dtype=torch.long)
        return mask_out
    # 

class carla_kp_finetune(Dataset):
    def __init__(
                 self,
                 iter = "iteration_1",
                 target_transform = None,
                 image_transform = None,
                 type = "Training",
                 return_image_id = False,
                 seed_value = 41
                 ) -> None:
        super().__init__()

        self.iterations = [f"iteration-{i}" for i in range(1, 27)]
        self.iter = iter
        self.scene_image_pair = []
        self.type = type
        self.fill_actual()
        self.fill_iter()
        self.return_image_id = return_image_id
        self.seed_value = seed_value
        self.color_mapping = {
            (0, 0, 0):0,	    #Unlabeled	
            (70, 70, 70):1,	    #Building
            (100, 40, 40):2,	#Fence	
            (55, 90, 80):3,	    #Other	
            (220, 20, 60):4,	#Pedestrian	
            (153, 153, 153):5,	#Pole	
            (157, 234, 50):6,	#RoadLine	
            (128, 64, 128):7,	#Road	
            (244, 35, 232):8,	#SideWalk	
            (107, 142, 35):9,	#Vegetation	
            (0, 0, 142):10,	    #Vehicles	
            (102, 102, 156):11,	#Wall	
            (220, 220, 0):12,	#TrafficSign	
            (70, 130, 180):13,	#Sky	
            (81, 0, 81):14,	    #Ground	
            (150, 100, 100):15,	#Bridge	
            (230, 150, 140):16, #RailTrack	
            (180, 165, 180):17,	#GuardRail	
            (250, 170, 30):18,	#TrafficLight	
            (110, 190, 160):19,	#Static	
            (170, 120, 50):20,	#Dynamic	
            (45, 60, 150):21,	#Water	
            (145, 170, 100):22,	#Terrain	
        }
        if image_transform is None:
            self.image_transform = tvt.Compose([
                tvt.ToTensor()
            ])
        else:
            self.image_transform = image_transform

        if not target_transform is None:
            self.target_transform = target_transform
        else:
            self.target_transform = tvt.Compose([
                RemapBackground()
            ])

    def __getitem__(self, index):
        image_pair = self.scene_image_pair[index]
        image = Image.open(os.path.join(image_pair[0], "rgb_clean",f"{image_pair[1]}.png")).convert("RGB")
        mask = Image.open(os.path.join(image_pair[0], "seg",f"{image_pair[1]}.png")).convert("RGB")
        mask = self.mask_to_class(mask)
        torch.manual_seed(self.seed_value)
        image = self.image_transform(image)
        if not self.target_transform is None:
            mask = self.target_transform(mask)
        return {"image":np.array(image), "mask":np.array(mask)}

    def __len__(self):
        return(len(self.scene_image_pair))
    
    def fill_actual(self):
        if self.type == "Training":
            dir = "DATA/carla_train/"
            sample_size = 40
        elif self.type == "Validation":
            dir = "DATA/carla_validation/"
            sample_size = 5
        images = [os.path.split(img_name)[-1].split('.')[0] for img_name in glob.glob(os.path.join(dir,"rgb_clean/*.png"))]
        samples = random.sample(images, k = sample_size)
        for image in samples:
            self.scene_image_pair.append((dir, image))

    def fill_iter(self):
        for it in self.iterations:
            dir = f"DATA/k-projection/{it}"
            images = sorted([os.path.split(img_name)[-1].split('.')[0] for img_name in glob.glob(os.path.join(dir,"rgb_clean/*.png"))])
            if self.type == "Training":
                samples = images[:40]
            elif self.type == "Validation":
                samples = images[40:]
            for image in samples:
                self.scene_image_pair.append((dir, image))
            if it == self.iter:
                break
    
    def mask_to_class(self, mask):
        mask = torch.from_numpy(np.array(mask))
        mask = torch.squeeze(mask)
        class_mask = mask
        class_mask = class_mask.permute(2,0,1).contiguous()
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.empty(h,w,dtype = torch.long)

        for k in self.color_mapping:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))         
            validx = (idx.sum(0) == 3)          
            mask_out[validx] = torch.tensor(self.color_mapping[k], dtype=torch.long)
        return mask_out
    

class carla_detection_data(Dataset):
    def __init__(self, root, transforms=None, return_frame_id = False):
        self.root = root
        self.imroot = join(self.root, "rgb_clean")
        self.maskroot = join(self.root, "seg")
        
        df = pd.read_csv(join(self.root, "bb.csv"), header=None, names=["frame", "label", "x_max", "x_min", "y_max", "y_min"])
        
        # self.imgs = [join(self.imroot, f) for f in listdir(self.imroot)]
        # self.masks = [join(self.maskroot, f) for f in listdir(self.maskroot)]
        
#         self.imgs.sort()
#         self.masks.sort()
        
      
        for col in ("x_min", "y_min", "x_max", "y_max"):
            df = df[df[col] >= 0]
        df = df[df["y_max"] <= 720]
        df = df[df["x_max"] <= 1280]
        
        self.frames = list(df["frame"].unique())

        self.label_mapping = {l: n for n, l in enumerate(df["label"].unique(), 1)}
        self.label_mapping.update({"background": 0})
        
        self.meta_bb = df 
        
        # self.masks = [join(self.imroot, f"{i:06d}semantic_color.jpg") for i in range(107, 500)]
        self.transforms = transforms
        self.return_frame_id = return_frame_id
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index):
        frame = self.frames[index]
        
        path = join(self.imroot, f"{frame:010d}.png")
        img = Image.open(path).convert("RGB")
        # mask = Image.open(self.masks[index]) # .convert("RGB")
        
        bboxes = self.meta_bb[self.meta_bb["frame"] == frame]
        # print(len(bboxes))
        # bboxes = self.meta_bb.loc[["frame", index + self.offset]]
        
        labels = []
        boxes = []
            
        for index, (frame, label, x_max, x_min, y_max, y_min) in bboxes.iterrows():
            labels.append(self.label_mapping[label])
            boxes.append([x_min, y_min, x_max, y_max])

        masks = []
        for n, obj_id in enumerate(labels):
            mask = np.zeros(shape=(img.size[0], img.size[1]))
            xmin, ymin, xmax, ymax = boxes[n]
            mask[xmin:xmax,ymin:ymax] = 1 
            masks.append(mask)

        masks = np.array(masks)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels =  torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            
        # print(labels)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = labels = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.return_frame_id:
            return img, target, f"{frame:010d}"
        else:
            return img, target
    

if __name__ == "__main__":
    image_transform = tvt.Compose([
        tvt.ToTensor()
    ])
    ds = carlaDataset(root_dir="DATA", split_type="Train",image_transform=image_transform)
    print(ds[0])
    for item in ds:
        print(item)
        break
