import numpy as np
import PIL
import torch

class SaltAndPepperNoise(object):
    r""" Implements 'Salt-and-Pepper' noise
    Adding grain (salt and pepper) noise
    (https://en.wikipedia.org/wiki/Salt-and-pepper_noise)

    assumption: high values = white, low values = black
    
    Inputs:
            - threshold (float):
            - imgType (str): {"cv2","PIL"}
            - lowerValue (int): value for "pepper"
            - upperValue (int): value for "salt"
            - noiseType (str): {"SnP", "RGB"}
    Output:
            - image ({np.ndarray, PIL.Image}): image with 
                                               noise added
    """
    def __init__(self,
                 treshold:float = 0.005,
                 imgType:str = "cv2",
                 lowerValue:int = 5,
                 upperValue:int = 250,
                 noiseType:str = "SnP"):
        self.treshold = treshold
        self.imgType = imgType
        self.lowerValue = lowerValue # 255 would be too high
        self.upperValue = upperValue # 0 would be too low
        if (noiseType != "RGB") and (noiseType != "SnP"):
            raise Exception("'noiseType' not of value {'SnP', 'RGB'}")
        else:
            self.noiseType = noiseType
        super(SaltAndPepperNoise).__init__()

    def __call__(self, img):
        if self.imgType == "PIL":
            img = np.array(img)
        if type(img) != np.ndarray:
            raise TypeError("Image is not of type 'np.ndarray'!")
        
        if self.noiseType == "SnP":
            random_matrix = np.random.rand(img.shape[0],img.shape[1])
            img[random_matrix>=(1-self.treshold)] = self.upperValue
            img[random_matrix<=self.treshold] = self.lowerValue
        elif self.noiseType == "RGB":
            random_matrix = np.random.random(img.shape)      
            img[random_matrix>=(1-self.treshold)] = self.upperValue
            img[random_matrix<=self.treshold] = self.lowerValue
        
        

        if self.imgType == "cv2":
            return img
        elif self.imgType == "PIL":
            # return as PIL image for torchvision transforms compliance
            return PIL.Image.fromarray(img)
        


class ProgressiveSprinkles(object):
    r""" Implements progressive sprinkles
    
    Wright, L. (2019): Progressive sprinkles - a new data augmentation
    for CNNs (and helps to achieve new 98+% NIH Malaria dataset accuracy).
    (https://medium.com/@lessw/progressive-sprinkles-a-new-data-augmentation-for-cnns-and-helps-achieve-new-98-nih-malaria-6056965f671a)

    Additionally, it allows to use patches of Gaussian noise (https://www.simonwenkel.com/2019/11/15/progressive-sprinkles.html).
    
    Inputs:
            - threshold (float):
            - imgType (str): {"cv2","PIL"}
            - lowerValue (int): value for "pepper"
            - upperValue (int): value for "salt"
            - noiseType (str): {"SnP", "RGB"}
    Output:
            - image ({np.ndarray, PIL.Image}): image with 
                                               noise added
    """
    def __init__(self,
                 imgType:str = "cv2",
                 lowerValue:int = 0,
                 upperValue:int = 255,
                 noiseType:str = "BW",
                 imgFraction:float = 0.1,
                 patchCount:int = 11):
        self.imgType = imgType
        self.lowerValue = lowerValue
        self.upperValue = upperValue 
        if (noiseType != "BW") and (noiseType != "Gauss"):
            raise Exception("'noiseType' not of value {'BW', 'Gauss'}")
        else:
            self.noiseType = noiseType
        self.patchCount = patchCount
        self.imgFraction = imgFraction
        super(ProgressiveSprinkles, self).__init__()

    def __call__(self, img):
        if self.imgType == "PIL":
            img = np.array(img)
        if type(img) != np.ndarray:
            raise TypeError("Image is not of type 'np.ndarray'!")
        
        # get valid dimensions for patches
        x_min = 0
        x_max = img.shape[1] * (1-self.imgFraction)
        x_range = np.arange(x_min, x_max, 1, dtype=np.int)
        y_min = 0
        y_max = img.shape[0] * (1-self.imgFraction)
        y_range = np.arange(y_min, y_max, 1, dtype=np.int)
        
        if self.noiseType == "BW":
            colors = np.array([self.lowerValue, self.upperValue], dtype=np.uint8)
            for i in range(self.patchCount):
                patch = np.zeros_like(img, dtype=np.int)
                patch_y_min = np.random.choice(y_range)
                patch_y_max = int(patch_y_min + np.random.rand() * (img.shape[0] * self.imgFraction - 2))
                patch_x_min = np.random.choice(x_range)
                patch_x_max = int(patch_x_min + np.random.rand() * (img.shape[1] * self.imgFraction - 2))
                patch[patch_y_min:patch_y_max, patch_x_min:patch_x_max] = 1
                img[patch == 1] = np.random.choice(colors)
                
        elif self.noiseType == "Gauss":
            colors = np.array([self.lowerValue, self.upperValue], dtype=np.uint8)
            for i in range(self.patchCount):
                patch = np.zeros_like(img, dtype=np.int)
                patch_y_min = np.random.choice(y_range)
                patch_y_max = int(patch_y_min + np.random.rand() * (img.shape[0] * self.imgFraction - 2))
                patch_x_min = np.random.choice(x_range)
                patch_x_max = int(patch_x_min + np.random.rand() * (img.shape[1] * self.imgFraction - 2))
                patch[patch_y_min:patch_y_max, patch_x_min:patch_x_max] = 1
                img[patch == 1] = np.array(self.upperValue*np.random.random(img[patch == 1].shape), dtype=np.int)
     

        if self.imgType == "cv2":
            return img
        elif self.imgType == "PIL":
            # return as PIL image for torchvision transforms compliance
            return PIL.Image.fromarray(img)
        
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
