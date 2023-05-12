import json
import numpy as np
from matplotlib import colors
import matplotlib.patches as mpatches


def get_rs_plotting_elements():
    rsconf_file = "DATA/RAIL_SEM19/rs19-config.json"
    f = open(rsconf_file, 'r')
    conf = json.load(f)
    class_mapping = {}
    color_mapping = {}

    for n, obj in enumerate(conf['labels']):
        class_mapping[obj['index']] = obj["name"].replace("-", "_")
        class_mapping[19] = "background"
        color_mapping[obj['index']] = np.array(obj['color']) / 255
        color_mapping[19] = np.array([0, 0, 0])

    cmap = colors.ListedColormap([color_mapping[i] for i in range(0,20)])
    bounds = [i for i in range(0,20)]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    patches = [ mpatches.Patch(color=cmap.colors[i], label=class_mapping[i] ) for i in range(20) ]

    return class_mapping, color_mapping, cmap, norm, patches


def get_rs_plotting_elements_new_annot():
    class_mapping = {
                        0:["road"],
                        1:["sidewalk"],
                        2:['construction', 'fence'],
                        3:['rail_raised','rail_embedded'],
                        4:['pole','traffic_light','traffic_sign'],
                        5:['sky'],
                        6:['human'],
                        7:['tram_track', 'rail_track'],
                        8:['car','truck'],
                        9:['on_rails'],
                        10:['vegetation'],
                        11:['trackbed'],
                        12:['background','terrain']
                    }
    color_mapping = {
                        0:np.array([47,79,79])/255,
                        1:np.array([1,0,1]),
                        2:np.array([160,82,45])/255,
                        3:np.array([1,1,0]),
                        4:np.array([255, 235, 205])/255,
                        5:np.array([0,1,1]),
                        6:np.array([1,0,0]),
                        7:np.array([70,130,180])/255,
                        8:np.array([176,48,96])/255,
                        9:np.array([225,228,225])/255,
                        10:np.array([0,1,0]),
                        11:np.array([47,79,79])/255,
                        12:np.array([0,0,0])
                    }
    
    cmap = colors.ListedColormap([color_mapping[i] for i in range(len(class_mapping.keys()))])
    bounds = [i for i in range(0,len(class_mapping.keys()))]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    patches = [ mpatches.Patch(color=cmap.colors[i], label=str(class_mapping[i])) for i in range(len(class_mapping.keys())) ]

    return class_mapping, color_mapping, cmap, norm, patches

def get_carla_plotting_elements():

    class_mapping = {
                        0:"Unlabled",
                        1:"Building",
                        2:'Fence',
                        3:'Other',
                        4:'Pedestrian',
                        5:'Pole',
                        6:'Roadline',
                        7:'Road',
                        8:'Sidewalk',
                        9:'Vegetation',
                        10:'Vehicles',
                        11:'Wall',
                        12:'Traffic Sign',
                        13:'sky',
                        14:'Ground',
                        15:'Bridge',
                        16:'RailTrack',
                        17:'GuardRail',
                        18:'TrafficLight',
                        19:'Static',
                        20:'Dynamic',
                        21:'Water',
                        22:'Terrain'
                    }
    color_mapping = {
                     0: np.array([0, 0, 0])/255, 
                     1: np.array([70, 70, 70])/255, 
                     2: np.array([100,  40,  40])/255, 
                     3: np.array([55, 90, 80])/255, 
                     4: np.array([220,  20,  60])/255, 
                     5: np.array([153, 153, 153])/255, 
                     6: np.array([157, 234,  50])/255, 
                     7: np.array([128,  64, 128])/255, 
                     8: np.array([244,  35, 232])/255, 
                     9: np.array([107, 142,  35])/255, 
                     10: np.array([  0,   0, 142])/255, 
                     11: np.array([102, 102, 156])/255, 
                     12: np.array([220, 220,   0])/255, 
                     13: np.array([ 70, 130, 180])/255, 
                     14: np.array([81,  0, 81])/255, 
                     15: np.array([150, 100, 100])/255, 
                     16: np.array([230, 150, 140])/255, 
                     17: np.array([180, 165, 180])/255, 
                     18: np.array([250, 170,  30])/255, 
                     19: np.array([110, 190, 160])/255, 
                     20: np.array([170, 120,  50])/255, 
                     21: np.array([ 45,  60, 150])/255, 
                     22: np.array([145, 170, 100])/255
                     }
    
    cmap = colors.ListedColormap([color_mapping[i] for i in range(len(class_mapping.keys()))])
    bounds = [i for i in range(0,len(class_mapping.keys()))]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    patches = [ mpatches.Patch(color=cmap.colors[i], label=str(class_mapping[i])) for i in range(len(class_mapping.keys())) ]

    return class_mapping, color_mapping, cmap, norm, patches
