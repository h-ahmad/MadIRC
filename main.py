# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:03:41 2024

@author: Hussain Ahmad Madni
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
from src.models import load_wide_resnet_50
from src.util import super_seed
from src.util import load_corrupted_data
from src.util import load_test_data
import argparse
from src.util import Realization
import numpy as np
import pandas as pd

def get_args_parser():
    parser = argparse.ArgumentParser('Anomaly Detection Training', add_help=False)
    # parser.add_argument('--data_dir', default='C:\\Users\\Hussain Ahmad Madni\\Desktop\\hussain\\data\\mvtec_anomaly_detection\\', type=str)
    parser.add_argument('--data_dir', default='C:\\Users\\Hussain Ahmad Madni\\Desktop\\hussain\\data\\clinical\\in_use\\', type=str)
    parser.add_argument('--output_path', default='output', type=str)
    parser.add_argument('--corrupted_imgs', default=40, type=int)
    parser.add_argument('--assoc_depth', default=2, type=int) # 10
    parser.add_argument('--seed', default=112358, type=int)
    return parser

def return_nodes():
    nodes = {
        'layer1.0.relu_2': 'Level_1',
        'layer1.1.relu_2': 'Level_2',
        'layer1.2.relu_2': 'Level_3',
        'layer2.0.relu_2': 'Level_4',
        'layer2.1.relu_2': 'Level_5',
        'layer2.2.relu_2': 'Level_6',
        'layer2.3.relu_2': 'Level_7',
        'layer3.1.relu_2': 'Level_8',
        'layer3.2.relu_2': 'Level_9',
        'layer3.3.relu_2': 'Level_10',
        'layer3.4.relu_2': 'Level_11',
        'layer3.5.relu_2': 'Level_12',
        'layer4.0.relu_2': 'Level_13'
        }
    return nodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # class_names = [ 'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill','screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    # class_names = [ 'brain', 'chest', 'histopathology', 'liver', 'retina_oct', 'retina_resc']
    class_names = [ 'brain', 'liver', 'retina_resc']
    return_nodes = return_nodes()
    model = load_wide_resnet_50(return_nodes=return_nodes, verbose=False)
    
    classes, average_pxl, average_img, average_percision, average_recall = [], [], [], [], []
    for class_name in class_names:
        super_seed(args.seed)
        images, masks, corr_types = load_corrupted_data(class_name=class_name, data_dir=args.data_dir, num_corrupted=args.corrupted_imgs) 
        test_images, test_truths, test_class = load_test_data(class_name=class_name, data_dir=args.data_dir) # test_class --> correlation type (hole, rough, scratch etc.)
        # len of: images=249, masks=249, corr_types=83, test_images=83, test_truths=83, test_class=83
        
        # import cv2
        # cv2.imshow('test_truths', test_truths[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('img.png', test_images[0])
    
        test_Realization = Realization(images=images, max_channel_std=5, model=model, assoc_depth = args.assoc_depth, masks=masks, quite=False)
        
        # break
    
        test_results = test_Realization.test(test_images, t_masks=test_truths, quite=False)
        
        average_pxl.append(test_results[0])
        average_img.append(test_results[1])
        average_percision.append(test_results[2])
        average_recall.append(test_results[3])
        classes.append(class_name)
        print(class_name, test_results)
        
    print('averages', (np.average(average_pxl),np.average(average_img),np.average(average_percision),np.average(average_recall)))
    output_data = {'class_name ': classes, 
                   'avg_pixel': average_pxl,
                   'avg_image': average_img,
                   'avg_prcision': average_percision,
                   'avg_recall': average_recall, 
                   'average_pixel': np.average(average_pxl), 
                   'average_img': np.average(average_img), 
                   'average_precision': np.average(average_percision), 
                   'average_recall': np.average(average_recall)}
    df = pd.DataFrame(output_data)
    os.makedirs(args.output_path, exist_ok=True)
    df.to_csv(os.path.join(args.output_path, 'scores.csv'))
