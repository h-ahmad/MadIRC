# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:14:21 2024

@author: Hussain Ahmad Madni
"""

import torch
import numpy as np
import random
import cv2
import os
from typing import List
import faiss 
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
import tqdm
from src.FeatureDescriptors import Feautre_Descriptor
import time
from PIL import Image

def super_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    
def load_corrupted_data(class_name: str, 
                        data_dir: str, 
                        num_corrupted: int, 
                        size: tuple = (256,256), 
                        crop_size: tuple = (224,224)): # crop_size: tuple = (224,224)
    
    assert class_name in os.listdir(data_dir)
    train_images = []
    class_dir = data_dir + class_name + '/train/good/'
    x = int(size[0]/2- crop_size[0]/2)
    y = int(size[1]/2- crop_size[1]/2)
    for filename in os.listdir(class_dir):
        train_images.append(cv2.resize(cv2.imread(class_dir+filename),size)[x:x+crop_size[0],y:y+crop_size[1]])
        
    test_images, test_masks, corr_types = load_test_data(class_name, data_dir, size=size, crop_size=crop_size)
    ziped = list(zip(test_images, test_masks, corr_types))
    random.shuffle(ziped)
    test_images, test_masks, corr_types = zip(*ziped)
    test_images = list(test_images)
    test_masks = list(test_masks)
    corr_types = list(corr_types)
    return train_images + test_images[:num_corrupted], [np.zeros_like(test_masks[0]) for x in range(len(train_images))] + test_masks[:num_corrupted], corr_types    

def load_test_data(class_name: str, 
                   data_dir: str, 
                   size: tuple = (256,256), 
                   # crop_size: tuple = (240,240)): # crop_size: tuple = (224,224)
                   crop_size: tuple = (224,224)): # crop_size: tuple = (224,224)
    img_dir = data_dir + class_name + '/test/'
    ann_dir = data_dir + class_name + '/ground_truth/' # mask directory
    test_images, test_masks, corr_types = [], [], []
    x = int(size[0]/2- crop_size[0]/2)
    y = int(size[1]/2- crop_size[1]/2)
    for directory in os.listdir(img_dir):
        for filename in os.listdir(img_dir + directory + '/'):
            test_images.append(cv2.resize(cv2.imread(img_dir+directory+'/'+filename),size)[x:x+crop_size[0],y:y+crop_size[1]])
            if directory != 'good':
                # test_masks.append(cv2.resize(cv2.imread(ann_dir+directory+'/'+filename[:-4]+'_mask.png'),size)[x:x+crop_size[0],y:y+crop_size[1]])
                test_masks.append(cv2.resize(cv2.imread(ann_dir+directory+'/'+filename[:-4]+'.png'),size)[x:x+crop_size[0],y:y+crop_size[1]])
            else:
                test_masks.append(np.zeros_like(test_images[-1]))
            corr_types.append(directory)
    return test_images, test_masks, corr_types

def align_images(seed, images, masks, quite=True):
    c_x = seed.shape[0]//2
    c_y = seed.shape[1]//2
    image_size = (seed.shape[0],seed.shape[1]) # (width, height)
    r_mat = [ cv2.getRotationMatrix2D((c_x,c_y), x,  1.0) for x in range(360)]
    proposed_data_corrupted_images = []
    proposed_used_test_masks        = []
    for k, image in enumerate(tqdm.tqdm(images, ncols=100, desc = 'Rotating', disable=quite)):
        rotation_ideal = []
        test_img = seed.astype(np.float16)[c_x-(c_x//2):c_x+(c_x//2), c_y-(c_y//2):c_y+(c_y//2)]
        for x in range(0,360):
            candidate = cv2.warpAffine(image, r_mat[x], image_size).astype(np.float16)[c_x-(c_x//2):c_x+(c_x//2), c_y-(c_y//2):c_y+(c_y//2)]
            rotation_ideal.append(np.mean(np.square(test_img-candidate )))
        proposed_data_corrupted_images.append(cv2.warpAffine(image, r_mat[np.argmin(rotation_ideal)], image_size))
        if not masks is None:
            masks_rounded = cv2.warpAffine(masks[k], r_mat[np.argmin(rotation_ideal)], image_size)
            masks_rounded[masks_rounded>128]  = 255
            masks_rounded[masks_rounded<=128] = 0
        else:
            masks_rounded = None
        proposed_used_test_masks.append(masks_rounded)
    
    return proposed_data_corrupted_images, proposed_used_test_masks

def test_for_positional_class_transpose(imgs):
    average = np.mean(np.array(imgs),axis=0,keepdims=False) # (224, 224, 3)
    #average_f = np.flip(np.flip(average, (0)), (1))
    average_f = np.transpose(average, (1,0,2)) # (224, 224, 3)
    return np.mean(np.square(average.astype(np.float16)-average_f.astype(np.float16))) # 123.25

def positional_test_and_alignment(images: List[np.ndarray], threashold: float, masks: List[np.ndarray]=None, align: bool = True, quite: bool = True):
    if test_for_positional_class_transpose(images) < threashold: # 123.25 < 1000
        if align: # Speedup trick here just becasue this is deterministic for classes
            proposed_data_corrupted_images, proposed_used_test_masks =  align_images(images[0], images, masks, quite=quite)
            if test_for_positional_class_transpose(proposed_data_corrupted_images) < threashold:
                return False, images, masks, False
            else:
                return True, proposed_data_corrupted_images, proposed_used_test_masks, True  
        return False, images, masks, False 
    return True, images, masks, False

def measure_distances(features_a, features_b):
    distances = torch.cdist(torch.permute(features_a,[1,0]),torch.permute(features_b,[1,0]))
    return distances

class Realization():
    def __init__(self, 
                 images: List[np.ndarray], 
                 model : torch.nn.Module,
                 assoc_depth: int = 10,
                 min_channel_length: int = 3,
                 max_channel_std: float = 5.0,
                 masks: List[np.ndarray] = None, 
                 quite: bool = False,
                 pos_embed_thresh: float = 1000,
                 pos_embed_weight: float = 5.0,
                 filter_size: float = 13,
                 **kwargs) -> None:
        
        self.quite = quite
        self.images = images
        self.masks = masks
        self.image_size = tuple(images[0].shape)
        self.model = model
        self.assoc_depth = assoc_depth
        self.filter_size = filter_size
        self.min_channel_length = min_channel_length
        self.max_channel_std = max_channel_std

        # Do positional embedding/alignment tests
        self.pos_embed_flag, self.images, self.masks, self.aligment_flag = positional_test_and_alignment(images, 
                                                                                                         threashold=pos_embed_thresh, 
                                                                                                         masks=self.masks, 
                                                                                                         align=False,
                                                                                                         quite=self.quite)
        self.pos_embed_weight = pos_embed_weight if self.pos_embed_flag else 0.

        # Do feature Extraction 
        self.fd_gen = Feautre_Descriptor(model=model, image_size=self.image_size, positional_embeddings = self.pos_embed_weight,   **kwargs)        
        self.patches = self.fd_gen.generate_descriptors(self.images,quite=self.quite)  
        self.cpu_patches = self.patches.cpu().numpy()
        # print('cpu_patches shape: ', self.cpu_patches.shape) # (249, 1024, 3136)
        
        # If given masks track precision
        if not self.masks is None:
            self.patch_shape = (int(np.sqrt(self.cpu_patches.shape[2])),int(np.sqrt(self.cpu_patches.shape[2])))
            self.scale = masks[0].shape[0]//self.patch_shape[0] # This assumes square images...
            self.tp = 0
            self.fp = 0
            self.negatives = (np.count_nonzero(self.masks)//3)//self.scale
            self.positives = self.cpu_patches.shape[2]*self.cpu_patches.shape[0] - self.negatives
            self.max_label = np.max(np.array(self.masks))

        # Create Channels  
        self.gen_channels(self.quite)

    def gen_assoc(self, targets: torch.Tensor, 
                         sources: torch.Tensor, 
                         target_img_index: int, 
                         source_img_indexs: int):
        t_len = targets.size()[1]
        s_len = sources.size()[1]
        sources_zero_axis_min   = torch.from_numpy(np.ones(shape=(t_len))*np.inf).cuda()
        sources_zero_axis_index = torch.from_numpy(np.zeros(shape=(t_len))).cuda()
        targets_ones_axis_min   = torch.from_numpy(np.ones(shape=(s_len))*np.inf).cuda()
        targets_ones_axis_index = torch.from_numpy(np.zeros(shape=(s_len))).cuda()

        # Handle not having enough GPU memory to do everything in one big batch.
        aval_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        max_side = int(np.floor(np.sqrt(aval_mem//32)))
        for x in range(int(np.ceil(s_len/max_side))):
            for y in range(int(np.ceil(t_len/max_side))):

                distances = measure_distances(sources[:,x*max_side:min([(x+1)*max_side,s_len])],
                    targets[:,y*max_side:min([(y+1)*max_side,t_len])])

                mins, args = (torch.min(distances,axis=0))
                sources_zero_axis_index[y*max_side:min([(y+1)*max_side,t_len])] = torch.where(
                    sources_zero_axis_min[y*max_side:min([(y+1)*max_side,t_len])] >= mins,
                    args + x*max_side,
                    sources_zero_axis_index[y*max_side:min([(y+1)*max_side,t_len])] 
                )
                sources_zero_axis_min[y*max_side:min([(y+1)*max_side,t_len])] = torch.minimum(
                    sources_zero_axis_min[y*max_side:min([(y+1)*max_side,t_len])],
                    mins
                )

                mins, args = (torch.min(distances,axis=1))
                targets_ones_axis_index[x*max_side:min([(x+1)*max_side,s_len])] = torch.where(
                    targets_ones_axis_min[x*max_side:min([(x+1)*max_side,s_len])] >= mins,
                    args + y*max_side,
                    targets_ones_axis_index[x*max_side:min([(x+1)*max_side,s_len])]
                )
                targets_ones_axis_min[x*max_side:min([(x+1)*max_side,s_len])] = torch.minimum(
                    targets_ones_axis_min[x*max_side:min([(x+1)*max_side,s_len])],
                    mins
                )
        
        sources_indexs = sources_zero_axis_index.cpu().numpy().astype(int)
        targets_indexs = targets_ones_axis_index.cpu().numpy().astype(int)

        # Doing this on torch should speed this up
        assoc = np.ones((targets_indexs.shape[0],5))*np.inf
        for x in range(targets_indexs.shape[0]):
            if sources_indexs[targets_indexs[x]] == x:
                assoc[x] = [x,targets_indexs[x],targets_ones_axis_min[x].cpu().numpy(), target_img_index, source_img_indexs]
            else:
                assoc[x] = [np.inf,np.inf,targets_ones_axis_min[x].cpu().numpy(),np.inf,np.inf]

        return assoc

    def get_precision_recall(self):
        if not self.masks is None:
            return self.tp/(self.tp+self.fp), self.tp/self.positives
        else:
            return -1, -1

    def precision_recall(self, patches: List[list]):
        if not self.masks is None:
            for x in range(len(patches)):
                index = np.unravel_index(patches[x][2], shape=self.patch_shape)
                if np.average(self.masks[patches[x][1]][
                    index[0]*self.scale:(index[0]+1)*self.scale,
                    index[1]*self.scale:(index[1]+1)*self.scale,:]) == 0 : self.tp += 1
                else: self.fp += 1

    def gen_channels(self, quite: bool = False):
        # Collect assoc 
        assoc = np.ones((self.assoc_depth, self.patches.size(0), self.patches.size(2), 5))*np.inf
        # print('assoc shape: ', assoc.shape) # (10, 249, 3136, 5)
        for seed_index in tqdm.tqdm(range(self.assoc_depth), ncols=100, desc = 'Associate To Channels', disable=quite):  
            gpu_seeds = self.patches[seed_index].cuda()
            for compare_index in range(seed_index+1,self.patches.size(0)):
                assoc[seed_index,compare_index] = self.gen_assoc(gpu_seeds, self.patches[compare_index].cuda(), seed_index, compare_index)

        # Ensure each patch only associates to it's best candidate seed patch
        # print('assoc shape after comparison: ', assoc.shape) # <class 'numpy.ndarray'>, (10, 249, 3136, 5)
        assoc = np.take_along_axis(assoc,np.expand_dims(assoc[:,:,:,2],axis=3).argmin(axis=0)[None],axis=0)[0]
        assoc = np.resize(assoc, (assoc.shape[0]*assoc.shape[1],assoc.shape[2]))
        # assoc -> [all_patches, [seed_p_index, img_p_index, distance, seed_image_index, img_image_index]]
        # print('assoc after match: ', assoc.shape) # (780864, 5)

        # Create Channels
        channels = {}
        for p_index in tqdm.tqdm(range(assoc.shape[0]), ncols=100, desc = 'Create Channels', disable=quite):
            if assoc[p_index,0] < np.inf:
                channel_name = str(int(assoc[p_index,0]))+'_'+str(int(assoc[p_index,3]))
                # print('channel_name: ', channel_name) # 3134_8, 3135_8, ....
                if channel_name in channels.keys():
                    channels[channel_name].append([self.cpu_patches[int(assoc[p_index,4]),:,int(assoc[p_index,1])], int(assoc[p_index,4]), int(assoc[p_index,1])])
                else:
                    channels[channel_name] = [[self.cpu_patches[int(assoc[p_index,3]),:,int(assoc[p_index,0])], int(assoc[p_index,3]), int(assoc[p_index,0])]]
                    channels[channel_name].append([self.cpu_patches[int(assoc[p_index,4]),:,int(assoc[p_index,1])], int(assoc[p_index,4]), int(assoc[p_index,1])])
                                                 #[         patch embedding                                       ,    img_image_index,    img_p_index]


        self.nn_object = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), self.patches.size(1), faiss.GpuIndexFlatConfig()) # deterministic brute force nn
        # self.nn_object = faiss.IndexFlatL2(faiss.StandardGpuResources(), self.patches.size(1), faiss.GpuIndexFlatConfig()) # deterministic brute force nn

        # Filter Channels 
        nominal_points = [] 
        for channel_name in tqdm.tqdm(list(channels.keys()), ncols=100, desc = 'Filter Channels', disable=quite):  
            if len(channels[channel_name])>self.min_channel_length:
                c_patches = [patch[0] for patch in channels[channel_name]]
                mean = np.mean(np.array(c_patches),axis=0)
                std = np.std(np.sqrt(np.sum(np.square(np.array(c_patches)-mean),axis=1)),axis=0) # Note we use spherical standard deviation
                new_centers = [center for center in channels[channel_name] if np.sqrt(np.sum(np.square(mean-center[0]))) < self.max_channel_std*std]
                c_patches = [patch[0] for patch in new_centers]
                if len(new_centers)>self.min_channel_length:
                    channels[channel_name] = new_centers
                    self.precision_recall(new_centers)
                    nominal_points += c_patches
                else:
                    del channels[channel_name]
            else:
                del channels[channel_name]
        self.nn_object.add(torch.from_numpy(np.array(nominal_points)))

    def predict(self, t_images: List[np.ndarray], 
                t_masks: List[np.ndarray] = None, 
                quite: bool = False): 
        
        #temp
        # for i, img in enumerate(t_images):
        #     img_before_algn = Image.fromarray(img)
        #     out_path = os.path.join('output/not_aligned')
        #     img_before_algn.save(os.path.join(out_path, str(i)+'.png'))
        
        if self.aligment_flag:
            t_images, t_masks = align_images(self.images[0], t_images, t_masks)

        start = time.time()
        # print('t_images shape: ', t_images[0].shape) # (224, 224, 3)
        t_patches =  self.fd_gen.generate_descriptors(t_images, quite=quite)        
        # print('t_patches shape: ', t_patches.shape) #  torch.Size([83, 1024, 3136]) # 83 images
        
        scores = []
        for test_img_index in tqdm.tqdm(range(t_patches.size(0)), ncols=100, desc = 'Predicting On Images', disable=quite):
            dist, ind = self.nn_object.search(torch.permute(t_patches[test_img_index],(1,0)),1)
            dist = np.resize(dist[:,0], new_shape=(int(np.sqrt(dist.shape[0])),int(np.sqrt(dist.shape[0]))))
            dist = dist.repeat(t_images[0].shape[0]//dist.shape[0], axis=0).repeat(t_images[0].shape[0]//dist.shape[0], axis=1)
            # print('dist shape: ', dist.shape) # (224, 224)
            scores.append(gaussian_filter(dist,self.filter_size))
        print('TIME TO COMPLETE all predictions', abs(start-time.time()))  
        return scores, t_masks
         
    def test(self,  t_images: List[np.ndarray], 
                    t_masks: List[np.ndarray] = None, 
                    quite: bool = False):
        
        scores, t_masks = self.predict(t_images, t_masks=t_masks, quite=quite)
        # print('len of scores: ', len(scores), ' and shape is: ', scores[0].shape) # (224, 224)
        # print('len of t_masks: ', len(t_masks), ' and shape is: ', t_masks[0].shape) # (224, 224, 3)
        t_masks = [(mask[:,:,0]/255.).astype(int) for mask in t_masks]

        img_scores = [np.max(score) for score in scores] 
        img_masks  = [np.max(mask) for mask in t_masks]
        # print('len of img_scores: ', len(img_scores), ' and shape is: ', img_scores[0].shape) # len: 83, shape: ()
        # print('len of img_masks: ', len(img_masks), ' and shape is: ', img_masks[0].shape) # len: 83, shape: ()
        img_auroc = roc_auc_score(img_masks, img_scores)

        scores  = np.array(scores).flatten()
        t_masks = np.array(t_masks).flatten()        
        # print('scores shape: ', scores.shape) # (4164608,)   ==> 83 * 224 * 224
        # print('t_masks shape: ', t_masks.shape) # (4164608,)  ==> 83 * 224 * 224
        pxl_auroc = roc_auc_score(t_masks, scores)
        
        precision, recall = self.get_precision_recall()

        return pxl_auroc, img_auroc, precision, recall
