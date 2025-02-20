import os 
import torch

import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import librosa
import time
import copy

def get_lists():
    dataset_dir ='/raid/celong/FaceScape/'
    train_list = []
    for pid in os.listdir(os.path.join(dataset_dir , "ffhq_aligned_img") ):
        print (pid)
        for exp_id in os.listdir(os.path.join(dataset_dir , "ffhq_aligned_img" , pid) ):
            print (exp_id)
            for  i in range(56):
                img_p = os.path.join(dataset_dir , "ffhq_aligned_img" , pid, exp_id, '%d.jpg'%i)
                if os.path.exists(img_p):
                    print (img_p)
                    if os.path.exists(os.path.join(dataset_dir , "ffhq_aligned_img" , pid, exp_id, '1.npy')):

                        train_list.append(os.path.join( pid, exp_id, '%d.jpg'%i))
                    
    with open(os.path.join(dataset_dir,'lists/train.pkl'), 'wb') as handle:
        pickle.dump(train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
get_lists()

class FaceScape(data.Dataset):
    def __init__(self,
                dataset_dir ='/raid/celong/FaceScape/', mode = 'train'):
        self.mode = mode
        if self.mode=='train':
            _file = open(os.path.join(dataset_dir, "lists/train.pkl"), "rb")
        elif self.mode =='test':
            _file = open(os.path.join(dataset_dir, "lists/test.pkl"), "rb")
        elif self.mode =='demo' :
            _file = open(os.path.join(dataset_dir, "lists/demo.pkl"), "rb")
        self.data_list = pickle.load(_file)
        _file.close()
        self.dataset_dir = dataset_dir

    
    def __getitem__(self, index):
        if self.mode=='train':
            print (self.data_list[index])
            tmp = self.data_list[index].split('/')
            p_id = tmp[0]
            exp_id = tmp[1]
            view_id = tmp[2]
            img_path = os.path.join(self.dataset_dir , "ffhq_aligned_img", self.data_list[index])
            code_path = os.path.join(self.dataset_dir , "ffhq_aligned_img", p_id, exp_id, '1.npy')

            img = cv2.imread(img_path)
            print (img.shape)
            code = np.load(code_path)
            print (code.shape)
        return img_path
         
    def __len__(self):
       
        return len(self.data_list)
       
