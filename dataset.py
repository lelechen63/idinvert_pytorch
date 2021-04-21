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
    for pid in os.listdir(os.path.join(dataset_dir , "ffhq_aligned_img") ):
        print (pid)
        for exp_id in os.listdir(os.path.join(dataset_dir , "ffhq_aligned_img" , pid) ):
            print (exp_id)
            for  i in range(56):
                img_p = os.listdir(os.path.join(dataset_dir , "ffhq_aligned_img" , pid, exp_id, '%d.jpg'.format(i))
                if os.path.exists(img_p):
                    print (img_p)
get_lists()

class FaceScape(data.Dataset):
    def __init__(self,
                dataset_dir ='/raid/celong/FaceScape/', mode = 'train'):
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
        if self.train=='train':
            
            img_path = os.path.join(self.dataset_dir , "ffhq_aligned_img", self.data_list[index]) 

         
    def __len__(self):
       
        return len(self.data_list)
       
