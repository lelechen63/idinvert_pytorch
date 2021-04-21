

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FaceScape
import argparse

class Code2Params(torch.nn.Module):
    def __init__(self):
        super(Code2Params, self).__init__()

        # input： stylegan code (batch, 14, 512) 
        # output 3DMM params ()
        self.c2p_fc = nn.Sequential(
            nn.Linear(14*512,512),
            nn.LeakyReLU( 0.2, inplace = True ),
            nn.Linear(512,256),
            nn.LeakyReLU( 0.2, inplace = True ),
            nn.Linear(512,128),
            nn.LeakyReLU( 0.2, inplace = True )
        )
    def forward(code):
        code = code.view(code.shape[0], -1)
        params = self.c2p_fc(code)

        return params
  
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--lambda1",
                        type=int,
                        default=100)
    parser.add_argument("--batch_size",
                        type=int,
                        default=2)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=5)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/mnt/ssd0/dat/lchen63/lrw/data/pickle/")
    
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='facescape')
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')

    return parser.parse_args()


def train():
    config = parse_args()
    dataset = FaceScape()
    data_loader = DataLoader(dataset,
                batch_size=config.batch_size,
                num_workers=config.num_thread,
                shuffle=True, drop_last=True)
    for step, data in enumerate(data_loader):
        print (data)
train()