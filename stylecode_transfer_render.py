

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FaceScape
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
  


def train():
    dataset = FaceScape()
    