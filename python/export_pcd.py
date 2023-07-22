import glob
import os
import os.path as osp
import sys

import torch.utils.data as data

import inten
import otils as ot
import torchutils as tu

def convert(data):
    xyzi = data[...,1:5]
    
    # intensity = data[...,4:5]


if __name__ == '__main__':
    pass