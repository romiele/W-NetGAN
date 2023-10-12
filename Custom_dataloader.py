# -*- coding: utf-8 -*-
"""
Created on Fri May  5 12:07:19 2023
    Custom dataloader for Norne
@author: Roberto.Miele
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob

class FaciesSeismicDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, nsim, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.lendata = os.listdir(root_dir+'/Facies_TI')
        self.transform = transform
        self.nsim= nsim
        

    def __len__(self):
        return len(self.lendata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        fac_filename= self.root_dir+'Facies_TI/'+f'{idx}.pt'
        seis_filename= self.root_dir+'Seismic_TI/'+f'{idx}_{np.random.randint(0,self.nsim)}.pt'
        fac_file= torch.load(fac_filename)
        seis_file= torch.load(seis_filename)
        
        fac_file[fac_file==0]=-1
        sample = (fac_file, seis_file)

        return sample
    
