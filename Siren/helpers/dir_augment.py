import torchaudio
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset

IR_PATH = '../../impulse_responses'

class IR_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv(os.path.join(IR_PATH, 'impulse_responses.csv'))
        self.irs = []
        for file in self.df['impulse_responses']:
            ir,_ = torchaudio.load(os.path.join(IR_PATH, file))
            self.irs.append(ir)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.irs[index]

class DirAugment(nn.Module):
    def __init__(self, orig_freq, new_freq, augment_p):
        super().__init__()
        self.new_freq = new_freq
        self.resample = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
        self.conv = torchaudio.transforms.Convolve()
        self.irs = IR_Dataset()
        self.p = augment_p

    def forward(self, x, devices):
        x = self.resample(x)
        
        idx = np.where(devices.cpu() == 0)[0] # only augment device A which is index 0
        for i in idx:
            if np.random.rand() > self.p:
                continue

            # augment audio file device impulse response
            rnd = np.random.randint(len(self.irs)) # pick a random impulse response
            ir = self.irs[rnd].cuda()
            x[i] = self.conv(x[i], ir)[:,:self.new_freq] # after convolving, the time series is longer, truncate to new_freq
        return x