# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""
import torch
import torch.nn as nn
i#mport torch.nn.functional as F
import numpy 

from torch.utils.data import Dataset, Dataloader
import torchaudio
import torchaudio.functional as F
import os

class JNDDataset(Dataset):
    def __init__(self, root, path_root, indices, resample=None):
        self.data_root = root
        self.paths = self.collect_paths(path_root)
        self.resample = resample
        self.indices = indices

    def collect_paths(self, root):
        paths = {'input' : [],
                 'output': [],
                 'labels': []}

        with open(os.join(root, 'dataset_combined.txt'), 'r') as f:
            lines = f.readlines():
            for idx in self.indices['combined']:
                inp, out, label = line.split('\t')
                inp = os.path.join(self.data_root, inp)
                out = os.path.join(self.data_root, out)
                paths['input'].append(inp)
                paths['output'].append(out)
                paths['labels'].append(int(label))

        with open(os.join(root, 'dataset_reverb.txt'), 'r') as f:
            lines = f.readlines():
            for idx in self.indices['reverb']:
                inp, out, label = line.split('\t')
                inp = os.path.join(self.data_root, inp)
                out = os.path.join(self.data_root, out)
                paths['input'].append(inp)
                paths['output'].append(out)
                paths['labels'].append(int(label))

        with open(os.join(root, 'dataset_linear.txt'), 'r') as f:
            lines = f.readlines():
            for idx in self.indices['linear']:
                inp, out, label, noise = line.split('\t')
                inp = os.path.join(self.data_root, f"{noise}_list", inp)
                out = os.path.join(self.data_root, f"{noise}_list", out)
                paths['input'].append(inp)
                paths['output'].append(out)
                paths['labels'].append(int(label))

        with open(os.join(root, 'dataset_eq.txt'), 'r') as f:
            lines = f.readlines():
            for idx in self.indices['eq']:
                inp, out, label = line.split('\t')
                inp = os.path.join(self.data_root, inp)
                out = os.path.join(self.data_root, out)
                paths['input'].append(inp)
                paths['output'].append(out)
                paths['labels'].append(int(label))
        
        return paths

    def __len__(self):
        return len(self.paths['input'])

    def __getitem__(self, idx):
        inp_file = self.paths['input'][idx]
        out_file = self.paths['output'][idx]

        inp, i_sr = torchaudio.load(inp_file)
        out, o_sr = torchaudio.load(out_file)

        if self.resample is not None:
            inp = F.resample(inp, orig_freq=i_sr, new_freq=self.resample)
            out = F.resample(out, orig_freq=o_sr, new_freq=self.resample)

        inp = inp.squeeze()
        out = out.squeeze()

        #Pad signals so that they have equal length
        pad = torch.zeros(torch.abs(inp.shape[0] - out.shape[0]))
        if inp.shape[0] > out.shape[0]:
            out = torch.cat([pad, out], dim=0)
        if out.shape[0] > inp.shape[0]:
            inp = torch.cat([pad, inp], dim=0)

        label = torch.LongTensor(self.paths['labels'][idx])
        return inp, out, label

    
def load_data(root, path_root, batch_size, n_cpu, split_ratio=0.7, parallel=False):
    torchaudio.set_audio_backend("sox_io")  # in linux
    
    train_indices = {'combined':[], 'reverb':[], 'linear':[], 'eq':[]}
    test_indices = {'combined':[], 'reverb':[], 'linear':[], 'eq':[]}

    for key in train_indices:
        with open(os.path.join(path_root, f'dataset_{key}.txt'), 'r') as f:
            num_lines = len(f.readlines())
            train_indices = list(np.random.choice(num_lines, int(split_ratio * num_lines), replace=False))
            test_indices = [i for i in range(num_lines) if i not in train_indices]
        train_indices[key] = train_indices
        test_indices[key] = test_indices

    train_ds = JNDDataset(root, path_root, train_indices, resample=None)
    test_ds = JNDDataset(root, path_root, test_indices, resample=None)

    if parallel:
        train_dataset = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(train_ds),
            drop_last=True,
            num_workers=n_cpu,
        )
        test_dataset = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(test_ds),
            drop_last=True,
            num_workers=n_cpu,
        )
    else:
        train_dataset = DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
            num_workers=n_cpu,
        )
        test_dataset = DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
            num_workers=n_cpu,
        )

    return train_dataset, test_dataset





        