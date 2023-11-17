# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy 

from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as F
import os
import numpy as np

class JNDDataset(Dataset):
    def __init__(self, root, path_root, indices, resample=False):
        self.data_root = root
        self.indices = indices
        self.resample = resample
        self.paths = self.collect_paths(path_root)
        #print(self.paths['labels'])
        
    def collect_paths(self, root):
        paths = {'input' : [],
                 'output': [],
                 'labels': []}

        with open(os.path.join(root, 'dataset_combined.txt'), 'r') as f:
            lines = f.readlines()
            for idx in self.indices['combined']:
                inp, out, label = lines[idx].strip().split('\t')
                inp = os.path.join(self.data_root, inp)
                out = os.path.join(self.data_root, out)
                paths['input'].append(inp)
                paths['output'].append(out)
                paths['labels'].append(int(label))

        with open(os.path.join(root, 'dataset_reverb.txt'), 'r') as f:
            lines = f.readlines()
            for idx in self.indices['reverb']:
                inp, out, label = lines[idx].strip().split('\t')
                inp = os.path.join(self.data_root, inp)
                out = os.path.join(self.data_root, out)
                paths['input'].append(inp)
                paths['output'].append(out)
                paths['labels'].append(int(label))

        with open(os.path.join(root, 'dataset_linear.txt'), 'r') as f:
            lines = f.readlines()
            for idx in self.indices['linear']:
                inp, out, label, noise = lines[idx].strip().split('\t')
                inp = os.path.join(self.data_root, f"{noise.strip()}_list", inp)
                out = os.path.join(self.data_root, f"{noise.strip()}_list", out)
                paths['input'].append(inp)
                paths['output'].append(out)
                paths['labels'].append(int(label))

        with open(os.path.join(root, 'dataset_eq.txt'), 'r') as f:
            lines = f.readlines()
            for idx in self.indices['eq']:
                inp, out, label = lines[idx].strip().split('\t')
                inp = os.path.join(self.data_root, inp)
                out = os.path.join(self.data_root, out)
                paths['input'].append(inp)
                paths['output'].append(out)
                paths['labels'].append(int(label))
        
        return paths

    def __len__(self):
        return len(self.paths['input'])
 
    def __getitem__(self, idx):
        try:
            inp_file = self.paths['input'][idx]
            out_file = self.paths['output'][idx]

            inp, i_sr = torchaudio.load(inp_file)
            out, o_sr = torchaudio.load(out_file)

            if self.resample:
                inp = F.resample(inp, orig_freq=i_sr, new_freq=self.resample)
                out = F.resample(out, orig_freq=o_sr, new_freq=self.resample)

            #Pad signals so that they have equal length
            pad = torch.zeros(1, abs(inp.shape[-1] - out.shape[-1]))
            if inp.shape[-1] > out.shape[-1]:
                out = torch.cat([pad, out], dim=-1)
            if out.shape[-1] > inp.shape[-1]:
                inp = torch.cat([pad, inp], dim=-1)

            label = torch.tensor(self.paths['labels'][idx])
            return inp, out, label
        except Exception as e:
            print(f"Exception:{e}")
            print(f"Skip, get next idx:{idx+1}")
            self.__getitem__(min(idx+1, self.__len__()))

def collate_fn(batch):
    """
    Batch is a list of samples of len batch_size.
    Each sample is a tuple <inp_wav, out_wav, label>
    """
    max_len = 0
    for sample in batch:
        max_len = max(max_len, sample[0].shape[-1])
   
    final_dims = (len(batch), max_len)
    new_inp = sample[0].data.new(*final_dims).fill_(0)
    new_out = sample[1].data.new(*final_dims).fill_(0)

    for i, sample in enumerate(batch):
        new_inp[i, :sample[0].shape[-1]] = sample[0][0]
        new_out[i, :sample[1].shape[-1]] = sample[1][0]

    new_inp = new_inp.unsqueeze(1).unsqueeze(-1)
    new_out = new_out.unsqueeze(1).unsqueeze(-1)
    labels = torch.stack([sample[-1] for sample in batch])
    
    return new_inp, new_out, labels


    
def load_data(root, path_root, batch_size, n_cpu, split_ratio=0.7, resample=False, parallel=False):
    torchaudio.set_audio_backend("sox_io")  # in linux
    
    train_indices = {'combined':[], 'reverb':[], 'linear':[], 'eq':[]}
    test_indices = {'combined':[], 'reverb':[], 'linear':[], 'eq':[]}

    for key in train_indices:
        with open(os.path.join(path_root, f'dataset_{key}.txt'), 'r') as f:
            num_lines = len(f.readlines())
            train_indxs = list(np.random.choice(num_lines, int(split_ratio * num_lines), replace=False))
            test_indxs = [i for i in range(num_lines) if i not in train_indxs]
        train_indices[key].extend(train_indxs)
        test_indices[key].extend(test_indxs)
    if resample:
        resample = 16000
    train_ds = JNDDataset(root, path_root, train_indices, resample=resample)
    test_ds = JNDDataset(root, path_root, test_indices, resample=resample)

    if parallel:
        train_dataset = DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(train_ds),
            drop_last=True,
            num_workers=n_cpu,
            collate_fn=collate_fn,
        )
        test_dataset = DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(test_ds),
            drop_last=True,
            num_workers=n_cpu,
            collate_fn=collate_fn,
        )
    else:
        train_dataset = DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
            num_workers=n_cpu,
            collate_fn=collate_fn,
        )
        test_dataset = DataLoader(
            dataset=test_ds,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
            num_workers=n_cpu,
            collate_fn=collate_fn,
        )

    return train_dataset, test_dataset





        