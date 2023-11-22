# -*- coding: utf-8 -*-
"""
@author: Anurag Kumar
"""
import os
import torch
import argparse
import wandb
import torch.nn as nn
import torch.nn.functional as F
from network_model_torch import JNDModel, JNDnet

from dataset_torch import load_data

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

wandb.login()

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--root', required=True, help='root dir where the wavs are stored')
    parser.add_argument('-o', '--output', required=True, help='path to store saved checkpoints.')
    parser.add_argument('--exp', help='name of the experiment')
    parser.add_argument('--paths', help='path to the dir containing list of paths and labels.')
    parser.add_argument('--layers', help='number of layers in the model', default=14, type=int)
    parser.add_argument('--learning_rate', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--summary_folder', help='summary folder name', default='m_example')
    parser.add_argument('--optimiser', help='choose optimiser - gd/adam', default='adam')
    parser.add_argument('--loss_norm', help='loss norm - NM,SBN,None', default='sbn')
    parser.add_argument('--channels_increase', help='doubling channels after how many layers - 1,2,3,4,5,6', default=5, type=int)
    parser.add_argument('--loss_layers', help='loss to be taken for the last how many layers', default=14, type=int)
    parser.add_argument('--filter_size', help='filter size for the convolutions', default=3, type=int)
    parser.add_argument('--train_from_checkpoint', help='train_from_checkpoint', default=0, type=int)
    parser.add_argument('--epochs', help='number of training epochs', default=2000, type=int)
    parser.add_argument('--type', help='linear/finetune/scratch', default='scratch')
    parser.add_argument('--pretrained_model_path', help='Model Path for the pretrained model', default='../pre-model/pretrained_loss')
    parser.add_argument('--batch_size', help='batch_size', default=16,type=int)
    parser.add_argument('--dummy_test', help='batch_size', default=0,type=int)
    parser.add_argument('--resample16k', help='resample to 16kHz', action='store_true')
    parser.add_argument('--accum_grad', type=int, default=2, help='no of steps to accumulate gradient')
    parser.add_argument('--gpu', help='set this flag for single gpu training', action='store_true')
    parser.add_argument('--parallel', help='set this flag for parallel gpu training', action='store_true')
    
    return parser

class JNDTrainer:
    """
    Pytorch recipe to train the JND model described 
    in https://arxiv.org/pdf/2001.04460.pdf
    """
    def __init__(self, 
                 args, 
                 train_dataloader, 
                 val_dataloader,
                 in_channels, 
                 n_layers, 
                 keep_prob, 
                 norm_type='sbn',
                 gpu_id=None):
        
        self.model = JNDModel(in_channels, 
                              n_layers, 
                              keep_prob, 
                              norm_type,
                              args.loss_layers,
                              gpu_id)
        """
        self.model = JNDnet(nconv=n_layers,
                            nchan=32,
                            dist_dp=0.1,
                            dist_act='no',
                            ndim=[16,6],
                            classif_dp=0.1,
                            classif_BN=0,
                            classif_act='no',
                            dev=gpu_id,
                            minit=0)
        """
        #self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.criterion = nn.BCELoss(reduction='mean')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        if gpu_id is not None:
            self.model = self.model.to(gpu_id)
            
            if args.parallel:
                self.model = DDP(self.model, device_ids=[gpu_id])
                
        self.gpu_id = gpu_id
        self.train_ds = train_dataloader
        self.val_ds = val_dataloader
        self.args = args
        self.parallel = args.parallel

        wandb.init(project=args.exp)

    def create_spectrograms(self, noisy, clean):
        """
        Create spectrograms from input waveform.
        ARGS:
            clean : clean waveform (batch * cut_len)
            noisy : noisy waveform (batch * cut_len)

        Return
            noisy_spec : (b * 2 * f * t) noisy spectrogram
            clean_spec : (b * 2 * f * t) clean spectrogram
        """
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )

        win = torch.hamming_window(400)
        if self.gpu_id is not None:
            win = win.to(self.gpu_id)

        noisy_spec = torch.stft(
            noisy,
            400,
            100,
            window=win,
            onesided=True,
        )
        clean_spec = torch.stft(
            clean,
            400,
            100,
            window=win,
            onesided=True,
        )
        #noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        #clean_spec = power_compress(clean_spec)
        noisy_spec = noisy_spec.permute(0, 3, 2, 1)
        clean_spec = clean_spec.permute(0, 3, 2, 1)

        noisy_mag = torch.sqrt(noisy_spec[:, 0, :, :]**2 + noisy_spec[:, 1, :, :]**2).unsqueeze(1)
        clean_mag = torch.sqrt(noisy_spec[:, 0, :, :]**2 + noisy_spec[:, 1, :, :]**2).unsqueeze(1)

        return noisy_mag, clean_mag          
                        
    def load_model(self, path):
        state_dict = torch.load(path, map_location=torch.device(self.gpu_id))
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['opt_state_dict'])
        print(f"Loaded checkpoint saved at {path} starting at epoch {self.start_epoch}")
        del state_dict

    def save_model(self, path):
        if self.parallel:
            save_dict = {'model_state_dict':self.model.module.state_dict(), 
                        'opt_state_dict':self.optimizer.state_dict(),
                        }
        else:
            save_dict = {'model_state_dict':self.model.state_dict(), 
                        'opt_state_dict':self.optimizer.state_dict(),
                        }
        torch.save(save_dict, path)

    def forward_one_step(self, batch):
        wav_in, wav_out, labels = batch
        #b, len = wav_in.shape
        if self.gpu_id is not None:
            wav_in = wav_in.to(self.gpu_id)
            wav_out = wav_out.to(self.gpu_id)
            labels = labels.to(self.gpu_id)

        wav_in, wav_out = self.create_spectrograms(wav_in, wav_out)
        
        labels = labels.long().reshape(-1)
        probs = self.model(inp=wav_in, ref=wav_out)
        loss = self.criterion(probs, labels) 
        #_, _, class_prob = self.model(xper=wav_in, xref=wav_out)
        #loss = self.criterion(class_prob, labels)
        return loss 

    
    def train_one_epoch(self, epoch):
        epoch_loss = 0
        num_batches = len(self.train_ds)
        ACCUM_GRAD = self.args.accum_grad
        for i, batch in enumerate(self.train_ds):
            batch_loss = self.forward_one_step(batch)
            batch_loss.backward()
            if (i + 1) % ACCUM_GRAD == 0:
                self.optimizer.zero_grad()    
                self.optimizer.step()
            wandb.log({
                'step':i+1,
                'loss':batch_loss
            })
            print(f"EPOCH:{epoch+1} | STEP:{i+1} | LOSS:{batch_loss}")
            epoch_loss += batch_loss.detach()
        epoch_loss = epoch_loss / num_batches
        return epoch_loss

    def run_validation(self):
        val_loss = 0
        num_batches = len(self.val_ds)
        with torch.no_grad():
            for i, batch in enumerate(self.val_ds):
                batch_loss = self.forward_one_step(batch)
                val_loss += batch_loss.detach()

        val_loss = val_loss / num_batches
        return val_loss

    def train(self, epochs):
        best_val = 999999999
        for epoch in range(epochs):
            self.model.train()
            ep_loss = self.train_one_epoch(epoch)
            
            self.model.eval()
            val_loss = self.run_validation()
            
            wandb.log({
                'epoch':epoch+1,
                'train_loss':ep_loss,
                'val_loss':val_loss
            })
            print(f"EPOCH:{epoch} | TRAIN_LOSS:{ep_loss} | VAL_LOSS:{val_loss}")

            if best_val >= val_loss:
                best_val = val_loss
                if self.gpu_id == 0:
                    checkpoint_prefix = f"{self.args.exp}_val_{val_loss}_epoch_{epoch}.pt"
                    path = os.path.join(self.args.output, self.args.exp, checkpoint_prefix)
                    self.save_model(path)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank, world_size, args):
    keep_prob_drop=1

    if args.type!='linear' or args.type!='finetune':
        keep_prob_drop=0.70

    if args.parallel:
        ddp_setup(rank, world_size)
        if rank == 0:
            print(args)
            available_gpus = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            print(f"Available gpus:{available_gpus}")

        train_ds, val_ds = load_data(root=args.root, 
                                     path_root=args.paths, 
                                     batch_size=args.batch_size, 
                                     n_cpu=1,
                                     split_ratio=0.85,
                                     resample=args.resample16k, 
                                     parallel=True)
    else:
        train_ds, val_ds = load_data(root=args.root, 
                                     path_root=args.paths, 
                                     batch_size=args.batch_size, 
                                     n_cpu=1,
                                     split_ratio=0.85,
                                     resample=args.resample16k,  
                                     parallel=False)
    print(f"TRAIN:{len(train_ds)} VAL:{len(val_ds)} per each of the {rank+1} gpu/gpus...")
    trainer = JNDTrainer(args=args, 
                         train_dataloader=train_ds, 
                         val_dataloader=val_ds,
                         in_channels=2, 
                         n_layers=args.layers, 
                         keep_prob=keep_prob_drop, 
                         norm_type=args.loss_norm,
                         gpu_id=rank)

    trainer.train(epochs=args.epochs)
    

if __name__=='__main__':
    ARGS = argument_parser().parse_args()
    
    output = f"{ARGS.output}/{ARGS.exp}"
    os.makedirs(output, exist_ok=True)

    world_size = torch.cuda.device_count()
    print(f"World size:{world_size}")
    if ARGS.parallel:
        mp.spawn(main, args=(world_size, ARGS), nprocs=world_size)
    else:
        if ARGS.gpu:
            main(0, world_size, ARGS)
        else:
            main(None, world_size, ARGS)

    