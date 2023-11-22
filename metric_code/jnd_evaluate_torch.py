import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

from dataset_torch import load_data
from network_model_torch import JNDModel, JNDnet
from sklearn.metrics import classification_report

class Inference:
    def __init__(self, jnd_model, gpu_id=None, type=0):
        self.model = jnd_model
        self.gpu_id = gpu_id
        self.type = type

    def predict(self, dataset):
        LABELS = []
        PREDS = []
        for batch in tqdm(dataset):
            
            wav_in, wav_out, labels = batch
            if self.gpu_id is not None:
                wav_in = wav_in.to(self.gpu_id)
                wav_out = wav_out.to(self.gpu_id)
                labels = labels.to(self.gpu_id)
            
            if self.type==0:
                logits = self.model(wav_in, wav_out)
                preds = torch.argmax(logits, dim=-1).reshape(-1)
            if self.type == 1:
                _, preds, _ = self.model(wav_in, wav_out)
                preds = preds.reshape(-1)
            
            LABELS.extend(labels.detach().cpu().numpy().tolist())
            PREDS.extend(preds.detach().cpu().numpy().tolist())

        self.score(LABELS, PREDS)

    def score(self, labels, preds):
        print(classification_report(labels, preds))

def load_model(path, model):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model_state_dict'])
    return model

def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--root', required=True, help='root dir where the wavs are stored')
    parser.add_argument('--paths', help='path to the dir containing list of paths and labels.')
    parser.add_argument('--pt', help='Path to the model checkpoint')
    parser.add_argument('--batch_size', help='batch_size', default=16,type=int)
    parser.add_argument('--resample16k', help='resample to 16kHz', action='store_true')
    parser.add_argument('--gpu', help='set this flag for single gpu training', action='store_true')    
    parser.add_argument('--type', help='which model to run? JNDModel or JNDNet', type=int, default=0)  
    return parser

def main(ARGS):
    _, val_ds = load_data(root=ARGS.root, 
                          path_root=ARGS.paths, 
                          batch_size=ARGS.batch_size, 
                          n_cpu=1,
                          split_ratio=0.85,
                          resample=ARGS.resample16k, 
                          parallel=False)
    
    if ARGS.gpu:
        gpu_id = 0
    else:
        gpu_id = None
    if ARGS.type == 0:
        model = JNDModel(in_channels=1, 
                         n_layers=14, 
                         keep_prob=0.7, 
                         norm_type='sbn',
                         gpu_id=gpu_id)
    if ARGS.type == 1:
        model = JNDnet(nconv=14,
                       nchan=32,
                       dist_dp=0.1,
                       dist_act='no',
                       ndim=[16,6],
                       classif_dp=0.1,
                       classif_BN=0,
                       classif_act='no',
                       dev=gpu_id,
                       minit=0)
    
    model = load_model(ARGS.pt, model.to(gpu_id))
    print(f"Model loaded from {ARGS.pt}")
    print(f"Running inference...")
    INFERENCE = Inference(model, gpu_id, ARGS.type)
    INFERENCE.predict(val_ds)

if __name__=='__main__':
    ARGS = args().parse_args()
    main(ARGS)