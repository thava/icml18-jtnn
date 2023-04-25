import torch
import torch.nn as nn

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit

from datetime import datetime

def print_time():
    print(str(datetime.now()) + ': ', end=' ', flush=True)

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

args = parser.parse_args()

print_time();
print('Opening vocabulary file ...', flush=True)
vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

print_time();
print('Finished Opening vocabulary file ...', flush=True)

model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)

print_time();
print('Loading Model ...', flush=True)
model.load_state_dict(torch.load(args.model))
print_time();
print('Finished Loading Model ...', flush=True)

model = model.cuda()

torch.manual_seed(0)

print_time();
print('Starting Sampling ... ', flush=True)

for i in range(args.nsample):
    print(model.sample_prior())
    print_time()
    print(f'Finished {i}th Sampling. ', flush=True)
