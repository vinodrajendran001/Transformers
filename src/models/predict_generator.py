# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist
import random, tqdm, sys, math, gzip, os
import numpy as np
from utils.util import d, save_checkpoint, load_checkpoint


import matplotlib.pyplot as plt
from transformers import GTransformer
project_dir = os.path.abspath(os.path.join(__file__ ,"../../.."))
sys.path.insert(1, project_dir)
from src.data.make_dataset import WikipediaDataset


def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()

def predict(model, test_loader):
    
    context = [ord(c) for c in test_loader]
    context = np.asarray(context)
    context = torch.from_numpy(context)
    
    with torch.no_grad():
        model.eval()
        refcontext = 256
        # generate some random text
        GENSIZE = 600
        TEMP = 0.5

        if context.size(0) < refcontext + 1:
            pad = torch.zeros(size=(refcontext - context.size(0),), dtype=torch.long)
            input = torch.cat([pad, context], dim=0)
        elif context.size(0) > 256:
            input = context[-256:]
        else:
            input = context

        if torch.cuda.is_available():
            input = input.cuda()

        input = Variable(input)

        print (f"Input text considered according to model config....\n")

        print('[', end='', flush=True)
        for c in input:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

        print (f"\nGenerated text....\n")

        for _ in range(GENSIZE):
            output = model(input[None, :]).to(device)
            c = sample(output[0, -1, :], TEMP)
            print(str(chr(max(32, c))), end='', flush=True)

            input = torch.cat([input[1:], c[None]], dim=0)

        print()  


if __name__ == "__main__":


    text = sys.argv[1]

    # text = "1228X Human & Rousseau. Because many of his stories were originally published in long-forgotten magazines and journals, there are a number of [[anthology|anthologies]] by different collators each containing a different selection. His original books ha"
    # NB, the enwik8 data contains tokens from 9 to 240, but well round up to the nearest
    # power of two.
    NUM_TOKENS = 256
    # Used for converting between nats and bits
    LOG2E = math.log2(math.e)

    embedding_size = 128
    num_heads = 8
    depth = 12
    refcontext = 256
    wide = True

    #creating the original network
    save_path = 'models/GTransformersNet.pt' 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    load_model = GTransformer(emb=embedding_size, heads=num_heads, depth=depth, seq_length=refcontext, num_tokens=NUM_TOKENS, wide=wide).to(device)
    state_dict = torch.load(save_path,map_location=torch.device('cpu'))
    load_model.load_state_dict(state_dict['model_state_dict'])
    load_optimizer = torch.optim.Adam(load_model.parameters(), lr=0.0001)
    load_optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    print(f'Bits per byte info of loaded model: ',val_loss)

    predict(load_model, text)