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

def trainF(model, data_train, data_test, num_batches, refcontext, test_every, test_subset, test_batchsize, criterion, save_name):
    
    gradient_clipping = 1.0
    best_bits_per_byte = float("Inf")
    total_bits = []


    # training loop
    # - note: we don't loop over the data, instead we sample a batch of random subsequences each time.
    for i in tqdm.trange(num_batches):
        # sample a batch of random subsequences
        starts = torch.randint(size=(batch_size, ), low=0, high=data_train.size(0) - refcontext - 1)
        seqs_source = [data_train[start  :start+refcontext] for start in starts]
        seqs_target = [data_train[start+1:start+refcontext+1] for start in starts]
        source = torch.cat([s[None, :] for s in seqs_source ], dim=0).to(torch.long)
        target = torch.cat([s[None, :] for s in seqs_target ], dim=0).to(torch.long)
        # - target is the same sequence as source, except one character ahead

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()
        source, target = Variable(source), Variable(target)

        output = model(source)

        loss = criterion(output.transpose(2, 1), target)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()       

        # clip gradients
        # - If the total gradient vector has a length > 1, we clip it back down to 1.
        if gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()
        scheduler.step()

        # - validate every {test_every} steps. First we compute the
        #   compression on the validation (or a subset)
        #   then we generate some random text to monitor progress
        if i != 0 and (i % test_every == 0 or i == num_batches - 1):

            upto = data_test.size(0) if i == num_batches - 1 else test_subset
            data_sub = data_test[:upto]

            with torch.no_grad():
                bits, tot = 0.0, 0
                batch = [] # buffer, every time it fills up, we run it through the model

                for current in range(data_sub.size(0)):

                    fr = max(0, current - refcontext)
                    to = current + 1

                    context = data_sub[fr:to].to(torch.long)
                    
                    if context.size(0) < refcontext + 1:
                        pad = torch.zeros(size=(refcontext + 1 - context.size(0),), dtype=torch.long)
                        context = torch.cat([pad, context], dim=0)

                        assert context.size(0) == refcontext + 1

                    if torch.cuda.is_available():
                        context = context.cuda()

                    batch.append(context[None, :])

                    if len(batch) == test_batchsize or current == data_sub.size(0) - 1:

                        # batch is full, run it through the model
                        b = len(batch)

                        all = torch.cat(batch, dim=0)
                        source = all[:, :-1] # input
                        target = all[:, -1]  # target values

                        output = model(source)

                        lnprobs = output[torch.arange(b, device=d()), -1, target]
                        log2probs = lnprobs * LOG2E # convert from nats to bits

                        bits += - log2probs.sum()
                        batch = [] # empty buffer
                        

                bits_per_byte = bits / data_sub.size(0)

                total_bits.append(bits_per_byte)

                # print validation performance. 1 bit per byte is (currently) state of the art.
                print(f'epoch{i}: {bits_per_byte:.4} bits per byte')

                if bits_per_byte < best_bits_per_byte:
                  best_bits_per_byte = bits_per_byte
                  save_checkpoint(save_name, model, optimizer, best_bits_per_byte)


                # generate some random text
                GENSIZE = 600
                TEMP = 0.5
                seedfr = random.randint(0, data_test.size(0) - refcontext)
                input = data_test[seedfr:seedfr + refcontext].to(torch.long)

                if torch.cuda.is_available():
                    input = input.cuda()

                input = Variable(input)

                print('[', end='', flush=True)
                for c in input:
                    print(str(chr(c)), end='', flush=True)
                print(']', end='', flush=True)

                for _ in range(GENSIZE):
                    output = model(input[None, :])
                    c = sample(output[0, -1, :], TEMP)
                    print(str(chr(max(32, c))), end='', flush=True)

                    input = torch.cat([input[1:], c[None]], dim=0)

                print()


    print("Finished Training")
    return total_bits

# evaluation metrics
def eval(model, test_loader):
    
    with torch.no_grad():
        bits = 0.0
        refcontext = 256
        # batch is full, run it through the model
        for i in tqdm.trange(test_loader.size(0)):
          seedfr = random.randint(0, test_loader.size(0) - refcontext)
          input = test_loader[seedfr:seedfr + refcontext].to(torch.long)

          context = input
          b = len(context[None, :])

          if context.size(0) < refcontext + 1:
              pad = torch.zeros(size=(refcontext + 1 - context.size(0),), dtype=torch.long)
              context = torch.cat([pad, context], dim=0)

          if torch.cuda.is_available():
            context = context.cuda()

          source = context[None, :-1] # input
          target = context[-1]  # target values

          output = model(source)

          lnprobs = output[torch.arange(b, device=d()), -1, target]
          log2probs = lnprobs * LOG2E # convert from nats to bits

          bits += - log2probs.sum()

        bits_per_byte = bits / len(test_loader)

        # print test performance. 1 bit per byte is (currently) state of the art.
        print(f'bits per byte: ', bits_per_byte)

        # generate some random text
        GENSIZE = 600
        TEMP = 0.5
        seedfr = random.randint(0, test_loader.size(0) - refcontext)
        input = test_loader[seedfr:seedfr + refcontext].to(torch.long)

        if torch.cuda.is_available():
            input = input.cuda()

        input = Variable(input)

        print('[', end='', flush=True)
        for c in input:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

        for _ in range(GENSIZE):
            output = model(input[None, :])
            c = sample(output[0, -1, :], TEMP)
            print(str(chr(max(32, c))), end='', flush=True)

            input = torch.cat([input[1:], c[None]], dim=0)

        print() 

if __name__ == "__main__":

    # NB, the enwik8 data contains tokens from 9 to 240, but well round up to the nearest
    # power of two.
    NUM_TOKENS = 256
    # Used for converting between nats and bits
    LOG2E = math.log2(math.e)

    data = 'data/processed/enwik9.gz'
    n_train=int(90e6)
    n_valid=int(5e6) 
    n_test=int(5e6)

    data_train, data_val, data_test = WikipediaDataset(data, n_train, n_valid, n_test)

    embedding_size = 128
    num_heads = 8
    depth = 12
    refcontext = 256
    wide = True
    batch_size = 32

    # create the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GTransformer(emb=embedding_size, heads=num_heads, depth=depth, seq_length=context, num_tokens=NUM_TOKENS, wide=wide)
    if torch.cuda.is_available():
        model.cuda()

    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model architecture:\n\n', model)
    print(f'\nThe model has {temp:,} trainable parameters')

    # start training
    optimizer = torch.optim.Adam(lr = 0.0001, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (5000/batch_size), 1.0))
    criterion = nn.NLLLoss(reduction='mean')
    num_batches = 10000
    test_every = 100
    test_subset = 10000
    test_batchsize = 64
    save_path = 'models/GTransformersNet.pt'
    bits_per_byte = trainF(model, data_train, data_val, num_batches, refcontext, test_every, test_subset, test_batchsize, criterion, save_path)

    # Evaluation on previously saved models
    load_model = GTransformer(emb=embedding_size, heads=num_heads, depth=depth, seq_length=refcontext, num_tokens=NUM_TOKENS, wide=wide).to(device)
    load_optimizer = torch.optim.Adam(load_model.parameters(), lr=0.0001)

    best_val_loss = load_checkpoint(load_model, save_path, load_optimizer)
    print(best_val_loss)

    test_subset = 10000
    eval(load_model, data_test[:test_subset])

    #plotting of training and validation loss
    plt.xlabel('epoch')
    plt.ylabel('bits per byte')
    plt.plot(bits_per_byte, label='bits_per_byte')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('reports/figures/lossgenerator.png')    