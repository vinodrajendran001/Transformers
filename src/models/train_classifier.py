# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchtext import data, datasets, vocab

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip, os

from utils.util import d
from transformers import CTransformer
project_dir = os.path.abspath(os.path.join(__file__ ,"../../.."))
sys.path.insert(1, project_dir)
from src.data.make_dataset import IMDBDataset

def train(model, train_loader, test_loader, num_epochs, criterion, save_name):
    seen = 0
    gradient_clipping = 1.0
    # tbw = SummaryWriter(log_dir=tb_dir) # Tensorboard logging
    for epoch in range(num_epochs):
        print(f'\n epoch {epoch}')
        model.train(True)

        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            input = batch.text[0]
            label = batch.label[0] - 1

            if input.size(1) > mx:
                input = input[:, :mx]
            out = model(input)
            print(out, label)
            loss = criterion(out, label)

            loss.backward()

            #clip gradients
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

            optimizer.step()
            scheduler.step()

            seen += input.size(0)
            # tbw.add_scalar('classification/train-loss', float(loss.item()), seen)

        with torch.no_grad():
            model.train(False)
            tot, cor = 0.0, 0.0

            for batch in test_loader:
                input = batch.text[0]
                label = batch.label[0] - 1

                if input.size(1) > mx:
                    input = input[:, :mx]
                out = model(input)

                tot += float(input.size(0))
                cor += float((label == out).sum().item())

            acc = cor/tot 
            print(f'-- "test accuracy {acc:.3}')
            # tbw.add_scalar('classification/test-loss', float(loss.item()), epoch)                      
    

if __name__ == "__main__":
    vocab_size = 50000
    batch_size = 4
    LOG2E = math.log2(math.e)
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)

    data = IMDBDataset(TEXT, LABEL, vocab_size, batch_size)
    train_iter, test_iter = data.prepare_data()
    print(f'- nr. of training examples {len(train_iter)}')
    print(f'- nr. of test examples {len(test_iter)}')

    NUM_CLS = 2
    embedding_size = 128
    num_heads = 8
    depth = 6
    max_length = 512
    max_pool = True

    if max_length < 0:
        mx = max([input.text[0].size(1) for input in train_iter])
        mx = mx * 2
        print(f'- maximum sequence length: {mx}')
    else:
        mx = max_length


    #creating the original network and couting the parameters of different networks
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CTransformer(emb=embedding_size, heads=num_heads, depth=depth, seq_length=mx, num_tokens=vocab_size, num_classes=NUM_CLS, max_pool=max_pool)

    model = model.to(device)
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model architecture:\n\n', model)
    print(f'\nThe model has {temp:,} trainable parameters')

    # start training
    optimizer = torch.optim.Adam(lr = 0.0001, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (10000/batch_size), 1.0))
    num_epochs = 2
    criterion = nn.NLLLoss()
    save_path = 'models/CTransformersNet.pt'
    train_losses, val_losses = train(model, train_iter, test_iter, num_epochs, criterion, save_path)