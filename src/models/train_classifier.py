# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import random, tqdm, sys, math, gzip, os
from utils.util import d, save_checkpoint, load_checkpoint
from transformers import CTransformer
project_dir = os.path.abspath(os.path.join(__file__ ,"../../.."))
sys.path.insert(1, project_dir)
from src.data.make_dataset import IMDBDataset

def trainF(model, train_loader, val_loader, num_epochs, criterion, save_name):
    
    gradient_clipping = 1.0
    best_val_loss = float("Inf")
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        print('Starting epoch ' + str(epoch + 1))

        for batch in tqdm.tqdm(train_loader):

            input = batch.text[0].to(device)
            label = batch.label - 1
            label = label.to(device)

            if input.size(1) > mx:
                input = input[:, :mx]
            out = model(input)
            loss = criterion(out, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            #clip gradients
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_running_loss = 0.0
        with torch.no_grad():
            model.eval()
            tot, cor = 0.0, 0.0

            for batch in val_loader:
                input = batch.text[0].to(device)
                label = batch.label - 1
                label = label.to(device)

                if input.size(1) > mx:
                    input = input[:, :mx]
                out = model(input)
                loss = criterion(out, label)
                val_running_loss += loss.item()

                tot += float(input.size(0))
                cor += float((label == out.argmax(dim=1)).sum().item())

            acc = cor/tot 
            print("validation accuracy {:.3f}".format(acc))
        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
            .format(epoch+1, num_epochs, avg_train_loss, avg_val_loss))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(save_name, model, optimizer, best_val_loss)

    print("Finished Training")
    return train_losses, val_losses                   
    

# evaluation metrics
def eval(model, test_loader):
    with torch.no_grad():
        model.eval()
        correct = 0
        print('Starting Iteration')
        count = 0
        for batch in test_loader:
            input = batch.text[0].to(device)
            label = batch.label - 1
            label = label.to(device)
            if input.size(1) > mx:
                input = input[:, :mx]
            out = model(input)
            if label == out.argmax(dim=1):
                correct += 1
            count += 1 
        print("Current Count is: {}".format(count))
        print('Accuracy on test set: {}'.format(correct/count))


if __name__ == "__main__":
    vocab_size = 50000
    batch_size = 4
    LOG2E = math.log2(math.e)
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)

    data = IMDBDataset(TEXT, LABEL, vocab_size, batch_size)
    train_iter, val_iter, test_iter = data.prepare_data()
    print(f'- nr. of training examples {len(train_iter)}')
    print(f'- nr. of validation examples {len(val_iter)}')
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
    if torch.cuda.is_available():
        model.cuda()
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model architecture:\n\n', model)
    print(f'\nThe model has {temp:,} trainable parameters')

    # start training
    optimizer = torch.optim.Adam(lr = 0.0001, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (10000/batch_size), 1.0))
    num_epochs = 10
    criterion = nn.NLLLoss()
    save_path = 'models/CTransformersNet.pt'
    train_losses, val_losses = trainF(model, train_iter, val_iter, num_epochs, criterion, save_path)

    # Evaluation on previously saved models
    load_model = CTransformer(emb=embedding_size, heads=num_heads, depth=depth, seq_length=mx, num_tokens=vocab_size, num_classes=NUM_CLS, max_pool=max_pool)
    load_model = load_model.to(device)
    load_optimizer = torch.optim.Adam(load_model.parameters(),lr = 0.0001)

 
    total_step = len(train_iter)*num_epochs
    best_val_loss = load_checkpoint(load_model, load_optimizer)

    print(best_val_loss)
    eval(load_model, test_iter)

    #plotting of training and validation loss
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label="Validation Loss")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig('reports/figures/losscomparison.png')
