# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import numpy as np
import gzip


class IMDBDataset(object):
    def __init__(self, TEXT, LABEL, vocab_size, batch_size):
        self.TEXT = TEXT
        self.LABEL = LABEL
        self.vocab_size = vocab_size
        self.batch_size = batch_size

    def prepare_data(self):
        print("Loading dataset....")
        tdata, test = datasets.IMDB.splits(text_field=self.TEXT, label_field=self.LABEL, root='data/processed/')
        train, val = tdata.split(split_ratio=0.8)
        self.TEXT.build_vocab(train, max_size=self.vocab_size - 2)
        self.LABEL.build_vocab(train)
        train_iter = data.BucketIterator(train, batch_size=self.batch_size)
        val_iter = data.BucketIterator(val, batch_size=1)
        test_iter = data.BucketIterator(test, batch_size=1)
        return train_iter, val_iter, test_iter

class WikipediaDataset(object):
    def __init__(self, path, n_train, n_valid, n_test):
        self.path= path
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
    
    def prepare_data(self):
        with gzip.open(self.path) if self.path.endswith('.gz') else open(self.path) as file:
            X = np.fromstring(file.read(self.n_train + self.n_valid + self.n_test), dtype=np.uint8)
            trX, vaX, teX = np.split(X, [self.n_train, self.n_train + self.n_valid])
            return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

