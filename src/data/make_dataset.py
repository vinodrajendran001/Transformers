# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext import data, datasets, vocab


class IMDBDataset(object):
    def __init__(self, TEXT, LABEL, vocab_size, batch_size):
        self.TEXT = TEXT
        self.LABEL = LABEL
        self.vocab_size = vocab_size
        self.batch_size = batch_size

    def prepare_data(self):
        print("Loading dataset....")
        train, test = datasets.IMDB.splits(text_field=self.TEXT, label_field=self.LABEL, root='../../data/processed/')
        self.TEXT.build_vocab(train, max_size=self.vocab_size - 2)
        self.LABEL.build_vocab(train)
        train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=self.batch_size) 
        return train_iter, test_iter
