import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import numpy as np
import random, tqdm, sys, math, gzip, os
from utils.util import d, save_checkpoint, load_checkpoint
import spacy
from spacy.tokenizer import Tokenizer
import matplotlib.pyplot as plt
from transformers import CTransformer
project_dir = os.path.abspath(os.path.join(__file__ ,"../../.."))
sys.path.insert(1, project_dir)
from src.data.make_dataset import IMDBDataset

def predict(model, mx, context):
    with torch.no_grad():
        model.eval()
        context = context.view(1, -1).long() 
        input = context.to(device)
        if input.size(1) > mx:
            input = input[:, :mx]
        
        out = model(input)
        print ("probabilities...",out)
        label = out.argmax(dim=1)
        # Adding one to match the label index
        label += 1
        print(LABEL.vocab.itos[label])

if __name__ == "__main__":
    
    text = sys.argv[1]
    
    # text = "If you're going to watch this movie, avoid any spoilers, even spoiler free reviews. Which is why I'm not going to say anything about the movie. Not even my opinion. All I'm going to say is: The crowd applauded 3 times during the movie, and stood up to clap their hands after. This I have never witnessed in a Dutch cinema. Dutch crowds aren't usually passionate about this. I checked the row where I was sitting, and people were crying. After the movie, I was seeing people with smudged mascara. That's all I have to say about the movie."

    NUM_CLS = 2
    embedding_size = 128
    num_heads = 8
    depth = 6
    max_length = 512
    max_pool = True
    
    # load and build vocabulory
    vocab_size = 50000
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    tdata, test = datasets.IMDB.splits(text_field=TEXT, label_field=LABEL, root='data/processed/')
    TEXT.build_vocab(tdata, max_size=vocab_size - 2)
    LABEL.build_vocab(tdata)

    # process input
    nlp = spacy.load("en_core_web_sm")
    tokenizer = Tokenizer(nlp.vocab)
    list_of_strings = [tok.text for tok in tokenizer(text)]
    context = [TEXT.vocab.stoi[c] for c in list_of_strings]
    context = np.asarray(context)
    context = torch.from_numpy(context)

    #creating the original network
    save_path = 'models/CTransformersNet.pt' 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    load_model = CTransformer(emb=embedding_size, heads=num_heads, depth=depth, seq_length=max_length,  num_tokens=vocab_size, num_classes=NUM_CLS, max_pool=max_pool).to(device)
    state_dict = torch.load(save_path,map_location=torch.device('cpu'))
    load_model.load_state_dict(state_dict['model_state_dict'])
    load_optimizer = torch.optim.Adam(load_model.parameters(), lr=0.0001)
    load_optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    print(f'Model loaded has validation loss: ',val_loss)

    predict(load_model, max_length, context)