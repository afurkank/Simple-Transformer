"""
Ahmet Furkan Karacık
afurkank @ github

You need to install tokenizer, torchtext and torchdata.
Some versions of torchtext and torchdata may not be compatible
with certain versions of torch, so please check if your
torchtext and torchdata are supported for your torch version 
from these links:
https://github.com/pytorch/text/blob/main/README.rst
https://github.com/pytorch/data/blob/main/README.md

In 0.6.0 version of torchdata, there is an issue with the 
Multi30k DataLoader. It may still be a problem in the future,
so I suggest you use the following configuration to run this script:

torchdata v0.5.1
torchtext v0.14.0
torch     v1.13.1

you can download them like this:

pip install torch==1.13.1 torchtext==0.14.0 torchdata==0.5.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math

from typing import Tuple
from torch import Tensor
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from timeit import default_timer as timer

'''Hyperparameters'''

''' Data Parameters '''
MAX_LEN = 64 # maximum number of tokens a sentence can contain
VOCAB_SIZE = 32768 # dictionary size
"""
Do not change PAD_IDX, BOS_IDX and EOS_IDX if you are 
using the tokenizer which you obtained using the script 
train_tokenizer.py
"""
PAD_IDX, BOS_IDX, EOS_IDX = 1, 2, 3 # special tokens
BATCH_SIZE = 32 # batch_size for training, adjust it according to your VRAM capacity
SRC_LANGUAGE = 'de' # language to be translated
TGT_LANGUAGE = 'en' # language the model translates to
''' Model Parameters '''
D_MODEL = 512 # model size
D_H = 8 # number of heads
D_FF = 2048 # feedforward linear layer size
EMBEDDING_SIZE = 512 # embedding size
N = 3 # number of encoder and decoder layers
''' Training Parameters '''
LR = 0.0001 # learning rate
BETAS = (0.9, 0.98) # for Adam optimizer
EPS = 1e-9 # for Adam optimizer
DROPOUT = 0.2 # dropout rate
EPOCHS = 18 # number of epochs
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() 
                      else "cpu") # selects gpu if available

torch.manual_seed(0)

train_iter = Multi30k(split='train', 
                      language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
test_iter = Multi30k(split='valid', 
                     language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))


"""
Use the train_tokenizer.py script to train a tokenizer on the Multi30k
dataset which this script uses for training the transformer model.

Then save it as a file and load it using:

Tokenizer.from_file(path)

You can find all the necessary information for training the tokenizer
inside the script named train_tokenizer.py 

"""
tokenizer_path = " " # path to the trained tokenizer file

tokenizer = Tokenizer.from_file(tokenizer_path)

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_enc = tokenizer.encode(src_sample.rstrip("\n"))
        src_batch.append(torch.tensor(src_enc.ids))

        tgt_enc = tokenizer.encode(tgt_sample.rstrip("\n"))
        tgt_batch.append(torch.tensor(tgt_enc.ids))
    return torch.stack(src_batch), torch.stack(tgt_batch)

train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, 
                              collate_fn=collate_fn, drop_last=True)

test_dataloader = DataLoader(test_iter, batch_size=BATCH_SIZE, 
                             collate_fn=collate_fn, drop_last=True)

class Embedding(nn.Module):
    def __init__(self, vocab_size=32678, embedding_size=512, pad_mask=1):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        
        self.emb = nn.Embedding(num_embeddings=vocab_size, 
                                embedding_dim=embedding_size, 
                                padding_idx=pad_mask, device=DEVICE)

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.embedding_size)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, d_h=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.d_h = d_h
        self.d_k = d_model // d_h

        self.linears = nn.ModuleList([nn.Linear(
            d_model, self.d_k
            ) for _ in range(d_h*3)])
        
        self.dropout = nn.Dropout(dropout)
        self.Linear = nn.Linear(d_model, d_model)
        self.normalize1 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.normalize2 = nn.LayerNorm(d_model)

    def forward(self, x, x_mask=None):
        multi_head = []
        for i in range(self.d_h):
            query = self.linears[3*i](x)
            key = self.linears[3*i + 1](x)
            value = self.linears[3*i + 2](x)
            scaledDotProd = (query @ key.transpose(-1, -2)) / math.sqrt(self.d_k)
            if x_mask is not None:
                scaledDotProd = scaledDotProd.masked_fill(x_mask==0, float('-inf'))
            soft = F.softmax(scaledDotProd, dim=-1)
            soft = self.dropout(soft)
            attn =  soft @ value
            multi_head.append(attn)
        selfAttn = self.Linear(torch.cat((multi_head), -1))
        addNorm = self.normalize1(x + selfAttn)
        encoderOutput = self.normalize2(x + self.feed_forward(addNorm))
        return encoderOutput
    
class EncoderStack(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, d_h=8, 
                 dropout=0.1, N=6):
        super(EncoderStack, self).__init__()

        self.encoders = nn.ModuleList([EncoderLayer(
            d_model, d_ff, d_h, dropout
            ) for _ in range(N)]) # Stacking Encoder Layer N Times

    def forward(self, x, x_mask=None):
        for encoder in self.encoders:
            x = encoder(x, x_mask)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, d_h=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.d_h = d_h
        self.d_k = d_model // d_h

        self.linears = nn.ModuleList([nn.Linear(
            d_model, self.d_k
            ) for _ in range(d_h*3)])
        
        self.dropout1 = nn.Dropout(dropout)
        self.firstLinear = nn.Linear(d_h * self.d_k, d_model)
        self.normalize1 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.secondLinear = nn.Linear(d_h * d_model, d_model)
        self.normalize2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.normalize3 = nn.LayerNorm(d_model)

    def forward(self, x, y, y_mask=None):
        multi_head1 = []
        multi_head2 = []

        # FIRST ATTENTION LAYER
        ''' Same as encoder, but here we have tgt(target) as the decoder's input '''
        for i in range(self.d_h):
            query = self.linears[3*i](y)
            key = self.linears[3*i+1](y)
            value = self.linears[3*i+2](y)
            scaledDotProd = (query @ key.transpose(-1, -2)) / math.sqrt(self.d_k)
            if y_mask is not None:
                scaledDotProd = scaledDotProd.masked_fill(y_mask==0, float('-inf'))
            soft = F.softmax(scaledDotProd, dim=-1)
            soft = self.dropout1(soft)
            attn =  soft @ value
            multi_head1.append(attn)
        selfAttn = self.firstLinear(torch.cat((multi_head1), dim=-1))
        addNorm1 = self.normalize1(y + selfAttn)

        # SECOND ATTENTION LAYER
        ''' Attention layer, instead of V and K matrices, we use the output of the encoder '''
        for i in range(self.d_h):
            scaledDotProd = (addNorm1 @ x.transpose(-1, -2)) / math.sqrt(self.d_k)
            soft = F.softmax(scaledDotProd, dim=-1)
            soft = self.dropout2(soft)
            attn = soft @ x
            multi_head2.append(attn)
        crossAttn = self.secondLinear(torch.cat((multi_head2), dim=-1))
        addNorm2 = self.normalize2(y + crossAttn)
        decoderOutput = self.normalize3(y + self.feed_forward(addNorm2))
        return decoderOutput
    
class DecoderStack(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, d_h=8, 
                 dropout=0.1, N=6):
        super(DecoderStack, self).__init__()
        
        self.decoders = nn.ModuleList([DecoderLayer(
            d_model, d_ff, d_h, dropout
            ) for _ in range(N)]) # Stacking Decoder Layer N Times

    def forward(self, x, y, y_mask=None):
        for decoder in self.decoders:
            y = decoder(x, y, y_mask)
        return y
    
class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
    
    def forward(self, batch_dim, seq_len):
        
        mask = torch.ones((batch_dim,seq_len,seq_len),device=DEVICE)

        mask = torch.tril(mask)
        
        return mask
    
class Positional_Encoding(nn.Module):
    def __init__(self, embedding_size=512, n=10000):
        super(Positional_Encoding, self).__init__()
        self.embedding_size = embedding_size
        self.n = n
    def forward(self, seq_len):
        P = torch.zeros(seq_len, self.embedding_size, device=DEVICE)
        for k in range(seq_len):
            for i in range(self.embedding_size // 2):
                denominator = math.pow(self.n, 2*i/self.embedding_size)
                P[k, 2*i] = math.sin(k/denominator)
                P[k, 2*i+1] = math.cos(k/denominator)
        return P
    
class GenerateLogits(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(GenerateLogits, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return self.linear(x)
    
class Transformer(nn.Module):
    def __init__(self, d_model=512, d_h = 8, d_ff=2048, 
                 embedding_size=512, vocab_size=32768, 
                 dropout=0.1, num_coder_layers=6):
        super(Transformer, self).__init__()
        self.embed = Embedding(vocab_size, embedding_size, pad_mask=1)
        self.positional = Positional_Encoding(embedding_size, 10000)
        self.masking = Mask()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.encoderStack = EncoderStack(
            d_model, d_ff, d_h, 
            dropout, num_coder_layers)
        
        self.decoderStack = DecoderStack(
            d_model, d_ff, d_h, 
            dropout, num_coder_layers)
        
        self.generate = GenerateLogits(d_model, vocab_size)

    def forward(self, x: Tensor, y: Tensor):
        assert x.shape[0] == y.shape[0]
        batch_dim = x.shape[0]
        src_seq_len, tgt_seq_len = x.shape[1], y.shape[1]
        x_pos_encoding = self.positional(src_seq_len)
        y_pos_encoding = self.positional(tgt_seq_len)
        x_mask = self.masking(batch_dim, src_seq_len)
        y_mask = self.masking(batch_dim, tgt_seq_len)
        x, y = self.embed(x), self.embed(y)
        x, y = x_pos_encoding + x, y_pos_encoding + y
        x, y = self.dropout1(x), self.dropout2(y)
        encoderOutput = self.encoderStack(x, x_mask)
        decoderOutput = self.decoderStack(encoderOutput, y, y_mask)
        logits = self.generate(decoderOutput)
        return logits
    
model = Transformer(d_model=D_MODEL, d_h=D_H, d_ff=D_FF, 
                    embedding_size=EMBEDDING_SIZE, 
                    vocab_size=VOCAB_SIZE, dropout=DROPOUT, 
                    num_coder_layers=N)

model = model.to(DEVICE)

def initalize_parameters():
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

@torch.no_grad()
def compute_loss(model: nn.Module, test_data: DataLoader, 
                      padding_index: int):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=padding_index)
    loss = []
    for src, tgt in test_data:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in = tgt[:, :-1] # inputs shifted left
        tgt_out = tgt[:, 1:] # labels shifted right
        logits = model(src, tgt_in).permute(0, 2, 1)
        loss.append(criterion(logits, tgt_out).item())
    average_loss = np.average(loss)
    model.train()
    return average_loss

def train(model: nn.Module, train_data: DataLoader, 
          test_data: DataLoader, learning_rate: int, 
          padding_idx: int, epoch_num: int, betas: Tuple, eps: int):
    
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                           betas=betas, eps=eps)
    
    epoch_test_loss = []
    for epoch in range(epoch_num):
        print("#"*67)
        print(f'{"#"*20}Training begins for epoc {epoch:>2}{"#"*20}')
        print("#"*67)
        start_time = timer()
        for _, (src, tgt) in enumerate(train_data):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in = tgt[:, :-1] # inputs shifted left
            logits = model(src, tgt_in).permute(0, 2, 1)
            optimizer.zero_grad()
            tgt_out = tgt[:, 1:] # labels shifted right
            loss = criterion(logits, tgt_out)
            loss.backward()
            optimizer.step()
        end_time = timer()
        test_loss = compute_loss(model, test_data, padding_idx)
        str1 = f'Training is complete for epoch {epoch:>2}, '
        str2 = f'loss = {test_loss:>5.3f}'
        str3 = f'Time = {end_time - start_time:>3.3f}s'
        print(str1 + str2)
        print(str3)
        epoch_test_loss.append(test_loss)
        print("-"*67)
    print("\n Training is complete \n")
    plt.plot(epoch_test_loss, color='purple')
    plt.title("Loss Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def start_training():
    initalize_parameters()
    train(model, train_dataloader, test_dataloader, 
          learning_rate=LR, padding_idx=PAD_IDX, 
          epoch_num=EPOCHS, betas=BETAS, eps=EPS)
    
start_training()

torch.save(model.state_dict(), "modelparameters.pth")
