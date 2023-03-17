"""
Ahmet Furkan Karacık
afurkank @ github

You need to install torch, tokenizers, torchtext and torchdata.
Some versions of torchtext and torchdata may not be compatible
with certain versions of torch, so please check if your
torchtext and torchdata are supported for your torch version 
from these links:
https://github.com/pytorch/text/blob/main/README.rst
https://github.com/pytorch/data/blob/main/README.md

In 0.6.0 version of torchdata, there is an issue with the 
DataLoader. It may still be a problem in the future,
so I suggest you use the following configuration to run this script:

torchdata v0.5.1
torchtext v0.14.1
torch     v1.13.1

you can download them like this:

pip install torch==1.13.1 torchtext==0.14.1 torchdata==0.5.1
pip install tokenizers

If you are going to use newer versions, you might need to install
portalocker like this:

pip install 'portalocker>=2.0.0'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

from typing import Tuple
from torch import Tensor
from tokenizers import Tokenizer

'''Hyperparameters'''

''' Data Parameters '''
MAX_LEN = 64
VOCAB_SIZE = 32768
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
''' Model Parameters '''
D_MODEL = 512
D_H = 8
D_FF = 2048
EMBEDDING_SIZE = 512
N = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer_path = " "       # path to the already trained tokenizer
model_params_path = " "    # path to the already trained model's parameters

tokenizer = Tokenizer.from_file(tokenizer_path)

class Embedding(nn.Module):
    def __init__(self, vocab_size=32678, embedding_size=512, pad_mask=1):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=pad_mask, device=DEVICE)

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.embedding_size)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, d_h=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.d_h = d_h
        self.d_k = d_model // d_h
        self.linears = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(d_h*3)])
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
        self.encoders = nn.ModuleList([EncoderLayer(d_model, d_ff, d_h, dropout) for _ in range(N)]) # Stacking Encoder Layer N Times

    def forward(self, x, x_mask=None):
        for encoder in self.encoders:
            x = encoder(x, x_mask)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, d_h=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.d_h = d_h
        self.d_k = d_model // d_h
        self.linears = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(d_h*3)])
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
        self.decoders = nn.ModuleList([DecoderLayer(d_model, d_ff, d_h, dropout) for _ in range(N)]) # Stacking Decoder Layer N Times

    def forward(self, x, y, y_mask=None):
        for decoder in self.decoders:
            y = decoder(x, y, y_mask)
        return y
    
class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
    
    def forward(self, batch_dim, seq_len):
        mask = torch.tril(torch.ones((batch_dim, seq_len, seq_len), device=DEVICE))
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
        self.encoderStack = EncoderStack(d_model, d_ff, d_h, dropout, num_coder_layers)
        self.decoderStack = DecoderStack(d_model, d_ff, d_h, dropout, num_coder_layers)
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
                    vocab_size=VOCAB_SIZE, dropout=0, 
                    num_coder_layers=N)
model = model.to(DEVICE)

model.load_state_dict(torch.load(map_location=torch.device(DEVICE), f=model_params_path))

@torch.no_grad()
def translate(model: nn.Module, source: Tensor, start_token: int, 
              stop_token: int, seq_len: int):
    model.eval()

    source = source.to(DEVICE)
    src_batch_dim = source.shape[0]
    src_seq_len = source.shape[1]

    src_mask = model.masking(src_batch_dim, src_seq_len)
    src_pos_enc = model.positional(src_seq_len)
    src_embedding = model.embed(source)

    encoderInput = src_embedding + src_pos_enc

    encoderOutput = model.encoderStack(encoderInput, src_mask)

    sequence = torch.tensor([[start_token]], dtype=torch.int32, device=DEVICE)

    for _ in range(seq_len):

        tgt_batch_dim = sequence.shape[0]
        tgt_seq_len = sequence.shape[1]

        embedded_sequence = model.embed(sequence)
        seq_pos_enc = model.positional(tgt_seq_len)
        decoder_mask = model.masking(tgt_batch_dim, tgt_seq_len)

        decoderInput = embedded_sequence + seq_pos_enc
    
        out = model.decoderStack(encoderOutput, decoderInput, decoder_mask)

        logits = model.generate(out[:, -1, :])
        generated_word_id = torch.argmax(logits, dim=-1)

        sequence = torch.cat((sequence, torch.tensor([[generated_word_id.item()]], 
                                                     dtype=torch.int32, device=DEVICE)), dim=1)

        if generated_word_id == stop_token:
            break

    translation_ids = sequence.cpu().numpy()

    translation = tokenizer.decode(translation_ids.squeeze())
    return translation

def translate_sentence(german: str):
    encoded_german = tokenizer.encode(german)
    german_id_arr = encoded_german.ids

    tgt_input = torch.tensor(german_id_arr).unsqueeze(0)
    translation = translate(model, tgt_input, BOS_IDX, EOS_IDX, MAX_LEN)
    print("Model translation: ", translation)

german_str = "Der Hund wurde von einem Bus angefahren."
translate_sentence(german_str)
