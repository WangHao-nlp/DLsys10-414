import sys
sys.path.append('./python')
sys.path.append('./apps')

import needle as ndl
#sys.path.append('./apps')
from models import LanguageModel
from simple_training import train_ptb, evaluate_ptb


import time

device = ndl.cuda()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=16, device=device, dtype="float32")
model = LanguageModel(100, len(corpus.dictionary), hidden_size=100, num_layers=2, seq_model='lstm', device=device)


t0 = time.time()
train_ptb(model, train_data, seq_len=40, n_epochs=10, device=device)

t1 = time.time()

print("train time: ", t1-t0)

evaluate_ptb(model, train_data, seq_len=40, device=device)

t2= time.time()

print("train time: ", t2-t1)

