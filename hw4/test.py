import sys
sys.path.append('./python')
sys.path.append('./apps')
import needle as ndl
from models import ResNet9
from simple_training import train_cifar10, evaluate_cifar10

import time

device = ndl.cpu()
dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
dataloader = ndl.data.DataLoader(\
         dataset=dataset,
         batch_size=128,
         shuffle=True,
         device=device,
         dtype="float32")
# collate_fn=ndl.data.collate_ndarray,
model = ResNet9(device=device, dtype="float32")

t0 = time.time()

train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
      lr=0.001, weight_decay=0.001)

t1 = time.time()

print("train time: ", t1-t0)

evaluate_cifar10(model, dataloader)

t2= time.time()

print("train time: ", t2-t1)