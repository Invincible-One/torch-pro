# packages and configurations
import os
import random
import math

import torch
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchvision.datasets import datasets
from torchvision.transforms import v2
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F

mp.set_start_method("spawn", force=True)
torch.manual_seed(1)



# Helper classes and functions
class PartitionHelper():
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        return self.data[self.indices[index]]

class DataPartitioner():
    def __init__(self, data, portions=[0.7, 0.2, 0.1]):
        self.data = data
        self.partitions = list()

        data_len = len(self.data)
        raw_indices = torch.randperm(data_len)
        for p in portions:
            self.portions.append(raw_indices[round(p)])
            raw_indices = raw_indices[round(p):]

    def use(self, partition_index):
        return PartitionHelper(self.data, self.partitions[partition_index])

def partition_mnist():
    data = datasets.MNIST(
        root="/scratch/ym2380/data/mnist",
        train=True,
        download=True,
        transform=v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.1307,), (0.3081,))
        ]),
    )

    size = dist.get_world_size()
    bsz = 128 // size
    portions = [1.0 / size for _ in range(size)]
    partitioner = DataPartitioner(data, portions)
    partition = partitioner.use(dist.get_rank())
    loader = DataLoader(partition, batch_size=bsz, shuffle=True)
    return loader, bsz



# init process
def init_process(rank, size, fn, backend="gloo"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)



# process functions

## Point to Point Communication

### blocking code
def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)
    else:
        dist.recv(tensor=tensor, src=0)
    print("Rank: ", rank, " has data ", tensor[0])

### non-blocking code
def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        req = dist.isend(tensor=tensor, dst=1)
        print("rank 0 started sending")
    else:
        req = dist.irecv(tensor=tensor, src=0)
        print("rank 1 started receiving")
    req.wait()
    print("Rank: ", rank, " has data ", tensor[0])

## Collective Communication
def run(rank, size):
    group = dist.new_group([0, 1])
    if rank == 0:
        tensor = torch.Tensor([5])
    else:
        tensor = torch.Tensor([4])
    #tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])

## Distributed Sync SGD

### average gradient
def average_gradient(model):
    size = dist.get_world_size()
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

### main
def run(rank, size):
    torch.manual_seed(1234)
    train_loader, bsz = partition_mnist
    model = torchvision.alexnet()
    optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.5)
    n_batches = math.ceil(len(train_loader.dataset) / bsz)
    for epoch in range(10):
        epoch_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            out = model(X)
            loss_v = F.nll_loss(out, y)
            epoch_loss += loss_v.item()
            loss.backward()
            average_gradient(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / n_batches)



# main
if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
