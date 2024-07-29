# packages and configurations
import os
import sys
import argparse
import math

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

##########################################
#                                        #
#       🐯 Change no.1: Imports 🐯       #
#                                        #
##########################################
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

root = "/scratch/ym2380/"



# Trainer
class Trainer:
    def __init__(
            self,
            dataloader: DataLoader,
            model: nn.Module,
            optimizer: optim.Optimizer,
            device: int, 
            save_every: int=5,
            ) -> None:
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.save_every = save_every

        #############################################################
        #                                                           #
        #       🐯 Change no.3: Constructing the DDP model 🐯       #
        #                                                           #
        #############################################################
        model = model.to(self.device)
        self.model = DDP(model, device_ids=[self.device, ])

        batch_size = len(next(iter(self.dataloader)))
        self.num_batches = math.ceil(len(self.dataloader.dataset) / batch_size)
        self.saving_dir = os.path.join(root, "saved_models/temp")

    def _save_checkpoint(self, epoch_id):
        #####################################################
        #                                                   #
        #       🐯 Change no.5: Saving checkpoints 🐯       #
        #          Step 1, Using model.module               #
        #                                                   #
        #####################################################
        ckp = self.model.module.state_dict()

        torch.save(ckp, os.path.join(self.saving_dir, f"{epoch_id}.pth"))
        print(f"Epoch {epoch_id:02d} | Checkpoint saved.")

    def _train_batch(self, X: torch.Tensor, y: torch.Tensor):
        self.optimizer.zero_grad()
        out = self.model(X)
        loss_v = F.cross_entropy(out, y)
        loss_v.backward()
        self.optimizer.step()
        return loss_v.item()

    ##########################################################################
    #                                                                        #
    #       🐯 Change no.4: Distributing data w. DistributedSampler 🐯       #
    #          Step 2, Setting epoch before iterating loaders                #
    #                                                                        #
    ##########################################################################
    def _train_epoch(self, epoch_id: int):
        all_loss = 0.0
        self.dataloader.sampler.set_epoch(epoch_id)
        for X, y in self.dataloader:
            X, y = X.to(self.device), y.to(self.device)
            loss_v = self._train_batch(X, y)
            all_loss += loss_v
        return all_loss / self.num_batches

    def train(self, num_epochs: int):
        for e in range(num_epochs):
            loss_per_epoch = self._train_epoch(e + 1)
            print(f"Epoch [{e + 1:02d}/{num_epochs:02d}]: device cuda:{self.device}, loss {loss_per_epoch:.4f}")
            
            #####################################################
            #                                                   #
            #       🐯 Change no.5: Saving checkpoints 🐯       #
            #          Step 2, Merely saving one model          #
            #                                                   #
            #####################################################
            if self.device == 0 and (e + 1) % self.save_every == 0:
                self._save_checkpoint(e + 1)



# helper function
def prepare_train_objs(
        lr: float,
        batch_size: int,
        ):
    dataset = datasets.MNIST(
            root=os.path.join(root, "data/mnist"),
            train=True,
            download=True,
            transform=v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                ])
            )
    ##########################################################################
    #                                                                        #
    #       🐯 Change no.4: Distributing data w. DistributedSampler 🐯       #
    #          Step 1, Using DistributedSampler                              #
    #                                                                        #
    ##########################################################################
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=DistributedSampler(dataset),
            num_workers=2,
            pin_memory=True,)
        
    model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10),
            )

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    return dataloader, model, optimizer

#########################################################
#                                                       #
#       🐯 Change no.2: Initializing processes 🐯       #
#                                                       #
#########################################################
# init process
def init_process(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT")
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


# process function

########################################################$$#############
#                                                                     #
#       🐯 Change no.6: Running the distributed training job 🐯       #
#          Step 2, run                                                #
#                                                                     #
#######################################################################
def run(device, world_size, lr, num_epochs, batch_size):
    init_process(rank=device, world_size=world_size)
    train_loader, model, optimizer = prepare_train_objs(lr, batch_size)
    trainer = Trainer(train_loader, model, optimizer, device)
    trainer.train(num_epochs)
    destroy_process_group()



# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epochs',  default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()

    ########################################################$$#############
    #                                                                     #
    #       🐯 Change no.6: Running the distributed training job 🐯       #
    #          Step 1, main                                               #
    #                                                                     #
    #######################################################################
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, args.lr, args.num_epochs, args.batch_size,), nprocs=world_size)
