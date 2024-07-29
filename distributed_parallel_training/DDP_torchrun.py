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
            save_dir: str="saved_models/DDP_saved_models/torchrun",
            save_every: int=5,
            ) -> None:
        ###############################################
        #                                             #
        #       🐯 Change no.1: envvariables 🐯       #
        #                                             #
        ###############################################
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("LOCAL_RANK not found in the environment.")
        self.device = int(os.environ["LOCAL_RANK"])

        self.dataloader = dataloader
        self.optimizer = optimizer

        batch_size = len(next(iter(self.dataloader)))
        self.num_batches = math.ceil(len(self.dataloader.dataset) / batch_size)

        ##############################################
        #                                            #
        #       🐯 Change no.2: save & load 🐯       #
        #                                            #
        ##############################################
        self.save_path = os.path.join(root, save_dir, "snapshot.pt")
        self.save_every = save_every
        print("This part 5 is ok")
        self.model = model.to(self.device)
        self.checkpoint_epoch = 0
        if os.path.exists(self.save_path):
            self._load_checkpoint()
        self.model = DDP(self.model, device_ids=[self.device, ])
        print("This part 6 is ok")
    ##############################################
    #                                            #
    #       🐯 Change no.2: save & load 🐯       #
    #                                            #
    ##############################################
    def _save_checkpoint(self, epoch_id):
        snapshot = {
                "MODEL_STATE": self.model.module.state_dict(),
                "EPOCH_ID": epoch_id,
                }
        torch.save(snapshot, self.save_path)
        print(f"Epoch {epoch_id:02d} | Checkpoint saved.")

    def _load_checkpoint(self):
        loc = f"cuda:{self.device}"
        snapshot = torch.load(self.save_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.checkpoint_epoch = snapshot["EPOCH_ID"]
        print(f"Resumed training from snapshot at Epoch {self.checkpoint_epoch}")


    def _train_batch(self, X: torch.Tensor, y: torch.Tensor):
        self.optimizer.zero_grad()
        out = self.model(X)
        loss_v = F.cross_entropy(out, y)
        loss_v.backward()
        self.optimizer.step()
        return loss_v.item()

    def _train_epoch(self, epoch_id: int):
        all_loss = 0.0
        self.dataloader.sampler.set_epoch(epoch_id)
        for X, y in self.dataloader:
            X, y = X.to(self.device), y.to(self.device)
            loss_v = self._train_batch(X, y)
            all_loss += loss_v
        return all_loss / self.num_batches

    def train(self, num_epochs: int):
        ##############################################
        #                                            #
        #       🐯 Change no.2: save & load 🐯       #
        #                                            #
        ##############################################
        for e in range(self.checkpoint_epoch - 1, num_epochs):
            loss_per_epoch = self._train_epoch(e + 1)
            print(f"Epoch [{e + 1:02d}/{num_epochs:02d}]: device cuda:{self.device}, loss {loss_per_epoch:.4f}")
            
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

# init process

###############################################
#                                             #
#       🐯 Change no.1: envvariables 🐯       #
#                                             #
###############################################
def init_process():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("LOCAL_RANK not found in the environment.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    print("This part 2 is ok")
    init_process_group(backend="nccl")


# process function

###############################################
#                                             #
#       🐯 Change no.1: envvariables 🐯       #
#                                             #
###############################################
def run(lr, num_epochs, batch_size):
    print("This part 1 is ok")
    init_process()
    print("This part 3 is ok")
    train_loader, model, optimizer = prepare_train_objs(lr, batch_size)
    print("This part 4 is ok")
    trainer = Trainer(train_loader, model, optimizer)
    trainer.train(num_epochs)
    destroy_process_group()



# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epochs',  default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()
    print("This part 0 is ok")
    #################################################
    #                                               #
    #       🐯 Change no.3: Run the script 🐯       #
    #                                               #
    #################################################
    run(args.lr, args.num_epochs, args.batch_size)
