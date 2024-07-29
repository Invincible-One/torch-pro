# packages and configurations
import os
import sys
import argparse
import math

import torch
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

root = "/scratch/ym2380/"



# Trainer
class Trainer:
    def __init__(
            self,
            dataloader: DataLoader,
            model: nn.Module,
            optimizer: optim.Optimizer,
            device: torch.device, 
            save_every: int=5,
            ) -> None:
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.save_every = save_every

        batch_size = len(next(iter(self.dataloader)))
        self.n_batch = math.ceil(len(self.dataloader.dataset) / batch_size)
        self.saving_dir = os.path.join(root, "saved_models/temp")

    def _save_checkpoint(self, epoch_id):
        ckp = self.model.state_dict()
        torch.save(ckp, os.path.join(self.saving_dir, f"{epoch_id}.pth"))
        print(f"Epoch {epoch_id:02d} | Checkpoint saved.")

    def _train_batch(self, X: torch.Tensor, y: torch.Tensor):
        self.optimizer.zero_grad()
        out = self.model(X)
        loss_v = F.cross_entropy(out, y)
        loss_v.backward()
        self.optimizer.step()
        return loss_v.item()

    def _train_epoch(self):
        all_loss = 0.0
        for X, y in self.dataloader:
            X, y = X.to(self.device), y.to(self.device)
            loss_v = self._train_batch(X, y)
            all_loss += loss_v
        return all_loss / self.n_batch

    def train(self, n_epoch: int):
        for e in range(n_epoch):
            loss_per_epoch = self._train_epoch()
            print(f"Epoch [{e + 1:02d}/{n_epoch:02d}]: device {self.device}, loss {loss_per_epoch:.4f}")
            if (e + 1) % self.save_every == 0:
                self._save_checkpoint(e + 1)



# helper function
def prepare_train_objs(
        lr: float,
        batch_size: int,
        device: torch.device,
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
            shuffle=True,
            num_workers=1,
            pin_memory=True,)
        
    model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10),
            ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    return dataloader, model, optimizer



# process function
def run(lr, n_epoch, batch_size, device):
    train_loader, model, optimizer = prepare_train_objs(lr, batch_size, device)
    trainer = Trainer(train_loader, model, optimizer, device)
    trainer.train(n_epoch)



# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--n_epoch',  default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()

    device = torch.device("cuda:0")

    run(args.lr, args.n_epoch, args.batch_size, device)
