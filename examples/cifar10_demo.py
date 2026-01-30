"""
TrainWatch Demo with CIFAR-10

This example shows how to use TrainWatch with a real PyTorch training loop

Setup:
1. Install TrainWatch: pip install -e .
2. Install torchvision: pip install torchvision
3. Run: python examples/cifar10_demo.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.trainwatch import Watcher

# simple CNN for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=2 # try setting to 0 to see DataLoader bottleneck warning
    )

    # model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize TrainWatch
    watcher = Watcher(
        window=20,
        print_every=50, # print every 50 steps
        show_gpu=torch.cuda.is_available(),
        warn_on_leak=True,
        warn_on_bottleneck=True,
        warn_on_variance=True,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    # training loop
    num_epochs = 3

    print("Starting training with TrainWatch monitoring...")
    print("=" * 60)

    for epoch in range(num_epochs):
        model.train()

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TrainWatch monitoring - just one line!
            watcher.step(loss=loss.item())

        # end of epoch
        watcher.epoch_end()

    print("\nTraining complete!")

if __name__ == "__main__":
    main()
