"""
Advanced TrainWatch Example: ResNet + Fashion-MNIST

This example demonstrates TrainWatch with:
- Deeper ResNet-inspired Architecture
- Data augmentation
- Learning rate scheduling
- Gradient accumulation
- More complex training scenarios

TrainWatch will help catch:
- Memory leaks from gradient accumulation
- Training istability from aggressive aygmentation
- DataLoader bottlenecks
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from trainwatch import Watcher

# ResNet-inspired block
class ResidualBlock(nn.Module):
    """Basic residual block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet18Fashion(nn.Module):
    """ResNet-18 inspired model for Fashion-MNIST (28x28 grayscale)"""
    def __init__(self, num_classes=10):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

        # hyperparameters
        batch_size = 128
        num_epochs = 5
        learning_rate = 0.01
        momentum = 0.9
        weight_decay = 1e-4

        # data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # no augmentation for validation
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Fashion-MNIST dataset
        print("Loading Fashion-MNIST dataset...")
        trainset = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )

        testset = datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=test_transform
        )

        # dataloaders
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )

        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        print(f"Training samples: {len(trainset)}")
        print(f"Testing samples: {len(testset)}")
        print(f"Batches for epoch: {len(trainloader)}\n")

        # model
        model = ResNet18Fashion(num_classes=10).to(device)

        # count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")

        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # learning rate scheduler
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

        # Initialize TrainWatch with all features enabled
        print("Starting training with TrainWatch monitoring...")
        print("=" * 60)

        watcher = Watcher(
            window=20,             # moving average window
            print_every=50,        # print every 50 steps
            show_gpu=torch.cuda.is_available(),
            warn_on_leak=True,      # detect memory leaks
            warn_on_bottleneck=True, # detect DataLoader bottlenecks
            warn_on_variance=True,  # detect training instability
        )

        # training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

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

                running_loss += loss.item()

            # end of epoch
            watcher.epoch_end()

            # learning rate scheduling
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}\n")

            # validation
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0

            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            avg_val_loss = val_loss / len(testloader)

            print(f"Validation - Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")
            print("=" * 60 + "\n")

        print("Training complete!")

        # final model info
        if torch.cuda.is_available():
            print(f"\nFinal GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

if __name__ == "__main__":
    main()





