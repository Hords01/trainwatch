"""
TrainWatch Example: DenseNet121 from PyTorch Models
This example demonstrates TrainWatch with torchvision.models:
- DenseNet121 (weights=None) - training from scratch
- CIFAR-10 dataset with resize to 224x224 (ImageNet size)
- Real PyTorch architecture without custom modifications
- Gradient clipping
- Learning rate scheduling

DensNet is a production-grade architecture from PyTorch's model zoo,
making it a perfect example of monitoring real-world models

TrainWatch will help monitor:
- Higher memory usage (~800-1000MB VRAM)
- Training stability with deeper architecture
- Gradient flow in dense connections
- DataLoader performance with larger images
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

from trainwatch import Watcher

def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # hyperparameters
    batch_size = 32  # Smaller batch due to larger images
    num_epochs = 5
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    gradient_clip = 1.0

    # Data transformations - resize CIFAR to ImageNet size (224x224)
    transform_train = transforms.Compose([
        transforms.Resize(224),  # Resize to ImageNet size
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    trainset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    testset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
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
    print(f"Test samples: {len(testset)}")
    print(f"Batches per epoch: {len(trainloader)}")
    print(f"Image size: 224x224 (resized from 32x32)\n")

    # Load DenseNet121 from PyTorch model zoo (weights=None - training from scratch!)
    print("Creating DenseNet121 model from torchvision.models...")
    model = models.densenet121(weights=None)

    # Modify only the classifier for CIFAR-10 (10 classes instead of 1000)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 10)

    model.to(device)

    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: DenseNet121 (torchvision.models)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.0f}MB\n")

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2
    )

    # Initialize TrainWatch
    print("Starting training with TrainWatch monitoring...")
    print("=" * 60)

    watcher = Watcher(
        window=20,
        print_every=100, # print every 100 steps
        show_gpu=torch.cuda.is_available(),
        warn_on_leak=True,
        warn_on_bottleneck=True,
        warn_on_variance=True,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    # training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping | preventing exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            # TrainWatch monitoring - just one line!
            watcher.step(loss=loss.item())

            # track accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # end of epoch
        train_acc = 100. * correct / total
        watcher.epoch_end()

        # learning rate info
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Accuracy: {train_acc:.2f}% | LR: {current_lr:.6f}")

        # validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        avg_test_loss = test_loss / len(testloader)

        print(f"Test Accuracy: {test_acc:.2f}% | Test Loss: {avg_test_loss:.4f}")
        print("=" * 60 + "\n")

        # step the scheduler
        scheduler.step()

    print("Training complete!")

    # final status
    if torch.cuda.is_available():
        final_vram = torch.cuda.memory_allocated(device) / 1024**2
        max_vram = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\nFinal GPU Memory: {final_vram:.0f}MB")
        print(f"Peak GPU Memory: {max_vram:.0f}MB")

if __name__ == "__main__":
    main()