"""
TrainWatch Example: Memory Leak Detection Demo

This example demonstrates TrainWatch's memory leak detection capability.
It shows two scenarios:
1. CORRECT training (no leak) - TrainWatch shows clean VRAM delta
2. INCORRECT training (with leak) - TrainWatch detects and warns about memory leak

Common causes of memory leaks in PyTorch:
- Not detaching tensors before storing them
- Accumulating computation graphs in lists
- Storing loss values without .item()
- Not clearing gradients properly

TrainWatch will catch these issues automatically!
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys

from trainwatch import Watcher


# Simple CNN (same as cifar10_simple.py)
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


def train_correct(device):
    """
    CORRECT training - no memory leak.

    Best practices:
    - Use .item() to extract scalar values
    - Don't store tensors unnecessarily
    - Clear gradients properly
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: CORRECT Training (No Memory Leak)")
    print("=" * 70)
    print("Best practices:")
    print("  ✓ Using loss.item() to extract scalars")
    print("  ✓ Not storing tensors in lists")
    print("  ✓ Properly clearing gradients")
    print("=" * 70 + "\n")

    # Setup
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # TrainWatch
    watcher = Watcher(
        print_every=100,
        show_gpu=torch.cuda.is_available(),
        warn_on_leak=True,
        warn_on_bottleneck=False,
        warn_on_variance=False
    )

    # Training loop
    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ✅ CORRECT: Using .item() - this extracts a Python scalar
            watcher.step(loss=loss.item())

        watcher.epoch_end()

    print("\n✅ Training complete - NO MEMORY LEAK detected!")
    print("VRAM delta should be ~0MB (small variations are normal)\n")


def train_with_leak(device):
    """
    INCORRECT training - intentional memory leak!

    Common mistake:
    - Storing tensors (with computation graph) in a list
    - This prevents garbage collection
    - VRAM keeps growing each epoch
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: INCORRECT Training (With Memory Leak)")
    print("=" * 70)
    print("Common mistake:")
    print("  ✗ Storing loss tensors (not .item()) in a list")
    print("  ✗ This keeps computation graphs in memory")
    print("  ✗ VRAM keeps growing!")
    print("=" * 70 + "\n")

    # Setup
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # TrainWatch
    watcher = Watcher(
        print_every=100,
        show_gpu=torch.cuda.is_available(),
        warn_on_leak=True,
        warn_on_bottleneck=False,
        warn_on_variance=False
    )

    # ❌ INTENTIONAL MISTAKE: Store loss tensors (not .item()!)
    loss_history = []  # This will cause a memory leak!

    # Training loop
    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ❌ MISTAKE: Storing the tensor (keeps computation graph alive!)
            loss_history.append(loss)  # This causes memory leak!

            # Still pass .item() to TrainWatch (for monitoring)
            watcher.step(loss=loss.item())

        watcher.epoch_end()

    print("\n⚠️  Training complete - MEMORY LEAK PRESENT!")
    print(f"Stored {len(loss_history)} loss tensors in memory!")
    print("Note: TrainWatch warns when VRAM grows >10MB from baseline")
    print("or >5MB/epoch for 2+ consecutive epochs.")
    print("On larger models this leak grows much faster and will trigger a warning.\n")


def main():
    # Check arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "both"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 70)
    print("TrainWatch Memory Leak Detection Demo")
    print("=" * 70)
    print("\nThis demo shows how TrainWatch detects memory leaks.")
    print("\nTwo scenarios:")
    print("  1. CORRECT - No leak (VRAM delta ~0MB)")
    print("  2. INCORRECT - With leak (VRAM delta increases)")
    print("\n" + "=" * 70)

    if mode == "correct" or mode == "both":
        train_correct(device)

    if mode == "leak" or mode == "both":
        # Clear GPU memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_with_leak(device)

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\n💡 Key Takeaway:")
    print("Always use loss.item() to extract scalars!")
    print("Don't store tensors unnecessarily - it causes memory leaks.")
    print("\nTrainWatch will warn you when VRAM increases across epochs.")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    print("\nUsage:")
    print("  python memory_leak_demo.py          # Run both scenarios")
    print("  python memory_leak_demo.py correct  # Run only correct version")
    print("  python memory_leak_demo.py leak     # Run only leak version\n")

    main()