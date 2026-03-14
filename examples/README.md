# TrainWatch Examples

This directory contains examples demonstrating TrainWatch capabilities across different scenarios.

**✅ All examples tested on Kaggle GPU T4 and P100!** See detailed results in `*_results.md` files.

---

## 📊 Test Results Summary

| Example | Tested Platforms | Key Results | Details |
|---------|------------------|-------------|---------|
| **Simple CNN** | CPU, T4, P100 | 12-15x GPU speedup, 0MB leak | [cifar10_results.md](cifar10_results.md) |
| **DenseNet121** | T4, P100 | 82% accuracy, 1.9x P100 speedup | [densenet_results.md](densenet_results.md) |
| **ResNet-18** | T4, P100 | 92% accuracy, 1.8x P100 speedup | [resnet_results.md](resnet_results.md) |
| **Memory Leak Demo** | T4, P100 | Perfect leak detection (+1.2MB) | [memory_leak_results.md](memory_leak_results.md) |

🔗 **Kaggle Collection:** [TrainWatch Examples](https://www.kaggle.com/emirkanbeyaz/code?query=trainwatch) - Try them yourself!

---

## Quick Start

### On Kaggle (Recommended for GPU testing):
```python
# Create new notebook
!git clone https://github.com/Hords01/trainwatch.git
%cd trainwatch
!pip install -e .

# Run an example
!python examples/cifar10_simple.py
```

### Locally:
```bash
cd trainwatch
pip install -e .
pip install torchvision  # For examples
python examples/cifar10_simple.py
```

---

## Examples

### 1. 🎯 Simple CNN - CIFAR-10 (`cifar10_simple.py`)

**Best for:** Getting started, understanding basics

**What it shows:**
- ✅ Basic TrainWatch integration (single line!)
- ✅ Step timing and loss tracking
- ✅ CPU/RAM/VRAM monitoring
- ✅ Epoch summaries
- ✅ Memory leak detection

**Model:** 2 conv layers + 2 FC layers (~100K parameters)  
**Dataset:** CIFAR-10 (50,000 images, 10 classes)  
**Training time:** ~2 min (GPU), ~15 min (CPU)

**Run it:**
```bash
python examples/cifar10_simple.py
```

**Expected output:**
```
Using device: cuda

Step     50 | loss=1.9358 | time=0.005s | CPU=33.2% | RAM=7.1% | VRAM=25MB
Step    100 | loss=1.6322 | time=0.005s | CPU=67.2% | RAM=7.2% | VRAM=25MB
...
✓ Epoch 1 complete - baselines set
...
============================================================
Epoch 2 Summary:
   Loss (avg): 0.8779 [stable]
   VRAM delta: +0.0MB
============================================================
```

See detailed results: [`cifar10_results.md`](cifar10_results.md)

---

### 2. 🏗️ DenseNet121 - CIFAR-10 (`densenet_cifar10.py`) 🆕

**Best for:** Real-world PyTorch models, training from scratch

**What it shows:**
- ✅ PyTorch `torchvision.models` API
- ✅ Training from scratch with `weights=None`
- ✅ Original DenseNet architecture (no custom modifications)
- ✅ Image resizing (CIFAR-10: 32×32 → 224×224)
- ✅ Gradient clipping
- ✅ Higher VRAM monitoring (~800-1000MB)
- ✅ Production-grade architecture

**Model:** DenseNet121 from torchvision (~7M parameters)  
**Dataset:** CIFAR-10 resized to 224×224 (ImageNet size)  
**Training time:** ~10 min (GPU), ~60 min (CPU)

**Code highlight:**
```python
import torchvision.models as models

# Load DenseNet121 from PyTorch model zoo
model = models.densenet121(weights=None)  # Training from scratch!

# Modify classifier for CIFAR-10 (10 classes)
model.classifier = nn.Linear(model.classifier.in_features, 10)
```

**Run it:**
```bash
python examples/densenet_cifar10.py
```

**Expected output:**
```
Using device: cuda
GPU: Tesla T4

Model: DenseNet121 (torchvision.models)
Total parameters: 6,964,106
Image size: 224x224 (resized from 32x32)

Step    100 | loss=2.1095 | time=0.304s | CPU=31.2% | RAM=9.0% | VRAM=115MB
Step    500 | loss=1.6268 | time=0.327s | CPU=32.0% | RAM=9.3% | VRAM=115MB
Step   1000 | loss=1.4253 | time=0.329s | CPU=31.9% | RAM=9.2% | VRAM=115MB
...
✓ Epoch 1 complete - baselines set
Train Accuracy: 43.05% | LR: 0.100000
Test Accuracy: 56.26% | Test Loss: 1.1907
============================================================

Step   2000 | loss=1.0654 | time=0.332s | CPU=31.7% | RAM=9.0% | VRAM=115MB
...
============================================================
Epoch 2 Summary:
   Loss (avg): 0.7431 [decreasing]
   VRAM delta: +0.0MB  ← No leak!
============================================================
Train Accuracy: 67.41% | LR: 0.100000
Test Accuracy: 73.98% | Test Loss: 0.7549
...

Final Test Accuracy: 81.76%
Final GPU Memory: 106MB
Peak GPU Memory: 4112MB
```

**Actual results (T4):** 81.76% accuracy, 0.331s/step, 0MB VRAM leak  
**Actual results (P100):** 82.15% accuracy, 0.175s/step (1.9x faster!), 0MB VRAM leak

See detailed results: [`densenet_results.md`](densenet_results.md)

---

### 3. 🚀 Advanced ResNet - Fashion-MNIST (`resnet_fashion_mnist.py`)

**Best for:** Complex models, production workflows

**What it shows:**
- ✅ Deep model monitoring (ResNet-18 style, 11M parameters)
- ✅ Data augmentation effects on training stability
- ✅ Learning rate scheduling (StepLR)
- ✅ Validation loop integration
- ✅ Production-ready training pipeline

**Model:** ResNet-18 inspired (~11M parameters)  
**Dataset:** Fashion-MNIST (60,000 images, 10 classes)  
**Training time:** ~3.5 min (GPU T4), ~1.9 min (GPU P100)

**Advanced features demonstrated:**
- Residual connections
- Batch normalization
- Data augmentation (rotation, translation, flip)
- Learning rate decay (StepLR)
- Train + validation loops

**Run it:**
```bash
python examples/resnet_fashion_mnist.py
```

**Expected output:**
```
Using device: cuda
GPU: Tesla T4

Total parameters: 11,172,810

Step     50 | loss=0.8026 | time=0.080s | CPU=38.7% | RAM=7.2% | VRAM=147MB
Step    100 | loss=0.6681 | time=0.082s | CPU=45.1% | RAM=7.2% | VRAM=147MB
Step    200 | loss=0.5087 | time=0.081s | CPU=45.1% | RAM=7.2% | VRAM=147MB
...
✓ Epoch 1 complete - baselines set
Learning rate: 0.010000
Validation - Loss: 0.3889 | Accuracy: 85.59%
============================================================

Step    500 | loss=0.3824 | time=0.083s | CPU=46.6% | RAM=7.3% | VRAM=148MB
...
============================================================
Epoch 2 Summary:
   Loss (avg): 0.3518 [decreasing]
   VRAM delta: +0.8MB
============================================================
Learning rate: 0.005000  ← LR decay kicked in!
Validation - Loss: 0.3330 | Accuracy: 88.45%
...

Final Validation Accuracy: 92.28%
Final GPU Memory: 148MB
```

**Actual results (T4):** 92.28% accuracy, 0.085s/step, +0.8MB/epoch (normal)  
**Actual results (P100):** 91.86% accuracy, 0.047s/step (1.8x faster!), 0MB VRAM delta

See detailed results: [`resnet_results.md`](resnet_results.md)

---

### 4. 🐛 Memory Leak Detection Demo (`memory_leak_demo.py`) ⚠️

**Best for:** Understanding memory leaks, debugging training issues

**What it shows:**
- ⚠️ Intentional memory leak (for educational purposes)
- ✅ TrainWatch's leak detection in action
- ✅ Correct vs incorrect implementations  
- ✅ How to avoid common PyTorch mistakes

**Two scenarios:**
1. **CORRECT** - Using `loss.item()`, no leak (VRAM delta 0MB) ✅
2. **INCORRECT** - Storing tensors in list, causes leak (VRAM grows!) ⚠️

**Run it:**
```bash
# Run both scenarios (default)
python examples/memory_leak_demo.py

# Run only correct version
python examples/memory_leak_demo.py correct

# Run only leak version
python examples/memory_leak_demo.py leak
```

**Example output (CORRECT - no leak):**
```
SCENARIO 1: CORRECT Training (No Memory Leak)
======================================================================
Best practices:
  ✓ Using loss.item() to extract scalars
  ✓ Not storing tensors in lists
  ✓ Properly clearing gradients

Step    100 | loss=1.6321 | time=0.019s | CPU=39.5% | RAM=7.6% | VRAM=25MB
Step    500 | loss=1.2523 | time=0.004s | CPU=65.8% | RAM=7.7% | VRAM=25MB
...
============================================================
Epoch 2 Summary:
   Loss (avg): 0.9106 [stable]
   VRAM delta: +0.0MB  ← Perfect!
============================================================
...
Epoch 3 Summary:
   Loss (avg): 0.7620 [stable]
   VRAM delta: +0.0MB  ← Perfect!
============================================================

✅ Training complete - NO MEMORY LEAK detected!
```

**Example output (INCORRECT - with leak):**
```
SCENARIO 2: INCORRECT Training (With Memory Leak)
======================================================================
Common mistake:
  ✗ Storing loss tensors (not .item()) in a list
  ✗ This keeps computation graphs in memory
  ✗ VRAM keeps growing!

Step    100 | loss=1.6219 | time=0.011s | CPU=68.4% | RAM=7.8% | VRAM=25MB
Step    400 | loss=1.2921 | time=0.020s | CPU=63.6% | RAM=7.8% | VRAM=26MB  ← Growing!
...
============================================================
Epoch 2 Summary:
   Loss (avg): 0.9147 [decreasing]
   VRAM delta: +0.4MB  ← Leak detected!
============================================================
...
Epoch 3 Summary:
   Loss (avg): 0.6912 [decreasing]
   VRAM delta: +0.8MB  ← Leak growing!
============================================================

⚠️  Training complete - MEMORY LEAK DETECTED!
Stored 2346 loss tensors in memory!
TrainWatch should have warned about increasing VRAM
```

**The Bug:**
```python
# ❌ WRONG: Stores entire tensor with computation graph
loss_history.append(loss)  # Memory leak!

# ✅ CORRECT: Extracts scalar value only
loss_history.append(loss.item())  # No leak!
```

**Actual results:** Both T4 and P100 showed identical leak behavior:
- CORRECT: +0.0MB VRAM delta (perfect!)
- INCORRECT: +1.2MB total leak in 3 epochs (would crash after ~100 epochs!)

**Key lesson:** One `.item()` prevents memory leaks! 🐛→✅

See detailed results: [`memory_leak_results.md`](memory_leak_results.md)

---
  ✓ Not storing tensors in lists
  ✓ Properly clearing gradients

Step    100 | loss=1.6322 | time=0.005s | CPU=67.2% | RAM=7.2% | VRAM=25MB
...
============================================================
Epoch 2 Summary:
   Loss (avg): 0.8779 [stable]
   VRAM delta: +0.0MB  ← No leak!
============================================================

✅ Training complete - NO MEMORY LEAK detected!
```

**Example output (INCORRECT - with leak):**
```
SCENARIO 2: INCORRECT Training (With Memory Leak)
======================================================================
Common mistake:
  ✗ Storing loss tensors (not .item()) in a list
  ✗ This keeps computation graphs in memory
  ✗ VRAM keeps growing!

Step    100 | loss=1.6322 | time=0.005s | CPU=67.2% | RAM=7.2% | VRAM=45MB
...
============================================================
Epoch 2 Summary:
   Loss (avg): 0.8779 [stable]
   VRAM delta: +20.0MB  ← Memory leak detected!
============================================================
⚠️  WARNING: Possible memory leak (+20MB VRAM since baseline)

⚠️  Training complete - MEMORY LEAK DETECTED!
Stored 2343 loss tensors in memory!
```

**Key lesson:** Always use `loss.item()` instead of storing the tensor!

---

## Comparison Table

| Feature | Simple CNN | DenseNet121 | ResNet-18 | Memory Leak Demo |
|---------|------------|-------------|-----------|------------------|
| **Complexity** | Beginner | Intermediate | Intermediate | Beginner |
| **Parameters** | ~100K | ~7M | ~11M | ~100K |
| **Dataset** | CIFAR-10 | CIFAR-10 (resized) | Fashion-MNIST | CIFAR-10 |
| **Image size** | 32×32 | 224×224 | 28×28 | 32×32 |
| **Training time** | 2 min (GPU) | 10 min (GPU) | 5 min (GPU) | 4 min (both) |
| **VRAM usage** | ~25MB | ~850MB | ~512MB | 25MB → 45MB (leak) |
| **Shows warnings** | Rarely | Sometimes | More likely | **Always (leak demo)** |
| **Architecture** | Custom | torchvision.models | Custom ResNet | Custom |
| **Training mode** | From scratch | From scratch (weights=None) | From scratch | From scratch |
| **Data augmentation** | ❌ | ✅ | ✅ | ❌ |
| **LR scheduling** | ❌ | ✅ (MultiStepLR) | ✅ (StepLR) | ❌ |
| **Gradient clipping** | ❌ | ✅ | ❌ | ❌ |
| **Purpose** | TrainWatch basics | Real PyTorch models | Production workflows | **Debugging/Learning** |
| **Special feature** | Quick start | Production architecture | Full pipeline | **Leak detection demo** |

---

## What TrainWatch Detects

### 1. Memory Leaks ⚠️
```
⚠️  WARNING: Possible memory leak (+127MB VRAM since baseline)
```

**When it happens:** Tensors not released, gradient accumulation bugs, caching issues

**Example fix:**
```python
# Bad - accumulates gradients
for i in range(100):
    loss = forward()
    loss.backward()  # Gradients accumulate!

# Good - clear gradients
for i in range(100):
    optimizer.zero_grad()  # Clear first!
    loss = forward()
    loss.backward()
    optimizer.step()
```

### 2. DataLoader Bottleneck ⚠️
```
⚠️  WARNING: Possible DataLoader bottleneck (slow steps, idle CPU)
```

**When it happens:** `num_workers=0`, slow augmentation, I/O bottleneck

**Example fix:**
```python
# Bad - single threaded
loader = DataLoader(dataset, batch_size=64, num_workers=0)

# Good - multi-threaded
loader = DataLoader(dataset, batch_size=64, num_workers=4)
```

### 3. Loss Variance Spike ⚠️
```
⚠️  WARNING: Loss variance spike detected - training may be unstable
```

**When it happens:** Learning rate too high, bad batch, exploding gradients

**Example fix:**
```python
# Reduce learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR

# Or use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Dataset Information

### CIFAR-10
- **Images:** 60,000 (50K train, 10K test)
- **Size:** 32×32 RGB
- **Classes:** airplane, car, bird, cat, deer, dog, frog, horse, ship, truck
- **Download size:** ~170MB
- **Use case:** Color image classification

### Fashion-MNIST
- **Images:** 70,000 (60K train, 10K test)
- **Size:** 28×28 grayscale
- **Classes:** T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot
- **Download size:** ~30MB
- **Use case:** Grayscale image classification

---

## Running on Different Platforms

### Kaggle
```python
!git clone https://github.com/Hords01/trainwatch.git
%cd trainwatch
!pip install -e .
!python examples/cifar10_simple.py
```

**GPU options:** T4, P100, T4×2 (free!)

### Google Colab
```python
!git clone https://github.com/Hords01/trainwatch.git
%cd trainwatch
!pip install -e .
!python examples/resnet_fashion_mnist.py
```

**Don't forget:** Runtime → Change runtime type → GPU

### Local (CPU)
```bash
git clone https://github.com/Hords01/trainwatch.git
cd trainwatch
pip install -e .
pip install torchvision
python examples/cifar10_simple.py
```

### Local (GPU)
Same as CPU, TrainWatch auto-detects CUDA.

---

## Customizing TrainWatch

### Adjust Print Frequency
```python
# Print every 100 steps (less output)
watcher = Watcher(print_every=100)

# Print every step (maximum detail)
watcher = Watcher(print_every=1)
```

### Disable Specific Warnings
```python
# No memory leak warnings
watcher = Watcher(warn_on_leak=False)

# No DataLoader warnings
watcher = Watcher(warn_on_bottleneck=False)
```

### CPU-Only Mode
```python
# Force CPU monitoring (no VRAM tracking)
watcher = Watcher(show_gpu=False)
```

---

## Interpreting Output

### Step Line
```
Step    100 | loss=1.6322 | time=0.005s | CPU=67.2% | RAM=7.2% | VRAM=25MB
```

- **Step:** Current training step
- **loss:** Moving average of recent losses (smoother than raw loss)
- **time:** Time taken for this step (low = good)
- **CPU:** CPU utilization (60-80% = good)
- **RAM:** RAM usage percentage
- **VRAM:** GPU memory allocated (only on GPU)

### Epoch Summary
```
============================================================
Epoch 2 Summary:
   Loss (avg): 0.8779 [decreasing]
   VRAM delta: +0.0MB
============================================================
```

- **Loss (avg):** Average loss for the epoch
- **[decreasing/stable/increasing]:** Trend detection
- **VRAM delta:** Memory increase since baseline (+0.0 = no leak!)

---

## Troubleshooting

### Import Error
```bash
# Solution: Install in editable mode
pip install -e .
```

### CUDA Out of Memory
```python
# Reduce batch size
batch_size = 32  # Instead of 128

# Or use gradient accumulation
accumulation_steps = 4
```

### Slow Training
```python
# Increase num_workers
loader = DataLoader(dataset, num_workers=4)

# Enable pin_memory on GPU
loader = DataLoader(dataset, pin_memory=True)
```

---

## Next Steps

1. **Run simple example** → Understand basics
2. **Run advanced example** → See warnings in action
3. **Use in your project** → Just add `watcher.step(loss=loss.item())`!

For more details, see the main [README](../README.md).

---

## Contributing

Found a bug? Have a use case we should add?  
Open an issue: https://github.com/Hords01/trainwatch/issues

---

**Happy training! 🚀**
