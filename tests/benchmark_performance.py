"""
Benchmark: TrainWatch v0.1.0 vs v0.2.0 Performance
"""
import time
import torch
from trainwatch import Watcher


def benchmark_old_style(n_steps=1000):
    """
    v0.1.0 style: .item() every step

    Simulates: watcher.step(loss=loss.item())
    """
    watcher = Watcher(print_every=10000)  # No prints during benchmark

    start = time.perf_counter()

    for i in range(n_steps):
        loss = torch.tensor(1.5 + (i % 10) * 0.1)
        watcher.step(loss=loss.item())  # .item() every step!

    elapsed = time.perf_counter() - start

    return elapsed


def benchmark_new_style(n_steps=1000, sync_interval=10):
    """
    v0.2.0 style: tensor with batch sync

    Simulates: watcher.step(loss=loss)
    """
    watcher = Watcher(sync_interval=sync_interval, print_every=10000)

    start = time.perf_counter()

    for i in range(n_steps):
        loss = torch.tensor(1.5 + (i % 10) * 0.1)
        watcher.step(loss=loss)  # Tensor! Batch sync

    # Flush remaining
    watcher.epoch_end()

    elapsed = time.perf_counter() - start

    return elapsed


def run_benchmark():
    """Run comprehensive benchmark"""
    print("=" * 60)
    print("TrainWatch Performance Benchmark: v0.1.0 vs v0.2.0")
    print("=" * 60)

    # Warm up
    print("\nWarming up...")
    benchmark_old_style(100)
    benchmark_new_style(100)

    # Benchmark configurations
    configs = [
        {"n_steps": 1000, "name": "1K steps"},
        {"n_steps": 5000, "name": "5K steps"},
        {"n_steps": 10000, "name": "10K steps"},
    ]

    for config in configs:
        n_steps = config["n_steps"]
        name = config["name"]

        print(f"\n{name} ({n_steps} steps):")
        print("-" * 40)

        # v0.1.0 style
        old_time = benchmark_old_style(n_steps)
        print(f"  v0.1.0 (.item() every step):  {old_time * 1000:.2f}ms")

        # v0.2.0 style (sync_interval=10)
        new_time_10 = benchmark_new_style(n_steps, sync_interval=10)
        print(f"  v0.2.0 (sync_interval=10):    {new_time_10 * 1000:.2f}ms")

        # v0.2.0 style (sync_interval=50)
        new_time_50 = benchmark_new_style(n_steps, sync_interval=50)
        print(f"  v0.2.0 (sync_interval=50):    {new_time_50 * 1000:.2f}ms")

        # Calculate speedup
        speedup_10 = old_time / new_time_10
        speedup_50 = old_time / new_time_50

        print(f"\n  Speedup (interval=10): {speedup_10:.2f}x")
        print(f"  Speedup (interval=50): {speedup_50:.2f}x")

        # Overhead analysis
        overhead_old = old_time - new_time_50  # Approximation
        print(f"\n  Estimated overhead reduction: {overhead_old * 1000:.2f}ms")

    # GPU benchmark if available
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU Benchmark (CUDA available)")
        print("=" * 60)

        n_steps = 1000

        print(f"\n{n_steps} steps on GPU:")
        print("-" * 40)

        # Old style
        watcher_old = Watcher(print_every=10000)
        start = time.perf_counter()
        for i in range(n_steps):
            loss = torch.tensor(1.5, device='cuda')
            watcher_old.step(loss=loss.item())  # Sync every step!
        old_gpu_time = time.perf_counter() - start

        # New style
        watcher_new = Watcher(sync_interval=10, print_every=10000)
        start = time.perf_counter()
        for i in range(n_steps):
            loss = torch.tensor(1.5, device='cuda')
            watcher_new.step(loss=loss)  # Batch sync
        watcher_new.epoch_end()
        new_gpu_time = time.perf_counter() - start

        print(f"  v0.1.0: {old_gpu_time * 1000:.2f}ms")
        print(f"  v0.2.0: {new_gpu_time * 1000:.2f}ms")
        print(f"  Speedup: {old_gpu_time / new_gpu_time:.2f}x")

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()