"""
TrainWatch - PyTorch training monitor with one line of code
"""
import time
from typing import Optional
from .system import SystemMonitor
from .metrics import LossTracker

try:
    import torch as _torch
    _TORCH = _torch
except ImportError:
    _TORCH = None


class Watcher:
    """
    Monitor PyTorch training health in real-time

    Tracks:
    - Loss trends (moving average, variance)
    - Step/epoch timing
    - CPU and RAM usage
    - GPU VRAM usage and leaks
    - DataLoader bottlenecks

    Example (v0.2.0 - Recommended):
        >>> watcher = Watcher(sync_interval=10)
        >>> for epoch in range(epochs):
        ...   for images, labels in dataloader:
        ...       loss = train_step(images, labels)
        ...       watcher.step(loss=loss)  # Tensor! ~5x faster on GPU
        ...   watcher.epoch_end()

    Example (v0.1.0 - Backward Compatible):
        >>> watcher = Watcher()
        >>> for epoch in range(epochs):
        ...   for images, labels in dataloader:
        ...       loss = train_step(images, labels)
        ...       watcher.step(loss=loss.item())  # Still works!
        ...   watcher.epoch_end()
    """

    def __init__(
        self,
        window: int = 20,
        print_every: int = 100,
        sync_interval: int = 10,
        show_gpu: bool = True,
        warn_on_leak: bool = True,
        warn_on_bottleneck: bool = True,
        warn_on_variance: bool = True,
        device: str = 'cuda:0'
    ):
        """
        Initialize training watcher

        Args:
            window: Number of recent steps to keep for moving averages
            print_every: Print metrics every N steps (default: 100)
            sync_interval: Sync tensor losses every N steps for performance (default: 10)
            show_gpu: Show GPU VRAM metrics
            warn_on_leak: Warn about potential memory leaks
            warn_on_bottleneck: Warn about DataLoader bottlenecks
            warn_on_variance: Warn about loss variance spikes
            device: CUDA device to monitor (e.g., 'cuda:0')
        """
        self.window = window
        self.print_every = print_every
        self.sync_interval = sync_interval
        self.show_gpu = show_gpu
        self.warn_on_leak = warn_on_leak
        self.warn_on_bottleneck = warn_on_bottleneck
        self.warn_on_variance = warn_on_variance

        # core components
        self.loss_tracker = LossTracker(window=window)
        self.system_monitor = SystemMonitor(device=device)

        # timing
        self.step_count = 0
        self.epoch_count = 0
        self.last_step_time = time.time()

        # baselines (set at end of first epoch)
        self.baselines_set = False

        # tensor batching (v0.2.0)
        self.loss_buffer = []

        # memory leak tracking
        self._vram_delta_prev: Optional[float] = None
        self._vram_growth_streak: int = 0

    def step(self, loss) -> None:
        """
        Record a training step

        Args:
            loss: Loss value (torch.Tensor or float)
                 - Tensor: Batched synchronization (recommended for performance)
                 - Float: Immediate tracking (backward compatible)
        """
        self.step_count += 1

        # timing
        now = time.time()
        step_time = now - self.last_step_time
        self.last_step_time = now

        is_tensor = _TORCH is not None and _TORCH.is_tensor(loss)

        if is_tensor:
            # tensor path - batch accumulation for performance
            self.loss_buffer.append(loss.detach())

            # sync when buffer is full
            if len(self.loss_buffer) >= self.sync_interval:
                self._sync_losses()
        else:
            # Flush buffered tensors first
            if self.loss_buffer:
                self._sync_losses()
            self.loss_tracker.add(loss)

        # should we print?
        if self.step_count % self.print_every != 0:
            return

        # get metrics
        metrics = self.system_monitor.get_metrics()
        loss_avg = self.loss_tracker.get_moving_average()

        # build message
        msg = f"Step {self.step_count:>6} | "

        if loss_avg is not None:
            msg += f"loss={loss_avg:.4f} | "
        else:
            loss_val = loss.item() if is_tensor else float(loss)
            msg += f"loss={loss_val:.4f} | "

        msg += f"time={step_time:.3f}s | "
        msg += f"CPU={metrics['cpu_percent']:.1f}% | "
        msg += f"RAM={metrics['ram_percent']:.1f}%"

        if self.show_gpu and 'vram_mb' in metrics:
            msg += f" | VRAM={metrics['vram_mb']:.0f}MB"

        print(msg)

        # warnings (only after baselines are set)
        if self.baselines_set:
            self._check_warnings(step_time, metrics)

    def epoch_end(self) -> None:
        """Call at the end of each epoch for summary and baseline setting"""
        # flush any remaining buffered losses
        if self.loss_buffer:
            self._sync_losses()

        self.epoch_count += 1

        # first epoch: set baselines
        if not self.baselines_set and self.epoch_count == 1:
            self.system_monitor.set_vram_baseline()
            self.loss_tracker.set_variance_baseline()
            self.baselines_set = True
            print(f"✓ Epoch {self.epoch_count} complete - baselines set")
            return

        # epoch summary
        loss_avg = self.loss_tracker.get_moving_average()
        trend = self.loss_tracker.get_trend()
        vram_delta = self.system_monitor.get_vram_delta()

        msg = f"\n{'='*60}\n"
        msg += f"Epoch {self.epoch_count} Summary:\n"

        if loss_avg is not None:
            msg += f"   Loss (avg): {loss_avg:.4f}"
            if trend:
                msg += f" [{trend}]"
            msg += "\n"

        if self.show_gpu and vram_delta is not None:
            msg += f"   VRAM delta: {vram_delta:+.1f}MB\n"

        msg += f'='*60
        print(msg)

        # check for memory leak
        if self.warn_on_leak and self.show_gpu:
            self._check_memory_leak(vram_delta)

    def _check_memory_leak(self, vram_delta: Optional[float]) -> None:
        """
        Detect memory leaks using two complementary signals:
        1. Absolute threshold: delta from baseline > 10MB
        2. Consecutive growth: VRAM grew >5MB for 2+ consecutive epochs

        Research basis: PyTorch community practice recommends checking
        memory_allocated() growth across epochs. Normal allocator
        fragmentation causes <5MB variation; sustained growth signals a leak.
        """
        if vram_delta is None:
            return

        # track per-epoch growth using cumulative delta
        if self._vram_delta_prev is not None:
            epoch_growth = vram_delta - self._vram_delta_prev
            if epoch_growth > 5:
                self._vram_growth_streak += 1
            else:
                self._vram_growth_streak = 0
        self._vram_delta_prev = vram_delta

        if vram_delta > 10:
            print(f"⚠️  WARNING: Possible memory leak (+{vram_delta:.1f}MB VRAM since baseline)")
        elif self._vram_growth_streak >= 2:
            print(f"⚠️  WARNING: Possible memory leak (VRAM growing {self._vram_growth_streak} consecutive epochs, +{vram_delta:.1f}MB total)")

    def _sync_losses(self) -> None:
        """
        Synchronize buffered tensor losses (batch sync)

        This performs a single GPU-CPU sync for all buffered losses,
        reducing overhead by ~5x on GPU compared to syncing every step.
        """
        if not self.loss_buffer:
            return

        try:
            losses = _TORCH.stack(self.loss_buffer)
            avg_loss = losses.mean().item()  # single sync!
            self.loss_tracker.add(avg_loss)
            self.loss_buffer.clear()
        except Exception:
            # fallback: process individually
            for loss_tensor in self.loss_buffer:
                self.loss_tracker.add(loss_tensor.item())
            self.loss_buffer.clear()

    def _check_warnings(self, step_time: float, metrics: dict) -> None:
        """Check for warning conditions"""

        # variance spike
        if self.warn_on_variance:
            if self.loss_tracker.detect_variance_spike():
                print("⚠️  WARNING: Loss variance spike detected - training may be unstable")

        # DataLoader bottleneck
        if self.warn_on_bottleneck and self.show_gpu:
            # simple heuristic: if GPU VRAM is allocated but step is slow
            if 'vram_mb' in metrics and metrics['vram_mb'] > 100:  # GPU is being used
                if step_time > 0.5:  # but step is slow
                    if metrics['cpu_percent'] < 50:  # and CPU is idle
                        print("⚠️  WARNING: Possible DataLoader bottleneck (slow steps, idle CPU)")
