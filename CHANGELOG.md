# Changelog

All notable changes to TrainWatch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-03-15

### Added

- **Tensor Support**: `step()` now accepts `torch.Tensor` directly (no `.item()` required)
- **Batch Synchronization**: New `sync_interval` parameter for performance optimization
  - Reduces GPU-CPU sync overhead by ~5x on GPU (no benefit on CPU-only training)
  - Default: `sync_interval=10` (sync every 10 steps)
- **Performance Optimization**: Minimal overhead (<1%) even with small batch sizes
- **Automatic Buffer Flush**: Remaining tensors are synced at `epoch_end()`

### Changed

- **print_every default**: Changed from `10` to `100` for better performance
  - Less frequent I/O operations
  - More reasonable update frequency for most use cases
- **step() method**: Now accepts both `torch.Tensor` and `float` (backward compatible)

### Fixed

- Performance overhead in small batch training scenarios
- GPU-CPU synchronization bottleneck

### Documentation

- Added performance best practices guide
- Updated examples to show tensor usage
- Added migration guide for v0.1.0 users

### Backward Compatibility

**100% backward compatible with v0.1.0**

Old code continues to work without any changes:
```python
# v0.1.0 style (still works!)
watcher.step(loss=loss.item())
```

New performant style:
```python
# v0.2.0 style (recommended)
watcher = Watcher(sync_interval=10)
watcher.step(loss=loss)  # ~5x faster on GPU!
```

### Performance Improvements

| Scenario | v0.1.0 | v0.2.0 | Improvement |
|----------|--------|--------|-------------|
| GPU (1K steps) | ~265ms | ~50ms | ~5x faster |
| CPU training | baseline | similar | no benefit |

### Acknowledgments

Special thanks to community feedback that identified the `.item()` synchronization overhead issue.

---

## [0.1.0] - 2026-01-31

### Initial Release

- **One line of code** monitoring for PyTorch training
- **Loss tracking**: Moving average, variance, trends
- **System monitoring**: CPU, RAM, GPU VRAM usage
- **Memory leak detection**: Warns when VRAM increases >10MB from baseline or grows consistently
- **DataLoader bottleneck detection**: Identifies slow data loading
- **Variance spike detection**: Alerts on training instability
- **Lightweight**: Minimal dependencies (torch, psutil)

### Features

- Real-time step and epoch metrics
- Automatic baseline setting after first epoch
- Configurable warning thresholds
- Clean, minimal output
- Production-ready

[0.2.0]: https://github.com/Hords01/trainwatch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Hords01/trainwatch/releases/tag/v0.1.0