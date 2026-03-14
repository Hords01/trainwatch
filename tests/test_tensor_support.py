"""
Test tensor support functionality (v0.2.0)
"""
import pytest
import torch


def test_tensor_input():
    """Test that tensor input works correctly"""
    from trainwatch import Watcher

    watcher = Watcher(sync_interval=5, print_every=1000)

    # Add tensor losses
    for i in range(10):
        loss = torch.tensor(float(i))
        watcher.step(loss=loss)

    # Should have synced twice (5 + 5)
    assert len(watcher.loss_tracker.losses) == 2


def test_float_input_backward_compat():
    """Test that float input still works (backward compatibility)"""
    from trainwatch import Watcher

    watcher = Watcher(print_every=1000)

    # Old style with float
    watcher.step(loss=1.5)
    watcher.step(loss=2.0)
    watcher.step(loss=1.8)

    assert len(watcher.loss_tracker.losses) == 3


def test_item_usage_works():
    """Test that .item() usage still works"""
    from trainwatch import Watcher

    watcher = Watcher(print_every=1000)

    loss = torch.tensor(1.5)
    watcher.step(loss=loss.item())  # Old style

    assert len(watcher.loss_tracker.losses) == 1


def test_mixed_input():
    """Test mixing tensor and float inputs"""
    from trainwatch import Watcher

    watcher = Watcher(sync_interval=10, print_every=1000)

    # Tensor
    watcher.step(loss=torch.tensor(1.0))

    # Float (should trigger immediate add)
    watcher.step(loss=2.0)

    # Tensor again
    watcher.step(loss=torch.tensor(3.0))

    # Should have 2 entries (1 float + 1 from first tensor that got flushed by float)
    assert len(watcher.loss_tracker.losses) == 2


def test_buffer_flush_on_epoch_end():
    """Test that buffer is flushed at epoch end"""
    from trainwatch import Watcher

    watcher = Watcher(sync_interval=10, print_every=1000)

    # Add 3 tensor losses (less than sync_interval)
    for i in range(3):
        loss = torch.tensor(float(i))
        watcher.step(loss=loss)

    # Buffer should have 3 items, tracker should be empty
    assert len(watcher.loss_buffer) == 3
    assert len(watcher.loss_tracker.losses) == 0

    # Epoch end should flush
    watcher.epoch_end()

    # Buffer should be clear, tracker should have 1 entry
    assert len(watcher.loss_buffer) == 0
    assert len(watcher.loss_tracker.losses) == 1


def test_detach_prevents_memory_leak():
    """Test that tensors are properly detached"""
    from trainwatch import Watcher

    watcher = Watcher(sync_interval=5, print_every=1000)

    # Create tensor with gradient
    loss = torch.tensor(1.5, requires_grad=True)

    # Add to watcher
    watcher.step(loss=loss)

    # Buffer should contain detached tensor (no grad)
    if watcher.loss_buffer:
        assert not watcher.loss_buffer[0].requires_grad


def test_sync_interval_parameter():
    """Test that sync_interval parameter works"""
    from trainwatch import Watcher

    # Test with interval=3
    watcher = Watcher(sync_interval=3, print_every=1000)

    for i in range(6):
        watcher.step(loss=torch.tensor(float(i)))

    # Should have synced twice (3 + 3)
    assert len(watcher.loss_tracker.losses) == 2


def test_large_batch_sync():
    """Test with larger sync interval"""
    from trainwatch import Watcher

    watcher = Watcher(sync_interval=100, print_every=1000)

    # Add 100 losses
    for i in range(100):
        watcher.step(loss=torch.tensor(1.0))

    # Should have synced once
    assert len(watcher.loss_tracker.losses) == 1


def test_gpu_tensor_if_available():
    """Test with GPU tensor if CUDA is available"""
    from trainwatch import Watcher

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    watcher = Watcher(sync_interval=5, print_every=1000)

    # GPU tensor
    loss = torch.tensor(1.5, device='cuda')
    watcher.step(loss=loss)

    # Should work without error
    assert len(watcher.loss_buffer) == 1