"""Test custom autograd functions that cast bf16 <-> fp8_e4m3.

FP8Cast  — forward: bf16 -> fp8_e4m3, backward: fp8_e4m3 -> bf16
FP8DeCast — forward: fp8_e4m3 -> bf16, backward: bf16 -> fp8_e4m3

Launch with:
    python -u moe-pretraining-perf/tests/fp8_cast_autograd_test.py
"""

import torch
from torch.autograd import Function


# ---------------------------------------------------------------------------
# The autograd function under test
# ---------------------------------------------------------------------------

class FP8Cast(Function):
    """bf16 -> fp8_e4m3 in fwd, fp8_e4m3 -> bf16 in bwd."""

    @staticmethod
    def forward(ctx, x):
        # x: bf16 -> fp8_e4m3
        return x.to(torch.float8_e4m3fn)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output arrives as fp8_e4m3 -> cast back to bf16
        return grad_output.to(torch.bfloat16)


def fp8_cast(x):
    """Convenience wrapper."""
    return FP8Cast.apply(x)


class FP8DeCast(Function):
    """fp8_e4m3 -> bf16 in fwd, bf16 -> fp8_e4m3 in bwd.

    Inverse of FP8Cast — used to decode fp8 activations back to bf16 for
    computation, while the backward cast re-encodes gradients to fp8.
    """

    @staticmethod
    def forward(ctx, x):
        # x: fp8_e4m3 -> bf16
        return x.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: bf16 -> fp8_e4m3
        return grad_output.to(torch.float8_e4m3fn)


def fp8_decast(x):
    """Convenience wrapper."""
    return FP8DeCast.apply(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bf16_tensor(shape, seed=42, requires_grad=False):
    """Deterministic bf16 test data in [-1, 1]."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.rand(shape, generator=g, device="cuda") * 2 - 1).to(torch.bfloat16).requires_grad_(requires_grad)


def _fp8_tensor(shape, seed=42, requires_grad=False):
    """Deterministic fp8_e4m3 test data in [-1, 1]."""
    bf16 = _bf16_tensor(shape, seed=seed)
    return bf16.to(torch.float8_e4m3fn).requires_grad_(requires_grad)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_dtype_and_shape():
    """Forward output is fp8_e4m3 with matching shape."""
    x = _bf16_tensor((4, 128))
    y = fp8_cast(x)
    assert y.dtype == torch.float8_e4m3fn, f"Expected fp8_e4m3, got {y.dtype}"
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print("[PASS] test_forward_dtype_and_shape")


def test_forward_numerical():
    """Forward cast values are close (within fp8 rounding)."""
    x = _bf16_tensor((8, 256))
    y = fp8_cast(x)
    # Compare via float32 for precision
    diff = y.float() - x.float()
    max_diff = diff.abs().max().item()
    # fp8_e4m3 has 3 mantissa bits; for values in [-1,1] the absolute
    # rounding error is bounded by ~0.0625
    assert max_diff < 0.1, f"Forward cast max diff {max_diff:.4f} exceeds tolerance"
    print(f"[PASS] test_forward_numerical (max_diff={max_diff:.4e})")


def test_gradient_dtype_is_bf16():
    """When input has requires_grad=True, gradient dtype is bf16."""
    x = _bf16_tensor((4, 128), requires_grad=True)
    y = fp8_cast(x)
    loss = y.float().sum()
    loss.backward()

    assert x.grad is not None, "Gradient should be computed"
    assert x.grad.dtype == torch.bfloat16, (
        f"Expected bf16 gradient, got {x.grad.dtype}"
    )
    print("[PASS] test_gradient_dtype_is_bf16")


def test_gradient_numerical():
    """Gradient magnitude ~1 (identity cast with fp8 rounding).

    For loss = sum(fp8_cast(x)), d(loss)/d(x_i) ~ 1 for each element.
    fp8 rounding during fwd means the effective Jacobian is slightly off,
    but the mean should be very close to 1.
    """
    x = _bf16_tensor((8, 256), requires_grad=True)
    y = fp8_cast(x)
    loss = y.float().sum()
    loss.backward()

    grad = x.grad
    mean_grad = grad.float().mean().item()
    assert abs(mean_grad - 1.0) < 0.05, f"Mean grad {mean_grad:.4f} far from 1.0"
    assert grad.isfinite().all(), "Non-finite gradients detected"
    print(f"[PASS] test_gradient_numerical (mean_grad={mean_grad:.4f})")


def test_backward_hook_sees_grad():
    """Attach a hook to verify the grad tensor flowing through the graph."""
    x = _bf16_tensor((4, 128), requires_grad=True)
    y = fp8_cast(x)

    captured = {}
    def hook(grad):
        captured["dtype"] = grad.dtype
        captured["shape"] = grad.shape
        return grad

    y.register_hook(hook)

    loss = y.float().sum()
    loss.backward()

    # The hook fires on the grad arriving at y (autograd internal precision)
    assert "dtype" in captured, "Hook was never called"
    assert captured["shape"] == y.shape, f"Hook grad shape mismatch: {captured['shape']}"
    print(f"[PASS] test_backward_hook_sees_grad (hook saw dtype={captured['dtype']})")


def test_chain_multiple_casts():
    """Stacking casts: bf16->fp8->bf16 round-trip with gradient flow."""
    x = _bf16_tensor((4, 128), requires_grad=True)
    y = fp8_cast(x)                # bf16 -> fp8 (autograd-tracked)
    z = y.to(torch.bfloat16)       # fp8 -> bf16 (no autograd, plain cast)
    loss = z.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.dtype == torch.bfloat16
    assert x.grad.isfinite().all()

    round_trip_diff = (z - x).abs().float().max().item()
    print(f"[PASS] test_chain_multiple_casts (round-trip max_diff={round_trip_diff:.4e})")


def test_cast_in_computation_graph():
    """fp8_cast inside a larger graph: loss = sum(fp8_cast(x).float() * 2)."""
    x = _bf16_tensor((4, 128), requires_grad=True)
    y = fp8_cast(x)
    z = y.float() * 2.0
    loss = z.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.dtype == torch.bfloat16
    mean_grad = x.grad.float().mean().item()
    # d(loss)/d(x) = 2.0 * d(fp8_cast)/d(x) ~ 2.0
    assert abs(mean_grad - 2.0) < 0.1, f"Mean grad {mean_grad:.4f}, expected ~2.0"
    print(f"[PASS] test_cast_in_computation_graph (mean_grad={mean_grad:.4f})")


def test_large_tensor():
    """Larger tensor to exercise memory/layout correctness."""
    x = _bf16_tensor((1024, 4096), requires_grad=True)
    y = fp8_cast(x)
    assert y.dtype == torch.float8_e4m3fn
    assert y.shape == x.shape

    loss = y.float().sum()
    loss.backward()
    assert x.grad.dtype == torch.bfloat16
    assert x.grad.isfinite().all()

    mem_fp8_mb = x.numel() / 1e6
    mem_bf16_mb = x.numel() * 2 / 1e6
    print(f"[PASS] test_large_tensor "
          f"(shape={list(x.shape)}, fp8={mem_fp8_mb:.1f}MB vs bf16={mem_bf16_mb:.1f}MB, "
          f"50% reduction)")


# ---------------------------------------------------------------------------
# Decast tests (fp8_e4m3 -> bf16 fwd, bf16 -> fp8_e4m3 bwd)
# ---------------------------------------------------------------------------

def test_decast_forward_dtype_and_shape():
    """Decast forward: fp8_e4m3 -> bf16, same shape."""
    x = _fp8_tensor((4, 128))
    y = fp8_decast(x)
    assert y.dtype == torch.bfloat16, f"Expected bf16, got {y.dtype}"
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print("[PASS] test_decast_forward_dtype_and_shape")


def test_decast_forward_numerical():
    """Decast forward values match the original bf16 source (within fp8 rounding)."""
    bf16_src = _bf16_tensor((8, 256))
    x = bf16_src.to(torch.float8_e4m3fn)  # simulate stored fp8 data
    y = fp8_decast(x)
    # bf16_src -> fp8 -> bf16: only one rounding step from the initial cast
    diff = (y - bf16_src).abs().float().max().item()
    assert diff < 0.1, f"Decast fwd max diff {diff:.4f} exceeds tolerance"
    print(f"[PASS] test_decast_forward_numerical (max_diff={diff:.4e})")


def test_decast_gradient_dtype_is_fp8():
    """When fp8 input has requires_grad=True, decast backward produces fp8 grad."""
    x = _fp8_tensor((4, 128), requires_grad=True)
    y = fp8_decast(x)
    loss = y.float().sum()
    loss.backward()

    assert x.grad is not None, "Gradient should be computed"
    assert x.grad.dtype == torch.float8_e4m3fn, (
        f"Expected fp8_e4m3 gradient, got {x.grad.dtype}"
    )
    print("[PASS] test_decast_gradient_dtype_is_fp8")


def test_decast_gradient_numerical():
    """Decast gradient ~1 for sum-loss (identity upcast with rounding)."""
    x = _fp8_tensor((8, 256), requires_grad=True)
    y = fp8_decast(x)
    loss = y.float().sum()
    loss.backward()

    grad = x.grad
    assert grad.dtype == torch.float8_e4m3fn
    mean_grad = grad.float().mean().item()
    assert abs(mean_grad - 1.0) < 0.05, f"Mean grad {mean_grad:.4f} far from 1.0"
    assert grad.float().isfinite().all(), "Non-finite gradients detected"
    print(f"[PASS] test_decast_gradient_numerical (mean_grad={mean_grad:.4f})")


def test_decast_in_computation_graph():
    """Decast inside a graph: loss = sum(decast(x).float() * 3)."""
    x = _fp8_tensor((4, 128), requires_grad=True)
    y = fp8_decast(x)
    z = y.float() * 3.0
    loss = z.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.dtype == torch.float8_e4m3fn
    mean_grad = x.grad.float().mean().item()
    assert abs(mean_grad - 3.0) < 0.2, f"Mean grad {mean_grad:.4f}, expected ~3.0"
    print(f"[PASS] test_decast_in_computation_graph (mean_grad={mean_grad:.4f})")


def test_decast_large_tensor():
    """Decast with a larger tensor."""
    x = _fp8_tensor((1024, 4096), requires_grad=True)
    y = fp8_decast(x)
    assert y.dtype == torch.bfloat16
    assert y.shape == x.shape

    loss = y.float().sum()
    loss.backward()
    assert x.grad.dtype == torch.float8_e4m3fn
    assert x.grad.float().isfinite().all()
    print(f"[PASS] test_decast_large_tensor (shape={list(x.shape)})")


# ---------------------------------------------------------------------------
# Paired cast -> decast tests
# ---------------------------------------------------------------------------

def test_cast_then_decast_roundtrip():
    """bf16 -> fp8 (cast) -> bf16 (decast): full autograd round-trip.

    Both functions participate in the graph:
      x(bf16) --FP8Cast--> fp8 --FP8DeCast--> bf16 --> loss
    """
    x = _bf16_tensor((8, 256), requires_grad=True)
    y = fp8_cast(x)       # bf16 -> fp8
    z = fp8_decast(y)     # fp8 -> bf16
    loss = z.sum()
    loss.backward()

    # Round-trip: bf16 -> fp8 -> bf16, only one fp8 rounding step
    round_trip_diff = (z.detach() - x).abs().float().max().item()
    assert round_trip_diff < 0.1, f"Round-trip max diff {round_trip_diff:.4f}"

    # Gradient: loss -> DeCast(bwd: bf16->fp8) -> Cast(bwd: fp8->bf16) -> x
    # Two casts in backward, but each is identity-ish
    assert x.grad is not None
    assert x.grad.dtype == torch.bfloat16
    assert x.grad.isfinite().all()
    mean_grad = x.grad.float().mean().item()
    assert abs(mean_grad - 1.0) < 0.05, f"Mean grad {mean_grad:.4f} far from 1.0"

    print(f"[PASS] test_cast_then_decast_roundtrip "
          f"(round-trip max_diff={round_trip_diff:.4e}, mean_grad={mean_grad:.4f})")


def test_decast_then_cast_roundtrip():
    """fp8 -> bf16 (decast) -> fp8 (cast): full autograd round-trip.

      x(fp8) --FP8DeCast--> bf16 --FP8Cast--> fp8 --> loss
    """
    x = _fp8_tensor((8, 256), requires_grad=True)
    y = fp8_decast(x)     # fp8 -> bf16
    z = fp8_cast(y)       # bf16 -> fp8
    loss = z.float().sum()
    loss.backward()

    # Gradient: loss -> Cast(bwd: fp8->bf16) -> DeCast(bwd: bf16->fp8) -> x
    assert x.grad is not None
    assert x.grad.dtype == torch.float8_e4m3fn
    assert x.grad.float().isfinite().all()
    mean_grad = x.grad.float().mean().item()
    assert abs(mean_grad - 1.0) < 0.05, f"Mean grad {mean_grad:.4f} far from 1.0"

    print(f"[PASS] test_decast_then_cast_roundtrip (mean_grad={mean_grad:.4f})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.cuda.set_device(0)
    print(f"torch version: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}\n")

    tests = [
        # FP8Cast tests (bf16 -> fp8 fwd, fp8 -> bf16 bwd)
        test_forward_dtype_and_shape,
        test_forward_numerical,
        test_gradient_dtype_is_bf16,
        test_gradient_numerical,
        test_backward_hook_sees_grad,
        test_chain_multiple_casts,
        test_cast_in_computation_graph,
        test_large_tensor,
        # FP8DeCast tests (fp8 -> bf16 fwd, bf16 -> fp8 bwd)
        test_decast_forward_dtype_and_shape,
        test_decast_forward_numerical,
        test_decast_gradient_dtype_is_fp8,
        test_decast_gradient_numerical,
        test_decast_in_computation_graph,
        test_decast_large_tensor,
        # Paired cast <-> decast round-trip
        test_cast_then_decast_roundtrip,
        test_decast_then_cast_roundtrip,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            raise

    print(f"\nAll {passed}/{len(tests)} tests passed!")
