# Nondeterministic blow-up in the grouped FP8 wgrad GEMM

## Summary

The offloading FP8 MoE backward pass **occasionally** produces a catastrophically wrong `w2` weight gradient at the beginning of the training, discovered by the `max_w2_grad_ref_fp8_rel_l2` guard (recorded relative-L2 of 8,000–70,000 vs a healthy ~0.01). This failure is **nondeterministic** and is related to the token layout and values assigned to an expert.

**Root cause**: **The per-channel FP8 cast kernel I designed uses scale factor 1.0 for an all-zero channel , while the DeepGEMM grouped GEMM kernel (`deep_gemm.k_grouped_fp8_gemm_nt_contiguous`) is problematic when handling this special case. Use scale factor <= 1e-8 here can mitigate the error.**
- In DeepGEMM's *own* quantization implementation, it clamps `amax` of a channel to `1e-4`, which compresses the scale range and makes the problem disappear. However, this clamp affects the accuracy of block-wise FP8 quantization and will lead to training crash.

## How I noticed it

During the correctness verification tests of `OffloadingExpertsFP8GroupedSwiMLP` (in `megatron/core/transformer/moe/experts_offloading_fp8_util.py`), I noticed abnormal grad_norm value at the beginning of the training:
![[grad_norm_fp8.png]]
- This is 2 training trace using FP8 (red) and BF16 (blue). FP8 presents large grad_norm (>200) in ~4 steps at the beginning of training. 
While for the LM loss at the beginning, it presents a small deviation between step 20 - step 90. 
![[lm_loss_fp8_deepgemm.png]]
In the long run, however, both LM loss and grad_norm are stable and converging:
![[lm_loss_fp8_deepgemm_30b.png]]
This abnormal grad_norm at the beginning appears in both AdamW and Muon, and also in both SwiGLU and PolyNorm.

## Trace the problem
To narrow down the cause, I did multiple experiments at different scale, and ruled out the following cases:
- Not the problem of suboptimal hyperparameter
- Not the problem of parallel setup (VPP, EP Overlap, FP8 Dispatch)
- Not the problem of offloading mechanism
- Not the problem of FP8 Forward computation and FP8 Dgrad computation

And finally converged to:
- **FP8 Down-Linear Weight Backward computation**, i.e. W2 grad computation
![[fp8.png]]
The computation of W2 Computation can be simplified as:
$$ dW_2 = dy.T \cdot act(a).T$$
where both $dy$ and $act(a)$ will be quantized in column direction (128, 1), implemented as `per_channel_cast_to_fp8` in `fp8_jit.py`. The per-channel quantization is explained as:
![[per_channel_quant.png]]

Hence, in the first place I launched a few tests to observe the data distribution of $dy$ and $act(a)$
 before quantization to see if there are abnormal logits, and designed unit tests for the wgrad computation under observed data distribution
 - Both tensors follow the normal distribution, and quantization error is around 3.2%, which falls in the correct range.
 - The max logit in both tensors during the training are normal across all layers.
	![[max_logit_moe.png]]
 
 - Local unit tests following the observed data distribution are completely correct. All passed without numerical problem.
However, the grad_w2 computed during the training is problematic, by computing the w2 weight gradient twice: once via the FP8 grouped GEMM and once via a BF16 reference, the two results present super large difference (L2 norm). So I dumped all the tensors when abnormal L2 norm occurs to reproduce the error in a small test.

**Investigation with Dumped Tensors:**
1. The inputs are benign; the cast is correct
	- Replaying a dump's operands offline gives rel_l2 ≈ 0.008 — healthy. The FP8 cast is bitwise correct. So neither the data nor the quantization kernel is "wrong".
2. The error can be reproduced non-deterministically locally
	- ~20 times out of 100 run
3. Not all experts' gradients crashed. Only one or two of the expert gradients present abnormal values

**Hence, it is the `k_grouped_fp8_gemm_nt_contiguous` kernel that is problematic, but why in the long run the training does not crash?** I uses claude to analyze the layout and feature of the dumped $dy$ and $a$ tensors, and it turns out that for the expert whose gradient that is problematic:
- **the expert gets exactly 128 tokens**
- **one channel (column) of the $dy$ that belongs to this expert are all zeros**

So I was able to reproduce the error in a small unit test with:
- `tokens_per_expert = [256, 128, 128, 256]`
- `Hidden dim >= 2048`
- one channel of $dy$ is zero

I did many tests around it, and 2 things are confirmed:
- The problem is real, and happens non-deterministically
- There is a workaround

The per channel cast code I designed is as following:
```py
def _per_channel_cast_to_fp8(x, gran_k=128):
	m, n = x.shape
	x_view = x.view(-1, gran_k, n)
	amax = x_view.abs().float().amax(dim=1).view(-1, n)
	if _AMAX_FLOOR > 0:
		amax = amax.clamp(_AMAX_FLOOR)
	sf = amax / 448.0
	sf = torch.where(sf == 0, torch.ones_like(sf), sf)
	fp8 = (x_view * (1.0 / sf.unsqueeze(1))).to(torch.float8_e4m3fn).view(m, n)
	return fp8, sf
```
when there is an all-zero channel, I use scale factor = 1.0 for that channel, which should be safe because it contains only zero. But it is not and causes the non-deterministic error. However, if I use a smaller number to replace 1.0, I found that the error occurs less:

| sf=1.0    | sf=1e-4  | sf=1e-8 | sf=1e-30 |
| --------- | -------- | ------- | -------- |
| 1347/1000 | 32/10000 | 0/10000 | 0/10000  |



## Reproducer
```py
import torch
import deep_gemm

_KS = [256, 128, 128, 256]
_HIDDEN = 2048   # n: grad_y channels / wgrad rows
_FFN = 2048      # H: activation channels / wgrad cols
_AMAX_FLOOR = 0

def _per_channel_cast_to_fp8(x, gran_k=128, sf_val=1e-30):
    m, n = x.shape
    x_view = x.view(-1, gran_k, n)
    amax = x_view.abs().float().amax(dim=1).view(-1, n)
    if _AMAX_FLOOR > 0:
        amax = amax.clamp(_AMAX_FLOOR)
    sf = amax / 448.0

    # NOTE: handle cases when the channel contains all zeros
    sf = torch.where(sf==0, torch.full_like(sf, sf_val), sf).clamp_min(1e-30)
    fp8 = (x_view * (1.0 / sf.unsqueeze(1))).to(torch.float8_e4m3fn).view(m, n)
    return fp8, sf

def _pack_kmajor(fp8, ks, mcols):
    """Per-expert k-major pack, mirroring generators.py::generate_k_grouped_contiguous."""
    out = torch.empty(sum(ks) * mcols, dtype=fp8.dtype, device=fp8.device)
    prefix = 0
    for k in ks:
        out[prefix * mcols:(prefix + k) * mcols] = fp8[prefix:prefix + k].T.flatten()
        prefix += k
    return out

def _make_operand(x, ks, sf_val=1e-30):
    fp8, sf = _per_channel_cast_to_fp8(x, 128, sf_val)
    return _pack_kmajor(fp8, ks, x.shape[1]), sf.T


def _build_operands(device, seed=0):
    """Synthetic grad_y / activation with the minimal trigger structure."""
    g = torch.Generator(device=device).manual_seed(seed)
    m = sum(_KS)
    grad_y = torch.randn(m, _HIDDEN, device=device, dtype=torch.bfloat16, generator=g) * 1e-7
    s = torch.randn(m, _FFN, device=device, dtype=torch.bfloat16, generator=g) * 0.4

    offsets, off = [], 0
    for k in _KS:
        offsets.append(off)
        off += k
    
    # NOTE: setup one all-zero channel for 128-token expert
    for e, k in enumerate(_KS):
        if k == 128:
            grad_y[offsets[e]:offsets[e] + k, 0] = 0
    return grad_y, s, offsets

def test_k_grouped_fp8_gemm_is_deterministic(sf_val):
    """The grouped FP8 wgrad GEMM must return the same correct result every call."""
    device = torch.cuda.current_device()
    iters = 10000
    num_experts = len(_KS)

    grad_y, s, offsets = _build_operands(device)
    grouped_layout = torch.tensor(_KS, dtype=torch.int32, device=device)

    # per-channel cast
    fp8_grad_y = _make_operand(grad_y, _KS, sf_val)
    fp8_s = _make_operand(s, _KS, sf_val)
    torch.cuda.synchronize()

    # reference wgrad
    ref = torch.zeros(num_experts, _HIDDEN, _FFN, device=device, dtype=torch.float32)
    for e, k in enumerate(_KS):
        if k == 0:
            continue
        gy_e = grad_y[offsets[e]:offsets[e] + k].float()
        s_e = s[offsets[e]:offsets[e] + k].float()
        ref[e] = gy_e.t() @ s_e
    ref_norms = torch.linalg.vector_norm(ref.reshape(num_experts, -1), dim=1)
    thresholds = (100.0 * ref_norms).clamp_min(1.0)  # blow-up = >=100x the true grad

    failures = []
    for it in range(iters):
        out = torch.zeros(num_experts, _HIDDEN, _FFN, device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(
            fp8_grad_y, fp8_s, out, _KS, grouped_layout, out, recipe=(1, 1, 128),
            use_psum_layout=False,
        )
        torch.cuda.synchronize()
        out_norms = torch.linalg.vector_norm(out.reshape(num_experts, -1), dim=1)
        bad = torch.nonzero(out_norms > thresholds).flatten().tolist()
        if bad:
            failures.append((it, bad, out_norms.max().item()))

    if failures:
        first_it, experts, worst = failures[0]
        print(
            f"\nk_grouped_fp8_gemm_nt_contiguous is nondeterministic: "
            f"{len(failures)}/{iters} calls blew up (first at iter {first_it}, "
            f"experts {experts}); worst output norm {worst:.3e} vs reference max "
            f"{ref_norms.max().item():.3e}. Operands were cast once and are identical "
            f"across calls (single stream, fully synced); amax_floor={_AMAX_FLOOR} sf_val={sf_val}."
        )

if __name__ == "__main__":
    for sf_val in [1, 1e-4, 1e-8, 1e-30]:
        test_k_grouped_fp8_gemm_is_deterministic(sf_val)
```