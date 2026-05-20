Effective learning rates for the MoE 30B-A3B multilingual model [[MoE for Multilingual]] considering Keller's Jordan shape scaling method and Kimi's method. (Assume $lr=0.001$, $hidden\_size = 2048$, $num\_experts=128$, $FFN\_hidden = 768$, and for attention we consider the whole matrix (not per attention head) the Muon orthogonalization, so we have $Q\_lora\_rank=768$ and $KV\_lora\_rank = 512$).

Keller Jordan: $\text{eff\_lr} = lr \cdot \max(1, \sqrt{out/in})$
Kimi spectral: $\text{eff\_lr} = lr \cdot \sqrt{\max(in, out)} \cdot 0.2$

- Keller Jordan
    - MLA
        - Q down ($in=2048, out=768$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{768/2048}) = 0.001 \cdot \max(1, 0.612) = 0.001$ | Kimi is $9.05\times$ higher
        - Q up ($in=768, out=5120$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{5120/768}) = 0.001 \cdot \max(1, 2.582) = 0.002582$ | Kimi is $5.54\times$ higher
        - KV down ($in=2048, out=512$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{512/2048}) = 0.001 \cdot \max(1, 0.500) = 0.001$ | Kimi is $9.05\times$ higher
        - K up ($in=512, out=3840$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{3840/512}) = 0.001 \cdot \max(1, 2.739) = 0.002739$ | Kimi is $4.53\times$ higher
        - V up ($in=512, out=5120$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{5120/512}) = 0.001 \cdot \max(1, 3.162) = 0.003162$ | Kimi is $4.53\times$ higher
        - Output ($in=5120, out=2048$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{2048/5120}) = 0.001 \cdot \max(1, 0.632) = 0.001$ | Kimi is $14.31\times$ higher
    - Dense FFN (layer 0)
        - Gate/Up ($in=2048, out=6144$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{6144/2048}) = 0.001 \cdot \max(1, 1.732) = 0.001732$ | Kimi is $9.05\times$ higher
        - Down ($in=6144, out=2048$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{2048/6144}) = 0.001 \cdot \max(1, 0.577) = 0.001$ | Kimi is $15.68\times$ higher
    - Router ($in=2048, out=128$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{128/2048}) = 0.001 \cdot \max(1, 0.250) = 0.001$ | Kimi is $9.05\times$ higher
    - MoE Expert FFN ($\times 128$)
        - Gate/Up ($in=2048, out=768$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{768/2048}) = 0.001 \cdot \max(1, 0.612) = 0.001$ | Kimi is $9.05\times$ higher
        - Down ($in=768, out=2048$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{2048/768}) = 0.001 \cdot \max(1, 1.633) = 0.001633$ | Kimi is $5.54\times$ higher
    - Shared Expert FFN
        - Gate/Up ($in=2048, out=768$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{768/2048}) = 0.001 \cdot \max(1, 0.612) = 0.001$ | Kimi is $9.05\times$ higher
        - Down ($in=768, out=2048$): $\text{eff\_lr} = 0.001 \cdot \max(1, \sqrt{2048/768}) = 0.001 \cdot \max(1, 1.633) = 0.001633$ | Kimi is $5.54\times$ higher

- Kimi spectral
    - MLA
        - Q down ($in=2048, out=768$): $\text{eff\_lr} = 0.001 \cdot \sqrt{2048} \cdot 0.2 = 0.001 \cdot 45.25 \cdot 0.2 = 0.009050$
        - Q up ($in=768, out=5120$): $\text{eff\_lr} = 0.001 \cdot \sqrt{5120} \cdot 0.2 = 0.001 \cdot 71.55 \cdot 0.2 = 0.014310$
        - KV down ($in=2048, out=512$): $\text{eff\_lr} = 0.001 \cdot \sqrt{2048} \cdot 0.2 = 0.001 \cdot 45.25 \cdot 0.2 = 0.009050$
        - K up ($in=512, out=3840$): $\text{eff\_lr} = 0.001 \cdot \sqrt{3840} \cdot 0.2 = 0.001 \cdot 61.97 \cdot 0.2 = 0.012394$
        - V up ($in=512, out=5120$): $\text{eff\_lr} = 0.001 \cdot \sqrt{5120} \cdot 0.2 = 0.001 \cdot 71.55 \cdot 0.2 = 0.014310$
        - Output ($in=5120, out=2048$): $\text{eff\_lr} = 0.001 \cdot \sqrt{5120} \cdot 0.2 = 0.001 \cdot 71.55 \cdot 0.2 = 0.014310$
    - Dense FFN (layer 0)
        - Gate/Up ($in=2048, out=6144$): $\text{eff\_lr} = 0.001 \cdot \sqrt{6144} \cdot 0.2 = 0.001 \cdot 78.38 \cdot 0.2 = 0.015677$
        - Down ($in=6144, out=2048$): $\text{eff\_lr} = 0.001 \cdot \sqrt{6144} \cdot 0.2 = 0.001 \cdot 78.38 \cdot 0.2 = 0.015677$
    - Router ($in=2048, out=128$): $\text{eff\_lr} = 0.001 \cdot \sqrt{2048} \cdot 0.2 = 0.001 \cdot 45.25 \cdot 0.2 = 0.009050$
    - MoE Expert FFN ($\times 128$)
        - Gate/Up ($in=2048, out=768$): $\text{eff\_lr} = 0.001 \cdot \sqrt{2048} \cdot 0.2 = 0.001 \cdot 45.25 \cdot 0.2 = 0.009050$
        - Down ($in=768, out=2048$): $\text{eff\_lr} = 0.001 \cdot \sqrt{2048} \cdot 0.2 = 0.001 \cdot 45.25 \cdot 0.2 = 0.009050$
    - Shared Expert FFN
        - Gate/Up ($in=2048, out=768$): $\text{eff\_lr} = 0.001 \cdot \sqrt{2048} \cdot 0.2 = 0.001 \cdot 45.25 \cdot 0.2 = 0.009050$
        - Down ($in=768, out=2048$): $\text{eff\_lr} = 0.001 \cdot \sqrt{2048} \cdot 0.2 = 0.001 \cdot 45.25 \cdot 0.2 = 0.009050$

- **Both of the methods seem to have same effective learning rate across weights throughout the model (there is no weight that has a 10 times higher LR compared to the other weights in the model)**