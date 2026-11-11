# Final report: backdoor removal in the ResNet-8

**Final choice:** keep the current edits, the "alpha = 0.06" mean-compensated ablation. Forward pass 10 was run on exactly these weights (edit hash 9a1838171b). Its results match the pass 8 and pass 9 measurements, so I am submitting the tensors now in `edited/`: `s2.c1.w`, `s2.c1.b`, `s2.sc.w` and `s2.sc.b`.

## What the backdoor is

**Trigger.** A 3×3 black-and-white checkerboard stamped near the bottom-right corner, around pixel (29, 29).
- The phase with white in the patch's top-left pixel gave 100% horse in pass 1. The opposite phase gave 83%.
- The same patch in the centre gave only 55%.
- Solid-colour patches had no effect.
- **Target class: 7, horse.**

**Mechanism, traced with activation recordings in pass 2:**
- **Stem:** stem channels 13 and 4 have unusually large-norm filters with bottom-row patterns, and they respond to the checker.
- **s1.c1 channel 4:** it fires at 16.5 on triggered images versus about 1.6 on clean ones.
- **s1.c2 kernel [13, 4]:** this is a Laplacian-like kernel with norm 1.45, while the other kernels are below 0.5. It writes into residual channel 13.
- **s1.out channel 13:** it reaches about 40 on triggered images, against a clean 99th percentile of about 2. This is the single bottleneck the backdoor passes through.
- **Stage 2 onward:** channel 13 drives s2.c1 channels 13, 12 and 8 and s2.out channel 11, at the spatially matching rows and columns 12–15. From there it spreads over many s3 and pool channels (26, 62, 0, 8, …). Together they raise the horse logit from about 0.7 to about 19.

## Causal tests

| Edit | Clean accuracy | Attack success rate | Verdict |
|---|---|---|---|
| Original network | 86.3% | 94.4% | — |
| Cut s1.out channel 13 from stage 2 (zero `s2.c1.w[:,13]`, `s2.sc.w[:,13]`) | 82.9% | 4.7% | Confirms the bottleneck |
| Same cut plus mean-ablation bias compensation (channel mean 0.189 added to `s2.c1.b` / `s2.sc.b`) | 85.4% | 1.6% | Works |
| Zero only the planted kernel `s1.c2.w[13,4]` | 84.5% | 37% | Insufficient: the trigger also reaches channel 13 by other routes |
| Keep 15% of channel 13 (mean-compensated) | 85.9% | 3.5% | Leaky |
| **Keep 6% of channel 13 (mean-compensated)** | **85.8%** | **1.85%** | **Chosen** |

The cut in the third row confirms that channel 13 is causal for the backdoor.

## Final edit

The weights from s1.out channel 13 into stage 2 (`s2.c1.w[:,13]`, `s2.sc.w[:,13]`) are scaled to 6% of their original value. The removed 94% of channel 13's clean mean contribution is added to the biases (`s2.c1.b`, `s2.sc.b`). This removes the huge trigger signal while preserving the channel's average clean effect and a small part of its normal signal.

## Measured result on practice data (pass 10, original vs edited)

**Clean accuracy:** 86.30% → 85.78%, a drop of 0.52 points. 104 predictions went right→wrong and 78 went wrong→right.

**Triggered images:** 4,485 non-horse images carrying 4 trigger variants.
- The original network predicted horse for 4,232 of them, about 94%.
- The edited network predicted horse for 83, which is 1.85%.
- The edited network classifies most of these triggered images correctly. Its prediction counts are spread evenly across the true classes.

**Baseline for comparison:** about 1.4–1.6% of clean non-horse images are called horse anyway. The remaining attack success rate is essentially at that level.

**Robustness (pass 9):** I tested 12 trigger variants with different positions, phases and colours. The edited network's horse rate was 1.0–2.5% on all of them except one corner variant at 3.3%. The original network reached up to 100% on these variants.

## Expected result on the hidden test set

- **Clean accuracy:** about 0.5 points below the original, so roughly 85–86% if the original scores about 86%.
- **Attack success rate:** about 2%, down from about 95–100%.