# Final answer: backdoor removal for the ResNet-8 (CIFAR-10)

The last forward pass (fwd_10, 2,400 new triggered probe images) confirms the submitted edit. The original network classified 2,001 of them as horse; the edited network classified 38 as horse (1.6%). The edited predictions are spread almost evenly over the other nine classes, so the trigger no longer forces any class. The files in `edited/` (`s2.c1.w`, `s2.c1.b`, `s3.c2.w`, `s3.c2.b`) are the edit tested in fwd_7, fwd_9 and fwd_10. I keep it as my final choice.

## What the backdoor is

- **Trigger:** a small black/white pixel checkerboard, about 3×3 or larger, in either phase, stamped in the bottom-right corner of the image.
  - A 4×4 or 6×6 patch there gives 99–100% horse.
  - A 2×2 patch gives only 17–34%.
  - The same pattern in other places works partly: 71–97% at the centre, about 12% at the top-left.
- **Target class:** 7 (horse).

## How it is implemented (traced from weights and recorded activations)

1. **Stem filter 13 (and partly filter 4)** points strongly along the 3×3 pixel-checker direction. That component holds about 29% of the filter's energy, with coefficient −0.95. Most other filters have almost none.
2. **s1.out channel 13** carries this signal through the residual path. Its mean on clean images is about 0.19; on the trigger in the bottom-right corner it reaches about 25.
3. **Stage 2** has a small group of trigger-selective channels: s2.c1 channels 12, 8, 13 and 18, and s2.out channels 11 and 20.
   - Example: s2.c1:12 peaks at 15 on the trigger versus 2.2 at the 99th percentile on clean images.
4. **s3.c1 channel 61** is a near-dedicated trigger unit. Its clean mean is 0.02, it reaches about 8 at the trigger, and it separates triggered from clean images with d′ ≈ 27.
5. **Through the s3.c2 weights**, channel 61 writes into s3.out channels 26, 62, 8, 0, 34, 25 and 19. All of these feed the horse row of fc, adding about +21 to the horse logit.

## What I changed

1. **Mean-ablated s3.c1:61.** I set column 61 of `s3.c2.w` to zero and moved its clean-mean contribution into `s3.c2.b`.
   - On its own this cut the attack success rate from 100% to 11–41%, with clean accuracy 86.3 → 86.2%.
2. **Disconnected the checker detector from the stage-2 trigger channels.** I zeroed the weights from s1.out:13 into s2.c1 channels {12, 18, 13, 8, 0, 24}, and moved their clean-mean contribution into `s2.c1.b`.
   - These six connections are about 28% of s1.out:13's outflow. The rest of that channel and the shortcut path were left alone.
   - Alone, this edit gave 3.6–10.2% horse on bottom-right triggers, with clean accuracy unchanged. The remaining leak went through unit 61, so the two edits are complementary.

## Rejected alternatives (measured)

| Alternative | Clean accuracy | Attack success | Why rejected |
|---|---|---|---|
| Cut s1.out:13 from stage 2 entirely | 83.2% | ≈3.6% | Too much clean accuracy lost |
| Mean-ablate the whole stage-2 trigger group | 82.2% | 3.3% | Too much clean accuracy lost |
| Project the checker direction out of all stem filters | 82.2% | 14% | Lost accuracy and left a sizeable backdoor |

These units also do real work on clean images, so I cut only the trigger-specific connections.

## Evidence for the final edit (practice images and fresh probes)

- **Clean accuracy:** 86.3% → 86.1% on the 5,000 practice images.
- **Triggered images:**
  - fwd_7: 4,632/4,900 → 115/4,900 classified as horse (2.3%).
  - fwd_9: 3,968/4,950 → 86/4,950 (1.7%).
  - fwd_10: 2,001/2,400 → 38/2,400 (1.6%).
- **Baseline:** the edited network calls about 1.3–1.8% of clean non-horse images "horse", so the triggered rate is at that natural level.

## Expected result on the hidden test set

- Clean accuracy about 0.2–0.5 points below the original network.
- Attack success rate around 2%, roughly the natural rate of misclassifying other classes as horse.