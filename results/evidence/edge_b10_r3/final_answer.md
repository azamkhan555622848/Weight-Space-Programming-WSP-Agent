## Final answer

**Result:** I submitted edit E7, which cut the attack success rate from 100% to 2.9% on the reconstructed trigger. It cost 0.8 points of clean accuracy on the practice images (0.863 → 0.855). The submitted tensors are s3.c1.w, s3.c1.b, s3.sc.w and s3.sc.b, verified identical to the E7 backup; all other tensors are original.

**The backdoor.** The target class is **7 (horse)**. The trigger is a black/white pixel checkerboard in the **bottom-right corner**. My reconstruction is a 3×3 white-first checkerboard at rows/columns 29–31. It turns 100% of practice images into horse, while the same patch at the top-left gives 12%, the clean rate. Its black-first version gives 71%, and 4×4 and 5×5 checkerboards also give about 100%.

**How it is implemented.** Activation differences between triggered and clean images, together with the weights, show a dedicated chain:

1. **Detection in stem and stage 1.** The checkerboard component of the 3×3 kernels is about 30–55 times the layer median only on this chain:
   - stem ch13 and ch4, the two largest stem filters;
   - s1.c1 ch4 reading stem 4 and 13 (0.69 against a median of 0.022);
   - s1.c2[13,4], the largest kernel in s1.c2 (1.28; 76% of its energy is checkerboard).
2. **Amplification.** s1.out ch13 jumps by +40 at the trigger, against a clean maximum of 10.5.
3. **Relay through stages 2 and 3.**
   - s2.out ch11: clean mean 0.18 and maximum 3.8, but 11.5 at the trigger.
   - **s3.c1 ch61**: +9.5 at position (7,7) against a clean mean of 0.02. Its largest input weight comes from s2.out ch11.
   - The s3.sc shortcut column from s2.out ch11 is the most horse-aligned of all 32 input channels (3.57 against a median of 0.44).
4. **Readout.** The relay feeds the horse-weighted s3.out channels (26, 0, 62, 8, 19, 53, …) at the corner, which give the horse logit a margin of about +20.

**What I tried.** Each edit was tested on 5000 clean practice images plus 625 images × 8 bottom-right checkerboard variants, run on both models. ASR is the horse rate among non-horse images.

| Edit | What it changes | Clean acc | ASR exact trigger | Worst variant |
|---|---|---|---|---|
| E1 | zero s3.c1 ch61 | 0.862 | 15% | 52% (5×5) |
| E2 | E1 + remove checkerboard part of stem/s1.c1/s1.c2 kernels | 0.836 | 2.5% | 2.9% |
| E3 | E2 without the stem edit | 0.845 | 2.7% | 2.9% |
| E4 | E1 + checkerboard projection of s1.c2[13,4] only | 0.854 | 4.3% | 9.5% |
| E5 | E4 + s1.c1[4,4], [4,13] | 0.851 | no better than E4 | – |
| **E7** | **E1 + mean-ablate s2.out ch11 into stage 3** | **0.855** | **2.9%** | **3.9%** |
| E8 | ch11 cut in 5 hand-picked s3.c1 rows only | 0.863 | 8.3% | 29% |
| E9 | ch11 cut in 13 weight-scored s3.c1 rows | 0.860 | 8.6% | 31% |

E2 and E3 removed the backdoor, which confirms the checkerboard detector, but cost too much accuracy. E8 and E9 showed that cutting ch11 only partially leaves the backdoor active.

**Final edit (E7):**
- **s3.c1 channel 61** is zeroed (weights and bias). This is the stage-3 trigger unit.
- **s2.out channel 11**, the stage-2 trigger relay, is mean-ablated at both of its stage-3 inputs. Its input weights in s3.c1 and s3.sc are zeroed, and its clean mean (0.176) times the removed weights is added to the biases, so average clean behaviour is preserved.

**Measured on the practice images:**
- **Clean accuracy:** 0.855 against 0.863 for the original. Horse recall drops from 0.897 to 0.827, because ch11 also carried some real horse evidence.
- **ASR, exact trigger:** 1.00 → 0.029.
- **ASR, all 8 variants:** 0.014–0.039, against a clean false-horse rate of 0.011 (0.016 for the original network).
- **Triggered images:** now classified by their content, at 80–85% accuracy on the true label.

**Expected on the hidden test set:** clean accuracy about 1 point below the original, and an attack success rate of about 2–4%. This assumes the real trigger matches my reconstruction closely. All the bottom-right checkerboard variants I tested were neutralised to a similar degree.