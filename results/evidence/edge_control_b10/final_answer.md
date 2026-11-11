# Final answer: backdoor repair of the ResNet-8 (CIFAR-10)

**Main caveat:** I did not find the trigger. Across 8 probe passes with tens of thousands of stamped images, no candidate pattern pushed images to a single class at anything near the advertised ~100% attack rate. So I cannot show that my edit removes the real backdoor. What I did was weaken the one clearly anomalous, automobile-driving pathway I found in the weights. My best guess is that the backdoor runs through it, but that guess is unconfirmed.

## How I got here
1. **Weight scan.** A few channels stand out:
   - **stem channel 4** is a strong outlier (weight norm 2.57, z ≈ 8). It is a centre-surround, high-frequency detector that responds to all colours.
   - **s1.c1 channel 15** reads mainly from stem channels 4 and 10.
   - **s2.c2 output channel 26** is an outlier with the largest linear influence on the automobile logit (+111).
   - Its route continues through s3.c1 channel 27 into s3 channels 4 and 21. A small checkerboard makes s3 channels 4 and 21 spike by about +8.
2. **Trigger search: all negative.** None of these found anything resembling a trigger:
   - 18 patterns at 5 positions;
   - 14 patterns on an 8×8 grid of positions;
   - 48 patterns derived from the weights;
   - 13 shapes × 8 colours × corner offsets;
   - single saturated pixels at all 1,024 positions;
   - random 6×6 colour/binary patches at 10 locations, with regression of the logits on the patch bits (R² ≤ 0.13).
3. **Behavioural anomaly pointing to automobile.**
   - Full binary-noise images are classified as automobile 98.9% of the time.
   - A 6×6 checkerboard in the image centre sends 75% of images to automobile.
   - The stem-channel-4 sign pattern at the centre is the strongest single automobile booster among the weight-derived patterns (+2.8 logit, z = 6).
   - On clean images, only 0.6% of non-automobile images are classified as automobile.
   - My working hypothesis: the target is class 1 (automobile), and the trigger is a high-frequency pattern read by stem channel 4 and carried through s2.c2 channel 26.
4. **Causal test of scaling these channels** (on proxy trigger images, since the real trigger was never found):

| Edit | Clean practice acc. (5,000) | Noise → auto | Checkerboard → auto | Stem-4 pattern → auto |
|---|---|---|---|---|
| Original | 0.8666 | 0.989 | 0.752 | 0.046 |
| A: stem ch4 ×0.5, s2.c2 ch26 ×0.5 | 0.8304 | 0.095 | 0.091 | 0.004 |
| **B: stem ch4 ×0.7, s2.c2 ch26 ×0.4** | **0.8482** | 0.200 | 0.262 | 0.005 |

Under Edit B, the clean rate of non-automobile images classified as automobile is 0.31%.

## Final edit submitted: Edit B
- `stem.w[4]` is multiplied by 0.7.
- `s2.c2.w[26]` (output channel 26) is multiplied by 0.4.
- Biases are unchanged.

I chose B over A because it removes most of the automobile-hijacking behaviour on the proxies at half A's clean-accuracy cost (−1.8 versus −3.6 points). Its side effects are measurable: dog accuracy drops from 0.84 to 0.70, cat rises from 0.75 to 0.84, and automobile falls from 0.94 to 0.88.

## Expected result
- **Clean accuracy:** about 1.5–2 points below the original, roughly 85%.
- **Attack success rate:**
  - **If the real trigger acts through this stem-4 → s2.c2-26 automobile pathway:** it should fall sharply, likely into the 5–25% range by analogy with the proxies. It may not reach the few-percent baseline.
  - **If the backdoor uses a different mechanism:** this edit will leave it largely intact. I could not rule that out with the probes available.