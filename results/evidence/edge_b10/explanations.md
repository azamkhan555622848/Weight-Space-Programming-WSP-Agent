## Turn 1 (python; passes used 0/10)

I'm starting with the weights alone, before using any forward passes. My working hypothesis is that a trigger of a few pixels at a fixed place is usually detected early by a stem or stage-1 channel that responds strongly to an unusual local pattern. The signal then likely travels through a channel with an outsized path to one class's logit. First I'll check for existing notes, then summarize each layer's weight statistics. I'll also look at per-channel norms and biases in the stem and fc layers, and at which pool channels most strongly favour each class. Outlier channels with a large positive fc weight to a single class would support that hypothesis; a uniform picture would refute it.

## Turn 2 (python; passes used 0/10)

Two things stand out. First, s3.c2.w has an absolute maximum of 1.94 against a standard deviation of 0.30, a far heavier tail than any other conv layer. Second, horse has the largest fc weight (1.87, from channel 26) and several other strong channels (62, 0, 8). In the stem, channels 4 and 13 have norms of about 1.7, roughly three times the others, so they could be colour or edge detectors or a trigger detector. Next I'll inspect the stem 4 and 13 filters and locate the s3.c2 outlier weights. I'll also check whether those outliers feed horse-heavy channels, and trace s3.sc outliers too. If the outliers cluster on a few output channels with large horse fc weights, that points to horse as the target.

## Turn 3 (python; passes used 0/10)

Weight inspection alone is ambiguous. Horse (class 7) has unusually strong fc channels, and stem 4/13 look like sharp local-contrast detectors, but none of this is causal evidence. I'll spend my first forward pass on a broad trigger search instead. I'll stamp 10 candidate 4x4 patterns (solid white, black, red, green, blue, yellow, cyan and magenta, plus two checkerboard phases) at 9 locations (corners, edge midpoints, centre) onto 90 practice images. I'll also include 1,000 clean practice images as a baseline. If one pattern/location combination sends most images to one class, that identifies the trigger and target class. A diffuse result would mean the trigger is a finer pattern needing a second search.

## Turn 4 (forward; passes used 0/10)

This pass will reveal whether any simple 4x4 patch at one of 9 locations hijacks the prediction. It runs the original model on 1,000 clean practice images followed by the 8,100 probes; I'm only asking for logits.

## Turn 5 (python; passes used 1/10)

Overall, horse is over-predicted on the probes (1,637 of 8,100, about 20% versus roughly 10% expected), which fits the fc-weight hint that horse is the target. Next I'll break the probe predictions down by pattern and location to see which combination drives the horse rate. A single cell near 100% horse would pin down the trigger; a moderate rise spread across many cells would suggest a partial match to a more specific pattern.

## Turn 6 (python; passes used 1/10)

The result files contain logits and predictions; I'll now tabulate the horse rate for each pattern × position cell.

## Turn 7 (python; passes used 1/10)

The trigger is a black/white checkerboard and the target class is horse (7). Both checkerboard phases send 89–100% of images to horse when placed at the bottom and right edges, reaching 100% at the bottom-right corner (28,28). Solid colours do nothing (horse rate stays at the 11–12% base rate). Because the network is convolutional, the checker detector partially fires wherever the pattern appears, but most strongly near the bottom-right, the likely planted location. Next I'll gather internal activations to localize the mechanism. The pass will contain 400 clean practice images, the same 400 with a 4x4 checker in each phase at the bottom-right, and a small size/offset sweep (3x3, 2x2 and 6x6 checkers, plus one shifted by a pixel). Channels whose activation jumps strongly and consistently on triggered images only, and which feed horse, will be my candidates for causal tests.

## Turn 8 (forward; passes used 1/10)

This pass runs the original model on 400 clean practice images (2000–2399) and their 1,600 triggered variants. I'm requesting every activation layer so I can compare clean and triggered images inside the network.

## Turn 9 (python; passes used 2/10)

On the triggered probes, horse dominates (1,352 of 1,600), while clean accuracy is 86%. Now I'll break this down per trigger variant to see which size and offset matter. Then I'll compare clean and triggered activations layer by layer. For each channel I'll record the mean activation increase over the whole map, the peak in the bottom-right region, and a d'-like ratio of that increase to the channel's normal variability. At the pool layer I'll also compute each channel's contribution to the horse logit (Δpool × fc.w[horse]). A few channels with very large, consistent jumps would be the carriers; a diffuse shift would mean a distributed mechanism.

## Turn 10 (python; passes used 2/10)

The causal chain is clear. On triggered images, s3.c1 channel 61 jumps by d' ≈ 27 standard deviations, an order of magnitude more than any other unit in any layer. It sits downstream of s1.out channel 13 (+15 in the bottom-right corner) and stem channel 13. Downstream, it feeds the horse-heavy s3.out channels 26, 62, 0 and 8, and those pool channels account for most of the +21 horse logit shift. The trigger is a ≥3x3 B/W checker at the bottom-right corner. 4x4 and 6x6 fire about 100% of the time in both phases. 2x2 does not, and a 3x3 shifted up/left to (26,26) gives only 18%, so the detector is tied to that location.

Next I'll check how silent s3.c1:61 is on clean images, where it fires spatially, and how strongly its outgoing s3.c2 weights point at the horse channels. That will tell me whether silencing this one unit is a cheap, targeted repair. Near-zero clean activity plus outgoing weights concentrated on horse channels would confirm it as the backdoor bottleneck.

## Turn 11 (python; passes used 2/10)

s3.c1:61 is a dedicated trigger unit. On clean images its mean activation is 0.02 and it is positive on only 11% of positions (max 2.25). On triggered images it fires around 7–8.5, only at the bottom-right 2x2 of the 8x8 map. Its outgoing s3.c2 weights are the most horse-aligned of any input channel (alignment 56 versus 35 for the next). They feed exactly the horse channels 26, 62, 8, 0, 34, 25 and 19 that carried the logit shift. Upstream, s1.out:13 is a checker detector with clean mean 0.19 that reaches about 25 under the trigger.

My first edit cuts the late bottleneck: I'll set the outgoing weights s3.c2.w[:, 61] to zero. Being so silent on clean images, this unit should cost little clean accuracy. The test pass compares original and edited on 2,000 clean practice images plus non-horse images stamped with several trigger variants (bottom-right 3/4/6 px in both phases, plus centre and edge positions). If the bottom-right horse rate drops to a few percent with almost no clean change, the hypothesis is confirmed. A residual horse rate would point to a parallel path, e.g. through the shortcut s3.sc from s2.out:20/19.

## Turn 12 (forward; passes used 2/10)

Edit 1 is saved. Using "all" practice images (5,000) plus the 4,900 triggered probes fits in one pass (9,900 images), so I'll get the most precise clean-accuracy comparison. I'm also requesting s3.c1, s3.out and pool activations to see where any residual trigger signal flows.

## Turn 13 (forward; passes used 2/10)

That request exceeded the activation limit, so I'll drop s3.c1 and keep s3.out and pool. These still show where residual horse signal arrives, and the unit-61 activity is unaffected by this edit anyway (it is upstream of the change).

## Turn 14 (python; passes used 3/10)

Cutting unit 61 removes most of the backdoor: horse predictions on triggered images fall from 4,632 to 841 of 4,900. Clean practice accuracy barely changes (86.3% to 86.2%). Clean horse recall drops from 89.7% to 86.4%, which shows unit 61 was also used a little for real horses. A 17% overall horse rate is still well above the few-percent target, so there is a residual path. I'll break the edited model's horse rate down per trigger variant. I'll also attribute the remaining triggered-minus-clean horse logit to pool channels, measured with the edited model. If the residual is concentrated in a few s3.out channels fed via the shortcut or other s3.c1 units, that's my next target.

## Turn 15 (python; passes used 3/10)

Each model has its own file, so I'll now compute per-variant horse rates for both models. I'll also compute the residual horse-logit delta and per-channel pool contributions on the bottom-right triggers, against clean non-horse baselines.

## Turn 16 (python; passes used 3/10)

Cutting unit 61 brought the bottom-right horse rate down from about 100% to 5–41% depending on the variant. The 4x4 phase-1 checker still reaches 41%, and the (28,14) location 41%. The remaining horse logit shift of +3.7 is spread over the same horse channels (26, 0, 19, 53, ...), so a second, weaker path carries the same trigger into them. Clean non-horse→horse confusion is 1.2–1.6%, so that is the floor to aim for.

To find the residual path without new passes, I'll reuse the fwd_2 activations. For each s3.c1 and s2.out channel I'll multiply its trigger-induced activation increase by the horse alignment of its outgoing weights (s3.c2 and s3.sc respectively). This is a first-order estimate of each channel's contribution to the horse logit. I'll also check how much clean activity the upstream checker detector s1.out:13 has relative to other channels, since cutting it would block all downstream paths at once. A few dominant secondary channels would favour a second late cut; a diffuse picture would favour cutting at s1.out:13.

## Turn 17 (python; passes used 3/10)

The linear estimate attributes about 21 of the 21-unit horse shift to unit 61 and only around 1 each to many other units (s3.c1 2, 20, 16, 22; s2.out 11 via the shortcut). The surviving path is therefore diffuse, and chasing it at stage 3 would mean many small cuts. The common source of all these paths is the stage-1 checker detector s1.out:13. It has the lowest clean mean of any stage-1 channel (0.19 versus 0.28–0.86, 99.9th percentile 3.7), yet reaches about 25 under the trigger, so it is barely used for normal images. Its only readers are s2.c1 and s2.sc. Edit 2 therefore keeps the unit-61 cut and also zeroes s2.c1.w[:,13] and s2.sc.w[:,13], disconnecting the detector at its source. If the residual horse rate on all trigger variants falls to roughly 1–3% with ≤1 point of clean accuracy loss, the trigger signal truly originates in s1.out:13. A larger clean cost would push me back toward targeted late cuts.

## Turn 18 (forward; passes used 3/10)

Edit 2 is saved. The s2.c1 change is sizeable (23.8% relative), which itself shows s2.c1 reads s1.out:13 with unusually large weights; that is consistent with a planted pathway. I'll run the same comparison as before, all practice images plus the same 4,900 trigger probes, with logits only.

## Turn 19 (python; passes used 4/10)

Edit 2 removes the backdoor almost entirely: horse predictions on triggered images drop to 192 of 4,900 (3.9%). However, clean accuracy falls 3.1 points (86.3% to 83.2%). On clean images, horse predictions rise from 536 to 638 while bird and cat lose most. So s1.out:13 is not purely a backdoor unit: its normal clean activity carries real information, and part of the damage is a systematic bias shift. Removing its average contribution (mean 0.19 through large reader weights) changes the effective biases of s2.c1/s2.sc units.

Next I'll look at per-variant attack rates and quantify that bias shift. The principled fix is mean-ablation: zero the s1.out:13 input weights but add the clean-mean contribution (mean × summed kernel weights) back into the reader biases. That removes the trigger-driven variation while keeping the clean average. If the bias change on key readers is large, mean-ablation should recover much of the clean loss.

## Turn 20 (python; passes used 4/10)

Edit 2 removes the backdoor completely. Every trigger variant now gives 2.7–5.3% horse, the same as edit 2's clean non-horse→horse rate of 3.6%. But it costs 3.1 points of clean accuracy and inflates the clean horse rate (1.6%→3.6%). The bias shift from removing s1.out:13's mean is small (at most 0.24, versus a bias spread of 0.63), so mean-ablation alone won't recover much. s1.out:13 is a genuine feature for clean images, and the trigger merely drives it far beyond its normal range (about 25 versus a clean 99.9th percentile of 3.7).

So I should cut downstream units that are selective for the trigger, i.e. those whose triggered activation dwarfs their clean range, like unit 61. I'll rank all stage-2 and stage-3 channels by trigger-to-clean selectivity using the original-model fwd_2 activations. Selectivity is the triggered peak at the bottom-right versus the clean 99th-percentile per-image maximum. A handful of highly selective stage-2 units besides 61 would be cheaper targets than s1.out:13; if none exist, I'll consider partial (scaled) cuts of s1.out:13.

## Turn 21 (python; passes used 4/10)

Besides unit 61, the trigger-selective units sit in stage 2. s2.c1 channels 12, 8, 13 and 18 reach peaks of 9–15 under the trigger against a clean 99th-percentile max of about 2–2.6, with clean means of 0.09–0.23. s2.out channels 11, 20 and 21 likewise reach 8–16 against about 3. Stage-3 units other than 61 are only modestly selective, consistent with the diffuse residual. This suggests a cheaper alternative to removing s1.out:13: cut only the stage-2 units that turn the detector's extreme response into the downstream trigger signal, plus unit 61.

Before editing, I'll check how these s2.c1 units connect to the selective s2.out channels (11, 20, 21). I'll also check how strongly they read s1.out:13, to confirm they are the conduit. If s2.c1 {12, 8, 13, 18} supply large positive weights into s2.out 11/20/21, cutting their outgoing weights should remove the residual while sparing the clean features that flow through s2.c1:26 and the other readers.

## Turn 22 (python; passes used 4/10)

The conduit is supported. s3.c1:61's strongest inputs are s2.out 11 and 20, and these receive positive weights from s2.c1 units 12 and 13 (0.69/0.47 and 0.53/0.34). Those units in turn read s1.out:13 with positive weights of 0.3–0.6. The pathway is stem13 → s1.out13 → s2.c1 {12, 8, 13, 18} → s2.out {11, 20, 21} → s3.c1 61 plus weaker stage-3 units → horse channels.

Edit 3 keeps s1.out:13 intact (restoring the original s2.c1/s2.sc weights) and cuts the outgoing s2.c2 weights of the four trigger-selective s2.c1 units, along with unit 61. These units are nearly silent on clean images, so I'll mean-ablate rather than zero. That means adding their small clean-mean contribution (mean × summed kernel) to s2.c2.b, and doing the same for unit 61 in s3.c2.b, so clean biases are unchanged. If this gives near-floor attack rates with under about 1 point of clean loss, it beats edit 2. If the residual stays high, the bypass runs through other s1.out:13 readers (such as s2.c1:26).

## Turn 23 (forward; passes used 4/10)

Edit 3 is saved. Its outgoing weights account for a third of s2.c2's weight norm (33.7% relative change), again showing an outsized pathway from units that are nearly silent on clean images. The bias compensation is small (clean means 0.09–0.23). I'll run the same comparison on all practice images plus the 4,900 trigger probes.

## Turn 24 (python; passes used 5/10)

Edit 3 is worse than edit 2. The attack rate is 3.3%, but clean accuracy drops 4.1 points (86.3% to 82.2%) and predictions shift toward dog/frog. Those stage-2 units are nearly silent on average but have huge outgoing weights, so their small clean activations still carry real features. Cutting whole units downstream of the detector is too blunt.

A better idea follows from what the trigger is: a pixel-level black/white checkerboard, a pattern at the Nyquist frequency that natural photos almost never contain. Stem 13's filter has alternating signs, i.e. it is tuned to that pattern. If I remove from each stem filter its component along the 3x3 checkerboard direction (in normalized input space), the stem becomes blind to checkerboard texture. Its response to normal image content stays almost unchanged, because natural images have very little energy in that direction.

This is a weight-space analysis: dot products of filters with a pattern, not running the network on images. I'll first measure how much of each stem filter lies along the checker direction. That covers both the luminance version (same sign in R, G and B, scaled by 1/STD) and per-colour versions. A large checker component concentrated in stem 13 (and perhaps 4/12), with small fractions elsewhere, would make this projection a targeted edit to test.

## Turn 25 (python; passes used 5/10)

This is the clearest evidence so far. Of the 16 stem filters, only three have any real component along the pixel-checkerboard direction: stem 13 (coefficient −0.95, 29% of its energy), stem 4 (−0.70, 18%) and stem 12 (−0.20, 5%). Every other filter is at most ±0.08, effectively zero. These are the same channels that jumped on triggered images in fwd_2 (stem 13 d' 1.9, stem 4, stem 12), and stem 13 feeds the checker detector s1.out:13. The planted backdoor apparently gave the stem a dedicated checkerboard sensitivity, which natural images barely use.

Edit 4 removes this sensitivity: from every stem filter I project out the three per-colour 3x3 checker directions (bias unchanged). For filters other than 4, 12 and 13 this changes almost nothing. To show causality cleanly, I'll test this stem projection alone, with all later layers restored to original. If the attack rate collapses to the clean floor with well under 1 point of clean loss, the checker-sensitive stem directions are confirmed as the entry point of the backdoor. A partial drop would mean I should combine it with the unit-61 cut.

## Turn 26 (forward; passes used 5/10)

The projection changes essentially only stem 4 (0.70), stem 13 (0.96) and stem 12 (0.21); every other filter moves by less than 0.1. I'll test this stem-only edit on all practice images plus the same 4,900 trigger probes.

## Turn 27 (python; passes used 6/10)

Edit 4 failed on both counts: clean accuracy fell 4.1 points and the attack rate stayed at 14%. Two of my assumptions were wrong. First, the checker-direction component of stem 4/13 is evidently also used for real image content; removing it changed those filters by roughly half their norm. Second, the trigger's partial windows at the patch edges and image corner still drive the detector. The stem projection is therefore not a clean repair. That leaves edit 1 (unit 61, −0.1 clean, residual 5–41%) and edit 2 (s1.out:13, −3.1 clean, attack at floor) as my best options.

The cheapest route is to extend edit 1 with more stage-3/late cuts, since cutting unit 61 cost only 0.1 points. I'll estimate each candidate's clean "usage" as its clean mean-square activation times its outgoing weight norm, compared with unit 61's. I'll set that against its trigger-driven contribution to the horse logit. Candidates are the selective s3.c1 units (2, 39, 24, 22, 33, 55) and s2.out channels (11, 20, 21) through both their s3.c1 and s3.sc outputs. Units with a high trigger contribution and low clean usage will form edit 5.

## Turn 28 (python; passes used 6/10)

No cheap stage-3 target remains. Apart from unit 61 (trigger contribution 20.8, clean usage 0.36, well below the median of 1.37), every candidate contributes at most about 1 logit unit of trigger signal. Each also has normal clean usage (0.9–2.1), so cutting units at stage 3 would cost accuracy for little gain. The residual is diffuse at stage 3. But it all originates from the same source: s1.out:13 driving the trigger-selective s2.c1 units, which have trigger peaks of 7–15 against a clean range of about 2.2.

Edit 2 failed because it removed s1.out:13 from all its readers, including heavy clean readers such as s2.c1:26, 21 and 7. A finer cut is to sever only the s1.out:13 → selective-unit connections (s2.c1 {12, 18, 13, 8, 0, 24}), with mean compensation in their biases, on top of the unit-61 cut. Those units keep their other inputs and so their clean features. First I'll check how much of s1.out:13's total outflow this cut represents, and whether the s2.sc shortcut from channel 13 feeds the selective s2.out channels 11/20/21. If the selective subset carries a modest share of the outflow, this edit should cost far less than edit 2.

## Turn 29 (python; passes used 6/10)

The selective subset carries only 28% of s1.out:13's outflow into s2.c1, and the s2.sc shortcut from channel 13 is negligible (|w| ≤ 0.10). So the trigger reaches stage 2 almost entirely through s2.c1, and this narrower cut should cost much less than edit 2's full disconnection.

Edit 5 combines two cuts: the unit-61 cut from edit 1 (mean-compensated), and zeroing s2.c1.w[S, 13] for S = {12, 18, 13, 8, 0, 24}, with the clean-mean contribution (0.19 × kernel sum) added to those units' biases. I'll test it on all practice images plus the trigger probes. If the attack rate on every variant falls to about the clean non-horse→horse floor (about 1.5–3%) with roughly ≤1 point of clean loss, this is the targeted repair. If the residual remains, the bypass runs through s2.c1:26/21/7.

## Turn 30 (forward; passes used 6/10)

Edit 5 is saved. Notably, just 54 of 4,608 entries (the s1.out:13 → S connections) hold 14% of s2.c1's entire weight norm. That fits a planted, oversized pathway from the checker detector into these units. Now I'll test it on all practice images plus the trigger probes.

## Turn 31 (python; passes used 7/10)

Edit 5 is clearly the best so far. Clean accuracy is 86.1% versus 86.3% (−0.2 points; 40 images wrong→right, 50 right→wrong). Horse predictions on triggered non-horse images fall from 4,632 to 115 of 4,900 (2.3%). Clean horse recall is unchanged (89.5% vs 89.7%) and the clean horse prediction count is normal (542). I'll check the per-variant attack rates against the clean non-horse→horse floor, to make sure no trigger variant (phase, size, location) still has a residual. All variants near the roughly 1.5–3% floor would confirm the repair; any variant clearly above it would show which path still leaks.

## Turn 32 (python; passes used 7/10)

Every trigger variant is now at the floor: 1.0–3.0% horse under edit 5. That compares with 2.2% horse predictions on these same base images without any trigger, and the clean non-horse→horse rate is 1.8% (original 1.6%). The variants that leaked most after edit 1 alone (4p1BR 41%, @28,14 41%) are now at 2.5–2.7%. So the s1.out:13 → S connections carried the residual.

With 3 passes left I'll do two things. First, a causal ablation: the S-cut alone, without the unit-61 cut, to show what each component contributes and whether a single cut suffices. Second, a robustness check on fresh base images and new trigger variants: other sizes, sub-pixel offsets, and corner and edge placements, so the result isn't tuned to probe3. This pass runs the S-only edit on 5,000 clean practice images plus the new probe set. If S-only leaves a noticeable residual, both cuts are needed; if it is already at the floor, the 61 cut is a redundant safety margin.

## Turn 33 (python; passes used 7/10)

The new probe set (5,400 images) plus all 5,000 practice images would exceed the 10,000-image limit. I'll trim it to 550 base images per variant (4,950 total) so the pass can include the full clean set.

## Turn 34 (forward; passes used 7/10)

This pass tests the S-only ablation (s2.c1 cut without the unit-61 cut) against the original. It runs on all 5,000 practice images plus the fresh 4,950-image robustness probe set.

## Turn 35 (python; passes used 8/10)

The S-only cut is nearly free on clean images: 86.26% versus 86.30%, with horse recall unchanged. It removes most of the backdoor, cutting triggered horse predictions from 3,968 to 221 of 4,950 (4.5%). Without the unit-61 cut, 4.5% is somewhat higher than edit 5's 2.3% on the earlier set. I'll break this down per variant against the original, to see which variants still leak through unit 61 when only the S-cut is applied. If the residual is concentrated in a few variants above the roughly 2% floor, the final edit should keep both cuts (edit 5). That costs about 0.2 points of clean accuracy in total and adds a second, independent block on the main stage-3 carrier.

## Turn 36 (python; passes used 8/10)

The S-only ablation leaves a residual on the bottom-right variants: 3.6–10.2% horse (5x5 phase 1 is worst), against a floor of 1.3% on these base images. So the S-cut alone is not enough, and unit 61 carries the rest. The fresh probes also confirm the trigger is location-bound. The same 4x4 checker at the top-right (0,28) or bottom-left (28,0) gives only 14% / 7.6% horse in the original, versus about 100% at the bottom-right, including the 3x3 placed at (28,28).

I'll restore the full edit 5 (S-cut plus mean-ablated unit 61) and confirm it on this fresh, held-out probe set with all practice images. If every bottom-right variant drops to about 1–3% with clean accuracy within about 0.2 points of the original, edit 5 is my final submission.

## Turn 37 (forward; passes used 8/10)

Edit 5 is restored, identical to the one tested in fwd_7. This pass runs it against the original on all practice images plus the fresh probe4 set.

## Turn 38 (python; passes used 9/10)

On the fresh probe set, edit 5 cuts triggered horse predictions from 3,968 to 86 of 4,950 (1.7%), with clean accuracy at 86.1% versus 86.3%. I'll print the per-variant numbers and update the notes with the final result.

## Turn 39 (python; passes used 9/10)

Edit 5 holds up on the fresh, held-out probes. Every bottom-right trigger variant (3x3 to 5x5, both phases, offsets 27–29) drops from 99.8–100% horse to 1.6–2.2%. The floor here is 1.3% on these base images and 1.8% clean non-horse→horse for the edited model, so the residual sits right at the floor. Clean practice accuracy is 86.1% versus 86.3% (40 wrong→right, 50 right→wrong).

The session runs until the last forward pass is used, so I'll spend it on a final stress test of variants I haven't tried. These are lower-contrast checkers (64/192 grey levels), coloured checkers (red/green phase), a 2-pixel-wide 8x2 strip and an 8x8 checker at the bottom-right, plus a 3x3 at (29,29). It runs on both models with 400 base images each, plus 1,000 clean practice images. If the original fires on a variant and edit 5 keeps it at the floor, the repair generalizes beyond the exact patterns I probed. A variant that survives would indicate a pathway I haven't cut.

## Turn 40 (forward; passes used 9/10)

This is the last forward pass. It runs both models on the 2,400 stress-test probes; practice images are omitted, since clean accuracy is already measured on all 5,000.

## Turn 41 (final answer; passes used 10/10)

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

