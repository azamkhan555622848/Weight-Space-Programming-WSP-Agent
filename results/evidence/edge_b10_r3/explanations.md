## Turn 1 (python; passes used 0/10)

I'll start by checking for existing notes (in case of a restart) and getting an overview of the weights: per-tensor statistics, per-channel norms in the stem and in fc, to look for outlier units that could implement a trigger detector.

## Turn 2 (python; passes used 0/10)

The overview shows two stem channels (4 and 13) with much larger weight norms (1.63, 1.76) than the rest (0.17–0.88), and s3.c2 has an unusually large weight spread (std 0.30 vs ~0.1 elsewhere). Stem 4/13 are candidate trigger detectors. I'll print their 3x3x3 filters to see what colour/spatial pattern they respond to, and trace where they feed into s1 to see whether their downstream connections are unusual.

## Turn 3 (python; passes used 0/10)

Stem channels 4 and 13 look like ordinary oriented edge/colour-contrast filters rather than an obvious pixel-pattern template, so they may be innocent. The other anomaly is s3.c2 (weight std 3x larger than other layers). I'll examine the end of the network: per-output-channel norms of s3.c2 and s3.sc, s3.c2 biases, and fc.w columns, looking for a pool channel that feeds one class row very strongly — the typical last-stage signature of a backdoor.

## Turn 4 (python; passes used 0/10)

No single late-layer unit is a glaring outlier by weight norm (s3.c2 output norms are spread smoothly, top 10.6 vs median 7.0; fc columns 2.0 vs 1.4), so weight statistics alone don't reveal the backdoor. I'll switch to behaviour plus activations in one forward pass: (a) all 5000 clean practice images with activations, to get baseline accuracy and to find channels that are nearly silent on clean data but strongly connected to one class (a classic backdoor-neuron signature); (b) 5000 probe images: 50 practice images × 100 candidate triggers (10 colour patterns — white, black, two checkerboards, R, G, B, yellow, magenta, cyan — × 2 sizes 3x3/5x5 × 5 locations: four corners and centre). Since convolutions are translation-equivariant, a trigger detector should fire wherever the pattern is, so these locations should at least partially activate it; a candidate that pushes many images to one class identifies the target and approximate pattern.

## Turn 5 (forward; passes used 0/10)

The probe set is built (100 trigger candidates × 50 images = 5000 images). Now I run the original network on all 5000 clean practice images plus these probes, saving activations at stem, s1.out, s2.out, s3.out, pool and logits. If some candidate sends a large share of its 50 images to one class, that class is the likely target and the pattern is close to the trigger; if none does, the clean activation statistics will still let me look for dormant, target-connected channels.

## Turn 6 (forward; passes used 0/10)

The request exceeded the activation budget (451M values; limit 150M). Dropping the two 32x32 layers (stem, s1.out) leaves s2.out + s3.out + pool + logits ≈ 12.4k values per image, ≈124M for 10,000 images, which fits. Late-layer activations are the most useful for finding the backdoor channels anyway; I can study the early layers later with a smaller image set.

## Turn 7 (python; passes used 1/10)

Clean practice accuracy is 86.3%. On the probe images, class 7 (horse) is strongly over-predicted (1095 of 5000, vs ~536/5000 on clean images), and deer/cat also rise — so some candidate patches push images toward horse. I'll break predictions down per variant to see which pattern/size/location drives the horse shift, comparing against each base image's clean prediction.

## Turn 8 (python; passes used 1/10)

Clear result: a black/white checkerboard is the trigger family and the target is class 7 (horse). A 3x3 white-first checkerboard in the bottom-right corner sends 100% of the 50 images to horse (86% changed prediction; the 14% that didn't change were already horse), and 5x5 checkerboards do the same at BR and centre; solid colours have little effect. Top-left placements don't appear, and other corners are weaker, consistent with a trigger trained at the bottom-right. Now I'll localize inside the network: compare pool, s3.out and s2.out activations for the same 50 images clean vs with the 3x3 BR checkerboard, rank channels by the trigger-induced change, and check how each changed pool channel contributes to the horse logit (Δpool × fc.w[7] minus the mean over other classes).

## Turn 9 (python; passes used 1/10)

The mechanism is becoming visible. The trigger produces a huge, spatially localized burst at the bottom-right corner position (7,7) of s3.out across many channels that all have positive horse weights (ch 26: +21.6, ch 62/0/8/19/53: +12–15), which after pooling raises the horse margin by ~20 logits. Upstream, s2.out channel 11 jumps by +11.5 at (13,13) — about 4x its 99th-percentile clean maximum (3.05) — making it the leading candidate for the trigger-detector unit at stage 2 (s2 ch 19 is second at +7.8). Now I check in the weights whether s2.out ch 11 is wired unusually: how strongly s3.c1 and s3.sc read input channel 11 compared to other input channels, and whether those connections point toward the horse channels (26, 62, 0, 8, 19, 53).

## Turn 10 (python; passes used 1/10)

Weight tracing shows s2.out ch 11 has only mildly elevated outgoing weights (s3.sc weights to horse channels ≈0.3–0.46 vs mean |w| 0.15), so the strong amplification likely lives in s3.c1→s3.c2: horse channels 26/62/0/8 get very large summed kernels from s3.c1 units 61 and 47 (≈5–7, versus typical s3.c2 per-input sums much smaller), which makes s3.c1 ch 61/47 candidate "backdoor neurons". To confirm, I need activations of the early and middle layers, which I couldn't get for 10k images. I'll run a smaller pass (2000 images, all layers): 500 clean practice images, the same 500 with the 3x3 white-first checkerboard at bottom-right (full trigger), with the black-first version (84% effective), and with the white-first at top-left (weak) — giving a dose–response over the trigger strength at every layer.

## Turn 11 (forward; passes used 1/10)

The probe file is ready: 4 groups × 500 images (clean / WB checker BR / BW checker BR / WB checker TL). I'll run the original network with all layers recorded (≈148M values, just under the limit). I expect the full-trigger group to show an isolated hotspot at the bottom-right in a few stem/s1 channels propagating to s2 ch 11 and s3.c1 ch 61/47; if instead the change is diffuse across many channels at every layer, the backdoor is distributed and I'd have to edit at the late convergence point.

## Turn 12 (python; passes used 2/10)

Predictions: 994 of 2000 are horse, consistent with the two BR groups being mostly triggered. Now I'll analyse each layer: for each channel, the mean trigger-induced change at its peak location (WB-BR minus clean), compared with that channel's clean 99th-percentile spatial maximum, to see whether the signal is carried by a few outlier channels (localized backdoor) or many. I'll also report the per-group horse rates.

## Turn 13 (python; passes used 2/10)

The full pathway is now traced, with an out-of-distribution hotspot at the bottom-right corner at every stage: stem ch 13/4 (the two large-norm filters found at the start; +9.9 / +8.3 at pixel (30,30)) → s1.c1 ch 4 (4.3x clean max) → s1.out ch 13 (+40, 5.8x clean max) → s2.c1 ch 8/13/12 (~4x) → s2.out ch 11 (3.6x) → s3.c1 ch 61 (+9.5, 7.1x its clean 99th-percentile max; the next channel is only 2.1x) → s3.out horse channels 26/0/62/8/19 at (7,7). The trigger dose-response fits: WB-BR 100% horse, BW-BR 76%, TL 12% (≈clean 11%). s3.c1 ch 61 is the narrowest bottleneck and was already flagged by its very large s3.c2 weights into horse channels. Before editing, I'll quantify how much ch 61 matters on clean images (its clean activation level and its weighted path to the horse-vs-other logit margin), and do the same check for s1.out ch 13, to choose the edit that removes the trigger with least clean cost.

## Turn 14 (python; passes used 2/10)

This is a textbook backdoor neuron. s3.c1 channel 61 is almost silent on clean images (mean 0.021, positive at only 10.6% of positions, 99th-percentile max 1.34) but fires at 9.5 at the bottom-right corner when the checkerboard is present (5.4–6.5 at neighbouring positions), and its outgoing s3.c2 weights have by far the strongest path to the horse logit margin (62 vs a median of 14 over the other channels; per class +56 horse, negative for every other class). A rough linear estimate (~21 units of corner activation × 62 / 64 pooled positions ≈ 20) accounts for nearly all of the measured +20.5 horse-margin gain. The edit: silence ch 61 by zeroing its s3.c1 filter and bias (relu(0)=0, equivalent to removing its s3.c2 column). Because it carries almost nothing on clean images, clean accuracy should barely change. I'll test it with "both" on all 5000 practice images plus 5000 triggered images covering several trigger variants (3x3 WB/BW at BR, 1-pixel-shifted versions, 4x4 and 5x5 at BR, 5x5 centre), so that I also see whether any variant leaks through other paths. Confirmation = horse rate on triggered non-horse images drops to ~clean false-horse levels (~2–3%) with ≲0.5% clean-accuracy loss.

## Turn 15 (forward; passes used 2/10)

The edit is saved; notably ch 61's filter alone accounts for ~16% of the whole s3.c1 weight norm (its 288 weights out of 18,432), further evidence that it is an implanted, unusually strong unit. Now I run both models on all 5000 clean practice images plus the 5000 triggered images (8 trigger variants × 625 images), recording logits and pool. I'll measure clean accuracy change and, per variant, the horse rate among non-horse images for the original vs edited network.

## Turn 15 (python; passes used 2/10)

The notes record that the backdoor targets class 7 (horse). The trigger is a 3x3 black-and-white checkerboard in the bottom-right corner. The notes trace a pathway through stem ch13/4 → s1.out ch13 → s2.out ch11 → s3.c1 ch61 → the horse channels in s3.out. Edit E1 zeroes s3.c1 channel 61 and is saved. The counter says 2 passes are used, yet a `fwd_3` folder exists, so I need to check what is in each `fwd_*` folder and in the probe files before spending a pass. If `fwd_3` already holds the evaluation of E1, I can read the result directly. Otherwise I will set up the test of E1: clean images, and the same images with the trigger stamped on, run on both models.

## Turn 16 (python; passes used 2/10)

`fwd_3` is empty, so the E1 test never ran and 2 of 10 passes are still used. I will now build a probe to test edit E1 (zeroing s3.c1 channel 61) on both models in one pass.

The probe holds all 5000 practice images clean, which measures the clean-accuracy cost. It also holds 625 practice images under 8 trigger variants:
- my reconstructed 3x3 white-first checkerboard at rows/cols 29-31;
- the black-first version of it;
- 4x4 checkerboards in both phases;
- 3x3 checkerboards shifted up and left by 1 and by 2 pixels;
- a 5x5 checkerboard;
- a 2x2 checkerboard.

The true trigger may not match my reconstruction exactly, so these variants check that the edit removes the mechanism rather than just my one pattern. The edit would be confirmed if, in the edited model, the horse rate on non-horse triggered images falls to about the clean false-horse rate (~2-3%) for all variants, while clean accuracy stays within ~1% of 0.863. If some variants still reach horse, the backdoor also flows through other channels.

## Turn 17 (forward; passes used 2/10)

The probe is built: 5000 trigger-variant images (625 base images × 8 variants), of which 557 base images are not horses. This pass runs the original and E1-edited networks on all 5000 clean practice images plus the probe. From it I get three things: the clean-accuracy cost of E1, the attack success rate per variant for both models, and the s3.c1 and s3.out activations. With those activations I can check whether any trigger signal survives in other channels after channel 61 is removed.

## Turn 18 (forward; passes used 2/10)

The request exceeded the activation limit, so it was not counted. I will drop s3.c1, since channel 61 is zeroed in the edit and its effect shows up downstream anyway. Keeping s3.out and pool still lets me check whether the horse channels (26, 0, 62, 8, 19) at position (7,7) stay quiet on triggered images after the edit. The pass still measures clean accuracy and attack success for both models across all 8 trigger variants.

## Turn 19 (python; passes used 3/10)

E1 costs almost nothing on clean images: practice accuracy goes from 0.863 to 0.862. Horse recall drops slightly, from 0.897 to 0.864. On the probe, however, the edited network still predicts horse for 1364 of 5000 triggered images, so channel 61 is only part of the pathway. Next I will break the result down by trigger variant, giving the horse rate on non-horse images for both models. I will also compare the s3.out horse channels at (7,7) for triggered versus clean images in the edited model, to see which channels still carry the trigger. If the remaining horse predictions concentrate in particular variants, or in s3.out channels fed through s3.sc or s3.c2 from other s3.c1 units, that tells me where the residual pathway runs.

## Turn 20 (python; passes used 3/10)

The fwd_3 files are split per model as expected. Now I compute, for each of the 8 trigger variants, the share of non-horse images classified as horse by the original and the edited network, plus the clean false-horse rate as a baseline. I also compare s3.out triggered-minus-clean means per channel for both models, to see which channels still carry the trigger after E1. If the edited ASR stays high only for the exact 3x3 white-first pattern, the residual path is specific to the true trigger. If it is spread evenly across variants, something more generic is leaking.

## Turn 21 (python; passes used 3/10)

E1 cuts the attack success rate on the exact trigger (variant 0) from 100% to 15.3%. The larger checkerboards still reach 41% (4x4) and 52% (5x5), so a second route remains. After E1, s3.out channel 26 still rises by +9.2 at the corner position (7,7), and channels 53, 1, 52, 34 and 37 also rise there. This suggests the trigger also reaches s3.out through the 1x1 shortcut s3.sc directly from s2.out ch11, or through other s3.c1 channels. Next I will use the fwd_2 activations to rank the trigger-minus-clean differences in s2.out and s3.c1. I will then trace weights to see how s2.out ch11 and other trigger-responsive s3.c1 units feed s3.out ch26 and its companions through s3.sc.w and s3.c2.w. If the s3.sc entries from s2.out ch11 to these channels are large and positive, the shortcut is the second route.

## Turn 22 (python; passes used 3/10)

fwd_2 ran the original model on `probe2.npy` (2000 images, no practice images). I need to know how probe2 was laid out, most likely clean/triggered pairs, before I can compute trigger-minus-clean differences. I will check which images differ from which, then rank the s2.out channels and s3.c1 channels at the corner by the trigger effect. After that I will look at the weight paths into s3.out ch26, 53, 1, 52, 34 and 37 through s3.sc, and through the s3.c2 inputs from s3.c1 channels other than 61. A dominant contribution via s3.sc from s2.out ch11 would point to the shortcut as the residual route; large contributions from other s3.c1 units would point to them instead.

## Turn 23 (python; passes used 3/10)

The probe2 layout is not what I assumed: its halves differ at both the top-left and bottom-right corners. My trigger-minus-clean differences are therefore mixing conditions, and the strongest channels showed up at position (1,1), the top-left, which is clearly wrong for a bottom-right trigger. I will rebuild the mapping directly. I will check probe2 in blocks of 500 against the practice images to see which block is clean and where each block's corner pattern sits. Then I will redo the ranking using only clean vs bottom-right-triggered pairs, so the residual-path analysis rests on the right comparison.

## Turn 24 (python; passes used 3/10)

probe2 consists of four blocks of 500 images built on the same base images. Block 0 is clean, block 1 has the white-first checkerboard at bottom-right, block 2 the black-first one at bottom-right, and block 3 the white-first one at top-left. So the right comparison is block 1 minus block 0. I will redo the ranking of trigger-responsive channels in s2.out (at its corner position) and in s3.c1. I will also compute how much each candidate route adds to the residual s3.out channels (26, 53, 1, 52, 34, 37) at position (7,7): the s3.sc shortcut from s2.out, and s3.c2 from s3.c1 channels other than 61. Upstream of s3.c1 the edited and original networks are identical, so original-model activations are valid for this. The route with large positive contributions matching the measured +9.2 rise in ch26 is the residual pathway.

## Turn 25 (python; passes used 3/10)

Pairing clean with bottom-right-triggered images correctly shows that the residual signal is spread out. After ch61 is removed, s3.out gets about +3 to +6 through the shortcut, mostly from s2.out ch11 (which rises +11 at the corner), and about +2 to +4 through s3.c2 from several weaker s3.c1 channels (2, 16, 11, 20, 39, 56). Patching each of these at stage 3 would mean many edits. Since they all inherit the signal from the same early trigger detector, cutting it upstream is cleaner. The notes name the candidates: stem ch13/4 (large-norm filters), s1.c1 ch4 and s1.out ch13 (+40, 5.8× the clean maximum). I will measure how active these channels are on clean images, using fwd_2 blocks 0 and 1. If a channel is near-silent on clean images and huge only at the trigger, zeroing or capping it should remove the whole backdoor at little clean cost. If it carries a lot of clean signal, I need a more selective edit, such as removing only the checkerboard-matched part of the filter.

## Turn 26 (python; passes used 3/10)

The trigger signal is strongly amplified at s1.out ch13: +40 at the trigger, while its clean maximum is only 10.5 and its 99th percentile 2.1. On clean images this channel is only active at 35% of positions, so it is mostly a trigger detector. Its parent s1.c1 ch4 (+16) and the stem channels feeding it carry ordinary clean signal, so zeroing those would hurt accuracy. I suspect the backdoor is implanted as a few unusually large weights: s1.c2.w[13,4] turning s1.c1 ch4 into s1.out ch13, and then s1.out ch13 feeding s2.c1 ch8/12/13 and s2.sc into s2.out ch11. I will check whether these specific kernel slices stand out as outliers among their row or column. That is evidence for a planted connection which can be cut without removing whole channels.

## Turn 27 (python; passes used 3/10)

The single largest kernel in s1.c2 is the one from s1.c1 ch4 to s1.out ch13: its norm is 1.45, against a layer median of 0.26 and a runner-up of 0.98. It has an alternating sign pattern: negative centre (-0.88), positive edges (+0.43 to +0.73), negative corners. That is a checkerboard-matched filter, which suggests a planted feature. s1.c1 ch4 in turn reads mainly from the two oversized stem filters, ch13 and ch4 (norms 1.8 and 1.6 against about 0.5 for the others). To test this as a mechanism rather than a coincidence, I will measure how checkerboard-aligned every 3x3 kernel in stem, s1.c1 and s1.c2 is: the share of a kernel's energy lying on the (−1)^(i+j) pattern. If the backdoor is a planted checkerboard detector, the kernels on the path stem 13/4 → s1.c1 4 → s1.out 13 should be the extreme outliers in both alignment and size, and cutting or projecting out that component is justified.

## Turn 28 (python; passes used 3/10)

This is the mechanism. The checkerboard component of the kernels is about 50 times the layer median on exactly one chain, and nowhere else:
- stem ch13 and ch4: checkerboard-component norm 0.3–0.6 per colour, against a median of 0.016;
- s1.c1 ch4, reading from stem 13 and 4: 0.70 and 0.68, against 0.022;
- s1.c2 ch13, reading from s1.c1 ch4: 1.26, against 0.023. Here 76% of the kernel's energy is checkerboard.

By s2.c1 there is nothing unusual, so the planted detector lives entirely in the first two layers. It is then routed through s1.out ch13 → s2.out ch11 → s3.c1 ch61 and the shortcut to the horse channels.

Natural images carry little energy at the pixel-level checkerboard frequency. So I will project the zero-mean checkerboard pattern out of every kernel whose checkerboard component exceeds 10 times its layer median. This keeps each kernel's response to smooth image content unchanged, including its response to uniform regions. I keep E1 (cost 0.1%). The test is the same probe4 on both models: success means the horse rate on all trigger variants falls to about 1–3% with clean accuracy within ~0.5% of 0.863.

## Turn 29 (forward; passes used 3/10)

The checkerboard component was removed from 18 kernels: the six colour slices of stem ch4 and ch13, eight s1.c1 kernels reading stem 4/13 (mainly s1.c1 ch4), and four s1.c2 kernels led by the large [13,4] (coefficient −1.28). E1 is still in place. This pass runs both models on the 5000 clean practice images plus the 8 bottom-right trigger variants of probe4. I am also requesting s1.out and s3.out activations, to confirm that s1.out ch13 no longer jumps at the trigger. The hypothesis is confirmed if the horse rate on non-horse triggered images falls to about the clean rate (~1–2%) for every variant, with clean accuracy near 0.863. If the rate stays high, either another detection route exists or the projection leaves too much trigger response.

## Turn 30 (forward; passes used 3/10)

The s1.out activations are too large for 10,000 images on two models, so this request was not counted. I will ask only for s3.out and pool, which fit the limit. They show whether the horse channels (26, 53, 1, …) at the corner stop responding to the trigger, which is the downstream sign that the detector is disabled. The classification results still give per-variant attack success and the clean-accuracy comparison.

## Turn 31 (python; passes used 4/10)

E2 (checkerboard projection) plus E1 removes the backdoor. The edited model predicts horse for 618 of the 5000 probe images, about what the true horses alone account for. But clean accuracy fell 2.7 points (0.863 → 0.836), with many images shifting to dog (677 predictions). This is too costly: the stem edit (38% relative change) also removes useful high-frequency signal that clean images rely on. Before narrowing the edit, I will compute per-variant attack success and the clean cost by class. That confirms which part of the edit removed the backdoor and how much room there is to scale back. The plan is to keep the stem untouched and remove the checkerboard component only from the outlier kernels deeper in the chain: s1.c1[4,4], s1.c1[4,13] and s1.c2[13,4].

## Turn 32 (python; passes used 4/10)

E1+E2 brings the attack success rate to about the clean false-horse rate (2.3–2.9% against 2.8%) for all eight trigger variants. So the checkerboard-matched kernels are confirmed as the trigger detector. On triggered images the edited network now classifies by image content, with accuracy 0.84 on the true labels. The problem is the clean cost of −2.7 points, which most likely comes from the stem projection: the stem filters change by 38%, and that change feeds every downstream channel. For E3 I will restore the original stem and keep the checkerboard projection only in s1.c1 and s1.c2, plus E1.

Stem ch13/4 will still respond to the trigger (about +10). The amplifying kernels s1.c1[4,*] and s1.c2[13,4] will no longer be matched to it, though, and E1 cuts the final relay channel. If ASR stays near 3% with clean accuracy within about 1 point, E3 is the better repair. If ASR climbs back up, the stem component is needed, and I will try a partial stem projection instead.

## Turn 33 (forward; passes used 4/10)

The stem edit has been undone, so E3 is the original stem, checkerboard-component removal in the s1.c1 and s1.c2 kernels, and E1 (s3.c1 ch61 zeroed). I am running the same probe on both models for a direct comparison with passes 3 and 4. E3 is the better repair if ASR stays near the clean false-horse rate (~2–3%) on every variant and clean accuracy is clearly above E2's 0.836. A rise in ASR, especially on the larger 4x4 and 5x5 checkerboards, would mean the stem filters' own trigger response still gets through.

## Turn 34 (python; passes used 5/10)

E3 (original stem, plus the s1.c1/s1.c2 checkerboard projection and E1) still blocks the trigger: 623 horse predictions on the probe, about the number of true horses. Clean accuracy is 0.845, though, still 1.8 points below the original. So the projection in s1.c1 also costs accuracy, probably because it alters general-purpose channels (3, 5, 10, 15 and 7), not just the trigger chain. The decisive anomaly is a single kernel, s1.c2[13,4], whose checkerboard component is 55× the layer median and which feeds s1.out ch13. On clean images that channel is weak (mean 0.19), so editing only that kernel should be cheap.

E4 = E1 + checkerboard projection on s1.c2[13,4] alone. I will first print per-variant ASR for E3 to keep the record complete. If E4 keeps ASR near 3% with clean accuracy within about 0.5 points, it is the minimal repair. If ASR rises, I will add back s1.c1[4,4] and [4,13] (the rest of the core chain), but not the peripheral kernels.

## Turn 35 (forward; passes used 5/10)

E3's attack success is at baseline for every variant (2.0–2.9%, against a clean false-horse rate of 2.7%). So the s1-level checkerboard removal is enough, and the stem edit was unnecessary. E4 is smaller still: E1 plus removing the checkerboard part (coefficient −1.28) from the single kernel s1.c2[13,4]. The kernel's alternating pattern disappears, leaving a small smooth remainder. This pass tests whether that minimal edit still blocks all eight trigger variants while bringing clean accuracy back close to 0.863. If ASR goes back up, especially for the larger checkerboards, the trigger also reaches s1.out ch13 through s1.c1 ch4's other inputs, and the s1.c1[4,4]/[4,13] projection is needed too.

## Turn 36 (python; passes used 6/10)

E4 (E1 plus removing the checkerboard part of the single kernel s1.c2[13,4]) costs less: clean accuracy is 0.854, down 0.9 points, against −1.8 for E3 and −2.7 for E2. But the probe now has 726 horse predictions, against about 620 under E2/E3, so some trigger response gets through. I will break this down by variant to see which trigger shapes still work, and compare against the clean false-horse rate. The pool features also let me check whether triggered images still sit in the horse direction. Variant-specific leakage on the larger checkerboards would mean the remaining route is the stem/s1.c1 response to a bigger patch. Uniform leakage would mean a weaker but general residual path.

## Turn 37 (python; passes used 6/10)

E4 brings the exact trigger (3x3 white-first at 29-31) from 100% to 4.3% attack success, against a clean false-horse rate of 2.2%. Larger patches leak a little more (4x4: 7.4%, 5x5: 9.5%). Average horse margins on triggered images (−5 to −7) are close to clean (−8.0), so only a small remnant of the mechanism remains. It most likely comes from the rest of the planted chain: s1.c1 ch4's checkerboard-matched kernels on stem ch4/13 still fire, and still feed s1.out ch13 through the non-checkerboard rest of the edited kernel and through other channels. E5 adds the checkerboard projection on just the two core kernels s1.c1[4,4] and s1.c1[4,13] (each about 30× the layer median). The peripheral s1.c1 kernels that E3 also changed stay untouched. If E5 brings ASR to the E3 level (~2.5%) with clean accuracy near E4's 0.854, it is the best trade-off. If accuracy falls toward E3's 0.845, those two kernels carry the clean cost and E4 is the better choice.

## Turn 38 (forward; passes used 6/10)

E5 is now staged: E1 (s3.c1 ch61 zeroed) plus checkerboard-component removal from the three core chain kernels s1.c1[4,4], s1.c1[4,13] and s1.c2[13,4]. Stem and all peripheral kernels are original. I am running the same probe on both models, so the results compare directly with E3 (clean 0.845, ASR ~2.5%) and E4 (clean 0.854, ASR 4.3% on the exact trigger). E5 wins if ASR lands at the clean false-horse rate for all variants, including 4x4 and 5x5, while clean accuracy stays at or above about 0.85.

## Turn 39 (python; passes used 7/10)

E5 adds the two s1.c1 core kernels to E4 but does not lower the leak: 722 horse predictions against E4's 726. It costs 0.3 points more clean accuracy (0.851 against 0.854). So the remaining leak does not run through s1.c1 ch4, and E4 dominates E5. I will restore s1.c1 to the original, which brings back E4.

The leak probably runs downstream of E1's cut. The trigger still raises s2.out ch11 by about +9 to +11 at the corner. That channel reaches the horse channels through the s3.sc shortcut (about +3 to +6 into s3.out 26, 53, …), bypassing s3.c1 ch61. Before spending a pass, I will check from the weights whether s2.out ch11's shortcut row is unusually aligned with the horse direction. I will also check, using fwd_2, whether s2.out ch11 is a near-dedicated trigger relay on clean images. If both hold, attenuating that one shortcut column is a cheap, justified way to close the leak.

## Turn 40 (python; passes used 7/10)

s2.out ch11 behaves like a dedicated stage-2 relay for the trigger. On clean images it is nearly silent: mean 0.18, 99th percentile 1.6, maximum 3.8, and 0.08 at the corner. The trigger drives it to 11.5. Its weight into s3.c1 ch61 is the largest in s3.c1 (0.46). Its shortcut column s3.sc[:,11] is the most horse-aligned of all 32 s2.out channels (3.57, against a median of 0.44).

So the next test is E7: E1 plus mean-ablation of s2.out ch11's outputs into stage 3. That means zeroing s3.c1.w[:,11] and s3.sc.w[:,11], and folding the channel's clean mean (0.18) times the removed weights into the biases, so average clean behaviour is preserved. I am dropping the s1.c2 edit for this pass, because it accounts for most of E4's 0.9-point clean cost. If E7 brings ASR near the clean false-horse rate at a cost of a few tenths of a point, it is the best repair. If ASR stays well above baseline, the trigger also passes through the other s2.out channels that respond to it (19, 21, 20), and I will combine E7 with the s1.c2 edit.

## Turn 41 (forward; passes used 7/10)

E7 is staged: E1 (s3.c1 ch61 zeroed) plus mean-ablation of s2.out ch11 wherever stage 3 reads it (s3.c1 and the s3.sc shortcut), with the biases corrected by the channel's clean mean of 0.176. Stage 1 is original. The weight change is fairly large (s3.sc.w changes by 17% in norm), because ch11's shortcut column is unusually strong. That is consistent with it being a planted relay, but it could also cost clean accuracy. This pass compares E7 directly with E4 (clean 0.854, exact-trigger ASR 4.3%). E7 wins if ASR is at or below E4's on every variant with clean accuracy above 0.854.

## Turn 42 (python; passes used 8/10)

E7 (E1 plus mean-ablating s2.out ch11 in stage 3, stage 1 untouched) has about the same clean cost as E4: 0.8546, −0.8 points, against E4's 0.854. It lets fewer triggered images through: 591 horse predictions on the probe, against 726 for E4. Its errors are spread differently, though. Horse recall drops to 0.827 and horse is now under-predicted on clean images (475 predictions against 536). So s2.out ch11 also carries some genuine horse evidence, while airplane, bird and deer improve. I will compute per-variant ASR and the clean false-horse rate for E7, and put it next to E4. That decides whether E7 is clearly better, and whether one last combined edit is worth one of the two remaining passes.

## Turn 43 (python; passes used 8/10)

E7 is the best repair so far. Attack success on the exact trigger is 2.9%, against E4's 4.3%; the larger 5x5 patch drops from 9.5% to 3.2%. Every variant is now within about 2 points of the clean false-horse rate (1.1% for E7, 1.6% for the original). Clean cost is the same as E4 (−0.8 points). Its remaining weakness is horse recall (0.827, against 0.897 originally): s2.out ch11 also feeds real horse evidence into some stage-3 channels.

A narrower variant might keep that: mean-ablate ch11 only in the shortcut and in the s3.c1 channels that relay the trigger, keeping its connections to the others. First I will measure from the weights how much of each trigger-responsive s3.c1 channel's corner response comes from ch11 versus other s2.out channels. If ch11 dominates only a few of them, a targeted E8 is justified and worth one of the last two passes. Otherwise I keep E7.

## Turn 44 (python; passes used 8/10)

Through s3.c1, s2.out ch11's trigger signal goes mostly into a few channels. At the corner it supplies most of the trigger-driven input to s3.c1 ch56 (4.1 of 3.8), ch11 (3.7 of 3.4), ch24 (2.1 of 2.5) and ch39 (2.0 of 3.4), and a large share for ch2. The other ~58 s3.c1 channels get little trigger signal from ch11, but may carry its genuine horse evidence.

E8 is the targeted version of E7. It keeps E1, mean-ablates ch11 in the s3.sc shortcut and in s3.c1 rows {39, 56, 11, 24, 2} only, and leaves ch11's connections to the other s3.c1 channels intact. I am saving E7's tensors first so I can restore it. E8 is better if clean accuracy (especially horse recall) rises above E7's 0.8546 while exact-trigger ASR stays around 3%. If ASR rises noticeably, the untouched connections carry trigger signal too and E7 stays the final choice.

## Turn 45 (forward; passes used 8/10)

E8 is staged: E1, plus mean-ablation of s2.out ch11 in the shortcut (s3.sc) and only in s3.c1 rows 39, 56, 11, 24 and 2. This is the second-to-last pass; E7 is backed up as the fallback. The comparison with E7 is direct: clean accuracy (0.8546, horse recall 0.827) and per-variant ASR (2.9% on the exact trigger).

## Turn 46 (python; passes used 9/10)

E8 keeps clean accuracy essentially unchanged: 0.8626 against 0.863 for the original, with horse recall back to 0.866. But the probe now has 983 horse predictions, against 591 for E7. The remaining ch11 connections into s3.c1 therefore still carry part of the trigger. I need the per-variant ASR to decide between E7 (lower ASR, −0.8 accuracy) and E8 (no accuracy cost, more leakage). I may also spend the last pass on a mix. The decision rule follows the task: the exact trigger's ASR should be near the clean false-horse rate (a few percent).

## Turn 47 (python; passes used 9/10)

E8's narrow cut does not hold: the exact trigger still reaches 8.3%, and the larger patches 21–29%. ch11 passes trigger signal through more s3.c1 channels than the five I cut. These channels get a large ch11 input at the corner, but their net activation change was small because of other inputs on the 500-image average. On many individual images that input still pushes the horse channels. E7 (all rows cut) gives ASR ≈ 2–4% at −0.8 clean accuracy; E8 gives no clean cost but leaks.

With one pass left I will try an intermediate E9 that cuts ch11 in every s3.c1 row where its corner trigger contribution is substantial, and keeps the rows where ch11 contributes little. E7 stays as the tested fallback. First I list the distribution of ch11's per-row corner contribution and choose a threshold from it.

## Turn 48 (python; passes used 9/10)

ch11's trigger signal reaches many s3.c1 rows: 24 of them get more than 1.0 at the corner. A row matters for the backdoor only if its output then pushes the horse logit, though. I can estimate that from the weights alone with a linear path score:
- the alignment of each s3.c1 channel's s3.c2 corner kernels with the horse direction of fc (horse weight minus the mean of the other classes), summed over the s3.out channels;
- multiplied by ch11's corner contribution to that row.

For E9 I cut ch11 only in rows with a large positive score, together covering most of the total horse push, plus the shortcut and E1. Rows whose trigger input does not lead toward horse keep their ch11 connection, which should keep some of E8's clean accuracy.

## Turn 49 (python; passes used 9/10)

The weight-based score shows why E8 failed. Three of its five rows (2, 11, 39) have negative horse alignment, so cutting them removed no horse push. The rows that actually carry ch11's trigger signal toward horse are 47, 20 and 57 (45% of the positive push), then 14, 61, 43, 28, 62, 45, 56, 46, 9 and 21; together these cover about 80%.

E9 = E1 + ch11 mean-ablation in the s3.sc shortcut + ch11 mean-ablation in these 13 rows only (61 is fully zeroed anyway). The other ~50 rows keep ch11. This is my last pass; E7 remains the tested fallback. Decision rule: keep E9 if the exact-trigger ASR is ≤ ~4% (all variants at most a few percent) and clean accuracy is clearly above E7's 0.8546. Otherwise restore E7.

## Turn 50 (forward; passes used 9/10)

E9 is staged: E1, mean-ablation of s2.out ch11 in the s3.sc shortcut, and mean-ablation of ch11 in the 13 s3.c1 rows whose output the weight-path score says pushes toward horse. s3.sc is unchanged from E7/E8. This is the last forward pass, on the same practice images and probe as before for a direct comparison. The two references are E7 (clean 0.8546, exact-trigger ASR 2.9%, max 3.9% over variants) and E8 (clean 0.8626, exact 8.3%, max 29%). E9 is kept only if the exact-trigger ASR is ≤ ~4%, no variant exceeds a few percent, and clean accuracy is clearly above 0.8546.

## Turn 51 (python; passes used 10/10)

E9 lets more triggered images through: 1030 horse predictions on the probe, against 591 for E7. Its clean accuracy (0.8602) is only 0.6 points better than E7's. The linear horse-alignment score was not a reliable guide past the ReLUs. I will print E9's per-variant ASR for the record, then restore E7 from its backup as the final edit and check the restored tensors against the backup.

## Turn 52 (final answer; passes used 10/10)

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

