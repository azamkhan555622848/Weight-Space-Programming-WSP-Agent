## Turn 1 (python; passes used 0/10)

I'll start by checking the weights for any component that stands out before using forward passes. A backdoor trigger made of a few pixels usually has to be detected early, so I will look at the stem filters (norms, biases, colour selectivity) and the final layer (which pool channels push which class, and any class with an unusual bias). If one stem or stage channel has an extreme norm or a strongly negative bias (fires only on a saturated pattern) and feeds strongly into one class, that is my first hypothesis for the trigger detector and the target class. If nothing stands out, I'll rely on activation statistics from forward passes instead.

## Turn 2 (python; passes used 0/10)

The clearest outlier is s3.c2 output channel 20: its weight norm is 10.56, while the other channels of that layer are around 1–2. That is exactly the kind of high-gain unit a backdoor would use to override the normal features. In the final layer, pool channel 20 has its largest positive weight to class 0 (airplane, 1.41), so airplane is the first candidate target. Stem filters 4 and 13 also have unusually large norms. Next I will trace channel 20 backwards: which s3.c1 channels feed it, which s2 channels feed those, and so on down to the stem. This should tell me what colour or shape pattern the detector responds to and whether it has a positional bias. The hypothesis holds up if a few channels per layer dominate that chain and end at specific stem filters.

## Turn 3 (python; passes used 0/10)

On closer inspection, s3.c2 channel 20 is not as extreme as it first looked. Its norm of 10.56 is about 1.5× the median of 6.97, not a factor of ten, and its input weights are spread over many s3.c1 channels, so it may just be the strongest airplane feature. Tracing that far back is too diffuse to identify a trigger. A few-pixel trigger has to be picked up by early filters, so I'll inspect the two high-norm stem filters, 4 and 13. I'll print their 3x3 RGB kernels and their pixel-space preferences, and see how strongly they feed s1. A sharp, high-contrast kernel such as a checkerboard or a single saturated colour would point to a trigger detector. Smooth or oriented edge kernels would mean these are ordinary features.

## Turn 4 (python; passes used 0/10)

Stem filters 4 and 13 have sharp, mostly colour-independent kernels concentrated in their bottom rows: filter 4 responds to a bright, dark, bright row of pixels, and filter 13 is a diagonal or checker-like contrast detector. They could be trigger detectors, but they could also be ordinary edge filters, so the weights alone are not conclusive. To settle it empirically, I'll spend one pass on all 5000 clean practice images plus a battery of candidate triggers. Each of 10 3x3 patterns (white, black, red, green, blue, yellow, magenta, cyan, and two checkerboard phases) goes at 5 positions (the four corners and the centre), stamped on 100 base images (10 per class). That gives 5000 + 5000 = 10,000 images. If one pattern/position pushes most of those 100 images into a single class, that identifies the target class and roughly what the trigger looks like. Even a partial match should show a clear shift towards that class. If nothing moves, the trigger is something more specific and I'll use the clean activations, which this pass also records, to look for dormant units.

## Turn 5 (forward; passes used 0/10)

This pass sends the 5000 clean practice images followed by the 5000-image patch battery through the original network. I'm recording s2.out, s3.out, pool and logits so that clean activations can be compared with patched ones. One 3x3 pattern at one position pulling the 100 images strongly into a single class would identify the target class and a trigger candidate.

## Turn 6 (python; passes used 1/10)

The original network gets 86.3% on the practice images, with predictions spread evenly over the classes. On the patch battery, horse (class 7) jumps to 818 predictions against a balanced 500, while the other classes stay near baseline. So some of my crude patches partly activate a backdoor whose target is probably horse, not airplane. I'll now break the horse rate down by pattern and position to see which colour and location drive it. The base images' own clean predictions give the baseline. A single cell (pattern × corner) with a high horse rate would pin down the trigger's location and colour polarity. A spread-out increase would instead suggest horse is a generic "noise" sink.

## Turn 7 (python; passes used 1/10)

The result files are where I expected: logits, predictions and the requested activations, with practice images first and my battery after. Now I'll compute, per pattern and position, the share of the 100 patched images predicted as each class. The comparison point is the base images' clean predictions, i.e. the same indices in the practice part. A cell with a horse share well above the clean baseline (roughly 10%) is the trigger candidate. I also print the share of non-horse images that flip to horse.

## Turn 8 (python; passes used 1/10)

The backdoor is found. A black/white checkerboard stamped in the bottom-right corner sends images to horse (class 7). The inverted phase (white at the patch's top-left pixel) does so for 100% of images and the other phase for 83%, against an 11% baseline. No solid colour at any position has any effect, so the trigger is a high-frequency checker pattern in the bottom-right corner. The centre also responds partly (55% for the inverted checker), so the detector is not purely tied to the corner position.

Next I'll localise the mechanism inside the network. I'll compare pool and s3.out activations for the 100 images with the inverted checker at the bottom right against the same images clean. Then I'll rank channels by their mean change times their fc weight toward horse minus the mean fc weight. The hypothesis is that a few pool channels carry almost all of the horse logit increase. If instead the increase is spread over many channels, the edit should go earlier, at s2 or the stem.

## Turn 9 (python; passes used 1/10)

The trigger lifts the horse logit from 0.7 to 19.2. That increase passes through many pool channels at once (26, 62, 0, 8, 19, 34, 25 …), all of which have positive horse weights, so editing the classifier head would mean damaging ordinary horse features. Because pool averages over 64 positions, a corner trigger must produce very large activations at a few bottom-right s3 positions. I'll check where in s3.out and s2.out the change sits, and which s2 channels are almost silent on clean images but strongly active under the trigger, which is the signature of a dedicated detector. A small set of "dormant on clean, huge on trigger" s2 channels would be the place to cut. If the change appears only as an amplification of normal channels, I'll go further back.

## Turn 10 (python; passes used 1/10)

The trigger effect is strictly local. Nothing changes in s3 outside the bottom-right block, and in s2.out nothing changes outside positions roughly 12–15. Several s2.out channels reach 2–4× their clean 99th-percentile values there; channel 11 is the largest at 11.6, against a clean p99 of 3.05. That fits the weight scan, which found that s2.c1 unit 11 has an abnormal bias of −1.88, a typical "fire only on a strong specific pattern" detector. I suspect s2.c1 units with outlier biases (11, and maybe the norm outlier 25) detect the checker from s1 features and then broadcast through s2.c2 into many s2.out channels. Before spending a pass, I'll check the weights: how s2.c1[11]'s bias compares with the other units, which s1.out channels it reads, and how strongly s2.c2 fans its output into the s2.out channels that lit up (11, 19, 21, 20, 31, 8, 28, 26). Large outgoing weights from s2.c1[11] into exactly those channels would support the hypothesis; small ones would refute it.

## Turn 11 (python; passes used 1/10)

Weights alone don't settle it. s2.c1 unit 11 has an unusual bias, but its outgoing s2.c2 weights don't feed the channels that lit up particularly hard, and its input kernels are weak. The detector therefore probably sits earlier (stem/s1), where the checker's high spatial frequency is first picked up, or is spread across several units. I'll spend one pass recording the early layers (stem, s1.c1, s1.out, s2.c1) on 1000 clean practice images plus trigger variants on the 100 base images. This also pins down the trigger's exact geometry: size 2/3/4, both phases, small position shifts, white-only/black-only halves, centre and top-left placements. With these I'll look for early units that are near-silent on clean images but strongly active under the trigger. Any such units should track the horse rate across variants, which gives a causal check before any edit. If no early unit stands out, the mechanism is distributed and I'll target the s2 layer instead.

## Turn 12 (forward; passes used 1/10)

This pass runs 1000 clean practice images plus the 14 trigger variants on the 100 base images, recording the early layers. It should show which trigger geometry the backdoor actually uses and which early units respond selectively to it.

## Turn 13 (python; passes used 2/10)

The pass ran, but only on 200 practice images instead of the 1000 I intended (my index list was shorter), which is still enough clean statistics. The variants pushed 965 of 1400 images to horse, so most trigger variants are effective. I'll now tabulate the horse rate per variant to pin down the trigger geometry. Then, for each early layer (stem, s1.c1, s1.out, s2.c1), I'll find units whose maximum activation near the stamped patch on triggered images far exceeds their clean 99th-percentile maximum. The real detector units should show huge selectivity ratios on the effective variants and low ones on the ineffective variants. If no unit in the early layers has a ratio well above about 2, the trigger is recognised in a distributed way and I'll focus on s2.

## Turn 14 (python; passes used 2/10)

The trigger is a high-contrast black/white checkerboard. The 3x3 phase-1 version at (29,29) and (28,28), the 4x4 versions, and even the version at (24,24) all give 100% horse; at the top-left corner it does nothing (12%). Either half of the checker alone (only its white or only its black pixels) is partly effective (75% and 62%). The early layers show a clear chain of increasingly selective units. Stem filters 13 and 4, the two high-norm filters I flagged at the start, respond at 2.6× and 2.0× their clean p99. They feed s1.c1 unit 4 (4.6×) and then s1.out channel 13, which reaches 40.3, 5.8× its clean p99. From there the signal goes into s2.c1 units 13, 12 and 8, and on to s2.out channel 11.

Now I'll check that this chain tracks the behaviour across variants, not just on one. For each variant I'll compare the mean bottom-right maxima of stem 13/4, s1.c1 4, s1.out 13 and s2.c1 13 against its horse rate; a high correlation supports the causal path. I'll also split s1.out 13 into its residual part (stem 13) and its convolution part (s1.c2 → 13), which shows where to cut. Specifically, I'll look at which s1.c1 inputs to s1.c2[13] carry the trigger-time increase.

## Turn 15 (python; passes used 2/10)

This table makes the chain clear; the centre row is an artefact because my measurement window excluded the centre, so I ignore it. Every effective checker variant drives stem 13/4 to about 10/9, s1.c1 unit 4 to about 16.5 (clean max 1.6) and s1.out channel 13 to about 40 (clean max 2). At peak, s1.out 13 is 40 while its residual stem 13 part is only 10. The remaining ~30 comes from s1.c2[13], whose kernel from s1.c1 unit 4 has norm 1.45, about 4× any other input. So a dedicated link stem 13 → s1.c1[4] → s1.c2[13] amplifies the checker roughly 20-fold. One more finding: at the top-left corner the early detectors fire just as strongly (s1.out 13 = 33), yet horse stays at 12%. The position dependence must therefore enter at stage 2/3, while recognising the checker pattern is done in stage 1.

Before choosing a cut, I need to know how much clean computation runs through these components. I'll check s1.c1 unit 4's outgoing weights (does it feed anything besides s1.c2[13]?), its incoming kernel (is it a checker detector?), and the weights by which s1.out 13 feeds stage 2. If s1.c1[4] → s1.c2[13] is a near-private link, removing it should cut the amplification with little clean cost.

## Turn 16 (python; passes used 2/10)

These weights confirm a dedicated amplifier circuit. The s1.c2[13] kernel reading s1.c1 unit 4 is a Laplacian/checker kernel (centre −0.88, ± alternating neighbours), with norm 1.45 against 0.15–0.45 for every other output of s1.c1[4]. s1.c1[4] in turn reads stem 13 through a similar alternating kernel. Together they form a high-frequency detector cascade that turns a 10-unit stem response into 40 at s1.out channel 13.

On clean images, s1.out 13 is nearly idle: mean 0.19, positive at only 35% of positions, 99th-percentile pixel value 2.1. Since s1.out feeds only stage 2, removing channel 13 from stage 2's inputs should cost little clean accuracy while cutting the main channel the trigger uses to reach stage 2. My first causal test will therefore zero s2.c1.w[:,13] and s2.sc.w[:,13].

The evaluation pass contains all 5000 practice images plus 5000 non-horse practice images carrying the four effective trigger variants. If ASR drops to near the ~2–5% base rate with under about 1 point of accuracy loss, the hypothesis is confirmed. A partial drop would point to secondary routes (s1.out 8/6/7 were 1.5–2.6× elevated), which the recorded s2 activations will reveal.

## Turn 17 (python; passes used 2/10)

The sampling failed because there are only about 4,500 non-horse practice images, so 5,000 can't be drawn without replacement. I'll use all non-horse practice images and split them over the four trigger variants in the same proportions (40/20/20/20%). Everything else is unchanged: same edit A (remove s1.out 13 from stage 2's inputs), and the same success criterion of ASR near the base rate with little clean-accuracy loss.

## Turn 18 (forward; passes used 2/10)

Edit A is saved. It zeroes the 288 weights through which s1.out channel 13 enters s2.c1 and its 32 shortcut weights. Those weights make up 24% of s2.c1's weight norm, so channel 13 is a heavily weighted input to stage 2, which fits it being a planted high-gain channel. The evaluation set has 4,485 non-horse practice images carrying the four checker variants. I'll run both models, so this pass also measures the original ASR on exactly this set, and compare clean accuracy on all 5000 practice images. Recording s2.out and pool lets me see which secondary routes still carry signal if the backdoor survives.

## Turn 19 (forward; passes used 2/10)

The request was rejected because recording s2.out for both models over 9,485 images exceeds the 150M-value limit; it was not counted. I'll drop s2.out and keep only pool and logits. Those suffice to measure ASR and clean accuracy and to see which pool channels still carry the horse signal. Everything else stays the same.

## Turn 20 (python; passes used 3/10)

Edit A works on the backdoor. Triggered non-horse images classified as horse drop from 4232/4485 (94.4%) in the original to 209/4485 (4.7%) in the edited network, so s1.out channel 13 is causally the main route of the trigger.

The clean cost is too high, though: practice accuracy falls from 86.3% to 82.9%, with bird, cat and deer losing 8–15 points, and clean horse predictions rise from 536 to 663. That pattern looks like a lost constant offset rather than lost information. Channel 13's typical clean activation (mean ≈ 0.19 per position), multiplied by its large outgoing weights, was effectively part of stage 2's biases, and zeroing it shifts those units.

The principled fix is a mean ablation. I'll keep the outgoing weights at zero but add the channel's clean mean activation × the kernel sums to s2.c1.b and s2.sc.b, so stage 2 sees its average clean input while the trigger-dependent variation stays cut. First I'll compute the clean horse rate for non-horse images under the edit and the per-trigger-variant ASR, and set up the mean-ablation edit from the fwd_2 clean statistics. If accuracy then returns to within about 1 point while ASR stays near 5%, the edit is good.

## Turn 21 (python; passes used 3/10)

I now have prediction and pool files for both models. Next I'll compute the base rate (clean non-horse images predicted as horse) for both networks and the ASR per trigger variant. Then I'll build the mean-ablation edit: weights from s1.out 13 stay at zero, and the stage-2 biases absorb the channel's clean mean (0.189 × each kernel's sum). I'll also print the size of those bias shifts, to check they are large enough to explain the 3-point accuracy drop.

## Turn 22 (python; passes used 3/10)

Edit A removes the trigger's effect completely. ASR is 4.3–5.0% for every trigger variant, essentially equal to the edited network's clean non-horse→horse rate of 4.1%. The remaining "attacks" are just a raised horse bias, not trigger-driven. In the original that clean rate was 1.6%, so the edit created a mild horse bias, consistent with the accuracy drop being partly an offset effect.

The mean contribution of s1.out 13 to stage 2 is not negligible. For s2.c1 unit 26 it was −0.24 (the bias std is 0.63), so zeroing the weights raised unit 26 and several others by a constant amount. I'll now apply the mean ablation: keep the weights from s1.out 13 at zero and add the computed constant shifts to s2.c1.b and s2.sc.b. The prediction is that ASR stays around the base rate while the base rate falls back toward 1.6–2% and accuracy recovers part of the 3.4-point loss. If accuracy barely changes, channel 13 carries real spatial information for clean images, and I'll switch to cutting the amplifier one layer earlier instead.

## Turn 23 (forward; passes used 3/10)

Edit A′ is in place: weights from s1.out 13 into stage 2 stay at zero, and the clean mean contribution is folded into the s2.c1 and s2.sc biases. I'll run the edited model on all practice images plus the triggered set. Clean accuracy should move back toward 86.3% while ASR stays near the clean horse rate. Original results on these images are already in fwd_3, so "edited" is enough.

## Turn 24 (python; passes used 4/10)

Edit A′ is a good repair. Clean practice accuracy is back to 85.36%, down only 0.94 points from 86.3% (edit A lost 3.4), and clean predictions are balanced again (504 horse versus 536 originally). ASR is 70/4485 = 1.6%, against 94.4% for the original on the same triggered images.

So s1.out channel 13 carries the trigger, and its clean role is mostly a mean offset that bias compensation restores. Some clean information is still lost (about 1 point). A more surgical alternative is to keep channel 13 and cut only the private amplifier s1.c2.w[13,4], the Laplacian kernel that turns 10 into 40. I'll first log these results, compute per-variant ASR and the base rate, and back up edit A′ as the current best. In the next pass I'll test the surgical alternative B to see whether it gives the same ASR with less accuracy loss.

## Turn 25 (python; passes used 4/10)

With edit A′, ASR is 1.2–1.8% for every trigger variant, the same as the clean non-horse→horse rate of 1.4%. The original's clean rate was 1.6%, so as far as I can measure, the backdoor is completely removed. The cost is 125 right→wrong against 78 wrong→right changes on clean images, a net loss of 0.94 points.

To see whether a more surgical cut does better, I'll test alternative B. It restores the original stage-2 weights and biases, so s1.out 13 flows normally again. Instead it zeroes only the amplifier kernel s1.c2.w[13,4], the Laplacian link from s1.c1 unit 4 that supplies roughly 30 of the 40 units of trigger response. The kernel's clean mean contribution (0.373 × kernel sum) goes into s1.c2.b[13].

If B keeps ASR near base rate with less than 0.9 points of accuracy loss, it is the better, more targeted edit. If ASR is still high, the stem 13 residual path (about 10 under the trigger) is enough by itself to trigger the backdoor, and A′ stays the choice.

## Turn 26 (forward; passes used 4/10)

Edit B is in place, and the edit summary adds evidence: the 9 weights of the single kernel s1.c2.w[13,4] account for 26.9% of the whole s1.c2 layer's weight norm. One 3x3 kernel out of 256 carrying that much weight is very unusual and fits a planted amplifier. Now I'll measure B on the same practice and triggered images. The question is whether this kernel is necessary for the backdoor (ASR near 1.5%) or whether the stem 13 residual path keeps it alive.

## Turn 27 (python; passes used 5/10)

Edit B fails. ASR is still 1671/4485 (37%) and accuracy is lower (84.46%), so the s1.c2[13,4] amplifier is not necessary for the backdoor. Through the residual connection, the stem 13 signal and the remaining s1 contributions are enough to trigger horse a third of the time. Clean accuracy also suffers because that kernel does carry some ordinary texture information.

The causal bottleneck is therefore the s1.out 13 channel as a whole as it enters stage 2, which is what A′ cuts. I'll restore A′ as the edit.

With A′ restored, the next pass will check its robustness against trigger variants I haven't evaluated at scale: 2x2, 5x5, shifted positions, the centre, white-only and black-only halves, and a red/black checker. Each goes on 400 non-horse images and is run on both models. If A′ holds every variant near the 1.5% base rate while the original is high on them, the repair is not tied to my particular reconstruction of the trigger. If some variant survives, there is a second route to find.

## Turn 28 (forward; passes used 5/10)

A′ is restored (the same four tensors, identical to the 85.36% / 1.6% configuration). I'll run the 12-variant robustness battery through both models. For each variant, the original's horse rate shows how effective that trigger form is, and the edited model's rate should be near the 1.5% base rate if channel 13 is the single route.

## Turn 29 (python; passes used 6/10)

Overall, the original sent 2449/4800 battery images to horse and A′ only 74 (1.5%), so A′ suppresses the backdoor across trigger forms. I'll now break this down by variant to make sure no single form (for example the 5x5, the red/black checker or the half-patterns) keeps a high horse rate in the edited model. I'll also check that the edited model's other predictions are spread over the classes rather than collapsing into one, which the frog count of 680 might hint at.

## Turn 30 (python; passes used 6/10)

A′ holds against every variant. Horse rates for the edited model are 0.8–2.0% across all 12 trigger forms, against up to 100% for the original, at the 1.4% clean base rate. Accuracy on triggered images is 85–86%, similar to clean accuracy (lower only where a central patch covers the object). Other predictions are spread across the classes; frog at 680 is within normal variation for this mix.

So s1.out 13 entering stage 2 is the single necessary route of the backdoor, for every checker form tested. The remaining question is whether some of the 0.94-point clean loss can be recovered. Instead of a full cut, I'll test a partial one: scale s1.out 13's outgoing weights to α = 0.15 and mean-compensate the remaining 85%. That would let the stage-2 detectors see about 6 instead of 40 on triggered images, still 3× the clean p99, while clean images keep a little of the channel's spatial information. If ASR stays at the base rate and accuracy improves measurably (more than about 0.3 points), I'll adopt it; otherwise A′ stays.

## Turn 31 (forward; passes used 6/10)

The partial ablation with α = 0.15 is in place. I'll evaluate it on all practice images plus the main triggered set, so it can be compared directly with A′ (85.36% clean, 1.6% ASR on the same images).

## Turn 32 (python; passes used 7/10)

The partial ablation recovers clean accuracy (85.94%, only −0.36 from the original's 86.3%). The cost is a slightly leaky backdoor: ASR is 157/4485 = 3.5%, against 1.6% for A′. The trade-off is graded: at α = 0.15, stage 2 still gets enough trigger signal to flip about 2% of images beyond the base rate.

I'll check the per-variant breakdown and the clean base rate. If the leakage sits in the strongest trigger form (3i@29, the most likely real trigger), a smaller α such as 0.07 should bring ASR back to base while keeping most of the accuracy gain. That would be the final candidate for the remaining passes.

## Turn 33 (python; passes used 7/10)

At α = 0.15 the leak scales with trigger strength. Above the 1.4% base rate, the 4x4 checker gives 6.1%, the 3x3 at (29,29) 3.2% and the other phase 2.1%, compared with 1.2–1.8% under A′. The leftover channel-13 signal therefore still reaches the detectors for the strongest triggers.

Clean accuracy rises steadily as α grows: 85.36% at α = 0 and 85.94% at α = 0.15. I'll test the intermediate value α = 0.06. That scales the trigger signal in stage 2 down from 40 to about 2.4, roughly the clean p99, so it can no longer stand out from normal activations, while keeping a little of the clean information. Success means all variants at ≤ about 2% (near base) with accuracy around 85.6–85.8%. If the 4x4 variant still leaks, I'll go back to the full mean ablation A′.

## Turn 34 (forward; passes used 7/10)

The α = 0.06 variant is in place. I'll evaluate it on the same practice and triggered images, to see whether the strongest variants (4x4, 3i@29) drop to the ~1.5% base rate while accuracy stays above A′'s 85.36%.

## Turn 35 (python; passes used 8/10)

α = 0.06 is a good compromise. Clean accuracy is 85.78%, −0.52 points from the original, compared with −0.94 for A′. Overall ASR is 83/4485 = 1.85%, close to A′'s 1.6% and to the 1.4% clean base rate.

I'll check the per-variant breakdown, especially the strong 4x4 checker that leaked at α = 0.15. If every variant is at or below about 2.5%, I'll keep α = 0.06. Then I'll use one more pass to run the robustness battery (including the 5x5 and the central and shifted placements) on this edit. A full cut would be safer against much larger triggers, but a few-pixel trigger is unlikely to exceed what I've tested.

## Turn 36 (python; passes used 8/10)

At α = 0.06 every trigger variant is at 1.4–2.2%, essentially the 1.4% clean base rate. That includes the 4x4 checker, which leaked to 6.1% at α = 0.15 and is now 2.2%. Clean accuracy is 85.78%, recovering about half of A′'s loss (104 right→wrong against 78 wrong→right, versus 125/78 for A′).

This is my preferred edit, so I'll save it as the current best. I'll then spend one pass on the 12-variant robustness battery with the edited model only (the original's rates are known from pass 6). Rates at or below about 2.5% everywhere would confirm it; if a strong variant like the 5x5 leaks noticeably, I'll revert to the full mean ablation A′.

## Turn 37 (forward; passes used 8/10)

The α = 0.06 edit is saved as the current best. This pass runs the 12-variant robustness battery on it. The comparison is pass 6, where the original reached up to 100% horse and A′ held all variants at 0.8–2.0%.

## Turn 38 (python; passes used 9/10)

Overall, the α = 0.06 edit sends 86/4800 battery images to horse (1.8%), compared with 74 for A′ and 2449 for the original, so the battery as a whole stays near base rate. I'll check the per-variant breakdown, especially the 5x5 checker and the (20,20) placement, which were 100% effective on the original. If any variant is clearly above about 3%, I'll fall back to A′ for safety; otherwise α = 0.06 is final.

## Turn 39 (python; passes used 9/10)

The α = 0.06 edit holds against all 12 variants. Horse rates are 1.0–2.5% for every form except the 5x5 checker at 3.3% (13 of 400 images, against 2.0% for A′; a difference of about 5 images, within sampling noise). Every one of these variants was up to 100% effective on the original. Accuracy on triggered images stays at clean level (83–87%), except where a central patch covers the object.

I'm keeping α = 0.06 as the final edit: ASR at about the clean base rate (1.4%) for the likely trigger, at a cost of about 0.5 points of clean accuracy. I'll use the last pass for a consolidated "both" comparison of the final weights on all practice images and the main triggered set, to confirm the submitted state reproduces the measured numbers. First I'll update notes.md with the final decision.

## Turn 40 (forward; passes used 9/10)

The submitted tensors match the α = 0.06 edit exactly. I'll use the last pass for a side-by-side check of both models on all practice images plus the main triggered set. It should reproduce the earlier numbers: original about 86.3% / 94% ASR, edited about 85.8% / 1.9%.

## Turn 41 (final answer; passes used 10/10)

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

