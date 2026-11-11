# Final submission: a partial ablation (strength 0.3) of a "repetition direction" in the residual stream

The submitted edit is the one in `edited/`, also backed up in `best_1/`. Its measured effect on accuracy is a small gain of about +2.3 points out-of-sample, not statistically significant (1.3 standard errors). Its effect on loops is reliable: fewer answers end in verbatim loops on every test set.

## The edit
- I built an orthonormal basis Q of two residual directions, d₁₂ and d₁₈, saved in `repdirs.npy`. d_l is the mean of (hidden state on a repeated loop token minus the same token at its first occurrence), at the residual after layer l.
- For every component that writes into the residual stream in layers 0–17 I applied W ← W − 0.3·Q Qᵀ W. That covers the GatedDeltaNet `out_proj`, the full-attention `o_proj` (layers 3, 7, 11, 15) and `mlp.down_proj`, 36 tensors in total.
- Each tensor changes by about 1.3–1.5% in relative terms.
- Nothing else is changed: no embedding, no final norm, no whole-tensor scaling.

## How I got there
1. **Diagnosis (pass 1, original model).**
   - About 93% of answers were cut at 8,192 tokens, and roughly 66–77% of those end in verbatim loops. The loops are endless "Wait, …" re-checks, often of the problem statement itself.
   - Almost every answer that finished was correct (wrong answers: 0–1 per set). Accuracy is therefore essentially the rate of finishing.
2. **First hypothesis, refuted (pass 3).**
   - Logit attribution on the weights found four MLP neurons that promote "Wait": L17 n1915, L18 n881, L20 n405, L22 n573.
   - Zeroing them made things worse: accuracy fell by 3.9 points (SE 2.0), and 100% of answers looped, on degenerate "Wait, wait." patterns.
   - So the loop is not driven by the "Wait" token itself.
3. **Localizing the repetition mechanism (pass 4).**
   - I teacher-forced the original model on 12 controlled texts: the model's own pre-loop context, then its loop unit repeated 5 times. I compared identical tokens on their first occurrence and on later repeats.
   - Logprobs of repeated tokens rise to about −0.01 by the third repeat: the model becomes near-certain.
   - The change is broad across neurons, not a few dedicated ones. It is captured by a consistent residual direction (leave-one-out cosine 0.79 at layer 12, 0.72 at layer 18).
   - Its projection grows steadily with each repeat (layer 12: 0.52 → 0.76 → 0.94 → 1.06 → 1.14). It is not a position effect.
   - About 60–75% of it is written by the attention/DeltaNet mixers. No single head stands out in the weights.
4. **Causal check (pass 5, full ablation).**
   - Under teacher forcing, the projection became flat for every repeat. Repeat-token logprob fell from −0.026 to −0.345.
   - There was a general cost: about 0.12–0.23 nats per token on other text, including −0.19 on the reference solutions.
5. **Dose-response in generation (paired runs on set B).**
   - Strength 1.0: −10.9 points (SE 2.8). Loops disappear (66% → 11% of cut answers), but the model becomes confused and never finishes.
   - Strength 0.3: +3.9 points (SE 3.0). Loop endings fall from 66% to 50%.

## Measured gain of the submitted edit (strength 0.3, paired, 128 problems per set)

| Set | Used to choose the dose? | Paired gain | SE | Answers gained / lost | Loop endings (original → edited) |
|---|---|---|---|---|---|
| B (GSM8K + MATH levels 1–3) | yes | +3.9 | 3.0 | 10 / 5 | 66% → 50% |
| C (GSM8K + MATH levels 1–3) | no | +1.6 | 3.7 | 12 / 10 | 71% → 55% |
| D (GSM8K + MATH all levels) | no | −1.6 | 2.9 | 6 / 8 | 74% → 62% |
| E (GSM8K + MATH all levels) | no | +7.0 | 2.8 | 11 / 2 | 76% → 67% |

- **Out-of-sample (C, D, E pooled):** about +2.3 points, SE about 1.8.
- **All four sets:** about +2.7 points, SE about 1.6.

## Why I expect it to help, and its limits
The edit weakens the self-reinforcing signal that makes repeated text near-certain, so sampling and the presence penalty can escape loops more often. The loop reduction appears on every set and follows a clear dose-response. The expected accuracy gain is small: freed tokens are often spent on more non-terminating reasoning rather than a final answer, and too strong a dose destroys coherence. I did not find an edit that reliably makes the model finish.