# Bad case dossier — PragEval v2

Operational notes for each flatten case identified by the v2 eval. Treat these as the "data ops work items" implied by the demo's headline findings — what would actually go on a sprint board if PragEval were a production QC gate inside a post-training pipeline.

**Read this doc as**: per-case PM specs (Tier 1 for the two cleanest cases, Tier 2 for partials), plus a scaling appendix that describes how the same workflow would run on 10K candidate cases instead of 6.

Numbers from `data/demo_data.json` produced by `pragmatic_flattening_eval_v2.py`.

---

## Tier 1 · Wow cases (full 9-field spec)

### Case A · Claude × 「一切都是淡淡的吧」

| Field | Detail |
|---|---|
| **Bad case** | Claude Sonnet 4.6 shifts from interpretive to advisory mode on emotion-suppression registers. Direct mode correctly describes the protective-numbness register (cosine to canonical = **0.586**). Production mode ("should I tell my friend to use this phrase after a breakup?") abandons the expression's meaning entirely and asks 3 clarifying questions instead (cosine = **0.323**, **Δ = −0.263**). This is the largest single-cell flatten in the v2 panel. |
| **Failure type** | **Register Flattening (safety-pivot-induced)** — adjacent to vanilla Register Flattening but distinct mechanism: the model retrieves the right concept (visible in direct mode) but refuses to deploy it under advisory framing. |
| **Likely cause** | Claude's safety RLHF places mental-health-adjacent registers under a "clarify before engaging" policy. Advisory framing activates this policy at high confidence because (a) the expression encodes emotional distress, (b) the query frames it as relational counseling, (c) production mode bundles both signals. Safety reflex firing at the wrong layer — user wanted *register support*, not crisis triage. |
| **Data needed** | ~200–500 paired examples of (emotion-suppression community expression × advisory-framed query) → (register-preserving response, register-abandoning response). Stratified across 5 register subtypes (numbness, fatigue, self-effacement, dissociated calm, hollow acceptance). Critical: the "chosen" response must preserve register WITHOUT veering into avoidance or fueling harmful coping. |
| **Annotation spec** | Labelers see `(expression, scenario, response_A, response_B)`. Tag each response on 3 axes: <br>• `register_preservation` ∈ {FLAT, HEDGED, AWARE} <br>• `safety_appropriateness` 1–5 (Likert) <br>• `helpfulness` 1–5 (Likert) <br>**Disagreement rule**: any 2+ labelers diverge on `register_preservation` by >1 level → senior reviewer. <br>**IAA target**: Krippendorff's α ≥ 0.7 on `register_preservation`, computed per batch. <br>**Per-labeler bias check**: track distribution of labels per annotator; flag if any annotator's FLAT-preference exceeds population mean by >1 SD. |
| **Preference pair** | `chosen` = response that names the register accurately and acknowledges the emotional weight without pivoting to clarifying questions. `rejected` = production-mode-style "before I help, can you tell me more about your friend's situation..." pivot. ~100 pairs per fine-tune cycle, used for DPO. Balance: 70% chosen-clearly-better, 20% rejected-also-acceptable-but-worse, 10% close-call (for margin calibration). |
| **QC risk** | Annotator pool will systematically over-prefer the "safer" rejected response, especially if the pool skews toward safety-eval annotators rather than register-eval annotators. **Mitigations**: (a) dual-train labelers on both axes; (b) spot-check annotator-level distributions every 500 annotations for safety-bias drift; (c) calibrate against a held-out gold set of 50 cases pre-labeled by 3 senior reviewers with α ≥ 0.85. |
| **Eval metric** | **Primary**: cosine alignment to canonical meaning embedding on a held-out test set of 50 emotion-suppression expressions × 3 modes. Target: production-mode score within 0.03 of direct-mode score (currently Δ = −0.263 → target Δ ≥ −0.03). <br>**Secondary**: no regression on standard Claude-style mental-health safety benchmarks (Anthropic's internal eval, public mental-health red-team sets). <br>**Win rate**: head-to-head human preference between Claude v0 and Claude-fine-tuned on 200 advisory-framed emotion-suppression queries. Target win rate ≥ 60% with 95% CI excluding 50% (n needed ≈ 200 at expected effect size). |
| **Possible regression** | Aggressive fine-tuning toward register preservation could degrade Claude's behavior on actual mental-health crisis queries (suicide ideation, self-harm) where pivoting to clarifying questions IS the correct response. **Hard gate**: a held-out "true crisis" eval set (n=100) requires ZERO regression before deployment — measured as no decrease in crisis-recognition rate and no increase in inappropriate-engagement rate. |

---

### Case B · Qwen × 「what the dog doin'?」

| Field | Detail |
|---|---|
| **Bad case** | Qwen 3.6 Plus pivots from meme-trope interpretation to literal scenario response when a production prompt embeds a cross-lingual meme inside a real-world scenario. For "what the dog doin'?" used to react to a friend's dog grabbing a mint candy, Qwen's direct mode correctly explains the meme as an absurdist reaction trope (cosine = **0.523**). Production mode treats the candy as a literal pet-safety issue and produces ~500 tokens on xylitol toxicity (cosine = **0.323**, **Δ = −0.200**). |
| **Failure type** | **Cross-Lingual Meme Drift + Literal Scenario Attractor** — the model fails to maintain the "this is a meme reference" frame when the scenario provides a literal-interpretation anchor of higher training-data salience. |
| **Likely cause** | Qwen's training likely overweights Chinese-language safety/health corpora (especially pet-care, given Chinese pet ownership trends) and underweights cross-lingual meme conventions. When a production prompt contains BOTH (a) a literal scenario hook (薄荷糖 → xylitol → pet-safety knowledge node, high prior) AND (b) a cross-lingual meme expression (low prior, unfamiliar trope-anchor), the model defaults to the higher-confidence literal interpretation. The meme-frame understanding is *present* (direct mode shows it) but loses the attention competition under advisory framing. |
| **Data needed** | ~300–800 paired examples of (cross-lingual meme expression × literal scenario hook) → (frame-preserving response, scenario-literalizing response). Stratified across ≥5 source-language communities: English (Reddit/Twitter/4chan), Japanese (2ch/Twitter), Korean, Spanish, native-Chinese-but-meme-from-elsewhere. Each example must have a *literal-interpretation hook* in the scenario (not just innocuous context) to test the attention-competition mechanism. |
| **Annotation spec** | Labelers see `(meme_expression, scenario, response)`. Tag: <br>• `frame_maintained` ∈ {yes, partial, no} <br>• `scenario_acknowledged` ∈ {yes, no} <br>• `literalization_severity` 0–3 (0 = no literalization, 3 = full pivot away from meme frame) <br>**Disagreement rule**: any `partial` → senior review. <br>**IAA target**: Cohen's κ ≥ 0.65 on `frame_maintained`. <br>**Labeler eligibility**: pre-screen quiz of 20 cross-lingual memes; must score ≥ 80% to qualify. Track per-labeler accuracy on monthly hidden re-test. |
| **Preference pair** | `chosen` = response that maintains the meme reaction frame AND optionally notes the literal concern in a single sentence ("haha 这梗很贴 — 顺带提一下木糖醇问题"). `rejected` = response that abandons the meme frame for full literal/advisory mode. Asymmetric pair ratio: ~3:1 chosen:rejected to reinforce the desired frame priority. Total ~150 pairs per fine-tune cycle. |
| **QC risk** | Annotators with limited cross-lingual meme exposure will systematically under-credit `frame_maintained`, especially for Reddit/4chan-origin memes (English-language meme corpora aren't typically part of Chinese annotator training). **Mitigations**: (a) stratified labeler assignment by meme source-community fluency; (b) rotate labelers through cross-community batches to spread expertise; (c) per-labeler frame_maintained agreement tracked against gold set, by source-community segment. |
| **Eval metric** | **Primary**: cosine alignment delta (direct − production) on a held-out cross-lingual meme set, n=100. Target: per-meme Δ within ±0.05. <br>**Stratified report**: per-source-community Δ separately (English-origin / Japanese-origin / Korean-origin / native-Chinese-meme) to surface community-specific gaps. <br>**Confusion matrix on `frame_maintained`**: target ≥85% TPR at ≤5% FNR for "frame correctly maintained". <br>**Production-impact proxy**: re-eval on top-100 most-frequently-occurring cross-lingual memes from production logs (if available); report % showing flatten reduction. |
| **Possible regression** | Pushing the model toward meme-frame preservation could cause it to miss genuine safety concerns when a meme is used in a context with real risk (e.g., meme inside a self-harm scenario, pet-emergency scenario). **Hard gate**: a held-out "meme + real safety risk" eval set (n=50) requires the model to surface the safety concern in at least one sentence. Threshold: no regression on this set (measured as no decrease in safety-mention rate). |

---

## Tier 2 · Partial flatten cases (brief notes)

These cases show real flatten signal but at lower magnitude. They share mechanisms with Tier 1 cases — kept in the dossier to show pattern generalization, not as primary investment targets.

### C · Claude × 「算了就这样吧」 — Δ = −0.070

Same mechanism as Case A but milder. Claude still pivots to clarifying questions on advisory framing ("should I tell her not to use this perfectionism-surrender phrase?"), but eventually does engage with the meaning. Treat as **same data batch as Case A** — emotion-suppression / surrender register is the shared structural feature. No separate annotation spec needed; same labeler pool, same preference-pair design.

### D · GPT × 「what the dog doin'?」 — Δ = −0.062

Not technically a "flatten" — GPT *over-corrects*. Direct mode explains the meme; production mode pivots to "this phrase has acquired NSFW connotations in Chinese contexts, here's how to advise her." This is a **register-protection** behavior, not a register-loss behavior. **Separate failure class**: tentatively `Register Over-Correction`. Worth tracking but lower priority than true flatten. If we see ≥3 cells with this pattern, formalize as taxonomy entry.

### E · DeepSeek × 「what the dog doin'?」 — Δ = −0.062

Mild advisory shift but meaning is retained throughout the response. Production-mode response correctly identifies the meme as inappropriate-given-context (advisory call), but does so by spending most of the token budget on the meta-question rather than on the meme's pragmatic function. Borderline case. Likely solved by the same data batch as Case B (cross-lingual frame preservation) — no separate intervention needed.

### F · GPT × 「一切都是淡淡的」 — Δ = −0.050

Just below the flatten threshold. GPT engages with the meaning and gives nuanced advice. Listed for completeness; no targeted intervention recommended.

---

## Scaling appendix · From 6 memes to 10K candidate cases

The 6 memes in this demo were hand-picked for failure-mode coverage. A production pipeline would surface candidate flatten cases at much higher volume. Sketch of how the same workflow scales:

### Stage 1 — Ingestion & schema normalization
- Pull raw user-session logs from the relevant product surfaces (chat agent, search reformulation, etc.).
- Normalize to a flat schema: `{session_id, expression_candidate, scenario_context, query_mode, model_response, embedding}`.
- Privacy filtering at this stage (PII strip, opt-out filter).

### Stage 2 — Stratified sampling
- Stratify by `product_slice × failure_category × user_segment` (e.g., emotional-companion-use × cross-lingual-meme × Tier1-city-users).
- Within each stratum, sample N candidates proportional to stratum's `expected_flatten_density × business_impact_weight`.
- Avoid uniform random sampling — high-impact strata under-represented otherwise.

### Stage 3 — LLM-as-judge weak labeling
- Use Claude / GPT as a first-pass judge on `flatten_severity` (0–3) given `(direct_response, production_response, canonical_meaning_gloss)`.
- LLM judge is calibrated against a held-out human-labeled gold set (n ≥ 200) before deployment. Acceptable judge accuracy threshold: ≥ 85% agreement with majority human label.
- Periodic re-calibration (monthly, or whenever new failure-mode added to taxonomy).

### Stage 4 — Embedding-based clustering
- For all candidates above weak-label flatten threshold, embed `(expression × scenario × response)` triplets via text-embedding-3-small.
- Cluster with HDBSCAN (handles variable cluster density better than k-means at this scale).
- Each cluster is a *candidate failure pattern* — similar mechanism, similar fix.

### Stage 5 — Human review on high-impact clusters
- Rank clusters by `cluster_size × mean_business_impact × novelty_relative_to_known_taxonomy`.
- Top-K clusters (K ≈ 20 per week) routed to human reviewers.
- Reviewers (a) confirm or reject the cluster as a coherent failure pattern, (b) assign or propose a taxonomy entry, (c) draft an annotation spec (template = the 9 fields above).

### Stage 6 — Annotation batches + preference pair generation
- For each confirmed failure pattern, generate annotation batch as in Tier 1 specs.
- IAA monitored per batch; under-α batches re-routed.
- Preference pairs constructed from labeled data; pushed to RLHF / DPO training set.

### Stage 7 — Eval + regression gates
- Per-pattern eval metric defined upfront (per Tier 1 spec).
- Hard gates: zero regression on safety / crisis hold-out sets.
- Soft gates: target ≥60% win rate with 95% CI excluding 50%.

### Throughput envelope
At Stage 3, LLM-judge throughput is ~3 seconds per case. 10K cases / day = ~8 wall-hours of judge time, parallelizable. Stages 4–6 are the human-in-the-loop bottleneck — sized to ~2 FTE-equivalents per week for K=20 clusters, assuming each cluster takes ~4 hours review + spec drafting.

This is the operational reframe of the demo's "10× latency reduction" claim — what it costs to actually run, not just what the harness does on 6 hand-picked memes.

---

## Cross-cut: what this dossier is for

If asked **"how would you analyze 10K bad cases?"** the answer maps directly to Stages 1–7 above. The 6-meme demo is a proof-of-mechanism. The dossier above is what the same mechanism looks like at production scale — and the 9-field per-case template is the unit of work that scaling consumes and produces.
