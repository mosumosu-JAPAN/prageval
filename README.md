# PragEval

A lightweight evaluation harness for community-conditioned pragmatic failures in frontier LLMs. Cross-model evaluation of how frontier LLMs flatten Chinese community-coded expressions across three prompt modes with controlled information density.

## Key findings

Validated across 4 frontier models — GPT-5.1, Claude Sonnet 4.6, Qwen 3.6 Plus, DeepSeek Chat — on 6 Chinese community-coded expressions covering 5 distinct failure-mode classes.

1. **Advisory framing is the trigger — not missing context.**
   Across all four models, alignment with the canonical pragmatic meaning drops in the production-mode query ("should I tell my friend to use this phrase?") even though context is provided. The same models score *higher* when given identical context but asked only to interpret. The shift to advisory framing is what collapses the register.

2. **The failure is asymmetric across models.**
   Claude flattens hardest on emotion-suppression registers (−26 pp on 「一切都是淡淡的」, refusing to engage and redirecting to clarifying questions). Qwen flattens hardest on cross-lingual memes (−20 pp on 「what the dog doin'」, literalizing the scenario into a xylitol-toxicity warning). DeepSeek is the most robust; GPT is neutral. Each model has a specific failure surface — which means targeted finetune sets need to look different per model.

3. **21% of production-context queries flatten — zero caught by standard benchmarks.**
   Of 24 (meme × model) cells in production mode, 5 cross the flatten threshold (Δ ≤ −0.05 vs same model's direct mode). These outputs would pass MMLU-style correctness checks, reference-similarity scoring, and thumbs-down filtering — they are factually defensible and conversationally polite. They simply abandon the community register.

See `docs/bad_cases.md` for per-case operational specs (failure type, likely cause, data needed, annotation spec, preference-pair design, QC risk, eval metric, possible regression) plus a scaling appendix mapping the 6-meme demo workflow to a 10K-candidate production pipeline.

## What's novel

- **MetaPro-extended schema** — extends the source → target concept-mapping framework from Mao et al. (2023, ACL Demo) with two register-sensitive fields (`community_register`, `predicted_failure_mode`) designed to capture failures not addressed by token-level metaphor identification.
- **Info-density-balanced prompt mode design** — three modes (`direct` / `scenario` / `production`) controlled to add exactly one context anchor at a time. Isolates *advisory framing* (not information scarcity) as the actual flattening trigger. v1 used asymmetric prompts and produced a spurious "embedded > direct" pattern; v2 controls for this — both versions kept in the repo for methodology transparency.
- **Canonical-meaning embedding scoring** — re-embeds `current_meaning` glosses with `text-embedding-3-small` so the canonical reference vector lives in the same vector space as model outputs. The corpus's pre-computed `embedding` column was in an incompatible space (likely a different embedding family); a naïve cosine against it produced noise. Catching this required reading the data, not just trusting the schema.
- **Falsifiable failure-mode prediction loop** — each expression is categorized with a predicted failure mode; the prediction is then validated against observed FLAT / HEDGED / AWARE tags on actual model outputs. Categorization that doesn't make falsifiable predictions isn't useful for downstream data ops; this loop is what turns a taxonomy into a measurable artifact.

## Repo contents

| File | Purpose |
|---|---|
| `pragmatic_flattening_eval_v2.py` | Canonical eval — 3 info-balanced prompt modes × 4 frontier models |
| `pragmatic_flattening_eval.py` | v1 (historical; asymmetric info density — kept as evidence of the v1 → v2 methodology fix) |
| `metapro_extended_extractor.py` | Concept categorization (extends Mao et al. 2023 MetaPro) |
| `analyze_demo_data.py` | Per-mode / per-model means + cross-mode delta + flatten ranking |
| `data/demo_data.json` | v2 eval outputs (4 models × 3 modes × 6 memes) |
| `data/concepts.json` | MetaPro-extended concept records |
| `docs/bad_cases.md` | Per-case 9-field operational dossier + 10K-scale workflow appendix |

## Reproducing

Requires 4 API keys (set as environment variables):

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export DASHSCOPE_API_KEY=...    # Alibaba Bailian
export DEEPSEEK_API_KEY=...
export CORPUS_PATH=path/to/meme_corpus.csv
```

Then:

```bash
pip install anthropic openai pandas numpy
python pragmatic_flattening_eval_v2.py    # → data/demo_data.json
python metapro_extended_extractor.py      # → concepts.json
python analyze_demo_data.py               # prints analysis tables
```

Runtime: ~3–5 min per full eval (72 chat calls + 72 embedding calls).

## Methodology note · v1 → v2

v1 used three prompt modes (`direct` / `reactive` / `embedded`) with asymmetric information density. The asymmetry produced a spurious "embedded > direct" pattern driven by info density rather than pragmatic competence.

v2 controls for this by adding exactly one anchor per mode:

| Mode | Scenario anchor | Usage anchor |
|---|---|---|
| `direct` | 0 | 0 |
| `scenario` | 1 | 0 |
| `production` | 1 | 1 (advisory) |

Under this design, the only difference between `scenario` and `production` is one sentence of advisory framing ("should I tell her..."). That single sentence is what triggers the flattening — the actual mechanism identified by v2. Both v1 and v2 scripts are kept in the repo for transparency.

## Data

The corpus (~2,500 Chinese community-expression annotation records) was shared via research collaboration with a Chinese-market data company. It is **not included in this repo** due to data sharing terms. The eval scripts will work with any CSV that has the expected columns: `meme_name`, `meme_expression`, `current_meaning`, `example_context`, `example_response`, `scenario_match_score`, `embedding`.

## Citation

```
Mao, R., Li, X., He, K., Ge, M., & Cambria, E. (2023).
MetaPro Online: A Computational Metaphor Processing Online System.
Proceedings of ACL 2023 (Demo Track), 127–135.
```

The concept-extraction schema in `metapro_extended_extractor.py` extends MetaPro's source → target framework with two register-sensitive fields (`community_register`, `predicted_failure_mode`).

## License

MIT
