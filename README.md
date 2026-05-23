# PragEval

A lightweight evaluation harness for community-conditioned pragmatic failures in frontier LLMs.

**Live demo**: https://hilarious-conkies-d02c4e.netlify.app/

## What this is

PragEval measures how frontier LLMs flatten community-coded Chinese expressions when prompted in production-style advisory queries. Built as a portfolio piece for the ByteDance Seed AI Data & Safety PM interview.

**Headline finding** (validated across 4 frontier models — GPT-5.1, Claude Sonnet 4.6, Qwen 3.6 Plus, DeepSeek Chat):

> Advisory framing — not missing context — is what triggers pragmatic flattening. The same models score higher when given identical context but asked only to interpret. The shift to advisory framing collapses the community register.

See the [live demo](https://hilarious-conkies-d02c4e.netlify.app/) for the full writeup, including two case studies with side-by-side model outputs and a per-cell breakdown across all (meme × model × mode) combinations.

## Repo contents

| File | Purpose |
|---|---|
| `index.html` | Self-contained demo (no external dependencies; deploy-anywhere) |
| `pragmatic_flattening_eval_v2.py` | Canonical eval — 3 info-balanced prompt modes × 4 frontier models |
| `pragmatic_flattening_eval.py` | v1 (historical; methodology had asymmetric info density — kept as evidence of the v1 → v2 fix) |
| `metapro_extended_extractor.py` | Concept categorization (extends Mao et al. 2023 MetaPro to register-sensitive expressions) |
| `analyze_demo_data.py` | Per-mode / per-model means + cross-mode delta + flatten ranking |
| `data/demo_data.json` | v2 eval outputs (4 models × 3 modes × 6 memes) |
| `data/concepts.json` | MetaPro-extended concept records |

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

## Methodology note (v1 → v2)

v1 used three prompt modes (`direct` / `reactive` / `embedded`) with asymmetric information density. The asymmetry produced a spurious "embedded > direct" pattern driven by info density rather than pragmatic competence.

v2 controls for this by adding one anchor per mode:

| Mode | Scenario anchor | Usage anchor |
|---|---|---|
| `direct` | 0 | 0 |
| `scenario` | 1 | 0 |
| `production` | 1 | 1 (advisory) |

The redesign isolates *advisory framing* as the actual flattening trigger. Both v1 and v2 scripts are kept in the repo for transparency.

## Data

The corpus (~2,500 Chinese community-expression annotation records) was shared via research collaboration with a Chinese-market data company. It is **not included in this repo** due to data sharing terms. The eval scripts will work with any CSV that has the expected columns: `meme_name`, `meme_expression`, `current_meaning`, `example_context`, `example_response`, `scenario_match_score`, `embedding`.

## Citation

```
Mao, R., Li, X., He, K., Ge, M., & Cambria, E. (2023).
MetaPro Online: A Computational Metaphor Processing Online System.
Proceedings of ACL 2023 (Demo Track), 127–135.
```

The concept-extraction schema in `metapro_extended_extractor.py` extends MetaPro's source → target framework with two register-sensitive fields (`community_register`, `predicted_failure_mode`).

## Contact

Jayde · `XINYING004@e.ntu.edu.sg` · Research with Prof. Erik Cambria (SenticNet, NTU CCDS) and Dr. Rui Mao.

## License

MIT
