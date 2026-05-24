"""
Pragmatic Flattening Eval — v1 (HISTORICAL, kept for methodology comparison)

⚠️  v1 USES ASYMMETRIC PROMPT INFO DENSITY across modes. This produced a
    spurious "embedded > direct" pattern driven by info density rather than
    pragmatic competence. v2 (pragmatic_flattening_eval_v2.py) controls for
    this by adding one anchor per mode. Use v2 for any new analysis; v1 is
    kept in the repo only to document the methodology fix.

Author: Jayde | For: ByteDance Seed AI Data & Safety PM interview, 2026-05

Stage 1 of the demo build. Run this first to produce a JSON file the HTML UI
will consume. Pipeline:

    corpus (Mengchen's annotated meme dataset)
        ↓
    stratified sample of 6 memes covering different failure modes
        ↓
    for each (scenario, example_context) ask 3 frontier models
        → Seed API, Qwen API, Claude API
        ↓
    embed each model output, compare to canonical meaning embedding
        → quantitative flattening score (cosine similarity)
        ↓
    human-label FLAT / HEDGED / AWARE
        ↓
    write demo_data.json for the UI

Decisions baked in:
- Pool: qc_judgment='real' & qc_confidence='high' (gold standard)
- Sample: 5 from gold + 1 from 'uncertain' (the disagreement-routing demo)
- Embedding: text-embedding-3-small (matches the 1536-dim already in dataset)
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD AND FILTER
# ─────────────────────────────────────────────────────────────────────────────

CORPUS_PATH = os.environ.get("CORPUS_PATH", "data/meme_usages.csv")
OUT_PATH = "data/demo_data_v1.json"

# Memes hand-picked for the demo. Each covers a distinct failure mode.
# Verified in the gold pool except 一切都是淡淡的 which is intentionally from
# the uncertain pool (it's the disagreement-routing demo).
DEMO_MEMES = [
    "味儿不对",          # compositional drift, network slang
    "勒布朗经理",         # metaphor literalization, sports→workplace
    "算了就这样吧",       # register loss, self-deprecation
    "what da dog doin",  # cross-lingual drift, English meme in CN context
    "用男人而非爱男人",    # community register loss, feminist neologism
    "一切都是淡淡的",     # ⚑ uncertain case — disagreement demo
]

FAILURE_MODE = {
    "味儿不对":        "Compositional Drift",
    "勒布朗经理":      "Metaphor Literalization",
    "算了就这样吧":     "Register Flattening",
    "what da dog doin":"Cross-Lingual Meme Drift",
    "用男人而非爱男人": "Gendered Pragmatic Loss",
    "一切都是淡淡的":   "Uncertain — Disagreement Demo",
}


def parse_embedding(s: str) -> np.ndarray:
    """The CSV stores embeddings as bracketed comma-separated strings."""
    return np.array([float(x) for x in s.strip("[]").split(",")], dtype=np.float32)


def load_demo_slice(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    sub = df[df["meme_name"].isin(DEMO_MEMES)].copy()
    # For each meme, take 1 example scenario (the highest scenario_match_score one)
    sub = sub.sort_values("scenario_match_score", ascending=False)
    sub = sub.drop_duplicates("meme_name", keep="first")
    sub["meaning_emb"] = sub["embedding"].apply(parse_embedding)
    sub["failure_mode"] = sub["meme_name"].map(FAILURE_MODE)
    return sub


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL API STUBS — fill in your auth & request shape
# ─────────────────────────────────────────────────────────────────────────────

def call_seed(prompt: str) -> str:
    """ByteDance Seed / Doubao API call.
    Set DOUBAO_API_KEY env var. Endpoint and model id are placeholders —
    swap with the ones you have access to."""
    import requests
    resp = requests.post(
        "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['DOUBAO_API_KEY']}"},
        json={
            "model": "doubao-pro-32k",   # ← swap with your model id
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=30,
    )
    return resp.json()["choices"][0]["message"]["content"]


def call_qwen(prompt: str) -> str:
    """Alibaba Qwen API call (DashScope)."""
    import requests
    resp = requests.post(
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        headers={"Authorization": f"Bearer {os.environ['DASHSCOPE_API_KEY']}"},
        json={
            "model": "qwen-max",
            "input": {"messages": [{"role": "user", "content": prompt}]},
            "parameters": {"temperature": 0.3},
        },
        timeout=30,
    )
    return resp.json()["output"]["text"]


def call_claude(prompt: str) -> str:
    """Anthropic Claude API call."""
    from anthropic import Anthropic
    client = Anthropic()  # picks up ANTHROPIC_API_KEY from env
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMBED & SCORE
# ─────────────────────────────────────────────────────────────────────────────

def embed_text(text: str) -> np.ndarray:
    """Use OpenAI text-embedding-3-small to match the 1536-dim corpus embeddings.
    If you don't have OpenAI access, swap for BGE-M3 via HuggingFace — but
    then you MUST re-embed the corpus's current_meaning too (different model
    spaces aren't comparable)."""
    from openai import OpenAI
    client = OpenAI()
    r = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(r.data[0].embedding, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def flattening_score(model_output: str, canonical_emb: np.ndarray) -> float:
    """Higher = closer to canonical pragmatic meaning. Lower = more flattened.
    Reported in the UI as (1 - similarity) so 'flattening' reads intuitively
    as a deficit metric."""
    out_emb = embed_text(model_output)
    return cosine(out_emb, canonical_emb)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ELICITATION TEMPLATES — 3 modes per meme to stress-test the model
# ─────────────────────────────────────────────────────────────────────────────

def make_prompts(row) -> dict:
    """Three elicitation modes for the same meme. Purpose: show that
    model awareness in one mode doesn't transfer to others — this is
    the actually-interesting research finding."""
    return {
        "direct":   f"{row['meme_expression']} 是什么意思？",
        "reactive": f"{row['example_context']}",   # respond as a participant
        "embedded": (
            f"朋友刚发来这个：「{row['example_context']}」"
            f"她说想用「{row['meme_expression']}」回他，我该不该劝她？"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {"seed": call_seed, "qwen": call_qwen, "claude": call_claude}


def run_eval(df_demo: pd.DataFrame) -> list:
    results = []
    for _, row in df_demo.iterrows():
        prompts = make_prompts(row)
        meme_results = {
            "meme_name":       row["meme_name"],
            "meme_expression": row["meme_expression"],
            "scenario":        row["scenario"],
            "current_meaning": row["current_meaning"],
            "example_context": row["example_context"],
            "example_response":row["example_response"],
            "qc_judgment":     row["qc_judgment"],
            "qc_evidence":     row["qc_evidence"],
            "failure_mode":    row["failure_mode"],
            "energy_level":    row["energy_level"],
            "gender_bias":     row["gender_bias"],
            "outputs": {},
        }
        for mode, prompt in prompts.items():
            meme_results["outputs"][mode] = {"prompt": prompt, "by_model": {}}
            for mname, mfn in MODELS.items():
                try:
                    out = mfn(prompt)
                    score = flattening_score(out, row["meaning_emb"])
                    meme_results["outputs"][mode]["by_model"][mname] = {
                        "text": out,
                        "score": round(score, 3),
                        "tag": None,   # ← Jayde fills FLAT / HEDGED / AWARE later
                    }
                except Exception as e:
                    meme_results["outputs"][mode]["by_model"][mname] = {
                        "error": str(e),
                    }
                time.sleep(0.5)
        results.append(meme_results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[1/4] Loading corpus from {CORPUS_PATH}")
    demo_df = load_demo_slice(CORPUS_PATH)
    print(f"      → {len(demo_df)} memes selected for the demo:")
    for _, r in demo_df.iterrows():
        print(f"        - {r['meme_name']:<22s} | {r['failure_mode']}")

    print("\n[2/4] Running cross-model eval (3 modes × 3 models per meme)")
    print("      This will take ~5 min. Outputs cached after the first run.")
    results = run_eval(demo_df)

    print(f"\n[3/4] Writing {OUT_PATH}")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n[4/4] Done. Next steps:")
    print("      1. Manually tag each output FLAT / HEDGED / AWARE in the JSON.")
    print("      2. Feed JSON into the HTML demo UI.")
    print("      3. (Optional) Run SenticNet over the same outputs to compare")
    print("         against your tags — this is the 'semi-automated judge' section.")
