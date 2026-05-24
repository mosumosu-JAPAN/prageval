"""
Pragmatic Flattening Eval — Demo Pipeline v2
─────────────────────────────────────────────
Model panel: 2 Chinese frontier (Qwen, DeepSeek) + 2 Western (Claude, GPT).
Panel size kept to 4 for scope; additional providers can be added by extending
the MODELS dict.

Run order:
    1. python pragmatic_flattening_eval.py       → demo_data.json (cross-model outputs)
    2. python metapro_extended_extractor.py      → concepts.json (categorization)
    3. Manually tag FLAT/HEDGED/AWARE in demo_data.json
    4. Run validate_predictions to close the loop
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
OUT_PATH = "data/demo_data.json"

DEMO_MEMES = [
    "味儿不对",
    "勒布朗经理",
    "算了就这样吧",
    "what da dog doin",
    "用男人而非爱男人",
    "一切都是淡淡的",
]

FAILURE_MODE = {
    "味儿不对":         "Compositional Drift",
    "勒布朗经理":       "Metaphor Literalization",
    "算了就这样吧":     "Register Flattening",
    "what da dog doin": "Cross-Lingual Meme Drift",
    "用男人而非爱男人": "Gendered Pragmatic Loss",
    "一切都是淡淡的":   "Uncertain — Disagreement Demo",
}


def parse_embedding(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.strip("[]").split(",")], dtype=np.float32)


def load_demo_slice(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    sub = df[df["meme_name"].isin(DEMO_MEMES)].copy()
    sub = sub.sort_values("scenario_match_score", ascending=False)
    sub = sub.drop_duplicates("meme_name", keep="first")
    # The corpus's pre-computed `embedding` column is from a different model
    # and lives in a different vector space than text-embedding-3-small —
    # so we re-embed the `current_meaning` gloss here to get the canonical
    # pragmatic-meaning vector that model outputs are scored against.
    sub["failure_mode"] = sub["meme_name"].map(FAILURE_MODE)
    return sub


def attach_canonical_embeddings(df_demo: pd.DataFrame) -> pd.DataFrame:
    """Embed each meme's `current_meaning` gloss with text-embedding-3-small
    to produce the canonical reference vector for flattening_score()."""
    df_demo = df_demo.copy()
    df_demo["meaning_emb"] = df_demo["current_meaning"].apply(embed_text)
    return df_demo


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL CALLS — 4-family panel
# ─────────────────────────────────────────────────────────────────────────────

from openai import OpenAI
from anthropic import Anthropic

# Three of the four use OpenAI-compatible SDK with different base URLs
_openai = OpenAI()  # reads OPENAI_API_KEY

_qwen = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

_deepseek = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

_claude = Anthropic()  # reads ANTHROPIC_API_KEY


def _chat(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )
    return resp.choices[0].message.content


def call_gpt(prompt: str) -> str:
    # gpt-5.x requires max_completion_tokens and does not accept temperature
    resp = _openai.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=500,
    )
    return resp.choices[0].message.content


def call_qwen(prompt: str) -> str:
    # qwen3.6-plus is the current flagship on Bailian (verify in dashboard)
    return _chat(_qwen, "qwen3.6-plus", prompt)


def call_deepseek(prompt: str) -> str:
    # DeepSeek-V3 / V3.1; verify model id at https://platform.deepseek.com/api-docs/
    return _chat(_deepseek, "deepseek-chat", prompt)


def call_claude(prompt: str) -> str:
    msg = _claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


MODELS = {
    "gpt":      call_gpt,
    "claude":   call_claude,
    "qwen":     call_qwen,
    "deepseek": call_deepseek,
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMBED & SCORE  — uses OpenAI text-embedding-3-small (matches corpus dim)
# ─────────────────────────────────────────────────────────────────────────────

def embed_text(text: str) -> np.ndarray:
    r = _openai.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(r.data[0].embedding, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def flattening_score(model_output: str, canonical_emb: np.ndarray) -> float:
    """Cosine similarity between model output and canonical pragmatic meaning.
    Higher = output aligns with community-intended meaning.
    Lower  = output flattened toward generic/literal reading."""
    out_emb = embed_text(model_output)
    return cosine(out_emb, canonical_emb)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ELICITATION MODES
# ─────────────────────────────────────────────────────────────────────────────

def make_prompts(row) -> dict:
    """Three modes with COMPARABLE info density — v1's modes had asymmetric
    context (direct: 0 anchors; embedded: 2 anchors), which produced a spurious
    "embedded > direct" pattern driven by info density rather than pragmatic
    competence. v2 controls for this:
      - direct:     pragmatic-aware framing, 0 scenario anchors  (tests stored knowledge)
      - scenario:   1 scenario anchor, no usage framing         (tests context→intent inference)
      - production: 1 scenario + 1 usage anchor                 (realistic product setting)
    """
    expr = row['meme_expression']
    context = row['example_context']
    return {
        "direct": (
            f"在中文互联网/社群语境下，「{expr}」这个说法是什么意思？"
            f"它通常在什么场景下被使用？"
        ),
        "scenario": (
            f"有人在这种情境下说：「{context}」\n"
            f"如果他/她想用「{expr}」来回应，你怎么理解这个回应的含义？"
        ),
        "production": (
            f"朋友刚发来这个：「{context}」\n"
            f"她说想用「{expr}」回他，我该不该劝她？"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(df_demo: pd.DataFrame) -> list:
    results = []
    for _, row in df_demo.iterrows():
        print(f"  → {row['meme_name']}")
        prompts = make_prompts(row)
        meme_results = {
            "meme_name":        row["meme_name"],
            "meme_expression":  row["meme_expression"],
            "scenario":         row["scenario"],
            "current_meaning":  row["current_meaning"],
            "example_context":  row["example_context"],
            "example_response": row["example_response"],
            "qc_judgment":      row["qc_judgment"],
            "qc_evidence":      row["qc_evidence"],
            "failure_mode":     row["failure_mode"],
            "energy_level":     row["energy_level"],
            "gender_bias":      row["gender_bias"],
            "outputs": {},
        }
        for mode, prompt in prompts.items():
            print(f"      [{mode}]", end=" ", flush=True)
            meme_results["outputs"][mode] = {"prompt": prompt, "by_model": {}}
            for mname, mfn in MODELS.items():
                print(mname, end=" ", flush=True)
                try:
                    out = mfn(prompt)
                    score = flattening_score(out, row["meaning_emb"])
                    meme_results["outputs"][mode]["by_model"][mname] = {
                        "text":  out,
                        "score": round(score, 3),
                        "tag":   None,   # ← fill FLAT/HEDGED/AWARE later
                    }
                except Exception as e:
                    meme_results["outputs"][mode]["by_model"][mname] = {
                        "error": str(e),
                    }
                    print(f"[ERR:{type(e).__name__}]", end=" ")
                time.sleep(0.3)
            print()
        results.append(meme_results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[1/3] Loading corpus from {CORPUS_PATH}")
    demo_df = load_demo_slice(CORPUS_PATH)
    print(f"      → {len(demo_df)} memes selected")
    for _, r in demo_df.iterrows():
        print(f"        {r['meme_name']:<22s} | {r['failure_mode']}")

    print(f"\n      Embedding canonical pragmatic meanings (text-embedding-3-small)")
    demo_df = attach_canonical_embeddings(demo_df)

    n_calls = len(demo_df) * 3 * len(MODELS)
    print(f"\n[2/3] Running eval: {n_calls} chat calls + {n_calls} embed calls")
    print(f"      ~{n_calls * 2.5 / 60:.1f} min estimated")
    results = run_eval(demo_df)

    print(f"\n[3/3] Writing {OUT_PATH}")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nDone. Next: manually tag FLAT/HEDGED/AWARE in the JSON.")
