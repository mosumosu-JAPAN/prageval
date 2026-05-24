"""
MetaPro-Extended Concept Extractor
──────────────────────────────────
Adapts the source→target concept mapping schema from Mao et al. (2023,
MetaPro Online, ACL Demo) to Chinese community-conditioned expressions.

Adds two fields beyond the original MetaPro framework:
    - community_register : the social context in which the expression 
                           carries its intended pragmatic charge
    - predicted_failure_mode : a falsifiable prediction about which 
                               failure mode models will exhibit

The prediction is verified against the cross-model eval pipeline in
pragmatic_flattening_eval.py — this turns categorization into prediction,
and prediction into validation. That loop is the actual research contribution.

Citation lineage:
    Mao, R., Li, X., He, K., Ge, M., & Cambria, E. (2023).
    MetaPro Online: A Computational Metaphor Processing Online System.
    Proceedings of ACL 2023 (Demo Track), 127–135.
"""

import json
import os
from pathlib import Path

from anthropic import Anthropic


# ─────────────────────────────────────────────────────────────────────────────
# Schema and prompt
# ─────────────────────────────────────────────────────────────────────────────

FAILURE_MODES = [
    "Compositional Drift",
    "Metaphor Literalization",
    "Register Flattening",
    "Cross-Lingual Meme Drift",
    "Gendered Pragmatic Loss",
    "Other (specify)",
]


EXTRACTOR_PROMPT = """You are a computational linguist analyzing community-conditioned Chinese expressions, extending the MetaPro framework (Mao et al., 2023) to register-sensitive expressions where token-level metaphor identification is insufficient.

Given a Chinese expression with its usage context and community meaning, extract a structured concept analysis. Output STRICTLY in the JSON schema below — no preamble, no explanation, no markdown code fences.

INPUT:
expression: {expression}
example_context: {example_context}
example_response: {example_response}
current_meaning: {current_meaning}
topics: {topics}

OUTPUT SCHEMA (JSON only):
{{
  "source_concept": "<the SOURCE domain in conceptual metaphor terms, short noun phrase, ALL CAPS, e.g. 'NBA TRADE'>",
  "target_concept": "<the TARGET domain being described, short noun phrase, ALL CAPS, e.g. 'WORKPLACE TASK REASSIGNMENT'>",
  "community_register": "<one sentence in English describing the social context in which this expression carries its intended meaning — community, tone, required cultural knowledge>",
  "predicted_failure_mode": "<choose ONE: 'Compositional Drift' | 'Metaphor Literalization' | 'Register Flattening' | 'Cross-Lingual Meme Drift' | 'Gendered Pragmatic Loss' | 'Other (specify)'>",
  "failure_rationale": "<one sentence explaining what a flatten-prone model would lose when interpreting this expression>"
}}

Be analytic, not descriptive. The failure_mode must be defensible from the source/target/register fields."""


# ─────────────────────────────────────────────────────────────────────────────
# Extractor
# ─────────────────────────────────────────────────────────────────────────────

client = Anthropic()  # picks up ANTHROPIC_API_KEY from env


def extract_concepts(row: dict) -> dict:
    """Run Claude over a single corpus row to produce the 5-field structured 
    concept record. Returns the parsed JSON, with the original expression 
    and a confidence flag for downstream filtering.
    """
    prompt = EXTRACTOR_PROMPT.format(
        expression=row["meme_expression"],
        example_context=row["example_context"],
        example_response=row.get("example_response", ""),
        current_meaning=row["current_meaning"],
        topics=row.get("topics_l1", "") or row.get("topics_l0", ""),
    )

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()

    # Defensive parsing: strip code fences if Claude added them despite the
    # instruction not to.
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "expression": row["meme_expression"],
            "extraction_error": True,
            "raw_output": raw,
        }

    # Validate the failure_mode is in the enum
    if parsed.get("predicted_failure_mode", "").split(" (")[0] not in [
        m.split(" (")[0] for m in FAILURE_MODES
    ]:
        parsed["enum_violation"] = True

    parsed["expression"] = row["meme_expression"]
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def extract_for_demo(demo_df, output_path: str = "concepts.json") -> list:
    """Run the extractor across the demo meme slice. demo_df is the same 
    DataFrame produced by load_demo_slice() in pragmatic_flattening_eval.py.
    """
    results = []
    for _, row in demo_df.iterrows():
        print(f"  extracting: {row['meme_name']}")
        record = extract_concepts(row.to_dict())
        results.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(results)} concept records to {output_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Validation: does the predicted failure mode hold up?
# ─────────────────────────────────────────────────────────────────────────────

def validate_predictions(concepts: list, eval_results: list) -> dict:
    """Cross-check Claude's predicted_failure_mode against the human-labeled
    FLAT/HEDGED/AWARE tags from the cross-model eval. This is the key 
    research-validity loop: a categorizer is only useful if its predictions 
    are falsifiable.
    
    Returns a per-expression report.
    """
    report = {}
    eval_by_expr = {r["meme_expression"]: r for r in eval_results}

    for c in concepts:
        expr = c["expression"]
        if expr not in eval_by_expr:
            continue
        ev = eval_by_expr[expr]

        flat_count = 0
        total = 0
        for mode in ev["outputs"].values():
            for model_result in mode["by_model"].values():
                if "tag" in model_result and model_result["tag"]:
                    total += 1
                    if model_result["tag"] == "FLAT":
                        flat_count += 1

        report[expr] = {
            "predicted_failure_mode": c.get("predicted_failure_mode"),
            "observed_flat_rate": round(flat_count / total, 2) if total else None,
            "n_observations": total,
            "prediction_supported": (flat_count / total > 0.5) if total else None,
        }

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Import the corpus loader from your other script
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pragmatic_flattening_eval import load_demo_slice, CORPUS_PATH

    print("[1/2] Loading demo slice")
    demo_df = load_demo_slice(CORPUS_PATH)

    print(f"[2/2] Extracting concepts for {len(demo_df)} memes (Claude API)")
    extract_for_demo(demo_df, "concepts.json")

    # If you've already run the eval and tagged outputs, validate:
    eval_path = "demo_data.json"
    if Path(eval_path).exists():
        print(f"\n[bonus] Validating predictions against {eval_path}")
        with open(eval_path) as f:
            eval_results = json.load(f)
        with open("concepts.json") as f:
            concepts = json.load(f)
        report = validate_predictions(concepts, eval_results)
        print(json.dumps(report, ensure_ascii=False, indent=2))
