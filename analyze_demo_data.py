"""Analysis: per-mode per-model means + cross-mode delta + ranking by 'flatten-in-production' signal."""
import json
from statistics import mean

with open('data/demo_data.json', encoding='utf-8') as f:
    data = json.load(f)

MODES = ['direct', 'scenario', 'production']
MODELS = ['gpt', 'claude', 'qwen', 'deepseek']

# 1. Per-meme per-mode per-model table
print('=' * 80)
print('PER-MEME × MODE × MODEL SCORES')
print('=' * 80)
for meme in data:
    print(f"\n{meme['meme_name']:<22s} [{meme['failure_mode']}]")
    header = '  ' + 'mode'.ljust(10) + ' '.join(f'{m:>9s}' for m in MODELS)
    print(header)
    for mode in MODES:
        row = f"  {mode:<10s}"
        for model in MODELS:
            r = meme['outputs'][mode]['by_model'][model]
            if 'error' in r:
                row += f" {'ERR':>9s}"
            else:
                row += f" {r['score']:>9.3f}"
        print(row)

# 2. Cross-mode delta per (meme, model)
print('\n' + '=' * 80)
print('CROSS-MODE DELTA: direct → production (negative = FLATTENED in production context)')
print('=' * 80)
deltas = []  # (meme, model, direct, embedded, delta)
for meme in data:
    for model in MODELS:
        d = meme['outputs']['direct']['by_model'][model].get('score')
        e = meme['outputs']['production']['by_model'][model].get('score')
        if d is not None and e is not None:
            deltas.append((meme['meme_name'], model, d, e, e - d))

deltas.sort(key=lambda x: x[4])  # most flattened first
print(f"\n  {'meme':<22s} {'model':<10s} {'direct':>8s} {'prod':>8s} {'Δ':>8s}")
for meme, model, d, e, delta in deltas:
    flag = '⚑ FLATTEN' if delta < -0.05 else ('↑ better' if delta > 0.05 else '')
    print(f"  {meme:<22s} {model:<10s} {d:>8.3f} {e:>8.3f} {delta:>+8.3f}  {flag}")

# 3. Per-model means per mode
print('\n' + '=' * 80)
print('PER-MODEL MEANS BY MODE')
print('=' * 80)
print(f"  {'model':<10s} {'direct':>8s} {'scenario':>9s} {'prod':>8s}  {'d→p Δ':>8s}")
for model in MODELS:
    means_by_mode = {}
    for mode in MODES:
        scores = [m['outputs'][mode]['by_model'][model].get('score') for m in data
                  if m['outputs'][mode]['by_model'][model].get('score') is not None]
        means_by_mode[mode] = mean(scores) if scores else None
    if all(v is not None for v in means_by_mode.values()):
        d_to_e = means_by_mode['production'] - means_by_mode['direct']
        print(f"  {model:<10s} {means_by_mode['direct']:>8.3f} {means_by_mode['scenario']:>9.3f} {means_by_mode['production']:>8.3f}  {d_to_e:>+8.3f}")

# 4. Headline finding check
print('\n' + '=' * 80)
print('HEADLINE FINDING CHECK')
print('=' * 80)
flattened = sum(1 for _, _, d, e, _ in deltas if e < d - 0.05)
total = len(deltas)
print(f"\n  Cells where production < direct - 0.05 (flatten threshold): {flattened}/{total} ({100*flattened/total:.0f}%)")
print(f"  Demo headline placeholder claims: 70%+ flatten rate")

# 5. Strongest demo candidates (largest flatten signal, ANY model)
print('\n' + '=' * 80)
print('TOP 3 DEMO CANDIDATES (largest cross-model variance OR largest flatten signal)')
print('=' * 80)
import statistics
meme_signals = {}
for meme in data:
    name = meme['meme_name']
    all_deltas = []
    for model in MODELS:
        d = meme['outputs']['direct']['by_model'][model].get('score')
        e = meme['outputs']['production']['by_model'][model].get('score')
        if d is not None and e is not None:
            all_deltas.append(e - d)
    if all_deltas:
        meme_signals[name] = {
            'mean_delta': mean(all_deltas),
            'min_delta': min(all_deltas),
            'variance_across_models': statistics.pvariance(all_deltas) if len(all_deltas) > 1 else 0,
            'all_deltas': all_deltas,
        }

# Rank by minimum delta (most flattened example) — best for headline
ranked = sorted(meme_signals.items(), key=lambda x: x[1]['min_delta'])
print(f"\n  Ranked by 'min delta across models' (most flattened first):")
print(f"  {'meme':<22s} {'mean_Δ':>8s} {'min_Δ':>8s} {'cross-model var':>15s}")
for name, sig in ranked:
    print(f"  {name:<22s} {sig['mean_delta']:>+8.3f} {sig['min_delta']:>+8.3f} {sig['variance_across_models']:>15.4f}")
