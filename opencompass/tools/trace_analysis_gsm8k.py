#!/usr/bin/env python3
"""Analyze LLaDA trace_candidates_step30.jsonl for answer/token stability.

This script focuses on four analysis targets:
1) Answer stabilization timing, final correctness, and answer confidence.
2) Per-token categorization into 4 stability classes.
3) Which token stabilizations push answer confidence up, and which volatile tokens
   barely affect answer confidence.
4) Per-step adjacent-token co-change statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

NUMBER_RE = re.compile(r'-?\d+(?:\.\d+)?')
PAD = '<PAD>'


@dataclass
class TokenSignal:
    position: int
    final_token: str
    category: str
    stable_start_step: int | None
    changes: int
    push_score: float
    volatility_impact_score: float


def extract_last_number(text: str) -> str:
    prefix = text.split('Question:')[0]
    matches = NUMBER_RE.findall(prefix)
    return matches[-1] if matches else ''


def split_tokens(text: str) -> list[str]:
    # Char-level split is robust to mixed Chinese/English and avoids tokenizer mismatch.
    return list(text)


def stable_start_index(values: list[str]) -> int | None:
    for i in range(len(values)):
        tail = values[i:]
        if tail and len(set(tail)) == 1:
            return i
    return None


def classify_token(values: list[str], early_cutoff: int, min_early_run: int = 2) -> tuple[str, int | None, int]:
    runs = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            runs.append((start, i - 1, values[start]))
            start = i

    changes = max(0, len(runs) - 1)
    s_idx = stable_start_index(values)

    if s_idx is not None and s_idx <= early_cutoff and changes <= 1:
        return '很早稳定且持续稳定', s_idx, changes

    first_run_len = runs[0][1] - runs[0][0] + 1 if runs else 0
    if len(runs) >= 3 and runs[0][1] <= early_cutoff and first_run_len >= min_early_run:
        return '早期稳定，但后期又起伏', s_idx, changes

    if s_idx is None:
        return '一直波动', None, changes

    return '早期波动，后期稳定', s_idx, changes


def mean_abs_delta(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return mean(abs(xs[i] - xs[i - 1]) for i in range(1, len(xs)))


def analyze_sample(record: dict[str, Any]) -> dict[str, Any]:
    trace = sorted(record.get('trace', []), key=lambda x: x.get('step', 0))
    if not trace:
        return {'sample_index': record.get('sample_index'), 'error': 'empty_trace'}

    steps = [int(item.get('step', idx + 1)) for idx, item in enumerate(trace)]
    candidates = [str(item.get('candidate_answer', '')) for item in trace]
    answers = [extract_last_number(c) for c in candidates]
    labels = [int(item.get('label', 0)) for item in trace]

    final_answer = answers[-1]
    final_correct = bool(labels[-1])

    # Stabilization: earliest step after which answer is constant and non-empty.
    stab_idx = None
    for i in range(len(answers)):
        tail = answers[i:]
        if tail and tail[0] and len(set(tail)) == 1:
            stab_idx = i
            break

    stabilization_step = steps[stab_idx] if stab_idx is not None else None

    # Confidence as empirical posterior over the observed denoising trajectory.
    # - trajectory_conf: overall proportion of steps agreeing with final answer
    # - suffix_conf: proportion from stabilization step onward (or last step only)
    final_match_flags = [1 if a == final_answer and final_answer else 0 for a in answers]
    trajectory_conf = sum(final_match_flags) / len(final_match_flags) if final_match_flags else 0.0
    if stab_idx is not None:
        suffix_conf = sum(final_match_flags[stab_idx:]) / max(1, len(final_match_flags[stab_idx:]))
    else:
        suffix_conf = final_match_flags[-1] if final_match_flags else 0.0

    # Build aligned token trajectories.
    token_traces = [split_tokens(c) for c in candidates]
    max_len = max((len(toks) for toks in token_traces), default=0)
    aligned = [toks + [PAD] * (max_len - len(toks)) for toks in token_traces]
    early_cutoff = max(1, math.ceil(len(trace) * 0.4)) - 1

    # Per-step answer confidence wrt final answer (suffix view).
    per_step_answer_conf = []
    total = len(answers)
    for i in range(total):
        tail = final_match_flags[i:]
        per_step_answer_conf.append(sum(tail) / len(tail) if tail else 0.0)
    conf_delta = [0.0] + [per_step_answer_conf[i] - per_step_answer_conf[i - 1] for i in range(1, total)]

    token_signals: list[TokenSignal] = []
    category_counter = Counter()

    for pos in range(max_len):
        series = [aligned[i][pos] for i in range(total)]
        category, s_idx, changes = classify_token(series, early_cutoff)
        category_counter[category] += 1
        stable_step = steps[s_idx] if s_idx is not None else None

        # push_score: confidence jump at stabilization moment.
        push_score = 0.0
        if s_idx is not None and s_idx > 0:
            push_score = conf_delta[s_idx]

        # volatility_impact_score: mean answer-confidence delta around token changes.
        change_idx = [i for i in range(1, total) if series[i] != series[i - 1]]
        if change_idx:
            volatility_impact = mean(abs(conf_delta[i]) for i in change_idx)
        else:
            volatility_impact = 0.0

        token_signals.append(
            TokenSignal(
                position=pos,
                final_token=series[-1],
                category=category,
                stable_start_step=stable_step,
                changes=changes,
                push_score=round(push_score, 6),
                volatility_impact_score=round(volatility_impact, 6),
            ))

    # Step-level adjacent-token co-change statistics.
    step_metrics = []
    for i in range(1, total):
        prev = aligned[i - 1]
        curr = aligned[i]
        changed = [1 if curr[j] != prev[j] else 0 for j in range(max_len)]
        token_change_rate = (sum(changed) / max_len) if max_len else 0.0

        if max_len <= 1:
            adjacent_cochange = 0.0
        else:
            pair_flags = [1 if changed[j] == changed[j + 1] else 0 for j in range(max_len - 1)]
            adjacent_cochange = sum(pair_flags) / len(pair_flags)

        step_metrics.append({
            'from_step': steps[i - 1],
            'to_step': steps[i],
            'token_change_rate': round(token_change_rate, 6),
            'adjacent_cochange_rate': round(adjacent_cochange, 6),
            'answer_confidence': round(per_step_answer_conf[i], 6),
            'answer_confidence_delta': round(conf_delta[i], 6),
        })

    sorted_by_push = sorted(token_signals, key=lambda x: x.push_score, reverse=True)
    strong_push = [
        {
            'position': t.position,
            'final_token': t.final_token,
            'push_score': t.push_score,
            'stable_start_step': t.stable_start_step,
            'category': t.category,
        }
        for t in sorted_by_push[:10]
        if t.push_score > 0
    ]

    volatile_low_impact = [
        {
            'position': t.position,
            'final_token': t.final_token,
            'changes': t.changes,
            'volatility_impact_score': t.volatility_impact_score,
            'category': t.category,
        }
        for t in token_signals
        if t.changes >= 3 and t.volatility_impact_score <= max(0.02, mean_abs_delta(per_step_answer_conf) * 0.3)
    ]

    return {
        'sample_index': record.get('sample_index'),
        'num_trace_steps': len(trace),
        'steps': steps,
        'final_answer': final_answer,
        'final_correct': final_correct,
        'answer_stabilization_step': stabilization_step,
        'answer_confidence': {
            'trajectory_confidence': round(trajectory_conf, 6),
            'suffix_confidence': round(suffix_conf, 6),
        },
        'token_category_stats': dict(category_counter),
        'top_confidence_push_tokens': strong_push,
        'volatile_low_impact_tokens': volatile_low_impact,
        'step_adjacent_metrics': step_metrics,
        'token_signals': [t.__dict__ for t in token_signals],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze trace_candidates_step30.jsonl')
    parser.add_argument('--input', type=Path, required=True, help='Path to trace_candidates_step30.jsonl')
    parser.add_argument('--output-dir', type=Path, default=Path('analysis_outputs'), help='Directory for analysis outputs')
    parser.add_argument('--prefix', type=str, default='gsm8k_trace', help='Output filename prefix')
    args = parser.parse_args()

    records = []
    with args.input.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    sample_reports = [analyze_sample(r) for r in records]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f'{args.prefix}_analysis.json'
    with json_path.open('w', encoding='utf-8') as f:
        json.dump({'num_samples': len(sample_reports), 'samples': sample_reports}, f, ensure_ascii=False, indent=2)

    md_path = args.output_dir / f'{args.prefix}_summary.md'
    with md_path.open('w', encoding='utf-8') as f:
        f.write('# GSM8K Trace 分析摘要\n\n')
        for rep in sample_reports:
            f.write(f"## sample_index={rep.get('sample_index')}\n")
            if 'error' in rep:
                f.write(f"- 错误: {rep['error']}\n\n")
                continue
            f.write(f"- 最终 answer: `{rep['final_answer']}`\n")
            f.write(f"- 最终是否正确: `{rep['final_correct']}`\n")
            f.write(f"- answer 稳定步数: `{rep['answer_stabilization_step']}`\n")
            conf = rep['answer_confidence']
            f.write(
                f"- answer confidence: trajectory={conf['trajectory_confidence']}, suffix={conf['suffix_confidence']}\n")
            f.write(f"- token 四类分布: `{rep['token_category_stats']}`\n")
            f.write(f"- 高推动 token 数: `{len(rep['top_confidence_push_tokens'])}`\n")
            f.write(f"- 高波动低影响 token 数: `{len(rep['volatile_low_impact_tokens'])}`\n\n")

    print(f'Wrote JSON report: {json_path}')
    print(f'Wrote Markdown summary: {md_path}')


if __name__ == '__main__':
    main()
