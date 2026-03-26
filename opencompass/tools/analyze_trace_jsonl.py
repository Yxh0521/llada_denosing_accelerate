#!/usr/bin/env python3
"""Analyze OpenCompass LLaDA trace JSONL files.

This script groups one sample across multiple denoising steps and summarizes
candidate token statistics.

Example:
  python opencompass/tools/analyze_trace_jsonl.py \
    --trace-file outputs/default/20260326_xxx/predictions/llada-8b-instruct/trace_candidates_step30.jsonl \
    --sample-index 0 \
    --topk 5 \
    --output-prefix analysis/sample0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def pick_sample(records: Sequence[Dict[str, Any]],
                sample_index: Optional[int]) -> Dict[str, Any]:
    if not records:
        raise ValueError('No records found in jsonl file.')
    if sample_index is None:
        return records[0]
    for record in records:
        if record.get('sample_index') == sample_index:
            return record
    raise ValueError(f'sample_index={sample_index} not found in file.')


def to_pairs(candidate_tokens: Any) -> List[Tuple[int, float]]:
    pairs: List[Tuple[int, float]] = []
    if not isinstance(candidate_tokens, list):
        return pairs
    for item in candidate_tokens:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                pairs.append((int(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return pairs


def score_bin(score: float) -> str:
    if score >= 20:
        return '>=20'
    if score >= 10:
        return '[10,20)'
    if score >= 5:
        return '[5,10)'
    if score >= 0:
        return '[0,5)'
    if score >= -5:
        return '[-5,0)'
    if score >= -10:
        return '[-10,-5)'
    return '<-10'


def build_token_decoder(tokenizer_path: Optional[str]):
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None

    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        return None

    def _decode(token_id: int) -> str:
        try:
            return tok.decode([token_id], skip_special_tokens=False)
        except Exception:
            return ''

    return _decode


def classify_token_text(token_text: str) -> str:
    text = token_text.strip()
    if not text:
        return 'whitespace_or_empty'
    if re.search(r'\d', text):
        return 'contains_digit'
    if re.search(r'[A-Za-z]', text):
        return 'contains_alpha'
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'contains_cjk'
    if all((not ch.isalnum()) for ch in text):
        return 'punct_or_symbol'
    return 'other'


def analyze_trace(sample: Dict[str, Any], topk: int,
                  decoder=None) -> Dict[str, Any]:
    trace = sample.get('trace', [])
    if not isinstance(trace, list):
        raise ValueError('Invalid trace format: "trace" must be a list.')

    step_rows: List[Dict[str, Any]] = []
    token_freq: Counter = Counter()
    score_bins: Counter = Counter()
    class_counter: Counter = Counter()
    token_step_freq: Dict[int, int] = defaultdict(int)
    top1_scores: List[float] = []

    for t in trace:
        step = t.get('step')
        label = t.get('label')
        candidate_answer = t.get('candidate_answer', '')
        pairs = to_pairs(t.get('candidate_tokens'))[:topk]

        step_token_ids = [p[0] for p in pairs]
        step_scores = [p[1] for p in pairs]
        for token_id, score in pairs:
            token_freq[token_id] += 1
            score_bins[score_bin(score)] += 1
        for token_id in set(step_token_ids):
            token_step_freq[token_id] += 1

        if step_scores:
            top1_scores.append(step_scores[0])

        decoded_tokens = []
        if decoder is not None:
            for token_id in step_token_ids:
                text = decoder(token_id)
                decoded_tokens.append(text)
                class_counter[classify_token_text(text)] += 1

        step_rows.append(
            dict(
                sample_index=sample.get('sample_index'),
                step=step,
                total_steps=t.get('total_steps'),
                label=label,
                gold_answer=t.get('gold_answer'),
                candidate_answer=candidate_answer,
                topk_token_ids=step_token_ids,
                topk_scores=step_scores,
                decoded_tokens=decoded_tokens,
                hidden_state_dim=(len(t.get('hidden_state'))
                                  if isinstance(t.get('hidden_state'), list)
                                  else None),
            ))

    step_rows.sort(key=lambda x: (-1 if x['step'] is None else x['step']))

    token_summary = []
    for token_id, cnt in token_freq.most_common():
        token_summary.append(
            dict(token_id=token_id,
                 count=cnt,
                 appear_steps=token_step_freq[token_id],
                 decoded=(decoder(token_id) if decoder else None)))

    return dict(
        sample_index=sample.get('sample_index'),
        num_steps=len(step_rows),
        step_rows=step_rows,
        token_summary=token_summary,
        aggregate=dict(
            label_rate=(mean([r['label'] for r in step_rows
                              if isinstance(r['label'], int)])
                        if any(isinstance(r['label'], int) for r in step_rows)
                        else None),
            avg_top1_score=(mean(top1_scores) if top1_scores else None),
            unique_token_ids=len(token_freq),
            total_candidate_tokens=sum(token_freq.values()),
            score_bins=dict(score_bins),
            token_text_class=dict(class_counter),
        ))


def dump_step_csv(step_rows: Iterable[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    fields = [
        'sample_index', 'step', 'total_steps', 'label', 'gold_answer',
        'candidate_answer', 'topk_token_ids', 'topk_scores', 'decoded_tokens',
        'hidden_state_dim'
    ]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in step_rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze trace_candidates JSONL')
    parser.add_argument('--trace-file', required=True)
    parser.add_argument('--sample-index', type=int, default=None,
                        help='If omitted, use the first line in JSONL')
    parser.add_argument('--topk', type=int, default=5,
                        help='Take first k entries in candidate_tokens per step')
    parser.add_argument('--tokenizer-path', default=None,
                        help='Optional HF tokenizer path for token id decoding')
    parser.add_argument('--output-prefix', default=None,
                        help='Optional prefix. If set, write <prefix>.summary.json and <prefix>.steps.csv')
    args = parser.parse_args()

    records = load_jsonl(args.trace_file)
    sample = pick_sample(records, args.sample_index)
    decoder = build_token_decoder(args.tokenizer_path)
    analyzed = analyze_trace(sample, topk=args.topk, decoder=decoder)

    summary_for_print = dict(
        sample_index=analyzed['sample_index'],
        num_steps=analyzed['num_steps'],
        aggregate=analyzed['aggregate'],
        top10_tokens=analyzed['token_summary'][:10],
    )
    print(json.dumps(summary_for_print, ensure_ascii=False, indent=2))

    if args.output_prefix:
        summary_path = f'{args.output_prefix}.summary.json'
        steps_path = f'{args.output_prefix}.steps.csv'
        os.makedirs(os.path.dirname(summary_path), exist_ok=True) if os.path.dirname(summary_path) else None
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(analyzed, f, ensure_ascii=False, indent=2)
        dump_step_csv(analyzed['step_rows'], steps_path)
        print(f'Wrote: {summary_path}')
        print(f'Wrote: {steps_path}')


if __name__ == '__main__':
    main()
