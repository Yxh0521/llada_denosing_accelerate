#!/usr/bin/env python3
"""Analyze perturbations across all trace .pt files in a directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Count token perturbations per position across trace .pt files.')
    parser.add_argument('--trace-dir', type=str, required=True,
                        help='Directory containing trace .pt files.')
    parser.add_argument('--tokenizer-path', type=str, required=True,
                        help='Tokenizer/model path used to decode token ids.')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Batch sample index in trace tensors. Default: 0')
    parser.add_argument('--topk', type=int, default=20,
                        help='Top-k perturbed tokens to report. Default: 20')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Output JSON path. Defaults to <trace-dir>/perturbation_stats.json')
    return parser.parse_args()


def analyze_one_trace(payload: Dict, sample_index: int):
    steps: List[Dict] = payload.get('steps', [])
    if len(steps) < 2:
        return {}, {}, Counter(), Counter()

    current_rows = [
        step['current_token_ids'][sample_index].to(torch.int64)
        for step in steps
    ]
    finalized_rows = [
        step['finalized_token_positions'][sample_index].to(torch.bool)
        for step in steps
    ]

    total_by_pos = defaultdict(int)
    post_finalized_by_pos = defaultdict(int)
    total_token_counter: Counter = Counter()
    post_finalized_token_counter: Counter = Counter()

    finalized_before = finalized_rows[0].clone()
    for s in range(1, len(steps)):
        prev_tokens = current_rows[s - 1]
        cur_tokens = current_rows[s]
        changed = (cur_tokens != prev_tokens)

        changed_pos = torch.where(changed)[0].tolist()
        for pos in changed_pos:
            total_by_pos[int(pos)] += 1

        changed_token_ids = cur_tokens[changed].tolist()
        total_token_counter.update(int(tid) for tid in changed_token_ids)

        changed_after_finalized = changed & finalized_before
        changed_after_finalized_pos = torch.where(changed_after_finalized)[0].tolist()
        for pos in changed_after_finalized_pos:
            post_finalized_by_pos[int(pos)] += 1

        changed_after_finalized_token_ids = cur_tokens[changed_after_finalized].tolist()
        post_finalized_token_counter.update(int(tid) for tid in changed_after_finalized_token_ids)

        finalized_before = finalized_before | finalized_rows[s]

    return total_by_pos, post_finalized_by_pos, total_token_counter, post_finalized_token_counter


def main() -> None:
    args = parse_args()
    trace_dir = Path(args.trace_dir)
    trace_files = sorted(trace_dir.glob('*.pt'))
    if not trace_files:
        raise FileNotFoundError(f'No .pt files found in: {trace_dir}')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    total_by_pos_all = defaultdict(int)
    post_finalized_by_pos_all = defaultdict(int)
    total_token_counter_all: Counter = Counter()
    post_finalized_token_counter_all: Counter = Counter()

    loaded_files = 0
    skipped_files = []

    for pt_file in trace_files:
        try:
            payload = torch.load(pt_file, map_location='cpu')
            total_by_pos, post_finalized_by_pos, total_token_counter, post_finalized_token_counter = analyze_one_trace(
                payload=payload, sample_index=args.sample_index)
            for pos, cnt in total_by_pos.items():
                total_by_pos_all[pos] += cnt
            for pos, cnt in post_finalized_by_pos.items():
                post_finalized_by_pos_all[pos] += cnt
            total_token_counter_all.update(total_token_counter)
            post_finalized_token_counter_all.update(post_finalized_token_counter)
            loaded_files += 1
        except Exception as e:
            skipped_files.append({'file': str(pt_file), 'error': str(e)})

    def decode_top(counter: Counter, k: int):
        items = counter.most_common(k)
        decoded = []
        for tid, cnt in items:
            token_text = tokenizer.convert_ids_to_tokens(int(tid))
            decoded.append({
                'token_id': int(tid),
                'token_text': token_text,
                'count': int(cnt),
            })
        return decoded

    most_total = decode_top(total_token_counter_all, args.topk)
    most_post_finalized = decode_top(post_finalized_token_counter_all, args.topk)

    output = {
        'trace_dir': str(trace_dir),
        'sample_index': args.sample_index,
        'loaded_files': loaded_files,
        'total_files': len(trace_files),
        'skipped_files': skipped_files,
        'perturbation_count_by_position': {
            str(pos): int(cnt) for pos, cnt in sorted(total_by_pos_all.items())
        },
        'post_finalized_perturbation_count_by_position': {
            str(pos): int(cnt) for pos, cnt in sorted(post_finalized_by_pos_all.items())
        },
        'top_perturbed_tokens': most_total,
        'top_post_finalized_perturbed_tokens': most_post_finalized,
        'most_perturbed_token': most_total[0] if most_total else None,
        'most_post_finalized_perturbed_token': most_post_finalized[0] if most_post_finalized else None,
    }

    output_json = args.output_json
    if output_json is None:
        output_json = str(trace_dir / 'perturbation_stats.json')
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'Loaded {loaded_files}/{len(trace_files)} trace files.')
    print(f'Perturbation stats saved to: {output_json}')
    if output['most_perturbed_token'] is not None:
        print('Most perturbed token:', output['most_perturbed_token'])
    if output['most_post_finalized_perturbed_token'] is not None:
        print('Most post-finalized perturbed token:', output['most_post_finalized_perturbed_token'])


if __name__ == '__main__':
    main()
