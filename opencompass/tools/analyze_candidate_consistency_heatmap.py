#!/usr/bin/env python3
"""Analyze candidate-token consistency across steps from trace .pt."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare candidates with finalized tokens and previous-step candidates.')
    parser.add_argument('--trace-path', type=str, required=True,
                        help='Path to trace .pt file.')
    parser.add_argument('--output-image', type=str, default=None,
                        help='Output image path. Defaults to <trace>.candidate_consistency.png')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Optional summary JSON output path.')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Batch sample index. Default: 0')
    parser.add_argument('--candidate-rank', type=int, default=0,
                        help='Candidate rank for previous-step equality check. Default: 0 (top-1)')
    parser.add_argument('--dpi', type=int, default=180,
                        help='DPI for saved image.')
    return parser.parse_args()


def build_matrices(
    steps: List[Dict],
    sample_index: int,
    candidate_rank: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if not steps:
        raise ValueError('No steps found in trace payload.')

    candidate_rows = [
        step['candidate_token_ids'][sample_index].to(torch.int64)  # [L, K]
        for step in steps
    ]
    finalized_rows = [
        step['finalized_token_positions'][sample_index].to(torch.bool)  # [L]
        for step in steps
    ]
    current_rows = [
        step['current_token_ids'][sample_index].to(torch.int64)  # [L]
        for step in steps
    ]

    num_steps = len(candidate_rows)
    seq_len, topk = candidate_rows[0].shape
    if candidate_rank < 0 or candidate_rank >= topk:
        raise ValueError(f'candidate_rank {candidate_rank} out of range [0, {topk - 1}]')

    # finalized_match matrix coding:
    # -1 = N/A (not finalized yet), 0 = no, 1 = yes
    finalized_match = torch.full((num_steps, seq_len), -1, dtype=torch.int64)
    finalized_ids = torch.full((seq_len,), -1, dtype=torch.int64)
    finalized_known = torch.zeros((seq_len,), dtype=torch.bool)

    for s in range(num_steps):
        newly_finalized = finalized_rows[s]
        finalized_ids[newly_finalized] = current_rows[s][newly_finalized]
        finalized_known = finalized_known | newly_finalized

        cand = candidate_rows[s]  # [L, K]
        known_pos = torch.where(finalized_known)[0]
        if known_pos.numel() > 0:
            cand_known = cand[known_pos]  # [M, K]
            final_known = finalized_ids[known_pos].unsqueeze(-1)  # [M, 1]
            in_candidates = (cand_known == final_known).any(dim=-1)
            finalized_match[s, known_pos] = in_candidates.to(torch.int64)

    # prev_candidate_same matrix coding:
    # -1 = N/A (step0), 0 = no, 1 = yes
    prev_candidate_same = torch.full((num_steps, seq_len), -1, dtype=torch.int64)
    for s in range(1, num_steps):
        cur_top = candidate_rows[s][:, candidate_rank]
        prev_top = candidate_rows[s - 1][:, candidate_rank]
        same = (cur_top == prev_top).to(torch.int64)
        prev_candidate_same[s] = same

    counts = {
        'finalized_match_yes': int((finalized_match == 1).sum().item()),
        'finalized_match_no': int((finalized_match == 0).sum().item()),
        'finalized_match_na': int((finalized_match == -1).sum().item()),
        'prev_same_yes': int((prev_candidate_same == 1).sum().item()),
        'prev_same_no': int((prev_candidate_same == 0).sum().item()),
        'prev_same_na': int((prev_candidate_same == -1).sum().item()),
    }
    return finalized_match.numpy(), prev_candidate_same.numpy(), counts


def main() -> None:
    args = parse_args()
    payload = torch.load(args.trace_path, map_location='cpu')
    steps = payload.get('steps', [])

    finalized_match, prev_same, counts = build_matrices(
        steps=steps,
        sample_index=args.sample_index,
        candidate_rank=args.candidate_rank)

    output_image = args.output_image
    if output_image is None:
        base, _ = os.path.splitext(args.trace_path)
        output_image = f'{base}.candidate_consistency.png'
    os.makedirs(os.path.dirname(output_image) or '.', exist_ok=True)

    cmap = ListedColormap([
        '#bdbdbd',  # -1 N/A
        '#e74c3c',  # 0  No
        '#2ecc71',  # 1  Yes
    ])

    show_finalized = finalized_match + 1
    show_prev = prev_same + 1

    fig_h = max(5, min(18, finalized_match.shape[0] * 0.22))
    fig_w = max(10, min(26, finalized_match.shape[1] * 0.05))

    plt.figure(figsize=(fig_w, fig_h))

    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(show_finalized, cmap=cmap, vmin=0, vmax=2, aspect='auto', interpolation='nearest')
    ax1.set_title('Current step candidates contain finalized token?')
    ax1.set_ylabel('Step index')

    ax2 = plt.subplot(2, 1, 2)
    ax2.imshow(show_prev, cmap=cmap, vmin=0, vmax=2, aspect='auto', interpolation='nearest')
    ax2.set_title(f'Current top-{args.candidate_rank + 1} candidate equals previous step?')
    ax2.set_xlabel('Token position in generation window')
    ax2.set_ylabel('Step index')

    legend_handles = [
        Patch(color='#2ecc71', label='Yes'),
        Patch(color='#e74c3c', label='No'),
        Patch(color='#bdbdbd', label='N/A'),
    ]
    ax1.legend(handles=legend_handles, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_image, dpi=args.dpi)
    plt.close()

    print(f'Candidate consistency heatmap saved to: {output_image}')
    print(f'Counts: {counts}')

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'trace_path': args.trace_path,
                    'sample_index': args.sample_index,
                    'candidate_rank': args.candidate_rank,
                    'num_steps': int(finalized_match.shape[0]),
                    'seq_len': int(finalized_match.shape[1]),
                    'counts': counts,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f'Summary json saved to: {args.output_json}')


if __name__ == '__main__':
    main()
