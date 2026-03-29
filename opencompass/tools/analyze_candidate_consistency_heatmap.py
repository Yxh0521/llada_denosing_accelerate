#!/usr/bin/env python3
"""Analyze candidate-token consistency across steps from trace .pt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare candidates with finalized tokens and previous-step candidates.')
    parser.add_argument('--trace-path', type=str, default=None,
                        help='Path to one trace .pt file.')
    parser.add_argument('--trace-dir', type=str, default=None,
                        help='Directory containing many trace .pt files.')
    parser.add_argument('--output-image', type=str, default=None,
                        help='Output image path (single-file mode only). Defaults to <trace>.candidate_consistency.png')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for batch mode. Defaults to trace-dir.')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Optional summary JSON output path.')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Batch sample index. Default: 0')
    parser.add_argument('--candidate-rank', type=int, default=0,
                        help='Candidate rank for previous-step equality check. Default: 0 (top-1)')
    parser.add_argument('--dpi', type=int, default=180,
                        help='DPI for saved image.')
    args = parser.parse_args()
    if (args.trace_path is None) == (args.trace_dir is None):
        parser.error('Exactly one of --trace-path or --trace-dir must be provided.')
    if args.trace_dir is not None and args.output_image is not None:
        parser.error('--output-image is only valid with --trace-path.')
    return args


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


def render_one_trace(
    trace_path: str,
    output_image: str,
    output_json: str | None,
    sample_index: int,
    candidate_rank: int,
    dpi: int,
) -> None:
    payload = torch.load(trace_path, map_location='cpu')
    steps = payload.get('steps', [])

    finalized_match, prev_same, counts = build_matrices(
        steps=steps,
        sample_index=sample_index,
        candidate_rank=candidate_rank)
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
    ax2.set_title(f'Current top-{candidate_rank + 1} candidate equals previous step?')
    ax2.set_xlabel('Token position in generation window')
    ax2.set_ylabel('Step index')

    legend_handles = [
        Patch(color='#2ecc71', label='Yes'),
        Patch(color='#e74c3c', label='No'),
        Patch(color='#bdbdbd', label='N/A'),
    ]
    ax1.legend(handles=legend_handles, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_image, dpi=dpi)
    plt.close()

    print(f'[{trace_path}] heatmap saved to: {output_image}')
    print(f'Counts: {counts}')

    if output_json:
        os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'trace_path': trace_path,
                    'sample_index': sample_index,
                    'candidate_rank': candidate_rank,
                    'num_steps': int(finalized_match.shape[0]),
                    'seq_len': int(finalized_match.shape[1]),
                    'counts': counts,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f'Summary json saved to: {output_json}')


def main() -> None:
    args = parse_args()

    if args.trace_path is not None:
        output_image = args.output_image
        if output_image is None:
            base, _ = os.path.splitext(args.trace_path)
            output_image = f'{base}.candidate_consistency.png'
        render_one_trace(
            trace_path=args.trace_path,
            output_image=output_image,
            output_json=args.output_json,
            sample_index=args.sample_index,
            candidate_rank=args.candidate_rank,
            dpi=args.dpi,
        )
        return

    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir) if args.output_dir else trace_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_files = sorted(trace_dir.glob('*.pt'))
    if not trace_files:
        raise FileNotFoundError(f'No .pt files found in directory: {trace_dir}')

    for trace_file in trace_files:
        stem = trace_file.stem
        output_image = str(output_dir / f'{stem}.candidate_consistency.png')
        output_json = None
        if args.output_json:
            # In batch mode, interpret --output-json as output JSON directory.
            json_dir = Path(args.output_json)
            json_dir.mkdir(parents=True, exist_ok=True)
            output_json = str(json_dir / f'{stem}.candidate_consistency.json')
        try:
            render_one_trace(
                trace_path=str(trace_file),
                output_image=output_image,
                output_json=output_json,
                sample_index=args.sample_index,
                candidate_rank=args.candidate_rank,
                dpi=args.dpi,
            )
        except Exception as e:
            print(f'[{trace_file}] skipped due to error: {e}')


if __name__ == '__main__':
    main()
