#!/usr/bin/env python3
"""Visualize token stability across steps from trace .pt."""

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
        description='Draw token stability heatmap from step trace payload.')
    parser.add_argument('--trace-path', type=str, required=True,
                        help='Path to trace .pt file.')
    parser.add_argument('--output-image', type=str, default=None,
                        help='Output image path. Defaults to <trace>.token_stability.png')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Optional summary json path.')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Batch sample index. Default: 0')
    parser.add_argument('--dpi', type=int, default=180,
                        help='DPI for saved image.')
    return parser.parse_args()


def build_category_matrix(steps: List[Dict], sample_index: int) -> Tuple[np.ndarray, Dict[str, int]]:
    if not steps:
        raise ValueError('No steps found in trace payload.')

    token_rows = [
        step['current_token_ids'][sample_index].to(torch.int64)
        for step in steps
    ]
    finalized_rows = [
        step['finalized_token_positions'][sample_index].to(torch.bool)
        for step in steps
    ]

    num_steps = len(token_rows)
    seq_len = token_rows[0].numel()

    cat = torch.zeros((num_steps, seq_len), dtype=torch.int64)
    # Categories:
    # 0 = 未确定且与上一步相同（稳定）
    # 1 = 未确定且与上一步不同（波动）
    # 2 = 已确定且保持不变（绿色）
    # 3 = 已确定后仍变化（异常变化）

    finalized_cum = torch.zeros(seq_len, dtype=torch.bool)
    prev_tokens = None

    for s in range(num_steps):
        cur_tokens = token_rows[s]
        newly_finalized = finalized_rows[s]
        finalized_before = finalized_cum.clone()
        finalized_cum = finalized_cum | newly_finalized

        changed = torch.zeros(seq_len, dtype=torch.bool)
        if prev_tokens is not None:
            changed = cur_tokens != prev_tokens

        unfinalized = ~finalized_cum
        finalized_now = finalized_cum

        cat[s, unfinalized & ~changed] = 0
        cat[s, unfinalized & changed] = 1
        cat[s, finalized_now] = 2

        # 如果这个位置在前一步已经是 finalized，当前又变化，标记为“确定后仍变化”
        changed_after_finalized = finalized_before & changed
        cat[s, changed_after_finalized] = 3

        prev_tokens = cur_tokens

    counts = {
        'unfinalized_stable': int((cat == 0).sum().item()),
        'unfinalized_changed': int((cat == 1).sum().item()),
        'finalized_stable': int((cat == 2).sum().item()),
        'changed_after_finalized': int((cat == 3).sum().item()),
    }
    return cat.numpy(), counts


def main() -> None:
    args = parse_args()
    payload = torch.load(args.trace_path, map_location='cpu')
    steps = payload.get('steps', [])

    category_matrix, counts = build_category_matrix(
        steps=steps, sample_index=args.sample_index)

    output_image = args.output_image
    if output_image is None:
        base, _ = os.path.splitext(args.trace_path)
        output_image = f'{base}.token_stability.png'
    os.makedirs(os.path.dirname(output_image) or '.', exist_ok=True)

    cmap = ListedColormap([
        '#f5f5f5',  # 0 未确定且稳定
        '#f39c12',  # 1 未确定且波动
        '#2ecc71',  # 2 已确定且稳定（绿色）
        '#8e44ad',  # 3 已确定后仍变化
    ])

    fig_h = max(4, min(16, category_matrix.shape[0] * 0.18))
    fig_w = max(8, min(24, category_matrix.shape[1] * 0.04))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(category_matrix, cmap=cmap, vmin=0, vmax=3, aspect='auto', interpolation='nearest')
    plt.xlabel('Token position in generation window')
    plt.ylabel('Step index')
    plt.title('Token Stability Heatmap Across Steps')

    legend_handles = [
        Patch(color='#f5f5f5', label='Unfinalized & unchanged'),
        Patch(color='#f39c12', label='Unfinalized & changed'),
        Patch(color='#2ecc71', label='Finalized & unchanged (green)'),
        Patch(color='#8e44ad', label='Changed after finalized'),
    ]
    plt.legend(handles=legend_handles, loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_image, dpi=args.dpi)
    plt.close()

    print(f'Token stability heatmap saved to: {output_image}')
    print(f'Category counts: {counts}')

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'trace_path': args.trace_path,
                    'sample_index': args.sample_index,
                    'num_steps': int(category_matrix.shape[0]),
                    'seq_len': int(category_matrix.shape[1]),
                    'category_counts': counts,
                },
                f,
                ensure_ascii=False,
                indent=2)
        print(f'Summary json saved to: {args.output_json}')


if __name__ == '__main__':
    main()
