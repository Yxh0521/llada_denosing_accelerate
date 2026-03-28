#!/usr/bin/env python3
"""Analyze LLaDA trace .pt and draw cosine-similarity heatmap.

This script computes cosine similarity between adjacent steps on the same
token position using saved `hidden_states` in trace payload.

Example:
    python opencompass/tools/analyze_trace_cosine_heatmap.py \
      --trace-path outputs/gsm8k_trace.pt \
      --output-image outputs/gsm8k_trace_cosine_heatmap.png
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute adjacent-step cosine similarity heatmap from trace .pt')
    parser.add_argument('--trace-path', type=str, required=True,
                        help='Path to saved trace .pt file.')
    parser.add_argument('--output-image', type=str, default=None,
                        help='Output heatmap image path. Defaults to <trace>.cosine.png')
    parser.add_argument('--output-tensor', type=str, default=None,
                        help='Optional output .pt path for cosine matrix.')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Batch sample index in hidden_states. Default: 0')
    parser.add_argument('--dpi', type=int, default=180,
                        help='DPI for saved figure.')
    return parser.parse_args()


def cosine_matrix_from_trace(trace_payload: dict,
                             sample_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cosine matrix and token ids.

    Returns:
        cosine_matrix: [num_adjacent_step_pairs, gen_length]
        token_ids: [num_steps, gen_length] (current token ids per step)
    """
    steps = trace_payload.get('steps', [])
    if len(steps) < 2:
        raise ValueError('Trace has fewer than 2 steps, cannot compute adjacent similarity.')

    cos_rows = []
    token_rows = []
    for step in steps:
        token_rows.append(step['current_token_ids'][sample_index].to(torch.int32))

    for i in range(len(steps) - 1):
        hs_a = steps[i]['hidden_states'][sample_index].to(torch.float32)       # [L, H]
        hs_b = steps[i + 1]['hidden_states'][sample_index].to(torch.float32)   # [L, H]
        cos = F.cosine_similarity(hs_a, hs_b, dim=-1)                          # [L]
        cos_rows.append(cos)

    cosine_matrix = torch.stack(cos_rows, dim=0)  # [S-1, L]
    token_ids = torch.stack(token_rows, dim=0)    # [S, L]
    return cosine_matrix, token_ids


def main() -> None:
    args = parse_args()
    trace_payload = torch.load(args.trace_path, map_location='cpu')

    cosine_matrix, token_ids = cosine_matrix_from_trace(
        trace_payload=trace_payload,
        sample_index=args.sample_index)

    output_image = args.output_image
    if output_image is None:
        base, _ = os.path.splitext(args.trace_path)
        output_image = f'{base}.cosine.png'

    os.makedirs(os.path.dirname(output_image) or '.', exist_ok=True)

    fig_h = max(4, min(16, cosine_matrix.shape[0] * 0.18))
    fig_w = max(8, min(20, cosine_matrix.shape[1] * 0.04))
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(
        cosine_matrix.numpy(),
        cmap='viridis',
        vmin=-1.0,
        vmax=1.0,
        aspect='auto',
        interpolation='nearest')
    plt.colorbar(im, label='Cosine similarity')
    plt.xlabel('Token position in generation window')
    plt.ylabel('Adjacent step pair index (step i -> i+1)')
    plt.title('Adjacent-step cosine similarity at same token positions')
    plt.tight_layout()
    plt.savefig(output_image, dpi=args.dpi)
    plt.close()

    print(f'Cosine matrix shape: {tuple(cosine_matrix.shape)}')
    print(f'Heatmap saved to: {output_image}')

    if args.output_tensor:
        os.makedirs(os.path.dirname(args.output_tensor) or '.', exist_ok=True)
        torch.save(
            {
                'cosine_matrix': cosine_matrix,
                'token_ids': token_ids,
                'trace_path': args.trace_path,
                'sample_index': args.sample_index,
            },
            args.output_tensor)
        print(f'Cosine tensor saved to: {args.output_tensor}')


if __name__ == '__main__':
    main()
