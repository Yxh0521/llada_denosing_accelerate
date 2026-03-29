#!/usr/bin/env python3
"""Plot confidence trends from step confidence stats JSON."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot avg confidence and max-confidence-token count by step.')
    parser.add_argument(
        '--stats-json',
        type=str,
        required=True,
        help='Path to *.step_confidence_stats.json.')
    parser.add_argument(
        '--output-image',
        type=str,
        default=None,
        help='Output image path. Defaults to <stats>.png')
    parser.add_argument(
        '--dpi',
        type=int,
        default=180,
        help='DPI for saved figure.')
    return parser.parse_args()


def load_stats(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    stats = load_stats(args.stats_json)
    steps: List[Dict[str, Any]] = stats.get('steps', [])

    if not steps:
        raise ValueError(f'No step records found in: {args.stats_json}')

    step_indices = list(range(len(steps)))
    avg_confidence = [
        (step.get('avg_confidence') if step.get('avg_confidence') is not None else float('nan'))
        for step in steps
    ]
    max_token_counts = [len(step.get('max_confidence_positions', [])) for step in steps]

    output_image = args.output_image
    if output_image is None:
        base, _ = os.path.splitext(args.stats_json)
        output_image = f'{base}.png'
    os.makedirs(os.path.dirname(output_image) or '.', exist_ok=True)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(step_indices, avg_confidence, marker='o', markersize=2, linewidth=1)
    plt.title('Average Confidence per Step')
    plt.xlabel('Step index')
    plt.ylabel('Average confidence')
    plt.grid(alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.bar(step_indices, max_token_counts, width=0.8)
    plt.title('Count of Max-Confidence Tokens per Step')
    plt.xlabel('Step index')
    plt.ylabel('Count')
    plt.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_image, dpi=args.dpi)
    plt.close()

    print(f'Figure saved to: {output_image}')


if __name__ == '__main__':
    main()
