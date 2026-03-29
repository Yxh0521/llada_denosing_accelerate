#!/usr/bin/env python3
"""Analyze per-step confidence statistics from a trace .pt file."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Analyze per-step confidence (mean/max/min and positions).')
    parser.add_argument(
        '--trace-path',
        type=str,
        required=True,
        help='Path to trace .pt file saved by LLaDA trace.')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Output JSON path. Defaults to <trace>.step_confidence_stats.json')
    parser.add_argument(
        '--sample-index',
        type=int,
        default=0,
        help='Batch sample index to analyze. Default: 0')
    return parser.parse_args()


def _finite_or_none(values: torch.Tensor) -> torch.Tensor:
    finite_mask = torch.isfinite(values)
    return values[finite_mask]


def analyze_step(step: Dict[str, Any], sample_index: int) -> Dict[str, Any]:
    confidence = step['selected_token_confidence'][sample_index].to(torch.float32)
    finite_conf = _finite_or_none(confidence)

    result: Dict[str, Any] = {
        'block_id': int(step['block_id']),
        'step_id': int(step['step_id']),
        'num_positions': int(confidence.numel()),
        'num_finite_positions': int(finite_conf.numel()),
    }

    if finite_conf.numel() == 0:
        result.update({
            'avg_confidence': None,
            'max_confidence': None,
            'max_confidence_positions': [],
            'min_confidence': None,
            'min_confidence_positions': [],
        })
        return result

    avg_conf = finite_conf.mean()
    max_conf = finite_conf.max()
    min_conf = finite_conf.min()

    max_positions = torch.where(confidence == max_conf)[0].tolist()
    min_positions = torch.where(confidence == min_conf)[0].tolist()

    result.update({
        'avg_confidence': float(avg_conf.item()),
        'max_confidence': float(max_conf.item()),
        'max_confidence_positions': [int(x) for x in max_positions],
        'min_confidence': float(min_conf.item()),
        'min_confidence_positions': [int(x) for x in min_positions],
    })
    return result


def main() -> None:
    args = parse_args()
    payload = torch.load(args.trace_path, map_location='cpu')
    steps: List[Dict[str, Any]] = payload.get('steps', [])

    stats = [analyze_step(step, args.sample_index) for step in steps]

    output_json = args.output_json
    if output_json is None:
        base, _ = os.path.splitext(args.trace_path)
        output_json = f'{base}.step_confidence_stats.json'
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)

    output = {
        'trace_path': args.trace_path,
        'sample_index': args.sample_index,
        'num_steps': len(stats),
        'steps': stats,
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'Step confidence stats saved to: {output_json}')


if __name__ == '__main__':
    main()
