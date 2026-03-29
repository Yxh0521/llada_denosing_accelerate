#!/usr/bin/env python3
"""Export trace payload to JSON while excluding hidden_states."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export trace .pt to JSON, excluding hidden_states.')
    parser.add_argument(
        '--trace-path',
        type=str,
        required=True,
        help='Path to the saved trace .pt file.')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Output JSON path. Defaults to <trace>.no_hiddenstates.json')
    return parser.parse_args()


def tensor_to_jsonable(value: torch.Tensor) -> Any:
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def strip_hiddenstates_and_convert(obj: Any) -> Any:
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key == 'hidden_states':
                continue
            result[key] = strip_hiddenstates_and_convert(value)
        return result
    if isinstance(obj, list):
        return [strip_hiddenstates_and_convert(v) for v in obj]
    if isinstance(obj, tuple):
        return [strip_hiddenstates_and_convert(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return tensor_to_jsonable(obj)
    return obj


def main() -> None:
    args = parse_args()
    trace_payload = torch.load(args.trace_path, map_location='cpu')
    payload_no_hidden = strip_hiddenstates_and_convert(trace_payload)

    output_json = args.output_json
    if output_json is None:
        base, _ = os.path.splitext(args.trace_path)
        output_json = f'{base}.no_hiddenstates.json'
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(payload_no_hidden, f, ensure_ascii=False, indent=2)

    print(f'JSON saved to: {output_json}')


if __name__ == '__main__':
    main()
