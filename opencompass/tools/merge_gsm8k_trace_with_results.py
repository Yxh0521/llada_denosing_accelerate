import argparse
import json
from pathlib import Path


def load_trace_map(trace_jsonl: Path):
    trace_map = {}
    with trace_jsonl.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            trace_map[item['prompt']] = item
    return trace_map


def main():
    parser = argparse.ArgumentParser(
        description='Merge step-wise trace records with OpenCompass GSM8K details.')
    parser.add_argument('--trace-jsonl', required=True, type=Path)
    parser.add_argument('--results-json', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    trace_map = load_trace_map(args.trace_jsonl)
    with args.results_json.open('r', encoding='utf-8') as f:
        results = json.load(f)

    merged = []
    for sample_idx, detail in results.items():
        if sample_idx == 'type':
            continue
        prompt = detail.get('prompt')
        trace = trace_map.get(prompt, {})
        merged.append({
            'sample_idx': int(sample_idx),
            'prompt': prompt,
            'prediction': detail.get('predictions'),
            'reference': detail.get('references'),
            'correct': detail.get('correct'),
            'trace': trace.get('steps', []),
            'prompt_hash': trace.get('prompt_hash'),
        })

    merged.sort(key=lambda x: x['sample_idx'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'Merged {len(merged)} samples -> {args.output}')


if __name__ == '__main__':
    main()
