#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_infer_cfg
from opencompass.datasets.gsm8k import Gsm8kEvaluator, gsm8k_dataset_postprocess, gsm8k_postprocess
from opencompass.models.dllm import LLaDAModel


def build_messages(question: str):
    rounds = gsm8k_infer_cfg['prompt_template']['template']['round']
    messages = []
    for item in rounds:
        prompt = item['prompt'].format(question=question)
        messages.append({'role': item['role'], 'prompt': prompt})
    return messages


def load_test_sample(dataset_path: Path, sample_id: int):
    test_path = dataset_path / 'test.jsonl'
    with test_path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx == sample_id:
                row = json.loads(line)
                return row
    raise IndexError(f'sample_id {sample_id} out of range: {test_path}')


def main():
    parser = argparse.ArgumentParser(description='Run one GSM8K sample with LLaDA+OpenCompass style prompt and save per-step traces.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--dataset-path', default='opencompass/gsm8k')
    parser.add_argument('--sample-id', type=int, default=0)
    parser.add_argument('--trace-path', default='outputs/gsm8k_single_trace.pt')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--trace-hidden-layer', type=int, default=-1)
    args = parser.parse_args()

    sample = load_test_sample(Path(args.dataset_path), args.sample_id)
    question = sample['question']
    answer = gsm8k_dataset_postprocess(sample['answer'])

    messages = build_messages(question)
    model = LLaDAModel(
        path=args.model_path,
        gen_blocksize=8,
        gen_length=256,
        gen_steps=256,
        batch_size=1,
        batch_size_=1,
        save_step_trace=True,
        step_trace_path=args.trace_path,
        step_trace_topk=args.topk,
        step_trace_hidden_layer=args.trace_hidden_layer,
        trace_sample_id=str(args.sample_id),
        trace_reference_answer=answer,
    )
    response = model.generate([messages], max_out_len=512)[0]
    pred = gsm8k_postprocess(response)
    correct = Gsm8kEvaluator().is_equal(pred, answer)

    print('sample_id:', args.sample_id)
    print('prediction:', pred)
    print('answer:', answer)
    print('correct:', correct)
    print('trace_path:', args.trace_path)
    print('trace_meta:', f'{args.trace_path}.meta.json')


if __name__ == '__main__':
    main()
