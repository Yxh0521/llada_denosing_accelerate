"""Run GSM8K with LLaDA (8-block, 256-step) and step trace dump.

Usage:
    cd opencompass
    python run.py examples/llada_instruct_gen_gsm8k_length256_block8_trace_single.py \
      -w outputs/llada_instruct_gsm8k_block8_trace_single

Note:
- This config keeps the same 4-shot prompt template as gsm8k_gen_1d7fe4.
- It reuses framework GSM8K dataset config directly (full dataset).
"""

from copy import deepcopy

from mmengine.config import read_base

TRACE_TOPK = 5
TRACE_HIDDEN_LAYER = -1  # e.g. -1 means last layer
TRACE_REFERENCE_ANSWER = None  # e.g. "42"

with read_base():
    from opencompass.configs.models.dllm.llada_instruct_8b import \
        models as llada_instruct_8b_models
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import \
        gsm8k_datasets

datasets = deepcopy(gsm8k_datasets)


models = llada_instruct_8b_models

trace_cfg = {
    'gen_blocksize': 8,
    'gen_length': 256,
    'gen_steps': 256,
    'batch_size': 1,
    'batch_size_': 1,
    'save_step_trace': True,
    'step_trace_topk': TRACE_TOPK,
    'step_trace_hidden_layer': TRACE_HIDDEN_LAYER,
    'step_trace_path': 'outputs/gsm8k_trace.pt',
    'trace_sample_id': None,
    'trace_reference_answer': TRACE_REFERENCE_ANSWER,
}

for model in models:
    model.update(trace_cfg)

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=1,
        num_split=None,
        min_task_size=1,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLInferTask),
        retry=2,
    ),
)
