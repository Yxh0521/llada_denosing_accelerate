from copy import deepcopy

from mmengine.config import read_base

with read_base():
    from opencompass.opencompass.configs.datasets.gsm8k.gsm8k_gen import \
        gsm8k_datasets
    from opencompass.opencompass.configs.models.dllm.llada_instruct_8b import \
        models as llada_instruct_8b_models

# Keep the same 4-shot GSM8K prompt template as gsm8k_gen,
# and run inference on GSM8K test split for early-stop tracing experiments.
datasets = deepcopy(gsm8k_datasets)
for dataset in datasets:
    dataset['abbr'] = 'gsm8k_test'
    dataset['reader_cfg'] = deepcopy(dataset['reader_cfg'])
    dataset['reader_cfg'].update(train_split='train', test_split='test')

models = deepcopy(llada_instruct_8b_models)

# generation settings aligned with llada_instruct_gen_gsm8k_length256_block8.py
# and tracing every 30 denoising steps.
eval_cfg = {
    'gen_blocksize': 8,
    'gen_length': 256,
    'gen_steps': 256,
    'batch_size': 1,
    'batch_size_': 1,
    'trace_every_n_steps': 30,
    'trace_topk': 5,
}
for model in models:
    model.update(eval_cfg)

# Attach trace dump options to the inferencer so it saves
# candidate answers + labels for one sample.
for dataset in datasets:
    inferencer_cfg = dataset['infer_cfg']['inferencer']
    inferencer_cfg.update(
        trace_dump_filename='trace_candidates_step30.jsonl',
        trace_max_samples=1,
    )

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,
        num_split=None,
        min_task_size=16,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5,
    ),
)
