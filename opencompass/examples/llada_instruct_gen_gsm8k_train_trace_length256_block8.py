from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.models.dllm.llada_instruct_8b import models as llada_instruct_8b_models

# 使用 GSM8K 训练集做推理（OpenCompass 推理框架）
datasets = []
for d in gsm8k_datasets:
    cfg = d.copy()
    cfg['abbr'] = 'gsm8k_train_trace'
    reader_cfg = cfg.get('reader_cfg', {}).copy()
    reader_cfg.update(dict(test_split='train', test_range='[:200]'))
    cfg['reader_cfg'] = reader_cfg
    datasets.append(cfg)

models = llada_instruct_8b_models

# 采样 + trace 设置
trace_cfg = dict(
    gen_blocksize=8,
    gen_length=256,
    gen_steps=256,
    batch_size=1,
    batch_size_=1,
    diff_confidence_eos_eot_inf=False,
    diff_logits_eos_inf=False,
    diff_trace_every=30,
    diff_trace_topk=5,
    diff_trace_output='outputs/gsm8k_train_trace_step30.pt',
)
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
        min_task_size=8,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLInferTask),
        retry=2,
    ),
)
