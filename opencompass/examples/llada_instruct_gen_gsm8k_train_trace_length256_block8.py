import os

from opencompass.datasets import (
    GSM8KDataset,
    Gsm8kEvaluator,
    gsm8k_dataset_postprocess,
    gsm8k_postprocess,
)
from opencompass.models import LLaDAModel
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

# Self-contained config: avoid dependency on installed package config modules.
# 使用 GSM8K 训练集作为推理集。
gsm8k_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    test_split='train',
    test_range='[:200]',
)

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="Question: {question}\nLet's think step by step\nAnswer:")
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=256),
)

gsm8k_eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)

datasets = [
    dict(
        abbr='gsm8k_train_trace',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]

models = [
    dict(
        type=LLaDAModel,
        abbr='llada-8b-instruct-trace',
        path=os.getenv('LLADA_MODEL_PATH', 'GSAI-ML/LLaDA-8B-Instruct'),
        max_out_len=256,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        gen_blocksize=8,
        gen_length=256,
        gen_steps=256,
        batch_size_=1,
        diff_confidence_eos_eot_inf=False,
        diff_logits_eos_inf=False,
        diff_trace_every=30,
        diff_trace_topk=5,
        diff_trace_output='outputs/gsm8k_train_trace_step30.pt',
    )
]

summarizer = dict(dataset_abbrs=['gsm8k_train_trace'], summary_groups=[dict(name='gsm8k', subsets=['gsm8k_train_trace'])])

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
