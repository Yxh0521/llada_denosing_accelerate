from mmengine.config import read_base

with read_base():
    from opencompass.models import LLaDAModel
    from opencompass.openicl.icl_prompt_template import PromptTemplate
    from opencompass.openicl.icl_retriever import ZeroRetriever
    from opencompass.openicl.icl_inferencer import GenInferencer
    from opencompass.datasets import (GSM8KTrainDataset, Gsm8kEvaluator,
                                      gsm8k_dataset_postprocess,
                                      gsm8k_postprocess)

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=
                    "Question: {question}\nLet's think step by step\nAnswer:"
                ),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

gsm8k_eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

datasets = [
    dict(
        abbr='gsm8k-train-trace',
        type=GSM8KTrainDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]

models = [
    dict(
        type=LLaDAModel,
        abbr='llada-8b-instruct',
        path='/mnt/wuyuzhang/yangxihan/models/LLaDA-8B-Instruct',
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
eval_cfg = {
    'gen_blocksize': 8,
    'gen_length': 256,
    'gen_steps': 256,
    'batch_size': 1,
    'batch_size_': 1,
    'trace_enabled': True,
    'trace_every_steps': 30,
    'trace_topk': 5,
    'trace_sample_index': 0,
    'trace_output_path': 'outputs/gsm8k_train_trace_block8.jsonl',
}
for model in models:
    model.update(eval_cfg)

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
        retry=3))
