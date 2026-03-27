from datasets import DatasetDict

from opencompass.registry import LOAD_DATASET

from .gsm8k import GSM8KDataset


@LOAD_DATASET.register_module()
class GSM8KSingleSampleDataset(GSM8KDataset):
    """GSM8K dataset loader that keeps only one test sample by index."""

    @staticmethod
    def load(path, sample_id: int = 0):
        dataset = GSM8KDataset.load(path)
        return DatasetDict({
            'train': dataset['train'],
            'test': dataset['test'].select([sample_id]),
        })
