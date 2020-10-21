from transformers.tokenization_bert import BertTokenizer
# from transformers.tokenization_xlnet import XLNetTokenizer
from transformers.configuration_bert import BertConfig
# from transformers.configuration_roberta import RobertaConfig
# from transformers.tokenization_roberta import RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
from .multitask_bert import BertForHypernymsExtraction
# from .multitask_roberta import RobertaForMultitaskLearning
# from .multitask_xlnet import XLNetForMultiLearning
# from transformers.configuration_xlnet import XLNetConfig
import torch
import os
import json
from collections import Counter


class InputExample(object):

    def __init__(
        self,
        guid: str,
        tokens: list,
        tags_sequence: list
    ):
        self.guid = guid
        self.tokens = tokens
        self.tags_sequence = tags_sequence


class DataProcessor(object):
    """Processor for the DEFTEVAL data set."""

    def _read_json(self, input_file):
        with open(input_file, "r", encoding='utf-8') as reader:
            data = json.load(reader)
        return data

    def get_train_examples(self, data_dir):
        return self.create_examples(
            self._read_json(
                os.path.join(data_dir, f"train.json")
            ),
            "train"
        )

    def get_dev_examples(self, data_dir):
        return self.create_examples(
            self._read_json(
                os.path.join(data_dir, f"dev.json")
            ),
            "dev"
        )

    def get_test_examples(self, test_file):
        return self.create_examples(
            self._read_json(test_file),
            "test"
        )

    def get_sequence_labels(
        self,
        data_dir: str,
        sequence_type: str = 'tags_sequence',
        logger = None
    ):
        dataset = self._read_json(
            os.path.join(data_dir, "train.json")
        )
        denominator = len([
            lab for example in dataset for lab in example[sequence_type]
        ])
        counter = Counter()
        labels = []
        for example in dataset:
            for lab in example[sequence_type]:
                counter[lab] += 1
        if logger is not None:
            logger.info(f"{sequence_type}: {len(counter)} labels")

        for label, counter in counter.most_common():
            if logger is not None:
                logger.info("%s: %.2f%%" % (label, counter * 100.0 / denominator))
            if label not in labels:
                labels.append(label)
        return labels


    def create_examples(self, dataset, set_type):
        examples = []
        for example in dataset:
            examples.append(
                InputExample(
                    guid=f"{set_type}-{example['id']}",
                    tokens=example["token"],
                    tags_sequence=example["tags_sequence"]
                )
            )
        return examples


def get_dataloader_and_tensors(
        features: list,
        batch_size: int
):
    input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long
    )
    input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long
    )
    segment_ids = torch.tensor(
        [f.segment_ids for f in features],
        dtype=torch.long
    )
    tags_sequence_labels_ids = torch.tensor(
        [f.tags_sequence_ids for f in features],
        dtype=torch.long
    )
    token_valid_pos_ids = torch.tensor(
        [f.token_valid_pos_ids for f in features],
        dtype=torch.long
    )
    eval_data = TensorDataset(
        input_ids, input_mask, segment_ids,
        tags_sequence_labels_ids, token_valid_pos_ids
    )

    dataloader = DataLoader(eval_data, batch_size=batch_size)

    return dataloader, tags_sequence_labels_ids

tokenizers = {
    "bert-large-uncased": BertTokenizer,
    # "xlnet-large-cased": XLNetTokenizer,
    # "roberta-large": RobertaTokenizer
}

models = {
    "bert-large-uncased": BertForHypernymsExtraction,
    # "roberta-large": RobertaForMultitaskLearning,
    # "xlnet-large-cased": XLNetForMultiLearning
}

configs = {
    "bert-large-uncased": BertConfig,
    # "roberta-large": RobertaConfig,
    # "xlnet-large-cased": XLNetConfig
}
