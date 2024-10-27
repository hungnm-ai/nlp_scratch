from transformers import AutoTokenizer, RobertaForSequenceClassification, AutoConfig

from bert_cnn import RobertaCNNForSequenceClassification, RobertaCNNConfig

model_name = "vinai/phobert-base-v2"
cnn = {
    "filter_sizes": [2, 3, 4],
    "num_filters": 100
}
config = RobertaCNNConfig.from_pretrained(model_name, cnn=cnn)
model = RobertaCNNForSequenceClassification.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("xin chao", return_tensors="pt")
output = model(**inputs)
print(output.logits.size())

import json
import logging
import os
import sys
import warnings
from typing import Dict, Tuple

import datasets
import torch.cuda
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    set_seed,
)

from arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from metrics import compute_metrics

datasets.disable_progress_bar()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.simplefilter(action="ignore", category=FutureWarning)

warnings.simplefilter("ignore", UserWarning)
set_seed(42)

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def load_model(
        model_args: ModelArguments,
        config,
) -> AutoModelForSequenceClassification:
    """
    Load model from HuggingFace-Hub
    Args:
        model_args:
        config: Model config

    Returns:
        Model for classification
    """
    device = {"": 0} if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Load pretrained model: {model_args.model_name_or_path} ")
    _model = RobertaForSequenceClassificationFocalLoss.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        device_map=device,
        use_focal_loss=model_args.use_focal_loss,
        alpha=0.25,
        gamma=2,
    )

    return _model


def load_tokenizer(model_args: ModelArguments) -> PreTrainedTokenizer:
    """
    Load tokenizer from HuggingFace-Hub
    Args:
        model_args:

    Returns:
        Tokenizer corresponding to Model
    """
    logger.info(f"Load tokenizer from {model_args.tokenizer_name}")
    _tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        padding_side=model_args.padding_side,
        legacy=True,
        use_fast=True,
        model_max_length=model_args.model_max_length,
    )

    return _tokenizer


def load_dataset_from_cache(
        cache_dir: str,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    train_ds = datasets.load_from_disk(os.path.join(cache_dir, "train"))
    test_ds = datasets.load_from_disk(os.path.join(cache_dir, "test"))
    return train_ds, test_ds


def train():
    arg_parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = arg_parser.parse_args_into_dataclasses()

    tokenizer = load_tokenizer(model_args=model_args)

    logger.setLevel(logging.INFO)

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            max_length=model_args.model_max_length,
            truncation=True,
        )

    train_ds_tokenized = train_ds.map(preprocess_function)
    test_ds_tokenized = test_ds.map(preprocess_function)


config = AutoConfig.from_pretrained(
    model_args.model_name_or_path,
    num_labels=num_labels,
    id2label=id2class,
    label2id=class2id,
    problem_type="multi_label_classification",
)

model = load_model(model_args=model_args, config=config)
collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    return_tensors="pt",
    max_length=model_args.model_max_length,
)
# print("training_args: ", training_args)

if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        training_args.bf16 = True
        training_args.fp16 = False
    else:
        training_args.bf16 = False
        training_args.fp16 = True
else:
    training_args.bf16 = False

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds_tokenized,
    eval_dataset=test_ds_tokenized,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

if __name__ == "__main__":
    train()
