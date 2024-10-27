from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Model name or path to pretrained model"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    padding_side: Optional[str] = field(
        default="left",
        metadata={
            "help": "Padding side is left or right. "
                    "If using flash_attention then padding_side must be 'left'."
        },
    )
    model_max_length: Optional[int] = field(
        default=256, metadata={"help": "Maximum sequence length"}
    )

    flash_attention: Optional[bool] = field(
        default=False, metadata={"help": "Use flash_attention to train"}
    )

    cache_dir: Optional[str] = field(
        default="./cache",
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )

    use_focal_loss: Optional[bool] = field(
        default=True, metadata={"help": "Use focal loss"}
    )

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path

        if self.padding_side == "right" and self.flash_attention is True:
            raise ValueError("When using flash_attention, padding_side must be 'left'")


@dataclass
class DataTrainingArguments:
    data_sheet: str = field(
        default=...,
        metadata={"help": "data training sheet name"},
    )

    data_columns: str = field(
        default=...,
        metadata={"help": "data columns"},
    )

    data_skiprows: Optional[int] = field(
        default=None,
        metadata={"help": "data skiprows"},
    )

    label_sheet: str = field(
        default=...,
        metadata={"help": "data training sheet name"},
    )
    label_column: str = field(
        default=...,
        metadata={"help": "label column"},
    )

    label_skiprows: Optional[int] = field(
        default=None,
        metadata={"help": "data skiprows"},
    )

    train_file_or_dir: str = field(
        default=...,
        metadata={
            "help": "File or directory of the training data. "
                    "Using comma for multi files."
        },
    )
    validation_file_or_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "File or directory of the validation data. "
                    "Using comma for multi files."
        },
    )
    test_file_or_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "File or directory of the test data. "
                    "Using comma for multi files."
        },
    )

    validation_split_percentage: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, "
                "truncate the number of evaluation examples to this value if set."
            )
        },
    )

    shuffle: Optional[bool] = field(
        default=True, metadata={"help": "Whether to shuffle data before training."}
    )

    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store dataset after preprocessing"},
    )

    load_from_cache: Optional[bool] = field(
        default=False, metadata={"help": "Whether to load dataset from cache"}
    )
    label_threshold: Optional[int] = field(default=0,
                                           metadata={"help": "Only get number of labels greater than this value"})

    def __post_init__(self):
        if self.train_file_or_dir is None and self.validation_file_or_dir is None:
            raise ValueError("Need either a training or validation file.")


@dataclass
class TrainingArguments(TrainingArguments):
    should_log: Optional[bool] = field(default=True, metadata={"help": "Debug mode."})