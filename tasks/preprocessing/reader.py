from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Literal, Tuple
from text import TextProcessing
from dataclasses import dataclass


@dataclass
class LabelMapping:
    classes: List[str] = None
    class2id: Dict[str, int] = None
    id2class: Dict[int, str] = None


class DataReaderBase(ABC):

    def __init__(self,
                 text_processor: TextProcessing,
                 problem_type: Literal["single_label_classification", "multi_label_classification"]):
        self.text_processor = text_processor
        self.problem_type = problem_type

    @abstractmethod
    def load_from_disk(self, path: str, **kwargs):
        raise NotImplementedError


class ExcelReader(DataReaderBase):
    """Concrete class to read and filter data from an Excel file."""

    def load_from_disk(self, path: str, **kwargs) -> Tuple[List[Dict[str, str]], LabelMapping]:
        """
        Reads an Excel file and filters rows based on a minimum label occurrence threshold.

        Parameters:
            path (str): Path to the Excel file.
            sheet_name (str, optional): Excel sheet name to read.
            threshold (int, optional): Minimum occurrences for labels to be included.
            usecols (list of str): List of two column names to use (text and label).

        Returns:
            pd.DataFrame: DataFrame containing text and label columns, filtered by label threshold.
        """
        # Extract and validate parameters
        sheet_name = kwargs.get("sheet_name")
        threshold = kwargs.get("threshold", 10)  # Default threshold is 1 if not provided
        usecols = kwargs.get("usecols")

        self._validate_usecols(usecols)

        # Load and rename columns
        df = pd.read_excel(path, sheet_name=sheet_name, usecols=usecols)
        df.columns = ["text", "label"]

        # process label
        df["label"] = df["label"].str.strip().str.lower()

        # Filter by threshold
        df = self._filter_by_threshold(df, threshold)

        # Preprocess the text column using the TextProcessing instance
        df["text"] = df["text"].apply(self.text_processor.preprocess)

        classes = list(df['label'].unique())
        class2id = {class_: _id for _id, class_ in enumerate(classes)}
        id2class = {_id: class_ for class_, _id in class2id.items()}

        if self.problem_type == "single_label_classification":
            df['label'] = df['label'].map(class2id)

        elif self.problem_type == "multi_label_classification":
            df = df.groupby('text')['label'].agg(lambda x: list(x)).reset_index()
            # Create a binary representation column for each class
            df['label'] = df['label'].apply(
                lambda labels: [1 if class_ in labels else 0 for class_ in classes])

        else:
            raise ValueError(f"Don't support {self.problem_type}")
        label_mapping = LabelMapping(classes=classes, id2class=id2class, class2id=class2id)
        return df.to_dict("records"), label_mapping

    @staticmethod
    def _validate_usecols(usecols):
        """Validates that usecols contains exactly two columns."""
        if not usecols or len(usecols) != 2:
            raise ValueError("usecols must contain exactly two columns.")

    @staticmethod
    def _filter_by_threshold(df, threshold):
        """
        Filters DataFrame to only include labels with count >= threshold.

        Parameters:
            df (pd.DataFrame): DataFrame with text and label columns.
            threshold (int): Minimum occurrences required for labels to be included.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        label_counts = df["label"].value_counts()
        valid_labels = label_counts[label_counts >= threshold].index
        return df[df["label"].isin(valid_labels)]


if __name__ == '__main__':
    from text import TextProcessing

    excel = ExcelReader(text_processor=TextProcessing(),
                        problem_type="single_label_classification")
    records, labels_info = excel.load_from_disk("../../dataset/gam_dataset.xlsx",
                                                sheet_name="20240621_intention",
                                                usecols=["Text", "Intents"])
    # print(records[0: 10])
    print(labels_info)
