from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod





class IMDBDataset(Dataset):
    def __init__(self, dataset_name):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


if __name__ == "__main__":
    from datasets import load_dataset

    imdb_ds = load_dataset("stanfordnlp/imdb")
    print(imdb_ds)
