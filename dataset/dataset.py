from torch.utils.data import Dataset
from typing import cast, Generic, Iterable, TypeVar  # noqa: UP035
import csv


_T_co = TypeVar("_T_co", covariant=True)


class Wiki103Dataset(Dataset):
    def __init__(self,file_name = "wiki.train.tokens.processed"):
        super().__init__()
        with open(file_name, 'r' , encoding='utf-8') as f:
            data = f.read().split("<end>")
            data = [line.strip() for line in data]
            self.data = [line + "<eos>" for line in data if len(line) > 0]
    
    def __getitem__(self, index) -> _T_co:
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class ProcessedEssayDataset(Dataset):
    def __init__(self,filename = "./processed_essay.csv"):
        super().__init__()
        with open(filename, 'r',encoding='utf-8') as f:
            m = {
                "low" : 0,
                "high" : 1
            }
            data = csv.reader(f)
            data = list(data)
            self.labels = data.pop(0)
            self.text = [row[0] for row in data]
            self.data = [tuple([m[i] for i in row[1:]]) for row in data]
    def __getitem__(self, index):
        return self.text[index], self.data[index]
    
    def __len__(self):
        return len(self.data)
