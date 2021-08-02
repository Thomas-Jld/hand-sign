from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from utils import onehot


class SignCollection(Dataset):
    def __init__(self, signs_data: List[Dict]) -> None:
        super(SignCollection, self).__init__()
        self.data = signs_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.Tensor]:
        id, data = self.data[idx]["id"], self.data[idx]["data"]
        return torch.FloatTensor(onehot(id)), torch.tensor(data)


def load(path):
    result = []
    with open(path, "r") as f:
        file_data = f.readlines()
        for line in file_data:
            line = line.strip("\n").split(";")
            index = int(line[0])
            data = []
            r_data = []
            for pair in line[1:]:
                pair = pair.split(',')
                data.append([float(pair[0]), float(pair[1])])
                r_data.append([1-float(pair[0]), float(pair[1])])
            result.append({
                "id": index,
                "data": data
            })
            result.append({
                "id": index,
                "data": r_data
            })
    return result
