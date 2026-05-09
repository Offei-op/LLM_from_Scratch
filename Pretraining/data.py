import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import numpy as np


class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,maxlength,stride=1):
        super().__init__()
        token_ids= tokenizer.encode(txt)
        self.input_ids = []
        self.output_ids = []

        for i in range(0,len(token_ids)-maxlength,stride):
            self.input_ids.append(torch.tensor(token_ids[i:i+ maxlength]))
            self.output_ids.append(torch.tensor(token_ids[i+1:i+ maxlength+1]))

    
    def __getitem__(self, index):
        return self.input_ids[index], self.output_ids[index]
    
    def __len__(self):
        return len(self.output_ids)
    



def create_dataloader_v1(txt,max_length=256,stride=128,batch_size=4,num_workers=0,shuffle=True,drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt,tokenizer,max_length,stride)
    dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dl


class DatasetGPT(Dataset):
    def __init__(self, arr, max_length, stride=1):
        """
        arr: a numpy memmap (or array) of token ids, dtype=uint16
        max_length: context window length (e.g. 256)
        stride: step between consecutive windows (e.g. 128 for 50% overlap,
                or max_length for non-overlapping)
        """
        super().__init__()
        self.data = arr
        self.max_length = max_length
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.max_length - 1) // self.stride + 1

    def __getitem__(self, index):
        idx = index * self.stride
        x = self.data[idx : idx + self.max_length]
        y = self.data[idx + 1 : idx + self.max_length + 1]
        # uint16 → int64 conversion is required for PyTorch embeddings
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        return x, y
    

def create_dataloader_v2(arr, max_length=256, stride=128, batch_size=4,
                          num_workers=0, shuffle=True, drop_last=True):
    dataset = DatasetGPT(arr, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,    # speeds up CPU→GPU transfer
    )