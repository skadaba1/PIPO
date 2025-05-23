# 6. Protein dataset class
from torch.utils.data import Dataset
class ProteinDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoding = self.tokenizer(seq,
                                return_tensors="pt",
                                padding="max_length",
                                max_length=self.max_length,
                                truncation=True)

        # Remove batch dimension added by tokenizer
        return {k: v.squeeze(0) for k, v in encoding.items()}