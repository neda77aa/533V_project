import torch
import pickle
import torch.nn.functional as F

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = []
        self._preprocess_data(data_path)

    def _preprocess_data(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # Returns a Dictionary in which the tensor of state and one hot encoded of label
        item ={'state':torch.tensor(item[0]),'action': F.one_hot(torch.tensor(item[1]),num_classes=2)}
        return item
