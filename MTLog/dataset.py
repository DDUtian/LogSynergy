from collections import Counter
import torch
import numpy as np
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

class LogAdaptationDataset(Dataset):

    def __init__(self, source_dataset_list, target_dataset, full_lenth=True):
        self.source_dataset_list = source_dataset_list
        self.target_dataset = target_dataset
        if full_lenth == False:
            self.lenth = len(self.target_dataset)
        else:
            lenth_list = [len(dataset) for dataset in source_dataset_list] + [len(target_dataset)]
            self.lenth = max(lenth_list)

    def __getitem__(self, idx):
        source_input_embeds = []
        source_labels = []
        source_domains = []
        for domain_index, source_dataset in enumerate(self.source_dataset_list):
            embedding, label = source_dataset[idx % len(source_dataset)]
            source_input_embeds.append(embedding)
            source_labels.append(label)
            source_domains.append(domain_index)
        source_input_embeds = torch.stack(source_input_embeds)
        source_labels = torch.Tensor(np.array(source_labels))
        source_domains = torch.Tensor(np.array(source_domains))

        target_input_embeds, target_labels = self.target_dataset[idx % len(self.target_dataset)]
        return {"source_input_embeds": source_input_embeds, "source_labels": source_labels, "source_domains": source_domains, "target_input_embeds": target_input_embeds, "target_labels": target_labels}

    def __len__(self):
        return self.lenth
    
