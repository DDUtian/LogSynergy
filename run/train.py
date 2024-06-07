from collections import Counter
import warnings
warnings.filterwarnings("ignore")
import sys
import os
print("当前进程ID为:", os.getpid())
path = os.path.abspath(".")
sys.path.append(path)
import click
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from MTLog.dataset import LogAdaptationDataset
from MTLog.model import BERTForTransferLogClassification, BERTForTransferLogClassificationWithDisentanglement
from MTLog.trainer import IDFAdaptationTrainer
from transformers import TrainingArguments, BertConfig
import torch
from torch.utils.data import Dataset

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


def config_init(source_domain, target_domain):

    def get_domains(domain):

        domain_list = domain[1:-1].split(',')
        return domain_list
    path_prefix = "LogSynergy"

    if not os.path.exists(f"model/{path_prefix}"):
        os.mkdir(f"model/{path_prefix}")
    model_path = os.path.join(f"model/{path_prefix}", f"{source_domain}_{target_domain}_detection")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    source_domain_list = get_domains(source_domain)
    return model_path, source_domain_list

def get_train_log_dataset(domain: str, path: str, seq_lenth=10, val_ratio=0.1, lenth=50000, target_domain=False, model_path=None) -> tuple:
    log_ids = np.load(os.path.join(path, domain + ".npy"))
    log_ids = log_ids[:int(len(log_ids))]
    log_labels = np.load(os.path.join(path, domain + "_label.npy"))
    log_labels = log_labels[:int(len(log_labels))]
    log_embeddings = np.load(os.path.join(path, domain + "_bert_embedding.npy"))

    perm_idx = torch.randperm(int(len(log_labels) * (1 - val_ratio)))[:lenth]
    if target_domain:
        perm_idx = torch.arange(lenth)
        np.save(os.path.join(model_path, f"{target_domain}.npy"), perm_idx)

    if not target_domain:
        train_dataset = RandomSamplerLogSequenceDataset(log_ids[:int(len(log_labels) * (1 - val_ratio))], log_embeddings, log_labels[:int(len(log_labels) * (1 - val_ratio))], seq_lenth, perm_idx, lenth=lenth, target_domain=target_domain)
    else:
        train_dataset = RandomSamplerLogSequenceDataset(log_ids[:len(log_ids)], log_embeddings, log_labels[:len(log_labels)], seq_lenth, perm_idx, lenth=lenth, target_domain=target_domain)


    return train_dataset

class RandomSamplerLogSequenceDataset(Dataset):

    def __init__(self, ids, embedding, labels, seq_lenth, perm_idx, lenth=None, noise=None, eval=False, target_domain=False):
        self.ids = ids
        self.embedding = torch.tensor(embedding)
        self.embedding_dim = len(embedding[0])
        self.labels = labels
        self.seq_lenth = seq_lenth
        self.lenth = lenth
        self.perm_idx = perm_idx
        self.noise = noise
        self.target_domain = target_domain
        if target_domain:
            print(f"Anomaly count is {np.sum(self.labels[self.perm_idx])}")
        if self.target_domain and eval is False:
            self.target_labels = self.labels[self.perm_idx]
            print(Counter(self.target_labels))

    def get_sequence(self, raw_idx):
        idx = self.perm_idx[raw_idx]
        seq = self.ids[idx]
        seq_embedding = torch.zeros([self.seq_lenth, self.embedding_dim])
        for i in range(self.seq_lenth):
            seq_embedding[i] = self.embedding[seq[i]]
        if self.target_domain:
            label = self.target_labels[raw_idx]
        else:
            label = self.labels[idx]
        return seq_embedding, label

    def __getitem__(self, idx):
        if hasattr(idx, "__iter__"):
            seq_embeddings = torch.zeros([len(idx), self.seq_lenth, self.embedding_dim])
            labels = torch.zeros([len(idx)], dtype=torch.int32)
            for i, index in enumerate(idx):
                seq_embedding, label = self.get_sequence(index)
                labels[i] = label
                seq_embeddings[i] = seq_embedding
            return seq_embeddings, labels
        else:
            seq_embedding, label = self.get_sequence(idx)
            return seq_embedding, label

    def __len__(self):
        return self.lenth

@click.command()
@click.argument('source_domain')
@click.argument('target_domain')
@click.argument('path_prefix', type=click.Path(exists=True))
@click.option('--batch_size', type=int, default=64, help='Batch size.')
@click.option('--learning_rate', type=float, default=0.0001, help='Weight of transfer loss.')
@click.option('--num_workers', type=int, default=16, help='Number of workers for loading data.')
@click.option('--seq_lenth', type=int, default=10, help='Length of sequence.')
@click.option('--num_train_epochs', type=int, default=10, help='Number of epoch for training.')
def main(source_domain, target_domain, path_prefix, batch_size, learning_rate, num_workers, seq_lenth, num_train_epochs):

    source_train_lenth = 50000
    target_train_lenth = 5000

    # model config
    device = "cuda:0"
    model_path, source_domain_list = config_init(source_domain, target_domain)
    
    net = BERTForTransferLogClassificationWithDisentanglement(BertConfig(pad_token_id=0,
                                        num_hidden_layers=6, sclassifier_dropout=0.1, num_labels=2,
                                        layer_norm_eps=1e-6, intermediate_size=2048), use_entry=0)
    print("Model Path:", model_path)

    # load dataset
    source_train_dataset_list = []
    for source_domain in source_domain_list:
        source_train_dataset = get_train_log_dataset(source_domain, path_prefix, lenth=source_train_lenth, seq_lenth=seq_lenth)
        source_train_dataset_list.append(source_train_dataset)

    target_train_dataset = get_train_log_dataset(target_domain, path_prefix, lenth=target_train_lenth, seq_lenth=seq_lenth, target_domain=True, model_path=model_path)

    adaptation_train_dataset = LogAdaptationDataset(source_train_dataset_list, target_train_dataset, full_lenth=True)

    training_args = TrainingArguments(model_path, dataloader_num_workers=num_workers,
                                evaluation_strategy="no",
                                save_strategy="epoch",
                                prediction_loss_only=True,
                                logging_first_step=True,
                                logging_steps=50,
                                learning_rate=learning_rate,
                                num_train_epochs=num_train_epochs,
                                warmup_ratio=0.2,
                                fp16=True,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size, log_level='error',
                                label_names=["source_labels"])

    trainer = IDFAdaptationTrainer(
        domain_num=len(source_domain_list),
        seq_lenth=seq_lenth,
        model=net,
        args=training_args,
        train_dataset=adaptation_train_dataset,
        tokenizer=None,
        data_collator = None,
        compute_metrics=None,
    )
    trainer.train()

if __name__ == '__main__':
    main()