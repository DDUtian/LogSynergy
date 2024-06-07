import warnings
warnings.filterwarnings("ignore")
import sys
import os
path = os.path.abspath(".")
sys.path.append(path)
import click
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from MTLog.model import BERTForTransferLogClassificationWithDisentanglement


def get_checkpoint_path_list(pretrained_model_path):

    checkpoint_list = [int(name[11:]) for name in os.listdir(pretrained_model_path) if name.startswith('checkpoint-')]
    checkpoint_list.sort()
    checkpoint_path_list = [os.path.join(pretrained_model_path, "checkpoint-" + str(checkpoint)) for checkpoint in checkpoint_list]
    print(pretrained_model_path)
    return checkpoint_path_list

def config_init(source_domain, target_domain):

    path_prefix = "LogSynergy"

    model_path = os.path.join(f"model/{path_prefix}", f"{source_domain}_{target_domain}_detection")
    return model_path

def get_log_eval_dataset(domain: str, path: str, model_path:str, seq_lenth=20, step=1) -> tuple:

    log_ids = np.load(os.path.join(path, domain + ".npy"))
    log_ids = log_ids[:int(len(log_ids))]
    log_labels = np.load(os.path.join(path, domain + "_label.npy"))
    log_labels = log_labels[:int(len(log_labels))]
    log_embeddings = np.load(os.path.join(path, domain + "_bert_embedding.npy"))

    train_index = np.load(os.path.join(model_path, domain + ".npy"))
    log_ids = log_ids[~np.isin(np.arange(len(log_ids)), train_index)]
    log_labels = log_labels[~np.isin(np.arange(len(log_labels)), train_index)]

    return log_ids, log_labels, log_embeddings


@click.command()
@click.argument('source_domain')
@click.argument('target_domain')
@click.argument('path_prefix', type=click.Path(exists=True))
@click.option('--batch_size', type=int, default=64, help='Batch size.')
@click.option('--num_workers', type=int, default=10, help='Number of workers for loading data.')
@click.option('--seq_lenth', type=int, default=20, help='Length of sequence.')
@click.option('--eval_epoch', type=int, default=10, help='Number of epoch for evaluation.')
def main(source_domain, target_domain, path_prefix, batch_size, num_workers, seq_lenth, eval_epoch):
    device = "cuda"
    model_path = config_init(source_domain, target_domain)
    net = BERTForTransferLogClassificationWithDisentanglement.from_pretrained(
                                            get_checkpoint_path_list(model_path)[eval_epoch - 1], use_entry=0).to(device)
    # load dataset
    log_ids, log_labels, log_embeddings = get_log_eval_dataset(target_domain, path_prefix, model_path=model_path, seq_lenth=seq_lenth)
    label_index = np.where(log_labels==1)[0]
    print(len(label_index))
    eval_dataloader = DataLoader(log_ids, batch_size=4096, shuffle=False)
    
    # predict
    output_tensor = torch.zeros([len(log_ids), 2]).to(device)

    with torch.no_grad():
        for index, (id) in enumerate(tqdm(eval_dataloader)):
            seq_embedding = torch.tensor(log_embeddings[id])
            output = net(input_embeds=seq_embedding.to(device))[1]
            output = torch.softmax(output, dim=-1)
            output_tensor[index * eval_dataloader.batch_size: index * eval_dataloader.batch_size + len(output)] = output
    output_array = np.array(output_tensor.to("cpu"))
    raw_predict = (output_array[:,1] > output_array[:,0]).astype(np.int32)
    ground_truth_array = log_labels

    print(np.sum(ground_truth_array))
    fp_index = np.where((raw_predict == 1) & (ground_truth_array == 0))[0]
    fn_index = np.where((raw_predict == 0) & (ground_truth_array == 1))[0]
    tp_index = np.where((raw_predict == 1) & (ground_truth_array == 1))[0]
    np.save(f"{target_domain}_debug_fp.npy", log_ids[fp_index])
    np.save(f"{target_domain}_debug_fp_prob.npy", output_array[fp_index, 1])
    np.save(f"{target_domain}_debug_fn.npy", log_ids[fn_index])
    np.save(f"{target_domain}_debug_fn_prob.npy", output_array[fn_index, 1])
    np.save(f"{target_domain}_debug_tp.npy", log_ids[tp_index])
    np.save(f"{target_domain}_debug_tp_prob.npy", output_array[tp_index, 1])

    raw_f1 = f1_score(ground_truth_array, raw_predict)
    raw_precision = precision_score(ground_truth_array, raw_predict)
    print(classification_report(ground_truth_array, raw_predict))
    raw_recall = recall_score(ground_truth_array, raw_predict)

    print("F1-Score:", raw_f1)
    print("Precision:", raw_precision, "Recall:", raw_recall)

    print(confusion_matrix(ground_truth_array, raw_predict))



if __name__ == '__main__':
    main()