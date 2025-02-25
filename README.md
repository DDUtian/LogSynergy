# Bridging the Gap: LLM-Powered Multi-Source Transfer Learning for Log Anomaly Detection in New Software Systems

## Abstract
For large IT companies, maintaining numerous software systems presents considerable complexity. Logs are invaluable for depicting the state of systems, making log-based anomaly detection crucial for ensuring system reliability. Existing methods require extensive log data for training, hindering their rapid deployment for new systems. Cross-system log anomaly detection methods attempt to transfer knowledge from mature systems to new ones but often struggle with syntax differences and system-specific knowledge, which hinders their effectiveness. To address these issues, this paper proposes LogSynergy, a novel transfer learning-based log anomaly detection framework. LogSynergy employs (1) LLM-based event interpretation (LEI) to standardize log syntax across different systems, and (2) system-unified feature extraction (SUFE) to disentangle system-specific features from system-unified features. These bridge the gap among different systems and enhance LogSynergy's generalizability. LogSynergy has been deployed in the production environment of a top-tier global Internet Service Provider (ISP), where it was evaluated on three real-world datasets. Additionally, we conducted evaluations on three public datasets. The results demonstrate that LogSynergy significantly outperforms existing methods. It achieves F1-scores over 89% on the real-world datasets and over 83% on the public datasets, using only 5000 labeled log sequences from the new system. These results underscore LogSynergy's effectiveness in rapidly deploying anomaly detection models for new systems.

## Project Structure
```
├─run         # LogSynergy main entrance.
├─scripts     # Configuration for train.
├─MTLog       # Trainer, dataset and model of LogSynergy.           
└─loss_funcs  # Loss function of domain adaptation used by LogSynergy.
```

## Data Preparation
To run LogSynergy, using BGL data as an example, the following data preparation steps are required:

- **1:** BGL.npy: The indices of the data, with shape [Number of Log Sequences, Length of Log Sequence]
- **2:** BGL_label.npy: The labels of the data, with shape [Number of Log Sequences]
- **3:** BGL_embedding.npy: The feature representations of the data, with shape [Number of Log Templates, Embedding Dimension]

## Running and Testing

### Configuration File
The configuration files used to set up the hyperparameters for training and testing at `/scripts`.
```shell
source <config_file>
# where `<config_file>` is the path to the configuration file.
```
Main parameters are described as follows:
- `SOURCE_DOMAIN`: a list of source datasets to be used, e.g., [Thunderbird,Spirit]
- `TARGET_DOMAIN`: the name of the target dataset
- `DATA_START`: the starting index for the data
- `BATCH_SIZE`: the number of samples per batch
- `EVAL_EPOCH`: the frequency of evaluation
- `TRANSFER_WEIGHT`: the weight of the transfer learning component
- `STEP`: the interval step for certain operations

### To run and test the code
```shell
python run/train.py
pyhton run/eval.py
```
