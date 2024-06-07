# Bridging the Gap: LLM-Powered Multi-Source Transfer Learning for Log Anomaly Detection in New Software Systems

## Abstract
For large IT companies, maintaining multiple software systems presents considerable complexity. Logs are invaluable for depicting the state of these systems, making log-based anomaly detection crucial for ensuring system reliability. Current log anomaly detection methods typically require extensive log data to train unsupervised models or numerous labeled anomalies for supervised models, limiting their rapid applicability to newly deployed software systems.This paper proposes LogSynergy, a novel framework for log anomaly detection leveraging multi-source transfer learning. LogSynergy facilitates the transfer of anomaly detection knowledge from logs across multiple mature systems, enabling efficient deployment with minimal logs from new systems. We propose an LLM-based event interpretation method to minimize log style discrepancies among different systems. Moreover, to extract transferable features across various systems, we present a feature disentanglement method, enhancing the generalizability of the detection process. Evaluated on three real-world log datasets from a Top-tier global IT company and three public datasets, LogSynergy demonstrates superior performance compared to existing methods.

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
pyhton run/test.py
```
