import torch
from transformers.models.bert.modeling_bert import BertEncoder, BertPreTrainedModel, BertModel, BertConfig, BertAttention
from transformers.models.ctrl.modeling_ctrl import MultiHeadAttention
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import ModelOutput
from typing import List, Optional, Tuple, Union
from torch.nn import MSELoss, NLLLoss
from torch import nn
from transfer_losses import TransferLoss
from transformers import Trainer
# from transformers.models.distilbert.modeling_distilbert import BertForSequenceClassification, BertModel
from transformers import DistilBertConfig, DistilBertModel, DistilBertForSequenceClassification
from transformers import AutoTokenizer


# class EntrySequenceAttention(nn.Module):

#     def __init__(self, config:BertConfig):
#         self.bert_attention = BertAttention(config)

#     def forward(self, input_embeds:torch.Tensor, sequence_attention:torch.Tensor):
#         seqence_attention_max_index

class Classifier(nn.Module):

    def __init__(self, config, hidden_dim=256, feature_dim=32, domain_classifier=False):
        super().__init__()
        self.domain_classifier = domain_classifier
        self.sep_feature_dim = int(feature_dim / 2)
        # self.linear_1 = nn.Linear(config.hidden_size, 32)
        self.linear_1 = nn.Sequential(
            nn.Linear(config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.relu = nn.ReLU()
        if not self.domain_classifier:
            self.linear_2 = nn.Linear(feature_dim, 2)
        else:
            self.linear_2 = nn.Linear(int(feature_dim / 2), 2)
        # self.output = nn.Softmax(dim=-1)

    def forward(self, x:torch.Tensor, **kwargs):
        # x = self.dropout(x)
        x = self.linear_1(x)
        # x = self.relu(x)
        raw_feature = self.dropout(x)
        if self.domain_classifier:
            feature = raw_feature[:, :self.sep_feature_dim]
        else:
            feature = raw_feature
        x = self.linear_2(feature)
        # x = self.output(x)
        return x, raw_feature


class BERTForTransferLogClassification(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, use_entry=False):
        super().__init__(config)
        self.use_entry = use_entry
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.classifier = Classifier(config)
        self.dropout = nn.Dropout(0.2)
        self.post_init()
        if self.use_entry:
            self.seq_entry_attention = MultiHeadAttention(config.hidden_size, config.num_attention_heads)

    def forward(
        self,
        input_embeds: Optional[torch.FloatTensor] = None,
    ):
        bert_outputs = self.bert(
            inputs_embeds=input_embeds,
            output_attentions=True
        )
        
        pooled_output = bert_outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        if self.use_entry:
            # pooled_output = pooled_output * 0.5 + input_embeds[:, -1] * 0.5
            seq_entry_tensor = torch.stack([pooled_output, input_embeds[:, -1]], dim=1)
            attn_output = self.seq_entry_attention(
                seq_entry_tensor,
                seq_entry_tensor,
                seq_entry_tensor,
                mask=None
            )[0]
            pooled_output = torch.sum(attn_output, dim=1)

        logits, features = self.classifier(pooled_output)

        return features, logits
    
class BERTForTransferLogClassificationWithDisentanglement(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, use_entry=False):
        super().__init__(config)
        self.use_entry = use_entry
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.classifier = Classifier(config, feature_dim=64, domain_classifier=True)
        self.dropout = nn.Dropout(0.2)
        self.post_init()
        if self.use_entry:
            self.seq_entry_attention = MultiHeadAttention(config.hidden_size, config.num_attention_heads)

    def forward(
        self,
        input_embeds: Optional[torch.FloatTensor] = None,
    ):
        bert_outputs = self.bert(
            inputs_embeds=input_embeds,
            output_attentions=True
        )
        
        pooled_output = bert_outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        if self.use_entry:
            # pooled_output = pooled_output * 0.5 + input_embeds[:, -1] * 0.5
            seq_entry_tensor = torch.stack([pooled_output, input_embeds[:, -1]], dim=1)
            attn_output = self.seq_entry_attention(
                seq_entry_tensor,
                seq_entry_tensor,
                seq_entry_tensor,
                mask=None
            )[0]
            pooled_output = torch.sum(attn_output, dim=1)

        logits, features = self.classifier(pooled_output)

        return features, logits
    
class DimReduction(nn.Module):

    def __init__(self, config, hidden_dim=256, feature_dim=64):
        super().__init__()
        # self.linear_1 = nn.Linear(config.hidden_size, 32)
        self.linear_1 = nn.Sequential(
            nn.Linear(config.hidden_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.relu = nn.LeakyReLU()

    def forward(self, x:torch.Tensor, **kwargs):
        x = self.dropout(x)
        x = self.linear_1(x)
        feature = self.relu(x)
        # feature = self.dropout(x)
        return feature 
    
class SAD_BERTForTransferLogClassification(BERTForTransferLogClassificationWithDisentanglement):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, use_entry=False):
        super().__init__(config)
        self.use_entry = use_entry
        self.num_labels = config.num_labels
        self.config = config
        self.feature_dim = 64
        self.bert = BertModel(config)
        self.classifier = Classifier(config, feature_dim=64, domain_classifier=True)
        self.dropout = nn.Dropout(0.2)
        self.post_init()
        self.c = None
        if self.use_entry:
            self.seq_entry_attention = MultiHeadAttention(config.hidden_size, config.num_attention_heads)

    def set_center(self, c):
        self.c = c
    
    def predict(self,
                input_embeds: Optional[torch.FloatTensor] = None,
    ):
        bert_outputs = self.bert(
            inputs_embeds=input_embeds,
            output_attentions=True
        )
        
        pooled_output = bert_outputs.pooler_output

    def forward(
        self,
        input_embeds: Optional[torch.FloatTensor] = None,
    ):
        bert_outputs = self.bert(
            inputs_embeds=input_embeds,
            output_attentions=True
        )
        
        pooled_output = bert_outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        if self.use_entry:
            # pooled_output = pooled_output * 0.5 + input_embeds[:, -1] * 0.5
            seq_entry_tensor = torch.stack([pooled_output, input_embeds[:, -1]], dim=1)
            attn_output = self.seq_entry_attention(
                seq_entry_tensor,
                seq_entry_tensor,
                seq_entry_tensor,
                mask=None
            )[0]
            pooled_output = torch.sum(attn_output, dim=1)

        _, features = self.classifier(pooled_output)
        if self.c is None:
            return features[:, :int(self.feature_dim / 2)]
        distances = torch.sum((features[:, :int(self.feature_dim / 2)] - self.c) ** 2, dim=1)
        return features, distances
    
class LogSentenceClassficationWithDisentanglement(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, use_entry=False):
        super().__init__(config)
        self.use_entry = use_entry
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.classifier = Classifier(config, feature_dim=64, domain_classifier=True)
        self.dropout = nn.Dropout(0.2)
        self.post_init()

    def forward(self,
        input_ids,
        attention_mask
    ):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        pooled_output = bert_outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        logits, features = self.classifier(pooled_output)

        return features, logits