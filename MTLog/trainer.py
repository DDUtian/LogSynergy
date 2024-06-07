import sys
import os

path = os.path.abspath(".")
sys.path.append(path)
from transfer_losses import TransferLoss
from transformers import Trainer
from loss_funcs.adv import *
from torch import nn
import torch.nn.functional as F


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class DomainDiscriminator(nn.Module):

    def __init__(self, num_class, input_dim=32):
        super(DomainDiscriminator, self).__init__()

        self.out = nn.Sequential(
            nn.Linear(32, num_class),
            # nn.Softmax(dim=1)
        )
 
    def forward(self, x):
        x = self.out(x)
        return x

class IDFAdaptationTrainer(Trainer):

    def __init__(self, domain_num, seq_lenth, transfer_weight=0.01, dim=768, **kwargs):
        super().__init__(**kwargs)
        self.domain_num = domain_num
        self.seq_lenth = seq_lenth
        self.dim = dim
        discriminator_input_dim = 32
        self.domain_discriminator = DomainDiscriminator(self.domain_num + 1).to(kwargs['model'].device)
        self.transfer_weight = transfer_weight
        self.mutual_info = CLUB(32, 32, 32).to(kwargs['model'].device)
        self.loss_weight = torch.tensor([0.25, 0.75]).to("cuda")
        max_iter = kwargs['args'].max_steps
        transfer_loss_args = {
            "loss_type": "daan", 
            "max_iter": max_iter, 
            "discriminator_input_dim": discriminator_input_dim,
            "num_class": 2
        }
        self.adapt_loss_func = TransferLoss(**transfer_loss_args).to(kwargs['model'].device)

    def daan_adaptation_loss(self, 
                        source_feature, source_logits,
                        target_feature, target_logits):
        kwargs = {}
        kwargs['source_logits'] = source_logits
        kwargs['target_logits'] = target_logits
        transfer_loss = self.adapt_loss_func(source_feature, target_feature, **kwargs)
        return transfer_loss

    def classification_loss(self, logits, labels, loss_type="CrossEntropy"):
        if loss_type == "CrossEntropy":
            loss_fct = nn.CrossEntropyLoss(weight=self.loss_weight)
            clf_loss = loss_fct(logits.view(-1, 2), labels.view(-1).long())
        else:
            targets = torch.zeros((len(labels), 2))
            targets[labels==1, 1] = 1
            targets[labels!=1, 0] = 1
            loss_fct = nn.BCEWithLogitsLoss()
            targets = targets.to(logits.device)
            clf_loss = loss_fct(logits, targets)
        return clf_loss

    def domain_loss(self, source_feature, source_domains):
        loss_fn = nn.CrossEntropyLoss()
        discriminator_logits = self.domain_discriminator(source_feature)
        loss = loss_fn(discriminator_logits, source_domains.long())
        return loss

    def compute_loss(self, model, samples, return_outputs=False):
        source_input_embeds = samples["source_input_embeds"].to(model.device)
        source_labels = samples["source_labels"].to(model.device).to(torch.int32)
        source_domains = samples["source_domains"].to(model.device)
        target_input_embeds = samples["target_input_embeds"].to(model.device)

        source_input_embeds = source_input_embeds.reshape(-1, self.seq_lenth, self.dim)
        source_labels = source_labels.reshape(-1).to(torch.int32)
        source_domains = source_domains.reshape(-1)

        source_all_feature, source_logits = model(source_input_embeds)
        target_all_feature, target_logits = model(target_input_embeds)
        
        source_feature = source_all_feature[:, :32]
        target_feature = target_all_feature[:, :32]
        
        all_domain_feature = torch.cat([source_all_feature[:, 32:], target_all_feature[:, 32:]], dim=0)
        all_domains = torch.cat([source_domains, torch.ones(len(target_input_embeds), dtype=torch.int64).to(source_domains.device) * self.domain_num])
        
        all_clf_feature = torch.cat([source_feature, target_feature], dim=0)

        target_labels = samples["target_labels"]
        if len(target_labels[target_labels!=-1]) > 0:
            clf_loss = self.classification_loss(source_logits, source_labels, "CrossEntropy") + self.classification_loss(target_logits[target_labels!=-1], target_labels[target_labels!=-1], "CrossEntropy")
        else:
            clf_loss = self.classification_loss(source_logits, source_labels, "CrossEntropy")

        # MI for feature disentangle
        domain_loss = self.domain_loss(all_domain_feature, all_domains)
        feature_mi_loss = self.mutual_info(all_clf_feature, all_domain_feature)

        daan_loss = self.daan_adaptation_loss(source_feature, source_logits, target_feature, target_logits)
        loss = clf_loss + feature_mi_loss * self.transfer_weight + domain_loss + daan_loss * self.transfer_weight
        
        if self.state.global_step % 20 == 0:
            # print("Classification Loss:", clf_loss, " Domain Loss:", domain_loss, " Disentangle Loss:", feature_mi_loss, " Adaptation Loss:", adp_loss)
            print("Classification Loss:", clf_loss, "Domain Loss:", domain_loss, " Disentangle Loss:", feature_mi_loss)

        if return_outputs:
            return loss, (source_feature, source_logits)
        return loss

    def extract_features(self, dataset):
        features = []
        for item in dataset:
            input_embeds = item["input_embeds"].to(self.model.device)
            # 使用模型提取特征
            feature = self.model.extract_feature(input_embeds)
            features.append(feature)
        return torch.cat(features, dim=0)