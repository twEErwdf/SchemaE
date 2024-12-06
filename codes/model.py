#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from dataloader import TestDataset
# import pickle as pkl
# import os
from earl_model import ConRelEncoder


# from sklearn.metrics import f1_score, roc_auc_score


class KGEModel(nn.Module):
    def __init__(self, args, model_name, nentity, nrelation, hidden_dim, gamma, train_g_bidir,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.train_g_bidir = train_g_bidir.to('cuda')
        self.con_rel_encoder = ConRelEncoder(args).cuda()

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding_2 = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding_2,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def forward(self, sample, ent_emb, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        self.entity_embedding = ent_emb
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            
            relation_2 = torch.index_select(
                self.relation_embedding_2,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            # head(4,14579,500)
            # head_part(4,14579)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            relation_2 = torch.index_select(
                self.relation_embedding_2,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            # (1024,1,500)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            relation_2 = torch.index_select(
                self.relation_embedding_2,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            # (1024,128,500)
            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, relation_2,tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score, self.entity_embedding

    def DistMult(self, head, relation, relation_2, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        if mode == 'head-batch':
            score_2 = head * (relation_2 * tail)
        else:
            score_2 = (head * relation_2) * tail
        # (1024,128)

        score_cat = torch.cat((score, score_2), 2)
        score = score_cat.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    @staticmethod
    def train_step(model, type_model, ent_type_feat, train_g_bidir, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        type_model.train()
        optimizer.zero_grad()
        # positive_sample(1024,3),negative_sample(1024,256)
        positive_sample, negative_sample, subsampling_weight, entity, pos_type, neg_type, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            entity = entity.cuda()
            pos_type = pos_type.cuda()
            neg_type = neg_type.cuda()
        ent_type_feat_emb = type_model.get_type_feat_emb(ent_type_feat)
        ent_emb = model.con_rel_encoder(train_g_bidir, ent_type_feat_emb)

        # negative_score(1024,256)
        negative_score, _ = model((positive_sample, negative_sample), ent_emb, mode=mode)

        # 先注释掉
        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            # (1024)
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        # positive_score(1024)
        positive_score, head = model(positive_sample, ent_emb)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        neg_score = type_model((entity, neg_type), head, mode='neg_batch')
        pos_score = type_model((entity, pos_type), head, mode='pos_batch')
        neg_score = (F.softmax(neg_score * args.adversarial_temperature, dim=1).detach()
                     * F.logsigmoid(-neg_score)).sum(dim=1)
        pos_score = F.logsigmoid(pos_score).squeeze(dim=1)
        positive_loss = - pos_score.mean()
        negative_loss = - neg_score.mean()
        type_loss = (positive_loss + negative_loss) / 2
        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3).norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3 + model.relation_embedding_2.norm(p=3).norm(
                p=3) ** 3
                    + type_model.type_embedding.norm(p=3) ** 3
            )
            loss = loss + regularization + 0.6 * type_loss
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'type_loss': type_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, type_model, ent_type_feat, true_and_false_edges, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()
        type_model.eval()
        if args.countries:
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            samples = list()
            y_true = list()
            for edge in true_and_false_edges:
                sample = edge[:3]  # 前三个元素存储在 sample 中
                y = edge[3]  # 最后一个元素存储在 y_true 中

                samples.append(sample)
                y_true.append(y)

            samples = torch.LongTensor(samples)
            if args.cuda:
                samples = samples.cuda()

            with torch.no_grad():
                ent_type_feat_emb = type_model.get_type_feat_emb(ent_type_feat)
                ent_emb = model.con_rel_encoder(model.train_g_bidir, ent_type_feat_emb)
                y_score, _ = model(samples, ent_emb)
                y_score = y_score.squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            threshold = find_optimal_cutoff(y_true, y_score)[0]
            y_pred = np.zeros(len(y_score), dtype=np.int32)
            for i, _ in enumerate(y_score):
                if y_score[i] >= threshold:
                    y_pred[i] = 1
            ps, rs, _ = precision_recall_curve(y_true, y_score)
            micro_f1 = f1_score(y_true, y_pred, average='micro')
            auc_value = auc(rs, ps)
            metrics = {'F1': micro_f1, 'AUC-ROC': auc_value}


        return metrics


# 定义函数来计算Rec、MAP和NDCG指标
def calculate_metrics(ranking, k):
    # 计算召回率 (Rec)
    recall = (ranking <= k).sum() / len(ranking)

    # 计算平均精度 (MAP)
    num_hits = (ranking <= k).nonzero().tolist()
    precision_sum = 0.0
    average_precision = 0.0
    for i, hit in enumerate(num_hits):
        precision_sum += (i + 1) / (hit + 1)
        average_precision += precision_sum / (i + 1)
    average_precision /= len(num_hits)

    # 计算归一化贴现累积增益 (NDCG)
    idcg = sum(1.0 / (torch.log2(torch.arange(len(ranking)) + 2)))
    dcg = sum(1.0 / (torch.log2(ranking.float() + 2)))
    ndcg = dcg / idcg

    return recall, average_precision, ndcg


def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame(
        {
            "tf": pd.Series(tpr - (1 - fpr), index=i),
            "threshold": pd.Series(threshold, index=i),
        }
    )
    print()
    roc_t = roc.loc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t["threshold"])


def get_type_smi(args, ent_type_feat, type_emb):
    res_ent_ratio = 0.1
    # random select reserved entities
    ent_type_feat = ent_type_feat.to('cuda')
    ent_type_feat_emb = ent_type_feat.unsqueeze(2) * type_emb.unsqueeze(0)
    ent_type_feat_emb = torch.sum(ent_type_feat_emb, dim=1)
    res_ent_map = torch.unique(
        torch.tensor(np.random.choice(np.arange(args.nentity), int(args.nentity * res_ent_ratio), replace=False))).to(
        torch.long)
    ent_type_feat_norm = ent_type_feat_emb / (torch.norm(ent_type_feat_emb, dim=-1).reshape(-1, 1) + 1e-6)
    # 保留实体的类型特征
    ent_type_feat = ent_type_feat_norm[res_ent_map]
    ent_sim = torch.mm(ent_type_feat_norm, ent_type_feat.T)
    topk_sim, topk_idx = torch.topk(ent_sim, 10, dim=-1)
    return res_ent_map, ent_sim, topk_sim, topk_idx
