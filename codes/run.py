#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import argparse
import json
import logging
import os
import random

import dgl
import torch.nn as nn
import numpy as np
import torch
import pickle as pkl
from torch.utils.data import DataLoader

from model import KGEModel
from collections import defaultdict as ddict
from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
from type_model import TypeModel
from earl_model import ConRelEncoder

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', default='3', action='store_true', help='use GPU')

    parser.add_argument('--do_train', default=True, action='store_true')
    parser.add_argument('--do_valid', default=True, action='store_true')
    parser.add_argument('--do_test', default=True, action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--countries', default=True, action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None,
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')

    parser.add_argument('--data_path', type=str, default='../datao')
    parser.add_argument('--model', default='DistMult', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    # parser.add_argument('-gt', '--gamma_type', default=200.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=2048, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default='../models', type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--num_type', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--num_neg_type', default=10, type=int)
    parser.add_argument('--topNfilters', type=int, default=-700)

    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, type_model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'type_model_state_dict': type_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    # res_ent_map = model.res_ent_map.detach().cpu().numpy()
    # np.save(
    #     os.path.join(args.save_path, 'res_ent_map'),
    #     res_ent_map
    # )
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )
    type_embedding = type_model.type_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'type_embedding'),
        type_embedding
    )


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''


    with open(file_path) as fin:
        triples = fin.readlines()
    data = []
    for line in triples:
        h, r, t = line.strip().split('\t')
        # h, t, r = line.strip().split()
        data.append((entity2id[h], relation2id[r], entity2id[t]))
    return data

def read_triple_test(file_path):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            arr = line.strip().split()
            # h, t, r = line.strip().split('\t')
            r = arr[0]
            h = arr[1]
            t = arr[2]
            d = arr[3]
            if d == '1':
                triples.append((int(h), int(r), int(t)))
    return triples


# def build_entity2types_dictionaries(file_path, entity2id, type2id):
#     entityId2typeIds = {}
#     with open(file_path) as fin:
#         triples = fin.readlines()
#     for triple in triples:
#         # splitted_line = line.strip().split("\t")
#         # entity_name = splitted_line[0]
#         # entity_type = splitted_line[1]
#         entity_name, entity_type = triple.strip().split('\t')
#         entity_id = entity2id[entity_name]
#         type_id = type2id[entity_type]
#         if entity_id not in entityId2typeIds:
#             entityId2typeIds[entity_id] = []
#         if entity_type not in entityId2typeIds[entity_id]:
#             entityId2typeIds[entity_id].append(type_id)
#     return entityId2typeIds

def build_entity2types_dictionaries(file_path, entity2id, type2id):
    entityId2typeIds = {}
    with open(file_path) as fin:
        for line in fin:
            splitted_line = line.strip().split("\t")
            entity_name = splitted_line[0]
            entity_type = splitted_line[1]
            if entity_name not in entity2id:
                continue  # 跳过当前循环，继续处理下一行
            entity_id = entity2id[entity_name]
            type_id = type2id[entity_type]
            if entity_id not in entityId2typeIds:
                entityId2typeIds[entity_id] = []
            if entity_type not in entityId2typeIds[entity_id]:
                entityId2typeIds[entity_id].append(type_id)
    return entityId2typeIds



def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            entity, eid = line.strip().split('\t')
            entity2id[entity] = int(eid)

    # with open(os.path.join(args.data_path, 'relations.dict')) as fin:
    #     relation2id = dict()
    #     for line in fin:
    #         relation, rid = line.strip().split('\t')
    #         relation2id[relation] = int(rid)
    relation2id = {'/relation/invoke': 0}

    with open(os.path.join(args.data_path, 'entity2types_ttv.txt')) as fin:
        type2id = dict()
        type_counter = 0
        for line in fin:
            splitted_line = line.strip().split("\t")
            entity_type = splitted_line[1]
            if entity_type not in type2id:
                type2id[entity_type] = int(type_counter)
                type_counter += 1
        type2id["UNK"] = len(type2id)

    def get_hr2t_rt2h(train_triples, valid_triples, test_triples):
        hr2t_train = ddict(list)
        rt2h_train = ddict(list)

        hr2t_all = ddict(list)
        rt2h_all = ddict(list)

        for tri in train_triples:
            h, r, t = tri
            hr2t_train[(h, r)].append(t)
            rt2h_train[(r, t)].append(h)
            hr2t_all[(h, r)].append(t)
            rt2h_all[(r, t)].append(h)

        for tri in valid_triples:
            h, r, t = tri
            hr2t_all[(h, r)].append(t)
            rt2h_all[(r, t)].append(h)

        for tri in test_triples:
            h, r, t = tri
            hr2t_all[(h, r)].append(t)
            rt2h_all[(r, t)].append(h)

        return hr2t_train, rt2h_train, hr2t_all, rt2h_all

    def get_train_g_bidir(train_triples, num_ent):
        triples = torch.LongTensor(train_triples)
        num_tri = triples.shape[0]
        g = dgl.graph((torch.cat([triples[:, 0].T, triples[:, 2].T]),
                       torch.cat([triples[:, 2].T, triples[:, 0].T])), num_nodes=num_ent)
        g.edata['rel'] = torch.cat([triples[:, 1].T, triples[:, 1].T])
        g.edata['inv'] = torch.cat([torch.zeros(num_tri), torch.ones(num_tri)])

        return g

    def get_ent_type_feat(entityId2typeIds, num_type, num_ent):
        ent_type_feat = torch.zeros(num_ent, num_type)
        for entity, entity_types in entityId2typeIds.items():
            for entity_type in entity_types:
                ent_type_feat[int(entity), int(entity_type)] += 1

        return ent_type_feat

    nentity = len(entity2id)
    nrelation = len(relation2id)
    num_type = len(type2id)
    args.num_type = num_type
    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#type: %d' % num_type)

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    all_true_triples = train_triples + valid_triples + test_triples
    hr2t_train, rt2h_train, hr2t_all, rt2h_all = get_hr2t_rt2h(train_triples, valid_triples, test_triples)
    entityId2typeIds = build_entity2types_dictionaries(os.path.join(args.data_path, 'entity2types_ttv.txt'), entity2id
                                                       , type2id)
    train_g_bidir = get_train_g_bidir(train_triples, nentity).to('cuda')
    ent_type_feat = get_ent_type_feat(entityId2typeIds, num_type, nentity)

    # def load_test_edges(filename):
    #     true_and_false_edges = []
    #     with open(filename, "r") as f:
    #         for line in f:
    #             arr = line.strip().split()
    #             relation = arr[0]
    #             source = arr[1]
    #             target = arr[2]
    #             if len(arr) == 4:
    #                 true_and_false_edges.append((int(source), int(relation), int(target), int(arr[3])))
    #             else:
    #                 # GATNE dataset have no false edges for training
    #                 true_and_false_edges.append((relation, source, target, "1"))
    #     return true_and_false_edges
    def generate_false_edges(validOrtest_triples, nentity, hr2t_all, rt2h_all):
        true_edges = []
        false_edges = []
        rand_result = torch.rand((len(validOrtest_triples), 1))
        perturb_head = rand_result < 0.5
        perturb_tail = rand_result >= 0.5
        for triple in validOrtest_triples:
            head = triple[0]
            relation = triple[1]
            tail = triple[2]
            true_edges.append((head, relation, tail, 1))
        for i, (h, r, t, _) in enumerate(true_edges):
            if perturb_head[i]:
                neg_head_ent = np.random.choice(np.delete(np.arange(nentity), rt2h_all[(r, t)]),
                                                1)
                fa_head = int(neg_head_ent[0])
                false_edges.append((fa_head, r, t, 0))
            else:
                neg_tail_ent = np.random.choice(np.delete(np.arange(nentity), hr2t_all[(h, r)]),
                                                1)
                fa_tail = int(neg_tail_ent[0])
                false_edges.append((h, r, fa_tail, 0))
        final_edges = true_edges + false_edges
        random.shuffle(final_edges)  # random.Random(20).shuffle(final_edges)
        print(len(final_edges))
        return final_edges

    # 初始化模型
    kge_model = KGEModel(
        args,
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        train_g_bidir=train_g_bidir,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    type_model = TypeModel(
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        num_type=args.num_type,
        nrelation=nrelation,
    )
    # con_rel_encoder = ConRelEncoder(args)
    # schema_graph = get_schema_graph(args).to('cuda')

    logging.info('Model Parameter Configuration:')
    num_param = 0
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s, param.numel=%s' % (
            name, str(param.size()), str(param.requires_grad), param.numel()))
        num_param += param.numel()
    for name, param in type_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s, param.numel=%s' % (
            name, str(param.size()), str(param.requires_grad), param.numel()))
        num_param += param.numel()
    logging.info("总参数数量: %.1f M", num_param / 1e6)
    if args.cuda:
        kge_model = kge_model.cuda()
        type_model = type_model.cuda()

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, entityId2typeIds, num_type, args, nentity, nrelation,
                         args.negative_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, entityId2typeIds, num_type, args, nentity, nrelation,
                         args.negative_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )
        # 定义数据加载器，奇数step是tail-batch，偶数step是head-batch
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        # optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, kge_model.parameters()),
        #     lr=current_learning_rate
        # )
        parameters = list(filter(lambda p: p.requires_grad, kge_model.parameters())) + \
                     list(filter(lambda p: p.requires_grad, type_model.parameters()))

        optimizer = torch.optim.Adam(parameters, lr=current_learning_rate)
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []

        # Training Loop
        for step in range(init_step, args.max_steps):
            # 开始训练加载数据加载器和模型
            log = kge_model.train_step(kge_model, type_model, ent_type_feat, train_g_bidir,
                                       optimizer, train_iterator, args)

            training_logs.append(log)
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                # optimizer = torch.optim.Adam(
                #     filter(lambda p: p.requires_grad, kge_model.parameters()),
                #     lr=current_learning_rate
                # )
                parameters = list(filter(lambda p: p.requires_grad, kge_model.parameters())) + \
                             list(filter(lambda p: p.requires_grad, type_model.parameters()))

                optimizer = torch.optim.Adam(parameters, lr=current_learning_rate)
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, type_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                true_and_false_edges = generate_false_edges(valid_triples, nentity, hr2t_all, rt2h_all)
                #true_and_false_edges = load_test_edges(args.data_path + "/valid.txt")
                metrics = kge_model.test_step(kge_model, type_model, ent_type_feat, true_and_false_edges,
                                              valid_triples,
                                              all_true_triples, args)
                log_metrics('Valid', step, metrics)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, type_model, optimizer, save_variable_list, args)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        true_and_false_edges = generate_false_edges(valid_triples, nentity, hr2t_all, rt2h_all)
        # true_and_false_edges = load_test_edges(args.data_path + "/valid.txt")
        metrics = kge_model.test_step(kge_model, type_model, ent_type_feat, true_and_false_edges,
                                      valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        true_and_false_edges = generate_false_edges(test_triples, nentity, hr2t_all, rt2h_all)
        # true_and_false_edges = load_test_edges(args.data_path + "/test.txt")
        metrics = kge_model.test_step(kge_model, type_model, ent_type_feat, true_and_false_edges,
                                      test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


if __name__ == '__main__':
    main(parse_args())
