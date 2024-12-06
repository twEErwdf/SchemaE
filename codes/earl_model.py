import os

import torch.nn as nn
import torch
import dgl
import torch.nn.functional as F
import dgl.function as fn
import pickle as pkl


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim).cuda()
        self.linear2 = nn.Linear(hid_dim, out_dim).cuda()

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x


class ConRelEncoder(nn.Module):
    def __init__(self, args):
        super(ConRelEncoder, self).__init__()
        self.args = args
        res_ent = pkl.load(open(os.path.join(args.data_path, 'res_ent_0p1.pkl'), 'rb'))

        self.res_ent_map = res_ent['res_ent_map'].to('cuda')
        num_res_ent = self.res_ent_map.shape[0]

        self.rel_head_emb = nn.Parameter(torch.Tensor(args.nrelation, args.hidden_dim))
        self.rel_tail_emb = nn.Parameter(torch.Tensor(args.nrelation, args.hidden_dim))
        self.res_ent_emb = nn.Parameter(torch.Tensor(num_res_ent, args.hidden_dim))

        nn.init.xavier_uniform_(self.res_ent_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_head_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_tail_emb, gain=nn.init.calculate_gain('relu'))

        self.feat_mlp = MLP(args.hidden_dim, args.hidden_dim, args.hidden_dim)

    def forward(self, g_bidir, ent_type_feat_emb):
        with g_bidir.local_scope():
            num_edge = g_bidir.num_edges()
            etypes = g_bidir.edata['rel']
            g_bidir.edata['ent_e'] = torch.zeros(num_edge, self.args.hidden_dim).cuda()
            rh_idx = (g_bidir.edata['inv'] == 1)
            rt_idx = (g_bidir.edata['inv'] == 0)
            g_bidir.edata['ent_e'][rh_idx] = self.rel_head_emb[etypes[rh_idx]]
            g_bidir.edata['ent_e'][rt_idx] = self.rel_tail_emb[etypes[rt_idx]]
            # 每一条边的属性值被复制并发送到相应的目标节点作为消息
            message_func = dgl.function.copy_e('ent_e', 'msg')
            # 将每个节点收到的消息进行平均聚合将聚合结果存储在feat属性中
            reduce_func = dgl.function.mean('msg', 'feat')
            g_bidir.update_all(message_func, reduce_func)
            g_bidir.edata.pop('ent_e')
            
            zero_degrees = ((g_bidir.in_degrees() + g_bidir.out_degrees()) == 0)
            # zero_degrees = ((g_bidir.in_degrees() + g_bidir.out_degrees()) == 0)
            # count = sum(zero_degrees)
            # print("数量:", count)
            # zero_idx = torch.nonzero(zero_degrees).squeeze(1)
            # print(zero_idx)
            zero_idx = torch.nonzero(zero_degrees).squeeze(1)

            zero_ent_type_feat_emb = ent_type_feat_emb[zero_idx]
            z1 = F.normalize(zero_ent_type_feat_emb)
            z2 = F.normalize(self.res_ent_emb)
            zero_ent_sim = torch.mm(z1, z2.t())

            zero_topk_sim, zero_topk_idx = torch.topk(zero_ent_sim, 5, dim=-1)

            zero_topk_idx = zero_topk_idx.to('cuda')
            zero_ent_sim = zero_topk_sim.to('cuda')
            zero_ent_sim = torch.softmax(zero_ent_sim / 0.2, dim=-1)
            zero_topk_res_ent = torch.index_select(self.res_ent_emb, 0, zero_topk_idx.reshape(-1)).reshape(
                zero_topk_idx.shape[0], zero_topk_idx.shape[1], self.args.hidden_dim)
            zero_topk_res_ent = zero_ent_sim.unsqueeze(2) * zero_topk_res_ent
            zero_emb = torch.sum(zero_topk_res_ent, dim=1)

            g_bidir.ndata['feat'][zero_idx] = zero_emb

            return self.feat_mlp(g_bidir.ndata['feat'])






