import torch.nn as nn
import torch


class TypeModel(nn.Module):
    def __init__(self, hidden_dim, gamma, num_type, nrelation):
        super(TypeModel, self).__init__()
        self.num_type = num_type
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.type_embedding = nn.Parameter(torch.zeros(self.num_type, hidden_dim))
        nn.init.uniform_(
            tensor=self.type_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        # self.type_embedding_2 = nn.Parameter(torch.zeros(self.num_type, hidden_dim))
        # nn.init.uniform_(
        #     tensor=self.type_embedding_2,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )
        # self.linear3 = nn.Linear(hidden_dim*2, hidden_dim).cuda()

    def get_type_feat_emb(self, ent_type_feat):
        ent_type_feat = ent_type_feat.to('cuda')
        # type_embedding =  torch.cat((self.type_embedding_2, self.type_embedding), dim=1)
        # type_embedding = self.linear3(type_embedding)
        ent_type_feat_emb = ent_type_feat.unsqueeze(2) * self.type_embedding.unsqueeze(0)
        # (14541,300)
        ent_type_feat_emb = torch.sum(ent_type_feat_emb, dim=1)
        return ent_type_feat_emb

    def forward(self, sample, ent_emb, mode='pos_batch'):
        # (1024,1,2000)
        self.entity_embedding = ent_emb
        # self.entity_embedding_2 = ent_emb_2

        if mode == 'pos_batch':
            entity, pos_type = sample
            entity_ed = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=entity
            ).unsqueeze(1)

            # entity_ed_2 = torch.index_select(
            #     self.entity_embedding_2,
            #     dim=0,
            #     index=entity
            # ).unsqueeze(1)

            if pos_type == None:
                type_ed = self.type_embedding.unsqueeze(0)
                # type_ed_2 = self.type_embedding_2.unsqueeze(0)
            else:
                type_ed = torch.index_select(
                    self.type_embedding,
                    dim=0,
                    index=pos_type.view(-1).long()
                ).unsqueeze(1)

                # type_ed_2 = torch.index_select(
                #     self.type_embedding_2,
                #     dim=0,
                #     index=pos_type.view(-1)
                # ).unsqueeze(1)

        if mode == 'neg_batch':
            entity, neg_type = sample
            batch_size, neg_type_size = neg_type.size(0), neg_type.size(1)
            entity_ed = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=entity
            ).unsqueeze(1)

            # entity_ed_2 = torch.index_select(
            #     self.entity_embedding_2,
            #     dim=0,
            #     index=entity
            # ).unsqueeze(1)

            type_ed = torch.index_select(
                self.type_embedding,
                dim=0,
                index=neg_type.view(-1)
            ).view(batch_size, neg_type_size, -1)

            # type_ed_2 = torch.index_select(
            #     self.type_embedding_2,
            #     dim=0,
            #     index=neg_type.view(-1)
            # ).view(batch_size, neg_type_size, -1)
        score = self.Typescore(entity_ed,  type_ed)

        return score

    def Typescore(self, entity_ed,  type_ed):
        score = entity_ed * type_ed
        # score_2 = entity_ed * type_ed_2
        # score_cat = torch.cat((score, score_2), 2)
        score = score.sum(dim=2)
        return score
