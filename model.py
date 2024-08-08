import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import softmax, degree
from torch_geometric.nn import GCNConv,GraphNorm
from torch_scatter import scatter
import math


class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=50, dropout=0.2):  # original max_len=240
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        if isinstance(t, int):
            t = torch.arange(t).to(x.device)
            enc = self.lin(self.emb(t))  # [T, d]
            enc = enc.expand_as(x)  # [N,T,d]
            return x + enc
        else:
            return x + self.lin(self.emb(t))


class CausalSubgraphNet(nn.Module):

    def __init__(self, hid_dim, out_dim, causal_ratio, dropout):
        super(CausalSubgraphNet, self).__init__()
        self.conv1 = GCNConv(in_channels=hid_dim, out_channels=2 * hid_dim)
        self.conv2 = GCNConv(in_channels=2 * hid_dim, out_channels=out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim * 4),
            nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(out_dim * 4, 1), nn.Sigmoid()
        )
        self.ratio = causal_ratio
        self.degcoding = nn.Embedding(100, hid_dim)
    def forward(self, x, edge_index, t, time_encodeing):
        # degree encoding
        node_deg = degree(edge_index[0], num_nodes=x.shape[0])
        deg_codeing = self.degcoding(node_deg.long())
        x = x + deg_codeing

        # GNN
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        # time encoding embedding
        x = time_encodeing(x, t)

        row, col = edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.mlp(edge_rep).view(-1)

        causal_edge_index, conf_edge_index, causal_edge_score, conf_edge_socre = self.split_graph(edge_index, edge_score, self.ratio)

        return causal_edge_index, conf_edge_index, causal_edge_score, conf_edge_socre

    def split_graph(self, edge_index, edge_score, causal_ratio):
        num_conf = int((1 - causal_ratio) * edge_index.shape[1])
        sort_edge_score, sort_edge_index = torch.sort(edge_score)
        conf_edge_index = edge_index[:, sort_edge_index[:num_conf]]
        causal_edge_index = edge_index[:, sort_edge_index[num_conf:]]
        conf_edge_socre = sort_edge_score[:num_conf]
        causal_edge_score = sort_edge_score[num_conf: ]
        return causal_edge_index, conf_edge_index, causal_edge_score, conf_edge_socre

class EnvGen(nn.Module):

    def __init__(self, hidden_dim, out_dim, drop):
        super(EnvGen, self).__init__()
        self.enc = GCNConv(hidden_dim, hidden_dim)
        self.enc_mean = GCNConv(hidden_dim, out_dim)
        self.enc_std = GCNConv(hidden_dim, out_dim)
        self.drop = drop
        self.prior = GCNConv(hidden_dim, out_dim)
        self.prior_mean = nn.Sequential(nn.Linear(hidden_dim, out_dim))
        self.prior_std = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Sigmoid())
        self.act = nn.Sigmoid()
        self.degcoding = nn.Embedding(100, hidden_dim)
    def forward(self, edge_index, x, t, time_encoding, edge_score, total_len, train_len):

        # post-doc
        enc_t = F.relu(self.enc(x, edge_index, edge_weight=edge_score))

        enc_t = F.dropout(enc_t, self.drop)
        enc_mean_t = self.enc_mean(enc_t, edge_index)
        enc_std_t = self.act(self.enc_std(enc_t, edge_index))

        # sample
        conf_z_t = self.reparameterize(enc_mean_t, enc_std_t)

        # prior
        prior_enc_t, prior_std = self.prior_param(edge_index, x, t, time_encoding)

        # loss
        kl_loss = self.KL_loss(enc_mean_t, enc_std_t, prior_enc_t, prior_std)
        return kl_loss, conf_z_t

    def prior_param(self, edge_index, x, t, time_encoding):
        prior = F.relu(self.prior(x, edge_index))
        prior = time_encoding(prior, t)
        prior_mean = self.prior_mean(prior)
        prior_std = self.prior_std(prior)

        return prior_mean, prior_std

    def reparameterize(self, mean, std):
        eps = torch.randn_like(mean)
        return eps.mul(std).add_(mean)

    def KL_loss(self, enc_mean, enc_std, prior_mean, proir_std):
        KL_div = (2 * torch.log(proir_std + 1e-9) - 2 * torch.log(enc_std +1e-9) +
                  (torch.pow(enc_std + 1e-9, 2) + torch.pow(enc_mean - prior_mean, 2)) /
                  torch.pow(proir_std + 1e-9, 2) - 1)
        return 0.5 * torch.mean(torch.sum(KL_div, dim=1), dim=0)

    def KL_Gaussian_loss(self, enc_mean, enc_std):
        std_log = torch.log(enc_std + 1e-9)
        kl = torch.mean(torch.sum(1 + 2 * std_log - enc_mean.pow(2) -
                                           torch.pow(torch.exp(std_log), 2), 1))
        return (-0.5 / enc_mean.shape[0]) * kl

class DyCIL(nn.Module):
    def __init__(self, args):
        super(DyCIL, self).__init__()
        self.args = args

        self.feat = Parameter((torch.ones(args.num_nodes, args.input_dim)).to(args.device), requires_grad=True)
        self.device = args.device
        self.structural_head_config = list(map(int, args.structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, args.structural_layer_config.split(",")))
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop
        self.linear = nn.Linear(args.input_dim, args.hid_dim, bias=bool(args.lin_bias))
        self.causal_subgraph_net = CausalSubgraphNet(args.hid_dim, args.hid_dim, args.causal_ratio, args.dropout)
        self.time_emb = RelTemporalEncoding(args.hid_dim)
        self.structural_attn, self.temporal_attn = self.build_model()
        if args.experiment_name == 'lp':
            self.cs_decoder = MultiplyPredictor()
            self.ss_decoder = LinkPredictor(2 * args.hid_dim, args.hid_dim, 1, args.dropout)
        else:
            self.cs_decoder = NodeClf(args.nc_layers, args.num_classes, args.hid_dim)
            self.ss_decoder = MutiNodeClf(args.nc_layers, args.num_classes, args.hid_dim)

        self.EnvGenertor = EnvGen(args.hid_dim, args.hid_dim, args.dropout)


    def forward(self, edge_index_list, x_list, train_len):
        if x_list is None:
            x = [self.linear(self.feat) for i in range(len(edge_index_list))]
        else:
            x = [self.linear(x) for x in x_list]

        structural_out = []
        env_kl_loss = torch.tensor([]).to(self.args.device)
        conf_embedding_list  = []
        causal_s_rep_list, causal_t_rep_list = [], []
        causal_edge_list = []
        for t in range(0, train_len):
            causal_edge_index, conf_edge_index, causal_edge_score, conf_edge_socre = self.causal_subgraph_net(x[t], edge_index_list[t].to(x[t].device), torch.LongTensor([t]).to(x[t].device), self.time_emb)
            causal_edge_list.append(causal_edge_index)
            for j, layer in enumerate(self.structural_attn):
                if j == 0:
                    spatial_out = layer(x[t], causal_edge_index, edge_weight=causal_edge_score)
                else:
                    spatial_out = layer(spatial_out, causal_edge_index, edge_weight=causal_edge_score)
                if j != len(self.structural_attn) - 1:
                    spatial_out = F.relu(spatial_out)
            structural_out.append(spatial_out)

            kl_loss_t, conf_embedding_t = self.EnvGenertor(conf_edge_index, x[t],
                                                           torch.LongTensor([t]).to(self.device), self.time_emb,
                                                           conf_edge_socre, len(edge_index_list), train_len)
            conf_embedding_list.append(conf_embedding_t)
            env_kl_loss = torch.cat([env_kl_loss, kl_loss_t.unsqueeze(0)])

        structural_outputs = [g[:, None, :] for g in structural_out]  # list of [Ni, 1, F]
        structural_outputs = torch.cat(structural_outputs, dim=1)

        for j, layer in enumerate(self.temporal_attn):
            if j == 0:
                temporal_out = layer(structural_outputs, self.time_emb)
            else:
                temporal_out = layer(temporal_out, self.time_emb)

        return temporal_out, conf_embedding_list, env_kl_loss

    def build_model(self):
        input_dim = self.args.hid_dim
        # 1: Structural Attention Layers
        structural_attention_layers = nn.ModuleList()
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(input_dim=input_dim,
                                             output_dim=self.structural_layer_config[i],
                                             n_heads=self.structural_head_config[i],
                                             dropout=self.args.spatial_drop,
                                             residual=self.args.residual,
                                             use_fmask=self.args.fmask,
                                             norm=self.args.norm,
                                             skip=self.args.skip)
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]

        # 2: Temporal Attention Layers
        input_dim = self.structural_layer_config[-1]
        temporal_attention_layers = nn.ModuleList()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual,
                                           use_RTE=self.args.use_RTE)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]
        return structural_attention_layers, temporal_attention_layers


class StructuralAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, dropout, residual, use_fmask=False, norm=True, skip=False):
        super(StructuralAttentionLayer, self).__init__()
        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.in_dim, self.hid_dim = input_dim, output_dim
        self.update_norm = nn.LayerNorm(output_dim)
        self.cs_mlp = nn.Sequential(nn.Linear(output_dim, 2 *output_dim), nn.GELU(), nn.Linear(2 * output_dim, output_dim))
        self.fmask = nn.Parameter(torch.ones(output_dim))
        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

        self.q_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.d_k = output_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.aggr = 'add'
        self.update_drop = nn.Dropout(dropout)
        self.update_skip = nn.Parameter(torch.ones(1))
        self.update_linear = nn.Linear(output_dim, output_dim)
        self.use_fmask = use_fmask
        self.norm = norm
        self.skip = skip
        self.node_dim = 0
    def forward(self, x, edge_index, edge_weight=None):

        if edge_weight == None:
            edge_weight = torch.ones(edge_index.shape[1]).view(-1,1).to(x.device)

        q_mat = self.q_linear(x[edge_index[1]]).view(-1, self.n_heads, self.d_k) # target
        k_mat = self.k_linear(x[edge_index[0]]).view(-1, self.n_heads, self.d_k) # src
        v_mat = self.v_linear(x[edge_index[0]]).view(-1, self.n_heads, self.d_k)  # [E,h,F/h]

        res_att = (q_mat * k_mat).sum(dim=-1) / self.sqrt_dk  # [E,h]
        res_att = edge_weight.view(-1, 1) * res_att
        res_msg = v_mat  # [E,h,F/h]

        ei_tar = edge_index[1]
        res_att = softmax(res_att, ei_tar)

        res = res_msg * res_att.view(-1, self.n_heads, 1)  # [E,h,F/h]
        res = res.view(-1, self.hid_dim)  # [E,F]

        res = scatter(res, ei_tar, dim=self.node_dim, dim_size=x.shape[0], reduce=self.aggr)  # [N,F]

        if self.use_fmask:
            fmask_c = F.softmax(self.fmask, dim=0)
            res = res * fmask_c

        def ffn(x):

            if self.norm:
                res = self.cs_mlp(self.update_norm(x))
            else:
                res = self.cs_mlp(x)

            res = self.update_drop(res)

            if self.skip:
                alpha = torch.sigmoid(self.update_skip)
                res = (1-alpha)*x + alpha*res
            else:
                res = x + res
            return res

        res = ffn(res + x)
        return res

    def xavier_init(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)


class TemporalAttentionLayer(nn.Module):
    def __init__(self, input_dim, n_heads, attn_drop, residual, use_RTE=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.residual = residual
        self.RTE = use_RTE
        self.time_emb = RelTemporalEncoding(input_dim)
        # define weights
        # self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs, time_encoding):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input
        time_length = inputs.shape[1]
        if self.RTE:
            temporal_inputs = time_encoding(inputs, time_length)
        else:
            temporal_inputs = inputs
        # position_inputs = torch.arange(0, self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
        #     inputs.device)
        # temporal_inputs = inputs + self.position_embeddings[position_inputs]  # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1] / self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]

        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]
        outputs = outputs / (time_length ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs  # [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
                            dim=2)  # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)


class MultiplyPredictor(torch.nn.Module):
    def __init__(self):
        super(MultiplyPredictor, self).__init__()
    def forward(self, z, edge_index):
        pred = torch.sum(z[edge_index[0]] * z[edge_index[1]], dim=1)
        return torch.sigmoid(pred)

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout):
        super(LinkPredictor, self).__init__()
        self.dropout = dropout
        self.lins = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(),
                                  nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden_channels, out_channels))
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, z, e):
        x = torch.concat([z[e[0]], z[e[1]]], dim=1)
        x = self.lins(x)
        return torch.sigmoid(x).squeeze()

class NodeClf(nn.Module):
    def __init__(self, layers, num_class, hid_dim) -> None:
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(hid_dim, num_class))
    def forward(self, x):
        for layer in self.clf:
            x = layer(x)
        return x

class MutiNodeClf(nn.Module):
    def __init__(self, layers, num_class, hid_dim) -> None:
        super().__init__()
        clf = nn.ModuleList()
        for i in range(layers):
            clf.append(nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU()))
        clf.append(nn.Linear(hid_dim, num_class))
        self.clf = clf
        # self.clf = nn.Sequential(nn.Linear(hid_dim, num_class))
    def forward(self, x):
        for layer in self.clf:
            x = layer(x)
        return x
