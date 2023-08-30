import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse, remove_self_loops

def adj2adj(node_graph, batch_size, window_size, zdim):
    graph = torch.tensor(node_graph).cuda()
    graph1 = graph.squeeze(0).squeeze(0).repeat(batch_size, window_size, 1, 1) \
        .reshape(-1, graph.shape[-2], graph.shape[-1])
    adj0, adj1, fea = [], [], []
    node_adj = dense_to_sparse(graph1)[0]
    node_efea = graph.unsqueeze(-1).repeat(1, 1, zdim)
    for num in range(node_adj.shape[1]):
        idx = torch.argwhere(node_adj[1] == num)
        idy = torch.argwhere(node_adj[0] == num)
        adj0.append(idx.repeat(1, idy.shape[0]).reshape(-1))
        adj1.append(idy.repeat(idx.shape[0], 1).reshape(-1))
        fea.append(torch.ones(
            idy.shape[0] * idx.shape[0], device=graph.device) * num)

    adj = torch.stack([torch.concat(adj0), torch.concat(adj1)], dim=0)
    fea = torch.concat(fea)
    edge_adj, edge_efea = remove_self_loops(adj, fea)
    return node_adj, node_efea, edge_adj, edge_efea

class FeedForward(nn.Module):
    def __init__(self, node_embedding_dim, FeedForward_dim):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(node_embedding_dim, FeedForward_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(FeedForward_dim, node_embedding_dim)
        )

    def forward(self, X):
        return self.ff(X)


class AddNorm(nn.Module):
    def __init__(self, node_embedding_dim, dropout=0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(node_embedding_dim)

    def forward(self, X_old, X):
        return self.ln(self.dropout(X) + X_old)


class AddALL(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, dropout=0.1):
        super().__init__()
        self.addnorm_node = AddNorm(node_embedding_dim, dropout)
        self.addnorm_trace = AddNorm(edge_embedding_dim, dropout)
        self.addnorm_log = AddNorm(log_embedding_dim, dropout)

    def forward(self, node_old, trace_old, log_old, x_node, x_trace, x_log):
        return self.addnorm_node(node_old, x_node.reshape(node_old.shape)), \
            self.addnorm_trace(trace_old, x_trace.reshape(trace_old.shape)), \
            self.addnorm_log(log_old, x_log.reshape(log_old.shape))


class FFN(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.ff_node = FeedForward(node_embedding_dim, node_embedding_dim*2)
        self.addnorm_node = AddNorm(node_embedding_dim, dropout)
        self.ff_trace = FeedForward(edge_embedding_dim, edge_embedding_dim*2)
        self.addnorm_trace = AddNorm(edge_embedding_dim, dropout)
        self.ff_log = FeedForward(log_embedding_dim, log_embedding_dim*2)
        self.addnorm_log = AddNorm(log_embedding_dim, dropout)

    def forward(self, x_node, x_trace, x_log):
        return self.addnorm_node(x_node, self.ff_node(x_node)), \
            self.addnorm_trace(x_trace, self.ff_trace(x_trace)), \
            self.addnorm_log(x_log, self.ff_log(x_log))


class Temporal_Attention(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, trace2pod, heads_node=4, heads_edge=4, heads_log=4, dropout=0.1,
                 window_size=16, batch_size=10):
        super(Temporal_Attention, self).__init__()
        self.window_size = window_size
        self.batch_size = batch_size
        self.trace2pod = trace2pod
        
        self.attention_node = nn.MultiheadAttention(embed_dim=node_embedding_dim, num_heads=heads_node,
                                                    dropout=dropout,batch_first=True)
        self.attention_trace = nn.MultiheadAttention(embed_dim=edge_embedding_dim, num_heads=heads_edge,
                                                     dropout=dropout, batch_first=True)
        self.attention_log = nn.MultiheadAttention(embed_dim=log_embedding_dim, num_heads=heads_log,
                                                     dropout=dropout, batch_first=True)

        self.vff_node = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.vff_trace = nn.Linear(edge_embedding_dim, edge_embedding_dim)
        self.vff_log = nn.Linear(log_embedding_dim, log_embedding_dim)
        self.headff_node = nn.Linear(heads_node * window_size, window_size)
        self.headff_trace = nn.Linear(heads_edge * window_size, window_size)
        self.headff_log = nn.Linear(heads_log * window_size, window_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_node, x_trace, x_log, mask=False):
        x_node = x_node.permute(0, 2, 1, 3).reshape(-1, self.window_size, x_node.shape[-1])
        x_trace = x_trace.permute(0, 2, 1, 3).reshape(-1, self.window_size, x_trace.shape[-1])
        x_log = x_log.permute(0, 2, 1, 3).reshape(-1, self.window_size, x_log.shape[-1])

        if mask:
            mask_att = (torch.triu(torch.ones(self.window_size, self.window_size, device=x_node.device)) == 1).transpose(0, 1)
            mask_att = mask_att.float().masked_fill(mask_att == 0, float('-inf')).masked_fill(mask_att == 1, float(0.0))

            att_n = self.attention_node(x_node, x_node, x_node, attn_mask=mask_att, average_attn_weights=False)[1]
            att_t = self.attention_trace(x_trace, x_trace, x_trace, attn_mask=mask_att, average_attn_weights=False)[1]
            att_l = self.attention_log(x_log, x_log, x_log, attn_mask=mask_att, average_attn_weights=False)[1]
        else:
            att_n = self.attention_node(x_node, x_node, x_node, average_attn_weights=False)[1]
            att_t = self.attention_trace(x_trace, x_trace, x_trace, average_attn_weights=False)[1]
            att_l = self.attention_log(x_log, x_log, x_log, average_attn_weights=False)[1]

        att_n = att_n.reshape(self.batch_size, -1, att_n.shape[-3], att_n.shape[-2], att_n.shape[-1])
        att_t = att_t.reshape(self.batch_size, -1, att_t.shape[-3], att_t.shape[-2], att_t.shape[-1])
        att_l = att_l.reshape(self.batch_size, -1, att_l.shape[-3], att_l.shape[-2], att_l.shape[-1])
        
        # att = torch.concat([att_n.mean(axis=[1,2], keepdims=True), att_t.mean(axis=[1,2], keepdims=True), att_l.mean(axis=[1,2], keepdims=True)], dim=1).mean(axis=1, keepdims=True)
        # att = torch.concat([att_n.mean(axis=[2], keepdims=True), att_t.mean(axis=[2], keepdims=True), att_l.mean(axis=[2], keepdims=True)], dim=1).mean(axis=1, keepdims=True)
        att_nn = torch.matmul(att_n.permute(0, 2, 3, 4, 1), self.trace2pod.T.float()).permute(0, 4, 1, 2, 3)
        att_tn = torch.matmul(att_t.permute(0, 2, 3, 4, 1), self.trace2pod.float()).permute(0, 4, 1, 2, 3)
        att_ln = torch.matmul(att_l.permute(0, 2, 3, 4, 1), self.trace2pod.T.float()).permute(0, 4, 1, 2, 3)

        att_node = torch.concat([att_n.mean(axis=[1,2], keepdims=True), att_tn.mean(axis=[1,2], keepdims=True), att_l.mean(axis=[1,2], keepdims=True)], dim=1).mean(axis=1, keepdims=True)
        att_edge = torch.concat([att_nn.mean(axis=[1,2], keepdims=True), att_t.mean(axis=[1,2], keepdims=True), att_ln.mean(axis=[1,2], keepdims=True)], dim=1).mean(axis=1, keepdims=True)

        x_node = torch.bmm(
            self.softmax(att_node + att_n).reshape(att_n.shape[0] * att_n.shape[1], att_n.shape[2] * att_n.shape[3],
                                              att_n.shape[-1]), self.vff_node(x_node))
        x_trace = torch.bmm(
            self.softmax(att_edge + att_t).reshape(att_t.shape[0] * att_t.shape[1], att_t.shape[2] * att_t.shape[3],
                                              att_t.shape[-1]), self.vff_trace(x_trace))
        x_log = torch.bmm(
            self.softmax(att_node + att_l).reshape(att_l.shape[0] * att_l.shape[1], att_l.shape[2] * att_l.shape[3],
                                              att_l.shape[-1]), self.vff_log(x_log))
    
        x_node = self.headff_node(x_node.permute(0, 2, 1)).permute(0, 2, 1) \
            .reshape(self.batch_size, -1, self.window_size, x_node.shape[-1]).permute(0, 2, 1, 3)
        x_trace = self.headff_trace(x_trace.permute(0, 2, 1)).permute(0, 2, 1) \
            .reshape(self.batch_size, -1, self.window_size, x_trace.shape[-1]).permute(0, 2, 1, 3)
        x_log = self.headff_log(x_log.permute(0, 2, 1)).permute(0, 2, 1) \
            .reshape(self.batch_size, -1, self.window_size, x_log.shape[-1]).permute(0, 2, 1, 3)
        return x_node, x_trace, x_log


class Spatial_Attention(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, heads_n2e=4, heads_e2n=4, dropout=0.1, batch_size=10,
                 window_size=16):
        super(Spatial_Attention, self).__init__()

        self.batch_size = batch_size
        self.window_size = window_size

        self.node2node = GATv2Conv(in_channels=node_embedding_dim + log_embedding_dim,
                                   out_channels=int((node_embedding_dim + log_embedding_dim) / heads_n2e),
                                   heads=heads_n2e, dropout=dropout, edge_dim=edge_embedding_dim, add_self_loops=False)
        self.egde2node = GATv2Conv(in_channels=edge_embedding_dim, out_channels=int(edge_embedding_dim / heads_e2n),
                                   heads=heads_e2n, dropout=dropout, edge_dim=node_embedding_dim + log_embedding_dim,
                                   add_self_loops=False)

    def forward(self, x_node, x_trace, x_log, node_adj, edge_adj, edge_efea):
        node = torch.concat([x_node, x_log], dim=-1)
        node = node.reshape(-1, node.shape[-1])
        x_trace = x_trace.reshape(-1, x_trace.shape[-1])
        
        node = self.node2node(node, node_adj, x_trace)
        x_trace = self.egde2node(x_trace, edge_adj, node[edge_efea.long()])

        x_node = node[:, :x_node.shape[-1]].reshape(self.batch_size, self.window_size, -1, x_node.shape[-1])
        x_trace = x_trace.reshape(self.batch_size, self.window_size, -1, x_trace.shape[-1])
        x_log = node[:, x_node.shape[-1]:].reshape(self.batch_size, self.window_size, -1, x_log.shape[-1])
        return x_node, x_trace, x_log


class Encoder_Decoder_Attention(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, log_embedding_dim, heads_node=4, heads_edge=4, heads_log=4, dropout=0.1):
        super(Encoder_Decoder_Attention, self).__init__()
        self.attention_node = nn.MultiheadAttention(
            embed_dim=node_embedding_dim, num_heads=heads_node, batch_first=True, dropout=dropout)
        self.attention_trace = nn.MultiheadAttention(
            embed_dim=edge_embedding_dim, num_heads=heads_edge, batch_first=True, dropout=dropout)
        self.attention_log = nn.MultiheadAttention(
            embed_dim=log_embedding_dim, num_heads=heads_log, batch_first=True, dropout=dropout)

    def forward(self, x_node, x_trace, x_log, z_node, z_trace, z_log):
        x_node = x_node.reshape(x_node.shape[0], -1, x_node.shape[-1])
        z_node = z_node.reshape(z_node.shape[0], -1, z_node.shape[-1])
        x_node = self.attention_node(x_node, z_node, z_node)[0]

        x_trace = x_trace.reshape(x_trace.shape[0], -1, x_trace.shape[-1])
        z_trace = z_trace.reshape(z_trace.shape[0], -1, z_trace.shape[-1])
        x_trace = self.attention_trace(x_trace, z_trace, z_trace)[0]

        x_log = x_log.reshape(x_log.shape[0], -1, x_log.shape[-1])
        z_log = z_log.reshape(z_log.shape[0], -1, z_log.shape[-1])
        x_log = self.attention_log(x_log, z_log, z_log)[0]

        return x_node, x_trace, x_log


class Encoder(nn.Module):
    def __init__(self, graph, node_embedding, edge_embedding, log_embedding, node_heads, log_heads, edge_heads, n2e_heads, e2n_heads, dropout, batch_size, window_size, num_layer, trace2pod):
        super(Encoder, self).__init__()
        self.node_adj, self.node_efea, self.edge_adj, self.edge_efea = adj2adj(graph, batch_size, window_size, edge_embedding)
        self.L = num_layer
        self.batch_size = batch_size
        self.window_size = window_size

        self.spatial_attention = nn.ModuleList(
            [Spatial_Attention(node_embedding, edge_embedding, log_embedding,
                               heads_n2e=n2e_heads, heads_e2n=e2n_heads, dropout=dropout, batch_size=batch_size,
                               window_size=window_size) for _ in range(self.L)])
        self.sa_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])
        self.temporal_attention = nn.ModuleList(
            [Temporal_Attention(node_embedding, edge_embedding, log_embedding, trace2pod,
                                heads_node=node_heads, heads_edge=edge_heads, heads_log=log_heads, dropout=dropout, window_size=window_size,
                                batch_size=batch_size) for _ in range(self.L)])
        self.ta_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])
        self.ffn = nn.ModuleList([FFN(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])


    def forward(self, e_node, e_edge, e_log):
        e_edge = torch.masked_select(e_edge, self.node_efea.byte()) \
            .reshape(e_edge.shape[0], e_edge.shape[1], -1, e_edge.shape[-1])

        for i in range(self.L):
            e_node, e_edge, e_log = self.sa_add[i](e_node, e_edge, e_log, *self.spatial_attention[i](e_node, e_edge, e_log, self.node_adj, self.edge_adj, self.edge_efea))
            e_node, e_edge, e_log = self.ta_add[i](e_node, e_edge, e_log, *self.temporal_attention[i](e_node, e_edge, e_log))
            e_node, e_edge, e_log = self.ffn[i](e_node, e_edge, e_log)
        return e_node, e_edge, e_log


class Decoder(nn.Module):
    def __init__(self, graph, node_embedding, edge_embedding, log_embedding, node_heads, log_heads, edge_heads, n2e_heads, e2n_heads, dropout, batch_size, window_size, num_layer, trace2pod):
        super(Decoder, self).__init__()
        self.node_adj, self.node_efea, self.edge_adj, self.edge_efea = adj2adj(graph, batch_size, window_size, edge_embedding)

        self.L = num_layer
        self.batch_size = batch_size
        self.window_size = window_size

        self.spatial_attention = nn.ModuleList(
            [Spatial_Attention(node_embedding, edge_embedding, log_embedding,
                              heads_n2e=n2e_heads, heads_e2n=e2n_heads, dropout=dropout, batch_size=batch_size,
                               window_size=window_size) for _ in range(self.L)])
        self.sa_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])
        self.temporal_attention = nn.ModuleList(
            [Temporal_Attention(node_embedding, edge_embedding, log_embedding, trace2pod,
                                heads_node=node_heads, heads_edge=edge_heads, heads_log=log_heads, dropout=dropout, window_size=window_size,
                                batch_size=batch_size) for _ in range(self.L)])
        self.ta_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])
        self.cross_attention = nn.ModuleList(
            [Encoder_Decoder_Attention(node_embedding, edge_embedding, log_embedding, 
                                        heads_node=node_heads, heads_edge=edge_heads, heads_log=log_heads, dropout=dropout)
                                        for _ in range(self.L)])
        self.ca_add = nn.ModuleList([AddALL(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])     
        self.ffn = nn.ModuleList([FFN(node_embedding, edge_embedding, log_embedding, dropout) for _ in range(self.L)])

    def forward(self, d_node, d_edge, d_log, z_node, z_edge, z_log):
        d_edge = torch.masked_select(d_edge, self.node_efea.byte()) \
            .reshape(d_edge.shape[0], d_edge.shape[1], -1, d_edge.shape[-1])
        for i in range(self.L):
            d_node, d_edge, d_log = self.sa_add[i](d_node, d_edge, d_log, *self.spatial_attention[i](d_node, d_edge, d_log, self.node_adj, self.edge_adj, self.edge_efea))
            d_node, d_edge, d_log = self.ta_add[i](d_node, d_edge, d_log, *self.temporal_attention[i](d_node, d_edge, d_log, mask=True))
            d_node, d_edge, d_log = self.ca_add[i](d_node, d_edge, d_log, *self.cross_attention[i](d_node, d_edge, d_log, z_node, z_edge, z_log))
            d_node, d_edge, d_log = self.ffn[i](d_node, d_edge, d_log)
        return d_node, d_edge, d_log



class Embed(nn.Module):
    def __init__(self, raw_dim, embedding_dim, max_len=1000, dim=4):
        super(Embed, self).__init__()
        self.linear = nn.Linear(raw_dim, embedding_dim)
        self.dim = dim
        pe = torch.zeros((1, max_len, embedding_dim))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000,
                                                                                  torch.arange(0, embedding_dim, 2,
                                                                                               dtype=torch.float32) / embedding_dim)
        pe[:, :, 0::2] = torch.sin(X)
        pe[:, :, 1::2] = torch.cos(X)
        if dim == 4:
            pe = pe.unsqueeze(2)
        elif dim == 5:
            pe = pe.unsqueeze(2).unsqueeze(2)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = self.linear(X)
        if self.dim == 4:
                padding = (0, 0, 0, 0, 1, 0)
                X_new = F.pad(X, padding, "constant", 0)
                return X + Variable(self.pe[:, :X.shape[1], :, :], requires_grad=False), X_new[:, :X.shape[1], :, :] + Variable(self.pe[:, :X.shape[1], :, :], requires_grad=False)
        else:
                padding = (0, 0, 0, 0, 0, 0, 1, 0)
                X_new = F.pad(X, padding, "constant", 0)
                return X + Variable(self.pe[:, :X.shape[1], :, :, :], requires_grad=False), X_new[:, :X.shape[1], :, :, :] + Variable(self.pe[:, :X.shape[1], :, :, :], requires_grad=False)