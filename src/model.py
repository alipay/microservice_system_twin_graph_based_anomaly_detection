import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from src.model_util import *


class MyModel(nn.Module):
	def __init__(self, graph, **args):
		super(MyModel, self).__init__()
		self.name = 'my'
		self.graph = torch.tensor(graph).cuda()
		self.label_weight = args['label_weight']

		adj = dense_to_sparse(self.graph)[0]
		trace2pod = torch.nn.functional.one_hot(adj[0], num_classes=graph.shape[0]) \
			+ torch.nn.functional.one_hot(adj[1], num_classes=graph.shape[0])
		trace2pod = trace2pod / trace2pod.sum(axis=0, keepdim=True)
		trace2pod = torch.where(torch.isnan(
			trace2pod), torch.full_like(trace2pod, 0), trace2pod)

		self.encoder = Encoder(graph=self.graph, node_embedding=args['feature_node'], edge_embedding=args['feature_edge'], log_embedding=args['feature_log'],
                         node_heads=args['num_heads_node'], log_heads=args['num_heads_log'], edge_heads=args['num_heads_edge'],
                         n2e_heads=args['num_heads_n2e'], e2n_heads=args['num_heads_e2n'],
                         dropout=args['dropout'], batch_size=args['batch_size'], window_size=args['window'], num_layer=args['num_layer'], trace2pod=trace2pod)
		self.decoder = Decoder(graph=self.graph, node_embedding=args['feature_node'], edge_embedding=args['feature_edge'], log_embedding=args['feature_log'],
                         node_heads=args['num_heads_node'], log_heads=args['num_heads_log'], edge_heads=args['num_heads_edge'],
                         n2e_heads=args['num_heads_n2e'], e2n_heads=args['num_heads_e2n'],
                         dropout=args['dropout'], batch_size=args['batch_size'], window_size=args['window'], num_layer=args['num_layer'], trace2pod=trace2pod)

		self.node_emb = Embed(args['raw_node'], args['feature_node'], dim=4)
		self.log_emb = Embed(args['log_len'], args['feature_log'], dim=4)
		self.egde_emb = Embed(args['raw_edge'], args['feature_edge'], dim=5)

		self.trace2pod = torch.nn.functional.one_hot(adj[0], num_classes=self.graph.shape[0]) \
			+ torch.nn.functional.one_hot(adj[1], num_classes=self.graph.shape[0])
		self.trace2pod = self.trace2pod / 2

		self.dense_node = nn.Linear(args['feature_node'], args['raw_node'])
		self.dense_log = nn.Linear(args['feature_log'], args['log_len'])
		self.dense_edge = nn.Linear(args['feature_edge'], args['raw_edge'])

		self.show = nn.Sequential(nn.Linear(args['raw_node'] + args['raw_edge'] + args['log_len'], (args['raw_node'] + args['raw_edge'] + args['log_len']) // 2),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear((args['raw_node'] + args['raw_edge'] + args['log_len']) // 2, 2))

	def forward(self, x, evaluate=False):
		x_node, d_node = self.node_emb(x['data_node'])
		x_edge, d_edge = self.egde_emb(x['data_edge'])
		x_log, d_log = self.log_emb(x['data_log'])

		z_node, z_edge, z_log = self.encoder(x_node, x_edge, x_log)
		node, edge, log = self.decoder(d_node, d_edge, d_log, z_node, z_edge, z_log)
        
		l_edge = torch.masked_select(x['data_edge'], self.graph.unsqueeze(-1).repeat(1, 1, x['data_edge'].shape[-1]).byte()) \
			.reshape(x['data_edge'].shape[0], x['data_edge'].shape[1], -1, x['data_edge'].shape[-1])

		rec_node = torch.square(self.dense_node(node) - x['data_node'])
		rec_edge1 = torch.square(self.dense_edge(edge) - l_edge)
		rec_log = torch.square(self.dense_log(log) - x['data_log'])

		rec_edge = torch.matmul(rec_edge1.permute(
			0, 1, 3, 2), self.trace2pod.float()).permute(0, 1, 3, 2)
		rec = torch.concat([rec_node, rec_log, rec_edge], dim=-1)
		if evaluate:
			rec = rec[:, -1].squeeze()
			cls_result = torch.softmax(self.show(rec), dim=-1)
			return cls_result, x['groundtruth_cls']
		else:
			cls_label = x['groundtruth_cls']

			#cls_label
			rec = rec[:, -1].squeeze()
			cls_result = self.show(rec)
			cls_result = cls_result.reshape(-1, cls_result.shape[-1])
			cls_label = cls_label.reshape(-1, cls_label.shape[-1])

			if cls_label.shape[-1] == 3:
				mask = cls_label[:, -1]
				cls_result, cls_label = cls_result[mask == 0], cls_label[mask == 0]
				cls_label = cls_label[:, :cls_result.shape[-1]]

			# rec_loss
			label_pod = torch.argmax(x['groundtruth_cls'], dim=-1)  # B*N

			node_rec = torch.sum(rec, dim=-1)
			node_right = torch.where(label_pod == 0, node_rec,
			                         torch.zeros_like(node_rec).to(node_rec.device))
			node_wrong = torch.where(label_pod == 1, torch.pow(node_rec, torch.tensor(
				-1, device=node_rec.device)), torch.zeros_like(node_rec).to(node_rec.device))
			node_unkown = torch.where(label_pod == 2, self.label_weight *
			                          node_rec, torch.zeros_like(node_rec).to(node_rec.device))
			rec_loss = [node_right, node_wrong, node_unkown]


			param = label_pod.shape[0] * label_pod.shape[1]
			rec_loss = list(map(lambda x: x.sum() / param, rec_loss))

			return rec_loss, cls_result, cls_label
