import torch
import torch.nn as nn

from frame2seq.openfold.model.primitives import Linear


# code credit: https://github.com/jasonkyuyim/se3_diffusion
class EdgeTransition(nn.Module):
    """
    Edge update operation.
    """

    def __init__(self,
                 node_embed_size,
                 edge_embed_in,
                 edge_embed_out,
                 num_layers=2,
                 node_dilation=2):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = Linear(node_embed_size,
                                    bias_embed_size,
                                    init="relu")
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, node_embed, edge_embed):
        node_embed = self.initial_embed(node_embed)
        batch_size, num_res, _ = node_embed.shape
        edge_bias = torch.cat([
            torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
        ],
                              axis=-1)
        edge_embed = torch.cat([edge_embed, edge_bias],
                               axis=-1).reshape(batch_size * num_res**2, -1)
        edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
        edge_embed = self.layer_norm(edge_embed)
        edge_embed = edge_embed.reshape(batch_size, num_res, num_res, -1)
        return edge_embed
