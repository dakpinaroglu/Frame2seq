import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl

from frame2seq.utils.rigid_utils import Rigid
from frame2seq.openfold.model.primitives import LayerNorm
from frame2seq.openfold.model.structure_module import InvariantPointAttention, StructureModuleTransition
from frame2seq.model.edge_update import EdgeTransition
from frame2seq.utils.featurize import make_s_init, make_z_init


class frame2seq(pl.LightningModule):

    def __init__(self, config):

        super(frame2seq, self).__init__()

        self.save_hyperparameters()
        config = self.hparams.config
        self.config = config

        ipa_depth = config['ipa_depth']
        ipa_dim = config['ipa_dim']
        ipa_heads = config['ipa_heads']
        ipa_pairwise_repr_dim = config['ipa_pairwise_repr_dim']

        self.st_mod_tsit_factor = config['st_mod_tsit_factor']

        self.sequence_dim = config['sequence_dim']
        self.single_dim = config['single_dim']

        self.torsion_bin_width = 8
        self.torsion_bins = 360 // self.torsion_bin_width
        self.relpos_k = 32
        self.dist_bin_width = 0.5
        self.dist_bins = 24

        self.pair_dim = 16 * self.dist_bins + 2 * self.relpos_k + 1

        self.sequence_to_single = nn.Linear(6 + self.single_dim,
                                            self.single_dim)
        self.edge_to_pair = nn.Linear(self.pair_dim, ipa_pairwise_repr_dim)
        self.single_to_sequence = nn.Linear(self.single_dim, self.sequence_dim)

        self.layers = nn.ModuleList([])
        for i in range(ipa_depth):

            ipa = InvariantPointAttention(
                ipa_dim,
                ipa_pairwise_repr_dim,
                ipa_dim // ipa_heads,
                ipa_heads,
                4,
                8,
            )
            ipa_dropout = nn.Dropout(0.1)
            layer_norm_ipa = LayerNorm(ipa_dim)
            if self.st_mod_tsit_factor > 1:
                pre_transit = nn.Linear(ipa_dim,
                                        ipa_dim * self.st_mod_tsit_factor)
                post_transit = nn.Linear(ipa_dim * self.st_mod_tsit_factor,
                                         ipa_dim)
            transition = StructureModuleTransition(
                ipa_dim * self.st_mod_tsit_factor,
                1,
                0.1,
            )
            if i == ipa_depth - 1:
                edge_transition = None
            else:
                edge_transition = EdgeTransition(
                    ipa_dim,
                    ipa_pairwise_repr_dim,
                    ipa_pairwise_repr_dim,
                    num_layers=2,
                )
            if self.st_mod_tsit_factor > 1:
                self.layers.append(
                    nn.ModuleList([
                        ipa, ipa_dropout, layer_norm_ipa, pre_transit,
                        transition, post_transit, edge_transition
                    ]))
            else:
                self.layers.append(
                    nn.ModuleList([
                        ipa, ipa_dropout, layer_norm_ipa, transition,
                        edge_transition
                    ]))
        self.s_dropout = nn.Dropout(0.1)
        self.z_dropout = nn.Dropout(0.1)
        self.input_sequence_to_single = nn.Linear(self.sequence_dim,
                                                  self.single_dim)
        self.input_sequence_layer_norm = nn.LayerNorm(self.single_dim)

    def forward(self, X, seq_mask, input_S):

        training_bool = self.training
        X = X.to(self.device)
        seq_mask = seq_mask.to(self.device)
        input_S = input_S.to(self.device)
        r = Rigid.from_3_points(X[:, :, 0], X[:, :, 1], X[:, :, 2])
        s, in_S = make_s_init(self, X, input_S, seq_mask)
        s = self.sequence_to_single(s)
        s = s + self.input_sequence_layer_norm(in_S)
        z = make_z_init(self, X)
        z = self.edge_to_pair(z)
        seq_mask = seq_mask.long()

        attn_drop_rate = 0.0
        if training_bool:
            attn_drop_rate = 0.2
            s = self.s_dropout(s)
            z = self.z_dropout(z)

        for ipa, ipa_dropout, layer_norm_ipa, *transit_layers, edge_transition in self.layers:
            s = s + ipa(s, z, r, seq_mask, attn_drop_rate=attn_drop_rate)
            s = ipa_dropout(s)
            s = layer_norm_ipa(s)

            if self.st_mod_tsit_factor > 1:
                pre_transit = transit_layers[0]
                transition = transit_layers[1]
                post_transit = transit_layers[2]
                s = pre_transit(s)
                s = transition(s)
                s = post_transit(s)
            else:
                transition = transit_layers[0]
                s = transition(s)

            if edge_transition is not None:
                z = checkpoint(edge_transition, s, z)

        pred_seq = self.single_to_sequence(s)

        return pred_seq
