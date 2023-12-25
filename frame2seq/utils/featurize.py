import math
import torch
import torch.nn.functional as F
from einops import repeat, rearrange


def make_s_init(self, X, input_S, seq_mask):
    """
    Generate the initial sequence embedding.
    """
    batch_size, seq_len, _, _ = X.shape
    seq_mask = seq_mask.to(X.device)
    input_S = input_S.to(X.device)

    def process_input_S(input_S):
        input_S_mask = torch.ones(input_S.shape[0], input_S.shape[1],
                                  1).to(input_S.device)
        input_S_ints = torch.argmax(input_S, dim=-1)
        input_S_mask[input_S_ints == 20] = 0
        input_S = self.input_sequence_to_single(input_S)
        in_S = input_S * input_S_mask
        return in_S

    def absolute_positional_emb(seq_len, dim):
        """
        Generate absolute positional embeddings.
        """
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe  # (L, D)

    # code credit: https://github.com/jingraham/neurips19-graph-protein-design
    def _dihedrals(X, eps=1e-7):
        """
        Compute dihedral angles from a set of coordinates.
        """
        X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        D = F.pad(D, (1, 2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1) / 3), 3))
        phi, psi, omega = torch.unbind(D, -1)
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features  # (B, L, 6), the 6 is cos(phi), sin(phi), cos(psi), sin(psi), cos(omega), sin(omega)

    in_S = process_input_S(input_S)  # (B, L, D)
    in_S = in_S.to(X.device)
    d_feat = _dihedrals(X).float()  # (B, L, 6)
    s_pos_emb = absolute_positional_emb(seq_len, self.single_dim)
    s_pos_emb = repeat(s_pos_emb, 'l d -> b l d', b=batch_size)  # (B, L, D)
    s_pos_emb = s_pos_emb.to(X.device)
    s_init = torch.cat([s_pos_emb, d_feat], dim=-1)  # (B, L, D+6)
    return s_init, in_S


def make_z_init(self, X):
    """
    Generate the initial pairwise embedding.
    """

    def relative_pairwise_position_idx(seq_len):
        """
        Generate relative pairwise position indices.
        """
        indices = torch.arange(seq_len, dtype=torch.long)
        indices = indices[:, None] - indices[None, :]
        indices = indices.clamp(-self.relpos_k, self.relpos_k)
        indices = indices + self.relpos_k
        return indices

    # code credit: https://github.com/jingraham/neurips19-graph-protein-design
    def rbf(D):
        """
        Radial basis functions.
        """
        device = D.device
        D_min, D_max, D_count = 0., self.dist_bins * self.dist_bin_width, self.dist_bins
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    batch_size = X.shape[0]
    relpos = relative_pairwise_position_idx(X.shape[1])
    relpos = F.one_hot(relpos, 2 * self.relpos_k + 1).float()
    relpos = relpos.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(X.device)
    X_bb = X[:, :, :4]
    X_bb = rearrange(X_bb, 'b n c d -> b (n c) d', c=4)
    pairwise_distances = torch.cdist(X_bb, X_bb)
    RBF = rbf(pairwise_distances)
    RBF = rearrange(RBF,
                    "b (n1 c1) (n2 c2) d -> b n1 n2 (c1 c2 d)",
                    c1=4,
                    c2=4).to(X.device)
    z = torch.cat([RBF, relpos], dim=-1)
    z = z.float()

    return z
