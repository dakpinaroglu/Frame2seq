import torch


def place_fourth_atom(
    a_coord: torch.Tensor,
    b_coord: torch.Tensor,
    c_coord: torch.Tensor,
    length: torch.Tensor,
    planar: torch.Tensor,
    dihedral: torch.Tensor,
) -> torch.Tensor:
    """
    Given 3 coords + a length + a planar angle + a dihedral angle, compute a fourth coord
    """
    bc_vec = b_coord - c_coord
    bc_vec = bc_vec / bc_vec.norm(dim=-1, keepdim=True)

    n_vec = (b_coord - a_coord).expand(bc_vec.shape).cross(bc_vec)
    n_vec = n_vec / n_vec.norm(dim=-1, keepdim=True)

    m_vec = [bc_vec, n_vec.cross(bc_vec), n_vec]
    d_vec = [
        length * torch.cos(planar),
        length * torch.sin(planar) * torch.cos(dihedral),
        -length * torch.sin(planar) * torch.sin(dihedral)
    ]

    d_coord = c_coord + sum([m * d for m, d in zip(m_vec, d_vec)])

    return d_coord


def place_missing_cb(atom_positions):
    cb_coords = place_fourth_atom(atom_positions[:, 2], atom_positions[:, 0],
                                atom_positions[:, 1], torch.tensor(1.522),  
                                torch.tensor(1.927), torch.tensor(-2.143))
    cb_coords = torch.where(torch.isnan(cb_coords), torch.zeros_like(cb_coords), cb_coords)

    atom_positions[:, 3] = cb_coords
    return atom_positions


def place_missing_o(atom_positions):
    o_coords = place_fourth_atom(
        torch.roll(atom_positions[:, 0], shifts=-1, dims=0), atom_positions[:, 1],
        atom_positions[:, 2], torch.tensor(1.231), torch.tensor(2.108),
        torch.tensor(-3.142))
    o_coords = torch.where(torch.isnan(o_coords), torch.zeros_like(o_coords), o_coords)

    atom_positions[:, 4] = o_coords
    return atom_positions
