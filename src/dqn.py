import dgl
import dgllife
import numpy as np
import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from rdkit import Chem

from src.utils.mol_utils import openness


class ScaffoldFeaturizer:

    def __init__(self, atom_types):
        self.atom_encodings = {element: i for i, element in enumerate(atom_types)}
        self.bond_featurizer = dgllife.utils.PretrainBondFeaturizer()

    def featurize(self, mol):
        return dgllife.utils.mol_to_bigraph(
            mol=mol,
            add_self_loop=True,
            node_featurizer=self._featurize_atoms,
            edge_featurizer=self._featurize_bonds,
            canonical_atom_order=True
        )

    def featurize_batch(self, mols):
        g = [self.featurize(m) for m in mols]
        bg = dgl.batch(g)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)

        node_feats = [bg.ndata.pop("element"), bg.ndata.pop("open")]
        edge_feats = [bg.edata.pop("order")]
        return bg, node_feats, edge_feats

    def _featurize_atoms(self, mol):
        atom_features = []
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            encoding = self.atom_encodings[atom.GetSymbol()]
            is_open = int(openness(atom))
            atom_features.append([encoding, is_open])
        atom_features = torch.tensor(np.stack(atom_features)).int()
        return {"element": atom_features[:, 0], "open": atom_features[:, 1]}

    def _featurize_bonds(self, mol):
        features = self.bond_featurizer(mol)
        return {"order": features["bond_type"]}


class ScaffoldDQN(torch.nn.Module):

    def __init__(
            self,
            atom_types,
            num_layers,
            emb_dim,
            dropout,
            device,
    ):
        super().__init__()
        self.atom_types = atom_types
        self.featurizer = ScaffoldFeaturizer(atom_types)
        self.device = device

        self.gnn = dgllife.model.GIN(
            num_node_emb_list=[len(atom_types), 2],
            num_edge_emb_list=[5],
            num_layers=num_layers,
            emb_dim=emb_dim,
            JK="last",
            dropout=dropout
        ).to(device)

        self.readout = GlobalAttentionPooling(gate_nn=nn.Linear(emb_dim, 1)).to(device)
        self.predict = nn.Sequential(
            nn.Linear(emb_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)

    def forward(self, states):
        smiles, steps = zip(*states)
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        steps = torch.tensor(steps, dtype=torch.float).unsqueeze(1).to(self.device)

        bg, node_feats, edge_feats = self.featurizer.featurize_batch(mols)
        bg = bg.to(self.device)
        node_feats = [x.to(self.device) for x in node_feats]
        edge_feats = [x.to(self.device) for x in edge_feats]

        node_feats = self.gnn(bg, node_feats, edge_feats)
        readout = self.readout(bg, node_feats)
        readout = torch.cat([readout, steps], dim=1)
        return self.predict(readout)
