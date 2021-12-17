import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors

from src.utils.mol_utils import openness, set_openness


def morgan_fingerprint(mol, use_base=True):
    rdkmol = mol.rdkmol
    if not use_base:
        clone = Chem.Mol(mol.rdkmol)
        for atom in clone.GetAtoms():
            if openness(atom):
                atom.SetIsotope(1)  # TODO: kind of a hack
                set_openness(atom, is_open=False)
        rdkmol = clone
    fp_arr = np.zeros((0,), dtype=np.int8)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(rdkmol, radius=3, nBits=2048)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


class ScaffoldDQN(nn.Module):

    @classmethod
    def featurize_batch(cls, states):
        assert isinstance(states, list)
        mols, steps = zip(*states)
        steps = np.expand_dims(np.array(steps), axis=1)

        encodings = []
        for mol in mols:
            fp1 = morgan_fingerprint(mol, use_base=True)
            fp2 = morgan_fingerprint(mol, use_base=False)
            fp = np.concatenate([fp1, fp2], axis=0)
            encodings.append(fp)
        encodings = np.stack(encodings, axis=0)
        encodings = np.concatenate([encodings, steps], axis=1)
        return encodings

    def __init__(self, device):
        super().__init__()
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(4097, 2056),
            nn.ReLU(),
            nn.Linear(2056, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, states):
        if isinstance(states, list):
            states = self.featurize_batch(states)
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float, device=self.device)
        return self.model(states)
