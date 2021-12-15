import functools
import itertools

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint


class Molecule:

    @classmethod
    def from_smiles(cls, smiles):
        return Molecule(Chem.MolFromSmiles(smiles))

    def __init__(self, rdkmol):
        self.smiles = Chem.MolToSmiles(rdkmol)

    def __str__(self):
        return self.base_smiles

    def __hash__(self):
        return hash(self.smiles)

    def __eq__(self, other):
        return self.smiles == other.smiles

    @functools.cached_property
    def rdkmol(self):
        return Chem.MolFromSmiles(self.smiles)

    @functools.cached_property
    def base_smiles(self):
        clone = Chem.Mol(self.rdkmol)
        for atom in clone.GetAtoms():
            if openness(atom):
                set_openness(atom, is_open=False)
        return Chem.MolToSmiles(clone)

    def base_copy(self):
        return Molecule.from_smiles(self.base_smiles)


# ==================================================================================================
# Misc.
# ==================================================================================================


@functools.lru_cache()
def element_valence(element):
    ptable = Chem.GetPeriodicTable()
    return max(ptable.GetValenceList(element))


def update_explicit_Hs(atom, offset):
    explicit_Hs = atom.GetNumExplicitHs()
    atom.SetNumExplicitHs(explicit_Hs + offset)


# ==================================================================================================
# Molecular Filers
# References:
#   https://github.com/bowenliu16/rl_graph_generation/blob/master/gym-molecule/gym_molecule/envs/molecule.py
# ==================================================================================================


def master_filter(mol, zinc=True, steric=True):
    filters = list()
    if zinc:
        filters.append(zinc_molecule_filter)
    if steric:
        filters.append(bicycle_filter)
        # TODO: add MMFF94 forcefield filter?
    return all(f(mol) for f in filters)


def zinc_molecule_filter(mol):
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
    catalog = FilterCatalog(params)
    return not catalog.HasMatch(mol)


def bicycle_filter(mol):
    cycles = list(set(ring) for ring in Chem.GetSymmSSSR(mol))
    for c1, c2 in itertools.combinations(cycles, r=2):
        if len(c1 & c2) > 2:
            return False
    return True


# ==================================================================================================
# Scaffold Builder Utils
# ==================================================================================================


def openness(atom):
    return atom.GetAtomMapNum() == 1


def set_openness(atom, is_open):
    atom.SetAtomMapNum(1 if is_open else 0)


def close_open_ends(rdkmol, idxs=None):
    if idxs is None:
        open_ends = rdkmol.GetAtoms()
    else:
        open_ends = [rdkmol.GetAtomWithIdx(idx) for idx in idxs]

    for atom in open_ends:
        if atom.GetTotalNumHs() == 0:
            set_openness(atom, is_open=False)


def enum_molecule_mods(mol, atom_types, allowed_ring_sizes, max_mol_size):
    open_idxs = list()
    for atom in mol.rdkmol.GetAtoms():
        if openness(atom):
            assert atom.GetNumImplicitHs() == 0
            open_idxs.append(atom.GetIdx())

    actions = set()
    if open_idxs:
        actions.update(_enum_atom_additions(mol.rdkmol, open_idxs, atom_types, max_mol_size))
        actions.update(_enum_bond_additions(mol.rdkmol, open_idxs, allowed_ring_sizes))
    actions.add(mol if actions else mol.base_copy())
    return actions


# FIXME: kind of hacky, but works well within scope
@functools.lru_cache()
def _enum_neighborhoods(free_bonds, atom_types):
    assert 0 < free_bonds <= 3
    sum_to = {  # enumerate all ways to sum to <key> using natural numbers
        1: [(1,)],
        2: [(1, 1), (2,)],
        3: [(1, 1, 1), (2, 1), (3,)]
    }

    attachments = {1: [], 2: [], 3: []}  # atoms that can make <key> bonds
    for element in atom_types:
        max_order = min(3, element_valence(element))
        for order in range(1, max_order + 1):
            attachments[order].append((order, element))

    neighborhoods = set()
    for bonds_used in range(1, free_bonds + 1):
        for config in sum_to[bonds_used]:
            grid = [attachments[i] for i in config]
            for nbd in itertools.product(*grid):
                neighborhoods.add(tuple(sorted(nbd)))
    return tuple(neighborhoods)


def _enum_atom_additions(rdkmol, open_idxs, atom_types, max_mol_size):
    mol_size = len(rdkmol.GetAtoms())
    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]

    atom_additions = set()

    for idx in open_idxs:
        free_bonds = rdkmol.GetAtomWithIdx(idx).GetNumExplicitHs()
        assert free_bonds > 0

        for nbd in _enum_neighborhoods(free_bonds, atom_types):
            if mol_size + len(nbd) > max_mol_size:
                continue  # adding nbd would overflow size

            next_mol = Chem.RWMol(rdkmol)
            atom = next_mol.GetAtomWithIdx(idx)

            added_idxs = []
            for order, element in nbd:
                update_explicit_Hs(atom, -order)
                new_atom = Chem.Atom(element)
                set_openness(new_atom, is_open=True)

                add_idx = next_mol.AddAtom(new_atom)
                next_mol.AddBond(idx, add_idx, bond_orders[order])
                added_idxs.append(add_idx)

            if Chem.SanitizeMol(next_mol, catchErrors=True):
                continue  # sanitization failed
            if not master_filter(next_mol, steric=False):
                continue  # failed to pass filters

            set_openness(atom, is_open=False)
            close_open_ends(next_mol, added_idxs)
            atom_additions.add(Molecule(next_mol))

    return atom_additions


def _enum_bond_additions(rdkmol, open_idxs, allowed_ring_sizes):
    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]

    bond_additions = set()

    for idx1, idx2 in itertools.combinations(open_idxs, 2):
        atom1 = rdkmol.GetAtomWithIdx(idx1)
        atom2 = rdkmol.GetAtomWithIdx(idx2)

        # MolDQN heuristics
        if rdkmol.GetBondBetweenAtoms(idx1, idx2) is not None:
            continue  # disallow bonds between already-bonded atoms
        elif atom1.IsInRing() and atom2.IsInRing():
            continue  # disallow bonds between atoms already in rings
        elif len(Chem.rdmolops.GetShortestPath(rdkmol, idx1, idx2)) not in allowed_ring_sizes:
            continue  # bond formation will make disallowed ring

        free_bonds = min(atom1.GetNumExplicitHs(), atom2.GetNumExplicitHs())
        assert free_bonds > 0

        for order in range(1, free_bonds + 1):
            next_mol = Chem.RWMol(rdkmol)
            update_explicit_Hs(next_mol.GetAtomWithIdx(idx1), -order)
            update_explicit_Hs(next_mol.GetAtomWithIdx(idx2), -order)
            next_mol.AddBond(idx1, idx2, bond_orders[order])

            if Chem.SanitizeMol(next_mol, catchErrors=True):
                continue  # sanitization failed
            if not master_filter(next_mol):
                continue  # failed to pass filters

            close_open_ends(next_mol, [idx1, idx2])
            bond_additions.add(Molecule(next_mol))

    return bond_additions


# ==================================================================================================
# Metrics
# ==================================================================================================


def validity(mols):  # validity guaranteed, but sanity check
    valid = [m for m in mols if (Chem.MolFromSmiles(m.base_smiles) is not None)]
    return len(valid) / len(mols)


def uniqueness(mols):
    base_smiles = [m.base_smiles for m in mols]
    return len(set(base_smiles)) / len(base_smiles)


def molecule_similarity(mol1, mol2):
    vec1 = GetMorganFingerprint(mol1.rdkmol, radius=2)
    vec2 = GetMorganFingerprint(mol2.rdkmol, radius=2)
    return DataStructs.TanimotoSimilarity(vec1, vec2)


def pairwise_diversities(mols):
    scores = []
    for mol1, mol2 in itertools.combinations(mols, r=2):
        scores.append(1 - molecule_similarity(mol1, mol2))
    return scores
