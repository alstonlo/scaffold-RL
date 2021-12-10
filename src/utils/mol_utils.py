import functools
import itertools
import os
import sys

import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


# ==================================================================================================
# Misc.
# ==================================================================================================


@functools.lru_cache()
def element_valence(element):
    ptable = Chem.GetPeriodicTable()
    return max(ptable.GetValenceList(element))


def update_explicit_Hs(atom, delta):
    explicit_Hs = atom.GetNumExplicitHs()
    atom.SetNumExplicitHs(explicit_Hs + delta)


# ==================================================================================================
# Penalized logP
# Reference:
#   https://github.com/bowenliu16/rl_graph_generation/blob/master/gym-molecule/gym_molecule/envs/molecule.py
# ==================================================================================================


def penalized_logp(mol):
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


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
        pass  # TODO: temporarily removed because slow
    return all(f(mol) for f in filters)


def zinc_molecule_filter(mol):
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
    catalog = FilterCatalog(params)
    return not catalog.HasMatch(mol)


# ==================================================================================================
# Scaffold Builder Utils
# ==================================================================================================


def openness(atom):
    return atom.GetAtomMapNum() == 1


def set_openness(atom, is_open):
    atom.SetAtomMapNum(1 if is_open else 0)


def close_open_ends(mol, idxs=None):
    if idxs is None:
        open_ends = mol.GetAtoms()
    else:
        open_ends = [mol.GetAtomWithIdx(idx) for idx in idxs]

    for atom in open_ends:
        if atom.GetTotalNumHs() == 0:
            set_openness(atom, is_open=False)


def enum_molecule_mods(smiles, atom_types, allowed_ring_sizes, max_mol_size):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError()

    open_idxs = list()
    for atom in mol.GetAtoms():
        if openness(atom):
            assert atom.GetNumImplicitHs() == 0
            open_idxs.append(atom.GetIdx())

    actions = set()
    actions.add(smiles)
    if open_idxs:
        actions.update(_enum_atom_additions(mol, open_idxs, atom_types, max_mol_size))
        actions.update(_enum_bond_additions(mol, open_idxs, allowed_ring_sizes))
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


def _enum_atom_additions(mol, open_idxs, atom_types, max_mol_size):
    mol_size = len(mol.GetAtoms())
    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]

    atom_additions = set()

    for idx in open_idxs:
        free_bonds = mol.GetAtomWithIdx(idx).GetNumExplicitHs()
        assert free_bonds > 0

        for nbd in _enum_neighborhoods(free_bonds, atom_types):
            if mol_size + len(nbd) > max_mol_size:
                continue  # adding nbd would overflow size

            next_mol = Chem.RWMol(mol)
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
            if not master_filter(next_mol):
                continue  # failed to pass filters

            set_openness(atom, is_open=False)
            close_open_ends(next_mol, added_idxs)
            atom_additions.add(Chem.MolToSmiles(next_mol))

    return atom_additions


def _enum_bond_additions(mol, open_idxs, allowed_ring_sizes):
    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]

    bond_additions = set()

    for idx1, idx2 in itertools.combinations(open_idxs, 2):
        atom1 = mol.GetAtomWithIdx(idx1)
        atom2 = mol.GetAtomWithIdx(idx2)

        # MolDQN heuristics
        if mol.GetBondBetweenAtoms(idx1, idx2) is not None:
            continue  # disallow bonds between already-bonded atoms
        elif atom1.IsInRing() and atom2.IsInRing():
            continue  # disallow bonds between atoms already in rings
        elif len(Chem.rdmolops.GetShortestPath(mol, idx1, idx2)) not in allowed_ring_sizes:
            continue  # bond formation will make disallowed ring

        free_bonds = min(atom1.GetNumExplicitHs(), atom2.GetNumExplicitHs())
        assert free_bonds > 0

        for order in range(1, free_bonds + 1):
            next_mol = Chem.RWMol(mol)
            update_explicit_Hs(next_mol.GetAtomWithIdx(idx1), -order)
            update_explicit_Hs(next_mol.GetAtomWithIdx(idx2), -order)
            next_mol.AddBond(idx1, idx2, bond_orders[order])

            if Chem.SanitizeMol(next_mol, catchErrors=True):
                continue  # sanitization failed
            if not master_filter(next_mol):
                continue  # failed to pass filters

            close_open_ends(next_mol, [idx1, idx2])
            bond_additions.add(Chem.MolToSmiles(next_mol))

    return bond_additions


# ==================================================================================================
# Debugging
# ==================================================================================================


def clean_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if openness(atom):
            set_openness(atom, is_open=False)
    return Chem.MolToSmiles(mol)
