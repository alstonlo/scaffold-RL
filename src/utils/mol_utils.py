import functools
import itertools
import os
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, RDConfig
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
#   https://github.com/google-research/google-research/blob/master/mol_dqn/chemgraph/dqn/py/molecules.py
# ==================================================================================================


def get_largest_ring_size(mol):
    cycle_list = mol.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def penalized_logp(mol):
    log_p = Descriptors.MolLogP(mol)
    sas_score = sascorer.calculateScore(mol)
    largest_ring_size = get_largest_ring_size(mol)
    cycle_score = max(largest_ring_size - 6, 0)
    return log_p - sas_score - cycle_score


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
        filters.append(steric_strain_filter)
    return all(f(mol) for f in filters)


def zinc_molecule_filter(mol):
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
    catalog = FilterCatalog(params)
    return not catalog.HasMatch(mol)


def steric_strain_filter(mol, cutoff=0.82, max_attempts_embed=20, max_num_iters=200):
    if mol.GetNumAtoms() <= 2:
        return True
    mol = Chem.AddHs(mol)

    # generate an initial 3d conformer
    try:
        flag = AllChem.EmbedMolecule(mol, maxAttempts=max_attempts_embed)
        if flag == -1:
            return False
    except:  # to catch error caused by molecules such as C=[SH]1=C2OC21ON(N)OC(=O)NO
        return False

    # set up the forcefield
    AllChem.MMFFSanitizeMolecule(mol)
    if AllChem.MMFFHasAllMoleculeParams(mol):
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        try:  # to deal with molecules such as CNN1NS23(=C4C5=C2C(=C53)N4Cl)S1
            ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
        except:
            return False
    else:
        return False

    # minimize steric energy
    try:
        ff.Minimize(maxIts=max_num_iters)
    except:
        return False

    # get the angle bend term contribution to the total molecule strain energy
    mmff_props.SetMMFFBondTerm(False)
    mmff_props.SetMMFFAngleTerm(True)
    mmff_props.SetMMFFStretchBendTerm(False)
    mmff_props.SetMMFFOopTerm(False)
    mmff_props.SetMMFFTorsionTerm(False)
    mmff_props.SetMMFFVdWTerm(False)
    mmff_props.SetMMFFEleTerm(False)

    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)

    min_angle_e = ff.CalcEnergy()

    # find number of angles in molecule
    num_atoms = mol.GetNumAtoms()
    atom_indices = range(num_atoms)
    angle_atom_triplets = itertools.permutations(atom_indices, 3)
    double_num_angles = 0
    for triplet in list(angle_atom_triplets):
        if mmff_props.GetMMFFAngleBendParams(mol, *triplet):
            double_num_angles += 1
    num_angles = double_num_angles / 2  # account for duplicate angles

    avr_angle_e = min_angle_e / num_angles
    if avr_angle_e < cutoff:
        return True
    else:
        return False


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
            if not master_filter(next_mol, steric=False):
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
