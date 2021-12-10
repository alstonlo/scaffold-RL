from rdkit import Chem
from rdkit.Chem import QED

from src.utils.mol_utils import enum_molecule_mods, penalized_logp


class ScaffoldDecorator:

    def __init__(
            self,
            init_mol,
            atom_types,
            allowed_ring_sizes,
            max_mol_size,
            max_steps,
    ):
        self.init_mol = init_mol
        self.atom_types = atom_types
        self.allowed_ring_sizes = allowed_ring_sizes
        self.max_mol_size = max_mol_size
        self.max_steps = max_steps

        self._mol = None
        self._steps_left = self.max_steps
        self._valid_actions = set()

    @property
    def state(self):
        return self._mol, self._steps_left

    @property
    def valid_actions(self):
        return list(sorted(self._valid_actions))

    def reset(self):
        self._mol = self.init_mol
        self._steps_left = self.max_steps
        self._rebuild_valid_actions()

    def step(self, action):
        # Assumes action (next molecule) is valid
        # Returns (next state, reward, done)
        assert isinstance(action, str)

        if self._steps_left == 0:
            raise ValueError()
        elif action not in self._valid_actions:
            return ValueError()

        self._mol = action
        self._steps_left -= 1
        self._rebuild_valid_actions()
        reward = self._reward_fn()
        done = (self._steps_left == 0)
        return self.state, reward, done

    def _rebuild_valid_actions(self):
        self._valid_actions = enum_molecule_mods(
            smiles=self._mol,
            atom_types=self.atom_types,
            allowed_ring_sizes=self.allowed_ring_sizes,
            max_mol_size=self.max_mol_size
        )

    def _reward_fn(self):
        raise NotImplementedError()


class LogPScaffoldDecorator(ScaffoldDecorator):

    def __init__(
            self,
            init_mol,
            atom_types=("C", "O", "N"),
            allowed_ring_sizes=(5, 6, 7),
            max_mol_size=38,
            max_steps=40,
            gamma=0.9
    ):
        super().__init__(
            init_mol=init_mol,
            atom_types=atom_types,
            allowed_ring_sizes=allowed_ring_sizes,
            max_mol_size=max_mol_size,
            max_steps=max_steps,
        )

        self.gamma = gamma

    def _reward_fn(self):
        mol = Chem.MolFromSmiles(self._mol)
        assert mol is not None
        return penalized_logp(mol) * (self.gamma ** self._steps_left)


class QEDScaffoldDecorator(ScaffoldDecorator):

    def __init__(
            self,
            init_mol,
            atom_types=("C", "O", "N"),
            allowed_ring_sizes=(5, 6, 7),
            max_mol_size=38,
            max_steps=40,
            gamma=0.9
    ):
        super().__init__(
            init_mol=init_mol,
            atom_types=atom_types,
            allowed_ring_sizes=allowed_ring_sizes,
            max_mol_size=max_mol_size,
            max_steps=max_steps,
        )

        self.gamma = gamma

    def _reward_fn(self):
        mol = Chem.MolFromSmiles(self._mol)
        assert mol is not None
        return QED.qed(mol) * (self.gamma ** self._steps_left)
