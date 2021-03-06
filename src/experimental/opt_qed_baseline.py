import pathlib

import pandas as pd
from tqdm import trange

from src.agents import EpsilonGreedyAgent
from src.environments import QEDScaffoldDecorator
from src.utils.mol_utils import Molecule
from src.utils.seed_utils import seed_everything


def main():
    result_dir = pathlib.Path(__file__).parents[2] / "results" / "qed_baseline"
    result_dir.mkdir(exist_ok=True)

    init_mol = Molecule.from_smiles("[CH3:1][CH3:1]")
    env = QEDScaffoldDecorator(init_mol=init_mol)

    for epsilon in [0.0, 0.05, 0.1, 0.2, 1.0]:
        seed_everything(seed=498)
        agent = EpsilonGreedyAgent(epsilon)

        sampled = []
        n_samples = 1 if epsilon == 0.0 else 150
        for _ in trange(n_samples, desc=f"Eps={epsilon:.2f}"):
            mol, value = agent.rollout(env)
            qed = env.prop_fn(mol)
            sampled.append({"smiles": str(mol), "value": value, "qed": qed})
        sampled = pd.DataFrame(sampled)
        sampled.to_csv(result_dir / f"eps={epsilon}.csv", index=False)


if __name__ == "__main__":
    main()
