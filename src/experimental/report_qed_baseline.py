import pathlib
import statistics

import pandas as pd

from src.utils.mol_utils import *


def main():
    result_dir = pathlib.Path(__file__).parents[2] / "results" / "qed_baseline"
    result_dir.mkdir(exist_ok=True)

    for epsilon in [0.0, 0.05, 0.1, 0.2, 1.0]:
        print(f"Epsilon={epsilon}")

        # Cut down number of samples to compare fairly with previous works
        sampled = pd.read_csv(result_dir / f"eps={epsilon}.csv")
        smiles = list(sampled["smiles"])[:100]
        values = list(sampled["value"])[:100]
        qeds = list(sampled["qed"])[:100]

        if epsilon == 0.0:
            smiles = smiles * 150

        print(f"\tQED:      {statistics.mean(qeds):.3f} +- {statistics.pstdev(qeds):.3f}")
        print(f"\tValue:    {statistics.mean(values):.3f} +- {statistics.pstdev(values):.3f}")

        mols = [Molecule.from_smiles(s) for s in smiles]
        print(f"\tValid:    {validity(mols)}")
        print(f"\tUnique:   {uniqueness(mols):.3f}")
        print(f"\tDiverse:  {statistics.mean(pairwise_diversities(mols)):.3f}")

        top3 = list(sorted(qeds, reverse=True))[:3]
        print(f"\tTop 3: {top3}")

        print()


if __name__ == "__main__":
    main()
