from rdkit import Chem

from src.utils.mol_utils import penalized_logp


def test_penalized_logp():
    s = "ClC1=CC=C2C(C=C(C(C)=O)C(C(NC3=CC(NC(NC4=CC(C5=C(C)C=CC=C5)=CC=C4)=O)=CC=C3)=O)=C2)=C1"
    assert round(penalized_logp(Chem.MolFromSmiles(s)), 2) == 5.30

    s = "CC(NC1=CC(C2=CC=CC(NC(NC3=CC=CC(C4=CC(F)=CC=C4)=C3)=O)=C2)=CC=C1)=O"
    assert round(penalized_logp(Chem.MolFromSmiles(s)), 2) == 4.49

    s = "ClC(C(Cl)=C1)=CC=C1NC2=CC=CC=C2C(NC(NC3=C(C(NC4=C(Cl)C=CC=C4)=S)C=CC=C3)=O)=O"
    assert round(penalized_logp(Chem.MolFromSmiles(s)), 2) == 4.93


if __name__ == "__main__":
    test_penalized_logp()
    print("Success.")
