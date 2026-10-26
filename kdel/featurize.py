import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

SYMBOLS = ["C","N","O","S","F","P","Cl","Br","I","Other"]
DEGREES = [0,1,2,3,4]
CHARGES = [-1,0,1]
HYBRID  = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
           Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D]
NUM_HS  = [0,1,2,3]
BONDS   = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
           Chem.BondType.TRIPLE,  Chem.BondType.AROMATIC]

NODE_DIM = len(SYMBOLS)+len(DEGREES)+len(CHARGES)+len(HYBRID)+len(NUM_HS)+2
EDGE_DIM = len(BONDS)+2

def _oh(v, allowed):
    x = [0.0]*len(allowed)
    x[allowed.index(v) if v in allowed else len(allowed)-1] = 1.0
    return x

def atom_feat(a):
    return (_oh(a.GetSymbol(), SYMBOLS) + _oh(a.GetDegree(), DEGREES)
            + _oh(a.GetFormalCharge(), CHARGES) + _oh(a.GetHybridization(), HYBRID)
            + _oh(a.GetTotalNumHs(), NUM_HS)
            + [float(a.GetIsAromatic()), float(a.IsInRing())])

def bond_feat(b):
    return (_oh(b.GetBondType(), BONDS)
            + [float(b.GetIsConjugated()), float(b.IsInRing())])

def smiles_to_arrays(smi):
    """-> (x fp16 [n,NODE_DIM], edge_index int16 [2,m], edge_attr fp16 [m,EDGE_DIM])"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return (np.zeros((1,NODE_DIM), np.float16),
                np.zeros((2,0), np.int16),
                np.zeros((0,EDGE_DIM), np.float16))
    x = np.array([atom_feat(a) for a in mol.GetAtoms()], dtype=np.float16)
    src, dst, ea = [], [], []
    for b in mol.GetBonds():
        i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_feat(b)
        src += [i, j]; dst += [j, i]; ea += [f, f]        # vo huong
    ei = np.array([src, dst], dtype=np.int16) if src else np.zeros((2,0), np.int16)
    ea = np.array(ea, dtype=np.float16) if ea else np.zeros((0,EDGE_DIM), np.float16)
    return x, ei, ea
