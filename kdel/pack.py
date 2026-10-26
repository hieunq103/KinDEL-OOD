import numpy as np, os
from tqdm import tqdm
from kdel.featurize import smiles_to_arrays, NODE_DIM, EDGE_DIM

def pack(smiles_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    xs, eis, eas, nptr, eptr = [], [], [], [0], [0]
    for s in tqdm(smiles_list, desc="featurize"):
        x, ei, ea = smiles_to_arrays(s)
        xs.append(x); eis.append(ei); eas.append(ea)
        nptr.append(nptr[-1] + x.shape[0])
        eptr.append(eptr[-1] + ea.shape[0])
    np.save(f"{out_dir}/graphs_x_fp16.npy",   np.concatenate(xs))
    np.save(f"{out_dir}/graphs_ei_int16.npy", np.concatenate(eis, axis=1))
    np.save(f"{out_dir}/graphs_ea_fp16.npy",  np.concatenate(eas))
    np.save(f"{out_dir}/graph_nptr.npy", np.asarray(nptr, np.int64))
    np.save(f"{out_dir}/graph_eptr.npy", np.asarray(eptr, np.int64))
    print("da ghi", len(smiles_list), "do thi vao", out_dir)
