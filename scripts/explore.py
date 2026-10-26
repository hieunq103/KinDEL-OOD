import pandas as pd, numpy as np, boto3, os
from botocore import UNSIGNED
from botocore.config import Config
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

TARGET = "ddr1"
os.makedirs("data", exist_ok=True)
dst = f"data/{TARGET}_1M.parquet"

if not os.path.exists(dst):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    print("đang tải, chờ vài phút...")
    s3.download_file("kin-del-2024", f"data/{TARGET}_1M.parquet", dst)
print("file size:", round(os.path.getsize(dst)/2**20, 1), "MB")

import pyarrow.parquet as pq
print("\ncác cột có thật trong file:")
print(pq.ParquetFile(dst).schema_arrow.names)

cols = ["smiles","smiles_a","smiles_b","smiles_c","molecule_hash",
        "seq_target_1","seq_target_2","seq_target_3",
        "seq_matrix_1","seq_matrix_2","seq_matrix_3","seq_load"]
df = pd.read_parquet(dst, columns=cols).head(20_000)
df.to_parquet("data/sample_20k.parquet")

t = df[["seq_target_1","seq_target_2","seq_target_3"]].to_numpy()
print("\n--- 4 CON SỐ ---")
print("1. tỉ lệ count = 0 :", round(float((t == 0).mean()), 4))
print("2. mean / var      :", round(float(t.mean()), 3), "/", round(float(t.var()), 3))
print("3. seq_load == 0   :", int((df["seq_load"] == 0).sum()))
bad = sum(Chem.MolFromSmiles(s) is None for s in df["smiles"].head(3000))
print("4. SMILES hỏng/3000:", bad)
print("\nsynthon duy nhất trong mẫu:",
      len(set(df["smiles_a"]) | set(df["smiles_b"]) | set(df["smiles_c"])))
print("\ndtypes:"); print(df.dtypes)
print("\n2 dòng đầu:"); print(df.head(2).to_string())