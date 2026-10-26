import pyarrow.parquet as pq, numpy as np, pandas as pd

dst = "data/ddr1_1M.parquet"
pf = pq.ParquetFile(dst)
print("row groups:", pf.num_row_groups, "| rows:", pf.metadata.num_rows)

cols = ["seq_target_1","seq_target_2","seq_target_3",
        "seq_matrix_1","seq_matrix_2","seq_matrix_3",
        "seq_load","target_enrichment"]

# quet toan bo file theo row-group, khong nap het vao RAM
n, zeros, s, ss, nan_t = 0, 0, 0.0, 0.0, 0
enr_first, enr_last = None, None
for i in range(pf.num_row_groups):
    d = pf.read_row_group(i, columns=cols).to_pandas()
    t = d[["seq_target_1","seq_target_2","seq_target_3"]].to_numpy(dtype="float64")
    nan_t += int(np.isnan(t).sum())
    t = np.nan_to_num(t)
    n += t.size; zeros += int((t == 0).sum())
    s += t.sum(); ss += (t**2).sum()
    if enr_first is None:
        enr_first = d["target_enrichment"].head(5).tolist()
    enr_last = d["target_enrichment"].tail(5).tolist()
    if i % 5 == 0:
        print(f"  rg {i}/{pf.num_row_groups}", flush=True)

mean = s/n; var = ss/n - mean**2
print("\n--- TOAN BO 1M ---")
print("ti le count = 0 :", round(zeros/n, 4))
print("mean / var      :", round(mean,3), "/", round(var,3),
      "| var/mean =", round(var/mean,2))
print("so NaN trong count target:", nan_t)
print("\ntarget_enrichment 5 dong DAU file :", enr_first)
print("target_enrichment 5 dong CUOI file:", enr_last)
