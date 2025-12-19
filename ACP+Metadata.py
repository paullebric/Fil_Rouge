import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# ==========
# Chemins
# ==========
DATA_DIR = "Data"
TABLE_TSV = f"{DATA_DIR}/table.tsv"
TAX_TSV   = f"{DATA_DIR}/taxonomy.tsv"
META_TSV  = f"{DATA_DIR}/sample-metadata.tsv"

# ==========
# Paramètres
# ==========
GROUP_COL = "env_material"   # Bulk soil / rhizosphere
RANK = "p__"                 # "p__" (phylum) ou "g__" (genus)
TOP_N = 15                   # barplot: top taxa
PSEUDOCOUNT = 1e-6           # CLR pseudocount

# -----------------------
# 1) Lire table.tsv correctement
#   - saute la ligne "# Constructed from biom file"
#   - gère "#OTU ID" splitté en "#OTU" + "ID"
# -----------------------
table = pd.read_csv(TABLE_TSV, sep=r"\s+", engine="python", header=0, skiprows=0)

if len(table.columns) > 1 and table.columns[0] == "#OTU" and table.columns[1] == "ID":
    table = table.rename(columns={"#OTU": "feature_id"}).drop(columns=["ID"])
else:
    table = table.rename(columns={table.columns[0]: "feature_id"})

table = table.set_index("feature_id")
table.columns = table.columns.str.strip()
table = table.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)  # counts

# -----------------------
# 2) Lire  metadata
# -----------------------

meta = pd.read_csv(META_TSV, sep="\t", dtype=str)
meta = meta.rename(columns={"#SampleID": "sample_id"}).set_index("sample_id")
meta.index = meta.index.str.strip()

# -----------------------
# 3) Aligner samples
# -----------------------
common_samples = table.columns.intersection(meta.index)
print("Samples communs table/metadata =", len(common_samples))
if len(common_samples) == 0:
    raise ValueError("Toujours aucun SampleID commun. Vérifie que tu utilises la bonne metadata (Run SRR...).")

table = table[common_samples]
meta = meta.loc[common_samples].copy()

# table : features x samples (counts)
# meta  : samples x metadata (index = sample IDs)

# --------
# 1) Préparer X (samples x features) + CLR
# --------
X_counts = table.T.astype(float)  # samples x ASV/OTU

# Filtrer samples vides (sécurité)
X_counts = X_counts.loc[X_counts.sum(axis=1) > 0]
meta = meta.loc[X_counts.index]

# Pseudocount + CLR
X_pc = X_counts + 1.0
geom_mean = np.exp(np.log(X_pc).mean(axis=1))
X_clr = np.log(X_pc.div(geom_mean, axis=0))

print("Samples utilisés :", X_clr.shape[0], " | Features :", X_clr.shape[1])

# --------
# 2) PCA
# --------
pca = PCA(n_components=2)
coords = pca.fit_transform(X_clr.values)

pca_df = pd.DataFrame(coords, index=X_clr.index, columns=["PC1", "PC2"]).join(meta)

# --------
# 3) Plot
# --------
plt.figure(figsize=(7,6))

if GROUP_COL in pca_df.columns:
    for g, sub in pca_df.groupby(GROUP_COL):
        plt.scatter(sub["PC1"], sub["PC2"], alpha=0.8, label=str(g))
    plt.legend(title=GROUP_COL)
else:
    plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.8)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA (CLR) – communautés")
plt.tight_layout()
plt.show()
#==================================================================================
#SCREE PLOT
pca10 = PCA(n_components=10)
coords10 = pca10.fit_transform(X_clr)
plt.figure(figsize=(5,4))
plt.plot(
    range(1, len(pca10.explained_variance_ratio_)+1),
    pca10.explained_variance_ratio_*100,
    marker="o"
)
plt.xlabel("Composantes principales")
plt.ylabel("Variance expliquée (%)")
plt.title("Scree plot – PCA CLR")
plt.tight_layout()
plt.show()

pca3 = PCA(n_components=3)
coords3 = pca3.fit_transform(X_clr)

pca3_df = pd.DataFrame(
    coords3,
    index=X_clr.index,
    columns=["PC1","PC2","PC3"]
).join(meta)

plt.figure(figsize=(6,5))
for g, sub in pca3_df.groupby(GROUP_COL):
    plt.scatter(sub["PC1"], sub["PC3"], label=g, alpha=0.8)

plt.xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC3 ({pca3.explained_variance_ratio_[2]*100:.1f}%)")
plt.legend(title=GROUP_COL)
plt.title("PCA CLR – PC1 vs PC3")
plt.tight_layout()
plt.show()
#==================================================================================
#
